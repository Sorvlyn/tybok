# Derived from LeRobot <https://github.com/huggingface/lerobot> fastwam/wan. Copyright 2024 The
# HuggingFace Inc. team; portions Copyright 2024-2025 The Alibaba Wan Team Authors. Licensed
# under the Apache License, Version 2.0; modified for TyBoK in 2026. See the NOTICE file.

"""FastWAM single layer (DiT block) core -- the unified forward entry shared by the video / action experts.
State keys ``mot.mixtures.<video|action>.blocks.<i>...`` align with the checkpoint.

Layer structure (once per layer)::

    x -- norm1 -- AdaLN(modulate) -- qkv proj -- qk-norm -- RoPE --+
    |                                                             | self SDPA
    +-------------- gate_msa x o-proj(attn) <----------------------+
    +-- norm3 -- cross-q -- cross SDPA(text context) -- o-proj ----> + residual
    +-- norm2 -- AdaLN -- ffn(up->GELU-tanh->down) -- gate_mlp ----> + residual -> out

Runtime tensors:
  - ``x``: [B, S, hidden] (video prefill S=120; action denoise S=32)
  - ``context``: [B, C, hidden] (C=129; text_embedding already applied before this block)
  - attention projection out = num_heads x attn_head_dim (video 3072->3072; action 1024->3072, projected attention)
  - ``t_mod``: [B, 6, hidden] (AdaLN modulation, produced by time_projection)

**Unified forward entry ``forward``**: all three forms -- video prefill / action denoise /
action cross-step cache -- call only it, with the differences expressed as optional parameters:
  - ``self_attn_mask``: bool [Lq, Lk]
  - ``video_kv``: (k, v), [B, L_video, n*d], concatenated before this block's self keys
  - ``context``(+``context_mask``) or ``context_kv``: cross context; the latter is the
    (k, v) precomputed by ``precompute_action_context``, already including qk-norm
  - ``return_kv=True``: returns (out, (k, v)), where k/v are this block's tokens' post-rope projections

Low-level sub-steps (``split_modulation`` / ``apply_norm*`` / ``project_self_attention`` /
``apply_cross_attention(_cached)`` / ``project_self_attention_output``) are kept as modular
interfaces (reference comparison / single-step debugging); regular inference goes through
``forward`` uniformly.

Weight precision: the block itself is pure bf16; fp8 conversion is done by the engine
replacing qkv/o/cross/ffn etc. Linear layers with ``FP8Linear`` before loading, and
quantization is transparent to forward (fp32 inputs are direct-quantized).

``pack_attention_qkv*`` are bf16/fp8 row-concatenation packing tools for attention
projections, called only at engine load time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as functional

if TYPE_CHECKING:
    # Only the two annotations below name it; the runtime import stays inside the functions
    # that build the packed matrices (``_cat_fp8_rows`` / ``pack_attention_qkv_fp8``).
    from .fp8_linear import FP8Linear

from .wan_base import (
    WanLayerNorm,
    WanRMSNorm,
    _linear_input,
    _wan_layer_norm,
    apply_dense_rope,
    rope_apply,
    rope_apply_grid,
)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Wan AdaLN modulation: ``x * (1 + scale) + shift``."""
    return x * (1 + scale) + shift


def fastwam_masked_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    ctx_mask: torch.Tensor | None = None,
    fp32_attention: bool = True,
) -> torch.Tensor:
    """SDPA attention kernel: reshape + optional fp32 upcast + SDPA + reshape.

    ``q/k/v`` are [B, S, n*d]; ``ctx_mask`` is bool [B?, Lq, Lk] (or [Lq, Lk]).
    When ``fp32_attention=False``, q/k are aligned to ``v.dtype`` (bf16 path).
    """
    b = q.shape[0]
    q = q.view(b, q.shape[1], num_heads, -1).permute(0, 2, 1, 3)
    k = k.view(b, k.shape[1], num_heads, -1).permute(0, 2, 1, 3)
    v = v.view(b, v.shape[1], num_heads, -1).permute(0, 2, 1, 3)
    if fp32_attention:
        q = q.float()
        k = k.float()
        v = v.float()
    else:
        q = q.to(dtype=v.dtype)
        k = k.to(dtype=v.dtype)
    x = functional.scaled_dot_product_attention(q, k, v, attn_mask=ctx_mask)
    x = x.permute(0, 2, 1, 3)
    return x.reshape(x.shape[0], x.shape[1], -1)


class _FastWAMProjectedAttention(nn.Module):
    """Attention projections when hidden_dim != attention_dim (action expert)."""

    def __init__(self, hidden_dim: int, attention_dim: int, num_heads: int, eps: float):
        super().__init__()
        self.dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        self.q = nn.Linear(hidden_dim, attention_dim)
        self.k = nn.Linear(hidden_dim, attention_dim)
        self.v = nn.Linear(hidden_dim, attention_dim)
        self.o = nn.Linear(attention_dim, hidden_dim)
        self.norm_q = WanRMSNorm(attention_dim, eps=eps)
        self.norm_k = WanRMSNorm(attention_dim, eps=eps)
        self.qkv = None
        self.kv = None

    def qkv_proj(self, x: torch.Tensor):
        if self.qkv is not None:
            return self.qkv(x).chunk(3, dim=-1)
        return self.q(x), self.k(x), self.v(x)

    def kv_proj(self, context: torch.Tensor):
        if self.kv is not None:
            return self.kv(context).chunk(2, dim=-1)
        return self.k(context), self.v(context)


class WanSelfAttention(nn.Module):
    """Attention projections when hidden_dim == attention_dim (video expert)."""

    def __init__(self, dim: int, num_heads: int, qk_norm: bool = True, eps: float = 1e-6):
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}).")
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.qkv = None  # packed qkv Linear (set by pack_attention_qkv)
        self.kv = None  # packed kv Linear (used by cross attention)

    def qkv_proj(self, x: torch.Tensor):
        """Self-attention's packed q/k/v projection (all three share input x)."""
        if self.qkv is not None:
            return self.qkv(x).chunk(3, dim=-1)
        return self.q(x), self.k(x), self.v(x)

    def kv_proj(self, context: torch.Tensor):
        """Cross-attention's packed k/v projection (k/v share input context)."""
        if self.kv is not None:
            return self.kv(context).chunk(2, dim=-1)
        return self.k(context), self.v(context)


def _build_attn(hidden_dim: int, attention_dim: int, num_heads: int, eps: float) -> nn.Module:
    if hidden_dim == attention_dim:
        return WanSelfAttention(hidden_dim, num_heads, qk_norm=True, eps=eps)
    return _FastWAMProjectedAttention(hidden_dim, attention_dim, num_heads, eps)


class FastWAMAttentionBlock(nn.Module):
    """Per-expert DiT block: Wan AdaLN modulation + SDPA attention.

    Layout: norm1 -> modulated self-attn -> gate residual; norm3 -> cross-attn over the
    text context; norm2 -> modulated gate FFN (GELU-tanh) -> gate residual. ``modulation``
    is (1, 6, dim); heads project hidden to attention_dim (video expert == hidden,
    action expert == num_heads*attn_head_dim). ``forward`` is the only inference entry.
    """

    def __init__(
        self,
        hidden_dim: int,
        attn_head_dim: int,
        num_heads: int,
        ffn_dim: int,
        eps: float = 1e-6,
        fp32_attention: bool = True,
    ):
        super().__init__()
        attention_dim = attn_head_dim * num_heads
        self.dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.qk_norm = True
        self.cross_attn_norm = True
        self.eps = eps
        self.norm1 = WanLayerNorm(hidden_dim, eps)
        self.self_attn = _build_attn(hidden_dim, attention_dim, num_heads, eps)
        self.norm3 = WanLayerNorm(hidden_dim, eps, elementwise_affine=True)
        self.cross_attn = _build_attn(hidden_dim, attention_dim, num_heads, eps)
        self.norm2 = WanLayerNorm(hidden_dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, hidden_dim) / hidden_dim**0.5)
        self.attn_head_dim = attn_head_dim
        self.fp32_attention = bool(fp32_attention)
        # fusion hooks (installed by `--cu-fused-vdit`); None = take the torch chain.
        # _fused_ffn_tail signature (x, shift_mlp, scale_mlp, gate_mlp) -> out;
        # _fused_attn takes over "self-attention segment + cross-attention segment", while
        # the FFN tail is still taken over by the former.
        self._fused_ffn_tail = None
        self._fused_attn = None
        # CUDA graph capture: the video token grid is a constant, so preset it here to
        # avoid the ``grid_sizes.tolist()`` D2H sync inside ``rope_apply`` during capture
        # (same mechanism as ``FusedAttnRunner.set_capture_grid``). None = production path.
        self._capture_grid = None

    # ------------------------------------------------------------------ #
    # low-level sub-steps (modular interface: for reference comparison / single-step debugging; regular inference uses forward)
    # ------------------------------------------------------------------ #
    @staticmethod
    def split_modulation(block, t_mod: torch.Tensor):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1

        base_mod = block.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (base_mod + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2),
                scale_msa.squeeze(2),
                gate_msa.squeeze(2),
                shift_mlp.squeeze(2),
                scale_mlp.squeeze(2),
                gate_mlp.squeeze(2),
            )
        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp

    def apply_norm1(self, x: torch.Tensor) -> torch.Tensor:
        return _wan_layer_norm(self.norm1, x)

    def apply_norm2(self, x: torch.Tensor) -> torch.Tensor:
        return _wan_layer_norm(self.norm2, x)

    def apply_norm3(self, x: torch.Tensor) -> torch.Tensor:
        return _wan_layer_norm(self.norm3, x)

    def project_self_attention(
        self, x: torch.Tensor, freqs: torch.Tensor | dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """q/k/v projection + qk-norm + RoPE -> (q, k, v), [B, S, n*d]."""
        q, k, v = self.self_attn.qkv_proj(x)
        q = self.self_attn.norm_q(q)
        k = self.self_attn.norm_k(k)
        if isinstance(freqs, dict):
            b, s = x.shape[:2]
            if self._capture_grid is not None:
                # CUDA-graph capture: no `.tolist()` D2H sync
                q = rope_apply_grid(
                    q.view(b, s, self.num_heads, self.attn_head_dim), self._capture_grid, freqs["freqs"]
                ).flatten(2)
                k = rope_apply_grid(
                    k.view(b, s, self.num_heads, self.attn_head_dim), self._capture_grid, freqs["freqs"]
                ).flatten(2)
            else:
                q = rope_apply(
                    q.view(b, s, self.num_heads, self.attn_head_dim), freqs["grid_sizes"], freqs["freqs"]
                ).flatten(2)
                k = rope_apply(
                    k.view(b, s, self.num_heads, self.attn_head_dim), freqs["grid_sizes"], freqs["freqs"]
                ).flatten(2)
        else:
            q = apply_dense_rope(q, freqs, self.num_heads)
            k = apply_dense_rope(k, freqs, self.num_heads)
        return q, k, v

    def project_self_attention_output(self, x: torch.Tensor) -> torch.Tensor:
        return self.self_attn.o(_linear_input(self.self_attn.o, x))

    def apply_cross_attention(
        self, x: torch.Tensor, context: torch.Tensor, context_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Cross-attention (k/v projected in place from context): called after norm3."""
        if context_mask is not None and context_mask.dim() == 3:
            context_mask = context_mask.unsqueeze(1)
        attn = self.cross_attn
        b, n, d = x.size(0), attn.num_heads, attn.head_dim
        q = attn.norm_q(attn.q(x)).view(b, -1, n * d)
        k, v = attn.kv_proj(context)
        k = attn.norm_k(k).view(b, -1, n * d)
        v = v.view(b, -1, n * d)
        x = fastwam_masked_attention(
            q=q,
            k=k,
            v=v,
            num_heads=n,
            ctx_mask=context_mask,
            fp32_attention=self.fp32_attention,
        )
        return attn.o(_linear_input(attn.o, x))

    def apply_cross_attention_cached(
        self,
        x: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Equivalent to ``apply_cross_attention``, but k/v are precomputed by
        ``precompute_action_context`` (including qk-norm); here only the cross-q projection
        + attention + o projection are done."""
        if context_mask is not None and context_mask.dim() == 3:
            context_mask = context_mask.unsqueeze(1)
        attn = self.cross_attn
        b, n, d = x.size(0), attn.num_heads, attn.head_dim
        q = attn.norm_q(attn.q(x)).view(b, -1, n * d)
        x = fastwam_masked_attention(
            q=q,
            k=k,
            v=v,
            num_heads=n,
            ctx_mask=context_mask,
            fp32_attention=self.fp32_attention,
        )
        return attn.o(_linear_input(attn.o, x))

    # ------------------------------------------------------------------ #
    # unified forward entry (MoT's video prefill / action denoise both come here)
    # ------------------------------------------------------------------ #
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None,
        t_mod: torch.Tensor,
        freqs: torch.Tensor | dict[str, torch.Tensor],
        context_mask: torch.Tensor | None = None,
        self_attn_mask: torch.Tensor | None = None,
        *,
        context_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        video_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_kv: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """One full layer forward: self-attention(gated residual) + cross-attention + FFN(gated residual).

        The form is determined by the optional parameters (see the module docstring):
        - default: single-block full forward, numerically consistent with the bf16/fp8 tiers.
        - action denoise: ``video_kv`` concatenates keys, ``context_kv`` uses the cross-step-cached cross k/v.
        - video prefill: ``return_kv=True`` returns ``(out, (k, v))`` (post-rope projections, not concatenated).
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.split_modulation(self, t_mod)
        # WARNING: the two paths **replace only the attention segment**; the FFN after it
        # is shared -- the fused branch must not `return` early, or the whole layer's FFN
        # would be skipped.
        if self._fused_attn is not None:
            # fused path: norm1+modulate -> qkv -> qk-norm/RoPE -> flash -> o + gate residual,
            # plus norm3 + cross-attention, all in one cooperative CUDA kernel (quantization drift tier).
            x, kv = self._fused_attn(
                x,
                context,
                shift_msa,
                scale_msa,
                gate_msa,
                freqs,
                context_mask=context_mask,
                context_kv=context_kv,
                video_kv=video_kv,
                self_attn_mask=self_attn_mask,
            )
        else:
            residual_x = x
            attn_input = modulate(self.apply_norm1(x), shift_msa, scale_msa)
            q, k, v = self.project_self_attention(attn_input, freqs)
            k_att, v_att = k, v
            if video_kv is not None:
                k_video, v_video = video_kv
                k_att = torch.cat([k_video, k], dim=1)
                v_att = torch.cat([v_video, v], dim=1)
            y = fastwam_masked_attention(
                q=q,
                k=k_att,
                v=v_att,
                num_heads=self.num_heads,
                ctx_mask=self_attn_mask,
                fp32_attention=self.fp32_attention,
            )
            x = residual_x + gate_msa * self.project_self_attention_output(y)
            if context_kv is not None:
                # precomputed cross k/v (action cross-step cache; includes qk-norm)
                kc, vc = context_kv
                x = x + self.apply_cross_attention_cached(self.apply_norm3(x), kc, vc, context_mask=context_mask)
            elif context is not None:
                x = x + self.apply_cross_attention(self.apply_norm3(x), context, context_mask=context_mask)
            kv = (k, v) if return_kv else None
        if self._fused_ffn_tail is not None:
            # fused path: norm2 + modulate + up->GELU->down + gate residual in one cooperative kernel (quantization drift tier).
            out = self._fused_ffn_tail(x, shift_mlp, scale_mlp, gate_mlp)
        else:
            mlp_input = modulate(self.apply_norm2(x), shift_mlp, scale_mlp)
            out = x + gate_mlp * self.ffn(mlp_input)
        if return_kv:
            return out, kv
        return out


# --------------------------------------------------------------------------- #
# attention projection packing tools (called at engine load time; not on the inference forward path)
# --------------------------------------------------------------------------- #
def _pack_self_qkv(attn: nn.Module) -> None:
    """Self-attention: q/k/v share input x, packed into a single [in -> 3*out] Linear."""
    in_f, out_f = attn.q.in_features, attn.q.out_features
    dtype, device = attn.q.weight.dtype, attn.q.weight.device
    w = torch.cat([attn.q.weight.data, attn.k.weight.data, attn.v.weight.data], dim=0)
    qkv = nn.Linear(in_f, out_f * 3, bias=attn.q.bias is not None).to(device=device, dtype=dtype)
    with torch.no_grad():
        qkv.weight.copy_(w)
        if attn.q.bias is not None:
            qkv.bias.copy_(torch.cat([attn.q.bias.data, attn.k.bias.data, attn.v.bias.data]))
    attn.qkv = qkv
    delattr(attn, "q")
    delattr(attn, "k")
    delattr(attn, "v")


def _pack_cross_kv(attn: nn.Module) -> None:
    """Cross-attention: k/v share input context, packed into a single [in -> 2*out] Linear; q is kept separate."""
    in_f, out_f = attn.k.in_features, attn.k.out_features
    dtype, device = attn.k.weight.dtype, attn.k.weight.device
    w = torch.cat([attn.k.weight.data, attn.v.weight.data], dim=0)
    kv = nn.Linear(in_f, out_f * 2, bias=attn.k.bias is not None).to(device=device, dtype=dtype)
    with torch.no_grad():
        kv.weight.copy_(w)
        if attn.k.bias is not None:
            kv.bias.copy_(torch.cat([attn.k.bias.data, attn.v.bias.data]))
    attn.kv = kv
    delattr(attn, "k")
    delattr(attn, "v")


def pack_attention_qkv(module: nn.Module) -> int:
    """Pack DiT attention's q/k/v into large matrices: self packs qkv (3x), cross packs kv (2x).

    Returns the number of packed attentions. The packed qkv/kv are used by qkv_proj/kv_proj,
    while q/k/v are removed to free the bf16 weights (so that later fp8 quantization applies
    only to the larger packed matrices).
    """
    count = 0
    for _name, child in list(module.named_children()):
        if isinstance(child, FastWAMAttentionBlock):
            _pack_self_qkv(child.self_attn)
            _pack_cross_kv(child.cross_attn)
            count += 2
        else:
            count += pack_attention_qkv(child)
    return count


def _cat_fp8_rows(parts: list[FP8Linear]) -> FP8Linear:
    """Concatenate multiple FP8Linear sharing the same input along output rows into one (row-concatenating fp8 weights / scale / bias)."""
    from .fp8_linear import FP8Linear

    first = parts[0]
    w = torch.cat([p.weight.data for p in parts], dim=0)
    sw = torch.cat([p.weight_scale.data for p in parts], dim=0)
    has_bias = first.bias is not None
    m = FP8Linear(first.in_features, sum(p.out_features for p in parts), bias=has_bias)
    m.weight = nn.Parameter(w)
    m.weight_scale = nn.Parameter(sw)
    if has_bias:
        m.bias = nn.Parameter(torch.cat([p.bias.data for p in parts], dim=0))
    else:
        m.register_parameter("bias", None)
    return m


def pack_attention_qkv_fp8(module: nn.Module) -> int:
    """fp8-resident packing: self q/k/v and cross k/v are already ``FP8Linear`` (fp8-resident),
    directly concatenated along fp8 rows into a single large GEMM (self 3x / cross 2x).

    After packing, each block's input is quantized only once (when unpacked, the three
    FP8Linear q/k/v would repeat per-token quantization 3 times on the same attn_input),
    and GEMM launches are merged. Requires all attention projections within the block to be
    FP8Linear; blocks that are not FP8Linear are skipped (left to bf16 ``pack_attention_qkv``).
    """
    from .fp8_linear import FP8Linear

    count = 0
    for _name, child in list(module.named_children()):
        if isinstance(child, FastWAMAttentionBlock):
            sa = child.self_attn
            if all(isinstance(getattr(sa, g), FP8Linear) for g in ("q", "k", "v")):
                sa.qkv = _cat_fp8_rows([sa.q, sa.k, sa.v])
                delattr(sa, "q")
                delattr(sa, "k")
                delattr(sa, "v")
                count += 1
            ca = child.cross_attn
            if all(isinstance(getattr(ca, g), FP8Linear) for g in ("k", "v")):
                ca.kv = _cat_fp8_rows([ca.k, ca.v])
                delattr(ca, "k")
                delattr(ca, "v")
                count += 1
        else:
            count += pack_attention_qkv_fp8(child)
    return count
