# Derived from transformers <https://github.com/huggingface/transformers>. Copyright 2024 Google
# Inc. HuggingFace Inc. team. Licensed under the Apache License, Version 2.0; modified for
# TyBoK in 2026. See the NOTICE file.

"""Gemma decoder building blocks for pi0.5 (transformers 5.5.4 semantics).

Norms run in fp32 (``x*rsqrt(mean(x^2)+eps)``, ``* (1+weight)``); with ``cond_dim``
they become the expert's AdaRMS norm (``dense(cond)`` -> scale/shift/gate, gated
residuals). Attention is GQA with an fp32 softmax over bf16 logits, an additive
4-D mask and ``scaling = head_dim**-0.5``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from .rope import apply_rotary_pos_emb


def gelu_pytorch_tanh(x: torch.Tensor) -> torch.Tensor:
    """Native CUDA tanh-approx GELU; the fused op is required for bit-exactness."""
    return F.gelu(x, approximate="tanh")


def gated_residual(x: torch.Tensor, y: torch.Tensor, gate: torch.Tensor | None) -> torch.Tensor:
    """x + y when gate is None, else x + y * gate (AdaRMS gated residual)."""
    if gate is None:
        return x + y
    return x + y * gate


class GemmaRMSNorm(nn.Module):
    """Gemma RMSNorm; with ``cond_dim`` set it behaves as the AdaRMS norm (expert)."""

    def __init__(self, dim: int, eps: float = 1e-6, cond_dim: int | None = None):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.cond_dim = cond_dim
        if cond_dim is not None:
            # AdaRMS modulation: scale/shift/gate from the conditioning vector.
            self.dense = nn.Linear(cond_dim, dim * 3, bias=True, dtype=torch.float32)
            nn.init.zeros_(self.dense.weight)
        else:
            # (non-adaptive) norms stay fp32 regardless of model dtype
            self.weight = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
            self.dense = None

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor | None = None,
        modulation: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Returns (normalized, gate); gate is None for plain Gemma norms.

        ``modulation`` is a precomputed ``dense(cond)`` ``[*, 3*dim]`` (values
        identical to the eager call); a ``[*, 2*dim]`` scale/shift-only form returns
        gate None for a norm whose gate is never consumed.
        """
        dtype = x.dtype
        normed = self._norm(x)
        if self.dense is None:
            # plain Gemma norm (fp32 weights)
            normed = normed * (1.0 + self.weight.float())
            return normed.type_as(x), None
        if modulation is None:
            modulation = self.dense(cond)
        if len(x.shape) == 3:
            modulation = modulation.unsqueeze(1)
        if modulation.shape[-1] == 2 * self.dim:
            # scale/shift-only modulation (final norm never consumes the gate)
            scale, shift = modulation.chunk(2, dim=-1)
            normed = normed * (1 + scale.float()) + shift.float()
            return normed.to(dtype), None
        scale, shift, gate = modulation.chunk(3, dim=-1)
        normed = normed * (1 + scale.float()) + shift.float()
        return normed.to(dtype), gate.to(dtype)


class GemmaMLP(nn.Module):
    def __init__(self, width: int, mlp_dim: int, dtype=torch.bfloat16):
        super().__init__()
        self.gate_proj = nn.Linear(width, mlp_dim, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(width, mlp_dim, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(mlp_dim, width, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(gelu_pytorch_tanh(self.gate_proj(x)) * self.up_proj(x))


class GemmaAttention(nn.Module):
    """GQA eager attention: fp32 softmax over bf16 logits, additive 4-D mask.

    GQA is expanded to full heads first (this torch build's fused SDPA backends
    reject GQA shapes).
    """

    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int, width: int, dtype=torch.bfloat16):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_key_value_groups = num_heads // num_kv_heads
        self.scaling = head_dim**-0.5
        # ``--tl-llm-flash-attn``: concat qkv GEMM (cuBLAS) + GQA-native flash path
        self.fused_prefill_attn = False
        self._qkv_weight: torch.Tensor | None = None

        self.q_proj = nn.Linear(width, num_heads * head_dim, bias=False, dtype=dtype)
        self.k_proj = nn.Linear(width, num_kv_heads * head_dim, bias=False, dtype=dtype)
        self.v_proj = nn.Linear(width, num_kv_heads * head_dim, bias=False, dtype=dtype)
        self.o_proj = nn.Linear(num_heads * head_dim, width, bias=False, dtype=dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, L, W]
        attention_mask: torch.Tensor,  # 4-D additive mask [B, 1, Lq, Lk] (f32)
        position_ids: torch.Tensor,  # [B, Lq]
        prefix_kv: tuple[torch.Tensor, torch.Tensor] | None,  # (k, v) [B, Hkv, Lk_pref, D] or None
        position_embeddings: tuple[torch.Tensor, torch.Tensor],  # (cos, sin) [B, L, D]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (attn_output, key_states, value_states) -- the post-RoPE K/V
        (concatenated with the prefix cache when ``prefix_kv`` is given)."""
        if self.fused_prefill_attn:
            return self._forward_fused_prefill(
                hidden_states, attention_mask, position_ids, prefix_kv, position_embeddings
            )
        return self._forward_eager(hidden_states, attention_mask, position_ids, prefix_kv, position_embeddings)

    def _project(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # [B, H, Lq, D]
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # [B, Hkv, Lq, D]
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        return query_states, key_states, value_states

    def _rope_and_cache(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        prefix_kv: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        query_states, key_states, value_states = self._project(hidden_states)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if prefix_kv is not None:
            # cross-attend over [prefix; suffix]: concat this layer's suffix K/V
            # after the shared prefix KV (matches the reference DynamicCache).
            prefix_k, prefix_v = prefix_kv
            key_states = torch.cat([prefix_k, key_states], dim=2)
            value_states = torch.cat([prefix_v, value_states], dim=2)

        # keep pre-repeat [B, Hkv, L, D] K/V for the cache capture (before GQA expansion)
        cache_k, cache_v = key_states, value_states
        return query_states, key_states, value_states, cache_k, cache_v

    def _forward_eager(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        prefix_kv: tuple[torch.Tensor, torch.Tensor] | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query_states, key_states, value_states, cache_k, cache_v = self._rope_and_cache(
            hidden_states, position_embeddings, prefix_kv
        )

        # k/v are repeated here; the cache stores the pre-repeat [B, Hkv, L, D] tensors
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*hidden_states.shape[:-1], -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, cache_k, cache_v

    def _forward_fused_prefill(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        prefix_kv: tuple[torch.Tensor, torch.Tensor] | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Concat qkv GEMM + RoPE + GQA-native flash over the pre-expand k/v.

        q/k/v collapse into one ``F.linear`` over the concatenated ``[H+G+G, D]``
        weight; attention runs over the pre-expand ``[L, G, D]`` k/v. o_proj stays
        cuBLAS.
        """
        from .triton_prefill_attn import triton_prefill_attn

        if self._qkv_weight is None:
            self._qkv_weight = torch.cat(
                [self.q_proj.weight.data, self.k_proj.weight.data, self.v_proj.weight.data], dim=0
            ).contiguous()
        H, G, D = self.num_heads, self.num_kv_heads, self.head_dim
        B, L, _ = hidden_states.shape

        qkv = F.linear(hidden_states, self._qkv_weight)  # [B, L, (H+G+G)*D]
        q = qkv[..., : H * D].view(B, L, H, D).transpose(1, 2)
        k = qkv[..., H * D : H * D + G * D].view(B, L, G, D).transpose(1, 2)
        v = qkv[..., H * D + G * D :].view(B, L, G, D).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        cache_k, cache_v = k, v  # [B, G, L, D] post-RoPE (pre-expand)

        mask2d = attention_mask.squeeze(1).squeeze(0) == 0.0  # [L, L] bool
        attn = triton_prefill_attn(
            q.squeeze(0).permute(1, 0, 2).contiguous(),
            k.squeeze(0).permute(1, 0, 2).contiguous(),
            v.squeeze(0).permute(1, 0, 2).contiguous(),
            mask2d,
            D,
        )  # [L, H*D]
        attn_output = self.o_proj(attn.unsqueeze(0))  # [B, L, K]
        return attn_output, cache_k, cache_v


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class GemmaDecoderLayer(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        width: int,
        mlp_dim: int,
        use_adarms: bool,
        dtype=torch.bfloat16,
    ):
        super().__init__()
        self.self_attn = GemmaAttention(num_heads, num_kv_heads, head_dim, width, dtype=dtype)
        self.mlp = GemmaMLP(width, mlp_dim, dtype=dtype)
        self.input_layernorm = GemmaRMSNorm(width, eps=1e-6, cond_dim=width if use_adarms else None)
        self.post_attention_layernorm = GemmaRMSNorm(width, eps=1e-6, cond_dim=width if use_adarms else None)
        # Fused Triton denoising attention / MLP tiers; ``triton_ctx`` carries the
        # RoPE tables / eps and is installed by the engine.
        self.fused_attn = False
        self.fused_mlp = False
        self.use_fp8_mlp = False
        self.triton_ctx: dict | None = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        prefix_kv: tuple[torch.Tensor, torch.Tensor] | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        adarms_cond: torch.Tensor | None = None,
        adarms_mods: tuple[torch.Tensor, torch.Tensor] | None = None,
        triton_prefix_pad: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # ---- attention block ----
        if self.fused_attn and prefix_kv is not None and adarms_mods is not None and triton_prefix_pad is not None:
            hidden_states, key_states, value_states = self._forward_fused_attn(
                hidden_states, position_ids, prefix_kv, adarms_mods[0], triton_prefix_pad
            )
        else:
            residual = hidden_states
            if adarms_mods is not None:
                hidden_states, gate = self.input_layernorm(hidden_states, modulation=adarms_mods[0])
            else:
                hidden_states, gate = self.input_layernorm(hidden_states, cond=adarms_cond)
            hidden_states, key_states, value_states = self.self_attn(
                hidden_states, attention_mask, position_ids, prefix_kv, position_embeddings
            )
            hidden_states = gated_residual(residual, hidden_states, gate)

        # ---- MLP block ----
        if self.use_fp8_mlp and adarms_mods is not None:
            hidden_states = self._forward_fp8_mlp(hidden_states, adarms_mods[1])
        elif self.fused_mlp and adarms_mods is not None:
            hidden_states = self._forward_fused_mlp(hidden_states, adarms_mods[1])
        else:
            residual = hidden_states
            if adarms_mods is not None:
                hidden_states, gate = self.post_attention_layernorm(hidden_states, modulation=adarms_mods[1])
            else:
                hidden_states, gate = self.post_attention_layernorm(hidden_states, cond=adarms_cond)
            hidden_states = self.mlp(hidden_states)
            hidden_states = gated_residual(residual, hidden_states, gate)
        return hidden_states, key_states, value_states

    def _gated_residual_buffers(self, shape: torch.Size, site: str, device) -> tuple[torch.Tensor, torch.Tensor]:
        """Persistent (out, prod) buffers for one fused-gated-residual site.

        Created on first use (before CUDA-graph capture) so replays reuse stable
        addresses; per-call allocations during capture have shown to alias/drift.
        """
        scratch = getattr(self, "_res_scratch", None)
        if scratch is None:
            scratch = {}
            self._res_scratch = scratch
        out = scratch.get(f"{site}_out")
        prod = scratch.get(f"{site}_prod")
        if out is None or out.shape != shape:
            out = torch.empty(shape, dtype=torch.bfloat16, device=device)
            scratch[f"{site}_out"] = out
        if prod is None or prod.shape != shape:
            prod = torch.empty(shape, dtype=torch.bfloat16, device=device)
            scratch[f"{site}_prod"] = prod
        return out, prod

    def _forward_fused_attn(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        prefix_kv: tuple[torch.Tensor, torch.Tensor],
        in_mod: torch.Tensor,
        triton_prefix_pad: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None]:
        """Denoising attention block: AdaRMS norm + fused qkv GEMM+RoPE + GQA flash,
        then o_proj + gated residual (fused qkv+RoPE+flash build)."""
        ctx = self.triton_ctx
        assert ctx is not None

        prefix_k, prefix_v = prefix_kv
        from .triton_expert_fused_qkv_rope_attn import triton_expert_attn

        scratch = getattr(self, "_fused_attn_scratch", None)
        if scratch is None:
            scratch = {}
            self._fused_attn_scratch = scratch
        attn = triton_expert_attn(
            hidden_states.squeeze(0),
            in_mod.reshape(-1),
            self.self_attn.q_proj,
            self.self_attn.k_proj,
            self.self_attn.v_proj,
            prefix_k.squeeze(0).squeeze(0),
            prefix_v.squeeze(0).squeeze(0),
            triton_prefix_pad.squeeze(0),
            position_ids.squeeze(0),
            ctx["cos"],
            ctx["sin"],
            ctx["eps"],
            scratch=scratch,
        )
        attn = self.self_attn.o_proj(attn).unsqueeze(0)  # [1, M, K] cuBLAS
        # fused gated residual (bit-exact with ``residual + attn*gate``)
        from .triton_expert_residual import gated_residual

        s = self._gated_residual_buffers(hidden_states.shape, "a", hidden_states.device)
        hidden_states = gated_residual(hidden_states, attn, in_mod.reshape(-1), out=s[0], prod=s[1])
        return hidden_states, None, None

    def _forward_fused_mlp(self, hidden_states: torch.Tensor, post_mod: torch.Tensor) -> torch.Tensor:
        """Fused denoising MLP block: post-norm + gate/up + GELU, then down_proj +
        gated residual (down_proj stays on cuBLAS)."""
        from .triton_expert_mlp import triton_norm_gate_up

        ctx = self.triton_ctx
        assert ctx is not None
        residual = hidden_states

        act = triton_norm_gate_up(
            hidden_states.squeeze(0),
            post_mod.reshape(-1),
            self.mlp.gate_proj.weight,
            self.mlp.up_proj.weight,
            ctx["eps"],
        )
        hidden_states = self.mlp.down_proj(act).unsqueeze(0)
        # fused gated residual (bit-exact with ``residual + down*post_gate``)
        from .triton_expert_residual import gated_residual

        s = self._gated_residual_buffers(hidden_states.shape, "m", hidden_states.device)
        hidden_states = gated_residual(residual, hidden_states, post_mod.reshape(-1), out=s[0], prod=s[1])
        return hidden_states

    def _forward_fp8_mlp(self, hidden_states: torch.Tensor, post_mod: torch.Tensor) -> torch.Tensor:
        """W8A8 fp8 denoising MLP block: fused fp8 norm+gate/up+GELU, then the
        fp8 down GEMM with a producer-fused activation quant (no host round
        trip), then gated residual (drift tier)."""
        from .triton_expert_mlp_fp8 import triton_mlp_fp8_chain

        ctx = self.triton_ctx
        assert ctx is not None
        residual = hidden_states

        down = triton_mlp_fp8_chain(
            hidden_states.squeeze(0),
            post_mod.reshape(-1),
            self.mlp.gate_proj.weight,
            self.mlp.up_proj.weight,
            self.mlp.down_proj.weight,
            ctx["eps"],
        ).unsqueeze(0)
        # fused gated residual (bit-exact with ``residual + down*post_gate``)
        from .triton_expert_residual import gated_residual

        s = self._gated_residual_buffers(hidden_states.shape, "m", hidden_states.device)
        return gated_residual(residual, down, post_mod.reshape(-1), out=s[0], prod=s[1])
