# Derived from transformers <https://github.com/huggingface/transformers>. Copyright 2023 Mesh
# TensorFlow authors, T5 Authors and HuggingFace Inc. team. Licensed under the Apache License,
# Version 2.0; modified for TyBoK in 2026. See the NOTICE file.

"""UMT5-XXL text encoder (encoder stack only).

UMT5 is a T5-family encoder: RMSNorm-style layer norm (scale only, no mean
removal), a per-layer learned relative-position bias
(``relative_attention_bias``, 32 buckets), gated GELU-new FFN
(``wi_0``/``wi_1``/``wo``), and no attention score scaling (softmax in fp32,
cast back to bf16).

Checkpoint weights are either bf16 shards or a per-row fp8 quantization
(``torch.float8_e4m3fn`` weights + ``*.scale_weight`` row scales). With
``load_umt5(fp8_resident=True)`` the quantized Linears are replaced by
``FP8Linear`` shells and the fp8 weight + per-row scale are copied **as-is**
(no dequant->requant double rounding); otherwise they are dequantised
``fp8.float() * scale[:, None]`` into bf16. Norms / embeddings / biases stay
bf16.

State-dict names match ``transformers.UMT5EncoderModel``; ``shared.weight``
is remapped onto ``encoder.embed_tokens.weight``.
"""

from __future__ import annotations

import glob
import json
import math
import os
import re

import torch
import torch.nn as nn

# key remap from the on-disk transformers state dict to our (decoder-free) tree
_KEY_REMAP = {
    "shared.weight": "encoder.embed_tokens.weight",
}

_FP8_WEIGHT_RE = re.compile(r"^(.*)\.weight$")
_FP8_SCALE_RE = re.compile(r"^(.*)\.scale_weight$")


class UMT5RMSNorm(nn.Module):
    """T5/UMT5 RMS-style layer norm (scale only, no shift / no mean removal)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        if self.weight.dtype in (torch.float16, torch.bfloat16):
            hidden_states = hidden_states.to(self.weight.dtype)
        return self.weight * hidden_states


class UMT5Attention(nn.Module):
    """Per-layer encoder self-attention (expects a pre-normed input)."""

    def __init__(
        self,
        d_model: int,
        d_kv: int,
        num_heads: int,
        num_buckets: int,
        max_distance: int,
    ):
        super().__init__()
        self.key_value_proj_dim = d_kv
        self.n_heads = num_heads
        self.inner_dim = self.n_heads * self.key_value_proj_dim
        self.relative_attention_num_buckets = num_buckets
        self.relative_attention_max_distance = max_distance
        self.is_decoder = False

        self.q = nn.Linear(d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, d_model, bias=False)

        self.relative_attention_bias = nn.Embedding(num_buckets, self.n_heads)

    def _relative_position_bucket(self, relative_position: torch.Tensor) -> torch.Tensor:
        """T5 mesh-tensorflow relative-position bucketing (encoder / bidirectional)."""
        num_buckets = self.relative_attention_num_buckets
        max_distance = self.relative_attention_max_distance

        num_buckets //= 2
        relative_buckets = (relative_position > 0).to(torch.long) * num_buckets
        relative_position = torch.abs(relative_position)

        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        log_ratio = torch.log(relative_position.float() / max_exact) / math.log(max_distance / max_exact)
        log_ratio = log_ratio * (num_buckets - max_exact)
        relative_position_if_large = max_exact + log_ratio.to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )
        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def _compute_bias(self, query_length: int, key_length: int) -> torch.Tensor:
        device = self.relative_attention_bias.weight.device
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position
        bucket = self._relative_position_bucket(relative_position)
        values = self.relative_attention_bias(bucket)  # (Lq, Lk, heads)
        return values.permute([2, 0, 1]).unsqueeze(0)  # (1, heads, Lq, Lk)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encoder self-attention over a *pre-normed* input.

        ``attention_mask``: additive [B, 1, 1, Lk] in the hidden-states dtype
        (the encoder stack pre-computes ``(1 - mask) * finfo(dtype).min``).
        """
        batch_size, seq_length = hidden_states.shape[:2]
        query_states = self.q(hidden_states).view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        key_states = self.k(hidden_states).view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        value_states = self.v(hidden_states).view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        scores = torch.matmul(query_states, key_states.transpose(3, 2))
        position_bias = self._compute_bias(seq_length, key_states.shape[-2])
        if attention_mask is not None:
            position_bias = position_bias + attention_mask
        scores += position_bias

        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)
        return self.o(attn_output)

    def pack_qkv(self) -> None:
        """Pack the q/k/v fp8 weights into a single qkv weight (for the fused kernel `tmt5.attn`).

        Packing frees the original q/k/v weights, so **only the fused path works after this**.
        """
        qkv_weight = torch.cat([self.q.weight.data, self.k.weight.data, self.v.weight.data], dim=0).contiguous()
        qkv_scale = torch.cat(
            [self.q.weight_scale.data, self.k.weight_scale.data, self.v.weight_scale.data], dim=0
        ).contiguous()
        self.qkv_weight = qkv_weight
        self.qkv_scale = qkv_scale
        self.q = None
        self.k = None
        self.v = None


class UMT5DenseGatedActDense(nn.Module):
    """Gated GELU-new FFN (``wi_0``/``wi_1``/``wo``)."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.wi_0 = nn.Linear(d_model, d_ff, bias=False)
        self.wi_1 = nn.Linear(d_model, d_ff, bias=False)
        self.wo = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # `gelu_new`: transformers ACT2FN['gelu_new'] python formula (NOT
        # `F.gelu(approximate='tanh')`; the two differ in bf16 rounding)
        hidden_gelu = self.wi_0(hidden_states)
        hidden_gelu = (
            0.5
            * hidden_gelu
            * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (hidden_gelu + 0.044715 * torch.pow(hidden_gelu, 3.0))))
        )
        hidden_linear = self.wi_1(hidden_states)
        return self.wo(hidden_gelu * hidden_linear)

    def pack_wi(self) -> None:
        """Pack the wi_0/wi_1 fp8 weights into a single wi weight (for the fused kernel `tmt5.ffn`).

        Packing frees the original wi_0/wi_1 weights, so **only the fused path works after this**.
        """
        wi_weight = torch.cat([self.wi_0.weight.data, self.wi_1.weight.data], dim=0).contiguous()
        wi_scale = torch.cat([self.wi_0.weight_scale.data, self.wi_1.weight_scale.data], dim=0).contiguous()
        self.wi_weight = wi_weight
        self.wi_scale = wi_scale
        self.wi_0 = None
        self.wi_1 = None


class UMT5LayerSelfAttention(nn.Module):
    """block.layer.0: pre-RMSNorm + self-attention + residual."""

    def __init__(self, d_model: int, d_kv: int, num_heads: int, num_buckets: int, max_distance: int, eps: float):
        super().__init__()
        self.SelfAttention = UMT5Attention(d_model, d_kv, num_heads, num_buckets, max_distance)
        self.layer_norm = UMT5RMSNorm(d_model, eps=eps)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        normed_hidden_states = self.layer_norm(hidden_states)
        return hidden_states + self.SelfAttention(normed_hidden_states, attention_mask=attention_mask)

    def fused_forward(self, hidden_states: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        """Fused: RMSNorm + packed qkv fp8 GEMM + attention + o-proj fp8 + residual.

        ``causal_mask`` is the additive column mask **without pos_bias**
        ``(1 - attention_mask) * finfo.min`` ([B,1,1,S]); it is taken over by the
        ``_fused_attn`` installed by `FusedAttnRunner.install()`, and the kernel consumes
        pos_bias and causal_mask separately.
        """
        fused = getattr(self, "_fused_attn", None)
        if fused is None:
            raise RuntimeError(
                "UMT5LayerSelfAttention.fused_forward requires the _fused_attn hook installed by FusedAttnRunner; "
                "pack_fused(attn=True) has freed the original q/k/v weights, so eager forward can no longer run"
            )
        return fused(hidden_states, self.SelfAttention._pos_bias, causal_mask)


class UMT5LayerFF(nn.Module):
    """block.layer.1: pre-RMSNorm + gated FFN + residual."""

    def __init__(self, d_model: int, d_ff: int, eps: float):
        super().__init__()
        self.DenseReluDense = UMT5DenseGatedActDense(d_model, d_ff)
        self.layer_norm = UMT5RMSNorm(d_model, eps=eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        forwarded_states = self.layer_norm(hidden_states)
        return hidden_states + self.DenseReluDense(forwarded_states)

    def fused_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Fused: RMSNorm + packed wi fp8 GEMM + gelu_new gating + wo fp8 + fp32 residual.

        Taken over by the ``_fused_ffn`` installed by `FusedTextRunner.install()`.
        """
        fused = getattr(self, "_fused_ffn", None)
        if fused is None:
            raise RuntimeError(
                "UMT5LayerFF.fused_forward requires the _fused_ffn hook installed by FusedTextRunner; "
                "pack_fused(ffn=True) has freed the original wi_0/wi_1 weights, so eager forward can no longer run"
            )
        return fused(hidden_states)


class UMT5Block(nn.Module):
    """One encoder layer: [SelfAttention, FFN]."""

    def __init__(
        self, d_model: int, d_kv: int, d_ff: int, num_heads: int, num_buckets: int, max_distance: int, eps: float
    ):
        super().__init__()
        self.layer = nn.ModuleList(
            [
                UMT5LayerSelfAttention(d_model, d_kv, num_heads, num_buckets, max_distance, eps),
                UMT5LayerFF(d_model, d_ff, eps),
            ]
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        hidden_states = self.layer[0](hidden_states, attention_mask)
        return self.layer[1](hidden_states)


class UMT5Stack(nn.Module):
    """Encoder stack (names under ``encoder.``)."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        d_kv: int,
        d_ff: int,
        num_heads: int,
        num_buckets: int,
        max_distance: int,
        eps: float,
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        self.block = nn.ModuleList(
            [UMT5Block(d_model, d_kv, d_ff, num_heads, num_buckets, max_distance, eps) for _ in range(num_layers)]
        )
        self.final_layer_norm = UMT5RMSNorm(d_model, eps=eps)

    def pack_fused(
        self, seq_len: int, dtype: torch.dtype = torch.bfloat16, attn: bool = True, ffn: bool = True
    ) -> None:
        """Pack qkv/wi weights + precompute each layer's position_bias; enable the fused
        forward per attn/ffn.

        ``attn`` controls the self-attention sublayer, ``ffn`` controls the FFN sublayer.
        Packing frees the original q/k/v or wi_0/wi_1 weights, so **only the fused path
        works after this**: the packed half must have the matching hook installed by
        `FusedAttnRunner` / `FusedTextRunner`, otherwise `fused_forward` errors directly
        (the eager forward no longer has weights).
        """
        self.fused_attn = attn
        self.fused_ffn = ffn
        self.fused = attn or ffn
        if not self.fused:
            return
        for block in self.block:
            if attn:
                sa = block.layer[0].SelfAttention
                sa.pack_qkv()
                sa._pos_bias = sa._compute_bias(seq_len, seq_len).to(dtype=dtype)
            if ffn:
                block.layer[1].DenseReluDense.pack_wi()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        # the embedding layer may stay on CPU (saves VRAM in deployment) while the blocks
        # are on GPU; move the embeddings and mask to the block's device here. Same device
        # is a no-cost no-op.
        hidden_states = self.embed_tokens(input_ids)
        block_device = next(self.block.parameters()).device
        hidden_states = hidden_states.to(block_device)
        dtype = hidden_states.dtype
        if attention_mask is None:
            attention_mask = torch.ones(
                (hidden_states.shape[0], hidden_states.shape[1]),
                dtype=torch.long,
                device=block_device,
            )
        elif attention_mask.device != block_device:
            attention_mask = attention_mask.to(block_device)
        # additive mask: eager and fused attention share the same one (fused's _pos_bias is a precomputed cache)
        causal_mask = attention_mask[:, None, None, :].to(dtype=dtype)
        causal_mask = (1.0 - causal_mask) * torch.finfo(dtype).min

        fused_attn = getattr(self, "fused_attn", False)
        fused_ffn = getattr(self, "fused_ffn", False)

        for block in self.block:
            attn = block.layer[0]
            ffn = block.layer[1]
            if fused_attn:
                # pos_bias is consumed by the fused kernel itself
                hidden_states = attn.fused_forward(hidden_states, causal_mask)
            else:
                hidden_states = attn.forward(hidden_states, causal_mask)
            if fused_ffn:
                hidden_states = ffn.fused_forward(hidden_states)
            else:
                hidden_states = ffn.forward(hidden_states)
        return self.final_layer_norm(hidden_states)


class UMT5Encoder(nn.Module):
    """UMT5-XXL encoder-only model (state-dict names follow transformers)."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        d_kv: int,
        d_ff: int,
        num_heads: int,
        num_buckets: int,
        max_distance: int,
        eps: float,
    ):
        super().__init__()
        self.encoder = UMT5Stack(vocab_size, d_model, num_layers, d_kv, d_ff, num_heads, num_buckets, max_distance, eps)
        self.dim = d_model  # FastWAM contract alias

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """``last_hidden_state`` [B, L, d_model] (like ``UMT5EncoderModel(...).last_hidden_state``)."""
        return self.encoder(input_ids, attention_mask=attention_mask)


# --------------------------------------------------------------------------- #
# weight loading
# --------------------------------------------------------------------------- #
def _weight_files(dir: str) -> list[str]:
    single = os.path.join(dir, "model.safetensors")
    if os.path.isfile(single):
        return [single]
    index = os.path.join(dir, "model.safetensors.index.json")
    if os.path.isfile(index):
        with open(index, "r", encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
        shards = sorted({os.path.join(dir, v) for v in weight_map.values()})
        return shards
    shards = sorted(glob.glob(os.path.join(dir, "model-*.safetensors")))
    if shards:
        return shards
    raise ValueError(f"no text-encoder weights (*.safetensors) found in {dir!r}")


def _is_fp8_file(path: str) -> bool:
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as f:
        return any(k.endswith(".scale_weight") for k in f.keys())


def _dequant_fp8_weight(w8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Row-scaled fp8 e4m3fn -> bf16. Supports scalar (per-tensor) and 1-D row scales."""
    if w8.dtype != torch.float8_e4m3fn:
        raise TypeError(f"expected float8_e4m3fn weight, got {w8.dtype}")
    if scale.ndim == 0:
        w = w8.float() * scale.float()
    elif scale.ndim == 1:
        w = w8.float() * scale.float().unsqueeze(1)
    else:
        raise ValueError(f"unsupported fp8 scale shape {tuple(scale.shape)}")
    return w.to(torch.bfloat16)


def _load_tensor_cpu(path: str, key: str) -> torch.Tensor:
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as f:
        return f.get_tensor(key)


def _load_fp8_resident(fp8_path: str, model: nn.Module, dtype: torch.dtype) -> int:
    """fp8 shell direct load: swap Linear for an ``FP8Linear`` shell, then **copy** the fp8
    weight + per-row scale straight from the fp8 file (skipping the dequant->bf16->requant
    double rounding). Returns the number of replaced Linears.

    Non-quantized weights (embedding / norm / position bias) are still copied as bf16;
    after copying, non-``FP8Linear`` parameters are unified to ``dtype``.
    """
    from .fp8_linear import FP8Linear, fp8ify_structural

    n = fp8ify_structural(model, min_dim=256)
    params = {name: p for name, p in model.named_parameters()}
    missing = list(params)
    scales = {m.group(1): _load_tensor_cpu(fp8_path, k) for k in _load_keys(fp8_path) if (m := _FP8_SCALE_RE.match(k))}
    state: dict[str, torch.Tensor] = {}
    for k in _load_keys(fp8_path):
        m = _FP8_WEIGHT_RE.match(k)
        if m is None:
            continue
        base = m.group(1)
        key = _KEY_REMAP.get(k, k)
        if base in scales:
            tgt = params.get(key)
            if tgt is not None and tgt.dtype == torch.float8_e4m3fn:
                state[key] = _load_tensor_cpu(fp8_path, k)  # w8 direct copy (bit-identical)
                state[base + ".weight_scale"] = scales[base]
            else:
                state[key] = _dequant_fp8_weight(_load_tensor_cpu(fp8_path, k), scales[base])
        else:
            # non-quantized weight (embedding, norms, relative bias)
            state[key] = _load_tensor_cpu(fp8_path, k)
    for key, tensor in state.items():
        if key not in params:
            continue
        params[key].data.copy_(tensor.reshape(params[key].shape))
        missing.remove(key)
    if missing:
        raise ValueError(f"text-encoder weights missing from checkpoint: {missing[:8]} ...")
    # FP8Linear keeps fp8 weight + fp32 scale; other parameters are unified to dtype
    for module in model.modules():
        if isinstance(module, FP8Linear):
            continue
        for p in module.parameters(recurse=False):
            if p.dtype != dtype:
                p.data = p.data.to(dtype=dtype)
    return n


def _load_keys(fp8_path: str) -> list[str]:
    from safetensors import safe_open

    with safe_open(fp8_path, framework="pt", device="cpu") as f:
        return list(f.keys())


def load_umt5(
    text_encoder_dir: str,
    dtype: torch.dtype = torch.bfloat16,
    fp8_resident: bool = False,
) -> UMT5Encoder:
    """Build + load the UMT5 encoder from a local dir (bf16 shards or fp8 file).

    ``fp8_resident=True``: if the dir contains an fp8 file, use "fp8 shell direct load";
    the weight bit pattern is bit-identical to the fp8 file with no dequant->requant
    double rounding; if the dir has only bf16 shards, load bf16 directly.
    """
    cfg_path = os.path.join(text_encoder_dir, "config.json")
    if not os.path.isfile(cfg_path):
        raise ValueError(f"no config.json in text-encoder dir {text_encoder_dir!r}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    model = UMT5Encoder(
        vocab_size=int(cfg.get("vocab_size", 256384)),
        d_model=int(cfg.get("d_model", 4096)),
        num_layers=int(cfg.get("num_layers", 24)),
        d_kv=int(cfg.get("d_kv", 64)),
        d_ff=int(cfg.get("d_ff", 10240)),
        num_heads=int(cfg.get("num_heads", 64)),
        num_buckets=int(cfg.get("relative_attention_num_buckets", 32)),
        max_distance=int(cfg.get("relative_attention_max_distance", 128)),
        eps=float(cfg.get("layer_norm_epsilon", 1e-6)),
    )

    files = _weight_files(text_encoder_dir)
    fp8_path = next((p for p in files if _is_fp8_file(p)), None)
    state: dict[str, torch.Tensor] = {}
    if fp8_path is not None:
        if fp8_resident:
            # fp8 direct load: the shell swap must come before weight copying
            _load_fp8_resident(fp8_path, model, dtype)
            return model
        scales = {
            m.group(1): _load_tensor_cpu(fp8_path, k) for k in _load_keys(fp8_path) if (m := _FP8_SCALE_RE.match(k))
        }
        for k in _load_keys(fp8_path):
            m = _FP8_WEIGHT_RE.match(k)
            if m is None:
                continue
            base = m.group(1)
            if base in scales:
                state[_KEY_REMAP.get(k, k)] = _dequant_fp8_weight(_load_tensor_cpu(fp8_path, k), scales[base])
            else:
                # non-quantized weight (embedding, norms, relative bias)
                t = _load_tensor_cpu(fp8_path, k)
                state[_KEY_REMAP.get(k, k)] = t.to(dtype=dtype) if t.dtype != dtype else t
    else:
        for path in files:
            from safetensors import safe_open

            with safe_open(path, framework="pt", device="cpu") as f:
                for k in f.keys():
                    t = f.get_tensor(k)
                    state[_KEY_REMAP.get(k, k)] = t.to(dtype=dtype) if t.dtype != dtype else t

    # copy into the module tree by parameter name (skip unknown keys)
    params = {name: p for name, p in model.named_parameters()}
    missing = list(params)
    for key, tensor in state.items():
        if key not in params:
            continue  # decoder keys in the bf16 shards etc.
        params[key].data.copy_(tensor.reshape(params[key].shape))
        missing.remove(key)
    if missing:
        raise ValueError(f"text-encoder weights missing from checkpoint: {missing[:8]} ...")
    model.to(dtype=dtype)
    return model
