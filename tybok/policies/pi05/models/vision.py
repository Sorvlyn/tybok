# Derived from transformers <https://github.com/huggingface/transformers>. Copyright 2024 Google
# AI and The HuggingFace Team. Licensed under the Apache License, Version 2.0; modified for
# TyBoK in 2026. See the NOTICE file.

"""SigLIP vision tower + PaliGemma projector for pi0.5 (transformers 5.5.4).

The whole vision path runs in float32 and the encoder attention uses SDPA; both
matter for bit-exactness. ``embed_image`` returns all 256 patches (no CLS token),
i.e. ``(B, 256, projection_dim)`` -- every patch becomes a prefix token.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from .layers import gelu_pytorch_tanh


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, hidden_size: int, image_size: int, patch_size: int, dtype=torch.float32):
        super().__init__()
        self.patch_embedding = nn.Conv2d(
            3, hidden_size, kernel_size=patch_size, stride=patch_size, padding="valid", dtype=dtype
        )
        self.num_patches = (image_size // patch_size) ** 2
        self.position_embedding = nn.Embedding(self.num_patches, hidden_size, dtype=dtype)
        self.register_buffer("position_ids", torch.arange(self.num_patches).expand((1, -1)), persistent=False)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # [B, W, grid, grid]
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class SiglipAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dtype=torch.float32):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.k_proj = nn.Linear(hidden_size, hidden_size, dtype=dtype)
        self.v_proj = nn.Linear(hidden_size, hidden_size, dtype=dtype)
        self.q_proj = nn.Linear(hidden_size, hidden_size, dtype=dtype)
        self.out_proj = nn.Linear(hidden_size, hidden_size, dtype=dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)
        queries = queries.view(hidden_shape).transpose(1, 2)
        keys = keys.view(hidden_shape).transpose(1, 2)
        values = values.view(hidden_shape).transpose(1, 2)

        # SDPA, no mask, is_causal=False.
        attn_output = F.scaled_dot_product_attention(
            queries, keys, values, attn_mask=None, dropout_p=0.0, scale=self.scale, is_causal=False
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1)
        return self.out_proj(attn_output)


class SiglipMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, dtype=torch.float32):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size, dtype=dtype)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, dtype=dtype)
        # Set by ``set_mlp_dtype``: when not None, only the two MLP GEMMs run in that dtype.
        self._mlp_dtype: torch.dtype | None = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._mlp_dtype is not None:
            # MLP-only low-precision tier: boundary casts keep every other f32 op
            # (LayerNorm / attention / residuals) bit-identical; only fc1/fc2 change
            # precision.
            dt = self._mlp_dtype
            hidden_states = self.fc1(hidden_states.to(dt))
            hidden_states = gelu_pytorch_tanh(hidden_states)
            return self.fc2(hidden_states).to(torch.float32)
        hidden_states = self.fc1(hidden_states)
        hidden_states = gelu_pytorch_tanh(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class SiglipEncoderLayer(nn.Module):
    def __init__(
        self, hidden_size: int, num_heads: int, intermediate_size: int, eps: float = 1e-6, dtype=torch.float32
    ):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=eps, dtype=dtype)
        self.self_attn = SiglipAttention(hidden_size, num_heads, dtype=dtype)
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=eps, dtype=dtype)
        self.mlp = SiglipMLP(hidden_size, intermediate_size, dtype=dtype)
        # Triton fused tiers: pre-packed weights + flags, set by the set_triton_* methods.
        self._fused_attn = False
        self._fused_mlp = False
        self._wqkv: torch.Tensor | None = None
        self._bqkv: torch.Tensor | None = None
        self._w1: torch.Tensor | None = None
        self._b1: torch.Tensor | None = None
        self._w2: torch.Tensor | None = None
        self._b2: torch.Tensor | None = None
        self._wout: torch.Tensor | None = None
        self._bout: torch.Tensor | None = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self._attn_block(hidden_states)
        hidden_states = self._mlp_block(hidden_states)
        return hidden_states

    # -- attention sub-block ------------------------------------------------ #
    def _attn_block(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._fused_attn:
            from .triton_vision import triton_vision_attention, triton_vision_out_proj

            residual = hidden_states
            q, k, v, B, L, C, attn = self._fused_qkv(hidden_states)
            attn_out = triton_vision_attention(q, k, v, attn.scale)
            attn_out = attn_out.transpose(1, 2).contiguous().reshape(B, L, C)
            return triton_vision_out_proj(attn_out, self._wout, self._bout, residual)
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        return residual + hidden_states

    def _fused_qkv(self, hidden_states: torch.Tensor):
        """Fused LayerNorm1 + q/k/v projection (fp16) -> q/k/v in (B,H,L,HD)."""
        from .triton_vision import triton_vision_qkv

        assert self._wqkv is not None and self._bqkv is not None
        B, L, C = hidden_states.shape
        attn = cast(SiglipAttention, self.self_attn)
        qkv = triton_vision_qkv(
            hidden_states.reshape(-1, C),
            self.layer_norm1.weight,
            self.layer_norm1.bias,
            self._wqkv,
            self._bqkv,
            self.layer_norm1.eps,
        )
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(B, L, attn.num_heads, attn.head_dim).transpose(1, 2).contiguous()
        k = k.reshape(B, L, attn.num_heads, attn.head_dim).transpose(1, 2).contiguous()
        v = v.reshape(B, L, attn.num_heads, attn.head_dim).transpose(1, 2).contiguous()
        return q, k, v, B, L, C, attn

    # -- MLP sub-block ------------------------------------------------------ #
    def _mlp_block(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._fused_mlp:
            from .triton_vision import triton_vision_mlp

            assert self._w1 is not None and self._b1 is not None and self._w2 is not None and self._b2 is not None
            B, L, C = hidden_states.shape
            return triton_vision_mlp(
                hidden_states.reshape(-1, C),
                self.layer_norm2.weight,
                self.layer_norm2.bias,
                self._w1,
                self._b1,
                self._w2,
                self._b2,
                self.layer_norm2.eps,
            ).reshape(B, L, C)
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class SiglipEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_layers: int,
        intermediate_size: int,
        eps: float = 1e-6,
        dtype=torch.float32,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(hidden_size, num_heads, intermediate_size, eps, dtype) for _ in range(num_layers)]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class SiglipVisionTransformer(nn.Module):
    """f32 vision tower: embeddings + 27 encoder layers + post_layernorm.

    ``forward`` returns the post-norm last hidden state ``(B, 256, 1152)`` (no pooler
    or head).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_layers: int,
        intermediate_size: int,
        image_size: int,
        patch_size: int,
        eps: float = 1e-6,
        dtype=torch.float32,
    ):
        super().__init__()
        self.embeddings = SiglipVisionEmbeddings(hidden_size, image_size, patch_size, dtype=dtype)
        self.encoder = SiglipEncoder(hidden_size, num_heads, num_layers, intermediate_size, eps, dtype)
        self.post_layernorm = nn.LayerNorm(hidden_size, eps=eps, dtype=dtype)

    def set_mlp_dtype(self, dtype: torch.dtype) -> None:
        """Convert every encoder-layer MLP (fc1/fc2 weights + biases) to ``dtype``.

        Only the MLP GEMMs change precision (norms / attention / residuals stay fp32);
        ``SiglipMLP.forward`` casts at the boundary.
        """
        for layer in self.encoder.layers:
            mlp = cast(SiglipMLP, layer.mlp)
            for lin in (mlp.fc1, mlp.fc2):
                lin.weight.data = lin.weight.data.to(dtype)
                if lin.bias is not None:
                    lin.bias.data = lin.bias.data.to(dtype)
            mlp._mlp_dtype = dtype

    def _pack_fused_qkv(self) -> None:
        """Pre-concatenate every layer's q/k/v weights into a fp16 ``[3C, C]`` block."""
        from .triton_vision import pack_qkv

        for layer in self.encoder.layers:
            attn = cast(SiglipAttention, layer.self_attn)
            wqkv, bqkv = pack_qkv(attn.q_proj, attn.k_proj, attn.v_proj)
            layer._wqkv = wqkv
            layer._bqkv = bqkv

    def set_fused_attn(self) -> None:
        """Replace ``layer_norm1 + q/k/v + SDPA`` with the fused Triton kernels.

        q/k/v become a pre-concatenated fp16 ``[3C, C]`` block and out-proj becomes bf16;
        every layer uses the fused ``_attn_block`` path (LayerNorm + qkv GEMM + flash
        attention + bf16 out-proj + fp32 residual). Norms and residuals stay fp32.
        """
        self._pack_fused_qkv()
        for layer in self.encoder.layers:
            attn = cast(SiglipAttention, layer.self_attn)
            layer._wout = attn.out_proj.weight.data.to(torch.bfloat16).contiguous()
            layer._bout = attn.out_proj.bias.data.to(torch.bfloat16).contiguous()
            layer._fused_attn = True

    def set_fused_mlp(self) -> None:
        """Fuse ``layer_norm2 + fc1 + gelu + fc2`` into two bf16 Triton kernels.

        The MLP GEMMs run bf16 but the fc1 output / gelu stay fp32 (the ``[M, F]``
        activation is only rounded to bf16 at the fc2 input), so this is more accurate
        than the eager MLP low-precision tier. The residual is added in the down kernel.
        """
        for layer in self.encoder.layers:
            mlp = cast(SiglipMLP, layer.mlp)
            layer._w1 = mlp.fc1.weight.data.to(torch.bfloat16).contiguous()
            layer._b1 = mlp.fc1.bias.data.to(torch.bfloat16).contiguous()
            layer._w2 = mlp.fc2.weight.data.to(torch.bfloat16).contiguous()
            layer._b2 = mlp.fc2.bias.data.to(torch.bfloat16).contiguous()
            layer._fused_mlp = True

    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run embeddings + the 27 encoder layers, return the pre-post_layernorm state.

        Used by the fused ``post_layernorm + projector`` Triton path (that kernel
        computes the LayerNorm itself); public ``forward`` returns the post_layernorm
        output.
        """
        return self.encoder(self.embeddings(pixel_values))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.encoder(hidden_states)
        return self.post_layernorm(hidden_states)  # [B, 256, 1152]
