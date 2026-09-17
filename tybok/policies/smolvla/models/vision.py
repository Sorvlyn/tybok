"""SmolVLM2 vision encoder (SigLIP variant) + connector.

- Conv2d patch embedding + learned position embeddings (bucketized position-id
  logic for arbitrary resolutions).
- Encoder layer: LayerNorm -> attention (SDPA on bf16 q/k/v) -> residual ->
  LayerNorm -> MLP (gelu_pytorch_tanh) -> residual; then ``post_layernorm``.
- Connector: pixel shuffle (scale_factor=4) + linear projection.

SDPA uses ``scale=head_dim**-0.5`` and an additive mask that is all zeros for
the fully-valid 512x512 deployment inputs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from .layers import VisionMLP


class VisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )
        self.num_patches_per_side = self.image_size // self.patch_size
        self.num_patches = self.num_patches_per_side**2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

    def forward(self, pixel_values: torch.Tensor, patch_attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, _, max_im_h, max_im_w = pixel_values.shape

        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        position_ids = self._position_ids(batch_size, max_im_h, max_im_w, patch_attention_mask, pixel_values)
        embeddings = embeddings + self.position_embedding(position_ids)
        return embeddings

    def _position_ids(self, batch_size, max_im_h, max_im_w, patch_attention_mask, pixel_values) -> torch.Tensor:
        """GPU-native bucketized position ids (no host sync, so CUDA-graph
        capturable); bit-identical to the reference for the masks that occur in
        practice. ``nb`` is clamped to >= 1 so an all-masked image stays finite.
        """
        max_nb_patches_h, max_nb_patches_w = max_im_h // self.patch_size, max_im_w // self.patch_size
        boundaries = torch.arange(
            1 / self.num_patches_per_side, 1.0, 1 / self.num_patches_per_side, device=pixel_values.device
        )
        row_idx = torch.arange(max_nb_patches_h, device=pixel_values.device, dtype=pixel_values.dtype)
        col_idx = torch.arange(max_nb_patches_w, device=pixel_values.device, dtype=pixel_values.dtype)
        nb_h = patch_attention_mask[:, :, 0].sum(dim=1).clamp(min=1)  # [B]
        nb_w = patch_attention_mask[:, 0, :].sum(dim=1).clamp(min=1)  # [B]

        fractional_coords_h = (row_idx[None, :] / nb_h[:, None]) * (1 - 1e-6)
        fractional_coords_w = (col_idx[None, :] / nb_w[:, None]) * (1 - 1e-6)

        bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)  # [B, H]
        bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)  # [B, W]

        position_ids = bucket_coords_h[:, :, None] * self.num_patches_per_side + bucket_coords_w[:, None, :]
        position_ids = position_ids * patch_attention_mask  # invalid patches -> 0 (matches reference)
        return position_ids.reshape(batch_size, -1)


class VisionAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        # ``--tl-fused-vit``: lazy [q|k|v] concatenated projection -- one GEMM instead
        # of three. NOT bit-exact (cuBLAS tiles the concatenated GEMM
        # differently), so it is gated behind ``--tl-fused-vit``.
        self.fused_qkv = False
        self._qkv_weight: torch.Tensor | None = None
        self._qkv_bias: torch.Tensor | None = None
        # Run the out_proj GEMM directly over the SDPA-output layout
        # (``triton_vision_out_proj``), skipping the transpose+contiguous copy.
        # Gated separately from ``fast`` so ``--compile --fast`` keeps the
        # eager copy path.
        self.fused_out_proj = False

    def build_qkv(self) -> None:
        """Pre-build the concatenated projection weights (called before CUDA
        graph capture -- building them lazily inside a capture would allocate
        and hang the capture)."""
        if self._qkv_weight is None:
            self._qkv_weight = torch.cat([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight], dim=0)
            self._qkv_bias = torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias], dim=0)

    def _qkv(self, hidden_states: torch.Tensor):
        """Concatenated q/k/v projection (``--tl-fused-vit``)."""
        w = self._qkv_weight
        b = self._qkv_bias
        if w is None or w.device != hidden_states.device or w.dtype != hidden_states.dtype:
            w = torch.cat([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight], dim=0)
            b = torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias], dim=0)
            self._qkv_weight = w
            self._qkv_bias = b
        qkv = torch.nn.functional.linear(hidden_states, w, b)  # [B, L, 3E]
        return qkv[..., : self.embed_dim], qkv[..., self.embed_dim : 2 * self.embed_dim], qkv[..., 2 * self.embed_dim :]

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None):
        batch_size, seq_length, embed_dim = hidden_states.shape

        if self.fused_qkv:
            queries, keys, values = self._qkv(hidden_states)
        else:
            queries = self.q_proj(hidden_states)
            keys = self.k_proj(hidden_states)
            values = self.v_proj(hidden_states)

        queries = queries.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # The reference model runs the vision encoder with the "sdpa" backend.
        attn_output = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=self.scale,
        )
        if self.fused_qkv and self.fused_out_proj:
            # out_proj reads the SDPA output in its native ``[B, H, L, D]``
            # layout, skipping the transpose+contiguous copy.
            from .triton_vision import triton_vision_out_proj

            return triton_vision_out_proj(attn_output, self.out_proj.weight, self.out_proj.bias)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_length, embed_dim).contiguous()
        attn_output = self.out_proj(attn_output)
        return attn_output


class VisionEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = VisionAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = VisionMLP(config.hidden_size, config.intermediate_size, activation=config.hidden_act)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class VisionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([VisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states, attention_mask)
        return hidden_states


def _prepare_4d_attention_mask(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Turn a bool [B, L] mask into the additive [B, 1, 1, L] mask (0 / -inf)."""
    expanded_mask = mask[:, None, None, :].to(dtype=dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class VisionTransformer(nn.Module):
    """SmolVLMVisionTransformer: embeddings -> encoder -> post_layernorm."""

    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        self.embeddings = VisionEmbeddings(config)
        self.encoder = VisionEncoder(config)
        self.patch_size = config.patch_size
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor, patch_attention_mask: torch.Tensor | None = None):
        batch_size = pixel_values.size(0)
        if patch_attention_mask is None:
            patch_size = self.patch_size
            patch_attention_mask = torch.ones(
                (
                    batch_size,
                    pixel_values.size(2) // patch_size,
                    pixel_values.size(3) // patch_size,
                ),
                dtype=torch.bool,
                device=pixel_values.device,
            )

        hidden_states = self.embeddings(pixel_values=pixel_values, patch_attention_mask=patch_attention_mask)

        patch_attention_mask = patch_attention_mask.view(batch_size, -1)
        attention_mask = _prepare_4d_attention_mask(patch_attention_mask, hidden_states.dtype)

        last_hidden_state = self.encoder(hidden_states, attention_mask=attention_mask)
        last_hidden_state = self.post_layernorm(last_hidden_state)
        return last_hidden_state


class Connector(nn.Module):
    """SmolVLMConnector: pixel shuffle (scale_factor) + linear projection."""

    def __init__(self, config):
        super().__init__()
        self.scale_factor = config.scale_factor
        input_size = config.vision_config.hidden_size * (config.scale_factor**2)
        output_size = config.text_config.hidden_size
        self.modality_projection = nn.Module()
        self.modality_projection.proj = nn.Linear(input_size, output_size, bias=False)

    def pixel_shuffle(self, x: torch.Tensor, scale_factor: int = 2) -> torch.Tensor:
        bsz, seq, embed_dim = x.size()
        height = width = int(seq**0.5)
        x = x.view(bsz, height, width, embed_dim)
        x = x.view(bsz, height, int(width / scale_factor), embed_dim * scale_factor)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(bsz, int(width / scale_factor), int(height / scale_factor), embed_dim * (scale_factor**2))
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(bsz, int(seq / (scale_factor**2)), embed_dim * (scale_factor**2))
        return x

    def forward(self, image_hidden_states: torch.Tensor) -> torch.Tensor:
        image_hidden_states = self.pixel_shuffle(image_hidden_states, self.scale_factor)
        image_hidden_states = self.modality_projection.proj(image_hidden_states)
        return image_hidden_states
