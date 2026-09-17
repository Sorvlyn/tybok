# Derived from transformers <https://github.com/huggingface/transformers>. Copyright 2024 Google
# Inc. HuggingFace Inc. team. Licensed under the Apache License, Version 2.0; modified for
# TyBoK in 2026. See the NOTICE file.

"""RoPE for the pi0.5 Gemma decoders (transformers 5.5.4 semantics).

``emb = cat((freqs, freqs))`` with cos/sin cast to the input dtype;
``apply_rotary_pos_emb`` rotates consecutive pairs with ``unsqueeze_dim=1`` on
``[B, H, L, D]``. Must use this Gemma variant, not the smolvla engine's
half-split ``apply_rope``, to stay bit-exact.
"""

from __future__ import annotations

import torch
from torch import nn


class GemmaRotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, rope_theta: float = 10000.0, max_position_embeddings: int = 8192):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len_cached = max_position_embeddings
        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).to(dtype=torch.float32) / head_dim)
        )
        # inv_freq is bf16-rounded (the whole decoder is cast to bf16 after
        # init), so cos/sin are computed from the bf16-rounded frequencies.
        self.register_buffer("inv_freq", inv_freq.to(dtype=torch.bfloat16), persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """position_ids: (B, L). Returns (cos, sin) of x's dtype, shape (B, L, D)."""
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def build_rope_tables(head_dim: int, max_pos: int, device, rope_theta: float = 10000.0):
    """Precomputed cos/sin tables for the Triton kernels.

    Same values as the eager ``GemmaRotaryEmbedding``: inverse frequencies go
    through the bf16 buffer cast, then fp32 ``cos(freq * pos)`` is tabulated
    over ``[0, max_pos)`` x ``[0, head_dim)``. Returns ``(cos, sin)`` of shape
    ``[max_pos, head_dim]`` f32 on ``device``.
    """
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).to(dtype=torch.float32) / head_dim)
    )
    inv_freq = inv_freq.to(dtype=torch.bfloat16).to(dtype=torch.float32)  # bf16-rounded freqs
    pos = torch.arange(max_pos, dtype=torch.float32, device=device)
    freqs = pos[:, None] * inv_freq[None, :].to(device)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate q/k ([B, H, L, D]) with cos/sin ([B, L, D]) -- transformers convention."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
