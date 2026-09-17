"""Action expert: a smaller decoder interleaved with the VLM.

``attention_mode="cross_attn"`` with ``self_attn_every_n_layers=2``:

- even layer indexes are *self-attention* over ``[prefix KV cache; own suffix
  KV]``
- odd layer indexes are *cross-attention* over the VLM prefix KV only; their
  ``k_proj``/``v_proj`` project the 320-dim VLM prefix K/V, hence the
  non-square ``(320, 320)`` weights

The fp32 cross-attention k/v projections are a checkpoint quirk that must be
preserved for numerical parity (weight layout ``lm_expert.layers.{i}.*``).
"""

from __future__ import annotations

import torch
from torch import nn

from .layers import MLP, RMSNorm


class ExpertAttention(nn.Module):
    def __init__(self, config, *, cross_attn: bool, vlm_kv_dim: int, expert_hidden: int):
        super().__init__()
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = config.head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        # q always comes from the expert hidden state
        self.q_proj = nn.Linear(expert_hidden, num_heads * head_dim, bias=False)
        if cross_attn:
            # Project the VLM prefix K/V (vlm_kv_dim) to the expert kv size.
            # fp32 for numerical parity with the reference.
            self.k_proj = nn.Linear(vlm_kv_dim, num_kv_heads * head_dim, bias=False, dtype=torch.float32)
            self.v_proj = nn.Linear(vlm_kv_dim, num_kv_heads * head_dim, bias=False, dtype=torch.float32)
        else:
            self.k_proj = nn.Linear(expert_hidden, num_kv_heads * head_dim, bias=False)
            self.v_proj = nn.Linear(expert_hidden, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, expert_hidden, bias=False)


class ExpertDecoderLayer(nn.Module):
    def __init__(self, config, *, cross_attn: bool, vlm_kv_dim: int, expert_hidden: int):
        super().__init__()
        self.self_attn = ExpertAttention(
            config, cross_attn=cross_attn, vlm_kv_dim=vlm_kv_dim, expert_hidden=expert_hidden
        )
        self.mlp = MLP(expert_hidden, config.intermediate_size, activation=config.hidden_act)
        self.input_layernorm = RMSNorm(expert_hidden, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(expert_hidden, eps=config.rms_norm_eps)


class ExpertModel(nn.Module):
    """``lm_expert``: num_expert_layers decoder layers, hidden size scaled by
    ``expert_width_multiplier``."""

    def __init__(self, config, text_config, expert_hidden_size: int):
        super().__init__()
        self.config = config
        num_layers = config.n_expert_layers
        vlm_kv_dim = text_config.num_key_value_heads * text_config.head_dim
        self.layers = nn.ModuleList(
            [
                ExpertDecoderLayer(
                    text_config,
                    cross_attn=(
                        "cross" in config.attention_mode
                        and not (config.self_attn_every_n_layers > 0 and i % config.self_attn_every_n_layers == 0)
                    ),
                    vlm_kv_dim=vlm_kv_dim,
                    expert_hidden=expert_hidden_size,
                )
                for i in range(num_layers)
            ]
        )
        self.norm = RMSNorm(expert_hidden_size, eps=text_config.rms_norm_eps)
        # Keep ``embed_tokens`` absent (reference sets it to None) so the loader
        # ignores any spurious key.
