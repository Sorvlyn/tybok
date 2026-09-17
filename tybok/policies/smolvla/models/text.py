"""Gemma/Llama-style text decoder of the SmolVLM backbone.

Only the layer norms, q/k/v/o projections and MLPs are used: attention runs in
``vlm_with_expert.py``, not in the ``transformers`` layers. Weight names match
the checkpoint 1:1::

    text_model.embed_tokens.weight
    text_model.layers.{i}.input_layernorm.weight
    text_model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
    text_model.layers.{i}.post_attention_layernorm.weight
    text_model.layers.{i}.mlp.{gate,up,down}_proj.weight
    text_model.norm.weight
"""

from __future__ import annotations

import torch
from torch import nn

from .layers import MLP, RMSNorm


class TextAttention(nn.Module):
    """Holds the q/k/v/o projections; attention itself runs in
    ``vlm_with_expert.py`` so it can be shared between the VLM prefix prefill
    and the action expert's cross/self attention."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * config.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * config.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * config.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * config.head_dim, config.hidden_size, bias=config.attention_bias
        )


class TextDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = TextAttention(config)
        self.mlp = MLP(config.hidden_size, config.intermediate_size, activation=config.hidden_act)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Provided for completeness; the inference path drives the layers
        # manually (shared attention between VLM and expert).
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # (self-attention omitted on purpose)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class TextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([TextDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.norm(hidden_states)
