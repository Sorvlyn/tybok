"""Gemma decoder stack shared by the pi0.5 VLM and action expert.

- ``inputs_embeds`` are cast to bf16 when the first layer is bf16 (the prefix
  embeddings are fp32 after the type-promoting ``torch.cat``, the suffix
  embeddings are fp32 from the fp32 projections)
- one shared RoPE ``cos/sin`` for all layers
- per-layer forward with an optional shared prefix KV (cross-attention over the
  VLM prefix for the expert) and an optional ``adarms_cond`` (AdaRMS)
- final norm

When ``use_cache`` is set (prefix prefill), each layer's post-RoPE K/V is
collected into ``past_key_values``.
"""

from __future__ import annotations

import torch
from torch import nn

from .layers import GemmaDecoderLayer, GemmaRMSNorm
from .rope import GemmaRotaryEmbedding


class GemmaDecoder(nn.Module):
    def __init__(
        self,
        width: int,
        depth: int,
        mlp_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        *,
        use_adarms: bool,
        vocab_size: int | None = None,
        dtype=torch.bfloat16,
    ):
        super().__init__()
        self.width = width
        self.depth = depth
        self.use_adarms = use_adarms
        self.rotary_emb = GemmaRotaryEmbedding(head_dim)
        self.layers = nn.ModuleList(
            [
                GemmaDecoderLayer(num_heads, num_kv_heads, head_dim, width, mlp_dim, use_adarms, dtype=dtype)
                for _ in range(depth)
            ]
        )
        self.norm = GemmaRMSNorm(width, eps=1e-6, cond_dim=width if use_adarms else None)
        # When set, ``forward`` skips the final AdaRMS norm and returns the
        # pre-norm hidden -- the flow-matching tail fuses the norm into
        # ``action_out_proj`` (``fuse_final_tail``). Only the action-expert
        # decoder sets this; the VLM decoder keeps its norm.
        self.skip_final_norm = False
        # Only the VLM has an embedding table (the expert's is None). The VLM
        # table is fed by the checkpoint's tied lm_head.
        self.embed_tokens = None
        if vocab_size is not None:
            self.embed_tokens = nn.Embedding(vocab_size, width, dtype=dtype)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        prefix_kv: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        adarms_cond: torch.Tensor | None = None,
        adarms_mods: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        triton_prefix_pad: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Returns (last_hidden_state, past_key_values).

        ``past_key_values`` is a per-layer list of (k, v) [B, Hkv, L, D], filled
        only when ``use_cache=True`` (prefix prefill); on the denoising path
        ``prefix_kv`` carries the VLM prefix cache and the list is empty.

        ``adarms_cond`` is the AdaRMS conditioning vector (time embedding);
        ``adarms_mods`` instead carries the per-layer precomputed ``dense(cond)``
        outputs ``[(mod_in, mod_post), ...]``. ``triton_prefix_pad`` is the
        prefix padding mask.
        """
        hidden_states = inputs_embeds
        if self.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.bfloat16)

        # The fused (triton) denoising branch gathers its own cos/sin inside
        # ``_fused_gemm_rope_kernel`` and never consumes the rotary embeddings --
        # skip their per-step computation when every layer is on that branch
        # (VLM prefill keeps prefix_kv=None, so it still computes them).
        full_fused_attn = (
            prefix_kv is not None
            and adarms_mods is not None
            and triton_prefix_pad is not None
            and all(l.fused_attn for l in self.layers)
        )
        if full_fused_attn:
            position_embeddings = None
        else:
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, layer in enumerate(self.layers):
            kv = prefix_kv[i] if prefix_kv is not None else None
            hidden_states, key_states, value_states = layer(
                hidden_states,
                attention_mask,
                position_ids,
                kv,
                position_embeddings,
                adarms_cond,
                adarms_mods[i] if adarms_mods is not None else None,
                triton_prefix_pad if triton_prefix_pad is not None else None,
            )
            if use_cache:
                past_key_values.append((key_states, value_states))

        if adarms_mods is not None and getattr(self, "skip_final_norm", False):
            # final AdaRMS norm is fused into the flow tail
            pass
        else:
            hidden_states, _ = self.norm(
                hidden_states, adarms_cond, adarms_mods[-1][0] if adarms_mods is not None else None
            )
        return hidden_states, past_key_values
