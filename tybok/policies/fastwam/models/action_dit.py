"""FastWAM action-expert DiT.

hidden 1024, attention dim num_heads*attn_head_dim (projected q/k/v); cross-attn
consumes the text context, sinusoidal time embedding + Wan AdaLN modulation, Linear
action head. State keys ``mot.mixtures.action.*`` map one-to-one to this module.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ..config import FastWAMDiTSpec
from .dit_block import FastWAMAttentionBlock
from .wan_base import precompute_freqs_cis, sinusoidal_embedding_1d


class ActionDiT(nn.Module):
    def __init__(self, spec: FastWAMDiTSpec):
        super().__init__()
        hidden_dim = spec.hidden_dim
        action_dim = spec.action_dim
        ffn_dim = spec.ffn_dim
        num_heads = spec.num_heads
        attn_head_dim = spec.attn_head_dim
        num_layers = spec.num_layers
        eps = spec.eps
        if attn_head_dim % 2 != 0:
            raise ValueError(f"`attn_head_dim` must be even for RoPE, got {attn_head_dim}")

        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.ffn_dim = ffn_dim
        self.text_dim = spec.text_dim
        self.freq_dim = spec.freq_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.num_layers = num_layers
        self.eps = eps

        self.action_encoder = nn.Linear(action_dim, hidden_dim)
        self.text_embedding = nn.Sequential(
            nn.Linear(spec.text_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(spec.freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        self.blocks = nn.ModuleList(
            [
                FastWAMAttentionBlock(
                    hidden_dim=hidden_dim,
                    attn_head_dim=attn_head_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    eps=eps,
                    fp32_attention=spec.fp32_attention,
                )
                for _ in range(num_layers)
            ]
        )
        self.head = nn.Linear(hidden_dim, action_dim)
        # dense rotary table (plain attribute, not registered as a buffer)
        self.freqs = precompute_freqs_cis(attn_head_dim, end=1024)
        # device-side memo (constant table; unpinned H2D is **not allowed** during CUDA
        # graph capture). Values are bitwise-identical.
        self._freqs_dev_cache: dict = {}
        self.fp32_attention = spec.fp32_attention
        self.use_gradient_checkpointing = spec.use_gradient_checkpointing

    # ------------------------------------------------------------------ #
    def precompute_action_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Precompute the context-side work that is invariant across action denoise steps
        (once per chunk).

        The text embedding and each layer's cross k/v (including qk-norm) depend only on
        the context, so they are hoisted out of the denoise loop; this is bitwise-
        identical to recomputing them step by step. Returns
        ``{"emb", "kv", "mask"}`` (``kv`` is a per-layer list of (k, v))."""
        model_dtype = self.action_encoder.weight.dtype
        context = context.to(dtype=model_dtype)
        _fused_ctx = getattr(self, "_fused_ctx_pre", None)
        if _fused_ctx is not None:
            return _fused_ctx(context, context_mask)
        context_emb = self.text_embedding(context)
        kv: list[tuple[torch.Tensor, torch.Tensor]] = []
        for block in self.blocks:
            attn = block.cross_attn
            n, d = attn.num_heads, attn.head_dim
            k, v = attn.kv_proj(context_emb)
            k = attn.norm_k(k).view(context.shape[0], -1, n * d)
            v = v.view(context.shape[0], -1, n * d)
            kv.append((k, v))
        return {"emb": context_emb, "kv": kv, "mask": context_mask}

    def pre_dit(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        context_emb: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Embed noisy action tokens + text/time conditioning.

        Passing ``context_emb`` (from ``precompute_action_context``) skips recomputing
        ``text_embedding(context)`` without changing the values.
        """
        if action_tokens.ndim != 3 or action_tokens.shape[2] != self.action_dim:
            raise ValueError(
                f"`action_tokens` must be 3D [B, T, action_dim={self.action_dim}], got {tuple(action_tokens.shape)}"
            )
        if context.ndim != 3:
            raise ValueError(f"`context` must be 3D [B, L, D], got shape {tuple(context.shape)}")
        batch_size = action_tokens.shape[0]
        if context.shape[0] != batch_size:
            raise ValueError(
                f"Batch mismatch between action tokens and text context: {batch_size} vs {context.shape[0]}"
            )
        if timestep.ndim == 0:
            timestep = timestep.unsqueeze(0)
        if timestep.ndim != 1 or timestep.shape[0] not in (1, batch_size):
            raise ValueError(f"`timestep` must be 1D [1] or [B], got {tuple(timestep.shape)}")
        if context_mask is None:
            context_mask = torch.ones((batch_size, context.shape[1]), dtype=torch.bool, device=context.device)
        elif context_mask.ndim != 2 or context_mask.shape != context.shape[:2]:
            raise ValueError("`context_mask` must be 2D [B, L] matching `context`")

        seq_len = action_tokens.shape[1]
        if seq_len > self.freqs.shape[0]:
            raise ValueError(f"Action token length {seq_len} exceeds RoPE cache {self.freqs.shape[0]}.")

        model_dtype = self.action_encoder.weight.dtype
        action_tokens = action_tokens.to(dtype=model_dtype)
        _fused_t = getattr(self, "_fused_tmod", None)
        if _fused_t is not None:
            # fused time path: sinusoidal + 3xFP8Linear + 2xSiLU packed into 4 small launches.
            # t is only a placeholder (post_dit doesn't use t, only tokens).
            t_mod = _fused_t(timestep)
            t = t_mod.reshape(t_mod.shape[0], -1)
        else:
            t_emb = sinusoidal_embedding_1d(self.freq_dim, timestep).to(dtype=model_dtype)
            t = self.time_embedding(t_emb)
            t_mod = self.time_projection(t).unflatten(1, (6, self.hidden_dim))

        tokens = self.action_encoder(action_tokens)
        if context_emb is None:
            context = context.to(dtype=model_dtype)
            context_emb = self.text_embedding(context)
        context_attn_mask = context_mask.unsqueeze(1).expand(-1, seq_len, -1)
        freqs = self._device_freqs(seq_len, tokens.device)

        return {
            "tokens": tokens,
            "freqs": freqs,
            "t": t,
            "t_mod": t_mod,
            "context": context_emb,
            "context_mask": context_attn_mask,
            "meta": {"batch_size": batch_size, "seq_len": seq_len},
        }

    def post_dit(self, tokens: torch.Tensor, pre_state: dict[str, Any]) -> torch.Tensor:
        return self.head(tokens)

    def _device_freqs(self, seq_len: int, device) -> torch.Tensor:
        """Device-side cache of the constant dense RoPE table (equivalent to `self.freqs[:S].view(S,1,-1).to(device)`)."""
        key = (int(seq_len), str(device))
        t = self._freqs_dev_cache.get(key)
        if t is None:
            t = self.freqs[:seq_len].view(seq_len, 1, -1).to(device)
            self._freqs_dev_cache[key] = t
        return t
