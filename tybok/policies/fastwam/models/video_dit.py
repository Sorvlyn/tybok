# Derived from LeRobot <https://github.com/huggingface/lerobot> fastwam/wan. Copyright 2024 The
# HuggingFace Inc. team; portions Copyright 2024-2025 The Alibaba Wan Team Authors. Licensed
# under the Apache License, Version 2.0; modified for TyBoK in 2026. See the NOTICE file.

"""Wan2.2 TI2V video-expert DiT (FastWAM video expert).

Inference interface:
- ``pre_dit``: patchify latents + time/text embed, produces tokens/modulations
- ``build_video_to_video_mask``: first-frame-causal / per-frame-causal masks
- ``post_dit`` / ``unpatchify``: video reconstruction (unused by action inference,
  present only for weight completeness)

State keys ``mot.mixtures.video.*`` map one-to-one to this module.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ..config import FastWAMDiTSpec
from .dit_block import FastWAMAttentionBlock
from .wan_base import WanLayerNorm, rope_params, sinusoidal_embedding_1d


def create_group_causal_attn_mask(
    num_temporal_groups: int, num_query_per_group: int, num_key_per_group: int, mode: str = "causal"
) -> torch.Tensor:
    if mode not in ["causal", "group_diagonal"]:
        raise ValueError(f"`mode` must be 'causal' or 'group_diagonal', got {mode}.")
    query_time_indices = torch.arange(num_temporal_groups).repeat_interleave(num_query_per_group).unsqueeze(1)
    key_time_indices = torch.arange(num_temporal_groups).repeat_interleave(num_key_per_group).unsqueeze(0)
    if mode == "causal":
        attn_mask = query_time_indices >= key_time_indices
    else:
        attn_mask = query_time_indices == key_time_indices
    return attn_mask


class WanHead(nn.Module):
    """Video-expert unpatchify head (Wan ``Head``): modulated output projection."""

    def __init__(self, dim: int, out_dim: int, patch_size: tuple[int, int, int], eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps
        out_dim = patch_size[0] * patch_size[1] * patch_size[2] * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast("cuda", dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            x = self.head(self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2))
        return x


class WanVideoDiT(nn.Module):
    """FastWAM video expert (Wan2.2 TI2V DiT)."""

    def __init__(self, spec: FastWAMDiTSpec):
        super().__init__()
        if spec.attn_head_dim != spec.hidden_dim // spec.num_heads:
            raise ValueError(
                "`attn_head_dim` must match the upstream Wan head dimension "
                f"`hidden_dim // num_heads`; got {spec.attn_head_dim} vs {spec.hidden_dim // spec.num_heads}."
            )
        if spec.has_image_input:
            raise ValueError("FastWAM currently expects Wan2.2 TI2V latents with fused image conditioning.")
        hidden_dim = spec.hidden_dim
        ffn_dim = spec.ffn_dim
        num_heads = spec.num_heads
        num_layers = spec.num_layers
        eps = spec.eps

        self.patch_size = spec.patch_size
        self.in_dim = spec.in_dim
        self.dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.freq_dim = spec.freq_dim
        self.text_dim = spec.text_dim
        self.out_dim = spec.out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.qk_norm = True
        self.cross_attn_norm = True
        self.eps = eps
        self.text_len = 512  # unused by FastWAM (context is fixed length)

        # embeddings
        self.patch_embedding = nn.Conv3d(spec.in_dim, hidden_dim, kernel_size=spec.patch_size, stride=spec.patch_size)
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
                    attn_head_dim=spec.attn_head_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    eps=eps,
                    fp32_attention=spec.fp32_attention,
                )
                for _ in range(num_layers)
            ]
        )
        self.head = WanHead(hidden_dim, spec.out_dim, spec.patch_size, eps)

        # rotary table (plain attribute, not registered as a buffer)
        d = hidden_dim // num_heads
        self.freqs = torch.cat(
            [
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
            ],
            dim=1,
        )

        # config knobs used by the MoT / pre_dit
        self.hidden_dim = hidden_dim
        self.attn_head_dim = spec.attn_head_dim
        self.seperated_timestep = spec.seperated_timestep
        self.fuse_vae_embedding_in_latents = spec.fuse_vae_embedding_in_latents
        self.video_attention_mask_mode = spec.video_attention_mask_mode
        self.action_conditioned = spec.action_conditioned
        self.action_dim = spec.action_dim
        self.action_group_causal_mask_mode = spec.action_group_causal_mask_mode
        self.fp32_attention = spec.fp32_attention
        self.use_gradient_checkpointing = spec.use_gradient_checkpointing

        # video pre-fusion (`--video-pre-fused`); None takes the original path.
        self._pre_fused = None
        # device-side memo (constant RoPE table / grid; values are bitwise-identical).
        # Unpinned H2D is **not allowed** during CUDA graph capture, so these two constant
        # tables must be staged to VRAM outside the graph.
        self._freqs_dev_cache: dict = {}
        self._grid_dev_cache: dict = {}

    # ------------------------------------------------------------------ #
    # masking / patching
    # ------------------------------------------------------------------ #
    def build_video_to_video_mask(
        self, video_seq_len: int, video_tokens_per_frame: int, device: torch.device
    ) -> torch.Tensor:
        pre_fused = getattr(self, "_pre_fused", None)
        if pre_fused is not None:
            # both masks are config constants, cached by (mode, S, TPF, device) (values are bitwise-identical)
            return pre_fused.video_mask(self.video_attention_mask_mode, video_seq_len, video_tokens_per_frame, device)
        return self._build_video_to_video_mask_raw(video_seq_len, video_tokens_per_frame, device)

    def _build_video_to_video_mask_raw(
        self, video_seq_len: int, video_tokens_per_frame: int, device: torch.device
    ) -> torch.Tensor:
        if self.video_attention_mask_mode == "bidirectional":
            return torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)
        if self.video_attention_mask_mode == "per_frame_causal":
            num_video_frames = video_seq_len // video_tokens_per_frame
            frame_causal = torch.tril(torch.ones((num_video_frames, num_video_frames), dtype=torch.bool, device=device))
            return frame_causal.repeat_interleave(video_tokens_per_frame, dim=0).repeat_interleave(
                video_tokens_per_frame, dim=1
            )
        if self.video_attention_mask_mode == "first_frame_causal":
            video_mask = torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)
            first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
            video_mask[:first_frame_tokens, first_frame_tokens:] = False
            return video_mask
        raise ValueError(f"Unsupported video attention mask mode: {self.video_attention_mask_mode}")

    def unpatchify(self, x_tokens: torch.Tensor, grid_sizes: torch.Tensor) -> list[torch.Tensor]:
        """Reconstruct latents from patchified tokens (one tensor per sample)."""
        c = self.out_dim
        out = []
        for u, v in zip(x_tokens, grid_sizes.tolist(), strict=False):
            u = u[: v[0] * v[1] * v[2]].view(*v, *self.patch_size, c)
            u = torch.einsum("fhwpqrc->cfphqwr", u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size, strict=False)])
            out.append(u)
        return out

    # ------------------------------------------------------------------ #
    # pre / post DiT
    # ------------------------------------------------------------------ #
    def pre_dit(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
        fuse_vae_embedding_in_latents: bool = False,
    ) -> dict[str, Any]:
        """Embed latents + text/time conditioning into MoT inputs (includes the fp32-autocast time modulation section)."""
        if x.ndim != 5:
            raise ValueError(f"`latents` must be 5D [B, C, T, H, W], got shape {tuple(x.shape)}")
        if context.ndim != 3:
            raise ValueError(f"`context` must be 3D [B, L, D], got shape {tuple(context.shape)}")
        if timestep.ndim != 1:
            raise ValueError(f"`timestep` must be 1D [B] or [1], got shape {tuple(timestep.shape)}")
        if context_mask is None:
            context_mask = torch.ones((context.shape[0], context.shape[1]), dtype=torch.bool, device=context.device)
        elif context_mask.ndim != 2 or context_mask.shape != context.shape[:2]:
            raise ValueError("`context_mask` must be 2D [B, L] matching `context`")

        batch_size = x.shape[0]
        model_dtype = self.patch_embedding.weight.dtype
        x = x.to(dtype=model_dtype)
        context = context.to(dtype=model_dtype)
        if action is not None:
            raise ValueError("fastWAM action-conditioned video experts are not supported")

        patch_h, patch_w = int(self.patch_size[1]), int(self.patch_size[2])
        tokens_per_frame = (x.shape[3] // patch_h) * (x.shape[4] // patch_w)
        if not (self.seperated_timestep and fuse_vae_embedding_in_latents):
            raise NotImplementedError("FastWAM currently requires separated timesteps with fused VAE latents.")

        pre_fused = getattr(self, "_pre_fused", None)
        with torch.amp.autocast("cuda", dtype=torch.float32):
            if pre_fused is not None:
                # prebuilt fp64 frequency table + removing the per-call arange/pow/outer (bitwise-identical)
                token_t_emb = pre_fused.time_embed(timestep, batch_size, x.shape[2], tokens_per_frame)
            else:
                token_timesteps = torch.ones(
                    (batch_size, x.shape[2], tokens_per_frame),
                    dtype=model_dtype,
                    device=timestep.device,
                ) * timestep.to(dtype=model_dtype).view(batch_size, 1, 1)
                token_timesteps[:, 0, :] = 0
                token_timesteps = token_timesteps.reshape(batch_size, -1)
                token_t_emb = sinusoidal_embedding_1d(self.freq_dim, token_timesteps.reshape(-1)).float()
            t = self.time_embedding(token_t_emb).reshape(batch_size, -1, self.hidden_dim)
            t_mod = self.time_projection(t).unflatten(2, (6, self.hidden_dim))
        t_mod = t_mod.to(dtype=model_dtype)

        x = self.patchify(x)
        f, h, w = x.shape[2:]
        context = self.text_embedding(context)
        seq_len = f * h * w
        context_mask = context_mask.unsqueeze(1).expand(-1, seq_len, -1)

        x_tokens = x.permute(0, 2, 3, 4, 1).reshape(batch_size, seq_len, self.hidden_dim).contiguous()
        if pre_fused is not None:
            # grid_sizes / freqs are constants (freqs stay resident in VRAM, avoiding a per-step H2D)
            grid_sizes = pre_fused.grid_sizes(f, h, w, batch_size, x_tokens.device)
            freqs_table = pre_fused.device_freqs(x_tokens.device)
        else:
            grid_sizes = self._device_grid_sizes(f, h, w, batch_size, x_tokens.device)
            freqs_table = self._device_freqs(x_tokens.device)
        freqs = {"grid_sizes": grid_sizes, "freqs": freqs_table}
        return {
            "tokens": x_tokens,
            "freqs": freqs,
            "t": t,
            "t_mod": t_mod,
            "context": context,
            "context_mask": context_mask,
            "meta": {
                "grid_sizes": grid_sizes,
                "tokens_per_frame": tokens_per_frame,
                "batch_size": batch_size,
            },
        }

    def _device_freqs(self, device) -> torch.Tensor:
        """Device-side cache of the constant RoPE table (equivalent to `self.freqs.to(device)`)."""
        key = str(device)
        t = self._freqs_dev_cache.get(key)
        if t is None:
            t = self.freqs.to(device)
            self._freqs_dev_cache[key] = t
        return t

    def _device_grid_sizes(self, f: int, h: int, w: int, batch: int, device) -> torch.Tensor:
        """Device-side cache of the constant grid_sizes (equivalent to `torch.tensor([[f,h,w]]*B, device=...)`)."""
        key = (int(f), int(h), int(w), int(batch), str(device))
        t = self._grid_dev_cache.get(key)
        if t is None:
            t = torch.tensor([[f, h, w]] * int(batch), dtype=torch.long, device=device)
            self._grid_dev_cache[key] = t
        return t

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        return self.patch_embedding(x)

    def post_dit(self, x_tokens: torch.Tensor, pre_state: dict[str, Any]) -> torch.Tensor:
        x = self.head(x_tokens, pre_state["t"])
        return torch.stack(self.unpatchify(x, pre_state["meta"]["grid_sizes"]))
