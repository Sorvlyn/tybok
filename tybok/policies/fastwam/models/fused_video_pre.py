"""video DiT non-block component fusion (``--video-pre-fused``) -- VideoPreRunner.

**Purely "exact" optimization** (bit-identical to the production chain): keep
request-invariant quantities device-resident / cached by key:
`freqs` (RoPE complex frequency table) and the prebuilt fp64 sinusoidal frequency
table stay resident in memory; `grid_sizes`, `video_mask` /
`mot_attention_mask` are cached by key; and the `--cu-fused-adit` entry's repeated
`.to().contiguous()` traversal over already-contiguous bf16 KV is removed.

Numerics: all bit-identical to production (`t_emb`/`tokens`/`t`/`t_mod`/`context`
and the two masks).
"""

from __future__ import annotations

import torch


class VideoPreRunner:
    """Pre-cache + exact time embedding attached to the video expert / MoT."""

    def __init__(self, model) -> None:
        self.model = model
        self.ve = model.video_expert
        self.mot = model.mot
        half = int(self.ve.freq_dim) // 2
        if int(self.ve.freq_dim) % 2 != 0:
            raise ValueError(f"freq_dim must be even, got {self.ve.freq_dim}")
        self._half = half
        # bit-identical to the wan_base.sinusoidal_embedding_1d frequency table (fp64)
        self._freq_table_cpu = torch.pow(10000.0, -torch.arange(half, dtype=torch.float64) / half)
        self._ftab_by_device: dict[str, torch.Tensor] = {}
        self._freqs_by_device: dict[str, torch.Tensor] = {}
        self._grids: dict[tuple, torch.Tensor] = {}
        self._vmask: dict[tuple, torch.Tensor] = {}
        self._mmask: dict[tuple, torch.Tensor] = {}
        self._installed = False

    # ------------------------------------------------------------------ #
    def install(self) -> "VideoPreRunner":
        self.ve._pre_fused = self
        self.mot._pre_fused = self
        self._installed = True
        return self

    def uninstall(self) -> None:
        self.ve._pre_fused = None
        self.mot._pre_fused = None
        self._installed = False

    # ------------------------------------------------------------------ #
    # device-resident constants
    # ------------------------------------------------------------------ #
    def _freq_table(self, device) -> torch.Tensor:
        key = str(device)
        t = self._ftab_by_device.get(key)
        if t is None:
            t = self._freq_table_cpu.to(device)
            self._ftab_by_device[key] = t
        return t

    def device_freqs(self, device) -> torch.Tensor:
        """RoPE frequency table resident in memory (replaces production's per-call `self.freqs.to(device)`)."""
        key = str(device)
        t = self._freqs_by_device.get(key)
        if t is None:
            # no-op when on the same device; one copy like production when crossing devices; not repeated after
            t = self.ve.freqs.to(device)
            self._freqs_by_device[key] = t
        return t

    def grid_sizes(self, f: int, h: int, w: int, batch: int, device) -> torch.Tensor:
        key = (f, h, w, int(batch), str(device))
        t = self._grids.get(key)
        if t is None:
            t = torch.tensor([[f, h, w]] * int(batch), dtype=torch.long, device=device)
            self._grids[key] = t
        return t

    # ------------------------------------------------------------------ #
    # time embedding (exact)
    # ------------------------------------------------------------------ #
    def time_embed(self, timestep: torch.Tensor, batch_size: int, frames: int, tokens_per_frame: int) -> torch.Tensor:
        """Equivalent to `sinusoidal_embedding_1d(freq_dim, token_timesteps.reshape(-1)).float()`.

        token_timesteps = ones(B, T, TPF) * timestep (first frame set to 0).
        """
        model_dtype = self.ve.patch_embedding.weight.dtype
        ts = timestep.to(dtype=model_dtype).view(batch_size, 1, 1)
        token_ts = torch.ones((batch_size, frames, tokens_per_frame), dtype=model_dtype, device=timestep.device) * ts
        token_ts[:, 0, :] = 0
        pos = token_ts.reshape(-1).double().unsqueeze(1)
        v = pos * self._freq_table(timestep.device).unsqueeze(0)
        return torch.cat([torch.cos(v), torch.sin(v)], dim=1).float()

    # ------------------------------------------------------------------ #
    # mask (config constants)
    # ------------------------------------------------------------------ #
    def video_mask(self, mode: str, video_seq_len: int, tokens_per_frame: int, device) -> torch.Tensor:
        key = (mode, int(video_seq_len), int(tokens_per_frame), str(device))
        m = self._vmask.get(key)
        if m is None:
            m = self.ve._build_video_to_video_mask_raw(video_seq_len, tokens_per_frame, device)
            self._vmask[key] = m
        return m

    def mot_mask(self, video_seq_len: int, action_seq_len: int, tokens_per_frame: int, device) -> torch.Tensor:
        key = (int(video_seq_len), int(action_seq_len), int(tokens_per_frame), str(device))
        m = self._mmask.get(key)
        if m is None:
            video_mask = self.video_mask(self.ve.video_attention_mask_mode, video_seq_len, tokens_per_frame, device)
            m = self.mot._build_mot_attention_mask_raw(
                video_seq_len, action_seq_len, tokens_per_frame, video_mask, device
            )
            self._mmask[key] = m
        return m
