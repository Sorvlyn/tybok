"""RoPE: ``max_wavelength``-style with base 10_000, half-split rotation, computed
in float32.

Deliberately NOT the ``transformers`` Llama rope, which the SmolVLA checkpoint
was not trained with.
"""

from __future__ import annotations

import torch

# Frequency grid ``max_wavelength ** ((2/d) * arange(d/2))`` depends only on
# (head_dim, device), so it is cached per key and reused across all attention
# layers / denoising steps. Same values as per-call computation (same op, same
# inputs), so caching preserves bit-exactness.
_timescale_cache: dict[tuple[int, str, float], torch.Tensor] = {}

# sin/cos lookup tables over positions ``[0, _TABLE_POSITIONS)``, computed with
# the same division/transcendentals as the per-call ``sin(positions / timescale)``
# on the same device (bit-identical). Positions are bounded by
# ``max_position_embeddings`` (8192 for SmolVLM2); the table has headroom and
# positions are clamped defensively (clamping never touches valid positions;
# masked positions are garbage anyway).
_TABLE_POSITIONS = 16384
_tables_cache: dict[tuple[int, str, float], tuple[torch.Tensor, torch.Tensor]] = {}


def _get_timescale(d_half: int, device: torch.device, max_wavelength: float) -> torch.Tensor:
    key = (d_half, str(device), max_wavelength)
    ts = _timescale_cache.get(key)
    if ts is None:
        freq_exponents = (2.0 / (d_half * 2)) * torch.arange(d_half, dtype=torch.float32, device=device)
        ts = max_wavelength**freq_exponents
        _timescale_cache[key] = ts
    return ts


def _get_rope_tables(d_half: int, device: torch.device, max_wavelength: float) -> tuple[torch.Tensor, torch.Tensor]:
    """sin/cos tables over the position range; bit-identical to per-call computation."""
    key = (d_half, str(device), max_wavelength)
    tables = _tables_cache.get(key)
    if tables is None:
        timescale = _get_timescale(d_half, device, max_wavelength)
        pos = torch.arange(_TABLE_POSITIONS, dtype=torch.float32, device=device)
        radians = pos[:, None] / timescale[None, :]
        tables = (torch.sin(radians), torch.cos(radians))
        _tables_cache[key] = tables
    return tables


def apply_rope(x: torch.Tensor, positions: torch.Tensor, max_wavelength: float = 10_000.0) -> torch.Tensor:
    """Apply RoPE positions [B, L] to x [B, L, H, D].

    Half of the head dim is rotated with cos/sin at ``position / base**(2i/d)``.
    """
    d_half = x.shape[-1] // 2
    device = x.device
    dtype = x.dtype
    x = x.to(torch.float32)

    # sin/cos at ``position / timescale``, gathered from a precomputed table
    # (bit-identical to the per-call computation).
    sin_tab, cos_tab = _get_rope_tables(d_half, device, max_wavelength)
    p = positions.clamp(0, _TABLE_POSITIONS - 1)
    sin = sin_tab[p][..., None, :]  # [B, L] -> [B, L, d_half] -> [B, L, 1, d_half]
    cos = cos_tab[p][..., None, :]

    x1, x2 = x.split(d_half, dim=-1)
    res = torch.empty_like(x)
    res[..., :d_half] = x1 * cos - x2 * sin
    res[..., d_half:] = x2 * cos + x1 * sin

    return res.to(dtype)
