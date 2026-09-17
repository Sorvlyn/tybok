# Derived from LeRobot <https://github.com/huggingface/lerobot> fastwam/wan. Copyright 2024 The
# HuggingFace Inc. team; portions Copyright 2024-2025 The Alibaba Wan Team Authors. Licensed
# under the Apache License, Version 2.0; modified for TyBoK in 2026. See the NOTICE file.

"""Wan / FastWAM shared **math primitives** (pure functions + norm modules).

Holds only low-level primitives reused across multiple model files, no DiT layer
structure: sinusoidal / rotary helpers, fp32-statistics norms, and the Linear input
dtype convention helper ``_linear_input`` (bf16 Linear cast; FP8Linear passes
fp32/bf16 through as-is -- quantization is already fused inside ``FP8Linear.forward``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional


# --------------------------------------------------------------------------- #
# sinusoidal / rotary helpers
# --------------------------------------------------------------------------- #
def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    """Wan time embedding: fp64 cos/sin table over ``position``."""
    if dim % 2 != 0:
        raise ValueError(f"dim must be even, got {dim}.")
    half = dim // 2
    position = position.type(torch.float64)
    # arange is built directly on position's device: unpinned H2D is not allowed during CUDA graph capture.
    freq = torch.pow(10000, -torch.arange(half, device=position.device, dtype=torch.float64) / half)
    sinusoid = torch.outer(position, freq)
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


def rope_params(max_seq_len: int, dim: int, theta: float = 10000) -> torch.Tensor:
    """Complex polar rotary frequencies, shape (max_seq_len, dim // 2)."""
    if dim % 2 != 0:
        raise ValueError(f"dim must be even, got {dim}.")
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)),
    )
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0) -> torch.Tensor:
    return rope_params(end, dim, theta)


def rope_apply(x: torch.Tensor, grid_sizes: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Wan video rope over a 3D (f, h, w) token grid.

    ``x`` is [B, S, n, d] (n heads). The rotary table is split into
    f / h / w groups and each spatial index of every sample rotates its own
    group. fp64 complex math, fp32 output.

    ``grid_sizes.tolist()`` is a D2H sync, so this cannot be used inside CUDA graph
    capture; use :func:`rope_apply_grid` there.
    """
    return _rope_apply_grids(x, grid_sizes.tolist(), freqs)


def rope_apply_grid(x: torch.Tensor, grid: tuple[int, int, int], freqs: torch.Tensor) -> torch.Tensor:
    """``rope_apply`` with an explicit ``(f, h, w)`` tuple (no ``.tolist()`` D2H sync).

    CUDA-graph-safe variant: the video token grid is a constant, so it can be passed in
    from Python instead of read back from the device tensor.
    """
    return _rope_apply_grids(x, [(int(grid[0]), int(grid[1]), int(grid[2]))] * x.size(0), freqs)


def _rope_apply_grids(x: torch.Tensor, grids: list, freqs: torch.Tensor) -> torch.Tensor:
    n, c = x.size(2), x.size(3) // 2
    # split freqs into f / h / w groups
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    output = []
    for i, (f, h, w) in enumerate(grids):
        seq_len = f * h * w
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        freqs_i = torch.cat(
            [
                freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).float()


def apply_dense_rope(x: torch.Tensor, freqs: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Dense (non-grid) rotary embedding used by the action expert.

    ``freqs`` is the complex polar table sliced to the sequence length
    (S, 1, d/2); rotation happens per adjacent pair in fp32.
    """
    b, s, _ = x.shape
    x = x.view(b, s, num_heads, -1)
    x_out = torch.view_as_complex(x.to(torch.float32).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2))
    freqs = freqs.to(torch.complex64)
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


# --------------------------------------------------------------------------- #
# norms
# --------------------------------------------------------------------------- #
class WanRMSNorm(nn.Module):
    """RMS norm (no mean removal): fp32 statistics, cast back to the input dtype after computing."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS over the last dim, computed in fp32, cast back to x's dtype
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):
    """Wan LayerNorm: fp32 statistics, cast back to the input dtype after computing (non-affine by default)."""

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # fp32 statistics, cast back
        return super().forward(x.float()).type_as(x)


def _wan_layer_norm(norm: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Apply a Wan layer norm with fp32 weights/bias."""
    if isinstance(norm, WanLayerNorm) and norm.weight is not None:
        weight = norm.weight.float()
        bias = norm.bias.float() if norm.bias is not None else None
        return functional.layer_norm(x.float(), norm.normalized_shape, weight, bias, norm.eps).to(dtype=x.dtype)
    return norm(x)


def _linear_input(linear: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    """Match the input dtype semantics of the reference torch Linear.

    bf16 weights: cast to the weight dtype (under autocast bf16 the reference rounds
    the fp32 SDPA output back to bf16 before the GEMM, which is the reference's own
    rounding point).

    fp8 weights (``FP8Linear``): **no cast to the weight dtype**. Quantization + GEMM
    are already fused inside ``FP8Linear.forward`` (per-token ``fp32->fp8`` RNE direct
    quantization, a single bit-level rounding); casting to fp8 here would first hard-
    quantize at the full-scale 448 (double quantization), and casting to bf16 would
    lose a round of bf16 precision first. The upstream **fp32 SDPA output of
    o/cross-o keeps full fp32 precision** going into forward, where it is quantized
    once to the fp8 grid (high-precision tier).
    """
    weight_dtype = linear.weight.dtype
    if weight_dtype in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2fnuz,
    ):
        return x  # FP8Linear: fp32/bf16 passed through as-is (fp8 input is rejected in forward)
    return x.to(dtype=weight_dtype)
