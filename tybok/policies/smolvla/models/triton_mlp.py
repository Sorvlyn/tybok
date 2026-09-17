"""Triton fused expert MLP norm+gate/up (``--tl-fused-expert``).

One kernel replaces the denoising chain
``post_attention_layernorm -> gate_proj -> silu -> up_proj``:

- RMSNorm is fused via the linearity trick: the norm weight is folded into the
  dot input (bf16, like the eager norm output) and the per-row rstd scale is
  applied to the accumulated fp32 gate/up results afterwards.
- ``silu(gate) * up`` is computed in-kernel, so only the ``[M, intermediate]``
  activation is written; the ``down_proj`` GEMM stays on cuBLAS.

The kernel is parallelised over the intermediate dim and the (tiny, M=50)
sequence dim (2-D grid, one program per ``BLOCK_M x BLOCK_N`` tile). The
variance is per-row over the full K dim (each program's K loop covers all of
K), so the M split is exact, and ``sum_sq`` is accumulated in the same pass as
the dots.

Numerics drift from the eager chain by design (bf16 MMA accumulation order and
the in-kernel sigmoid differ from cuBLAS + torch), so this is gated behind
``--tl-fused-expert``; the eager/bit-exact path is untouched.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _norm_gate_up_kernel(
    x_ptr,
    norm_w_ptr,
    gate_w_ptr,
    up_w_ptr,
    out_ptr,
    M,
    K,
    N,
    EPS,
    stride_xm,
    stride_xk,
    stride_gwn,
    stride_gwk,
    stride_uwn,
    stride_uwk,
    stride_nw,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    n0 = pid_n * BLOCK_N
    m0 = pid_m * BLOCK_M
    offs_m = m0 + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    n_mask = (n0 + offs_n) < N

    # per-row rstd over the raw x (RMSNorm variance, fp32), accumulated in the
    # same K loop as the dots (x loaded once; the norm weight is folded into
    # the dot input, and rstd is applied to the fp32 results afterwards)
    sum_sq = tl.zeros((BLOCK_M,), tl.float32)
    gate = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    up = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k0 in range(0, K, BLOCK_K):
        km = (k0 + offs_k) < K
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + (k0 + offs_k)[None, :] * stride_xk,
            mask=m_mask[:, None] & km[None, :],
            other=0.0,
        )
        xf = x.to(tl.float32)
        sum_sq += tl.sum(xf * xf, 1)
        wn = tl.load(norm_w_ptr + (k0 + offs_k), mask=km, other=0.0)
        a_w = (xf * wn[None, :]).to(tl.bfloat16)
        wg = tl.load(
            gate_w_ptr + (n0 + offs_n)[:, None] * stride_gwn + (k0 + offs_k)[None, :] * stride_gwk,
            mask=n_mask[:, None] & km[None, :],
            other=0.0,
        )
        gate = tl.dot(a_w, tl.trans(wg), acc=gate)
        wu = tl.load(
            up_w_ptr + (n0 + offs_n)[:, None] * stride_uwn + (k0 + offs_k)[None, :] * stride_uwk,
            mask=n_mask[:, None] & km[None, :],
            other=0.0,
        )
        up = tl.dot(a_w, tl.trans(wu), acc=up)

    rstd = tl.rsqrt(sum_sq / K + EPS)
    gate = gate * rstd[:, None]
    up = up * rstd[:, None]
    act = gate * (1.0 / (1.0 + tl.exp(-gate))) * up  # silu(gate) * up
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + (n0 + offs_n)[None, :] * stride_on,
        act.to(tl.bfloat16),
        mask=m_mask[:, None] & n_mask[None, :],
    )


def triton_norm_gate_up(x, norm_w, eps, gate_w, up_w, out: torch.Tensor | None = None) -> torch.Tensor:
    """Fused ``RMSNorm -> gate_proj -> silu(gate)*up``.

    Args:
        x: ``[M, K]`` bf16, the post-attention residual output (pre-norm).
        norm_w: ``[K]`` bf16 RMSNorm weight; eps: its epsilon.
        gate_w/up_w: ``[N, K]`` bf16 MLP gate/up projection weights.

    Returns ``[M, N]`` bf16 ``silu(gate) * up`` (the down_proj input), same
    layout as the eager chain.
    """
    M, K = x.shape
    N = gate_w.shape[0]
    if out is None:
        out = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    _norm_gate_up_kernel[(triton.cdiv(N, 64), triton.cdiv(M, 32))](
        x,
        norm_w,
        gate_w,
        up_w,
        out,
        M,
        K,
        N,
        float(eps),
        x.stride(0),
        x.stride(1),
        gate_w.stride(0),
        gate_w.stride(1),
        up_w.stride(0),
        up_w.stride(1),
        norm_w.stride(0),
        out.stride(0),
        out.stride(1),
        BLOCK_M=32,
        BLOCK_N=64,
        BLOCK_K=64,
        num_warps=4,
    )
    return out
