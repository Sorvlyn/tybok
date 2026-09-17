"""Triton fused expert MLP norm+gate/up for pi0.5.

One kernel replaces the denoising chain
``post_attention_layernorm -> gate_proj -> gelu_tanh -> up_proj``:

- the AdaRMS norm runs in two passes: the variance pass (fp32), then the
  normed input ``round_bf16(x * rsqrt(var+eps) * (1+scale) + shift)`` built
  exactly like the eager norm (single rounding point, shift included);
- ``gelu_tanh(gate) * up`` is computed in-kernel (``F.gelu(approximate="tanh")``
  semantics via ``tanhf``), so only the ``[M, intermediate]`` activation is
  written (the ``down_proj`` GEMM stays on cuBLAS).

Parallelised over the intermediate dimension AND the (tiny) sequence dim (2-D
grid, one program per ``BLOCK_M x BLOCK_N`` tile). The variance is per-row over
the FULL K dim (each program's K loop covers all of K), so the split is exact.

Numerics drift from the eager chain by design (bf16 MMA accumulation order and
the in-kernel tanhf differ from cuBLAS + torch), so this is gated behind a tier
flag; the eager/bit-exact path is untouched.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _norm_gate_up_kernel(
    x_ptr,          # [M, K] bf16 pre-norm residual
    mod_ptr,        # [3*K] f32 AdaRMS modulation (scale, shift, gate)
    gate_w_ptr,     # [N, K] bf16
    up_w_ptr,       # [N, K] bf16
    out_ptr,        # [M, N] bf16 gelu(gate)*up (down_proj input)
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

    # two-pass AdaRMS norm with the exact eager rounding: pass 1 accumulates
    # the per-row variance (fp32) over the full K; pass 2 builds the normed
    # input ``round_bf16(x * rsqrt(var+eps) * (1+scale) + shift)`` (single
    # rounding point, shift included) and projects gate/up with bf16 dots.
    sum_sq = tl.zeros((BLOCK_M,), tl.float32)
    for k0 in range(0, K, BLOCK_K):
        km = (k0 + offs_k) < K
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + (k0 + offs_k)[None, :] * stride_xk,
            mask=m_mask[:, None] & km[None, :],
            other=0.0,
        )
        xf = x.to(tl.float32)
        sum_sq += tl.sum(xf * xf, 1)
    rstd = tl.rsqrt(sum_sq / K + EPS)

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
        sc = tl.load(mod_ptr + (k0 + offs_k), mask=km, other=0.0)
        sh = tl.load(mod_ptr + K + (k0 + offs_k), mask=km, other=0.0)
        a_w = (xf * rstd[:, None] * (1.0 + sc[None, :]) + sh[None, :]).to(tl.bfloat16)
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
    # F.gelu(x, approximate="tanh") = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3)))
    c = 0.7978845608028654  # sqrt(2/pi)
    t = tl.extra.cuda.libdevice.tanh(c * (gate + 0.044715 * gate * gate * gate))
    act = 0.5 * gate * (1.0 + t) * up
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + (n0 + offs_n)[None, :] * stride_on,
        act.to(tl.bfloat16),
        mask=m_mask[:, None] & n_mask[None, :],
    )


def triton_norm_gate_up(
    x: torch.Tensor,           # [M, K] bf16 pre-norm residual
    modulation: torch.Tensor,  # [3*K] f32 (scale, shift, gate)
    gate_w: torch.Tensor,      # [N, K] bf16
    up_w: torch.Tensor,        # [N, K] bf16
    eps: float,
    out: torch.Tensor | None = None,  # [M, N] bf16
) -> torch.Tensor:
    """Fused AdaRMS norm -> gate/up + gelu_tanh (drift tier)."""
    M, K = x.shape
    N = gate_w.shape[0]
    if out is None:
        out = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    _norm_gate_up_kernel[(triton.cdiv(N, 64), triton.cdiv(M, 32))](
        x,
        modulation,
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
        out.stride(0),
        out.stride(1),
        BLOCK_M=32,
        BLOCK_N=64,
        BLOCK_K=64,
        num_warps=4,
    )
    return out
