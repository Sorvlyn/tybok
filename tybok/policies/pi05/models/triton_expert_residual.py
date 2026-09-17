"""Fused gated residual for the pi0.5 denoising layers (``--tl-fused-expert``).

The eager gated residual ``residual + out * gate`` materialises three small
kernels per site (gate slice cast fp32->bf16, the bf16 multiply, the bf16 add).
One tiny kernel replaces the chain with bit-identical arithmetic:

    gate_b = round_bf16(gate_f32[k])          (the layer's ``.to(dtype)``)
    prod   = round_bf16(out * gate_b)          (bf16 mul, torch semantics)
    dst    = round_bf16(residual + prod)       (bf16 add, torch semantics)

The intermediate product is rounded to bf16 through a scratch store/load on
purpose: Triton would otherwise prove the ``f32 -> bf16 -> f32`` round trip an
identity, fuse mul+add into one fp32 FMA and round once, which differs from the
eager chain's double rounding by 1 bf16 ulp. Crossing global memory keeps both
roundings, so the result is bit-equal to the eager chain given the same inputs
(no drift is added).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gated_residual_kernel(
    res_ptr,        # [M, K] bf16 residual
    out_ptr,        # [M, K] bf16 branch output (attn-o / down-proj out)
    gate_ptr,       # [K] f32 gate slice (modulation[2K:3K], contiguous)
    prod_ptr,       # [M, K] bf16 scratch (bf16-rounding barrier)
    dst_ptr,        # [M, K] bf16 out
    M,
    K,
    stride_rm,
    stride_rk,
    stride_om,
    stride_ok,
    stride_dm,
    stride_dk,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_k = pid_n * BLOCK_K + tl.arange(0, BLOCK_K)
    m_mask = offs_m < M

    res = tl.load(res_ptr + offs_m[:, None] * stride_rm + offs_k[None, :] * stride_rk,
                  mask=m_mask[:, None], other=0.0)
    out = tl.load(out_ptr + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok,
                  mask=m_mask[:, None], other=0.0)
    gb = tl.load(gate_ptr + offs_k).to(tl.bfloat16)  # exact fp32 -> bf16 cast
    # round the product to bf16 via global memory (see module docstring)
    prod = (out.to(tl.float32) * gb.to(tl.float32)[None, :]).to(tl.bfloat16)
    tl.store(prod_ptr + offs_m[:, None] * K + offs_k[None, :], prod, mask=m_mask[:, None])
    prod2 = tl.load(prod_ptr + offs_m[:, None] * K + offs_k[None, :], mask=m_mask[:, None], other=0.0)
    dst = (res.to(tl.float32) + prod2.to(tl.float32)).to(tl.bfloat16)
    tl.store(dst_ptr + offs_m[:, None] * stride_dm + offs_k[None, :] * stride_dk,
             dst, mask=m_mask[:, None])


def gated_residual(
    residual: torch.Tensor,   # [1, M, K] bf16 (or [M, K])
    branch_out: torch.Tensor,  # [1, M, K] bf16
    gate_mod: torch.Tensor,   # [3*K] f32 modulation (scale/shift/gate layout)
    *,
    out: torch.Tensor,        # [1, M, K] bf16 destination (caller-owned, persistent)
    prod: torch.Tensor,       # [1, M, K] bf16 scratch (caller-owned, persistent)
    block_k: int = 256,
    num_warps: int = 4,
) -> torch.Tensor:
    """Bit-exact ``residual + branch_out * gate`` where ``gate`` is the
    ``[2K:3K]`` slice of ``gate_mod`` (the AdaRMS gate of this site).

    ``out``/``prod`` are caller-owned persistent buffers (graph-capture safe);
    ``out`` is returned.
    """
    K = gate_mod.numel() // 3
    if residual.shape[-1] != K or branch_out.shape[-1] != K:
        raise ValueError(f"gated residual needs width {K} (got "
                         f"{residual.shape[-1]}, {branch_out.shape[-1]})")
    if K % block_k:
        raise ValueError(f"gated residual needs K % block_k == 0 (K={K}, block_k={block_k})")
    m = residual.numel() // K
    view = lambda t: t.reshape(m, K)  # noqa: E731
    res, bout, dst, prd = view(residual), view(branch_out), view(out), view(prod)
    _gated_residual_kernel[(K // block_k,)](
        res, bout, gate_mod[2 * K:], prd, dst,
        m, K,
        res.stride(0), res.stride(1),
        bout.stride(0), bout.stride(1),
        dst.stride(0), dst.stride(1),
        BLOCK_M=64, BLOCK_K=block_k,
        num_warps=num_warps,
    )
    return out
