"""Triton vision kernel (``--tl-fused-vit``).

``triton_vision_out_proj`` runs the vision attention's ``out_proj`` GEMM
directly over the SDPA output layout ``[B, H, L, D]``, so the eager path's
``transpose(1, 2).contiguous()`` materialisation (a ``[B*L, E]`` bf16 copy per
layer) disappears. With ``BLOCK_D == head_dim`` the per-head load is one
contiguous ``[BLOCK_M, D]`` block (the rows of a head's ``[L, D]`` slab are
consecutive), so the loads stay coalesced; the head loop is unrolled at
compile time (``H`` is a constexpr). Still gated behind ``--tl-fused-vit``: it is a
hand-written GEMM, so the K-accumulation order is not guaranteed to match
cuBLAS on every shape.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _vision_out_proj_kernel(
    att_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    L,
    D,
    E,
    stride_ab,
    stride_ah,
    stride_al,
    stride_ad,
    stride_w,
    stride_ob,
    stride_ol,
    stride_oe,
    H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_n = tl.program_id(2)
    offs_l = pid_l * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    l_mask = offs_l < L
    n_mask = offs_n < E
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for h in range(H):
        a = tl.load(
            att_ptr + b * stride_ab + h * stride_ah + offs_l[:, None] * stride_al + offs_d[None, :] * stride_ad,
            mask=l_mask[:, None],
            other=0.0,
        )
        w = tl.load(
            w_ptr + offs_n[:, None] * stride_w + (h * BLOCK_D + offs_d)[None, :],
            mask=n_mask[:, None],
            other=0.0,
        )
        acc = tl.dot(a, tl.trans(w), acc=acc)
    bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + bias[None, :]
    tl.store(
        out_ptr + b * stride_ob + offs_l[:, None] * stride_ol + offs_n[None, :] * stride_oe,
        acc.to(tl.bfloat16),
        mask=l_mask[:, None] & n_mask[None, :],
    )


def triton_vision_out_proj(att, weight, bias, out=None):
    """out_proj over the SDPA-output layout.

    Args:
        att: ``[B, H, L, D]`` bf16 contiguous (SDPA output).
        weight: ``[E, H*D]`` bf16, bias: ``[E]`` bf16 (out_proj parameters).

    Returns ``[B, L, E]`` bf16 -- what the eager path's transpose+copy+out_proj
    produces, without the materialised ``[B, L, H*D]`` copy.
    """
    B, H, L, D = att.shape
    E = weight.shape[0]
    if out is None:
        out = torch.empty((B, L, E), dtype=torch.bfloat16, device=att.device)
    _vision_out_proj_kernel[(B, triton.cdiv(L, 64), triton.cdiv(E, 64))](
        att,
        weight,
        bias,
        out,
        L,
        D,
        E,
        att.stride(0),
        att.stride(1),
        att.stride(2),
        att.stride(3),
        weight.stride(0),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        H=H,
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_D=D,
        num_warps=4,
    )
    return out
