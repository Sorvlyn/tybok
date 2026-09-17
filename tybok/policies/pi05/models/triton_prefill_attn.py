"""Triton GQA-native flash attention for the pi0.5 LLM backbone prefill (drift tier).

The VLM prefill attention (``GemmaAttention``, PaliGemma 2B: H=8, G=1, D=256,
K=2048) runs ``input_layernorm -> q/k/v projections -> RoPE -> GQA expand ->
bf16 SDPA``. ``_prefill_attn_kernel`` (``triton_prefill_attn``) replaces the
``expand->reshape`` GQA materialisation + bf16 SDPA with a GQA-native flash
attention (one program per query head, all heads share the single kv group),
paired with the *cuBLAS* concatenated qkv GEMM. Used as the attention half of
``--tl-llm-flash-attn`` (``GemmaAttention._forward_fused_prefill``).

This is a **drift tier**: the Triton online softmax / tf32 scores differ from
the eager reference.

.. note::
   ``triton_prefill_attn`` (GQA flash) does **not** reproduce the bf16 SDPA's
   drift cancellation: in the fp8+pad-free triple it FAILS. Its tf32-score error
   mode differs from the bf16-SDPA error mode that coincidentally cancels the
   fp8+pad-free error. Use it only standalone, not inside the fp8+pad-free triple.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _prefill_attn_kernel(
    q_ptr,       # [L, H, D] bf16 post-RoPE queries
    k_ptr,       # [L, G, D] bf16 keys
    v_ptr,       # [L, G, D] bf16 values
    mask_ptr,    # [L, L] bool 2-D mask
    out_ptr,     # [L, H*D] bf16 pre-o_proj attention output
    L,
    SCALE,
    stride_qm,
    stride_qh,
    stride_qd,
    stride_km,
    stride_kg,
    stride_kd,
    stride_vm,
    stride_vg,
    stride_vd,
    stride_mask_m,
    stride_mask_a,
    stride_om,
    stride_oh,
    stride_od,
    H: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_A: tl.constexpr,
    BLOCK_D: tl.constexpr,
    DH: tl.constexpr,
):
    """GQA-native flash attention; one program per (query head, M block).

    The score dot runs as two half-width bf16 dots over the RoPE halves
    ``[0, DH)`` / ``[DH, D)``, avoiding a full-width bf16 ``tl.dot`` over
    BLOCK_D=256.
    """
    h = tl.program_id(0)
    pid_m = tl.program_id(1)
    g = h // (H // G)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    offs_h = tl.arange(0, DH)
    m_mask = offs_m < L

    q1 = tl.load(
        q_ptr + offs_m[:, None] * stride_qm + h * stride_qh + offs_h[None, :] * stride_qd,
        mask=m_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    q2 = tl.load(
        q_ptr + offs_m[:, None] * stride_qm + h * stride_qh + (DH + offs_h)[None, :] * stride_qd,
        mask=m_mask[:, None],
        other=0.0,
    ).to(tl.float32)

    m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    for a0 in range(0, L, BLOCK_A):
        offs_a = a0 + tl.arange(0, BLOCK_A)
        a_mask = offs_a < L
        vv = tl.load(
            v_ptr + offs_a[:, None] * stride_vm + g * stride_vg + offs_d[None, :] * stride_vd,
            mask=a_mask[:, None],
            other=0.0,
        )
        mk = tl.load(
            mask_ptr + offs_m[:, None] * stride_mask_m + offs_a[None, :] * stride_mask_a,
            mask=m_mask[:, None] & a_mask[None, :],
            other=False,
        )
        kk_t1 = tl.load(
            k_ptr + offs_h[:, None] * stride_kd + g * stride_kg + offs_a[None, :] * stride_km,
            mask=a_mask[None, :],
            other=0.0,
        )  # [DH, BLOCK_A] transposed first half
        kk_t2 = tl.load(
            k_ptr + (DH + offs_h)[:, None] * stride_kd + g * stride_kg + offs_a[None, :] * stride_km,
            mask=a_mask[None, :],
            other=0.0,
        )  # [DH, BLOCK_A] transposed second half
        s = (tl.dot(q1.to(tl.bfloat16), kk_t1) + tl.dot(q2.to(tl.bfloat16), kk_t2)) * SCALE
        # big_neg (not -inf) matches the eager ``finfo.min`` mask: fully-masked
        # rows then softmax to a uniform distribution instead of 0/0 = NaN.
        s = tl.where(mk, s, -3.4028234663852886e38)
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.exp(s - m_new[:, None])
        alpha = tl.where(m_i == float("-inf"), 0.0, tl.exp(m_i - m_new))
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), vv)
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + h * stride_oh + offs_d[None, :] * stride_od,
        acc.to(tl.bfloat16),
        mask=m_mask[:, None],
    )


def triton_prefill_attn(
    q: torch.Tensor,             # [L, H, D] bf16
    k: torch.Tensor,             # [L, G, D] bf16
    v: torch.Tensor,             # [L, G, D] bf16
    mask2d: torch.Tensor,        # [L, L] bool
    head_dim: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """GQA-native flash prefill attention -> [L, H*D] bf16."""
    L, H, D = q.shape
    G = k.shape[1]
    if out is None:
        out = torch.empty(L, H * D, dtype=torch.bfloat16, device=q.device)
    grid = (H, triton.cdiv(L, 16))
    _prefill_attn_kernel[grid](
        q, k, v, mask2d, out, L, head_dim ** -0.5,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        mask2d.stride(0), mask2d.stride(1),
        H * D, D, 1,  # out [L, H*D]: row stride H*D, head stride D, dim stride 1
        H=H, G=G, D=D, BLOCK_M=16, BLOCK_A=32, BLOCK_D=256, DH=D // 2,
        num_warps=4, num_stages=1,
    )
    return out
