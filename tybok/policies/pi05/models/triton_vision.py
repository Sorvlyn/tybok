"""Fused Triton GEMMs for the pi0.5 SigLIP vision tower (drift tier).

The vision tower runs 27 ``SiglipEncoderLayer`` blocks in fp32; the GEMMs never
touch the tensor cores (fp32 -> FFMA). This module provides low-precision fused
kernels that keep the **residual stream in fp32** (the ``residual +
hidden_states`` adds stay exactly on torch).

- ``triton_vision_qkv`` -- ``layer_norm1 -> q_proj | k_proj | v_proj`` as one fp16
  tensor-core GEMM over the concatenated ``[3C, C]`` weight. LayerNorm stats are
  computed in fp32 (like the eager ``nn.LayerNorm``), the normalised input is cast
  to fp16 at the fused-kernel boundary, the dots accumulate in fp32,
  and the output is written fp32 so the SDPA / softmax path stays the unchanged
  f32 mem-eff implementation.
- ``triton_vision_mlp`` -- ``layer_norm2 -> fc1 -> gelu_tanh -> fc2`` as two bf16
  tensor-core kernels (up then down), with the ``[M, F]`` gelu activation held in
  bf16 between them; the down kernel adds the fp32 residual and writes fp32 (only
  the two GEMMs + gelu are bf16).
- ``triton_vision_attention`` -- full (unmasked, ``is_causal=False``) flash
  attention: fp16 QK^T / PV tensor-core dots with fp32 accumulation, fp32
  online-softmax, fp32 output. head_dim 72 is padded to BLOCK_D 128 (pad lanes
  are 0).

The q/k/v projections need a >=10-bit mantissa, so q/k/v use fp16 (bf16 is not
precise enough); the MLP is smooth enough for bf16. The residual adds, norms and
SDPA stay fp32.

This is a **drift tier**: Triton's bf16/fp16 MMA accumulation order differs from
cuBLAS, so the eager/bit-exact f32 path is untouched and this is opt-in.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _vision_qkv_kernel(
    x_ptr,          # [M, C] fp32 (post-embeddings residual, pre-norm)
    lnw_ptr,        # [C] fp32
    lnb_ptr,        # [C] fp32
    wqkv_ptr,       # [3C, C] fp16 (q|k|v concatenated along rows)
    bqkv_ptr,       # [3C] fp16
    out_ptr,        # [M, 3C] fp32
    M,
    C,
    N,              # 3 * C
    EPS,
    stride_xm,
    stride_xc,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    n_mask = offs_n < N

    # two-pass fp32 LayerNorm (matches the eager nn.LayerNorm boundary cast).
    s = tl.zeros((BLOCK_M,), tl.float32)
    ss = tl.zeros((BLOCK_M,), tl.float32)
    for k0 in range(0, C, BLOCK_K):
        km = (k0 + offs_k) < C
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + (k0 + offs_k)[None, :] * stride_xc,
            mask=m_mask[:, None] & km[None, :],
            other=0.0,
        )
        s += tl.sum(x, 1)
        ss += tl.sum(x * x, 1)
    mean = s / C
    var = ss / C - mean * mean
    rstd = tl.rsqrt(var + EPS)

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k0 in range(0, C, BLOCK_K):
        km = (k0 + offs_k) < C
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + (k0 + offs_k)[None, :] * stride_xc,
            mask=m_mask[:, None] & km[None, :],
            other=0.0,
        )
        lnw = tl.load(lnw_ptr + (k0 + offs_k), mask=km, other=0.0)
        lnb = tl.load(lnb_ptr + (k0 + offs_k), mask=km, other=0.0)
        n = (x - mean[:, None]) * rstd[:, None] * lnw[None, :] + lnb[None, :]
        a = n.to(tl.float16)
        # load W^T tile directly: [BLOCK_K, BLOCK_N] -> dot(a, w) needs no tl.trans.
        w = tl.load(
            wqkv_ptr + (k0 + offs_k)[:, None] * stride_wk + offs_n[None, :] * stride_wn,
            mask=km[:, None] & n_mask[None, :],
            other=0.0,
        )
        acc = tl.dot(a, w, acc)  # fp16 MMA, fp32 accumulate
    bias = tl.load(bqkv_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    acc += bias[None, :]
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc,
        mask=m_mask[:, None] & n_mask[None, :],
    )


@triton.jit
def _vision_mlp_up_kernel(
    x_ptr,          # [M, C] fp32 (post-attention residual, pre-norm)
    lnw_ptr,        # [C] fp32
    lnb_ptr,        # [C] fp32
    w1_ptr,         # [F, C] bf16
    b1_ptr,         # [F] bf16
    act_ptr,        # [M, F] bf16 (gelu activation, down-proj input)
    M,
    C,
    F,
    EPS,
    stride_xm,
    stride_xc,
    stride_w1n,
    stride_w1k,
    stride_am,
    stride_an,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    n_mask = offs_n < F

    s = tl.zeros((BLOCK_M,), tl.float32)
    ss = tl.zeros((BLOCK_M,), tl.float32)
    for k0 in range(0, C, BLOCK_K):
        km = (k0 + offs_k) < C
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + (k0 + offs_k)[None, :] * stride_xc,
            mask=m_mask[:, None] & km[None, :],
            other=0.0,
        )
        s += tl.sum(x, 1)
        ss += tl.sum(x * x, 1)
    mean = s / C
    var = ss / C - mean * mean
    rstd = tl.rsqrt(var + EPS)

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k0 in range(0, C, BLOCK_K):
        km = (k0 + offs_k) < C
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + (k0 + offs_k)[None, :] * stride_xc,
            mask=m_mask[:, None] & km[None, :],
            other=0.0,
        )
        lnw = tl.load(lnw_ptr + (k0 + offs_k), mask=km, other=0.0)
        lnb = tl.load(lnb_ptr + (k0 + offs_k), mask=km, other=0.0)
        n = (x - mean[:, None]) * rstd[:, None] * lnw[None, :] + lnb[None, :]
        a = n.to(tl.bfloat16)
        w1 = tl.load(
            w1_ptr + (k0 + offs_k)[:, None] * stride_w1k + offs_n[None, :] * stride_w1n,
            mask=km[:, None] & n_mask[None, :],
            other=0.0,
        )
        acc = tl.dot(a, w1, acc)  # bf16 MMA, fp32 accumulate
    bias = tl.load(b1_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    acc += bias[None, :]
    # gelu_tanh in fp32 (F.gelu(x, approximate="tanh")):
    #   0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    c = 0.7978845608028654
    t = tl.extra.cuda.libdevice.tanh(c * acc * (1.0 + 0.044715 * acc * acc))
    act = 0.5 * acc * (1.0 + t)
    tl.store(
        act_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an,
        act.to(tl.bfloat16),
        mask=m_mask[:, None] & n_mask[None, :],
    )


@triton.jit
def _vision_mlp_down_kernel(
    act_ptr,        # [M, F] bf16
    w2_ptr,         # [C, F] bf16
    b2_ptr,         # [C] bf16
    res_ptr,        # [M, C] fp32 (the residual stream)
    out_ptr,        # [M, C] fp32
    M,
    F,
    C,
    stride_am,
    stride_an,
    stride_w2n,
    stride_w2k,
    stride_rm,
    stride_rc,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    n_mask = offs_n < C

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k0 in range(0, F, BLOCK_K):
        km = (k0 + offs_k) < F
        a = tl.load(
            act_ptr + offs_m[:, None] * stride_am + (k0 + offs_k)[None, :] * stride_an,
            mask=m_mask[:, None] & km[None, :],
            other=0.0,
        )
        w2 = tl.load(
            w2_ptr + (k0 + offs_k)[:, None] * stride_w2k + offs_n[None, :] * stride_w2n,
            mask=km[:, None] & n_mask[None, :],
            other=0.0,
        )
        acc = tl.dot(a, w2, acc)  # bf16 MMA, fp32 accumulate
    bias = tl.load(b2_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    acc += bias[None, :]
    res = tl.load(
        res_ptr + offs_m[:, None] * stride_rm + offs_n[None, :] * stride_rc,
        mask=m_mask[:, None] & n_mask[None, :],
        other=0.0,
    )
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc + res,
        mask=m_mask[:, None] & n_mask[None, :],
    )


@triton.jit
def _vision_attn_kernel(
    q_ptr,          # [B*H, L, HD] fp32
    k_ptr,          # [B*H, L, HD] fp32
    v_ptr,          # [B*H, L, HD] fp32
    out_ptr,        # [B*H, L, HD] fp32
    L,
    HD,
    SCALE,
    stride_qb,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_km,
    stride_kd,
    stride_vb,
    stride_vm,
    stride_vd,
    stride_ob,
    stride_om,
    stride_od,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Full (unmasked) flash attention per (batch*head, M-block).

    head_dim 72 is padded to BLOCK_D (128) with masked loads; the pad lanes are
    0 so the extra dot work is wasted FLOPs but not wrong.  QK^T and PV use fp16
    tensor cores with fp32 accumulation; softmax stays fp32.  The PV output
    (``acc / l_i``) is rounded to bf16 before the store, because the downstream
    out-proj GEMM is bf16 and never needs the fp32 attention output.
    """
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    m_mask = offs_m < L
    d_mask = offs_d < HD

    q = tl.load(
        q_ptr + pid_bh * stride_qb + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
        mask=m_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float16)

    m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    for n0 in range(0, L, BLOCK_N):
        # load k already transposed as (BLOCK_D, BLOCK_N) so the score dot needs
        # no tl.trans shared-memory round trip.
        k = tl.load(
            k_ptr + pid_bh * stride_kb + offs_d[:, None] * stride_kd + (n0 + offs_n)[None, :] * stride_km,
            mask=d_mask[:, None],
            other=0.0,
        ).to(tl.float16)
        v = tl.load(
            v_ptr + pid_bh * stride_vb + (n0 + offs_n)[:, None] * stride_vm + offs_d[None, :] * stride_vd,
            mask=d_mask[None, :],
            other=0.0,
        ).to(tl.float16)
        s = tl.dot(q, k) * SCALE  # (BLOCK_M, BLOCK_N) fp32
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(
        out_ptr + pid_bh * stride_ob + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
        acc.to(tl.bfloat16),
        mask=m_mask[:, None] & d_mask[None, :],
    )


# --------------------------------------------------------------------------- #
# host-side wrappers
# --------------------------------------------------------------------------- #
def pack_qkv(q_proj, k_proj, v_proj) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate q/k/v weights+bias into ``[3C, C]`` / ``[3C]`` fp16."""
    w = torch.cat([q_proj.weight, k_proj.weight, v_proj.weight], dim=0).to(torch.float16).contiguous()
    b = torch.cat([q_proj.bias, k_proj.bias, v_proj.bias], dim=0).to(torch.float16).contiguous()
    return w, b


def triton_vision_qkv(
    x: torch.Tensor,               # [M, C] fp32
    ln_w: torch.Tensor,            # [C] fp32
    ln_b: torch.Tensor,            # [C] fp32
    wqkv: torch.Tensor,            # [3C, C] fp16
    bqkv: torch.Tensor,            # [3C] fp16
    eps: float,
    out: torch.Tensor | None = None,  # [M, 3C] fp32
    block_m: int = 64,
    block_n: int = 128,
    block_k: int = 64,
    num_warps: int = 4,
    num_stages: int = 1,
) -> torch.Tensor:
    M, C = x.shape
    N = wqkv.shape[0]
    assert N == 3 * C and wqkv.shape[1] == C
    if out is None:
        out = torch.empty((M, N), dtype=torch.float32, device=x.device)
    _vision_qkv_kernel[(triton.cdiv(N, block_n), triton.cdiv(M, block_m))](
        x,
        ln_w,
        ln_b,
        wqkv,
        bqkv,
        out,
        M,
        C,
        N,
        float(eps),
        x.stride(0),
        x.stride(1),
        wqkv.stride(0),
        wqkv.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


def triton_vision_attention(
    q: torch.Tensor,               # [B, H, L, HD] fp32
    k: torch.Tensor,               # [B, H, L, HD] fp32
    v: torch.Tensor,               # [B, H, L, HD] fp32
    scale: float,
    out: torch.Tensor | None = None,  # [B, H, L, HD] bf16
    block_m: int = 64,
    block_n: int = 64,
    block_d: int = 128,
    num_warps: int = 8,
    num_stages: int = 2,
) -> torch.Tensor:
    """Full (unmasked, is_causal=False) flash attention over the head dim.

    The PV output is rounded to bf16 in-kernel (the downstream out-proj GEMM is
    bf16), so ``out`` is bf16.
    """
    B, H, L, HD = q.shape
    assert L % block_n == 0, f"L={L} must be divisible by block_n={block_n}"
    assert block_d >= HD, f"block_d={block_d} must cover head_dim={HD}"
    if out is None:
        out = torch.empty((B, H, L, HD), dtype=torch.bfloat16, device=q.device)
    qf = q.reshape(B * H, L, HD)
    kf = k.reshape(B * H, L, HD)
    vf = v.reshape(B * H, L, HD)
    of = out.reshape(B * H, L, HD)
    _vision_attn_kernel[(B * H, triton.cdiv(L, block_m))](
        qf,
        kf,
        vf,
        of,
        L,
        HD,
        float(scale),
        qf.stride(0),
        qf.stride(1),
        qf.stride(2),
        kf.stride(0),
        kf.stride(1),
        kf.stride(2),
        vf.stride(0),
        vf.stride(1),
        vf.stride(2),
        of.stride(0),
        of.stride(1),
        of.stride(2),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


@triton.jit
def _vision_attn_out_kernel(
    a_ptr,          # [M, C] fp32 (flash output, already transposed to [B,L,C])
    w_ptr,          # [C, C] bf16 (out_proj weight)
    b_ptr,          # [C] bf16 (out_proj bias)
    res_ptr,        # [M, C] fp32 (the residual stream)
    out_ptr,        # [M, C] fp32
    M,
    C,
    stride_am,
    stride_ak,
    stride_wk,
    stride_wn,
    stride_rm,
    stride_rc,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fused out_proj (bf16 GEMM) + bias + fp32 residual add, fp32 output.

    Replaces the eager ``out_proj(attn_out) + residual`` (a fp32 cuBLAS GEMM +
    a separate elementwise add) with ONE bf16 tensor-core GEMM whose epilogue
    adds the bias and the fp32 residual before the single fp32 store.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    n_mask = offs_n < C

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k0 in range(0, C, BLOCK_K):
        km = (k0 + offs_k) < C
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + (k0 + offs_k)[None, :] * stride_ak,
            mask=m_mask[:, None] & km[None, :],
            other=0.0,
        ).to(tl.bfloat16)
        w = tl.load(
            w_ptr + (k0 + offs_k)[:, None] * stride_wk + offs_n[None, :] * stride_wn,
            mask=km[:, None] & n_mask[None, :],
            other=0.0,
        )
        acc = tl.dot(a, w, acc=acc)  # bf16 MMA, fp32 accumulate
    bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    res = tl.load(
        res_ptr + offs_m[:, None] * stride_rm + offs_n[None, :] * stride_rc,
        mask=m_mask[:, None] & n_mask[None, :],
        other=0.0,
    )
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc + bias[None, :] + res,
        mask=m_mask[:, None] & n_mask[None, :],
    )


def triton_vision_out_proj(
    attn_out: torch.Tensor,      # [B, L, C] bf16 (flash output, transposed)
    w: torch.Tensor,             # [C, C] bf16 (out_proj weight)
    b: torch.Tensor,             # [C] bf16 (out_proj bias)
    residual: torch.Tensor,      # [B, L, C] fp32
    out: torch.Tensor | None = None,  # [B, L, C] fp32
    block_m: int = 32,
    block_n: int = 64,
    block_k: int = 128,
    num_warps: int = 4,
    num_stages: int = 2,
) -> torch.Tensor:
    """Fused out_proj (bf16) + bias + fp32 residual -> fp32 (drift tier).

    The flash attention output must already be transposed to ``[B, L, C]`` (the
    caller does ``attn.transpose(1, 2).contiguous().reshape(B, L, C)``); this
    kernel then runs the out-projection GEMM in bf16 and adds the fp32 residual
    in the epilogue, so the residual stream never leaves fp32.
    """
    B, L, C = attn_out.shape
    M = B * L
    af = attn_out.reshape(M, C)
    rf = residual.reshape(M, C)
    if out is None:
        out = torch.empty((B, L, C), dtype=torch.float32, device=attn_out.device)
    of = out.reshape(M, C)
    _vision_attn_out_kernel[(triton.cdiv(M, block_m), triton.cdiv(C, block_n))](
        af,
        w,
        b,
        rf,
        of,
        M,
        C,
        af.stride(0),
        af.stride(1),
        w.stride(1),
        w.stride(0),
        rf.stride(0),
        rf.stride(1),
        of.stride(0),
        of.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


def triton_vision_mlp(
    x: torch.Tensor,               # [M, C] fp32 (pre-norm residual)
    ln_w: torch.Tensor,            # [C] fp32
    ln_b: torch.Tensor,            # [C] fp32
    w1: torch.Tensor,              # [F, C] bf16
    b1: torch.Tensor,              # [F] bf16
    w2: torch.Tensor,              # [C, F] bf16
    b2: torch.Tensor,              # [C] bf16
    eps: float,
    act: torch.Tensor | None = None,  # [M, F] bf16 scratch
    out: torch.Tensor | None = None,  # [M, C] fp32
    block_m: int = 32,
    block_n_up: int = 128,
    block_n_down: int = 64,
    block_k: int = 128,
    num_warps_up: int = 8,
    num_stages_up: int = 1,
    num_warps_down: int = 4,
    num_stages_down: int = 2,
) -> torch.Tensor:
    M, C = x.shape
    F = w1.shape[0]
    if act is None:
        act = torch.empty((M, F), dtype=torch.bfloat16, device=x.device)
    if out is None:
        out = torch.empty((M, C), dtype=torch.float32, device=x.device)
    _vision_mlp_up_kernel[(triton.cdiv(F, block_n_up), triton.cdiv(M, block_m))](
        x,
        ln_w,
        ln_b,
        w1,
        b1,
        act,
        M,
        C,
        F,
        float(eps),
        x.stride(0),
        x.stride(1),
        w1.stride(0),
        w1.stride(1),
        act.stride(0),
        act.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n_up,
        BLOCK_K=block_k,
        num_warps=num_warps_up,
        num_stages=num_stages_up,
    )
    _vision_mlp_down_kernel[(triton.cdiv(C, block_n_down), triton.cdiv(M, block_m))](
        act,
        w2,
        b2,
        x,
        out,
        M,
        F,
        C,
        act.stride(0),
        act.stride(1),
        w2.stride(0),
        w2.stride(1),
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n_down,
        BLOCK_K=block_k,
        num_warps=num_warps_down,
        num_stages=num_stages_down,
    )
    return out


@triton.jit
def _vision_projector_kernel(
    x_ptr,          # [M, C] fp32 (encoder output, pre-post_layernorm)
    lnw_ptr,        # [C] fp32 (post_layernorm weight)
    lnb_ptr,        # [C] fp32 (post_layernorm bias)
    w_ptr,          # [N, C] bf16 (projector weight)
    b_ptr,          # [N] bf16 (projector bias)
    out_ptr,        # [M, N] fp32
    M,
    C,
    N,
    EPS,
    stride_xm,
    stride_xc,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fused post_layernorm + multimodal-projector linear (bf16 GEMM), fp32 out.

    Same two-pass fp32 LayerNorm + low-precision GEMM pattern as the qkv kernel,
    but the projection runs in bf16 (the projector is a smooth path, not the
    precision-critical attention path) and writes fp32 so the VLM prefix stays
    fp32.
    """
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    n_mask = offs_n < N

    s = tl.zeros((BLOCK_M,), tl.float32)
    ss = tl.zeros((BLOCK_M,), tl.float32)
    for k0 in range(0, C, BLOCK_K):
        km = (k0 + offs_k) < C
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + (k0 + offs_k)[None, :] * stride_xc,
            mask=m_mask[:, None] & km[None, :],
            other=0.0,
        )
        s += tl.sum(x, 1)
        ss += tl.sum(x * x, 1)
    mean = s / C
    var = ss / C - mean * mean
    rstd = tl.rsqrt(var + EPS)

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k0 in range(0, C, BLOCK_K):
        km = (k0 + offs_k) < C
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + (k0 + offs_k)[None, :] * stride_xc,
            mask=m_mask[:, None] & km[None, :],
            other=0.0,
        )
        lnw = tl.load(lnw_ptr + (k0 + offs_k), mask=km, other=0.0)
        lnb = tl.load(lnb_ptr + (k0 + offs_k), mask=km, other=0.0)
        n = (x - mean[:, None]) * rstd[:, None] * lnw[None, :] + lnb[None, :]
        a = n.to(tl.bfloat16)
        w = tl.load(
            w_ptr + (k0 + offs_k)[:, None] * stride_wk + offs_n[None, :] * stride_wn,
            mask=km[:, None] & n_mask[None, :],
            other=0.0,
        )
        acc = tl.dot(a, w, acc=acc)  # bf16 MMA, fp32 accumulate
    bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    acc += bias[None, :]
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc,
        mask=m_mask[:, None] & n_mask[None, :],
    )


def triton_vision_projector(
    x: torch.Tensor,              # [M, C] fp32 (encoder output, pre-norm)
    ln_w: torch.Tensor,           # [C] fp32
    ln_b: torch.Tensor,           # [C] fp32
    w: torch.Tensor,              # [N, C] bf16
    b: torch.Tensor,              # [N] bf16
    eps: float,
    out: torch.Tensor | None = None,  # [M, N] fp32
    block_m: int = 32,
    block_n: int = 64,
    block_k: int = 128,
    num_warps: int = 4,
    num_stages: int = 2,
) -> torch.Tensor:
    """Fused post_layernorm + projector linear (bf16) -> fp32 (drift tier)."""
    M, C = x.shape
    N = w.shape[0]
    if out is None:
        out = torch.empty((M, N), dtype=torch.float32, device=x.device)
    _vision_projector_kernel[(triton.cdiv(N, block_n), triton.cdiv(M, block_m))](
        x,
        ln_w,
        ln_b,
        w,
        b,
        out,
        M,
        C,
        N,
        float(eps),
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
