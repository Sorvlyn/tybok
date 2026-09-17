"""FP8 (W8A8) expert MLP kernels for the pi0.5 denoising path (drift tier).

The expert MLP is DRAM-bound on its bf16 weights, re-read every denoising step.
FP8 halves the weight bytes. The wired entry point is ``triton_mlp_fp8_chain``,
a **producer-fused** pipeline with no host-side activation round trip:

1. ``triton_norm_gate_up_fp8`` -- fused AdaRMS norm + fp8 gate/up + gelu_tanh
   (per-output-channel cached weight scales, per-token dynamic activation scale
   computed in the norm pass), writing the bf16 activation AND the per-CTA
   per-token partial |act| max;
2. ``triton_quant_act_fp8`` -- reduce the partials to the per-token scale and cast
   the activation to fp8 (one read of the act buffer, L2-warm);
3. ``triton_down_proj_fp8_splitk`` -- fp8 down GEMM split over K + fp32 merge.

Numerics: 8-bit weights + per-token activations are a drift tier.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# e4m3 finite max (the fp8 range bound used for the per-row scale)
FP8_MAX = 448.0

# weight quant caches: data_ptr-keyed with strong refs
_wq_cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
_wqt_cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def quantize_weight_rows(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """bf16 [N, K] -> (fp8 e4m3 [N, K], per-row scale [N] fp32, strong ref).

    Per-output-channel (row) scale over the full K: ``wq = w / s`` with
    ``s = amax(row) / FP8_MAX`` (so ``|wq| <= FP8_MAX``, no saturation loss).
    """
    ptr = w.data_ptr()
    e = _wq_cache.get(ptr)
    if e is None:
        amax = w.abs().amax(dim=1, keepdim=True).float()
        s = (amax / FP8_MAX).clamp_min(1e-12)
        wq = (w.float() / s).to(torch.float8_e4m3fn).contiguous()
        if len(_wq_cache) > 64:
            _wq_cache.clear()
        _wq_cache[ptr] = (wq, s.squeeze(1).contiguous(), w)
        return _wq_cache[ptr]
    return e[0], e[1], e[2]


# --------------------------------------------------------------------------- #
# fused AdaRMS norm -> fp8 gate/up -> gelu_tanh (activation fp8 per token)
# --------------------------------------------------------------------------- #
@triton.jit
def _norm_gate_up_fp8_kernel(
    x_ptr,          # [M, K] bf16 pre-norm residual
    mod_ptr,        # [3*K] f32 AdaRMS modulation (scale, shift, gate)
    g8t_ptr,        # [K, N] fp8 gate weight (pre-transposed, B-operand layout)
    gs_ptr,         # [N] f32 gate per-output-channel scale
    u8t_ptr,        # [K, N] fp8 up weight
    us_ptr,         # [N] f32 up per-output-channel scale
    out_ptr,        # [M, N] bf16 gelu(gate)*up (down_proj input)
    part_ptr,       # [n_tiles, rows_pad] f32 per-token partial max |act| (per CTA)
    stride_part,    # rows_pad (row stride of part_ptr)
    M,
    K,
    N,
    EPS,
    stride_xm,
    stride_xk,
    stride_gwk,
    stride_gwn,
    stride_uwk,
    stride_uwn,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    """Fused AdaRMS norm -> fp8 gate/up -> gelu_tanh.

    Expert shapes guarantee ``K % BLOCK_K == 0`` and ``N % BLOCK_N == 0``, so
    the K loop is mask-free and the weight loads need no n mask; only the row
    mask survives. The weights are pre-transposed to ``[K, N]`` fp8 so each
    iteration loads the B operand directly in the ``[BK, BN]`` dot shape -- no
    per-iteration ``tl.trans`` smem round trip.
    """
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M

    # pass 1: per-row variance over the full K (fp32)
    sum_sq = tl.zeros((BLOCK_M,), tl.float32)
    for k0 in range(0, K, BLOCK_K):
        x = tl.load(x_ptr + offs_m[:, None] * stride_xm + (k0 + offs_k)[None, :] * stride_xk,
                    mask=m_mask[:, None], other=0.0)
        xf = x.to(tl.float32)
        sum_sq += tl.sum(xf * xf, 1)
    rstd = tl.rsqrt(sum_sq / K + EPS)

    # pass 2: max |a| over the full K -> per-token fp8 activation scale
    amax = tl.zeros((BLOCK_M,), tl.float32)
    for k0 in range(0, K, BLOCK_K):
        x = tl.load(x_ptr + offs_m[:, None] * stride_xm + (k0 + offs_k)[None, :] * stride_xk,
                    mask=m_mask[:, None], other=0.0)
        xf = x.to(tl.float32)
        sc = tl.load(mod_ptr + (k0 + offs_k))
        sh = tl.load(mod_ptr + K + (k0 + offs_k))
        a = xf * rstd[:, None] * (1.0 + sc[None, :]) + sh[None, :]
        amax = tl.maximum(amax, tl.max(tl.abs(a), 1))

    gate = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    up = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    gs = tl.load(gs_ptr + offs_n)
    us = tl.load(us_ptr + offs_n)
    inv = tl.where(amax == 0.0, 0.0, FP8_MAX / amax)  # per-token 1/sA
    for k0 in range(0, K, BLOCK_K):
        x = tl.load(x_ptr + offs_m[:, None] * stride_xm + (k0 + offs_k)[None, :] * stride_xk,
                    mask=m_mask[:, None], other=0.0)
        xf = x.to(tl.float32)
        sc = tl.load(mod_ptr + (k0 + offs_k))
        sh = tl.load(mod_ptr + K + (k0 + offs_k))
        a = xf * rstd[:, None] * (1.0 + sc[None, :]) + sh[None, :]
        aq = (a * inv[:, None]).to(tl.float8e4nv)  # [BM, BK] fp8
        wg = tl.load(g8t_ptr + (k0 + offs_k)[:, None] * stride_gwk + offs_n[None, :] * stride_gwn)
        gate = tl.dot(aq, wg, acc=gate)
        wu = tl.load(u8t_ptr + (k0 + offs_k)[:, None] * stride_uwk + offs_n[None, :] * stride_uwn)
        up = tl.dot(aq, wu, acc=up)
    # dequantise per-token x per-output-channel (scaling is exact after the dot:
    # aq = a * inv[m], so acc = (sum aq*wq) and gate = acc * sa[m] * sW[n])
    sa = amax / FP8_MAX
    gate = gate * sa[:, None] * gs[None, :]
    up = up * sa[:, None] * us[None, :]
    # F.gelu(x, approximate="tanh") = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3)))
    c = 0.7978845608028654  # sqrt(2/pi)
    t = tl.extra.cuda.libdevice.tanh(c * (gate + 0.044715 * gate * gate * gate))
    act = 0.5 * gate * (1.0 + t) * up
    actb = act.to(tl.bfloat16)
    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
             actb, mask=m_mask[:, None])
    # producer-fused act quant: this CTA owns columns [n0, n0+BN) of every row,
    # so emit the per-token partial max |act| of its own tile; a later cast
    # kernel reduces the n_tiles partials -> final per-token scale. Maxing the
    # bf16-rounded act keeps the chain bit-identical to the host-side quant.
    row_max = tl.max(tl.abs(actb).to(tl.float32), 1)
    tl.store(part_ptr + pid_n * stride_part + offs_m, row_max, mask=m_mask)


def triton_norm_gate_up_fp8(
    x: torch.Tensor,           # [M, K] bf16 pre-norm residual
    modulation: torch.Tensor,  # [3*K] f32
    gate_w: torch.Tensor,      # [N, K] bf16
    up_w: torch.Tensor,        # [N, K] bf16
    eps: float,
    out: torch.Tensor | None = None,  # [M, N] bf16
    *,
    block_n: int = 64,
    block_m: int = 64,
    block_k: int = 64,
    num_warps: int = 4,
    num_stages: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Producer kernel + per-token partial |act| max buffer for the fp8 down chain.

    Returns ``(out, part, n_tiles)``: ``out`` [M, N] bf16 activation, ``part``
    flat fp32 [n_tiles * rows_pad] partial row-max (one row per CTA per token,
    row index = pid_n * rows_pad + m), ``n_tiles`` = N / block_n.
    """
    M, K = x.shape
    N = gate_w.shape[0]
    if N % block_n or K % block_k:
        raise ValueError(f"fp8 kernel needs N%block_n==0 and K%block_k==0 "
                         f"(N={N}%{block_n}, K={K}%{block_k})")
    g8t, gs, _ = quantize_weight_rows_transposed(gate_w)  # [K, N] fp8
    u8t, us, _ = quantize_weight_rows_transposed(up_w)
    if out is None:
        out = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    n_tiles = triton.cdiv(N, block_n)
    rows_pad = triton.cdiv(M, block_m) * block_m
    part = torch.empty(n_tiles * rows_pad, dtype=torch.float32, device=x.device)
    _norm_gate_up_fp8_kernel[(n_tiles, triton.cdiv(M, block_m))](
        x, modulation, g8t, gs, u8t, us, out, part, rows_pad,
        M, K, N, float(eps),
        x.stride(0), x.stride(1),
        g8t.stride(0), g8t.stride(1),
        u8t.stride(0), u8t.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, FP8_MAX=FP8_MAX,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out, part, n_tiles


def quantize_weight_rows_transposed(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """bf16 [A, B] -> fp8 [B, A] (kernel B-operand layout) + per-A scale [A].

    Quantise per row of the linear weight (the output channel dim = dim 0 of
    ``w``), then transpose the fp8 result so a GEMM kernel can load its B
    operand directly in the ``[BK, BN]`` dot shape. The fp8 ``[B, A]`` copy is
    cached (the transpose+contiguous is otherwise a full extra pass over the
    weight on every call).
    """
    ptr = w.data_ptr()
    e = _wqt_cache.get(ptr)
    if e is None:
        wq, s, ref = quantize_weight_rows(w)  # fp8 [A, B] (also cached)
        wqt = wq.transpose(0, 1).contiguous()
        if len(_wqt_cache) > 64:
            _wqt_cache.clear()
        _wqt_cache[ptr] = (wqt, s, ref)
        return _wqt_cache[ptr]
    return e[0], e[1], e[2]


# --------------------------------------------------------------------------- #
# producer-fused fp8 down chain (no host-side activation round trip)
#
#   1. ``triton_norm_gate_up_fp8``      -- fused norm+gate/up+gelu writes the
#      bf16 act AND the per-CTA per-token partial max |act| (epilogue
#      byproduct, no extra pass over act);
#   2. ``triton_quant_act_fp8``           -- tiny kernel: reduce the n_tiles
#      partials to the final per-token scale, cast act -> fp8 (one read of the
#      act buffer, L2-warm);
#   3. ``triton_down_proj_fp8_splitk``    -- fp8 down GEMM split over K + fp32
#      partial merge; the K split avoids the latency-bound N-split shape at
#      M=50.
# --------------------------------------------------------------------------- #
@triton.jit
def _quant_act_fp8_kernel(
    act_ptr,        # [M, I] bf16 gelu(gate)*up (producer output)
    part_ptr,       # [N_TILES, rows_pad] f32 partial per-token max |act|
    a8_ptr,         # [M, I] fp8 quantised activation (out)
    sa_ptr,         # [M] f32 per-token activation scale (out)
    M,
    I,
    N_TILES,
    stride_part,    # rows_pad
    stride_am,
    stride_ak,
    stride_om,
    stride_ok,
    BLOCK_M: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_T: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    """Reduce the producer's per-token partial maxes, then cast act to fp8.

    One program per I-block: each one re-reduces the tiny [N_TILES, BM] partial
    buffer (L2-resident) to the final per-token scale, then quantises its own
    column slice of the bf16 act. N_TILES <= BLOCK_T (64 for BN=64) and
    I % BLOCK_I == 0 are enforced by the wrapper.
    """
    pid_i = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    offs_t = tl.arange(0, BLOCK_T)
    m_mask = offs_m < M

    part = tl.load(part_ptr + offs_t[:, None] * stride_part + offs_m[None, :],
                   mask=(offs_t[:, None] < N_TILES) & m_mask[None, :], other=0.0)
    amax = tl.max(part, 0)  # [BM] final per-token max over the full I row
    sa = amax / FP8_MAX
    inv = tl.where(amax == 0.0, 0.0, FP8_MAX / amax)  # per-token 1/sA
    tl.store(sa_ptr + offs_m, sa, mask=m_mask)

    a = tl.load(act_ptr + offs_m[:, None] * stride_am + offs_i[None, :] * stride_ak,
                mask=m_mask[:, None], other=0.0)
    aq = (a.to(tl.float32) * inv[:, None]).to(tl.float8e4nv)
    tl.store(a8_ptr + offs_m[:, None] * stride_om + offs_i[None, :] * stride_ok,
             aq, mask=m_mask[:, None])


def triton_quant_act_fp8(
    act: torch.Tensor,        # [M, I] bf16
    part: torch.Tensor,       # [n_tiles, rows_pad] f32 flat
    n_tiles: int,
    a8: torch.Tensor | None = None,  # [M, I] fp8 out
    sa: torch.Tensor | None = None,  # [M] f32 out
    *,
    block_i: int = 256,
    num_warps: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cast kernel: final per-token scale (from the producer partials) + fp8 act."""
    M, I = act.shape
    if part.numel() % n_tiles or I % block_i:
        raise ValueError("quant act needs part flat divisible by n_tiles and I%block_i==0")
    rows_pad = part.numel() // n_tiles
    if rows_pad < M or n_tiles > 64:
        raise ValueError(f"quant act: rows_pad {rows_pad} < M or n_tiles {n_tiles} > 64")
    if a8 is None:
        a8 = torch.empty((M, I), dtype=torch.float8_e4m3fn, device=act.device)
    if sa is None:
        sa = torch.empty(M, dtype=torch.float32, device=act.device)
    _quant_act_fp8_kernel[(triton.cdiv(I, block_i),)](
        act, part, a8, sa,
        M, I, n_tiles, rows_pad,
        act.stride(0), act.stride(1),
        a8.stride(0), a8.stride(1),
        BLOCK_M=64, BLOCK_I=block_i, BLOCK_T=64, FP8_MAX=FP8_MAX,
        num_warps=num_warps,
    )
    return a8, sa


@triton.jit
def _down_fp8_splitk_kernel(
    a8_ptr,         # [M, I] fp8 quantised activation
    w8t_ptr,        # [I, W] fp8 transposed down weight
    wks_ptr,        # [S, W/BN, BM, BN] fp32 partial sums
    M,
    I,
    W,
    S,
    Ks,             # I // S
    stride_am,
    stride_ak,
    stride_wk,
    stride_wn,
    n_tiles,        # W // BN
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Split-K partial: acc over the [Ks, BN] slice of one (n-tile, K split).

    I % S == 0, Ks % BLOCK_K == 0 and W % BLOCK_N == 0 are enforced by the
    wrapper, so the K loop and the weight loads need no masks; only the row
    mask survives.
    """
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    k0 = pid_s * Ks

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k in range(k0, k0 + Ks, BLOCK_K):
        a = tl.load(a8_ptr + offs_m[:, None] * stride_am + (k + offs_k)[None, :] * stride_ak,
                    mask=m_mask[:, None], other=0.0)
        w = tl.load(w8t_ptr + (k + offs_k)[:, None] * stride_wk + offs_n[None, :] * stride_wn)
        acc = tl.dot(a, w, acc=acc)
    base = wks_ptr + (pid_s * n_tiles + pid_n) * BLOCK_M * BLOCK_N
    tl.store(base + offs_m[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :],
             tl.where(m_mask[:, None], acc, 0.0))


@triton.jit
def _down_fp8_splitk_reduce_kernel(
    wks_ptr,        # [S, W/BN, BM, BN] fp32 partial sums
    sa_ptr,         # [M] f32 per-token activation scale
    ws_ptr,         # [W] f32 per-output-channel weight scale
    out_ptr,        # [M, W] bf16
    M,
    W,
    S,
    n_tiles,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Merge the S fp32 partials in fixed order, dequant, single bf16 rounding."""
    pid_n = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for s in range(0, S):
        base = wks_ptr + (s * n_tiles + pid_n) * BLOCK_M * BLOCK_N
        acc += tl.load(base + offs_m[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :])
    sa = tl.load(sa_ptr + offs_m, mask=m_mask, other=1.0)
    ws = tl.load(ws_ptr + offs_n)
    acc = acc * sa[:, None] * ws[None, :]
    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
             acc.to(tl.bfloat16), mask=m_mask[:, None])


def triton_down_proj_fp8_splitk(
    a8: torch.Tensor,         # [M, I] fp8 quantised activation
    sa: torch.Tensor,         # [M] f32 per-token activation scale
    down_w: torch.Tensor,     # [W, I] bf16 (quantised internally, cached)
    out: torch.Tensor | None = None,  # [M, W] bf16
    *,
    split_k: int = 4,
    block_n: int = 64,
    block_k: int = 128,
    num_warps: int = 4,
    num_stages: int = 3,
    wks: torch.Tensor | None = None,  # [S, W/BN, BM, BN] fp32 workspace
) -> torch.Tensor:
    """fp8 down GEMM with K split: [M, I] x [I, W], grid (W/BN, S).

    The K split parallelises the small-M GEMM, which is latency-bound as a
    plain N-split. Constraints are enforced by the wrapper.
    """
    M, I = a8.shape
    W = down_w.shape[0]
    S = split_k
    if I % S or (I // S) % block_k or W % block_n:
        raise ValueError(
            f"split-K down needs I%S==0, (I/S)%block_k==0, W%block_n==0 "
            f"(I={I}, S={S}, block_k={block_k}, W={W}, block_n={block_n})")
    w8t, ws, _ = quantize_weight_rows_transposed(down_w)  # [I, W] fp8
    n_tiles = W // block_n
    if out is None:
        out = torch.empty((M, W), dtype=torch.bfloat16, device=a8.device)
    if wks is None:
        wks = torch.empty((S, n_tiles, 64, block_n), dtype=torch.float32, device=a8.device)
    _down_fp8_splitk_kernel[(n_tiles, S)](
        a8, w8t, wks,
        M, I, W, S, I // S,
        a8.stride(0), a8.stride(1),
        w8t.stride(0), w8t.stride(1),
        n_tiles,
        BLOCK_M=64, BLOCK_N=block_n, BLOCK_K=block_k,
        num_warps=num_warps, num_stages=num_stages,
    )
    _down_fp8_splitk_reduce_kernel[(n_tiles,)](
        wks, sa, ws, out,
        M, W, S, n_tiles,
        out.stride(0), out.stride(1),
        BLOCK_M=64, BLOCK_N=block_n,
        num_warps=num_warps, num_stages=1,
    )
    return out


def triton_mlp_fp8_chain(
    x: torch.Tensor,           # [M, K] bf16 pre-norm residual
    modulation: torch.Tensor,  # [3*K] f32
    gate_w: torch.Tensor,      # [I, K] bf16
    up_w: torch.Tensor,        # [I, K] bf16
    down_w: torch.Tensor,      # [W, I] bf16
    eps: float,
    out: torch.Tensor | None = None,  # [M, W] bf16
    *,
    split_k: int = 4,
    block_n: int = 64,
    block_m: int = 64,
    block_k: int = 64,
    num_warps: int = 4,
    num_stages: int = 3,
) -> torch.Tensor:
    """Full fp8 expert MLP block: norm+gate/up+gelu -> fp8 act -> down GEMM.

    The activation quantisation is producer-fused: the gate/up epilogue emits
    the per-token partial maxes and a tiny cast kernel reduces them, so the
    bf16 act buffer is only ever read once (no host-side amax/div/cast chain).
    """
    act, part, n_tiles = triton_norm_gate_up_fp8(
        x, modulation, gate_w, up_w, eps,
        block_n=block_n, block_m=block_m, block_k=block_k,
        num_warps=num_warps, num_stages=num_stages,
    )
    a8, sa = triton_quant_act_fp8(act, part, n_tiles)
    return triton_down_proj_fp8_splitk(
        a8, sa, down_w, out, split_k=split_k,
        block_n=block_n, num_warps=num_warps, num_stages=num_stages,
    )
