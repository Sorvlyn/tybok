"""Self-contained expert attention for the pi0.5 denoising path with the fused
GEMM+RoPE qkv stage (``--tl-fused-expert``).

One public entry point with everything self-contained below it -- no operators
are imported from sibling model files.

Pipeline (3 kernels per layer):
1. ``triton_qkv_rope_fused`` -- ONE kernel: AdaRMS norm prologue (per-row fp32
   variance over K, folded in because the freshly written residual is
   L2-resident so the redundant variance pass is cheap L2 traffic) + packed
   q/k/v projection (q/k output columns interleaved ``(p, p+DH) -> (2p, 2p+1)``,
   v untouched) with the normalized tiles built inline + in-register epilogue
   RoPE on the BN=64 GEMM tile;
2. ``_splitk_flash_kernel`` -- split-K GQA flash over [interleaved prefix keys;
   interleaved suffix k] (the score dot is invariant under any bijection
   applied to both q and k);
3. ``_splitk_merge_kernel`` -- merge the partials, single bf16 rounding.

Layout contract (keep both ends consistent):
    qr [H,M,D], kr [M,D], pkT [D,Lp]  -- INTERLEAVED column order
    vr [M,D]                          -- normal order (never permuted)
    out [M, H*D]                      -- normal order

Numerics: bf16 in / fp32 accumulate / per-op bf16 rounding in the rotation,
matching the eager AdaRMS+RoPE recipe (drift tier -- validate with a tolerance).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# --------------------------------------------------------------------------- #
# host helpers: interleaved layouts (cached, data_ptr-keyed with strong refs)
# --------------------------------------------------------------------------- #
# ``weight.data`` returns a fresh python object per access, so id()-keyed
# caches miss under CUDA-graph capture and record pack ops INTO the graph;
# data_ptr keys + strong refs to the source tensors pin the storage instead.
# The LAYOUT tag keeps these caches disjoint from any normal-order cache.
LAYOUT = "interleaved_rope_v1"

_qkv_i_cache: dict[tuple[int, int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _interleave_rows(t: torch.Tensor, groups: int, d: int) -> torch.Tensor:
    """[groups*D, K] -> interleave each D-block's rows: row 2p <- p, 2p+1 <- p+D/2."""
    dh = d // 2
    g = t.reshape(groups, d, -1)
    a, b = g[:, :dh, :], g[:, dh:, :]
    return torch.stack((a, b), dim=-2).reshape(groups * d, -1)


def _packed_qkv_interleaved(q_proj, k_proj, v_proj, h: int, g: int, d: int) -> torch.Tensor:
    """Cached interleaved ``[q_i; k_i; v]`` weight, pre-transposed to ``[K, N]``.

    q/k rows are interleaved (dim=0, the output dim); v rows are untouched and
    form the trailing band. N = H*D + 2*G*D stays 2560.
    """
    wq, wk, wv = q_proj.weight.data, k_proj.weight.data, v_proj.weight.data
    key = (wq.data_ptr(), wk.data_ptr(), wv.data_ptr())
    entry = _qkv_i_cache.get(key)
    if entry is None:
        wq_i = _interleave_rows(wq, h, d)
        wk_i = _interleave_rows(wk, g, d)
        w = torch.cat([wq_i, wk_i, wv], dim=0).contiguous()
        w = w.transpose(0, 1).contiguous()  # [K, N]
        if len(_qkv_i_cache) > 64:
            _qkv_i_cache.clear()
        _qkv_i_cache[key] = (w, wq, wk, wv)
        return w
    return entry[0]


def _prefix_kT_interleaved(prefix_k: torch.Tensor, g: int, d: int) -> torch.Tensor:
    """``[Lp, G*D]`` -> interleaved per-group D dim -> ``[G*D, Lp]``.

    The pair axis must land between G and D: stack along the LAST axis, then
    reshape. stack(..., dim=-2) would flatten to [a-block, b-block] -- an
    identity permutation that silently defeats the interleaved contract.

    WARNING: do NOT cache this by ``prefix_k.data_ptr()`` (the earlier version
    did). ``prefix_k`` is an **activation** whose content changes per request and
    per pipeline stage, and a data_ptr-keyed cache does two bad things:
      * it can return the layout built from a previous request's keys, and
      * during CUDA-graph capture it *hides the pack ops from the graph* entirely,
        so every replay keeps reading that stale buffer.
    The second effect made the fused prefill x step-0 overlap path wrong
    (~5-9% per expert layer, 2.4e-1 on step-0 ``x1``).
    Constant weights are still cached; see :func:`_packed_qkv_interleaved`.
    """
    dh = d // 2
    pk = prefix_k.reshape(-1, g, d)
    a, b = pk[..., :dh], pk[..., dh:]  # slice the D dim of each group
    pk_i = torch.stack((a, b), dim=-1).reshape(pk.shape[0], g * d)
    return pk_i.transpose(0, 1).contiguous()


# --------------------------------------------------------------------------- #
# 1. fused AdaRMS norm + GEMM + epilogue RoPE
#    (grid = (N/BN,), band-specialised epilogue)
# --------------------------------------------------------------------------- #
@triton.jit
def _fused_gemm_rope_kernel(
    x_ptr,          # [M, K] bf16 pre-norm residual (raw, un-normalised)
    mod_ptr,        # [3*K] f32 AdaRMS modulation (scale, shift, gate), contiguous
    wt_ptr,         # [K, N] bf16 row-major (q/k interleaved, v normal)
    qr_ptr,         # [H, M, D] bf16 interleaved rotated q (out)
    kr_ptr,         # [M, D] bf16 interleaved rotated k (out)
    vr_ptr,         # [M, D] bf16 normal v (out)
    pos_ptr,        # [M] int64 suffix position ids
    cos_ptr,        # [max_pos, D] f32 cos table (normal columns)
    sin_ptr,        # [max_pos, D] f32 sin table
    M,
    K,
    N,
    EPS,
    stride_xm,
    stride_xk,
    stride_wk,
    stride_wn,
    stride_qr_h,
    stride_qr_m,
    stride_qr_d,
    stride_kr_m,
    stride_kr_d,
    stride_vr_m,
    stride_vr_d,
    stride_pos,
    stride_cos_p,
    stride_cos_d,
    H: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    VAR_BLOCK_K: tl.constexpr,
):
    """One 64-col GEMM tile per program; the tile always lies inside one band.

    AdaRMS norm is folded in as a prologue: each CTA computes the per-row fp32
    variance over the full K, then the GEMM mainloop builds each bf16 tile as
    ``round_bf16(x * rstd * (1+scale) + shift)`` (the exact single-rounding
    recipe of the eager norm) before the dot. The redundant variance pass reads
    only the L2-resident residual; ``VAR_BLOCK_K`` (>= BLOCK_K) keeps the
    variance loop short. Rows past M are zero (pad rows load other=0), so their
    variance is 0/eps and their (masked) stores are discarded.

    pid_n layout: [0, H*D/BN) q tiles (one head each), then (G*D)/BN k tiles,
    then (G*D)/BN v tiles. q/k epilogues rotate in the bf16 domain on the
    ``(2p, 2p+1)`` adjacent pair produced by the reshape/split; v is stored
    straight. cos/sin are gathered at the ORIGINAL pair columns (p, p+DH).
    """
    pid_n = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    n0 = pid_n * BLOCK_N
    m_mask = offs_m < M

    # ---- AdaRMS variance prologue (folded from the standalone norm kernel) ----
    offs_vk = tl.arange(0, VAR_BLOCK_K)
    sum_sq = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k0 in range(0, K, VAR_BLOCK_K):
        xv = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + (k0 + offs_vk)[None, :] * stride_xk,
            mask=m_mask[:, None],
            other=0.0,
            eviction_policy="evict_last",
        )
        xf = xv.to(tl.float32)
        sum_sq += tl.sum(xf * xf, 1)
    rstd = tl.rsqrt(sum_sq / K + EPS)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        xv = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + (k0 + offs_k)[None, :] * stride_xk,
            mask=m_mask[:, None],
            other=0.0,
            eviction_policy="evict_last",
        )
        xf = xv.to(tl.float32)
        sc = tl.load(mod_ptr + (k0 + offs_k))
        sh = tl.load(mod_ptr + K + (k0 + offs_k))
        a = (xf * rstd[:, None] * (1.0 + sc[None, :]) + sh[None, :]).to(tl.bfloat16)
        w = tl.load(
            wt_ptr + (k0 + offs_k)[:, None] * stride_wk + (n0 + tl.arange(0, BLOCK_N))[None, :] * stride_wn,
            eviction_policy="evict_first",
        )
        acc = tl.dot(a, w, acc=acc)
    acc_b = acc.to(tl.bfloat16)

    Q_CTAS = H * (D // BLOCK_N)
    K_CTAS = (G * D) // BLOCK_N

    if pid_n < Q_CTAS:  # ---- q head pid_n // (D/BN), 32-pair tile ----
        h = pid_n // (D // BLOCK_N)
        tile_in_head = pid_n % (D // BLOCK_N)
        pair_base = tile_in_head * (BLOCK_N // 2)
        offs_p = pair_base + tl.arange(0, BLOCK_N // 2)  # original column p
        pos = tl.load(pos_ptr + offs_m * stride_pos, mask=m_mask, other=0).to(tl.int64)
        c1 = tl.load(cos_ptr + pos[:, None] * stride_cos_p + offs_p[None, :] * stride_cos_d,
                     mask=m_mask[:, None], other=0.0).to(tl.bfloat16)
        s1 = tl.load(sin_ptr + pos[:, None] * stride_cos_p + offs_p[None, :] * stride_cos_d,
                     mask=m_mask[:, None], other=0.0).to(tl.bfloat16)
        c2 = tl.load(cos_ptr + pos[:, None] * stride_cos_p + (D // 2 + offs_p)[None, :] * stride_cos_d,
                     mask=m_mask[:, None], other=0.0).to(tl.bfloat16)
        s2 = tl.load(sin_ptr + pos[:, None] * stride_cos_p + (D // 2 + offs_p)[None, :] * stride_cos_d,
                     mask=m_mask[:, None], other=0.0).to(tl.bfloat16)
        first, second = tl.split(tl.reshape(acc_b, (BLOCK_M, BLOCK_N // 2, 2)))
        rot_first = first * c1 - second * s1
        rot_second = second * c2 + first * s2
        rot = tl.reshape(tl.join(rot_first, rot_second), (BLOCK_M, BLOCK_N))
        col = tile_in_head * BLOCK_N + tl.arange(0, BLOCK_N)
        tl.store(
            qr_ptr + h * stride_qr_h + offs_m[:, None] * stride_qr_m + col[None, :] * stride_qr_d,
            rot,
            mask=m_mask[:, None],
        )
    elif pid_n < Q_CTAS + K_CTAS:  # ---- k (shared kv group) ----
        tile_in_head = pid_n - Q_CTAS
        pair_base = tile_in_head * (BLOCK_N // 2)
        offs_p = pair_base + tl.arange(0, BLOCK_N // 2)
        pos = tl.load(pos_ptr + offs_m * stride_pos, mask=m_mask, other=0).to(tl.int64)
        c1 = tl.load(cos_ptr + pos[:, None] * stride_cos_p + offs_p[None, :] * stride_cos_d,
                     mask=m_mask[:, None], other=0.0).to(tl.bfloat16)
        s1 = tl.load(sin_ptr + pos[:, None] * stride_cos_p + offs_p[None, :] * stride_cos_d,
                     mask=m_mask[:, None], other=0.0).to(tl.bfloat16)
        c2 = tl.load(cos_ptr + pos[:, None] * stride_cos_p + (D // 2 + offs_p)[None, :] * stride_cos_d,
                     mask=m_mask[:, None], other=0.0).to(tl.bfloat16)
        s2 = tl.load(sin_ptr + pos[:, None] * stride_cos_p + (D // 2 + offs_p)[None, :] * stride_cos_d,
                     mask=m_mask[:, None], other=0.0).to(tl.bfloat16)
        first, second = tl.split(tl.reshape(acc_b, (BLOCK_M, BLOCK_N // 2, 2)))
        rot_first = first * c1 - second * s1
        rot_second = second * c2 + first * s2
        rot = tl.reshape(tl.join(rot_first, rot_second), (BLOCK_M, BLOCK_N))
        col = tile_in_head * BLOCK_N + tl.arange(0, BLOCK_N)
        tl.store(
            kr_ptr + offs_m[:, None] * stride_kr_m + col[None, :] * stride_kr_d,
            rot,
            mask=m_mask[:, None],
        )
    else:  # ---- v (no RoPE, normal order) ----
        tile_in_head = pid_n - Q_CTAS - K_CTAS
        col = tile_in_head * BLOCK_N + tl.arange(0, BLOCK_N)
        tl.store(
            vr_ptr + offs_m[:, None] * stride_vr_m + col[None, :] * stride_vr_d,
            acc_b,
            mask=m_mask[:, None],
        )


def triton_qkv_rope_fused(
    x: torch.Tensor,            # [M, K] bf16 pre-norm residual (raw)
    modulation: torch.Tensor,   # [3*K] f32 AdaRMS modulation (scale, shift, gate)
    eps: float,
    q_proj, k_proj, v_proj,
    position_ids: torch.Tensor,  # [M] int64
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    qr: torch.Tensor,           # [H, M, D] bf16 out (interleaved)
    kr: torch.Tensor,           # [M, D] bf16 out (interleaved)
    vr: torch.Tensor,           # [M, D] bf16 out (normal)
    *,
    block_n: int = 64,
    block_k: int = 64,
    var_block_k: int = 64,  # AdaRMS variance-prologue chunk
    num_warps: int = 4,
    num_stages: int = 4,
) -> None:
    """One kernel: AdaRMS norm prologue + packed q/k/v GEMM + epilogue RoPE."""
    M, K = x.shape
    H = q_proj.weight.shape[0] // 256
    G = k_proj.weight.shape[0] // 256
    D = 256
    N = H * D + 2 * G * D
    wt = _packed_qkv_interleaved(q_proj, k_proj, v_proj, H, G, D)
    _fused_gemm_rope_kernel[(triton.cdiv(N, block_n),)](
        x,
        modulation.reshape(-1),
        wt,
        qr,
        kr,
        vr,
        position_ids,
        cos_table,
        sin_table,
        M,
        K,
        N,
        float(eps),
        x.stride(0),
        x.stride(1),
        wt.stride(0),
        wt.stride(1),
        qr.stride(0),
        qr.stride(1),
        qr.stride(2),
        kr.stride(0),
        kr.stride(1),
        vr.stride(0),
        vr.stride(1),
        position_ids.stride(0),
        cos_table.stride(0),
        cos_table.stride(1),
        H=H,
        G=G,
        D=D,
        BLOCK_M=block_n,  # BM covers M; keep BN-symmetric tiles (64x64)
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        VAR_BLOCK_K=var_block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )


# --------------------------------------------------------------------------- #
# 3. split-K GQA flash + merge (M-split enabled)
# --------------------------------------------------------------------------- #
@triton.jit
def _splitk_flash_kernel(
    qr_ptr, kr_ptr, vr_ptr,
    pkT_ptr, pv_ptr, pad_ptr,
    m_ptr, l_ptr, acc_ptr,
    M, Lp, SCALE,
    stride_qr_h, stride_qr_m, stride_qr_d,
    stride_kr_m, stride_kr_d, stride_vr_m, stride_vr_d,
    stride_pk_l, stride_pk_d, stride_pv_l, stride_pv_d,
    stride_mb, stride_mh, stride_lb, stride_lh,
    stride_ab, stride_ah, stride_am, stride_ad,
    H: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_A: tl.constexpr, PREFIX_BLOCKS: tl.constexpr,
    BLOCK_MK: tl.constexpr,
):
    h = tl.program_id(0)
    ms = tl.program_id(1)
    b = tl.program_id(2)
    offs_m = ms * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    offs_a = tl.arange(0, BLOCK_A)
    m_mask = offs_m < M

    qr = tl.load(qr_ptr + h * stride_qr_h + offs_m[:, None] * stride_qr_m + offs_d[None, :] * stride_qr_d,
                 mask=m_mask[:, None], other=0.0)

    m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, D), tl.float32)

    if b < PREFIX_BLOCKS:
        # ceil split so no tail keys are dropped when Lp % PREFIX_BLOCKS != 0
        blk = (Lp + PREFIX_BLOCKS - 1) // PREFIX_BLOCKS
        start = b * blk
        end = min(start + blk, Lp)
        for p0 in range(start, end, BLOCK_A):
            offs_p = p0 + offs_a
            a_mask = offs_p < end
            pk = tl.load(pkT_ptr + offs_d[:, None] * stride_pk_l + offs_p[None, :] * stride_pk_d,
                         mask=a_mask[None, :], other=0.0)
            pv = tl.load(pv_ptr + offs_p[:, None] * stride_pv_l + offs_d[None, :] * stride_pv_d,
                         mask=a_mask[:, None], other=0.0)
            pad = tl.load(pad_ptr + offs_p, mask=a_mask, other=False)
            s = tl.dot(qr, pk) * SCALE
            s = tl.where(pad[None, :], s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            p = tl.where(s == float("-inf"), 0.0, tl.exp(s - m_new[:, None]))
            alpha = tl.where(m_i == float("-inf"), 0.0, tl.exp(m_i - m_new))
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), pv)
            m_i = m_new
    else:
        # suffix self-attention: the key/v rows are the FULL [0, M) token range,
        # NOT the query tile rows.  Under M-split the query tile only holds its
        # own row slice, so the keys need their own full-width BLOCK_MK tile
        # (BLOCK_MK >= M) with a separate mask.
        offs_kr = tl.arange(0, BLOCK_MK)
        km_mask = offs_kr < M
        kr = tl.load(kr_ptr + offs_kr[:, None] * stride_kr_m + offs_d[None, :] * stride_kr_d,
                     mask=km_mask[:, None], other=0.0)
        vr = tl.load(vr_ptr + offs_kr[:, None] * stride_vr_m + offs_d[None, :] * stride_vr_d,
                     mask=km_mask[:, None], other=0.0)
        s = tl.dot(qr, tl.trans(kr)) * SCALE
        s = tl.where(km_mask[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.exp(s - m_new[:, None])
        alpha = tl.where(m_i == float("-inf"), 0.0, tl.exp(m_i - m_new))
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), vr)
        m_i = m_new

    tl.store(m_ptr + b * stride_mb + h * stride_mh + offs_m, m_i, mask=m_mask)
    tl.store(l_ptr + b * stride_lb + h * stride_lh + offs_m, l_i, mask=m_mask)
    tl.store(acc_ptr + b * stride_ab + h * stride_ah + offs_m[:, None] * stride_am + offs_d[None, :] * stride_ad,
             acc, mask=m_mask[:, None])


@triton.jit
def _splitk_merge_kernel(
    m_ptr, l_ptr, acc_ptr, out_ptr,
    M, num_blocks,
    stride_om, stride_on,
    stride_mb, stride_mh, stride_lb, stride_lh,
    stride_ab, stride_ah, stride_am, stride_ad,
    H: tl.constexpr, D: tl.constexpr, BLOCK_M: tl.constexpr,
):
    h = tl.program_id(0)
    ms = tl.program_id(1)
    offs_m = ms * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    m_mask = offs_m < M

    m_i = tl.load(m_ptr + 0 * stride_mb + h * stride_mh + offs_m, mask=m_mask, other=float("-inf"))
    l_i = tl.load(l_ptr + 0 * stride_lb + h * stride_lh + offs_m, mask=m_mask, other=0.0)
    acc = tl.load(acc_ptr + 0 * stride_ab + h * stride_ah + offs_m[:, None] * stride_am + offs_d[None, :] * stride_ad,
                  mask=m_mask[:, None], other=0.0)

    for b in range(1, num_blocks):
        m_b = tl.load(m_ptr + b * stride_mb + h * stride_mh + offs_m, mask=m_mask, other=float("-inf"))
        l_b = tl.load(l_ptr + b * stride_lb + h * stride_lh + offs_m, mask=m_mask, other=0.0)
        acc_b = tl.load(acc_ptr + b * stride_ab + h * stride_ah + offs_m[:, None] * stride_am + offs_d[None, :] * stride_ad,
                        mask=m_mask[:, None], other=0.0)
        m_new = tl.maximum(m_i, m_b)
        alpha = tl.where(m_i == float("-inf"), 0.0, tl.exp(m_i - m_new))
        beta = tl.where(m_b == float("-inf"), 0.0, tl.exp(m_b - m_new))
        l_i = l_i * alpha + l_b * beta
        acc = acc * alpha[:, None] + acc_b * beta[:, None]
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(out_ptr + offs_m[:, None] * stride_om + (h * D + offs_d)[None, :] * stride_on,
             acc.to(tl.bfloat16), mask=m_mask[:, None])


# --------------------------------------------------------------------------- #
# 4. the fused attention pipeline
# --------------------------------------------------------------------------- #
def triton_expert_attn(
    hidden: torch.Tensor,          # [M, K] bf16 pre-norm residual
    modulation: torch.Tensor,      # [3*K] f32
    q_proj, k_proj, v_proj,
    prefix_k: torch.Tensor,        # [Lp, G*D] bf16 (normal order)
    prefix_v: torch.Tensor,        # [Lp, G*D] bf16
    prefix_pad_mask: torch.Tensor,  # [Lp] bool
    position_ids: torch.Tensor,    # [M] int64
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    eps: float,
    out: torch.Tensor | None = None,  # [M, H*D] bf16 pre-o_proj
    prefix_blocks: int = 4,
    block_m: int = 32,  # M-split: 2 row-tiles (bit-identical)
    block_k: int = 64,
    block_a: int = 64,
    flash_stages: int = 1,
    flash_warps: int = 8,
    gemm_warps: int = 4,
    gemm_stages: int = 4,
    scratch: dict | None = None,
) -> torch.Tensor:
    """Fused AdaRMS norm + (interleaved) qkv GEMM+RoPE -> split-K flash (drift tier)."""
    M, K = hidden.shape
    H = q_proj.weight.shape[0] // 256
    G = k_proj.weight.shape[0] // 256
    D = 256
    Lp = prefix_k.shape[0]
    num_blocks = prefix_blocks + 1

    dev = hidden.device
    if scratch is None:
        scratch = {}
    if "qr" not in scratch:
        scratch.update(
            qr=torch.empty(H, M, D, dtype=torch.bfloat16, device=dev),
            kr=torch.empty(M, D, dtype=torch.bfloat16, device=dev),
            vr=torch.empty(M, D, dtype=torch.bfloat16, device=dev),
            m=torch.empty(num_blocks, H, M, dtype=torch.float32, device=dev),
            l=torch.empty(num_blocks, H, M, dtype=torch.float32, device=dev),
            acc=torch.empty(num_blocks, H, M, D, dtype=torch.float32, device=dev),
        )
    qr = scratch["qr"]
    kr = scratch["kr"]
    vr = scratch["vr"]
    m_part = scratch["m"]
    l_part = scratch["l"]
    acc_part = scratch["acc"]
    if out is None:
        out = torch.empty(M, H * D, dtype=torch.bfloat16, device=dev)

    triton_qkv_rope_fused(hidden, modulation, eps, q_proj, k_proj, v_proj, position_ids,
                          cos_table, sin_table, qr, kr, vr, block_k=block_k,
                          num_warps=gemm_warps, num_stages=gemm_stages)

    pkT = _prefix_kT_interleaved(prefix_k, G, D)
    # grid: (H, M-splits, num_blocks).  M-split is exact: every row's online
    # softmax chain runs over the same key blocks in the same order, so the
    # per-row math is bit-identical to the unsplit kernel; it only halves the
    # register-hungry fp32 accumulator per CTA (acc [BLOCK_M, D]).
    _splitk_flash_kernel[(H, triton.cdiv(M, block_m), num_blocks)](
        qr, kr, vr, pkT, prefix_v, prefix_pad_mask,
        m_part, l_part, acc_part,
        M, Lp, D ** -0.5,
        qr.stride(0), qr.stride(1), qr.stride(2),
        kr.stride(0), kr.stride(1), vr.stride(0), vr.stride(1),
        pkT.stride(0), pkT.stride(1), prefix_v.stride(0), prefix_v.stride(1),
        m_part.stride(0), m_part.stride(1), l_part.stride(0), l_part.stride(1),
        acc_part.stride(0), acc_part.stride(1), acc_part.stride(2), acc_part.stride(3),
        H=H, D=D, BLOCK_M=block_m, BLOCK_A=block_a, PREFIX_BLOCKS=prefix_blocks,
        BLOCK_MK=triton.next_power_of_2(M),
        num_warps=flash_warps, num_stages=flash_stages,
    )
    _splitk_merge_kernel[(H, triton.cdiv(M, block_m))](
        m_part, l_part, acc_part, out,
        M, num_blocks,
        out.stride(0), out.stride(1),
        m_part.stride(0), m_part.stride(1), l_part.stride(0), l_part.stride(1),
        acc_part.stride(0), acc_part.stride(1), acc_part.stride(2), acc_part.stride(3),
        H=H, D=D, BLOCK_M=block_m,
        num_warps=8, num_stages=1,
    )
    return out
