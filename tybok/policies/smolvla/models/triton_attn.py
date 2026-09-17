"""Triton fused GQA attention for the denoising path (``--tl-fused-expert``).

One kernel per layer type replaces the eager attention chain of a denoising
step (``forward_attn_layer`` / ``forward_cross_attn_layer`` with
``fill_kv_cache=False``), fusing:

- input RMSNorm (single-pass: norm weight folded into the dot input, rstd
  applied to the accumulated fp32 results),
- the q (self: also k/v) projections (bf16 GEMMs, fp32 accumulate),
- RoPE on q and the suffix k (fp32 half-split rotation, gathered from the
  precomputed sin/cos tables),
- the suffix K/V buffer write (nothing else reads it back: the suffix keys are
  consumed inside this kernel),
- the GQA ``expand -> reshape`` materialisation,
- the boolean 2-D mask construction and the fp32 softmax.

The kernels attend over ``[prefix KV; suffix k/v]`` (self) or the expert prefix
KV (cross) with flash-style online softmax, GQA handled natively (one program
per query head, kv heads shared by ``num_heads // num_kv_heads`` heads).

The LLM prefill gets the same treatment from ``_gqa_prefill_qkv_kernel`` +
``_gqa_prefill_kernel`` (``triton_prefill_layer``): fused norm+qkv+RoPE writing
the prefix K/V into the static cache and the query into a scratch buffer, then
the GQA flash kernel attending over the cache.

Numerics drift from the fp32 eager reference (Triton reduction order,
``tl.exp`` (exp2-based) and TF32 scores), so the output is validated with the
chunk tolerance (like the other ``--tl-fused-*`` tiers); the eager/bit-exact path
is untouched.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .rope import _get_rope_tables

_MAX_WAVELENGTH = 10_000.0


@triton.jit
def _gqa_self_attn_kernel(
    hidden_ptr,
    wq_ptr,
    wk_ptr,
    wv_ptr,
    pk_ptr,
    pv_ptr,
    sin_ptr,
    cos_ptr,
    pos_ptr,
    pad_ptr,
    norm_w_ptr,
    out_ptr,
    K,
    PREFIX,
    H,
    G,
    D,
    SUFFIX,
    SCALE,
    EPS,
    stride_hm,
    stride_hk,
    stride_wq_n,
    stride_wq_k,
    stride_wk_n,
    stride_wk_k,
    stride_wv_n,
    stride_wv_k,
    stride_pk_l,
    stride_pk_g,
    stride_pk_d,
    stride_pv_l,
    stride_pv_g,
    stride_pv_d,
    stride_pos,
    stride_pad,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_A: tl.constexpr,
    DH: tl.constexpr,
):
    """One program per query head ``h`` (kv group ``g = h // (H // G)``)."""

    h = tl.program_id(0)
    g = h // (H // G)

    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)
    m_mask = offs_m < SUFFIX

    # ---- fused input_layernorm (RMSNorm) + q/k/v projections ----------------
    # The dot is linear in the per-row scale, so fold the norm weight into the
    # dot input (bf16, like the eager norm output) and apply the rstd scale to
    # the accumulated fp32 results -- single pass over K. The variance is over
    # the *raw* hidden (``variance = x.pow(2).mean(-1)``), accumulated in fp32.
    sum_sq = tl.zeros((BLOCK_M,), tl.float32)
    q = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    ks = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    vs = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        a = tl.load(
            hidden_ptr + offs_m[:, None] * stride_hm + (k0 + offs_k)[None, :] * stride_hk,
            mask=m_mask[:, None] & ((k0 + offs_k)[None, :] < K),
            other=0.0,
        )
        af = a.to(tl.float32)
        sum_sq += tl.sum(af * af, 1)
        wn = tl.load(norm_w_ptr + (k0 + offs_k), mask=(k0 + offs_k)[:] < K, other=0.0)
        a_w = (af * wn[None, :]).to(tl.bfloat16)
        bq = tl.load(
            wq_ptr + (h * BLOCK_D + offs_d)[None, :] * stride_wq_n + (k0 + offs_k)[:, None] * stride_wq_k,
            mask=(k0 + offs_k)[:, None] < K,
            other=0.0,
        )
        q = tl.dot(a_w, bq, acc=q)
        bk = tl.load(
            wk_ptr + (g * BLOCK_D + offs_d)[None, :] * stride_wk_n + (k0 + offs_k)[:, None] * stride_wk_k,
            mask=(k0 + offs_k)[:, None] < K,
            other=0.0,
        )
        ks = tl.dot(a_w, bk, acc=ks)
        bv = tl.load(
            wv_ptr + (g * BLOCK_D + offs_d)[None, :] * stride_wv_n + (k0 + offs_k)[:, None] * stride_wv_k,
            mask=(k0 + offs_k)[:, None] < K,
            other=0.0,
        )
        vs = tl.dot(a_w, bv, acc=vs)
    rstd = tl.rsqrt(sum_sq / K + EPS)
    q = q * rstd[:, None]
    ks = ks * rstd[:, None]
    vs = vs * rstd[:, None]

    # ---- RoPE (fp32 half-split, sin/cos gathered from the tables) ----
    pos = tl.load(pos_ptr + offs_m * stride_pos, mask=m_mask, other=0).to(tl.int64)
    offs_h = tl.arange(0, DH)
    sin = tl.load(sin_ptr + pos[:, None] * DH + offs_h[None, :], mask=m_mask[:, None], other=0.0)
    cos = tl.load(cos_ptr + pos[:, None] * DH + offs_h[None, :], mask=m_mask[:, None], other=0.0)

    q = tl.where(m_mask[:, None], q, 0.0)
    ks = tl.where(m_mask[:, None], ks, 0.0)
    vs = tl.where(m_mask[:, None], vs, 0.0)

    # half-split RoPE (reference layout: pair (x[i], x[i+DH]), NOT adjacent)
    q3 = tl.reshape(q, (BLOCK_M, 2, DH))
    qt = tl.trans(q3, (0, 2, 1))
    q1, q2 = tl.split(qt)  # q1 = q[:, :DH], q2 = q[:, DH:]
    qr1 = q1 * cos - q2 * sin
    qr2 = q2 * cos + q1 * sin
    q = tl.reshape(tl.trans(tl.join(qr1, qr2), (0, 2, 1)), (BLOCK_M, BLOCK_D))
    # same for the suffix k
    k3 = tl.reshape(ks, (BLOCK_M, 2, DH))
    kt = tl.trans(k3, (0, 2, 1))
    k1, k2 = tl.split(kt)
    kr1 = k1 * cos - k2 * sin
    kr2 = k2 * cos + k1 * sin
    ks = tl.reshape(tl.trans(tl.join(kr1, kr2), (0, 2, 1)), (BLOCK_M, BLOCK_D))
    # Round q and the suffix k to bf16 like the eager path (``apply_rope``
    # casts back to bf16, and the suffix K/V is written to the bf16 cache),
    # then upcast for the fp32 scores: the kernel's score inputs are then the
    # same values the eager reference feeds its fp32 matmul. The softmax output
    # is cast to bf16 before the value matmul, mirroring the eager's
    # ``probs.to(value_states.dtype)``.
    q = q.to(tl.bfloat16).to(tl.float32)
    kb = ks.to(tl.bfloat16).to(tl.float32)

    # ---- attention over [prefix KV; suffix k/v], flash-style online softmax ----
    m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    for p0 in range(0, PREFIX, BLOCK_A):
        offs_a = p0 + tl.arange(0, BLOCK_A)
        a_mask = offs_a < PREFIX
        pk = tl.load(
            pk_ptr + offs_a[:, None] * stride_pk_l + g * stride_pk_g + offs_d[None, :] * stride_pk_d,
            mask=a_mask[:, None],
            other=0.0,
        )
        pv = tl.load(
            pv_ptr + offs_a[:, None] * stride_pv_l + g * stride_pv_g + offs_d[None, :] * stride_pv_d,
            mask=a_mask[:, None],
            other=0.0,
        )
        pad = tl.load(pad_ptr + offs_a * stride_pad, mask=a_mask, other=False)
        # TF32 tensor-core dot: the eager reference upcasts to fp32, but a true
        # fp32 (ieee) dot has no tensor-core path on Ada; TF32 keeps a small
        # relative score error (10-bit mantissa).
        s = tl.dot(q, tl.trans(pk.to(tl.float32)), input_precision="tf32") * SCALE
        s = tl.where(pad[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), pv)
        m_i = m_new

    for b0 in range(0, SUFFIX, BLOCK_A):
        offs_a = b0 + tl.arange(0, BLOCK_A)
        a_mask = offs_a < SUFFIX
        s = tl.dot(q, tl.trans(kb), input_precision="tf32") * SCALE
        # The reference's suffix 2-D mask (``make_att_2d_masks``) is causal:
        # query i attends suffix keys j <= i, not the full suffix. With scores
        # up to ~20 the peaked softmax amplifies any extra attended key, so
        # this mask is required for numerical agreement.
        s = tl.where(a_mask[None, :] & (offs_a[None, :] <= offs_m[:, None]), s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), vs.to(tl.bfloat16))
        m_i = m_new

    acc = acc / l_i[:, None]
    acc = tl.where(m_mask[:, None], acc, 0.0)
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + (h * BLOCK_D + offs_d)[None, :] * stride_on,
        acc.to(tl.bfloat16),
        mask=m_mask[:, None],
    )


@triton.jit
def _gqa_prefill_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    mask_ptr,
    out_ptr,
    L,
    H,
    G,
    D,
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
    BLOCK_M: tl.constexpr,
    BLOCK_A: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """GQA-native flash attention for the LLM prefill.

    Replaces the bf16 SDPA path's materialised ``expand -> reshape`` GQA
    expansion with native kv-group sharing, plus the flash-style blocked online
    softmax. grid = (H, M blocks). Fully-masked rows softmax to a uniform
    distribution like the eager reference's big_neg softmax (their outputs are
    masked out downstream either way). Note: the output pointer uses ``h * D``
    for the head stride (not ``out.stride(2)``) -- the out layout is
    ``[B, L, H*D]``.
    """
    h = tl.program_id(0)
    pid_m = tl.program_id(1)
    g = h // (H // G)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    m_mask = offs_m < L

    q = tl.load(
        q_ptr + offs_m[:, None] * stride_qm + h * stride_qh + offs_d[None, :] * stride_qd,
        mask=m_mask[:, None],
        other=0.0,
    ).to(tl.float32)

    m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    for a0 in range(0, L, BLOCK_A):
        offs_a = a0 + tl.arange(0, BLOCK_A)
        a_mask = offs_a < L
        kk = tl.load(
            k_ptr + offs_a[:, None] * stride_km + g * stride_kg + offs_d[None, :] * stride_kd,
            mask=a_mask[:, None],
            other=0.0,
        )
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
        s = tl.dot(q, tl.trans(kk.to(tl.float32)), input_precision="tf32") * SCALE
        # big_neg (not -inf) like the eager reference: fully-masked rows then
        # softmax to a uniform distribution instead of NaN (the reference's
        # ``torch.where(mask, w, finfo.min)`` + softmax semantics).
        s = tl.where(mk, s, tl.full((), -3.4028234663852886e38, tl.float32))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), vv)
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + h * stride_oh + offs_d[None, :] * stride_od,
        acc.to(tl.bfloat16),
        mask=m_mask[:, None],
    )


@triton.jit
def _gqa_prefill_qkv_kernel(
    hidden_ptr,
    norm_w_ptr,
    wq_ptr,
    wk_ptr,
    wv_ptr,
    sin_ptr,
    cos_ptr,
    pos_ptr,
    kc_ptr,
    vc_ptr,
    q_ptr,
    L,
    H,
    G,
    D,
    K,
    EPS,
    stride_hm,
    stride_hk,
    stride_wq_n,
    stride_wq_k,
    stride_wk_n,
    stride_wk_k,
    stride_wv_n,
    stride_wv_k,
    stride_kc_l,
    stride_kc_g,
    stride_kc_d,
    stride_vc_l,
    stride_vc_g,
    stride_vc_d,
    stride_qm,
    stride_qh,
    stride_qd,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    DH: tl.constexpr,
):
    """Fused prefill norm+qkv+RoPE+KV-write (one program per (query head ``h``,
    M block); kv group ``g = h // (H // G)``).

    Replaces the eager prefill chain ``input_layernorm -> q/k/v projections ->
    apply_rope -> KV cache fill`` with one kernel: the RMSNorm uses the
    linearity trick (norm weight folded into the dot input, rstd applied to the
    fp32 results), q is projected per head, k/v per kv group (computed
    redundantly by the group's heads, like the denoise kernels), RoPE is the
    fp32 half-split gather, and the post-RoPE k/v (bf16, like the eager cache)
    are written into the static KV cache -- which ``_gqa_prefill_kernel`` then
    attends over. q is written to a scratch buffer (bf16, same rounding as
    ``apply_rope``'s cast back).
    """
    h = tl.program_id(0)
    pid_m = tl.program_id(1)
    g = h // (H // G)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)
    m_mask = offs_m < L

    # ---- fused RMSNorm + q/k/v projections (single pass) ----
    sum_sq = tl.zeros((BLOCK_M,), tl.float32)
    q = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    ks = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    vs = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        km = (k0 + offs_k) < K
        a = tl.load(
            hidden_ptr + offs_m[:, None] * stride_hm + (k0 + offs_k)[None, :] * stride_hk,
            mask=m_mask[:, None] & km[None, :],
            other=0.0,
        )
        af = a.to(tl.float32)
        sum_sq += tl.sum(af * af, 1)
        wn = tl.load(norm_w_ptr + (k0 + offs_k), mask=km, other=0.0)
        a_w = (af * wn[None, :]).to(tl.bfloat16)
        bq = tl.load(
            wq_ptr + (h * BLOCK_D + offs_d)[None, :] * stride_wq_n + (k0 + offs_k)[:, None] * stride_wq_k,
            mask=km[:, None],
            other=0.0,
        )
        q = tl.dot(a_w, bq, acc=q)
        bk = tl.load(
            wk_ptr + (g * BLOCK_D + offs_d)[None, :] * stride_wk_n + (k0 + offs_k)[:, None] * stride_wk_k,
            mask=km[:, None],
            other=0.0,
        )
        ks = tl.dot(a_w, bk, acc=ks)
        bv = tl.load(
            wv_ptr + (g * BLOCK_D + offs_d)[None, :] * stride_wv_n + (k0 + offs_k)[:, None] * stride_wv_k,
            mask=km[:, None],
            other=0.0,
        )
        vs = tl.dot(a_w, bv, acc=vs)
    rstd = tl.rsqrt(sum_sq / K + EPS)
    q = q * rstd[:, None]
    ks = ks * rstd[:, None]
    vs = vs * rstd[:, None]

    # ---- RoPE (fp32 half-split, gathered from the tables) ----
    pos = tl.load(pos_ptr + offs_m, mask=m_mask, other=0).to(tl.int64)
    offs_h = tl.arange(0, DH)
    sin = tl.load(sin_ptr + pos[:, None] * DH + offs_h[None, :], mask=m_mask[:, None], other=0.0)
    cos = tl.load(cos_ptr + pos[:, None] * DH + offs_h[None, :], mask=m_mask[:, None], other=0.0)
    q = tl.where(m_mask[:, None], q, 0.0)
    ks = tl.where(m_mask[:, None], ks, 0.0)
    vs = tl.where(m_mask[:, None], vs, 0.0)
    q3 = tl.reshape(q, (BLOCK_M, 2, DH))
    qt = tl.trans(q3, (0, 2, 1))
    q1, q2 = tl.split(qt)
    q = tl.reshape(tl.trans(tl.join(q1 * cos - q2 * sin, q2 * cos + q1 * sin), (0, 2, 1)), (BLOCK_M, BLOCK_D))
    k3 = tl.reshape(ks, (BLOCK_M, 2, DH))
    kt = tl.trans(k3, (0, 2, 1))
    k1, k2 = tl.split(kt)
    ks = tl.reshape(tl.trans(tl.join(k1 * cos - k2 * sin, k2 * cos + k1 * sin), (0, 2, 1)), (BLOCK_M, BLOCK_D))
    # bf16 rounding like the eager path (``apply_rope`` casts back to bf16;
    # the KV cache is bf16)
    qb = q.to(tl.bfloat16)
    kb = ks.to(tl.bfloat16)
    vb = vs.to(tl.bfloat16)

    # ---- write k/v into the static KV cache, q into the attention scratch ----
    tl.store(
        kc_ptr + offs_m[:, None] * stride_kc_l + g * stride_kc_g + offs_d[None, :] * stride_kc_d,
        kb,
        mask=m_mask[:, None],
    )
    tl.store(
        vc_ptr + offs_m[:, None] * stride_vc_l + g * stride_vc_g + offs_d[None, :] * stride_vc_d,
        vb,
        mask=m_mask[:, None],
    )
    tl.store(
        q_ptr + offs_m[:, None] * stride_qm + h * stride_qh + offs_d[None, :] * stride_qd,
        qb,
        mask=m_mask[:, None],
    )


def triton_prefill_layer(
    hidden, norm_w, eps, wq, wk, wv, positions, mask2d, cache, layer_idx, head_dim: int, out=None
):
    """Fused LLM prefill layer attention (``--tl-llm-fused-attn``).

    Replaces the eager prefill chain for one layer -- ``input_layernorm`` +
    q/k/v projections + ``apply_rope`` + KV-cache fill + GQA flash attention --
    with two kernels: ``_gqa_prefill_qkv_kernel`` (norm+qkv+RoPE, writing the
    prefix K/V into the static cache and q into a scratch buffer) and
    ``_gqa_prefill_kernel`` (the existing GQA flash kernel, attending over the
    cache). The K/V written here is exactly what the denoising cross/self
    attention reads later.

    Args:
        hidden: ``[1, L, K]`` raw (pre-``input_layernorm``) prefix hidden
            (layer 0 is fp32 -- the type-promoted embedding -- the kernel
            normalises it in fp32 before projecting).
        norm_w: ``[K]`` input-layernorm weight, eps: its epsilon.
        wq/wk/wv: projection weights ``[H*D, K]`` / ``[G*D, K]`` bf16.
        positions: ``[1, L]`` int64 prefix positions.
        mask2d: ``[L, L]`` bool 2-D attention mask (``make_att_2d_masks``).
        cache: the ``KVCache`` (k/v are written in-kernel; buffers are
            allocated on first use).

    Returns ``[1, L, H*D]`` bf16 (o_proj input), same layout as the eager path.
    """
    L, K = hidden.shape[1], hidden.shape[2]
    H = wq.shape[0] // head_dim
    G = wk.shape[0] // head_dim
    D = head_dim
    dev = hidden.device
    if cache.key_buf is None:
        # Lazily allocate the cache buffers (validate/eager paths). The dummy
        # values are overwritten by the kernel below.
        dummy = torch.empty(1, L, G, D, dtype=torch.bfloat16, device=dev)
        cache.fill(layer_idx, dummy, dummy)
    else:
        cache.prefix_len = L
        cache.fill_count += 1
    kc = cache.key_buf[layer_idx, :, :L]
    vc = cache.value_buf[layer_idx, :, :L]
    if out is None:
        out = torch.empty((1, L, H * D), dtype=torch.bfloat16, device=dev)
    q_scratch = torch.empty((1, L, H * D), dtype=torch.bfloat16, device=dev)
    sin_tab, cos_tab = _get_rope_tables(D // 2, dev, _MAX_WAVELENGTH)
    pos = positions[0] if positions.shape[0] == 1 else positions  # [L]
    scale = D**-0.5
    _gqa_prefill_qkv_kernel[(H, triton.cdiv(L, 64))](
        hidden,
        norm_w,
        wq,
        wk,
        wv,
        sin_tab,
        cos_tab,
        pos,
        kc,
        vc,
        q_scratch,
        L,
        H,
        G,
        D,
        K,
        float(eps),
        hidden.stride(1),
        hidden.stride(2),
        wq.stride(0),
        wq.stride(1),
        wk.stride(0),
        wk.stride(1),
        wv.stride(0),
        wv.stride(1),
        kc.stride(1),
        kc.stride(2),
        kc.stride(3),
        vc.stride(1),
        vc.stride(2),
        vc.stride(3),
        q_scratch.stride(1),
        D,
        q_scratch.stride(2),
        BLOCK_M=64,
        BLOCK_K=64,
        BLOCK_D=64,
        DH=32,
        num_warps=4,
    )
    _gqa_prefill_kernel[(H, triton.cdiv(L, 64))](
        q_scratch,
        kc,
        vc,
        mask2d,
        out,
        L,
        H,
        G,
        D,
        scale,
        q_scratch.stride(1),
        D,
        q_scratch.stride(2),
        kc.stride(1),
        kc.stride(2),
        kc.stride(3),
        vc.stride(1),
        vc.stride(2),
        vc.stride(3),
        mask2d.stride(0),
        mask2d.stride(1),
        out.stride(1),
        D,
        out.stride(2),
        BLOCK_M=64,
        BLOCK_A=64,
        BLOCK_D=64,
        num_warps=4,
    )
    return out


def triton_self_attn(hidden, norm_w, eps, wq, wk, wv, prefix_k, prefix_v, positions, prefix_pad, head_dim: int, out=None):
    """Fused RMSNorm + GQA self-attention for one denoising self-attention layer.

    Args:
        hidden: ``[S, K]`` raw (pre-``input_layernorm``) hidden, bf16 (or fp32
            for layer 0), contiguous.
        norm_w: ``[K]`` input-layernorm weight, eps: its epsilon.
        wq/wk/wv: projection weights ``[Hq*D, K]`` / ``[G*D, K]`` bf16.
        prefix_k/prefix_v: ``[P, G, D]`` bf16 prefix KV (static buffer slice).
        positions: ``[S]`` int64 denoising suffix positions.
        prefix_pad: ``[P]`` bool prefix padding mask (all-True in deployment).
        head_dim: ``D``.

    Returns ``[S, Hq*D]`` bf16 (o_proj input), same layout as the eager path.
    """
    S, K = hidden.shape
    P = prefix_k.shape[0]
    Hq = wq.shape[0] // head_dim
    G = wk.shape[0] // head_dim
    D = head_dim
    scale = D**-0.5
    if out is None:
        out = torch.empty((S, Hq * D), dtype=torch.bfloat16, device=hidden.device)
    sin_tab, cos_tab = _get_rope_tables(D // 2, hidden.device, _MAX_WAVELENGTH)
    _gqa_self_attn_kernel[(Hq,)](
        hidden,
        wq,
        wk,
        wv,
        prefix_k,
        prefix_v,
        sin_tab,
        cos_tab,
        positions,
        prefix_pad,
        norm_w,
        out,
        K,
        P,
        Hq,
        G,
        D,
        S,
        scale,
        float(eps),
        hidden.stride(0),
        hidden.stride(1),
        wq.stride(0),
        wq.stride(1),
        wk.stride(0),
        wk.stride(1),
        wv.stride(0),
        wv.stride(1),
        prefix_k.stride(0),
        prefix_k.stride(1),
        prefix_k.stride(2),
        prefix_v.stride(0),
        prefix_v.stride(1),
        prefix_v.stride(2),
        positions.stride(0),
        prefix_pad.stride(0),
        out.stride(0),
        out.stride(1),
        BLOCK_M=64,
        BLOCK_K=64,
        BLOCK_D=64,
        BLOCK_A=64,
        DH=32,
    )
    return out


@triton.jit
def _gqa_cross_attn_kernel(
    hidden_ptr,
    wq_ptr,
    ek_ptr,
    ev_ptr,
    sin_ptr,
    cos_ptr,
    pos_ptr,
    pad_ptr,
    norm_w_ptr,
    out_ptr,
    K,
    PREFIX,
    H,
    G,
    D,
    SUFFIX,
    SCALE,
    EPS,
    stride_hm,
    stride_hk,
    stride_wq_n,
    stride_wq_k,
    stride_ek_l,
    stride_ek_g,
    stride_ek_d,
    stride_ev_l,
    stride_ev_g,
    stride_ev_d,
    stride_pos,
    stride_pad,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_A: tl.constexpr,
    DH: tl.constexpr,
):
    """One program per (query head ``h``, M block) -- kv group ``g = h // (H // G)``.

    Cross-attention attends over the *expert* prefix KV (fp32 projections of the
    VLM prefix KV, loop-invariant and cached per inference), so this kernel only
    projects q, applies RoPE and runs the GQA attention over the prefix -- no
    suffix K/V at all. Each program's q rows attend over the full shared
    prefix, so no redundant work is introduced.
    """

    h = tl.program_id(0)
    pid_m = tl.program_id(1)
    g = h // (H // G)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)
    m_mask = offs_m < SUFFIX

    # ---- fused input_layernorm (RMSNorm) + q projection (single pass, see
    # the self-attention kernel for the linearity argument) ----
    sum_sq = tl.zeros((BLOCK_M,), tl.float32)
    q = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        a = tl.load(
            hidden_ptr + offs_m[:, None] * stride_hm + (k0 + offs_k)[None, :] * stride_hk,
            mask=m_mask[:, None] & ((k0 + offs_k)[None, :] < K),
            other=0.0,
        )
        af = a.to(tl.float32)
        sum_sq += tl.sum(af * af, 1)
        wn = tl.load(norm_w_ptr + (k0 + offs_k), mask=(k0 + offs_k)[:] < K, other=0.0)
        a_w = (af * wn[None, :]).to(tl.bfloat16)
        bq = tl.load(
            wq_ptr + (h * BLOCK_D + offs_d)[None, :] * stride_wq_n + (k0 + offs_k)[:, None] * stride_wq_k,
            mask=(k0 + offs_k)[:, None] < K,
            other=0.0,
        )
        q = tl.dot(a_w, bq, acc=q)
    rstd = tl.rsqrt(sum_sq / K + EPS)
    q = q * rstd[:, None]

    # ---- RoPE (fp32 half-split, positions are 0-based for cross layers) ----
    pos = tl.load(pos_ptr + offs_m * stride_pos, mask=m_mask, other=0).to(tl.int64)
    offs_h = tl.arange(0, DH)
    sin = tl.load(sin_ptr + pos[:, None] * DH + offs_h[None, :], mask=m_mask[:, None], other=0.0)
    cos = tl.load(cos_ptr + pos[:, None] * DH + offs_h[None, :], mask=m_mask[:, None], other=0.0)
    q = tl.where(m_mask[:, None], q, 0.0)
    q3 = tl.reshape(q, (BLOCK_M, 2, DH))
    qt = tl.trans(q3, (0, 2, 1))
    q1, q2 = tl.split(qt)
    q = tl.reshape(tl.trans(tl.join(q1 * cos - q2 * sin, q2 * cos + q1 * sin), (0, 2, 1)), (BLOCK_M, BLOCK_D))
    # same bf16 rounding as the eager path (``apply_rope`` casts back to bf16)
    q = q.to(tl.bfloat16).to(tl.float32)

    # ---- attention over the expert prefix KV (fp32), flash-style online softmax ----
    m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    for p0 in range(0, PREFIX, BLOCK_A):
        offs_a = p0 + tl.arange(0, BLOCK_A)
        a_mask = offs_a < PREFIX
        ek = tl.load(
            ek_ptr + offs_a[:, None] * stride_ek_l + g * stride_ek_g + offs_d[None, :] * stride_ek_d,
            mask=a_mask[:, None],
            other=0.0,
        )  # fp32
        ev = tl.load(
            ev_ptr + offs_a[:, None] * stride_ev_l + g * stride_ev_g + offs_d[None, :] * stride_ev_d,
            mask=a_mask[:, None],
            other=0.0,
        )  # fp32
        pad = tl.load(pad_ptr + offs_a * stride_pad, mask=a_mask, other=False)
        s = tl.dot(q, tl.trans(ek), input_precision="tf32") * SCALE
        s = tl.where(pad[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        # The eager cross path keeps probs fp32 (``probs.to(value_states.dtype)``
        # with fp32 expert V), so the value matmul is fp32 (TF32 tensor cores).
        acc = acc * alpha[:, None] + tl.dot(p, ev, input_precision="tf32")
        m_i = m_new

    acc = acc / l_i[:, None]
    acc = tl.where(m_mask[:, None], acc, 0.0)
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + (h * BLOCK_D + offs_d)[None, :] * stride_on,
        acc.to(tl.bfloat16),
        mask=m_mask[:, None],
    )


def triton_cross_attn(hidden, norm_w, eps, wq, expert_k, expert_v, positions, prefix_pad, head_dim: int, out=None):
    """Fused RMSNorm + GQA cross-attention for one denoising cross-attention layer.

    Args:
        hidden: ``[S, K]`` raw (pre-``input_layernorm``) hidden, bf16 (or fp32
            for layer 0), contiguous.
        norm_w: ``[K]`` input-layernorm weight, eps: its epsilon.
        wq: q projection weights ``[Hq*D, K]`` bf16.
        expert_k/expert_v: ``[P, G, D]`` fp32 expert prefix KV (cached per
            inference, values identical to the eager recomputation).
        positions: ``[S]`` int64 0-based positions (``position_ids - min``).
        prefix_pad: ``[P]`` bool prefix padding mask.
        head_dim: ``D``.

    Returns ``[S, Hq*D]`` bf16 (o_proj input), same layout as the eager path.
    """
    S, K = hidden.shape
    P = expert_k.shape[0]
    Hq = wq.shape[0] // head_dim
    G = expert_k.shape[1]
    D = head_dim
    scale = D**-0.5
    if out is None:
        out = torch.empty((S, Hq * D), dtype=torch.bfloat16, device=hidden.device)
    sin_tab, cos_tab = _get_rope_tables(D // 2, hidden.device, _MAX_WAVELENGTH)
    _gqa_cross_attn_kernel[(Hq, triton.cdiv(S, 32))](
        hidden,
        wq,
        expert_k,
        expert_v,
        sin_tab,
        cos_tab,
        positions,
        prefix_pad,
        norm_w,
        out,
        K,
        P,
        Hq,
        G,
        D,
        S,
        scale,
        float(eps),
        hidden.stride(0),
        hidden.stride(1),
        wq.stride(0),
        wq.stride(1),
        expert_k.stride(0),
        expert_k.stride(1),
        expert_k.stride(2),
        expert_v.stride(0),
        expert_v.stride(1),
        expert_v.stride(2),
        positions.stride(0),
        prefix_pad.stride(0),
        out.stride(0),
        out.stride(1),
        BLOCK_M=32,
        BLOCK_K=64,
        BLOCK_D=64,
        BLOCK_A=64,
        DH=32,
    )
    return out
