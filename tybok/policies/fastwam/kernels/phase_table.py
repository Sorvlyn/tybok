"""Phase table: contract declarations for 8 kernels / 41 phases.

**The positional-argument order corresponds one-to-one with bindings' `py::arg`**, and `phases.validate()` reconciles it against the signature pybind writes into
`__doc__` -- copying the order wrong fails on the spot, never becoming a wrong answer.

`reads` / `writes` list only **tensors**; scalars (stop_phase / norm_eps / phase_ts / dbg / L / B / ...)
are marked with `s` in `KernelIO.args`.
"""

from __future__ import annotations

from .phases import Buffer, KernelIO, Numerics, PhaseSpec, register_kernel, register_phase

# text encoder: tmt5.ffn (4 phases)
# The rest of this module is the registration table: one kernel/phase per statement, laid out to mirror the pybind arg order. Formatter off for the body; the docstring and import above stay formatted.
# fmt: off
register_kernel(KernelIO(
    kernel="tmt5.ffn", ext="tmt5", ext_fn="tmt5_ffn",
    args=(("t", "x_in"), ("t", "nw"), ("t", "w0"), ("t", "sw0"), ("t", "b0"),
          ("t", "w1"), ("t", "sw1"), ("t", "b1"), ("t", "out"),
          ("t", "a8buf"), ("t", "sa0buf"), ("t", "gbuf"), ("t", "a8gbuf"), ("t", "sa1buf"),
          ("s", "stop_phase"), ("s", "norm_eps"), ("s", "phase_ts"),
          ("r", "stream"), ("p", "only_phase")),
    buffers={
        "x_in":    Buffer("MxH", "bf16", "FFN input (also the F_PHASE_4 residual term)"),
        "nw":      Buffer("H", "bf16", "RMSNorm weight"),
        "w0":      Buffer("2FxH", "fp8", "packed wi_0|wi_1"),
        "sw0":     Buffer("2F", "fp32", "per-row scale of w0"),
        "b0":      Buffer("2F", "bf16", "all zeros (same convention as production; must not pass nullptr)"),
        "w1":      Buffer("HxF", "fp8", "wo"),
        "sw1":     Buffer("H", "fp32", "per-row scale of w1"),
        "b1":      Buffer("H", "bf16", "all zeros"),
        "out":     Buffer("MxH", "bf16", "output of this layer"),
        "a8buf":   Buffer("MxH", "fp8", "F_PHASE_1 output"),
        "sa0buf":  Buffer("M", "fp32", "F_PHASE_1 output"),
        "gbuf":    Buffer("Mx2F", "bf16", "F_PHASE_2 output (= K4's wi)"),
        "a8gbuf":  Buffer("MxF", "fp8", "F_PHASE_3 output"),
        "sa1buf":  Buffer("M", "fp32", "F_PHASE_3 output"),
    },
    geometry=dict(M=128, H=4096, F=10240, NT=256, SMEM=98304),
    grid_policy="capacity", min_grid=1, coop=True, phases=(1, 2, 3, 4),
))

register_phase(PhaseSpec(
    kernel="tmt5.ffn", index=1, does="RMSNorm + per-token fp8 quantization",
    reads=("x_in", "nw"), writes=("a8buf", "sa0buf"),
    numerics=Numerics(
        norm_kind="rms", norm_eps=1e-6, determinism="fixed",
        rounding_points=("rstd = 1/sqrtf(sum(x^2)/K + eps)  fp32 (**not rsqrtf**, that is an approximate instruction)",
                         "n  = bf16RN(x * rstd)     bf16 rounding point 1",
                         "nb = bf16RN(w_bf16 * n)   bf16 rounding point 2"),
        fp8_recipe="amax/448 + software RNE + true division; sa = max(amax*(1/448), 1e-12)",
        input_precision="bf16", accum_width="fp32", divide="div.rn",
    ),
))

register_phase(PhaseSpec(
    kernel="tmt5.ffn", index=2, does="up fp8 GEMM (packed wi) -> bf16 output",
    reads=("a8buf", "sa0buf", "w0", "sw0", "b0"), writes=("gbuf",), tiles=("F_PHASE_2",),
    numerics=Numerics(
        determinism="fixed", accum_width="fp32",
        rounding_points=("epilogue: bf16RN(acc*sa*sw + bias)  single rounding",),
        input_precision="bf16", divide="div.rn",
    ),
))

register_phase(PhaseSpec(
    kernel="tmt5.ffn", index=3, does="gelu_new gating + row amax + quantization",
    reads=("gbuf",), writes=("a8gbuf", "sa1buf"),
    numerics=Numerics(
        determinism="fixed",
        activation="gelu_new(u) = 0.5*u*(1+tanh(c*(u + 0.044715*u^3))), c = sqrt(2/pi), "
                   "**gating is computed in fp32**: act = gelu_fp32(bf16f(g0)) * bf16f(g1)",
        rounding_points=("act = bf16RN(gelu_fp32 * bf16f)   single bf16 rounding (rounding gelu first and then gating is wrong)",
                         "fp8 quantization"),
        fp8_recipe="amax/448 + software RNE + true division; sa1 = max(amax*(1/448), 1e-12)",
        accum_width="fp32", divide="div.rn",
    ),
))

register_phase(PhaseSpec(
    kernel="tmt5.ffn", index=4, does="down fp8 GEMM + residual",
    reads=("a8gbuf", "sa1buf", "w1", "sw1", "b1", "x_in"), writes=("out",), tiles=("F_PHASE_4",),
    numerics=Numerics(
        determinism="fixed", accum_width="fp32",
        residual="single_round: bf16(acc*sa*sw + f32(x_in)) (**not** double rounding)",
        rounding_points=("out = bf16RN(acc*sa*sw + f32(res))  single",),
        input_precision="bf16", divide="div.rn",
    ),
))

# text encoder: tmt5.attn (5 phases)
register_kernel(KernelIO(
    kernel="tmt5.attn", ext="tmt5", ext_fn="tmt5_attn",
    args=(("t", "x_in"), ("t", "nw"), ("t", "wqkv"), ("t", "sqkv"), ("t", "bqkv"),
          ("t", "wo"), ("t", "swo"), ("t", "bo"), ("t", "posb"), ("t", "amask"),
          ("t", "out"), ("t", "a8buf"), ("t", "sa0buf"),
          ("t", "qkvbuf"), ("t", "attnbuf"), ("t", "a8obuf"), ("t", "sa1buf"),
          ("s", "stop_phase"), ("s", "norm_eps"), ("s", "phase_ts"), ("s", "dbg"),
          ("r", "stream"), ("p", "only_phase")),
    buffers={
        "x_in":    Buffer("MxH", "bf16", "self-attention input (also the S_PHASE_5 residual term)"),
        "nw":      Buffer("H", "bf16", "RMSNorm weight"),
        "wqkv":    Buffer("3HxH", "fp8", "packed qkv"),
        "sqkv":    Buffer("3H", "fp32", "per-row scale of wqkv"),
        "bqkv":    Buffer("3H", "bf16", "all zeros"),
        "wo":      Buffer("HxH", "fp8", "o projection"),
        "swo":     Buffer("H", "fp32", "per-row scale of wo"),
        "bo":      Buffer("H", "bf16", "all zeros"),
        "posb":    Buffer("1xNHxSxS", "bf16", "per-layer relative position bias (**contiguous copy**)"),
        "amask":   Buffer("S", "bf16", "additive column mask (1-mask)*finfo.min"),
        "out":     Buffer("MxH", "bf16", "output of this layer"),
        "a8buf":   Buffer("MxH", "fp8", "S_PHASE_1 output"),
        "sa0buf":  Buffer("M", "fp32", "S_PHASE_1 output"),
        "qkvbuf":  Buffer("Mx3H", "bf16", "S_PHASE_2 output"),
        "attnbuf": Buffer("MxH", "bf16", "S_PHASE_3 output"),
        "a8obuf":  Buffer("MxH", "fp8", "S_PHASE_4 output"),
        "sa1buf":  Buffer("M", "fp32", "S_PHASE_4 output"),
    },
    geometry=dict(M=128, H=4096, N=12288, NH=64, HD=64, NT=256, SMEM=98304),
    grid_policy="capacity", min_grid=1, coop=True, phases=(1, 2, 3, 4, 5),
))

register_phase(PhaseSpec(
    kernel="tmt5.attn", index=1, does="RMSNorm + per-token fp8 quantization",
    reads=("x_in", "nw"), writes=("a8buf", "sa0buf"),
    numerics=Numerics(
        norm_kind="rms", norm_eps=1e-6, determinism="fixed",
        rounding_points=("rstd = 1/sqrtf(sum(x^2)/K + eps)  fp32 (not rsqrtf)",
                         "n  = bf16RN(x * rstd)", "nb = bf16RN(w_bf16 * n)"),
        fp8_recipe="amax/448 + software RNE + true division; sa = max(amax*(1/448), 1e-12)",
        input_precision="bf16", accum_width="fp32", divide="div.rn",
    ),
))

register_phase(PhaseSpec(
    kernel="tmt5.attn", index=2, does="packed-qkv fp8 GEMM", tiles=("S_PHASE_2",),
    reads=("a8buf", "sa0buf", "wqkv", "sqkv", "bqkv"), writes=("qkvbuf",),
    numerics=Numerics(
        determinism="fixed", accum_width="fp32",
        rounding_points=("epilogue: bf16RN(acc*sa*sw + bias)",),
        input_precision="bf16", divide="div.rn",
    ),
))

register_phase(PhaseSpec(
    kernel="tmt5.attn", index=3, does="bf16 attention (one CTA per head)",
    reads=("qkvbuf", "posb", "amask"), writes=("attnbuf",),
    numerics=Numerics(
        determinism="fixed", accum_width="fp32",
        attention="**no 1/sqrt(d) scaling** (the UMT5 reference has none); scores = bf16RN(q@k^T), "
                  "then add bf16RN(pos_bias + amask) (two bf16 RN additions); "
                  "softmax is computed in fp32 and then cast back to bf16",
        rounding_points=("bf16RN(q@k^T)", "bf16RN(scores + bf16RN(pos+mask))",
                         "attn_weights = bf16RN(softmax_fp32)", "PV output bf16"),
        rope=None,
    ),
))

register_phase(PhaseSpec(
    kernel="tmt5.attn", index=4, does="attn row amax + fp8 quantization",
    reads=("attnbuf",), writes=("a8obuf", "sa1buf"),
    numerics=Numerics(
        determinism="fixed",
        rounding_points=("fp8 quantization (amax taken in fp32)",),
        fp8_recipe="amax/448 + software RNE + true division; sa1 = max(amax*(1/448), 1e-12)",
        accum_width="fp32", divide="div.rn",
    ),
))

register_phase(PhaseSpec(
    kernel="tmt5.attn", index=5, does="o fp8 GEMM + residual", tiles=("S_PHASE_5",),
    reads=("a8obuf", "sa1buf", "wo", "swo", "bo", "x_in"), writes=("out",),
    numerics=Numerics(
        determinism="fixed", accum_width="fp32",
        residual="single_round: bf16(acc*sa*sw + f32(x_in))",
        rounding_points=("out = bf16RN(acc*sa*sw + f32(res))  single",),
        input_precision="bf16", divide="div.rn",
    ),
))

# action DiT: adit.attn_self (S, 6 phases)
register_kernel(KernelIO(
    kernel="adit.attn_self", ext="action", ext_fn="adit_attn_self",
    args=(("t", "x_in"), ("t", "shift_msa"), ("t", "scale_msa"),
          ("t", "wqkv"), ("t", "swqkv"), ("t", "bqkv"),
          ("t", "kv_cache"), ("t", "v_cache"), ("t", "wnq"), ("t", "wnk"),
          ("t", "cos_t"), ("t", "sin_t"), ("t", "gate_msa"),
          ("t", "wo"), ("t", "swo"), ("t", "bo"),
          ("t", "out"), ("t", "qkv"), ("t", "attn"),
          ("t", "a8x"), ("t", "sa0x"), ("t", "a8a"), ("t", "sa0a"),
          ("t", "rstd_scratch"), ("s", "L"), ("r", "stream"), ("p", "only_phase")),
    buffers={
        "x_in":         Buffer("MxH", "bf16", "block input (also the S_PHASE_6 residual)"),
        "shift_msa":    Buffer("H", "bf16", "AdaLN shift"),
        "scale_msa":    Buffer("H", "bf16", "AdaLN scale"),
        "wqkv":         Buffer("NxH", "fp8", "packed qkv (N = 3xH3)"),
        "swqkv":        Buffer("N", "fp32", "per-row scale of wqkv"),
        "bqkv":         Buffer("N", "bf16", "all zeros"),
        "kv_cache":     Buffer("LxH3", "bf16", "video KV (already norm+RoPE)"),
        "v_cache":      Buffer("LxH3", "bf16", "video V"),
        "wnq":          Buffer("H3", "bf16", "qk-norm weight"),
        "wnk":          Buffer("H3", "bf16", "qk-norm weight"),
        "cos_t":        Buffer("Mx64", "fp32", "RoPE table"),
        "sin_t":        Buffer("Mx64", "fp32", "RoPE table"),
        "gate_msa":     Buffer("H", "bf16", "gate_msa of S_PHASE_6"),
        "wo":           Buffer("HxH3", "fp8", "o projection"),
        "swo":          Buffer("H", "fp32", "per-row scale of wo"),
        "bo":           Buffer("H", "bf16", "all zeros"),
        "out":          Buffer("MxH", "bf16", "output of this layer"),
        "qkv":          Buffer("MxN", "bf16", "S_PHASE_2 output / S_PHASE_3-S_PHASE_4 input"),
        "attn":         Buffer("MxH3", "bf16", "S_PHASE_4 output"),
        "a8x":          Buffer("MxH", "fp8", "S_PHASE_1 output"),
        "sa0x":         Buffer("M", "fp32", "S_PHASE_1 output"),
        "a8a":          Buffer("MxH3", "fp8", "S_PHASE_5 output"),
        "sa0a":         Buffer("M", "fp32", "S_PHASE_5 output"),
        "rstd_scratch": Buffer("2x24xM", "fp32", "per-head sum(q^2)/sum(k^2) shard slots (need no zeroing)"),
    },
    geometry=dict(M=32, H=1024, H3=3072, N=9216, HEADS=24, BD=128, NT=256, GRID=96),
    grid_policy="problem", min_grid=96, coop=True, phases=(1, 2, 3, 4, 5, 6),
))

register_phase(PhaseSpec(
    kernel="adit.attn_self", index=1, does="norm1(LN without affine) + modulate + row quantization",
    reads=("x_in", "shift_msa", "scale_msa"), writes=("a8x", "sa0x"),
    numerics=Numerics(
        norm_kind="layer_plain (LN, **no affine**)", determinism="fixed",
        rounding_points=("modulate: bf16RN step by step (same convention as dit_block)", "fp8 quantization"),
        fp8_recipe="amax/448 + software RNE + true division; sa = max(amax*(1/448), 1e-12)",
        input_precision="bf16", accum_width="fp32", divide="div.rn",
    ),
))
register_phase(PhaseSpec(
    kernel="adit.attn_self", index=2, does="qkv fp8 GEMM + epilogue",
    reads=("a8x", "sa0x", "wqkv", "swqkv", "bqkv"), writes=("qkv",),
    numerics=Numerics(
        determinism="fixed", accum_width="fp32",
        rounding_points=("epilogue: bf16RN(acc*sa*sw + bias)",),
        input_precision="bf16", divide="div.rn",
    ),
))
register_phase(PhaseSpec(
    kernel="adit.attn_self", index=3, does="q/k full-row RMS shard scan (one CTA per head)",
    reads=("qkv",), writes=("rstd_scratch",),
    numerics=Numerics(
        determinism="fixed",
        reduction_notes="each (head,row) owns a slot, plain write; downstream S_PHASE_4 sums over h=0..23 in "
                        "**fixed order** (no atomicAdd: not reproducible for the same input)",
        rounding_points=("sum accumulates in fp32, no intermediate cast",),
        accum_width="fp32",
    ),
))
register_phase(PhaseSpec(
    kernel="adit.attn_self", index=4, does="rstd fixed reduction + qk-norm(bf16)/RoPE + bf16 flash",
    reads=("qkv", "rstd_scratch", "wnq", "wnk", "cos_t", "sin_t", "kv_cache", "v_cache"),
    writes=("attn",),
    numerics=Numerics(
        determinism="fixed", accum_width="fp32",
        attention="budget-gated flash (online softmax) replaces fp32 SDPA -- pure bf16 semantics tier; "
                  "score must be multiplied by 1/sqrt(128) (SATT)",
        rope="adjacent-pair complex multiply (action side)",
        rounding_points=("bf16RN(q*rstd*w) for q/k", "flash output bf16RN",),
    ),
))
register_phase(PhaseSpec(
    kernel="adit.attn_self", index=5, does="attn row amax + quantization",
    reads=("attn",), writes=("a8a", "sa0a"),
    numerics=Numerics(
        determinism="fixed", rounding_points=("fp8 quantization",),
        fp8_recipe="amax/448 + software RNE + true division", accum_width="fp32", divide="div.rn",
    ),
))
register_phase(PhaseSpec(
    kernel="adit.attn_self", index=6, does="o fp8 GEMM + gate_msa residual",
    reads=("a8a", "sa0a", "wo", "swo", "bo", "gate_msa", "x_in"), writes=("out",),
    numerics=Numerics(
        determinism="fixed", accum_width="fp32",
        residual="with gate_msa: out = bf16RN(x + bf16RN(gate_msa * bf16RN(acc*sa*sw+b)))",
        rounding_points=("epilogue three bf16RN (acc/gate_msa/residual)",),
        input_precision="bf16", divide="div.rn",
    ),
))

# action DiT: adit.attn_cross (G, 6 phases)
register_kernel(KernelIO(
    kernel="adit.attn_cross", ext="action", ext_fn="adit_attn_cross",
    args=(("t", "x_in"), ("t", "w3"), ("t", "b3"),
          ("t", "wq"), ("t", "swq"), ("t", "bq"),
          ("t", "qp"), ("t", "a8q"), ("t", "sa0q"), ("t", "wnq"),
          ("t", "k_cache"), ("t", "v_cache"), ("t", "mask"), ("t", "attn"),
          ("t", "wo"), ("t", "swo"), ("t", "bo"), ("t", "out"),
          ("t", "a8o"), ("t", "sa0o"), ("t", "rstd_scratch"),
          ("s", "L"), ("r", "stream"), ("p", "only_phase")),
    buffers={
        "x_in":         Buffer("MxH", "bf16", "block input (cross residual)"),
        "w3":           Buffer("H", "bf16", "norm3 weight (**affine LN**)"),
        "b3":           Buffer("H", "bf16", "norm3 bias"),
        "wq":           Buffer("NQxH", "fp8", "cross q"),
        "swq":          Buffer("NQ", "fp32", "per-row scale"),
        "bq":           Buffer("NQ", "bf16", "all zeros"),
        "qp":           Buffer("MxNQ", "bf16", "C_PHASE_2 output / C_PHASE_3-C_PHASE_4 input"),
        "a8q":          Buffer("MxH", "fp8", "C_PHASE_1 output"),
        "sa0q":         Buffer("M", "fp32", "C_PHASE_1 output"),
        "wnq":          Buffer("NQ", "bf16", "q row RMS weight"),
        "k_cache":      Buffer("LxNQ", "bf16", "context k (already qk-norm)"),
        "v_cache":      Buffer("LxNQ", "bf16", "context v"),
        "mask":         Buffer("L", "bool", "context mask"),
        "attn":         Buffer("MxNQ", "bf16", "C_PHASE_4 output"),
        "wo":           Buffer("NOxNQ", "fp8", "cross o"),
        "swo":          Buffer("NO", "fp32", "per-row scale"),
        "bo":           Buffer("NO", "bf16", "all zeros"),
        "out":          Buffer("MxNO", "bf16", "output of this layer"),
        "a8o":          Buffer("MxNQ", "fp8", "C_PHASE_5 output"),
        "sa0o":         Buffer("M", "fp32", "C_PHASE_5 output"),
        "rstd_scratch": Buffer("24xM", "fp32", "per-head sum(q^2) shard slots (need no zeroing)"),
    },
    geometry=dict(M=32, H=1024, NQ=3072, NO=1024, HEADS=24, BD=128, NT=256, GRID=48, C=129),
    grid_policy="problem", min_grid=48, coop=True, phases=(1, 2, 3, 4, 5, 6),
))

register_phase(PhaseSpec(
    kernel="adit.attn_cross", index=1, does="norm3(affine LayerNorm) + row quantization",
    reads=("x_in", "w3", "b3"), writes=("a8q", "sa0q"),
    numerics=Numerics(
        norm_kind="layer_affine (**has** weight/bias)", determinism="fixed",
        rounding_points=("LN output bf16RN", "fp8 quantization"),
        fp8_recipe="amax/448 + software RNE + true division", input_precision="bf16",
        accum_width="fp32", divide="div.rn",
    ),
))
register_phase(PhaseSpec(
    kernel="adit.attn_cross", index=2, does="cross-q fp8 GEMM",
    reads=("a8q", "sa0q", "wq", "swq", "bq"), writes=("qp",),
    numerics=Numerics(determinism="fixed", accum_width="fp32",
                      rounding_points=("epilogue: bf16RN(acc*sa*sw + bias)",),
                      input_precision="bf16", divide="div.rn"),
))
register_phase(PhaseSpec(
    kernel="adit.attn_cross", index=3, does="q full-row RMS shard scan + q'/kv staging",
    reads=("qp", "k_cache", "v_cache", "mask"), writes=("rstd_scratch",),
    numerics=Numerics(
        determinism="fixed", accum_width="fp32",
        reduction_notes="same as adit.attn_self S_PHASE_3: shard slots + downstream fixed-order reduction",
        rounding_points=("sum accumulates in fp32",),
    ),
))
register_phase(PhaseSpec(
    kernel="adit.attn_cross", index=4, does="qs scaling + masked bf16 flash",
    reads=("qp", "k_cache", "v_cache", "mask", "rstd_scratch"), writes=("attn",),
    numerics=Numerics(
        determinism="fixed", accum_width="fp32",
        attention="masked flash; the split-phase form **replays** C_PHASE_3's staging (wns/mask/qs rebuilt "
                  "from global, but the RMS is not redone)",
        rounding_points=("qs = bf16RN(q*rq*wn)", "flash output bf16RN",),
    ),
))
register_phase(PhaseSpec(
    kernel="adit.attn_cross", index=5, does="attn row amax + quantization",
    reads=("attn",), writes=("a8o", "sa0o"),
    numerics=Numerics(determinism="fixed", rounding_points=("fp8 quantization",),
                      fp8_recipe="amax/448 + software RNE + true division",
                      accum_width="fp32", divide="div.rn"),
))
register_phase(PhaseSpec(
    kernel="adit.attn_cross", index=6, does="cross-o fp8 GEMM + ungated residual",
    reads=("a8o", "sa0o", "wo", "swo", "bo", "x_in"), writes=("out",),
    numerics=Numerics(determinism="fixed", accum_width="fp32",
                      residual="out = bf16RN(x + bf16RN(acc*sa*sw+b))",
                      rounding_points=("epilogue two bf16RN",),
                      input_precision="bf16", divide="div.rn"),
))

# action DiT: adit.ffn (4 phases)
register_kernel(KernelIO(
    kernel="adit.ffn", ext="action", ext_fn="adit_ffn",
    args=(("t", "x_in"), ("t", "shift_mlp"), ("t", "scale_mlp"), ("t", "gate_mlp"),
          ("t", "w0"), ("t", "sw0"), ("t", "b0"), ("t", "w1"), ("t", "sw1"), ("t", "b1"),
          ("t", "out"), ("t", "a8buf"), ("t", "sa0buf"), ("t", "gbuf"),
          ("t", "a8gbuf"), ("t", "raw1"), ("s", "B"), ("r", "stream"), ("p", "only_phase")),
    buffers={
        "x_in":      Buffer("BxMxH", "bf16", "block input (also the F_PHASE_4 residual)"),
        "shift_mlp": Buffer("BxH", "bf16", "AdaLN shift"),
        "scale_mlp": Buffer("BxH", "bf16", "AdaLN scale"),
        "gate_mlp":  Buffer("BxH", "bf16", "gate_mlp of F_PHASE_4"),
        "w0":        Buffer("FxH", "fp8", "up"),
        "sw0":       Buffer("F", "fp32", "per-row scale"),
        "b0":        Buffer("F", "bf16", "all zeros"),
        "w1":        Buffer("HxF", "fp8", "down"),
        "sw1":       Buffer("H", "fp32", "per-row scale"),
        "b1":        Buffer("H", "bf16", "all zeros"),
        "out":       Buffer("BxMxH", "bf16", "output of this layer"),
        "a8buf":     Buffer("BxMxH", "fp8", "F_PHASE_1 output"),
        "sa0buf":    Buffer("BxM", "fp32", "F_PHASE_1 output"),
        "gbuf":      Buffer("BxMxF", "bf16", "F_PHASE_2 output"),
        "a8gbuf":    Buffer("BxMxF", "fp8", "F_PHASE_3 output"),
        "raw1":      Buffer("BxM", "fp32 bit pattern", "row |g| amax (written by F_PHASE_2, read by F_PHASE_3/F_PHASE_4)"),
    },
    geometry=dict(M=32, H=1024, F=4096, NT=256, BNU=64, GRID0=64),
    grid_policy="problem", min_grid=64, coop=True, phases=(1, 2, 3, 4),
))

register_phase(PhaseSpec(
    kernel="adit.ffn", index=1, does="norm2 + modulate + row quantization (one CTA per row)",
    reads=("x_in", "shift_mlp", "scale_mlp"), writes=("a8buf", "sa0buf", "raw1"),
    numerics=Numerics(
        norm_kind="layer_plain + modulate", determinism="fixed",
        rounding_points=("modulate bf16RN step by step", "fp8 quantization"),
        fp8_recipe="amax/448 + software RNE + true division", input_precision="bf16",
        accum_width="fp32", divide="div.rn",
    ),
))
register_phase(PhaseSpec(
    kernel="adit.ffn", index=2, does="up fp8 GEMM + GELU + row amax",
    reads=("a8buf", "sa0buf", "w0", "sw0", "b0"), writes=("gbuf", "raw1"),
    numerics=Numerics(
        determinism="fixed", accum_width="fp32",
        activation="0.5u(1+tanh(c1(u+c2u^3))), c1=sqrt(2/pi) = torch nn.GELU(approximate='tanh'); "
                   "shared helper kernels/common.h `gelu_tanh`, same as video/text/lerobot",
        rounding_points=("epilogue: bf16RN(acc*sa*sw + bias)",),
        input_precision="bf16", divide="div.rn",
    ),
))
register_phase(PhaseSpec(
    kernel="adit.ffn", index=3, does="gbuf quantization (by the column slice between w0/w1)",
    reads=("gbuf", "raw1"), writes=("a8gbuf",),
    numerics=Numerics(
        determinism="fixed", rounding_points=("fp8 quantization (amax provided by raw1)",),
        fp8_recipe="sa = max(raw1/448, 1e-12); software RNE + true division",
        accum_width="fp32", divide="div.rn",
    ),
))
register_phase(PhaseSpec(
    kernel="adit.ffn", index=4, does="down fp8 GEMM + gate_mlp residual",
    reads=("a8gbuf", "raw1", "w1", "sw1", "b1", "gate_mlp", "x_in"), writes=("out",),
    numerics=Numerics(
        determinism="fixed", accum_width="fp32",
        residual="with gate_mlp (same convention as adit.attn_self S_PHASE_6)",
        rounding_points=("epilogue three bf16RN",),
        input_precision="bf16", divide="div.rn",
    ),
))

# video DiT: vdit.attn_self (6 phases)
register_kernel(KernelIO(
    kernel="vdit.attn_self", ext="video", ext_fn="vdit_attn_self",
    args=(("t", "x_in"), ("t", "shift_msa"), ("t", "scale_msa"), ("t", "gate_msa"),
          ("t", "wqkv"), ("t", "swqkv"), ("t", "bqkv"), ("t", "wnq"), ("t", "wnk"),
          ("t", "cos_t"), ("t", "sin_t"), ("t", "wo"), ("t", "swo"), ("t", "bo"),
          ("t", "out"), ("t", "qkv"), ("t", "qc"), ("t", "kc"), ("t", "vc"), ("t", "attn"),
          ("t", "a8x"), ("t", "sa0x"), ("t", "a8a"), ("t", "sa0a"),
          ("s", "norm_eps"), ("s", "mod_srow"), ("s", "stop_phase"),
          ("p", "only_phase"), ("s", "dbg"), ("s", "phase_ts"), ("r", "stream")),
    buffers={
        "x_in":      Buffer("MxH", "bf16", "block input (norm1 input + residual)"),
        "shift_msa": Buffer("MxH or H", "bf16", "AdaLN shift"),
        "scale_msa": Buffer("same as above", "bf16", "AdaLN scale"),
        "gate_msa":  Buffer("same as above", "bf16", "gate_msa of S_PHASE_6"),
        "wqkv":      Buffer("9216xH", "fp8", "packed qkv"),
        "swqkv":     Buffer("9216", "fp32", "per-row scale"),
        "bqkv":      Buffer("9216", "bf16", "all zeros"),
        "wnq":       Buffer("H3", "bf16", "qk-norm (full row, not per-head)"),
        "wnk":       Buffer("H3", "bf16", "qk-norm"),
        "cos_t":     Buffer("Mx64", "fp32", "3-D grid RoPE table"),
        "sin_t":     Buffer("Mx64", "fp32", "3-D grid RoPE table"),
        "wo":        Buffer("HxH3", "fp8", "o projection"),
        "swo":       Buffer("H", "fp32", "per-row scale"),
        "bo":        Buffer("H", "bf16", "all zeros"),
        "out":       Buffer("MxH", "bf16", "output of this layer"),
        "qkv":       Buffer("Mx9216", "bf16", "S_PHASE_2 output"),
        "qc":        Buffer("MxH3", "bf16", "S_PHASE_3 output: q after norm+RoPE"),
        "kc":        Buffer("MxH3", "bf16", "S_PHASE_3 output: k after norm+RoPE = KV cache"),
        "vc":        Buffer("MxH3", "bf16", "S_PHASE_3 output: v as-is = KV cache"),
        "attn":      Buffer("MxH3", "**fp32**", "S_PHASE_4 output (note it is fp32, not bf16)"),
        "a8x":       Buffer("MxH", "fp8", "S_PHASE_1 output"),
        "sa0x":      Buffer("M", "fp32", "S_PHASE_1 output"),
        "a8a":       Buffer("MxH3", "fp8", "S_PHASE_5 output"),
        "sa0a":      Buffer("M", "fp32", "S_PHASE_5 output"),
    },
    geometry=dict(M=120, H=3072, N=9216, NH=24, D=128, NT=256, SMEM=49152),
    grid_policy="capacity", min_grid=120, coop=True, phases=(1, 2, 3, 4, 5, 6),
))

register_phase(PhaseSpec(
    kernel="vdit.attn_self", index=1, does="norm1 + modulate + row quantization",
    reads=("x_in", "shift_msa", "scale_msa"), writes=("a8x", "sa0x"),
    numerics=Numerics(
        norm_kind="layer_plain + modulate", determinism="fixed",
        rounding_points=("modulate bf16RN step by step", "fp8 quantization"),
        fp8_recipe="amax/448 + software RNE + true division", input_precision="bf16",
        accum_width="fp32", divide="div.rn",
    ),
))
register_phase(PhaseSpec(
    kernel="vdit.attn_self", index=2, does="qkv fp8 GEMM (N=9216)", tiles=("S_PHASE_2",),
    reads=("a8x", "sa0x", "wqkv", "swqkv", "bqkv"), writes=("qkv",),
    numerics=Numerics(determinism="fixed", accum_width="fp32",
                      rounding_points=("epilogue: bf16RN(acc*sa*sw + bias)",),
                      input_precision="bf16", divide="div.rn"),
))
register_phase(PhaseSpec(
    kernel="vdit.attn_self", index=3, does="q/k full-row RMS + qk-norm + 3-D RoPE; v as-is",
    reads=("qkv", "wnq", "wnk", "cos_t", "sin_t"), writes=("qc", "kc", "vc"),
    numerics=Numerics(
        norm_kind="rms (across the **entire 3072-wide row**, not per-head)", determinism="fixed",
        reduction_notes="**one CTA per full row, no atomic, no intermediate sync** (120 rows <= grid) --"
                        "this and the action kernel's per-head sharding + atomicAdd are two different solutions",
        rope="3-D grid RoPE",
        rounding_points=("bf16RN(q*rstd)", "bf16RN after RoPE",),
        accum_width="fp32",
    ),
))
register_phase(PhaseSpec(
    kernel="vdit.attn_self", index=4, does="bf16 flash (24 heads x 4 q-tiles)",
    reads=("qc", "kc", "vc"), writes=("attn",),
    numerics=Numerics(
        determinism="fixed", accum_width="fp32",
        attention="**the output is fp32** (S_PHASE_5 takes amax / quantizes directly on fp32, without going back to bf16)",
        rounding_points=("bf16RN for score/PV inside flash; output stays fp32",),
    ),
))
register_phase(PhaseSpec(
    kernel="vdit.attn_self", index=5, does="attn row amax + quantization",
    reads=("attn",), writes=("a8a", "sa0a"),
    numerics=Numerics(determinism="fixed",
                      input_precision="fp32_preserved (input is fp32, single rounding to fp8)",
                      rounding_points=("fp8 quantization (quantize fp32 directly, no prior bf16 cast)",),
                      fp8_recipe="amax/448 + software RNE + true division",
                      accum_width="fp32", divide="div.rn"),
))
register_phase(PhaseSpec(
    kernel="vdit.attn_self", index=6, does="o fp8 GEMM + gate_msa residual", tiles=("S_PHASE_6",),
    reads=("a8a", "sa0a", "wo", "swo", "bo", "gate_msa", "x_in"), writes=("out",),
    numerics=Numerics(determinism="fixed", accum_width="fp32",
                      residual="with gate_msa: out = bf16RN(x + bf16RN(gate_msa * bf16RN(acc*sa*sw+b)))",
                      rounding_points=("epilogue three bf16RN",),
                      input_precision="bf16", divide="div.rn"),
))

# video DiT: vdit.attn_cross (6 phases)
register_kernel(KernelIO(
    kernel="vdit.attn_cross", ext="video", ext_fn="vdit_attn_cross",
    args=(("t", "x_in"), ("t", "context"), ("t", "w3"), ("t", "b3"),
          ("t", "wq"), ("t", "swq"), ("t", "bq"),
          ("t", "wkv"), ("t", "swkv"), ("t", "bkv"),
          ("t", "wnq"), ("t", "wnk"), ("t", "wo"), ("t", "swo"), ("t", "bo"), ("t", "ones"),
          ("t", "out"), ("t", "qp"), ("t", "kvp"), ("t", "qc"), ("t", "kc"), ("t", "vc"),
          ("t", "attn"), ("t", "a8q"), ("t", "sa0q"), ("t", "a8c"), ("t", "sa0c"),
          ("t", "a8o"), ("t", "sa0o"),
          ("s", "norm_eps"), ("s", "stop_phase"), ("p", "only_phase"),
          ("s", "dbg"), ("s", "phase_ts"), ("r", "stream")),
    buffers={
        "x_in":    Buffer("MxH", "bf16", "block input (cross residual)"),
        "context": Buffer("CxH", "bf16", "text context"),
        "w3":      Buffer("H", "bf16", "norm3 weight (affine LN)"),
        "b3":      Buffer("H", "bf16", "norm3 bias"),
        "wq":      Buffer("H3xH", "fp8", "cross q"),
        "swq":     Buffer("H3", "fp32", "per-row scale"),
        "bq":      Buffer("H3", "bf16", "all zeros"),
        "wkv":     Buffer("6144xH", "fp8", "cross k/v (packed)"),
        "swkv":    Buffer("6144", "fp32", "per-row scale"),
        "bkv":     Buffer("6144", "bf16", "all zeros"),
        "wnq":     Buffer("H3", "bf16", "q row RMS weight"),
        "wnk":     Buffer("H3", "bf16", "k row RMS weight"),
        "wo":      Buffer("HxH3", "fp8", "cross o"),
        "swo":     Buffer("H", "fp32", "per-row scale"),
        "bo":      Buffer("H", "bf16", "all zeros"),
        "ones":    Buffer("H", "bf16", "ones used for the ungated residual"),
        "out":     Buffer("MxH", "bf16", "output of this layer"),
        "qp":      Buffer("MxH3", "bf16", "C_PHASE_2 output"),
        "kvp":     Buffer("Cx6144", "bf16", "C_PHASE_2 output"),
        "qc":      Buffer("MxH3", "bf16", "C_PHASE_3 output"),
        "kc":      Buffer("CxH3", "bf16", "C_PHASE_3 output"),
        "vc":      Buffer("CxH3", "bf16", "C_PHASE_3 output (v as-is)"),
        "attn":    Buffer("MxH3", "**fp32**", "C_PHASE_4 output"),
        "a8q":     Buffer("MxH", "fp8", "C_PHASE_1 output (x side)"),
        "sa0q":    Buffer("M", "fp32", "C_PHASE_1 output"),
        "a8c":     Buffer("CxH", "fp8", "C_PHASE_1 output (context side)"),
        "sa0c":    Buffer("C", "fp32", "C_PHASE_1 output"),
        "a8o":     Buffer("MxH3", "fp8", "C_PHASE_5 output"),
        "sa0o":    Buffer("M", "fp32", "C_PHASE_5 output"),
    },
    geometry=dict(M=120, C=129, H=3072, H3=3072, NKV=6144, NH=24, D=128, NT=256, SMEM=49152),
    grid_policy="capacity", min_grid=120, coop=True, phases=(1, 2, 3, 4, 5, 6),
))

register_phase(PhaseSpec(
    kernel="vdit.attn_cross", index=1, does="norm3+quantize x (<120 rows) / quantize context (>=120 rows)",
    reads=("x_in", "context", "w3", "b3"), writes=("a8q", "sa0q", "a8c", "sa0c"),
    numerics=Numerics(
        norm_kind="layer_affine", determinism="fixed",
        rounding_points=("LN output bf16RN", "fp8 quantization (once per branch)"),
        fp8_recipe="amax/448 + software RNE + true division", input_precision="bf16",
        accum_width="fp32", divide="div.rn",
    ),
))
register_phase(PhaseSpec(
    kernel="vdit.attn_cross", index=2, does="cross-q GEMM (M=120,N=3072) immediately followed by cross-kv GEMM (M=129,N=6144)",
    # two GEMMs in one phase share a single tile (marked by the label suffix _2G): sweep it with `--geom C_PHASE_2_2G`
    tiles=("C_PHASE_2_2G",),
    reads=("a8q", "sa0q", "wq", "swq", "bq", "a8c", "sa0c", "wkv", "swkv", "bkv"),
    writes=("qp", "kvp"),
    numerics=Numerics(determinism="fixed", accum_width="fp32",
                      rounding_points=("epilogue: bf16RN(acc*sa*sw + bias)",),
                      input_precision="bf16", divide="div.rn"),
))
register_phase(PhaseSpec(
    kernel="vdit.attn_cross", index=3, does="q row RMS+wnq -> qc; kv row RMS+wnk (only the k half) -> kc, v as-is",
    reads=("qp", "kvp", "wnq", "wnk"), writes=("qc", "kc", "vc"),
    numerics=Numerics(
        norm_kind="rms (full row)", determinism="fixed",
        reduction_notes="one CTA per full row (249 rows grid-stride), no atomic",
        rounding_points=("bf16RN(q*rstd*wn)",), accum_width="fp32",
    ),
))
register_phase(PhaseSpec(
    kernel="vdit.attn_cross", index=4, does="bf16 flash (q 120 x kv 129)",
    reads=("qc", "kc", "vc"), writes=("attn",),
    numerics=Numerics(
        determinism="fixed", accum_width="fp32",
        attention="kv 129 rows -> 9 tiles; the tail tile has only 1 valid row (the other 15 rows must be set to -inf); output fp32",
        rounding_points=("bf16RN inside flash; output stays fp32",),
    ),
))
register_phase(PhaseSpec(
    kernel="vdit.attn_cross", index=5, does="attn row amax + quantization",
    reads=("attn",), writes=("a8o", "sa0o"),
    numerics=Numerics(determinism="fixed",
                      input_precision="fp32_preserved",
                      rounding_points=("fp8 quantization (quantize fp32 directly)",),
                      fp8_recipe="amax/448 + software RNE + true division",
                      accum_width="fp32", divide="div.rn"),
))
register_phase(PhaseSpec(
    kernel="vdit.attn_cross", index=6, does="cross-o GEMM + **ungated** residual", tiles=("C_PHASE_6",),
    reads=("a8o", "sa0o", "wo", "swo", "bo", "ones", "x_in"), writes=("out",),
    numerics=Numerics(
        determinism="fixed", accum_width="fp32",
        residual="ungated: reuse the RESID path (pass ones as gate_mlp, gate_srow=0)",
        rounding_points=("epilogue bf16RN",),
        input_precision="bf16", divide="div.rn",
    ),
))

# video DiT: vdit.ffn (4 phases)
register_kernel(KernelIO(
    kernel="vdit.ffn", ext="video", ext_fn="vdit_ffn",
    args=(("t", "x_in"), ("t", "gate_mlp"), ("t", "shift_mlp"), ("t", "scale_mlp"),
          ("t", "w0"), ("t", "sw0"), ("t", "b0"), ("t", "w1"), ("t", "sw1"), ("t", "b1"),
          ("t", "out"), ("t", "a8buf"), ("t", "sa0buf"), ("t", "gbuf"),
          ("t", "a8gbuf"), ("t", "sa1buf"), ("t", "raw1"),
          ("s", "stop_phase"), ("p", "only_phase"), ("s", "use_norm"), ("s", "norm_eps"),
          ("s", "phase_ts"), ("r", "stream"), ("s", "mod_srow")),
    buffers={
        "x_in":      Buffer("MxH", "bf16", "block input (norm2 input + residual)"),
        "gate_mlp":      Buffer("H", "bf16", "gate_mlp of F_PHASE_4"),
        "shift_mlp": Buffer("H", "bf16", "AdaLN shift"),
        "scale_mlp": Buffer("H", "bf16", "AdaLN scale"),
        "w0":        Buffer("FxH", "fp8", "up"),
        "sw0":       Buffer("F", "fp32", "per-row scale"),
        "b0":        Buffer("F", "bf16", "all zeros"),
        "w1":        Buffer("HxF", "fp8", "down"),
        "sw1":       Buffer("H", "fp32", "per-row scale"),
        "b1":        Buffer("H", "bf16", "all zeros"),
        "out":       Buffer("MxH", "bf16", "output of this layer"),
        "a8buf":     Buffer("MxH", "fp8", "F_PHASE_1 output"),
        "sa0buf":    Buffer("M", "fp32", "F_PHASE_1 output"),
        "gbuf":      Buffer("MxF", "bf16", "F_PHASE_2 output"),
        "a8gbuf":    Buffer("MxF", "fp8", "F_PHASE_3 output"),
        "sa1buf":    Buffer("M", "fp32", "F_PHASE_3 output"),
        "raw1":      Buffer("M", "fp32 bit pattern", "row |g| amax (written by F_PHASE_2, read by F_PHASE_3)"),
    },
    geometry=dict(M=120, H=3072, F=14336, NT=256, SMEM=49152),
    grid_policy="capacity", min_grid=120, coop=True, phases=(1, 2, 3, 4),
))

register_phase(PhaseSpec(
    kernel="vdit.ffn", index=1, does="input quantization (includes norm2 + modulate when use_norm=1)",
    reads=("x_in", "gate_mlp", "shift_mlp", "scale_mlp"), writes=("a8buf", "sa0buf", "raw1"),
    numerics=Numerics(
        norm_kind="layer_plain + modulate (only when use_norm=1)", determinism="fixed",
        rounding_points=("modulate bf16RN step by step", "fp8 quantization"),
        fp8_recipe="amax/448 + software RNE + true division", input_precision="bf16",
        accum_width="fp32", divide="div.rn",
    ),
))
register_phase(PhaseSpec(
    kernel="vdit.ffn", index=2, does="up fp8 GEMM + GELU(tanh) + row amax", tiles=("F_PHASE_2",),
    reads=("a8buf", "sa0buf", "w0", "sw0", "b0"), writes=("gbuf", "raw1"),
    numerics=Numerics(
        determinism="fixed", accum_width="fp32",
        activation="0.5x(1+tanh(c1(x+c2x^3))), c1=sqrt(2/pi) = torch nn.GELU(approximate='tanh') "
                   "and lerobot policies/fastwam/wan; fixed 2026-09-14 (the shared helper "
                   "kernels/common.h `gelu_tanh` used to drop the c1 factor on the cubic term)",
        rounding_points=("bf16RN after dequant+GELU in EPI",),
        input_precision="bf16", divide="div.rn",
    ),
))
register_phase(PhaseSpec(
    kernel="vdit.ffn", index=3, does="gbuf quantization",
    reads=("gbuf", "raw1"), writes=("a8gbuf", "sa1buf"),
    numerics=Numerics(
        determinism="fixed", rounding_points=("fp8 quantization",),
        fp8_recipe="sa1 = max(raw1/448, 1e-12); software RNE + true division",
        accum_width="fp32", divide="div.rn",
    ),
))
register_phase(PhaseSpec(
    kernel="vdit.ffn", index=4, does="down fp8 GEMM + gate_mlp residual", tiles=("F_PHASE_4",),
    reads=("a8gbuf", "sa1buf", "w1", "sw1", "b1", "gate_mlp", "x_in"), writes=("out",),
    numerics=Numerics(
        determinism="fixed", accum_width="fp32",
        residual="**double rounding** (RESID=1): out = bf16RN(bf16RN(z*gate_mlp) + x); "
                 "UMT5's wo_residual is single -- the two chains differ",
        rounding_points=("dequant -> bf16RN(z*gate_mlp) -> bf16RN(+x)",),
        input_precision="bf16", divide="div.rn",
    ),
))
# fmt: on
