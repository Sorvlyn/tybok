// Device-side helpers shared by video DiT (vdit) and UMT5 (tmt5): the part common to the two fp8 GEMM cores.
//
// Provides fwam_bf16r / fwam_resid_bf16 / fwam_cp16 / fwam_gt / fwam_gemm_extra,
// plus the phase-mark idiom FWAM_MARK(ts, p).
//
// ⚠️ tmt5's gelu / fp8-RNE / division use their own convention (tmt5_ffn_*, a bit-exact replica of its
// production chain); do not replace them with the gelu_tanh / fp8_rne from this header.
#ifndef FASTWAM_KERNELS_GEMM_COMMON_H_
#define FASTWAM_KERNELS_GEMM_COMMON_H_

#include <cstdint>

#include "common.h"

// Extra inputs for the RESID epilogue (the non-fused path passes nullptr). The old vdit_gemm.h / tmt5_gemm.h
// each defined a **verbatim-identical** struct under the same name; it now lives here (both family headers were deleted).
//   x_in:      bf16 [M, H] -- both the input (of the FFN / attention) and the residual term
//   gate:      bf16        —— out = x_in + gate * z
//   gate_srow: row stride of gate (in elements); 0 = broadcast (a single [H] row is shared)
struct fwam_gemm_extra {
    const void* x_in;
    const void* gate;
    int H;
    int gate_srow;
};

// Safety guard: bf16f takes a bf16 **bit pattern** (unsigned short), not a float. Passing a float/double gets
// silently truncated by the implicit integer conversion (this has bitten us repeatedly). The deleted overloads turn misuse into a compile error.
__device__ float bf16f(float) = delete;
__device__ float bf16f(double) = delete;
__device__ __forceinline__ float fwam_bf16r(float v) {
    return bf16f(__bfloat16_as_ushort(__float2bfloat16_rn(v)));
}
// FFN residual (RESID=1): o = bf16RN( bf16RN(z * gate) + x )
// z is an fp32 value already rounded to bf16; gate/x are given as raw bf16 bit patterns. Matches torch's
// `x + gate * ffn(x)` bit for bit (bf16 element-wise, opmath=fp32, RN at every step), same as the action kernels.
// NOTE: the fwam_bf16r in this file returns a bf16 value **already converted back to float** (not a bit pattern),
// so it must **not** be wrapped in another bf16f here -- that takes a bf16 bit pattern (unsigned short) and would
// truncate the float to an integer and reinterpret it as a bit pattern (a past bug: the result was always a subnormal near 9e-41).
__device__ __forceinline__ float fwam_resid_bf16(float z, unsigned short g, unsigned short x) {
    const float zg = fwam_bf16r(z * bf16f(g));   // product rounded to bf16
    return fwam_bf16r(zg + bf16f(x));            // add the residual and round to bf16
}

// Unified 16B cp.async wrapper: CG selects .cg (bypasses L1) / .ca; the eviction policy only goes through the
// cache_hint form, whose policy register pol is a runtime value (evict_normal is the default semantics) -- this
// guarantees that in the hot loop every cp.async form is a single instruction with zero branches (a runtime bool
// used to select hint / no-hint, and ptxas emitted both unrolled paths, taking the instruction count from 10.3M to 13.0M).
// zfill is used for zero-filling out-of-bounds rows (src-size=0).
template <bool CG, bool HINT, int PF = 0>
__device__ __forceinline__ void fwam_cp16(unsigned dst, const uint8_t* src,
                                          uint64_t pol, bool zfill) {
    // PF: the .level::prefetch_size suffix (0 = none, 128/256 = L2 prefetch granularity); supported only on the non-zfill path
    #define FWAM_CP_BODY(OPBASE, PFSUF, HSUF, HASPOL)                          \
    do {                                                                       \
        if (zfill) {                                                           \
            if (HASPOL) asm volatile(OPBASE HSUF " [%0], [%1], 16, 0, %2;\n"   \
                                     :: "r"(dst), "l"(src), "l"(pol));         \
            else        asm volatile(OPBASE " [%0], [%1], 16, 0;\n"            \
                                     :: "r"(dst), "l"(src));                   \
        } else if (HASPOL) {                                                   \
            asm volatile(OPBASE PFSUF HSUF " [%0], [%1], 16, %2;\n"            \
                         :: "r"(dst), "l"(src), "l"(pol));                     \
        } else {                                                               \
            asm volatile(OPBASE PFSUF " [%0], [%1], 16;\n"                     \
                         :: "r"(dst), "l"(src));                               \
        }                                                                      \
    } while (0)
    constexpr bool P128 = (PF == 128), P256 = (PF == 256);
    if (CG) {
        if (P128)      { if (HINT) { FWAM_CP_BODY("cp.async.cg.shared.global", ".L2::128B", ".L2::cache_hint", true); }
                         else     { FWAM_CP_BODY("cp.async.cg.shared.global", ".L2::128B", "", false); } }
        else if (P256) { if (HINT) { FWAM_CP_BODY("cp.async.cg.shared.global", ".L2::256B", ".L2::cache_hint", true); }
                         else     { FWAM_CP_BODY("cp.async.cg.shared.global", ".L2::256B", "", false); } }
        else           { if (HINT) { FWAM_CP_BODY("cp.async.cg.shared.global", "", ".L2::cache_hint", true); }
                         else     { FWAM_CP_BODY("cp.async.cg.shared.global", "", "", false); } }
    } else {
        if (P128)      { if (HINT) { FWAM_CP_BODY("cp.async.ca.shared.global", ".L2::128B", ".L2::cache_hint", true); }
                         else     { FWAM_CP_BODY("cp.async.ca.shared.global", ".L2::128B", "", false); } }
        else if (P256) { if (HINT) { FWAM_CP_BODY("cp.async.ca.shared.global", ".L2::256B", ".L2::cache_hint", true); }
                         else     { FWAM_CP_BODY("cp.async.ca.shared.global", ".L2::256B", "", false); } }
        else           { if (HINT) { FWAM_CP_BODY("cp.async.ca.shared.global", "", ".L2::cache_hint", true); }
                         else     { FWAM_CP_BODY("cp.async.ca.shared.global", "", "", false); } }
    }
    #undef FWAM_CP_BODY
}

// Phase timing mark (tick = 1ns): the caller (the timing script) allocates the buffer according to the **actual grid**.
// Old names: va_gt/vg_gt/vf_gt in vdit, ta_gt/tmt5_gt in tmt5 -- five verbatim-identical implementations.
__device__ __forceinline__ unsigned long long fwam_gt() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// Phase timing mark: writes (phase, CTA) into phase_ts[p * gridDim.x + cta] (gridDim.x = the actual grid).
// Each kernel keeps its own `<FAM>_<BLOCK>_MARK(p)` name, which merely aliases this one (five verbatim-identical macro bodies before).
#define FWAM_MARK(ts, p)                                                       \
    do {                                                                       \
        if ((ts) && threadIdx.x == 0)                                          \
            (ts)[(p) * (int)gridDim.x + blockIdx.x] = fwam_gt();               \
    } while (0)

#endif  // FASTWAM_KERNELS_GEMM_COMMON_H_
