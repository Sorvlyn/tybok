// Single-kernel fusion of the UMT5 text encoder's self-attention sublayer (cooperative + non-cooperative split-phase).
//
// Shapes M=128 / H=4096 / NH=64 × HD=64 / S≡128; weights packed qkv fp8[12288,4096] + o fp8[4096,4096].
//
// Phases:
//   S_PHASE_1 RMSNorm + per-token fp8 quantization → a8 + sa0
//   S_PHASE_2 qkv GEMM (EPI=0) → qkv bf16
//   S_PHASE_3 attention (one CTA per head): k/v/pos_bias staged in smem → bf16 mma QKᵀ → +pos_bias+causal
//      → fp32 online softmax (replicates torch's warp version) → bf16 mma PV → attnbuf
//   S_PHASE_4 attn row amax + fp8 quantization → a8o + sa1
//   S_PHASE_5 o GEMM (RESID=2 single-round residual) → out
//   phases are separated by grid.sync.
//
// Numerics: bitwise-aligned with the production chain (not the drift tier). The softmax replicates the
// WARP_BATCH=2 / WARP_ITERATIONS=4 reduction tree of torch's `softmax_warp_forward<...>`, ending in an **IEEE
// division** e/sum; the only non-bitwise part is S_PHASE_1's row reduction order (1ulp rstd on a few layers).
// S_PHASE_3's replication is **zero data movement**: the mma C fragment's lane grouping is isomorphic to torch's {L, L+32, L+64, L+96} at the register level; only the butterfly's offsets 4/2 need shfl_xor.
//
// ⚠️ `bf16f` takes a bf16 **bit pattern** (unsigned short); a float passed in gets truncated; use `fwam_bf16r`.
// ⚠️ Putting the butterfly's offset 1 (the b dimension) inside `for (b)` double-counts the sum; offsets 4/2 must use shfl_xor(2)/(1).
// ⚠️ Do not enable --use_fast_math (expf→__expf, division→reciprocal multiply; the softmax stops being bitwise at once).
// Geometry: S_PHASE_2/S_PHASE_5 use BM128/BN64/BK128/SA=SB=4 (96KB smem) ⇒ 1 CTA/SM; S_PHASE_3 reuses the same dynamic smem.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cstdint>
#include <cstdio>

#include "coop_grid.h"
#include "geom_report.h"          // device capacity query / raise on failure (no longer exit)
#include "common.h"
#include "tmt5_gemm_core.cu"

// ---- geometry (GEMM template parameters): instantiation and self-report use the **same macro set**; editing here = changing geometry ----
#define FWAM_TMT5_ATTN_S_PHASE_2_TILES 128, 64, 128, 4, 4, 2, 4, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, \
                                       0, 0, 1, 0
#define FWAM_TMT5_ATTN_S_PHASE_5_TILES 128, 64, 128, 4, 4, 2, 4, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, \
                                       0, 0, 1, 2

namespace cg = cooperative_groups;

// ------------------------------------------------------------------ //
// shape and geometry constants
// ------------------------------------------------------------------ //
constexpr int TMT5_ATTN_M = 128;      // token count (fixed-length padding)
constexpr int TMT5_ATTN_S = 128;      // sequence length
constexpr int TMT5_ATTN_H = 4096;     // d_model
constexpr int TMT5_ATTN_NH = 64;      // heads
constexpr int TMT5_ATTN_HD = 64;      // head_dim
constexpr int TMT5_ATTN_INNER = TMT5_ATTN_NH * TMT5_ATTN_HD;   // 4096
constexpr int TMT5_ATTN_NT = 256;     // threads/CTA
constexpr int TMT5_ATTN_GRID = 66;    // **nominal** value (1 CTA/SM × 66 SM on this machine); the launcher takes the actual grid from device capacity
// Structural lower bound. The row phases (S_PHASE_1/S_PHASE_4) claim rows with `r = blockIdx.x; r += gridDim.x`; the head phase
// (S_PHASE_3) is `h = blockIdx.x` under an `if (h < TMT5_ATTN_NH)` guard; the GEMM phases (S_PHASE_2/S_PHASE_5) are a 1-D grid-stride
// loop ⇒ all of them cover the whole work, so **there is no structural lower bound**; the grid only affects speed, not numerics.
constexpr int TMT5_ATTN_MIN_GRID = 1;
constexpr int TMT5_ATTN_NWARP = TMT5_ATTN_NT / 32;
constexpr int TMT5_ATTN_SMEM = 98304; // same amount as S_PHASE_2/S_PHASE_5's (128+64)*128*4

// S_PHASE_3's smem layout (in units of bf16 elements; **every row stride gets a pad of 8** to kill bank
// conflicts; pad 8 bf16 = 16B keeps 16B alignment, so whole ranges can be copied as 16B)
constexpr int TMT5_ATTN_SROW = TMT5_ATTN_HD + 8;        // k/v row stride 72
constexpr int TMT5_ATTN_MSROW = TMT5_ATTN_S + 8;        // pos_bias row stride 136
constexpr int TMT5_ATTN_KO = 0;                                    // k  [128][72] = 18432 B
constexpr int TMT5_ATTN_VO = TMT5_ATTN_KO + TMT5_ATTN_S * TMT5_ATTN_SROW * 2;           // v  same as above
constexpr int TMT5_ATTN_PO = TMT5_ATTN_VO + TMT5_ATTN_S * TMT5_ATTN_SROW * 2;           // pos [128][136] = 34816 B
constexpr int TMT5_ATTN_AO = TMT5_ATTN_PO + TMT5_ATTN_S * TMT5_ATTN_MSROW * 2;          // causal mask [128] = 256 B
constexpr int TMT5_ATTN_A3_BYTES = TMT5_ATTN_AO + TMT5_ATTN_S * 2;
static_assert(TMT5_ATTN_A3_BYTES <= TMT5_ATTN_SMEM, "A3 smem over budget");

// Phase timing marks (tick = 1ns): ts[p*gridDim.x + cta]. fwam_gt() is in gemm_common.h (shared by vdit+tmt5).
// The caller (timing script) must size the buffer by the **actual grid**: `nphases * ext.attn_grid()`.
#define TMT5_ATTN_MARK(p)  FWAM_MARK(phase_ts, p)

// The two switches of the non-cooperative fallback (same as the video kernel):
//   TMT5_ATTN_PHASE(k)     —— with only_phase>=0 only that phase is true; with <0 always true (keeps the old behavior).
//   TMT5_ATTN_GRID_SYNC()  —— unchanged on the cooperative path (fence + grid.sync); the split path does nothing.
// NOTE: the split path **never** builds `cg::this_grid()` —— only building/calling a grid barrier imposes co-residency.
#define TMT5_ATTN_PHASE(k)     (only_phase < 0 || only_phase == (k))
#define TMT5_ATTN_GRID_SYNC()  do { if (only_phase < 0) { __threadfence(); cg::this_grid().sync(); } } while (0)

// ------------------------------------------------------------------ //
// bf16 mma m16n8k16 (the fragment mapping is verbatim-identical to attn_probe.cu — that one has already
// been bit-compared against cublas on the real shapes, so do not change any of the mappings here)
// ------------------------------------------------------------------ //
__device__ __forceinline__ uint32_t tmt5_attn_pk2(__nv_bfloat16 lo, __nv_bfloat16 hi) {
    return (uint32_t)__bfloat16_as_ushort(lo) | ((uint32_t)__bfloat16_as_ushort(hi) << 16);
}
__device__ __forceinline__ uint32_t tmt5_attn_pk2f(float lo, float hi) {
    return tmt5_attn_pk2(__float2bfloat16_rn(lo), __float2bfloat16_rn(hi));
}
__device__ __forceinline__ void tmt5_attn_mma_bf16(float* d, const uint32_t* a,
                                            const uint32_t* b, const float* c) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

// ------------------------------------------------------------------ //
// Note: the B-operand delivery of S_PHASE_3's two matmuls was swept over the parameters (scalar LDS / ldmatrix.x4 / cp.async / q through
// smem) — the ldmatrix version is bitwise and 7% faster on the hot convention, but ties on the chain convention (S_PHASE_3 is covered by the
// cold pos_bias traffic), which is why the scalar version is kept.

// ------------------------------------------------------------------ //
// fused kernel
// ------------------------------------------------------------------ //
extern "C" __global__ void __launch_bounds__(TMT5_ATTN_NT, 1)
tmt5_attn_kernel(const uint8_t* __restrict__ x_in,    // bf16 [M,H] attention input (also the residual)
                     const uint8_t* __restrict__ nw,      // bf16 [H] RMSNorm weight
                     const uint8_t* __restrict__ wqkv,    // fp8  [3H,H] packed q|k|v
                     const float*   __restrict__ sqkv,    // fp32 [3H]
                     const uint8_t* __restrict__ bqkv,    // bf16 [3H] (all zeros, same convention as production)
                     const uint8_t* __restrict__ wo,      // fp8  [H,H]
                     const float*   __restrict__ swo,     // fp32 [H]
                     const uint8_t* __restrict__ bo,      // bf16 [H] (all zeros)
                     const uint8_t* __restrict__ posb,    // bf16 [NH,S,S] per-layer relative bias
                     const uint8_t* __restrict__ amask,   // bf16 [S] (causal/pad, broadcast by column)
                     uint8_t* __restrict__ out,           // bf16 [M,H] (**must not alias x_in**)
                     uint8_t* __restrict__ a8buf,         // fp8  [M,H]  S_PHASE_1 output
                     float*   __restrict__ sa0buf,        // fp32 [M]
                     uint8_t* __restrict__ qkvbuf,        // bf16 [M,3H]  S_PHASE_2 output
                     uint8_t* __restrict__ attnbuf,       // bf16 [M,H]   S_PHASE_3 output
                     uint8_t* __restrict__ a8obuf,        // fp8  [M,H]  S_PHASE_4 output
                     float*   __restrict__ sa1buf,        // fp32 [M]
                     int stop_phase,                      // debug: <=0 = run all; 1..5 = run only the first N phases
                     int only_phase,                      // non-cooperative fallback: >=0 = run only that phase; <0 = the full cooperative flow
                     float norm_eps,
                     unsigned long long* phase_ts,
                     float* __restrict__ dbg)             // debug (nullable): head0's [2,S,S] = scores/p
{
    // scratch borrows the last 64B of dynamic smem (same as tmt5_ffn: sm_89 allocates at 8KB
    // granularity, so a static __shared__ is not possible)
    extern __shared__ __align__(16) uint8_t tmt5_smem[];
    float* red = reinterpret_cast<float*>(tmt5_smem + TMT5_ATTN_SMEM - 64);
    float* bc  = red + TMT5_ATTN_NWARP;

    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;

    const int UPTO = (stop_phase <= 0) ? 5 : stop_phase;
    TMT5_ATTN_MARK(0);

    // ---------------- S_PHASE_1: RMSNorm + per-token fp8 quantization -------------------- //
    // **Line-for-line identical** to tmt5_ffn's F_PHASE_1 (the same production operator, norm_quantize).
    if (UPTO >= 1 && TMT5_ATTN_PHASE(1)) {
        for (int r = blockIdx.x; r < TMT5_ATTN_M; r += (int)gridDim.x) {
            const unsigned short* __restrict__ xr =
                reinterpret_cast<const unsigned short*>(x_in + (size_t)r * TMT5_ATTN_H * 2);
            const unsigned short* __restrict__ nwu =
                reinterpret_cast<const unsigned short*>(nw);
            float ss = 0.f;
            for (int i = tid; i < TMT5_ATTN_H; i += TMT5_ATTN_NT) {
                const float f = bf16f(xr[i]);
                ss = fmaf(f, f, ss);
            }
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
            if (lane == 0) red[wid] = ss;
            __syncthreads();
            if (wid == 0) {
                float v = (lane < TMT5_ATTN_NWARP) ? red[lane] : 0.f;
                #pragma unroll
                for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
                if (lane == 0) bc[0] = 1.0f / sqrtf(v / (float)TMT5_ATTN_H + norm_eps);
            }
            __syncthreads();
            const float rstd = bc[0];

            float am = 0.f;
            for (int i = tid; i < TMT5_ATTN_H; i += TMT5_ATTN_NT) {
                const float xf = bf16f(xr[i]);
                const float n = fwam_bf16r(xf * rstd);
                const float nb = fwam_bf16r(bf16f(nwu[i]) * n);
                am = fmaxf(am, fabsf(nb));
            }
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
            if (lane == 0) red[wid] = am;
            __syncthreads();
            if (wid == 0) {
                float v = (lane < TMT5_ATTN_NWARP) ? red[lane] : 0.f;
                #pragma unroll
                for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
                if (lane == 0) bc[1] = fmaxf(v * (1.0f / 448.0f), 1e-12f);
            }
            __syncthreads();
            const float sa = bc[1];
            if (tid == 0) sa0buf[r] = sa;

            uint8_t* qr = a8buf + (size_t)r * TMT5_ATTN_H;
            for (int i = tid; i < TMT5_ATTN_H; i += TMT5_ATTN_NT) {
                const float xf = bf16f(xr[i]);
                const float n = fwam_bf16r(xf * rstd);
                const float nb = fwam_bf16r(bf16f(nwu[i]) * n);
                qr[i] = tmt5_ffn_fp8_rne(tmt5_ffn_div_full(nb, sa));
            }
            __syncthreads();
        }
    }
    TMT5_ATTN_MARK(1);
    TMT5_ATTN_GRID_SYNC();
    TMT5_ATTN_MARK(2);

    // ---------------- S_PHASE_2: qkv GEMM (N=3H=12288, K=4096) -------------------- //
    if (UPTO >= 2 && TMT5_ATTN_PHASE(2)) {
    fwam_fp8_gemm_body<FWAM_TMT5_ATTN_S_PHASE_2_TILES>(
        /*A=*/a8buf, /*W=*/wqkv, /*sa=*/sa0buf, /*sw=*/sqkv,
        /*bias=*/reinterpret_cast<const __nv_bfloat16*>(bqkv),
        /*Cout=*/reinterpret_cast<__nv_bfloat16*>(qkvbuf),
        /*partial=*/nullptr, /*counters=*/nullptr,
        /*M=*/TMT5_ATTN_M, /*N=*/3 * TMT5_ATTN_H, /*K=*/TMT5_ATTN_H, /*Kseg=*/TMT5_ATTN_H,
        /*wp=*/11, /*ap=*/0, /*col0=*/0, /*ffn_ex=*/nullptr);
    }
    TMT5_ATTN_MARK(3);
    TMT5_ATTN_GRID_SYNC();
    TMT5_ATTN_MARK(4);

    // ---------------- S_PHASE_3: attention (one CTA per head) --------------------- //
    // Bitwise replication of production's three steps (`triton_text_encoder_attn.eager_attention`):
    //   scores = bf16RN(q @ kᵀ)                 ← mma, the k order accumulates over 16-blocks
    //   scores = bf16RN(scores + bf16RN(pos + mask))   ← torch's two bf16 RN additions
    //   p = softmax_fp32(scores);  out = bf16RN(p @ v)
    // The softmax reduction tree: see the "zero data movement" argument in the file header comment.
    if (UPTO >= 3 && TMT5_ATTN_PHASE(3)) {
        const int h = blockIdx.x;
        if (h < TMT5_ATTN_NH) {
            const int warp = tid >> 5, l = tid & 31;
            const int g = l >> 2, t = l & 3;

            __nv_bfloat16* ksm = reinterpret_cast<__nv_bfloat16*>(tmt5_smem + TMT5_ATTN_KO);
            __nv_bfloat16* vsm = reinterpret_cast<__nv_bfloat16*>(tmt5_smem + TMT5_ATTN_VO);
            // pos/mask are read as **bf16 bit patterns (unsigned short)**: bf16f takes a bit pattern
            const unsigned short* psu = reinterpret_cast<const unsigned short*>(tmt5_smem + TMT5_ATTN_PO);
            const unsigned short* csu = reinterpret_cast<const unsigned short*>(tmt5_smem + TMT5_ATTN_AO);

            // ---- stage k / v (the two qkv sections, 64 columns per head; row stride 3H bf16 = 24576 B)
            //      plus pos_bias[h] (contiguous 32KB); everything is 16B-aligned and copied as whole ranges ----
            const uint8_t* kb = qkvbuf + (size_t)TMT5_ATTN_INNER * 2 + (size_t)h * TMT5_ATTN_HD * 2;
            const uint8_t* vb = qkvbuf + (size_t)2 * TMT5_ATTN_INNER * 2 + (size_t)h * TMT5_ATTN_HD * 2;
            for (int i = tid; i < TMT5_ATTN_S * 8; i += TMT5_ATTN_NT) {
                const int row = i >> 3, ch = (i & 7) << 4;
                const size_t so = (size_t)row * (3 * TMT5_ATTN_INNER * 2) + ch;
                *reinterpret_cast<uint4*>(tmt5_smem + TMT5_ATTN_KO + row * (TMT5_ATTN_SROW * 2) + ch) =
                    *reinterpret_cast<const uint4*>(kb + so);
                *reinterpret_cast<uint4*>(tmt5_smem + TMT5_ATTN_VO + row * (TMT5_ATTN_SROW * 2) + ch) =
                    *reinterpret_cast<const uint4*>(vb + so);
            }
            const uint8_t* pb = posb + (size_t)h * TMT5_ATTN_S * TMT5_ATTN_S * 2;
            for (int i = tid; i < TMT5_ATTN_S * 16; i += TMT5_ATTN_NT) {
                const int row = i >> 4, ch = (i & 15) << 4;
                *reinterpret_cast<uint4*>(tmt5_smem + TMT5_ATTN_PO + row * (TMT5_ATTN_MSROW * 2) + ch) =
                    *reinterpret_cast<const uint4*>(pb + (size_t)row * TMT5_ATTN_S * 2 + ch);
            }
            if (tid < 16)
                *reinterpret_cast<uint4*>(tmt5_smem + TMT5_ATTN_AO + tid * 16) =
                    *reinterpret_cast<const uint4*>(amask + tid * 16);
            __syncthreads();

            // ---- QKᵀ: A reads the q section of qkv directly (L2 hit, each element read once), B reads ksm ----
            float sv[16][4];
            #pragma unroll
            for (int nt = 0; nt < 16; nt++)
                #pragma unroll
                for (int i = 0; i < 4; i++) sv[nt][i] = 0.f;

            const int row = warp * 16 + g;   // the global row this lane owns (8 more rows below)
            const __nv_bfloat16* qrow =
                reinterpret_cast<const __nv_bfloat16*>(qkvbuf) + (size_t)row * (3 * TMT5_ATTN_INNER) + h * TMT5_ATTN_HD;
            #pragma unroll
            for (int kk = 0; kk < 4; kk++) {
                const int d0 = kk * 16 + 2 * t;
                uint32_t a[4];
                a[0] = tmt5_attn_pk2(qrow[d0], qrow[d0 + 1]);
                a[1] = tmt5_attn_pk2(qrow[8 * 3 * TMT5_ATTN_INNER + d0], qrow[8 * 3 * TMT5_ATTN_INNER + d0 + 1]);
                a[2] = tmt5_attn_pk2(qrow[d0 + 8], qrow[d0 + 9]);
                a[3] = tmt5_attn_pk2(qrow[8 * 3 * TMT5_ATTN_INNER + d0 + 8], qrow[8 * 3 * TMT5_ATTN_INNER + d0 + 9]);
                #pragma unroll
                for (int nt = 0; nt < 16; nt++) {
                    const int j = nt * 8 + g;
                    uint32_t b[2];
                    b[0] = tmt5_attn_pk2(ksm[j * TMT5_ATTN_SROW + d0], ksm[j * TMT5_ATTN_SROW + d0 + 1]);
                    b[1] = tmt5_attn_pk2(ksm[j * TMT5_ATTN_SROW + d0 + 8], ksm[j * TMT5_ATTN_SROW + d0 + 9]);
                    tmt5_attn_mma_bf16(sv[nt], a, b, sv[nt]);
                }
            }

            // ---- + pos_bias + causal (two bf16 RN, the same shape as torch's two elementwise ops) ----
            #pragma unroll
            for (int nt = 0; nt < 16; nt++)
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    const int rr = row + ((i >> 1) & 1) * 8;
                    const int cc = nt * 8 + 2 * t + (i & 1);
                    const float mv = fwam_bf16r(bf16f(psu[rr * TMT5_ATTN_MSROW + cc]) +
                                                bf16f(csu[cc]));
                    // ⚠️ NOTE: fwam_bf16r returns a **bf16 value already converted back to float** (not a bit pattern) —
                    // it **must not** be wrapped in bf16f here (that one takes unsigned short, so it would truncate the float to an integer and read it as a bit pattern)
                    sv[nt][i] = fwam_bf16r(fwam_bf16r(sv[nt][i]) + mv);
                }
            // Debug: dump the masked scores (head 0)
            if (dbg != nullptr && h == 0) {
                #pragma unroll
                for (int nt = 0; nt < 16; nt++)
                    #pragma unroll
                    for (int i = 0; i < 4; i++)
                        dbg[(size_t)(row + ((i >> 1) & 1) * 8) * TMT5_ATTN_S + nt * 8 + 2 * t + (i & 1)]
                            = sv[nt][i];
            }

            // ---- softmax (fp32, replicates torch's softmax_warp_forward) ----
            // Virtual lanes: L = 8·nt0 + 2t + b, holding the columns {L, L+32, L+64, L+96} = this lane's
            // sv[nt0 + 4i][b + 2·rs]. The butterfly offsets 16/8/1 stay inside this lane's registers;
            // only 4/2 are shfl_xor(8)/(4).
            float mxs[2][2][4];
            #pragma unroll
            for (int rs = 0; rs < 2; rs++)
                #pragma unroll
                for (int b = 0; b < 2; b++)
                    #pragma unroll
                    for (int nt0 = 0; nt0 < 4; nt0++) {
                        const int off = b + 2 * rs;
                        float v = sv[nt0][off];
                        v = fmaxf(v, sv[nt0 + 4][off]);
                        v = fmaxf(v, sv[nt0 + 8][off]);
                        v = fmaxf(v, sv[nt0 + 12][off]);
                        mxs[rs][b][nt0] = v;
                    }
            #pragma unroll
            for (int rs = 0; rs < 2; rs++) {
                // ---- offsets 16 / 8: the nt0 dimension, inside this lane's registers ----
                #pragma unroll
                for (int b = 0; b < 2; b++) {
                    { const float x0 = mxs[rs][b][0], x2 = mxs[rs][b][2];
                      mxs[rs][b][0] = fmaxf(x0, x2); mxs[rs][b][2] = fmaxf(x2, x0); }
                    { const float x1 = mxs[rs][b][1], x3 = mxs[rs][b][3];
                      mxs[rs][b][1] = fmaxf(x1, x3); mxs[rs][b][3] = fmaxf(x3, x1); }
                    { const float x0 = mxs[rs][b][0], x1 = mxs[rs][b][1];
                      mxs[rs][b][0] = fmaxf(x0, x1); mxs[rs][b][1] = fmaxf(x1, x0); }
                    { const float x2 = mxs[rs][b][2], x3 = mxs[rs][b][3];
                      mxs[rs][b][2] = fmaxf(x2, x3); mxs[rs][b][3] = fmaxf(x3, x2); }
                    // ---- offsets 4 / 2: the t dimension, across lanes ----
                    // ⚠️ lane = 4·g + t (g = lane>>2, t = lane&3), so
                    //    t^2 = lane^2 and t^1 = lane^1 — **not** ^8/^4 (those move g and would mix data across rows)
                    #pragma unroll
                    for (int nt0 = 0; nt0 < 4; nt0++) {
                        const float o4 = __shfl_xor_sync(0xffffffffu, mxs[rs][b][nt0], 2);
                        mxs[rs][b][nt0] = fmaxf(mxs[rs][b][nt0], o4);
                    }
                    #pragma unroll
                    for (int nt0 = 0; nt0 < 4; nt0++) {
                        const float o2 = __shfl_xor_sync(0xffffffffu, mxs[rs][b][nt0], 1);
                        mxs[rs][b][nt0] = fmaxf(mxs[rs][b][nt0], o2);
                    }
                }
                // ---- offset 1: the b dimension, inside this lane's registers ----
                // ⚠️ This step **must stay outside the b loop**: it updates both sides at once, so
                // inside for(b) it would run twice (invisible for the idempotent max, but the sum would double-count).
                #pragma unroll
                for (int nt0 = 0; nt0 < 4; nt0++) {
                    const float x0 = mxs[rs][0][nt0], x1 = mxs[rs][1][nt0];
                    mxs[rs][0][nt0] = fmaxf(x0, x1); mxs[rs][1][nt0] = fmaxf(x1, x0);
                }
            }

            float sms[2][2][4];
            #pragma unroll
            for (int rs = 0; rs < 2; rs++)
                #pragma unroll
                for (int b = 0; b < 2; b++)
                    #pragma unroll
                    for (int nt0 = 0; nt0 < 4; nt0++) sms[rs][b][nt0] = 0.f;
            #pragma unroll
            for (int nt = 0; nt < 16; nt++)
                #pragma unroll
                for (int rs = 0; rs < 2; rs++)
                    #pragma unroll
                    for (int b = 0; b < 2; b++) {
                        const int off = b + 2 * rs;
                        const float e = expf(sv[nt][off] - mxs[rs][b][nt & 3]);
                        sv[nt][off] = e;
                        sms[rs][b][nt & 3] += e;
                    }
            #pragma unroll
            for (int rs = 0; rs < 2; rs++) {
                #pragma unroll
                for (int b = 0; b < 2; b++) {
                    { const float x0 = sms[rs][b][0], x2 = sms[rs][b][2];
                      sms[rs][b][0] = x0 + x2; sms[rs][b][2] = x2 + x0; }
                    { const float x1 = sms[rs][b][1], x3 = sms[rs][b][3];
                      sms[rs][b][1] = x1 + x3; sms[rs][b][3] = x3 + x1; }
                    { const float x0 = sms[rs][b][0], x1 = sms[rs][b][1];
                      sms[rs][b][0] = x0 + x1; sms[rs][b][1] = x1 + x0; }
                    { const float x2 = sms[rs][b][2], x3 = sms[rs][b][3];
                      sms[rs][b][2] = x2 + x3; sms[rs][b][3] = x3 + x2; }
                    #pragma unroll
                    for (int nt0 = 0; nt0 < 4; nt0++) {   // offset 4 → t^2 = lane^2
                        const float o4 = __shfl_xor_sync(0xffffffffu, sms[rs][b][nt0], 2);
                        sms[rs][b][nt0] = sms[rs][b][nt0] + o4;
                    }
                    #pragma unroll
                    for (int nt0 = 0; nt0 < 4; nt0++) {   // offset 2 → t^1 = lane^1
                        const float o2 = __shfl_xor_sync(0xffffffffu, sms[rs][b][nt0], 1);
                        sms[rs][b][nt0] = sms[rs][b][nt0] + o2;
                    }
                }
                // offset 1 (the b dimension) — as with max: outside the b loop, done once
                #pragma unroll
                for (int nt0 = 0; nt0 < 4; nt0++) {
                    const float x0 = sms[rs][0][nt0], x1 = sms[rs][1][nt0];
                    sms[rs][0][nt0] = x0 + x1; sms[rs][1][nt0] = x1 + x0;
                }
            }
            // p = e / sum (**IEEE division**; production torch is e/sum, do not change it to a reciprocal multiply)
            #pragma unroll
            for (int nt = 0; nt < 16; nt++)
                #pragma unroll
                for (int rs = 0; rs < 2; rs++)
                    #pragma unroll
                    for (int b = 0; b < 2; b++)
                        sv[nt][b + 2 * rs] = sv[nt][b + 2 * rs] / sms[rs][b][nt & 3];
            if (dbg != nullptr && h == 0) {
                #pragma unroll
                for (int nt = 0; nt < 16; nt++)
                    #pragma unroll
                    for (int i = 0; i < 4; i++)
                        dbg[(size_t)TMT5_ATTN_S * TMT5_ATTN_S +
                            (row + ((i >> 1) & 1) * 8) * TMT5_ATTN_S + nt * 8 + 2 * t + (i & 1)]
                            = sv[nt][i];
            }

            // ---- PV: the A fragment comes straight from sv (the same mma layout, zero movement), B reads vsm ----
            float oacc[8][4];
            #pragma unroll
            for (int nt = 0; nt < 8; nt++)
                #pragma unroll
                for (int i = 0; i < 4; i++) oacc[nt][i] = 0.f;
            #pragma unroll
            for (int kk = 0; kk < 8; kk++) {
                uint32_t a[4];
                a[0] = tmt5_attn_pk2f(sv[kk * 2][0], sv[kk * 2][1]);
                a[1] = tmt5_attn_pk2f(sv[kk * 2][2], sv[kk * 2][3]);
                a[2] = tmt5_attn_pk2f(sv[kk * 2 + 1][0], sv[kk * 2 + 1][1]);
                a[3] = tmt5_attn_pk2f(sv[kk * 2 + 1][2], sv[kk * 2 + 1][3]);
                #pragma unroll
                for (int nt = 0; nt < 8; nt++) {
                    const int d = nt * 8 + g;
                    uint32_t b[2];
                    b[0] = tmt5_attn_pk2(vsm[(kk * 16 + 2 * t) * TMT5_ATTN_SROW + d],
                                  vsm[(kk * 16 + 2 * t + 1) * TMT5_ATTN_SROW + d]);
                    b[1] = tmt5_attn_pk2(vsm[(kk * 16 + 2 * t + 8) * TMT5_ATTN_SROW + d],
                                  vsm[(kk * 16 + 2 * t + 9) * TMT5_ATTN_SROW + d]);
                    tmt5_attn_mma_bf16(oacc[nt], a, b, oacc[nt]);
                }
            }
            __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(attnbuf) + h * TMT5_ATTN_HD;
            #pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                const int d = nt * 8 + 2 * t;
                *reinterpret_cast<__nv_bfloat162*>(ob + (size_t)row * TMT5_ATTN_INNER + d) =
                    __floats2bfloat162_rn(oacc[nt][0], oacc[nt][1]);
                *reinterpret_cast<__nv_bfloat162*>(ob + (size_t)(row + 8) * TMT5_ATTN_INNER + d) =
                    __floats2bfloat162_rn(oacc[nt][2], oacc[nt][3]);
            }
        }
        __syncthreads();   // smem must be handed back before S_PHASE_4/S_PHASE_5
    }
    TMT5_ATTN_MARK(5);
    TMT5_ATTN_GRID_SYNC();
    TMT5_ATTN_MARK(6);

    // ---------------- S_PHASE_4: attn row amax + fp8 quantization ------------------------- //
    // `quantize_act(attn_out)` inside production's `o_proj_residual`: per-token amax → sa →
    // true division + RNE. attn_out was just written by S_PHASE_3 (1MB, L2 hit), scanned twice.
    if (UPTO >= 4 && TMT5_ATTN_PHASE(4)) {
        const unsigned short* ab = reinterpret_cast<const unsigned short*>(attnbuf);
        for (int r = blockIdx.x; r < TMT5_ATTN_M; r += (int)gridDim.x) {
            const unsigned short* ar = ab + (size_t)r * TMT5_ATTN_H;
            float am = 0.f;
            for (int i = tid; i < TMT5_ATTN_H; i += TMT5_ATTN_NT) am = fmaxf(am, fabsf(bf16f(ar[i])));
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
            if (lane == 0) red[wid] = am;
            __syncthreads();
            if (wid == 0) {
                float v = (lane < TMT5_ATTN_NWARP) ? red[lane] : 0.f;
                #pragma unroll
                for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
                if (lane == 0) bc[1] = fmaxf(v * (1.0f / 448.0f), 1e-12f);
            }
            __syncthreads();
            const float sa = bc[1];
            if (tid == 0) sa1buf[r] = sa;
            uint8_t* qr = a8obuf + (size_t)r * TMT5_ATTN_H;
            for (int i = tid; i < TMT5_ATTN_H; i += TMT5_ATTN_NT)
                qr[i] = tmt5_ffn_fp8_rne(tmt5_ffn_div_full(bf16f(ar[i]), sa));
            __syncthreads();
        }
    }
    TMT5_ATTN_MARK(7);
    TMT5_ATTN_GRID_SYNC();
    TMT5_ATTN_MARK(8);

    // ---------------- S_PHASE_5: o GEMM (RESID=2 single-round residual) ------------------- //
    // S_PHASE_5 originally had no UPTO guard (stop_phase only gates S_PHASE_1..S_PHASE_4; the last phase always runs);
    // only the "is this the current phase" test is done here, and it is always true when only_phase<0 ⇒ verbatim-identical to the old behavior.
    if (TMT5_ATTN_PHASE(5)) {
    fwam_gemm_extra ex{x_in, nullptr, TMT5_ATTN_H, 0};
    fwam_fp8_gemm_body<FWAM_TMT5_ATTN_S_PHASE_5_TILES>(
        /*A=*/a8obuf, /*W=*/wo, /*sa=*/sa1buf, /*sw=*/swo,
        /*bias=*/reinterpret_cast<const __nv_bfloat16*>(bo),
        /*Cout=*/reinterpret_cast<__nv_bfloat16*>(out),
        /*partial=*/nullptr, /*counters=*/nullptr,
        /*M=*/TMT5_ATTN_M, /*N=*/TMT5_ATTN_H, /*K=*/TMT5_ATTN_H, /*Kseg=*/TMT5_ATTN_H,
        /*wp=*/11, /*ap=*/0, /*col0=*/0, /*ffn_ex=*/&ex);
    }
    TMT5_ATTN_MARK(9);
}

// ------------------------------------------------------------------ //
// host: the grid is decided by device capacity (occupancy × SM count); 66 = this machine's SM count is no longer hard-coded
// ------------------------------------------------------------------ //
static int g_tmt5_attn_split_grid = -1;
static int tmt5_attn_grid_split() {
    return cached_split_grid(g_tmt5_attn_split_grid, (const void*)tmt5_attn_kernel, TMT5_ATTN_NT,
                             TMT5_ATTN_SMEM, TMT5_ATTN_MIN_GRID, "tmt5_attn");
}
static int g_tmt5_attn_coop_grid = -1;
static int tmt5_attn_grid_coop() {
    return cached_coop_grid(g_tmt5_attn_coop_grid, (const void*)tmt5_attn_kernel, TMT5_ATTN_NT,
                            TMT5_ATTN_SMEM, TMT5_ATTN_MIN_GRID, "tmt5_attn");
}
// Lets the timing script size the phase_ts buffer by the **actual grid** (the span is gridDim.x, not a hard-coded 66).

// ---- geometry self-report (shape constants + GEMM template parameters; the instantiation uses the same macro set) ----
extern "C" const char* tmt5_attn_geom() {
    static const std::string s = fwam_geom_join({
        fwam_shapes({{"M", TMT5_ATTN_M}, {"S", TMT5_ATTN_S}, {"H", TMT5_ATTN_H}, {"NH", TMT5_ATTN_NH}, {"HD", TMT5_ATTN_HD}, {"NT", TMT5_ATTN_NT}, {"SMEM", TMT5_ATTN_SMEM}, {"GRID", TMT5_ATTN_GRID}}),
        fwam_tiles("S_PHASE_2", FWAM_TILES(FWAM_TMT5_ATTN_S_PHASE_2_TILES)),
        fwam_tiles("S_PHASE_5", FWAM_TILES(FWAM_TMT5_ATTN_S_PHASE_5_TILES)),
    });
    return s.c_str();
}

extern "C" int tmt5_attn_grid() { return tmt5_attn_grid_split(); }
extern "C" int tmt5_attn_coop_grid() { return tmt5_attn_grid_coop(); }

// ------------------------------------------------------------------ //
// host: cooperative / non-cooperative split-phase launch
// ------------------------------------------------------------------ //
extern "C" void tmt5_attn_cuda(const void* x_in, const void* nw,
                                   const void* wqkv, const float* sqkv, const void* bqkv,
                                   const void* wo, const float* swo, const void* bo,
                                   const void* posb, const void* amask,
                                   void* out, void* a8buf, float* sa0buf,
                                   void* qkvbuf, void* attnbuf, void* a8obuf, float* sa1buf,
                                   int stop_phase, float norm_eps,
                                   unsigned long long* phase_ts, float* dbg,
                                   int only_phase, cudaStream_t stream)
{
    dim3 block(TMT5_ATTN_NT);
    void* args[] = {
        (void*)&x_in, (void*)&nw, (void*)&wqkv, (void*)&sqkv, (void*)&bqkv,
        (void*)&wo, (void*)&swo, (void*)&bo, (void*)&posb, (void*)&amask,
        (void*)&out, (void*)&a8buf, (void*)&sa0buf,
        (void*)&qkvbuf, (void*)&attnbuf, (void*)&a8obuf, (void*)&sa1buf,
        (void*)&stop_phase, (void*)&only_phase, (void*)&norm_eps, (void*)&phase_ts,
        (void*)&dbg,
    };
    cudaError_t e;
    dim3 grid;
    if (only_phase < 0) {
        grid = dim3(tmt5_attn_grid_coop());
        e = cudaLaunchCooperativeKernel((void*)tmt5_attn_kernel, grid,
                                        block, args, TMT5_ATTN_SMEM, stream);
    } else {
        grid = dim3(tmt5_attn_grid_split());
        e = cudaLaunchKernel((void*)tmt5_attn_kernel, grid, block, args,
                             TMT5_ATTN_SMEM, stream);
    }
    if (e != cudaSuccess) {
        coop_fail("tmt5_attn",
                  "launch failed (only_phase=" + std::to_string(only_phase)
                  + ", grid=" + std::to_string((int)grid.x) + "): "
                  + cudaGetErrorString(e));
    }
}
