// One cooperative kernel fusing the whole video-expert cross-attention chain (6 phases / 5 grid.sync).
//
// Replaces the torch chain: norm3(affine LN) → cross q/k/v projection → qk-norm(q,k) → SDPA → cross o residual.
//
// Differences from the self version:
//   - norm3 is an **affine** LayerNorm (with weight/bias).
//   - k/v are computed on the fly from context (C_PHASE_1 additionally quantizes context).
//   - no RoPE.
//   - kv is 129 rows (9 tiles; the tail tile has only 1 valid row → 15 rows are set to -inf).
//   - the residual is **ungated**: out = x + z, reusing the RESID path with ones as the gate.
//
// Phases:
//   C_PHASE_1 [249 rows grid-stride] norm3+quantize x / quantize context → a8q/sa0q, a8c/sa0c
//   C_PHASE_2 [full grid] cross-q GEMM (M=120,N=3072) immediately followed by cross-kv GEMM (M=129,N=6144)
//   C_PHASE_3 [249 rows grid-stride] q row RMS+wnq → qc; kv row RMS+wnk (k half only) → kc, v as-is → vc
//   C_PHASE_4 [b<96]   bf16 flash (q 120 × kv 129)
//   C_PHASE_5 [b<120]  attn row quantization
//   C_PHASE_6 [full grid] cross-o GEMM + ungated residual → out
//
// only_phase (non-cooperative fallback) has the same semantics as in the self version.
#include <cooperative_groups.h>
#include "vdit_gemm_core.cu"   // fwam_fp8_gemm_body + device-side helpers (shared by the three vdit kernels)
#include "coop_grid.h"
#include "geom_report.h"            // grid comes from the device capacity (occupancy × SM)
#include "dit_common.h"

// ---- Geometry (GEMM template parameters): instantiation and self-report share **one macro set**; changing this = changing the geometry ----
#define FWAM_VDIT_CROSS_C_PHASE_2_2G_TILES 64, 64, 128, 3, 3, 4, 2, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, \
                                           0, 1, 0
#define FWAM_VDIT_CROSS_C_PHASE_6_TILES 64, 64, 128, 3, 3, 4, 2, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, \
                                        0, 1, 1

namespace cg = cooperative_groups;

constexpr int VDIT_CROSS_M = 120;         // video tokens
constexpr int VDIT_CROSS_C = 129;         // context tokens
constexpr int VDIT_CROSS_H = 3072;        // hidden
constexpr int VDIT_CROSS_H3 = 3072;       // attention width
constexpr int VDIT_CROSS_NKV = 6144;      // packed cross k/v output width (2 × H3)
constexpr int VDIT_CROSS_ROWS = VDIT_CROSS_M + VDIT_CROSS_C;   // 249
constexpr int VDIT_CROSS_NT = 256;
constexpr int VDIT_CROSS_GRID = 132;   // nominal value; the actual grid is taken by the launcher from the device capacity (see vg_grid())
constexpr int VDIT_CROSS_NWARP = VDIT_CROSS_NT / 32;
constexpr int VDIT_CROSS_SMEM = 49152;

// flash geometry (isomorphic to the self version; kv becomes 129 rows → 9 tiles)
constexpr int VDIT_CROSS_QT = 32;
constexpr int VDIT_CROSS_NQT = (VDIT_CROSS_M + VDIT_CROSS_QT - 1) / VDIT_CROSS_QT;   // 4
constexpr int VDIT_CROSS_NH = 24;
constexpr int VDIT_CROSS_D = 128;
constexpr int VDIT_CROSS_FGRID = VDIT_CROSS_NH * VDIT_CROSS_NQT;             // 96
constexpr int VDIT_CROSS_TILE = 16;
constexpr int VDIT_CROSS_BP = 136;
constexpr int VDIT_CROSS_NSTAGE = 2;

// Phase timing marks: fwam_gt() in gemm_common.h (shared by vdit+tmt5)
#define VDIT_CROSS_MARK(p)  FWAM_MARK(phase_ts, p)

#define VDIT_CROSS_PHASE(k)     (only_phase < 0 || only_phase == (k))
#define VDIT_CROSS_GRID_SYNC()  do { if (only_phase < 0) { __threadfence(); cg::this_grid().sync(); } } while (0)


// Row RMS of one row (across the whole 3072): returns rstd. All 256 threads take part.
__device__ __forceinline__ float vdit_cross_row_rms(const __nv_bfloat16* __restrict__ row,
                                            float* __restrict__ sc, int tid, int lane,
                                            int wid, float eps) {
    float ss = 0.f;
    #pragma unroll
    for (int i = 0; i < VDIT_CROSS_H3 / VDIT_CROSS_NT; ++i) {
        const float v = bf16f(__bfloat16_as_ushort(row[tid + VDIT_CROSS_NT * i]));
        ss += v * v;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off);
    if (lane == 0) sc[wid] = ss;
    __syncthreads();
    if (tid == 0) {
        float a = 0.f;
        #pragma unroll
        for (int i = 0; i < VDIT_CROSS_NWARP; ++i) a += sc[i];
        sc[VDIT_CROSS_NWARP] = a;                  // only the **sum of squares** is broadcast; each thread computes the rstd itself
    }
    __syncthreads();
    // Each thread derives the rstd from the broadcast sum itself: one fewer dependent write-read smem round trip,
    // and it avoids the inconsistency that "tid0 writes sc[8] → everyone reads sc[8]" produces under __restrict__
    // (measured 2026-09: the sum was correct but the rstd read back was garbage).
    // volatile: forces this read to happen **after** __syncthreads(). With __restrict__ on sc the compiler
    // believes "only this thread writes it" and hoists the read ahead of tid0's write (measured: the sum was written correctly
    // but the value read back was garbage/NaN).
    const float rq = rsqrtf(((volatile float*)sc)[VDIT_CROSS_NWARP] / (float)VDIT_CROSS_H3 + eps);
    __syncthreads();          // sync before the next row (grid-stride) reuses sc
    return rq;
}

__global__ void __launch_bounds__(VDIT_CROSS_NT, 2)
vdit_attn_cross_kernel(
    const uint8_t* __restrict__ x_in,       // bf16 [M,H] block input (cross residual)
    const uint8_t* __restrict__ context,    // bf16 [C,H]
    const uint8_t* __restrict__ w3,         // bf16 [H]  norm3 weight
    const uint8_t* __restrict__ b3,         // bf16 [H]  norm3 bias
    const uint8_t* __restrict__ wq,         // fp8  [H3,H]  cross q
    const float* __restrict__ swq,          // fp32 [H3]
    const uint8_t* __restrict__ bq,         // bf16 [H3]
    const uint8_t* __restrict__ wkv,        // fp8  [6144,H] cross k/v (packed)
    const float* __restrict__ swkv,         // fp32 [6144]
    const uint8_t* __restrict__ bkv,        // bf16 [6144]
    const uint8_t* __restrict__ wnq,        // bf16 [H3]
    const uint8_t* __restrict__ wnk,        // bf16 [H3]
    const uint8_t* __restrict__ wo,         // fp8  [H,H3]  cross o
    const float* __restrict__ swo,          // fp32 [H]
    const uint8_t* __restrict__ bo,         // bf16 [H]
    const uint8_t* __restrict__ ones,       // bf16 [H]  ones used by the ungated residual
    uint8_t* __restrict__ out,              // bf16 [M,H]
    uint8_t* __restrict__ qp,               // bf16 [M,H3]     C_PHASE_2 output
    uint8_t* __restrict__ kvp,              // bf16 [C,NKV]    C_PHASE_2 output
    uint8_t* __restrict__ qc,               // bf16 [M,H3]     C_PHASE_3 output (q after norm)
    uint8_t* __restrict__ kc,               // bf16 [C,H3]     C_PHASE_3 output (k after norm)
    uint8_t* __restrict__ vc,               // bf16 [C,H3]     C_PHASE_3 output (v as-is)
    uint8_t* __restrict__ attn,             // fp32 [M,H3]
    uint8_t* __restrict__ a8q, float* __restrict__ sa0q,   // fp8/fp32 [M]
    uint8_t* __restrict__ a8c, float* __restrict__ sa0c,   // fp8/fp32 [C]
    uint8_t* __restrict__ a8o, float* __restrict__ sa0o,   // fp8/fp32 [M]
    float norm_eps,
    int stop_phase,
    int only_phase,           // non-cooperative fallback: >=0 runs that phase only; <0 = the cooperative full pipeline
    float* dbg,
    unsigned long long* phase_ts)
{
    extern __shared__ __align__(16) uint8_t vg_smem[];
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    float* sc = reinterpret_cast<float*>(vg_smem);   // row-level scratch (borrowed by each phase)

    VDIT_CROSS_MARK(0);

    // ================= C_PHASE_1: norm3 + quantize x / quantize context (249 rows grid-stride) =================
    if (VDIT_CROSS_PHASE(1)) {
        constexpr int PER = VDIT_CROSS_H / 2 / VDIT_CROSS_NT;        // u32s per thread (= 6)
        for (int u = b; u < VDIT_CROSS_ROWS; u += gridDim.x) {
            const bool is_x = (u < VDIT_CROSS_M);
            const uint8_t* src = is_x ? (x_in + (size_t)u * VDIT_CROSS_H * 2)
                                      : (context + (size_t)(u - VDIT_CROSS_M) * VDIT_CROSS_H * 2);
            uint8_t* dst = is_x ? (a8q + (size_t)u * VDIT_CROSS_H) : (a8c + (size_t)(u - VDIT_CROSS_M) * VDIT_CROSS_H);
            float* sad = is_x ? sa0q : sa0c;
            const int r = is_x ? u : (u - VDIT_CROSS_M);
            const uint32_t* __restrict__ xu = reinterpret_cast<const uint32_t*>(src);
            constexpr int NH = VDIT_CROSS_H / 2;

            float mean_r = 0.f, rstd_r = 1.f;
            if (is_x) {
                // norm3 = affine LayerNorm: fp32 statistics + fp32 weight/bias, then a single cast back to bf16
                if (wid == 0) {
                    float ss = 0.f, qq = 0.f;
                    #pragma unroll 4
                    for (int i = lane; i < NH; i += 32) {
                        const uint32_t wd = xu[i];
                        const float v0 = bf16f((unsigned short)(wd & 0xffffu));
                        const float v1 = bf16f((unsigned short)(wd >> 16));
                        ss += v0; ss += v1;
                        qq += v0 * v0; qq += v1 * v1;
                    }
                    #pragma unroll
                    for (int off = 16; off > 0; off >>= 1) {
                        ss += __shfl_down_sync(0xffffffffu, ss, off);
                        qq += __shfl_down_sync(0xffffffffu, qq, off);
                    }
                    if (lane == 0) {
                        const float mean = ss / (float)VDIT_CROSS_H;
                        const float var = fmaxf(qq / (float)VDIT_CROSS_H - mean * mean, 0.f);
                        sc[0] = mean;
                        sc[1] = rsqrtf(var + norm_eps);
                    }
                }
                __syncthreads();
                mean_r = sc[0];
                rstd_r = sc[1];
            }

            float mxl = 0.f;
            #pragma unroll 4
            for (int i = tid; i < NH; i += VDIT_CROSS_NT) {
                const uint32_t wd = xu[i];
                float v0 = bf16f((unsigned short)(wd & 0xffffu));
                float v1 = bf16f((unsigned short)(wd >> 16));
                if (is_x) {
                    const int c = 2 * i;
                    const float w0 = bf16f(__bfloat16_as_ushort(
                        reinterpret_cast<const __nv_bfloat16*>(w3)[c]));
                    const float w1 = bf16f(__bfloat16_as_ushort(
                        reinterpret_cast<const __nv_bfloat16*>(w3)[c + 1]));
                    const float g0 = bf16f(__bfloat16_as_ushort(
                        reinterpret_cast<const __nv_bfloat16*>(b3)[c]));
                    const float g1 = bf16f(__bfloat16_as_ushort(
                        reinterpret_cast<const __nv_bfloat16*>(b3)[c + 1]));
                    v0 = fwam_bf16r((v0 - mean_r) * rstd_r * w0 + g0);
                    v1 = fwam_bf16r((v1 - mean_r) * rstd_r * w1 + g1);
                }
                mxl = fmaxf(mxl, fmaxf(fabsf(v0), fabsf(v1)));
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                mxl = fmaxf(mxl, __shfl_xor_sync(0xffffffffu, mxl, off));
            if (lane == 0) sc[4 + wid] = mxl;
            __syncthreads();
            if (wid == 0 && lane == 0) {
                float m = 0.f;
                #pragma unroll
                for (int i = 0; i < VDIT_CROSS_NWARP; ++i) m = fmaxf(m, sc[4 + i]);
                const float sa_v = fmaxf(m / 448.0f, 1e-12f);
                sad[r] = sa_v;
                sc[2] = sa_v;
            }
            __syncthreads();
            const float sa_r = sc[2];

            #pragma unroll 4
            for (int i = tid; i < NH; i += VDIT_CROSS_NT) {
                const uint32_t wd = xu[i];
                float v0 = bf16f((unsigned short)(wd & 0xffffu));
                float v1 = bf16f((unsigned short)(wd >> 16));
                if (is_x) {
                    const int c = 2 * i;
                    const float w0 = bf16f(__bfloat16_as_ushort(
                        reinterpret_cast<const __nv_bfloat16*>(w3)[c]));
                    const float w1 = bf16f(__bfloat16_as_ushort(
                        reinterpret_cast<const __nv_bfloat16*>(w3)[c + 1]));
                    const float g0 = bf16f(__bfloat16_as_ushort(
                        reinterpret_cast<const __nv_bfloat16*>(b3)[c]));
                    const float g1 = bf16f(__bfloat16_as_ushort(
                        reinterpret_cast<const __nv_bfloat16*>(b3)[c + 1]));
                    v0 = fwam_bf16r((v0 - mean_r) * rstd_r * w0 + g0);
                    v1 = fwam_bf16r((v1 - mean_r) * rstd_r * w1 + g1);
                }
                const unsigned short pair = (unsigned short)fp8_rne(v0 / sa_r) |
                                            ((unsigned short)fp8_rne(v1 / sa_r) << 8);
                *reinterpret_cast<unsigned short*>(dst + 2 * i) = pair;
            }
            __syncthreads();     // sc is reused by the next iteration
        }
    }
    VDIT_CROSS_MARK(1);
    VDIT_CROSS_GRID_SYNC();
    VDIT_CROSS_MARK(2);

    // ================= C_PHASE_2: cross-q GEMM → cross-kv GEMM (back to back, no sync in between) =================
    // The two GEMMs are independent and each grows PERSISTently from blockIdx.x. q: M=120,N=3072 → 96 tiles;
    // kv: M=129,N=6144 → 3×96 = 288 tiles. CTAs that go idle when the first step ends are filled in by the second step.

    if (VDIT_CROSS_PHASE(2) && (stop_phase == 0 || stop_phase > 2)) {
        fwam_fp8_gemm_body<FWAM_VDIT_CROSS_C_PHASE_2_2G_TILES>(
            /*A=*/a8q, /*W=*/wq, /*sa=*/sa0q, /*sw=*/swq,
            /*bias=*/reinterpret_cast<const __nv_bfloat16*>(bq),
            /*Cout=*/reinterpret_cast<__nv_bfloat16*>(qp),
            /*partial=*/nullptr, /*counters=*/nullptr,
            /*M=*/VDIT_CROSS_M, /*N=*/VDIT_CROSS_H3, /*K=*/VDIT_CROSS_H, /*Kseg=*/VDIT_CROSS_H,
            /*wp=*/11, /*ap=*/0, /*col0=*/0, /*ffn_ex=*/nullptr);
        // ⚠️ NOTE: there must be a block barrier between the two body calls. The body's PERSIST tile loop only calls
        // `__syncthreads()` when `_t != T0_` (its purpose being "the previous tile's epilogue has finished reading the smem staging area"),
        // whereas **each call's own first tile is exactly T0_, which sets no barrier**. So while call #1's epilogue
        // is still reading the staging area (= the head of smem_raw, overlapping the A operand ring [0, 24576)), the threads that
        // have already finished call #1 enter call #2's prefetch prologue and issue cp.async into that same smem → overwriting the
        // staged accumulator values with operands. Measured at ~1e-4~1e-5 per call, one contiguous 32B block in `qp` is computed wrongly (racecheck
        // reports exactly read vdit_gemm_core.cu:1070 vs write gemm_common.h:76).
        __syncthreads();
        fwam_fp8_gemm_body<FWAM_VDIT_CROSS_C_PHASE_2_2G_TILES>(
            /*A=*/a8c, /*W=*/wkv, /*sa=*/sa0c, /*sw=*/swkv,
            /*bias=*/reinterpret_cast<const __nv_bfloat16*>(bkv),
            /*Cout=*/reinterpret_cast<__nv_bfloat16*>(kvp),
            /*partial=*/nullptr, /*counters=*/nullptr,
            /*M=*/VDIT_CROSS_C, /*N=*/VDIT_CROSS_NKV, /*K=*/VDIT_CROSS_H, /*Kseg=*/VDIT_CROSS_H,
            /*wp=*/11, /*ap=*/0, /*col0=*/0, /*ffn_ex=*/nullptr);
    }
    VDIT_CROSS_MARK(3);
    VDIT_CROSS_GRID_SYNC();
    VDIT_CROSS_MARK(4);

    // ================= C_PHASE_3: qk-norm (no RoPE) + split out k/v (249 rows grid-stride) =================
    // q rows: full-row RMS of qp[r] + wnq → qc[r]
    // kv rows: full-row RMS + wnk over the **first 3072 columns** of kvp[r] (k) → kc[r]; the last 3072 columns (v) as-is → vc[r]
    //   (k/v are packed 6144 wide in kvp; splitting them into two contiguous 3072-wide buffers gives the flash a uniform row stride)
    if (VDIT_CROSS_PHASE(3) && (stop_phase == 0 || stop_phase > 3)) {
        for (int u = b; u < VDIT_CROSS_ROWS; u += gridDim.x) {
            const bool is_q = (u < VDIT_CROSS_M);
            const int r = is_q ? u : (u - VDIT_CROSS_M);
            const __nv_bfloat16* src = is_q
                ? reinterpret_cast<const __nv_bfloat16*>(qp + (size_t)r * VDIT_CROSS_H3 * 2)
                : reinterpret_cast<const __nv_bfloat16*>(kvp + (size_t)r * VDIT_CROSS_NKV * 2);
            const float rr = vdit_cross_row_rms(src, sc, tid, lane, wid, norm_eps);
            const __nv_bfloat16* wn = is_q
                ? reinterpret_cast<const __nv_bfloat16*>(wnq)
                : reinterpret_cast<const __nv_bfloat16*>(wnk);
            __nv_bfloat16* dst = is_q
                ? reinterpret_cast<__nv_bfloat16*>(qc + (size_t)r * VDIT_CROSS_H3 * 2)
                : reinterpret_cast<__nv_bfloat16*>(kc + (size_t)r * VDIT_CROSS_H3 * 2);
            constexpr int NPAIR = (VDIT_CROSS_H3 / 2) / VDIT_CROSS_NT;    // 6
            #pragma unroll
            for (int i = 0; i < NPAIR; ++i) {
                const int m = tid + VDIT_CROSS_NT * i;
                const int c = 2 * m;
                const float a0 = bf16f(__bfloat16_as_ushort(src[c]));
                const float a1 = bf16f(__bfloat16_as_ushort(src[c + 1]));
                const float na = fwam_bf16r(fwam_bf16r(a0 * rr) *
                                            bf16f(__bfloat16_as_ushort(wn[c])));
                const float nb = fwam_bf16r(fwam_bf16r(a1 * rr) *
                                            bf16f(__bfloat16_as_ushort(wn[c + 1])));
                dst[c] = __float2bfloat16_rn(na);
                dst[c + 1] = __float2bfloat16_rn(nb);
            }
            if (!is_q) {     // v copied as-is (the second half of kvp)
                __nv_bfloat16* vo = reinterpret_cast<__nv_bfloat16*>(vc + (size_t)r * VDIT_CROSS_H3 * 2);
                const __nv_bfloat16* vs = src + VDIT_CROSS_H3;
                #pragma unroll
                for (int i = 0; i < VDIT_CROSS_H3 / 2 / VDIT_CROSS_NT; ++i)
                    reinterpret_cast<uint32_t*>(vo)[tid + VDIT_CROSS_NT * i] =
                        reinterpret_cast<const uint32_t*>(vs)[tid + VDIT_CROSS_NT * i];
            }
            __syncthreads();
        }
    }
    VDIT_CROSS_MARK(5);
    VDIT_CROSS_GRID_SYNC();
    VDIT_CROSS_MARK(6);

    // ================= C_PHASE_4: bf16 flash (q 120 × kv 129) =================
    if (b < VDIT_CROSS_FGRID && VDIT_CROSS_PHASE(4) && (stop_phase == 0 || stop_phase > 4)) {
        const int h = b >> 2;
        const int qt = b & 3;
        const int w = tid >> 5;
        const int lane2 = tid & 31;

        __nv_bfloat16* fqs = reinterpret_cast<__nv_bfloat16*>(vg_smem);
        __nv_bfloat16* fkt = fqs + VDIT_CROSS_QT * VDIT_CROSS_BP;
        __nv_bfloat16* fvt = fkt + VDIT_CROSS_NSTAGE * VDIT_CROSS_TILE * VDIT_CROSS_BP;
        float* fsS = reinterpret_cast<float*>(fvt + VDIT_CROSS_NSTAGE * VDIT_CROSS_TILE * VDIT_CROSS_BP);
        constexpr int VDIT_CROSS_SS = 2 * 2 * 16 * 4;

        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int chunk = tid + VDIT_CROSS_NT * i;
            const int rr = chunk >> 4;
            const int j = chunk & 15;
            const int grow = qt * VDIT_CROSS_QT + rr;
            uint4 val = make_uint4(0u, 0u, 0u, 0u);
            if (grow < VDIT_CROSS_M) {
                const uint8_t* src = qc + ((size_t)grow * VDIT_CROSS_H3 + h * VDIT_CROSS_D) * 2 + j * 16;
                val = *reinterpret_cast<const uint4*>(src);
            }
            *reinterpret_cast<uint4*>(fqs + rr * VDIT_CROSS_BP + j * 8) = val;
        }

        const int mh = w >> 2;
        const int dq = w & 3;
        const int gid = lane2 >> 2;
        const int tig = lane2 & 3;

        float acc[4][4];
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt)
            #pragma unroll
            for (int i = 0; i < 4; ++i) acc[nt][i] = 0.f;
        float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;

        auto flash_tile = [&](const __nv_bfloat16* ktile, const __nv_bfloat16* vtile,
                              int nrows) {
            const __nv_bfloat16* qb = fqs + 16 * mh * VDIT_CROSS_BP;
            float s[2][4];
            #pragma unroll
            for (int n8 = 0; n8 < 2; ++n8)
                #pragma unroll
                for (int i = 0; i < 4; ++i) s[n8][i] = 0.f;
            const int q_row = lane2 & 15, q_c16 = lane2 >> 4;
            if (dq < 2) {
                const int n8q = dq;
                const int k2_row = lane2 & 7, k2_col = (lane2 >> 3) & 1;
                const __nv_bfloat16* kpb = ktile + (n8q * 8 + k2_row) * VDIT_CROSS_BP + 8 * k2_col;
                float c0v[4] = {0.f, 0.f, 0.f, 0.f}, c1v[4] = {0.f, 0.f, 0.f, 0.f};
                #pragma unroll
                for (int kc = 0; kc < 8; ++kc) {
                    unsigned a0, a1, a2, a3, b0, b1;
                    ldm_x4(qb + q_row * VDIT_CROSS_BP + 16 * kc + 8 * q_c16, a0, a1, a2, a3);
                    ldm_x2(kpb + 16 * kc, b0, b1);
                    if (kc & 1) mma_f32(c1v, a0, a1, a2, a3, b0, b1);
                    else        mma_f32(c0v, a0, a1, a2, a3, b0, b1);
                }
                c0v[0] += c1v[0]; c0v[1] += c1v[1]; c0v[2] += c1v[2]; c0v[3] += c1v[3];
                float* sp0 = &fsS[((mh * 2 + n8q) * 16 + gid) * 4 + tig];
                float* sp1 = &fsS[VDIT_CROSS_SS + ((mh * 2 + n8q) * 16 + gid) * 4 + tig];
                sp0[0] = c0v[0]; sp1[0] = c0v[1];
                sp0[8 * 4] = c0v[2]; sp1[8 * 4] = c0v[3];
            }
            __syncthreads();
            #pragma unroll
            for (int n8 = 0; n8 < 2; ++n8) {
                const int rb = ((mh * 2 + n8) * 16 + gid) * 4 + tig;
                s[n8][0] = fsS[rb];              s[n8][1] = fsS[VDIT_CROSS_SS + rb];
                s[n8][2] = fsS[rb + 8 * 4];      s[n8][3] = fsS[VDIT_CROSS_SS + rb + 8 * 4];
            }
            #pragma unroll
            for (int n8 = 0; n8 < 2; ++n8)
                #pragma unroll
                for (int i = 0; i < 4; ++i) s[n8][i] *= 0.08838834764831845f;
            // ⚠️ NOTE: the kv pad rows (129 = 8×16 + 1 → the tail tile has 15 padded rows) must be set to -inf,
            // otherwise they enter the softmax denominator with an exp(0-m) weight —— a **silent error**.
            #pragma unroll
            for (int n8 = 0; n8 < 2; ++n8) {
                const int cb = 8 * n8 + 2 * tig;
                if (cb >= nrows) { s[n8][0] = -INFINITY; s[n8][2] = -INFINITY; }
                if (cb + 1 >= nrows) { s[n8][1] = -INFINITY; s[n8][3] = -INFINITY; }
            }
            float mm0 = fmaxf(fmaxf(s[0][0], s[0][1]), fmaxf(s[1][0], s[1][1]));
            float mm1 = fmaxf(fmaxf(s[0][2], s[0][3]), fmaxf(s[1][2], s[1][3]));
            mm0 = fmaxf(mm0, __shfl_xor_sync(0xffffffffu, mm0, 1));
            mm0 = fmaxf(mm0, __shfl_xor_sync(0xffffffffu, mm0, 2));
            mm1 = fmaxf(mm1, __shfl_xor_sync(0xffffffffu, mm1, 1));
            mm1 = fmaxf(mm1, __shfl_xor_sync(0xffffffffu, mm1, 2));
            const float mn0 = fmaxf(m0, mm0), mn1 = fmaxf(m1, mm1);
            const float al0 = __expf(m0 - mn0), al1 = __expf(m1 - mn1);
            float e[2][4];
            #pragma unroll
            for (int n8 = 0; n8 < 2; ++n8) {
                e[n8][0] = __expf(s[n8][0] - mn0); e[n8][1] = __expf(s[n8][1] - mn0);
                e[n8][2] = __expf(s[n8][2] - mn1); e[n8][3] = __expf(s[n8][3] - mn1);
            }
            float ls0 = e[0][0] + e[0][1] + e[1][0] + e[1][1];
            float ls1 = e[0][2] + e[0][3] + e[1][2] + e[1][3];
            ls0 += __shfl_xor_sync(0xffffffffu, ls0, 1);
            ls0 += __shfl_xor_sync(0xffffffffu, ls0, 2);
            ls1 += __shfl_xor_sync(0xffffffffu, ls1, 1);
            ls1 += __shfl_xor_sync(0xffffffffu, ls1, 2);
            l0 = l0 * al0 + ls0;
            l1 = l1 * al1 + ls1;
            m0 = mn0; m1 = mn1;
            #pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                acc[nt][0] *= al0; acc[nt][1] *= al0;
                acc[nt][2] *= al1; acc[nt][3] *= al1;
            }
            unsigned pa[4];
            pa[0] = (unsigned)bf16_bits(e[0][0]) | ((unsigned)bf16_bits(e[0][1]) << 16);
            pa[1] = (unsigned)bf16_bits(e[0][2]) | ((unsigned)bf16_bits(e[0][3]) << 16);
            pa[2] = (unsigned)bf16_bits(e[1][0]) | ((unsigned)bf16_bits(e[1][1]) << 16);
            pa[3] = (unsigned)bf16_bits(e[1][2]) | ((unsigned)bf16_bits(e[1][3]) << 16);
            unsigned bf[4][2];
            {
                const int v_row = lane2 & 15, v_c16 = lane2 >> 4;
                const __nv_bfloat16* vbp = vtile + v_row * VDIT_CROSS_BP + dq * 32 + 8 * v_c16;
                unsigned t0, t1, t2, t3;
                ldm_x4_t(vbp, t0, t1, t2, t3);
                bf[0][0] = t0; bf[0][1] = t1; bf[1][0] = t2; bf[1][1] = t3;
                ldm_x4_t(vbp + 16, t0, t1, t2, t3);
                bf[2][0] = t0; bf[2][1] = t1; bf[3][0] = t2; bf[3][1] = t3;
            }
            #pragma unroll
            for (int nt = 0; nt < 4; ++nt)
                mma_f32(acc[nt], pa[0], pa[1], pa[2], pa[3], bf[nt][0], bf[nt][1]);
        };

        const int ntiles = (VDIT_CROSS_C + VDIT_CROSS_TILE - 1) / VDIT_CROSS_TILE;    // 9 (129 = 8×16 + 1)
        const uint32_t* kvu = reinterpret_cast<const uint32_t*>(kc);
        const uint32_t* vvu = reinterpret_cast<const uint32_t*>(vc);
        auto issue_tile = [&](int stage, int tb) {
            const int cc = tid >> 4;
            const int j = tid & 15;
            const int g = tb * VDIT_CROSS_TILE + cc;
            __nv_bfloat16* kdst = &fkt[stage * VDIT_CROSS_TILE * VDIT_CROSS_BP + cc * VDIT_CROSS_BP + j * 8];
            __nv_bfloat16* vdst = &fvt[stage * VDIT_CROSS_TILE * VDIT_CROSS_BP + cc * VDIT_CROSS_BP + j * 8];
            if (g < VDIT_CROSS_C) {
                const uint32_t* gk = kvu + (size_t)g * (VDIT_CROSS_H3 / 2) + h * (VDIT_CROSS_D / 2) + j * 4;
                unsigned dk = (unsigned)__cvta_generic_to_shared(kdst);
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dk), "l"(gk));
                const uint32_t* gv = vvu + (size_t)g * (VDIT_CROSS_H3 / 2) + h * (VDIT_CROSS_D / 2) + j * 4;
                unsigned dv = (unsigned)__cvta_generic_to_shared(vdst);
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dv), "l"(gv));
            } else {
                unsigned long long* kz = reinterpret_cast<unsigned long long*>(kdst);
                unsigned long long* vz = reinterpret_cast<unsigned long long*>(vdst);
                kz[0] = 0ull; kz[1] = 0ull;
                vz[0] = 0ull; vz[1] = 0ull;
            }
            asm volatile("cp.async.commit_group;\n" ::);
        };
        __syncthreads();
        issue_tile(0, 0);
        if (ntiles == 1) asm volatile("cp.async.commit_group;\n" ::);
        else if (ntiles > 1) issue_tile(1, 1);
        for (int tb = 0; tb < ntiles; ++tb) {
            const int stage = tb & 1;
            if (tb + 1 == ntiles) asm volatile("cp.async.wait_group 0;\n" ::);
            else                  asm volatile("cp.async.wait_group 1;\n" ::);
            __syncthreads();
            flash_tile(fkt + stage * VDIT_CROSS_TILE * VDIT_CROSS_BP, fvt + stage * VDIT_CROSS_TILE * VDIT_CROSS_BP,
                       min(VDIT_CROSS_TILE, VDIT_CROSS_C - tb * VDIT_CROSS_TILE));
            __syncthreads();
            if (tb + 2 < ntiles) issue_tile(stage, tb + 2);
            else                 asm volatile("cp.async.commit_group;\n" ::);
        }

        const float inv0 = 1.0f / l0, inv1 = 1.0f / l1;
        float* ou = reinterpret_cast<float*>(attn);
        const int orow = qt * VDIT_CROSS_QT + 16 * mh + gid;
        const int colbase = h * VDIT_CROSS_D + dq * 32;
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const int cp = colbase + 8 * nt + 2 * tig;
            if (orow < VDIT_CROSS_M) {
                ou[(size_t)orow * VDIT_CROSS_H3 + cp] = acc[nt][0] * inv0;
                ou[(size_t)orow * VDIT_CROSS_H3 + cp + 1] = acc[nt][1] * inv0;
            }
            if (orow + 8 < VDIT_CROSS_M) {
                ou[(size_t)(orow + 8) * VDIT_CROSS_H3 + cp] = acc[nt][2] * inv1;
                ou[(size_t)(orow + 8) * VDIT_CROSS_H3 + cp + 1] = acc[nt][3] * inv1;
            }
        }
    }
    VDIT_CROSS_MARK(7);
    VDIT_CROSS_GRID_SYNC();

    // ================= C_PHASE_5: attn row quantization (b<120) =================
    if (b < VDIT_CROSS_M && VDIT_CROSS_PHASE(5) && (stop_phase == 0 || stop_phase > 5)) {
        const float* __restrict__ ar = reinterpret_cast<const float*>(attn) +
                                       (size_t)b * VDIT_CROSS_H3;
        constexpr int NV = VDIT_CROSS_H3 / VDIT_CROSS_NT;      // 12
        float v[NV];
        float mx = 0.f;
        #pragma unroll
        for (int i = 0; i < NV; ++i) {
            v[i] = ar[tid + VDIT_CROSS_NT * i];
            mx = fmaxf(mx, fabsf(v[i]));
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
        if (lane == 0) sc[wid] = mx;
        __syncthreads();
        if (tid == 0) {
            float m = 0.f;
            #pragma unroll
            for (int i = 0; i < VDIT_CROSS_NWARP; ++i) m = fmaxf(m, sc[i]);
            sc[VDIT_CROSS_NWARP] = fmaxf(m / 448.0f, 1e-12f);
        }
        __syncthreads();
        const float sa = sc[VDIT_CROSS_NWARP];
        sa0o[b] = sa;
        uint8_t* __restrict__ ao = a8o + (size_t)b * VDIT_CROSS_H3;
        #pragma unroll
        for (int i = 0; i < NV; ++i) ao[tid + VDIT_CROSS_NT * i] = fp8_rne(v[i] / sa);
    }
    VDIT_CROSS_MARK(8);
    VDIT_CROSS_GRID_SYNC();

    // ================= C_PHASE_6: cross-o GEMM + ungated residual =================
    if (VDIT_CROSS_PHASE(6) && (stop_phase == 0 || stop_phase > 6)) {
        // Ungated residual out = x + z: reuse the RESID path, passing a single ones as the gate with gate_srow=0 (broadcast)
        fwam_gemm_extra ex{x_in, ones, VDIT_CROSS_H, 0};
        fwam_fp8_gemm_body<FWAM_VDIT_CROSS_C_PHASE_6_TILES>(
            /*A=*/a8o, /*W=*/wo, /*sa=*/sa0o, /*sw=*/swo,
            /*bias=*/reinterpret_cast<const __nv_bfloat16*>(bo),
            /*Cout=*/reinterpret_cast<__nv_bfloat16*>(out),
            /*partial=*/nullptr, /*counters=*/nullptr,
            /*M=*/VDIT_CROSS_M, /*N=*/VDIT_CROSS_H, /*K=*/VDIT_CROSS_H3, /*Kseg=*/VDIT_CROSS_H3,
            /*wp=*/11, /*ap=*/0, /*col0=*/0, /*ffn_ex=*/&ex);
    }
    VDIT_CROSS_MARK(9);
    (void)dbg;
}

// ---- The grid comes from the device capacity (occupancy × SM count), no longer hard-coded to 132 ----
// Structural lower bound = VDIT_CROSS_M (C_PHASE_1's per-row reduction and C_PHASE_5's `if (b < VDIT_CROSS_M)`; grid<120 drops rows).
static int g_vg_grid = -1;
static int vg_grid() {
    return cached_coop_grid(g_vg_grid, (const void*)vdit_attn_cross_kernel, VDIT_CROSS_NT,
                            VDIT_CROSS_SMEM, VDIT_CROSS_M, "vdit_attn_cross");
}
extern "C" int vdit_attn_cross_grid() { return vg_grid(); }
// The actual grid of the non-cooperative split-phase form (lower bound = VDIT_CROSS_M; a plain launch may take multiple waves, so small GPUs can run it too)
static int vg_split_grid();   // defined further down in this file (one each for the cooperative/split grid)
extern "C" int vdit_attn_cross_split_grid() { return vg_split_grid(); }

// The non-cooperative fallback's grid (no co-residency required): lower bound = VDIT_CROSS_M (C_PHASE_5 `b<VDIT_CROSS_M`; C_PHASE_1/C_PHASE_3 are grid-stride and any grid covers them).
static int g_vg_split_grid = -1;
static int vg_split_grid() {
    return cached_split_grid(g_vg_split_grid, (const void*)vdit_attn_cross_kernel, VDIT_CROSS_NT,
                             VDIT_CROSS_SMEM, VDIT_CROSS_M, "vdit_attn_cross");
}

// ------------------------------------------------------------------ //

// ---- Geometry self-report (shape constants + GEMM template parameters; the template parameters use the same macro set as the instantiation) ----
extern "C" const char* vdit_attn_cross_geom() {
    static const std::string s = fwam_geom_join({
        fwam_shapes({{"M", VDIT_CROSS_M}, {"C", VDIT_CROSS_C}, {"H", VDIT_CROSS_H}, {"NKV", VDIT_CROSS_NKV}, {"NT", VDIT_CROSS_NT}, {"SMEM", VDIT_CROSS_SMEM}, {"GRID", VDIT_CROSS_GRID}}),
        fwam_tiles("C_PHASE_2_2G", FWAM_TILES(FWAM_VDIT_CROSS_C_PHASE_2_2G_TILES)),
        fwam_tiles("C_PHASE_6", FWAM_TILES(FWAM_VDIT_CROSS_C_PHASE_6_TILES)),
    });
    return s.c_str();
}

extern "C" void vdit_attn_cross_cuda(
    const void* x_in, const void* context, const void* w3, const void* b3,
    const void* wq, const float* swq, const void* bq,
    const void* wkv, const float* swkv, const void* bkv,
    const void* wnq, const void* wnk,
    const void* wo, const float* swo, const void* bo, const void* ones,
    void* out, void* qp, void* kvp, void* qc, void* kc, void* vc, void* attn,
    void* a8q, float* sa0q, void* a8c, float* sa0c, void* a8o, float* sa0o,
    float norm_eps, int stop_phase, int only_phase, float* dbg,
    unsigned long long* phase_ts, cudaStream_t stream)
{
    dim3 block(VDIT_CROSS_NT);
    void* args[] = {
        (void*)&x_in, (void*)&context, (void*)&w3, (void*)&b3,
        (void*)&wq, (void*)&swq, (void*)&bq,
        (void*)&wkv, (void*)&swkv, (void*)&bkv,
        (void*)&wnq, (void*)&wnk,
        (void*)&wo, (void*)&swo, (void*)&bo, (void*)&ones,
        (void*)&out, (void*)&qp, (void*)&kvp, (void*)&qc, (void*)&kc,
        (void*)&vc, (void*)&attn,
        (void*)&a8q, (void*)&sa0q, (void*)&a8c, (void*)&sa0c, (void*)&a8o, (void*)&sa0o,
        (void*)&norm_eps, (void*)&stop_phase, (void*)&only_phase, (void*)&dbg, (void*)&phase_ts,
    };
    cudaError_t e;
    dim3 grid;                       // declared outside: the error path needs to report the grid
    if (only_phase < 0) {
        grid = dim3(vg_grid());
        e = cudaLaunchCooperativeKernel((void*)vdit_attn_cross_kernel, grid,
                                        block, args, VDIT_CROSS_SMEM, stream);
    } else {
        grid = dim3(vg_split_grid());
        e = cudaLaunchKernel((void*)vdit_attn_cross_kernel, grid, block, args,
                             VDIT_CROSS_SMEM, stream);
    }
    if (e != cudaSuccess) {
        // Throw instead of exit(1): exit would kill the worker process outright and the caller would get no handleable error.
        coop_fail("vdit_attn_cross",
                  "launch failed (only_phase=" + std::to_string(only_phase)
                  + ", grid=" + std::to_string((int)grid.x) + "): "
                  + cudaGetErrorString(e));
    }
}
