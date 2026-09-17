// G: one cooperative kernel for the whole action cross-attention path
// (norm3+q' GEMM → RMS+masked flash → o GEMM + ungated residual).
//
// Phases (grid = 48 CTA × 256 threads, all co-resident):
//   C_PHASE_1 [b<32] norm3 (affine LN) + row quantization → a8q/sa0q
//   C_PHASE_2 [b<48] q' fp8 GEMM → qp[32, 3072]
//   C_PHASE_3 [b<24] q full-row RMS shard scan → rstd_scratch
//   C_PHASE_4 [b<24] qs scaling + masked bf16 flash → attn
//   C_PHASE_5 [b<32] attn row quantization → a8o/sa0o
//   C_PHASE_6 [b<32] o fp8 GEMM + ungated residual → out (b≥32 idles until the end)
//
// Phases are separated by grid.sync. Numerics: fp8 single RNE / bf16 RN, bit-exact with the split
// form.
// smem: a large union buffer (gemm1/gemm2/flash reuse it mutually exclusively, ~41.5KB max) + a
// small static area; dynamic smem = an L-byte mask (static+dynamic ≤ 48KB → L ≤ ~6K keys).

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cstdint>
#include <cstdio>
#include <cassert>
#include "coop_grid.h"
#include "geom_report.h"            // geometry self-report (instantiation and self-report share one macro set)
#include "dit_common.h"                      // adit device-side helpers (pulls in common.h)

namespace cg = cooperative_groups;

constexpr int ADIT_CROSS_M = 32;       // token rows
constexpr int ADIT_CROSS_H = 1024;     // hidden (the K of the q' GEMM / the norm3 width)
constexpr int ADIT_CROSS_NQ = 3072;    // q' width (= flash full width / the K of the o GEMM)
constexpr int ADIT_CROSS_NO = 1024;    // o output width
constexpr int ADIT_CROSS_NTHREADS = 256;
constexpr int ADIT_CROSS_NWARPS = ADIT_CROSS_NTHREADS / 32;
constexpr float ADIT_CROSS_EPS = 1e-6f;

// ---- C_PHASE_2 (q' GEMM) geometry: BN=64/BK=128/STAGES=3 -> grid=48 ----
constexpr int ADIT_CROSS_BN_A = 64;
constexpr int ADIT_CROSS_BK_A = 128;
constexpr int ADIT_CROSS_STAGES = 3;
constexpr int ADIT_CROSS_ARS_A = ADIT_CROSS_BK_A + 16;                 // 144
constexpr int ADIT_CROSS_BRS_A = ADIT_CROSS_BK_A + 16;
constexpr int ADIT_CROSS_U_A2 = ADIT_CROSS_M * ADIT_CROSS_ARS_A + ADIT_CROSS_BN_A * ADIT_CROSS_BRS_A;   // 13824
constexpr int ADIT_CROSS_GRID = ADIT_CROSS_NQ / ADIT_CROSS_BN_A;                  // 48
constexpr int ADIT_CROSS_KtA = ADIT_CROSS_H / ADIT_CROSS_BK_A;                    // 8

// ---- C_PHASE_6 (o GEMM) geometry: BN=32/BK=192/STAGES=3 -> grid=32 ----
constexpr int ADIT_CROSS_BN_C = 32;
constexpr int ADIT_CROSS_BK_C = 192;
constexpr int ADIT_CROSS_ARS_C = ADIT_CROSS_BK_C + 16;                 // 208
constexpr int ADIT_CROSS_BRS_C = ADIT_CROSS_BK_C + 16;
constexpr int ADIT_CROSS_U_C2 = ADIT_CROSS_M * ADIT_CROSS_ARS_C + ADIT_CROSS_BN_C * ADIT_CROSS_BRS_C;   // 13312
constexpr int ADIT_CROSS_KtC = ADIT_CROSS_NQ / ADIT_CROSS_BK_C;                   // 16

// ---- C_PHASE_4 (flash) geometry ----
constexpr int ADIT_CROSS_BD = 128;      // head_dim
constexpr int ADIT_CROSS_BP = 136;
constexpr int ADIT_CROSS_HEADS = ADIT_CROSS_NQ / ADIT_CROSS_BD;  // 24
constexpr int ADIT_CROSS_TILE = 16;
constexpr int ADIT_CROSS_NSTAGE = 2;
constexpr float ADIT_CROSS_SATT = 0.08838834764831845f;  // 1/sqrt(128): torch SDPA's default scale (the flash score must be multiplied)

static_assert(ADIT_CROSS_STAGES * ADIT_CROSS_U_A2 <= 48 * 1024, "q' GEMM smem over budget");
static_assert(ADIT_CROSS_STAGES * ADIT_CROSS_U_C2 <= 48 * 1024, "o GEMM smem over budget");

// ------------------------------------------------------------------ //
// Device-side helpers: bf16f in common.h; bf16_bits/fp8_rne/ldm_x2/x4/x4_t/mma_f32 in dit_common.h.
// ------------------------------------------------------------------ //

// Non-cooperative fallback (only_phase >= 0): each phase is launched separately and visibility
// between phases comes from stream ordering. The split path builds no grid barrier, so a plain
// `cudaLaunchKernel` can be used (no co-residency requirement).
#define ADIT_CROSS_PHASE(k)     (only_phase < 0 || only_phase == (k))
#define ADIT_CROSS_GRID_SYNC()  do { if (only_phase < 0) { __threadfence(); cg::this_grid().sync(); } } while (0)

// ------------------------------------------------------------------ //
__global__ void __launch_bounds__(ADIT_CROSS_NTHREADS)
adit_attn_cross_kernel(
    // C_PHASE_1/C_PHASE_2 (E1) inputs: norm3 + q' GEMM
    const uint8_t* __restrict__ x_in,    // bf16 [M, H]
    const uint8_t* __restrict__ w3,      // bf16 [H]
    const uint8_t* __restrict__ b3,      // bf16 [H]
    const uint8_t* __restrict__ wq,      // fp8 [NQ, H]
    const float* __restrict__ swq,       // fp32 [NQ]
    const uint8_t* __restrict__ bq,      // bf16 [NQ]
    uint8_t* __restrict__ qp,            // bf16 [M, NQ] (C_PHASE_2 output / B input)
    uint8_t* __restrict__ a8q,           // fp8 [M, H]
    float* __restrict__ sa0q,            // fp32 [M]
    // B (E2) inputs
    const uint8_t* __restrict__ wnq,     // bf16 [NQ]
    const uint8_t* __restrict__ k_cache, // bf16 [L, NQ]
    const uint8_t* __restrict__ v_cache, // bf16 [L, NQ]
    const unsigned char* __restrict__ mask,   // [L] bool
    uint8_t* __restrict__ attn,          // bf16 [M, NQ] (B output / C_PHASE_5 input)
    // C (E3) input/output
    const uint8_t* __restrict__ wo,      // fp8 [NO, NQ]
    const float* __restrict__ swo,       // fp32 [NO]
    const uint8_t* __restrict__ bo,      // bf16 [NO]
    uint8_t* __restrict__ out,           // bf16 [M, NO]
    uint8_t* __restrict__ a8o,           // fp8 [M, NQ]
    float* __restrict__ sa0o,            // fp32 [M]
    float* __restrict__ rstd_scratch,    // fp32 [HEADS*M]: per-head Σq² shard slots (no zeroing needed)
    int L, int only_phase)               // non-cooperative fallback: >=0 runs that phase only; <0 = the cooperative full pipeline
{
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;

    // ---------------- smem: large union buffer + small static area ----------------
    __shared__ __align__(16) union {
        uint8_t gemm1[ADIT_CROSS_STAGES][ADIT_CROSS_U_A2];              // C_PHASE_2 q' GEMM stages (41472B)
        uint8_t gemm2[ADIT_CROSS_STAGES][ADIT_CROSS_U_C2];              // C_PHASE_6 o GEMM stages (39936B)
        struct {
            __nv_bfloat16 kt[ADIT_CROSS_NSTAGE][ADIT_CROSS_TILE * ADIT_CROSS_BP];
            __nv_bfloat16 vt[ADIT_CROSS_NSTAGE][ADIT_CROSS_TILE * ADIT_CROSS_BP];
            __nv_bfloat16 qs[ADIT_CROSS_M * ADIT_CROSS_BP];
            uint32_t wns[ADIT_CROSS_BD / 2];
        } fl;                                     // B flash (~26.4KB)
    } s;
    __shared__ float q_shred[2 * ADIT_CROSS_NWARPS];          // C_PHASE_1 norm3 statistics
    // The S buffer shared by QK: split into two arrays by even/odd column so that each lane's
    // address = its lane number ⇒ zero bank conflicts (the original [row*18 + 2*tig] row stride
    // was a stable 2-way conflict with no way around it).
    // sS[e][ ((mh*2+n8)*16 + row)*4 + tig ], e=0 holds the 2t columns, e=1 the 2t+1 columns.
    __shared__ float sS[2][2 * 2 * 16 * 4];
    __shared__ float o_shmax[ADIT_CROSS_NWARPS];              // C_PHASE_5 amax
    __shared__ float sh_sa;                        // C_PHASE_1/C_PHASE_5 block-wide scale broadcast
    __shared__ float rq[ADIT_CROSS_M];                        // C_PHASE_4 RMS result
    extern __shared__ unsigned char msk[];         // dynamic: an L-byte mask

    const int lane_g = lane >> 2;
    const int lane_t = lane & 3;
    const int m_slice = wid / (ADIT_CROSS_NWARPS / 2);
    const int n_grp = wid % (ADIT_CROSS_NWARPS / 2);

    // ================= C_PHASE_1: norm3 (affine LayerNorm) + row quantization =================
    if (ADIT_CROSS_PHASE(1)) {
        const uint32_t* xu = reinterpret_cast<const uint32_t*>(x_in);
        if (b < ADIT_CROSS_M) {
            const int r = b;
            uint32_t xv[2];
            float ps = 0.f, pq = 0.f;
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                xv[i] = xu[(size_t)r * (ADIT_CROSS_H / 2) + tid + i * ADIT_CROSS_NTHREADS];
                float a = bf16f((unsigned short)(xv[i] & 0xffffu));
                float bb = bf16f((unsigned short)(xv[i] >> 16));
                ps += a + bb;
                pq += a * a + bb * bb;
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                ps += __shfl_down_sync(0xffffffffu, ps, off);
                pq += __shfl_down_sync(0xffffffffu, pq, off);
            }
            if (lane == 0) { q_shred[wid] = ps; q_shred[ADIT_CROSS_NWARPS + wid] = pq; }
            __syncthreads();
            float mean = 0.f, rstd = 0.f;
            if (wid == 0 && lane == 0) {
                float s2 = 0.f, q2 = 0.f;
                #pragma unroll
                for (int i = 0; i < ADIT_CROSS_NWARPS; ++i) { s2 += q_shred[i]; q2 += q_shred[ADIT_CROSS_NWARPS + i]; }
                mean = s2 / (float)ADIT_CROSS_H;
                float var = fmaxf(q2 / (float)ADIT_CROSS_H - mean * mean, 0.f);
                rstd = rsqrtf(var + ADIT_CROSS_EPS);
            }
            if (tid == 0) { q_shred[0] = mean; q_shred[1] = rstd; }
            __syncthreads();
            mean = q_shred[0]; rstd = q_shred[1];
            __syncthreads();

            const uint32_t* wu = reinterpret_cast<const uint32_t*>(w3);
            const uint32_t* bu = reinterpret_cast<const uint32_t*>(b3);
            uint32_t wv[2], bv[2];
            float mx = 0.f;
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int u = tid + i * ADIT_CROSS_NTHREADS;
                wv[i] = wu[u]; bv[i] = bu[u];
                float wl = bf16f((unsigned short)(wv[i] & 0xffffu)), wh = bf16f((unsigned short)(wv[i] >> 16));
                float bl = bf16f((unsigned short)(bv[i] & 0xffffu)), bh = bf16f((unsigned short)(bv[i] >> 16));
                float a = bf16f((unsigned short)(xv[i] & 0xffffu));
                float bb = bf16f((unsigned short)(xv[i] >> 16));
                float ya = bf16f(bf16_bits((a - mean) * rstd * wl + bl));
                float yb = bf16f(bf16_bits((bb - mean) * rstd * wh + bh));
                mx = fmaxf(mx, fmaxf(fabsf(ya), fabsf(yb)));
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                mx = fmaxf(mx, __shfl_down_sync(0xffffffffu, mx, off));
            if (lane == 0) q_shred[wid] = mx;
            __syncthreads();
            if (wid == 0 && lane == 0) {
                float m = 0.f;
                #pragma unroll
                for (int i = 0; i < ADIT_CROSS_NWARPS; ++i) m = fmaxf(m, q_shred[i]);
                const float sa_v = fmaxf(m / 448.0f, 1e-12f);
                sa0q[r] = sa_v;
                sh_sa = sa_v;
            }
            __syncthreads();
            const float sa_r = sh_sa;
            unsigned char* a8r = a8q + (size_t)r * ADIT_CROSS_H;
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                float wl = bf16f((unsigned short)(wv[i] & 0xffffu)), wh = bf16f((unsigned short)(wv[i] >> 16));
                float bl = bf16f((unsigned short)(bv[i] & 0xffffu)), bh = bf16f((unsigned short)(bv[i] >> 16));
                float a = bf16f((unsigned short)(xv[i] & 0xffffu));
                float bb = bf16f((unsigned short)(xv[i] >> 16));
                float ya = bf16f(bf16_bits((a - mean) * rstd * wl + bl));
                float yb = bf16f(bf16_bits((bb - mean) * rstd * wh + bh));
                unsigned short pair = (unsigned short)fp8_rne(ya / sa_r)
                                    | ((unsigned short)fp8_rne(yb / sa_r) << 8);
                *reinterpret_cast<unsigned short*>(a8r + 2 * (tid + i * ADIT_CROSS_NTHREADS)) = pair;
            }
        }
    }
    ADIT_CROSS_GRID_SYNC();    // #1: a8q / sa0q visible grid-wide

    // ================= C_PHASE_2: q' fp8 GEMM + epilogue (BN=64/BK=128) =================
    if (ADIT_CROSS_PHASE(2)) {
        constexpr int n8 = ADIT_CROSS_BN_A / (ADIT_CROSS_NWARPS / 2) / 8;   // 2
        constexpr int k_steps = ADIT_CROSS_BK_A / 32;            // 4
        constexpr int KQ = ADIT_CROSS_BK_A / 16;                 // 8
        constexpr int A_CH = ADIT_CROSS_M * ADIT_CROSS_BK_A / 16;           // 256
        constexpr int W_CH = ADIT_CROSS_BN_A * ADIT_CROSS_BK_A / 16;        // 512
        constexpr int TOT = A_CH + W_CH;              // 768
        uint8_t (&st)[ADIT_CROSS_STAGES][ADIT_CROSS_U_A2] = s.gemm1;

        auto issue = [&](int stage, int kt) {
            uint8_t* base = st[stage];
            #pragma unroll
            for (int c = tid; c < TOT; c += ADIT_CROSS_NTHREADS) {
                if (c < A_CH) {
                    int r = c / KQ;
                    int kq = (c % KQ) * 16;
                    const uint8_t* src = a8q + (size_t)r * ADIT_CROSS_H + kt * ADIT_CROSS_BK_A + kq;
                    unsigned sd = (unsigned)__cvta_generic_to_shared(base + r * ADIT_CROSS_ARS_A + kq);
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sd), "l"(src));
                } else {
                    int wc = c - A_CH;
                    int n = wc / KQ;
                    int kq = (wc % KQ) * 16;
                    const uint8_t* src = wq + ((size_t)b * ADIT_CROSS_BN_A + n) * ADIT_CROSS_H + kt * ADIT_CROSS_BK_A + kq;
                    unsigned sd = (unsigned)__cvta_generic_to_shared(base + ADIT_CROSS_M * ADIT_CROSS_ARS_A + n * ADIT_CROSS_BRS_A + kq);
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sd), "l"(src));
                }
            }
            asm volatile("cp.async.commit_group;\n" ::);
        };
        auto issue_empty = []() { asm volatile("cp.async.commit_group;\n" ::); };

        #pragma unroll
        for (int s2 = 0; s2 < ADIT_CROSS_STAGES - 1; ++s2)
            if (s2 < ADIT_CROSS_KtA) issue(s2, s2);

        float acc[n8][4];
        #pragma unroll
        for (int s2 = 0; s2 < n8; ++s2)
            #pragma unroll
            for (int i = 0; i < 4; ++i) acc[s2][i] = 0.f;

        for (int kt = 0; kt < ADIT_CROSS_KtA; ++kt) {
            const int stage = kt % ADIT_CROSS_STAGES;
            if (kt + 1 == ADIT_CROSS_KtA) asm volatile("cp.async.wait_group 0;\n" ::);
            else              asm volatile("cp.async.wait_group %0;\n" :: "n"(ADIT_CROSS_STAGES - 2));
            __syncthreads();

            const uint8_t* As_s = st[stage];
            const uint8_t* Ws_s = st[stage] + ADIT_CROSS_M * ADIT_CROSS_ARS_A;
            #pragma unroll
            for (int kk = 0; kk < k_steps; ++kk) {
                const int k_off = kk * 32;
                const int arow0 = m_slice * 16 + lane_g;
                const int arow1 = arow0 + 8;
                uint32_t a0 = *reinterpret_cast<const uint32_t*>(As_s + (size_t)arow0 * ADIT_CROSS_ARS_A + k_off + lane_t * 4);
                uint32_t a1 = *reinterpret_cast<const uint32_t*>(As_s + (size_t)arow1 * ADIT_CROSS_ARS_A + k_off + lane_t * 4);
                uint32_t a2 = *reinterpret_cast<const uint32_t*>(As_s + (size_t)arow0 * ADIT_CROSS_ARS_A + k_off + lane_t * 4 + 16);
                uint32_t a3 = *reinterpret_cast<const uint32_t*>(As_s + (size_t)arow1 * ADIT_CROSS_ARS_A + k_off + lane_t * 4 + 16);
                #pragma unroll
                for (int s2 = 0; s2 < n8; ++s2) {
                    const int bn = n_grp * (ADIT_CROSS_BN_A / (ADIT_CROSS_NWARPS / 2)) + s2 * 8 + lane_g;
                    const uint32_t* bp = reinterpret_cast<const uint32_t*>(Ws_s + (size_t)bn * ADIT_CROSS_BRS_A + k_off);
                    uint32_t b0w = bp[lane_t];
                    uint32_t b1w = bp[lane_t + 4];
                    asm volatile(
                        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                        : "+f"(acc[s2][0]), "+f"(acc[s2][1]), "+f"(acc[s2][2]), "+f"(acc[s2][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0w), "r"(b1w));
                }
            }
            const int nkt = kt + ADIT_CROSS_STAGES - 1;
            if (nkt < ADIT_CROSS_KtA) issue((kt - 1 + ADIT_CROSS_STAGES) % ADIT_CROSS_STAGES, nkt);
            else           issue_empty();
        }

        // epilogue: qp = bf16(acc*sa0q*swq + bq)
        const int r0 = m_slice * 16 + lane_g;
        const int r1 = r0 + 8;
        const float sa0_r0 = sa0q[r0], sa0_r1 = sa0q[r1];
        #pragma unroll
        for (int s2 = 0; s2 < n8; ++s2) {
            const int col0 = b * ADIT_CROSS_BN_A + n_grp * (ADIT_CROSS_BN_A / (ADIT_CROSS_NWARPS / 2)) + s2 * 8 + lane_t * 2;
            float swc0 = swq[col0], swc1 = swq[col0 + 1];
            uint32_t bc = *reinterpret_cast<const uint32_t*>(bq + (size_t)col0 * 2);
            float bc0 = bf16f((unsigned short)(bc & 0xffffu)), bc1 = bf16f((unsigned short)(bc >> 16));
            float u0 = bf16f(bf16_bits(acc[s2][0] * sa0_r0 * swc0 + bc0));
            float u1 = bf16f(bf16_bits(acc[s2][1] * sa0_r0 * swc1 + bc1));
            *reinterpret_cast<uint32_t*>(qp + ((size_t)r0 * ADIT_CROSS_NQ + col0) * 2) =
                (uint32_t)bf16_bits(u0) | ((uint32_t)bf16_bits(u1) << 16);
            float u2 = bf16f(bf16_bits(acc[s2][2] * sa0_r1 * swc0 + bc0));
            float u3 = bf16f(bf16_bits(acc[s2][3] * sa0_r1 * swc1 + bc1));
            *reinterpret_cast<uint32_t*>(qp + ((size_t)r1 * ADIT_CROSS_NQ + col0) * 2) =
                (uint32_t)bf16_bits(u2) | ((uint32_t)bf16_bits(u3) << 16);
        }
    }
    ADIT_CROSS_GRID_SYNC();    // #2: qp visible grid-wide

    // ================= C_PHASE_3: RMS scan + q' slice staging (head h=b<24) =================
    if (b < ADIT_CROSS_HEADS && ADIT_CROSS_PHASE(3)) {
        const int h = b;
        __nv_bfloat16 (&kt)[ADIT_CROSS_NSTAGE][ADIT_CROSS_TILE * ADIT_CROSS_BP] = s.fl.kt;
        __nv_bfloat16 (&vt)[ADIT_CROSS_NSTAGE][ADIT_CROSS_TILE * ADIT_CROSS_BP] = s.fl.vt;
        __nv_bfloat16 (&qs)[ADIT_CROSS_M * ADIT_CROSS_BP] = s.fl.qs;
        uint32_t (&wns)[ADIT_CROSS_BD / 2] = s.fl.wns;

        // 0) wnq slice + mask preload
        {
            const uint32_t* wnu = reinterpret_cast<const uint32_t*>(wnq);
            if (tid < ADIT_CROSS_BD / 2) wns[tid] = wnu[h * (ADIT_CROSS_BD / 2) + tid];
            for (int i = tid; i < L; i += ADIT_CROSS_NTHREADS) msk[i] = mask[i];
        }
        // kv tile double-buffer prefetch: issued before the RMS grid.sync (to cover the RMS global latency)
        const int ntiles = (L + ADIT_CROSS_TILE - 1) / ADIT_CROSS_TILE;
        const uint32_t* kvu = reinterpret_cast<const uint32_t*>(k_cache);
        const uint32_t* vvu = reinterpret_cast<const uint32_t*>(v_cache);
        auto issue_tile = [&](int stage, int tb) {
            const int cc = tid >> 4;
            const int j = tid & 15;
            const int g = tb * ADIT_CROSS_TILE + cc;
            __nv_bfloat16* kdst = &kt[stage][cc * ADIT_CROSS_BP + j * 8];
            __nv_bfloat16* vdst = &vt[stage][cc * ADIT_CROSS_BP + j * 8];
            if (g < L) {
                const uint32_t* gk = kvu + (size_t)g * (ADIT_CROSS_NQ / 2) + h * (ADIT_CROSS_BD / 2) + j * 4;
                unsigned dk = (unsigned)__cvta_generic_to_shared(kdst);
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dk), "l"(gk));
                const uint32_t* gv = vvu + (size_t)g * (ADIT_CROSS_NQ / 2) + h * (ADIT_CROSS_BD / 2) + j * 4;
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
        issue_tile(0, 0);
        if (ntiles == 1) asm volatile("cp.async.commit_group;\n" ::);  // empty group pads the count: with a single tile, C_PHASE_4's wait_group 1 waits for the real tile
        else if (ntiles > 1) issue_tile(1, 1);

        // 1) RMS rstd: scan this head's 128-column slice, staging the raw q' into qs
        const uint32_t* qu = reinterpret_cast<const uint32_t*>(qp);
        {
            const int r = tid >> 3;
            const int s2 = tid & 7;
            float sq = 0.f;
            const uint32_t* base = qu + (size_t)r * (ADIT_CROSS_NQ / 2) + h * (ADIT_CROSS_BD / 2);
            #pragma unroll
            for (int u = s2; u < ADIT_CROSS_BD / 2; u += 8) {
                uint32_t v = base[u];
                float a = bf16f((unsigned short)(v & 0xffffu));
                float bb = bf16f((unsigned short)(v >> 16));
                sq += a * a + bb * bb;
                *reinterpret_cast<uint32_t*>(qs + r * ADIT_CROSS_BP + 2 * u) = v;
            }
            #pragma unroll
            for (int off = 1; off < 8; off <<= 1)
                sq += __shfl_xor_sync(0xffffffffu, sq, off);
            // Each (head,row) owns its own slot ⇒ a plain write suffices (it was fp32 atomicAdd, ordered by CTA arrival)
            if (s2 == 0) rstd_scratch[h * ADIT_CROSS_M + r] = sq;
        }
    }
    ADIT_CROSS_GRID_SYNC();    // #3: all 24 heads' Σq² shards have been written

    // ================= C_PHASE_4: qs scaling + masked flash + write attn =================
    if (b < ADIT_CROSS_HEADS && ADIT_CROSS_PHASE(4)) {
        const int h = b;
        const int w = tid >> 5;
        const int lane2 = tid & 31;
        const int rowgrp = lane2 >> 3;
        const int d8 = lane2 & 7;
        const int row = w * 4 + rowgrp;
        __nv_bfloat16 (&kt)[ADIT_CROSS_NSTAGE][ADIT_CROSS_TILE * ADIT_CROSS_BP] = s.fl.kt;
        __nv_bfloat16 (&vt)[ADIT_CROSS_NSTAGE][ADIT_CROSS_TILE * ADIT_CROSS_BP] = s.fl.vt;
        __nv_bfloat16 (&qs)[ADIT_CROSS_M * ADIT_CROSS_BP] = s.fl.qs;
        uint32_t (&wns)[ADIT_CROSS_BD / 2] = s.fl.wns;

        // Non-cooperative split phases: the wns/msk/qs that C_PHASE_3 wrote to smem before this
        // phase do not survive across a launch, so rebuild the staging from global exactly as
        // C_PHASE_3 writes it (bytes only, no redoing of the RMS scan -- the shard slots were
        // already filled by phase 3).
        if (only_phase >= 0) {
            const uint32_t* wnu = reinterpret_cast<const uint32_t*>(wnq);
            if (tid < ADIT_CROSS_BD / 2) wns[tid] = wnu[h * (ADIT_CROSS_BD / 2) + tid];
            for (int i = tid; i < L; i += ADIT_CROSS_NTHREADS) msk[i] = mask[i];
            {
                const uint32_t* qu = reinterpret_cast<const uint32_t*>(qp);
                const int r = tid >> 3;
                const int s2 = tid & 7;
                const uint32_t* base = qu + (size_t)r * (ADIT_CROSS_NQ / 2) + h * (ADIT_CROSS_BD / 2);
                #pragma unroll
                for (int u = s2; u < ADIT_CROSS_BD / 2; u += 8) {
                    const uint32_t v = base[u];
                    *reinterpret_cast<uint32_t*>(qs + r * ADIT_CROSS_BP + 2 * u) = v;
                }
            }
            __syncthreads();
        }

        // rq read-back + in-place qs scaling. The full-row statistic = the 24 heads' partials
        // summed in **fixed order** (replacing the former fp32 atomicAdd from 24 CTAs), so the
        // summation order is unique ⇒ bit-exact reproducible.
        if (tid < 32) {
            float s = 0.f;
            #pragma unroll
            for (int hh = 0; hh < ADIT_CROSS_HEADS; ++hh) s += rstd_scratch[hh * ADIT_CROSS_M + tid];
            rq[tid] = rsqrtf(s / (float)ADIT_CROSS_NQ + ADIT_CROSS_EPS);
        }
        __syncthreads();
        {
            const int r = tid >> 3;
            const int s2 = tid & 7;
            const float rq_r = rq[r];
            #pragma unroll
            for (int u = s2; u < ADIT_CROSS_BD / 2; u += 8) {
                uint32_t v = *reinterpret_cast<const uint32_t*>(qs + r * ADIT_CROSS_BP + 2 * u);
                uint32_t w01 = wns[u];
                float xa = bf16f((unsigned short)(v & 0xffffu));
                float xb = bf16f((unsigned short)(v >> 16));
                float wa = bf16f((unsigned short)(w01 & 0xffffu));
                float wb = bf16f((unsigned short)(w01 >> 16));
                unsigned short na = bf16_bits(xa * rq_r * wa);
                unsigned short nb = bf16_bits(xb * rq_r * wb);
                *reinterpret_cast<uint32_t*>(qs + r * ADIT_CROSS_BP + 2 * u) =
                    (uint32_t)na | ((uint32_t)nb << 16);
            }
            __syncthreads();
        }

        // 3) flash over the cached kv
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

        float qr[16];
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            int d = d8 + 8 * i;
            qr[i] = bf16f(__bfloat16_as_ushort(qs[row * ADIT_CROSS_BP + d]));
        }

        auto flash_tile = [&](const __nv_bfloat16* ktile, const __nv_bfloat16* vtile,
                              int c0, int nrows) {
            bool mv[ADIT_CROSS_TILE];
            #pragma unroll
            for (int cc = 0; cc < ADIT_CROSS_TILE; ++cc)
                mv[cc] = (c0 + cc < nrows) && (msk[c0 + cc] != 0);
            const __nv_bfloat16* qb = qs + 16 * mh * ADIT_CROSS_BP;
            float s2v[2][4], s2w[2][4];
            #pragma unroll
            for (int n8 = 0; n8 < 2; ++n8)
                #pragma unroll
                for (int i = 0; i < 4; ++i) { s2v[n8][i] = 0.f; s2w[n8][i] = 0.f; }
            const int q_row = lane2 & 15, q_c16 = lane2 >> 4;
            // ---- QK^T: only the 4 warps with dq∈{0,1} compute one n8 block (two per warp was 4x redundant) ----
            if (dq < 2) {
                const int n8q = dq;
                const int k2_row = lane2 & 7, k2_col = (lane2 >> 3) & 1;
                const __nv_bfloat16* kpb = ktile + (n8q * 8 + k2_row) * ADIT_CROSS_BP + 8 * k2_col;
                float c0v[4] = {0.f, 0.f, 0.f, 0.f}, c1v[4] = {0.f, 0.f, 0.f, 0.f};
                #pragma unroll
                for (int kc = 0; kc < 8; ++kc) {
                    unsigned a0, a1, a2, a3, b0, b1;
                    ldm_x4(qb + q_row * ADIT_CROSS_BP + 16 * kc + 8 * q_c16, a0, a1, a2, a3);
                    ldm_x2(kpb + 16 * kc, b0, b1);
                    if (kc & 1) mma_f32(c1v, a0, a1, a2, a3, b0, b1);
                    else        mma_f32(c0v, a0, a1, a2, a3, b0, b1);
                }
                c0v[0] += c1v[0]; c0v[1] += c1v[1]; c0v[2] += c1v[2]; c0v[3] += c1v[3];
                float* sp0 = &sS[0][((mh * 2 + n8q) * 16 + gid) * 4 + tig];
                float* sp1 = &sS[1][((mh * 2 + n8q) * 16 + gid) * 4 + tig];
                sp0[0] = c0v[0]; sp1[0] = c0v[1];
                sp0[8 * 4] = c0v[2]; sp1[8 * 4] = c0v[3];
            }
            __syncthreads();
            #pragma unroll
            for (int n8 = 0; n8 < 2; ++n8) {
                const int rb = ((mh * 2 + n8) * 16 + gid) * 4 + tig;
                s2v[n8][0] = sS[0][rb];         s2v[n8][1] = sS[1][rb];
                s2v[n8][2] = sS[0][rb + 8 * 4]; s2v[n8][3] = sS[1][rb + 8 * 4];
            }
            // Aligned with torch SDPA: score *= 1/sqrt(head_dim) (fixed 2026-09: the missing factor
            // made p too sharp on real data, disagreeing with the engine/reference semantics -- the
            // old reference was missing it too, so self-consistency hid it)
            #pragma unroll
            for (int n8 = 0; n8 < 2; ++n8)
                #pragma unroll
                for (int i = 0; i < 4; ++i) s2v[n8][i] *= ADIT_CROSS_SATT;
            #pragma unroll
            for (int n8 = 0; n8 < 2; ++n8) {
                const int cbase = 8 * n8 + 2 * tig;
                if (!mv[cbase]) { s2v[n8][0] = -INFINITY; s2v[n8][2] = -INFINITY; }
                if (!mv[cbase + 1]) { s2v[n8][1] = -INFINITY; s2v[n8][3] = -INFINITY; }
            }
            float mm0 = fmaxf(fmaxf(s2v[0][0], s2v[0][1]), fmaxf(s2v[1][0], s2v[1][1]));
            float mm1 = fmaxf(fmaxf(s2v[0][2], s2v[0][3]), fmaxf(s2v[1][2], s2v[1][3]));
            mm0 = fmaxf(mm0, __shfl_xor_sync(0xffffffffu, mm0, 1));
            mm0 = fmaxf(mm0, __shfl_xor_sync(0xffffffffu, mm0, 2));
            mm1 = fmaxf(mm1, __shfl_xor_sync(0xffffffffu, mm1, 1));
            mm1 = fmaxf(mm1, __shfl_xor_sync(0xffffffffu, mm1, 2));
            const float mn0 = fmaxf(m0, mm0), mn1 = fmaxf(m1, mm1);
            const float al0 = __expf(m0 - mn0), al1 = __expf(m1 - mn1);
            float e[2][4];
            #pragma unroll
            for (int n8 = 0; n8 < 2; ++n8) {
                e[n8][0] = __expf(s2v[n8][0] - mn0); e[n8][1] = __expf(s2v[n8][1] - mn0);
                e[n8][2] = __expf(s2v[n8][2] - mn1); e[n8][3] = __expf(s2v[n8][3] - mn1);
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
                const __nv_bfloat16* vbp = vtile + v_row * ADIT_CROSS_BP + dq * 32 + 8 * v_c16;
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

        const int ntiles = (L + ADIT_CROSS_TILE - 1) / ADIT_CROSS_TILE;
        const uint32_t* kvu = reinterpret_cast<const uint32_t*>(k_cache);
        const uint32_t* vvu = reinterpret_cast<const uint32_t*>(v_cache);
        auto issue_tile = [&](int stage, int tb) {
            const int cc = tid >> 4;
            const int j = tid & 15;
            const int g = tb * ADIT_CROSS_TILE + cc;
            __nv_bfloat16* kdst = &kt[stage][cc * ADIT_CROSS_BP + j * 8];
            __nv_bfloat16* vdst = &vt[stage][cc * ADIT_CROSS_BP + j * 8];
            if (g < L) {
                const uint32_t* gk = kvu + (size_t)g * (ADIT_CROSS_NQ / 2) + h * (ADIT_CROSS_BD / 2) + j * 4;
                unsigned dk = (unsigned)__cvta_generic_to_shared(kdst);
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dk), "l"(gk));
                const uint32_t* gv = vvu + (size_t)g * (ADIT_CROSS_NQ / 2) + h * (ADIT_CROSS_BD / 2) + j * 4;
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

        // Cooperative path: tiles 0/1 were already issued by C_PHASE_3's prefetch (covering the
        // RMS+grid.sync latency), so they are not issued again here; re-issuing the same kt/vt smem
        // would form a write-write race with C_PHASE_3's in-flight cp.async (racecheck reports
        // 98304 hazards).
        // Non-cooperative split phases: C_PHASE_3's prefetch ends with the phase-3 launch and smem
        // does not survive a launch, so tiles 0/1 are re-issued here; the commit/counting is
        // verbatim-identical to C_PHASE_3's, keeping the pending group count seen by the
        // wait_group below unchanged.
        if (only_phase >= 0) {
            issue_tile(0, 0);
            if (ntiles == 1) asm volatile("cp.async.commit_group;\n" ::);
            else if (ntiles > 1) issue_tile(1, 1);
        }
        for (int tb = 0; tb < ntiles; ++tb) {
            const int stage = tb & 1;
            // The tail iteration uses wait_group 0 (as in the GEMM stretch): a completed empty
            // group occupies no pending slot, so wait_group 1 would let through and read smem that
            // has not landed on the last tile, which has only 1 real group left (fixed 2026-09).
            if (tb + 1 == ntiles) asm volatile("cp.async.wait_group 0;\n" ::);
            else                  asm volatile("cp.async.wait_group 1;\n" ::);
            __syncthreads();
            flash_tile(kt[stage], vt[stage], tb * ADIT_CROSS_TILE, L);
            __syncthreads();
            if (tb + 2 < ntiles) issue_tile(stage, tb + 2);
            else                 asm volatile("cp.async.commit_group;\n" ::);
        }

        // 4) normalize + write attn
        const float inv0 = 1.0f / l0, inv1 = 1.0f / l1;
        unsigned* ou = reinterpret_cast<unsigned*>(attn);
        const int orow = 16 * mh + gid;
        const int colbase = h * ADIT_CROSS_BD + dq * 32;
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const int cp = colbase + 8 * nt + 2 * tig;
            const unsigned v0 = (unsigned)bf16_bits(acc[nt][0] * inv0)
                              | ((unsigned)bf16_bits(acc[nt][1] * inv0) << 16);
            const unsigned v1 = (unsigned)bf16_bits(acc[nt][2] * inv1)
                              | ((unsigned)bf16_bits(acc[nt][3] * inv1) << 16);
            ou[(size_t)orow * (ADIT_CROSS_NQ / 2) + cp / 2] = v0;
            ou[(size_t)(orow + 8) * (ADIT_CROSS_NQ / 2) + cp / 2] = v1;
        }
    }
    ADIT_CROSS_GRID_SYNC();    // #4: attn visible grid-wide

    // ================= C_PHASE_5: attn row quantization (row r=b<32) =================
    if (ADIT_CROSS_PHASE(5)) {
        const uint32_t* au = reinterpret_cast<const uint32_t*>(attn);
        if (b < ADIT_CROSS_M) {
            const int r = b;
            uint32_t xv[6];
            float mx = 0.f;
            #pragma unroll
            for (int i = 0; i < 6; ++i) {
                xv[i] = au[(size_t)r * (ADIT_CROSS_NQ / 2) + wid * 192 + i * 32 + lane];
                float a = bf16f((unsigned short)(xv[i] & 0xffffu));
                float bb = bf16f((unsigned short)(xv[i] >> 16));
                mx = fmaxf(mx, fmaxf(fabsf(a), fabsf(bb)));
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                mx = fmaxf(mx, __shfl_down_sync(0xffffffffu, mx, off));
            if (lane == 0) o_shmax[wid] = mx;
            __syncthreads();
            if (wid == 0 && lane == 0) {
                float m = 0.f;
                #pragma unroll
                for (int i = 0; i < ADIT_CROSS_NWARPS; ++i) m = fmaxf(m, o_shmax[i]);
                const float sa_v = fmaxf(m / 448.0f, 1e-12f);
                sa0o[r] = sa_v;
                sh_sa = sa_v;
            }
            __syncthreads();
            const float sa_r = sh_sa;
            unsigned char* a8r = a8o + (size_t)r * ADIT_CROSS_NQ;
            #pragma unroll
            for (int i = 0; i < 6; ++i) {
                float a = bf16f((unsigned short)(xv[i] & 0xffffu));
                float bb = bf16f((unsigned short)(xv[i] >> 16));
                unsigned short pair = (unsigned short)fp8_rne(a / sa_r)
                                    | ((unsigned short)fp8_rne(bb / sa_r) << 8);
                *reinterpret_cast<unsigned short*>(a8r + 2 * (wid * 192 + i * 32 + lane)) = pair;
            }
        }
    }
    ADIT_CROSS_GRID_SYNC();    // #5: a8o / sa0o visible grid-wide

    // ================= C_PHASE_6: o fp8 GEMM + residual epilogue (slab=b<32) =================
    if (b < ADIT_CROSS_M && ADIT_CROSS_PHASE(6)) {
        constexpr int n8 = ADIT_CROSS_BN_C / (ADIT_CROSS_NWARPS / 2) / 8;   // 1
        constexpr int k_steps = ADIT_CROSS_BK_C / 32;            // 6
        constexpr int KQ = ADIT_CROSS_BK_C / 16;                 // 12
        constexpr int A_CH = ADIT_CROSS_M * ADIT_CROSS_BK_C / 16;           // 384
        constexpr int W_CH = ADIT_CROSS_BN_C * ADIT_CROSS_BK_C / 16;        // 384
        constexpr int TOT = A_CH + W_CH;              // 768
        uint8_t (&st)[ADIT_CROSS_STAGES][ADIT_CROSS_U_C2] = s.gemm2;

        auto issue = [&](int stage, int kt) {
            uint8_t* base = st[stage];
            #pragma unroll
            for (int c = tid; c < TOT; c += ADIT_CROSS_NTHREADS) {
                if (c < A_CH) {
                    int r = c / KQ;
                    int kq = (c % KQ) * 16;
                    const uint8_t* src = a8o + (size_t)r * ADIT_CROSS_NQ + kt * ADIT_CROSS_BK_C + kq;
                    unsigned sd = (unsigned)__cvta_generic_to_shared(base + r * ADIT_CROSS_ARS_C + kq);
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sd), "l"(src));
                } else {
                    int wc = c - A_CH;
                    int n = wc / KQ;
                    int kq = (wc % KQ) * 16;
                    const uint8_t* src = wo + ((size_t)b * ADIT_CROSS_BN_C + n) * ADIT_CROSS_NQ + kt * ADIT_CROSS_BK_C + kq;
                    unsigned sd = (unsigned)__cvta_generic_to_shared(base + ADIT_CROSS_M * ADIT_CROSS_ARS_C + n * ADIT_CROSS_BRS_C + kq);
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sd), "l"(src));
                }
            }
            asm volatile("cp.async.commit_group;\n" ::);
        };
        auto issue_empty = []() { asm volatile("cp.async.commit_group;\n" ::); };

        #pragma unroll
        for (int s2 = 0; s2 < ADIT_CROSS_STAGES - 1; ++s2)
            if (s2 < ADIT_CROSS_KtC) issue(s2, s2);

        float acc[n8][4];
        #pragma unroll
        for (int s2 = 0; s2 < n8; ++s2)
            #pragma unroll
            for (int i = 0; i < 4; ++i) acc[s2][i] = 0.f;

        for (int kt = 0; kt < ADIT_CROSS_KtC; ++kt) {
            const int stage = kt % ADIT_CROSS_STAGES;
            if (kt + 1 == ADIT_CROSS_KtC) asm volatile("cp.async.wait_group 0;\n" ::);
            else              asm volatile("cp.async.wait_group %0;\n" :: "n"(ADIT_CROSS_STAGES - 2));
            __syncthreads();

            const uint8_t* As_s = st[stage];
            const uint8_t* Ws_s = st[stage] + ADIT_CROSS_M * ADIT_CROSS_ARS_C;
            #pragma unroll
            for (int kk = 0; kk < k_steps; ++kk) {
                const int k_off = kk * 32;
                const int arow0 = m_slice * 16 + lane_g;
                const int arow1 = arow0 + 8;
                uint32_t a0 = *reinterpret_cast<const uint32_t*>(As_s + (size_t)arow0 * ADIT_CROSS_ARS_C + k_off + lane_t * 4);
                uint32_t a1 = *reinterpret_cast<const uint32_t*>(As_s + (size_t)arow1 * ADIT_CROSS_ARS_C + k_off + lane_t * 4);
                uint32_t a2 = *reinterpret_cast<const uint32_t*>(As_s + (size_t)arow0 * ADIT_CROSS_ARS_C + k_off + lane_t * 4 + 16);
                uint32_t a3 = *reinterpret_cast<const uint32_t*>(As_s + (size_t)arow1 * ADIT_CROSS_ARS_C + k_off + lane_t * 4 + 16);
                #pragma unroll
                for (int s2 = 0; s2 < n8; ++s2) {
                    const int bn = n_grp * (ADIT_CROSS_BN_C / (ADIT_CROSS_NWARPS / 2)) + s2 * 8 + lane_g;
                    const uint32_t* bp = reinterpret_cast<const uint32_t*>(Ws_s + (size_t)bn * ADIT_CROSS_BRS_C + k_off);
                    uint32_t b0w = bp[lane_t];
                    uint32_t b1w = bp[lane_t + 4];
                    asm volatile(
                        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                        : "+f"(acc[s2][0]), "+f"(acc[s2][1]), "+f"(acc[s2][2]), "+f"(acc[s2][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0w), "r"(b1w));
                }
            }
            const int nkt = kt + ADIT_CROSS_STAGES - 1;
            if (nkt < ADIT_CROSS_KtC) issue((kt - 1 + ADIT_CROSS_STAGES) % ADIT_CROSS_STAGES, nkt);
            else           issue_empty();
        }

        // epilogue: out = bf16(x + bf16(acc*sa0o*swo + bo)) (no gate)
        const int r0 = m_slice * 16 + lane_g;
        const int r1 = r0 + 8;
        const float sa0_r0 = sa0o[r0], sa0_r1 = sa0o[r1];
        const uint32_t* xu = reinterpret_cast<const uint32_t*>(x_in);   // residual = the raw x
        const uint32_t* bu = reinterpret_cast<const uint32_t*>(bo);
        #pragma unroll
        for (int s2 = 0; s2 < n8; ++s2) {
            const int col0 = b * ADIT_CROSS_BN_C + n_grp * (ADIT_CROSS_BN_C / (ADIT_CROSS_NWARPS / 2)) + s2 * 8 + lane_t * 2;
            const int col0h = col0 / 2;
            const float swc0 = swo[col0], swc1 = swo[col0 + 1];
            uint32_t bcp = bu[col0h];
            float bc0 = bf16f((unsigned short)(bcp & 0xffffu));
            float bc1 = bf16f((unsigned short)(bcp >> 16));
            uint32_t xp0 = xu[(size_t)r0 * (ADIT_CROSS_NO / 2) + col0h];
            uint32_t xp1 = xu[(size_t)r1 * (ADIT_CROSS_NO / 2) + col0h];
            float xa0 = bf16f((unsigned short)(xp0 & 0xffffu));
            float xa1 = bf16f((unsigned short)(xp0 >> 16));
            float xb0 = bf16f((unsigned short)(xp1 & 0xffffu));
            float xb1 = bf16f((unsigned short)(xp1 >> 16));
            unsigned short u0 = bf16_bits(acc[s2][0] * sa0_r0 * swc0 + bc0);
            unsigned short u1 = bf16_bits(acc[s2][1] * sa0_r0 * swc1 + bc1);
            unsigned short y0 = bf16_bits(xa0 + bf16f(u0));
            unsigned short y1 = bf16_bits(xa1 + bf16f(u1));
            *reinterpret_cast<unsigned*>(out + ((size_t)r0 * ADIT_CROSS_NO + col0) * 2) =
                (uint32_t)y0 | ((uint32_t)y1 << 16);
            unsigned short u2 = bf16_bits(acc[s2][2] * sa0_r1 * swc0 + bc0);
            unsigned short u3 = bf16_bits(acc[s2][3] * sa0_r1 * swc1 + bc1);
            unsigned short y2 = bf16_bits(xb0 + bf16f(u2));
            unsigned short y3 = bf16_bits(xb1 + bf16f(u3));
            *reinterpret_cast<unsigned*>(out + ((size_t)r1 * ADIT_CROSS_NO + col0) * 2) =
                (uint32_t)y2 | ((uint32_t)y3 << 16);
        }
    }
}

// ------------------------------------------------------------------ //
static void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        // Raise instead of exit(1): exit would kill the worker process outright, leaving the caller no handleable error.
        coop_fail(what, std::string("CUDA error: ") + cudaGetErrorString(e));
    }
}

// ---- Queries for registry/probe (pure query: raises if it does not fit; the caller catches it via try/except for the reason) ----
// This kernel's grid is set by the **problem size** (GRID = NQ/BN_A = 48); the cooperative form
// only additionally requires the whole grid to be co-resident.
// Dynamic smem = an L-byte mask (L = context length, at runtime), so the capacity is computed from
// the L given at call time.

// ---- Geometry self-report (shape constants + GEMM template parameters; template parameters and instantiation share one macro set) ----
extern "C" const char* adit_attn_cross_geom() {
    static const std::string s = fwam_geom_join({
        fwam_shapes({{"M", ADIT_CROSS_M}, {"H", ADIT_CROSS_H}, {"NQ", ADIT_CROSS_NQ}, {"NO", ADIT_CROSS_NO}, {"HEADS", ADIT_CROSS_HEADS}, {"BN_A", ADIT_CROSS_BN_A}, {"BK_A", ADIT_CROSS_BK_A}, {"STAGES", ADIT_CROSS_STAGES}, {"NT", ADIT_CROSS_NTHREADS}, {"GRID", ADIT_CROSS_GRID}}),
    });
    return s.c_str();
}

extern "C" int adit_attn_cross_grid() { return ADIT_CROSS_GRID; }
extern "C" int adit_attn_cross_coop_grid(int L) {
    // Not cached: L is a runtime parameter (the context length), and caching would make a query
    // with a different L use the wrong capacity. This entry point is only for probe (low
    // frequency); the launcher has its own static.
    coop_grid((const void*)adit_attn_cross_kernel, ADIT_CROSS_NTHREADS, L, ADIT_CROSS_GRID, "fused_cross_g");
    return ADIT_CROSS_GRID;
}

extern "C" void adit_attn_cross_cuda(
    const void* x_in, const void* w3, const void* b3,
    const void* wq, const float* swq, const void* bq,
    void* qp, void* a8q, float* sa0q,
    const void* wnq, const void* k_cache, const void* v_cache, const void* mask,
    void* attn,
    const void* wo, const float* swo, const void* bo,
    void* out, void* a8o, float* sa0o, float* rstd_scratch,
    int L, int only_phase, cudaStream_t stream)
{
    dim3 grid(ADIT_CROSS_GRID), block(ADIT_CROSS_NTHREADS);
    void* args[] = {
        (void*)&x_in, (void*)&w3, (void*)&b3,
        (void*)&wq, (void*)&swq, (void*)&bq,
        (void*)&qp, (void*)&a8q, (void*)&sa0q,
        (void*)&wnq, (void*)&k_cache, (void*)&v_cache, (void*)&mask,
        (void*)&attn,
        (void*)&wo, (void*)&swo, (void*)&bo,
        (void*)&out, (void*)&a8o, (void*)&sa0o, (void*)&rstd_scratch,
        (void*)&L, (void*)&only_phase,
    };
    cudaError_t e;
    if (only_phase < 0) {
        // Device capacity check: grid = NQ/BN_A is decided by the problem size (phases claim by
        // slab=b) and the cooperative kernel requires the whole grid to be co-resident; report it
        // clearly here when the capacity falls short. Dynamic smem = an L-byte mask.
        static const int cap = coop_grid((const void*)adit_attn_cross_kernel, ADIT_CROSS_NTHREADS, (int)L, ADIT_CROSS_GRID,
                                         "fused_cross_g");
        (void)cap;
        e = cudaLaunchCooperativeKernel((void*)adit_attn_cross_kernel, grid, block,
                                        args, (size_t)L, stream);
    } else {
        // Non-cooperative split phases: the grid stays GRID (= NQ/BN_A, decided by the tile
        // geometry). Dynamic smem = an L-byte mask; the static part is ~41.7KB, so when the 48KB
        // default limit is exceeded the dynamic smem limit must be raised explicitly.
        cudaFuncSetAttribute((const void*)adit_attn_cross_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)L);
        e = cudaLaunchKernel((void*)adit_attn_cross_kernel, grid, block, args, (size_t)L, stream);
    }
    check(e, only_phase < 0 ? "cooperative launch adit_attn_cross" : "split launch adit_attn_cross");
}
