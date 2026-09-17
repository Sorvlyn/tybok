// S: one cooperative kernel for the whole action self-attention chain
// (norm1+modulate+quantization → qkv GEMM → qk-norm/RoPE/bf16 flash → o-proj + gate residual).
//
// Phases (grid = 96 CTA × 256 threads, __launch_bounds__(256, 3)):
//   S_PHASE_1 [b<M]  norm1 (LN without affine) + modulate + row quantization → a8x/sa0x
//   S_PHASE_2 [b<96] qkv fp8 GEMM → qkv[32, 9216]
//   S_PHASE_3 [b<24] q/k full-row RMS shard scan (each (head, row) owns a slot, plain write) → rstd_scratch
//   S_PHASE_4 [b<24] rstd fixed-order reduction + qk-norm(bf16)/RoPE + bf16 flash → attn
//   S_PHASE_5 [b<M]  attn row quantization → a8a/sa0a
//   S_PHASE_6 [b<32] o fp8 GEMM + gate residual → out
//
// Phases are separated by grid.sync; S_PHASE_3 uses plain writes (not atomicAdd) so that the same
// input is reproducible.
// Numerics: bf16 RN + fp8 software single RNE.
// smem: a large union buffer (the A/D stage and the BC flash phases reuse it mutually exclusively)
// + a small static area, ~46.4KB ≤ 48KB total.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cstdint>
#include <cstdio>
#include <cassert>
#include "coop_grid.h"
#include "geom_report.h"            // geometry self-report (instantiation and self-report share one macro set)
#include "adit_common.h"                       // adit device-side helpers (pulls in dit_common.h / common.h)

namespace cg = cooperative_groups;

// A phase (qkv GEMM) geometry: BN_QKV=96 → grid=96 (2 CTA/SM can be co-resident; the element-wise
// arithmetic is bit-exact with BN=64, only the CTA column ownership changes). D phase tile:
// BN_O=32 / BK_O=128 / STAGES_O=5 (bit-level equivalent).
constexpr int ADIT_SELF_M = 32;
constexpr int ADIT_SELF_H = 1024;
constexpr int ADIT_SELF_N = 9216;      // packed qkv output
constexpr int ADIT_SELF_H3 = 3072;     // per-branch width / attn width
constexpr int ADIT_SELF_BN_QKV = 96;
constexpr int ADIT_SELF_BN_O = 32;      // 64 -> 32: S_PHASE_6's grid goes from 16 CTA to 32 CTA
constexpr int ADIT_SELF_BK_Q = 64;      // qkv phase (S_PHASE_2) K shard
constexpr int ADIT_SELF_BK_O = 128;     // o phase (S_PHASE_6) K shard: 64 -> 128, k-tile 48 -> 24
constexpr int ADIT_SELF_STAGES_Q = 3;   // S_PHASE_2 pipeline stages (carried over from the original)
constexpr int ADIT_SELF_STAGES_O = 5;   // S_PHASE_6: 3 -> 5 (smem 5*9216 = 45KB < the 48KB static limit)
constexpr int ADIT_SELF_NTHREADS = 256;
constexpr int ADIT_SELF_NWARPS = ADIT_SELF_NTHREADS / 32;
constexpr float ADIT_SELF_EPS = 1e-6f;

constexpr int ADIT_SELF_GRID = ADIT_SELF_N / ADIT_SELF_BN_QKV;              // 96
constexpr int ADIT_SELF_ARS_Q = ADIT_SELF_BK_Q + 16;              // 80
constexpr int ADIT_SELF_BRS_Q = ADIT_SELF_BK_Q + 16;
constexpr int ADIT_SELF_ARS_O = ADIT_SELF_BK_O + 16;              // 144
constexpr int ADIT_SELF_BRS_O = ADIT_SELF_BK_O + 16;
constexpr int ADIT_SELF_U_A = ADIT_SELF_M * ADIT_SELF_ARS_Q + ADIT_SELF_BN_QKV * ADIT_SELF_BRS_Q;   // 10240 (A qkv GEMM stage)
constexpr int ADIT_SELF_U_D = ADIT_SELF_M * ADIT_SELF_ARS_O + ADIT_SELF_BN_O * ADIT_SELF_BRS_O;     // 9216 (D o GEMM stage)
constexpr int ADIT_SELF_KtQKV = ADIT_SELF_H / ADIT_SELF_BK_Q;               // 16
constexpr int ADIT_SELF_KtO = ADIT_SELF_H3 / ADIT_SELF_BK_O;                // 24

// ---- BC flash geometry ----
constexpr int ADIT_SELF_BD = 128;
constexpr int ADIT_SELF_BP = 136;
constexpr int ADIT_SELF_HEADS = ADIT_SELF_H3 / ADIT_SELF_BD;                // 24
// Per-head partial slots for qk-norm: S_PHASE_3 writes [g/q side × head × row], S_PHASE_4 reduces
// them into the full-row statistic in fixed order. Each slot is written exactly once by one thread
// ⇒ **no zeroing needed**, and no atomic either (the order is decided by S_PHASE_4's for).
constexpr int ADIT_SELF_RSTD_SLOTS = 2 * ADIT_SELF_HEADS * ADIT_SELF_M;     // 1536
constexpr int ADIT_SELF_TILE = 16;
constexpr int ADIT_SELF_NSTAGE = 2;
constexpr float ADIT_SELF_SATT = 0.08838834764831845f;  // 1/sqrt(128): torch SDPA's default scale (the flash score must be multiplied)

static_assert(ADIT_SELF_STAGES_Q * ADIT_SELF_U_A <= 48 * 1024, "A GEMM smem over budget");
static_assert(ADIT_SELF_STAGES_O * ADIT_SELF_U_D <= 48 * 1024, "D GEMM smem over budget");
static_assert(ADIT_SELF_BN_O % 32 == 0 && ADIT_SELF_BK_O % 32 == 0, "BN/BK must be a multiple of 32");

// ------------------------------------------------------------------ //
// Device-side helpers: bf16f/gelu_tanh in common.h (shared by all three); bf16_bits/fp8_rne/
// mod_bf16/ldm_x2/x4/x4_t/mma_f32 in dit_common.h (adit+vdit); bf16_pair in adit_common.h.
// ------------------------------------------------------------------ //

// Non-cooperative fallback (only_phase >= 0): each phase is launched separately and visibility
// between phases comes from stream ordering. The split path builds no grid barrier, so a plain
// `cudaLaunchKernel` can be used (no co-residency requirement).
//   ADIT_SELF_PHASE(k)     -- true for that phase only when only_phase>=0; always true when <0 (keeps the old behavior).
//   ADIT_SELF_GRID_SYNC()  -- unchanged on the cooperative path (fence + grid.sync); a no-op on the split path.
#define ADIT_SELF_PHASE(k)     (only_phase < 0 || only_phase == (k))
#define ADIT_SELF_GRID_SYNC()  do { if (only_phase < 0) { __threadfence(); cg::this_grid().sync(); } } while (0)

// ------------------------------------------------------------------ //
__global__ void __launch_bounds__(ADIT_SELF_NTHREADS, 3)
adit_attn_self_kernel(
    // A inputs
    const uint8_t* __restrict__ x_in,        // bf16 [M, H] (block input = the D/FFN residual)
    const uint8_t* __restrict__ shift_msa,   // bf16 [H]
    const uint8_t* __restrict__ scale_msa,   // bf16 [H]
    const uint8_t* __restrict__ wqkv,        // fp8 [N, H]
    const float* __restrict__ swqkv,         // fp32 [N]
    const uint8_t* __restrict__ bqkv,        // bf16 [N]
    // BC inputs
    const uint8_t* __restrict__ kv_cache,    // bf16 [L, H3] (video cache, already normed + RoPE'd)
    const uint8_t* __restrict__ v_cache,     // bf16 [L, H3]
    const uint8_t* __restrict__ wnq,         // bf16 [H3]
    const uint8_t* __restrict__ wnk,         // bf16 [H3]
    const float* __restrict__ cos_t,         // fp32 [M, 64]
    const float* __restrict__ sin_t,         // fp32 [M, 64]
    // D inputs
    const uint8_t* __restrict__ gate_msa,    // bf16 [H]
    const uint8_t* __restrict__ wo,          // fp8 [H, H3]
    const float* __restrict__ swo,           // fp32 [H]
    const uint8_t* __restrict__ bo,          // bf16 [H]
    // Outputs
    uint8_t* __restrict__ out,               // bf16 [M, H] (self-chain output)
    // Internal scratch (host-allocated; must be globally visible across grid.sync)
    uint8_t* __restrict__ qkv,               // bf16 [M, N]
    uint8_t* __restrict__ attn,              // bf16 [M, H3]
    uint8_t* __restrict__ a8x,               // fp8 [M, H]
    float* __restrict__ sa0x,                // fp32 [M]
    uint8_t* __restrict__ a8a,               // fp8 [M, H3]
    float* __restrict__ sa0a,                // fp32 [M]
    float* __restrict__ rstd_scratch,        // fp32 [RSTD_SLOTS]: per-head Σq²/Σk² shard slots (no zeroing needed)
    int L, int only_phase)                  // non-cooperative fallback: >=0 runs that phase only; <0 = the cooperative full pipeline
{
    const int b = blockIdx.x;               // 0..143
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;

    // ---------------- smem: large union buffer + small static area ----------------
    __shared__ __align__(16) union {
        uint8_t gA[ADIT_SELF_STAGES_Q][ADIT_SELF_U_A];                    // A qkv GEMM stages (30720B)
        uint8_t gD[ADIT_SELF_STAGES_O][ADIT_SELF_U_D];                    // D o GEMM stages (46080B)
        struct {
            __nv_bfloat16 qs[ADIT_SELF_M * ADIT_SELF_BP];
            __nv_bfloat16 ks[ADIT_SELF_M * ADIT_SELF_BP];
            __nv_bfloat16 vs[ADIT_SELF_M * ADIT_SELF_BP];
            __nv_bfloat16 kt[ADIT_SELF_NSTAGE][ADIT_SELF_TILE * ADIT_SELF_BP];
            __nv_bfloat16 vt[ADIT_SELF_NSTAGE][ADIT_SELF_TILE * ADIT_SELF_BP];
        } fl;                                         // BC flash (43520B)
    } s;
    __shared__ float ph_mean, ph_rstd;       // S_PHASE_1 row statistic broadcast (b<M)
    __shared__ float ph_red[ADIT_SELF_NWARPS];         // S_PHASE_1 cross-warp amax reduction
    __shared__ float ph_sa;                  // S_PHASE_1 scale broadcast
    __shared__ float rq[ADIT_SELF_M], rk[ADIT_SELF_M];           // BC (b<24)
    // The S buffer shared by QK: split into two arrays by even/odd column so that each lane's
    // address = its lane number ⇒ zero bank conflicts (the original [row*18 + 2*tig] row stride
    // was a stable 2-way conflict with no way around it).
    // sS[e][ ((mh*2+n8)*16 + row)*4 + tig ], e=0 holds the 2t columns, e=1 the 2t+1 columns.
    __shared__ float sS[2][2 * 2 * 16 * 4];
    __shared__ float shmax[ADIT_SELF_NWARPS];          // S_PHASE_5 (b<M)
    __shared__ float sh_sa;                  // S_PHASE_5 block-wide broadcast

    const int lane_g = lane >> 2;
    const int lane_t = lane & 3;
    const int m_slice = wid / (ADIT_SELF_NWARPS / 2);
    const int n_grp = wid % (ADIT_SELF_NWARPS / 2);

    // ================= S_PHASE_1: norm1 + modulate + row quantization (row-parallel, b<M) =================
    if (b < ADIT_SELF_M && ADIT_SELF_PHASE(1)) {                      // row-parallel: one CTA per row
        const int r = b;
        const uint32_t* xu = reinterpret_cast<const uint32_t*>(x_in);
        const uint32_t* scu = reinterpret_cast<const uint32_t*>(scale_msa);
        const uint32_t* shu = reinterpret_cast<const uint32_t*>(shift_msa);

        // ---- pass 1: row statistics (warp0; the lane→i mapping and the shfl tree are bit-exact
        //      with the original, so psum's summation order is unchanged) ----
        if (wid == 0) {
            float ss = 0.f, qq = 0.f;
            #pragma unroll
            for (int i = lane; i < ADIT_SELF_H / 2; i += 32) {
                uint32_t wd = xu[(size_t)r * (ADIT_SELF_H / 2) + i];
                float a = bf16f((unsigned short)(wd & 0xffffu));
                float bb = bf16f((unsigned short)(wd >> 16));
                ss += a; ss += bb;
                qq += a * a; qq += bb * bb;
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                ss += __shfl_down_sync(0xffffffffu, ss, off);
                qq += __shfl_down_sync(0xffffffffu, qq, off);
            }
            if (lane == 0) {
                float mean = ss / (float)ADIT_SELF_H;
                float var = fmaxf(qq / (float)ADIT_SELF_H - mean * mean, 0.f);
                ph_mean = mean;
                ph_rstd = rsqrtf(var + ADIT_SELF_EPS);
            }
        }
        __syncthreads();
        const float mean_r = ph_mean, rstd_r = ph_rstd;

        // ---- pass 2: modulate + row amax (fmaxf is exact and order-independent) ----
        float mxl = 0.f;
        #pragma unroll 4
        for (int i = tid; i < ADIT_SELF_H / 2; i += ADIT_SELF_NTHREADS) {
            uint32_t sc01 = scu[i], sh01 = shu[i];
            float sc_a = bf16f((unsigned short)(sc01 & 0xffffu));
            float sc_b = bf16f((unsigned short)(sc01 >> 16));
            float sh_a = bf16f((unsigned short)(sh01 & 0xffffu));
            float sh_b = bf16f((unsigned short)(sh01 >> 16));
            uint32_t wd = xu[(size_t)r * (ADIT_SELF_H / 2) + i];
            float xa = bf16f((unsigned short)(wd & 0xffffu));
            float xb = bf16f((unsigned short)(wd >> 16));
            float ma = mod_bf16(xa, mean_r, rstd_r, sc_a, sh_a);
            float mb = mod_bf16(xb, mean_r, rstd_r, sc_b, sh_b);
            mxl = fmaxf(mxl, fmaxf(fabsf(ma), fabsf(mb)));
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            mxl = fmaxf(mxl, __shfl_down_sync(0xffffffffu, mxl, off));
        if (lane == 0) ph_red[wid] = mxl;
        __syncthreads();
        if (wid == 0 && lane == 0) {
            float m = 0.f;
            #pragma unroll
            for (int i = 0; i < ADIT_SELF_NWARPS; ++i) m = fmaxf(m, ph_red[i]);
            const float sa_v = fmaxf(m / 448.0f, 1e-12f);
            sa0x[r] = sa_v;
            ph_sa = sa_v;
        }
        __syncthreads();
        const float sa_r = ph_sa;

        // ---- pass 3: quantization (element-wise, order-independent) ----
        unsigned char* a8r = a8x + (size_t)r * ADIT_SELF_H;
        #pragma unroll 4
        for (int i = tid; i < ADIT_SELF_H / 2; i += ADIT_SELF_NTHREADS) {
            uint32_t sc01 = scu[i], sh01 = shu[i];
            float sc_a = bf16f((unsigned short)(sc01 & 0xffffu));
            float sc_b = bf16f((unsigned short)(sc01 >> 16));
            float sh_a = bf16f((unsigned short)(sh01 & 0xffffu));
            float sh_b = bf16f((unsigned short)(sh01 >> 16));
            uint32_t wd = xu[(size_t)r * (ADIT_SELF_H / 2) + i];
            float xa = bf16f((unsigned short)(wd & 0xffffu));
            float xb = bf16f((unsigned short)(wd >> 16));
            float ma = mod_bf16(xa, mean_r, rstd_r, sc_a, sh_a);
            float mb = mod_bf16(xb, mean_r, rstd_r, sc_b, sh_b);
            unsigned short pair = (unsigned short)fp8_rne(ma / sa_r) |
                                  ((unsigned short)fp8_rne(mb / sa_r) << 8);
            *reinterpret_cast<unsigned short*>(a8r + 2 * i) = pair;
        }
    }

    ADIT_SELF_GRID_SYNC();    // #1

    // ================= S_PHASE_2: qkv fp8 GEMM + epilogue (BN_QKV=96/grid 96, slab=b) =================
    if (ADIT_SELF_PHASE(2)) {
        constexpr int n8 = ADIT_SELF_BN_QKV / (ADIT_SELF_NWARPS / 2) / 8;   // 3
        constexpr int k_steps = ADIT_SELF_BK_Q / 32;                // 2
        constexpr int KQ = ADIT_SELF_BK_Q / 16;                     // 4
        constexpr int A_CH = ADIT_SELF_M * ADIT_SELF_BK_Q / 16;               // 128
        constexpr int W_CH = ADIT_SELF_BN_QKV * ADIT_SELF_BK_Q / 16;          // 384
        constexpr int TOT = A_CH + W_CH;                // 512 = 2 x NTHREADS
        uint8_t (&st)[ADIT_SELF_STAGES_Q][ADIT_SELF_U_A] = s.gA;
        auto issue = [&](int stage, int kt) {
            uint8_t* base = st[stage];
            #pragma unroll
            for (int c = tid; c < TOT; c += ADIT_SELF_NTHREADS) {
                if (c < A_CH) {
                    int r = c / KQ;
                    int kq = (c % KQ) * 16;
                    const uint8_t* src = a8x + (size_t)r * ADIT_SELF_H + kt * ADIT_SELF_BK_Q + kq;
                    unsigned sd = (unsigned)__cvta_generic_to_shared(base + r * ADIT_SELF_ARS_Q + kq);
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sd), "l"(src));
                } else {
                    int wc = c - A_CH;
                    int n = wc / KQ;
                    int kq = (wc % KQ) * 16;
                    const uint8_t* src = wqkv + ((size_t)b * ADIT_SELF_BN_QKV + n) * ADIT_SELF_H + kt * ADIT_SELF_BK_Q + kq;
                    unsigned sd = (unsigned)__cvta_generic_to_shared(base + ADIT_SELF_M * ADIT_SELF_ARS_Q + n * ADIT_SELF_BRS_Q + kq);
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sd), "l"(src));
                }
            }
            asm volatile("cp.async.commit_group;\n" ::);
        };
        auto issue_empty = []() { asm volatile("cp.async.commit_group;\n" ::); };

        #pragma unroll
        for (int st2 = 0; st2 < ADIT_SELF_STAGES_Q - 1; ++st2)
            if (st2 < ADIT_SELF_KtQKV) issue(st2, st2);

        float acc[n8][4];
        #pragma unroll
        for (int s2 = 0; s2 < n8; ++s2)
            #pragma unroll
            for (int i = 0; i < 4; ++i) acc[s2][i] = 0.f;

        for (int kt = 0; kt < ADIT_SELF_KtQKV; ++kt) {
            const int stage = kt % ADIT_SELF_STAGES_Q;
            if (kt + 1 == ADIT_SELF_KtQKV) asm volatile("cp.async.wait_group 0;\n" ::);
            else                asm volatile("cp.async.wait_group %0;\n" :: "n"(ADIT_SELF_STAGES_Q - 2));
            __syncthreads();
            const uint8_t* As_s = st[stage];
            const uint8_t* Ws_s = st[stage] + ADIT_SELF_M * ADIT_SELF_ARS_Q;
            #pragma unroll
            for (int kk = 0; kk < k_steps; ++kk) {
                const int k_off = kk * 32;
                const int arow0 = m_slice * 16 + lane_g;
                const int arow1 = arow0 + 8;
                uint32_t a0 = *reinterpret_cast<const uint32_t*>(As_s + (size_t)arow0 * ADIT_SELF_ARS_Q + k_off + lane_t * 4);
                uint32_t a1 = *reinterpret_cast<const uint32_t*>(As_s + (size_t)arow1 * ADIT_SELF_ARS_Q + k_off + lane_t * 4);
                uint32_t a2 = *reinterpret_cast<const uint32_t*>(As_s + (size_t)arow0 * ADIT_SELF_ARS_Q + k_off + lane_t * 4 + 16);
                uint32_t a3 = *reinterpret_cast<const uint32_t*>(As_s + (size_t)arow1 * ADIT_SELF_ARS_Q + k_off + lane_t * 4 + 16);
                #pragma unroll
                for (int s2 = 0; s2 < n8; ++s2) {
                    const int bn = n_grp * (ADIT_SELF_BN_QKV / (ADIT_SELF_NWARPS / 2)) + s2 * 8 + lane_g;
                    const uint32_t* bp = reinterpret_cast<const uint32_t*>(Ws_s + (size_t)bn * ADIT_SELF_BRS_Q + k_off);
                    uint32_t b0w = bp[lane_t];
                    uint32_t b1w = bp[lane_t + 4];
                    asm volatile(
                        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                        : "+f"(acc[s2][0]), "+f"(acc[s2][1]), "+f"(acc[s2][2]), "+f"(acc[s2][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0w), "r"(b1w));
                }
            }
            const int nkt = kt + ADIT_SELF_STAGES_Q - 1;
            if (nkt < ADIT_SELF_KtQKV) issue((kt - 1 + ADIT_SELF_STAGES_Q) % ADIT_SELF_STAGES_Q, nkt);
            else             issue_empty();
        }
        // epilogue: qkv = bf16(acc*sa0x*swqkv + bqkv)
        const int r0 = m_slice * 16 + lane_g;
        const int r1 = r0 + 8;
        const float sa0_r0 = sa0x[r0], sa0_r1 = sa0x[r1];
        #pragma unroll
        for (int s2 = 0; s2 < n8; ++s2) {
            const int col0 = b * ADIT_SELF_BN_QKV + n_grp * (ADIT_SELF_BN_QKV / (ADIT_SELF_NWARPS / 2)) + s2 * 8 + lane_t * 2;
            float swc0 = swqkv[col0], swc1 = swqkv[col0 + 1];
            uint32_t bc = *reinterpret_cast<const uint32_t*>(bqkv + (size_t)col0 * 2);
            float bc0 = bf16f((unsigned short)(bc & 0xffffu)), bc1 = bf16f((unsigned short)(bc >> 16));
            float u0 = bf16f(bf16_bits(acc[s2][0] * sa0_r0 * swc0 + bc0));
            float u1 = bf16f(bf16_bits(acc[s2][1] * sa0_r0 * swc1 + bc1));
            *reinterpret_cast<uint32_t*>(qkv + ((size_t)r0 * ADIT_SELF_N + col0) * 2) = bf16_pair(u0, u1);
            float u2 = bf16f(bf16_bits(acc[s2][2] * sa0_r1 * swc0 + bc0));
            float u3 = bf16f(bf16_bits(acc[s2][3] * sa0_r1 * swc1 + bc1));
            *reinterpret_cast<uint32_t*>(qkv + ((size_t)r1 * ADIT_SELF_N + col0) * 2) = bf16_pair(u2, u3);
        }
    }
    ADIT_SELF_GRID_SYNC();    // #2

    // ================= S_PHASE_3: q/k full-row RMS shard scan (head h=b<24) =================
    // The math is unchanged; only the **publication scheme** changed: each row used to be collected
    // by one fp32 accumulator (atomicAdd, 24 heads fighting over the same address); now each head
    // writes its own slot and the order is left to S_PHASE_4.
    if (b < ADIT_SELF_HEADS && ADIT_SELF_PHASE(3)) {
        const int h = b;
        const uint32_t* qkvu = reinterpret_cast<const uint32_t*>(qkv);
        constexpr int W2 = ADIT_SELF_N / 2;
        constexpr int H32 = ADIT_SELF_H3 / 2;
        const int r = tid >> 3;
        const int s2 = tid & 7;
        float sq = 0.f, sk = 0.f;
        const uint32_t* base = qkvu + (size_t)r * W2 + h * (ADIT_SELF_BD / 2);
        #pragma unroll
        for (int u = s2; u < ADIT_SELF_BD / 2; u += 8) {
            uint32_t vq = base[u];
            uint32_t vk = base[H32 + u];
            float aq = bf16f((unsigned short)(vq & 0xffffu));
            float bq = bf16f((unsigned short)(vq >> 16));
            float ak = bf16f((unsigned short)(vk & 0xffffu));
            float bk = bf16f((unsigned short)(vk >> 16));
            sq += aq * aq + bq * bq;
            sk += ak * ak + bk * bk;
        }
        #pragma unroll
        for (int off = 1; off < 8; off <<= 1) {
            sq += __shfl_xor_sync(0xffffffffu, sq, off);
            sk += __shfl_xor_sync(0xffffffffu, sk, off);
        }
        if (s2 == 0) {
            // Each (head,row) owns its own slot ⇒ a plain write suffices (no cross-CTA accumulation, hence no ordering concern).
            rstd_scratch[h * ADIT_SELF_M + r] = sq;
            rstd_scratch[ADIT_SELF_HEADS * ADIT_SELF_M + h * ADIT_SELF_M + r] = sk;
        }
    }
    ADIT_SELF_GRID_SYNC();    // #3

    // ================= S_PHASE_4: norm/RoPE + flash + write attn (head h=b<24) =================
    if (b < ADIT_SELF_HEADS && ADIT_SELF_PHASE(4)) {
        const int h = b;
        const int w = tid >> 5;
        const int lane2 = tid & 31;
        const int rowgrp = lane2 >> 3;
        const int d8 = lane2 & 7;
        const int row = w * 4 + rowgrp;
        __nv_bfloat16 (&qs)[ADIT_SELF_M * ADIT_SELF_BP] = s.fl.qs;
        __nv_bfloat16 (&ks)[ADIT_SELF_M * ADIT_SELF_BP] = s.fl.ks;
        __nv_bfloat16 (&vs)[ADIT_SELF_M * ADIT_SELF_BP] = s.fl.vs;
        __nv_bfloat16 (&kt)[ADIT_SELF_NSTAGE][ADIT_SELF_TILE * ADIT_SELF_BP] = s.fl.kt;
        __nv_bfloat16 (&vt)[ADIT_SELF_NSTAGE][ADIT_SELF_TILE * ADIT_SELF_BP] = s.fl.vt;

        // The full-row statistic = the 24 heads' partials summed in **fixed order** (h = 0..23).
        // This step replaces the former fp32 atomicAdd from 24 CTAs, so the summation order is
        // uniquely decided by the for here ⇒ bit-exact reproducible.
        // 24 global reads per row: the 6KB slot area was just written by S_PHASE_3 and hits L1/L2;
        // 64 threads x 24 each is a negligible cost.
        if (tid < 32) {
            float s = 0.f;
            #pragma unroll
            for (int hh = 0; hh < ADIT_SELF_HEADS; ++hh) s += rstd_scratch[hh * ADIT_SELF_M + tid];
            rq[tid] = rsqrtf(s / (float)ADIT_SELF_H3 + ADIT_SELF_EPS);
        } else if (tid < 64) {
            const int r = tid - 32;
            float s = 0.f;
            #pragma unroll
            for (int hh = 0; hh < ADIT_SELF_HEADS; ++hh) s += rstd_scratch[ADIT_SELF_HEADS * ADIT_SELF_M + hh * ADIT_SELF_M + r];
            rk[r] = rsqrtf(s / (float)ADIT_SELF_H3 + ADIT_SELF_EPS);
        }
        __syncthreads();

        // ---- norm (bf16 rounding) + dense rope -> qs/ks; fresh v -> vs ----
        {
            const int d0 = h * ADIT_SELF_BD;
            const __nv_bfloat16* qkvp = reinterpret_cast<const __nv_bfloat16*>(qkv);
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                int idx = tid + k * ADIT_SELF_NTHREADS;
                int rr = idx >> 6;
                int j = idx & 63;
                int ca = d0 + 2 * j;
                float rq_r = rq[rr];
                const __nv_bfloat16* wq = reinterpret_cast<const __nv_bfloat16*>(wnq) + ca;
                float xa = bf16f(__bfloat16_as_ushort(qkvp[(size_t)rr * ADIT_SELF_N + ca]));
                float xb = bf16f(__bfloat16_as_ushort(qkvp[(size_t)rr * ADIT_SELF_N + ca + 1]));
                float wa = bf16f(__bfloat16_as_ushort(wq[0]));
                float wb = bf16f(__bfloat16_as_ushort(wq[1]));
                float na = bf16f(bf16_bits(xa * rq_r * wa));
                float nb = bf16f(bf16_bits(xb * rq_r * wb));
                float cth = cos_t[rr * 64 + j], sth = sin_t[rr * 64 + j];
                float re = na * cth - nb * sth;
                float im = na * sth + nb * cth;
                qs[rr * ADIT_SELF_BP + 2 * j] = __float2bfloat16_rn(re);
                qs[rr * ADIT_SELF_BP + 2 * j + 1] = __float2bfloat16_rn(im);
                float rk_r = rk[rr];
                const __nv_bfloat16* wk = reinterpret_cast<const __nv_bfloat16*>(wnk) + ca;
                float ka = bf16f(__bfloat16_as_ushort(qkvp[(size_t)rr * ADIT_SELF_N + ADIT_SELF_H3 + ca]));
                float kbv = bf16f(__bfloat16_as_ushort(qkvp[(size_t)rr * ADIT_SELF_N + ADIT_SELF_H3 + ca + 1]));
                float kwa = bf16f(__bfloat16_as_ushort(wk[0]));
                float kwb = bf16f(__bfloat16_as_ushort(wk[1]));
                float kna = bf16f(bf16_bits(ka * rk_r * kwa));
                float knb = bf16f(bf16_bits(kbv * rk_r * kwb));
                ks[rr * ADIT_SELF_BP + 2 * j] = __float2bfloat16_rn(kna * cth - knb * sth);
                ks[rr * ADIT_SELF_BP + 2 * j + 1] = __float2bfloat16_rn(kna * sth + knb * cth);
            }
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                int d = d8 + 8 * i;
                vs[row * ADIT_SELF_BP + d] = qkvp[(size_t)row * ADIT_SELF_N + 2 * ADIT_SELF_H3 + d0 + d];
            }
            __syncthreads();
        }

        // ---- flash: the video-cache stretch (FA2-style mma + cp.async double buffering) ----
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

        auto flash_tile = [&](const __nv_bfloat16* ktile, const __nv_bfloat16* vtile, int nrows) {
            const __nv_bfloat16* qb = qs + 16 * mh * ADIT_SELF_BP;
            float s[2][4], s2[2][4];
            #pragma unroll
            for (int n8 = 0; n8 < 2; ++n8)
                #pragma unroll
                for (int i = 0; i < 4; ++i) { s[n8][i] = 0.f; s2[n8][i] = 0.f; }
            const int q_row = lane2 & 15, q_c16 = lane2 >> 4;
            // ---- QK^T: only the 4 warps with dq∈{0,1} compute one n8 block (two per warp was 4x redundant) ----
            if (dq < 2) {
                const int n8q = dq;
                const int k2_row = lane2 & 7, k2_col = (lane2 >> 3) & 1;
                const __nv_bfloat16* kpb = ktile + (n8q * 8 + k2_row) * ADIT_SELF_BP + 8 * k2_col;
                float c0v[4] = {0.f, 0.f, 0.f, 0.f}, c1v[4] = {0.f, 0.f, 0.f, 0.f};
                #pragma unroll
                for (int kc = 0; kc < 8; ++kc) {
                    unsigned a0, a1, a2, a3, b0, b1;
                    ldm_x4(qb + q_row * ADIT_SELF_BP + 16 * kc + 8 * q_c16, a0, a1, a2, a3);
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
                s[n8][0] = sS[0][rb];         s[n8][1] = sS[1][rb];
                s[n8][2] = sS[0][rb + 8 * 4]; s[n8][3] = sS[1][rb + 8 * 4];
            }
            // Aligned with torch SDPA: score *= 1/sqrt(head_dim) (fixed 2026-09: the missing factor
            // made p too sharp on real data, disagreeing with the engine/reference semantics -- the
            // old reference was missing it too, so self-consistency hid it)
            #pragma unroll
            for (int n8 = 0; n8 < 2; ++n8)
                #pragma unroll
                for (int i = 0; i < 4; ++i) s[n8][i] *= ADIT_SELF_SATT;
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
                const __nv_bfloat16* vbp = vtile + v_row * ADIT_SELF_BP + dq * 32 + 8 * v_c16;
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

        const int ntiles = (L + ADIT_SELF_TILE - 1) / ADIT_SELF_TILE;
        const uint32_t* kvu = reinterpret_cast<const uint32_t*>(kv_cache);
        const uint32_t* vvu = reinterpret_cast<const uint32_t*>(v_cache);
        auto issue_tile = [&](int stage, int tb) {
            const int cc = tid >> 4;
            const int j = tid & 15;
            const int g = tb * ADIT_SELF_TILE + cc;
            __nv_bfloat16* kdst = &kt[stage][cc * ADIT_SELF_BP + j * 8];
            __nv_bfloat16* vdst = &vt[stage][cc * ADIT_SELF_BP + j * 8];
            if (g < L) {
                const uint32_t* gk = kvu + (size_t)g * (ADIT_SELF_H3 / 2) + h * (ADIT_SELF_BD / 2) + j * 4;
                unsigned dk = (unsigned)__cvta_generic_to_shared(kdst);
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dk), "l"(gk));
                const uint32_t* gv = vvu + (size_t)g * (ADIT_SELF_H3 / 2) + h * (ADIT_SELF_BD / 2) + j * 4;
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
        if (ntiles == 1) asm volatile("cp.async.commit_group;\n" ::);  // empty group pads the count: with a single tile, wait_group 1 waits for the real tile
        else if (ntiles > 1) issue_tile(1, 1);
        for (int tb = 0; tb < ntiles; ++tb) {
            const int stage = tb & 1;
            // The tail iteration uses wait_group 0 (as in the GEMM stretch): padding the count with
            // an empty group is unreliable for the last tile -- a completed empty group occupies no
            // pending slot, so wait_group 1 would let through and read smem that has not landed
            // (fixed 2026-09: the root cause of the tail-tile dirty read / race that reproduced
            // reliably in a debug context).
            if (tb + 1 == ntiles) asm volatile("cp.async.wait_group 0;\n" ::);
            else                  asm volatile("cp.async.wait_group 1;\n" ::);
            __syncthreads();
            flash_tile(kt[stage], vt[stage], min(ADIT_SELF_TILE, L - tb * ADIT_SELF_TILE));
            __syncthreads();
            if (tb + 2 < ntiles) issue_tile(stage, tb + 2);
            else                 asm volatile("cp.async.commit_group;\n" ::);
        }
        flash_tile(ks, vs, 16);
        flash_tile(ks + 16 * ADIT_SELF_BP, vs + 16 * ADIT_SELF_BP, 16);

        // normalize + write attn
        const float inv0 = 1.0f / l0, inv1 = 1.0f / l1;
        unsigned* ou = reinterpret_cast<unsigned*>(attn);
        const int orow = 16 * mh + gid;
        const int colbase = h * ADIT_SELF_BD + dq * 32;
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const int cp = colbase + 8 * nt + 2 * tig;
            const unsigned v0 = (unsigned)bf16_bits(acc[nt][0] * inv0)
                              | ((unsigned)bf16_bits(acc[nt][1] * inv0) << 16);
            const unsigned v1 = (unsigned)bf16_bits(acc[nt][2] * inv1)
                              | ((unsigned)bf16_bits(acc[nt][3] * inv1) << 16);
            ou[(size_t)orow * (ADIT_SELF_H3 / 2) + cp / 2] = v0;
            ou[(size_t)(orow + 8) * (ADIT_SELF_H3 / 2) + cp / 2] = v1;
        }
    }
    ADIT_SELF_GRID_SYNC();    // #4

    // ================= S_PHASE_5: attn row quantization (row-parallel, b<M: one CTA per row) =================
    if (ADIT_SELF_PHASE(5)) {
        const uint32_t* au = reinterpret_cast<const uint32_t*>(attn);
        if (b < ADIT_SELF_M) {
            {
                const int rr = b;
                uint32_t xv[6];                       // staging: the quantization pass does not re-read
                float mx = 0.f;
                #pragma unroll
                for (int i = 0; i < 6; ++i) {
                    uint32_t v = au[(size_t)rr * (ADIT_SELF_H3 / 2) + wid * 192 + i * 32 + lane];
                    xv[i] = v;
                    float a = bf16f((unsigned short)(v & 0xffffu));
                    float bb = bf16f((unsigned short)(v >> 16));
                    mx = fmaxf(mx, fmaxf(fabsf(a), fabsf(bb)));
                }
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    mx = fmaxf(mx, __shfl_down_sync(0xffffffffu, mx, off));
                if (lane == 0) shmax[wid] = mx;
                __syncthreads();
                if (wid == 0 && lane == 0) {
                    float m = 0.f;
                    #pragma unroll
                    for (int i = 0; i < ADIT_SELF_NWARPS; ++i) m = fmaxf(m, shmax[i]);
                    const float sa_v = fmaxf(m / 448.0f, 1e-12f);
                    sa0a[rr] = sa_v;
                    sh_sa = sa_v;
                }
                __syncthreads();
                const float sa_r = sh_sa;
                unsigned char* a8r = a8a + (size_t)rr * ADIT_SELF_H3;
                #pragma unroll
                for (int i = 0; i < 6; ++i) {
                    const int idx = wid * 192 + i * 32 + lane;
                    uint32_t v = xv[i];
                    float a = bf16f((unsigned short)(v & 0xffffu));
                    float bb = bf16f((unsigned short)(v >> 16));
                    unsigned short pair = (unsigned short)fp8_rne(a / sa_r)
                                        | ((unsigned short)fp8_rne(bb / sa_r) << 8);
                    *reinterpret_cast<unsigned short*>(a8r + 2 * idx) = pair;
                }
            }
        }
    }
    ADIT_SELF_GRID_SYNC();    // #5

    // ================= S_PHASE_6: o fp8 GEMM + gate residual epilogue (slab=b<32) =================
    if (b < 32 && ADIT_SELF_PHASE(6)) {
        constexpr int n8 = ADIT_SELF_BN_O / (ADIT_SELF_NWARPS / 2) / 8;   // 1
        constexpr int k_steps = ADIT_SELF_BK_O / 32;          // 4
        constexpr int KQ = ADIT_SELF_BK_O / 16;               // 8
        constexpr int A_CH = ADIT_SELF_M * ADIT_SELF_BK_O / 16;         // 256
        constexpr int W_CH = ADIT_SELF_BN_O * ADIT_SELF_BK_O / 16;      // 256
        uint8_t (&st)[ADIT_SELF_STAGES_O][ADIT_SELF_U_D] = s.gD;
        auto issue = [&](int stage, int kt) {
            uint8_t* base = st[stage];
            if (tid < A_CH) {
                int r = tid / KQ;
                int kq = (tid % KQ) * 16;
                const uint8_t* src = a8a + (size_t)r * ADIT_SELF_H3 + kt * ADIT_SELF_BK_O + kq;
                unsigned sd = (unsigned)__cvta_generic_to_shared(base + r * ADIT_SELF_ARS_O + kq);
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sd), "l"(src));
            }
            if (tid < W_CH) {
                int n = tid / KQ;
                int kq = (tid % KQ) * 16;
                const uint8_t* src = wo + ((size_t)b * ADIT_SELF_BN_O + n) * ADIT_SELF_H3 + kt * ADIT_SELF_BK_O + kq;
                unsigned sd = (unsigned)__cvta_generic_to_shared(base + ADIT_SELF_M * ADIT_SELF_ARS_O + n * ADIT_SELF_BRS_O + kq);
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sd), "l"(src));
            }
            asm volatile("cp.async.commit_group;\n" ::);
        };
        auto issue_empty = []() { asm volatile("cp.async.commit_group;\n" ::); };

        #pragma unroll
        for (int st2 = 0; st2 < ADIT_SELF_STAGES_O - 1; ++st2)
            if (st2 < ADIT_SELF_KtO) issue(st2, st2);

        float acc[n8][4];
        #pragma unroll
        for (int s2 = 0; s2 < n8; ++s2)
            #pragma unroll
            for (int i = 0; i < 4; ++i) acc[s2][i] = 0.f;

        for (int kt = 0; kt < ADIT_SELF_KtO; ++kt) {
            const int stage = kt % ADIT_SELF_STAGES_O;
            if (kt + 1 == ADIT_SELF_KtO) asm volatile("cp.async.wait_group 0;\n" ::);
            else              asm volatile("cp.async.wait_group %0;\n" :: "n"(ADIT_SELF_STAGES_O - 2));
            __syncthreads();
            const uint8_t* As_s = st[stage];
            const uint8_t* Ws_s = st[stage] + ADIT_SELF_M * ADIT_SELF_ARS_O;
            #pragma unroll
            for (int kk = 0; kk < k_steps; ++kk) {
                const int k_off = kk * 32;
                const int arow0 = m_slice * 16 + lane_g;
                const int arow1 = arow0 + 8;
                uint32_t a0 = *reinterpret_cast<const uint32_t*>(As_s + (size_t)arow0 * ADIT_SELF_ARS_O + k_off + lane_t * 4);
                uint32_t a1 = *reinterpret_cast<const uint32_t*>(As_s + (size_t)arow1 * ADIT_SELF_ARS_O + k_off + lane_t * 4);
                uint32_t a2 = *reinterpret_cast<const uint32_t*>(As_s + (size_t)arow0 * ADIT_SELF_ARS_O + k_off + lane_t * 4 + 16);
                uint32_t a3 = *reinterpret_cast<const uint32_t*>(As_s + (size_t)arow1 * ADIT_SELF_ARS_O + k_off + lane_t * 4 + 16);
                #pragma unroll
                for (int s2 = 0; s2 < n8; ++s2) {
                    const int bn = n_grp * (ADIT_SELF_BN_O / (ADIT_SELF_NWARPS / 2)) + s2 * 8 + lane_g;
                    const uint32_t* bp = reinterpret_cast<const uint32_t*>(Ws_s + (size_t)bn * ADIT_SELF_BRS_O + k_off);
                    uint32_t b0w = bp[lane_t];
                    uint32_t b1w = bp[lane_t + 4];
                    asm volatile(
                        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                        : "+f"(acc[s2][0]), "+f"(acc[s2][1]), "+f"(acc[s2][2]), "+f"(acc[s2][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0w), "r"(b1w));
                }
            }
            const int nkt = kt + ADIT_SELF_STAGES_O - 1;
            if (nkt < ADIT_SELF_KtO) issue((kt - 1 + ADIT_SELF_STAGES_O) % ADIT_SELF_STAGES_O, nkt);
            else           issue_empty();
        }

        // epilogue: out = bf16(x + bf16(gate_msa * bf16(acc*sa0a*swo + bo))) (no re-quantization)
        const int r0 = m_slice * 16 + lane_g;
        const int r1 = r0 + 8;
        const float sa0_r0 = sa0a[r0], sa0_r1 = sa0a[r1];
        const uint32_t* xu = reinterpret_cast<const uint32_t*>(x_in);
        const uint32_t* gu = reinterpret_cast<const uint32_t*>(gate_msa);
        const uint32_t* bu = reinterpret_cast<const uint32_t*>(bo);
        #pragma unroll
        for (int s2 = 0; s2 < n8; ++s2) {
            const int col0 = b * ADIT_SELF_BN_O + n_grp * (ADIT_SELF_BN_O / (ADIT_SELF_NWARPS / 2)) + s2 * 8 + lane_t * 2;
            const int col0h = col0 / 2;
            const float swc0 = swo[col0], swc1 = swo[col0 + 1];
            uint32_t bcp = bu[col0h];
            float bc0 = bf16f((unsigned short)(bcp & 0xffffu));
            float bc1 = bf16f((unsigned short)(bcp >> 16));
            uint32_t gcp = gu[col0h];
            float gc0 = bf16f((unsigned short)(gcp & 0xffffu));
            float gc1 = bf16f((unsigned short)(gcp >> 16));
            uint32_t xp0 = xu[(size_t)r0 * (ADIT_SELF_H / 2) + col0h];
            uint32_t xp1 = xu[(size_t)r1 * (ADIT_SELF_H / 2) + col0h];
            float xa0 = bf16f((unsigned short)(xp0 & 0xffffu));
            float xa1 = bf16f((unsigned short)(xp0 >> 16));
            float xb0 = bf16f((unsigned short)(xp1 & 0xffffu));
            float xb1 = bf16f((unsigned short)(xp1 >> 16));
            unsigned short u0 = bf16_bits(acc[s2][0] * sa0_r0 * swc0 + bc0);
            unsigned short u1 = bf16_bits(acc[s2][1] * sa0_r0 * swc1 + bc1);
            unsigned short t0 = bf16_bits(gc0 * bf16f(u0));
            unsigned short t1 = bf16_bits(gc1 * bf16f(u1));
            unsigned short y0 = bf16_bits(xa0 + bf16f(t0));
            unsigned short y1 = bf16_bits(xa1 + bf16f(t1));
            *reinterpret_cast<unsigned*>(out + ((size_t)r0 * ADIT_SELF_H + col0) * 2) =
                (uint32_t)y0 | ((uint32_t)y1 << 16);
            unsigned short u2 = bf16_bits(acc[s2][2] * sa0_r1 * swc0 + bc0);
            unsigned short u3 = bf16_bits(acc[s2][3] * sa0_r1 * swc1 + bc1);
            unsigned short t2 = bf16_bits(gc0 * bf16f(u2));
            unsigned short t3 = bf16_bits(gc1 * bf16f(u3));
            unsigned short y2 = bf16_bits(xb0 + bf16f(t2));
            unsigned short y3 = bf16_bits(xb1 + bf16f(t3));
            *reinterpret_cast<unsigned*>(out + ((size_t)r1 * ADIT_SELF_H + col0) * 2) =
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
// This kernel's grid is set by the **problem size** (GRID = N/BN_QKV = 96); the cooperative form
// only additionally requires the whole grid to be co-resident.

// ---- Geometry self-report (shape constants + GEMM template parameters; template parameters and instantiation share one macro set) ----
extern "C" const char* adit_attn_self_geom() {
    static const std::string s = fwam_geom_join({
        fwam_shapes({{"M", ADIT_SELF_M}, {"H", ADIT_SELF_H}, {"N", ADIT_SELF_N}, {"H3", ADIT_SELF_H3}, {"HEADS", ADIT_SELF_HEADS}, {"BN_QKV", ADIT_SELF_BN_QKV}, {"BN_O", ADIT_SELF_BN_O}, {"BK_Q", ADIT_SELF_BK_Q}, {"BK_O", ADIT_SELF_BK_O}, {"STAGES_Q", ADIT_SELF_STAGES_Q}, {"STAGES_O", ADIT_SELF_STAGES_O}, {"NT", ADIT_SELF_NTHREADS}, {"GRID", ADIT_SELF_GRID}}),
    });
    return s.c_str();
}

extern "C" int adit_attn_self_grid() { return ADIT_SELF_GRID; }
extern "C" int adit_attn_self_coop_grid() {
    static const int cap = coop_grid((const void*)adit_attn_self_kernel, ADIT_SELF_NTHREADS, 0, ADIT_SELF_GRID,
                                     "adit_attn_self");
    (void)cap;          // capacity check only; the actual launch still uses the problem-size GRID
    return ADIT_SELF_GRID;
}

extern "C" void adit_attn_self_cuda(
    const void* x_in, const void* shift_msa, const void* scale_msa,
    const void* wqkv, const float* swqkv, const void* bqkv,
    const void* kv_cache, const void* v_cache,
    const void* wnq, const void* wnk,
    const void* cos_t, const void* sin_t,
    const void* gate_msa, const void* wo, const float* swo, const void* bo,
    void* out, void* qkv, void* attn,
    void* a8x, float* sa0x, void* a8a, float* sa0a, float* rstd_scratch,
    int L, int only_phase, cudaStream_t stream)
{
    dim3 grid(ADIT_SELF_GRID), block(ADIT_SELF_NTHREADS);
    void* args[] = {
        (void*)&x_in, (void*)&shift_msa, (void*)&scale_msa,
        (void*)&wqkv, (void*)&swqkv, (void*)&bqkv,
        (void*)&kv_cache, (void*)&v_cache,
        (void*)&wnq, (void*)&wnk,
        (void*)&cos_t, (void*)&sin_t,
        (void*)&gate_msa, (void*)&wo, (void*)&swo, (void*)&bo,
        (void*)&out, (void*)&qkv, (void*)&attn,
        (void*)&a8x, (void*)&sa0x, (void*)&a8a, (void*)&sa0a, (void*)&rstd_scratch,
        (void*)&L, (void*)&only_phase,
    };
    cudaError_t e;
    if (only_phase < 0) {
        // Device capacity check: grid = GRID (= N/BN_QKV) is decided by the problem size and phases
        // claim by slab=b, so grid<GRID undercounts; the cooperative kernel additionally requires
        // the whole grid to be co-resident. Report it clearly here when the capacity falls short,
        // rather than letting the driver throw a bare cudaErrorCooperativeLaunchTooLarge at launch.
        static const int cap = coop_grid((const void*)adit_attn_self_kernel, ADIT_SELF_NTHREADS, 0, ADIT_SELF_GRID,
                                         "adit_attn_self");
        (void)cap;
        e = cudaLaunchCooperativeKernel((void*)adit_attn_self_kernel, grid, block,
                                        args, 0, stream);
    } else {
        // Non-cooperative split phases: the grid must stay GRID (decided by the tile geometry); a
        // plain launch may run in multiple waves.
        e = cudaLaunchKernel((void*)adit_attn_self_kernel, grid, block, args, 0, stream);
    }
    check(e, only_phase < 0 ? "cooperative launch adit_attn_self"
                            : "split launch adit_attn_self");
}
