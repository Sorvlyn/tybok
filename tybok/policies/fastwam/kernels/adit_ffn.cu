// Single-kernel fusion of the action FFN
// (norm2+modulate+quantization → up GEMM+GELU → quantization → down GEMM + gate residual).
//
// Flow (GRID0=64 slab CTAs per instance, grid = 64 × B):
//   raw1 is zeroed
//   F_PHASE_1 [slab<M] norm2 + modulate + row quantization → a8buf/sa0buf (row-parallel: slab = row index)
//   F_PHASE_2 fp8 GEMM up + dequant/GELU epilogue → gbuf + raw1(amax)
//   F_PHASE_3 gbuf column-slice quantization → a8gbuf
//   F_PHASE_4 fp8 GEMM down + gate residual → out (only the first SLAB1 slabs are used)
//   phases are separated by grid.sync.
//
// Numerics: bf16 RN / fp8 software single RNE / bias added in fp32 after dequant.
// GELU goes through `gelu_tanh` in common.h = torch's nn.GELU(approximate='tanh')
// (0.5u(1+tanh(c1(u+c2u³))), c1=sqrt(2/pi)), the same convention as video/text/lerobot.
// History: before 2026-09-14 it was written u*(c1+c2u²) (the cubic term was missing the c1 factor,
// up to 6.3e-3 off from torch); fixed, do not revert.

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

constexpr int ADIT_FFN_M = 32;
constexpr int ADIT_FFN_H = 1024;
constexpr int ADIT_FFN_F = 4096;
constexpr int ADIT_FFN_BNU = 64;      // 128 -> 64: the U phase's slab count goes 32 -> 64 (= SM count)
constexpr int ADIT_FFN_BN1 = 32;
constexpr float ADIT_FFN_EPS = 1e-6f;

constexpr int ADIT_FFN_BKU = 128;                   // up phase K shard: 64 -> 128
constexpr int ADIT_FFN_BKD = 128;                   // down phase K shard
constexpr int ADIT_FFN_STAGES = 3;
constexpr int ADIT_FFN_NTHREADS = 256;
constexpr int ADIT_FFN_NWARPS = ADIT_FFN_NTHREADS / 32;

static_assert(ADIT_FFN_F / ADIT_FFN_BNU >= ADIT_FFN_H / ADIT_FFN_BN1, "grid must fit every slab of both phases");
constexpr int ADIT_FFN_GRID0 = ADIT_FFN_F / ADIT_FFN_BNU;             // 64 slab-CTAs / instance (the U phase is full)
constexpr int ADIT_FFN_SLAB1 = ADIT_FFN_H / ADIT_FFN_BN1;             // 32: the D phase only uses the first SLAB1 slabs

// smem: one union region sized by the U-phase stage layout (M*ARS_U + BNU*BRS_U = 12.8KB)
constexpr int ADIT_FFN_ARS_U = ADIT_FFN_BKU + 16;            // 144
constexpr int ADIT_FFN_BRS_U = ADIT_FFN_BKU + 16;
constexpr int ADIT_FFN_ARS_D = ADIT_FFN_BKD + 16;            // 144
constexpr int ADIT_FFN_BRS_D = ADIT_FFN_BKD + 16;
constexpr int ADIT_FFN_U_STAGE = ADIT_FFN_M * ADIT_FFN_ARS_U + ADIT_FFN_BNU * ADIT_FFN_BRS_U;   // 13824
static_assert(ADIT_FFN_STAGES * ADIT_FFN_U_STAGE <= 48 * 1024, "smem over budget");
// The D phase's per-stage usage (As 32x144 + W1 32x144 = 9216) must fit into U_STAGE
static_assert(ADIT_FFN_M * ADIT_FFN_ARS_D + ADIT_FFN_BN1 * ADIT_FFN_BRS_D <= ADIT_FFN_U_STAGE, "D stage does not fit in the union region");

// Device helpers: bf16f/gelu_tanh → common.h; bf16_bits/fp8_rne/mod_bf16 → dit_common.h; bf16_pair → adit_common.h.
// Non-cooperative fallback (only_phase >= 0): each phase is launched separately and visibility
// between phases comes from stream ordering. The split path builds no grid barrier, so a plain
// `cudaLaunchKernel` can be used (no co-residency requirement).
#define ADIT_FFN_PHASE(k)     (only_phase < 0 || only_phase == (k))
#define ADIT_FFN_GRID_SYNC()  do { if (only_phase < 0) { __threadfence(); cg::this_grid().sync(); } } while (0)

// ------------------------------------------------------------------ //
__global__ void __launch_bounds__(ADIT_FFN_NTHREADS)
adit_ffn_kernel(const uint8_t* __restrict__ x_in,       // bf16 [B, M, H]
                 const uint8_t* __restrict__ shift_mlp,  // bf16 [B, H]
                 const uint8_t* __restrict__ scale_mlp,  // bf16 [B, H]
                 const uint8_t* __restrict__ gate_mlp,   // bf16 [B, H]
                 const uint8_t* __restrict__ w0,         // fp8 [F,H]
                 const float* __restrict__ sw0,          // fp32 [F]
                 const uint8_t* __restrict__ b0,         // bf16 [F]
                 const uint8_t* __restrict__ w1,         // fp8 [H,F]
                 const float* __restrict__ sw1,          // fp32 [H]
                 const uint8_t* __restrict__ b1,         // bf16 [H]
                 uint8_t* __restrict__ out,              // bf16 [B, M, H]
                 uint8_t* __restrict__ a8buf,            // fp8 [B, M, H]
                 float* __restrict__ sa0buf,             // fp32 [B, M]
                 uint8_t* __restrict__ gbuf,             // bf16 [B, M, F]
                 uint8_t* __restrict__ a8gbuf,           // fp8 [B, M, F]
                 uint32_t* __restrict__ raw1,            // [B*M]
                 int B, int only_phase)         // non-cooperative fallback: >=0 runs that phase only; <0 = the cooperative full pipeline
{
    const int inst = blockIdx.x / ADIT_FFN_GRID0;
    const int slab = blockIdx.x - inst * ADIT_FFN_GRID0;
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    const int lane_g = lane >> 2;
    const int lane_t = lane & 3;
    const int m_slice = wid / (ADIT_FFN_NWARPS / 2);
    const int n_grp = wid % (ADIT_FFN_NWARPS / 2);

    constexpr int n8u = ADIT_FFN_BNU / (ADIT_FFN_NWARPS / 2) / 8;   // 2
    constexpr int n8d = ADIT_FFN_BN1 / (ADIT_FFN_NWARPS / 2) / 8;   // 1
    constexpr int k_steps_u = ADIT_FFN_BKU / 32;           // 4
    constexpr int k_steps_d = ADIT_FFN_BKD / 32;           // 4

    __shared__ __align__(16) uint8_t smem[ADIT_FFN_STAGES][ADIT_FFN_U_STAGE];
    __shared__ float ph_mean, ph_rstd;       // F_PHASE_1 row statistic broadcast (slab<M)
    __shared__ float ph_red[ADIT_FFN_NWARPS];         // F_PHASE_1 cross-warp amax reduction
    __shared__ float ph_sa;                  // F_PHASE_1 scale broadcast

    const uint8_t* xi = x_in + (size_t)inst * (ADIT_FFN_M * ADIT_FFN_H * 2);
    const uint8_t* sci = scale_mlp + (size_t)inst * ADIT_FFN_H * 2;
    const uint8_t* shi = shift_mlp + (size_t)inst * ADIT_FFN_H * 2;
    const uint8_t* gti = gate_mlp + (size_t)inst * ADIT_FFN_H * 2;
    uint8_t* a8i = a8buf + (size_t)inst * (ADIT_FFN_M * ADIT_FFN_H);
    float* sa0i = sa0buf + (size_t)inst * ADIT_FFN_M;
    uint8_t* gi = gbuf + (size_t)inst * (ADIT_FFN_M * ADIT_FFN_F * 2);
    uint8_t* a8gi = a8gbuf + (size_t)inst * (ADIT_FFN_M * ADIT_FFN_F);
    uint8_t* oi = out + (size_t)inst * (ADIT_FFN_M * ADIT_FFN_H * 2);
    uint32_t* rawi = raw1 + (size_t)inst * ADIT_FFN_M;

    // ---------------- F_PHASE_1: row-parallel (one slab-CTA per row); writes a8buf + sa0buf ----------------
    if (ADIT_FFN_PHASE(1)) {
    // raw1 zeroing (F_PHASE_2's EPI atomicMax needs a zero start): folded into the F_PHASE_1 phase
    // and published by the barrier after it; on the split path it runs once in the phase-1 launch,
    // after which the kernel boundary orders it before F_PHASE_2.
    if (blockIdx.x == 0) {
        for (int r = 0; r < ADIT_FFN_M * B; ++r) raw1[r] = 0u;
    }
    if (slab < ADIT_FFN_M) {                              // row-parallel: one slab-CTA per row
        const int r = slab;
        const uint32_t* xu = reinterpret_cast<const uint32_t*>(xi);
        const uint32_t* scu = reinterpret_cast<const uint32_t*>(sci);
        const uint32_t* shu = reinterpret_cast<const uint32_t*>(shi);

        // ---- pass 1: row statistics (warp0; bit-exact with the original) ----
        if (wid == 0) {
            float ss = 0.f, qq = 0.f;
            #pragma unroll
            for (int i = lane; i < ADIT_FFN_H / 2; i += 32) {
                uint32_t wd = xu[(size_t)r * (ADIT_FFN_H / 2) + i];
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
                float mean = ss / (float)ADIT_FFN_H;
                float var = fmaxf(qq / (float)ADIT_FFN_H - mean * mean, 0.f);
                ph_mean = mean;
                ph_rstd = rsqrtf(var + ADIT_FFN_EPS);
            }
        }
        __syncthreads();
        const float mean_r = ph_mean, rstd_r = ph_rstd;

        // ---- pass 2: modulate + row amax ----
        float mxl = 0.f;
        #pragma unroll 4
        for (int i = tid; i < ADIT_FFN_H / 2; i += ADIT_FFN_NTHREADS) {
            uint32_t sc01 = scu[i], sh01 = shu[i];
            float sc_a = bf16f((unsigned short)(sc01 & 0xffffu));
            float sc_b = bf16f((unsigned short)(sc01 >> 16));
            float sh_a = bf16f((unsigned short)(sh01 & 0xffffu));
            float sh_b = bf16f((unsigned short)(sh01 >> 16));
            uint32_t wd = xu[(size_t)r * (ADIT_FFN_H / 2) + i];
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
            for (int i = 0; i < ADIT_FFN_NWARPS; ++i) m = fmaxf(m, ph_red[i]);
            const float sa_v = fmaxf(m / 448.0f, 1e-12f);
            sa0i[r] = sa_v;
            ph_sa = sa_v;
        }
        __syncthreads();
        const float sa_r = ph_sa;

        // ---- pass 3: quantization ----
        unsigned char* a8r = a8i + (size_t)r * ADIT_FFN_H;
        #pragma unroll 4
        for (int i = tid; i < ADIT_FFN_H / 2; i += ADIT_FFN_NTHREADS) {
            uint32_t sc01 = scu[i], sh01 = shu[i];
            float sc_a = bf16f((unsigned short)(sc01 & 0xffffu));
            float sc_b = bf16f((unsigned short)(sc01 >> 16));
            float sh_a = bf16f((unsigned short)(sh01 & 0xffffu));
            float sh_b = bf16f((unsigned short)(sh01 >> 16));
            uint32_t wd = xu[(size_t)r * (ADIT_FFN_H / 2) + i];
            float xa = bf16f((unsigned short)(wd & 0xffffu));
            float xb = bf16f((unsigned short)(wd >> 16));
            float ma = mod_bf16(xa, mean_r, rstd_r, sc_a, sh_a);
            float mb = mod_bf16(xb, mean_r, rstd_r, sc_b, sh_b);
            unsigned short pair = (unsigned short)fp8_rne(ma / sa_r) |
                                  ((unsigned short)fp8_rne(mb / sa_r) << 8);
            *reinterpret_cast<unsigned short*>(a8r + 2 * i) = pair;
        }
    }
    }

    ADIT_FFN_GRID_SYNC();    // a8buf / sa0buf visible grid-wide

    // Row indices shared by the F_PHASE_2 epilogue and F_PHASE_4 + the empty-commit lambda (function
    // scope originally; wrapping them in the if block would shadow them).
    const int r0 = m_slice * 16 + lane_g;
    const int r1 = r0 + 8;
    auto issue_empty = []() { asm volatile("cp.async.commit_group;\n" ::); };

    // ---------------- F_PHASE_2: fp8 GEMM up (BK=128) ----------------
    if (ADIT_FFN_PHASE(2)) {
    constexpr int KQ_U = ADIT_FFN_BKU / 16;                 // 8
    constexpr int A_U = ADIT_FFN_M * ADIT_FFN_BKU / 16;              // 256
    constexpr int W0_U = ADIT_FFN_BNU * ADIT_FFN_BKU / 16;           // 512
    constexpr int Kt0 = ADIT_FFN_H / ADIT_FFN_BKU;                   // 8
    auto issue_u = [&](int stage, int kt) {
        uint8_t* base = smem[stage];
        if (tid < A_U) {
            int r = tid / KQ_U;
            int kq = (tid % KQ_U) * 16;
            const uint8_t* src = a8i + (size_t)r * ADIT_FFN_H + kt * ADIT_FFN_BKU + kq;
            unsigned sd = (unsigned)__cvta_generic_to_shared(base + r * ADIT_FFN_ARS_U + kq);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sd), "l"(src));
        }
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            int chunk = tid + j * ADIT_FFN_NTHREADS;   // < 512
            int n = chunk / KQ_U;
            int kq = (chunk % KQ_U) * 16;
            const uint8_t* src = w0 + ((size_t)slab * ADIT_FFN_BNU + n) * ADIT_FFN_H + kt * ADIT_FFN_BKU + kq;
            unsigned sd = (unsigned)__cvta_generic_to_shared(base + ADIT_FFN_M * ADIT_FFN_ARS_U + n * ADIT_FFN_BRS_U + kq);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sd), "l"(src));
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };

    #pragma unroll
    for (int s = 0; s < ADIT_FFN_STAGES - 1; ++s)
        if (s < Kt0) issue_u(s, s);

    float acc0[n8u][4];
    #pragma unroll
    for (int s = 0; s < n8u; ++s)
        #pragma unroll
        for (int i = 0; i < 4; ++i) acc0[s][i] = 0.f;

    for (int kt = 0; kt < Kt0; ++kt) {
        const int stage = kt % ADIT_FFN_STAGES;
        if (kt + 1 == Kt0) asm volatile("cp.async.wait_group 0;\n" ::);
        else               asm volatile("cp.async.wait_group %0;\n" :: "n"(ADIT_FFN_STAGES - 2));
        __syncthreads();

        const uint8_t* As_s = smem[stage];
        const uint8_t* Ws_s = smem[stage] + ADIT_FFN_M * ADIT_FFN_ARS_U;
        #pragma unroll
        for (int kk = 0; kk < k_steps_u; ++kk) {
            const int k_off = kk * 32;
            const int arow0 = m_slice * 16 + lane_g;
            const int arow1 = arow0 + 8;
            uint32_t a0 = *reinterpret_cast<const uint32_t*>(As_s + (size_t)arow0 * ADIT_FFN_ARS_U + k_off + lane_t * 4);
            uint32_t a1 = *reinterpret_cast<const uint32_t*>(As_s + (size_t)arow1 * ADIT_FFN_ARS_U + k_off + lane_t * 4);
            uint32_t a2 = *reinterpret_cast<const uint32_t*>(As_s + (size_t)arow0 * ADIT_FFN_ARS_U + k_off + lane_t * 4 + 16);
            uint32_t a3 = *reinterpret_cast<const uint32_t*>(As_s + (size_t)arow1 * ADIT_FFN_ARS_U + k_off + lane_t * 4 + 16);
            #pragma unroll
            for (int s = 0; s < n8u; ++s) {
                const int bn = n_grp * (ADIT_FFN_BNU / (ADIT_FFN_NWARPS / 2)) + s * 8 + lane_g;
                const uint32_t* bp = reinterpret_cast<const uint32_t*>(Ws_s + (size_t)bn * ADIT_FFN_BRS_U + k_off);
                uint32_t b0w = bp[lane_t];
                uint32_t b1w = bp[lane_t + 4];
                asm volatile(
                    "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(acc0[s][0]), "+f"(acc0[s][1]), "+f"(acc0[s][2]), "+f"(acc0[s][3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0w), "r"(b1w));
            }
        }
        const int nkt = kt + ADIT_FFN_STAGES - 1;
        if (nkt < Kt0) issue_u((kt - 1 + ADIT_FFN_STAGES) % ADIT_FFN_STAGES, nkt);
        else           issue_empty();
    }

    // ---------------- F_PHASE_2 epilogue: dequant(sa0buf)+GELU -> gbuf; row amax -> raw1 ----------------
    const float sa0_r0 = sa0i[r0], sa0_r1 = sa0i[r1];
    float rowmax_r0 = 0.f, rowmax_r1 = 0.f;
    #pragma unroll
    for (int s = 0; s < n8u; ++s) {
        const int col0 = slab * ADIT_FFN_BNU + n_grp * (ADIT_FFN_BNU / (ADIT_FFN_NWARPS / 2)) + s * 8 + lane_t * 2;
        float swc0 = sw0[col0], swc1 = sw0[col0 + 1];
        uint32_t bc = *reinterpret_cast<const uint32_t*>(b0 + (size_t)col0 * 2);
        float bc0 = bf16f((unsigned short)(bc & 0xffffu)), bc1 = bf16f((unsigned short)(bc >> 16));
        float u0 = bf16f(bf16_bits(acc0[s][0] * sa0_r0 * swc0 + bc0));
        float u1 = bf16f(bf16_bits(acc0[s][1] * sa0_r0 * swc1 + bc1));
        float g0 = bf16f(bf16_bits(gelu_tanh(u0)));
        float g1 = bf16f(bf16_bits(gelu_tanh(u1)));
        rowmax_r0 = fmaxf(rowmax_r0, fmaxf(fabsf(g0), fabsf(g1)));
        *reinterpret_cast<uint32_t*>(gi + ((size_t)r0 * ADIT_FFN_F + col0) * 2) = bf16_pair(g0, g1);
        float u2 = bf16f(bf16_bits(acc0[s][2] * sa0_r1 * swc0 + bc0));
        float u3 = bf16f(bf16_bits(acc0[s][3] * sa0_r1 * swc1 + bc1));
        float g2 = bf16f(bf16_bits(gelu_tanh(u2)));
        float g3 = bf16f(bf16_bits(gelu_tanh(u3)));
        rowmax_r1 = fmaxf(rowmax_r1, fmaxf(fabsf(g2), fabsf(g3)));
        *reinterpret_cast<uint32_t*>(gi + ((size_t)r1 * ADIT_FFN_F + col0) * 2) = bf16_pair(g2, g3);
    }
    if (rowmax_r0 > 0.f) atomicMax(rawi + r0, __float_as_uint(rowmax_r0));
    if (rowmax_r1 > 0.f) atomicMax(rawi + r1, __float_as_uint(rowmax_r1));
    }

    ADIT_FFN_GRID_SYNC();

    // ---------------- F_PHASE_3: quantize its own gbuf column slice -> a8gbuf ----------------
    if (ADIT_FFN_PHASE(3)) {
        const uint32_t col_lo = slab * ADIT_FFN_BNU, col_hi = col_lo + ADIT_FFN_BNU;
        for (int idx = tid; idx < ADIT_FFN_M * ADIT_FFN_BNU; idx += ADIT_FFN_NTHREADS) {
            const int row = idx / ADIT_FFN_BNU;
            const int col = col_lo + idx % ADIT_FFN_BNU;
            unsigned short gw = *reinterpret_cast<const unsigned short*>(gi + ((size_t)row * ADIT_FFN_F + col) * 2);
            float gv = bf16f(gw);
            float sa1 = fmaxf(__uint_as_float(rawi[row]) / 448.0f, 1e-12f);
            a8gi[(size_t)row * ADIT_FFN_F + col] = fp8_rne(gv / sa1);
        }
    }

    ADIT_FFN_GRID_SYNC();

    // ---------------- F_PHASE_4: fp8 GEMM down (BK=128) ----------------
    if (ADIT_FFN_PHASE(4)) {
    constexpr int KQ_D = ADIT_FFN_BKD / 16;             // 8
    constexpr int A_D = ADIT_FFN_M * ADIT_FFN_BKD / 16;          // 256
    constexpr int W1_D = ADIT_FFN_BN1 * ADIT_FFN_BKD / 16;       // 256
    constexpr int Kt1 = ADIT_FFN_F / ADIT_FFN_BKD;               // 32
    auto issue_d = [&](int stage, int kt) {
        if (slab >= ADIT_FFN_SLAB1) return;                 // spare slabs do not take part in the D phase
        uint8_t* base = smem[stage];
        if (tid < A_D) {
            int r = tid / KQ_D;
            int kq = (tid % KQ_D) * 16;
            const uint8_t* src = a8gi + (size_t)r * ADIT_FFN_F + kt * ADIT_FFN_BKD + kq;
            unsigned sd = (unsigned)__cvta_generic_to_shared(base + r * ADIT_FFN_ARS_D + kq);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sd), "l"(src));
        }
        if (tid < W1_D) {
            int n = tid / KQ_D;
            int kq = (tid % KQ_D) * 16;
            const uint8_t* src = w1 + ((size_t)slab * ADIT_FFN_BN1 + n) * ADIT_FFN_F + kt * ADIT_FFN_BKD + kq;
            unsigned sd = (unsigned)__cvta_generic_to_shared(base + ADIT_FFN_M * ADIT_FFN_ARS_D + n * ADIT_FFN_BRS_D + kq);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sd), "l"(src));
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };

    const bool d_active = slab < ADIT_FFN_SLAB1;        // D phase: only the first 32 slabs do any work
    #pragma unroll
    for (int s = 0; s < ADIT_FFN_STAGES - 1; ++s)
        if (d_active && s < Kt1) issue_d(s, s);

    float acc1[n8d][4];
    #pragma unroll
    for (int s = 0; s < n8d; ++s)
        #pragma unroll
        for (int i = 0; i < 4; ++i) acc1[s][i] = 0.f;

    const float sa1r0 = fmaxf(__uint_as_float(rawi[r0]) / 448.0f, 1e-12f);
    const float sa1r1 = fmaxf(__uint_as_float(rawi[r1]) / 448.0f, 1e-12f);

    for (int kt = 0; d_active && kt < Kt1; ++kt) {
        const int stage = kt % ADIT_FFN_STAGES;
        if (kt + 1 == Kt1) asm volatile("cp.async.wait_group 0;\n" ::);
        else               asm volatile("cp.async.wait_group %0;\n" :: "n"(ADIT_FFN_STAGES - 2));
        __syncthreads();

        const uint8_t* As_s = smem[stage];
        const uint8_t* Ws_s = smem[stage] + ADIT_FFN_M * ADIT_FFN_ARS_D;
        #pragma unroll
        for (int kk = 0; kk < k_steps_d; ++kk) {
            const int k_off = kk * 32;
            const int arow0 = m_slice * 16 + lane_g;
            const int arow1 = arow0 + 8;
            uint32_t a0 = *reinterpret_cast<const uint32_t*>(As_s + (size_t)arow0 * ADIT_FFN_ARS_D + k_off + lane_t * 4);
            uint32_t a1 = *reinterpret_cast<const uint32_t*>(As_s + (size_t)arow1 * ADIT_FFN_ARS_D + k_off + lane_t * 4);
            uint32_t a2 = *reinterpret_cast<const uint32_t*>(As_s + (size_t)arow0 * ADIT_FFN_ARS_D + k_off + lane_t * 4 + 16);
            uint32_t a3 = *reinterpret_cast<const uint32_t*>(As_s + (size_t)arow1 * ADIT_FFN_ARS_D + k_off + lane_t * 4 + 16);
            #pragma unroll
            for (int s = 0; s < n8d; ++s) {
                const int bn = n_grp * (ADIT_FFN_BN1 / (ADIT_FFN_NWARPS / 2)) + lane_g;
                const uint32_t* bp = reinterpret_cast<const uint32_t*>(Ws_s + (size_t)bn * ADIT_FFN_BRS_D + k_off);
                uint32_t b0w = bp[lane_t];
                uint32_t b1w = bp[lane_t + 4];
                asm volatile(
                    "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(acc1[s][0]), "+f"(acc1[s][1]), "+f"(acc1[s][2]), "+f"(acc1[s][3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0w), "r"(b1w));
            }
        }
        const int nkt = kt + ADIT_FFN_STAGES - 1;
        if (nkt < Kt1) issue_d((kt - 1 + ADIT_FFN_STAGES) % ADIT_FFN_STAGES, nkt);
        else           issue_empty();
    }

    // ---------------- F_PHASE_4 epilogue: gate*z + x_in ----------------
    const uint32_t* xu_out = reinterpret_cast<const uint32_t*>(xi);
    #pragma unroll
    for (int s = 0; d_active && s < n8d; ++s) {
        const int col0 = slab * ADIT_FFN_BN1 + n_grp * (ADIT_FFN_BN1 / (ADIT_FFN_NWARPS / 2)) + lane_t * 2;
        float swc0 = sw1[col0], swc1 = sw1[col0 + 1];
        uint32_t bc = *reinterpret_cast<const uint32_t*>(b1 + (size_t)col0 * 2);
        float bc0 = bf16f((unsigned short)(bc & 0xffffu)), bc1 = bf16f((unsigned short)(bc >> 16));
        uint32_t gc = *reinterpret_cast<const uint32_t*>(gti + (size_t)col0 * 2);
        float gc0 = bf16f((unsigned short)(gc & 0xffffu)), gc1 = bf16f((unsigned short)(gc >> 16));
        uint32_t x0 = xu_out[(size_t)r0 * (ADIT_FFN_H / 2) + col0 / 2];
        float xa0 = bf16f((unsigned short)(x0 & 0xffffu)), xa1 = bf16f((unsigned short)(x0 >> 16));
        float z0 = bf16f(bf16_bits(acc1[s][0] * sa1r0 * swc0 + bc0));
        float z1 = bf16f(bf16_bits(acc1[s][1] * sa1r0 * swc1 + bc1));
        float o0 = bf16f(bf16_bits(bf16f(bf16_bits(z0 * gc0)) + xa0));
        float o1 = bf16f(bf16_bits(bf16f(bf16_bits(z1 * gc1)) + xa1));
        *reinterpret_cast<uint32_t*>(oi + ((size_t)r0 * ADIT_FFN_H + col0) * 2) = bf16_pair(o0, o1);
        uint32_t x1 = xu_out[(size_t)r1 * (ADIT_FFN_H / 2) + col0 / 2];
        float xb0 = bf16f((unsigned short)(x1 & 0xffffu)), xb1 = bf16f((unsigned short)(x1 >> 16));
        float z2 = bf16f(bf16_bits(acc1[s][2] * sa1r1 * swc0 + bc0));
        float z3 = bf16f(bf16_bits(acc1[s][3] * sa1r1 * swc1 + bc1));
        float o2 = bf16f(bf16_bits(bf16f(bf16_bits(z2 * gc0)) + xb0));
        float o3 = bf16f(bf16_bits(bf16f(bf16_bits(z3 * gc1)) + xb1));
        *reinterpret_cast<uint32_t*>(oi + ((size_t)r1 * ADIT_FFN_H + col0) * 2) = bf16_pair(o2, o3);
    }
    }
}

// ------------------------------------------------------------------ //
// host wrapper: cooperative launch
// ------------------------------------------------------------------ //
static void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        // Raise instead of exit(1): exit would kill the worker process outright, leaving the caller no handleable error.
        coop_fail(what, std::string("CUDA error: ") + cudaGetErrorString(e));
    }
}

// ---- Queries for registry/probe (pure query: raises if it does not fit; the caller catches it via try/except for the reason) ----
// grid = GRID0 × B (B = the number of instances, at runtime); the cooperative form requires the
// whole grid to be co-resident.
extern "C" int adit_ffn_grid(int B) { return ADIT_FFN_GRID0 * B; }
extern "C" int adit_ffn_coop_grid(int B) {
    // Not cached: the grid grows linearly with B, and caching would make a query with a different B use the wrong capacity.
    const int cap = coop_grid((const void*)adit_ffn_kernel, ADIT_FFN_NTHREADS, 0, ADIT_FFN_GRID0,
                              "adit_ffn");
    if (cap < ADIT_FFN_GRID0 * B) coop_fail("adit_ffn",
        "device capacity " + std::to_string(cap) + " CTA < grid " + std::to_string(ADIT_FFN_GRID0 * B)
        + " (=GRID0 " + std::to_string(ADIT_FFN_GRID0) + " × B " + std::to_string(B) + ")");
    return ADIT_FFN_GRID0 * B;
}


// ---- Geometry self-report (shape constants + GEMM template parameters; template parameters and instantiation share one macro set) ----
extern "C" const char* adit_ffn_geom() {
    static const std::string s = fwam_geom_join({
        fwam_shapes({{"M", ADIT_FFN_M}, {"H", ADIT_FFN_H}, {"F", ADIT_FFN_F}, {"BNU", ADIT_FFN_BNU}, {"BN1", ADIT_FFN_BN1}, {"BKU", ADIT_FFN_BKU}, {"BKD", ADIT_FFN_BKD}, {"STAGES", ADIT_FFN_STAGES}, {"NT", ADIT_FFN_NTHREADS}, {"GRID0", ADIT_FFN_GRID0}, {"SLAB1", ADIT_FFN_SLAB1}}),
    });
    return s.c_str();
}

extern "C" void adit_ffn_cuda(const void* x_in, const void* shift_mlp,
                                  const void* scale_mlp, const void* gate_mlp,
                                  const void* w0, const float* sw0, const void* b0,
                                  const void* w1, const float* sw1, const void* b1,
                                  void* out, void* a8buf, float* sa0buf,
                                  void* gbuf, void* a8gbuf, uint32_t* raw1,
                                  int B, int only_phase, cudaStream_t stream)
{
    dim3 grid(ADIT_FFN_GRID0 * B), block(ADIT_FFN_NTHREADS);
    void* args[] = {
        (void*)&x_in, (void*)&shift_mlp, (void*)&scale_mlp, (void*)&gate_mlp,
        (void*)&w0, (void*)&sw0, (void*)&b0,
        (void*)&w1, (void*)&sw1, (void*)&b1,
        (void*)&out, (void*)&a8buf, (void*)&sa0buf,
        (void*)&gbuf, (void*)&a8gbuf, (void*)&raw1,
        (void*)&B, (void*)&only_phase,
    };
    cudaError_t e;
    if (only_phase < 0) {
        // Device capacity check: grid = GRID0*B is decided by the slab geometry (inst/slab are
        // decomposed from blockIdx.x) and cannot be resized; the cooperative kernel requires the
        // whole grid to be co-resident, so report it clearly here when it does not fit.
        static const int cap = coop_grid((const void*)adit_ffn_kernel, ADIT_FFN_NTHREADS, 0, ADIT_FFN_GRID0,
                                         "adit_ffn");
        if (cap < ADIT_FFN_GRID0 * B) {
            coop_fail("adit_ffn",
                      "device capacity " + std::to_string(cap) + " CTA < grid "
                      + std::to_string(ADIT_FFN_GRID0 * B) + " (=GRID0 " + std::to_string(ADIT_FFN_GRID0)
                      + " × B " + std::to_string(B) + "): the grid grows linearly with the instance count "
                        "B, so too little capacity means reducing B or using the non-cooperative "
                        "split-phase form");
        }
        e = cudaLaunchCooperativeKernel((void*)adit_ffn_kernel, grid, block,
                                        args, 0, stream);
    } else {
        // Non-cooperative split phases: the grid must stay GRID0*B (decided by the slab/tile
        // geometry); a plain launch may run in multiple waves.
        e = cudaLaunchKernel((void*)adit_ffn_kernel, grid, block, args, 0, stream);
    }
    check(e, only_phase < 0 ? "cooperative launch adit_ffn"
                            : "split launch adit_ffn");
}
