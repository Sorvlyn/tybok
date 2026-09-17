// Single-kernel fusion of the UMT5 text encoder's FFN sublayer (cooperative + non-cooperative split-phase).
//
// Shapes M=128 / H=4096 / F=10240; two fp8 GEMMs:
//   up  : A8[128,4096]  × W0[20480,4096]ᵀ → bf16[128,20480] (packed wi_0|wi_1)
//   down: A8[128,10240] × W1[4096,10240]ᵀ → bf16[128,4096] (+ x residual)
//
// Phases:
//   F_PHASE_1 RMSNorm + per-token fp8 quantization → a8 + sa0
//   F_PHASE_2 up GEMM (EPI=0 staged: u = bf16RN(acc*sa*sw+bias)) → gbuf[M,2F]
//   F_PHASE_3 gelu_new gating + row amax + quantization: act = bf16RN(gelu_fp32(gbuf[:,:F]) * bf16f(gbuf[:,F:]))
//      → a8g + sa1 = max(amax/448, 1e-12)
//   F_PHASE_4 down GEMM (RESID=2: single-round fp32 residual, no gate) → out
//   phases are separated by grid.sync.
//
// Numerics: bitwise-aligned with the production Triton chain. The GEMM f32acc is bit-identical to Triton; the gelu uses tmt5_ffn_gelu_new
// (production's association order); the fp8 cast uses tmt5_ffn_fp8_rne.
//
// ⚠️ FA=0 (in-stage fp32 accumulation): production quantization uses up the whole e4m3 range, a 128-term dot product has rms ≈1.8e5 ≫ 65504,
// so FA=1 must produce NaN.
// ⚠️ Do not enable --use_fast_math.
// Geometry: BM128/BN64/BK128/SA=SB=4 → smem 96KB ⇒ 1 CTA/SM.

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
#define FWAM_TMT5_FFN_F_PHASE_2_TILES 128, 64, 128, 4, 4, 2, 4, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, \
                                      0, 0, 1, 0
#define FWAM_TMT5_FFN_F_PHASE_4_TILES 128, 64, 128, 4, 4, 2, 4, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, \
                                      0, 0, 1, 2

namespace cg = cooperative_groups;

// ------------------------------------------------------------------ //
// shape and geometry constants
// ------------------------------------------------------------------ //
constexpr int TMT5_FFN_M = 128;      // text-encoder token count (fixed-length padding, B=1, S≡128)
constexpr int TMT5_FFN_H = 4096;     // d_model
constexpr int TMT5_FFN_F = 10240;    // d_ff (the packed wi has N = 2*TMT5_FFN_F = 20480)
constexpr int TMT5_FFN_NT = 256;     // threads/CTA
constexpr int TMT5_FFN_GRID = 66;    // **nominal** value (1 CTA/SM × 66 SM on this machine); the launcher takes the actual grid from device capacity
// Structural lower bound. The row phases (F_PHASE_1/F_PHASE_3) claim rows with `r = blockIdx.x; r += gridDim.x`, while the GEMM
// phases (F_PHASE_2/F_PHASE_4) are a 1-D grid-stride loop inside fwam_fp8_gemm_body (PERSIST=1) ⇒ both cover the whole work, so
// **there is no structural lower bound**; the grid only affects speed, not numerics (a row/tile is claimed by one CTA and finished inside it in a fixed order).
constexpr int TMT5_FFN_MIN_GRID = 1;
constexpr int TMT5_FFN_NWARP = TMT5_FFN_NT / 32;

// Both phases are (BM+BN)*BK*SA = (128+64)*128*4 = 98304
constexpr int TMT5_FFN_SMEM = 98304;

// Phase timing: %globaltimer marks (tick = 1ns). ts[p*gridDim.x + cta]; fwam_gt() is in gemm_common.h.
// The caller (timing script) must size the buffer by the **actual grid**: `nphases * ext.ffn_grid()`.
#define TMT5_FFN_MARK(p)  FWAM_MARK(phase_ts, p)

// The two switches of the non-cooperative fallback (same as the video kernel):
//   TMT5_FFN_PHASE(k)     —— with only_phase>=0 only that phase is true; with <0 always true (keeps the old behavior).
//   TMT5_FFN_GRID_SYNC()  —— unchanged on the cooperative path (fence + grid.sync); the split path does nothing.
// NOTE: the split path **never** builds `cg::this_grid()` —— only building/calling a grid barrier imposes co-residency.
#define TMT5_FFN_PHASE(k)     (only_phase < 0 || only_phase == (k))
#define TMT5_FFN_GRID_SYNC()  do { if (only_phase < 0) { __threadfence(); cg::this_grid().sync(); } } while (0)

// ------------------------------------------------------------------ //
// fused kernel
// ------------------------------------------------------------------ //
// Template order (see fwam_fp8_gemm_body in tmt5_gemm_core.cu):
//   <BM,BN,BK,SA,SB,WM,WN,SPLIT,EV,G2,LIN,ORD,APATH,EPI,FA,PH,LDB,PD,XS,F16P,FAP,
//    ACG,WCG,AHINT,WHINT,BORD,WPF,PERSIST,RESID>
//   F_PHASE_2: 128,64,128, 4,4, 2,4, 1, 0,0,1,0,0, 0,0, 0,1,0, 1, 0,1, 1,1,0,1, 0,0, 1,0
//   F_PHASE_4: 128,64,128, 4,4, 2,4, 1, 0,0,1,0,0, 0,0, 1,1,0, 1, 0,1, 1,1,0,1, 0,0, 1,2
extern "C" __global__ void __launch_bounds__(TMT5_FFN_NT, 1)
tmt5_ffn_kernel(const uint8_t* __restrict__ x_in,   // bf16 [M,H] FFN input (also the residual term)
                    const uint8_t* __restrict__ nw,     // bf16 [H] RMSNorm weight
                    const uint8_t* __restrict__ w0,     // fp8  [2F,H] packed wi_0|wi_1
                    const float*   __restrict__ sw0,    // fp32  [2F]
                    const uint8_t* __restrict__ b0,     // bf16  [2F] (all zeros; same convention as production)
                    const uint8_t* __restrict__ w1,     // fp8  [H,F]  wo
                    const float*   __restrict__ sw1,    // fp32  [H]
                    const uint8_t* __restrict__ b1,     // bf16  [H] (all zeros)
                    uint8_t* __restrict__ out,          // bf16  [M,H] output (**must not alias x_in**)
                    uint8_t* __restrict__ a8buf,        // fp8   [M,H]   F_PHASE_1 output
                    float*   __restrict__ sa0buf,       // fp32  [M]
                    uint8_t* __restrict__ gbuf,         // bf16  [M,2F]  F_PHASE_2 output (= the wi of K4)
                    uint8_t* __restrict__ a8gbuf,       // fp8   [M,F]   F_PHASE_3 output
                    float*   __restrict__ sa1buf,       // fp32  [M]
                    int stop_phase,                     // debug: <=0 = run all; 1..4 = run only the first N phases
                    int only_phase,                     // non-cooperative fallback: >=0 = run only that phase; <0 = the full cooperative flow
                    float norm_eps,                     // eps of UMT5RMSNorm (default 1e-6)
                    unsigned long long* phase_ts)       // phase timing marks (nullable)
{
    // ⚠️ NOTE: the F_PHASE_1/F_PHASE_3 scratch must live in the **dynamic** smem, not in a static __shared__ variable:
    // sm_89 allocates shared memory in **8KB granularity**, so dynamic 98304 + any static amount rounds up past
    // 100KB → 1 CTA/SM will not compile. The dynamic region is empty during F_PHASE_1/F_PHASE_3 (the F_PHASE_2/F_PHASE_4 rings are only
    // written in their own phase), so borrowing the **tail** 64B fits — the tail rather than the head: F_PHASE_3's whole-row staging
    // (40KB) is laid out from the head, so the two must not overlap.
    // Layout: red[0..7] = one reduction slot per warp; bc[0] = rstd/sa slot, bc[1] = sa
    extern __shared__ __align__(16) uint8_t tmt5_smem[];
    float* red = reinterpret_cast<float*>(tmt5_smem + TMT5_FFN_SMEM - 64);
    float* bc  = red + TMT5_FFN_NWARP;
    unsigned short* rowbuf = reinterpret_cast<unsigned short*>(tmt5_smem);   // F_PHASE_3 row staging

    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;

    const int UPTO = (stop_phase <= 0) ? 4 : stop_phase;
    TMT5_FFN_MARK(0);

    // ---------------- F_PHASE_1: RMSNorm + per-token fp8 quantization -------------------- //
    // Replicates the **rounding points** of Triton's `_norm_kernel` (triton_text_encoder_attn.py:29):
    //   ① rstd = 1/sqrt(Σx²/K + eps)            fp32 (**not rsqrtf**, that is an approximate instruction)
    //   ② n  = bf16RN(x * rstd)                 bf16 rounding point 1
    //   ③ nb = bf16RN(w_bf16 * n)               bf16 rounding point 2
    //   ④ sa = max(amax(|nb|) * (1/448), 1e-12) fp32 (reciprocal multiply, the same semantics as torch/Triton scalar division)
    //   ⑤ a8 = tmt5_ffn_fp8_rne(nb / sa)              true division + a single RNE
    // The reduction order differs from Triton's `tl.sum` tree (not cheaply replicable), so this phase is **not bitwise**,
    // expecting ~1ulp rstd; the other three phases are bitwise.
    if (UPTO >= 1 && TMT5_FFN_PHASE(1)) {
        for (int r = blockIdx.x; r < TMT5_FFN_M; r += (int)gridDim.x) {
            const unsigned short* __restrict__ xr =
                reinterpret_cast<const unsigned short*>(x_in + (size_t)r * TMT5_FFN_H * 2);
            // ⚠️ nw is a byte pointer and must be indexed as bf16 (unsigned short); nw[i] taken directly is a single byte.
            const unsigned short* __restrict__ nwu =
                reinterpret_cast<const unsigned short*>(nw);
            // pass 1: Σx² (fmaf, same order as the reference kernel)
            float ss = 0.f;
            for (int i = tid; i < TMT5_FFN_H; i += TMT5_FFN_NT) {
                const float f = bf16f(xr[i]);
                ss = fmaf(f, f, ss);
            }
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
            if (lane == 0) red[wid] = ss;
            __syncthreads();
            if (wid == 0) {
                float v = (lane < TMT5_FFN_NWARP) ? red[lane] : 0.f;
                #pragma unroll
                for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
                if (lane == 0) bc[0] = 1.0f / sqrtf(v / (float)TMT5_FFN_H + norm_eps);
            }
            __syncthreads();
            const float rstd = bc[0];

            // pass 2: normalization (two levels of bf16 rounding) + row amax
            float am = 0.f;
            for (int i = tid; i < TMT5_FFN_H; i += TMT5_FFN_NT) {
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
                float v = (lane < TMT5_FFN_NWARP) ? red[lane] : 0.f;
                #pragma unroll
                for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
                if (lane == 0) bc[1] = fmaxf(v * (1.0f / 448.0f), 1e-12f);
            }
            __syncthreads();
            const float sa = bc[1];
            if (tid == 0) sa0buf[r] = sa;

            // pass 3: cast (deterministically recompute nb, bitwise identical to pass 2)
            uint8_t* qr = a8buf + (size_t)r * TMT5_FFN_H;
            for (int i = tid; i < TMT5_FFN_H; i += TMT5_FFN_NT) {
                const float xf = bf16f(xr[i]);
                const float n = fwam_bf16r(xf * rstd);
                const float nb = fwam_bf16r(bf16f(nwu[i]) * n);
                qr[i] = tmt5_ffn_fp8_rne(tmt5_ffn_div_full(nb, sa));
            }
            __syncthreads();   // red/bc are reused for the next row
        }
    }
    TMT5_FFN_MARK(1);
    TMT5_FFN_GRID_SYNC();
    TMT5_FFN_MARK(2);

    // ---------------- F_PHASE_2: up GEMM (packed wi, N=2F) ------------------------ //
    if (UPTO >= 2 && TMT5_FFN_PHASE(2)) {
    // EPI=0 + STAGED: the epilogue emits u = bf16RN(acc*sa*sw+bias) as-is — this is the bf16 output wi of
    // production's K4. **Neither GELU nor gating may be done here**: EPI=1 would round the gelu to bf16 first
    // (one extra rounding the production chain does not have), and the gate must wait until the wi_1 half is in — both
    // belong in F_PHASE_3. PH=0: gbuf is an intermediate that F_PHASE_3 is about to re-read, so it **must not** get evict_first (video's pitfall 2).
    fwam_fp8_gemm_body<FWAM_TMT5_FFN_F_PHASE_2_TILES>(
        /*A=*/a8buf, /*W=*/w0, /*sa=*/sa0buf, /*sw=*/sw0,
        /*bias=*/reinterpret_cast<const __nv_bfloat16*>(b0),
        /*Cout=*/reinterpret_cast<__nv_bfloat16*>(gbuf),
        /*partial=*/nullptr, /*counters=*/nullptr,
        /*M=*/TMT5_FFN_M, /*N=*/2 * TMT5_FFN_F, /*K=*/TMT5_FFN_H, /*Kseg=*/TMT5_FFN_H,
        /*wp=*/11, /*ap=*/0, /*col0=*/0, /*ffn_ex=*/nullptr);
    }
    TMT5_FFN_MARK(3);
    TMT5_FFN_GRID_SYNC();
    TMT5_FFN_MARK(4);

    // ---------------- F_PHASE_3: gelu_new gating + row amax + quantization ------------------- //
    // Replicates production's K5 (triton_text_encoder_ffn.py:31):
    //   act_f32 = tmt5_ffn_gelu_new(bf16f(gbuf[r,c])) * bf16f(gbuf[r, c+F])   ← all fp32
    //   act     = bf16RN(act_f32)                                       ← a single bf16 rounding
    // Then act is treated as K5's output and goes through quantize_act's semantics (amax → sa → true division + RNE).
    // Implementation: **one CTA for the whole row**, the row staged into smem (20480 bf16 = 40KB ≤ 96KB),
    // one DRAM pass end to end → gating/amax → overwritten in place with act → quantized by sa into a8g.
    // A single pass = 5.24MB read + 1.25MB write; two sweeps (one for amax, one for cast) would read 5.24MB more ≈ 11µs.
    if (UPTO >= 3 && TMT5_FFN_PHASE(3)) {
        for (int r = blockIdx.x; r < TMT5_FFN_M; r += (int)gridDim.x) {
            const unsigned short* __restrict__ gr =
                reinterpret_cast<const unsigned short*>(gbuf + (size_t)r * 2 * TMT5_FFN_F * 2);
            // The whole row into smem (16B vectorized: 8 bf16 per access, (2F)/8 = 2560 accesses / 256 threads = 10 rounds)
            {
                const uint4* src = reinterpret_cast<const uint4*>(gr);
                uint4* dst = reinterpret_cast<uint4*>(rowbuf);
                #pragma unroll
                for (int i = tid; i < (2 * TMT5_FFN_F) / 8; i += TMT5_FFN_NT) dst[i] = src[i];
            }
            __syncthreads();
            // gating + row amax
            float am = 0.f;
            for (int c = tid; c < TMT5_FFN_F; c += TMT5_FFN_NT) {
                const float g0 = bf16f(rowbuf[c]);
                const float u1 = bf16f(rowbuf[c + TMT5_FFN_F]);
                const float act = fwam_bf16r(tmt5_ffn_gelu_new(g0) * u1);
                rowbuf[c] = __bfloat16_as_ushort(__float2bfloat16_rn(act));
                am = fmaxf(am, fabsf(act));
            }
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
            if (lane == 0) red[wid] = am;
            __syncthreads();
            if (wid == 0) {
                float v = (lane < TMT5_FFN_NWARP) ? red[lane] : 0.f;
                #pragma unroll
                for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
                if (lane == 0) bc[1] = fmaxf(v * (1.0f / 448.0f), 1e-12f);
            }
            __syncthreads();
            const float sa = bc[1];
            if (tid == 0) sa1buf[r] = sa;
            // quantization (recomputed from smem, DRAM untouched)
            uint8_t* qr = a8gbuf + (size_t)r * TMT5_FFN_F;
            for (int c = tid; c < TMT5_FFN_F; c += TMT5_FFN_NT)
                qr[c] = tmt5_ffn_fp8_rne(tmt5_ffn_div_full(bf16f(rowbuf[c]), sa));
            __syncthreads();   // rowbuf/red/bc are reused for the next row
        }
    }
    TMT5_FFN_MARK(5);
    TMT5_FFN_GRID_SYNC();
    TMT5_FFN_MARK(6);

    // ---------------- F_PHASE_4: down GEMM (wo + single-round residual) -------------------- //
    // F_PHASE_4 originally had no UPTO guard (stop_phase only gates F_PHASE_1..F_PHASE_3; the last phase always runs); only the
    // "is this the current phase" test is done here, and it is always true when only_phase<0 ⇒ verbatim-identical to the old behavior.
    if (TMT5_FFN_PHASE(4)) {
    fwam_gemm_extra ex{x_in, nullptr, TMT5_FFN_H, 0};   // RESID=2 does not use gate
    fwam_fp8_gemm_body<FWAM_TMT5_FFN_F_PHASE_4_TILES>(
        /*A=*/a8gbuf, /*W=*/w1, /*sa=*/sa1buf, /*sw=*/sw1,
        /*bias=*/reinterpret_cast<const __nv_bfloat16*>(b1),
        /*Cout=*/reinterpret_cast<__nv_bfloat16*>(out),
        /*partial=*/nullptr, /*counters=*/nullptr,
        /*M=*/TMT5_FFN_M, /*N=*/TMT5_FFN_H, /*K=*/TMT5_FFN_F, /*Kseg=*/TMT5_FFN_F,
        // wp=11: W1 is also a 42MB single-pass stream, so it carries evict_first 0.75 just like F_PHASE_2 (the largest single item on this machine).
        // PH=1: out is no longer reused by this kernel, so its stores carry evict_first and do not steal L2 ways from W/A.
        /*wp=*/11, /*ap=*/0, /*col0=*/0, /*ffn_ex=*/&ex);
    }
    TMT5_FFN_MARK(7);
}

// ------------------------------------------------------------------ //
// host: cooperative launch
// ------------------------------------------------------------------ //
// ---- the grid is decided by device capacity (occupancy × SM count); 66 = this machine's SM count is no longer hard-coded ----
// Both forms share the same "capacity" convention; the difference is only that the cooperative form additionally requires the whole grid to be co-resident.
static int g_tmt5_ffn_split_grid = -1;
static int tmt5_ffn_grid_split() {
    return cached_split_grid(g_tmt5_ffn_split_grid, (const void*)tmt5_ffn_kernel, TMT5_FFN_NT,
                             TMT5_FFN_SMEM, TMT5_FFN_MIN_GRID, "tmt5_ffn");
}
static int g_tmt5_ffn_coop_grid = -1;
static int tmt5_ffn_grid_coop() {
    return cached_coop_grid(g_tmt5_ffn_coop_grid, (const void*)tmt5_ffn_kernel, TMT5_FFN_NT,
                            TMT5_FFN_SMEM, TMT5_FFN_MIN_GRID, "tmt5_ffn");
}
// Lets the timing script size the phase_ts buffer by the **actual grid** (the span is gridDim.x, not a hard-coded 66).

// ---- geometry self-report (shape constants + GEMM template parameters; the instantiation uses the same macro set) ----
extern "C" const char* tmt5_ffn_geom() {
    static const std::string s = fwam_geom_join({
        fwam_shapes({{"M", TMT5_FFN_M}, {"H", TMT5_FFN_H}, {"F", TMT5_FFN_F}, {"NT", TMT5_FFN_NT}, {"SMEM", TMT5_FFN_SMEM}, {"GRID", TMT5_FFN_GRID}}),
        fwam_tiles("F_PHASE_2", FWAM_TILES(FWAM_TMT5_FFN_F_PHASE_2_TILES)),
        fwam_tiles("F_PHASE_4", FWAM_TILES(FWAM_TMT5_FFN_F_PHASE_4_TILES)),
    });
    return s.c_str();
}

extern "C" int tmt5_ffn_grid() { return tmt5_ffn_grid_split(); }
extern "C" int tmt5_ffn_coop_grid() { return tmt5_ffn_grid_coop(); }

extern "C" void tmt5_ffn_cuda(const void* x_in, const void* nw,
                                  const void* w0, const float* sw0, const void* b0,
                                  const void* w1, const float* sw1, const void* b1,
                                  void* out, void* a8buf, float* sa0buf,
                                  void* gbuf, void* a8gbuf, float* sa1buf,
                                  int stop_phase, float norm_eps,
                                  unsigned long long* phase_ts, int only_phase,
                                  cudaStream_t stream)
{
    dim3 block(TMT5_FFN_NT);
    void* args[] = {
        (void*)&x_in, (void*)&nw, (void*)&w0, (void*)&sw0, (void*)&b0,
        (void*)&w1, (void*)&sw1, (void*)&b1,
        (void*)&out, (void*)&a8buf, (void*)&sa0buf,
        (void*)&gbuf, (void*)&a8gbuf, (void*)&sa1buf,
        (void*)&stop_phase, (void*)&only_phase, (void*)&norm_eps, (void*)&phase_ts,
    };
    cudaError_t e;
    dim3 grid;
    if (only_phase < 0) {
        grid = dim3(tmt5_ffn_grid_coop());
        e = cudaLaunchCooperativeKernel((void*)tmt5_ffn_kernel, grid,
                                        block, args, TMT5_FFN_SMEM, stream);
    } else {
        grid = dim3(tmt5_ffn_grid_split());
        e = cudaLaunchKernel((void*)tmt5_ffn_kernel, grid, block, args,
                             TMT5_FFN_SMEM, stream);
    }
    if (e != cudaSuccess) {
        coop_fail("tmt5_ffn",
                  "launch failed (only_phase=" + std::to_string(only_phase)
                  + ", grid=" + std::to_string((int)grid.x) + "): "
                  + cudaGetErrorString(e));
    }
}
