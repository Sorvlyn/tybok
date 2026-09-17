// Single-kernel fusion of the video DiT FFN (cooperative, 4 phases / 3 grid.sync).
//
// Shapes M=120 / H=3072 / F=14336; two fp8 GEMMs:
//   up  : A8[120,H] × W0[F,H]ᵀ → bf16[120,F]
//   down: A8[120,F] × W1[H,F]ᵀ → bf16[120,H]
//
// Flow:
//   F_PHASE_1 input quantization x_in → a8 + sa0 (per-token amax/448 + software RNE)
//   F_PHASE_2 up GEMM → epilogue dequant+GELU(tanh)+bf16 → gbuf; row |g| amax → raw1 (atomicMax)
//   F_PHASE_3 quantize gbuf → a8g + sa1 (sa1 = max(raw1/448, 1e-12))
//   F_PHASE_4 down GEMM → RESID: dequant → gate*z + x_in → out
//   phases are separated by grid.sync; raw1 is zeroed at the start.
//
// Structural lower bound VDIT_FFN_M=120 (F_PHASE_1/F_PHASE_3 claim one row each); both phases must fit in ≤48KB to keep 2 CTA/SM.
// Numerics: quantization-drift tier (rel ≲3e-2); GELU uses 0.5x(1+tanh(c1(x+c2x³))), c1=sqrt(2/π);
// fp8 uses a single software RNE.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cstdint>
#include <cstdio>

#include "vdit_gemm_core.cu"  // fwam_fp8_gemm_body + device-side tools (shared by the three vdit kernels)
#include "coop_grid.h"
#include "geom_report.h"            // grid comes from the device capacity (occupancy × SM)
#include "dit_common.h"

// ---- Geometry (GEMM template parameters): instantiation and self-report share **one macro set**; changing this = changing the geometry ----
#define FWAM_VDIT_FFN_F_PHASE_2_TILES 64, 64, 128, 3, 3, 4, 2, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, \
                                       0, 1, 0
#define FWAM_VDIT_FFN_F_PHASE_4_TILES 64, 48, 128, 3, 3, 4, 2, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 2, 0, 1, 0, 0, 1, 1, 0, \
                                       0, 1, 1

namespace cg = cooperative_groups;

// ------------------------------------------------------------------ //
// Shapes and geometry constants
// ------------------------------------------------------------------ //
constexpr int VDIT_FFN_M = 120;      // video prefill tokens
constexpr int VDIT_FFN_H = 3072;     // hidden
constexpr int VDIT_FFN_F = 14336;    // ffn
constexpr int VDIT_FFN_NT = 256;     // threads/CTA
constexpr int VDIT_FFN_GRID = 132;   // nominal value; the actual grid is taken by the launcher from the device capacity (see vf_grid())
constexpr int VDIT_FFN_NWARP = VDIT_FFN_NT / 32;

// up phase smem = (64+64)*128*3 = 49152 B (XS=1, no pad)
// down phase smem = (64+48)*144*3 = 48384 B (XS=2 dense+pad)
constexpr int VDIT_FFN_SMEM = 49152;

// Phase timing: %globaltimer marks (tick = 1ns). ts[p*VDIT_FFN_GRID + cta]; fwam_gt() in gemm_common.h.
#define VDIT_FFN_MARK(p)  FWAM_MARK(phase_ts, p)

// Non-cooperative fallback (only_phase >= 0): each phase is launched separately and visibility between phases comes from stream ordering;
// the split path builds no grid barrier, so a plain `cudaLaunchKernel` can be used (no co-residency requirement).
// The fine-grained fusion inside each phase (F_PHASE_1 norm+modulate+quantization; F_PHASE_2 GEMM+dequant+GELU+amax;
// F_PHASE_3 quantization; F_PHASE_4 GEMM+dequant+gate_mlp residual) is all kept.
#define VDIT_FFN_PHASE(k)     (only_phase < 0 || only_phase == (k))
#define VDIT_FFN_GRID_SYNC()  do { if (only_phase < 0) { __threadfence(); cg::this_grid().sync(); } } while (0)



// ------------------------------------------------------------------ //
// Fused kernel
// ------------------------------------------------------------------ //
// The body's template order for F_PHASE_2 (see up_gemm.cu):
//   <BM,BN,BK,SA,SB,WM,WN,SPLIT,EV,G2,LIN,ORD,APATH,EPI,FA,PH,LDB,PD,XS,F16P,FAP,
//    ACG,WCG,AHINT,WHINT,BORD,WPF,PERSIST,RESID>
//   up  : 64,64,128, 3,3, 4,2, 1, 0,0,1,0,0, 1,1, 1,1,0, 1, 0,1, 1,1,0,1, 0,0, 1,0
//   down: 64,48,128, 3,3, 4,2, 1, 0,0,1,0,0, 0,1, 1,1,0, 2, 0,1, 0,0,1,1, 0,0, 1,1
extern "C" __global__ void __launch_bounds__(VDIT_FFN_NT, 2)
vdit_ffn_kernel(const uint8_t* __restrict__ x_in,   // bf16 [M,H] block input (norm2 input + residual term)
                      const uint8_t* __restrict__ gate_mlp,   // bf16 [H]
                      const uint8_t* __restrict__ shift_mlp, // bf16 [H] AdaLN shift
                      const uint8_t* __restrict__ scale_mlp, // bf16 [H] AdaLN scale
                      const uint8_t* __restrict__ w0,     // fp8 [F,H]
                      const float* __restrict__ sw0,      // fp32 [F]
                      const uint8_t* __restrict__ b0,     // bf16 [F]
                      const uint8_t* __restrict__ w1,     // fp8 [H,F]
                      const float* __restrict__ sw1,      // fp32 [H]
                      const uint8_t* __restrict__ b1,     // bf16 [H]
                      uint8_t* __restrict__ out,          // bf16 [M,H]
                      uint8_t* __restrict__ a8buf,        // fp8  [M,H]  F_PHASE_1 output
                      float* __restrict__ sa0buf,         // fp32 [M]
                      uint8_t* __restrict__ gbuf,         // bf16 [M,F]  F_PHASE_2 output
                      uint8_t* __restrict__ a8gbuf,       // fp8  [M,F]  F_PHASE_3 output
                      float* __restrict__ sa1buf,         // fp32 [M]
                      uint32_t* __restrict__ raw1,        // [M] fp32 bit pattern: row |g| amax
                      int stop_phase,                     // debug: 0 = run all; <0 runs F_PHASE_4 only
                      int only_phase,                     // non-cooperative fallback: 1/2/3/4 = run only F_PHASE_1/F_PHASE_2/F_PHASE_3/F_PHASE_4; <0 = the cooperative full pipeline
                      int use_norm,                       // 1 = do norm2+modulate; 0 = only the input quantization
                      float norm_eps,                     // same eps as WanLayerNorm (default 1e-6)
                      int mod_srow,                       // row stride of shift/scale/gate_mlp (elements); 0 = broadcast
                      unsigned long long* phase_ts)       // phase timing marks (nullable)
{
    // ⚠️ NOTE: F_PHASE_1's scratch must live in the **dynamic** smem, not in a static __shared__ variable:
    // sm_89 allocates shared memory in **8KB granularity**, so dynamic 49152 + static 48 → rounded up to 57344,
    // and 2 CTA/SM becomes 1 CTA/SM (a straight +57µs on the up phase). The dynamic area is empty during F_PHASE_1
    // (the GEMM's ring is not written until F_PHASE_2), so borrowing the first 64B is exactly right.
    // Layout: sc[0]=mean sc[1]=rstd sc[2]=sa sc[4+wid]=per-warp amax
    extern __shared__ __align__(16) uint8_t vf_smem[];
    float* sc = reinterpret_cast<float*>(vf_smem);
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;

    VDIT_FFN_MARK(0);
    // Cooperative path: stop_phase < 0 → skip F_PHASE_1/F_PHASE_2/F_PHASE_3 and run only F_PHASE_4 (using the caller-prefilled a8gbuf/sa1buf to isolate the down phase).
    // Split path: run_up = whether this launch belongs to F_PHASE_1/F_PHASE_2/F_PHASE_3 (F_PHASE_4 is a separate launch).
    const bool split = (only_phase >= 0);
    const bool run_up = split ? (only_phase >= 1 && only_phase <= 3) : (stop_phase >= 0);

    // ⚠️ NOTE: raw1 zeroing (EPI's atomicMax needs a zero start). The zeroing must happen on a **launch different from
    // the F_PHASE_2 atomicMax**, or with a real grid barrier between the two:
    //   cooperative path: place it before F_PHASE_1 and let the P1→P2 grid.sync() (which includes __threadfence) publish it;
    //   split path: the P2 launch has **no** barrier (VDIT_FFN_GRID_SYNC does nothing when only_phase>=0),
    //     so never clear in P2 —— otherwise block0's 120 plain writes race against the other CTAs' epilogue atomicMax
    //     and wipe out an amax that was already written (that row's sa1 comes out too small → the quantization scale is wrong).
    //     Instead clear it in the F_PHASE_1 launch and rely on the kernel boundary for ordering (the host always issues 1,2,3,4 in order).
    // The old `VDIT_FFN_PHASE(2) && (split || run_up)` lands exactly in the P2 launch on the split path —— that is the
    // source of the non-cooperative form's occasional drift (~2/600).
    const bool clear_raw1 = split ? (only_phase == 1) : (VDIT_FFN_PHASE(2) && run_up);
    if (clear_raw1 && blockIdx.x == 0 && tid < VDIT_FFN_M) raw1[tid] = 0u;

    // ---------------- F_PHASE_1: norm2 + modulate + quantization (one row per CTA, 120 rows ≤ 132 CTAs)--
    // Isomorphic to F_PHASE_1 in the action kernel adit_ffn.cu (the v5 row-parallel version): three passes
    //   pass1 row statistics (warp0 only; mean/rstd go through the shfl tree) —— corresponds to dit_block.apply_norm2
    //         = _wan_layer_norm(WanLayerNorm, x): fp32 statistics, cast back to bf16 after the norm, no affine
    //   pass2 modulate + row |·| amax (all threads) —— corresponds to dit_block.modulate: x*(1+scale)+shift
    //   pass3 quantization (all threads)
    // With use_norm=0, mean=0/rstd=1/scale=shift=0 and mod_bf16 degenerates to the identity (bf16→bf16),
    // so the whole stretch is equivalent to "input per-token quantization only" —— both paths share the same code.
    if (run_up && VDIT_FFN_PHASE(1)) {
        const int r = blockIdx.x;
        if (r < VDIT_FFN_M) {
            const uint32_t* __restrict__ xu =
                reinterpret_cast<const uint32_t*>(x_in + (size_t)r * VDIT_FFN_H * 2);
            // mod_srow=0 → broadcast (a single [H]); =H → one set per token (the video t_mod is 4-D and
            // split_modulation squeezes it into [B,S,H], so here it is indexed with a row offset).
            const int mrow = r * (mod_srow >> 1);              // row offset in u32 units
            const uint32_t* __restrict__ scu =
                reinterpret_cast<const uint32_t*>(scale_mlp) + mrow;
            const uint32_t* __restrict__ shu =
                reinterpret_cast<const uint32_t*>(shift_mlp) + mrow;
            constexpr int NH = VDIT_FFN_H / 2;               // number of u32s (each u32 = 2 bf16)
            constexpr int PER = NH / VDIT_FFN_NT;            // u32s per thread (pass2/3)

            // ---- pass 1: row statistics (warp0) ----
            if (use_norm && wid == 0) {
                float ss = 0.f, qq = 0.f;
                #pragma unroll 4
                for (int i = lane; i < NH; i += 32) {
                    const uint32_t wd = xu[i];
                    const float va = bf16f((unsigned short)(wd & 0xffffu));
                    const float vb = bf16f((unsigned short)(wd >> 16));
                    ss += va; ss += vb;
                    qq += va * va; qq += vb * vb;
                }
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    ss += __shfl_down_sync(0xffffffffu, ss, off);
                    qq += __shfl_down_sync(0xffffffffu, qq, off);
                }
                if (lane == 0) {
                    const float mean = ss / (float)VDIT_FFN_H;
                    const float var = fmaxf(qq / (float)VDIT_FFN_H - mean * mean, 0.f);
                    sc[0] = mean;
                    sc[1] = rsqrtf(var + norm_eps);
                }
            }
            __syncthreads();
            const float mean_r = use_norm ? sc[0] : 0.f;
            const float rstd_r = use_norm ? sc[1] : 1.f;

            // ---- pass 2: modulate + row amax ----
            float mxl = 0.f;
            #pragma unroll 4
            for (int i = tid; i < NH; i += VDIT_FFN_NT) {
                const uint32_t sc01 = scu[i], sh01 = shu[i], wd = xu[i];
                const float ma = mod_bf16(
                    bf16f((unsigned short)(wd & 0xffffu)), mean_r, rstd_r,
                    bf16f((unsigned short)(sc01 & 0xffffu)),
                    bf16f((unsigned short)(sh01 & 0xffffu)));
                const float mb = mod_bf16(
                    bf16f((unsigned short)(wd >> 16)), mean_r, rstd_r,
                    bf16f((unsigned short)(sc01 >> 16)),
                    bf16f((unsigned short)(sh01 >> 16)));
                mxl = fmaxf(mxl, fmaxf(fabsf(ma), fabsf(mb)));
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                mxl = fmaxf(mxl, __shfl_xor_sync(0xffffffffu, mxl, off));
            if (lane == 0) sc[4 + wid] = mxl;
            __syncthreads();
            if (wid == 0 && lane == 0) {
                float m = 0.f;
                #pragma unroll
                for (int i = 0; i < VDIT_FFN_NWARP; ++i) m = fmaxf(m, sc[4 + i]);
                const float sa_v = fmaxf(m / 448.0f, 1e-12f);
                sa0buf[r] = sa_v;
                sc[2] = sa_v;
            }
            __syncthreads();
            const float sa_r = sc[2];

            // ---- pass 3: quantization (write 2 bytes as a pair) ----
            unsigned char* ar = a8buf + (size_t)r * VDIT_FFN_H;
            #pragma unroll 4
            for (int i = tid; i < NH; i += VDIT_FFN_NT) {
                const uint32_t sc01 = scu[i], sh01 = shu[i], wd = xu[i];
                const float ma = mod_bf16(
                    bf16f((unsigned short)(wd & 0xffffu)), mean_r, rstd_r,
                    bf16f((unsigned short)(sc01 & 0xffffu)),
                    bf16f((unsigned short)(sh01 & 0xffffu)));
                const float mb = mod_bf16(
                    bf16f((unsigned short)(wd >> 16)), mean_r, rstd_r,
                    bf16f((unsigned short)(sc01 >> 16)),
                    bf16f((unsigned short)(sh01 >> 16)));
                const unsigned short pair = (unsigned short)fp8_rne(ma / sa_r) |
                                            ((unsigned short)fp8_rne(mb / sa_r) << 8);
                *reinterpret_cast<unsigned short*>(ar + 2 * i) = pair;
            }
        }
    }
    VDIT_FFN_MARK(1);
    VDIT_FFN_GRID_SYNC();
    VDIT_FFN_MARK(2);

    // ---------------- F_PHASE_2: up GEMM (EPI: dequant+GELU+bf16 → gbuf; row amax → raw1) ----
    if (run_up && VDIT_FFN_PHASE(2)) {
    // FA=0 (fp32 accumulation): the production-quantized fp8 uses the full e4m3 range, so with in-stage fp16 accumulation the rms of the 128-term dot
    // product is ~1.8e5 ≫ 65504 and FA=1 always gives NaN (both shapes blew up in measurement); a 2^-4 prescale is not enough either (~5.6σ).
    // PH=0: gbuf is an intermediate that F_PHASE_3 re-reads right away, so it **must not** be marked evict_first the way a standalone kernel would
    // (that would evict gbuf itself from L2 and force F_PHASE_3 back to DRAM reads → measured 15.4µs for F_PHASE_3).
    fwam_fp8_gemm_body<FWAM_VDIT_FFN_F_PHASE_2_TILES>(
        /*A=*/a8buf, /*W=*/w0, /*sa=*/sa0buf, /*sw=*/sw0,
        /*bias=*/reinterpret_cast<const __nv_bfloat16*>(b0),
        /*Cout=*/reinterpret_cast<__nv_bfloat16*>(gbuf),
        /*partial=*/reinterpret_cast<float*>(raw1),   // EPI atomically writes the amax here
        /*counters=*/nullptr,
        /*M=*/VDIT_FFN_M, /*N=*/VDIT_FFN_F, /*K=*/VDIT_FFN_H, /*Kseg=*/VDIT_FFN_H,
        /*wp=*/11, /*ap=*/0, /*col0=*/0, /*ffn_ex=*/nullptr);
    }
    VDIT_FFN_MARK(3);
    VDIT_FFN_GRID_SYNC();
    VDIT_FFN_MARK(4);

    // ---------------- F_PHASE_3: quantize gbuf → a8g + sa1 ----------------
    if (run_up && VDIT_FFN_PHASE(3)) {
        // sa1 = max(raw1/448, 1e-12). Only CTA0 writes it (an earlier version had all 132 CTAs write the same value,
        // so 15840 threads hammered 120 addresses → L2 write contention; this stretch measured 16.4µs, the third largest cost in the whole run).
        if (blockIdx.x == 0 && tid < VDIT_FFN_M)
            sa1buf[tid] = fmaxf(__uint_as_float(raw1[tid]) / 448.0f, 1e-12f);
        // Split by row: each CTA claims a whole row (256 threads cover it contiguously), which guarantees the row's sa is computed once and
        // the accesses are fully contiguous. The previous grid-stride write changed rows at every step (stride 33792 > row width 14336),
        // so every element had to recompute its row index and re-read raw1.
        const unsigned short* __restrict__ gu =
            reinterpret_cast<const unsigned short*>(gbuf);
        for (int r = blockIdx.x; r < VDIT_FFN_M; r += gridDim.x) {
            const float sa = fmaxf(__uint_as_float(raw1[r]) / 448.0f, 1e-12f);
            const unsigned short* gr = gu + (size_t)r * VDIT_FFN_F;
            uint8_t* ar = a8gbuf + (size_t)r * VDIT_FFN_F;
            // Element-wise **true division**, the same semantics as the production Triton `x.to(f32) / scale`.
            // (it was once changed to `x * (1/sa)` to save ALU: measured **not faster at all** —— this loop is memory-bound
            //   and the division hides completely —— yet it donated a ~1ulp rounding bias, so it was reverted.)
            #pragma unroll 4
            for (int c = tid; c < VDIT_FFN_F; c += VDIT_FFN_NT)
                ar[c] = fp8_rne(bf16f(gr[c]) / sa);
        }
    }
    VDIT_FFN_MARK(5);
    VDIT_FFN_GRID_SYNC();
    VDIT_FFN_MARK(6);

    // ---------------- F_PHASE_4: down GEMM (RESID: dequant → gate_mlp*z + x_in → out) --------
    if (VDIT_FFN_PHASE(4)) {
        fwam_gemm_extra ex{x_in, gate_mlp, VDIT_FFN_H, mod_srow};
        fwam_fp8_gemm_body<FWAM_VDIT_FFN_F_PHASE_4_TILES>(
            /*A=*/a8gbuf, /*W=*/w1, /*sa=*/sa1buf, /*sw=*/sw1,
            /*bias=*/reinterpret_cast<const __nv_bfloat16*>(b1),
            /*Cout=*/reinterpret_cast<__nv_bfloat16*>(out),
            /*partial=*/nullptr, /*counters=*/nullptr,
            /*M=*/VDIT_FFN_M, /*N=*/VDIT_FFN_H, /*K=*/VDIT_FFN_F, /*Kseg=*/VDIT_FFN_F,
            // wp=11: W1 is also a 44MB single-pass stream, marked evict_first 0.75 like the up phase.
            // The isolated test saves 10.3µs (dirty-cold 118.8→108.5); hot/net does not move at all (see down_wp.py).
            /*wp=*/11, /*ap=*/0, /*col0=*/0, /*ffn_ex=*/&ex);
    }
    VDIT_FFN_MARK(7);
}

// Single-pass bf16→fp8 quantization (for the unfused baseline; equivalent to the F_PHASE_3 stretch, and it avoids the torch version's
// distortion from materializing a 6.9MB fp32 temporary for `g.float()/sa`).
__global__ void vf_cast_gbuf_kernel(const __nv_bfloat16* __restrict__ g,
                                    const float* __restrict__ sa1,
                                    uint8_t* __restrict__ a8g, int Mn, int Fn) {
    const size_t total = (size_t)Mn * Fn;
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int r = (int)(i / Fn);
    a8g[i] = fp8_rne(__bfloat162float(g[i]) / sa1[r]);
}

extern "C" void vf_cast_gbuf_cuda(const void* g, const float* sa1, void* a8g,
                                  int Mn, int Fn, cudaStream_t stream) {
    const size_t total = (size_t)Mn * Fn;
    const int threads = 256;
    const int blocks = (int)((total + threads - 1) / threads);
    vf_cast_gbuf_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(g), sa1,
        reinterpret_cast<uint8_t*>(a8g), Mn, Fn);
}

// ------------------------------------------------------------------ //
// host: cooperative launch
// ------------------------------------------------------------------ //
// ---- The grid comes from the device capacity (occupancy × SM count), no longer hard-coded to 132 ----
// Structural lower bound = VDIT_FFN_M (F_PHASE_1 is `if (r < VDIT_FFN_M)` with one row per CTA; grid<120 drops rows).
static int g_vf_grid = -1;
static int vf_grid() {
    return cached_coop_grid(g_vf_grid, (const void*)vdit_ffn_kernel, VDIT_FFN_NT,
                            VDIT_FFN_SMEM, VDIT_FFN_M, "vdit_ffn");
}

// ---- Geometry self-report (shape constants + GEMM template parameters; the template parameters use the same macro set as the instantiation) ----
extern "C" const char* vdit_ffn_geom() {
    static const std::string s = fwam_geom_join({
        fwam_shapes({{"M", VDIT_FFN_M}, {"H", VDIT_FFN_H}, {"F", VDIT_FFN_F}, {"NT", VDIT_FFN_NT}, {"SMEM", VDIT_FFN_SMEM}, {"GRID", VDIT_FFN_GRID}}),
        fwam_tiles("F_PHASE_2", FWAM_TILES(FWAM_VDIT_FFN_F_PHASE_2_TILES)),
        fwam_tiles("F_PHASE_4", FWAM_TILES(FWAM_VDIT_FFN_F_PHASE_4_TILES)),
    });
    return s.c_str();
}

extern "C" int vdit_ffn_grid() { return vf_grid(); }
// The actual grid of the non-cooperative split-phase form (lower bound = VDIT_FFN_M; a plain launch may take multiple waves, so small GPUs can run it too)
static int vf_split_grid();   // defined further down in this file (one each for the cooperative/split grid)
extern "C" int vdit_ffn_split_grid() { return vf_split_grid(); }

// The non-cooperative fallback's grid (no co-residency required): lower bound = VDIT_FFN_M (F_PHASE_1's `r < VDIT_FFN_M` and F_PHASE_3's per-row write-back).
static int g_vf_split_grid = -1;
static int vf_split_grid() {
    return cached_split_grid(g_vf_split_grid, (const void*)vdit_ffn_kernel, VDIT_FFN_NT,
                             VDIT_FFN_SMEM, VDIT_FFN_M, "vdit_ffn");
}

extern "C" void vdit_ffn_cuda(const void* x_in, const void* gate_mlp,
                                    const void* shift_mlp, const void* scale_mlp,
                                    const void* w0, const float* sw0, const void* b0,
                                    const void* w1, const float* sw1, const void* b1,
                                    void* out, void* a8buf, float* sa0buf,
                                    void* gbuf, void* a8gbuf, float* sa1buf,
                                    uint32_t* raw1, int stop_phase, int only_phase,
                                    int use_norm, float norm_eps, int mod_srow,
                                    unsigned long long* phase_ts, cudaStream_t stream)
{
    dim3 block(VDIT_FFN_NT);
    void* args[] = {
        (void*)&x_in, (void*)&gate_mlp, (void*)&shift_mlp, (void*)&scale_mlp,
        (void*)&w0, (void*)&sw0, (void*)&b0,
        (void*)&w1, (void*)&sw1, (void*)&b1,
        (void*)&out, (void*)&a8buf, (void*)&sa0buf,
        (void*)&gbuf, (void*)&a8gbuf, (void*)&sa1buf, (void*)&raw1,
        (void*)&stop_phase, (void*)&only_phase, (void*)&use_norm, (void*)&norm_eps,
        (void*)&mod_srow, (void*)&phase_ts,
    };
    cudaError_t e;
    dim3 grid;                       // declared outside: the error path needs to report the grid
    if (only_phase < 0) {
        grid = dim3(vf_grid());
        e = cudaLaunchCooperativeKernel((void*)vdit_ffn_kernel, grid,
                                        block, args, VDIT_FFN_SMEM, stream);
    } else {
        grid = dim3(vf_split_grid());
        e = cudaLaunchKernel((void*)vdit_ffn_kernel, grid, block, args,
                             VDIT_FFN_SMEM, stream);
    }
    if (e != cudaSuccess) {
        // Throw instead of exit(1): exit would kill the worker process outright and the caller would get no handleable error.
        coop_fail("vdit_ffn",
                  "launch failed (only_phase=" + std::to_string(only_phase)
                  + ", grid=" + std::to_string((int)grid.x) + "): "
                  + cudaGetErrorString(e));
    }
}
