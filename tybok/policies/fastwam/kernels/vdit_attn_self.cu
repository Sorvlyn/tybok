// One cooperative kernel fusing the whole video-expert self-attention chain (6 phases / 5 grid.sync).
//
// Replaces the torch chain: modulate(norm1) → qkv projection + qk-norm + 3-D RoPE → SDPA → gate residual.
//
// Phases:
//   S_PHASE_1 [b<120]   norm1 + modulate + row quantization → a8x/sa0x
//   S_PHASE_2 [full grid] qkv fp8 GEMM → qkv bf16
//   S_PHASE_3 [b<120]   q/k full-row RMS + qk-norm + 3-D RoPE; v as-is → qc/kc/vc (kc/vc = KV cache)
//   S_PHASE_4 [b<96]    bf16 flash (24 heads × 4 q-tiles) → attn (**fp32**)
//   S_PHASE_5 [b<120]   row amax + quantization → a8a/sa0a
//   S_PHASE_6 [full grid] o fp8 GEMM + gate residual → out bf16
//
// only_phase (non-cooperative fallback): >=0 runs that phase only and skips the grid barrier, so each
// phase can be issued with a plain launch (stream order keeps it visible); <0 is verbatim-identical to the cooperative form.
//
// Numerics: quantization-drift tier (rel ≲3e-2); the attn output stays fp32 and is quantized directly, with no intermediate bf16 rounding.
#include <cooperative_groups.h>
#include "vdit_gemm_core.cu"   // fwam_fp8_gemm_body + device-side helpers (shared by the three vdit kernels)
#include "coop_grid.h"
#include "geom_report.h"            // grid comes from the device capacity (occupancy × SM)
#include "dit_common.h"

// ---- Geometry (GEMM template parameters): instantiation and self-report share **one macro set**; changing this = changing the geometry ----
#define FWAM_VDIT_SELF_S_PHASE_2_TILES 64, 64, 128, 3, 3, 4, 2, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, \
                                          0, 1, 0
#define FWAM_VDIT_SELF_S_PHASE_6_TILES 64, 64, 128, 3, 3, 4, 2, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, \
                                          0, 1, 1

namespace cg = cooperative_groups;

// ------------------------------------------------------------------ //
// Geometry (all fixed at compile time; other shapes need a recompile)
// ------------------------------------------------------------------ //
constexpr int VDIT_SELF_M = 120;      // video prefill tokens
constexpr int VDIT_SELF_H = 3072;     // hidden
constexpr int VDIT_SELF_H3 = 3072;    // attention width (= 24 × 128, same as hidden)
constexpr int VDIT_SELF_N = 9216;     // packed qkv output width (3 × H3)
constexpr int VDIT_SELF_NH = 24;      // heads
constexpr int VDIT_SELF_D = 128;      // head_dim
constexpr int VDIT_SELF_NT = 256;     // threads/CTA
constexpr int VDIT_SELF_GRID = 132;   // nominal value for documentation only; the actual grid is taken by the launcher from the device capacity (see va_grid())
constexpr int VDIT_SELF_NWARP = VDIT_SELF_NT / 32;

// flash
constexpr int VDIT_SELF_QT = 32;                     // q rows per CTA
// ⚠️ NOTE: 120 = 3×32 + **24** is not divisible! It must be ceil (=4), otherwise VDIT_SELF_FGRID
// computes to 72, only the first 18 heads are written and the last 6 are left as-is (actually hit in 2026-09:
// integer division truncated 120/32 to 3). The last q-tile has only 24 valid rows, and both the q load and the write-back must bounds-check.
constexpr int VDIT_SELF_NQT = (VDIT_SELF_M + VDIT_SELF_QT - 1) / VDIT_SELF_QT;   // 4
constexpr int VDIT_SELF_FGRID = VDIT_SELF_NH * VDIT_SELF_NQT;      // 96 CTAs run the flash
constexpr int VDIT_SELF_TILE = 16;                   // kv tile rows
constexpr int VDIT_SELF_BP = 136;                    // flash smem row stride (17 16B units ≡ 1 mod 8)
constexpr int VDIT_SELF_NSTAGE = 2;

// ⚠️ NOTE: the dynamic smem must be exactly 49152 (= 6 × 8192, right on the 8KB granularity → 2 CTA/SM).
// **This file must have no static `__shared__` at all**: 49152 + any static bytes is rounded up by sm_89's
// 8KB granularity to 57344 → 2 CTA/SM drops to 1 CTA/SM, silently twice as slow (hit twice in the FFN round).
constexpr int VDIT_SELF_SMEM = 49152;

// Phase timing marks: fwam_gt() in gemm_common.h (shared by vdit+tmt5)
#define VDIT_SELF_MARK(p)  FWAM_MARK(phase_ts, p)

// The two switches of the non-cooperative fallback:
//   VDIT_SELF_PHASE(k)     —— with only_phase>=0 only that phase is true; with <0 always true (keeps the old behavior).
//   VDIT_SELF_GRID_SYNC()  —— unchanged on the cooperative path (fence + grid.sync); a no-op on the split path.
// NOTE: the split path **never** builds `cg::this_grid()` —— only building/calling a grid barrier imposes co-residency.
#define VDIT_SELF_PHASE(k)     (only_phase < 0 || only_phase == (k))
#define VDIT_SELF_GRID_SYNC()  do { if (only_phase < 0) { __threadfence(); cg::this_grid().sync(); } } while (0)





// ------------------------------------------------------------------ //
__global__ void __launch_bounds__(VDIT_SELF_NT, 2)
vdit_attn_self_kernel(
    const uint8_t* __restrict__ x_in,       // bf16 [M,H]  block input (norm1 input + residual)
    const uint8_t* __restrict__ shift_msa,  // bf16 [M,H] or [H]
    const uint8_t* __restrict__ scale_msa,  // bf16 same as above
    const uint8_t* __restrict__ gate_msa,   // bf16 same as above
    const uint8_t* __restrict__ wqkv,       // fp8  [9216, H]
    const float* __restrict__ swqkv,        // fp32 [9216]
    const uint8_t* __restrict__ bqkv,       // bf16 [9216]
    const uint8_t* __restrict__ wnq,        // bf16 [H3]  qk-norm weight (full row, not per-head)
    const uint8_t* __restrict__ wnk,        // bf16 [H3]
    const float* __restrict__ cos_t,        // fp32 [M, 64]  3-D grid RoPE table
    const float* __restrict__ sin_t,        // fp32 [M, 64]
    const uint8_t* __restrict__ wo,         // fp8  [H, H3]
    const float* __restrict__ swo,          // fp32 [H]
    const uint8_t* __restrict__ bo,         // bf16 [H]
    uint8_t* __restrict__ out,              // bf16 [M,H]
    // scratch (host-allocated; must be globally visible across grid.sync)
    uint8_t* __restrict__ qkv,              // bf16 [M, 9216]
    uint8_t* __restrict__ qc,               // bf16 [M, H3]  q after norm+rope
    uint8_t* __restrict__ kc,               // bf16 [M, H3]  k after norm+rope = KV cache
    uint8_t* __restrict__ vc,               // bf16 [M, H3]  v as-is = KV cache
    uint8_t* __restrict__ attn,             // fp32 [M, H3]
    uint8_t* __restrict__ a8x,              // fp8  [M, H]
    float* __restrict__ sa0x,               // fp32 [M]
    uint8_t* __restrict__ a8a,              // fp8  [M, H3]
    float* __restrict__ sa0a,               // fp32 [M]
    float norm_eps,                         // same eps as WanLayerNorm / WanRMSNorm
    int mod_srow,                           // shift/scale/gate row stride (elements); 0 = broadcast
    int stop_phase,                         // debug: 0 = run all; >0 = stop after that phase
    int only_phase,                         // non-cooperative fallback: >=0 runs that phase only; <0 = the cooperative full pipeline
    float* dbg,                             // debug probe (nullable), 8 slots per CTA
    unsigned long long* phase_ts)
{
    extern __shared__ __align__(16) uint8_t va_smem[];
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;

    VDIT_SELF_MARK(0);

    // ================= S_PHASE_1: norm1 + modulate + row quantization (one row per CTA, b<120) =================
    if (b < VDIT_SELF_M && VDIT_SELF_PHASE(1)) {
        const int r = b;
        const uint32_t* __restrict__ xu =
            reinterpret_cast<const uint32_t*>(x_in + (size_t)r * VDIT_SELF_H * 2);
        const int mrow = r * (mod_srow >> 1);      // row offset in u32 units; srow=0 → broadcast
        const uint32_t* __restrict__ scu =
            reinterpret_cast<const uint32_t*>(scale_msa) + mrow;
        const uint32_t* __restrict__ shu =
            reinterpret_cast<const uint32_t*>(shift_msa) + mrow;
        constexpr int NH = VDIT_SELF_H / 2;               // number of u32s
        constexpr int PER = NH / VDIT_SELF_NT;            // u32s per thread (= 6)
        float* sc = reinterpret_cast<float*>(va_smem);   // borrowed from the front of the dynamic area (the ring is not written yet)

        // ---- pass 1: row statistics (warp0) ----
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
                const float mean = ss / (float)VDIT_SELF_H;
                const float var = fmaxf(qq / (float)VDIT_SELF_H - mean * mean, 0.f);
                sc[0] = mean;
                sc[1] = rsqrtf(var + norm_eps);
            }
        }
        __syncthreads();
        const float mean_r = sc[0];
        const float rstd_r = sc[1];

        // ---- pass 2: modulate + row amax ----
        float mxl = 0.f;
        #pragma unroll 4
        for (int i = tid; i < NH; i += VDIT_SELF_NT) {
            const uint32_t sc01 = scu[i], sh01 = shu[i], wd = xu[i];
            const float ma = mod_bf16(bf16f((unsigned short)(wd & 0xffffu)),
                                         mean_r, rstd_r,
                                         bf16f((unsigned short)(sc01 & 0xffffu)),
                                         bf16f((unsigned short)(sh01 & 0xffffu)));
            const float mb = mod_bf16(bf16f((unsigned short)(wd >> 16)),
                                         mean_r, rstd_r,
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
            for (int i = 0; i < VDIT_SELF_NWARP; ++i) m = fmaxf(m, sc[4 + i]);
            const float sa_v = fmaxf(m / 448.0f, 1e-12f);
            sa0x[r] = sa_v;
            sc[2] = sa_v;
        }
        __syncthreads();
        const float sa_r = sc[2];

        // ---- pass 3: quantization (write 2 bytes as a pair) ----
        unsigned char* ar = a8x + (size_t)r * VDIT_SELF_H;
        #pragma unroll 4
        for (int i = tid; i < NH; i += VDIT_SELF_NT) {
            const uint32_t sc01 = scu[i], sh01 = shu[i], wd = xu[i];
            const float ma = mod_bf16(bf16f((unsigned short)(wd & 0xffffu)),
                                         mean_r, rstd_r,
                                         bf16f((unsigned short)(sc01 & 0xffffu)),
                                         bf16f((unsigned short)(sh01 & 0xffffu)));
            const float mb = mod_bf16(bf16f((unsigned short)(wd >> 16)),
                                         mean_r, rstd_r,
                                         bf16f((unsigned short)(sc01 >> 16)),
                                         bf16f((unsigned short)(sh01 >> 16)));
            const unsigned short pair = (unsigned short)fp8_rne(ma / sa_r) |
                                        ((unsigned short)fp8_rne(mb / sa_r) << 8);
            *reinterpret_cast<unsigned short*>(ar + 2 * i) = pair;
        }
    }
    VDIT_SELF_MARK(1);
    VDIT_SELF_GRID_SYNC();
    VDIT_SELF_MARK(2);

    // ================= S_PHASE_2: qkv fp8 GEMM (N=9216) =================
    if (VDIT_SELF_PHASE(2) && (stop_phase == 0 || stop_phase > 2)) {
        // XS=1 (XOR swizzle, no pad): (64+64)*128*3 = 49152, right on the budget.
        // XS=2 would need (64+64)*144*3 = 55296 → over budget, and it would push out the 2 CTA/SM.
        // FA=0 is required: the production-quantized e4m3 uses the full range, and with in-stage fp16 accumulation the rms of the 128-term dot
        //   product is ~1.8e5 ≫ 65504, so FA=1 goes straight to NaN (both shapes blew up in the FFN round).
        // PH=0: qkv is re-read right away by S_PHASE_3/S_PHASE_4, so it must not be marked evict_first.
        fwam_fp8_gemm_body<FWAM_VDIT_SELF_S_PHASE_2_TILES>(
            /*A=*/a8x, /*W=*/wqkv, /*sa=*/sa0x, /*sw=*/swqkv,
            /*bias=*/reinterpret_cast<const __nv_bfloat16*>(bqkv),
            /*Cout=*/reinterpret_cast<__nv_bfloat16*>(qkv),
            /*partial=*/nullptr, /*counters=*/nullptr,
            /*M=*/VDIT_SELF_M, /*N=*/VDIT_SELF_N, /*K=*/VDIT_SELF_H, /*Kseg=*/VDIT_SELF_H,
            /*wp=*/11, /*ap=*/0, /*col0=*/0, /*ffn_ex=*/nullptr);
    }
    VDIT_SELF_MARK(3);
    VDIT_SELF_GRID_SYNC();
    VDIT_SELF_MARK(4);

    // ================= S_PHASE_3: full-row qk-norm + 3-D RoPE (one row per CTA, b<120) =================
    // Follows `WanRMSNorm`: the RMS spans the **whole 3072-wide row** (not per-head), with an fp32 statistic.
    // Unlike the action kernel's S_PHASE_3+S_PHASE_4: there 32 rows need 24 CTAs to each scan one head and then atomicAdd,
    // while here 120 rows ≤ grid → **one CTA per full row, no atomic, no intermediate sync**.
    if (b < VDIT_SELF_M && VDIT_SELF_PHASE(3) && (stop_phase == 0 || stop_phase > 3)) {
        const int r = b;
        const __nv_bfloat16* __restrict__ qr =
            reinterpret_cast<const __nv_bfloat16*>(qkv + (size_t)r * VDIT_SELF_N * 2);
        float* sc = reinterpret_cast<float*>(va_smem);

        // ---- pass 1: read q/k into registers (in pairs) + row sum of squares ----
        constexpr int NPAIR = (VDIT_SELF_H3 / 2) / VDIT_SELF_NT;     // 1536/256 = 6
        float qa[NPAIR], qb[NPAIR], ka[NPAIR], kb[NPAIR];
        float ssq = 0.f, ssk = 0.f;
        #pragma unroll
        for (int i = 0; i < NPAIR; ++i) {
            const int m = tid + VDIT_SELF_NT * i;
            const int c = 2 * m;
            qa[i] = bf16f(__bfloat16_as_ushort(qr[c]));
            qb[i] = bf16f(__bfloat16_as_ushort(qr[c + 1]));
            ka[i] = bf16f(__bfloat16_as_ushort(qr[VDIT_SELF_H3 + c]));
            kb[i] = bf16f(__bfloat16_as_ushort(qr[VDIT_SELF_H3 + c + 1]));
            ssq += qa[i] * qa[i] + qb[i] * qb[i];
            ssk += ka[i] * ka[i] + kb[i] * kb[i];
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            ssq += __shfl_xor_sync(0xffffffffu, ssq, off);
            ssk += __shfl_xor_sync(0xffffffffu, ssk, off);
        }
        if (lane == 0) { sc[wid] = ssq; sc[VDIT_SELF_NWARP + wid] = ssk; }
        __syncthreads();
        if (tid == 0) {
            float a = 0.f, v = 0.f;
            #pragma unroll
            for (int i = 0; i < VDIT_SELF_NWARP; ++i) { a += sc[i]; v += sc[VDIT_SELF_NWARP + i]; }
            sc[2 * VDIT_SELF_NWARP] = rsqrtf(a / (float)VDIT_SELF_H3 + norm_eps);   // rq
            sc[2 * VDIT_SELF_NWARP + 1] = rsqrtf(v / (float)VDIT_SELF_H3 + norm_eps);  // rk
        }
        __syncthreads();
        // volatile (the same form as `vdit_cross_row_rms` in `vdit_attn_cross.cu`): forces this read
        // to happen **after** `__syncthreads()`. That kernel measured it: if sc is taken as "written only by this thread", the compiler hoists
        // the read ahead of tid0's write, and the full-row q/k rstd reads back the stale value left in smem by the previous round.
        // This is the same "single-thread write + whole-block broadcast read" structure; the form is kept aligned to guard against the same miscompile;
        // NOTE: it is **not** the root cause of the known ~1e-4/layer probabilistic error (a static reorder would fail every time, not intermittently).
        const float rq = ((volatile float*)sc)[2 * VDIT_SELF_NWARP];
        const float rk = ((volatile float*)sc)[2 * VDIT_SELF_NWARP + 1];

        // ---- pass 2: qk-norm (two roundings, aligned with torch) + RoPE (fp32) + write ----
        __nv_bfloat16* __restrict__ qo =
            reinterpret_cast<__nv_bfloat16*>(qc + (size_t)r * VDIT_SELF_H3 * 2);
        __nv_bfloat16* __restrict__ ko =
            reinterpret_cast<__nv_bfloat16*>(kc + (size_t)r * VDIT_SELF_H3 * 2);
        const __nv_bfloat16* __restrict__ wn = reinterpret_cast<const __nv_bfloat16*>(wnq);
        const __nv_bfloat16* __restrict__ wk = reinterpret_cast<const __nv_bfloat16*>(wnk);
        const float* __restrict__ ct = cos_t + r * 64;
        const float* __restrict__ st = sin_t + r * 64;
        #pragma unroll
        for (int i = 0; i < NPAIR; ++i) {
            const int m = tid + VDIT_SELF_NT * i;
            const int c = 2 * m;
            const int p = m & 63;                 // pair index within the head (64 pairs/head)
            const float cth = ct[p], sth = st[p];
            // Follows `WanRMSNorm.forward` = `_norm(x.float()).type_as(x) * weight`:
            // the normalization is cast back to bf16 first and only then multiplied by the bf16 weight —— **two roundings**.
            // (the action fused kernel writes a triple product with a single rounding, a known last-ulp bias source;
            //   here the rounding points follow torch verbatim, at zero extra cost.)
            const float nqa = fwam_bf16r(fwam_bf16r(qa[i] * rq) *
                                         bf16f(__bfloat16_as_ushort(wn[c])));
            const float nqb = fwam_bf16r(fwam_bf16r(qb[i] * rq) *
                                         bf16f(__bfloat16_as_ushort(wn[c + 1])));
            qo[c] = __float2bfloat16_rn(nqa * cth - nqb * sth);
            qo[c + 1] = __float2bfloat16_rn(nqa * sth + nqb * cth);
            const float nka = fwam_bf16r(fwam_bf16r(ka[i] * rk) *
                                         bf16f(__bfloat16_as_ushort(wk[c])));
            const float nkb = fwam_bf16r(fwam_bf16r(kb[i] * rk) *
                                         bf16f(__bfloat16_as_ushort(wk[c + 1])));
            ko[c] = __float2bfloat16_rn(nka * cth - nkb * sth);
            ko[c + 1] = __float2bfloat16_rn(nka * sth + nkb * cth);
        }
        // ---- v as-is (not normalized, no RoPE) ----
        __nv_bfloat16* __restrict__ vo =
            reinterpret_cast<__nv_bfloat16*>(vc + (size_t)r * VDIT_SELF_H3 * 2);
        #pragma unroll
        for (int i = 0; i < VDIT_SELF_H3 / 2 / VDIT_SELF_NT; ++i)
            reinterpret_cast<uint32_t*>(vo)[tid + VDIT_SELF_NT * i] =
                reinterpret_cast<const uint32_t*>(qr + 2 * VDIT_SELF_H3)[tid + VDIT_SELF_NT * i];
    }
    VDIT_SELF_MARK(5);
    VDIT_SELF_GRID_SYNC();
    VDIT_SELF_MARK(6);

    // ================= S_PHASE_4: bf16 flash (24 heads × 4 q-tiles = 96 CTAs) =================
    // CTA = (head, q-tile), 32 rows per q-tile → reuses the action kernel's (mh, dq) warp split and
    // the "warps with dq<2 share QKᵀ through sS" trick (the QK mma drops to 1/4).
    // k/v stream entirely from the global kc/vc (the action kernel's "video-cache prefix + 2 fresh tiles"
    // degenerates here to a plain 8 tiles: 120 = 7×16 + 8).
    if (b < VDIT_SELF_FGRID && VDIT_SELF_PHASE(4) && (stop_phase == 0 || stop_phase > 4)) {
        const int h = b >> 2;
        const int qt = b & 3;
        const int w = tid >> 5;
        const int lane2 = tid & 31;

        __nv_bfloat16* fqs = reinterpret_cast<__nv_bfloat16*>(va_smem);              // 32*136
        __nv_bfloat16* fkt = fqs + VDIT_SELF_QT * VDIT_SELF_BP;                                    // 2*16*136
        __nv_bfloat16* fvt = fkt + VDIT_SELF_NSTAGE * VDIT_SELF_TILE * VDIT_SELF_BP;                      // 2*16*136
        // ⚠️ sS is **2×256 float** (the action kernel declares `float sS[2][2*2*16*4]`),
        // not 2×128 —— when flattened each half spans 256 floats, so writing +128 goes out of bounds and corrupts
        // and the value read back is the tail of the other half (every score wrong → exp overflow → l=inf → the output is always 0).
        float* fsS = reinterpret_cast<float*>(fvt + VDIT_SELF_NSTAGE * VDIT_SELF_TILE * VDIT_SELF_BP);
        constexpr int VDIT_SELF_SS = 2 * 2 * 16 * 4;   // 256: the length of each even/odd column buffer

        // Load this CTA's 32 q rows (this head's 128 columns): 32×128 bf16 = 8KB
        // each row is 128 columns × 2B = 256B = **16 chunks of 16B**, so 32 rows give 512 chunks → each of the 256 threads moves 2.
        // (missing the factor 2 moves only the first 64 columns —— that error does not crash, it just makes the second half of the attention all 0.)
        // The last q-tile has only 24 valid rows; the extra 8 **must be zero-filled rather than reading** qc out of bounds.
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int chunk = tid + VDIT_SELF_NT * i;  // 0..511
            const int rr = chunk >> 4;          // 0..31
            const int j = chunk & 15;           // 0..15 → the 16 chunks of 16B
            const int grow = qt * VDIT_SELF_QT + rr;
            uint4 val = make_uint4(0u, 0u, 0u, 0u);
            if (grow < VDIT_SELF_M) {
                const uint8_t* src = qc +
                    ((size_t)grow * VDIT_SELF_H3 + h * VDIT_SELF_D) * 2 + j * 16;
                val = *reinterpret_cast<const uint4*>(src);
            }
            *reinterpret_cast<uint4*>(fqs + rr * VDIT_SELF_BP + j * 8) = val;
        }

        const int mh = w >> 2;                  // 0..1
        const int dq = w & 3;                   // 0..3
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
            const __nv_bfloat16* qb = fqs + 16 * mh * VDIT_SELF_BP;
            float s[2][4], s2[2][4];
            #pragma unroll
            for (int n8 = 0; n8 < 2; ++n8)
                #pragma unroll
                for (int i = 0; i < 4; ++i) { s[n8][i] = 0.f; s2[n8][i] = 0.f; }
            const int q_row = lane2 & 15, q_c16 = lane2 >> 4;
            if (dq < 2) {   // only the 4 warps with dq∈{0,1} each compute one n8 block
                const int n8q = dq;
                const int k2_row = lane2 & 7, k2_col = (lane2 >> 3) & 1;
                const __nv_bfloat16* kpb = ktile + (n8q * 8 + k2_row) * VDIT_SELF_BP + 8 * k2_col;
                float c0v[4] = {0.f, 0.f, 0.f, 0.f}, c1v[4] = {0.f, 0.f, 0.f, 0.f};
                #pragma unroll
                for (int kc = 0; kc < 8; ++kc) {
                    unsigned a0, a1, a2, a3, b0, b1;
                    ldm_x4(qb + q_row * VDIT_SELF_BP + 16 * kc + 8 * q_c16, a0, a1, a2, a3);
                    ldm_x2(kpb + 16 * kc, b0, b1);
                    if (kc & 1) mma_f32(c1v, a0, a1, a2, a3, b0, b1);
                    else        mma_f32(c0v, a0, a1, a2, a3, b0, b1);
                }
                c0v[0] += c1v[0]; c0v[1] += c1v[1]; c0v[2] += c1v[2]; c0v[3] += c1v[3];
                float* sp0 = &fsS[((mh * 2 + n8q) * 16 + gid) * 4 + tig];
                float* sp1 = &fsS[VDIT_SELF_SS + ((mh * 2 + n8q) * 16 + gid) * 4 + tig];
                sp0[0] = c0v[0]; sp1[0] = c0v[1];
                sp0[8 * 4] = c0v[2]; sp1[8 * 4] = c0v[3];
            }
            __syncthreads();
            #pragma unroll
            for (int n8 = 0; n8 < 2; ++n8) {
                const int rb = ((mh * 2 + n8) * 16 + gid) * 4 + tig;
                s[n8][0] = fsS[rb];              s[n8][1] = fsS[VDIT_SELF_SS + rb];
                s[n8][2] = fsS[rb + 8 * 4];      s[n8][3] = fsS[VDIT_SELF_SS + rb + 8 * 4];
            }
            // Aligned with torch SDPA: score *= 1/sqrt(head_dim)
            #pragma unroll
            for (int n8 = 0; n8 < 2; ++n8)
                #pragma unroll
                for (int i = 0; i < 4; ++i) s[n8][i] *= 0.08838834764831845f;
            // ⚠️ kv pad rows must be set to -inf: this phase has **no mask** (production prefill is all True),
            // so skipping this step lets pad columns enter the softmax denominator with an exp(0-m) weight —— a **silent error**.
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
            // ⚠️ the packing **must use the bit pattern**: the action kernel's `bf16_bits` returns unsigned short (the bit pattern), whereas
            // the reusable `fwam_bf16r` in this file returns a **value converted back to float**. Writing
            // `(unsigned)fwam_bf16r(v)` truncates 1.0f to the integer 1 (= the smallest subnormal bf16 bit pattern),
            // so P entering the PV mma is about 0 —— the output is always 0 and nothing errors out (actually hit in 2026-09).
            pa[0] = (unsigned)bf16_bits(e[0][0]) | ((unsigned)bf16_bits(e[0][1]) << 16);
            pa[1] = (unsigned)bf16_bits(e[0][2]) | ((unsigned)bf16_bits(e[0][3]) << 16);
            pa[2] = (unsigned)bf16_bits(e[1][0]) | ((unsigned)bf16_bits(e[1][1]) << 16);
            pa[3] = (unsigned)bf16_bits(e[1][2]) | ((unsigned)bf16_bits(e[1][3]) << 16);
            unsigned bf[4][2];
            {
                const int v_row = lane2 & 15, v_c16 = lane2 >> 4;
                const __nv_bfloat16* vbp = vtile + v_row * VDIT_SELF_BP + dq * 32 + 8 * v_c16;
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

        const int ntiles = (VDIT_SELF_M + VDIT_SELF_TILE - 1) / VDIT_SELF_TILE;    // 8
        const uint32_t* kvu = reinterpret_cast<const uint32_t*>(kc);
        const uint32_t* vvu = reinterpret_cast<const uint32_t*>(vc);
        auto issue_tile = [&](int stage, int tb) {
            const int cc = tid >> 4;
            const int j = tid & 15;
            const int g = tb * VDIT_SELF_TILE + cc;
            __nv_bfloat16* kdst = &fkt[stage * VDIT_SELF_TILE * VDIT_SELF_BP + cc * VDIT_SELF_BP + j * 8];
            __nv_bfloat16* vdst = &fvt[stage * VDIT_SELF_TILE * VDIT_SELF_BP + cc * VDIT_SELF_BP + j * 8];
            if (g < VDIT_SELF_M) {
                const uint32_t* gk = kvu + (size_t)g * (VDIT_SELF_H3 / 2) + h * (VDIT_SELF_D / 2) + j * 4;
                unsigned dk = (unsigned)__cvta_generic_to_shared(kdst);
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dk), "l"(gk));
                const uint32_t* gv = vvu + (size_t)g * (VDIT_SELF_H3 / 2) + h * (VDIT_SELF_D / 2) + j * 4;
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
        __syncthreads();          // only enter the pipeline after fqs is written
        issue_tile(0, 0);
        if (ntiles == 1) asm volatile("cp.async.commit_group;\n" ::);
        else if (ntiles > 1) issue_tile(1, 1);
        for (int tb = 0; tb < ntiles; ++tb) {
            const int stage = tb & 1;
            // The tail iteration uses wait_group 0: a completed empty commit group occupies no pending slot, so wait_group 1 would let through
            // smem that has not landed (the tail-tile dirty-read root cause the action kernel pinned down in 2026-09, carried over as-is).
            if (tb + 1 == ntiles) asm volatile("cp.async.wait_group 0;\n" ::);
            else                  asm volatile("cp.async.wait_group 1;\n" ::);
            __syncthreads();
            flash_tile(fkt + stage * VDIT_SELF_TILE * VDIT_SELF_BP, fvt + stage * VDIT_SELF_TILE * VDIT_SELF_BP,
                       min(VDIT_SELF_TILE, VDIT_SELF_M - tb * VDIT_SELF_TILE));
            if (dbg && tb == 0 && tid == 0) {
                const __nv_bfloat16* kz = fkt + stage * VDIT_SELF_TILE * VDIT_SELF_BP;
                const __nv_bfloat16* vz = fvt + stage * VDIT_SELF_TILE * VDIT_SELF_BP;
                dbg[b * 8 + 0] = bf16f(__bfloat16_as_ushort(fqs[0]));
                dbg[b * 8 + 1] = bf16f(__bfloat16_as_ushort(fqs[127]));
                dbg[b * 8 + 2] = bf16f(__bfloat16_as_ushort(kz[0]));
                dbg[b * 8 + 3] = bf16f(__bfloat16_as_ushort(kz[127]));
                dbg[b * 8 + 4] = bf16f(__bfloat16_as_ushort(vz[0]));
                dbg[b * 8 + 5] = bf16f(__bfloat16_as_ushort(vz[127]));
                dbg[b * 8 + 6] = l0;
                dbg[b * 8 + 7] = l1;
            }
            __syncthreads();
            if (tb + 2 < ntiles) issue_tile(stage, tb + 2);
            else                 asm volatile("cp.async.commit_group;\n" ::);
        }

        // ---- normalize + write attn (**fp32**: the reference o-proj quantizes from fp32 with a single RNE) ----
        const float inv0 = 1.0f / l0, inv1 = 1.0f / l1;
        float* ou = reinterpret_cast<float*>(attn);
        const int orow = qt * VDIT_SELF_QT + 16 * mh + gid;
        const int colbase = h * VDIT_SELF_D + dq * 32;
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const int cp = colbase + 8 * nt + 2 * tig;
            // orow ≤ 119 always holds (qt≤3 → 96+16+7), but orow+8 reaches 127 → the last slot must bounds-check
            if (orow < VDIT_SELF_M) {
                ou[(size_t)orow * VDIT_SELF_H3 + cp] = acc[nt][0] * inv0;
                ou[(size_t)orow * VDIT_SELF_H3 + cp + 1] = acc[nt][1] * inv0;
            }
            if (orow + 8 < VDIT_SELF_M) {
                ou[(size_t)(orow + 8) * VDIT_SELF_H3 + cp] = acc[nt][2] * inv1;
                ou[(size_t)(orow + 8) * VDIT_SELF_H3 + cp + 1] = acc[nt][3] * inv1;
            }
        }
    }
    VDIT_SELF_MARK(7);
    VDIT_SELF_GRID_SYNC();

    // ================= S_PHASE_5: attn row quantization (one row per CTA, b<120) =================
    if (b < VDIT_SELF_M && VDIT_SELF_PHASE(5) && (stop_phase == 0 || stop_phase > 5)) {
        const int r = b;
        const float* __restrict__ ar = reinterpret_cast<const float*>(attn) +
                                       (size_t)r * VDIT_SELF_H3;
        float* sc = reinterpret_cast<float*>(va_smem);
        constexpr int NV = VDIT_SELF_H3 / VDIT_SELF_NT;      // 12
        float v[NV];
        float mx = 0.f;
        #pragma unroll
        for (int i = 0; i < NV; ++i) {
            v[i] = ar[tid + VDIT_SELF_NT * i];
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
            for (int i = 0; i < VDIT_SELF_NWARP; ++i) m = fmaxf(m, sc[i]);
            sc[VDIT_SELF_NWARP] = fmaxf(m / 448.0f, 1e-12f);
        }
        __syncthreads();
        const float sa = sc[VDIT_SELF_NWARP];
        sa0a[r] = sa;
        uint8_t* __restrict__ ao = a8a + (size_t)r * VDIT_SELF_H3;
        // Element-wise **true division**, the same semantics as the production Triton `x.to(f32)/scale`
        // (division→multiply-by-reciprocal was measured: not faster, and it adds one rounding; see the FFN round's conclusion).
        #pragma unroll
        for (int i = 0; i < NV; ++i) ao[tid + VDIT_SELF_NT * i] = fp8_rne(v[i] / sa);
    }
    VDIT_SELF_MARK(8);
    VDIT_SELF_GRID_SYNC();

    // ================= S_PHASE_6: o fp8 GEMM + gate residual (RESID) =================
    if (VDIT_SELF_PHASE(6) && (stop_phase == 0 || stop_phase > 6)) {
        fwam_gemm_extra ex{x_in, gate_msa, VDIT_SELF_H, mod_srow};
        fwam_fp8_gemm_body<FWAM_VDIT_SELF_S_PHASE_6_TILES>(
            /*A=*/a8a, /*W=*/wo, /*sa=*/sa0a, /*sw=*/swo,
            /*bias=*/reinterpret_cast<const __nv_bfloat16*>(bo),
            /*Cout=*/reinterpret_cast<__nv_bfloat16*>(out),
            /*partial=*/nullptr, /*counters=*/nullptr,
            /*M=*/VDIT_SELF_M, /*N=*/VDIT_SELF_H, /*K=*/VDIT_SELF_H3, /*Kseg=*/VDIT_SELF_H3,
            /*wp=*/11, /*ap=*/0, /*col0=*/0, /*ffn_ex=*/&ex);
    }
    VDIT_SELF_MARK(9);
}

// ---- The grid comes from the device capacity (occupancy × SM count), no longer hard-coded to 132 = 2×66 ----
// Structural lower bound = VDIT_SELF_M (S_PHASE_1/S_PHASE_3/S_PHASE_5 are all `if (b < VDIT_SELF_M)`, one row per CTA; grid<120 drops rows).
static int g_va_grid = -1;
static int va_grid() {
    return cached_coop_grid(g_va_grid, (const void*)vdit_attn_self_kernel, VDIT_SELF_NT,
                            VDIT_SELF_SMEM, VDIT_SELF_M, "vdit_attn_self");
}
extern "C" int vdit_attn_self_grid() { return va_grid(); }
// The actual grid of the non-cooperative split-phase form (lower bound = VDIT_SELF_M; a plain launch may take multiple waves, so small GPUs can run it too)
static int va_split_grid();   // defined further down in this file (one each for the cooperative/split grid)
extern "C" int vdit_attn_self_split_grid() { return va_split_grid(); }

// The non-cooperative fallback's grid: it does not require co-residency, so it may take multiple waves; but the row/flash phases are fixed
// one-CTA-per-item claims (`if (b < VDIT_SELF_M)` / `b < VDIT_SELF_FGRID`), so the grid must be >= the structural lower bound or it **silently drops work**.
static int g_va_split_grid = -1;
static int va_split_grid() {
    return cached_split_grid(g_va_split_grid, (const void*)vdit_attn_self_kernel, VDIT_SELF_NT,
                             VDIT_SELF_SMEM, VDIT_SELF_M, "vdit_attn_self");
}

// ------------------------------------------------------------------ //
// host: cooperative launch
// ------------------------------------------------------------------ //

// ---- Geometry self-report (shape constants + GEMM template parameters; the template parameters use the same macro set as the instantiation) ----
extern "C" const char* vdit_attn_self_geom() {
    static const std::string s = fwam_geom_join({
        fwam_shapes({{"M", VDIT_SELF_M}, {"H", VDIT_SELF_H}, {"N", VDIT_SELF_N}, {"NH", VDIT_SELF_NH}, {"D", VDIT_SELF_D}, {"NT", VDIT_SELF_NT}, {"SMEM", VDIT_SELF_SMEM}, {"GRID", VDIT_SELF_GRID}}),
        fwam_tiles("S_PHASE_2", FWAM_TILES(FWAM_VDIT_SELF_S_PHASE_2_TILES)),
        fwam_tiles("S_PHASE_6", FWAM_TILES(FWAM_VDIT_SELF_S_PHASE_6_TILES)),
    });
    return s.c_str();
}

extern "C" void vdit_attn_self_cuda(
    const void* x_in, const void* shift_msa, const void* scale_msa, const void* gate_msa,
    const void* wqkv, const float* swqkv, const void* bqkv,
    const void* wnq, const void* wnk,
    const void* cos_t, const void* sin_t,
    const void* wo, const float* swo, const void* bo,
    void* out, void* qkv, void* qc, void* kc, void* vc, void* attn,
    void* a8x, float* sa0x, void* a8a, float* sa0a,
    float norm_eps, int mod_srow, int stop_phase, int only_phase,
    float* dbg, unsigned long long* phase_ts, cudaStream_t stream)
{
    dim3 block(VDIT_SELF_NT);
    void* args[] = {
        (void*)&x_in, (void*)&shift_msa, (void*)&scale_msa, (void*)&gate_msa,
        (void*)&wqkv, (void*)&swqkv, (void*)&bqkv,
        (void*)&wnq, (void*)&wnk, (void*)&cos_t, (void*)&sin_t,
        (void*)&wo, (void*)&swo, (void*)&bo,
        (void*)&out, (void*)&qkv, (void*)&qc, (void*)&kc, (void*)&vc, (void*)&attn,
        (void*)&a8x, (void*)&sa0x, (void*)&a8a, (void*)&sa0a,
        (void*)&norm_eps, (void*)&mod_srow, (void*)&stop_phase, (void*)&only_phase,
        (void*)&dbg, (void*)&phase_ts,
    };
    cudaError_t e;
    dim3 grid;                       // declared outside: the error path needs to report the grid
    if (only_phase < 0) {
        grid = dim3(va_grid());
        e = cudaLaunchCooperativeKernel((void*)vdit_attn_self_kernel, grid,
                                        block, args, VDIT_SELF_SMEM, stream);
    } else {
        grid = dim3(va_split_grid());
        e = cudaLaunchKernel((void*)vdit_attn_self_kernel, grid, block, args,
                             VDIT_SELF_SMEM, stream);
    }
    if (e != cudaSuccess) {
        // Throw instead of exit(1): exit would kill the worker process outright and the caller would get no handleable error.
        coop_fail("vdit_attn_self",
                  "launch failed (only_phase=" + std::to_string(only_phase)
                  + ", grid=" + std::to_string((int)grid.x) + "): "
                  + cudaGetErrorString(e));
    }
}
