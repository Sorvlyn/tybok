// Shared GEMM core of the video fused kernels (used by self/cross/ffn): device-side helpers +
// fwam_fp8_gemm_body, without host dispatch / sweep instantiations.
//
// Semantics: out[m,n] = RN( (Σ_k A8[m,k]·W8[n,k]) · sa[m] · sw[n] + bias[n] ).
//
// Structure:
//   * One template kernel serves both the plain form (SPLIT=1, writes bf16 directly) and split-K
//     (SPLIT>1, fp32 partial + per-tile atomic counter, the last block reduces); no second launch,
//     and the merge data stays inside L2.
//   * 2D warp partitioning WM × WN (NWARPS = WM*WN); with 8 warps, 4×2 has the least B redundancy.
//   * smem crosswise layout (A 16x32 core 512B / B 8x32 core 256B, no PAD), the ldmatrix fragment
//     mapping of the fp8 mma m16n8k32.
//   * PERSIST=1: 1-D grid (gridDim.x = G), each CTA grid-strides serially over several tiles along
//     t = m + n*MT (m varies fastest → the two m-tiles of the same n are adjacent, so the W tile is shared through L2).
//   * RESID=1: the epilogue appends o = bf16RN(bf16RN(z*gate) + x_in) (= torch x + gate*ffn(x)).
//   * wp/ap encode the runtime L2 eviction policy; tagging W with evict_first 0.75 saves ~10µs.
//   * BM row mask: when M < BM the out-of-bounds rows are cp.async zero-filled, and the epilogue
//     only writes r < M.
//
// WARNING: do not enable --use_fast_math (it replaces tanhf with __tanhf and x/sa with a reciprocal multiply).
// Epilogue: acc*sa[m]*sw[n] (left-associative) + bf16 bias (promoted to fp32), then RN to bf16.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cassert>

#include "common.h"
#include "gemm_common.h"   // fwam_bf16r / fwam_resid_bf16 / fwam_cp16 / fwam_gt (shared by vdit+tmt5)

// --------------------------------------------------------------------------- //
// Main kernel
// --------------------------------------------------------------------------- //
// [L2 eviction policy -- the core increment of this kernel over the main kernels/ directory]
// up's W=44MB is streaming data that is read once and thrown away, yet L2 holds only 48MB. With the
// default (no policy) every W miss claims an L2 way by LRU, and under a "dirty L2" (the do_bench
// convention: cache.zero_() leaves 48MB of dirty lines) it pushes the dirty lines out → 44MB of
// writeback traffic competes with the read stream for the DRAM bus. Measured:
//   dram__bytes_write 13.23MB → 3.04MB (writeback essentially disappears), duration 110.1 → 95.8μs.
// Two ways to apply it (cfg.wp / cfg.ap, runtime):
//   wp=1..3  : cp.async.L2::cache_hint inside the kernel, eviction priority evict_first/last/unchanged
//   wp>=100  : a host-side cudaStreamSetAttribute access-policy window blankets W's address range
//              with hitProp=Streaming; the ones digit of wp-100 picks the property (1=Streaming
//              2=Persisting), the tens digit picks hitRatio (0=1.0 1=0.75 2=0.5 3=0.25).
//              **Zero instruction overhead**, and measured not to hurt the hot convention (the
//              instruction-level hint pins W to evict_first, degrading hot from 72.7 to 88).
//              wp=101 is recommended.
// Note: the window is per-stream persistent state and must be reset after launch (see launch_cfg).
// --------------------------------------------------------------------------- //
// upgemm structural surgery: a single stage ring → independent depths for the A ring (SA stages)
// and the B ring (SB stages).
// Rationale: A is reused by N/BN=224 n-tiles and mostly hits in L2 (low latency, so a shallow ring
// is enough), while B=W carries the bulk of the DRAM traffic and, when cold, must get past
// dirty-line writeback (a deeper ring = a longer prefetch distance = a smoother request stream).
// Commit is still one commit group per round (A and B in the same group), so the FIFO semantics of
// wait_group(SB-2) automatically guarantee that the A ring has completed too (SB >= SA).
template <int BM, int BN, int BK, int SA, int SB, int WM, int WN, int SPLIT, int EV,
          int G2 = 0,    // G2=1: split each stage into two cp.async commits, A/B (triton cadence)
          int LIN = 1,   // LIN=1: cp.async stores in dst-linear order (eliminates store bank conflicts)
          int ORD = 0,   // ORD=0: issue the next stage after mma (measured better); 1: issue right after the barrier
          int APATH = 0, // APATH=1: A does not use cp.async -- LDG→reg (1 stage ahead)→STS into smem; B still uses cp.async
          int EPI = 0,   // EPI=1: fused epilogue -- u=bf16RN(acc*sa*sw+bias) → g=bf16RN(gelu(u))
                         //        → write gbuf(Cout) + per-row |g| atomicMax (partial slot → raw[M]); SPLIT=1 only
          int FA = 0,   // FA=1: in-stage fp16 accumulation (sm_89 fp8+f16acc = 2× tensor throughput), promoted to fp32 at stage end
                         //        numeric precondition: each stage's k-segment partial sum |Σ| < 65504 (the fp16 range)
          int PH = 0,    // PH=1: partial/Cout stores carry an L2::evict_first hint (stored data does not steal W/A's L2 ways)
          int LDB = 0,   // LDB=1: double-buffer the ldmatrix fragments (overlaps the LSU with the tensor pipeline)
          int PD = 0,    // PD=1: in split mode write the fragment straight to partial (skips smem staging + one barrier)
          int XS = 0,    // XS=1: triton-style XOR-swizzle smem layout (4KB window, src row-major contiguous,
                         //       best cp.async store / L2 coalescing; requires BK%64==0, BM%64==0)
          int F16P = 0,  // F16P=1: fp16 partials (halves the traffic; requires |Kseg partial sums| < 65504)
          int FAP = 1,   // FAP>1: the promotion period of FA (promote once every FAP stages, saving promotion ALU;
                         //          the cost is an fp16 accumulation length of ×FAP)
          int ACG = 0,   // A ring cp.async uses .cg (bypasses L1) instead of .ca
          int WCG = 0,   // W ring cp.async uses .cg (bypasses L1) instead of .ca
          int AHINT = 1, // whether the A ring cp.async carries L2::cache_hint (off = the eviction policy stops working)
          int WHINT = 1, // same as above for the W ring
          int BORD = 0,  // 0=A issued first; 1=B(W) issued first (W is on the DRAM critical path)
          int WPF = 0,   // L2 prefetch granularity of the W ring cp.async: 0=none 128/256
          // PERSIST=1: persistent form -- the grid becomes 1D (gridDim.x = G) and each CTA serially
          // processes several (m,n) tiles by grid-stride. Tile linearization t = m + n*MT (m varies
          // fastest) keeps the two m-tiles of the same n on adjacent CTAs → the W tile is shared in
          // L2 (with BM=64, W is nominally read twice while DRAM still reads it once). It answers the
          // question "how much does the cooperative launch's grid upper bound (≤132) lose" -- with
          // PERSIST=0 it degrades at compile time and the SASS/behavior is verbatim-identical to before.
          int PERSIST = 0,
          // RESID=1: append the FFN residual at the end of the epilogue, o = bf16RN(bf16RN(z*gate[c]) + x_in[r,c])
          // (matches torch: out = x + gate * ffn(x); same shape as the action kernel). Requires ffn_ex.
          int RESID = 0>
__device__ __forceinline__ void fwam_fp8_gemm_body(
                     const uint8_t* __restrict__ A,          // [M, K] fp8 row-major
                     const uint8_t* __restrict__ W,          // [N, K] fp8 row-major
                     const float* __restrict__ sa,           // [M]
                     const float* __restrict__ sw,           // [N]
                     const __nv_bfloat16* __restrict__ bias, // [N]
                     __nv_bfloat16* __restrict__ Cout,       // [M, N]
                     float* __restrict__ partial,            // [SPLIT][M][N] fp32
                     int* __restrict__ counters,             // per output tile
                     const int M, const int N, const int K,  // K: gmem row stride
                     const int Kseg,
                     const int wp,   // W ring L2 policy: 0=evict_normal 1=evict_first 2=evict_last 3=evict_unchanged
                     const int ap,   // A ring L2 policy: same as above
                     const int col0, // N column offset of this launch (used when splitting a mixed tile)
                     const fwam_gemm_extra* __restrict__ ffn_ex)  // used by RESID (nullptr otherwise)
{
    constexpr int NTHREADS = WM * WN * 32;
    constexpr int NWARPS = WM * WN;
    static_assert(SA >= 2 && SB >= 2 && BK % 32 == 0 && BM % (16 * WM) == 0 &&
                  BN % (8 * WN) == 0, "tile/warp constraint");
    static_assert(!APATH || (LIN && !G2), "APATH variant needs LIN && !G2");
    static_assert(!EPI || SPLIT == 1, "EPI (fused epilogue) only in plain mode");
    static_assert(!PD || SPLIT > 1, "PD only in split mode (direct partial write)");

    constexpr int M16 = BM / 16;             // total number of m16 row bands
    constexpr int N8 = BN / 8;               // total number of n8 column bands
    static_assert(M16 % WM == 0 && N8 % WN == 0, "warp partition must divide evenly");
    constexpr int M16PW = M16 / WM;          // m16 row bands per warp
    constexpr int N8PW = N8 / WN;            // n8 column bands per warp
    constexpr int KCORES = BK / 32;          // 16x32 cores per stage along K

    // ---- XS (XOR-swizzle windowed layout) constants: window = 64 rows × 64B; cell within the window = t^((t>>3)&7) ----
    constexpr int KPARTS = BK / 64;                  // windows per 64-row block (XS requires BK%64==0)
    constexpr int NWA = (BM / 64) * KPARTS;          // number of windows in the A region
    constexpr int NWB = ((BN + 63) / 64) * KPARTS;   // number of windows in the B region (the tail row block may be partly invalid)
    static_assert(!XS || XS == 2 || (BK % 64 == 0 && BM % 64 == 0),
                  "XS=1 needs BK%64==0 and BM%64==0");
    static_assert(XS != 2 || BK % 32 == 0, "XS=2 needs BK%32==0");

    // ---- XS=2 (dense+pad): row-major dense, each row padded by 16B → SROWD=BK+16.
    //   36 words ≡ 4 (mod 32): 8 rows accessed at the same time land on banks {0,4,...,28}, 4 words
    //   each = full coverage of the 32 banks, so LDSM and cp.async conflict in neither direction;
    //   and the address is fully affine in (m16g, kk, lane) (loop invariants + a constant increment
    //   in kk), removing XS=1's ~85 swizzle integer ops per stage (measured loop body 399→~310).
    //   src is still row-major 16B contiguous (the loader is on a par with XS=1), a full 128B row per
    //   warp (better than XS=1's 64B).
    constexpr int SROWD = BK + 16;

    // crosswise: A core 16x32=512B, B core 8x32=256B (no PAD, zero bank conflicts)
    constexpr int A_BYTES = (XS == 2) ? BM * SROWD
                                       : (XS ? NWA * 4096 : BM * BK);   // B region offset
    constexpr int B_BYTES = (XS == 2) ? BN * SROWD
                           : (XS ? NWB * 4096 : BN * BK);              // one level of the B ring
    constexpr int STAGE_BYTES = A_BYTES + B_BYTES;                     // nominal size of one stage
    constexpr int SMEM_BYTES = A_BYTES * SA + B_BYTES * SB;            // independent A/B ring depths
    static_assert((size_t)SMEM_BYTES <= 200 * 1024, "smem over budget");

    extern __shared__ __align__(16) uint8_t smem_raw[];

    const int k_base = blockIdx.z * Kseg;
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    const int lane_g = lane >> 2;
    const int lane_t = lane & 3;
    const int warp_m = wid / WN;
    const int warp_n = wid % WN;

    // ---- PERSIST: the string of (m,n) tiles this CTA owns (exactly 1 when PERSIST=0) ----
    const int MT_ = (M + BM - 1) / BM;
    const int NT_ = (N + BN - 1) / BN;
    const int NTILES_ = MT_ * NT_;
    const int T0_ = PERSIST ? (int)blockIdx.x : (int)(blockIdx.x + blockIdx.y * MT_);
    const int TSTEP_ = PERSIST ? (int)gridDim.x : 1;
    const int TEND_ = PERSIST ? NTILES_ : T0_ + 1;
    for (int _t = T0_; _t < TEND_; _t += TSTEP_) {
    const int block_row = (_t % MT_) * BM;
    const int block_col = col0 + (_t / MT_) * BN;
    if (PERSIST && _t != T0_) __syncthreads();   // the previous tile's epilogue has finished reading the smem staging area

    const int Ktiles = Kseg / BK;

    constexpr int KQ = BK / 16;
    constexpr int A_CHUNKS = (XS == 1) ? NWA * 256 : BM * BK / 16;   // XS=1: 256 16B cells per window
    constexpr int B_CHUNKS = (XS == 1) ? NWB * 256 : BN * BK / 16;
    constexpr int A_ROUNDS = (A_CHUNKS + NTHREADS - 1) / NTHREADS;
    constexpr int B_ROUNDS = (B_CHUNKS + NTHREADS - 1) / NTHREADS;

    uint64_t pol_w = 0;
    if constexpr (EV == 1) {
        asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol_w));
    }
    // APATH==3: A's LDG carries L2::evict_last (A is broadcast and fully L2-resident, so the 42MB W stream cannot push it out)
    uint64_t pol_a = 0;
    if constexpr (APATH == 3) {
        asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol_a));
    }
    // PH=1: partial/Cout stores use evict_first -- the stored data does not steal L2 ways from the
    // W/A streams, so when hot W(44MB)+A stay resident in the 48MB L2 across iterations, avoiding the DRAM refill caused by self-eviction
    uint64_t pol_o = 0;
    if constexpr (PH == 1) {
        asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol_o));
    }
    // Read-stream L2 eviction priority (runtime): W is 44MB of single-pass streaming data, so let it
    // evict itself first rather than pushing out A (368KB, read repeatedly by 224 n-tiles) and the rest of L2.
    // Encoding: p%10 = priority (0 normal/1 first/2 last/3 unchanged); p/10 = hit-ratio tier
    //   0→1.0  1→0.75  2→0.5  3→0.25  4→0.125
    // Example: wp=1 → evict_first 1.0; wp=12 → evict_first 0.5.
    auto mkpol = [](int p) -> uint64_t {
        uint64_t v = 0;
        const int frac = p / 10;
        switch (p % 10) {
            case 1:
                if (frac == 1)      asm("createpolicy.fractional.L2::evict_first.b64 %0, 0.75;\n" : "=l"(v));
                else if (frac == 2) asm("createpolicy.fractional.L2::evict_first.b64 %0, 0.5;\n"  : "=l"(v));
                else if (frac == 3) asm("createpolicy.fractional.L2::evict_first.b64 %0, 0.25;\n" : "=l"(v));
                else if (frac == 4) asm("createpolicy.fractional.L2::evict_first.b64 %0, 0.125;\n": "=l"(v));
                else                asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n"  : "=l"(v));
                break;
            case 2:
                if (frac == 2)      asm("createpolicy.fractional.L2::evict_last.b64 %0, 0.5;\n"  : "=l"(v));
                else if (frac == 3) asm("createpolicy.fractional.L2::evict_last.b64 %0, 0.25;\n" : "=l"(v));
                else                asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n"  : "=l"(v));
                break;
            case 3: asm("createpolicy.fractional.L2::evict_unchanged.b64 %0, 1.0;\n" : "=l"(v)); break;
            default: asm("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;\n" : "=l"(v)); break;
        }
        return v;
    };
    const uint64_t pol_wrd = mkpol(wp);   // W ring read stream
    const uint64_t pol_ard = mkpol(ap);   // A ring read stream

    // ---- dst-linear chunk assignment (LIN=1): cp.async stores in smem-destination linear order →
    //      each warp instruction covers a contiguous 512B (8 lanes / 128B phase), eliminating
    //      cp.async store bank conflicts (measured: ~4.9M conflicts in the original mapping, all
    //      from op_ldgsts; triton ≈0).
    // ---- XS=1: windowed decomposition (src row-major contiguous, dst XOR-swizzle): a probe measured
    //      38.9μs of hot fetch vs 56.3μs for LIN-crosswise (4×128B full rows per warp of src vs 16×32B
    //      fragments).
    // Per-thread precomputation (loop-invariant): A's (gmem row, 16B column within k, destination 16B cell, mask)
    int arowk[A_ROUNDS], akgc[A_ROUNDS], adstu[A_ROUNDS];
    bool aval[A_ROUNDS], amask[A_ROUNDS];
    int browk[B_ROUNDS], bkgc[B_ROUNDS], bdstu[B_ROUNDS];
    bool bval[B_ROUNDS];
    if constexpr (XS == 2) {
        // dense+pad: u lays out the tile in row-major order; dst = row*SROWD + column (bytes), src is
        // row-major 16B contiguous. 8 consecutive threads = one full 128B row (KQ=8) → 4 whole rows of
        // src per warp (better than XS=1's 8×64B).
        #pragma unroll
        for (int j = 0; j < A_ROUNDS; ++j) {
            const int u = tid + j * NTHREADS;
            aval[j] = u < A_CHUNKS;
            if (aval[j]) {
                const int r = u / KQ;
                const int c16 = (u % KQ) * 16;
                arowk[j] = block_row + r;
                akgc[j] = c16;
                adstu[j] = (r * SROWD + c16) / 16;      // 16B cell index
                amask[j] = arowk[j] < M;
            }
        }
        #pragma unroll
        for (int j = 0; j < B_ROUNDS; ++j) {
            const int u = tid + j * NTHREADS;
            bval[j] = u < B_CHUNKS;
            if (bval[j]) {
                const int n = u / KQ;
                const int c16 = (u % KQ) * 16;
                browk[j] = block_col + n;
                bkgc[j] = c16;
                bdstu[j] = (n * SROWD + c16) / 16;
            }
        }
    } else if constexpr (XS) {
        // General chunk→window decomposition: u = tid + j*NTHREADS; w = u>>8; cellb = u&255
        // (supports 128/256/512 threads; dst = swizzle(cellb), src row-major)
        #pragma unroll
        for (int j = 0; j < A_ROUNDS; ++j) {
            const int u = tid + j * NTHREADS;
            const int w = u >> 8;
            aval[j] = w < NWA;
            if (aval[j]) {
                const int cellb = u & 255;
                const int cell = cellb ^ ((cellb >> 3) & 7);
                const int grow = (w / KPARTS) * 64 + (cellb >> 2);
                arowk[j] = block_row + grow;
                akgc[j] = (w % KPARTS) * 64 + (cellb & 3) * 16;
                adstu[j] = w * 256 + cell;                 // 16B cell index (×16 = bytes)
                amask[j] = grow < BM && arowk[j] < M;
            }
        }
        #pragma unroll
        for (int j = 0; j < B_ROUNDS; ++j) {
            const int u = tid + j * NTHREADS;
            const int w = u >> 8;
            const int cellb = u & 255;
            const int nl = (w / KPARTS) * 64 + (cellb >> 2);   // row within the window (W's local row index)
            bval[j] = w < NWB && nl < BN;
            if (bval[j]) {
                const int cell = cellb ^ ((cellb >> 3) & 7);
                browk[j] = block_col + nl;
                bkgc[j] = (w % KPARTS) * 64 + (cellb & 3) * 16;
                bdstu[j] = w * 256 + cell;
            }
        }
    } else if constexpr (LIN) {
        constexpr int A_UNITS_PER_ROWG = KCORES * 32;   // 16B cells per r_core
        #pragma unroll
        for (int j = 0; j < A_ROUNDS; ++j) {
            const int u = tid + j * NTHREADS;
            aval[j] = u < A_CHUNKS;
            if (aval[j]) {
                const int r_core = u / A_UNITS_PER_ROWG;
                const int t1 = u % A_UNITS_PER_ROWG;
                const int c_core = t1 / 32;
                const int t2 = t1 % 32;
                const int blk = t2 / 8;
                const int row8 = t2 % 8;
                const int r = r_core * 16 + ((blk & 1) << 3) + row8;
                const int kq = c_core * 2 + (blk >> 1);
                arowk[j] = block_row + r;
                akgc[j] = kq * 16;
                adstu[j] = u;
                amask[j] = arowk[j] < M;
            }
        }
    }
    if constexpr (LIN && !XS) {
        constexpr int B_UNITS_PER_NG = KCORES * 16;
        #pragma unroll
        for (int j = 0; j < B_ROUNDS; ++j) {
            const int u = tid + j * NTHREADS;
            if (u < B_CHUNKS) {
                const int n_core = u / B_UNITS_PER_NG;
                const int t1 = u % B_UNITS_PER_NG;
                const int c_core = t1 / 16;
                const int t2 = t1 % 16;
                const int col_half = t2 / 8;
                const int n8 = t2 % 8;
                const int n = n_core * 8 + n8;
                const int kq = c_core * 2 + col_half;
                browk[j] = block_col + n;
                bkgc[j] = kq * 16;
                bdstu[j] = u;
                bval[j] = true;
            }
        }
    }

    auto issue_stage_impl = [&](int aslot, int bslot, int kt, int part, auto self) {
        // part: 0=A half tile; 1=B half tile; 2=the whole stage
        uint8_t* baseA = smem_raw + (size_t)aslot * A_BYTES;
        uint8_t* baseB = smem_raw + (size_t)A_BYTES * SA + (size_t)bslot * B_BYTES;
        const int k_off = k_base + kt * BK;
        if (part != 1) {
            if constexpr (XS) {
                #pragma unroll
                for (int j = 0; j < A_ROUNDS; ++j) {
                    if (aval[j]) {
                        unsigned dst = (unsigned)__cvta_generic_to_shared(
                            baseA + (size_t)adstu[j] * 16);
                        const uint8_t* src = A + (size_t)arowk[j] * K + k_off + akgc[j];
                        fwam_cp16<ACG != 0, AHINT != 0>(dst, src, pol_ard, !amask[j]);
                    }
                }
            } else if constexpr (LIN) {
                #pragma unroll
                for (int j = 0; j < A_ROUNDS; ++j) {
                    if (aval[j]) {
                        unsigned dst = (unsigned)__cvta_generic_to_shared(
                            baseA + (size_t)adstu[j] * 16);
                        const uint8_t* src = A + (size_t)arowk[j] * K + k_off + akgc[j];
                        // EV=1: .cg + W evict_first (triton style; .ca is better on this machine, control only);
                        // EV=3: pure .cg (control); default (0): .ca (measured best on this machine).
                        if constexpr (EV == 1 || EV == 3) {
                            if (!amask[j]) {
                                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, 0;\n"
                                             :: "r"(dst), "l"(src));
                            } else {
                                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                                             :: "r"(dst), "l"(src));
                            }
                        } else {
                            if (!amask[j]) {
                                asm volatile("cp.async.ca.shared.global [%0], [%1], 16, 0;\n"
                                             :: "r"(dst), "l"(src));
                            } else {
                                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                                             :: "r"(dst), "l"(src));
                            }
                        }
                    }
                }
            } else {
                #pragma unroll
                for (int j = 0; j < A_ROUNDS; ++j) {
                    int chunk = tid + j * NTHREADS;
                    if (chunk < A_CHUNKS) {
                        int r = chunk / KQ;
                        int c = (chunk % KQ) * 16;
                        const int grow = block_row + r;
                        const int r_core = r >> 4;
                        const int c_core = c >> 5;
                        const int r16 = r & 15;
                        const int c32 = c & 31;
                        const int blk = (r16 >> 3) + ((c32 >> 4) << 1);
                        const int dst_off =
                            (r_core * KCORES + c_core) * 512 + blk * 128 + (r16 & 7) * 16;
                        unsigned dst = (unsigned)__cvta_generic_to_shared(baseA + dst_off);
                        const uint8_t* src = A + (size_t)grow * K + k_off + c;
                        if (grow >= M) {
                            asm volatile("cp.async.ca.shared.global [%0], [%1], 16, 0;\n"
                                         :: "r"(dst), "l"(src));
                        } else if (EV) {
                            asm volatile("cp.async.cg.shared.global.L2::cache_hint "
                                         "[%0], [%1], 16, %2;\n"
                                         :: "r"(dst), "l"(src), "l"(pol_w));
                        } else {
                            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                                         :: "r"(dst), "l"(src));
                        }
                    }
                }
            }
        }
        if (part != 0) {
            if constexpr (XS) {
                #pragma unroll
                for (int j = 0; j < B_ROUNDS; ++j) {
                    if (bval[j]) {
                        unsigned dst = (unsigned)__cvta_generic_to_shared(
                            baseB + (size_t)bdstu[j] * 16);
                        const uint8_t* src = W + (size_t)browk[j] * K + k_off + bkgc[j];
                        fwam_cp16<WCG != 0, WHINT != 0, WPF>(dst, src, pol_wrd, false);
                    }
                }
            } else if constexpr (LIN) {
                #pragma unroll
                for (int j = 0; j < B_ROUNDS; ++j) {
                    const int u = tid + j * NTHREADS;
                    if (u < B_CHUNKS) {
                        unsigned dst = (unsigned)__cvta_generic_to_shared(
                            baseB + (size_t)bdstu[j] * 16);
                        const uint8_t* src = W + (size_t)browk[j] * K + k_off + bkgc[j];
                        if constexpr (EV == 1) {  // W is 42MB streaming → evict_first protects A/intermediates
                            asm volatile("cp.async.cg.shared.global.L2::cache_hint "
                                         "[%0], [%1], 16, %2;\n"
                                         :: "r"(dst), "l"(src), "l"(pol_w));
                        } else if constexpr (EV == 3) {
                            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                                         :: "r"(dst), "l"(src));
                        } else {
                            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                                         :: "r"(dst), "l"(src));
                        }
                    }
                }
            } else {
                #pragma unroll
                for (int j = 0; j < B_ROUNDS; ++j) {
                    int chunk = tid + j * NTHREADS;
                    if (chunk < B_CHUNKS) {
                        int n = chunk / KQ;
                        int c = (chunk % KQ) * 16;
                        const int n_core = n >> 3;
                        const int c_core = c >> 5;
                        const int n8 = n & 7;
                        const int col_half = (c & 31) >> 4;
                        const int dst_off =
                            (n_core * KCORES + c_core) * 256 + col_half * 128 + n8 * 16;
                        unsigned dst = (unsigned)__cvta_generic_to_shared(
                            baseB + dst_off);
                        const uint8_t* src = W + (size_t)(block_col + n) * K + k_off + c;
                        if (EV) {
                            asm volatile("cp.async.cg.shared.global.L2::cache_hint "
                                         "[%0], [%1], 16, %2;\n"
                                         :: "r"(dst), "l"(src), "l"(pol_w));
                        } else {
                            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                                         :: "r"(dst), "l"(src));
                        }
                    }
                }
            }
        }
    };
    // upgemm: make the commit explicit -- commit manually after every issue (whether A, B, paired or
    // empty), so that each round has a constant 1 commit group (the FIFO depth is the accounting basis for the prefetch distance).
    auto commit_grp = []() { asm volatile("cp.async.commit_group;\n" ::); };

    // BORD=1: B(W) issued first -- W is on the DRAM critical path, so let it grab the memory queues first
    auto issue_stage = [&](int aslot, int bslot, int kt, int part) {
        if constexpr (BORD) {
            if (part == 2) { issue_stage_impl(aslot, bslot, kt, 1, issue_stage_impl);
                             issue_stage_impl(aslot, bslot, kt, 0, issue_stage_impl); }
            else           { issue_stage_impl(aslot, bslot, kt, part, issue_stage_impl); }
        } else {
            issue_stage_impl(aslot, bslot, kt, part, issue_stage_impl);
        }
    };
    auto issue_pair = [&](int aslot, int bslot, int kt) { issue_stage(aslot, bslot, kt, 2); };
    auto issue_A = [&](int aslot, int kt) { issue_stage(aslot, 0, kt, 0); };
    auto issue_B = [&](int bslot, int kt) { issue_stage(0, bslot, kt, 1); };
    // APATH has been removed (measured ineffective); only the cp.async path is kept.

    // Prefill: first fill up to min(SA,SB)-1 stages pairwise, then top the deeper side up on its own
    constexpr int MINRING = SA < SB ? SA : SB;
    #pragma unroll
    for (int j = 0; j < MINRING - 1; ++j) {
        if (j < Ktiles) issue_pair(j % SA, j % SB, j);
        commit_grp();
    }
    #pragma unroll
    for (int j = MINRING - 1; j < SA - 1; ++j) {
        if (j < Ktiles) issue_A(j % SA, j);
        commit_grp();
    }
    #pragma unroll
    for (int j = MINRING - 1; j < SB - 1; ++j) {
        if (j < Ktiles) issue_B(j % SB, j);
        commit_grp();
    }

    float acc[N8PW][M16PW][4];
    // FA=1: in-stage fp16 accumulator (half2 bit pattern), promoted into the fp32 acc at the end of the stage
    uint32_t acc16[N8PW][M16PW][2];
    #pragma unroll
    for (int ni = 0; ni < N8PW; ++ni)
        #pragma unroll
        for (int mi = 0; mi < M16PW; ++mi) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) acc[ni][mi][i] = 0.f;
            if constexpr (FA) { acc16[ni][mi][0] = 0u; acc16[ni][mi][1] = 0u; }
        }

    for (int kt = 0; kt < Ktiles; ++kt) {
        const int aslot = kt % SA, bslot = kt % SB;
        // Wait for the same kt block in A and B: with 1 commit group per round, whichever of A_kt/B_kt
        // has the shorter prefetch distance (distance = min(SA,SB)-1) comes later, so wait_group(min(SA,SB)-2) guarantees both have completed.
        if (kt + 1 == Ktiles) asm volatile("cp.async.wait_group 0;\n" ::);
        else                  asm volatile("cp.async.wait_group %0;\n" :: "n"(MINRING - 2));
#ifndef UPGEMM_NO_SYNC
        __syncthreads();
#endif
        // Diagnostics (-DUPGEMM_NO_SYNC): drop the whole-block barrier of every stage. The result is
        // no longer guaranteed to be correct (another warp's cp.async data may not be visible); it is
        // only used to measure how much time "barrier + inter-warp coupling" is actually worth.
        // ORD=1: issue right after the barrier (maximum lead); ORD=0: issue after mma.
        // The target slots (kt+SA-1)%SA / (kt+SB-1)%SB were already consumed in round kt-1, so neither order has a hazard.
        if constexpr (ORD) {
            const int nktA = kt + SA - 1, nktB = kt + SB - 1;
            if (nktA < Ktiles) issue_A(nktA % SA, nktA);
            if (nktB < Ktiles) issue_B(nktB % SB, nktB);
            commit_grp();
        }

        const uint8_t* As_s = smem_raw + (size_t)aslot * A_BYTES;
        const uint8_t* Ws_s = smem_raw + (size_t)A_BYTES * SA + (size_t)bslot * B_BYTES;

        // ldmatrix addressing: XS = XOR-swizzle within the window (t = r*4+q, cell = t^((t>>3)&7));
        // otherwise the crosswise core. Both paths are the inverse of their own cp.async placement,
        // which keeps the fragment semantics consistent.
        // XS=2 addressing: row-major + SROWD pad, every lane-dependent term is a loop invariant, so
        // entering the loop leaves only one increment, base + kk*32 (vs XS=1's ~85 swizzle integer
        // ops per stage).
        // The matrix→(row, k half) map is the inverse of the LIN loader's dst decomposition (the same
        // mma fragment semantics):
        //   A (512B core): blk = 2*k_half + row_half → matrix m=lane>>3: m0=(rows 0-7,k0-15)
        //                m1=(rows 8-15,k0-15) m2=(rows 0-7,k16-31) m3=(rows 8-15,k16-31)
        //   B (256B core): matrix m: core=lane>>4 (kk / kk+1), k half=(lane>>3)&1, row=lane&7
        auto a_addr = [&](int m16g, int kk) -> unsigned {
            if constexpr (XS == 2) {
                const int r = ((lane >> 3) & 1) * 8 + (lane & 7);   // row half + within-row
                return (unsigned)__cvta_generic_to_shared(
                    As_s + (size_t)(m16g * 16 + r) * SROWD + (size_t)kk * 32
                    + (size_t)(lane >> 4) * 16);                    // k half
            } else if constexpr (XS) {
                const int t = ((((m16g & 3) << 4) + (lane & 15)) << 2)
                            + (((kk & 1) << 1) | (lane >> 4));
                const int cell = t ^ ((t >> 3) & 7);
                return (unsigned)__cvta_generic_to_shared(
                    As_s + (size_t)((m16g >> 2) * KPARTS + (kk >> 1)) * 4096
                    + (size_t)cell * 16);
            } else {
                return (unsigned)__cvta_generic_to_shared(
                    As_s + (size_t)(m16g * KCORES + kk) * 512 + (size_t)lane * 16);
            }
        };
        auto b_addr = [&](int n8g, int kk) -> unsigned {
            if constexpr (XS == 2) {
                return (unsigned)__cvta_generic_to_shared(
                    Ws_s + (size_t)(n8g * 8 + (lane & 7)) * SROWD + (size_t)kk * 32
                    + (size_t)(lane >> 4) * 32 + (size_t)((lane >> 3) & 1) * 16);
            } else if constexpr (XS) {
                const int t = ((((n8g & 7) << 3) + (lane & 7)) << 2) + ((lane >> 3) & 3);
                const int cell = t ^ ((t >> 3) & 7);
                return (unsigned)__cvta_generic_to_shared(
                    Ws_s + (size_t)((n8g >> 3) * KPARTS + (kk >> 1)) * 4096
                    + (size_t)cell * 16);
            } else {
                return (unsigned)__cvta_generic_to_shared(
                    Ws_s + (size_t)(n8g * KCORES + kk) * 256 + (size_t)lane * 16);
            }
        };

        if constexpr (LDB) {
            // ---- ldmatrix double buffering: prefetch the next fragment so that the ldm latency/LSU overlaps the mma tensor pipeline ----
            constexpr int NSTEP = KCORES / 2;
            uint32_t abuf[2][M16PW][2][4];
            uint32_t bcur[4], bnxt[4];
            // Prefetch A(step0) + B(step0, ni0)
            #pragma unroll
            for (int mi = 0; mi < M16PW; ++mi) {
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int m16g = warp_m * M16PW + mi;
                    const unsigned addr = a_addr(m16g, h);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(abuf[0][mi][h][0]), "=r"(abuf[0][mi][h][1]),
                                   "=r"(abuf[0][mi][h][2]), "=r"(abuf[0][mi][h][3])
                                 : "r"(addr));
                }
            }
            {
                const int n8g = warp_n * N8PW;
                const unsigned addr = b_addr(n8g, 0);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(bcur[0]), "=r"(bcur[1]), "=r"(bcur[2]), "=r"(bcur[3])
                             : "r"(addr));
            }
            #pragma unroll
            for (int ks = 0; ks < NSTEP; ++ks) {
                const int cur = ks & 1;
                if (ks + 1 < NSTEP) {
                    #pragma unroll
                    for (int mi = 0; mi < M16PW; ++mi) {
                        #pragma unroll
                        for (int h = 0; h < 2; ++h) {
                            const int m16g = warp_m * M16PW + mi;
                            const unsigned addr = a_addr(m16g, (ks + 1) * 2 + h);
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                         : "=r"(abuf[cur ^ 1][mi][h][0]), "=r"(abuf[cur ^ 1][mi][h][1]),
                                           "=r"(abuf[cur ^ 1][mi][h][2]), "=r"(abuf[cur ^ 1][mi][h][3])
                                         : "r"(addr));
                        }
                    }
                }
                #pragma unroll
                for (int ni = 0; ni < N8PW; ++ni) {
                    // B prefetch: the next ni of this step, otherwise ni0 of the next step
                    if (ni + 1 < N8PW || ks + 1 < NSTEP) {
                        const int ni2 = (ni + 1 < N8PW) ? ni + 1 : 0;
                        const int ks2 = (ni + 1 < N8PW) ? ks : ks + 1;
                        const int n8g = warp_n * N8PW + ni2;
                        const unsigned addr = b_addr(n8g, ks2 * 2);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                     : "=r"(bnxt[0]), "=r"(bnxt[1]),
                                       "=r"(bnxt[2]), "=r"(bnxt[3])
                                     : "r"(addr));
                    }
                    #pragma unroll
                    for (int mi = 0; mi < M16PW; ++mi) {
                        if constexpr (FA) {
                            asm volatile(
                                "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
                                "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                                : "+r"(acc16[ni][mi][0]), "+r"(acc16[ni][mi][1])
                                : "r"(abuf[cur][mi][0][0]), "r"(abuf[cur][mi][0][1]),
                                  "r"(abuf[cur][mi][0][2]), "r"(abuf[cur][mi][0][3]),
                                  "r"(bcur[0]), "r"(bcur[1]));
                            asm volatile(
                                "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
                                "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                                : "+r"(acc16[ni][mi][0]), "+r"(acc16[ni][mi][1])
                                : "r"(abuf[cur][mi][1][0]), "r"(abuf[cur][mi][1][1]),
                                  "r"(abuf[cur][mi][1][2]), "r"(abuf[cur][mi][1][3]),
                                  "r"(bcur[2]), "r"(bcur[3]));
                        } else {
                            asm volatile(
                                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                                : "+f"(acc[ni][mi][0]), "+f"(acc[ni][mi][1]),
                                  "+f"(acc[ni][mi][2]), "+f"(acc[ni][mi][3])
                                : "r"(abuf[cur][mi][0][0]), "r"(abuf[cur][mi][0][1]),
                                  "r"(abuf[cur][mi][0][2]), "r"(abuf[cur][mi][0][3]),
                                  "r"(bcur[0]), "r"(bcur[1]));
                            asm volatile(
                                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                                : "+f"(acc[ni][mi][0]), "+f"(acc[ni][mi][1]),
                                  "+f"(acc[ni][mi][2]), "+f"(acc[ni][mi][3])
                                : "r"(abuf[cur][mi][1][0]), "r"(abuf[cur][mi][1][1]),
                                  "r"(abuf[cur][mi][1][2]), "r"(abuf[cur][mi][1][3]),
                                  "r"(bcur[2]), "r"(bcur[3]));
                        }
                    }
                    bcur[0] = bnxt[0]; bcur[1] = bnxt[1];
                    bcur[2] = bnxt[2]; bcur[3] = bnxt[3];
                }
            }
        } else {
        #pragma unroll
        for (int kk = 0; kk < KCORES; kk += 2) {
            // A fragment: every m16 row band of this warp → ldmatrix.x4 (16x32 core, one per kk)
            uint32_t a[M16PW][2][4];
            #pragma unroll
            for (int mi = 0; mi < M16PW; ++mi) {
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int m16g = warp_m * M16PW + mi;
                    const unsigned addr = a_addr(m16g, kk + h);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(a[mi][h][0]), "=r"(a[mi][h][1]),
                                   "=r"(a[mi][h][2]), "=r"(a[mi][h][3])
                                 : "r"(addr));
                }
            }
            #pragma unroll
            for (int ni = 0; ni < N8PW; ++ni) {
                const int n8g = warp_n * N8PW + ni;
                // B: one x4 spans the two adjacent 8x32 cores (kk, kk+1) → 4 fragment registers
                uint32_t b0, b1, b2, b3;
                {
                    const unsigned addr = b_addr(n8g, kk);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                                 "{%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3) : "r"(addr));
                }
                #pragma unroll
                for (int mi = 0; mi < M16PW; ++mi) {
                    if constexpr (FA) {
                        asm volatile(
                            "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
                            "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                            : "+r"(acc16[ni][mi][0]), "+r"(acc16[ni][mi][1])
                            : "r"(a[mi][0][0]), "r"(a[mi][0][1]),
                              "r"(a[mi][0][2]), "r"(a[mi][0][3]),
                              "r"(b0), "r"(b1));
                        asm volatile(
                            "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
                            "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                            : "+r"(acc16[ni][mi][0]), "+r"(acc16[ni][mi][1])
                            : "r"(a[mi][1][0]), "r"(a[mi][1][1]),
                              "r"(a[mi][1][2]), "r"(a[mi][1][3]),
                              "r"(b2), "r"(b3));
                    } else {
                        asm volatile(
                            "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                            : "+f"(acc[ni][mi][0]), "+f"(acc[ni][mi][1]),
                              "+f"(acc[ni][mi][2]), "+f"(acc[ni][mi][3])
                            : "r"(a[mi][0][0]), "r"(a[mi][0][1]), "r"(a[mi][0][2]), "r"(a[mi][0][3]),
                              "r"(b0), "r"(b1));
                        asm volatile(
                            "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                            : "+f"(acc[ni][mi][0]), "+f"(acc[ni][mi][1]),
                              "+f"(acc[ni][mi][2]), "+f"(acc[ni][mi][3])
                            : "r"(a[mi][1][0]), "r"(a[mi][1][1]), "r"(a[mi][1][2]), "r"(a[mi][1][3]),
                              "r"(b2), "r"(b3));
                    }
                }
            }
        }
        }  // LDB

        // FA=1: at the end of the stage promote the fp16 partial sums to fp32 (and clear them for the
        // next stage). FAP>1: promote once every FAP stages (the last stage always promotes); with
        // FAP==1 the condition is always true at compile time, so existing instantiations keep exactly the same code and cost.
        if constexpr (FA) {
            const bool do_promote = (FAP == 1) || ((kt + 1) % FAP == 0) || (kt + 1 == Ktiles);
            if (do_promote) {
            #pragma unroll
            for (int ni = 0; ni < N8PW; ++ni)
                #pragma unroll
                for (int mi = 0; mi < M16PW; ++mi) {
                    const float2 f0 = __half22float2(
                        *reinterpret_cast<const __half2*>(&acc16[ni][mi][0]));
                    const float2 f1 = __half22float2(
                        *reinterpret_cast<const __half2*>(&acc16[ni][mi][1]));
                    acc[ni][mi][0] += f0.x;
                    acc[ni][mi][1] += f0.y;
                    acc[ni][mi][2] += f1.x;
                    acc[ni][mi][3] += f1.y;
                    acc16[ni][mi][0] = 0u;
                    acc16[ni][mi][1] = 0u;
                }
            }
        }

        if constexpr (!ORD) {
            const int nktA = kt + SA - 1, nktB = kt + SB - 1;
            if (nktA < Ktiles) issue_A(nktA % SA, nktA);
            if (nktB < Ktiles) issue_B(nktB % SB, nktB);
            commit_grp();           // exactly one commit group per round (keeps the FIFO count)
        }
    }

    // ------------------------------------------------------------------ //
    // epilogue (staged through smem in row-major layout -> fully coalesced vector stores; falls back to direct stores when smem is insufficient)
    // ------------------------------------------------------------------ //
    // After the main loop every thread has finished consuming the stage smem:
    __syncthreads();
    // Staging row stride (+4 pad removes the 8-way bank conflict of frag->staging)
    constexpr int SROW = BN + 4;
    constexpr bool STAGED =
        (size_t)SMEM_BYTES >= (size_t)BM * SROW * 4 + 1024;

    if constexpr (SPLIT > 1) {
        // ---- split-K: acc is written to partial[z] (an independent slot per z, zero contention) ----
        const size_t s_off = (size_t)blockIdx.z * M * N;
        __half* part16 = reinterpret_cast<__half*>(partial);   // used by F16P
        if constexpr (PD || !STAGED) {
            // Fragment direct write (F16P: half2 4B; otherwise float2 8B); skips smem staging and
            // the barrier; 4 lanes of a warp cover the same 32B sector (a 16B half sector for F16P); PH=1 adds evict_first
            #pragma unroll
            for (int ni = 0; ni < N8PW; ++ni)
                #pragma unroll
                for (int mi = 0; mi < M16PW; ++mi) {
                    const int m16g = warp_m * M16PW + mi;
                    const int n8g = warp_n * N8PW + ni;
                    const int r0 = block_row + m16g * 16 + lane_g;
                    const int c0 = block_col + n8g * 8 + lane_t * 2;
                    if (r0 < M) {
                        if constexpr (F16P) {
                            __half* p = part16 + s_off + (size_t)r0 * N + c0;
                            const uint32_t h01 = *reinterpret_cast<const uint32_t*>(
                                &(const __half2&)__float22half2_rn(
                                    make_float2(acc[ni][mi][0], acc[ni][mi][1])));
                            if constexpr (PH) {
                                asm volatile("st.global.L2::cache_hint.b32 [%0], %1, %2;\n"
                                             :: "l"(p), "r"(h01), "l"(pol_o));
                            } else {
                                *reinterpret_cast<uint32_t*>(p) = h01;
                            }
                            if (r0 + 8 < M) {
                                __half* p2 = part16 + s_off + (size_t)(r0 + 8) * N + c0;
                                const uint32_t h23 = *reinterpret_cast<const uint32_t*>(
                                    &(const __half2&)__float22half2_rn(
                                        make_float2(acc[ni][mi][2], acc[ni][mi][3])));
                                if constexpr (PH) {
                                    asm volatile("st.global.L2::cache_hint.b32 [%0], %1, %2;\n"
                                                 :: "l"(p2), "r"(h23), "l"(pol_o));
                                } else {
                                    *reinterpret_cast<uint32_t*>(p2) = h23;
                                }
                            }
                        } else {
                        float* p = partial + s_off + (size_t)r0 * N + c0;
                        if constexpr (PH) {
                            asm volatile("st.global.L2::cache_hint.v2.f32 [%0], {%1,%2}, %3;\n"
                                         :: "l"(p), "f"(acc[ni][mi][0]), "f"(acc[ni][mi][1]),
                                            "l"(pol_o));
                        } else {
                            *reinterpret_cast<float2*>(p) =
                                make_float2(acc[ni][mi][0], acc[ni][mi][1]);
                        }
                        if (r0 + 8 < M) {
                            float* p2 = partial + s_off + (size_t)(r0 + 8) * N + c0;
                            if constexpr (PH) {
                                asm volatile("st.global.L2::cache_hint.v2.f32 [%0], {%1,%2}, %3;\n"
                                             :: "l"(p2), "f"(acc[ni][mi][2]), "f"(acc[ni][mi][3]),
                                                "l"(pol_o));
                            } else {
                                *reinterpret_cast<float2*>(p2) =
                                    make_float2(acc[ni][mi][2], acc[ni][mi][3]);
                            }
                        }
                        }
                    }
                }
        } else {
            float* st = reinterpret_cast<float*>(smem_raw);
            #pragma unroll
            for (int ni = 0; ni < N8PW; ++ni)
                #pragma unroll
                for (int mi = 0; mi < M16PW; ++mi) {
                    const int m16g = warp_m * M16PW + mi;
                    const int n8g = warp_n * N8PW + ni;
                    const int r0t = m16g * 16 + lane_g;
                    const int c0t = n8g * 8 + lane_t * 2;
                    st[(size_t)r0t * SROW + c0t] = acc[ni][mi][0];
                    st[(size_t)r0t * SROW + c0t + 1] = acc[ni][mi][1];
                    st[(size_t)(r0t + 8) * SROW + c0t] = acc[ni][mi][2];
                    st[(size_t)(r0t + 8) * SROW + c0t + 1] = acc[ni][mi][3];
                }
            __syncthreads();
            // Each warp covers one row (BN/4 = 32 float4) → fully coalesced 16B per thread
            const int col4_per_row = BN / 4;
            #pragma unroll
            for (int q = tid; q < BM * col4_per_row; q += NTHREADS) {
                const int r = q / col4_per_row;          // row within the tile
                const int c4 = (q % col4_per_row) * 4;   // column within the tile
                if (block_row + r < M) {
                    const float4 v = *reinterpret_cast<const float4*>(
                        st + (size_t)r * SROW + c4);
                    if constexpr (F16P) {
                        // 4 × fp16 = 8B (uint2); 4 adjacent threads = a full 32B sector
                        const __half2 h01 = __float22half2_rn(make_float2(v.x, v.y));
                        const __half2 h23 = __float22half2_rn(make_float2(v.z, v.w));
                        uint2 packed = make_uint2(
                            *reinterpret_cast<const uint32_t*>(&h01),
                            *reinterpret_cast<const uint32_t*>(&h23));
                        void* p = part16 + s_off +
                                  (size_t)(block_row + r) * N + block_col + c4;
                        if constexpr (PH) {
                            asm volatile("st.global.L2::cache_hint.v2.b32 [%0], {%1,%2}, %3;\n"
                                         :: "l"(p), "r"(packed.x), "r"(packed.y), "l"(pol_o));
                        } else {
                            *reinterpret_cast<uint2*>(p) = packed;
                        }
                    } else {
                    float* p = partial + s_off +
                               (size_t)(block_row + r) * N + block_col + c4;
                    if constexpr (PH) {
                        asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1,%2,%3,%4}, %5;\n"
                                     :: "l"(p), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w),
                                        "l"(pol_o));
                    } else {
                        *reinterpret_cast<float4*>(p) = v;
                    }
                    }
                }
            }
        }
        // ---- per-tile counter: the last block reduces and writes bf16 directly.
        //      counters==nullptr = separate merge mode: skip fence/counter/merge and let a later
        //      standalone kernel reduce (a kernel boundary is device-visible, no membar needed) ----
        if (counters != nullptr) {
        __threadfence();          // this thread's partial stores are device-visible (once per thread)
        __syncthreads();
        __shared__ int s_merge;
        if (counters != nullptr) {
            const int tile_id = blockIdx.x * gridDim.y + blockIdx.y;
            if (tid == 0) {
                const int old = atomicAdd(&counters[tile_id], 1);
                s_merge = (old == SPLIT - 1);
            }
        }
        __syncthreads();
        if (s_merge) {
            // The last arrival of this tile: sum the SPLIT partials in fixed order (float4/half4
            // vectorized), multiply by sa*sw, add bias and write bf16 RN directly (only real rows r<M).
            auto ld4 = [&](size_t idx) -> float4 {
                if constexpr (F16P) {
                    const uint2 u = *reinterpret_cast<const uint2*>(part16 + idx);
                    const __half2 h01 = *reinterpret_cast<const __half2*>(&u.x);
                    const __half2 h23 = *reinterpret_cast<const __half2*>(&u.y);
                    const float2 f01 = __half22float2(h01);
                    const float2 f23 = __half22float2(h23);
                    return make_float4(f01.x, f01.y, f23.x, f23.y);
                } else {
                    return *reinterpret_cast<const float4*>(partial + idx);
                }
            };
            const int col4_per_row = BN / 4;
            const int ncells = BM * col4_per_row;   // float4 cells (over the padded tile)
            for (int q = tid; q < ncells; q += NTHREADS) {
                const int rl = q / col4_per_row;
                const int c4 = (q % col4_per_row) * 4;
                const int r = block_row + rl;
                const int c = block_col + c4;
                if (r < M) {
                    float4 v = ld4((size_t)r * N + c);
                    #pragma unroll
                    for (int s = 1; s < SPLIT; ++s) {
                        const float4 t = ld4(((size_t)s * M + r) * N + c);
                        v.x += t.x; v.y += t.y; v.z += t.z; v.w += t.w;
                    }
                    const float4 swv = *reinterpret_cast<const float4*>(sw + c);
                    const float sr = sa[r];
                    float b0 = v.x * sr * swv.x;
                    float b1 = v.y * sr * swv.y;
                    float b2 = v.z * sr * swv.z;
                    float b3 = v.w * sr * swv.w;
                    if (bias != nullptr) {
                        const float2 f0 = __bfloat1622float2(
                            *reinterpret_cast<const __nv_bfloat162*>(bias + c));
                        const float2 f1 = __bfloat1622float2(
                            *reinterpret_cast<const __nv_bfloat162*>(bias + c + 2));
                        b0 += f0.x; b1 += f0.y; b2 += f1.x; b3 += f1.y;
                    }
                    const __nv_bfloat162 h0 = __floats2bfloat162_rn(b0, b1);
                    const __nv_bfloat162 h1 = __floats2bfloat162_rn(b2, b3);
                    const uint32_t w0 = *reinterpret_cast<const uint32_t*>(&h0);
                    const uint32_t w1 = *reinterpret_cast<const uint32_t*>(&h1);
                    if constexpr (PH) {
                        asm volatile("st.global.L2::cache_hint.v2.b32 [%0], {%1,%2}, %3;\n"
                                     :: "l"(Cout + (size_t)r * N + c), "r"(w0), "r"(w1),
                                        "l"(pol_o));
                    } else {
                        *reinterpret_cast<uint2*>(Cout + (size_t)r * N + c) =
                            make_uint2(w0, w1);
                    }
                }
            }
        }
        __syncthreads();
        if (counters != nullptr && tid == 0 && s_merge) {
            counters[blockIdx.x * gridDim.y + blockIdx.y] = 0;
        }
        }  // counters != nullptr
    } else {
        // ---- Plain form: acc*sa*sw+bias -> bf16 (staged through smem in row-major -> 8B coalesced store per thread) ----
        if constexpr (EPI) {
            // Fused epilogue: u=bf16RN(acc*sa*sw+bias) → g=bf16RN(gelu_tanh(u))
            // → write gbuf(Cout); the per-row max |g| is atomicMax'ed into raw (= the partial slot, fp32 bit pattern)
            static_assert(STAGED, "EPI needs staged epilogue");
            float* st = reinterpret_cast<float*>(smem_raw);
            #pragma unroll
            for (int ni = 0; ni < N8PW; ++ni)
                #pragma unroll
                for (int mi = 0; mi < M16PW; ++mi) {
                    const int m16g = warp_m * M16PW + mi;
                    const int n8g = warp_n * N8PW + ni;
                    const int r0t = m16g * 16 + lane_g;
                    const int c0t = n8g * 8 + lane_t * 2;
                    st[(size_t)r0t * SROW + c0t] = acc[ni][mi][0];
                    st[(size_t)r0t * SROW + c0t + 1] = acc[ni][mi][1];
                    st[(size_t)(r0t + 8) * SROW + c0t] = acc[ni][mi][2];
                    st[(size_t)(r0t + 8) * SROW + c0t + 1] = acc[ni][mi][3];
                }
            __syncthreads();
            uint32_t* raw = reinterpret_cast<uint32_t*>(partial);  // [M] fp32 bit pattern
            const int col4_per_row = BN / 4;
            constexpr int SEG = (BN / 4) < 32 ? (BN / 4) : 32;   // segment width of the amax warp reduction
            static_assert(SEG == 16 || SEG == 32, "EPI amax reduction needs BN ∈ {64,128}");
            #pragma unroll
            for (int q = tid; q < BM * col4_per_row; q += NTHREADS) {
                const int rl = q / col4_per_row;
                const int c4 = (q % col4_per_row) * 4;
                const int r = block_row + rl;
                float mx = 0.f;
                if (r < M) {
                    const float4 v = *reinterpret_cast<const float4*>(st + (size_t)rl * SROW + c4);
                    const float sc = sa[r];
                    const int cg = block_col + c4;
                    float u0 = v.x * sc * sw[cg];
                    float u1 = v.y * sc * sw[cg + 1];
                    float u2 = v.z * sc * sw[cg + 2];
                    float u3 = v.w * sc * sw[cg + 3];
                    if (bias != nullptr) {
                        u0 += __bfloat162float(bias[cg]);
                        u1 += __bfloat162float(bias[cg + 1]);
                        u2 += __bfloat162float(bias[cg + 2]);
                        u3 += __bfloat162float(bias[cg + 3]);
                    }
                    const float g0 = fwam_bf16r(gelu_tanh(fwam_bf16r(u0)));
                    const float g1 = fwam_bf16r(gelu_tanh(fwam_bf16r(u1)));
                    const float g2 = fwam_bf16r(gelu_tanh(fwam_bf16r(u2)));
                    const float g3 = fwam_bf16r(gelu_tanh(fwam_bf16r(u3)));
                    const __nv_bfloat162 h0 = __floats2bfloat162_rn(g0, g1);
                    const __nv_bfloat162 h1 = __floats2bfloat162_rn(g2, g3);
                    uint32_t w0b = *reinterpret_cast<const uint32_t*>(&h0);
                    uint32_t w1b = *reinterpret_cast<const uint32_t*>(&h1);
                    *reinterpret_cast<uint2*>(Cout + (size_t)r * N + cg) = make_uint2(w0b, w1b);
                    mx = fmaxf(fmaxf(fabsf(g0), fabsf(g1)),
                               fmaxf(fabsf(g2), fabsf(g3)));
                }
                // Row-level amax: a segmented warp reduction followed by a single atomic (previously
                // one atomic per float4 → ~430K serial RMWs hammering 120 slots ≈ 90μs; now ~128 per
                // CTA). max is associative, so with col4>32 several warps atomically updating separate row
                // segments is still correct; invalid rows contribute 0 and do not affect the max.
                #pragma unroll
                for (int off = SEG / 2; off > 0; off >>= 1)
                    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off, SEG));
                if ((lane & (SEG - 1)) == 0 && r < M && mx > 0.f)
                    atomicMax(raw + r, __float_as_uint(mx));  // |g|≥0 keeps the bit pattern monotonic
            }
        } else if constexpr (STAGED) {
            float* st = reinterpret_cast<float*>(smem_raw);
            #pragma unroll
            for (int ni = 0; ni < N8PW; ++ni)
                #pragma unroll
                for (int mi = 0; mi < M16PW; ++mi) {
                    const int m16g = warp_m * M16PW + mi;
                    const int n8g = warp_n * N8PW + ni;
                    const int r0t = m16g * 16 + lane_g;
                    const int c0t = n8g * 8 + lane_t * 2;
                    st[(size_t)r0t * SROW + c0t] = acc[ni][mi][0];
                    st[(size_t)r0t * SROW + c0t + 1] = acc[ni][mi][1];
                    st[(size_t)(r0t + 8) * SROW + c0t] = acc[ni][mi][2];
                    st[(size_t)(r0t + 8) * SROW + c0t + 1] = acc[ni][mi][3];
                }
            __syncthreads();
            // Fix: this path used to only stage without writing back to Cout. staging → scale+bias → coalesced bf16 store
            const int col4_per_row = BN / 4;
            #pragma unroll
            for (int q = tid; q < BM * col4_per_row; q += NTHREADS) {
                const int rl = q / col4_per_row;
                const int c4 = (q % col4_per_row) * 4;
                const int r = block_row + rl;
                if (r < M) {
                    const float4 v = *reinterpret_cast<const float4*>(
                        st + (size_t)rl * SROW + c4);
                    const float sc = sa[r];
                    const int cg = block_col + c4;
                    float b0 = v.x * sc * sw[cg];
                    float b1 = v.y * sc * sw[cg + 1];
                    float b2 = v.z * sc * sw[cg + 2];
                    float b3 = v.w * sc * sw[cg + 3];
                    if (bias != nullptr) {
                        b0 += __bfloat162float(bias[cg]);
                        b1 += __bfloat162float(bias[cg + 1]);
                        b2 += __bfloat162float(bias[cg + 2]);
                        b3 += __bfloat162float(bias[cg + 3]);
                    }
                    if constexpr (RESID) {   // o = bf16RN(bf16RN(z*gate) + x_in)
                        const unsigned short* gu =
                            reinterpret_cast<const unsigned short*>(ffn_ex->gate) +
                            (size_t)r * ffn_ex->gate_srow;   // srow=0 → broadcast
                        const unsigned short* xu =
                            reinterpret_cast<const unsigned short*>(
                                static_cast<const uint8_t*>(ffn_ex->x_in)) +
                            (size_t)r * ffn_ex->H + cg;
                        b0 = fwam_resid_bf16(b0, gu[cg + 0], xu[0]);
                        b1 = fwam_resid_bf16(b1, gu[cg + 1], xu[1]);
                        b2 = fwam_resid_bf16(b2, gu[cg + 2], xu[2]);
                        b3 = fwam_resid_bf16(b3, gu[cg + 3], xu[3]);
                    }
                    const __nv_bfloat162 h0 = __floats2bfloat162_rn(b0, b1);
                    const __nv_bfloat162 h1 = __floats2bfloat162_rn(b2, b3);
                    *reinterpret_cast<uint2*>(Cout + (size_t)r * N + cg) =
                        make_uint2(*reinterpret_cast<const uint32_t*>(&h0),
                                   *reinterpret_cast<const uint32_t*>(&h1));
                }
            }
        } else if constexpr (RESID) {
            // ---- Small-smem fallback + FFN residual (fragment direct write 4B) ----
            const unsigned short* gu =
                reinterpret_cast<const unsigned short*>(ffn_ex->gate);
            const unsigned short* xu =
                reinterpret_cast<const unsigned short*>(
                    static_cast<const uint8_t*>(ffn_ex->x_in));
            #pragma unroll
            for (int ni = 0; ni < N8PW; ++ni)
                #pragma unroll
                for (int mi = 0; mi < M16PW; ++mi) {
                    const int m16g = warp_m * M16PW + mi;
                    const int n8g = warp_n * N8PW + ni;
                    const int r0 = block_row + m16g * 16 + lane_g;
                    const int c0 = block_col + n8g * 8 + lane_t * 2;
                    if (r0 < M) {
                        float v0 = acc[ni][mi][0] * sa[r0] * sw[c0];
                        float v1 = acc[ni][mi][1] * sa[r0] * sw[c0 + 1];
                        if (bias != nullptr) {
                            v0 += __bfloat162float(bias[c0]);
                            v1 += __bfloat162float(bias[c0 + 1]);
                        }
                        const size_t xo0 = (size_t)r0 * ffn_ex->H + c0;
                        const __nv_bfloat162 o0 = __floats2bfloat162_rn(
                            fwam_resid_bf16(v0, gu[c0], xu[xo0]),
                            fwam_resid_bf16(v1, gu[c0 + 1], xu[xo0 + 1]));
                        *reinterpret_cast<uint32_t*>(Cout + (size_t)r0 * N + c0) =
                            *reinterpret_cast<const uint32_t*>(&o0);
                        if (r0 + 8 < M) {
                            float v2 = acc[ni][mi][2] * sa[r0 + 8] * sw[c0];
                            float v3 = acc[ni][mi][3] * sa[r0 + 8] * sw[c0 + 1];
                            if (bias != nullptr) {
                                v2 += __bfloat162float(bias[c0]);
                                v3 += __bfloat162float(bias[c0 + 1]);
                            }
                            const size_t xo1 = (size_t)(r0 + 8) * ffn_ex->H + c0;
                            const __nv_bfloat162 o1 = __floats2bfloat162_rn(
                                fwam_resid_bf16(v2, gu[c0], xu[xo1]),
                                fwam_resid_bf16(v3, gu[c0 + 1], xu[xo1 + 1]));
                            *reinterpret_cast<uint32_t*>(Cout + (size_t)(r0 + 8) * N + c0) =
                                *reinterpret_cast<const uint32_t*>(&o1);
                        }
                    }
                }
        } else {
            // ---- Small-smem fallback: fragment direct write 4B ----
            #pragma unroll
            for (int ni = 0; ni < N8PW; ++ni)
                #pragma unroll
                for (int mi = 0; mi < M16PW; ++mi) {
                    const int m16g = warp_m * M16PW + mi;
                    const int n8g = warp_n * N8PW + ni;
                    const int r0 = block_row + m16g * 16 + lane_g;
                    const int c0 = block_col + n8g * 8 + lane_t * 2;
                    if (r0 < M) {
                        float v0 = acc[ni][mi][0] * sa[r0] * sw[c0];
                        float v1 = acc[ni][mi][1] * sa[r0] * sw[c0 + 1];
                        if (bias != nullptr) {
                            v0 += __bfloat162float(bias[c0]);
                            v1 += __bfloat162float(bias[c0 + 1]);
                        }
                        const __nv_bfloat162 h0 = __floats2bfloat162_rn(v0, v1);
                        *reinterpret_cast<uint32_t*>(Cout + (size_t)r0 * N + c0) =
                            *reinterpret_cast<const uint32_t*>(&h0);
                        if (r0 + 8 < M) {
                            float v2 = acc[ni][mi][2] * sa[r0 + 8] * sw[c0];
                            float v3 = acc[ni][mi][3] * sa[r0 + 8] * sw[c0 + 1];
                            if (bias != nullptr) {
                                v2 += __bfloat162float(bias[c0]);
                                v3 += __bfloat162float(bias[c0 + 1]);
                            }
                            const __nv_bfloat162 h1 = __floats2bfloat162_rn(v2, v3);
                            *reinterpret_cast<uint32_t*>(Cout + (size_t)(r0 + 8) * N + c0) =
                                *reinterpret_cast<const uint32_t*>(&h1);
                        }
                    }
                }
        }
    }
    }  // for (_t) PERSIST tile loop
}  // fwam_fp8_gemm_body
