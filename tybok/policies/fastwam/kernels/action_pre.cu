// Fused kernels for the action DiT pre-components:
//   1) Time path (time_embedding/time_projection): an M=1 chain of 3 small GEMMs, squeezed into 4
//      small launches (sinusoidal → FC0+SiLU → FC1+SiLU → FC2).
//   2) Context precompute (precompute_action_context): kv via one large GEMM over the stacked
//      weights ([L,1024] × [30*6144,1024]ᵀ), then a repack kernel for the per-layer split + norm_k.
//
// Numerics: step-for-step identical in shape to production FP8Linear / WanRMSNorm (per-row
// amax/448 + software single RNE, bf16 RN, SiLU in fp32); the only difference is the fp32
// reduction order (which stays inside the bf16 rounding surface).
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>

#include "dit_common.h"   // fp8_rne: the same software single RNE as adit+vdit (no longer rewritten locally)

// --------------------------------------------------------------------------- //
// Basic conversions
// --------------------------------------------------------------------------- //
__device__ __forceinline__ float ap_bf16f(const __nv_bfloat16 h) {
    return __bfloat162float(h);
}

// The fp8 e4m3 software single RNE is used directly from `fp8_rne` in dit_common.h (the former local copy was verbatim-identical).

// e4m3 byte → float (bit manipulation, no ldexpf)
__device__ __forceinline__ float ap_e4m3f(unsigned char b) {
    const unsigned u = (unsigned)b;
    const unsigned sign = (u & 0x80u) << 24;
    const unsigned e = (u >> 3) & 0xFu;
    const unsigned m = u & 0x7u;
    const float norm = __uint_as_float(((e + 120u) << 23) | (m << 20));
    const float sub = (float)m * (1.0f / 512.0f);
    const float v = (e == 0u) ? sub : norm;
    return sign ? -v : v;
}

// --------------------------------------------------------------------------- //
// 1) Time path: sinusoidal (1 block)
//    x0 = bf16(cat(cos(pos*f), sin(pos*f))), amax|x0| → amax0; also zeroes amax1/amax2
// --------------------------------------------------------------------------- //
__global__ void ap_sin_kernel(const float* __restrict__ ts,
                              const double* __restrict__ freq,   // [NH]
                              __nv_bfloat16* __restrict__ y0,    // [NF]
                              float* __restrict__ amax0,
                              float* __restrict__ amax1,
                              float* __restrict__ amax2) {
    const int NF = blockDim.x;          // = 2*NH
    const int NH = NF >> 1;
    const int j = threadIdx.x;
    if (j == 0) { amax1[0] = 0.f; amax2[0] = 0.f; }
    // One output per thread (j<NH → cos, j>=NH → sin)
    const int k = (j < NH) ? j : (j - NH);
    const double pos = (double)ts[0];
    const double v = pos * freq[k];
    const double r = (j < NH) ? cos(v) : sin(v);
    const __nv_bfloat16 h = __double2bfloat16(r);
    y0[j] = h;
    float a = fabsf(ap_bf16f(h));
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    __shared__ float sh[32];
    const int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (lane == 0) sh[wid] = a;
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = 0.f;
        for (int i = 0; i < (int)(blockDim.x >> 5); ++i) m = fmaxf(m, sh[i]);
        amax0[0] = m;
    }
}

// --------------------------------------------------------------------------- //
// Generic FC: read y_in (bf16) → quantize (sa=amax_in/448) → wt[K,N] fp8 matmul → +bias →
//   optional SiLU (output side) → write y_out (bf16) + optional atomicMax(amax_out, |y_out|)
// Thread mapping: global thread n owns output neuron n (grid-stride); the input is quantized
//   once in smem and shared.
// --------------------------------------------------------------------------- //
template <int K, int N, bool SILU, bool AMAX>
__global__ void ap_fc_kernel(const __nv_bfloat16* __restrict__ y_in,
                             const float* __restrict__ amax_in,
                             const unsigned char* __restrict__ wt,   // [K,N]
                             const float* __restrict__ sw,           // [N]
                             const __nv_bfloat16* __restrict__ bias, // [N]
                             __nv_bfloat16* __restrict__ y_out,      // [N]
                             float* __restrict__ amax_out) {
    extern __shared__ float s_af[];   // K floats (pre-decoded input activations)
    const float sa = fmaxf(amax_in[0] / 448.0f, 1e-12f);
    for (int k = threadIdx.x; k < K; k += blockDim.x)
        s_af[k] = ap_e4m3f(fp8_rne(ap_bf16f(y_in[k]) / sa));
    __syncthreads();

    for (int n = blockIdx.x * blockDim.x + threadIdx.x; n < N;
         n += gridDim.x * blockDim.x) {
        float acc = 0.f;
        #pragma unroll 8
        for (int k = 0; k < K; ++k)
            acc = fmaf(s_af[k], ap_e4m3f(wt[(size_t)k * N + n]), acc);
        float y = acc * sa * sw[n] + ap_bf16f(bias[n]);
        __nv_bfloat16 h = __float2bfloat16_rn(y);
        if (SILU) {
            const float yf = ap_bf16f(h);
            h = __float2bfloat16_rn(yf / (1.0f + expf(-yf)));
        }
        y_out[n] = h;
        if (AMAX) {
            const float a = fabsf(ap_bf16f(h));
            atomicMax(reinterpret_cast<unsigned int*>(amax_out), __float_as_uint(a));
        }
    }
}

// --------------------------------------------------------------------------- //
// 2) context kv repack + norm_k
//    out_big [L, NB*6144] (column layout [l*6144 + (k|v)*3072]) → kv_out [NB,2,L,3072]
//    k goes through WanRMSNorm (two bf16 rounding steps), v is passed through as-is.
// --------------------------------------------------------------------------- //
__global__ void ap_ctx_repack_kernel(const __nv_bfloat16* __restrict__ out_big,
                                     int L, int NB,
                                     const __nv_bfloat16* __restrict__ wnk, // [NB,3072]
                                     float eps,
                                     __nv_bfloat16* __restrict__ kv_out) {  // [NB,2,L,3072]
    const int l = blockIdx.x;
    const int r = blockIdx.y;
    const size_t row = (size_t)NB * 6144;
    const __nv_bfloat16* krow = out_big + (size_t)r * row + (size_t)l * 6144;
    const __nv_bfloat16* vrow = krow + 3072;

    float ss = 0.f;
    for (int c = threadIdx.x; c < 3072; c += blockDim.x) {
        const float x = ap_bf16f(krow[c]);
        ss = fmaf(x, x, ss);
    }
    __shared__ float sh[32];
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x < 32) {
        const int nw = (blockDim.x + 31) >> 5;
        float v = (threadIdx.x < nw) ? sh[threadIdx.x] : 0.f;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (threadIdx.x == 0) sh[0] = v;
    }
    __syncthreads();
    const float rms = rsqrtf(sh[0] / 3072.0f + eps);

    const __nv_bfloat16* w = wnk + (size_t)l * 3072;
    __nv_bfloat16* kod = kv_out + (((size_t)l * 2 + 0) * L + r) * 3072;
    __nv_bfloat16* vod = kv_out + (((size_t)l * 2 + 1) * L + r) * 3072;
    for (int c = threadIdx.x; c < 3072; c += blockDim.x) {
        const float x = ap_bf16f(krow[c]);
        const __nv_bfloat16 n = __float2bfloat16_rn(x * rms);       // normalize → bf16
        kod[c] = __float2bfloat16_rn(ap_bf16f(n) * ap_bf16f(w[c])); // * weight → bf16
        vod[c] = vrow[c];
    }
}

// --------------------------------------------------------------------------- //
// host
// --------------------------------------------------------------------------- //
extern "C" void ap_time_path_cuda(const float* ts, const double* freq,
                                  const unsigned char* w0t, const float* sw0,
                                  const __nv_bfloat16* b0,
                                  const unsigned char* w1t, const float* sw1,
                                  const __nv_bfloat16* b1,
                                  const unsigned char* w2t, const float* sw2,
                                  const __nv_bfloat16* b2,
                                  __nv_bfloat16* y0, __nv_bfloat16* y1,
                                  __nv_bfloat16* y2,
                                  float* amax0, float* amax1, float* amax2,
                                  __nv_bfloat16* tmod, cudaStream_t stream) {
    constexpr int NF = 256, H = 1024, OUT = 6144;
    ap_sin_kernel<<<1, NF, 0, stream>>>(ts, freq, y0, amax0, amax1, amax2);
    ap_fc_kernel<NF, H, true, true><<<4, 256, 4 * NF, stream>>>(
        y0, amax0, w0t, sw0, b0, y1, amax1);
    ap_fc_kernel<H, H, true, true><<<4, 256, 4 * H, stream>>>(
        y1, amax1, w1t, sw1, b1, y2, amax2);
    ap_fc_kernel<H, OUT, false, false><<<24, 256, 4 * H, stream>>>(
        y2, amax2, w2t, sw2, b2, tmod, amax2);
    (void)y1; (void)y2;
}

extern "C" void ap_ctx_repack_cuda(const void* out_big, int L, int NB,
                                   const void* wnk, float eps, void* kv_out,
                                   cudaStream_t stream) {
    dim3 grid(NB, L), block(256);
    ap_ctx_repack_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(out_big), L, NB,
        reinterpret_cast<const __nv_bfloat16*>(wnk), eps,
        reinterpret_cast<__nv_bfloat16*>(kv_out));
}
