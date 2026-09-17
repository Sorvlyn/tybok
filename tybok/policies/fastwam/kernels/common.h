// Device-side math helpers shared by all three families (adit / vdit / tmt5).
#ifndef FASTWAM_KERNELS_COMMON_H_
#define FASTWAM_KERNELS_COMMON_H_

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstddef>

__device__ __forceinline__ float bf16f(unsigned short u) {
    return __bfloat162float(__ushort_as_bfloat16(u));
}

// tanh approximation of GELU, equal to torch's `nn.GELU(approximate="tanh")`:
//     gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// Same convention as `tmt5_ffn_gelu_new` in `tmt5_gemm_core.cu`, the PyTorch side of this repo
// (`nn.GELU(approximate="tanh")` in `dit_block.py` / `video_dit.py`), and lerobot
// (`nn.GELU(approximate="tanh")` in `lerobot/policies/fastwam/wan/*.py`).
//
// History (fixed in 2026-09): this once read `0.5*x*(1+tanh(x*(c1 + c2*x*x)))` -- the cubic term
// was missing the c1 factor, i.e. the coefficient was 0.044715 instead of c1*0.044715 ~= 0.035677.
// The worst deviation from torch was 6.3e-3 (x~=1.8, ~0.36% relative) -- a systematic bias, not rounding noise; adit_ffn and vdit_ffn both use it.
__device__ __forceinline__ float gelu_tanh(float x) {
    constexpr float c1 = 0.7978845608028654f;   // sqrt(2/pi)
    constexpr float c2 = 0.044715f;
    return 0.5f * x * (1.0f + tanhf(c1 * (x + c2 * x * x * x)));
}

#endif  // FASTWAM_KERNELS_COMMON_H_
