// Device-side helpers specific to the action DiT (adit).
#ifndef FASTWAM_KERNELS_ADIT_COMMON_H_
#define FASTWAM_KERNELS_ADIT_COMMON_H_

#include "dit_common.h"

__device__ __forceinline__ uint32_t bf16_pair(float lo, float hi) {
    return (uint32_t)bf16_bits(lo) | ((uint32_t)bf16_bits(hi) << 16);
}

#endif  // FASTWAM_KERNELS_ADIT_COMMON_H_
