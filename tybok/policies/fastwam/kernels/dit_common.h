// Device-side helpers shared by action DiT (adit) and video DiT (vdit): bf16/fp8 conversion, ldmatrix, mma.
#ifndef FASTWAM_KERNELS_DIT_COMMON_H_
#define FASTWAM_KERNELS_DIT_COMMON_H_

#include "common.h"

__device__ __forceinline__ unsigned short bf16_bits(float v) {
    return __bfloat16_as_ushort(__float2bfloat16_rn(v));
}

__device__ __forceinline__ unsigned char fp8_rne(float x) {
    unsigned bits = __float_as_uint(x);
    unsigned sign = (bits >> 24) & 0x80u;
    unsigned a = bits & 0x7fffffffu;
    bool ovf = a >= 0x43E80000u;
    unsigned res_sub = __float_as_uint(__uint_as_float(a) + 24576.0f) - 0x46800000u;
    unsigned mant_odd = (a >> 20) & 1u;
    unsigned n = a + 0xC4000000u + 0x7FFFFu + mant_odd;
    unsigned res = ovf ? 0x7Fu : (a < 0x3C800000u ? res_sub : (n >> 20));
    return (unsigned char)(res | sign);
}

// Per-step bf16 rounding of modulate (the engine's modulate semantics); scale/shift are bf16 values held exactly as fp32
__device__ __forceinline__ float mod_bf16(float x, float mean, float rstd, float scale, float shift) {
    float nb = bf16f(bf16_bits((x - mean) * rstd));
    float op1 = bf16f(bf16_bits(1.0f + scale));
    return bf16f(bf16_bits(bf16f(bf16_bits(nb * op1)) + shift));
}

// ---- ldmatrix: replaces the 4 scalar LDS loads of an A/B fragment with 1 (only the read path changes, the smem layout does not) ----
// The row stride comes from the caller (BP=136: 17 16B units, 17 ≡ 1 mod 8), so the 8 rows land in 8 distinct 16B-bank groups
// ⇒ ldmatrix has no bank conflicts. The values in the fragment are bit-identical to the originals.
__device__ __forceinline__ void ldm_x2(const __nv_bfloat16* p, unsigned& r0, unsigned& r1) {
    unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1) : "r"(a));
}
__device__ __forceinline__ void ldm_x4(const __nv_bfloat16* p, unsigned& r0,
                                       unsigned& r1, unsigned& r2, unsigned& r3) {
    unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ void ldm_x4_t(const __nv_bfloat16* p, unsigned& r0,
                                         unsigned& r1, unsigned& r2, unsigned& r3) {
    unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}

__device__ __forceinline__ void mma_f32(float c[4], unsigned a0, unsigned a1,
                                        unsigned a2, unsigned a3,
                                        unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

#endif  // FASTWAM_KERNELS_DIT_COMMON_H_
