/**
 * @file
 * @brief Shared CDNA3 (gfx942) MFMA fragment types and storage traits.
 *
 * The 16x16x16 MFMA fragment layout is easy to get subtly wrong and expensive
 * to re-derive, so it lives here once. The layout below is the one validated by
 * the landed self-attention kernel
 * (`kernels/attention/gqa/variants/rocm_cdna3/attn_mfma.cuh`) against SDPA.
 *
 * For `v_mfma_f32_16x16x16*` with `lane` in [0,64), `lo = lane/16`, `li = lane%16`:
 *
 *     a[v] = A[m = li      ][k = 4*lo + v]      // 16x16, row-major operand A
 *     b[v] = B[k = 4*lo + v][n = li      ]      // 16x16, operand B
 *     c[v] = C[m = 4*lo + v][n = li      ]      // 16x16 fp32 accumulator
 *
 * Note the asymmetry: A and B are indexed by `li` in their non-contraction
 * dimension, but the accumulator's row index is `4*lo + v`. Writing C back with
 * A's indexing is the classic bug -- it transposes the result in a way that
 * still "looks plausible" on square inputs.
 *
 * Traits let one kernel body serve bf16 and fp16: the intrinsic and the
 * 4-element fragment type differ, everything else does not.
 */
#ifndef QUIXICORE_ROCM_CDNA3_MFMA_CUH
#define QUIXICORE_ROCM_CDNA3_MFMA_CUH

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

namespace qc {

using bf16 = __hip_bfloat16;
using fp16 = __half;

typedef __attribute__((__vector_size__(4 * sizeof(short)))) short short4_t;
typedef __attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16 half4_t;
typedef __attribute__((__vector_size__(4 * sizeof(float)))) float float4_t;

/// MFMA lane decomposition: `lo` selects the 4-element k-group, `li` the
/// row/column within the 16x16 tile.
__device__ __forceinline__ int mfma_lo(int lane) { return lane >> 4; }
__device__ __forceinline__ int mfma_li(int lane) { return lane & 15; }

struct Bf16Traits {
    using storage = bf16;
    using frag = short4_t;
    static constexpr const char *name = "bf16";
    __device__ static float to_float(storage v) { return __bfloat162float(v); }
    __device__ static storage from_float(float v) { return __float2bfloat16(v); }
    __device__ static void put(frag &f, int v, storage x) {
        short s;
        __builtin_memcpy(&s, &x, 2);
        ((short *)&f)[v] = s;
    }
    __device__ static float4_t mma(frag a, frag b, float4_t c) {
        return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, c, 0, 0, 0);
    }
};

struct Fp16Traits {
    using storage = fp16;
    using frag = half4_t;
    static constexpr const char *name = "fp16";
    __device__ static float to_float(storage v) { return __half2float(v); }
    __device__ static storage from_float(float v) { return __float2half(v); }
    __device__ static void put(frag &f, int v, storage x) {
        _Float16 h;
        __builtin_memcpy(&h, &x, 2);
        ((_Float16 *)&f)[v] = h;
    }
    __device__ static float4_t mma(frag a, frag b, float4_t c) {
        return __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c, 0, 0, 0);
    }
};

/// Zeroed fp32 accumulator fragment.
__device__ __forceinline__ float4_t mfma_zero() { return float4_t{0.f, 0.f, 0.f, 0.f}; }

}  // namespace qc

#endif  // QUIXICORE_ROCM_CDNA3_MFMA_CUH
