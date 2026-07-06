/**
 * @file
 * @brief CDNA3 (gfx942) MFMA replacement for the CUDA tm_qmm.cuh m16n8k16
 * tensor-core path. The CUDA version used the NVIDIA PTX
 * `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` with a 32-lane fragment
 * layout; this uses `v_mfma_f32_16x16x16_f16` (one 16x16x16 op per K=16 step)
 * across a full 64-wide wavefront.
 *
 * MFMA lane layout for v_mfma_f32_16x16x16_f16 (64 lanes, 4 regs/lane):
 *   A[M=16,K=16] : lane l, reg v -> A[m = l%16][k = 4*(l/16) + v]
 *   B[K=16,N=16] : lane l, reg v -> B[k = 4*(l/16) + v][n = l%16]
 *   D[M=16,N=16] : lane l, reg v -> D[m = 4*(l/16) + v][n = l%16]
 *
 * For qgemm Y(M,N) = X(M,K) @ dequant(W(N,K))^T we set A=X (A[m][k]) and
 * B=W^T (B[k][n] = W[n][k]). So each lane reads 4 contiguous K of one X row and
 * 4 contiguous K of one W row, and owns a 4-row column of the output.
 */
#pragma once
#include "quant_formats.cuh"
#include "quant_formats_tables.cuh"
#include <hip/hip_fp16.h>

namespace tmq {

typedef __attribute__((__vector_size__(4 * sizeof(__fp16)))) __fp16 half4_t;
typedef __attribute__((__vector_size__(4 * sizeof(float))))  float  float4_t;

// fp16 passthrough "format": lets the full-dequant route reuse the same kernel.
struct fp16_raw {
    static constexpr int block_k = 16, block_bytes = 32;   // 16 halfs
    __device__ static float dequant(const uint8_t* base, int col) {
        return __half2float(reinterpret_cast<const __half*>(base)[col]);
    }
};

// A operand: 4 contiguous K of one X row (row-major (M,K), 8-byte aligned since
// k is a multiple of 4). __half and __fp16 share IEEE fp16 bits -> raw reinterpret.
__device__ __forceinline__ half4_t load_xfrag(const __half* X, int K, int m0, int k0) {
    const int l = threadIdx.x & 63;
    const int m = m0 + (l & 15);
    const int k = k0 + (l >> 4) * 4;
    return *reinterpret_cast<const half4_t*>(X + size_t(m) * K + k);
}

// B operand: 4 contiguous K of one W row (n = n0 + l%16), dequantized. The 4 k's
// (k0 + 4*(l/16) .. +3) stay inside one quant block: k-in-block is 0/4/8/12 and
// every format has block_k a multiple of 16, so col..col+3 is within the block.
template<typename FMT>
__device__ __forceinline__ half4_t load_wfrag(const uint8_t* Wq, int bpr, int n0, int k0) {
    const int l = threadIdx.x & 63;
    const int n = n0 + (l & 15);
    const int k = k0 + (l >> 4) * 4;
    const int kb = k / FMT::block_k, cin = k % FMT::block_k;
    const uint8_t* base = Wq + (size_t(n) * bpr + kb) * FMT::block_bytes;
    half4_t b;
    b[0] = (__fp16)FMT::dequant(base, cin);
    b[1] = (__fp16)FMT::dequant(base, cin + 1);
    b[2] = (__fp16)FMT::dequant(base, cin + 2);
    b[3] = (__fp16)FMT::dequant(base, cin + 3);
    return b;
}

__device__ __forceinline__ float4_t mma_16x16x16(half4_t a, half4_t b, float4_t acc) {
    return __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, acc, 0, 0, 0);
}

}  // namespace tmq
