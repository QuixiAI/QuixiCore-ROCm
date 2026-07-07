#include "hip/hip_runtime.h"
/**
 * @file
 * @brief Additional qgemm variants missing from the first CDNA3 MFMA port.
 */
#pragma once

#include "tm_qmm_mfma.cuh"
#include <hip/hip_fp16.h>

namespace tmq {

__device__ __forceinline__ half4_t load_xfrag_actorder(const __half* X,
                                                       const int* perm,
                                                       int K,
                                                       int m0,
                                                       int k0) {
    const int l = threadIdx.x & 63;
    const int m = m0 + (l & 15);
    const int k = k0 + (l >> 4) * 4;
    half4_t a;
    a[0] = (__fp16)__half2float(X[size_t(m) * K + perm[k + 0]]);
    a[1] = (__fp16)__half2float(X[size_t(m) * K + perm[k + 1]]);
    a[2] = (__fp16)__half2float(X[size_t(m) * K + perm[k + 2]]);
    a[3] = (__fp16)__half2float(X[size_t(m) * K + perm[k + 3]]);
    return a;
}

template<typename FMT>
__device__ __forceinline__ half4_t load_wfrag_blockscale(const uint8_t* Wq,
                                                         const __half* scale2d,
                                                         int bpr,
                                                         int k_tiles,
                                                         int n0,
                                                         int k0) {
    const int l = threadIdx.x & 63;
    const int n = n0 + (l & 15);
    const int k = k0 + (l >> 4) * 4;
    const int kb = k / FMT::block_k;
    const int cin = k % FMT::block_k;
    const uint8_t* base = Wq + (size_t(n) * bpr + kb) * FMT::block_bytes;
    const __half s = scale2d[size_t(n / 128) * k_tiles + k / 128];
    half4_t b;
    b[0] = (__fp16)__half2float(__hmul(__float2half(FMT::dequant(base, cin + 0)), s));
    b[1] = (__fp16)__half2float(__hmul(__float2half(FMT::dequant(base, cin + 1)), s));
    b[2] = (__fp16)__half2float(__hmul(__float2half(FMT::dequant(base, cin + 2)), s));
    b[3] = (__fp16)__half2float(__hmul(__float2half(FMT::dequant(base, cin + 3)), s));
    return b;
}

template<typename FMT>
__global__ void qgemm_actorder(float* Y,
                               const __half* X,
                               const uint8_t* Wq,
                               const int* perm,
                               int M,
                               int N,
                               int K) {
    const int n0 = blockIdx.x * 16;
    const int m0 = blockIdx.y * 16;
    const int bpr = K / FMT::block_k;

    float4_t acc = {0, 0, 0, 0};
    for (int k0 = 0; k0 < K; k0 += 16) {
        half4_t a = load_xfrag_actorder(X, perm, K, m0, k0);
        half4_t b = load_wfrag<FMT>(Wq, bpr, n0, k0);
        acc = mma_16x16x16(a, b, acc);
    }

    const int l = threadIdx.x & 63;
    const int n = n0 + (l & 15);
    const int mrow = m0 + (l >> 4) * 4;
    #pragma unroll
    for (int v = 0; v < 4; v++)
        if (mrow + v < M) Y[size_t(mrow + v) * N + n] = acc[v];
}

template<typename FMT>
__global__ void qgemm_blockscale(float* Y,
                                 const __half* X,
                                 const uint8_t* Wq,
                                 const __half* scale2d,
                                 int M,
                                 int N,
                                 int K) {
    const int n0 = blockIdx.x * 16;
    const int m0 = blockIdx.y * 16;
    const int bpr = K / FMT::block_k;
    const int k_tiles = K / 128;

    float4_t acc = {0, 0, 0, 0};
    for (int k0 = 0; k0 < K; k0 += 16) {
        half4_t a = load_xfrag(X, K, m0, k0);
        half4_t b = load_wfrag_blockscale<FMT>(Wq, scale2d, bpr, k_tiles, n0, k0);
        acc = mma_16x16x16(a, b, acc);
    }

    const int l = threadIdx.x & 63;
    const int n = n0 + (l & 15);
    const int mrow = m0 + (l >> 4) * 4;
    #pragma unroll
    for (int v = 0; v < 4; v++)
        if (mrow + v < M) Y[size_t(mrow + v) * N + n] = acc[v];
}

} // namespace tmq
