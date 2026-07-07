#include "hip/hip_runtime.h"
/**
 * @file
 * @brief CDNA3 port of norm+quant epilogues and asymmetric int8 quantizers.
 *
 * This is the ROCm counterpart to QuixiCore-CUDA/kernels/elementwise/
 * tm_norm_quant_kernels.cuh. It keeps the CUDA behavior but replaces CUDA
 * headers/types with HIP and uses the local ROCm quant format layer for FP8.
 */
#pragma once

#include "../../../../quantization/qgemm/variants/rocm_cdna3/quant_formats.cuh"
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <cstdint>

namespace tmnq {

template <typename T> __device__ __forceinline__ float nf(T v);
template <> __device__ __forceinline__ float nf<float>(float v) { return v; }
template <> __device__ __forceinline__ float nf<__half>(__half v) { return __half2float(v); }
template <> __device__ __forceinline__ float nf<__hip_bfloat16>(__hip_bfloat16 v) { return __bfloat162float(v); }

template <typename T> __device__ __forceinline__ T fn(float v);
template <> __device__ __forceinline__ float fn<float>(float v) { return v; }
template <> __device__ __forceinline__ __half fn<__half>(float v) { return __float2half(v); }
template <> __device__ __forceinline__ __hip_bfloat16 fn<__hip_bfloat16>(float v) { return __float2bfloat16(v); }

__device__ __forceinline__ int8_t i8sat(float x) {
    return int8_t(int(fmaxf(-128.0f, fminf(127.0f, rintf(x)))));
}

__device__ __forceinline__ int8_t i8sat_i(int x) {
    return int8_t(max(-128, min(127, x)));
}

__device__ __forceinline__ float warp_sum32_f(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor(v, off);
    return v;
}

__device__ __forceinline__ float warp_max32_f(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor(v, off));
    return v;
}

__device__ __forceinline__ float warp_min32_f(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = fminf(v, __shfl_xor(v, off));
    return v;
}

__device__ __forceinline__ float block_sum_bcast(float v, float* partials, float* bcast) {
    v = warp_sum32_f(v);
    const int w = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int nw = blockDim.x >> 5;
    if (lane == 0) partials[w] = v;
    __syncthreads();
    if (w == 0) {
        float t = (lane < nw) ? partials[lane] : 0.0f;
        t = warp_sum32_f(t);
        if (lane == 0) *bcast = t;
    }
    __syncthreads();
    return *bcast;
}

__device__ __forceinline__ float block_max_bcast(float v, float* partials, float* bcast) {
    v = warp_max32_f(v);
    const int w = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int nw = blockDim.x >> 5;
    if (lane == 0) partials[w] = v;
    __syncthreads();
    if (w == 0) {
        float t = (lane < nw) ? partials[lane] : 0.0f;
        t = warp_max32_f(t);
        if (lane == 0) *bcast = t;
    }
    __syncthreads();
    return *bcast;
}

// RMSNorm (+optional residual) with quantized output. One block per row.
// FP8 emits e4m3 codes; otherwise codes stores int8 values byte-for-byte.
template <typename T, bool FP8, bool DYN, bool RESID>
__global__ void rms_norm_quant(uint8_t* __restrict__ codes,
                               float* __restrict__ scale_out,
                               T* __restrict__ res_out,
                               const T* __restrict__ x,
                               const T* __restrict__ residual,
                               const T* __restrict__ weight,
                               int D,
                               float eps,
                               float inv_static) {
    const long base = long(blockIdx.x) * D;
    const int tid = threadIdx.x;
    const int BLK = blockDim.x;
    __shared__ float part[32];
    __shared__ float bcast;

    float ss = 0.0f;
    float maxvw = 0.0f;
    for (int j = tid; j < D; j += BLK) {
        float v = nf(x[base + j]);
        if constexpr (RESID) {
            v += nf(residual[base + j]);
            res_out[base + j] = fn<T>(v);
        }
        ss += v * v;
        if constexpr (DYN) maxvw = fmaxf(maxvw, fabsf(v * nf(weight[j])));
    }

    ss = block_sum_bcast(ss, part, &bcast);
    const float inv_rms = rsqrtf(ss / float(D) + eps);
    const float qmax = FP8 ? 448.0f : 127.0f;

    float inv_scale = inv_static;
    if constexpr (DYN) {
        const float amax = block_max_bcast(maxvw, part, &bcast) * inv_rms;
        const float scale = amax / qmax;
        inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
        if (tid == 0) scale_out[blockIdx.x] = scale;
    }

    for (int j = tid; j < D; j += BLK) {
        const float v = RESID ? nf(res_out[base + j]) : nf(x[base + j]);
        const float y = v * inv_rms * nf(weight[j]) * inv_scale;
        codes[base + j] = FP8 ? tmq::e4m3_encode(y) : uint8_t(i8sat(y));
    }
}

// Asymmetric int8 quantization. Dynamic mode emits per-row scale and zero point.
template <typename T, bool DYN>
__global__ void azp_int8_quant(int8_t* __restrict__ codes,
                               float* __restrict__ scale_out,
                               int* __restrict__ azp_out,
                               const T* __restrict__ x,
                               int D,
                               float scale_static,
                               int azp_static) {
    const long base = long(blockIdx.x) * D;
    const int lane = threadIdx.x & 31;
    float scale = scale_static;
    int zp = azp_static;
    if constexpr (DYN) {
        float mn = 3.4028234663852886e38f;
        float mx = -3.4028234663852886e38f;
        for (int j = lane; j < D; j += 32) {
            const float v = nf(x[base + j]);
            mn = fminf(mn, v);
            mx = fmaxf(mx, v);
        }
        mn = warp_min32_f(mn);
        mx = warp_max32_f(mx);
        scale = (mx - mn) / 255.0f;
        zp = scale > 0.0f ? int(rintf(-128.0f - mn / scale)) : 0;
        if (lane == 0) {
            scale_out[blockIdx.x] = scale;
            azp_out[blockIdx.x] = zp;
        }
    }
    const float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
    for (int j = lane; j < D; j += 32)
        codes[base + j] = i8sat_i(int(rintf(nf(x[base + j]) * inv)) + zp);
}

// Symmetric int8 per token group. One 32-lane subgroup per (token, group).
template <typename T>
__global__ void per_token_group_int8_quant(int8_t* __restrict__ codes,
                                           float* __restrict__ scales,
                                           const T* __restrict__ x,
                                           int hidden,
                                           int group_size,
                                           int n_groups,
                                           float eps) {
    const int token = blockIdx.y;
    const int g = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int c0 = g * group_size;
    const T* row = x + long(token) * hidden;
    float amax = 0.0f;
    for (int c = c0 + lane; c < c0 + group_size && c < hidden; c += 32)
        amax = fmaxf(amax, fabsf(nf(row[c])));
    amax = warp_max32_f(fmaxf(amax, eps));
    const float scale = amax / 127.0f;
    const float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
    if (lane == 0) scales[long(token) * n_groups + g] = scale;
    for (int c = c0 + lane; c < c0 + group_size && c < hidden; c += 32)
        codes[long(token) * hidden + c] = i8sat(nf(row[c]) * inv);
}

} // namespace tmnq
