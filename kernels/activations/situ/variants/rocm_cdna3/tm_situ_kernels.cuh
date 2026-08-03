#include "hip/hip_runtime.h"
/**
 * @file
 * @brief SituGLU ("SITU"), the gated activation used by Kimi K3.
 *
 *   gate_out = beta * tanh(gate / beta) * sigmoid(gate)
 *   up_out   = linear_beta > 0 ? linear_beta * tanh(up / linear_beta) : up
 *   out      = gate_out * up_out
 *
 * Two shapes. The dense form takes [..., 2d] with gate and up in *separate
 * halves* (not interleaved) and writes [..., d]. The masked form is the MoE
 * one: [E, T, 2d] plus a per-expert token count, so experts that received no
 * tokens cost nothing and rows past a given expert's count are left untouched
 * rather than written with garbage.
 *
 * Both betas are soft clamps, not hard ones. `beta` bounds the gate through
 * tanh before the sigmoid; `linear_beta` optionally bounds the up half the
 * same way. Passing linear_beta <= 0 means "unset" and passes up through
 * unchanged -- the sentinel exists because the caller's config carries an
 * optional value and a separate bool would have to stay in sync with it.
 *
 * All compute is fp32 and results are written straight to `out`: the obvious
 * PyTorch spelling allocates roughly eight fp32 temporaries per call, which at
 * MoE width dominates both runtime and peak memory.
 *
 * Numerics note: accumulating in fp32 means the only meaningful error is the
 * single rounding on store, so measured error against an fp64 oracle sits at
 * about one ulp of the output dtype -- ~4e-3 relative for bf16, ~5e-4 for
 * fp16, ~3e-7 for fp32. Compare relatively, not absolutely: with Kimi K3's
 * beta=4 / linear_beta=25 the output reaches ~16, where one bf16 ulp is
 * already ~3e-2 in absolute terms.
 */
#pragma once

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

namespace quixicore {
namespace activations {

template <typename T> __host__ __device__ __forceinline__ float situ_tf(T v);
template <> __host__ __device__ __forceinline__ float situ_tf<float>(float v) { return v; }
template <> __host__ __device__ __forceinline__ float situ_tf<__half>(__half v) {
    return __half2float(v);
}
template <> __host__ __device__ __forceinline__ float situ_tf<__hip_bfloat16>(__hip_bfloat16 v) {
    return __bfloat162float(v);
}

template <typename T> __host__ __device__ __forceinline__ T situ_ft(float v);
template <> __host__ __device__ __forceinline__ float situ_ft<float>(float v) { return v; }
template <> __host__ __device__ __forceinline__ __half situ_ft<__half>(float v) {
    return __float2half(v);
}
template <> __host__ __device__ __forceinline__ __hip_bfloat16 situ_ft<__hip_bfloat16>(float v) {
    return __float2bfloat16(v);
}

/// Shared scalar body, so the dense and masked kernels cannot drift apart.
__device__ __forceinline__ float situ_apply(float g, float u, float beta,
                                            float inv_beta, bool clamp_up,
                                            float linear_beta,
                                            float inv_linear_beta) {
    const float gate_out = beta * tanhf(g * inv_beta) / (1.0f + expf(-g));
    const float up_out = clamp_up ? linear_beta * tanhf(u * inv_linear_beta) : u;
    return gate_out * up_out;
}

/// out[row, :d] = situ(in[row, :d], in[row, d:]); one block per row.
template <typename T>
__global__ void situ_and_mul(T* __restrict__ out, const T* __restrict__ input,
                             const int d, const float beta,
                             const float linear_beta) {
    const long long row = blockIdx.x;
    const T* gate_ptr = input + row * 2 * d;
    const T* up_ptr = gate_ptr + d;
    T* out_ptr = out + row * d;
    const bool clamp_up = linear_beta > 0.0f;
    const float inv_beta = 1.0f / beta;
    const float inv_linear_beta = clamp_up ? 1.0f / linear_beta : 0.0f;
    for (long long i = threadIdx.x; i < d; i += blockDim.x) {
        out_ptr[i] = situ_ft<T>(situ_apply(situ_tf<T>(gate_ptr[i]),
                                           situ_tf<T>(up_ptr[i]), beta, inv_beta,
                                           clamp_up, linear_beta, inv_linear_beta));
    }
}

/// MoE form. grid is (ceil(d/block), num_experts); expert_num_tokens[e] rows
/// of expert e are written and the rest are left alone.
template <typename T>
__global__ void masked_situ_and_mul(T* __restrict__ out,
                                    const T* __restrict__ input,
                                    const int* __restrict__ expert_num_tokens,
                                    const int max_num_tokens, const int d,
                                    const float beta, const float linear_beta) {
    const int expert = blockIdx.y;
    const int num_tokens = expert_num_tokens[expert];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d || num_tokens == 0) return;

    const bool clamp_up = linear_beta > 0.0f;
    const float inv_beta = 1.0f / beta;
    const float inv_linear_beta = clamp_up ? 1.0f / linear_beta : 0.0f;
    const long long expert_row = (long long)expert * max_num_tokens;
    for (int token = 0; token < num_tokens; ++token) {
        const long long row = expert_row + token;
        const T* gate_ptr = input + row * 2 * d;
        const T* up_ptr = gate_ptr + d;
        out[row * d + i] = situ_ft<T>(
            situ_apply(situ_tf<T>(gate_ptr[i]), situ_tf<T>(up_ptr[i]), beta,
                       inv_beta, clamp_up, linear_beta, inv_linear_beta));
    }
}

}  // namespace activations
}  // namespace quixicore
