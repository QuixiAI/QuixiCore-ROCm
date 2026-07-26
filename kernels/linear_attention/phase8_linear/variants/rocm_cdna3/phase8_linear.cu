/**
 * @file
 * @brief Phase 8 linear-attention, RWKV, and GatedDeltaNet parity ports.
 */
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include "../../../../common/cdna3_harness.cuh"

namespace {

constexpr int kMaxWaveItems = 4;

__host__ __device__ __forceinline__ size_t idx4(long long a, long long b,
                                                long long c, long long d,
                                                long long B, long long C,
                                                long long D) {
    return static_cast<size_t>(((a * B + b) * C + c) * D + d);
}

__host__ __device__ __forceinline__ size_t recurrent_offset(long long sequence,
                                                            long long token,
                                                            long long head,
                                                            long long dim,
                                                            long long tokens,
                                                            long long heads) {
    return static_cast<size_t>(((sequence * tokens + token) * heads + head) * dim);
}

__host__ __device__ __forceinline__ size_t recurrent_state_offset(
    long long sequence, long long head, long long row, long long col,
    long long heads, long long dim) {
    return static_cast<size_t>(((sequence * heads + head) * dim + row) * dim + col);
}

__device__ __forceinline__ float sigmoid_device(float value) {
    if (value >= 0.0f) return 1.0f / (1.0f + expf(-value));
    const float e = expf(value);
    return e / (1.0f + e);
}

__host__ float sigmoid_host(float value) {
    if (value >= 0.0f) return 1.0f / (1.0f + std::exp(-value));
    const float e = std::exp(value);
    return e / (1.0f + e);
}

__device__ __forceinline__ float softplus_device(float value) {
    if (value > 20.0f) return value;
    if (value < -20.0f) return expf(value);
    return log1pf(expf(value));
}

float softplus_host(float value) {
    if (value > 20.0f) return value;
    if (value < -20.0f) return std::exp(value);
    return std::log1p(std::exp(value));
}

std::vector<double> to_ref(const std::vector<float> &values) {
    return std::vector<double>(values.begin(), values.end());
}

__global__ void copy_float_kernel(const float *input, float *output,
                                  long long count) {
    const long long index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) output[index] = input[index];
}

// ---------------------------------------------------------------------------
// Gated linear attention and RWKV recurrences
// ---------------------------------------------------------------------------

__global__ void gated_linear_attention_scalar_kernel(
    const float *key, const float *value, const float *query, const float *gate,
    const float *initial_state, float *output, float *final_state,
    long long sequences, long long tokens, long long heads, long long dim,
    float scale) {
    const long long task = blockIdx.x;
    if (threadIdx.x != 0 || task >= sequences * heads) return;
    const long long sequence = task / heads;
    const long long head = task % heads;
    const long long matrix = dim * dim;
    float *state = final_state + task * matrix;
    const float *initial = initial_state + task * matrix;
    for (long long index = 0; index < matrix; ++index) state[index] = initial[index];
    for (long long token = 0; token < tokens; ++token) {
        const size_t offset = recurrent_offset(sequence, token, head, dim, tokens, heads);
        for (long long j = 0; j < dim; ++j) output[offset + j] = 0.0f;
        for (long long i = 0; i < dim; ++i) {
            const float key_value = key[offset + i];
            const float query_value = query[offset + i] * scale;
            const float gate_value = gate[offset + i];
            float *state_row = state + i * dim;
            for (long long j = 0; j < dim; ++j) {
                state_row[j] = state_row[j] * gate_value + key_value * value[offset + j];
                output[offset + j] += state_row[j] * query_value;
            }
        }
    }
}

__global__ void gated_linear_attention_wave_kernel(
    const float *key, const float *value, const float *query, const float *gate,
    const float *initial_state, float *output, float *final_state,
    long long sequences, long long tokens, long long heads, long long dim,
    float scale) {
    const long long task = blockIdx.x;
    const int lane = threadIdx.x;
    const long long j = task % dim;
    const long long head = (task / dim) % heads;
    const long long sequence = task / (dim * heads);
    if (sequence >= sequences || lane >= qc::kWave) return;
    float state[kMaxWaveItems]{};
    int count = 0;
    for (long long i = lane; i < dim && count < kMaxWaveItems; i += qc::kWave, ++count) {
        state[count] = initial_state[recurrent_state_offset(sequence, head, i, j, heads, dim)];
    }
    for (long long token = 0; token < tokens; ++token) {
        const size_t offset = recurrent_offset(sequence, token, head, dim, tokens, heads);
        const float value_j = value[offset + j];
        float partial = 0.0f;
        count = 0;
        for (long long i = lane; i < dim && count < kMaxWaveItems; i += qc::kWave, ++count) {
            state[count] = state[count] * gate[offset + i] + key[offset + i] * value_j;
            partial += state[count] * query[offset + i] * scale;
        }
        const float result = qc::wave_reduce_sum(partial);
        if (lane == 0) output[offset + j] = result;
    }
    count = 0;
    for (long long i = lane; i < dim && count < kMaxWaveItems; i += qc::kWave, ++count) {
        final_state[recurrent_state_offset(sequence, head, i, j, heads, dim)] = state[count];
    }
}

__global__ void rwkv_wkv6_scalar_kernel(
    const float *key, const float *value, const float *receptance,
    const float *time_first, const float *time_decay, const float *initial_state,
    float *output, float *final_state, long long sequences, long long tokens,
    long long heads, long long dim) {
    const long long task = blockIdx.x;
    if (threadIdx.x != 0 || task >= sequences * heads) return;
    const long long sequence = task / heads;
    const long long head = task % heads;
    const long long matrix = dim * dim;
    float *state = final_state + task * matrix;
    const float *initial = initial_state + task * matrix;
    for (long long index = 0; index < matrix; ++index) state[index] = initial[index];
    for (long long token = 0; token < tokens; ++token) {
        const size_t offset = recurrent_offset(sequence, token, head, dim, tokens, heads);
        for (long long j = 0; j < dim; ++j) output[offset + j] = 0.0f;
        for (long long i = 0; i < dim; ++i) {
            const float k = key[offset + i];
            const float r = receptance[offset + i];
            const float first = time_first[head * dim + i];
            const float decay = time_decay[offset + i];
            float *state_row = state + i * dim;
            for (long long j = 0; j < dim; ++j) {
                const float kv = k * value[offset + j];
                const float previous = state_row[j];
                output[offset + j] += (previous + kv * first) * r;
                state_row[j] = previous * decay + kv;
            }
        }
    }
}

__global__ void rwkv_wkv6_wave_kernel(
    const float *key, const float *value, const float *receptance,
    const float *time_first, const float *time_decay, const float *initial_state,
    float *output, float *final_state, long long sequences, long long tokens,
    long long heads, long long dim) {
    const long long task = blockIdx.x;
    const int lane = threadIdx.x;
    const long long j = task % dim;
    const long long head = (task / dim) % heads;
    const long long sequence = task / (dim * heads);
    if (sequence >= sequences || lane >= qc::kWave) return;
    float state[kMaxWaveItems]{};
    int count = 0;
    for (long long i = lane; i < dim && count < kMaxWaveItems; i += qc::kWave, ++count) {
        state[count] = initial_state[recurrent_state_offset(sequence, head, i, j, heads, dim)];
    }
    for (long long token = 0; token < tokens; ++token) {
        const size_t offset = recurrent_offset(sequence, token, head, dim, tokens, heads);
        const float value_j = value[offset + j];
        float partial = 0.0f;
        count = 0;
        for (long long i = lane; i < dim && count < kMaxWaveItems; i += qc::kWave, ++count) {
            const float kv = key[offset + i] * value_j;
            const float previous = state[count];
            partial += (previous + kv * time_first[head * dim + i]) * receptance[offset + i];
            state[count] = previous * time_decay[offset + i] + kv;
        }
        const float result = qc::wave_reduce_sum(partial);
        if (lane == 0) output[offset + j] = result;
    }
    count = 0;
    for (long long i = lane; i < dim && count < kMaxWaveItems; i += qc::kWave, ++count) {
        final_state[recurrent_state_offset(sequence, head, i, j, heads, dim)] = state[count];
    }
}

__global__ void rwkv_wkv7_scalar_kernel(
    const float *receptance, const float *decay, const float *key,
    const float *value, const float *a, const float *b,
    const float *initial_state, float *output, float *final_state,
    long long sequences, long long tokens, long long heads, long long dim) {
    const long long task = blockIdx.x;
    if (threadIdx.x != 0 || task >= sequences * heads) return;
    const long long sequence = task / heads;
    const long long head = task % heads;
    const long long matrix = dim * dim;
    float *state = final_state + task * matrix;
    const float *initial = initial_state + task * matrix;
    for (long long index = 0; index < matrix; ++index) state[index] = initial[index];
    for (long long token = 0; token < tokens; ++token) {
        const size_t offset = recurrent_offset(sequence, token, head, dim, tokens, heads);
        for (long long i = 0; i < dim; ++i) {
            float *state_row = state + i * dim;
            float state_a = 0.0f;
            for (long long j = 0; j < dim; ++j) state_a += a[offset + j] * state_row[j];
            float result = 0.0f;
            for (long long j = 0; j < dim; ++j) {
                state_row[j] = state_row[j] * decay[offset + j] +
                               value[offset + i] * key[offset + j] +
                               state_a * b[offset + j];
                result += state_row[j] * receptance[offset + j];
            }
            output[offset + i] = result;
        }
    }
}

__global__ void rwkv_wkv7_wave_kernel(
    const float *receptance, const float *decay, const float *key,
    const float *value, const float *a, const float *b,
    const float *initial_state, float *output, float *final_state,
    long long sequences, long long tokens, long long heads, long long dim) {
    const long long task = blockIdx.x;
    const int lane = threadIdx.x;
    const long long i_out = task % dim;
    const long long head = (task / dim) % heads;
    const long long sequence = task / (dim * heads);
    if (sequence >= sequences || lane >= qc::kWave) return;
    float state[kMaxWaveItems]{};
    int count = 0;
    for (long long j = lane; j < dim && count < kMaxWaveItems; j += qc::kWave, ++count) {
        state[count] =
            initial_state[recurrent_state_offset(sequence, head, i_out, j, heads, dim)];
    }
    for (long long token = 0; token < tokens; ++token) {
        const size_t offset = recurrent_offset(sequence, token, head, dim, tokens, heads);
        float partial_a = 0.0f;
        count = 0;
        for (long long j = lane; j < dim && count < kMaxWaveItems; j += qc::kWave, ++count) {
            partial_a += a[offset + j] * state[count];
        }
        const float state_a = qc::wave_reduce_sum(partial_a);
        float partial_out = 0.0f;
        count = 0;
        for (long long j = lane; j < dim && count < kMaxWaveItems; j += qc::kWave, ++count) {
            state[count] = state[count] * decay[offset + j] +
                           value[offset + i_out] * key[offset + j] +
                           state_a * b[offset + j];
            partial_out += state[count] * receptance[offset + j];
        }
        const float result = qc::wave_reduce_sum(partial_out);
        if (lane == 0) output[offset + i_out] = result;
    }
    count = 0;
    for (long long j = lane; j < dim && count < kMaxWaveItems; j += qc::kWave, ++count) {
        final_state[recurrent_state_offset(sequence, head, i_out, j, heads, dim)] =
            state[count];
    }
}

// ---------------------------------------------------------------------------
// Linear attention, unnormalized identity feature map
// ---------------------------------------------------------------------------

__global__ void linear_attention_unnormalized_scalar_kernel(
    const float *q, const float *k, const float *v, float *out, long long batch,
    long long heads, long long sequence, long long dim) {
    const long long bh = blockIdx.x;
    if (threadIdx.x != 0 || bh >= batch * heads) return;
    const size_t base = static_cast<size_t>(bh * sequence * dim);
    for (long long token = 0; token < sequence; ++token) {
        for (long long j = 0; j < dim; ++j) {
            double acc = 0.0;
            for (long long i = 0; i < dim; ++i) {
                double state = 0.0;
                for (long long source = 0; source < sequence; ++source) {
                    state += static_cast<double>(k[base + source * dim + i]) *
                             v[base + source * dim + j];
                }
                acc += static_cast<double>(q[base + token * dim + i]) * state;
            }
            out[base + token * dim + j] = static_cast<float>(acc);
        }
    }
}

__global__ void linear_attention_kv_kernel(const float *k, const float *v,
                                           double *kv, long long batch,
                                           long long heads, long long sequence,
                                           long long dim) {
    const long long i = blockIdx.x;
    const long long j = blockIdx.y;
    const long long bh = blockIdx.z;
    const int lane = threadIdx.x;
    if (bh >= batch * heads || i >= dim || j >= dim) return;
    const size_t base = static_cast<size_t>(bh * sequence * dim);
    double partial = 0.0;
    for (long long token = lane; token < sequence; token += qc::kWave) {
        partial += static_cast<double>(k[base + token * dim + i]) *
                   v[base + token * dim + j];
    }
    partial = qc::wave_reduce_sum(partial);
    if (lane == 0) kv[static_cast<size_t>((bh * dim + i) * dim + j)] = partial;
}

__global__ void linear_attention_out_kernel(const float *q, const double *kv,
                                            float *out, long long batch,
                                            long long heads, long long sequence,
                                            long long dim) {
    const long long token = blockIdx.x;
    const long long j = blockIdx.y;
    const long long bh = blockIdx.z;
    const int lane = threadIdx.x;
    if (bh >= batch * heads || token >= sequence || j >= dim) return;
    const size_t base = static_cast<size_t>(bh * sequence * dim);
    double partial = 0.0;
    for (long long i = lane; i < dim; i += qc::kWave) {
        partial += static_cast<double>(q[base + token * dim + i]) *
                   kv[static_cast<size_t>((bh * dim + i) * dim + j)];
    }
    partial = qc::wave_reduce_sum(partial);
    if (lane == 0) out[base + token * dim + j] = static_cast<float>(partial);
}

// ---------------------------------------------------------------------------
// GatedDeltaNet components
// ---------------------------------------------------------------------------

__global__ void gdn_recurrence_scalar_kernel(
    const float *q, const float *k, const float *v, const float *decay,
    const float *beta, float *state_pool_out, const int *cumulative_lengths,
    const int *slot_mapping, float *out, long long requests, long long key_heads,
    long long value_heads, long long key_dim, long long value_dim,
    bool load_initial) {
    const long long request = blockIdx.x;
    if (threadIdx.x != 0 || request >= requests) return;
    const long long head_group = value_heads / key_heads;
    const long long start = cumulative_lengths[request];
    const long long end = cumulative_lengths[request + 1];
    const int slot = slot_mapping[request];
    for (long long value_head = 0; value_head < value_heads; ++value_head) {
        const long long key_head = value_head / head_group;
        for (long long value_index = 0; value_index < value_dim; ++value_index) {
            float *state = state_pool_out +
                           ((static_cast<long long>(slot) * value_heads + value_head) *
                                value_dim +
                            value_index) *
                               key_dim;
            if (!load_initial) {
                for (long long dim = 0; dim < key_dim; ++dim) state[dim] = 0.0f;
            }
            for (long long token = start; token < end; ++token) {
                const float *key = k + (token * key_heads + key_head) * key_dim;
                const float *query = q + (token * key_heads + key_head) * key_dim;
                const float gate = decay[token * value_heads + value_head];
                double memory = 0.0;
                for (long long dim = 0; dim < key_dim; ++dim) {
                    state[dim] *= gate;
                    memory += static_cast<double>(state[dim]) * key[dim];
                }
                const double correction =
                    (v[(token * value_heads + value_head) * value_dim + value_index] -
                     memory) *
                    beta[token * value_heads + value_head];
                double result = 0.0;
                for (long long dim = 0; dim < key_dim; ++dim) {
                    state[dim] += static_cast<float>(key[dim] * correction);
                    result += static_cast<double>(state[dim]) * query[dim];
                }
                out[(token * value_heads + value_head) * value_dim + value_index] =
                    static_cast<float>(result);
            }
        }
    }
}

__global__ void gdn_recurrence_wave_kernel(
    const float *q, const float *k, const float *v, const float *decay,
    const float *beta, float *state_pool_out, const int *cumulative_lengths,
    const int *slot_mapping, float *out, long long requests, long long key_heads,
    long long value_heads, long long key_dim, long long value_dim,
    bool load_initial) {
    const long long value_index = blockIdx.x;
    const long long value_head = blockIdx.y;
    const long long request = blockIdx.z;
    const int lane = threadIdx.x;
    if (request >= requests || value_head >= value_heads ||
        value_index >= value_dim || lane >= qc::kWave) {
        return;
    }
    const long long head_group = value_heads / key_heads;
    const long long key_head = value_head / head_group;
    const long long start = cumulative_lengths[request];
    const long long end = cumulative_lengths[request + 1];
    const int slot = slot_mapping[request];
    float *state_ptr =
        state_pool_out +
        ((static_cast<long long>(slot) * value_heads + value_head) * value_dim +
         value_index) *
            key_dim;
    float state[kMaxWaveItems]{};
    int count = 0;
    for (long long dim = lane; dim < key_dim && count < kMaxWaveItems;
         dim += qc::kWave, ++count) {
        state[count] = load_initial ? state_ptr[dim] : 0.0f;
    }
    for (long long token = start; token < end; ++token) {
        const float *key = k + (token * key_heads + key_head) * key_dim;
        const float *query = q + (token * key_heads + key_head) * key_dim;
        const float gate = decay[token * value_heads + value_head];
        float partial_memory = 0.0f;
        count = 0;
        for (long long dim = lane; dim < key_dim && count < kMaxWaveItems;
             dim += qc::kWave, ++count) {
            state[count] *= gate;
            partial_memory += state[count] * key[dim];
        }
        const float memory = qc::wave_reduce_sum(partial_memory);
        const float correction =
            (v[(token * value_heads + value_head) * value_dim + value_index] -
             memory) *
            beta[token * value_heads + value_head];
        float partial_out = 0.0f;
        count = 0;
        for (long long dim = lane; dim < key_dim && count < kMaxWaveItems;
             dim += qc::kWave, ++count) {
            state[count] += key[dim] * correction;
            partial_out += state[count] * query[dim];
        }
        const float result = qc::wave_reduce_sum(partial_out);
        if (lane == 0) {
            out[(token * value_heads + value_head) * value_dim + value_index] = result;
        }
    }
    count = 0;
    for (long long dim = lane; dim < key_dim && count < kMaxWaveItems;
         dim += qc::kWave, ++count) {
        state_ptr[dim] = state[count];
    }
}

__global__ void gdn_short_conv_scalar_kernel(
    const float *x, const float *weight, float *state_pool_out,
    const int *cumulative_lengths, const int *slot_mapping, float *out,
    long long requests, long long channels, long long kernel_size,
    bool load_initial, bool apply_silu) {
    const long long request = blockIdx.x;
    if (threadIdx.x != 0 || request >= requests) return;
    const long long start = cumulative_lengths[request];
    const long long end = cumulative_lengths[request + 1];
    const int slot = slot_mapping[request];
    const long long history_size = kernel_size - 1;
    if (slot < 0) {
        for (long long token = start; token < end; ++token) {
            for (long long channel = 0; channel < channels; ++channel) {
                out[token * channels + channel] = 0.0f;
            }
        }
        return;
    }
    for (long long channel = 0; channel < channels; ++channel) {
        float history[7]{};
        float *state = state_pool_out + (static_cast<long long>(slot) * channels + channel) *
                                            history_size;
        if (load_initial) {
            for (long long item = 0; item < history_size; ++item) history[item] = state[item];
        }
        const float *channel_weight = weight + channel * kernel_size;
        for (long long token = start; token < end; ++token) {
            const float current = x[token * channels + channel];
            float value_out = current * channel_weight[kernel_size - 1];
            for (long long item = 0; item < history_size; ++item) {
                value_out += history[item] * channel_weight[item];
            }
            if (apply_silu) value_out *= sigmoid_device(value_out);
            out[token * channels + channel] = value_out;
            for (long long item = 0; item + 1 < history_size; ++item) {
                history[item] = history[item + 1];
            }
            history[history_size - 1] = current;
        }
        for (long long item = 0; item < history_size; ++item) state[item] = history[item];
    }
}

__global__ void gdn_short_conv_channel_kernel(
    const float *x, const float *weight, float *state_pool_out,
    const int *cumulative_lengths, const int *slot_mapping, float *out,
    long long requests, long long channels, long long kernel_size,
    bool load_initial, bool apply_silu) {
    const long long task = blockIdx.x * blockDim.x + threadIdx.x;
    if (task >= requests * channels) return;
    const long long request = task / channels;
    const long long channel = task % channels;
    const long long start = cumulative_lengths[request];
    const long long end = cumulative_lengths[request + 1];
    const int slot = slot_mapping[request];
    if (slot < 0) {
        for (long long token = start; token < end; ++token) {
            out[token * channels + channel] = 0.0f;
        }
        return;
    }
    const long long history_size = kernel_size - 1;
    float history[7]{};
    float *state = state_pool_out +
                   (static_cast<long long>(slot) * channels + channel) * history_size;
    if (load_initial) {
        for (long long item = 0; item < history_size; ++item) history[item] = state[item];
    }
    const float *channel_weight = weight + channel * kernel_size;
    for (long long token = start; token < end; ++token) {
        const float current = x[token * channels + channel];
        float value_out = current * channel_weight[kernel_size - 1];
        for (long long item = 0; item < history_size; ++item) {
            value_out += history[item] * channel_weight[item];
        }
        if (apply_silu) value_out *= sigmoid_device(value_out);
        out[token * channels + channel] = value_out;
        for (long long item = 0; item + 1 < history_size; ++item) {
            history[item] = history[item + 1];
        }
        history[history_size - 1] = current;
    }
    for (long long item = 0; item < history_size; ++item) state[item] = history[item];
}

__global__ void gdn_qkv_prepare_scalar_kernel(
    const float *mixed, float *q, float *k, float *v, long long tokens,
    long long key_heads, long long value_heads, long long key_dim,
    long long value_dim, float eps, float q_scale, float k_scale) {
    const long long token = blockIdx.x;
    if (threadIdx.x != 0 || token >= tokens) return;
    const long long qk_width = key_heads * key_dim;
    const long long value_width = value_heads * value_dim;
    const long long channels = 2 * qk_width + value_width;
    if (isnan(q_scale)) q_scale = 1.0f / static_cast<float>(key_dim);
    if (isnan(k_scale)) k_scale = rsqrtf(static_cast<float>(key_dim));
    const float *source = mixed + token * channels;
    for (long long head = 0; head < key_heads; ++head) {
        for (int is_key = 0; is_key < 2; ++is_key) {
            const float *input = source + is_key * qk_width + head * key_dim;
            float *output = (is_key ? k : q) + (token * key_heads + head) * key_dim;
            float sum_squares = 0.0f;
            for (long long dim = 0; dim < key_dim; ++dim) {
                sum_squares += input[dim] * input[dim];
            }
            const float scale =
                (is_key ? k_scale : q_scale) /
                sqrtf(sum_squares / static_cast<float>(key_dim) + eps);
            for (long long dim = 0; dim < key_dim; ++dim) output[dim] = input[dim] * scale;
        }
    }
    for (long long item = 0; item < value_width; ++item) {
        v[token * value_width + item] = source[2 * qk_width + item];
    }
}

__global__ void gdn_qkv_prepare_qk_kernel(
    const float *mixed, float *q, float *k, long long tokens,
    long long key_heads, long long value_heads, long long key_dim,
    long long value_dim, float eps, float q_scale, float k_scale) {
    const long long token = blockIdx.x;
    const long long head = blockIdx.y;
    const int is_key = blockIdx.z;
    const int lane = threadIdx.x;
    if (token >= tokens || head >= key_heads) return;
    const long long qk_width = key_heads * key_dim;
    const long long value_width = value_heads * value_dim;
    const long long channels = 2 * qk_width + value_width;
    if (isnan(q_scale)) q_scale = 1.0f / static_cast<float>(key_dim);
    if (isnan(k_scale)) k_scale = rsqrtf(static_cast<float>(key_dim));
    const float *input = mixed + token * channels + is_key * qk_width + head * key_dim;
    float *output = (is_key ? k : q) + (token * key_heads + head) * key_dim;
    float sum_squares = 0.0f;
    for (long long dim = lane; dim < key_dim; dim += blockDim.x) {
        sum_squares += input[dim] * input[dim];
    }
    __shared__ float scratch[4];
    const float total = qc::block_reduce_sum(sum_squares, scratch);
    const float scale =
        (is_key ? k_scale : q_scale) /
        sqrtf(total / static_cast<float>(key_dim) + eps);
    for (long long dim = lane; dim < key_dim; dim += blockDim.x) {
        output[dim] = input[dim] * scale;
    }
}

__global__ void gdn_qkv_prepare_v_kernel(const float *mixed, float *v,
                                         long long tokens, long long key_heads,
                                         long long value_heads,
                                         long long key_dim,
                                         long long value_dim) {
    const long long index = blockIdx.x * blockDim.x + threadIdx.x;
    const long long value_width = value_heads * value_dim;
    const long long total = tokens * value_width;
    if (index >= total) return;
    const long long token = index / value_width;
    const long long item = index - token * value_width;
    const long long qk_width = key_heads * key_dim;
    const long long channels = 2 * qk_width + value_width;
    v[index] = mixed[token * channels + 2 * qk_width + item];
}

__global__ void gdn_gate_beta_scalar_kernel(
    const float *a, const float *b, const float *a_log, const float *dt_bias,
    float *decay, float *beta, long long tokens, long long value_heads) {
    const long long token = blockIdx.x;
    if (threadIdx.x != 0 || token >= tokens) return;
    for (long long head = 0; head < value_heads; ++head) {
        const long long index = token * value_heads + head;
        const float alpha = a[index] + dt_bias[head];
        decay[index] = expf(-expf(a_log[head]) * softplus_device(alpha));
        beta[index] = sigmoid_device(b[index]);
    }
}

__global__ void gdn_gate_beta_kernel(const float *a, const float *b,
                                     const float *a_log,
                                     const float *dt_bias, float *decay,
                                     float *beta, long long count,
                                     long long value_heads) {
    const long long index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const long long head = index % value_heads;
    const float alpha = a[index] + dt_bias[head];
    decay[index] = expf(-expf(a_log[head]) * softplus_device(alpha));
    beta[index] = sigmoid_device(b[index]);
}

__global__ void gdn_gated_rmsnorm_scalar_kernel(
    const float *y, const float *z, const float *weight, float *out,
    long long rows, long long value_dim, float eps) {
    const long long row = blockIdx.x;
    if (threadIdx.x != 0 || row >= rows) return;
    const long long offset = row * value_dim;
    float sum_squares = 0.0f;
    for (long long dim = 0; dim < value_dim; ++dim) {
        sum_squares += y[offset + dim] * y[offset + dim];
    }
    const float inverse =
        1.0f / sqrtf(sum_squares / static_cast<float>(value_dim) + eps);
    for (long long dim = 0; dim < value_dim; ++dim) {
        const float gate = z[offset + dim];
        out[offset + dim] =
            y[offset + dim] * inverse * weight[dim] * gate * sigmoid_device(gate);
    }
}

__global__ void gdn_gated_rmsnorm_kernel(const float *y, const float *z,
                                         const float *weight, float *out,
                                         long long rows, long long value_dim,
                                         float eps) {
    const long long row = blockIdx.x;
    const int lane = threadIdx.x;
    if (row >= rows) return;
    const long long offset = row * value_dim;
    float sum_squares = 0.0f;
    for (long long dim = lane; dim < value_dim; dim += blockDim.x) {
        sum_squares += y[offset + dim] * y[offset + dim];
    }
    __shared__ float scratch[4];
    const float total = qc::block_reduce_sum(sum_squares, scratch);
    const float inverse =
        1.0f / sqrtf(total / static_cast<float>(value_dim) + eps);
    for (long long dim = lane; dim < value_dim; dim += blockDim.x) {
        const float gate = z[offset + dim];
        out[offset + dim] =
            y[offset + dim] * inverse * weight[dim] * gate * sigmoid_device(gate);
    }
}

// ---------------------------------------------------------------------------
// Host references
// ---------------------------------------------------------------------------

void gated_linear_attention_ref(const std::vector<float> &key,
                                const std::vector<float> &value,
                                const std::vector<float> &query,
                                const std::vector<float> &gate,
                                const std::vector<float> &initial_state,
                                std::vector<float> &output,
                                std::vector<float> &final_state,
                                long long sequences, long long tokens,
                                long long heads, long long dim, float scale) {
    output.assign(static_cast<size_t>(sequences * tokens * heads * dim), 0.0f);
    final_state = initial_state;
    for (long long sequence = 0; sequence < sequences; ++sequence) {
        for (long long head = 0; head < heads; ++head) {
            float *state = final_state.data() +
                           ((sequence * heads + head) * dim * dim);
            for (long long token = 0; token < tokens; ++token) {
                const size_t offset = recurrent_offset(sequence, token, head, dim, tokens, heads);
                for (long long i = 0; i < dim; ++i) {
                    const float key_value = key[offset + i];
                    const float query_value = query[offset + i] * scale;
                    const float gate_value = gate[offset + i];
                    float *state_row = state + i * dim;
                    for (long long j = 0; j < dim; ++j) {
                        state_row[j] = state_row[j] * gate_value + key_value * value[offset + j];
                        output[offset + j] += state_row[j] * query_value;
                    }
                }
            }
        }
    }
}

void rwkv_wkv6_ref(const std::vector<float> &key,
                   const std::vector<float> &value,
                   const std::vector<float> &receptance,
                   const std::vector<float> &time_first,
                   const std::vector<float> &time_decay,
                   const std::vector<float> &initial_state,
                   std::vector<float> &output, std::vector<float> &final_state,
                   long long sequences, long long tokens, long long heads,
                   long long dim) {
    output.assign(static_cast<size_t>(sequences * tokens * heads * dim), 0.0f);
    final_state = initial_state;
    for (long long sequence = 0; sequence < sequences; ++sequence) {
        for (long long head = 0; head < heads; ++head) {
            float *state = final_state.data() +
                           ((sequence * heads + head) * dim * dim);
            for (long long token = 0; token < tokens; ++token) {
                const size_t offset = recurrent_offset(sequence, token, head, dim, tokens, heads);
                for (long long i = 0; i < dim; ++i) {
                    const float k = key[offset + i];
                    const float r = receptance[offset + i];
                    const float first = time_first[head * dim + i];
                    const float decay = time_decay[offset + i];
                    float *state_row = state + i * dim;
                    for (long long j = 0; j < dim; ++j) {
                        const float kv = k * value[offset + j];
                        const float previous = state_row[j];
                        output[offset + j] += (previous + kv * first) * r;
                        state_row[j] = previous * decay + kv;
                    }
                }
            }
        }
    }
}

void rwkv_wkv7_ref(const std::vector<float> &receptance,
                   const std::vector<float> &decay,
                   const std::vector<float> &key,
                   const std::vector<float> &value,
                   const std::vector<float> &a, const std::vector<float> &b,
                   const std::vector<float> &initial_state,
                   std::vector<float> &output, std::vector<float> &final_state,
                   long long sequences, long long tokens, long long heads,
                   long long dim) {
    output.assign(static_cast<size_t>(sequences * tokens * heads * dim), 0.0f);
    final_state = initial_state;
    for (long long sequence = 0; sequence < sequences; ++sequence) {
        for (long long head = 0; head < heads; ++head) {
            float *state = final_state.data() +
                           ((sequence * heads + head) * dim * dim);
            for (long long token = 0; token < tokens; ++token) {
                const size_t offset = recurrent_offset(sequence, token, head, dim, tokens, heads);
                for (long long i = 0; i < dim; ++i) {
                    float *state_row = state + i * dim;
                    float state_a = 0.0f;
                    for (long long j = 0; j < dim; ++j) state_a += a[offset + j] * state_row[j];
                    float result = 0.0f;
                    for (long long j = 0; j < dim; ++j) {
                        state_row[j] = state_row[j] * decay[offset + j] +
                                       value[offset + i] * key[offset + j] +
                                       state_a * b[offset + j];
                        result += state_row[j] * receptance[offset + j];
                    }
                    output[offset + i] = result;
                }
            }
        }
    }
}

std::vector<float> linear_attention_unnormalized_ref(
    const std::vector<float> &q, const std::vector<float> &k,
    const std::vector<float> &v, long long batch, long long heads,
    long long sequence, long long dim) {
    std::vector<float> out(static_cast<size_t>(batch * heads * sequence * dim), 0.0f);
    std::vector<double> state(static_cast<size_t>(dim * dim));
    for (long long bh = 0; bh < batch * heads; ++bh) {
        std::fill(state.begin(), state.end(), 0.0);
        const size_t base = static_cast<size_t>(bh * sequence * dim);
        for (long long token = 0; token < sequence; ++token) {
            for (long long i = 0; i < dim; ++i) {
                for (long long j = 0; j < dim; ++j) {
                    state[static_cast<size_t>(i * dim + j)] +=
                        static_cast<double>(k[base + token * dim + i]) *
                        v[base + token * dim + j];
                }
            }
        }
        for (long long token = 0; token < sequence; ++token) {
            for (long long j = 0; j < dim; ++j) {
                double acc = 0.0;
                for (long long i = 0; i < dim; ++i) {
                    acc += static_cast<double>(q[base + token * dim + i]) *
                           state[static_cast<size_t>(i * dim + j)];
                }
                out[base + token * dim + j] = static_cast<float>(acc);
            }
        }
    }
    return out;
}

void gdn_recurrence_ref(const std::vector<float> &q, const std::vector<float> &k,
                        const std::vector<float> &v,
                        const std::vector<float> &decay,
                        const std::vector<float> &beta,
                        const std::vector<float> &state_pool,
                        const std::vector<int32_t> &cu,
                        const std::vector<int32_t> &slot_mapping,
                        std::vector<float> &out,
                        std::vector<float> &state_pool_out,
                        long long requests, long long key_heads,
                        long long value_heads, long long key_dim,
                        long long value_dim, bool load_initial) {
    out.assign(v.size(), 0.0f);
    state_pool_out = state_pool;
    const long long head_group = value_heads / key_heads;
    for (long long request = 0; request < requests; ++request) {
        const long long start = cu[request];
        const long long end = cu[request + 1];
        const long long slot = slot_mapping[request];
        for (long long value_head = 0; value_head < value_heads; ++value_head) {
            const long long key_head = value_head / head_group;
            for (long long value_index = 0; value_index < value_dim; ++value_index) {
                float *state =
                    state_pool_out.data() +
                    ((slot * value_heads + value_head) * value_dim + value_index) * key_dim;
                if (!load_initial) std::fill_n(state, key_dim, 0.0f);
                for (long long token = start; token < end; ++token) {
                    const float *key_row = k.data() + (token * key_heads + key_head) * key_dim;
                    const float *query = q.data() + (token * key_heads + key_head) * key_dim;
                    const float gate = decay[token * value_heads + value_head];
                    double memory = 0.0;
                    for (long long dim = 0; dim < key_dim; ++dim) {
                        state[dim] *= gate;
                        memory += static_cast<double>(state[dim]) * key_row[dim];
                    }
                    const double correction =
                        (v[(token * value_heads + value_head) * value_dim + value_index] -
                         memory) *
                        beta[token * value_heads + value_head];
                    double result = 0.0;
                    for (long long dim = 0; dim < key_dim; ++dim) {
                        state[dim] += static_cast<float>(key_row[dim] * correction);
                        result += static_cast<double>(state[dim]) * query[dim];
                    }
                    out[(token * value_heads + value_head) * value_dim + value_index] =
                        static_cast<float>(result);
                }
            }
        }
    }
}

void gdn_short_conv_ref(const std::vector<float> &x,
                        const std::vector<float> &weight,
                        const std::vector<float> &state_pool,
                        const std::vector<int32_t> &cu,
                        const std::vector<int32_t> &slot_mapping,
                        std::vector<float> &out,
                        std::vector<float> &state_pool_out,
                        long long requests, long long channels,
                        long long kernel_size, bool load_initial,
                        bool apply_silu) {
    out.assign(x.size(), 0.0f);
    state_pool_out = state_pool;
    const long long history_size = kernel_size - 1;
    for (long long request = 0; request < requests; ++request) {
        const long long start = cu[request];
        const long long end = cu[request + 1];
        const int slot = slot_mapping[request];
        if (slot < 0) {
            for (long long token = start; token < end; ++token) {
                std::fill_n(out.data() + token * channels, channels, 0.0f);
            }
            continue;
        }
        for (long long channel = 0; channel < channels; ++channel) {
            float history[7]{};
            float *state =
                state_pool_out.data() +
                (static_cast<long long>(slot) * channels + channel) * history_size;
            if (load_initial) std::copy_n(state, history_size, history);
            const float *channel_weight = weight.data() + channel * kernel_size;
            for (long long token = start; token < end; ++token) {
                const float current = x[token * channels + channel];
                float value = current * channel_weight[kernel_size - 1];
                for (long long item = 0; item < history_size; ++item) {
                    value += history[item] * channel_weight[item];
                }
                if (apply_silu) value *= sigmoid_host(value);
                out[token * channels + channel] = value;
                for (long long item = 0; item + 1 < history_size; ++item) {
                    history[item] = history[item + 1];
                }
                history[history_size - 1] = current;
            }
            std::copy_n(history, history_size, state);
        }
    }
}

void gdn_qkv_prepare_ref(const std::vector<float> &mixed, std::vector<float> &q,
                         std::vector<float> &k, std::vector<float> &v,
                         long long tokens, long long key_heads,
                         long long value_heads, long long key_dim,
                         long long value_dim, float eps, float q_scale,
                         float k_scale) {
    const long long qk_width = key_heads * key_dim;
    const long long value_width = value_heads * value_dim;
    const long long channels = 2 * qk_width + value_width;
    q.assign(static_cast<size_t>(tokens * qk_width), 0.0f);
    k.assign(static_cast<size_t>(tokens * qk_width), 0.0f);
    v.assign(static_cast<size_t>(tokens * value_width), 0.0f);
    if (std::isnan(q_scale)) q_scale = 1.0f / static_cast<float>(key_dim);
    if (std::isnan(k_scale)) k_scale = 1.0f / std::sqrt(static_cast<float>(key_dim));
    for (long long token = 0; token < tokens; ++token) {
        const float *source = mixed.data() + token * channels;
        for (long long head = 0; head < key_heads; ++head) {
            for (int is_key = 0; is_key < 2; ++is_key) {
                const float *input = source + is_key * qk_width + head * key_dim;
                float *output = (is_key ? k : q).data() + (token * key_heads + head) * key_dim;
                float sum_squares = 0.0f;
                for (long long dim = 0; dim < key_dim; ++dim) {
                    sum_squares += input[dim] * input[dim];
                }
                const float scale =
                    (is_key ? k_scale : q_scale) /
                    std::sqrt(sum_squares / static_cast<float>(key_dim) + eps);
                for (long long dim = 0; dim < key_dim; ++dim) output[dim] = input[dim] * scale;
            }
        }
        std::copy_n(source + 2 * qk_width, value_width, v.data() + token * value_width);
    }
}

void gdn_gate_beta_ref(const std::vector<float> &a, const std::vector<float> &b,
                       const std::vector<float> &a_log,
                       const std::vector<float> &dt_bias,
                       std::vector<float> &decay, std::vector<float> &beta,
                       long long tokens, long long value_heads) {
    decay.assign(static_cast<size_t>(tokens * value_heads), 0.0f);
    beta.assign(static_cast<size_t>(tokens * value_heads), 0.0f);
    for (long long token = 0; token < tokens; ++token) {
        for (long long head = 0; head < value_heads; ++head) {
            const long long index = token * value_heads + head;
            const float alpha = a[index] + dt_bias[head];
            decay[index] = std::exp(-std::exp(a_log[head]) * softplus_host(alpha));
            beta[index] = sigmoid_host(b[index]);
        }
    }
}

std::vector<float> gdn_gated_rmsnorm_ref(const std::vector<float> &y,
                                         const std::vector<float> &z,
                                         const std::vector<float> &weight,
                                         long long tokens,
                                         long long value_heads,
                                         long long value_dim, float eps) {
    const long long rows = tokens * value_heads;
    std::vector<float> out(y.size(), 0.0f);
    for (long long row = 0; row < rows; ++row) {
        const long long offset = row * value_dim;
        float sum_squares = 0.0f;
        for (long long dim = 0; dim < value_dim; ++dim) {
            sum_squares += y[offset + dim] * y[offset + dim];
        }
        const float inverse =
            1.0f / std::sqrt(sum_squares / static_cast<float>(value_dim) + eps);
        for (long long dim = 0; dim < value_dim; ++dim) {
            const float gate = z[offset + dim];
            out[offset + dim] =
                y[offset + dim] * inverse * weight[dim] * gate * sigmoid_host(gate);
        }
    }
    return out;
}

std::vector<int32_t> make_cu(const std::vector<int32_t> &lengths) {
    std::vector<int32_t> cu(lengths.size() + 1, 0);
    for (size_t i = 0; i < lengths.size(); ++i) cu[i + 1] = cu[i] + lengths[i];
    return cu;
}

template <typename Fn>
qc::Bench bench_per_launch(Fn &&fn, int warmups = 10, int iters = 50,
                           int repeats = 1) {
    qc::Bench b = qc::bench([&] {
        for (int repeat = 0; repeat < repeats; ++repeat) fn();
        QC_CHECK(hipGetLastError());
    }, warmups, iters);
    if (repeats > 1) {
        b.median_ms /= static_cast<double>(repeats);
        b.min_ms /= static_cast<double>(repeats);
        b.max_ms /= static_cast<double>(repeats);
        b.mean_ms /= static_cast<double>(repeats);
    }
    return b;
}

void report_pair(const char *label, const qc::Bench &baseline,
                 const qc::Bench &candidate, double metric,
                 bool compute = true) {
    if (compute) {
        baseline.report_compute(std::string(label) + " scalar", metric);
        candidate.report_compute(std::string(label) + " candidate", metric);
    } else {
        baseline.report_bandwidth(std::string(label) + " scalar", metric);
        candidate.report_bandwidth(std::string(label) + " candidate", metric);
    }
    qc::report_ab(label, baseline, candidate);
}

bool run_correctness() {
    bool ok = true;
    int checks = 0;
    qc::Rng rng(0x8008);

    {
        constexpr long long sequences = 2;
        constexpr long long tokens = 5;
        constexpr long long heads = 2;
        constexpr long long dim = 17;
        constexpr float scale = 0.25f;
        const size_t vector_count = static_cast<size_t>(sequences * tokens * heads * dim);
        const size_t state_count = static_cast<size_t>(sequences * heads * dim * dim);
        auto key = rng.uniforms(vector_count, -0.25f, 0.25f);
        auto value = rng.uniforms(vector_count, -0.25f, 0.25f);
        auto query = rng.uniforms(vector_count, -0.25f, 0.25f);
        auto gate = rng.uniforms(vector_count, 0.65f, 0.95f);
        auto initial = rng.uniforms(state_count, -0.02f, 0.02f);
        std::vector<float> ref_out, ref_state;
        gated_linear_attention_ref(key, value, query, gate, initial, ref_out,
                                   ref_state, sequences, tokens, heads, dim,
                                   scale);
        float *dk = qc::dnew(key);
        float *dv = qc::dnew(value);
        float *dq = qc::dnew(query);
        float *dg = qc::dnew(gate);
        float *di = qc::dnew(initial);
        float *do_ = qc::dzero<float>(vector_count);
        float *ds = qc::dzero<float>(state_count);
        gated_linear_attention_wave_kernel<<<sequences * heads * dim, qc::kWave>>>(
            dk, dv, dq, dg, di, do_, ds, sequences, tokens, heads, dim, scale);
        QC_SYNC();
        ok &= qc::compare(qc::d2h(do_, vector_count), to_ref(ref_out),
                          qc::Tol::fp32().with_elementwise(2e-4, 2e-5))
                  .report("gated_linear_attention output");
        ++checks;
        ok &= qc::compare(qc::d2h(ds, state_count), to_ref(ref_state),
                          qc::Tol::fp32().with_elementwise(2e-4, 2e-5))
                  .report("gated_linear_attention final_state");
        ++checks;
        qc::dfree(dk, dv, dq, dg, di, do_, ds);
    }

    {
        constexpr long long sequences = 2;
        constexpr long long tokens = 6;
        constexpr long long heads = 2;
        constexpr long long dim = 19;
        const size_t vector_count = static_cast<size_t>(sequences * tokens * heads * dim);
        const size_t state_count = static_cast<size_t>(sequences * heads * dim * dim);
        auto key = rng.uniforms(vector_count, -0.18f, 0.18f);
        auto value = rng.uniforms(vector_count, -0.18f, 0.18f);
        auto receptance = rng.uniforms(vector_count, -0.4f, 0.4f);
        auto time_first = rng.uniforms(static_cast<size_t>(heads * dim), 0.05f, 0.35f);
        auto time_decay = rng.uniforms(vector_count, 0.70f, 0.97f);
        auto initial = rng.uniforms(state_count, -0.02f, 0.02f);
        std::vector<float> ref_out, ref_state;
        rwkv_wkv6_ref(key, value, receptance, time_first, time_decay, initial,
                      ref_out, ref_state, sequences, tokens, heads, dim);
        float *dk = qc::dnew(key);
        float *dv = qc::dnew(value);
        float *dr = qc::dnew(receptance);
        float *df = qc::dnew(time_first);
        float *dd = qc::dnew(time_decay);
        float *di = qc::dnew(initial);
        float *do_ = qc::dzero<float>(vector_count);
        float *ds = qc::dzero<float>(state_count);
        rwkv_wkv6_wave_kernel<<<sequences * heads * dim, qc::kWave>>>(
            dk, dv, dr, df, dd, di, do_, ds, sequences, tokens, heads, dim);
        QC_SYNC();
        ok &= qc::compare(qc::d2h(do_, vector_count), to_ref(ref_out),
                          qc::Tol::fp32().with_elementwise(3e-4, 3e-5))
                  .report("rwkv_wkv6 output");
        ++checks;
        ok &= qc::compare(qc::d2h(ds, state_count), to_ref(ref_state),
                          qc::Tol::fp32().with_elementwise(3e-4, 3e-5))
                  .report("rwkv_wkv6 final_state");
        ++checks;
        qc::dfree(dk, dv, dr, df, dd, di, do_, ds);
    }

    {
        constexpr long long sequences = 2;
        constexpr long long tokens = 5;
        constexpr long long heads = 2;
        constexpr long long dim = 23;
        const size_t vector_count = static_cast<size_t>(sequences * tokens * heads * dim);
        const size_t state_count = static_cast<size_t>(sequences * heads * dim * dim);
        auto receptance = rng.uniforms(vector_count, -0.35f, 0.35f);
        auto decay = rng.uniforms(vector_count, 0.65f, 0.95f);
        auto key = rng.uniforms(vector_count, -0.15f, 0.15f);
        auto value = rng.uniforms(vector_count, -0.15f, 0.15f);
        auto a = rng.uniforms(vector_count, -0.10f, 0.10f);
        auto b = rng.uniforms(vector_count, -0.10f, 0.10f);
        auto initial = rng.uniforms(state_count, -0.01f, 0.01f);
        std::vector<float> ref_out, ref_state;
        rwkv_wkv7_ref(receptance, decay, key, value, a, b, initial, ref_out,
                      ref_state, sequences, tokens, heads, dim);
        float *dr = qc::dnew(receptance);
        float *dd = qc::dnew(decay);
        float *dk = qc::dnew(key);
        float *dv = qc::dnew(value);
        float *da = qc::dnew(a);
        float *db = qc::dnew(b);
        float *di = qc::dnew(initial);
        float *do_ = qc::dzero<float>(vector_count);
        float *ds = qc::dzero<float>(state_count);
        rwkv_wkv7_wave_kernel<<<sequences * heads * dim, qc::kWave>>>(
            dr, dd, dk, dv, da, db, di, do_, ds, sequences, tokens, heads, dim);
        QC_SYNC();
        ok &= qc::compare(qc::d2h(do_, vector_count), to_ref(ref_out),
                          qc::Tol::fp32().with_elementwise(3e-4, 3e-5))
                  .report("rwkv_wkv7 output");
        ++checks;
        ok &= qc::compare(qc::d2h(ds, state_count), to_ref(ref_state),
                          qc::Tol::fp32().with_elementwise(3e-4, 3e-5))
                  .report("rwkv_wkv7 final_state");
        ++checks;
        qc::dfree(dr, dd, dk, dv, da, db, di, do_, ds);
    }

    {
        constexpr long long batch = 2;
        constexpr long long heads = 2;
        constexpr long long sequence = 7;
        constexpr long long dim = 19;
        const size_t count = static_cast<size_t>(batch * heads * sequence * dim);
        auto q = rng.uniforms(count, -0.35f, 0.35f);
        auto k = rng.uniforms(count, -0.35f, 0.35f);
        auto v = rng.uniforms(count, -0.35f, 0.35f);
        const auto ref = linear_attention_unnormalized_ref(q, k, v, batch, heads,
                                                           sequence, dim);
        float *dq = qc::dnew(q);
        float *dk = qc::dnew(k);
        float *dv = qc::dnew(v);
        float *do_ = qc::dzero<float>(count);
        double *dkv = qc::dzero<double>(static_cast<size_t>(batch * heads * dim * dim));
        linear_attention_kv_kernel<<<dim3(dim, dim, batch * heads), qc::kWave>>>(
            dk, dv, dkv, batch, heads, sequence, dim);
        linear_attention_out_kernel<<<dim3(sequence, dim, batch * heads), qc::kWave>>>(
            dq, dkv, do_, batch, heads, sequence, dim);
        QC_SYNC();
        ok &= qc::compare(qc::d2h(do_, count), to_ref(ref),
                          qc::Tol::fp32().with_elementwise(2e-5, 2e-6))
                  .report("linear_attention_unnormalized");
        ++checks;
        qc::dfree(dq, dk, dv, do_, dkv);
    }

    {
        constexpr long long requests = 3;
        constexpr long long slots = 3;
        constexpr long long key_heads = 2;
        constexpr long long value_heads = 4;
        constexpr long long key_dim = 64;
        constexpr long long value_dim = 17;
        const std::vector<int32_t> cu = make_cu({4, 3, 5});
        const long long total_tokens = cu.back();
        const std::vector<int32_t> slot_mapping = {0, 2, 1};
        auto q = rng.uniforms(static_cast<size_t>(total_tokens * key_heads * key_dim), -0.12f, 0.12f);
        auto k = rng.uniforms(q.size(), -0.12f, 0.12f);
        auto v = rng.uniforms(static_cast<size_t>(total_tokens * value_heads * value_dim), -0.20f, 0.20f);
        auto decay = rng.uniforms(static_cast<size_t>(total_tokens * value_heads), 0.75f, 0.98f);
        auto beta = rng.uniforms(decay.size(), 0.10f, 0.80f);
        auto state_in = rng.uniforms(static_cast<size_t>(slots * value_heads * value_dim * key_dim), -0.01f, 0.01f);
        std::vector<float> ref_out, ref_state;
        gdn_recurrence_ref(q, k, v, decay, beta, state_in, cu, slot_mapping,
                           ref_out, ref_state, requests, key_heads, value_heads,
                           key_dim, value_dim, true);
        float *dq = qc::dnew(q);
        float *dk = qc::dnew(k);
        float *dv = qc::dnew(v);
        float *dd = qc::dnew(decay);
        float *db = qc::dnew(beta);
        float *dsi = qc::dnew(state_in);
        float *dso = qc::dzero<float>(state_in.size());
        float *do_ = qc::dzero<float>(v.size());
        int32_t *dcu = qc::dnew(cu);
        int32_t *dsl = qc::dnew(slot_mapping);
        copy_float_kernel<<<qc::grid_for(state_in.size(), 256), 256>>>(dsi, dso, state_in.size());
        gdn_recurrence_wave_kernel<<<dim3(value_dim, value_heads, requests), qc::kWave>>>(
            dq, dk, dv, dd, db, dso, dcu, dsl, do_, requests, key_heads,
            value_heads, key_dim, value_dim, true);
        QC_SYNC();
        ok &= qc::compare(qc::d2h(do_, v.size()), to_ref(ref_out),
                          qc::Tol::fp32().with_elementwise(4e-4, 4e-5))
                  .report("gdn_recurrence output");
        ++checks;
        ok &= qc::compare(qc::d2h(dso, state_in.size()), to_ref(ref_state),
                          qc::Tol::fp32().with_elementwise(4e-4, 4e-5))
                  .report("gdn_recurrence state");
        ++checks;
        qc::dfree(dq, dk, dv, dd, db, dsi, dso, do_, dcu, dsl);
    }

    {
        constexpr long long requests = 3;
        constexpr long long slots = 2;
        constexpr long long channels = 13;
        constexpr long long kernel_size = 4;
        const std::vector<int32_t> cu = make_cu({3, 4, 2});
        const long long total_tokens = cu.back();
        const std::vector<int32_t> slot_mapping = {0, -1, 1};
        auto x = rng.uniforms(static_cast<size_t>(total_tokens * channels), -0.35f, 0.35f);
        auto weight = rng.uniforms(static_cast<size_t>(channels * kernel_size), -0.20f, 0.20f);
        auto state_in = rng.uniforms(static_cast<size_t>(slots * channels * (kernel_size - 1)), -0.10f, 0.10f);
        std::vector<float> ref_out, ref_state;
        gdn_short_conv_ref(x, weight, state_in, cu, slot_mapping, ref_out,
                           ref_state, requests, channels, kernel_size, true,
                           true);
        float *dx = qc::dnew(x);
        float *dw = qc::dnew(weight);
        float *dsi = qc::dnew(state_in);
        float *dso = qc::dzero<float>(state_in.size());
        float *do_ = qc::dzero<float>(x.size());
        int32_t *dcu = qc::dnew(cu);
        int32_t *dsl = qc::dnew(slot_mapping);
        copy_float_kernel<<<qc::grid_for(state_in.size(), 256), 256>>>(dsi, dso, state_in.size());
        gdn_short_conv_channel_kernel<<<qc::grid_for(static_cast<size_t>(requests * channels), 256), 256>>>(
            dx, dw, dso, dcu, dsl, do_, requests, channels, kernel_size, true,
            true);
        QC_SYNC();
        ok &= qc::compare(qc::d2h(do_, x.size()), to_ref(ref_out), qc::Tol::fp32())
                  .report("gdn_short_conv output");
        ++checks;
        ok &= qc::compare(qc::d2h(dso, state_in.size()), to_ref(ref_state),
                          qc::Tol::fp32())
                  .report("gdn_short_conv state");
        ++checks;
        qc::dfree(dx, dw, dsi, dso, do_, dcu, dsl);
    }

    {
        constexpr long long tokens = 5;
        constexpr long long key_heads = 2;
        constexpr long long value_heads = 4;
        constexpr long long key_dim = 64;
        constexpr long long value_dim = 64;
        constexpr float eps = 1e-5f;
        const long long qk_width = key_heads * key_dim;
        const long long value_width = value_heads * value_dim;
        const long long channels = 2 * qk_width + value_width;
        auto mixed = rng.uniforms(static_cast<size_t>(tokens * channels), -0.4f, 0.4f);
        std::vector<float> ref_q, ref_k, ref_v;
        const float nan_scale = std::numeric_limits<float>::quiet_NaN();
        gdn_qkv_prepare_ref(mixed, ref_q, ref_k, ref_v, tokens, key_heads,
                            value_heads, key_dim, value_dim, eps, nan_scale,
                            nan_scale);
        float *dm = qc::dnew(mixed);
        float *dq = qc::dzero<float>(ref_q.size());
        float *dk = qc::dzero<float>(ref_k.size());
        float *dv = qc::dzero<float>(ref_v.size());
        gdn_qkv_prepare_qk_kernel<<<dim3(tokens, key_heads, 2), 128>>>(
            dm, dq, dk, tokens, key_heads, value_heads, key_dim, value_dim, eps,
            nan_scale, nan_scale);
        gdn_qkv_prepare_v_kernel<<<qc::grid_for(ref_v.size(), 256), 256>>>(
            dm, dv, tokens, key_heads, value_heads, key_dim, value_dim);
        QC_SYNC();
        ok &= qc::compare(qc::d2h(dq, ref_q.size()), to_ref(ref_q), qc::Tol::fp32())
                  .report("gdn_qkv_prepare q");
        ++checks;
        ok &= qc::compare(qc::d2h(dk, ref_k.size()), to_ref(ref_k), qc::Tol::fp32())
                  .report("gdn_qkv_prepare k");
        ++checks;
        ok &= qc::compare(qc::d2h(dv, ref_v.size()), to_ref(ref_v), qc::Tol::fp32())
                  .report("gdn_qkv_prepare v");
        ++checks;
        qc::dfree(dm, dq, dk, dv);
    }

    {
        constexpr long long tokens = 257;
        constexpr long long value_heads = 5;
        auto a = rng.uniforms(static_cast<size_t>(tokens * value_heads), -1.0f, 1.0f);
        auto b = rng.uniforms(a.size(), -1.0f, 1.0f);
        auto a_log = rng.uniforms(static_cast<size_t>(value_heads), -1.5f, 0.5f);
        auto dt_bias = rng.uniforms(static_cast<size_t>(value_heads), -0.4f, 0.4f);
        std::vector<float> ref_decay, ref_beta;
        gdn_gate_beta_ref(a, b, a_log, dt_bias, ref_decay, ref_beta, tokens,
                          value_heads);
        float *da = qc::dnew(a);
        float *db = qc::dnew(b);
        float *dal = qc::dnew(a_log);
        float *ddt = qc::dnew(dt_bias);
        float *dd = qc::dzero<float>(a.size());
        float *dbe = qc::dzero<float>(a.size());
        gdn_gate_beta_kernel<<<qc::grid_for(a.size(), 256), 256>>>(
            da, db, dal, ddt, dd, dbe, a.size(), value_heads);
        QC_SYNC();
        ok &= qc::compare(qc::d2h(dd, a.size()), to_ref(ref_decay),
                          qc::Tol::fp32().with_elementwise(2e-5, 2e-6))
                  .report("gdn_gate_beta decay");
        ++checks;
        ok &= qc::compare(qc::d2h(dbe, a.size()), to_ref(ref_beta),
                          qc::Tol::fp32().with_elementwise(2e-5, 2e-6))
                  .report("gdn_gate_beta beta");
        ++checks;
        qc::dfree(da, db, dal, ddt, dd, dbe);
    }

    {
        constexpr long long tokens = 5;
        constexpr long long value_heads = 3;
        constexpr long long value_dim = 64;
        constexpr float eps = 1e-5f;
        auto y = rng.uniforms(static_cast<size_t>(tokens * value_heads * value_dim), -0.5f, 0.5f);
        auto z = rng.uniforms(y.size(), -0.8f, 0.8f);
        auto weight = rng.uniforms(static_cast<size_t>(value_dim), 0.5f, 1.5f);
        const auto ref = gdn_gated_rmsnorm_ref(y, z, weight, tokens, value_heads,
                                               value_dim, eps);
        float *dy = qc::dnew(y);
        float *dz = qc::dnew(z);
        float *dw = qc::dnew(weight);
        float *do_ = qc::dzero<float>(y.size());
        gdn_gated_rmsnorm_kernel<<<tokens * value_heads, 128>>>(
            dy, dz, dw, do_, tokens * value_heads, value_dim, eps);
        QC_SYNC();
        ok &= qc::compare(qc::d2h(do_, y.size()), to_ref(ref),
                          qc::Tol::fp32().with_elementwise(2e-5, 2e-6))
                  .report("gdn_gated_rmsnorm");
        ++checks;
        qc::dfree(dy, dz, dw, do_);
    }

    std::printf("Phase 8 correctness checks: %d\n", checks);
    return ok;
}

void run_benchmarks() {
    std::printf("\n== Phase 8 benchmarks ==\n");
    std::printf("   Timing note: medians are per launch; fast kernels use inner repeats.\n");
    qc::Rng rng(0x8080);

    {
        constexpr long long sequences = 8;
        constexpr long long tokens = 16;
        constexpr long long heads = 4;
        constexpr long long dim = 64;
        constexpr float scale = 0.125f;
        const size_t vector_count = static_cast<size_t>(sequences * tokens * heads * dim);
        const size_t state_count = static_cast<size_t>(sequences * heads * dim * dim);
        auto key = rng.uniforms(vector_count, -0.20f, 0.20f);
        auto value = rng.uniforms(vector_count, -0.20f, 0.20f);
        auto query = rng.uniforms(vector_count, -0.20f, 0.20f);
        auto gate = rng.uniforms(vector_count, 0.70f, 0.97f);
        auto initial = rng.uniforms(state_count, -0.01f, 0.01f);
        float *dk = qc::dnew(key);
        float *dv = qc::dnew(value);
        float *dq = qc::dnew(query);
        float *dg = qc::dnew(gate);
        float *di = qc::dnew(initial);
        float *do_ = qc::dzero<float>(vector_count);
        float *ds = qc::dzero<float>(state_count);
        const double flops = double(sequences * heads * tokens * dim * dim) * 5.0;
        const auto scalar = bench_per_launch([&] {
            gated_linear_attention_scalar_kernel<<<sequences * heads, 1>>>(
                dk, dv, dq, dg, di, do_, ds, sequences, tokens, heads, dim, scale);
        }, 5, 20);
        const auto wave = bench_per_launch([&] {
            gated_linear_attention_wave_kernel<<<sequences * heads * dim, qc::kWave>>>(
                dk, dv, dq, dg, di, do_, ds, sequences, tokens, heads, dim, scale);
        }, 10, 40, 128);
        report_pair("gated_linear_attention", scalar, wave, flops);
        qc::dfree(dk, dv, dq, dg, di, do_, ds);
    }

    {
        constexpr long long sequences = 8;
        constexpr long long tokens = 16;
        constexpr long long heads = 4;
        constexpr long long dim = 64;
        const size_t vector_count = static_cast<size_t>(sequences * tokens * heads * dim);
        const size_t state_count = static_cast<size_t>(sequences * heads * dim * dim);
        auto key = rng.uniforms(vector_count, -0.18f, 0.18f);
        auto value = rng.uniforms(vector_count, -0.18f, 0.18f);
        auto receptance = rng.uniforms(vector_count, -0.4f, 0.4f);
        auto time_first = rng.uniforms(static_cast<size_t>(heads * dim), 0.05f, 0.35f);
        auto time_decay = rng.uniforms(vector_count, 0.70f, 0.97f);
        auto initial = rng.uniforms(state_count, -0.01f, 0.01f);
        float *dk = qc::dnew(key);
        float *dv = qc::dnew(value);
        float *dr = qc::dnew(receptance);
        float *df = qc::dnew(time_first);
        float *dd = qc::dnew(time_decay);
        float *di = qc::dnew(initial);
        float *do_ = qc::dzero<float>(vector_count);
        float *ds = qc::dzero<float>(state_count);
        const double flops = double(sequences * heads * tokens * dim * dim) * 6.0;
        const auto scalar = bench_per_launch([&] {
            rwkv_wkv6_scalar_kernel<<<sequences * heads, 1>>>(
                dk, dv, dr, df, dd, di, do_, ds, sequences, tokens, heads, dim);
        }, 5, 20);
        const auto wave = bench_per_launch([&] {
            rwkv_wkv6_wave_kernel<<<sequences * heads * dim, qc::kWave>>>(
                dk, dv, dr, df, dd, di, do_, ds, sequences, tokens, heads, dim);
        }, 10, 40, 128);
        report_pair("rwkv_wkv6", scalar, wave, flops);
        qc::dfree(dk, dv, dr, df, dd, di, do_, ds);
    }

    {
        constexpr long long sequences = 8;
        constexpr long long tokens = 16;
        constexpr long long heads = 4;
        constexpr long long dim = 64;
        const size_t vector_count = static_cast<size_t>(sequences * tokens * heads * dim);
        const size_t state_count = static_cast<size_t>(sequences * heads * dim * dim);
        auto receptance = rng.uniforms(vector_count, -0.35f, 0.35f);
        auto decay = rng.uniforms(vector_count, 0.65f, 0.95f);
        auto key = rng.uniforms(vector_count, -0.15f, 0.15f);
        auto value = rng.uniforms(vector_count, -0.15f, 0.15f);
        auto a = rng.uniforms(vector_count, -0.10f, 0.10f);
        auto b = rng.uniforms(vector_count, -0.10f, 0.10f);
        auto initial = rng.uniforms(state_count, -0.01f, 0.01f);
        float *dr = qc::dnew(receptance);
        float *dd = qc::dnew(decay);
        float *dk = qc::dnew(key);
        float *dv = qc::dnew(value);
        float *da = qc::dnew(a);
        float *db = qc::dnew(b);
        float *di = qc::dnew(initial);
        float *do_ = qc::dzero<float>(vector_count);
        float *ds = qc::dzero<float>(state_count);
        const double flops = double(sequences * heads * tokens * dim * dim) * 8.0;
        const auto scalar = bench_per_launch([&] {
            rwkv_wkv7_scalar_kernel<<<sequences * heads, 1>>>(
                dr, dd, dk, dv, da, db, di, do_, ds, sequences, tokens, heads, dim);
        }, 5, 20);
        const auto wave = bench_per_launch([&] {
            rwkv_wkv7_wave_kernel<<<sequences * heads * dim, qc::kWave>>>(
                dr, dd, dk, dv, da, db, di, do_, ds, sequences, tokens, heads, dim);
        }, 10, 40, 128);
        report_pair("rwkv_wkv7", scalar, wave, flops);
        qc::dfree(dr, dd, dk, dv, da, db, di, do_, ds);
    }

    {
        constexpr long long batch = 4;
        constexpr long long heads = 4;
        constexpr long long sequence = 32;
        constexpr long long dim = 32;
        const size_t count = static_cast<size_t>(batch * heads * sequence * dim);
        auto q = rng.uniforms(count, -0.30f, 0.30f);
        auto k = rng.uniforms(count, -0.30f, 0.30f);
        auto v = rng.uniforms(count, -0.30f, 0.30f);
        float *dq = qc::dnew(q);
        float *dk = qc::dnew(k);
        float *dv = qc::dnew(v);
        float *do_ = qc::dzero<float>(count);
        double *dkv = qc::dzero<double>(static_cast<size_t>(batch * heads * dim * dim));
        const double flops = double(batch * heads * sequence * dim * dim) * 4.0;
        const auto scalar = bench_per_launch([&] {
            linear_attention_unnormalized_scalar_kernel<<<batch * heads, 1>>>(
                dq, dk, dv, do_, batch, heads, sequence, dim);
        }, 3, 12);
        const auto staged = bench_per_launch([&] {
            linear_attention_kv_kernel<<<dim3(dim, dim, batch * heads), qc::kWave>>>(
                dk, dv, dkv, batch, heads, sequence, dim);
            linear_attention_out_kernel<<<dim3(sequence, dim, batch * heads), qc::kWave>>>(
                dq, dkv, do_, batch, heads, sequence, dim);
        }, 10, 40, 128);
        report_pair("linear_attention_unnormalized", scalar, staged, flops);
        qc::dfree(dq, dk, dv, do_, dkv);
    }

    {
        constexpr long long requests = 8;
        constexpr long long slots = 8;
        constexpr long long key_heads = 2;
        constexpr long long value_heads = 4;
        constexpr long long key_dim = 64;
        constexpr long long value_dim = 64;
        const std::vector<int32_t> cu = make_cu({8, 12, 10, 14, 9, 11, 7, 13});
        const long long total_tokens = cu.back();
        std::vector<int32_t> slots_h(requests);
        for (long long i = 0; i < requests; ++i) slots_h[i] = static_cast<int32_t>(i);
        auto q = rng.uniforms(static_cast<size_t>(total_tokens * key_heads * key_dim), -0.10f, 0.10f);
        auto k = rng.uniforms(q.size(), -0.10f, 0.10f);
        auto v = rng.uniforms(static_cast<size_t>(total_tokens * value_heads * value_dim), -0.18f, 0.18f);
        auto decay = rng.uniforms(static_cast<size_t>(total_tokens * value_heads), 0.78f, 0.98f);
        auto beta = rng.uniforms(decay.size(), 0.10f, 0.80f);
        auto state_in = rng.uniforms(static_cast<size_t>(slots * value_heads * value_dim * key_dim), -0.01f, 0.01f);
        float *dq = qc::dnew(q);
        float *dk = qc::dnew(k);
        float *dv = qc::dnew(v);
        float *dd = qc::dnew(decay);
        float *db = qc::dnew(beta);
        float *dsi = qc::dnew(state_in);
        float *dso = qc::dzero<float>(state_in.size());
        float *do_ = qc::dzero<float>(v.size());
        int32_t *dcu = qc::dnew(cu);
        int32_t *dsl = qc::dnew(slots_h);
        const double flops = double(total_tokens * value_heads * value_dim * key_dim) * 6.0;
        const auto scalar = bench_per_launch([&] {
            copy_float_kernel<<<qc::grid_for(state_in.size(), 256), 256>>>(dsi, dso, state_in.size());
            gdn_recurrence_scalar_kernel<<<requests, 1>>>(
                dq, dk, dv, dd, db, dso, dcu, dsl, do_, requests, key_heads,
                value_heads, key_dim, value_dim, true);
        }, 3, 12);
        const auto wave = bench_per_launch([&] {
            copy_float_kernel<<<qc::grid_for(state_in.size(), 256), 256>>>(dsi, dso, state_in.size());
            gdn_recurrence_wave_kernel<<<dim3(value_dim, value_heads, requests), qc::kWave>>>(
                dq, dk, dv, dd, db, dso, dcu, dsl, do_, requests, key_heads,
                value_heads, key_dim, value_dim, true);
        }, 10, 40, 128);
        report_pair("gdn_recurrence", scalar, wave, flops);
        qc::dfree(dq, dk, dv, dd, db, dsi, dso, do_, dcu, dsl);
    }

    {
        constexpr long long requests = 64;
        constexpr long long slots = 64;
        constexpr long long channels = 512;
        constexpr long long kernel_size = 4;
        std::vector<int32_t> lengths(static_cast<size_t>(requests), 4);
        const std::vector<int32_t> cu = make_cu(lengths);
        const long long total_tokens = cu.back();
        std::vector<int32_t> slots_h(static_cast<size_t>(requests));
        for (long long i = 0; i < requests; ++i) slots_h[i] = static_cast<int32_t>(i);
        auto x = rng.uniforms(static_cast<size_t>(total_tokens * channels), -0.35f, 0.35f);
        auto weight = rng.uniforms(static_cast<size_t>(channels * kernel_size), -0.20f, 0.20f);
        auto state_in = rng.uniforms(static_cast<size_t>(slots * channels * (kernel_size - 1)), -0.10f, 0.10f);
        float *dx = qc::dnew(x);
        float *dw = qc::dnew(weight);
        float *dsi = qc::dnew(state_in);
        float *dso = qc::dzero<float>(state_in.size());
        float *do_ = qc::dzero<float>(x.size());
        int32_t *dcu = qc::dnew(cu);
        int32_t *dsl = qc::dnew(slots_h);
        const double bytes = double(x.size() + weight.size() + state_in.size() * 2 + x.size()) * sizeof(float);
        const auto scalar = bench_per_launch([&] {
            copy_float_kernel<<<qc::grid_for(state_in.size(), 256), 256>>>(dsi, dso, state_in.size());
            gdn_short_conv_scalar_kernel<<<requests, 1>>>(
                dx, dw, dso, dcu, dsl, do_, requests, channels, kernel_size,
                true, true);
        }, 5, 20);
        const auto parallel = bench_per_launch([&] {
            copy_float_kernel<<<qc::grid_for(state_in.size(), 256), 256>>>(dsi, dso, state_in.size());
            gdn_short_conv_channel_kernel<<<qc::grid_for(static_cast<size_t>(requests * channels), 256), 256>>>(
                dx, dw, dso, dcu, dsl, do_, requests, channels, kernel_size,
                true, true);
        }, 10, 40, 1024);
        report_pair("gdn_short_conv", scalar, parallel, bytes, false);
        qc::dfree(dx, dw, dsi, dso, do_, dcu, dsl);
    }

    {
        constexpr long long tokens = 8192;
        constexpr long long key_heads = 4;
        constexpr long long value_heads = 8;
        constexpr long long key_dim = 64;
        constexpr long long value_dim = 64;
        constexpr float eps = 1e-5f;
        const long long qk_width = key_heads * key_dim;
        const long long value_width = value_heads * value_dim;
        const long long channels = 2 * qk_width + value_width;
        auto mixed = rng.uniforms(static_cast<size_t>(tokens * channels), -0.35f, 0.35f);
        float *dm = qc::dnew(mixed);
        float *dq = qc::dzero<float>(static_cast<size_t>(tokens * qk_width));
        float *dk = qc::dzero<float>(static_cast<size_t>(tokens * qk_width));
        float *dv = qc::dzero<float>(static_cast<size_t>(tokens * value_width));
        const float nan_scale = std::numeric_limits<float>::quiet_NaN();
        const double bytes = double(mixed.size() + tokens * qk_width * 2 + tokens * value_width) * sizeof(float);
        const auto scalar = bench_per_launch([&] {
            gdn_qkv_prepare_scalar_kernel<<<tokens, 1>>>(
                dm, dq, dk, dv, tokens, key_heads, value_heads, key_dim,
                value_dim, eps, nan_scale, nan_scale);
        }, 5, 20);
        const auto parallel = bench_per_launch([&] {
            gdn_qkv_prepare_qk_kernel<<<dim3(tokens, key_heads, 2), 128>>>(
                dm, dq, dk, tokens, key_heads, value_heads, key_dim, value_dim,
                eps, nan_scale, nan_scale);
            gdn_qkv_prepare_v_kernel<<<qc::grid_for(static_cast<size_t>(tokens * value_width), 256), 256>>>(
                dm, dv, tokens, key_heads, value_heads, key_dim, value_dim);
        }, 10, 40, 32);
        report_pair("gdn_qkv_prepare", scalar, parallel, bytes, false);
        qc::dfree(dm, dq, dk, dv);
    }

    {
        constexpr long long tokens = 65536;
        constexpr long long value_heads = 8;
        const size_t count = static_cast<size_t>(tokens * value_heads);
        auto a = rng.uniforms(count, -1.0f, 1.0f);
        auto b = rng.uniforms(count, -1.0f, 1.0f);
        auto a_log = rng.uniforms(static_cast<size_t>(value_heads), -1.5f, 0.5f);
        auto dt_bias = rng.uniforms(static_cast<size_t>(value_heads), -0.4f, 0.4f);
        float *da = qc::dnew(a);
        float *db = qc::dnew(b);
        float *dal = qc::dnew(a_log);
        float *ddt = qc::dnew(dt_bias);
        float *dd = qc::dzero<float>(count);
        float *dbe = qc::dzero<float>(count);
        const double bytes = double(count * 4 + value_heads * 2) * sizeof(float);
        const auto scalar = bench_per_launch([&] {
            gdn_gate_beta_scalar_kernel<<<tokens, 1>>>(
                da, db, dal, ddt, dd, dbe, tokens, value_heads);
        }, 5, 20);
        const auto parallel = bench_per_launch([&] {
            gdn_gate_beta_kernel<<<qc::grid_for(count, 256), 256>>>(
                da, db, dal, ddt, dd, dbe, count, value_heads);
        }, 10, 40, 1024);
        report_pair("gdn_gate_beta", scalar, parallel, bytes, false);
        qc::dfree(da, db, dal, ddt, dd, dbe);
    }

    {
        constexpr long long tokens = 4096;
        constexpr long long value_heads = 8;
        constexpr long long value_dim = 128;
        constexpr float eps = 1e-5f;
        const long long rows = tokens * value_heads;
        auto y = rng.uniforms(static_cast<size_t>(rows * value_dim), -0.5f, 0.5f);
        auto z = rng.uniforms(y.size(), -0.8f, 0.8f);
        auto weight = rng.uniforms(static_cast<size_t>(value_dim), 0.5f, 1.5f);
        float *dy = qc::dnew(y);
        float *dz = qc::dnew(z);
        float *dw = qc::dnew(weight);
        float *do_ = qc::dzero<float>(y.size());
        const double bytes = double(y.size() * 3 + weight.size() * rows) * sizeof(float);
        const auto scalar = bench_per_launch([&] {
            gdn_gated_rmsnorm_scalar_kernel<<<rows, 1>>>(
                dy, dz, dw, do_, rows, value_dim, eps);
        }, 5, 20);
        const auto parallel = bench_per_launch([&] {
            gdn_gated_rmsnorm_kernel<<<rows, 128>>>(dy, dz, dw, do_, rows,
                                                    value_dim, eps);
        }, 10, 40, 64);
        report_pair("gdn_gated_rmsnorm", scalar, parallel, bytes, false);
        qc::dfree(dy, dz, dw, do_);
    }
}

}  // namespace

int main(int argc, char **argv) {
    qc::print_environment("phase8_linear");
    const bool ok = run_correctness();
    if (qc::bench_requested(argc, argv)) run_benchmarks();
    return qc::finish(ok);
}
