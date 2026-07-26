/**
 * @file
 * @brief Phase 5 CPU/Metal parity ports for the canonical BaseQ family.
 *
 * BaseQ has no live Metal implementation; the canonical contract is the CPU
 * backend's base_q.h/base_q_ref.cpp. This file independently implements the
 * contract for CDNA3 and validates against fp64 host oracles.
 */
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "../../../../common/cdna3_harness.cuh"

namespace {

enum BaseQScaleType : int {
    kScaleBF16 = 0,
    kScaleF16 = 1,
    kScaleE8M0 = 2,
    kScaleE4M3 = 3,
};

enum StorageType : int {
    kStorageF32 = 0,
    kStorageF16 = 1,
    kStorageBF16 = 2,
};

struct BaseQView {
    const uint8_t *codes;
    const void *scales;
    const void *biases;
    long long rows;
    long long columns;
    int bits;
    int group_size;
    int scale_type;
    int symmetric;
};

__host__ __device__ __forceinline__ size_t storage_bytes(int type) {
    return type == kStorageF32 ? sizeof(float) : sizeof(uint16_t);
}

__host__ __device__ __forceinline__ float e4m3_decode_baseq(uint8_t v) {
    float mag;
    if (v & 0x78) {
        const int e = (v >> 3) & 0xF;
        const int m = v & 7;
#ifdef __HIP_DEVICE_COMPILE__
        mag = ldexpf(1.0f + float(m) * 0.125f, e - 7);
#else
        mag = std::ldexp(1.0f + float(m) * 0.125f, e - 7);
#endif
    } else {
        mag = float(v & 7) * 0.001953125f;
    }
    return (v & 0x80) ? -mag : mag;
}

__device__ __forceinline__ float load_scale_device(const void *storage,
                                                   size_t index, int type) {
    switch (type) {
        case kScaleBF16:
            return __bfloat162float(reinterpret_cast<const __hip_bfloat16 *>(storage)[index]);
        case kScaleF16:
            return __half2float(reinterpret_cast<const __half *>(storage)[index]);
        case kScaleE8M0:
            return exp2f(float(reinterpret_cast<const uint8_t *>(storage)[index]) - 127.0f);
        case kScaleE4M3:
            return e4m3_decode_baseq(reinterpret_cast<const uint8_t *>(storage)[index]);
        default:
            return 0.0f;
    }
}

__host__ float load_scale_host(const std::vector<__hip_bfloat16> &bf16,
                               const std::vector<__half> &f16,
                               const std::vector<uint8_t> &u8, size_t index,
                               int type) {
    switch (type) {
        case kScaleBF16:
            return __bfloat162float(bf16[index]);
        case kScaleF16:
            return __half2float(f16[index]);
        case kScaleE8M0:
            return std::ldexp(1.0f, int(u8[index]) - 127);
        case kScaleE4M3:
            return e4m3_decode_baseq(u8[index]);
        default:
            return 0.0f;
    }
}

__device__ __forceinline__ uint32_t load_code_device(BaseQView view,
                                                     long long row,
                                                     long long column) {
    const size_t row_bytes = size_t(view.columns * view.bits / 8);
    const long long bit_index = column * view.bits;
    const size_t byte_index = size_t(bit_index >> 3);
    const int shift = int(bit_index & 7);
    const uint8_t *row_codes = view.codes + size_t(row) * row_bytes;
    uint32_t word = row_codes[byte_index];
    if (shift + view.bits > 8) word |= uint32_t(row_codes[byte_index + 1]) << 8;
    return (word >> shift) & ((1u << view.bits) - 1u);
}

__device__ __forceinline__ float load_weight_device(BaseQView view,
                                                    long long row,
                                                    long long column) {
    const long long groups_per_row = view.columns / view.group_size;
    const size_t group =
        size_t(row * groups_per_row + column / view.group_size);
    const float scale = load_scale_device(view.scales, group, view.scale_type);
    const uint32_t code = load_code_device(view, row, column);
    if (view.symmetric) {
        return float(int(code) - (1 << (view.bits - 1))) * scale;
    }
    return float(code) * scale +
           load_scale_device(view.biases, group, view.scale_type);
}

__device__ __forceinline__ float load_storage_device(const void *storage,
                                                     int type, size_t index) {
    switch (type) {
        case kStorageF32:
            return reinterpret_cast<const float *>(storage)[index];
        case kStorageF16:
            return __half2float(reinterpret_cast<const __half *>(storage)[index]);
        case kStorageBF16:
            return __bfloat162float(reinterpret_cast<const __hip_bfloat16 *>(storage)[index]);
        default:
            return 0.0f;
    }
}

__device__ __forceinline__ float round_storage_device(float value, int type) {
    switch (type) {
        case kStorageF16:
            return __half2float(__float2half(value));
        case kStorageBF16:
            return __bfloat162float(__float2bfloat16(value));
        case kStorageF32:
        default:
            return value;
    }
}

__device__ __forceinline__ void store_storage_device(void *storage, int type,
                                                     size_t index, float value) {
    switch (type) {
        case kStorageF32:
            reinterpret_cast<float *>(storage)[index] = value;
            break;
        case kStorageF16:
            reinterpret_cast<__half *>(storage)[index] = __float2half(value);
            break;
        case kStorageBF16:
            reinterpret_cast<__hip_bfloat16 *>(storage)[index] =
                __float2bfloat16(value);
            break;
    }
}

__device__ __forceinline__ float silu_device(float x) {
    return x / (1.0f + expf(-x));
}

__global__ void base_q_dequant_kernel(BaseQView weights, void *output,
                                      int output_type) {
    const long long index = blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = weights.rows * weights.columns;
    if (index >= total) return;
    const long long row = index / weights.columns;
    const long long column = index - row * weights.columns;
    store_storage_device(output, output_type, size_t(index),
                         load_weight_device(weights, row, column));
}

__global__ void base_q_embedding_kernel(BaseQView weights,
                                        const int *__restrict__ ids,
                                        long long tokens, void *output,
                                        int output_type) {
    const long long index = blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = tokens * weights.columns;
    if (index >= total) return;
    const long long token = index / weights.columns;
    const long long column = index - token * weights.columns;
    const int row = ids[token];
    const float value =
        row < 0 || row >= weights.rows ? 0.0f
                                       : load_weight_device(weights, row, column);
    store_storage_device(output, output_type, size_t(index), value);
}

__global__ void base_q_gemm_scalar_kernel(BaseQView weights, const void *input,
                                          int input_type, void *output,
                                          int output_type, long long m) {
    const long long index = blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = m * weights.rows;
    if (index >= total) return;
    const long long input_row = index / weights.rows;
    const long long weight_row = index - input_row * weights.rows;
    float sums[4] = {};
    for (long long column = 0; column < weights.columns; ++column) {
        const float x = load_storage_device(
            input, input_type, size_t(input_row * weights.columns + column));
        sums[column & 3] += load_weight_device(weights, weight_row, column) * x;
    }
    const float value = (sums[0] + sums[1]) + (sums[2] + sums[3]);
    store_storage_device(output, output_type, size_t(index), value);
}

__global__ void base_q_gemm_wave64_kernel(BaseQView weights, const void *input,
                                          int input_type, void *output,
                                          int output_type, long long m) {
    const int lane = threadIdx.x & (qc::kWave - 1);
    const int wave = threadIdx.x >> 6;
    const int waves_per_block = blockDim.x >> 6;
    const long long index = static_cast<long long>(blockIdx.x) * waves_per_block + wave;
    const long long total = m * weights.rows;
    if (index >= total) return;
    const long long input_row = index / weights.rows;
    const long long weight_row = index - input_row * weights.rows;
    float acc = 0.0f;
    for (long long column = lane; column < weights.columns; column += qc::kWave) {
        const float x = load_storage_device(
            input, input_type, size_t(input_row * weights.columns + column));
        acc += load_weight_device(weights, weight_row, column) * x;
    }
    acc = qc::wave_reduce_sum(acc);
    if (lane == 0) store_storage_device(output, output_type, size_t(index), acc);
}

__global__ void base_q_gemv_qkv_scalar_kernel(
    BaseQView q_weights, BaseQView k_weights, BaseQView v_weights,
    const void *input, int input_type, void *q_output, int q_type,
    void *k_output, int k_type, void *v_output, int v_type) {
    const long long index = blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = q_weights.rows + k_weights.rows + v_weights.rows;
    if (index >= total) return;
    BaseQView weights = q_weights;
    void *output = q_output;
    int output_type = q_type;
    long long row = index;
    if (row >= q_weights.rows) {
        row -= q_weights.rows;
        weights = k_weights;
        output = k_output;
        output_type = k_type;
        if (row >= k_weights.rows) {
            row -= k_weights.rows;
            weights = v_weights;
            output = v_output;
            output_type = v_type;
        }
    }
    float sums[4] = {};
    for (long long column = 0; column < weights.columns; ++column) {
        sums[column & 3] +=
            load_weight_device(weights, row, column) *
            load_storage_device(input, input_type, size_t(column));
    }
    const float value = (sums[0] + sums[1]) + (sums[2] + sums[3]);
    store_storage_device(output, output_type, size_t(row), value);
}

__global__ void base_q_gemv_qkv_wave64_kernel(
    BaseQView q_weights, BaseQView k_weights, BaseQView v_weights,
    const void *input, int input_type, void *q_output, int q_type,
    void *k_output, int k_type, void *v_output, int v_type) {
    const int lane = threadIdx.x & (qc::kWave - 1);
    const int wave = threadIdx.x >> 6;
    const int waves_per_block = blockDim.x >> 6;
    const long long index = static_cast<long long>(blockIdx.x) * waves_per_block + wave;
    const long long total = q_weights.rows + k_weights.rows + v_weights.rows;
    if (index >= total) return;
    BaseQView weights = q_weights;
    void *output = q_output;
    int output_type = q_type;
    long long row = index;
    if (row >= q_weights.rows) {
        row -= q_weights.rows;
        weights = k_weights;
        output = k_output;
        output_type = k_type;
        if (row >= k_weights.rows) {
            row -= k_weights.rows;
            weights = v_weights;
            output = v_output;
            output_type = v_type;
        }
    }
    float acc = 0.0f;
    for (long long column = lane; column < weights.columns; column += qc::kWave) {
        acc += load_weight_device(weights, row, column) *
               load_storage_device(input, input_type, size_t(column));
    }
    acc = qc::wave_reduce_sum(acc);
    if (lane == 0) store_storage_device(output, output_type, size_t(row), acc);
}

__global__ void base_q_gemv_swiglu_scalar_kernel(BaseQView gate_weights,
                                                 BaseQView up_weights,
                                                 const void *input,
                                                 int input_type, void *output,
                                                 int output_type) {
    const long long row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= gate_weights.rows) return;
    float gate_sums[4] = {};
    float up_sums[4] = {};
    for (long long column = 0; column < gate_weights.columns; ++column) {
        const float x = load_storage_device(input, input_type, size_t(column));
        gate_sums[column & 3] += load_weight_device(gate_weights, row, column) * x;
        up_sums[column & 3] += load_weight_device(up_weights, row, column) * x;
    }
    const float gate =
        (gate_sums[0] + gate_sums[1]) + (gate_sums[2] + gate_sums[3]);
    const float up = (up_sums[0] + up_sums[1]) + (up_sums[2] + up_sums[3]);
    store_storage_device(output, output_type, size_t(row),
                         silu_device(gate) * up);
}

__global__ void base_q_gemv_swiglu_wave64_kernel(BaseQView gate_weights,
                                                 BaseQView up_weights,
                                                 const void *input,
                                                 int input_type, void *output,
                                                 int output_type) {
    const int lane = threadIdx.x & (qc::kWave - 1);
    const int wave = threadIdx.x >> 6;
    const int waves_per_block = blockDim.x >> 6;
    const long long row = static_cast<long long>(blockIdx.x) * waves_per_block + wave;
    if (row >= gate_weights.rows) return;
    float gate = 0.0f;
    float up = 0.0f;
    for (long long column = lane; column < gate_weights.columns; column += qc::kWave) {
        const float x = load_storage_device(input, input_type, size_t(column));
        gate += load_weight_device(gate_weights, row, column) * x;
        up += load_weight_device(up_weights, row, column) * x;
    }
    gate = qc::wave_reduce_sum(gate);
    up = qc::wave_reduce_sum(up);
    if (lane == 0) {
        store_storage_device(output, output_type, size_t(row),
                             silu_device(gate) * up);
    }
}

__global__ void base_q_lm_head_argmax_kernel(BaseQView weights,
                                             const void *input, int input_type,
                                             int *__restrict__ token_ids,
                                             long long batch) {
    const long long input_row = blockIdx.x;
    if (input_row >= batch) return;
    float best_value = -INFINITY;
    int best_token = std::numeric_limits<int>::max();
    for (long long token = threadIdx.x; token < weights.rows; token += blockDim.x) {
        float sums[4] = {};
        for (long long column = 0; column < weights.columns; ++column) {
            const float x = load_storage_device(
                input, input_type, size_t(input_row * weights.columns + column));
            sums[column & 3] += load_weight_device(weights, token, column) * x;
        }
        const float value = round_storage_device(
            (sums[0] + sums[1]) + (sums[2] + sums[3]), input_type);
        if (value > best_value || (value == best_value && token < best_token)) {
            best_value = value;
            best_token = int(token);
        }
    }

    const int lane = threadIdx.x & (qc::kWave - 1);
    const int wave = threadIdx.x >> 6;
    best_token = best_token == std::numeric_limits<int>::max()
                     ? std::numeric_limits<int>::max()
                     : best_token;
    qc::wave_reduce_argmax(best_value, best_token);

    __shared__ float wave_values[qc::kWavesPerBlock];
    __shared__ int wave_tokens[qc::kWavesPerBlock];
    if (lane == 0) {
        wave_values[wave] = best_value;
        wave_tokens[wave] = best_token;
    }
    __syncthreads();

    if (wave == 0) {
        float value =
            lane < qc::kWavesPerBlock ? wave_values[lane] : -INFINITY;
        int token = lane < qc::kWavesPerBlock
                        ? wave_tokens[lane]
                        : std::numeric_limits<int>::max();
        qc::wave_reduce_argmax(value, token);
        if (lane == 0) token_ids[input_row] = token;
    }
}

__global__ void base_q_materialized_argmax_kernel(const void *scores,
                                                  int score_type,
                                                  int *__restrict__ token_ids,
                                                  long long batch,
                                                  long long vocab) {
    const long long input_row = blockIdx.x;
    if (input_row >= batch) return;
    float best_value = -INFINITY;
    int best_token = std::numeric_limits<int>::max();
    for (long long token = threadIdx.x; token < vocab; token += blockDim.x) {
        const float value =
            load_storage_device(scores, score_type, size_t(input_row * vocab + token));
        if (value > best_value || (value == best_value && token < best_token)) {
            best_value = value;
            best_token = int(token);
        }
    }
    const int lane = threadIdx.x & (qc::kWave - 1);
    const int wave = threadIdx.x >> 6;
    qc::wave_reduce_argmax(best_value, best_token);
    __shared__ float wave_values[qc::kWavesPerBlock];
    __shared__ int wave_tokens[qc::kWavesPerBlock];
    if (lane == 0) {
        wave_values[wave] = best_value;
        wave_tokens[wave] = best_token;
    }
    __syncthreads();
    if (wave == 0) {
        float value =
            lane < qc::kWavesPerBlock ? wave_values[lane] : -INFINITY;
        int token = lane < qc::kWavesPerBlock
                        ? wave_tokens[lane]
                        : std::numeric_limits<int>::max();
        qc::wave_reduce_argmax(value, token);
        if (lane == 0) token_ids[input_row] = token;
    }
}

__global__ void base_q_moe_gemm_scalar_kernel(BaseQView weights,
                                              long long experts,
                                              const void *input, int input_type,
                                              const int *__restrict__ expert_of_tile,
                                              long long total_rows,
                                              void *output, int output_type) {
    const long long output_rows = weights.rows / experts;
    const long long index = blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = total_rows * output_rows;
    if (index >= total) return;
    const long long input_row = index / output_rows;
    const long long output_row = index - input_row * output_rows;
    const int expert = expert_of_tile[input_row / 32];
    if (expert < 0 || expert >= experts) {
        store_storage_device(output, output_type, size_t(index), 0.0f);
        return;
    }
    const long long weight_row = static_cast<long long>(expert) * output_rows + output_row;
    float sums[4] = {};
    for (long long column = 0; column < weights.columns; ++column) {
        const float x = load_storage_device(
            input, input_type, size_t(input_row * weights.columns + column));
        sums[column & 3] += load_weight_device(weights, weight_row, column) * x;
    }
    const float value = (sums[0] + sums[1]) + (sums[2] + sums[3]);
    store_storage_device(output, output_type, size_t(index), value);
}

__global__ void base_q_moe_gemm_wave64_kernel(BaseQView weights,
                                              long long experts,
                                              const void *input, int input_type,
                                              const int *__restrict__ expert_of_tile,
                                              long long total_rows,
                                              void *output, int output_type) {
    const long long output_rows = weights.rows / experts;
    const int lane = threadIdx.x & (qc::kWave - 1);
    const int wave = threadIdx.x >> 6;
    const int waves_per_block = blockDim.x >> 6;
    const long long index = static_cast<long long>(blockIdx.x) * waves_per_block + wave;
    const long long total = total_rows * output_rows;
    if (index >= total) return;
    const long long input_row = index / output_rows;
    const long long output_row = index - input_row * output_rows;
    const int expert = expert_of_tile[input_row / 32];
    if (expert < 0 || expert >= experts) {
        if (lane == 0) store_storage_device(output, output_type, size_t(index), 0.0f);
        return;
    }
    const long long weight_row = static_cast<long long>(expert) * output_rows + output_row;
    float acc = 0.0f;
    for (long long column = lane; column < weights.columns; column += qc::kWave) {
        const float x = load_storage_device(
            input, input_type, size_t(input_row * weights.columns + column));
        acc += load_weight_device(weights, weight_row, column) * x;
    }
    acc = qc::wave_reduce_sum(acc);
    if (lane == 0) store_storage_device(output, output_type, size_t(index), acc);
}

__global__ void base_q_moe_swiglu_scalar_kernel(
    BaseQView weights, long long experts, const void *input, int input_type,
    const int *__restrict__ expert_of_tile, long long total_rows, void *output,
    int output_type) {
    const long long intermediate = weights.rows / experts / 2;
    const long long index = blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = total_rows * intermediate;
    if (index >= total) return;
    const long long input_row = index / intermediate;
    const long long output_row = index - input_row * intermediate;
    const int expert = expert_of_tile[input_row / 32];
    if (expert < 0 || expert >= experts) {
        store_storage_device(output, output_type, size_t(index), 0.0f);
        return;
    }
    const long long expert_base = static_cast<long long>(expert) * 2 * intermediate;
    const long long gate_row = expert_base + output_row;
    const long long up_row = expert_base + intermediate + output_row;
    float gate_sums[4] = {};
    float up_sums[4] = {};
    for (long long column = 0; column < weights.columns; ++column) {
        const float x = load_storage_device(
            input, input_type, size_t(input_row * weights.columns + column));
        gate_sums[column & 3] += load_weight_device(weights, gate_row, column) * x;
        up_sums[column & 3] += load_weight_device(weights, up_row, column) * x;
    }
    const float gate =
        (gate_sums[0] + gate_sums[1]) + (gate_sums[2] + gate_sums[3]);
    const float up = (up_sums[0] + up_sums[1]) + (up_sums[2] + up_sums[3]);
    store_storage_device(output, output_type, size_t(index),
                         silu_device(gate) * up);
}

__global__ void base_q_moe_swiglu_wave64_kernel(
    BaseQView weights, long long experts, const void *input, int input_type,
    const int *__restrict__ expert_of_tile, long long total_rows, void *output,
    int output_type) {
    const long long intermediate = weights.rows / experts / 2;
    const int lane = threadIdx.x & (qc::kWave - 1);
    const int wave = threadIdx.x >> 6;
    const int waves_per_block = blockDim.x >> 6;
    const long long index = static_cast<long long>(blockIdx.x) * waves_per_block + wave;
    const long long total = total_rows * intermediate;
    if (index >= total) return;
    const long long input_row = index / intermediate;
    const long long output_row = index - input_row * intermediate;
    const int expert = expert_of_tile[input_row / 32];
    if (expert < 0 || expert >= experts) {
        if (lane == 0) store_storage_device(output, output_type, size_t(index), 0.0f);
        return;
    }
    const long long expert_base = static_cast<long long>(expert) * 2 * intermediate;
    const long long gate_row = expert_base + output_row;
    const long long up_row = expert_base + intermediate + output_row;
    float gate = 0.0f;
    float up = 0.0f;
    for (long long column = lane; column < weights.columns; column += qc::kWave) {
        const float x = load_storage_device(
            input, input_type, size_t(input_row * weights.columns + column));
        gate += load_weight_device(weights, gate_row, column) * x;
        up += load_weight_device(weights, up_row, column) * x;
    }
    gate = qc::wave_reduce_sum(gate);
    up = qc::wave_reduce_sum(up);
    if (lane == 0) {
        store_storage_device(output, output_type, size_t(index),
                             silu_device(gate) * up);
    }
}

struct Fixture {
    int bits = 0;
    long long rows = 3;
    long long columns = 128;
    int group_size = 32;
    int scale_type = kScaleBF16;
    bool symmetric = false;
    std::vector<uint8_t> codes;
    std::vector<__hip_bfloat16> scales_bf16;
    std::vector<__hip_bfloat16> biases_bf16;
    std::vector<__half> scales_f16;
    std::vector<__half> biases_f16;
    std::vector<uint8_t> scales8;
    std::vector<uint8_t> biases8;
    std::vector<double> decoded;

    size_t groups() const {
        return size_t(rows * (columns / group_size));
    }
    const void *scales() const {
        if (scale_type == kScaleBF16) return scales_bf16.data();
        if (scale_type == kScaleF16) return scales_f16.data();
        return scales8.data();
    }
    const void *biases() const {
        if (symmetric) return nullptr;
        if (scale_type == kScaleBF16) return biases_bf16.data();
        if (scale_type == kScaleF16) return biases_f16.data();
        return biases8.data();
    }
    size_t scale_bytes() const {
        return scale_type == kScaleBF16 || scale_type == kScaleF16 ? 2 : 1;
    }
};

void insert_code(std::vector<uint8_t> &bytes, size_t row_offset,
                 long long column, int bits, uint32_t code) {
    const int bit = int(column * bits);
    const size_t byte = row_offset + size_t(bit >> 3);
    const int shift = bit & 7;
    const uint16_t word = uint16_t(code << shift);
    bytes[byte] |= uint8_t(word);
    if (shift + bits > 8) bytes[byte + 1] |= uint8_t(word >> 8);
}

uint32_t load_code_host(const Fixture &fixture, long long row,
                        long long column) {
    const size_t row_bytes = size_t(fixture.columns * fixture.bits / 8);
    const long long bit_index = column * fixture.bits;
    const size_t byte_index = size_t(bit_index >> 3);
    const int shift = int(bit_index & 7);
    const uint8_t *row_codes =
        fixture.codes.data() + size_t(row) * row_bytes;
    uint32_t word = row_codes[byte_index];
    if (shift + fixture.bits > 8) word |= uint32_t(row_codes[byte_index + 1]) << 8;
    return (word >> shift) & ((1u << fixture.bits) - 1u);
}

float fixture_scale(const Fixture &fixture, size_t group, bool bias) {
    if (fixture.scale_type == kScaleBF16) {
        return __bfloat162float(
            bias ? fixture.biases_bf16[group] : fixture.scales_bf16[group]);
    }
    if (fixture.scale_type == kScaleF16) {
        return __half2float(
            bias ? fixture.biases_f16[group] : fixture.scales_f16[group]);
    }
    const auto &values = bias ? fixture.biases8 : fixture.scales8;
    if (fixture.scale_type == kScaleE8M0) {
        return std::ldexp(1.0f, int(values[group]) - 127);
    }
    return e4m3_decode_baseq(values[group]);
}

double fixture_weight(const Fixture &fixture, long long row, long long column) {
    const long long groups_per_row = fixture.columns / fixture.group_size;
    const size_t group =
        size_t(row * groups_per_row + column / fixture.group_size);
    const uint32_t code = load_code_host(fixture, row, column);
    const double scale = fixture_scale(fixture, group, false);
    if (fixture.symmetric) {
        return double(int(code) - (1 << (fixture.bits - 1))) * scale;
    }
    return double(code) * scale + fixture_scale(fixture, group, true);
}

Fixture make_fixture(int bits, bool symmetric = false,
                     int scale_type = kScaleBF16, int group_size = 32,
                     long long rows = 3, long long columns = 128) {
    Fixture fixture;
    fixture.bits = bits;
    fixture.symmetric = symmetric;
    fixture.scale_type = scale_type;
    fixture.group_size = group_size;
    fixture.rows = rows;
    fixture.columns = columns;
    const size_t row_bytes = size_t(columns * bits / 8);
    fixture.codes.assign(size_t(rows) * row_bytes, 0);
    const size_t group_count = fixture.groups();
    if (scale_type == kScaleBF16) {
        fixture.scales_bf16.resize(group_count);
        if (!symmetric) fixture.biases_bf16.resize(group_count);
    } else if (scale_type == kScaleF16) {
        fixture.scales_f16.resize(group_count);
        if (!symmetric) fixture.biases_f16.resize(group_count);
    } else {
        fixture.scales8.resize(group_count);
        if (!symmetric) fixture.biases8.resize(group_count);
    }

    for (size_t group = 0; group < group_count; ++group) {
        if (scale_type == kScaleBF16 || scale_type == kScaleF16) {
            const float scale = 0.0625f * float((group % 5) + 1);
            const float bias = -0.25f + 0.03125f * float(group % 7);
            if (scale_type == kScaleBF16) {
                fixture.scales_bf16[group] = __float2bfloat16(scale);
                if (!symmetric) fixture.biases_bf16[group] = __float2bfloat16(bias);
            } else {
                fixture.scales_f16[group] = __float2half(scale);
                if (!symmetric) fixture.biases_f16[group] = __float2half(bias);
            }
        } else if (scale_type == kScaleE8M0) {
            fixture.scales8[group] = uint8_t(124 + group % 4);
            if (!symmetric) fixture.biases8[group] = uint8_t(123 + group % 5);
        } else {
            constexpr uint8_t values[] = {0x28, 0x30, 0x38, 0x40};
            fixture.scales8[group] = values[group % 4];
            if (!symmetric) fixture.biases8[group] = values[(group + 1) % 4];
        }
    }

    fixture.decoded.resize(size_t(rows * columns));
    const uint32_t mask = (1u << bits) - 1u;
    for (long long row = 0; row < rows; ++row) {
        for (long long column = 0; column < columns; ++column) {
            const uint32_t code =
                uint32_t((row * 19 + column * 5 + 3) & mask);
            insert_code(fixture.codes, size_t(row) * row_bytes, column, bits,
                        code);
            fixture.decoded[size_t(row * columns + column)] =
                fixture_weight(fixture, row, column);
        }
    }
    return fixture;
}

struct DeviceFixture {
    uint8_t *codes = nullptr;
    void *scales = nullptr;
    void *biases = nullptr;
    BaseQView view{};
};

DeviceFixture upload_fixture(const Fixture &fixture) {
    DeviceFixture d;
    d.codes = qc::dnew(fixture.codes);
    QC_CHECK(hipMalloc(&d.scales, fixture.groups() * fixture.scale_bytes()));
    QC_CHECK(hipMemcpy(d.scales, fixture.scales(), fixture.groups() * fixture.scale_bytes(),
                       hipMemcpyHostToDevice));
    if (!fixture.symmetric) {
        QC_CHECK(hipMalloc(&d.biases, fixture.groups() * fixture.scale_bytes()));
        QC_CHECK(hipMemcpy(d.biases, fixture.biases(),
                           fixture.groups() * fixture.scale_bytes(),
                           hipMemcpyHostToDevice));
    }
    d.view = {d.codes, d.scales, d.biases, fixture.rows, fixture.columns,
              fixture.bits, fixture.group_size, fixture.scale_type,
              fixture.symmetric ? 1 : 0};
    return d;
}

void free_fixture(DeviceFixture &fixture) {
    if (fixture.codes) QC_CHECK(hipFree(fixture.codes));
    if (fixture.scales) QC_CHECK(hipFree(fixture.scales));
    if (fixture.biases) QC_CHECK(hipFree(fixture.biases));
    fixture = {};
}

template <typename T>
int storage_id();
template <>
int storage_id<float>() {
    return kStorageF32;
}
template <>
int storage_id<__half>() {
    return kStorageF16;
}
template <>
int storage_id<__hip_bfloat16>() {
    return kStorageBF16;
}

template <typename T>
qc::Tol storage_tol();
template <>
qc::Tol storage_tol<float>() {
    return qc::Tol::fp32().with_elementwise(5e-4, 5e-4);
}
template <>
qc::Tol storage_tol<__half>() {
    return qc::Tol::fp16_output().with_elementwise(6e-3, 6e-3);
}
template <>
qc::Tol storage_tol<__hip_bfloat16>() {
    return qc::Tol::bf16_output().with_elementwise(1.5e-2, 1.5e-2);
}

std::vector<double> make_input_values(size_t n, int stride = 23,
                                      float denom = 13.0f) {
    std::vector<double> input(n);
    for (size_t i = 0; i < n; ++i) {
        input[i] = double(int((i * size_t(stride) + 7) % 47) - 23) / denom;
    }
    return input;
}

template <typename T>
std::vector<T> to_storage_vec(const std::vector<double> &src) {
    std::vector<float> f(src.size());
    for (size_t i = 0; i < src.size(); ++i) f[i] = float(src[i]);
    return qc::to_storage<T>(f);
}

template <typename T>
std::vector<double> storage_to_double(const std::vector<T> &src) {
    std::vector<double> out(src.size());
    for (size_t i = 0; i < src.size(); ++i) out[i] = qc::to_double(src[i]);
    return out;
}

template <typename T>
std::vector<double> input_as_double(const std::vector<T> &src) {
    return storage_to_double(src);
}

std::vector<double> dense_projection(const Fixture &fixture,
                                     const std::vector<double> &input,
                                     long long m) {
    std::vector<double> output(size_t(m * fixture.rows));
    for (long long input_row = 0; input_row < m; ++input_row) {
        for (long long weight_row = 0; weight_row < fixture.rows; ++weight_row) {
            double acc = 0.0;
            for (long long column = 0; column < fixture.columns; ++column) {
                acc += fixture.decoded[size_t(weight_row * fixture.columns + column)] *
                       input[size_t(input_row * fixture.columns + column)];
            }
            output[size_t(input_row * fixture.rows + weight_row)] = acc;
        }
    }
    return output;
}

double silu_host(double x) {
    return x / (1.0 + std::exp(-x));
}

double round_for_storage(double value, int storage_type) {
    if (storage_type == kStorageF16) return qc::round_fp16(value);
    if (storage_type == kStorageBF16) return qc::round_bf16(value);
    return value;
}

template <typename Out>
bool run_dequant_case(const Fixture &fixture, const std::string &label) {
    DeviceFixture d = upload_fixture(fixture);
    Out *out = qc::dzero<Out>(fixture.decoded.size());
    base_q_dequant_kernel<<<qc::grid_for(fixture.decoded.size(), 256), 256>>>(
        d.view, out, storage_id<Out>());
    QC_SYNC();
    const auto got = qc::d2h(out, fixture.decoded.size());
    const bool ok =
        qc::compare(got, fixture.decoded, storage_tol<Out>()).report(label);
    qc::dfree(out);
    free_fixture(d);
    return ok;
}

template <typename In, typename Out>
bool run_gemm_case(const Fixture &fixture, long long m,
                   const std::string &label) {
    const auto src = make_input_values(size_t(m * fixture.columns));
    const auto h_in = to_storage_vec<In>(src);
    const auto ref_input = input_as_double(h_in);
    const auto ref = dense_projection(fixture, ref_input, m);
    DeviceFixture d = upload_fixture(fixture);
    In *din = qc::dnew(h_in);
    Out *dout = qc::dzero<Out>(ref.size());
    base_q_gemm_wave64_kernel<<<qc::wave_blocks(ref.size()), qc::kThreads>>>(
        d.view, din, storage_id<In>(), dout, storage_id<Out>(), m);
    QC_SYNC();
    const auto got = qc::d2h(dout, ref.size());
    const bool ok = qc::compare(got, ref, storage_tol<Out>()).report(label);
    qc::dfree(din, dout);
    free_fixture(d);
    return ok;
}

template <typename Out>
bool run_embedding_case(const Fixture &fixture, const std::string &label) {
    const std::vector<int> ids = {2, -1, 99, 0};
    std::vector<double> ref(size_t(ids.size()) * size_t(fixture.columns));
    for (size_t token = 0; token < ids.size(); ++token) {
        for (long long column = 0; column < fixture.columns; ++column) {
            const int row = ids[token];
            ref[token * size_t(fixture.columns) + size_t(column)] =
                row < 0 || row >= fixture.rows
                    ? 0.0
                    : fixture.decoded[size_t(row * fixture.columns + column)];
        }
    }
    DeviceFixture d = upload_fixture(fixture);
    int *dids = qc::dnew(ids);
    Out *dout = qc::dzero<Out>(ref.size());
    base_q_embedding_kernel<<<qc::grid_for(ref.size(), 256), 256>>>(
        d.view, dids, static_cast<long long>(ids.size()), dout, storage_id<Out>());
    QC_SYNC();
    const auto got = qc::d2h(dout, ref.size());
    const bool ok = qc::compare(got, ref, storage_tol<Out>()).report(label);
    qc::dfree(dids, dout);
    free_fixture(d);
    return ok;
}

template <typename In, typename Out>
bool run_qkv_case(const std::string &label) {
    Fixture q = make_fixture(5, false, kScaleBF16, 32, 2);
    Fixture k = make_fixture(5, false, kScaleBF16, 32, 3);
    Fixture v = make_fixture(5, false, kScaleBF16, 32, 4);
    const auto src = make_input_values(size_t(k.columns), 17, 9.0f);
    const auto h_in = to_storage_vec<In>(src);
    const auto ref_input = input_as_double(h_in);
    const auto q_ref = dense_projection(q, ref_input, 1);
    const auto k_ref = dense_projection(k, ref_input, 1);
    const auto v_ref = dense_projection(v, ref_input, 1);

    DeviceFixture dq = upload_fixture(q);
    DeviceFixture dk = upload_fixture(k);
    DeviceFixture dv = upload_fixture(v);
    In *din = qc::dnew(h_in);
    Out *dqo = qc::dzero<Out>(q_ref.size());
    Out *dko = qc::dzero<Out>(k_ref.size());
    Out *dvo = qc::dzero<Out>(v_ref.size());
    const long long total = q.rows + k.rows + v.rows;
    base_q_gemv_qkv_wave64_kernel<<<qc::wave_blocks(total), qc::kThreads>>>(
        dq.view, dk.view, dv.view, din, storage_id<In>(), dqo,
        storage_id<Out>(), dko, storage_id<Out>(), dvo, storage_id<Out>());
    QC_SYNC();
    bool ok = true;
    ok &= qc::compare(qc::d2h(dqo, q_ref.size()), q_ref, storage_tol<Out>())
              .report(label + " q");
    ok &= qc::compare(qc::d2h(dko, k_ref.size()), k_ref, storage_tol<Out>())
              .report(label + " k");
    ok &= qc::compare(qc::d2h(dvo, v_ref.size()), v_ref, storage_tol<Out>())
              .report(label + " v");
    qc::dfree(din, dqo, dko, dvo);
    free_fixture(dq);
    free_fixture(dk);
    free_fixture(dv);
    return ok;
}

template <typename In, typename Out>
bool run_swiglu_case(const Fixture &fixture, const std::string &label) {
    const auto src = make_input_values(size_t(fixture.columns), 17, 9.0f);
    const auto h_in = to_storage_vec<In>(src);
    const auto ref_input = input_as_double(h_in);
    const auto proj = dense_projection(fixture, ref_input, 1);
    std::vector<double> ref(size_t(fixture.rows));
    for (long long row = 0; row < fixture.rows; ++row)
        ref[size_t(row)] = silu_host(proj[size_t(row)]) * proj[size_t(row)];
    DeviceFixture d = upload_fixture(fixture);
    In *din = qc::dnew(h_in);
    Out *dout = qc::dzero<Out>(ref.size());
    base_q_gemv_swiglu_wave64_kernel<<<qc::wave_blocks(fixture.rows),
                                       qc::kThreads>>>(
        d.view, d.view, din, storage_id<In>(), dout, storage_id<Out>());
    QC_SYNC();
    const auto got = qc::d2h(dout, ref.size());
    const bool ok = qc::compare(got, ref, storage_tol<Out>()).report(label);
    qc::dfree(din, dout);
    free_fixture(d);
    return ok;
}

template <typename In>
bool run_lm_head_case(int bits, bool symmetric, const std::string &label) {
    Fixture fixture = make_fixture(bits, symmetric, kScaleF16, 32, 19);
    constexpr long long batch = 3;
    const auto src = make_input_values(size_t(batch * fixture.columns), 11, 29.0f);
    const auto h_in = to_storage_vec<In>(src);
    const auto ref_input = input_as_double(h_in);
    const auto logits = dense_projection(fixture, ref_input, batch);
    std::vector<double> ref(batch);
    for (long long input_row = 0; input_row < batch; ++input_row) {
        int best = 0;
        double best_value =
            round_for_storage(logits[size_t(input_row * fixture.rows)],
                              storage_id<In>());
        for (long long token = 1; token < fixture.rows; ++token) {
            const double value = round_for_storage(
                logits[size_t(input_row * fixture.rows + token)],
                storage_id<In>());
            if (value > best_value) {
                best_value = value;
                best = int(token);
            }
        }
        ref[size_t(input_row)] = best;
    }
    DeviceFixture d = upload_fixture(fixture);
    In *din = qc::dnew(h_in);
    In *dscores = qc::dzero<In>(size_t(batch * fixture.rows));
    int *dids = qc::dzero<int>(batch);
    base_q_gemm_wave64_kernel<<<qc::wave_blocks(size_t(batch * fixture.rows)),
                                qc::kThreads>>>(
        d.view, din, storage_id<In>(), dscores, storage_id<In>(), batch);
    base_q_materialized_argmax_kernel<<<batch, qc::kThreads>>>(
        dscores, storage_id<In>(), dids, batch, fixture.rows);
    QC_SYNC();
    const auto got = qc::d2h(dids, batch);
    bool ok = qc::compare(got, ref, qc::Tol::exact()).report(label);

    Fixture ties = make_fixture(4, false, kScaleF16, 32, 17);
    std::fill(ties.codes.begin(), ties.codes.end(), 0);
    std::fill(ties.scales_f16.begin(), ties.scales_f16.end(), __float2half(0.0f));
    std::fill(ties.biases_f16.begin(), ties.biases_f16.end(), __float2half(0.0f));
    DeviceFixture dt = upload_fixture(ties);
    std::vector<double> ones(size_t(2 * ties.columns), 1.0);
    const auto h_ones = to_storage_vec<In>(ones);
    In *dones = qc::dnew(h_ones);
    In *dtie_scores = qc::dzero<In>(size_t(2 * ties.rows));
    int *dtok = qc::dzero<int>(2);
    base_q_gemm_wave64_kernel<<<qc::wave_blocks(size_t(2 * ties.rows)),
                                qc::kThreads>>>(
        dt.view, dones, storage_id<In>(), dtie_scores, storage_id<In>(), 2);
    base_q_materialized_argmax_kernel<<<2, qc::kThreads>>>(
        dtie_scores, storage_id<In>(), dtok, 2, ties.rows);
    QC_SYNC();
    ok &= qc::compare(qc::d2h(dtok, 2), std::vector<double>{0.0, 0.0},
                      qc::Tol::exact())
              .report(label + " tie");

    qc::dfree(din, dscores, dids, dones, dtie_scores, dtok);
    free_fixture(d);
    free_fixture(dt);
    return ok;
}

std::vector<double> moe_gemm_ref(const Fixture &fixture,
                                 const std::vector<double> &input,
                                 const std::vector<int> &expert_of_tile,
                                 long long experts, long long total_rows) {
    const long long output_rows = fixture.rows / experts;
    std::vector<double> ref(size_t(total_rows * output_rows));
    for (long long row = 0; row < total_rows; ++row) {
        const int expert = expert_of_tile[size_t(row / 32)];
        for (long long output_row = 0; output_row < output_rows; ++output_row) {
            const long long weight_row = static_cast<long long>(expert) * output_rows + output_row;
            double acc = 0.0;
            for (long long column = 0; column < fixture.columns; ++column) {
                acc += input[size_t(row * fixture.columns + column)] *
                       fixture.decoded[size_t(weight_row * fixture.columns + column)];
            }
            ref[size_t(row * output_rows + output_row)] = acc;
        }
    }
    return ref;
}

std::vector<double> moe_swiglu_ref(const Fixture &fixture,
                                   const std::vector<double> &input,
                                   const std::vector<int> &expert_of_tile,
                                   long long experts, long long total_rows) {
    const long long intermediate = fixture.rows / experts / 2;
    std::vector<double> ref(size_t(total_rows * intermediate));
    for (long long row = 0; row < total_rows; ++row) {
        const int expert = expert_of_tile[size_t(row / 32)];
        const long long expert_base = static_cast<long long>(expert) * 2 * intermediate;
        for (long long output_row = 0; output_row < intermediate; ++output_row) {
            double gate = 0.0;
            double up = 0.0;
            for (long long column = 0; column < fixture.columns; ++column) {
                const double x = input[size_t(row * fixture.columns + column)];
                gate += x * fixture.decoded[size_t((expert_base + output_row) *
                                                   fixture.columns + column)];
                up += x * fixture.decoded[size_t((expert_base + intermediate +
                                                  output_row) *
                                                 fixture.columns + column)];
            }
            ref[size_t(row * intermediate + output_row)] = silu_host(gate) * up;
        }
    }
    return ref;
}

template <typename T>
std::vector<double> maybe_round_ref(const std::vector<double> &ref) {
    std::vector<double> rounded(ref.size());
    for (size_t i = 0; i < ref.size(); ++i)
        rounded[i] = round_for_storage(ref[i], storage_id<T>());
    return rounded;
}

template <typename T>
bool run_moe_cases(int bits, bool symmetric, const std::string &label) {
    constexpr long long experts = 2;
    constexpr long long output_rows = 64;
    constexpr long long intermediate = 32;
    constexpr long long total_rows = 64;
    const std::vector<int> expert_of_tile = {1, 0};
    Fixture rect = make_fixture(bits, symmetric, kScaleF16, 32,
                                experts * output_rows);
    Fixture swiglu = make_fixture(bits, symmetric, kScaleF16, 32,
                                  experts * 2 * intermediate);
    const auto src = make_input_values(size_t(total_rows * rect.columns), 7, 31.0f);
    const auto h_in = to_storage_vec<T>(src);
    const auto ref_input = input_as_double(h_in);
    const auto rect_ref = maybe_round_ref<T>(
        moe_gemm_ref(rect, ref_input, expert_of_tile, experts, total_rows));
    const auto swiglu_ref = maybe_round_ref<T>(
        moe_swiglu_ref(swiglu, ref_input, expert_of_tile, experts, total_rows));

    DeviceFixture drect = upload_fixture(rect);
    DeviceFixture dswiglu = upload_fixture(swiglu);
    T *din = qc::dnew(h_in);
    int *dexperts = qc::dnew(expert_of_tile);
    T *drect_out = qc::dzero<T>(rect_ref.size());
    T *dswiglu_out = qc::dzero<T>(swiglu_ref.size());
    base_q_moe_gemm_wave64_kernel<<<qc::wave_blocks(rect_ref.size()),
                                    qc::kThreads>>>(
        drect.view, experts, din, storage_id<T>(), dexperts, total_rows,
        drect_out, storage_id<T>());
    base_q_moe_swiglu_wave64_kernel<<<qc::wave_blocks(swiglu_ref.size()),
                                      qc::kThreads>>>(
        dswiglu.view, experts, din, storage_id<T>(), dexperts, total_rows,
        dswiglu_out, storage_id<T>());
    QC_SYNC();
    bool ok = true;
    ok &= qc::compare(qc::d2h(drect_out, rect_ref.size()), rect_ref,
                      storage_tol<T>())
              .report(label + " moe_gemm");
    ok &= qc::compare(qc::d2h(dswiglu_out, swiglu_ref.size()), swiglu_ref,
                      storage_tol<T>())
              .report(label + " moe_swiglu");
    qc::dfree(din, dexperts, drect_out, dswiglu_out);
    free_fixture(drect);
    free_fixture(dswiglu);
    return ok;
}

bool run_correctness() {
    bool ok = true;
    for (int group_size : {32, 64, 128}) {
        for (int bits : {2, 3, 4, 5, 6, 8}) {
            for (int scale_type : {kScaleBF16, kScaleF16, kScaleE8M0}) {
                for (bool symmetric : {false, true}) {
                    const Fixture fixture =
                        make_fixture(bits, symmetric, scale_type, group_size);
                    ok &= run_dequant_case<float>(
                        fixture, "base_q_dequant b" + std::to_string(bits) +
                                     " g" + std::to_string(group_size));
                }
            }
        }
        for (bool symmetric : {false, true}) {
            const Fixture fixture =
                make_fixture(8, symmetric, kScaleE4M3, group_size);
            ok &= run_dequant_case<float>(
                fixture, "base_q_dequant e4m3 g" + std::to_string(group_size));
        }
    }
    ok &= run_dequant_case<__half>(make_fixture(4), "base_q_dequant fp16 out");
    ok &= run_dequant_case<__hip_bfloat16>(make_fixture(4),
                                           "base_q_dequant bf16 out");

    for (int bits : {2, 3, 4, 5, 6, 8}) {
        const Fixture fixture = make_fixture(bits);
        ok &= run_gemm_case<__half, float>(
            fixture, 1, "base_q_gemv fp16->fp32 b" + std::to_string(bits));
        ok &= run_gemm_case<__hip_bfloat16, float>(
            fixture, 7, "base_q_gemm bf16->fp32 b" + std::to_string(bits));
        ok &= run_gemm_case<__half, __half>(
            fixture, 7, "base_q_gemm fp16->fp16 b" + std::to_string(bits));
        ok &= run_gemm_case<__hip_bfloat16, __hip_bfloat16>(
            fixture, 7, "base_q_gemm bf16->bf16 b" + std::to_string(bits));
    }

    ok &= run_embedding_case<float>(make_fixture(5), "base_q_embedding fp32");
    ok &= run_qkv_case<float, float>("base_q_gemv_qkv fp32");
    ok &= run_swiglu_case<float, float>(make_fixture(5),
                                        "base_q_gemv_swiglu fp32");

    for (int bits : {2, 3, 4, 5, 6, 8}) {
        for (bool symmetric : {false, true}) {
            ok &= run_lm_head_case<float>(
                bits, symmetric, "base_q_lm_head_argmax fp32 b" +
                                     std::to_string(bits));
            ok &= run_lm_head_case<__half>(
                bits, symmetric, "base_q_lm_head_argmax fp16 b" +
                                     std::to_string(bits));
            ok &= run_lm_head_case<__hip_bfloat16>(
                bits, symmetric, "base_q_lm_head_argmax bf16 b" +
                                     std::to_string(bits));
            ok &= run_moe_cases<float>(
                bits, symmetric, "base_q grouped fp32 b" + std::to_string(bits));
        }
    }
    ok &= run_moe_cases<__half>(4, false, "base_q grouped fp16 typed");
    ok &= run_moe_cases<__hip_bfloat16>(4, false, "base_q grouped bf16 typed");
    return ok;
}

struct BenchStats {
    double median_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
    int warmups = 0;
    int iters = 0;
    int repeats = 0;

    double spread() const { return min_ms > 0.0 ? max_ms / min_ms : 0.0; }
    double gbps(double bytes) const { return bytes / (median_ms * 1e-3) / 1e9; }
    double tflops(double flops) const { return flops / (median_ms * 1e-3) / 1e12; }

    void report_bandwidth(const std::string &label, double bytes) const {
        std::printf("  %-40s %8.4f ms  %8.1f GB/s  (min %.4f max %.4f spread %.2fx, w%d/i%d/r%d)\n",
                    label.c_str(), median_ms, gbps(bytes), min_ms, max_ms,
                    spread(), warmups, iters, repeats);
    }
    void report_compute(const std::string &label, double flops) const {
        std::printf("  %-40s %8.4f ms  %8.1f TFLOP/s  (min %.4f max %.4f spread %.2fx, w%d/i%d/r%d)\n",
                    label.c_str(), median_ms, tflops(flops), min_ms, max_ms,
                    spread(), warmups, iters, repeats);
    }
};

template <typename F>
BenchStats bench_launch(F &&launch, int warmups = 5, int iters = 20,
                        int repeats = 100) {
    for (int i = 0; i < warmups; ++i)
        for (int r = 0; r < repeats; ++r) launch();
    QC_SYNC();

    hipEvent_t start, stop;
    QC_CHECK(hipEventCreate(&start));
    QC_CHECK(hipEventCreate(&stop));
    std::vector<float> samples(size_t(iters), 0.0f);
    for (int i = 0; i < iters; ++i) {
        QC_CHECK(hipEventRecord(start));
        for (int r = 0; r < repeats; ++r) launch();
        QC_CHECK(hipEventRecord(stop));
        QC_CHECK(hipEventSynchronize(stop));
        QC_CHECK(hipEventElapsedTime(&samples[size_t(i)], start, stop));
        samples[size_t(i)] /= float(repeats);
    }
    QC_CHECK(hipEventDestroy(start));
    QC_CHECK(hipEventDestroy(stop));
    std::sort(samples.begin(), samples.end());

    BenchStats stats;
    stats.median_ms = samples[size_t(iters / 2)];
    stats.min_ms = samples.front();
    stats.max_ms = samples.back();
    stats.warmups = warmups;
    stats.iters = iters;
    stats.repeats = repeats;
    return stats;
}

void report_pair(const char *label, const BenchStats &scalar,
                 const BenchStats &candidate, double ops) {
    scalar.report_compute(std::string(label) + " scalar", ops);
    candidate.report_compute(std::string(label) + " wave64", ops);
    const double speedup = scalar.median_ms / candidate.median_ms;
    std::printf("  %-40s baseline %.4f ms -> candidate %.4f ms  = %.2fx (%+.1f%%)\n",
                label, scalar.median_ms, candidate.median_ms, speedup,
                (speedup - 1.0) * 100.0);
}

void run_benchmarks() {
    std::printf("\n== BaseQ Phase 5 benchmarks ==\n");
    std::printf("   Format: BaseQ4 affine, BF16 scales/biases, group_size=64, fp16 input unless noted.\n");

    {
        Fixture fixture = make_fixture(4, false, kScaleBF16, 64, 4096, 1024);
        DeviceFixture d = upload_fixture(fixture);
        float *dout = qc::dzero<float>(fixture.decoded.size());
        const auto current = bench_launch([&] {
            base_q_dequant_kernel<<<qc::grid_for(fixture.decoded.size(), 256), 256>>>(
                d.view, dout, kStorageF32);
        });
        current.report_bandwidth("base_q_dequant current",
                                 double(fixture.codes.size() +
                                        fixture.groups() * fixture.scale_bytes() * 2 +
                                        fixture.decoded.size() * sizeof(float)));
        qc::dfree(dout);
        free_fixture(d);
    }

    Fixture proj = make_fixture(4, false, kScaleBF16, 64, 2048, 1024);
    DeviceFixture dproj = upload_fixture(proj);
    const auto input_src = make_input_values(size_t(32 * proj.columns), 13, 17.0f);
    const auto input_h = to_storage_vec<__half>(input_src);
    __half *din = qc::dnew(input_h);
    float *dout = qc::dzero<float>(size_t(32 * proj.rows));

    {
        const long long m = 1;
        const double ops = 2.0 * double(m) * double(proj.rows) * double(proj.columns);
        const auto scalar = bench_launch([&] {
            base_q_gemm_scalar_kernel<<<qc::grid_for(size_t(m * proj.rows), 128), 128>>>(
                dproj.view, din, kStorageF16, dout, kStorageF32, m);
        });
        const auto wave = bench_launch([&] {
            base_q_gemm_wave64_kernel<<<qc::wave_blocks(size_t(m * proj.rows)),
                                        qc::kThreads>>>(
                dproj.view, din, kStorageF16, dout, kStorageF32, m);
        });
        report_pair("base_q_gemv", scalar, wave, ops);
    }
    {
        const long long m = 32;
        const double ops = 2.0 * double(m) * double(proj.rows) * double(proj.columns);
        const auto scalar = bench_launch([&] {
            base_q_gemm_scalar_kernel<<<qc::grid_for(size_t(m * proj.rows), 128), 128>>>(
                dproj.view, din, kStorageF16, dout, kStorageF32, m);
        });
        const auto wave = bench_launch([&] {
            base_q_gemm_wave64_kernel<<<qc::wave_blocks(size_t(m * proj.rows)),
                                        qc::kThreads>>>(
                dproj.view, din, kStorageF16, dout, kStorageF32, m);
        });
        report_pair("base_q_gemm", scalar, wave, ops);
    }

    {
        const std::vector<int> ids(8192, 7);
        int *dids = qc::dnew(ids);
        float *demb = qc::dzero<float>(size_t(ids.size()) * size_t(proj.columns));
        const auto current = bench_launch([&] {
            base_q_embedding_kernel<<<qc::grid_for(size_t(ids.size()) *
                                                       size_t(proj.columns),
                                                   256),
                                      256>>>(dproj.view, dids, static_cast<long long>(ids.size()),
                                             demb, kStorageF32);
        });
        current.report_bandwidth("base_q_embedding current",
                                 double(ids.size()) * double(proj.columns) *
                                     (sizeof(uint8_t) + sizeof(float)));
        qc::dfree(dids, demb);
    }

    {
        Fixture q = make_fixture(4, false, kScaleBF16, 64, 768, 1024);
        Fixture k = make_fixture(4, false, kScaleBF16, 64, 256, 1024);
        Fixture v = make_fixture(4, false, kScaleBF16, 64, 256, 1024);
        DeviceFixture dq = upload_fixture(q);
        DeviceFixture dk = upload_fixture(k);
        DeviceFixture dv = upload_fixture(v);
        float *dqout = qc::dzero<float>(q.rows);
        float *dkout = qc::dzero<float>(k.rows);
        float *dvout = qc::dzero<float>(v.rows);
        const long long total = q.rows + k.rows + v.rows;
        const double ops = 2.0 * double(total) * double(q.columns);
        const auto scalar = bench_launch([&] {
            base_q_gemv_qkv_scalar_kernel<<<qc::grid_for(size_t(total), 128), 128>>>(
                dq.view, dk.view, dv.view, din, kStorageF16, dqout, kStorageF32,
                dkout, kStorageF32, dvout, kStorageF32);
        });
        const auto wave = bench_launch([&] {
            base_q_gemv_qkv_wave64_kernel<<<qc::wave_blocks(size_t(total)),
                                           qc::kThreads>>>(
                dq.view, dk.view, dv.view, din, kStorageF16, dqout, kStorageF32,
                dkout, kStorageF32, dvout, kStorageF32);
        });
        report_pair("base_q_gemv_qkv", scalar, wave, ops);
        qc::dfree(dqout, dkout, dvout);
        free_fixture(dq);
        free_fixture(dk);
        free_fixture(dv);
    }

    {
        float *dswiglu = qc::dzero<float>(proj.rows);
        const double ops = 4.0 * double(proj.rows) * double(proj.columns);
        const auto scalar = bench_launch([&] {
            base_q_gemv_swiglu_scalar_kernel<<<qc::grid_for(size_t(proj.rows), 128), 128>>>(
                dproj.view, dproj.view, din, kStorageF16, dswiglu, kStorageF32);
        });
        const auto wave = bench_launch([&] {
            base_q_gemv_swiglu_wave64_kernel<<<qc::wave_blocks(size_t(proj.rows)),
                                              qc::kThreads>>>(
                dproj.view, dproj.view, din, kStorageF16, dswiglu, kStorageF32);
        });
        report_pair("base_q_gemv_swiglu", scalar, wave, ops);
        qc::dfree(dswiglu);
    }

    {
        constexpr long long batch = 4;
        Fixture vocab = make_fixture(4, false, kScaleBF16, 64, 4096, 1024);
        DeviceFixture dvocab = upload_fixture(vocab);
        auto lm_src = make_input_values(size_t(batch * vocab.columns), 19, 23.0f);
        auto lm_h = to_storage_vec<__half>(lm_src);
        __half *dlm = qc::dnew(lm_h);
        __half *dscores = qc::dzero<__half>(size_t(batch * vocab.rows));
        int *dids = qc::dzero<int>(batch);
        const double ops = 2.0 * double(batch) * double(vocab.rows) * double(vocab.columns);
        const auto materialized = bench_launch([&] {
            base_q_gemm_wave64_kernel<<<qc::wave_blocks(size_t(batch * vocab.rows)),
                                        qc::kThreads>>>(
                dvocab.view, dlm, kStorageF16, dscores, kStorageF16, batch);
            base_q_materialized_argmax_kernel<<<batch, qc::kThreads>>>(
                dscores, kStorageF16, dids, batch, vocab.rows);
        });
        const auto streaming = bench_launch([&] {
            base_q_lm_head_argmax_kernel<<<batch, qc::kThreads>>>(
                dvocab.view, dlm, kStorageF16, dids, batch);
        });
        materialized.report_compute("base_q_lm_head materialized", ops);
        streaming.report_compute("base_q_lm_head streaming", ops);
        const double speedup = materialized.median_ms / streaming.median_ms;
        std::printf("  %-40s baseline %.4f ms -> candidate %.4f ms  = %.2fx (%+.1f%%)\n",
                    "base_q_lm_head_argmax", materialized.median_ms,
                    streaming.median_ms, speedup, (speedup - 1.0) * 100.0);
        qc::dfree(dlm, dscores, dids);
        free_fixture(dvocab);
    }

    {
        constexpr long long experts = 4;
        constexpr long long total_rows = 128;
        constexpr long long output_rows = 512;
        constexpr long long intermediate = 512;
        Fixture moe = make_fixture(4, false, kScaleBF16, 64,
                                   experts * output_rows, 1024);
        Fixture moe_swiglu = make_fixture(4, false, kScaleBF16, 64,
                                          experts * 2 * intermediate, 1024);
        DeviceFixture dmoe = upload_fixture(moe);
        DeviceFixture dmoe_swiglu = upload_fixture(moe_swiglu);
        const auto moe_src = make_input_values(size_t(total_rows * moe.columns), 5, 19.0f);
        const auto moe_h = to_storage_vec<__half>(moe_src);
        __half *dmoe_in = qc::dnew(moe_h);
        const std::vector<int> expert_of_tile = {0, 1, 2, 3};
        int *dexpert = qc::dnew(expert_of_tile);
        float *dmoe_out = qc::dzero<float>(size_t(total_rows * output_rows));
        float *dmoe_swiglu_out = qc::dzero<float>(size_t(total_rows * intermediate));
        const double moe_ops =
            2.0 * double(total_rows) * double(output_rows) * double(moe.columns);
        const auto moe_scalar = bench_launch([&] {
            base_q_moe_gemm_scalar_kernel<<<qc::grid_for(size_t(total_rows * output_rows), 128), 128>>>(
                dmoe.view, experts, dmoe_in, kStorageF16, dexpert, total_rows,
                dmoe_out, kStorageF32);
        });
        const auto moe_wave = bench_launch([&] {
            base_q_moe_gemm_wave64_kernel<<<qc::wave_blocks(size_t(total_rows * output_rows)),
                                           qc::kThreads>>>(
                dmoe.view, experts, dmoe_in, kStorageF16, dexpert, total_rows,
                dmoe_out, kStorageF32);
        });
        report_pair("base_q_moe_gemm", moe_scalar, moe_wave, moe_ops);

        const double swiglu_ops =
            4.0 * double(total_rows) * double(intermediate) * double(moe.columns);
        const auto swiglu_scalar = bench_launch([&] {
            base_q_moe_swiglu_scalar_kernel<<<qc::grid_for(size_t(total_rows * intermediate), 128), 128>>>(
                dmoe_swiglu.view, experts, dmoe_in, kStorageF16, dexpert,
                total_rows, dmoe_swiglu_out, kStorageF32);
        });
        const auto swiglu_wave = bench_launch([&] {
            base_q_moe_swiglu_wave64_kernel<<<qc::wave_blocks(size_t(total_rows * intermediate)),
                                             qc::kThreads>>>(
                dmoe_swiglu.view, experts, dmoe_in, kStorageF16, dexpert,
                total_rows, dmoe_swiglu_out, kStorageF32);
        });
        report_pair("base_q_moe_swiglu", swiglu_scalar, swiglu_wave, swiglu_ops);
        qc::dfree(dmoe_in, dexpert, dmoe_out, dmoe_swiglu_out);
        free_fixture(dmoe);
        free_fixture(dmoe_swiglu);
    }

    qc::dfree(din, dout);
    free_fixture(dproj);
}

}  // namespace

int main(int argc, char **argv) {
    qc::print_environment("base_q");
    const bool ok = run_correctness();
    if (qc::bench_requested(argc, argv)) run_benchmarks();
    return qc::finish(ok);
}
