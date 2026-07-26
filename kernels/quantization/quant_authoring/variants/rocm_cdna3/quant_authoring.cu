/**
 * @file
 * @brief Phase 6 CPU/Metal parity ports for quant authoring and embedding.
 */
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "../../../../common/cdna3_harness.cuh"
#include "../../../qgemv/variants/rocm_cdna3/quant_formats.cuh"
#include "../../../qgemv/variants/rocm_cdna3/quant_formats_tables.cuh"

namespace {

using tmq::q4_0;
using tmq::q6_K;
using tmq::q8_0;

enum Float8Kind : int {
    kE4M3FN = 0,
    kE5M2 = 1,
};

enum PackedFormat : int {
    kFmtQ4_0 = 0,
    kFmtQ8_0 = 1,
    kFmtQ6_K = 2,
};

__host__ __device__ __forceinline__ uint32_t f32_bits(float v) {
#ifdef __HIP_DEVICE_COMPILE__
    return __float_as_uint(v);
#else
    uint32_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    return bits;
#endif
}

__host__ __device__ __forceinline__ uint32_t round_right_even_u32(uint32_t value,
                                                                  int shift) {
    if (shift <= 0) return value << -shift;
    if (shift > 31) return 0;
    const uint32_t truncated = value >> shift;
    const uint32_t remainder = value & ((uint32_t{1} << shift) - 1U);
    const uint32_t halfway = uint32_t{1} << (shift - 1);
    return truncated +
           (remainder > halfway ||
            (remainder == halfway && (truncated & 1U) != 0)
                ? 1U
                : 0U);
}

__host__ __device__ __forceinline__ float fp8_max_value(int kind) {
    return kind == kE4M3FN ? 448.0f : 57344.0f;
}

__host__ __device__ __forceinline__ uint8_t fp8_max_finite(int kind) {
    return kind == kE4M3FN ? 0x7e : 0x7b;
}

__host__ __device__ __forceinline__ float fp8_positive(uint8_t code, int kind) {
    if (kind == kE4M3FN) {
        const int exponent = (code >> 3) & 15;
        const int mantissa = code & 7;
        if (exponent == 15 && mantissa == 7) {
            return NAN;
        }
        if (exponent == 0) return ldexpf(float(mantissa), -9);
        return ldexpf(1.0f + float(mantissa) * 0.125f, exponent - 7);
    }
    const int exponent = (code >> 2) & 31;
    const int mantissa = code & 3;
    if (exponent == 31) {
        return mantissa == 0 ? INFINITY : NAN;
    }
    if (exponent == 0) return ldexpf(float(mantissa), -16);
    return ldexpf(1.0f + float(mantissa) * 0.25f, exponent - 15);
}

__host__ __device__ __forceinline__ float fp8_decode_ref(uint8_t code, int kind) {
    const float value = fp8_positive(code & 0x7f, kind);
    return (code & 0x80) ? -value : value;
}

__host__ __device__ __forceinline__ uint8_t fp8_encode_ref(float value,
                                                           int kind) {
    const uint32_t raw = f32_bits(value);
    if ((raw & 0x7fffffffU) > 0x7f800000U) {
        return kind == kE4M3FN ? 0x7f : 0x7d;
    }
    const bool negative = (raw >> 31) != 0;
    const float magnitude = fabsf(value);
    if (magnitude == 0.0f) return negative ? 0x80 : 0;
    if ((raw & 0x7fffffffU) >= 0x7f800000U ||
        magnitude >= fp8_max_value(kind)) {
        return fp8_max_finite(kind) | (negative ? 0x80 : 0);
    }
    const int source_exponent = int((raw >> 23) & 0xffU);
    if (source_exponent == 0) return negative ? 0x80 : 0;
    int unbiased_exponent = source_exponent - 127;
    const uint32_t significand = (uint32_t{1} << 23) | (raw & 0x7fffffU);
    const int mantissa_bits = kind == kE4M3FN ? 3 : 2;
    const int bias = kind == kE4M3FN ? 7 : 15;
    uint32_t code = 0;
    if (unbiased_exponent < 1 - bias) {
        const int shift = 24 - unbiased_exponent - bias - mantissa_bits;
        code = round_right_even_u32(significand, shift);
    } else {
        uint32_t rounded = round_right_even_u32(significand, 23 - mantissa_bits);
        if (rounded == (uint32_t{1} << (mantissa_bits + 1))) {
            rounded >>= 1;
            ++unbiased_exponent;
        }
        const int encoded_exponent = unbiased_exponent + bias;
        code = (uint32_t(encoded_exponent) << mantissa_bits) |
               (rounded - (uint32_t{1} << mantissa_bits));
    }
    code = min(code, uint32_t(fp8_max_finite(kind)));
    return uint8_t(code | (negative ? 0x80U : 0U));
}

__host__ __device__ __forceinline__ int round_nearest_even_int(float value) {
    return int(rintf(value));
}

__host__ __device__ __forceinline__ int round_half_away_int(float value) {
    return value < 0.0f ? -int(floorf(-value + 0.5f))
                        : int(floorf(value + 0.5f));
}

__host__ __device__ __forceinline__ float fp4_e2m1_decode_ref(uint8_t code) {
    const float table[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float magnitude = table[code & 7U];
    return (code & 8U) ? -magnitude : magnitude;
}

__host__ __device__ __forceinline__ float e8m0_decode_ref(uint8_t code) {
    return code == 0 ? ldexpf(1.0f, -127) : ldexpf(1.0f, int(code) - 127);
}

uint16_t half_bits(float value) {
    const __half h = __float2half(value);
    uint16_t bits = 0;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}

float half_at_host(const uint8_t *bytes) {
    __half h;
    std::memcpy(&h, bytes, sizeof(h));
    return __half2float(h);
}

void put_half_host(uint8_t *bytes, float value) {
    const uint16_t bits = half_bits(value);
    std::memcpy(bytes, &bits, sizeof(bits));
}

std::vector<double> to_ref(const std::vector<float> &values) {
    return std::vector<double>(values.begin(), values.end());
}

std::vector<double> to_ref_u8(const std::vector<uint8_t> &values) {
    std::vector<double> out(values.size());
    for (size_t i = 0; i < values.size(); ++i) out[i] = double(values[i]);
    return out;
}

std::vector<double> to_ref_i8(const std::vector<int8_t> &values) {
    std::vector<double> out(values.size());
    for (size_t i = 0; i < values.size(); ++i) out[i] = double(values[i]);
    return out;
}

std::vector<double> to_ref_u32(const std::vector<uint32_t> &values) {
    std::vector<double> out(values.size());
    for (size_t i = 0; i < values.size(); ++i) out[i] = double(values[i]);
    return out;
}

// ---------------------------------------------------------------------------
// Fake quantization
// ---------------------------------------------------------------------------

__global__ void fake_quant_int8_scalar_kernel(const float *x, float *out,
                                              int8_t *codes, float *scales,
                                              long long rows, long long dim) {
    const long long row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    float maximum = 0.0f;
    for (long long col = 0; col < dim; ++col) {
        maximum = fmaxf(maximum, fabsf(x[row * dim + col]));
    }
    const float scale = maximum / 127.0f;
    const float inverse = scale > 0.0f ? 1.0f / scale : 0.0f;
    scales[row] = scale;
    for (long long col = 0; col < dim; ++col) {
        const long long index = row * dim + col;
        const int code =
            max(-127, min(127, round_nearest_even_int(x[index] * inverse)));
        codes[index] = int8_t(code);
        out[index] = scale * float(code);
    }
}

__global__ void fake_quant_int8_wave_kernel(const float *x, float *out,
                                            int8_t *codes, float *scales,
                                            long long rows, long long dim) {
    const int lane = threadIdx.x & (qc::kWave - 1);
    const int wave = threadIdx.x >> 6;
    const long long row =
        (static_cast<long long>(blockIdx.x) * (blockDim.x / qc::kWave)) + wave;
    if (row >= rows) return;
    float maximum = 0.0f;
    for (long long col = lane; col < dim; col += qc::kWave) {
        maximum = fmaxf(maximum, fabsf(x[row * dim + col]));
    }
    maximum = qc::wave_reduce_max(maximum);
    const float scale = maximum / 127.0f;
    const float inverse = scale > 0.0f ? 1.0f / scale : 0.0f;
    if (lane == 0) scales[row] = scale;
    for (long long col = lane; col < dim; col += qc::kWave) {
        const long long index = row * dim + col;
        const int code =
            max(-127, min(127, round_nearest_even_int(x[index] * inverse)));
        codes[index] = int8_t(code);
        out[index] = scale * float(code);
    }
}

__global__ void fake_quant_float8_scalar_kernel(const float *x, float *out,
                                                uint8_t *codes, float *scale,
                                                long long count, int kind) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float maximum = 0.0f;
    for (long long i = 0; i < count; ++i) maximum = fmaxf(maximum, fabsf(x[i]));
    const float s = maximum / fp8_max_value(kind);
    const float inverse = s > 0.0f ? 1.0f / s : 0.0f;
    *scale = s;
    for (long long i = 0; i < count; ++i) {
        const uint8_t code = fp8_encode_ref(x[i] * inverse, kind);
        codes[i] = code;
        out[i] = s * fp8_decode_ref(code, kind);
    }
}

__global__ void fake_quant_float8_wave_kernel(const float *x, float *out,
                                              uint8_t *codes, float *scale,
                                              long long count, int kind) {
    const int lane = threadIdx.x & (qc::kWave - 1);
    float maximum = 0.0f;
    for (long long i = lane; i < count; i += qc::kWave) {
        maximum = fmaxf(maximum, fabsf(x[i]));
    }
    maximum = qc::wave_reduce_max(maximum);
    const float s = maximum / fp8_max_value(kind);
    const float inverse = s > 0.0f ? 1.0f / s : 0.0f;
    if (lane == 0) *scale = s;
    for (long long i = lane; i < count; i += qc::kWave) {
        const uint8_t code = fp8_encode_ref(x[i] * inverse, kind);
        codes[i] = code;
        out[i] = s * fp8_decode_ref(code, kind);
    }
}

void fake_quant_int8_ref(const std::vector<float> &x, std::vector<float> &out,
                         std::vector<int8_t> &codes, std::vector<float> &scales,
                         long long rows, long long dim) {
    out.assign(x.size(), 0.0f);
    codes.assign(x.size(), 0);
    scales.assign(size_t(rows), 0.0f);
    for (long long row = 0; row < rows; ++row) {
        float maximum = 0.0f;
        for (long long col = 0; col < dim; ++col) {
            maximum = std::max(maximum, std::fabs(x[row * dim + col]));
        }
        const float scale = maximum / 127.0f;
        const float inverse = scale > 0.0f ? 1.0f / scale : 0.0f;
        scales[row] = scale;
        for (long long col = 0; col < dim; ++col) {
            const long long index = row * dim + col;
            const int code = std::max(
                -127, std::min(127, int(std::nearbyint(x[index] * inverse))));
            codes[index] = int8_t(code);
            out[index] = scale * float(code);
        }
    }
}

void fake_quant_float8_ref(const std::vector<float> &x, std::vector<float> &out,
                           std::vector<uint8_t> &codes, float &scale,
                           int kind) {
    out.assign(x.size(), 0.0f);
    codes.assign(x.size(), 0);
    float maximum = 0.0f;
    for (float v : x) maximum = std::max(maximum, std::fabs(v));
    scale = maximum / fp8_max_value(kind);
    const float inverse = scale > 0.0f ? 1.0f / scale : 0.0f;
    for (size_t i = 0; i < x.size(); ++i) {
        codes[i] = fp8_encode_ref(x[i] * inverse, kind);
        out[i] = scale * fp8_decode_ref(codes[i], kind);
    }
}

// ---------------------------------------------------------------------------
// Ternary and TQ2_0 pack formats
// ---------------------------------------------------------------------------

__device__ __forceinline__ float half_at_device(const uint8_t *bytes) {
    return __half2float(*reinterpret_cast<const __half *>(bytes));
}

__device__ __forceinline__ void put_half_device(uint8_t *bytes, float value) {
    *reinterpret_cast<__half *>(bytes) = __float2half(value);
}

__global__ void ternary_pack_scalar_kernel(const float *weights, uint8_t *packed,
                                           float *dequantized, long long rows,
                                           long long k, long long group_k) {
    const long long group_index = blockIdx.x * blockDim.x + threadIdx.x;
    const long long groups_per_row = k / group_k;
    const long long total_groups = rows * groups_per_row;
    if (group_index >= total_groups) return;
    const long long row = group_index / groups_per_row;
    const long long group = group_index - row * groups_per_row;
    const long long group_base = row * k + group * group_k;
    double sum = 0.0;
    for (long long i = 0; i < group_k; ++i) sum += fabsf(weights[group_base + i]);
    const float scale = fmaxf(float(sum / double(group_k)), 1e-5f);
    const float rounded_scale = __half2float(__float2half(scale));
    for (long long block = 0; block < group_k / 32; ++block) {
        uint8_t *dst = packed + ((row * (k / 32)) + group * (group_k / 32) +
                                 block) *
                                    10;
        put_half_device(dst, scale);
        for (int byte = 0; byte < 8; ++byte) {
            uint8_t code = 0;
            for (int lane = 0; lane < 4; ++lane) {
                const long long local = block * 32 + byte * 4 + lane;
                const int q = max(
                    -1, min(1, round_nearest_even_int(weights[group_base + local] /
                                                       scale)));
                code |= uint8_t((q + 1) << (2 * lane));
                dequantized[group_base + local] = rounded_scale * float(q);
            }
            dst[2 + byte] = code;
        }
    }
}

__global__ void ternary_pack_group_kernel(const float *weights, uint8_t *packed,
                                          float *dequantized, long long rows,
                                          long long k, long long group_k) {
    const long long groups_per_row = k / group_k;
    const long long group_index = blockIdx.x;
    if (group_index >= rows * groups_per_row) return;
    const int tid = threadIdx.x;
    const long long row = group_index / groups_per_row;
    const long long group = group_index - row * groups_per_row;
    const long long group_base = row * k + group * group_k;
    __shared__ float partial[256];
    float sum = 0.0f;
    for (long long i = tid; i < group_k; i += blockDim.x) {
        sum += fabsf(weights[group_base + i]);
    }
    partial[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    const float scale = fmaxf(partial[0] / float(group_k), 1e-5f);
    const float rounded_scale = __half2float(__float2half(scale));
    const long long block_count = group_k / 32;
    for (long long item = tid; item < block_count * 8; item += blockDim.x) {
        const long long block = item / 8;
        const int byte = int(item - block * 8);
        uint8_t *dst = packed + ((row * (k / 32)) + group * block_count + block) *
                                    10;
        put_half_device(dst, scale);
        uint8_t code = 0;
        for (int lane = 0; lane < 4; ++lane) {
            const long long local = block * 32 + byte * 4 + lane;
            const int q = max(
                -1, min(1, round_nearest_even_int(weights[group_base + local] /
                                                   scale)));
            code |= uint8_t((q + 1) << (2 * lane));
            dequantized[group_base + local] = rounded_scale * float(q);
        }
        dst[2 + byte] = code;
    }
}

__global__ void ternary_unpack_kernel(const uint8_t *packed, float *weights,
                                      long long rows, long long k) {
    const long long index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= rows * k) return;
    const long long row = index / k;
    const long long col = index - row * k;
    const long long block = col / 32;
    const int local = int(col & 31);
    const uint8_t *src = packed + (row * (k / 32) + block) * 10;
    const float scale = half_at_device(src);
    const int code = (src[2 + local / 4] >> (2 * (local % 4))) & 3;
    weights[index] = scale * float(code - 1);
}

__global__ void ternary_stats_kernel(const uint8_t *packed, uint32_t *counts,
                                     long long rows, long long k) {
    const long long row = blockIdx.x;
    if (row >= rows) return;
    const int tid = threadIdx.x;
    __shared__ uint32_t local[3];
    if (tid < 3) local[tid] = 0;
    __syncthreads();
    const long long bytes_per_row = (k / 32) * 8;
    for (long long byte = tid; byte < bytes_per_row; byte += blockDim.x) {
        const long long block = byte / 8;
        const int b = int(byte - block * 8);
        const uint8_t value = packed[(row * (k / 32) + block) * 10 + 2 + b];
        for (int lane = 0; lane < 4; ++lane) {
            const int code = (value >> (2 * lane)) & 3;
            if (code < 3) atomicAdd(local + code, 1U);
        }
    }
    __syncthreads();
    if (tid < 3) counts[row * 3 + tid] = local[tid];
}

__global__ void ternary_flip_count_kernel(const uint8_t *a, const uint8_t *b,
                                          uint32_t *flips, long long rows,
                                          long long k) {
    const long long row = blockIdx.x;
    if (row >= rows) return;
    const int tid = threadIdx.x;
    __shared__ uint32_t partial[256];
    uint32_t count = 0;
    const long long bytes_per_row = (k / 32) * 8;
    for (long long byte = tid; byte < bytes_per_row; byte += blockDim.x) {
        const long long block = byte / 8;
        const int bidx = int(byte - block * 8);
        const long long base = (row * (k / 32) + block) * 10 + 2 + bidx;
        const uint8_t difference = a[base] ^ b[base];
        for (int lane = 0; lane < 4; ++lane) {
            count += ((difference >> (2 * lane)) & 3) != 0;
        }
    }
    partial[tid] = count;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    if (tid == 0) flips[row] = partial[0];
}

__global__ void tq2_0_pack_scalar_kernel(const float *weights, uint8_t *packed,
                                         float *dequantized, long long rows,
                                         long long k) {
    const long long block_index = blockIdx.x * blockDim.x + threadIdx.x;
    const long long blocks_per_row = k / 256;
    const long long total_blocks = rows * blocks_per_row;
    if (block_index >= total_blocks) return;
    const long long row = block_index / blocks_per_row;
    const long long block = block_index - row * blocks_per_row;
    const long long base = row * k + block * 256;
    float maximum = 0.0f;
    for (int i = 0; i < 256; ++i) maximum = fmaxf(maximum, fabsf(weights[base + i]));
    const float inverse = maximum > 0.0f ? 1.0f / maximum : 0.0f;
    const float rounded_scale = __half2float(__float2half(maximum));
    uint8_t *dst = packed + block_index * 66;
    for (int half = 0; half < 2; ++half) {
        for (int lane = 0; lane < 32; ++lane) {
            uint8_t byte = 0;
            for (int group = 0; group < 4; ++group) {
                const int local = 128 * half + 32 * group + lane;
                const int q = round_half_away_int(weights[base + local] * inverse);
                byte |= uint8_t((q + 1) << (2 * group));
                dequantized[base + local] = rounded_scale * float(q);
            }
            dst[32 * half + lane] = byte;
        }
    }
    put_half_device(dst + 64, maximum);
}

__global__ void tq2_0_pack_block_kernel(const float *weights, uint8_t *packed,
                                        float *dequantized, long long rows,
                                        long long k) {
    const long long blocks_per_row = k / 256;
    const long long block_index = blockIdx.x;
    if (block_index >= rows * blocks_per_row) return;
    const int tid = threadIdx.x;
    const long long row = block_index / blocks_per_row;
    const long long block = block_index - row * blocks_per_row;
    const long long base = row * k + block * 256;
    __shared__ float partial[256];
    float maximum = 0.0f;
    for (int i = tid; i < 256; i += blockDim.x) {
        maximum = fmaxf(maximum, fabsf(weights[base + i]));
    }
    partial[tid] = maximum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] = fmaxf(partial[tid], partial[tid + stride]);
        __syncthreads();
    }
    const float scale = partial[0];
    const float inverse = scale > 0.0f ? 1.0f / scale : 0.0f;
    const float rounded_scale = __half2float(__float2half(scale));
    uint8_t *dst = packed + block_index * 66;
    if (tid < 64) {
        const int half = tid / 32;
        const int lane = tid & 31;
        uint8_t byte = 0;
        for (int group = 0; group < 4; ++group) {
            const int local = 128 * half + 32 * group + lane;
            const int q = round_half_away_int(weights[base + local] * inverse);
            byte |= uint8_t((q + 1) << (2 * group));
            dequantized[base + local] = rounded_scale * float(q);
        }
        dst[tid] = byte;
    }
    if (tid == 0) put_half_device(dst + 64, scale);
}

__global__ void tq2_0_unpack_kernel(const uint8_t *packed, float *weights,
                                    long long rows, long long k) {
    const long long index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= rows * k) return;
    const long long row = index / k;
    const long long col = index - row * k;
    const long long block = col / 256;
    const int i = int(col & 255);
    const uint8_t *src = packed + (row * (k / 256) + block) * 66;
    const float scale = half_at_device(src + 64);
    const int half = i / 128;
    const int rest = i & 127;
    const int group = rest / 32;
    const int lane = rest & 31;
    const int code = (src[32 * half + lane] >> (2 * group)) & 3;
    weights[index] = scale * float(code - 1);
}

void ternary_pack_ref(const std::vector<float> &weights, std::vector<uint8_t> &packed,
                      std::vector<float> &dequantized, long long rows,
                      long long k, long long group_k) {
    packed.assign(size_t(rows * (k / 32) * 10), 0);
    dequantized.assign(weights.size(), 0.0f);
    for (long long row = 0; row < rows; ++row) {
        for (long long group = 0; group < k / group_k; ++group) {
            const long long group_base = row * k + group * group_k;
            double sum = 0.0;
            for (long long i = 0; i < group_k; ++i) {
                sum += std::fabs(weights[group_base + i]);
            }
            const float scale = std::max(float(sum / double(group_k)), 1e-5f);
            const float rounded_scale = __half2float(__float2half(scale));
            for (long long block = 0; block < group_k / 32; ++block) {
                uint8_t *dst = packed.data() +
                               ((row * (k / 32)) + group * (group_k / 32) +
                                block) *
                                   10;
                put_half_host(dst, scale);
                for (int byte = 0; byte < 8; ++byte) {
                    uint8_t code = 0;
                    for (int lane = 0; lane < 4; ++lane) {
                        const long long local = block * 32 + byte * 4 + lane;
                        const int q = std::max(
                            -1, std::min(1, int(std::nearbyint(
                                             weights[group_base + local] / scale))));
                        code |= uint8_t((q + 1) << (2 * lane));
                        dequantized[group_base + local] = rounded_scale * float(q);
                    }
                    dst[2 + byte] = code;
                }
            }
        }
    }
}

void ternary_unpack_ref(const std::vector<uint8_t> &packed,
                        std::vector<float> &weights, long long rows,
                        long long k) {
    weights.assign(size_t(rows * k), 0.0f);
    for (long long row = 0; row < rows; ++row) {
        for (long long block = 0; block < k / 32; ++block) {
            const uint8_t *src = packed.data() + (row * (k / 32) + block) * 10;
            const float scale = half_at_host(src);
            for (int i = 0; i < 32; ++i) {
                const int code = (src[2 + i / 4] >> (2 * (i % 4))) & 3;
                weights[row * k + block * 32 + i] = scale * float(code - 1);
            }
        }
    }
}

std::vector<uint32_t> ternary_stats_ref(const std::vector<uint8_t> &packed,
                                        long long rows, long long k) {
    std::vector<uint32_t> counts(size_t(rows * 3), 0);
    for (long long row = 0; row < rows; ++row) {
        for (long long block = 0; block < k / 32; ++block) {
            const uint8_t *src = packed.data() + (row * (k / 32) + block) * 10 + 2;
            for (int i = 0; i < 32; ++i) {
                const int code = (src[i / 4] >> (2 * (i % 4))) & 3;
                if (code < 3) ++counts[row * 3 + code];
            }
        }
    }
    return counts;
}

std::vector<uint32_t> ternary_flips_ref(const std::vector<uint8_t> &a,
                                        const std::vector<uint8_t> &b,
                                        long long rows, long long k) {
    std::vector<uint32_t> flips(size_t(rows), 0);
    for (long long row = 0; row < rows; ++row) {
        uint32_t count = 0;
        for (long long block = 0; block < k / 32; ++block) {
            const long long base = (row * (k / 32) + block) * 10 + 2;
            for (int byte = 0; byte < 8; ++byte) {
                const uint8_t difference = a[base + byte] ^ b[base + byte];
                for (int lane = 0; lane < 4; ++lane) {
                    count += ((difference >> (2 * lane)) & 3) != 0;
                }
            }
        }
        flips[row] = count;
    }
    return flips;
}

void tq2_0_pack_ref(const std::vector<float> &weights, std::vector<uint8_t> &packed,
                    std::vector<float> &dequantized, long long rows,
                    long long k) {
    packed.assign(size_t(rows * (k / 256) * 66), 0);
    dequantized.assign(weights.size(), 0.0f);
    for (long long row = 0; row < rows; ++row) {
        for (long long block = 0; block < k / 256; ++block) {
            const long long base = row * k + block * 256;
            float maximum = 0.0f;
            for (int i = 0; i < 256; ++i) {
                maximum = std::max(maximum, std::fabs(weights[base + i]));
            }
            const float inverse = maximum > 0.0f ? 1.0f / maximum : 0.0f;
            const float rounded_scale = __half2float(__float2half(maximum));
            uint8_t *dst = packed.data() + (row * (k / 256) + block) * 66;
            for (int half = 0; half < 2; ++half) {
                for (int lane = 0; lane < 32; ++lane) {
                    uint8_t byte = 0;
                    for (int group = 0; group < 4; ++group) {
                        const int local = 128 * half + 32 * group + lane;
                        const int q =
                            weights[base + local] * inverse < 0.0f
                                ? -int(std::floor(-weights[base + local] *
                                                       inverse +
                                                   0.5f))
                                : int(std::floor(weights[base + local] *
                                                     inverse +
                                                 0.5f));
                        byte |= uint8_t((q + 1) << (2 * group));
                        dequantized[base + local] = rounded_scale * float(q);
                    }
                    dst[32 * half + lane] = byte;
                }
            }
            put_half_host(dst + 64, maximum);
        }
    }
}

void tq2_0_unpack_ref(const std::vector<uint8_t> &packed,
                      std::vector<float> &weights, long long rows,
                      long long k) {
    weights.assign(size_t(rows * k), 0.0f);
    for (long long row = 0; row < rows; ++row) {
        for (long long block = 0; block < k / 256; ++block) {
            const uint8_t *src = packed.data() + (row * (k / 256) + block) * 66;
            const float scale = half_at_host(src + 64);
            for (int i = 0; i < 256; ++i) {
                const int half = i / 128;
                const int rest = i % 128;
                const int group = rest / 32;
                const int lane = rest % 32;
                const int code = (src[32 * half + lane] >> (2 * group)) & 3;
                weights[row * k + block * 256 + i] = scale * float(code - 1);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Calibration
// ---------------------------------------------------------------------------

__global__ void calibration_absmax_scalar_kernel(const float *x, const float *running,
                                                 float *out, long long tokens,
                                                 long long channels,
                                                 int has_running) {
    const long long channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= channels) return;
    float maximum = has_running ? running[channel] : 0.0f;
    if (!isnan(maximum)) {
        maximum = fabsf(maximum);
        for (long long token = 0; token < tokens; ++token) {
            const float value = fabsf(x[token * channels + channel]);
            if (isnan(value)) {
                maximum = value;
                break;
            }
            maximum = fmaxf(maximum, value);
        }
    }
    out[channel] = maximum;
}

__global__ void calibration_absmax_block_kernel(const float *x, const float *running,
                                                float *out, long long tokens,
                                                long long channels,
                                                int has_running) {
    const long long channel = blockIdx.x;
    if (channel >= channels) return;
    const int tid = threadIdx.x;
    __shared__ float partial[256];
    __shared__ int saw_nan;
    if (tid == 0) saw_nan = 0;
    __syncthreads();
    float maximum = 0.0f;
    for (long long token = tid; token < tokens; token += blockDim.x) {
        const float value = fabsf(x[token * channels + channel]);
        if (isnan(value)) {
            saw_nan = 1;
        } else {
            maximum = fmaxf(maximum, value);
        }
    }
    partial[tid] = maximum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] = fmaxf(partial[tid], partial[tid + stride]);
        __syncthreads();
    }
    if (tid == 0) {
        float value = has_running ? running[channel] : 0.0f;
        if (isnan(value) || saw_nan) {
            out[channel] = NAN;
        } else {
            out[channel] = fmaxf(fabsf(value), partial[0]);
        }
    }
}

std::vector<float> calibration_absmax_ref(const std::vector<float> &x,
                                          const std::vector<float> *running,
                                          long long tokens,
                                          long long channels) {
    std::vector<float> out(size_t(channels), 0.0f);
    for (long long channel = 0; channel < channels; ++channel) {
        float maximum = running ? (*running)[channel] : 0.0f;
        if (!std::isnan(maximum)) {
            maximum = std::fabs(maximum);
            for (long long token = 0; token < tokens; ++token) {
                const float value = std::fabs(x[token * channels + channel]);
                if (std::isnan(value)) {
                    maximum = value;
                    break;
                }
                maximum = std::max(maximum, value);
            }
        }
        out[channel] = maximum;
    }
    return out;
}

bool compare_nan_equal(const std::vector<float> &got, const std::vector<float> &ref,
                       const char *label) {
    size_t bad = 0;
    double max_abs = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        const bool gnan = std::isnan(got[i]);
        const bool rnan = std::isnan(ref[i]);
        if (gnan || rnan) {
            bad += gnan != rnan;
            continue;
        }
        const double d = std::fabs(double(got[i]) - double(ref[i]));
        max_abs = std::max(max_abs, d);
        bad += d > 1e-6;
    }
    std::printf("  %-52s [nan-exact] n=%zu bad=%zu max=%.3e  %s\n", label,
                got.size(), bad, max_abs, bad == 0 ? "PASS" : "FAIL");
    return bad == 0;
}

// ---------------------------------------------------------------------------
// Packed-table gather and embedding
// ---------------------------------------------------------------------------

int block_k_for(int fmt) {
    if (fmt == kFmtQ4_0) return 32;
    if (fmt == kFmtQ8_0) return 32;
    return 256;
}

int block_bytes_for(int fmt) {
    if (fmt == kFmtQ4_0) return 18;
    if (fmt == kFmtQ8_0) return 34;
    return 210;
}

void fill_q8_0_table(std::vector<uint8_t> &table, long long rows,
                     long long columns) {
    const long long blocks = columns / 32;
    table.assign(size_t(rows * blocks * 34), 0);
    for (long long row = 0; row < rows; ++row) {
        for (long long block = 0; block < blocks; ++block) {
            uint8_t *base = table.data() + (row * blocks + block) * 34;
            put_half_host(base, 0.015625f * float(1 + ((row + block) & 3)));
            for (int col = 0; col < 32; ++col) {
                reinterpret_cast<int8_t *>(base + 2)[col] =
                    int8_t(((row * 17 + block * 11 + col * 5) % 255) - 127);
            }
        }
    }
}

void fill_q4_0_table(std::vector<uint8_t> &table, long long rows,
                     long long columns) {
    const long long blocks = columns / 32;
    table.assign(size_t(rows * blocks * 18), 0);
    for (long long row = 0; row < rows; ++row) {
        for (long long block = 0; block < blocks; ++block) {
            uint8_t *base = table.data() + (row * blocks + block) * 18;
            put_half_host(base, 0.03125f * float(1 + ((row + block) & 1)));
            uint8_t *qs = base + 2;
            for (int col = 0; col < 32; ++col) {
                const uint8_t nib =
                    uint8_t((row * 7 + block * 13 + col * 3) & 0x0f);
                if (col < 16) {
                    qs[col] = uint8_t((qs[col] & 0xf0U) | nib);
                } else {
                    qs[col - 16] = uint8_t((qs[col - 16] & 0x0fU) | (nib << 4));
                }
            }
        }
    }
}

void fill_q6_K_table(std::vector<uint8_t> &table, long long rows,
                     long long columns) {
    const long long blocks = columns / 256;
    table.assign(size_t(rows * blocks * 210), 0);
    for (long long row = 0; row < rows; ++row) {
        for (long long block = 0; block < blocks; ++block) {
            uint8_t *base = table.data() + (row * blocks + block) * 210;
            put_half_host(base + 208, 0.0078125f * float(1 + ((row + block) & 3)));
            int8_t *scales = reinterpret_cast<int8_t *>(base + 192);
            for (int i = 0; i < 16; ++i) {
                scales[i] = int8_t(1 + ((row + block + i) % 5));
            }
            uint8_t *ql = base;
            uint8_t *qh = base + 128;
            for (int col = 0; col < 256; ++col) {
                const int chunk = col >> 7;
                const int pos = col & 127;
                const int group = pos >> 5;
                const int lane = pos & 31;
                const int encoded = (row * 19 + block * 23 + col * 7) & 63;
                const int nib = encoded & 15;
                const int high = (encoded >> 4) & 3;
                uint8_t &ql_byte = ql[chunk * 64 + lane + 32 * (group & 1)];
                if (group & 2) {
                    ql_byte = uint8_t((ql_byte & 0x0fU) | (nib << 4));
                } else {
                    ql_byte = uint8_t((ql_byte & 0xf0U) | nib);
                }
                qh[chunk * 32 + lane] =
                    uint8_t(qh[chunk * 32 + lane] | (high << (2 * group)));
            }
        }
    }
}

float dequant_host(const std::vector<uint8_t> &table, int fmt, long long row,
                   long long columns, long long column) {
    const int block_k = block_k_for(fmt);
    const int block_bytes = block_bytes_for(fmt);
    const long long blocks = columns / block_k;
    const uint8_t *base =
        table.data() + (row * blocks + column / block_k) * block_bytes;
    const int local = int(column % block_k);
    if (fmt == kFmtQ8_0) {
        return half_at_host(base) * float(reinterpret_cast<const int8_t *>(base + 2)[local]);
    }
    if (fmt == kFmtQ4_0) {
        const uint8_t *qs = base + 2;
        const int nib = local < 16 ? (qs[local] & 0x0f) : (qs[local - 16] >> 4);
        return half_at_host(base) * float(nib - 8);
    }
    const uint8_t *ql = base;
    const uint8_t *qh = base + 128;
    const int8_t *sca = reinterpret_cast<const int8_t *>(base + 192);
    const float d = half_at_host(base + 208);
    const int chunk = local >> 7;
    const int pos = local & 127;
    const int group = pos >> 5;
    const int lane = pos & 31;
    const int ql_byte = ql[chunk * 64 + lane + 32 * (group & 1)];
    const int nib = (group & 2) ? (ql_byte >> 4) : (ql_byte & 0x0f);
    const int hbits = (qh[chunk * 32 + lane] >> (2 * group)) & 3;
    const int q = (nib | (hbits << 4)) - 32;
    const int sc_idx = chunk * 8 + (lane >> 4) + group * 2;
    return d * float(int(sca[sc_idx])) * float(q);
}

template <typename FMT>
__global__ void dequant_gather_kernel(const uint8_t *table, const int *ids,
                                      float *out, long long rows,
                                      long long columns, long long tokens,
                                      float scale) {
    const long long index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= tokens * columns) return;
    const long long token = index / columns;
    const long long column = index - token * columns;
    const int row = ids[token];
    if (row < 0 || row >= rows) {
        out[index] = 0.0f;
        return;
    }
    const long long blocks = columns / FMT::block_k;
    const uint8_t *base =
        table + (static_cast<long long>(row) * blocks + column / FMT::block_k) *
                    FMT::block_bytes;
    out[index] = scale * FMT::dequant(base, int(column % FMT::block_k));
}

template <typename FMT, bool UseAdd>
__global__ void quantized_embedding_kernel(const uint8_t *table, const int *ids,
                                           const float *add, float *out,
                                           long long rows, long long columns,
                                           long long tokens, float scale) {
    const long long index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= tokens * columns) return;
    const long long token = index / columns;
    const long long column = index - token * columns;
    const int row = ids[token];
    if (row < 0 || row >= rows) {
        out[index] = 0.0f;
        return;
    }
    const long long blocks = columns / FMT::block_k;
    const uint8_t *base =
        table + (static_cast<long long>(row) * blocks + column / FMT::block_k) *
                    FMT::block_bytes;
    float value = scale * FMT::dequant(base, int(column % FMT::block_k));
    if (UseAdd) value += add[index];
    out[index] = value;
}

template <typename FMT>
__global__ void quantized_embedding_bag_kernel(
    const uint8_t *table, const int *ids, const int *offsets,
    const float *sample_weights, float *out, long long rows, long long columns,
    long long id_count, long long bags, float scale, int use_weights,
    int mean_mode) {
    const long long index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= bags * columns) return;
    const long long bag = index / columns;
    const long long column = index - bag * columns;
    const int begin = offsets[bag];
    const int end = offsets[bag + 1];
    float sum = 0.0f;
    int valid = 0;
    if (begin >= 0 && end >= begin && end <= id_count) {
        const long long blocks = columns / FMT::block_k;
        for (int item = begin; item < end; ++item) {
            const int row = ids[item];
            if (row < 0 || row >= rows) continue;
            const uint8_t *base =
                table + (static_cast<long long>(row) * blocks + column / FMT::block_k) *
                            FMT::block_bytes;
            const float coefficient = use_weights ? sample_weights[item] : 1.0f;
            sum += coefficient * FMT::dequant(base, int(column % FMT::block_k));
            ++valid;
        }
    }
    const float denominator = mean_mode && valid > 0 ? float(valid) : 1.0f;
    out[index] = scale * sum / denominator;
}

std::vector<float> dequant_gather_ref(const std::vector<uint8_t> &table,
                                      const std::vector<int> &ids,
                                      long long rows, long long columns,
                                      int fmt, float scale) {
    std::vector<float> out(ids.size() * size_t(columns), 0.0f);
    for (size_t token = 0; token < ids.size(); ++token) {
        const int row = ids[token];
        if (row < 0 || row >= rows) continue;
        for (long long column = 0; column < columns; ++column) {
            out[token * size_t(columns) + column] =
                scale * dequant_host(table, fmt, row, columns, column);
        }
    }
    return out;
}

std::vector<float> quantized_embedding_ref(const std::vector<uint8_t> &table,
                                           const std::vector<int> &ids,
                                           const std::vector<float> *add,
                                           long long rows, long long columns,
                                           int fmt, float scale) {
    std::vector<float> out = dequant_gather_ref(table, ids, rows, columns, fmt, scale);
    if (add) {
        for (size_t token = 0; token < ids.size(); ++token) {
            if (ids[token] < 0 || ids[token] >= rows) continue;
            for (long long column = 0; column < columns; ++column) {
                const size_t index = token * size_t(columns) + size_t(column);
                out[index] += (*add)[index];
            }
        }
    }
    return out;
}

std::vector<float> quantized_embedding_bag_ref(
    const std::vector<uint8_t> &table, const std::vector<int> &ids,
    const std::vector<int> &offsets, const std::vector<float> *sample_weights,
    long long rows, long long columns, long long bags, int fmt, float scale,
    bool use_weights, bool mean_mode) {
    std::vector<float> out(size_t(bags * columns), 0.0f);
    for (long long bag = 0; bag < bags; ++bag) {
        const int begin = offsets[bag];
        const int end = offsets[bag + 1];
        int valid = 0;
        if (begin >= 0 && end >= begin && end <= int(ids.size())) {
            for (int item = begin; item < end; ++item) {
                const int row = ids[item];
                if (row < 0 || row >= rows) continue;
                const float coefficient =
                    use_weights ? (*sample_weights)[item] : 1.0f;
                for (long long column = 0; column < columns; ++column) {
                    out[bag * columns + column] +=
                        coefficient *
                        dequant_host(table, fmt, row, columns, column);
                }
                ++valid;
            }
        }
        const float denominator = mean_mode && valid > 0 ? float(valid) : 1.0f;
        for (long long column = 0; column < columns; ++column) {
            out[bag * columns + column] = scale * out[bag * columns + column] /
                                          denominator;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// BitNet qgemm backward-input
// ---------------------------------------------------------------------------

__device__ __forceinline__ float ternary_weight_device(const uint8_t *packed,
                                                       long long row,
                                                       long long k,
                                                       long long column) {
    const long long block = column / 32;
    const int local = int(column & 31);
    const uint8_t *src = packed + (row * (k / 32) + block) * 10;
    const float scale = half_at_device(src);
    const int code = (src[2 + local / 4] >> (2 * (local % 4))) & 3;
    return scale * float(code - 1);
}

__global__ void qgemm_backward_input_scalar_kernel(const uint8_t *packed_weights,
                                                   const float *grad_y,
                                                   float *grad_x, long long m,
                                                   long long n, long long k) {
    const long long index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= m * k) return;
    const long long row = index / k;
    const long long column = index - row * k;
    double sum = 0.0;
    for (long long output = 0; output < n; ++output) {
        sum += double(grad_y[row * n + output]) *
               double(ternary_weight_device(packed_weights, output, k, column));
    }
    grad_x[index] = float(sum);
}

__global__ void qgemm_backward_input_wave_kernel(const uint8_t *packed_weights,
                                                 const float *grad_y,
                                                 float *grad_x, long long m,
                                                 long long n, long long k) {
    const int lane = threadIdx.x & (qc::kWave - 1);
    const int wave = threadIdx.x >> 6;
    const long long out_index =
        static_cast<long long>(blockIdx.x) * (blockDim.x / qc::kWave) + wave;
    if (out_index >= m * k) return;
    const long long row = out_index / k;
    const long long column = out_index - row * k;
    float sum = 0.0f;
    for (long long output = lane; output < n; output += qc::kWave) {
        sum += grad_y[row * n + output] *
               ternary_weight_device(packed_weights, output, k, column);
    }
    sum = qc::wave_reduce_sum(sum);
    if (lane == 0) grad_x[out_index] = sum;
}

std::vector<float> qgemm_backward_input_ref(const std::vector<uint8_t> &packed,
                                            const std::vector<float> &grad_y,
                                            long long m, long long n,
                                            long long k) {
    std::vector<float> dense_w;
    ternary_unpack_ref(packed, dense_w, n, k);
    std::vector<float> grad_x(size_t(m * k), 0.0f);
    for (long long row = 0; row < m; ++row) {
        for (long long input = 0; input < k; ++input) {
            double sum = 0.0;
            for (long long output = 0; output < n; ++output) {
                sum += double(grad_y[row * n + output]) *
                       double(dense_w[output * k + input]);
            }
            grad_x[row * k + input] = float(sum);
        }
    }
    return grad_x;
}

// ---------------------------------------------------------------------------
// MXFP4 GEMV
// ---------------------------------------------------------------------------

__global__ void mxfp4_gemv_scalar_kernel(const uint8_t *packed_weights,
                                         const uint8_t *scale_codes,
                                         const float *x, float *y,
                                         long long n, long long k) {
    const long long row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    const long long blocks = k / 32;
    double sum = 0.0;
    for (long long block = 0; block < blocks; ++block) {
        const float scale = e8m0_decode_ref(scale_codes[row * blocks + block]);
        double block_sum = 0.0;
        for (int item = 0; item < 32; ++item) {
            const long long column = block * 32 + item;
            const uint8_t byte = packed_weights[row * (k / 2) + column / 2];
            const uint8_t code =
                uint8_t((byte >> (4 * (column & 1))) & 0x0fU);
            block_sum += double(fp4_e2m1_decode_ref(code)) * double(x[column]);
        }
        sum += block_sum * double(scale);
    }
    y[row] = float(sum);
}

__global__ void mxfp4_gemv_wave_kernel(const uint8_t *packed_weights,
                                       const uint8_t *scale_codes,
                                       const float *x, float *y, long long n,
                                       long long k) {
    const int lane = threadIdx.x & (qc::kWave - 1);
    const int wave = threadIdx.x >> 6;
    const long long row =
        static_cast<long long>(blockIdx.x) * (blockDim.x / qc::kWave) + wave;
    if (row >= n) return;
    const long long blocks = k / 32;
    float sum = 0.0f;
    for (long long column = lane; column < k; column += qc::kWave) {
        const long long block = column / 32;
        const float scale = e8m0_decode_ref(scale_codes[row * blocks + block]);
        const uint8_t byte = packed_weights[row * (k / 2) + column / 2];
        const uint8_t code = uint8_t((byte >> (4 * (column & 1))) & 0x0fU);
        sum += scale * fp4_e2m1_decode_ref(code) * x[column];
    }
    sum = qc::wave_reduce_sum(sum);
    if (lane == 0) y[row] = sum;
}

void fill_mxfp4(std::vector<uint8_t> &packed, std::vector<uint8_t> &scale_codes,
                long long n, long long k) {
    packed.assign(size_t(n * k / 2), 0);
    scale_codes.assign(size_t(n * (k / 32)), 0);
    for (long long row = 0; row < n; ++row) {
        for (long long block = 0; block < k / 32; ++block) {
            scale_codes[row * (k / 32) + block] =
                uint8_t(121 + ((row * 3 + block * 5) % 12));
            for (int item = 0; item < 32; ++item) {
                const long long column = block * 32 + item;
                const uint8_t code =
                    uint8_t((row * 11 + block * 7 + item * 3) & 0x0f);
                uint8_t &byte = packed[row * (k / 2) + column / 2];
                if (column & 1) {
                    byte = uint8_t((byte & 0x0fU) | (code << 4));
                } else {
                    byte = uint8_t((byte & 0xf0U) | code);
                }
            }
        }
    }
}

std::vector<float> mxfp4_gemv_ref(const std::vector<uint8_t> &packed,
                                  const std::vector<uint8_t> &scale_codes,
                                  const std::vector<float> &x, long long n,
                                  long long k) {
    std::vector<float> y(size_t(n), 0.0f);
    const long long blocks = k / 32;
    for (long long row = 0; row < n; ++row) {
        double sum = 0.0;
        for (long long block = 0; block < blocks; ++block) {
            const float scale = e8m0_decode_ref(scale_codes[row * blocks + block]);
            double block_sum = 0.0;
            for (int item = 0; item < 32; ++item) {
                const long long column = block * 32 + item;
                const uint8_t byte = packed[row * (k / 2) + column / 2];
                const uint8_t code =
                    uint8_t((byte >> (4 * (column & 1))) & 0x0fU);
                block_sum += double(fp4_e2m1_decode_ref(code)) *
                             double(x[column]);
            }
            sum += block_sum * double(scale);
        }
        y[row] = float(sum);
    }
    return y;
}

// ---------------------------------------------------------------------------
// Correctness and benchmark drivers
// ---------------------------------------------------------------------------

template <typename FMT>
void launch_dequant_gather(const uint8_t *table, const int *ids, float *out,
                           long long rows, long long columns, long long tokens,
                           float scale) {
    dequant_gather_kernel<FMT><<<qc::grid_for(size_t(tokens * columns), 256), 256>>>(
        table, ids, out, rows, columns, tokens, scale);
}

template <typename FMT>
void launch_embedding(const uint8_t *table, const int *ids, const float *add,
                      float *out, long long rows, long long columns,
                      long long tokens, float scale, bool use_add) {
    if (use_add) {
        quantized_embedding_kernel<FMT, true>
            <<<qc::grid_for(size_t(tokens * columns), 256), 256>>>(
                table, ids, add, out, rows, columns, tokens, scale);
    } else {
        quantized_embedding_kernel<FMT, false>
            <<<qc::grid_for(size_t(tokens * columns), 256), 256>>>(
                table, ids, add, out, rows, columns, tokens, scale);
    }
}

template <typename FMT>
void launch_embedding_bag(const uint8_t *table, const int *ids,
                          const int *offsets, const float *sample_weights,
                          float *out, long long rows, long long columns,
                          long long id_count, long long bags, float scale,
                          bool use_weights, bool mean_mode) {
    quantized_embedding_bag_kernel<FMT>
        <<<qc::grid_for(size_t(bags * columns), 256), 256>>>(
            table, ids, offsets, sample_weights, out, rows, columns, id_count,
            bags, scale, use_weights ? 1 : 0, mean_mode ? 1 : 0);
}

bool run_correctness() {
    bool ok = true;
    int checks = 0;
    qc::Rng rng(0xC0FFEE);

    {
        constexpr long long rows = 5;
        constexpr long long dim = 129;
        const auto x = rng.uniforms(size_t(rows * dim), -3.0f, 3.0f);
        std::vector<float> ref_out, ref_scales;
        std::vector<int8_t> ref_codes;
        fake_quant_int8_ref(x, ref_out, ref_codes, ref_scales, rows, dim);
        float *dx = qc::dnew(x);
        float *dout = qc::dzero<float>(x.size());
        int8_t *dcodes = qc::dzero<int8_t>(x.size());
        float *dscales = qc::dzero<float>(rows);
        fake_quant_int8_wave_kernel<<<qc::wave_blocks(rows), qc::kThreads>>>(
            dx, dout, dcodes, dscales, rows, dim);
        QC_SYNC();
        ok &= qc::compare(qc::d2h(dout, x.size()), to_ref(ref_out),
                          qc::Tol::fp32())
                  .report("fake_quant_int8 output");
        ++checks;
        ok &= qc::compare(qc::d2h(dcodes, x.size()), to_ref_i8(ref_codes),
                          qc::Tol::exact())
                  .report("fake_quant_int8 codes");
        ++checks;
        ok &= qc::compare(qc::d2h(dscales, rows), to_ref(ref_scales),
                          qc::Tol::fp32())
                  .report("fake_quant_int8 scales");
        ++checks;
        qc::dfree(dx, dout, dcodes, dscales);
    }

    {
        constexpr long long count = 777;
        auto x = rng.uniforms(size_t(count), -11.0f, 11.0f);
        x[3] = -0.0f;
        for (int kind : {kE4M3FN, kE5M2}) {
            std::vector<float> ref_out;
            std::vector<uint8_t> ref_codes;
            float ref_scale = 0.0f;
            fake_quant_float8_ref(x, ref_out, ref_codes, ref_scale, kind);
            float *dx = qc::dnew(x);
            float *dout = qc::dzero<float>(x.size());
            uint8_t *dcodes = qc::dzero<uint8_t>(x.size());
            float *dscale = qc::dzero<float>(1);
            fake_quant_float8_wave_kernel<<<1, qc::kWave>>>(
                dx, dout, dcodes, dscale, count, kind);
            QC_SYNC();
            const char *name = kind == kE4M3FN ? "fake_quant_float8 e4m3"
                                               : "fake_quant_float8 e5m2";
            ok &= qc::compare(qc::d2h(dout, x.size()), to_ref(ref_out),
                              qc::Tol::fp8())
                      .report(std::string(name) + " output");
            ++checks;
            ok &= qc::compare(qc::d2h(dcodes, x.size()), to_ref_u8(ref_codes),
                              qc::Tol::exact())
                      .report(std::string(name) + " codes");
            ++checks;
            ok &= qc::compare(qc::d2h(dscale, 1), std::vector<double>{ref_scale},
                              qc::Tol::fp32())
                      .report(std::string(name) + " scale");
            ++checks;
            qc::dfree(dx, dout, dcodes, dscale);
        }
    }

    {
        constexpr long long rows = 4;
        constexpr long long k = 512;
        auto weights = rng.uniforms(size_t(rows * k), -2.0f, 2.0f);
        std::vector<uint8_t> ref_packed;
        std::vector<float> ref_dequant;
        tq2_0_pack_ref(weights, ref_packed, ref_dequant, rows, k);
        float *dw = qc::dnew(weights);
        uint8_t *dpacked = qc::dzero<uint8_t>(ref_packed.size());
        float *ddeq = qc::dzero<float>(weights.size());
        tq2_0_pack_block_kernel<<<rows * (k / 256), 256>>>(
            dw, dpacked, ddeq, rows, k);
        QC_SYNC();
        ok &= qc::compare(qc::d2h(dpacked, ref_packed.size()), to_ref_u8(ref_packed),
                          qc::Tol::exact())
                  .report("tq2_0_pack bytes");
        ++checks;
        ok &= qc::compare(qc::d2h(ddeq, weights.size()), to_ref(ref_dequant),
                          qc::Tol::fp32())
                  .report("tq2_0_pack dequantized");
        ++checks;
        float *dunpack = qc::dzero<float>(weights.size());
        tq2_0_unpack_kernel<<<qc::grid_for(weights.size(), 256), 256>>>(
            dpacked, dunpack, rows, k);
        QC_SYNC();
        std::vector<float> ref_unpacked;
        tq2_0_unpack_ref(ref_packed, ref_unpacked, rows, k);
        ok &= qc::compare(qc::d2h(dunpack, weights.size()), to_ref(ref_unpacked),
                          qc::Tol::fp32())
                  .report("tq2_0_unpack");
        ++checks;
        qc::dfree(dw, dpacked, ddeq, dunpack);
    }

    {
        constexpr long long rows = 6;
        constexpr long long k = 256;
        constexpr long long group_k = 64;
        auto weights = rng.uniforms(size_t(rows * k), -1.5f, 1.5f);
        std::vector<uint8_t> ref_packed;
        std::vector<float> ref_dequant;
        ternary_pack_ref(weights, ref_packed, ref_dequant, rows, k, group_k);
        float *dw = qc::dnew(weights);
        uint8_t *dpacked = qc::dzero<uint8_t>(ref_packed.size());
        float *ddeq = qc::dzero<float>(weights.size());
        ternary_pack_group_kernel<<<rows * (k / group_k), 256>>>(
            dw, dpacked, ddeq, rows, k, group_k);
        QC_SYNC();
        ok &= qc::compare(qc::d2h(dpacked, ref_packed.size()), to_ref_u8(ref_packed),
                          qc::Tol::exact())
                  .report("ternary_pack bytes");
        ++checks;
        ok &= qc::compare(qc::d2h(ddeq, weights.size()), to_ref(ref_dequant),
                          qc::Tol::fp32())
                  .report("ternary_pack dequantized");
        ++checks;
        float *dunpack = qc::dzero<float>(weights.size());
        ternary_unpack_kernel<<<qc::grid_for(weights.size(), 256), 256>>>(
            dpacked, dunpack, rows, k);
        QC_SYNC();
        std::vector<float> ref_unpacked;
        ternary_unpack_ref(ref_packed, ref_unpacked, rows, k);
        ok &= qc::compare(qc::d2h(dunpack, weights.size()), to_ref(ref_unpacked),
                          qc::Tol::fp32())
                  .report("ternary_unpack");
        ++checks;
        uint32_t *dstats = qc::dzero<uint32_t>(size_t(rows * 3));
        ternary_stats_kernel<<<rows, 256>>>(dpacked, dstats, rows, k);
        QC_SYNC();
        const auto ref_stats = ternary_stats_ref(ref_packed, rows, k);
        ok &= qc::compare(qc::d2h(dstats, size_t(rows * 3)), to_ref_u32(ref_stats),
                          qc::Tol::exact())
                  .report("ternary_stats");
        ++checks;
        auto other = weights;
        for (size_t i = 0; i < other.size(); i += 7) other[i] = -other[i];
        std::vector<uint8_t> ref_packed_b;
        std::vector<float> ignored;
        ternary_pack_ref(other, ref_packed_b, ignored, rows, k, group_k);
        uint8_t *dpacked_b = qc::dnew(ref_packed_b);
        uint32_t *dflips = qc::dzero<uint32_t>(rows);
        ternary_flip_count_kernel<<<rows, 256>>>(dpacked, dpacked_b, dflips,
                                                rows, k);
        QC_SYNC();
        const auto ref_flips = ternary_flips_ref(ref_packed, ref_packed_b, rows, k);
        ok &= qc::compare(qc::d2h(dflips, rows), to_ref_u32(ref_flips),
                          qc::Tol::exact())
                  .report("ternary_code_flip_count");
        ++checks;
        qc::dfree(dw, dpacked, ddeq, dunpack, dstats, dpacked_b, dflips);
    }

    {
        constexpr long long tokens = 33;
        constexpr long long channels = 70;
        auto x = rng.uniforms(size_t(tokens * channels), -4.0f, 4.0f);
        std::vector<float> running(size_t(channels), -0.25f);
        x[5 * channels + 9] = std::numeric_limits<float>::quiet_NaN();
        running[17] = std::numeric_limits<float>::infinity();
        const auto ref = calibration_absmax_ref(x, &running, tokens, channels);
        float *dx = qc::dnew(x);
        float *drunning = qc::dnew(running);
        float *dout = qc::dzero<float>(channels);
        calibration_absmax_block_kernel<<<channels, 256>>>(dx, drunning, dout,
                                                           tokens, channels, 1);
        QC_SYNC();
        ok &= compare_nan_equal(qc::d2h(dout, channels), ref, "calibration_absmax");
        ++checks;
        qc::dfree(dx, drunning, dout);
    }

    {
        constexpr long long rows = 9;
        constexpr long long columns = 256;
        const std::vector<int> ids = {0, 7, -1, 4, 8};
        const float scale = 0.75f;
        for (int fmt : {kFmtQ8_0, kFmtQ4_0, kFmtQ6_K}) {
            std::vector<uint8_t> table;
            if (fmt == kFmtQ8_0) fill_q8_0_table(table, rows, columns);
            if (fmt == kFmtQ4_0) fill_q4_0_table(table, rows, columns);
            if (fmt == kFmtQ6_K) fill_q6_K_table(table, rows, columns);
            uint8_t *dtable = qc::dnew(table);
            int *dids = qc::dnew(ids);
            float *dout = qc::dzero<float>(ids.size() * size_t(columns));
            if (fmt == kFmtQ8_0) {
                launch_dequant_gather<q8_0>(dtable, dids, dout, rows, columns,
                                            ids.size(), scale);
            } else if (fmt == kFmtQ4_0) {
                launch_dequant_gather<q4_0>(dtable, dids, dout, rows, columns,
                                            ids.size(), scale);
            } else {
                launch_dequant_gather<q6_K>(dtable, dids, dout, rows, columns,
                                            ids.size(), scale);
            }
            QC_SYNC();
            const auto ref = dequant_gather_ref(table, ids, rows, columns, fmt, scale);
            ok &= qc::compare(qc::d2h(dout, ids.size() * size_t(columns)),
                              to_ref(ref), qc::Tol::fp32())
                      .report(fmt == kFmtQ8_0 ? "dequant_gather q8_0"
                                              : (fmt == kFmtQ4_0
                                                     ? "dequant_gather q4_0"
                                                     : "dequant_gather q6_K"));
            ++checks;
            qc::dfree(dtable, dids, dout);
        }
    }

    {
        constexpr long long rows = 10;
        constexpr long long columns = 256;
        const std::vector<int> ids = {3, 1, 9, -3};
        const float scale = 1.25f;
        std::vector<uint8_t> table;
        fill_q8_0_table(table, rows, columns);
        auto add = rng.uniforms(ids.size() * size_t(columns), -0.2f, 0.2f);
        uint8_t *dtable = qc::dnew(table);
        int *dids = qc::dnew(ids);
        float *dadd = qc::dnew(add);
        float *dout = qc::dzero<float>(ids.size() * size_t(columns));
        launch_embedding<q8_0>(dtable, dids, dadd, dout, rows, columns,
                               ids.size(), scale, true);
        QC_SYNC();
        const auto ref =
            quantized_embedding_ref(table, ids, &add, rows, columns, kFmtQ8_0, scale);
        ok &= qc::compare(qc::d2h(dout, ids.size() * size_t(columns)),
                          to_ref(ref), qc::Tol::fp32())
                  .report("quantized_embedding q8_0 add");
        ++checks;
        qc::dfree(dtable, dids, dadd, dout);
    }

    {
        constexpr long long rows = 8;
        constexpr long long columns = 256;
        constexpr long long bags = 3;
        const std::vector<int> ids = {1, 5, -1, 2, 7, 0};
        const std::vector<int> offsets = {0, 2, 5, 6};
        const std::vector<float> weights = {1.0f, 0.5f, 2.0f, 0.25f, -1.0f, 3.0f};
        std::vector<uint8_t> table;
        fill_q6_K_table(table, rows, columns);
        uint8_t *dtable = qc::dnew(table);
        int *dids = qc::dnew(ids);
        int *doffsets = qc::dnew(offsets);
        float *dweights = qc::dnew(weights);
        float *dout = qc::dzero<float>(size_t(bags * columns));
        launch_embedding_bag<q6_K>(dtable, dids, doffsets, dweights, dout, rows,
                                   columns, ids.size(), bags, 0.5f, true, true);
        QC_SYNC();
        const auto ref = quantized_embedding_bag_ref(
            table, ids, offsets, &weights, rows, columns, bags, kFmtQ6_K, 0.5f,
            true, true);
        ok &= qc::compare(qc::d2h(dout, size_t(bags * columns)), to_ref(ref),
                          qc::Tol::fp32())
                  .report("quantized_embedding_bag q6_K weighted_mean");
        ++checks;
        qc::dfree(dtable, dids, doffsets, dweights, dout);
    }

    {
        constexpr long long m = 5;
        constexpr long long n = 37;
        constexpr long long k = 96;
        auto w = rng.uniforms(size_t(n * k), -1.0f, 1.0f);
        std::vector<uint8_t> packed;
        std::vector<float> ignored;
        ternary_pack_ref(w, packed, ignored, n, k, 32);
        auto grad_y = rng.uniforms(size_t(m * n), -0.5f, 0.5f);
        const auto ref = qgemm_backward_input_ref(packed, grad_y, m, n, k);
        uint8_t *dpacked = qc::dnew(packed);
        float *dgy = qc::dnew(grad_y);
        float *dgx = qc::dzero<float>(size_t(m * k));
        qgemm_backward_input_wave_kernel<<<qc::wave_blocks(size_t(m * k)),
                                          qc::kThreads>>>(dpacked, dgy, dgx, m,
                                                          n, k);
        QC_SYNC();
        ok &= qc::compare(qc::d2h(dgx, size_t(m * k)), to_ref(ref),
                          qc::Tol::fp32().with_elementwise(2e-5, 2e-5))
                  .report("qgemm_backward_input bitnet");
        ++checks;
        qc::dfree(dpacked, dgy, dgx);
    }

    {
        constexpr long long n = 71;
        constexpr long long k = 160;
        std::vector<uint8_t> packed, scale_codes;
        fill_mxfp4(packed, scale_codes, n, k);
        auto x = rng.uniforms(size_t(k), -0.75f, 0.75f);
        const auto ref = mxfp4_gemv_ref(packed, scale_codes, x, n, k);
        uint8_t *dpacked = qc::dnew(packed);
        uint8_t *dscales = qc::dnew(scale_codes);
        float *dx = qc::dnew(x);
        float *dy = qc::dzero<float>(n);
        mxfp4_gemv_wave_kernel<<<qc::wave_blocks(n), qc::kThreads>>>(
            dpacked, dscales, dx, dy, n, k);
        QC_SYNC();
        ok &= qc::compare(qc::d2h(dy, n), to_ref(ref),
                          qc::Tol::fp32().with_elementwise(2e-5, 2e-5))
                  .report("mxfp4_gemv");
        ++checks;
        qc::dfree(dpacked, dscales, dx, dy);
    }

    std::printf("Phase 6 correctness checks: %d\n", checks);
    return ok;
}

template <typename Fn>
qc::Bench bench_launch(Fn &&fn, int warmups = 10, int iters = 60,
                       int repeats = 1) {
    return qc::bench([&] {
        for (int repeat = 0; repeat < repeats; ++repeat) fn();
        QC_CHECK(hipGetLastError());
    }, warmups, iters);
}

template <typename Fn>
qc::Bench bench_per_launch(Fn &&fn, int warmups = 10, int iters = 60,
                           int repeats = 1) {
    qc::Bench b = bench_launch(fn, warmups, iters, repeats);
    if (repeats > 1) {
        b.median_ms /= double(repeats);
        b.min_ms /= double(repeats);
        b.max_ms /= double(repeats);
        b.mean_ms /= double(repeats);
    }
    return b;
}

void report_pair(const char *label, const qc::Bench &baseline,
                 const qc::Bench &candidate, double bytes_or_flops,
                 bool compute) {
    if (compute) {
        baseline.report_compute(std::string(label) + " scalar", bytes_or_flops);
        candidate.report_compute(std::string(label) + " wave64", bytes_or_flops);
    } else {
        baseline.report_bandwidth(std::string(label) + " scalar", bytes_or_flops);
        candidate.report_bandwidth(std::string(label) + " parallel", bytes_or_flops);
    }
    qc::report_ab(label, baseline, candidate);
}

void run_benchmarks() {
    std::printf("\n== Phase 6 benchmarks ==\n");
    std::printf("   Timing note: medians are reported per launch; fast kernels use "
                "inner launch repeats to stabilize HIP-event spread.\n");
    qc::Rng rng(0xBEEFFEED);

    {
        constexpr long long rows = 8192;
        constexpr long long dim = 1024;
        auto x = rng.uniforms(size_t(rows * dim), -3.0f, 3.0f);
        float *dx = qc::dnew(x);
        float *dout = qc::dzero<float>(x.size());
        int8_t *dcodes = qc::dzero<int8_t>(x.size());
        float *dscales = qc::dzero<float>(rows);
        const double bytes = double(x.size()) * (sizeof(float) + sizeof(float) + 1) +
                             double(rows) * sizeof(float);
        const auto scalar = bench_per_launch([&] {
            fake_quant_int8_scalar_kernel<<<qc::grid_for(rows, 128), 128>>>(
                dx, dout, dcodes, dscales, rows, dim);
        }, 10, 60, 8);
        const auto wave = bench_per_launch([&] {
            fake_quant_int8_wave_kernel<<<qc::wave_blocks(rows), qc::kThreads>>>(
                dx, dout, dcodes, dscales, rows, dim);
        }, 10, 60, 8);
        report_pair("fake_quant_int8", scalar, wave, bytes, false);
        qc::dfree(dx, dout, dcodes, dscales);
    }

    {
        constexpr long long count = 1024 * 1024;
        auto x = rng.uniforms(size_t(count), -8.0f, 8.0f);
        float *dx = qc::dnew(x);
        float *dout = qc::dzero<float>(x.size());
        uint8_t *dcodes = qc::dzero<uint8_t>(x.size());
        float *dscale = qc::dzero<float>(1);
        const double bytes = double(count) * (sizeof(float) + sizeof(float) + 1) +
                             sizeof(float);
        const auto scalar = bench_per_launch([&] {
            fake_quant_float8_scalar_kernel<<<1, 1>>>(dx, dout, dcodes, dscale,
                                                      count, kE4M3FN);
        }, 3, 20);
        const auto wave = bench_per_launch([&] {
            fake_quant_float8_wave_kernel<<<1, qc::kWave>>>(dx, dout, dcodes,
                                                            dscale, count,
                                                            kE4M3FN);
        }, 10, 60, 4);
        report_pair("fake_quant_float8", scalar, wave, bytes, false);
        qc::dfree(dx, dout, dcodes, dscale);
    }

    {
        constexpr long long rows = 32768;
        constexpr long long k = 512;
        auto weights = rng.uniforms(size_t(rows * k), -2.0f, 2.0f);
        const size_t packed_bytes = size_t(rows * (k / 256) * 66);
        float *dw = qc::dnew(weights);
        uint8_t *dpacked = qc::dzero<uint8_t>(packed_bytes);
        float *ddeq = qc::dzero<float>(weights.size());
        const double bytes = double(weights.size()) * sizeof(float) +
                             double(packed_bytes) +
                             double(weights.size()) * sizeof(float);
        const auto scalar = bench_per_launch([&] {
            tq2_0_pack_scalar_kernel<<<qc::grid_for(rows * (k / 256), 128), 128>>>(
                dw, dpacked, ddeq, rows, k);
        }, 10, 60, 16);
        const auto block = bench_per_launch([&] {
            tq2_0_pack_block_kernel<<<rows * (k / 256), 256>>>(dw, dpacked,
                                                               ddeq, rows, k);
        }, 10, 60, 16);
        report_pair("tq2_0_pack", scalar, block, bytes, false);
        const auto unpack = bench_per_launch([&] {
            tq2_0_unpack_kernel<<<qc::grid_for(weights.size(), 256), 256>>>(
                dpacked, ddeq, rows, k);
        }, 10, 60, 512);
        unpack.report_bandwidth("tq2_0_unpack", bytes);
        qc::dfree(dw, dpacked, ddeq);
    }

    {
        constexpr long long rows = 65536;
        constexpr long long k = 512;
        constexpr long long group_k = 128;
        auto weights = rng.uniforms(size_t(rows * k), -1.5f, 1.5f);
        const size_t packed_bytes = size_t(rows * (k / 32) * 10);
        float *dw = qc::dnew(weights);
        uint8_t *dpacked = qc::dzero<uint8_t>(packed_bytes);
        float *ddeq = qc::dzero<float>(weights.size());
        const double bytes = double(weights.size()) * sizeof(float) +
                             double(packed_bytes) +
                             double(weights.size()) * sizeof(float);
        const auto scalar = bench_per_launch([&] {
            ternary_pack_scalar_kernel<<<qc::grid_for(rows * (k / group_k), 128),
                                        128>>>(dw, dpacked, ddeq, rows, k,
                                               group_k);
        }, 10, 60, 16);
        const auto block = bench_per_launch([&] {
            ternary_pack_group_kernel<<<rows * (k / group_k), 256>>>(
                dw, dpacked, ddeq, rows, k, group_k);
        }, 10, 60, 16);
        report_pair("ternary_pack", scalar, block, bytes, false);
        const auto unpack = bench_per_launch([&] {
            ternary_unpack_kernel<<<qc::grid_for(weights.size(), 256), 256>>>(
                dpacked, ddeq, rows, k);
        }, 10, 60, 256);
        unpack.report_bandwidth("ternary_unpack", bytes);
        uint32_t *dstats = qc::dzero<uint32_t>(size_t(rows * 3));
        uint32_t *dflips = qc::dzero<uint32_t>(rows);
        const auto stats = bench_per_launch([&] {
            ternary_stats_kernel<<<rows, 256>>>(dpacked, dstats, rows, k);
        }, 10, 60, 256);
        stats.report_bandwidth("ternary_stats", double(packed_bytes));
        const auto flips = bench_per_launch([&] {
            ternary_flip_count_kernel<<<rows, 256>>>(dpacked, dpacked, dflips,
                                                     rows, k);
        }, 10, 60, 256);
        flips.report_bandwidth("ternary_code_flip_count", double(packed_bytes) * 2.0);
        qc::dfree(dw, dpacked, ddeq, dstats, dflips);
    }

    {
        constexpr long long tokens = 4096;
        constexpr long long channels = 4096;
        auto x = rng.uniforms(size_t(tokens * channels), -4.0f, 4.0f);
        auto running = rng.uniforms(size_t(channels), 0.0f, 5.0f);
        float *dx = qc::dnew(x);
        float *drunning = qc::dnew(running);
        float *dout = qc::dzero<float>(channels);
        const double bytes =
            double(x.size() + running.size() + running.size()) * sizeof(float);
        const auto scalar = bench_per_launch([&] {
            calibration_absmax_scalar_kernel<<<qc::grid_for(channels, 128), 128>>>(
                dx, drunning, dout, tokens, channels, 1);
        });
        const auto block = bench_per_launch([&] {
            calibration_absmax_block_kernel<<<channels, 256>>>(
                dx, drunning, dout, tokens, channels, 1);
        });
        report_pair("calibration_absmax", scalar, block, bytes, false);
        qc::dfree(dx, drunning, dout);
    }

    {
        constexpr long long rows = 65536;
        constexpr long long columns = 256;
        constexpr long long tokens = 65536;
        std::vector<uint8_t> table;
        fill_q8_0_table(table, rows, columns);
        std::vector<int> ids(static_cast<size_t>(tokens));
        for (long long i = 0; i < tokens; ++i) ids[i] = int((i * 7919) % rows);
        auto add = rng.uniforms(size_t(tokens * columns), -0.1f, 0.1f);
        uint8_t *dtable = qc::dnew(table);
        int *dids = qc::dnew(ids);
        float *dadd = qc::dnew(add);
        float *dout = qc::dzero<float>(size_t(tokens * columns));
        const double bytes = double(tokens) * columns * sizeof(float) +
                             double(tokens) * columns * sizeof(float) +
                             double(tokens) * (columns / q8_0::block_k) *
                                 q8_0::block_bytes;
        const auto gather = bench_per_launch([&] {
            launch_dequant_gather<q8_0>(dtable, dids, dout, rows, columns,
                                        tokens, 1.0f);
        }, 10, 60, 512);
        gather.report_bandwidth("dequant_gather q8_0", bytes);
        const auto emb = bench_per_launch([&] {
            launch_embedding<q8_0>(dtable, dids, dadd, dout, rows, columns,
                                   tokens, 1.0f, true);
        }, 10, 60, 512);
        emb.report_bandwidth("quantized_embedding q8_0 add", bytes);
        qc::dfree(dtable, dids, dadd, dout);
    }

    {
        constexpr long long rows = 32768;
        constexpr long long columns = 256;
        constexpr long long id_count = 65536;
        constexpr long long bags = 8192;
        std::vector<uint8_t> table;
        fill_q6_K_table(table, rows, columns);
        std::vector<int> ids(static_cast<size_t>(id_count));
        std::vector<int> offsets(size_t(bags + 1));
        std::vector<float> weights(size_t(id_count), 1.0f);
        for (long long i = 0; i < id_count; ++i) {
            ids[i] = int((i * 3571) % rows);
            weights[i] = 0.5f + 0.001f * float(i & 31);
        }
        for (long long bag = 0; bag <= bags; ++bag) {
            offsets[bag] = int((id_count * bag) / bags);
        }
        uint8_t *dtable = qc::dnew(table);
        int *dids = qc::dnew(ids);
        int *doffsets = qc::dnew(offsets);
        float *dweights = qc::dnew(weights);
        float *dout = qc::dzero<float>(size_t(bags * columns));
        const double bytes = double(id_count) * (columns / q6_K::block_k) *
                                 q6_K::block_bytes +
                             double(bags * columns) * sizeof(float);
        const auto bag = bench_per_launch([&] {
            launch_embedding_bag<q6_K>(dtable, dids, doffsets, dweights, dout,
                                       rows, columns, id_count, bags, 1.0f,
                                       true, true);
        }, 10, 60, 512);
        bag.report_bandwidth("quantized_embedding_bag q6_K", bytes);
        qc::dfree(dtable, dids, doffsets, dweights, dout);
    }

    {
        constexpr long long m = 64;
        constexpr long long n = 1024;
        constexpr long long k = 512;
        auto w = rng.uniforms(size_t(n * k), -1.0f, 1.0f);
        std::vector<uint8_t> packed;
        std::vector<float> ignored;
        ternary_pack_ref(w, packed, ignored, n, k, 32);
        auto grad_y = rng.uniforms(size_t(m * n), -0.5f, 0.5f);
        uint8_t *dpacked = qc::dnew(packed);
        float *dgy = qc::dnew(grad_y);
        float *dgx = qc::dzero<float>(size_t(m * k));
        const double flops = 2.0 * double(m) * double(n) * double(k);
        const auto scalar = bench_per_launch([&] {
            qgemm_backward_input_scalar_kernel<<<qc::grid_for(size_t(m * k), 128),
                                                128>>>(dpacked, dgy, dgx, m, n,
                                                       k);
        }, 10, 60, 8);
        const auto wave = bench_per_launch([&] {
            qgemm_backward_input_wave_kernel<<<qc::wave_blocks(size_t(m * k)),
                                             qc::kThreads>>>(dpacked, dgy, dgx,
                                                             m, n, k);
        }, 10, 60, 8);
        report_pair("qgemm_backward_input", scalar, wave, flops, true);
        qc::dfree(dpacked, dgy, dgx);
    }

    {
        constexpr long long n = 65536;
        constexpr long long k = 1024;
        std::vector<uint8_t> packed, scale_codes;
        fill_mxfp4(packed, scale_codes, n, k);
        auto x = rng.uniforms(size_t(k), -0.75f, 0.75f);
        uint8_t *dpacked = qc::dnew(packed);
        uint8_t *dscales = qc::dnew(scale_codes);
        float *dx = qc::dnew(x);
        float *dy = qc::dzero<float>(n);
        const double flops = 2.0 * double(n) * double(k);
        const auto scalar = bench_per_launch([&] {
            mxfp4_gemv_scalar_kernel<<<qc::grid_for(n, 128), 128>>>(
                dpacked, dscales, dx, dy, n, k);
        }, 10, 60, 256);
        const auto wave = bench_per_launch([&] {
            mxfp4_gemv_wave_kernel<<<qc::wave_blocks(n), qc::kThreads>>>(
                dpacked, dscales, dx, dy, n, k);
        }, 10, 60, 16);
        report_pair("mxfp4_gemv", scalar, wave, flops, true);
        qc::dfree(dpacked, dscales, dx, dy);
    }
}

}  // namespace

int main(int argc, char **argv) {
    qc::print_environment("quant_authoring_phase6");
    const bool ok = run_correctness();
    if (qc::bench_requested(argc, argv)) run_benchmarks();
    return qc::finish(ok);
}
