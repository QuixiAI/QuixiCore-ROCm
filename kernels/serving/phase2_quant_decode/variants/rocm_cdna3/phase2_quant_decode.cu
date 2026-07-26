/**
 * @file
 * @brief Remaining Phase 2 quantized KV-cache and decode-attention parity ports.
 */
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "../../../../common/cdna3_harness.cuh"

namespace {

constexpr int kThreads = qc::kThreads;
constexpr int kScoreTile = 16;

enum Kv3ScaleType { kKv3Fp16 = 0, kKv3Fp32 = 1 };
enum Kv3Signedness { kKv3Unsigned = 0, kKv3Signed = 1 };
enum Kv3ZeroMode { kKv3NoZero = 0, kKv3IntegerZero = 1 };

struct Kv3Config {
    int group_size;
    int scale_type;
    int signedness;
    int zero_mode;
};

enum QuantFormatLite { kFmtQ8_0 = 0, kFmtQ4_0 = 1 };

std::vector<double> to_ref(const std::vector<float> &x) {
    return std::vector<double>(x.begin(), x.end());
}

std::vector<double> to_ref_i32(const std::vector<int32_t> &x) {
    std::vector<double> out(x.size());
    for (size_t i = 0; i < x.size(); ++i) out[i] = static_cast<double>(x[i]);
    return out;
}

std::vector<double> to_ref_u8(const std::vector<uint8_t> &x) {
    std::vector<double> out(x.size());
    for (size_t i = 0; i < x.size(); ++i) out[i] = static_cast<double>(x[i]);
    return out;
}

std::vector<double> to_ref_u16(const std::vector<uint16_t> &x) {
    std::vector<double> out(x.size());
    for (size_t i = 0; i < x.size(); ++i) out[i] = static_cast<double>(x[i]);
    return out;
}

uint16_t f32_to_f16_bits_host(float value) {
    const __half h = __float2half(value);
    uint16_t bits;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}

float f16_bits_to_f32_host(uint16_t bits) {
    __half h;
    std::memcpy(&h, &bits, sizeof(bits));
    return __half2float(h);
}

__host__ __device__ int kv3_qmin(int signedness) {
    return signedness == kKv3Signed ? -4 : 0;
}

__host__ __device__ int kv3_qmax(int signedness) {
    return signedness == kKv3Signed ? 3 : 7;
}

__host__ __device__ long long kv3_packed_bytes(long long head_dim) {
    return (head_dim * 3 + 7) / 8;
}

__host__ __device__ unsigned unpack3_bits(const uint8_t *bytes,
                                          long long element) {
    const long long packet = (element >> 3) * 3;
    const int offset = static_cast<int>((element & 7) * 3);
    const unsigned raw = unsigned(bytes[packet]) |
                         (unsigned(bytes[packet + 1]) << 8) |
                         (unsigned(bytes[packet + 2]) << 16);
    return (raw >> offset) & 7u;
}

__host__ __device__ void pack3_bits(uint8_t *bytes, long long element,
                                    unsigned code) {
    const long long packet = (element >> 3) * 3;
    const int offset = static_cast<int>((element & 7) * 3);
    const unsigned shifted = (code & 7u) << offset;
    bytes[packet] |= static_cast<uint8_t>(shifted);
    bytes[packet + 1] |= static_cast<uint8_t>(shifted >> 8);
    bytes[packet + 2] |= static_cast<uint8_t>(shifted >> 16);
}

__host__ __device__ int kv3_decode_code(unsigned raw, int signedness) {
    return signedness == kKv3Signed && raw >= 4u ? static_cast<int>(raw) - 8
                                                 : static_cast<int>(raw);
}

__host__ __device__ unsigned kv3_encode_code(int code) {
    return static_cast<unsigned>(code) & 7u;
}

__device__ __forceinline__ void kv3_quant_params_device(
    float mn, float mx, int qmin, int qmax, bool integer_zero, int signedness,
    float *scale, int *zero) {
    if (integer_zero) {
        *scale = mx == mn ? 0.0f : (mx - mn) / static_cast<float>(qmax - qmin);
        *zero = *scale == 0.0f
                    ? 0
                    : max(qmin, min(qmax, static_cast<int>(
                                              nearbyintf(qmin - mn / *scale))));
    } else if (signedness == kKv3Signed) {
        *scale = fmaxf(mx > 0.0f ? mx / qmax : 0.0f,
                       mn < 0.0f ? mn / qmin : 0.0f);
        *zero = 0;
    } else {
        *scale = mx > 0.0f ? mx / qmax : 0.0f;
        *zero = 0;
    }
}

__device__ float load_scale_device(const void *scales, long long index,
                                   int type) {
    if (type == kKv3Fp32) return static_cast<const float *>(scales)[index];
    return __half2float(
        __ushort_as_half(static_cast<const uint16_t *>(scales)[index]));
}

__device__ float store_scale_device(void *scales, long long index, int type,
                                    float value) {
    if (type == kKv3Fp32) {
        static_cast<float *>(scales)[index] = value;
        return value;
    }
    const uint16_t bits = __half_as_ushort(__float2half(value));
    static_cast<uint16_t *>(scales)[index] = bits;
    return __half2float(__ushort_as_half(bits));
}

float load_scale_host_f32(const std::vector<float> &scales,
                          const std::vector<uint16_t> &, long long index) {
    return scales[index];
}

float load_scale_host_f16(const std::vector<float> &,
                          const std::vector<uint16_t> &scales,
                          long long index) {
    return f16_bits_to_f32_host(scales[index]);
}

__host__ __device__ unsigned unpack_bits(const uint8_t *bytes,
                                         long long element, int bits) {
    const long long bit_position = element * bits;
    const long long byte = bit_position / 8;
    const int offset = static_cast<int>(bit_position & 7);
    unsigned raw = bytes[byte];
    if (offset + bits > 8) raw |= unsigned(bytes[byte + 1]) << 8;
    return (raw >> offset) & ((1u << bits) - 1u);
}

void pack_bits_host(std::vector<uint8_t> &bytes, long long element, int bits,
                    unsigned code) {
    const long long bit_position = element * bits;
    const long long byte = bit_position / 8;
    const int offset = static_cast<int>(bit_position & 7);
    const unsigned mask = (1u << bits) - 1u;
    unsigned raw = (code & mask) << offset;
    bytes[byte] |= static_cast<uint8_t>(raw);
    if (offset + bits > 8) bytes[byte + 1] |= static_cast<uint8_t>(raw >> 8);
}

__host__ __device__ void fwht_inplace(float *values, long long count) {
    for (long long width = 1; width < count; width <<= 1) {
        for (long long base = 0; base < count; base += 2 * width) {
            for (long long item = 0; item < width; ++item) {
                const float a = values[base + item];
                const float b = values[base + width + item];
                values[base + item] = a + b;
                values[base + width + item] = a - b;
            }
        }
    }
}

__host__ __device__ float softcap_score(float score, float softcap) {
    return softcap > 0.0f ? softcap * tanhf(score / softcap) : score;
}

__device__ float qlite_dequant(const uint8_t *row, int format, int column) {
    if (format == kFmtQ8_0) {
        const float scale = __half2float(*reinterpret_cast<const __half *>(row));
        return scale * static_cast<float>(
                           reinterpret_cast<const int8_t *>(row + 2)[column]);
    }
    const float scale = __half2float(*reinterpret_cast<const __half *>(row));
    const uint8_t *qs = row + 2;
    const int nibble =
        column < 16 ? (qs[column] & 0x0f) : (qs[column - 16] >> 4);
    return scale * static_cast<float>(nibble - 8);
}

double qlite_dequant_host(const uint8_t *row, int format, int column) {
    const double scale = f16_bits_to_f32_host(
        uint16_t(row[0]) | (uint16_t(row[1]) << 8));
    if (format == kFmtQ8_0) {
        return scale * static_cast<double>(
                           reinterpret_cast<const int8_t *>(row + 2)[column]);
    }
    const uint8_t *qs = row + 2;
    const int nibble =
        column < 16 ? (qs[column] & 0x0f) : (qs[column - 16] >> 4);
    return scale * static_cast<double>(nibble - 8);
}

__host__ __device__ int qlite_block_bytes(int format) {
    return format == kFmtQ8_0 ? 34 : 18;
}

// ---------------------------------------------------------------------------
// BitNet-KV3 cache codec
// ---------------------------------------------------------------------------

__global__ void kv3_scatter_kernel(
    const float *key, const float *value, const int32_t *slots,
    uint8_t *key_cache, uint8_t *value_cache, void *key_scale_cache,
    void *value_scale_cache, int32_t *key_zero_cache,
    int32_t *value_zero_cache, long long max_slots, long long count,
    long long heads, long long head_dim, Kv3Config config, int32_t *invalid) {
    const long long task = blockIdx.x * blockDim.x + threadIdx.x;
    if (task >= max_slots * heads) return;
    const long long slot = task / heads;
    const long long head = task - slot * heads;
    long long token = -1;
    for (long long t = count - 1; t >= 0; --t) {
        if (slots[t] == slot) {
            token = t;
            break;
        }
    }
    if (token < 0) return;

    const long long packed = kv3_packed_bytes(head_dim);
    const long long groups = head_dim / config.group_size;
    const long long cache_row = slot * heads + head;
    uint8_t *key_codes = key_cache + cache_row * packed;
    uint8_t *value_codes = value_cache + cache_row * packed;
    for (long long i = 0; i < packed; ++i) {
        key_codes[i] = 0;
        value_codes[i] = 0;
    }

    const long long source_base = (token * heads + head) * head_dim;
    const int qmin = kv3_qmin(config.signedness);
    const int qmax = kv3_qmax(config.signedness);
    const bool integer_zero = config.zero_mode == kKv3IntegerZero;
    for (long long group = 0; group < groups; ++group) {
        const long long group_base = group * config.group_size;
        float key_min = INFINITY, key_max = -INFINITY;
        float value_min = INFINITY, value_max = -INFINITY;
        bool bad = false;
        for (long long i = 0; i < config.group_size; ++i) {
            const float kv = key[source_base + group_base + i];
            const float vv = value[source_base + group_base + i];
            if (!isfinite(kv) || !isfinite(vv)) bad = true;
            key_min = fminf(key_min, kv);
            key_max = fmaxf(key_max, kv);
            value_min = fminf(value_min, vv);
            value_max = fmaxf(value_max, vv);
        }
        if (bad) {
            atomicExch(invalid, 1);
            continue;
        }
        float key_scale = 0.0f, value_scale = 0.0f;
        int key_zero = 0, value_zero = 0;
        kv3_quant_params_device(key_min, key_max, qmin, qmax, integer_zero,
                                config.signedness, &key_scale, &key_zero);
        kv3_quant_params_device(value_min, value_max, qmin, qmax, integer_zero,
                                config.signedness, &value_scale, &value_zero);
        const long long meta = (cache_row * groups + group);
        key_scale =
            store_scale_device(key_scale_cache, meta, config.scale_type, key_scale);
        value_scale = store_scale_device(value_scale_cache, meta,
                                         config.scale_type, value_scale);
        if (!isfinite(key_scale) || !isfinite(value_scale)) {
            atomicExch(invalid, 1);
            continue;
        }
        if (integer_zero) {
            key_zero_cache[meta] = key_zero;
            value_zero_cache[meta] = value_zero;
        }
        for (long long i = 0; i < config.group_size; ++i) {
            const long long dim = group_base + i;
            const float kv = key[source_base + dim];
            const float vv = value[source_base + dim];
            const int kc = key_scale == 0.0f
                               ? key_zero
                               : max(qmin, min(qmax,
                                               static_cast<int>(
                                                   nearbyintf(kv / key_scale)) +
                                                   key_zero));
            const int vc = value_scale == 0.0f
                               ? value_zero
                               : max(qmin, min(qmax,
                                               static_cast<int>(nearbyintf(
                                                   vv / value_scale)) +
                                                   value_zero));
            pack3_bits(key_codes, dim, kv3_encode_code(kc));
            pack3_bits(value_codes, dim, kv3_encode_code(vc));
        }
    }
}

__global__ void kv3_scatter_scalar_kernel(
    const float *key, const float *value, const int32_t *slots,
    uint8_t *key_cache, uint8_t *value_cache, void *key_scale_cache,
    void *value_scale_cache, int32_t *key_zero_cache,
    int32_t *value_zero_cache, long long max_slots, long long count,
    long long heads, long long head_dim, Kv3Config config, int32_t *invalid) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (long long task = 0; task < max_slots * heads; ++task) {
        const long long slot = task / heads;
        const long long head = task - slot * heads;
        long long token = -1;
        for (long long t = count - 1; t >= 0; --t) {
            if (slots[t] == slot) {
                token = t;
                break;
            }
        }
        if (token < 0) continue;
        const long long packed = kv3_packed_bytes(head_dim);
        const long long groups = head_dim / config.group_size;
        const long long cache_row = slot * heads + head;
        uint8_t *key_codes = key_cache + cache_row * packed;
        uint8_t *value_codes = value_cache + cache_row * packed;
        for (long long i = 0; i < packed; ++i) {
            key_codes[i] = 0;
            value_codes[i] = 0;
        }
        const long long source_base = (token * heads + head) * head_dim;
        const int qmin = kv3_qmin(config.signedness);
        const int qmax = kv3_qmax(config.signedness);
        const bool integer_zero = config.zero_mode == kKv3IntegerZero;
        for (long long group = 0; group < groups; ++group) {
            const long long group_base = group * config.group_size;
            float key_min = INFINITY, key_max = -INFINITY;
            float value_min = INFINITY, value_max = -INFINITY;
            for (long long i = 0; i < config.group_size; ++i) {
                const float kv = key[source_base + group_base + i];
                const float vv = value[source_base + group_base + i];
                if (!isfinite(kv) || !isfinite(vv)) atomicExch(invalid, 1);
                key_min = fminf(key_min, kv);
                key_max = fmaxf(key_max, kv);
                value_min = fminf(value_min, vv);
                value_max = fmaxf(value_max, vv);
            }
            float key_scale = 0.0f, value_scale = 0.0f;
            int key_zero = 0, value_zero = 0;
            kv3_quant_params_device(key_min, key_max, qmin, qmax,
                                    integer_zero, config.signedness,
                                    &key_scale, &key_zero);
            kv3_quant_params_device(value_min, value_max, qmin, qmax,
                                    integer_zero, config.signedness,
                                    &value_scale, &value_zero);
            const long long meta = cache_row * groups + group;
            key_scale = store_scale_device(key_scale_cache, meta,
                                           config.scale_type, key_scale);
            value_scale = store_scale_device(value_scale_cache, meta,
                                             config.scale_type, value_scale);
            if (integer_zero) {
                key_zero_cache[meta] = key_zero;
                value_zero_cache[meta] = value_zero;
            }
            for (long long i = 0; i < config.group_size; ++i) {
                const long long dim = group_base + i;
                const int kc = key_scale == 0.0f
                                   ? key_zero
                                   : max(qmin, min(qmax,
                                                   static_cast<int>(nearbyintf(
                                                       key[source_base + dim] /
                                                       key_scale)) +
                                                       key_zero));
                const int vc = value_scale == 0.0f
                                   ? value_zero
                                   : max(qmin, min(qmax,
                                                   static_cast<int>(nearbyintf(
                                                       value[source_base + dim] /
                                                       value_scale)) +
                                                       value_zero));
                pack3_bits(key_codes, dim, kv3_encode_code(kc));
                pack3_bits(value_codes, dim, kv3_encode_code(vc));
            }
        }
    }
}

__global__ void kv3_gather_kernel(
    const uint8_t *key_cache, const uint8_t *value_cache,
    const int32_t *indices, const void *key_scale_cache,
    const void *value_scale_cache, const int32_t *key_zero_cache,
    const int32_t *value_zero_cache, float *key_out, float *value_out,
    long long count, long long heads, long long head_dim, Kv3Config config) {
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = count * heads * head_dim;
    if (idx >= total) return;
    const long long dim = idx % head_dim;
    const long long head = (idx / head_dim) % heads;
    const long long row = idx / (head_dim * heads);
    const long long slot = indices[row];
    const long long packed = kv3_packed_bytes(head_dim);
    const long long groups = head_dim / config.group_size;
    const long long cache_row = slot * heads + head;
    const long long group = dim / config.group_size;
    const long long meta = cache_row * groups + group;
    const int key_zero =
        config.zero_mode == kKv3IntegerZero ? key_zero_cache[meta] : 0;
    const int value_zero =
        config.zero_mode == kKv3IntegerZero ? value_zero_cache[meta] : 0;
    const float key_scale =
        load_scale_device(key_scale_cache, meta, config.scale_type);
    const float value_scale =
        load_scale_device(value_scale_cache, meta, config.scale_type);
    const uint8_t *key_codes = key_cache + cache_row * packed;
    const uint8_t *value_codes = value_cache + cache_row * packed;
    key_out[idx] = (kv3_decode_code(unpack3_bits(key_codes, dim),
                                    config.signedness) -
                    key_zero) *
                   key_scale;
    value_out[idx] = (kv3_decode_code(unpack3_bits(value_codes, dim),
                                      config.signedness) -
                      value_zero) *
                     value_scale;
}

__global__ void kv3_gather_scalar_kernel(
    const uint8_t *key_cache, const uint8_t *value_cache,
    const int32_t *indices, const void *key_scale_cache,
    const void *value_scale_cache, const int32_t *key_zero_cache,
    const int32_t *value_zero_cache, float *key_out, float *value_out,
    long long count, long long heads, long long head_dim, Kv3Config config) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (long long idx = 0; idx < count * heads * head_dim; ++idx) {
        const long long dim = idx % head_dim;
        const long long head = (idx / head_dim) % heads;
        const long long row = idx / (head_dim * heads);
        const long long slot = indices[row];
        const long long packed = kv3_packed_bytes(head_dim);
        const long long groups = head_dim / config.group_size;
        const long long cache_row = slot * heads + head;
        const long long group = dim / config.group_size;
        const long long meta = cache_row * groups + group;
        const int key_zero =
            config.zero_mode == kKv3IntegerZero ? key_zero_cache[meta] : 0;
        const int value_zero =
            config.zero_mode == kKv3IntegerZero ? value_zero_cache[meta] : 0;
        const float key_scale =
            load_scale_device(key_scale_cache, meta, config.scale_type);
        const float value_scale =
            load_scale_device(value_scale_cache, meta, config.scale_type);
        const uint8_t *key_codes = key_cache + cache_row * packed;
        const uint8_t *value_codes = value_cache + cache_row * packed;
        key_out[idx] = (kv3_decode_code(unpack3_bits(key_codes, dim),
                                        config.signedness) -
                        key_zero) *
                       key_scale;
        value_out[idx] = (kv3_decode_code(unpack3_bits(value_codes, dim),
                                          config.signedness) -
                          value_zero) *
                         value_scale;
    }
}

__device__ float kv3_dequant_value(const uint8_t *cache,
                                   const void *scale_cache,
                                   const int32_t *zero_cache, long long slot,
                                   long long head, long long dim,
                                   long long heads, long long head_dim,
                                   Kv3Config config) {
    const long long packed = kv3_packed_bytes(head_dim);
    const long long groups = head_dim / config.group_size;
    const long long cache_row = slot * heads + head;
    const long long group = dim / config.group_size;
    const long long meta = cache_row * groups + group;
    const int zero =
        config.zero_mode == kKv3IntegerZero ? zero_cache[meta] : 0;
    const uint8_t *codes = cache + cache_row * packed;
    return (kv3_decode_code(unpack3_bits(codes, dim), config.signedness) -
            zero) *
           load_scale_device(scale_cache, meta, config.scale_type);
}

__global__ void paged_attention_bitnet_kernel(
    const float *q, const uint8_t *key_cache, const uint8_t *value_cache,
    const void *key_scale_cache, const void *value_scale_cache,
    const int32_t *key_zero_cache, const int32_t *value_zero_cache,
    const int32_t *block_table, const int32_t *context_lens, float *out,
    long long batch, long long query_heads, long long kv_heads,
    long long head_dim, long long page_size, long long max_blocks,
    Kv3Config config, float scale, long long window) {
    const long long item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= batch * query_heads) return;
    const long long request = item / query_heads;
    const long long qhead = item - request * query_heads;
    const long long kvhead = qhead / (query_heads / kv_heads);
    const long long context = context_lens[request];
    const long long start = window > 0 ? max(0LL, context - window) : 0;
    const float score_scale =
        scale > 0.0f ? scale : rsqrtf(static_cast<float>(head_dim));
    const float *query = q + item * head_dim;
    float acc[256];
    for (long long d = 0; d < head_dim; ++d) acc[d] = 0.0f;
    float maximum = -INFINITY;
    double denominator = 0.0;
    float scores[kScoreTile];
    long long slots[kScoreTile];
    for (long long tile = start; tile < context; tile += kScoreTile) {
        const long long tile_end = min(context, tile + kScoreTile);
        long long valid = 0;
        float tile_max = -INFINITY;
        for (long long position = tile; position < tile_end; ++position) {
            const int32_t physical =
                block_table[request * max_blocks + position / page_size];
            const long long slot =
                static_cast<long long>(physical) * page_size +
                position % page_size;
            double dot = 0.0;
            for (long long d = 0; d < head_dim; ++d)
                dot += query[d] *
                       kv3_dequant_value(key_cache, key_scale_cache,
                                         key_zero_cache, slot, kvhead, d,
                                         kv_heads, head_dim, config);
            scores[valid] = static_cast<float>(dot * score_scale);
            slots[valid] = slot;
            tile_max = fmaxf(tile_max, scores[valid]);
            ++valid;
        }
        const float next_max = fmaxf(maximum, tile_max);
        const double old_weight =
            denominator > 0.0 ? exp(static_cast<double>(maximum - next_max))
                              : 0.0;
        denominator *= old_weight;
        if (old_weight != 1.0) {
            for (long long d = 0; d < head_dim; ++d)
                acc[d] *= static_cast<float>(old_weight);
        }
        for (long long i = 0; i < valid; ++i) {
            const double weight =
                exp(static_cast<double>(scores[i] - next_max));
            denominator += weight;
            for (long long d = 0; d < head_dim; ++d)
                acc[d] += static_cast<float>(
                    weight *
                    kv3_dequant_value(value_cache, value_scale_cache,
                                      value_zero_cache, slots[i], kvhead, d,
                                      kv_heads, head_dim, config));
        }
        maximum = next_max;
    }
    const double inv = denominator > 0.0 ? 1.0 / denominator : 0.0;
    for (long long d = 0; d < head_dim; ++d)
        out[item * head_dim + d] = static_cast<float>(acc[d] * inv);
}

__global__ void paged_attention_bitnet_scalar_kernel(
    const float *q, const uint8_t *key_cache, const uint8_t *value_cache,
    const void *key_scale_cache, const void *value_scale_cache,
    const int32_t *key_zero_cache, const int32_t *value_zero_cache,
    const int32_t *block_table, const int32_t *context_lens, float *out,
    long long batch, long long query_heads, long long kv_heads,
    long long head_dim, long long page_size, long long max_blocks,
    Kv3Config config, float scale, long long window) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (long long item = 0; item < batch * query_heads; ++item) {
        const long long request = item / query_heads;
        const long long qhead = item - request * query_heads;
        const long long kvhead = qhead / (query_heads / kv_heads);
        const long long context = context_lens[request];
        const long long start = window > 0 ? max(0LL, context - window) : 0;
        const float score_scale =
            scale > 0.0f ? scale : rsqrtf(static_cast<float>(head_dim));
        const float *query = q + item * head_dim;
        float acc[256];
        for (long long d = 0; d < head_dim; ++d) acc[d] = 0.0f;
        float maximum = -INFINITY;
        double denominator = 0.0;
        float scores[kScoreTile];
        long long slots[kScoreTile];
        for (long long tile = start; tile < context; tile += kScoreTile) {
            const long long tile_end = min(context, tile + kScoreTile);
            long long valid = 0;
            float tile_max = -INFINITY;
            for (long long position = tile; position < tile_end; ++position) {
                const int32_t physical =
                    block_table[request * max_blocks + position / page_size];
                const long long slot =
                    static_cast<long long>(physical) * page_size +
                    position % page_size;
                double dot = 0.0;
                for (long long d = 0; d < head_dim; ++d)
                    dot += query[d] *
                           kv3_dequant_value(key_cache, key_scale_cache,
                                             key_zero_cache, slot, kvhead, d,
                                             kv_heads, head_dim, config);
                scores[valid] = static_cast<float>(dot * score_scale);
                slots[valid] = slot;
                tile_max = fmaxf(tile_max, scores[valid]);
                ++valid;
            }
            const float next_max = fmaxf(maximum, tile_max);
            const double old_weight =
                denominator > 0.0 ? exp(static_cast<double>(maximum - next_max))
                                  : 0.0;
            denominator *= old_weight;
            if (old_weight != 1.0) {
                for (long long d = 0; d < head_dim; ++d)
                    acc[d] *= static_cast<float>(old_weight);
            }
            for (long long i = 0; i < valid; ++i) {
                const double weight =
                    exp(static_cast<double>(scores[i] - next_max));
                denominator += weight;
                for (long long d = 0; d < head_dim; ++d)
                    acc[d] += static_cast<float>(
                        weight * kv3_dequant_value(
                                     value_cache, value_scale_cache,
                                     value_zero_cache, slots[i], kvhead, d,
                                     kv_heads, head_dim, config));
            }
            maximum = next_max;
        }
        const double inv = denominator > 0.0 ? 1.0 / denominator : 0.0;
        for (long long d = 0; d < head_dim; ++d)
            out[item * head_dim + d] = static_cast<float>(acc[d] * inv);
    }
}

// ---------------------------------------------------------------------------
// TurboQuant decode
// ---------------------------------------------------------------------------

__device__ float turbo_dequant_key(const uint8_t *key_cache,
                                   const float *key_scale_cache,
                                   const float *key_zero_cache, long long slot,
                                   long long head, long long dim,
                                   long long heads, long long head_dim,
                                   int key_bits, bool key_signed) {
    const long long groups = head_dim / 32;
    const long long key_bytes = (head_dim * key_bits + 7) / 8;
    const long long row = slot * heads + head;
    const uint8_t *codes = key_cache + row * key_bytes;
    const unsigned code = unpack_bits(codes, dim, key_bits);
    const float quantized =
        key_signed && key_bits == 8
            ? static_cast<float>(static_cast<int8_t>(code))
            : static_cast<float>(code);
    const long long meta = row * groups + dim / 32;
    return (quantized + key_zero_cache[meta]) * key_scale_cache[meta];
}

__device__ float turbo_dequant_value_rotated(
    const uint8_t *value_cache, const float *value_scale_cache,
    const float *value_centroids, long long slot, long long head, long long dim,
    long long heads, long long head_dim, int value_bits) {
    const long long groups = head_dim / 32;
    const long long value_bytes = (head_dim * value_bits + 7) / 8;
    const long long row = slot * heads + head;
    const uint8_t *codes = value_cache + row * value_bytes;
    const unsigned code = unpack_bits(codes, dim, value_bits);
    const long long meta = row * groups + dim / 32;
    return value_centroids[code] * value_scale_cache[meta];
}

__global__ void paged_attention_turboquant_kernel(
    const float *q, const uint8_t *key_cache, const uint8_t *value_cache,
    const float *key_scale_cache, const float *value_scale_cache,
    const float *key_zero_cache, const float *value_centroids,
    const float *signs, const int32_t *block_table,
    const int32_t *context_lens, float *out, long long batch,
    long long query_heads, long long kv_heads, long long head_dim,
    long long page_size, long long max_blocks, int key_bits, bool key_signed,
    int value_bits, float scale, long long window) {
    const long long item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= batch * query_heads) return;
    const long long request = item / query_heads;
    const long long qhead = item - request * query_heads;
    const long long kvhead = qhead / (query_heads / kv_heads);
    const long long context = context_lens[request];
    const long long start = window > 0 ? max(0LL, context - window) : 0;
    const float score_scale =
        scale > 0.0f ? scale : rsqrtf(static_cast<float>(head_dim));
    const float norm = rsqrtf(static_cast<float>(head_dim));
    const float *query = q + item * head_dim;
    float rotated[256];
    for (long long d = 0; d < head_dim; ++d) rotated[d] = 0.0f;
    float maximum = -INFINITY;
    double denominator = 0.0;
    float scores[kScoreTile];
    long long slots[kScoreTile];
    for (long long tile = start; tile < context; tile += kScoreTile) {
        const long long tile_end = min(context, tile + kScoreTile);
        long long valid = 0;
        float tile_max = -INFINITY;
        for (long long position = tile; position < tile_end; ++position) {
            const int32_t physical =
                block_table[request * max_blocks + position / page_size];
            const long long slot =
                static_cast<long long>(physical) * page_size +
                position % page_size;
            double dot = 0.0;
            for (long long d = 0; d < head_dim; ++d)
                dot += query[d] * turbo_dequant_key(key_cache, key_scale_cache,
                                                    key_zero_cache, slot,
                                                    kvhead, d, kv_heads,
                                                    head_dim, key_bits,
                                                    key_signed);
            scores[valid] = static_cast<float>(dot * score_scale);
            slots[valid] = slot;
            tile_max = fmaxf(tile_max, scores[valid]);
            ++valid;
        }
        const float next_max = fmaxf(maximum, tile_max);
        const double old_weight =
            denominator > 0.0 ? exp(static_cast<double>(maximum - next_max))
                              : 0.0;
        denominator *= old_weight;
        if (old_weight != 1.0) {
            for (long long d = 0; d < head_dim; ++d)
                rotated[d] *= static_cast<float>(old_weight);
        }
        for (long long i = 0; i < valid; ++i) {
            const double weight =
                exp(static_cast<double>(scores[i] - next_max));
            denominator += weight;
            for (long long d = 0; d < head_dim; ++d)
                rotated[d] += static_cast<float>(
                    weight * turbo_dequant_value_rotated(
                                 value_cache, value_scale_cache,
                                 value_centroids, slots[i], kvhead, d,
                                 kv_heads, head_dim, value_bits));
        }
        maximum = next_max;
    }
    const double inv = denominator > 0.0 ? 1.0 / denominator : 0.0;
    if (denominator > 0.0) {
        for (long long d = 0; d < head_dim; ++d)
            rotated[d] = static_cast<float>(rotated[d] * inv);
        fwht_inplace(rotated, head_dim);
        for (long long d = 0; d < head_dim; ++d)
            out[item * head_dim + d] = rotated[d] * norm * signs[d];
    } else {
        for (long long d = 0; d < head_dim; ++d)
            out[item * head_dim + d] = 0.0f;
    }
}

__global__ void paged_attention_turboquant_scalar_kernel(
    const float *q, const uint8_t *key_cache, const uint8_t *value_cache,
    const float *key_scale_cache, const float *value_scale_cache,
    const float *key_zero_cache, const float *value_centroids,
    const float *signs, const int32_t *block_table,
    const int32_t *context_lens, float *out, long long batch,
    long long query_heads, long long kv_heads, long long head_dim,
    long long page_size, long long max_blocks, int key_bits, bool key_signed,
    int value_bits, float scale, long long window) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (long long item = 0; item < batch * query_heads; ++item) {
        const long long request = item / query_heads;
        const long long qhead = item - request * query_heads;
        const long long kvhead = qhead / (query_heads / kv_heads);
        const long long context = context_lens[request];
        const long long start = window > 0 ? max(0LL, context - window) : 0;
        const float score_scale =
            scale > 0.0f ? scale : rsqrtf(static_cast<float>(head_dim));
        const float norm = rsqrtf(static_cast<float>(head_dim));
        const float *query = q + item * head_dim;
        float rotated[256];
        for (long long d = 0; d < head_dim; ++d) rotated[d] = 0.0f;
        float maximum = -INFINITY;
        double denominator = 0.0;
        float scores[kScoreTile];
        long long slots[kScoreTile];
        for (long long tile = start; tile < context; tile += kScoreTile) {
            const long long tile_end = min(context, tile + kScoreTile);
            long long valid = 0;
            float tile_max = -INFINITY;
            for (long long position = tile; position < tile_end; ++position) {
                const int32_t physical =
                    block_table[request * max_blocks + position / page_size];
                const long long slot =
                    static_cast<long long>(physical) * page_size +
                    position % page_size;
                double dot = 0.0;
                for (long long d = 0; d < head_dim; ++d)
                    dot += query[d] * turbo_dequant_key(
                                        key_cache, key_scale_cache,
                                        key_zero_cache, slot, kvhead, d,
                                        kv_heads, head_dim, key_bits,
                                        key_signed);
                scores[valid] = static_cast<float>(dot * score_scale);
                slots[valid] = slot;
                tile_max = fmaxf(tile_max, scores[valid]);
                ++valid;
            }
            const float next_max = fmaxf(maximum, tile_max);
            const double old_weight =
                denominator > 0.0
                    ? exp(static_cast<double>(maximum - next_max))
                    : 0.0;
            denominator *= old_weight;
            if (old_weight != 1.0) {
                for (long long d = 0; d < head_dim; ++d)
                    rotated[d] *= static_cast<float>(old_weight);
            }
            for (long long i = 0; i < valid; ++i) {
                const double weight =
                    exp(static_cast<double>(scores[i] - next_max));
                denominator += weight;
                for (long long d = 0; d < head_dim; ++d)
                    rotated[d] += static_cast<float>(
                        weight * turbo_dequant_value_rotated(
                                     value_cache, value_scale_cache,
                                     value_centroids, slots[i], kvhead, d,
                                     kv_heads, head_dim, value_bits));
            }
            maximum = next_max;
        }
        const double inv = denominator > 0.0 ? 1.0 / denominator : 0.0;
        if (denominator > 0.0) {
            for (long long d = 0; d < head_dim; ++d)
                rotated[d] = static_cast<float>(rotated[d] * inv);
            fwht_inplace(rotated, head_dim);
            for (long long d = 0; d < head_dim; ++d)
                out[item * head_dim + d] = rotated[d] * norm * signs[d];
        } else {
            for (long long d = 0; d < head_dim; ++d)
                out[item * head_dim + d] = 0.0f;
        }
    }
}

// ---------------------------------------------------------------------------
// Advanced paged attention
// ---------------------------------------------------------------------------

__global__ void paged_attention_advanced_kernel(
    const float *q, const float *key_cache, const float *value_cache,
    const int32_t *block_table, const int32_t *context_lens,
    const int32_t *block_mask, const float *alibi_slopes,
    const float *sinks, float *out, long long batch, long long query_heads,
    long long kv_heads, long long head_dim, long long page_size,
    long long max_blocks, float scale, long long window, float softcap) {
    const long long item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= batch * query_heads) return;
    const long long request = item / query_heads;
    const long long qhead = item - request * query_heads;
    const long long kvhead = qhead / (query_heads / kv_heads);
    const long long context = context_lens[request];
    const long long start = window > 0 ? max(0LL, context - window) : 0;
    const float score_scale =
        scale > 0.0f ? scale : rsqrtf(static_cast<float>(head_dim));
    const float *query = q + item * head_dim;
    float *output = out + item * head_dim;
    for (long long d = 0; d < head_dim; ++d) output[d] = 0.0f;
    float maximum = sinks == nullptr ? -INFINITY : sinks[qhead];
    double denominator = sinks == nullptr ? 0.0 : 1.0;
    float scores[kScoreTile];
    long long bases[kScoreTile];
    for (long long tile = start; tile < context; tile += kScoreTile) {
        const long long tile_end = min(context, tile + kScoreTile);
        long long valid = 0;
        float tile_max = -INFINITY;
        for (long long position = tile; position < tile_end; ++position) {
            const long long logical_block = position / page_size;
            if (block_mask != nullptr &&
                block_mask[(request * query_heads + qhead) * max_blocks +
                           logical_block] == 0)
                continue;
            const int32_t physical =
                block_table[request * max_blocks + logical_block];
            const long long base =
                ((static_cast<long long>(physical) * page_size +
                  position % page_size) *
                     kv_heads +
                 kvhead) *
                head_dim;
            double dot = 0.0;
            for (long long d = 0; d < head_dim; ++d)
                dot += query[d] * key_cache[base + d];
            dot *= score_scale;
            if (alibi_slopes != nullptr)
                dot += alibi_slopes[qhead] * (position - (context - 1));
            const float score = softcap_score(static_cast<float>(dot), softcap);
            scores[valid] = score;
            bases[valid] = base;
            tile_max = fmaxf(tile_max, score);
            ++valid;
        }
        if (valid == 0) continue;
        const float next_max = fmaxf(maximum, tile_max);
        const double old_weight =
            denominator > 0.0 ? exp(static_cast<double>(maximum - next_max))
                              : 0.0;
        denominator *= old_weight;
        if (old_weight != 1.0) {
            for (long long d = 0; d < head_dim; ++d)
                output[d] *= static_cast<float>(old_weight);
        }
        for (long long i = 0; i < valid; ++i) {
            const double weight =
                exp(static_cast<double>(scores[i] - next_max));
            denominator += weight;
            const long long base = bases[i];
            for (long long d = 0; d < head_dim; ++d)
                output[d] +=
                    static_cast<float>(weight * value_cache[base + d]);
        }
        maximum = next_max;
    }
    if (denominator > 0.0) {
        const float inv = static_cast<float>(1.0 / denominator);
        for (long long d = 0; d < head_dim; ++d) output[d] *= inv;
    }
}

__global__ void paged_attention_advanced_scalar_kernel(
    const float *q, const float *key_cache, const float *value_cache,
    const int32_t *block_table, const int32_t *context_lens,
    const int32_t *block_mask, const float *alibi_slopes,
    const float *sinks, float *out, long long batch, long long query_heads,
    long long kv_heads, long long head_dim, long long page_size,
    long long max_blocks, float scale, long long window, float softcap) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (long long item = 0; item < batch * query_heads; ++item) {
        const long long request = item / query_heads;
        const long long qhead = item - request * query_heads;
        const long long kvhead = qhead / (query_heads / kv_heads);
        const long long context = context_lens[request];
        const long long start = window > 0 ? max(0LL, context - window) : 0;
        const float score_scale =
            scale > 0.0f ? scale : rsqrtf(static_cast<float>(head_dim));
        const float *query = q + item * head_dim;
        float *output = out + item * head_dim;
        for (long long d = 0; d < head_dim; ++d) output[d] = 0.0f;
        float maximum = sinks == nullptr ? -INFINITY : sinks[qhead];
        double denominator = sinks == nullptr ? 0.0 : 1.0;
        float scores[kScoreTile];
        long long bases[kScoreTile];
        for (long long tile = start; tile < context; tile += kScoreTile) {
            const long long tile_end = min(context, tile + kScoreTile);
            long long valid = 0;
            float tile_max = -INFINITY;
            for (long long position = tile; position < tile_end; ++position) {
                const long long logical_block = position / page_size;
                if (block_mask != nullptr &&
                    block_mask[(request * query_heads + qhead) * max_blocks +
                               logical_block] == 0)
                    continue;
                const int32_t physical =
                    block_table[request * max_blocks + logical_block];
                const long long base =
                    ((static_cast<long long>(physical) * page_size +
                      position % page_size) *
                         kv_heads +
                     kvhead) *
                    head_dim;
                double dot = 0.0;
                for (long long d = 0; d < head_dim; ++d)
                    dot += query[d] * key_cache[base + d];
                dot *= score_scale;
                if (alibi_slopes != nullptr)
                    dot += alibi_slopes[qhead] * (position - (context - 1));
                const float score =
                    softcap_score(static_cast<float>(dot), softcap);
                scores[valid] = score;
                bases[valid] = base;
                tile_max = fmaxf(tile_max, score);
                ++valid;
            }
            if (valid == 0) continue;
            const float next_max = fmaxf(maximum, tile_max);
            const double old_weight =
                denominator > 0.0
                    ? exp(static_cast<double>(maximum - next_max))
                    : 0.0;
            denominator *= old_weight;
            if (old_weight != 1.0) {
                for (long long d = 0; d < head_dim; ++d)
                    output[d] *= static_cast<float>(old_weight);
            }
            for (long long i = 0; i < valid; ++i) {
                const double weight =
                    exp(static_cast<double>(scores[i] - next_max));
                denominator += weight;
                const long long base = bases[i];
                for (long long d = 0; d < head_dim; ++d)
                    output[d] +=
                        static_cast<float>(weight * value_cache[base + d]);
            }
            maximum = next_max;
        }
        if (denominator > 0.0) {
            const float inv = static_cast<float>(1.0 / denominator);
            for (long long d = 0; d < head_dim; ++d) output[d] *= inv;
        }
    }
}

// ---------------------------------------------------------------------------
// Quantized prefill attention
// ---------------------------------------------------------------------------

__global__ void quantized_attention_kernel(
    const float *q, const uint8_t *packed_k, const uint8_t *packed_v,
    float *out, long long batch, long long heads, long long sequence,
    long long head_dim, int format, bool causal) {
    const long long item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= batch * heads * sequence) return;
    const long long query_pos = item % sequence;
    const long long prefix = item / sequence;
    const long long key_count = causal ? query_pos + 1 : sequence;
    const long long block_bytes = qlite_block_bytes(format);
    const long long blocks_per_row = head_dim / 32;
    const long long row_bytes = blocks_per_row * block_bytes;
    const float score_scale = rsqrtf(static_cast<float>(head_dim));
    const float *query = q + item * head_dim;
    float maximum = -INFINITY;
    for (long long key_pos = 0; key_pos < key_count; ++key_pos) {
        const uint8_t *key =
            packed_k + (prefix * sequence + key_pos) * row_bytes;
        double dot = 0.0;
        for (long long d = 0; d < head_dim; ++d)
            dot += query[d] *
                   qlite_dequant(key + (d / 32) * block_bytes, format,
                                 static_cast<int>(d % 32));
        maximum = fmaxf(maximum, static_cast<float>(dot * score_scale));
    }
    double denominator = 0.0;
    for (long long d = 0; d < head_dim; ++d) out[item * head_dim + d] = 0.0f;
    for (long long key_pos = 0; key_pos < key_count; ++key_pos) {
        const uint8_t *key =
            packed_k + (prefix * sequence + key_pos) * row_bytes;
        const uint8_t *value =
            packed_v + (prefix * sequence + key_pos) * row_bytes;
        double dot = 0.0;
        for (long long d = 0; d < head_dim; ++d)
            dot += query[d] *
                   qlite_dequant(key + (d / 32) * block_bytes, format,
                                 static_cast<int>(d % 32));
        const double prob = exp(dot * score_scale - maximum);
        denominator += prob;
        for (long long d = 0; d < head_dim; ++d)
            out[item * head_dim + d] += static_cast<float>(
                prob * qlite_dequant(value + (d / 32) * block_bytes, format,
                                     static_cast<int>(d % 32)));
    }
    const float inv = static_cast<float>(1.0 / denominator);
    for (long long d = 0; d < head_dim; ++d) out[item * head_dim + d] *= inv;
}

__global__ void quantized_attention_scalar_kernel(
    const float *q, const uint8_t *packed_k, const uint8_t *packed_v,
    float *out, long long batch, long long heads, long long sequence,
    long long head_dim, int format, bool causal) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (long long item = 0; item < batch * heads * sequence; ++item) {
        const long long query_pos = item % sequence;
        const long long prefix = item / sequence;
        const long long key_count = causal ? query_pos + 1 : sequence;
        const long long block_bytes = qlite_block_bytes(format);
        const long long blocks_per_row = head_dim / 32;
        const long long row_bytes = blocks_per_row * block_bytes;
        const float score_scale = rsqrtf(static_cast<float>(head_dim));
        const float *query = q + item * head_dim;
        float maximum = -INFINITY;
        for (long long key_pos = 0; key_pos < key_count; ++key_pos) {
            const uint8_t *key =
                packed_k + (prefix * sequence + key_pos) * row_bytes;
            double dot = 0.0;
            for (long long d = 0; d < head_dim; ++d)
                dot += query[d] *
                       qlite_dequant(key + (d / 32) * block_bytes, format,
                                     static_cast<int>(d % 32));
            maximum = fmaxf(maximum, static_cast<float>(dot * score_scale));
        }
        double denominator = 0.0;
        for (long long d = 0; d < head_dim; ++d)
            out[item * head_dim + d] = 0.0f;
        for (long long key_pos = 0; key_pos < key_count; ++key_pos) {
            const uint8_t *key =
                packed_k + (prefix * sequence + key_pos) * row_bytes;
            const uint8_t *value =
                packed_v + (prefix * sequence + key_pos) * row_bytes;
            double dot = 0.0;
            for (long long d = 0; d < head_dim; ++d)
                dot += query[d] *
                       qlite_dequant(key + (d / 32) * block_bytes, format,
                                     static_cast<int>(d % 32));
            const double prob = exp(dot * score_scale - maximum);
            denominator += prob;
            for (long long d = 0; d < head_dim; ++d)
                out[item * head_dim + d] += static_cast<float>(
                    prob *
                    qlite_dequant(value + (d / 32) * block_bytes, format,
                                  static_cast<int>(d % 32)));
        }
        const float inv = static_cast<float>(1.0 / denominator);
        for (long long d = 0; d < head_dim; ++d)
            out[item * head_dim + d] *= inv;
    }
}

// ---------------------------------------------------------------------------
// Host helpers and oracles
// ---------------------------------------------------------------------------

struct Kv3Cache {
    std::vector<uint8_t> key_codes;
    std::vector<uint8_t> value_codes;
    std::vector<uint16_t> key_scales_f16;
    std::vector<uint16_t> value_scales_f16;
    std::vector<float> key_scales_f32;
    std::vector<float> value_scales_f32;
    std::vector<int32_t> key_zero;
    std::vector<int32_t> value_zero;
};

float kv3_load_scale_host(const Kv3Cache &cache, bool key, long long index,
                          const Kv3Config &config) {
    if (config.scale_type == kKv3Fp32)
        return key ? cache.key_scales_f32[index] : cache.value_scales_f32[index];
    return f16_bits_to_f32_host(key ? cache.key_scales_f16[index]
                                    : cache.value_scales_f16[index]);
}

void kv3_store_scale_host(Kv3Cache &cache, bool key, long long index,
                          const Kv3Config &config, float value) {
    if (config.scale_type == kKv3Fp32) {
        (key ? cache.key_scales_f32 : cache.value_scales_f32)[index] = value;
    } else {
        (key ? cache.key_scales_f16 : cache.value_scales_f16)[index] =
            f32_to_f16_bits_host(value);
    }
}

Kv3Cache kv3_prefilled(long long max_slots, long long heads, long long head_dim,
                       const Kv3Config &config) {
    const long long packed = kv3_packed_bytes(head_dim);
    const long long groups = head_dim / config.group_size;
    Kv3Cache cache;
    cache.key_codes.assign(max_slots * heads * packed, 0xa5u);
    cache.value_codes.assign(cache.key_codes.size(), 0x5au);
    cache.key_scales_f16.assign(max_slots * heads * groups, 0x3c00u);
    cache.value_scales_f16.assign(max_slots * heads * groups, 0x3800u);
    cache.key_scales_f32.assign(max_slots * heads * groups, 1.0f);
    cache.value_scales_f32.assign(max_slots * heads * groups, 0.5f);
    cache.key_zero.assign(max_slots * heads * groups, 0);
    cache.value_zero.assign(max_slots * heads * groups, 0);
    return cache;
}

void kv3_scatter_ref(const std::vector<float> &key,
                     const std::vector<float> &value,
                     const std::vector<int32_t> &slots, Kv3Cache &cache,
                     long long max_slots, long long count, long long heads,
                     long long head_dim, const Kv3Config &config) {
    const long long packed = kv3_packed_bytes(head_dim);
    const long long groups = head_dim / config.group_size;
    const int qmin = kv3_qmin(config.signedness);
    const int qmax = kv3_qmax(config.signedness);
    const bool integer_zero = config.zero_mode == kKv3IntegerZero;
    for (long long slot = 0; slot < max_slots; ++slot) {
        long long token = -1;
        for (long long t = count - 1; t >= 0; --t) {
            if (slots[t] == slot) {
                token = t;
                break;
            }
        }
        if (token < 0) continue;
        for (long long head = 0; head < heads; ++head) {
            const long long row = slot * heads + head;
            std::fill_n(cache.key_codes.data() + row * packed, packed, 0);
            std::fill_n(cache.value_codes.data() + row * packed, packed, 0);
            const long long source = (token * heads + head) * head_dim;
            for (long long group = 0; group < groups; ++group) {
                const long long group_base = group * config.group_size;
                float key_min = std::numeric_limits<float>::infinity();
                float key_max = -std::numeric_limits<float>::infinity();
                float value_min = std::numeric_limits<float>::infinity();
                float value_max = -std::numeric_limits<float>::infinity();
                for (long long i = 0; i < config.group_size; ++i) {
                    const float kv = key[source + group_base + i];
                    const float vv = value[source + group_base + i];
                    key_min = std::min(key_min, kv);
                    key_max = std::max(key_max, kv);
                    value_min = std::min(value_min, vv);
                    value_max = std::max(value_max, vv);
                }
                auto params = [&](float mn, float mx, float *scale, int *zero) {
                    if (integer_zero) {
                        *scale = mx == mn
                                     ? 0.0f
                                     : (mx - mn) /
                                           static_cast<float>(qmax - qmin);
                        *zero = *scale == 0.0f
                                    ? 0
                                    : std::clamp(static_cast<int>(std::nearbyint(
                                                     qmin - mn / *scale)),
                                                 qmin, qmax);
                    } else if (config.signedness == kKv3Signed) {
                        *scale =
                            std::max(mx > 0.0f ? mx / qmax : 0.0f,
                                     mn < 0.0f ? mn / qmin : 0.0f);
                        *zero = 0;
                    } else {
                        *scale = mx > 0.0f ? mx / qmax : 0.0f;
                        *zero = 0;
                    }
                };
                float key_scale = 0.0f, value_scale = 0.0f;
                int key_zero = 0, value_zero = 0;
                params(key_min, key_max, &key_scale, &key_zero);
                params(value_min, value_max, &value_scale, &value_zero);
                const long long meta = row * groups + group;
                kv3_store_scale_host(cache, true, meta, config, key_scale);
                kv3_store_scale_host(cache, false, meta, config, value_scale);
                key_scale = kv3_load_scale_host(cache, true, meta, config);
                value_scale = kv3_load_scale_host(cache, false, meta, config);
                if (integer_zero) {
                    cache.key_zero[meta] = key_zero;
                    cache.value_zero[meta] = value_zero;
                }
                for (long long i = 0; i < config.group_size; ++i) {
                    const long long dim = group_base + i;
                    const int kc = key_scale == 0.0f
                                       ? key_zero
                                       : std::clamp(
                                             static_cast<int>(std::nearbyint(
                                                 key[source + dim] / key_scale)) +
                                                 key_zero,
                                             qmin, qmax);
                    const int vc = value_scale == 0.0f
                                       ? value_zero
                                       : std::clamp(
                                             static_cast<int>(std::nearbyint(
                                                 value[source + dim] /
                                                 value_scale)) +
                                                 value_zero,
                                             qmin, qmax);
                    pack3_bits(cache.key_codes.data() + row * packed, dim,
                               kv3_encode_code(kc));
                    pack3_bits(cache.value_codes.data() + row * packed, dim,
                               kv3_encode_code(vc));
                }
            }
        }
    }
}

double kv3_dequant_host(const Kv3Cache &cache, bool key, long long slot,
                        long long head, long long dim, long long heads,
                        long long head_dim, const Kv3Config &config) {
    const long long packed = kv3_packed_bytes(head_dim);
    const long long groups = head_dim / config.group_size;
    const long long row = slot * heads + head;
    const long long group = dim / config.group_size;
    const long long meta = row * groups + group;
    const int zero = config.zero_mode == kKv3IntegerZero
                         ? (key ? cache.key_zero[meta] : cache.value_zero[meta])
                         : 0;
    const uint8_t *codes =
        (key ? cache.key_codes.data() : cache.value_codes.data()) + row * packed;
    return (kv3_decode_code(unpack3_bits(codes, dim), config.signedness) -
            zero) *
           kv3_load_scale_host(cache, key, meta, config);
}

void kv3_gather_ref(const Kv3Cache &cache, const std::vector<int32_t> &indices,
                    std::vector<double> &key_out,
                    std::vector<double> &value_out, long long count,
                    long long heads, long long head_dim,
                    const Kv3Config &config) {
    key_out.assign(count * heads * head_dim, 0.0);
    value_out.assign(key_out.size(), 0.0);
    for (long long row = 0; row < count; ++row) {
        const long long slot = indices[row];
        for (long long head = 0; head < heads; ++head) {
            for (long long d = 0; d < head_dim; ++d) {
                const long long out = (row * heads + head) * head_dim + d;
                key_out[out] =
                    kv3_dequant_host(cache, true, slot, head, d, heads,
                                     head_dim, config);
                value_out[out] =
                    kv3_dequant_host(cache, false, slot, head, d, heads,
                                     head_dim, config);
            }
        }
    }
}

std::vector<double> paged_bitnet_ref(
    const std::vector<float> &q, const Kv3Cache &cache,
    const std::vector<int32_t> &block_table,
    const std::vector<int32_t> &context_lens, long long batch,
    long long query_heads, long long kv_heads, long long head_dim,
    long long page_size, long long max_blocks, const Kv3Config &config,
    float scale, long long window) {
    std::vector<double> out(batch * query_heads * head_dim, 0.0);
    const double score_scale =
        scale > 0.0f ? scale : 1.0 / std::sqrt(static_cast<double>(head_dim));
    for (long long item = 0; item < batch * query_heads; ++item) {
        const long long request = item / query_heads;
        const long long qhead = item % query_heads;
        const long long kvhead = qhead / (query_heads / kv_heads);
        const long long context = context_lens[request];
        const long long start = window > 0 ? std::max(0LL, context - window) : 0;
        const float *query = q.data() + item * head_dim;
        std::vector<double> acc(head_dim, 0.0);
        double maximum = -std::numeric_limits<double>::infinity();
        double denominator = 0.0;
        for (long long tile = start; tile < context; tile += kScoreTile) {
            const long long tile_end = std::min(context, tile + kScoreTile);
            std::vector<double> scores;
            std::vector<long long> slots;
            double tile_max = -std::numeric_limits<double>::infinity();
            for (long long pos = tile; pos < tile_end; ++pos) {
                const long long physical =
                    block_table[request * max_blocks + pos / page_size];
                const long long slot = physical * page_size + pos % page_size;
                double dot = 0.0;
                for (long long d = 0; d < head_dim; ++d)
                    dot += query[d] *
                           kv3_dequant_host(cache, true, slot, kvhead, d,
                                            kv_heads, head_dim, config);
                scores.push_back(dot * score_scale);
                slots.push_back(slot);
                tile_max = std::max(tile_max, scores.back());
            }
            const double next_max = std::max(maximum, tile_max);
            const double old_weight =
                denominator > 0.0 ? std::exp(maximum - next_max) : 0.0;
            denominator *= old_weight;
            if (old_weight != 1.0)
                for (double &v : acc) v *= old_weight;
            for (size_t i = 0; i < scores.size(); ++i) {
                const double weight = std::exp(scores[i] - next_max);
                denominator += weight;
                for (long long d = 0; d < head_dim; ++d)
                    acc[d] += weight * kv3_dequant_host(
                                           cache, false, slots[i], kvhead, d,
                                           kv_heads, head_dim, config);
            }
            maximum = next_max;
        }
        const double inv = denominator > 0.0 ? 1.0 / denominator : 0.0;
        for (long long d = 0; d < head_dim; ++d)
            out[item * head_dim + d] = acc[d] * inv;
    }
    return out;
}

double turbo_dequant_key_host(const std::vector<uint8_t> &cache,
                              const std::vector<float> &scale,
                              const std::vector<float> &zero, long long slot,
                              long long head, long long dim, long long heads,
                              long long head_dim, int bits, bool key_signed) {
    const long long groups = head_dim / 32;
    const long long row = slot * heads + head;
    const long long bytes = (head_dim * bits + 7) / 8;
    const unsigned code = unpack_bits(cache.data() + row * bytes, dim, bits);
    const double quantized =
        key_signed && bits == 8 ? static_cast<int8_t>(code)
                                : static_cast<double>(code);
    const long long meta = row * groups + dim / 32;
    return (quantized + zero[meta]) * scale[meta];
}

double turbo_dequant_value_host(const std::vector<uint8_t> &cache,
                                const std::vector<float> &scale,
                                const std::vector<float> &centroids,
                                long long slot, long long head, long long dim,
                                long long heads, long long head_dim, int bits) {
    const long long groups = head_dim / 32;
    const long long row = slot * heads + head;
    const long long bytes = (head_dim * bits + 7) / 8;
    const unsigned code = unpack_bits(cache.data() + row * bytes, dim, bits);
    return centroids[code] * scale[row * groups + dim / 32];
}

std::vector<double> paged_turbo_ref(
    const std::vector<float> &q, const std::vector<uint8_t> &key_cache,
    const std::vector<uint8_t> &value_cache,
    const std::vector<float> &key_scale,
    const std::vector<float> &value_scale, const std::vector<float> &key_zero,
    const std::vector<float> &centroids, const std::vector<float> &signs,
    const std::vector<int32_t> &block_table,
    const std::vector<int32_t> &context_lens, long long batch,
    long long query_heads, long long kv_heads, long long head_dim,
    long long page_size, long long max_blocks, int key_bits, bool key_signed,
    int value_bits, float scale, long long window) {
    std::vector<double> out(batch * query_heads * head_dim, 0.0);
    const double score_scale =
        scale > 0.0f ? scale : 1.0 / std::sqrt(static_cast<double>(head_dim));
    const double norm = 1.0 / std::sqrt(static_cast<double>(head_dim));
    for (long long item = 0; item < batch * query_heads; ++item) {
        const long long request = item / query_heads;
        const long long qhead = item % query_heads;
        const long long kvhead = qhead / (query_heads / kv_heads);
        const long long context = context_lens[request];
        const long long start = window > 0 ? std::max(0LL, context - window) : 0;
        const float *query = q.data() + item * head_dim;
        std::vector<double> rotated(head_dim, 0.0);
        double maximum = -std::numeric_limits<double>::infinity();
        double denominator = 0.0;
        for (long long tile = start; tile < context; tile += kScoreTile) {
            const long long tile_end = std::min(context, tile + kScoreTile);
            std::vector<double> scores;
            std::vector<long long> slots;
            double tile_max = -std::numeric_limits<double>::infinity();
            for (long long pos = tile; pos < tile_end; ++pos) {
                const long long physical =
                    block_table[request * max_blocks + pos / page_size];
                const long long slot = physical * page_size + pos % page_size;
                double dot = 0.0;
                for (long long d = 0; d < head_dim; ++d)
                    dot += query[d] *
                           turbo_dequant_key_host(key_cache, key_scale,
                                                  key_zero, slot, kvhead, d,
                                                  kv_heads, head_dim, key_bits,
                                                  key_signed);
                scores.push_back(dot * score_scale);
                slots.push_back(slot);
                tile_max = std::max(tile_max, scores.back());
            }
            const double next_max = std::max(maximum, tile_max);
            const double old_weight =
                denominator > 0.0 ? std::exp(maximum - next_max) : 0.0;
            denominator *= old_weight;
            if (old_weight != 1.0)
                for (double &v : rotated) v *= old_weight;
            for (size_t i = 0; i < scores.size(); ++i) {
                const double weight = std::exp(scores[i] - next_max);
                denominator += weight;
                for (long long d = 0; d < head_dim; ++d)
                    rotated[d] += weight * turbo_dequant_value_host(
                                               value_cache, value_scale,
                                               centroids, slots[i], kvhead, d,
                                               kv_heads, head_dim, value_bits);
            }
            maximum = next_max;
        }
        if (denominator > 0.0) {
            const double inv = 1.0 / denominator;
            std::vector<float> tmp(head_dim);
            for (long long d = 0; d < head_dim; ++d)
                tmp[d] = static_cast<float>(rotated[d] * inv);
            fwht_inplace(tmp.data(), head_dim);
            for (long long d = 0; d < head_dim; ++d)
                out[item * head_dim + d] = tmp[d] * norm * signs[d];
        }
    }
    return out;
}

std::vector<double> paged_advanced_ref(
    const std::vector<float> &q, const std::vector<float> &key_cache,
    const std::vector<float> &value_cache,
    const std::vector<int32_t> &block_table,
    const std::vector<int32_t> &context_lens,
    const std::vector<int32_t> *block_mask,
    const std::vector<float> *alibi_slopes, const std::vector<float> *sinks,
    long long batch, long long query_heads, long long kv_heads,
    long long head_dim, long long page_size, long long max_blocks,
    float scale, long long window, float softcap) {
    std::vector<double> out(batch * query_heads * head_dim, 0.0);
    const double score_scale =
        scale > 0.0f ? scale : 1.0 / std::sqrt(static_cast<double>(head_dim));
    for (long long item = 0; item < batch * query_heads; ++item) {
        const long long request = item / query_heads;
        const long long qhead = item % query_heads;
        const long long kvhead = qhead / (query_heads / kv_heads);
        const long long context = context_lens[request];
        const long long start = window > 0 ? std::max(0LL, context - window) : 0;
        const float *query = q.data() + item * head_dim;
        std::vector<double> acc(head_dim, 0.0);
        double maximum = sinks == nullptr ? -std::numeric_limits<double>::infinity()
                                          : (*sinks)[qhead];
        double denominator = sinks == nullptr ? 0.0 : 1.0;
        for (long long tile = start; tile < context; tile += kScoreTile) {
            const long long tile_end = std::min(context, tile + kScoreTile);
            std::vector<double> scores;
            std::vector<long long> bases;
            double tile_max = -std::numeric_limits<double>::infinity();
            for (long long pos = tile; pos < tile_end; ++pos) {
                const long long logical_block = pos / page_size;
                if (block_mask != nullptr &&
                    (*block_mask)[(request * query_heads + qhead) *
                                      max_blocks +
                                  logical_block] == 0)
                    continue;
                const long long physical =
                    block_table[request * max_blocks + logical_block];
                const long long base =
                    ((physical * page_size + pos % page_size) * kv_heads +
                     kvhead) *
                    head_dim;
                double dot = 0.0;
                for (long long d = 0; d < head_dim; ++d)
                    dot += query[d] * key_cache[base + d];
                dot *= score_scale;
                if (alibi_slopes != nullptr)
                    dot += (*alibi_slopes)[qhead] * (pos - (context - 1));
                const double score =
                    softcap > 0.0f ? softcap * std::tanh(dot / softcap) : dot;
                scores.push_back(score);
                bases.push_back(base);
                tile_max = std::max(tile_max, score);
            }
            if (scores.empty()) continue;
            const double next_max = std::max(maximum, tile_max);
            const double old_weight =
                denominator > 0.0 ? std::exp(maximum - next_max) : 0.0;
            denominator *= old_weight;
            if (old_weight != 1.0)
                for (double &v : acc) v *= old_weight;
            for (size_t i = 0; i < scores.size(); ++i) {
                const double weight = std::exp(scores[i] - next_max);
                denominator += weight;
                for (long long d = 0; d < head_dim; ++d)
                    acc[d] += weight * value_cache[bases[i] + d];
            }
            maximum = next_max;
        }
        if (denominator > 0.0) {
            const double inv = 1.0 / denominator;
            for (long long d = 0; d < head_dim; ++d)
                out[item * head_dim + d] = acc[d] * inv;
        }
    }
    return out;
}

void pack_q8_row(const float *src, uint8_t *dst) {
    float amax = 0.0f;
    for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(src[i]));
    const float scale = amax / 127.0f;
    const float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
    const uint16_t bits = f32_to_f16_bits_host(scale);
    dst[0] = static_cast<uint8_t>(bits);
    dst[1] = static_cast<uint8_t>(bits >> 8);
    for (int i = 0; i < 32; ++i) {
        const int code = std::clamp(static_cast<int>(std::rint(src[i] * inv)),
                                    -127, 127);
        dst[2 + i] = static_cast<uint8_t>(static_cast<int8_t>(code));
    }
}

void pack_q4_row(const float *src, uint8_t *dst) {
    float amax = 0.0f;
    for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(src[i]));
    const float scale = amax / 7.0f;
    const float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
    const uint16_t bits = f32_to_f16_bits_host(scale);
    dst[0] = static_cast<uint8_t>(bits);
    dst[1] = static_cast<uint8_t>(bits >> 8);
    uint8_t nibbles[32];
    for (int i = 0; i < 32; ++i)
        nibbles[i] = static_cast<uint8_t>(
            std::clamp(static_cast<int>(std::rint(src[i] * inv)) + 8, 0, 15));
    for (int i = 0; i < 16; ++i)
        dst[2 + i] = nibbles[i] | static_cast<uint8_t>(nibbles[i + 16] << 4);
}

std::vector<uint8_t> pack_quant_rows(const std::vector<float> &x, int format,
                                     long long rows, long long head_dim) {
    const long long block_bytes = qlite_block_bytes(format);
    const long long blocks = head_dim / 32;
    std::vector<uint8_t> packed(rows * blocks * block_bytes);
    for (long long row = 0; row < rows; ++row)
        for (long long block = 0; block < blocks; ++block) {
            uint8_t *dst =
                packed.data() + (row * blocks + block) * block_bytes;
            const float *src = x.data() + row * head_dim + block * 32;
            if (format == kFmtQ8_0)
                pack_q8_row(src, dst);
            else
                pack_q4_row(src, dst);
        }
    return packed;
}

std::vector<double> quantized_attention_ref(
    const std::vector<float> &q, const std::vector<uint8_t> &packed_k,
    const std::vector<uint8_t> &packed_v, long long batch, long long heads,
    long long sequence, long long head_dim, int format, bool causal) {
    std::vector<double> out(batch * heads * sequence * head_dim, 0.0);
    const long long block_bytes = qlite_block_bytes(format);
    const long long blocks = head_dim / 32;
    const long long row_bytes = blocks * block_bytes;
    const double score_scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
    for (long long item = 0; item < batch * heads * sequence; ++item) {
        const long long query_pos = item % sequence;
        const long long prefix = item / sequence;
        const long long key_count = causal ? query_pos + 1 : sequence;
        const float *query = q.data() + item * head_dim;
        double maximum = -std::numeric_limits<double>::infinity();
        std::vector<double> scores(key_count);
        for (long long key_pos = 0; key_pos < key_count; ++key_pos) {
            const uint8_t *key =
                packed_k.data() + (prefix * sequence + key_pos) * row_bytes;
            double dot = 0.0;
            for (long long d = 0; d < head_dim; ++d)
                dot += query[d] *
                       qlite_dequant_host(key + (d / 32) * block_bytes, format,
                                          static_cast<int>(d % 32));
            scores[key_pos] = dot * score_scale;
            maximum = std::max(maximum, scores[key_pos]);
        }
        double denominator = 0.0;
        for (double &score : scores) {
            score = std::exp(score - maximum);
            denominator += score;
        }
        for (long long key_pos = 0; key_pos < key_count; ++key_pos) {
            const uint8_t *value =
                packed_v.data() + (prefix * sequence + key_pos) * row_bytes;
            const double probability = scores[key_pos] / denominator;
            for (long long d = 0; d < head_dim; ++d)
                out[item * head_dim + d] +=
                    probability *
                    qlite_dequant_host(value + (d / 32) * block_bytes, format,
                                       static_cast<int>(d % 32));
        }
    }
    return out;
}

void make_block_table(std::vector<int32_t> &table,
                      std::vector<int32_t> &contexts, long long batch,
                      long long max_blocks, long long page_size,
                      long long cache_blocks) {
    contexts.resize(batch);
    table.resize(batch * max_blocks);
    for (long long b = 0; b < batch; ++b) {
        contexts[b] = static_cast<int32_t>(max_blocks * page_size - 3 - b * 5);
        for (long long m = 0; m < max_blocks; ++m)
            table[b * max_blocks + m] =
                static_cast<int32_t>((b * 5 + m * 3) % cache_blocks);
    }
}

template <typename T>
void fill_linear(std::vector<T> &x, T base, T step) {
    for (size_t i = 0; i < x.size(); ++i)
        x[i] = static_cast<T>(base + step * static_cast<T>(i));
}

// ---------------------------------------------------------------------------
// Correctness and benchmark
// ---------------------------------------------------------------------------

bool run_kv3_case(const char *name, Kv3Config config) {
    qc::Rng rng(32000 + config.group_size + config.scale_type * 17 +
                config.signedness * 31 + config.zero_mode * 47);
    const long long max_slots = 24, count = 10, heads = 2, head_dim = 64;
    const long long packed = kv3_packed_bytes(head_dim);
    const long long groups = head_dim / config.group_size;
    auto key = rng.uniforms(count * heads * head_dim, -1.0f, 1.0f);
    auto value = rng.uniforms(count * heads * head_dim, -1.0f, 1.0f);
    for (long long i = 0; i < config.group_size; ++i) key[i] = 0.0f;
    std::vector<int32_t> slots = {3, 7, -1, 11, 3, 14, 17, 7, 20, 22};
    Kv3Cache ref = kv3_prefilled(max_slots, heads, head_dim, config);
    kv3_scatter_ref(key, value, slots, ref, max_slots, count, heads, head_dim,
                    config);

    float *dkey = qc::dnew(key);
    float *dvalue = qc::dnew(value);
    int32_t *dslots = qc::dnew(slots);
    uint8_t *dkc = qc::dnew(ref.key_codes);
    uint8_t *dvc = qc::dnew(ref.value_codes);
    void *dks = nullptr;
    void *dvs = nullptr;
    if (config.scale_type == kKv3Fp32) {
        dks = qc::dnew(ref.key_scales_f32);
        dvs = qc::dnew(ref.value_scales_f32);
    } else {
        dks = qc::dnew(ref.key_scales_f16);
        dvs = qc::dnew(ref.value_scales_f16);
    }
    int32_t *dkz = qc::dnew(ref.key_zero);
    int32_t *dvz = qc::dnew(ref.value_zero);
    int32_t *dinvalid = qc::dzero<int32_t>(1);

    Kv3Cache pre = kv3_prefilled(max_slots, heads, head_dim, config);
    QC_CHECK(hipMemcpy(dkc, pre.key_codes.data(), pre.key_codes.size(),
                       hipMemcpyHostToDevice));
    QC_CHECK(hipMemcpy(dvc, pre.value_codes.data(), pre.value_codes.size(),
                       hipMemcpyHostToDevice));
    if (config.scale_type == kKv3Fp32) {
        QC_CHECK(hipMemcpy(dks, pre.key_scales_f32.data(),
                           pre.key_scales_f32.size() * sizeof(float),
                           hipMemcpyHostToDevice));
        QC_CHECK(hipMemcpy(dvs, pre.value_scales_f32.data(),
                           pre.value_scales_f32.size() * sizeof(float),
                           hipMemcpyHostToDevice));
    } else {
        QC_CHECK(hipMemcpy(dks, pre.key_scales_f16.data(),
                           pre.key_scales_f16.size() * sizeof(uint16_t),
                           hipMemcpyHostToDevice));
        QC_CHECK(hipMemcpy(dvs, pre.value_scales_f16.data(),
                           pre.value_scales_f16.size() * sizeof(uint16_t),
                           hipMemcpyHostToDevice));
    }
    QC_CHECK(hipMemcpy(dkz, pre.key_zero.data(),
                       pre.key_zero.size() * sizeof(int32_t),
                       hipMemcpyHostToDevice));
    QC_CHECK(hipMemcpy(dvz, pre.value_zero.data(),
                       pre.value_zero.size() * sizeof(int32_t),
                       hipMemcpyHostToDevice));
    kv3_scatter_kernel<<<qc::grid_for(max_slots * heads, kThreads), kThreads>>>(
        dkey, dvalue, dslots, dkc, dvc, dks, dvs, dkz, dvz, max_slots, count,
        heads, head_dim, config, dinvalid);
    QC_SYNC();

    bool ok = true;
    char label[128];
    std::snprintf(label, sizeof(label), "kv_cache_scatter_bitnet_kv3 %s key",
                  name);
    ok &= qc::compare(qc::d2h(dkc, ref.key_codes.size()),
                      to_ref_u8(ref.key_codes), qc::Tol::exact())
              .report(label);
    std::snprintf(label, sizeof(label), "kv_cache_scatter_bitnet_kv3 %s value",
                  name);
    ok &= qc::compare(qc::d2h(dvc, ref.value_codes.size()),
                      to_ref_u8(ref.value_codes), qc::Tol::exact())
              .report(label);
    if (config.scale_type == kKv3Fp32) {
        std::snprintf(label, sizeof(label),
                      "kv_cache_scatter_bitnet_kv3 %s key scale", name);
        ok &= qc::compare(qc::d2h(static_cast<float *>(dks),
                                  ref.key_scales_f32.size()),
                          to_ref(ref.key_scales_f32), qc::Tol::exact())
                  .report(label);
        std::snprintf(label, sizeof(label),
                      "kv_cache_scatter_bitnet_kv3 %s value scale", name);
        ok &= qc::compare(qc::d2h(static_cast<float *>(dvs),
                                  ref.value_scales_f32.size()),
                          to_ref(ref.value_scales_f32), qc::Tol::exact())
                  .report(label);
    } else {
        std::snprintf(label, sizeof(label),
                      "kv_cache_scatter_bitnet_kv3 %s key scale", name);
        ok &= qc::compare(qc::d2h(static_cast<uint16_t *>(dks),
                                  ref.key_scales_f16.size()),
                          to_ref_u16(ref.key_scales_f16), qc::Tol::exact())
                  .report(label);
        std::snprintf(label, sizeof(label),
                      "kv_cache_scatter_bitnet_kv3 %s value scale", name);
        ok &= qc::compare(qc::d2h(static_cast<uint16_t *>(dvs),
                                  ref.value_scales_f16.size()),
                          to_ref_u16(ref.value_scales_f16), qc::Tol::exact())
                  .report(label);
    }
    if (config.zero_mode == kKv3IntegerZero) {
        std::snprintf(label, sizeof(label),
                      "kv_cache_scatter_bitnet_kv3 %s key zero", name);
        ok &= qc::compare(qc::d2h(dkz, ref.key_zero.size()),
                          to_ref_i32(ref.key_zero), qc::Tol::exact())
                  .report(label);
        std::snprintf(label, sizeof(label),
                      "kv_cache_scatter_bitnet_kv3 %s value zero", name);
        ok &= qc::compare(qc::d2h(dvz, ref.value_zero.size()),
                          to_ref_i32(ref.value_zero), qc::Tol::exact())
                  .report(label);
    }

    std::vector<int32_t> indices = {3, 7, 11, 14, 17, 20};
    int32_t *dindices = qc::dnew(indices);
    float *dgk = qc::dzero<float>(indices.size() * heads * head_dim);
    float *dgv = qc::dzero<float>(indices.size() * heads * head_dim);
    kv3_gather_kernel<<<qc::grid_for(indices.size() * heads * head_dim,
                                      kThreads),
                         kThreads>>>(dkc, dvc, dindices, dks, dvs, dkz, dvz,
                                     dgk, dgv, indices.size(), heads, head_dim,
                                     config);
    QC_SYNC();
    std::vector<double> ref_key, ref_value;
    kv3_gather_ref(ref, indices, ref_key, ref_value, indices.size(), heads,
                   head_dim, config);
    std::snprintf(label, sizeof(label), "kv_cache_gather_bitnet_kv3 %s key",
                  name);
    ok &= qc::compare(qc::d2h(dgk, ref_key.size()), ref_key, qc::Tol::fp32())
              .report(label);
    std::snprintf(label, sizeof(label), "kv_cache_gather_bitnet_kv3 %s value",
                  name);
    ok &= qc::compare(qc::d2h(dgv, ref_value.size()), ref_value,
                      qc::Tol::fp32())
              .report(label);

    const long long batch = 2, qh = 4, kvh = heads, page = 4, max_blocks = 4,
                    cache_blocks = 6;
    std::vector<int32_t> block_table, context_lens;
    make_block_table(block_table, context_lens, batch, max_blocks, page,
                     cache_blocks);
    std::vector<float> query = rng.uniforms(batch * qh * head_dim, -0.5f, 0.5f);
    float *dq = qc::dnew(query);
    int32_t *dbt = qc::dnew(block_table);
    int32_t *dctx = qc::dnew(context_lens);
    float *dout = qc::dzero<float>(query.size());
    paged_attention_bitnet_kernel<<<qc::grid_for(batch * qh, kThreads),
                                    kThreads>>>(
        dq, dkc, dvc, dks, dvs, dkz, dvz, dbt, dctx, dout, batch, qh, kvh,
        head_dim, page, max_blocks, config, 0.0f, 11);
    QC_SYNC();
    std::snprintf(label, sizeof(label), "paged_attention_bitnet_kv3 %s", name);
    ok &= qc::compare(qc::d2h(dout, query.size()),
                      paged_bitnet_ref(query, ref, block_table, context_lens,
                                       batch, qh, kvh, head_dim, page,
                                       max_blocks, config, 0.0f, 11),
                      qc::Tol::quantized().with_elementwise(8e-4, 8e-4))
              .report(label);

    if (config.scale_type == kKv3Fp32) {
        qc::dfree(static_cast<float *>(dks), static_cast<float *>(dvs));
    } else {
        qc::dfree(static_cast<uint16_t *>(dks), static_cast<uint16_t *>(dvs));
    }
    qc::dfree(dkey, dvalue, dslots, dkc, dvc, dkz, dvz, dinvalid, dindices,
              dgk, dgv, dq, dbt, dctx, dout);
    return ok;
}

bool run_correctness() {
    std::printf("\n== Phase 2 remaining correctness ==\n");
    bool ok = true;
    int checks = 0;

    ok &= run_kv3_case("signed-fp16", Kv3Config{16, kKv3Fp16, kKv3Signed,
                                                kKv3NoZero});
    checks += 7;
    ok &= run_kv3_case("unsigned-zp-fp32",
                       Kv3Config{32, kKv3Fp32, kKv3Unsigned,
                                 kKv3IntegerZero});
    checks += 11;

    qc::Rng rng(33002);
    const long long batch = 2, qh = 4, kvh = 2, head_dim = 64, page = 8,
                    max_blocks = 4, cache_blocks = 8;
    std::vector<int32_t> block_table, context_lens;
    make_block_table(block_table, context_lens, batch, max_blocks, page,
                     cache_blocks);
    std::vector<float> query = rng.uniforms(batch * qh * head_dim, -0.35f, 0.35f);
    std::vector<float> key_cache =
        rng.uniforms(cache_blocks * page * kvh * head_dim, -0.4f, 0.4f);
    std::vector<float> value_cache =
        rng.uniforms(cache_blocks * page * kvh * head_dim, -0.6f, 0.6f);
    std::vector<int32_t> block_mask(batch * qh * max_blocks, 1);
    for (long long i = 0; i < static_cast<long long>(block_mask.size()); ++i)
        if ((i % 5) == 2) block_mask[i] = 0;
    std::vector<float> alibi = rng.uniforms(qh, -0.03f, 0.02f);
    std::vector<float> sinks = rng.uniforms(qh, -0.2f, 0.1f);
    float *dq = qc::dnew(query);
    float *dk = qc::dnew(key_cache);
    float *dv = qc::dnew(value_cache);
    int32_t *dbt = qc::dnew(block_table);
    int32_t *dctx = qc::dnew(context_lens);
    int32_t *dmask = qc::dnew(block_mask);
    float *dalibi = qc::dnew(alibi);
    float *dsinks = qc::dnew(sinks);
    float *dout = qc::dzero<float>(query.size());
    paged_attention_advanced_kernel<<<qc::grid_for(batch * qh, kThreads),
                                      kThreads>>>(
        dq, dk, dv, dbt, dctx, dmask, dalibi, dsinks, dout, batch, qh, kvh,
        head_dim, page, max_blocks, 0.0f, 19, 5.0f);
    QC_SYNC();
    ok &= qc::compare(
              qc::d2h(dout, query.size()),
              paged_advanced_ref(query, key_cache, value_cache, block_table,
                                 context_lens, &block_mask, &alibi, &sinks,
                                 batch, qh, kvh, head_dim, page, max_blocks,
                                 0.0f, 19, 5.0f),
              qc::Tol::fp32().with_elementwise(3e-5, 3e-5))
              .report("paged_attention_advanced mask/alibi/sink/softcap");
    ++checks;
    qc::dfree(dq, dk, dv, dbt, dctx, dmask, dalibi, dsinks, dout);

    const int key_bits = 4, value_bits = 3;
    const long long groups = head_dim / 32;
    const long long key_bytes = (head_dim * key_bits + 7) / 8;
    const long long value_bytes = (head_dim * value_bits + 7) / 8;
    const long long slots = cache_blocks * page;
    std::vector<uint8_t> tq_key(slots * kvh * key_bytes, 0);
    std::vector<uint8_t> tq_value(slots * kvh * value_bytes, 0);
    for (long long row = 0; row < slots * kvh; ++row) {
        for (long long d = 0; d < head_dim; ++d) {
            pack_bits_host(tq_key, row * head_dim + d, key_bits,
                           static_cast<unsigned>((row + d * 3) &
                                                 ((1 << key_bits) - 1)));
            pack_bits_host(tq_value, row * head_dim + d, value_bits,
                           static_cast<unsigned>((row * 5 + d) &
                                                 ((1 << value_bits) - 1)));
        }
    }
    std::vector<float> tq_ks(slots * kvh * groups);
    std::vector<float> tq_vs(slots * kvh * groups);
    std::vector<float> tq_kz(slots * kvh * groups);
    for (size_t i = 0; i < tq_ks.size(); ++i) {
        tq_ks[i] = 0.015f + 0.001f * static_cast<float>(i % 7);
        tq_vs[i] = 0.2f + 0.01f * static_cast<float>(i % 5);
        tq_kz[i] = -4.0f + static_cast<float>(i % 4);
    }
    std::vector<float> centroids(1 << value_bits);
    for (int i = 0; i < (1 << value_bits); ++i)
        centroids[i] = -1.0f + 2.0f * i / float((1 << value_bits) - 1);
    std::vector<float> signs(head_dim);
    for (long long d = 0; d < head_dim; ++d) signs[d] = (d % 3) == 0 ? -1.0f : 1.0f;
    float *dtq = qc::dnew(query);
    uint8_t *dtqk = qc::dnew(tq_key);
    uint8_t *dtqv = qc::dnew(tq_value);
    float *dtqks = qc::dnew(tq_ks);
    float *dtqvs = qc::dnew(tq_vs);
    float *dtqkz = qc::dnew(tq_kz);
    float *dcent = qc::dnew(centroids);
    float *dsign = qc::dnew(signs);
    int32_t *dtqbt = qc::dnew(block_table);
    int32_t *dtqctx = qc::dnew(context_lens);
    float *dtqout = qc::dzero<float>(query.size());
    paged_attention_turboquant_kernel<<<qc::grid_for(batch * qh, kThreads),
                                        kThreads>>>(
        dtq, dtqk, dtqv, dtqks, dtqvs, dtqkz, dcent, dsign, dtqbt, dtqctx,
        dtqout, batch, qh, kvh, head_dim, page, max_blocks, key_bits, false,
        value_bits, 0.0f, 17);
    QC_SYNC();
    ok &= qc::compare(qc::d2h(dtqout, query.size()),
                      paged_turbo_ref(query, tq_key, tq_value, tq_ks, tq_vs,
                                      tq_kz, centroids, signs, block_table,
                                      context_lens, batch, qh, kvh, head_dim,
                                      page, max_blocks, key_bits, false,
                                      value_bits, 0.0f, 17),
                      qc::Tol::quantized().with_elementwise(1e-3, 1e-3))
              .report("paged_attention_turboquant k4/v3");
    ++checks;
    qc::dfree(dtq, dtqk, dtqv, dtqks, dtqvs, dtqkz, dcent, dsign, dtqbt,
              dtqctx, dtqout);

    const long long qb = 1, qheads = 2, seq = 16, qdim = 64;
    std::vector<float> qa = rng.uniforms(qb * qheads * seq * qdim, -0.4f, 0.4f);
    std::vector<float> kf = rng.uniforms(qb * qheads * seq * qdim, -0.6f, 0.6f);
    std::vector<float> vf = rng.uniforms(qb * qheads * seq * qdim, -0.6f, 0.6f);
    for (int fmt : {kFmtQ8_0, kFmtQ4_0}) {
        std::vector<uint8_t> pk =
            pack_quant_rows(kf, fmt, qb * qheads * seq, qdim);
        std::vector<uint8_t> pv =
            pack_quant_rows(vf, fmt, qb * qheads * seq, qdim);
        float *dqa = qc::dnew(qa);
        uint8_t *dpk = qc::dnew(pk);
        uint8_t *dpv = qc::dnew(pv);
        float *dqout = qc::dzero<float>(qa.size());
        for (bool causal : {false, true}) {
            quantized_attention_kernel<<<qc::grid_for(qb * qheads * seq,
                                                       kThreads),
                                          kThreads>>>(
                dqa, dpk, dpv, dqout, qb, qheads, seq, qdim, fmt, causal);
            QC_SYNC();
            char label[96];
            std::snprintf(label, sizeof(label), "quantized_attention %s %s",
                          fmt == kFmtQ8_0 ? "q8_0" : "q4_0",
                          causal ? "causal" : "full");
            ok &= qc::compare(qc::d2h(dqout, qa.size()),
                              quantized_attention_ref(qa, pk, pv, qb, qheads,
                                                      seq, qdim, fmt, causal),
                              qc::Tol::quantized().with_elementwise(1e-3,
                                                                    1e-3))
                      .report(label);
            ++checks;
        }
        qc::dfree(dqa, dpk, dpv, dqout);
    }

    std::printf("Phase 2 remaining correctness checks: %d\n", checks);
    return ok;
}

void run_bench() {
    std::printf("\n== Phase 2 remaining benchmarks ==\n");
    std::printf("   Timing note: row/direct CDNA3 routes compared with scalar GPU baselines.\n");
    qc::Rng rng(34002);
    auto bench_repeated = [](auto &&fn, int repeats, int warmups, int iters) {
        qc::Bench b = qc::bench([&] {
            for (int r = 0; r < repeats; ++r) fn();
        }, warmups, iters);
        b.median_ms /= repeats;
        b.min_ms /= repeats;
        b.max_ms /= repeats;
        b.mean_ms /= repeats;
        return b;
    };

    const long long max_slots = 1024, count = 96, heads = 4, head_dim = 64;
    Kv3Config config{32, kKv3Fp32, kKv3Unsigned, kKv3IntegerZero};
    const long long packed = kv3_packed_bytes(head_dim);
    const long long groups = head_dim / config.group_size;
    auto key = rng.uniforms(count * heads * head_dim, -1.0f, 1.0f);
    auto value = rng.uniforms(key.size(), -1.0f, 1.0f);
    std::vector<int32_t> slots(count);
    for (long long i = 0; i < count; ++i) slots[i] = static_cast<int32_t>((i * 7) % max_slots);
    Kv3Cache pre = kv3_prefilled(max_slots, heads, head_dim, config);
    float *dkey = qc::dnew(key);
    float *dvalue = qc::dnew(value);
    int32_t *dslots = qc::dnew(slots);
    uint8_t *dkc = qc::dnew(pre.key_codes);
    uint8_t *dvc = qc::dnew(pre.value_codes);
    float *dks = qc::dnew(pre.key_scales_f32);
    float *dvs = qc::dnew(pre.value_scales_f32);
    int32_t *dkz = qc::dnew(pre.key_zero);
    int32_t *dvz = qc::dnew(pre.value_zero);
    int32_t *dinvalid = qc::dzero<int32_t>(1);
    auto scatter_s = qc::bench([&] {
        kv3_scatter_scalar_kernel<<<1, 1>>>(dkey, dvalue, dslots, dkc, dvc,
                                            dks, dvs, dkz, dvz, max_slots,
                                            count, heads, head_dim, config,
                                            dinvalid);
    }, 2, 3);
    auto scatter_c = bench_repeated([&] {
        kv3_scatter_kernel<<<qc::grid_for(max_slots * heads, kThreads),
                             kThreads>>>(dkey, dvalue, dslots, dkc, dvc, dks,
                                          dvs, dkz, dvz, max_slots, count,
                                          heads, head_dim, config, dinvalid);
    }, 20, 5, 20);
    const double scatter_bytes =
        static_cast<double>(count * heads * head_dim * 2 * sizeof(float) +
                            max_slots * heads * (packed * 2 + groups * 2 * 4));
    scatter_s.report_bandwidth("kv_cache_scatter_bitnet_kv3 scalar",
                               scatter_bytes);
    scatter_c.report_bandwidth("kv_cache_scatter_bitnet_kv3 candidate",
                               scatter_bytes);
    qc::report_ab("kv_cache_scatter_bitnet_kv3", scatter_s, scatter_c);

    std::vector<int32_t> indices(count);
    for (long long i = 0; i < count; ++i) indices[i] = slots[i];
    int32_t *dindices = qc::dnew(indices);
    float *dgk = qc::dzero<float>(count * heads * head_dim);
    float *dgv = qc::dzero<float>(count * heads * head_dim);
    auto gather_s = qc::bench([&] {
        kv3_gather_scalar_kernel<<<1, 1>>>(dkc, dvc, dindices, dks, dvs, dkz,
                                           dvz, dgk, dgv, count, heads,
                                           head_dim, config);
    }, 2, 3);
    auto gather_c = bench_repeated([&] {
        kv3_gather_kernel<<<qc::grid_for(count * heads * head_dim, kThreads),
                            kThreads>>>(dkc, dvc, dindices, dks, dvs, dkz,
                                        dvz, dgk, dgv, count, heads, head_dim,
                                        config);
    }, 1000, 5, 20);
    const double gather_bytes =
        static_cast<double>(count * heads * head_dim * 2 * sizeof(float));
    gather_s.report_bandwidth("kv_cache_gather_bitnet_kv3 scalar",
                              gather_bytes);
    gather_c.report_bandwidth("kv_cache_gather_bitnet_kv3 candidate",
                              gather_bytes);
    qc::report_ab("kv_cache_gather_bitnet_kv3", gather_s, gather_c);

    const long long batch = 4, qh = 8, kvh = heads, page = 16, max_blocks = 8,
                    cache_blocks = 64;
    std::vector<int32_t> block_table, context_lens;
    make_block_table(block_table, context_lens, batch, max_blocks, page,
                     cache_blocks);
    auto query = rng.uniforms(batch * qh * head_dim, -0.35f, 0.35f);
    float *dq = qc::dnew(query);
    int32_t *dbt = qc::dnew(block_table);
    int32_t *dctx = qc::dnew(context_lens);
    float *dout = qc::dzero<float>(query.size());
    auto bitnet_s = qc::bench([&] {
        paged_attention_bitnet_scalar_kernel<<<1, 1>>>(
            dq, dkc, dvc, dks, dvs, dkz, dvz, dbt, dctx, dout, batch, qh, kvh,
            head_dim, page, max_blocks, config, 0.0f, 64);
    }, 2, 3);
    auto bitnet_c = qc::bench([&] {
        paged_attention_bitnet_kernel<<<qc::grid_for(batch * qh, kThreads),
                                        kThreads>>>(
            dq, dkc, dvc, dks, dvs, dkz, dvz, dbt, dctx, dout, batch, qh, kvh,
            head_dim, page, max_blocks, config, 0.0f, 64);
    }, 5, 20);
    const double attn_flops =
        static_cast<double>(batch * qh * 64 * head_dim * 4);
    bitnet_s.report_compute("paged_attention_bitnet_kv3 scalar", attn_flops);
    bitnet_c.report_compute("paged_attention_bitnet_kv3 candidate", attn_flops);
    qc::report_ab("paged_attention_bitnet_kv3", bitnet_s, bitnet_c);

    auto fkey = rng.uniforms(cache_blocks * page * kvh * head_dim, -0.4f, 0.4f);
    auto fvalue =
        rng.uniforms(cache_blocks * page * kvh * head_dim, -0.6f, 0.6f);
    float *dfk = qc::dnew(fkey);
    float *dfv = qc::dnew(fvalue);
    std::vector<int32_t> mask(batch * qh * max_blocks, 1);
    for (size_t i = 0; i < mask.size(); ++i)
        if ((i % 7) == 3) mask[i] = 0;
    auto alibi = rng.uniforms(qh, -0.02f, 0.02f);
    auto sinks = rng.uniforms(qh, -0.2f, 0.2f);
    int32_t *dmask = qc::dnew(mask);
    float *dalibi = qc::dnew(alibi);
    float *dsinks = qc::dnew(sinks);
    auto adv_s = qc::bench([&] {
        paged_attention_advanced_scalar_kernel<<<1, 1>>>(
            dq, dfk, dfv, dbt, dctx, dmask, dalibi, dsinks, dout, batch, qh,
            kvh, head_dim, page, max_blocks, 0.0f, 64, 5.0f);
    }, 2, 3);
    auto adv_c = qc::bench([&] {
        paged_attention_advanced_kernel<<<qc::grid_for(batch * qh, kThreads),
                                          kThreads>>>(
            dq, dfk, dfv, dbt, dctx, dmask, dalibi, dsinks, dout, batch, qh,
            kvh, head_dim, page, max_blocks, 0.0f, 64, 5.0f);
    }, 5, 20);
    adv_s.report_compute("paged_attention_advanced scalar", attn_flops);
    adv_c.report_compute("paged_attention_advanced candidate", attn_flops);
    qc::report_ab("paged_attention_advanced", adv_s, adv_c);

    const int key_bits = 4, value_bits = 3;
    const long long tgroups = head_dim / 32;
    const long long key_bytes = (head_dim * key_bits + 7) / 8;
    const long long value_bytes = (head_dim * value_bits + 7) / 8;
    const long long tslots = cache_blocks * page;
    std::vector<uint8_t> tq_key(tslots * kvh * key_bytes, 0);
    std::vector<uint8_t> tq_value(tslots * kvh * value_bytes, 0);
    for (long long row = 0; row < tslots * kvh; ++row)
        for (long long d = 0; d < head_dim; ++d) {
            pack_bits_host(tq_key, row * head_dim + d, key_bits,
                           static_cast<unsigned>((row + d) &
                                                 ((1 << key_bits) - 1)));
            pack_bits_host(tq_value, row * head_dim + d, value_bits,
                           static_cast<unsigned>((row * 3 + d) &
                                                 ((1 << value_bits) - 1)));
        }
    std::vector<float> tq_ks(tslots * kvh * tgroups, 0.02f);
    std::vector<float> tq_vs(tslots * kvh * tgroups, 0.25f);
    std::vector<float> tq_kz(tslots * kvh * tgroups, -4.0f);
    std::vector<float> centroids(1 << value_bits);
    for (int i = 0; i < (1 << value_bits); ++i)
        centroids[i] = -1.0f + 2.0f * i / float((1 << value_bits) - 1);
    std::vector<float> signs(head_dim);
    for (long long d = 0; d < head_dim; ++d) signs[d] = (d & 1) ? -1.0f : 1.0f;
    uint8_t *dtqk = qc::dnew(tq_key);
    uint8_t *dtqv = qc::dnew(tq_value);
    float *dtqks = qc::dnew(tq_ks);
    float *dtqvs = qc::dnew(tq_vs);
    float *dtqkz = qc::dnew(tq_kz);
    float *dcent = qc::dnew(centroids);
    float *dsign = qc::dnew(signs);
    auto tq_s = qc::bench([&] {
        paged_attention_turboquant_scalar_kernel<<<1, 1>>>(
            dq, dtqk, dtqv, dtqks, dtqvs, dtqkz, dcent, dsign, dbt, dctx,
            dout, batch, qh, kvh, head_dim, page, max_blocks, key_bits, false,
            value_bits, 0.0f, 64);
    }, 2, 3);
    auto tq_c = qc::bench([&] {
        paged_attention_turboquant_kernel<<<qc::grid_for(batch * qh, kThreads),
                                            kThreads>>>(
            dq, dtqk, dtqv, dtqks, dtqvs, dtqkz, dcent, dsign, dbt, dctx,
            dout, batch, qh, kvh, head_dim, page, max_blocks, key_bits, false,
            value_bits, 0.0f, 64);
    }, 5, 20);
    tq_s.report_compute("paged_attention_turboquant scalar", attn_flops);
    tq_c.report_compute("paged_attention_turboquant candidate", attn_flops);
    qc::report_ab("paged_attention_turboquant", tq_s, tq_c);

    const long long qb = 2, qheads = 4, seq = 64, qdim = 64;
    auto qa = rng.uniforms(qb * qheads * seq * qdim, -0.4f, 0.4f);
    auto qkf = rng.uniforms(qb * qheads * seq * qdim, -0.6f, 0.6f);
    auto qvf = rng.uniforms(qb * qheads * seq * qdim, -0.6f, 0.6f);
    auto pk = pack_quant_rows(qkf, kFmtQ8_0, qb * qheads * seq, qdim);
    auto pv = pack_quant_rows(qvf, kFmtQ8_0, qb * qheads * seq, qdim);
    float *dqa = qc::dnew(qa);
    uint8_t *dpk = qc::dnew(pk);
    uint8_t *dpv = qc::dnew(pv);
    float *dqao = qc::dzero<float>(qa.size());
    auto qa_s = qc::bench([&] {
        quantized_attention_scalar_kernel<<<1, 1>>>(
            dqa, dpk, dpv, dqao, qb, qheads, seq, qdim, kFmtQ8_0, true);
    }, 2, 3);
    auto qa_c = qc::bench([&] {
        quantized_attention_kernel<<<qc::grid_for(qb * qheads * seq, kThreads),
                                     kThreads>>>(
            dqa, dpk, dpv, dqao, qb, qheads, seq, qdim, kFmtQ8_0, true);
    }, 5, 20);
    const double qa_flops = static_cast<double>(qb * qheads * seq * seq * qdim * 2);
    qa_s.report_compute("quantized_attention q8_0 scalar", qa_flops);
    qa_c.report_compute("quantized_attention q8_0 candidate", qa_flops);
    qc::report_ab("quantized_attention", qa_s, qa_c);

    qc::dfree(dkey, dvalue, dslots, dkc, dvc, dks, dvs, dkz, dvz, dinvalid,
              dindices, dgk, dgv, dq, dbt, dctx, dout, dfk, dfv, dmask,
              dalibi, dsinks, dtqk, dtqv, dtqks, dtqvs, dtqkz, dcent, dsign,
              dqa, dpk, dpv, dqao);
}

}  // namespace

int main(int argc, char **argv) {
    const bool do_bench = argc > 1 && std::string(argv[1]) == "--bench";
    qc::print_environment("phase2_quant_decode");
    const bool ok = run_correctness();
    if (do_bench) run_bench();
    std::printf("%s\n", ok ? "ALL PASS" : "FAIL");
    return ok ? 0 : 1;
}
