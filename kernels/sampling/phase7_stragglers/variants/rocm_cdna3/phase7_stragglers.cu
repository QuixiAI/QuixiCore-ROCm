/**
 * @file
 * @brief Phase 7 sampling, embedding, and root-collective parity ports.
 */
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

#include "../../../../common/cdna3_harness.cuh"

namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();

enum StorageType : int {
    kF32 = 0,
    kF16 = 1,
    kBF16 = 2,
};

__device__ __forceinline__ float load_storage(const void *ptr, int type,
                                              size_t index) {
    if (type == kF16) {
        return __half2float(reinterpret_cast<const __half *>(ptr)[index]);
    }
    if (type == kBF16) {
        return __bfloat162float(
            reinterpret_cast<const __hip_bfloat16 *>(ptr)[index]);
    }
    return reinterpret_cast<const float *>(ptr)[index];
}

__device__ __forceinline__ void store_storage(void *ptr, int type, size_t index,
                                              float value) {
    if (type == kF16) {
        reinterpret_cast<__half *>(ptr)[index] = __float2half(value);
    } else if (type == kBF16) {
        reinterpret_cast<__hip_bfloat16 *>(ptr)[index] = __float2bfloat16(value);
    } else {
        reinterpret_cast<float *>(ptr)[index] = value;
    }
}

template <typename T>
std::vector<double> to_ref(const std::vector<T> &values) {
    std::vector<double> out(values.size());
    for (size_t i = 0; i < values.size(); ++i) out[i] = qc::to_double(values[i]);
    return out;
}

std::vector<double> to_ref_float(const std::vector<float> &values) {
    return std::vector<double>(values.begin(), values.end());
}

template <typename T>
std::vector<double> storage_ref(const std::vector<T> &values) {
    std::vector<double> out(values.size());
    for (size_t i = 0; i < values.size(); ++i) out[i] = qc::to_double(values[i]);
    return out;
}

// ---------------------------------------------------------------------------
// Sampling renormalizers and softcap
// ---------------------------------------------------------------------------

__global__ void top_k_renorm_scalar_kernel(const float *probabilities,
                                           float *out, long long rows,
                                           long long vocab, int k) {
    const long long row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const long long base = row * vocab;
    int chosen_idx[64];
    float chosen_val[64];
    for (int i = 0; i < k; ++i) {
        chosen_idx[i] = vocab + i;
        chosen_val[i] = -1.0f;
    }
    for (long long token = 0; token < vocab; ++token) {
        const float value = probabilities[base + token];
        int pos = k;
        for (int item = 0; item < k; ++item) {
            if (value > chosen_val[item] ||
                (value == chosen_val[item] && token < chosen_idx[item])) {
                pos = item;
                break;
            }
        }
        if (pos < k) {
            for (int item = k - 1; item > pos; --item) {
                chosen_idx[item] = chosen_idx[item - 1];
                chosen_val[item] = chosen_val[item - 1];
            }
            chosen_idx[pos] = int(token);
            chosen_val[pos] = value;
        }
    }
    float sum = 0.0f;
    for (int i = 0; i < k; ++i) sum += chosen_val[i];
    const float inverse = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (long long token = 0; token < vocab; ++token) out[base + token] = 0.0f;
    for (int i = 0; i < k; ++i) out[base + chosen_idx[i]] = chosen_val[i] * inverse;
}

__global__ void top_k_renorm_rank_kernel(const float *probabilities, float *out,
                                         long long rows, long long vocab,
                                         int k) {
    const long long row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= rows) return;
    const long long base = row * vocab;
    __shared__ float partial[256];
    float local = 0.0f;
    for (long long token = tid; token < vocab; token += blockDim.x) {
        const float value = probabilities[base + token];
        int rank = 0;
        for (long long other = 0; other < vocab; ++other) {
            const float ov = probabilities[base + other];
            rank += (ov > value || (ov == value && other < token)) ? 1 : 0;
        }
        if (rank < k) local += value;
    }
    partial[tid] = local;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    const float inverse = partial[0] > 0.0f ? 1.0f / partial[0] : 0.0f;
    for (long long token = tid; token < vocab; token += blockDim.x) {
        const float value = probabilities[base + token];
        int rank = 0;
        for (long long other = 0; other < vocab; ++other) {
            const float ov = probabilities[base + other];
            rank += (ov > value || (ov == value && other < token)) ? 1 : 0;
        }
        out[base + token] = rank < k ? value * inverse : 0.0f;
    }
}

__global__ void top_p_renorm_scalar_kernel(const float *probabilities,
                                           float *out, long long rows,
                                           long long vocab, float p) {
    const long long row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const long long base = row * vocab;
    float total = 0.0f;
    for (long long token = 0; token < vocab; ++token) {
        out[base + token] = 0.0f;
        total += probabilities[base + token];
    }
    const float target = p * total;
    float sum = 0.0f;
    while (sum < target) {
        int best = -1;
        float best_value = -1.0f;
        for (long long token = 0; token < vocab; ++token) {
            if (out[base + token] != 0.0f) continue;
            const float value = probabilities[base + token];
            if (value > best_value ||
                (value == best_value && (best < 0 || token < best))) {
                best = int(token);
                best_value = value;
            }
        }
        if (best < 0) break;
        out[base + best] = best_value;
        sum += best_value;
    }
    const float inverse = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (long long token = 0; token < vocab; ++token) out[base + token] *= inverse;
}

__global__ void top_p_renorm_rank_kernel(const float *probabilities, float *out,
                                         long long rows, long long vocab,
                                         float p) {
    const long long row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= rows) return;
    const long long base = row * vocab;
    __shared__ float partial[256];
    __shared__ float total;
    float local_total = 0.0f;
    for (long long token = tid; token < vocab; token += blockDim.x) {
        local_total += probabilities[base + token];
    }
    partial[tid] = local_total;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    if (tid == 0) total = partial[0];
    __syncthreads();
    const float target = p * total;
    float kept_local = 0.0f;
    for (long long token = tid; token < vocab; token += blockDim.x) {
        const float value = probabilities[base + token];
        float prefix = 0.0f;
        for (long long other = 0; other < vocab; ++other) {
            const float ov = probabilities[base + other];
            if (ov > value || (ov == value && other < token)) prefix += ov;
        }
        if (prefix < target) kept_local += value;
    }
    partial[tid] = kept_local;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    const float inverse = partial[0] > 0.0f ? 1.0f / partial[0] : 0.0f;
    for (long long token = tid; token < vocab; token += blockDim.x) {
        const float value = probabilities[base + token];
        float prefix = 0.0f;
        for (long long other = 0; other < vocab; ++other) {
            const float ov = probabilities[base + other];
            if (ov > value || (ov == value && other < token)) prefix += ov;
        }
        out[base + token] = prefix < target ? value * inverse : 0.0f;
    }
}

__global__ void logits_softcap_scalar_kernel(const float *logits, float *out,
                                             long long count, float cap) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const float inverse = 1.0f / cap;
    for (long long index = 0; index < count; ++index) {
        out[index] = cap * tanhf(logits[index] * inverse);
    }
}

__global__ void logits_softcap_kernel(const float *logits, float *out,
                                      long long count, float cap) {
    const long long index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    out[index] = cap * tanhf(logits[index] / cap);
}

std::vector<float> top_k_ref(const std::vector<float> &probabilities,
                             long long rows, long long vocab, int k) {
    std::vector<float> out(probabilities.size(), 0.0f);
    for (long long row = 0; row < rows; ++row) {
        std::vector<int> ids(static_cast<size_t>(vocab));
        std::iota(ids.begin(), ids.end(), 0);
        const float *pr = probabilities.data() + row * vocab;
        std::stable_sort(ids.begin(), ids.end(), [&](int lhs, int rhs) {
            return pr[lhs] == pr[rhs] ? lhs < rhs : pr[lhs] > pr[rhs];
        });
        double sum = 0.0;
        for (int i = 0; i < k; ++i) sum += pr[ids[i]];
        for (int i = 0; i < k; ++i) out[row * vocab + ids[i]] = float(pr[ids[i]] / sum);
    }
    return out;
}

std::vector<float> top_p_ref(const std::vector<float> &probabilities,
                             long long rows, long long vocab, float p) {
    std::vector<float> out(probabilities.size(), 0.0f);
    for (long long row = 0; row < rows; ++row) {
        std::vector<int> ids(static_cast<size_t>(vocab));
        std::iota(ids.begin(), ids.end(), 0);
        const float *pr = probabilities.data() + row * vocab;
        double total = 0.0;
        for (long long token = 0; token < vocab; ++token) total += pr[token];
        std::stable_sort(ids.begin(), ids.end(), [&](int lhs, int rhs) {
            return pr[lhs] == pr[rhs] ? lhs < rhs : pr[lhs] > pr[rhs];
        });
        double sum = 0.0;
        size_t keep = 0;
        do {
            sum += pr[ids[keep++]];
        } while (keep < ids.size() && sum < p * total);
        for (size_t i = 0; i < keep; ++i) out[row * vocab + ids[i]] = float(pr[ids[i]] / sum);
    }
    return out;
}

std::vector<float> softcap_ref(const std::vector<float> &logits, float cap) {
    std::vector<float> out(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        out[i] = cap * std::tanh(logits[i] / cap);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Embedding lookup with token and type tables
// ---------------------------------------------------------------------------

__global__ void embedding_lookup_types_kernel(
    const int *token_ids, const int *type_ids, const void *token_table,
    const void *type_table, void *out, long long token_vocab,
    long long type_vocab, long long count, long long dim, float token_scale,
    int storage_type) {
    const long long index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count * dim) return;
    const long long token = index / dim;
    const long long feature = index - token * dim;
    const int token_id = token_ids[token];
    const int type_id = type_ids[token];
    const float token_value =
        (token_id >= 0 && token_id < token_vocab)
            ? load_storage(token_table, storage_type,
                           static_cast<size_t>(static_cast<long long>(token_id) * dim + feature))
            : 0.0f;
    const float type_value =
        (type_id >= 0 && type_id < type_vocab)
            ? load_storage(type_table, storage_type,
                           static_cast<size_t>(static_cast<long long>(type_id) * dim + feature))
            : 0.0f;
    store_storage(out, storage_type, static_cast<size_t>(index),
                  token_scale * token_value + type_value);
}

template <typename T>
std::vector<T> cast_storage(const std::vector<float> &values);

template <>
std::vector<float> cast_storage<float>(const std::vector<float> &values) {
    return values;
}

template <>
std::vector<__half> cast_storage<__half>(const std::vector<float> &values) {
    return qc::to_storage<__half>(values);
}

template <>
std::vector<__hip_bfloat16> cast_storage<__hip_bfloat16>(
    const std::vector<float> &values) {
    return qc::to_storage<__hip_bfloat16>(values);
}

std::vector<float> embedding_ref_f32(const std::vector<float> &token_table,
                                     const std::vector<float> &type_table,
                                     const std::vector<int> &token_ids,
                                     const std::vector<int> &type_ids,
                                     long long token_vocab,
                                     long long type_vocab, long long count,
                                     long long dim, float token_scale) {
    std::vector<float> out(size_t(count * dim), 0.0f);
    for (long long token = 0; token < count; ++token) {
        const int token_id = token_ids[token];
        const int type_id = type_ids[token];
        for (long long feature = 0; feature < dim; ++feature) {
            const float tv = token_id >= 0 && token_id < token_vocab
                                 ? token_scale * token_table[token_id * dim + feature]
                                 : 0.0f;
            const float ty = type_id >= 0 && type_id < type_vocab
                                 ? type_table[type_id * dim + feature]
                                 : 0.0f;
            out[token * dim + feature] = tv + ty;
        }
    }
    return out;
}

template <typename T>
std::vector<T> embedding_ref_typed(const std::vector<T> &token_table,
                                   const std::vector<T> &type_table,
                                   const std::vector<int> &token_ids,
                                   const std::vector<int> &type_ids,
                                   long long token_vocab,
                                   long long type_vocab, long long count,
                                   long long dim, float token_scale) {
    std::vector<float> widened(size_t(count * dim), 0.0f);
    for (long long token = 0; token < count; ++token) {
        const int token_id = token_ids[token];
        const int type_id = type_ids[token];
        for (long long feature = 0; feature < dim; ++feature) {
            const float token_value =
                token_id >= 0 && token_id < token_vocab
                    ? float(qc::to_double(
                          token_table[size_t(token_id) * size_t(dim) +
                                      size_t(feature)]))
                    : 0.0f;
            const float type_value =
                type_id >= 0 && type_id < type_vocab
                    ? float(qc::to_double(
                          type_table[size_t(type_id) * size_t(dim) +
                                     size_t(feature)]))
                    : 0.0f;
            widened[token * dim + feature] = token_scale * token_value + type_value;
        }
    }
    return cast_storage<T>(widened);
}

// ---------------------------------------------------------------------------
// Root collectives as single-device tensor kernels
// ---------------------------------------------------------------------------

__global__ void broadcast_scalar_kernel(const float *input, float *output,
                                        long long world, long long count,
                                        int root) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (long long rank = 0; rank < world; ++rank) {
        for (long long item = 0; item < count; ++item) {
            output[rank * count + item] =
                input[static_cast<long long>(root) * count + item];
        }
    }
}

__global__ void broadcast_kernel(const float *input, float *output,
                                 long long world, long long count, int root) {
    const long long index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= world * count) return;
    const long long item = index % count;
    output[index] = input[static_cast<long long>(root) * count + item];
}

__global__ void reduce_sum_scalar_kernel(const float *input, float *output,
                                         long long world, long long count,
                                         int root) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (long long item = 0; item < count; ++item) {
        float sum = 0.0f;
        for (long long rank = 0; rank < world; ++rank) {
            sum += input[rank * count + item];
        }
        output[static_cast<long long>(root) * count + item] = sum;
    }
}

__global__ void reduce_sum_kernel(const float *input, float *output,
                                  long long world, long long count, int root) {
    const long long item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= count) return;
    float sum = 0.0f;
    for (long long rank = 0; rank < world; ++rank) sum += input[rank * count + item];
    output[static_cast<long long>(root) * count + item] = sum;
}

std::vector<float> broadcast_ref(const std::vector<float> &input, long long world,
                                 long long count, int root) {
    std::vector<float> out(size_t(world * count), 0.0f);
    for (long long rank = 0; rank < world; ++rank) {
        std::copy_n(input.data() + static_cast<long long>(root) * count, count,
                    out.data() + rank * count);
    }
    return out;
}

std::vector<float> reduce_sum_ref(const std::vector<float> &input, long long world,
                                  long long count, int root) {
    std::vector<float> out(size_t(world * count), 0.0f);
    for (long long item = 0; item < count; ++item) {
        double sum = 0.0;
        for (long long rank = 0; rank < world; ++rank) sum += input[rank * count + item];
        out[static_cast<long long>(root) * count + item] = float(sum);
    }
    return out;
}

std::vector<float> normalized_probs(qc::Rng &rng, long long rows,
                                    long long vocab) {
    std::vector<float> p(size_t(rows * vocab));
    for (long long row = 0; row < rows; ++row) {
        double sum = 0.0;
        for (long long token = 0; token < vocab; ++token) {
            float value = 0.001f + std::fabs(rng.uniform(-1.0f, 1.0f));
            if (token % 97 == 0) value = 0.03125f;
            p[row * vocab + token] = value;
            sum += value;
        }
        for (long long token = 0; token < vocab; ++token) {
            p[row * vocab + token] = float(p[row * vocab + token] / sum);
        }
    }
    return p;
}

bool run_correctness() {
    bool ok = true;
    int checks = 0;
    qc::Rng rng(0x711);

    {
        constexpr long long rows = 5;
        constexpr long long vocab = 257;
        constexpr int k = 17;
        auto probs = normalized_probs(rng, rows, vocab);
        float *dp = qc::dnew(probs);
        float *dout = qc::dzero<float>(probs.size());
        top_k_renorm_rank_kernel<<<rows, 256>>>(dp, dout, rows, vocab, k);
        QC_SYNC();
        const auto ref = top_k_ref(probs, rows, vocab, k);
        ok &= qc::compare(qc::d2h(dout, probs.size()), to_ref_float(ref),
                          qc::Tol::fp32().with_elementwise(2e-5, 2e-6))
                  .report("top_k_renorm stable");
        ++checks;
        qc::dfree(dp, dout);
    }

    {
        constexpr long long rows = 4;
        constexpr long long vocab = 251;
        constexpr float p = 0.83f;
        auto probs = normalized_probs(rng, rows, vocab);
        float *dp = qc::dnew(probs);
        float *dout = qc::dzero<float>(probs.size());
        top_p_renorm_rank_kernel<<<rows, 256>>>(dp, dout, rows, vocab, p);
        QC_SYNC();
        const auto ref = top_p_ref(probs, rows, vocab, p);
        ok &= qc::compare(qc::d2h(dout, probs.size()), to_ref_float(ref),
                          qc::Tol::fp32().with_elementwise(2e-5, 2e-6))
                  .report("top_p_renorm stable");
        ++checks;
        qc::dfree(dp, dout);
    }

    {
        constexpr long long count = 4099;
        auto logits = rng.uniforms(size_t(count), -9.0f, 9.0f);
        constexpr float cap = 5.5f;
        float *dl = qc::dnew(logits);
        float *dout = qc::dzero<float>(logits.size());
        logits_softcap_kernel<<<qc::grid_for(logits.size(), 256), 256>>>(
            dl, dout, count, cap);
        QC_SYNC();
        const auto ref = softcap_ref(logits, cap);
        ok &= qc::compare(qc::d2h(dout, logits.size()), to_ref_float(ref),
                          qc::Tol::fp32())
                  .report("logits_softcap");
        ++checks;
        qc::dfree(dl, dout);
    }

    {
        constexpr long long token_vocab = 13;
        constexpr long long type_vocab = 4;
        constexpr long long count = 7;
        constexpr long long dim = 65;
        const std::vector<int> token_ids = {0, 3, -1, 12, 20, 6, 2};
        const std::vector<int> type_ids = {0, 1, 2, -1, 3, 99, 2};
        auto token_table = rng.uniforms(size_t(token_vocab * dim), -2.0f, 2.0f);
        auto type_table = rng.uniforms(size_t(type_vocab * dim), -0.5f, 0.5f);
        const float token_scale = 0.75f;
        const auto ref_f32 = embedding_ref_f32(token_table, type_table, token_ids,
                                               type_ids, token_vocab, type_vocab,
                                               count, dim, token_scale);
        for (int storage : {kF32, kF16, kBF16}) {
            int *dtok = qc::dnew(token_ids);
            int *dtyp = qc::dnew(type_ids);
            if (storage == kF32) {
                float *dtt = qc::dnew(token_table);
                float *dty = qc::dnew(type_table);
                float *dout = qc::dzero<float>(size_t(count * dim));
                embedding_lookup_types_kernel<<<qc::grid_for(size_t(count * dim), 256), 256>>>(
                    dtok, dtyp, dtt, dty, dout, token_vocab, type_vocab, count,
                    dim, token_scale, storage);
                QC_SYNC();
                ok &= qc::compare(qc::d2h(dout, size_t(count * dim)),
                                  to_ref_float(ref_f32), qc::Tol::fp32())
                          .report("embedding_lookup_types fp32");
                qc::dfree(dtt, dty, dout);
            } else if (storage == kF16) {
                auto htok = cast_storage<__half>(token_table);
                auto htyp = cast_storage<__half>(type_table);
                auto href = embedding_ref_typed(htok, htyp, token_ids, type_ids,
                                                token_vocab, type_vocab, count,
                                                dim, token_scale);
                __half *dtt = qc::dnew(htok);
                __half *dty = qc::dnew(htyp);
                __half *dout = qc::dzero<__half>(size_t(count * dim));
                embedding_lookup_types_kernel<<<qc::grid_for(size_t(count * dim), 256), 256>>>(
                    dtok, dtyp, dtt, dty, dout, token_vocab, type_vocab, count,
                    dim, token_scale, storage);
                QC_SYNC();
                ok &= qc::compare(qc::d2h(dout, size_t(count * dim)),
                                  storage_ref(href), qc::Tol::fp16_output())
                          .report("embedding_lookup_types fp16");
                qc::dfree(dtt, dty, dout);
            } else {
                auto htok = cast_storage<__hip_bfloat16>(token_table);
                auto htyp = cast_storage<__hip_bfloat16>(type_table);
                auto href = embedding_ref_typed(htok, htyp, token_ids, type_ids,
                                                token_vocab, type_vocab, count,
                                                dim, token_scale);
                __hip_bfloat16 *dtt = qc::dnew(htok);
                __hip_bfloat16 *dty = qc::dnew(htyp);
                __hip_bfloat16 *dout = qc::dzero<__hip_bfloat16>(size_t(count * dim));
                embedding_lookup_types_kernel<<<qc::grid_for(size_t(count * dim), 256), 256>>>(
                    dtok, dtyp, dtt, dty, dout, token_vocab, type_vocab, count,
                    dim, token_scale, storage);
                QC_SYNC();
                ok &= qc::compare(qc::d2h(dout, size_t(count * dim)),
                                  storage_ref(href), qc::Tol::bf16_output())
                          .report("embedding_lookup_types bf16");
                qc::dfree(dtt, dty, dout);
            }
            ++checks;
            qc::dfree(dtok, dtyp);
        }
    }

    {
        constexpr long long world = 4;
        constexpr long long count = 513;
        constexpr int root = 2;
        auto input = rng.uniforms(size_t(world * count), -2.0f, 2.0f);
        float *din = qc::dnew(input);
        float *dout = qc::dzero<float>(input.size());
        broadcast_kernel<<<qc::grid_for(input.size(), 256), 256>>>(
            din, dout, world, count, root);
        QC_SYNC();
        ok &= qc::compare(qc::d2h(dout, input.size()),
                          to_ref_float(broadcast_ref(input, world, count, root)),
                          qc::Tol::fp32())
                  .report("broadcast root");
        ++checks;
        QC_CHECK(hipMemset(dout, 0, input.size() * sizeof(float)));
        reduce_sum_kernel<<<qc::grid_for(size_t(count), 256), 256>>>(
            din, dout, world, count, root);
        QC_SYNC();
        ok &= qc::compare(qc::d2h(dout, input.size()),
                          to_ref_float(reduce_sum_ref(input, world, count, root)),
                          qc::Tol::fp32().with_elementwise(2e-5, 2e-6))
                  .report("reduce_sum root");
        ++checks;
        qc::dfree(din, dout);
    }

    std::printf("Phase 7 correctness checks: %d\n", checks);
    return ok;
}

template <typename Fn>
qc::Bench bench_per_launch(Fn &&fn, int warmups = 10, int iters = 50,
                           int repeats = 1) {
    qc::Bench b = qc::bench([&] {
        for (int repeat = 0; repeat < repeats; ++repeat) fn();
        QC_CHECK(hipGetLastError());
    }, warmups, iters);
    if (repeats > 1) {
        b.median_ms /= double(repeats);
        b.min_ms /= double(repeats);
        b.max_ms /= double(repeats);
        b.mean_ms /= double(repeats);
    }
    return b;
}

void report_pair(const char *label, const qc::Bench &baseline,
                 const qc::Bench &candidate, double bytes, bool compute = false) {
    if (compute) {
        baseline.report_compute(std::string(label) + " scalar", bytes);
        candidate.report_compute(std::string(label) + " candidate", bytes);
    } else {
        baseline.report_bandwidth(std::string(label) + " scalar", bytes);
        candidate.report_bandwidth(std::string(label) + " candidate", bytes);
    }
    qc::report_ab(label, baseline, candidate);
}

void run_benchmarks() {
    std::printf("\n== Phase 7 benchmarks ==\n");
    std::printf("   Timing note: medians are per launch; fast kernels use inner repeats.\n");
    qc::Rng rng(0x7007);

    {
        constexpr long long rows = 128;
        constexpr long long vocab = 1024;
        constexpr int k = 16;
        auto probs = normalized_probs(rng, rows, vocab);
        float *dp = qc::dnew(probs);
        float *dout = qc::dzero<float>(probs.size());
        const double bytes = double(probs.size()) * sizeof(float) * 2.0;
        const auto scalar = bench_per_launch([&] {
            top_k_renorm_scalar_kernel<<<qc::grid_for(rows, 128), 128>>>(
                dp, dout, rows, vocab, k);
        }, 10, 40);
        const auto rank = bench_per_launch([&] {
            top_k_renorm_rank_kernel<<<rows, 256>>>(dp, dout, rows, vocab, k);
        }, 10, 40);
        report_pair("top_k_renorm", scalar, rank, bytes);
        qc::dfree(dp, dout);
    }

    {
        constexpr long long rows = 64;
        constexpr long long vocab = 1024;
        constexpr float p = 0.9f;
        auto probs = normalized_probs(rng, rows, vocab);
        float *dp = qc::dnew(probs);
        float *dout = qc::dzero<float>(probs.size());
        const double bytes = double(probs.size()) * sizeof(float) * 2.0;
        const auto scalar = bench_per_launch([&] {
            top_p_renorm_scalar_kernel<<<qc::grid_for(rows, 128), 128>>>(
                dp, dout, rows, vocab, p);
        }, 10, 40);
        const auto rank = bench_per_launch([&] {
            top_p_renorm_rank_kernel<<<rows, 256>>>(dp, dout, rows, vocab, p);
        }, 10, 40);
        report_pair("top_p_renorm", scalar, rank, bytes);
        qc::dfree(dp, dout);
    }

    {
        constexpr long long count = 16 * 1024 * 1024;
        auto logits = rng.uniforms(size_t(count), -9.0f, 9.0f);
        float *dl = qc::dnew(logits);
        float *dout = qc::dzero<float>(logits.size());
        const double bytes = double(count) * sizeof(float) * 2.0;
        const auto scalar = bench_per_launch([&] {
            logits_softcap_scalar_kernel<<<1, 1>>>(dl, dout, count, 6.0f);
        }, 2, 12);
        const auto parallel = bench_per_launch([&] {
            logits_softcap_kernel<<<qc::grid_for(logits.size(), 256), 256>>>(
                dl, dout, count, 6.0f);
        }, 10, 50, 4);
        report_pair("logits_softcap", scalar, parallel, bytes);
        qc::dfree(dl, dout);
    }

    {
        constexpr long long token_vocab = 65536;
        constexpr long long type_vocab = 8;
        constexpr long long count = 65536;
        constexpr long long dim = 256;
        auto token_table = rng.uniforms(size_t(token_vocab * dim), -1.0f, 1.0f);
        auto type_table = rng.uniforms(size_t(type_vocab * dim), -0.1f, 0.1f);
        std::vector<int> token_ids(static_cast<size_t>(count));
        std::vector<int> type_ids(static_cast<size_t>(count));
        for (long long i = 0; i < count; ++i) {
            token_ids[i] = int((i * 4099) % token_vocab);
            type_ids[i] = int(i % type_vocab);
        }
        float *dtt = qc::dnew(token_table);
        float *dty = qc::dnew(type_table);
        int *dtok = qc::dnew(token_ids);
        int *dtyp = qc::dnew(type_ids);
        float *dout = qc::dzero<float>(size_t(count * dim));
        const double bytes = double(count * dim) * sizeof(float) * 2.0 +
                             double(count) * sizeof(int) * 2.0;
        const auto emb = bench_per_launch([&] {
            embedding_lookup_types_kernel<<<qc::grid_for(size_t(count * dim), 256), 256>>>(
                dtok, dtyp, dtt, dty, dout, token_vocab, type_vocab, count, dim,
                1.0f, kF32);
        }, 10, 50);
        emb.report_bandwidth("embedding_lookup_types fp32", bytes);
        qc::dfree(dtt, dty, dtok, dtyp, dout);
    }

    {
        constexpr long long world = 8;
        constexpr long long count = 1024 * 1024;
        constexpr int root = 3;
        auto input = rng.uniforms(size_t(world * count), -2.0f, 2.0f);
        float *din = qc::dnew(input);
        float *dout = qc::dzero<float>(input.size());
        const double broadcast_bytes = double(world * count + count) * sizeof(float);
        const auto bscalar = bench_per_launch([&] {
            broadcast_scalar_kernel<<<1, 1>>>(din, dout, world, count, root);
        }, 2, 20);
        const auto bparallel = bench_per_launch([&] {
            broadcast_kernel<<<qc::grid_for(input.size(), 256), 256>>>(
                din, dout, world, count, root);
        }, 10, 50, 128);
        report_pair("broadcast", bscalar, bparallel, broadcast_bytes);
        const double reduce_bytes = double(world * count + count) * sizeof(float);
        const auto rscalar = bench_per_launch([&] {
            reduce_sum_scalar_kernel<<<1, 1>>>(din, dout, world, count, root);
        }, 2, 20);
        const auto rparallel = bench_per_launch([&] {
            reduce_sum_kernel<<<qc::grid_for(size_t(count), 256), 256>>>(
                din, dout, world, count, root);
        }, 10, 50, 128);
        report_pair("reduce_sum", rscalar, rparallel, reduce_bytes);
        qc::dfree(din, dout);
    }
}

}  // namespace

int main(int argc, char **argv) {
    qc::print_environment("phase7_stragglers");
    const bool ok = run_correctness();
    if (qc::bench_requested(argc, argv)) run_benchmarks();
    return qc::finish(ok);
}
