/**
 * @file
 * @brief CDNA3 (gfx942) composite attention shapes: Swin windowed attention over
 *        packed QKV, fused decode-with-cache-append, and multi-level cascade
 *        attention over a prefix plus a paged cache.
 *
 * Semantic sources in ../QuixiCore-CPU/kernels/attention/:
 *   swin_attention_d32       attention_extended_ref.cpp:602
 *   decode_cache_attention   attention_composites_ref.cpp:332
 *   cascade_attention_multi  attention_composites_ref.cpp:470
 *
 * Grouped as one operation directory, matching `kernels/serving` and
 * `kernels/linear_attention`. All three are decode/composite shapes with a small
 * head dim and one output row per (request, head), so they share their geometry.
 *
 * What makes each one distinct, and where a careless port goes wrong:
 *
 * swin_attention_d32 -- QKV is interleaved at the token level, not split into
 *   three tensors: qkv[(((w*tokens + pos)*3 + which)*heads + head)*32 + d] with
 *   which in {0=Q, 1=K, 2=V}. The scale is a hardcoded 1/sqrt(32), the relative
 *   bias is per head, and the optional mask is indexed by `window %
 *   windows_per_image` -- the mask repeats per image, so using the raw window
 *   index silently applies the wrong shift to every window past the first image.
 *
 * decode_cache_attention -- two dependent stages. The K/V append RoPEs and
 *   optionally RMS-norms the new K, writes it plus V into the cache at slot
 *   context_lengths[item], and only then does the query attend over slots
 *   [0, context_lengths[item]] INCLUSIVE. The inclusive bound is the point: the
 *   token just appended must be visible to this same call. Two kernels with a
 *   launch boundary between them, because the second reads what the first wrote
 *   across workgroups.
 *
 * cascade_attention_multi -- one online softmax spans several prefix levels and
 *   then a paged cache, consumed in that order. Prefix levels are separate
 *   allocations reached through a pointer array; the paged tail is gathered via
 *   block_table. Both feed the same running max/denominator, so they cannot be
 *   computed independently and merged without a separate merge step.
 *
 *   hipcc -std=c++17 -O3 --offload-arch=gfx942 attn_composites.cu -o attn_composites.out
 */
#include "../../../../common/cdna3_harness.cuh"

using qc::StorageF32;

// A 32-lane row packs two rows per wavefront and leaves no lane idle. The RoPE
// A/B in kernels/attention/rope_variants established that for already-packed
// rows this beats a 64-lane row, so these kernels start there rather than
// re-deriving it.
static constexpr int kBlock = 256;
static constexpr int kLanes = 32;

template <int LANES>
__device__ __forceinline__ float row_reduce_sum(float v) {
#pragma unroll
    for (int offset = LANES / 2; offset > 0; offset >>= 1)
        v += __shfl_xor(v, offset, LANES);
    return v;
}

// ===========================================================================
// 1. swin_attention_d32
// ===========================================================================

constexpr int kSwinHeadDim = 32;
// The reference hardcodes this rather than computing 1/sqrt(head_dim); matched
// bit-for-bit so the scale cannot drift.
constexpr float kSwinScale = 0.1767766952966369f;

/// Baseline: K and V are re-read from global memory for every query position.
template <int LANES>
__global__ void k_swin_global(const float *__restrict__ qkv,
                              const float *__restrict__ relative_bias,
                              const float *__restrict__ mask, float *__restrict__ out,
                              int windows, int tokens, int heads,
                              int windows_per_image) {
    const long long row = (long long)blockIdx.x * (kBlock / LANES) + threadIdx.x / LANES;
    const long long rows = (long long)windows * heads * tokens;
    if (row >= rows) return;
    const int lane = threadIdx.x % LANES;

    const int query_position = row % tokens;
    const int head = (row / tokens) % heads;
    const int window = row / ((long long)tokens * heads);
    const int mask_window = windows_per_image > 0 ? window % windows_per_image : 0;

    auto qkv_at = [&](int position, int which) {
        return (long long)((((long long)window * tokens + position) * 3 + which) * heads +
                           head) *
               kSwinHeadDim;
    };
    const long long q_base = qkv_at(query_position, 0);
    const float q_val = lane < kSwinHeadDim ? qkv[q_base + lane] : 0.0f;

    float acc = 0.0f, running_max = -INFINITY, denominator = 0.0f;
    for (int key_position = 0; key_position < tokens; ++key_position) {
        const long long k_base = qkv_at(key_position, 1);
        float dot = lane < kSwinHeadDim ? q_val * qkv[k_base + lane] : 0.0f;
        dot = row_reduce_sum<LANES>(dot);
        float score =
            dot * kSwinScale +
            relative_bias[((long long)head * tokens + query_position) * tokens +
                          key_position];
        if (mask != nullptr)
            score += mask[((long long)mask_window * tokens + query_position) * tokens +
                          key_position];

        const float next_max = fmaxf(running_max, score);
        const float old_weight =
            (running_max == -INFINITY) ? 0.0f : __expf(running_max - next_max);
        const float new_weight = __expf(score - next_max);
        denominator = denominator * old_weight + new_weight;
        const long long v_base = qkv_at(key_position, 2);
        acc = acc * old_weight +
              (lane < kSwinHeadDim ? qkv[v_base + lane] : 0.0f) * new_weight;
        running_max = next_max;
    }
    if (lane < kSwinHeadDim) {
        const long long o_base =
            (long long)(((long long)window * tokens + query_position) * heads + head) *
            kSwinHeadDim;
        out[o_base + lane] = acc / denominator;
    }
}

/// Candidate: one workgroup owns a whole (window, head), staging that window's K
/// and V in LDS once and reusing them across all its query positions. A Swin
/// window is small (49 or 64 tokens), so 2 * tokens * 32 floats fits easily.
template <int MAX_TOKENS>
__global__ __launch_bounds__(256) void k_swin_lds(
    const float *__restrict__ qkv, const float *__restrict__ relative_bias,
    const float *__restrict__ mask, float *__restrict__ out, int windows, int tokens,
    int heads, int windows_per_image) {
    __shared__ float sK[MAX_TOKENS * kSwinHeadDim];
    __shared__ float sV[MAX_TOKENS * kSwinHeadDim];

    const int head = blockIdx.x % heads;
    const int window = blockIdx.x / heads;
    const int mask_window = windows_per_image > 0 ? window % windows_per_image : 0;

    auto qkv_at = [&](int position, int which) {
        return (long long)((((long long)window * tokens + position) * 3 + which) * heads +
                           head) *
               kSwinHeadDim;
    };

    for (int i = threadIdx.x; i < tokens * kSwinHeadDim; i += blockDim.x) {
        const int position = i / kSwinHeadDim, d = i % kSwinHeadDim;
        sK[i] = qkv[qkv_at(position, 1) + d];
        sV[i] = qkv[qkv_at(position, 2) + d];
    }
    __syncthreads();

    const int lane = threadIdx.x % kLanes;
    for (int query_position = threadIdx.x / kLanes; query_position < tokens;
         query_position += blockDim.x / kLanes) {
        const float q_val =
            lane < kSwinHeadDim ? qkv[qkv_at(query_position, 0) + lane] : 0.0f;
        float acc = 0.0f, running_max = -INFINITY, denominator = 0.0f;
        for (int key_position = 0; key_position < tokens; ++key_position) {
            float dot = lane < kSwinHeadDim
                            ? q_val * sK[key_position * kSwinHeadDim + lane]
                            : 0.0f;
            dot = row_reduce_sum<kLanes>(dot);
            float score =
                dot * kSwinScale +
                relative_bias[((long long)head * tokens + query_position) * tokens +
                              key_position];
            if (mask != nullptr)
                score += mask[((long long)mask_window * tokens + query_position) * tokens +
                              key_position];

            const float next_max = fmaxf(running_max, score);
            const float old_weight =
                (running_max == -INFINITY) ? 0.0f : __expf(running_max - next_max);
            const float new_weight = __expf(score - next_max);
            denominator = denominator * old_weight + new_weight;
            acc = acc * old_weight +
                  (lane < kSwinHeadDim ? sV[key_position * kSwinHeadDim + lane] : 0.0f) *
                      new_weight;
            running_max = next_max;
        }
        if (lane < kSwinHeadDim) {
            const long long o_base =
                (long long)(((long long)window * tokens + query_position) * heads + head) *
                kSwinHeadDim;
            out[o_base + lane] = acc / denominator;
        }
    }
}

// ===========================================================================
// 2. decode_cache_attention -- stage 1: RoPE + optional norm + cache append
// ===========================================================================

template <int LANES>
__global__ void k_decode_cache_append(
    const float *__restrict__ new_k, const float *__restrict__ new_v,
    const float *__restrict__ cosine, const float *__restrict__ sine,
    const int *__restrict__ positions, const int *__restrict__ context_lengths,
    const float *__restrict__ k_weight, float *__restrict__ key_cache,
    float *__restrict__ value_cache, int batch, int kv_heads, int cache_length,
    int head_dim, float eps, int do_k_norm, float gemma_offset) {
    const long long row = (long long)blockIdx.x * (kBlock / LANES) + threadIdx.x / LANES;
    if (row >= (long long)batch * kv_heads) return;
    const int lane = threadIdx.x % LANES;
    const int item = row / kv_heads;
    const int half = head_dim / 2;
    const long long source_offset = row * head_dim;

    float inverse = 1.0f;
    if (do_k_norm) {
        float squares = 0.0f;
        for (int d = lane; d < head_dim; d += LANES) {
            const float v = new_k[source_offset + d];
            squares = fmaf(v, v, squares);
        }
        squares = row_reduce_sum<LANES>(squares);
        inverse = 1.0f / sqrtf(squares / (float)head_dim + eps);
    }

    const long long table = (long long)positions[item] * half;
    const long long cache_offset =
        (row * cache_length + context_lengths[item]) * head_dim;
    for (int d = lane; d < half; d += LANES) {
        const float w0 = do_k_norm ? k_weight[d] + gemma_offset : 1.0f;
        const float w1 = do_k_norm ? k_weight[half + d] + gemma_offset : 1.0f;
        const float first = new_k[source_offset + d] * inverse * w0;
        const float second = new_k[source_offset + half + d] * inverse * w1;
        const float c = cosine[table + d], s = sine[table + d];
        key_cache[cache_offset + d] = first * c - second * s;
        key_cache[cache_offset + half + d] = second * c + first * s;
    }
    for (int d = lane; d < head_dim; d += LANES)
        value_cache[cache_offset + d] = new_v[source_offset + d];
}

// --- stage 2: rotate Q, then attend over cache slots [0, context] inclusive ---

/// Baseline: one thread owns an output row and walks head_dim scalar-wise. This
/// is the shape a direct transliteration of the CPU reference produces.
__global__ void k_decode_cache_attend_scalar(
    const float *__restrict__ q, const float *__restrict__ cosine,
    const float *__restrict__ sine, const int *__restrict__ positions,
    const int *__restrict__ context_lengths, const float *__restrict__ q_weight,
    const float *__restrict__ key_cache, const float *__restrict__ value_cache,
    float *__restrict__ out, int batch, int query_heads, int kv_heads, int cache_length,
    int head_dim, float eps, int do_q_norm, float gemma_offset, float multiplier) {
    const long long row = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= (long long)batch * query_heads) return;
    const int item = row / query_heads;
    const int head = row % query_heads;
    const int kv_head = head / (query_heads / kv_heads);
    const int half = head_dim / 2;
    extern __shared__ float scratch[];
    float *rotated = scratch + threadIdx.x * 0;  // unused; kept for symmetry

    const long long q_base = row * head_dim;
    float inverse = 1.0f;
    if (do_q_norm) {
        float squares = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            const float v = q[q_base + d];
            squares = fmaf(v, v, squares);
        }
        inverse = 1.0f / sqrtf(squares / (float)head_dim + eps);
    }
    const long long table = (long long)positions[item] * half;

    float running_max = -INFINITY, denominator = 0.0f;
    // Two passes over the cache so the rotated query need not be materialized:
    // recompute the rotated element inside the dot product instead.
    const long long kv_row = ((long long)item * kv_heads + kv_head) * cache_length;
    for (int token = 0; token <= context_lengths[item]; ++token) {
        const long long cache_offset = (kv_row + token) * head_dim;
        float score = 0.0f;
        for (int d = 0; d < half; ++d) {
            const float w0 = do_q_norm ? q_weight[d] + gemma_offset : 1.0f;
            const float w1 = do_q_norm ? q_weight[half + d] + gemma_offset : 1.0f;
            const float first = q[q_base + d] * inverse * w0;
            const float second = q[q_base + half + d] * inverse * w1;
            const float c = cosine[table + d], s = sine[table + d];
            score += (first * c - second * s) * key_cache[cache_offset + d];
            score += (second * c + first * s) * key_cache[cache_offset + half + d];
        }
        score *= multiplier;
        const float next_max = fmaxf(running_max, score);
        const float old_weight =
            (running_max == -INFINITY) ? 0.0f : __expf(running_max - next_max);
        const float new_weight = __expf(score - next_max);
        denominator = denominator * old_weight + new_weight;
        for (int d = 0; d < head_dim; ++d)
            out[q_base + d] = (token == 0 ? 0.0f : out[q_base + d]) * old_weight +
                              value_cache[cache_offset + d] * new_weight;
        running_max = next_max;
    }
    const float inv = denominator > 0.0f ? 1.0f / denominator : 0.0f;
    for (int d = 0; d < head_dim; ++d) out[q_base + d] *= inv;
    (void)rotated;
}

/// Candidate: one lane group per output row, head_dim split across lanes, the
/// rotated query held in registers across the whole cache walk.
template <int LANES, int MAX_ITEMS>
__global__ void k_decode_cache_attend(
    const float *__restrict__ q, const float *__restrict__ cosine,
    const float *__restrict__ sine, const int *__restrict__ positions,
    const int *__restrict__ context_lengths, const float *__restrict__ q_weight,
    const float *__restrict__ key_cache, const float *__restrict__ value_cache,
    float *__restrict__ out, int batch, int query_heads, int kv_heads, int cache_length,
    int head_dim, float eps, int do_q_norm, float gemma_offset, float multiplier) {
    const long long row = (long long)blockIdx.x * (kBlock / LANES) + threadIdx.x / LANES;
    if (row >= (long long)batch * query_heads) return;
    const int lane = threadIdx.x % LANES;
    const int item = row / query_heads;
    const int head = row % query_heads;
    const int kv_head = head / (query_heads / kv_heads);
    const int half = head_dim / 2;
    const long long q_base = row * head_dim;

    float inverse = 1.0f;
    if (do_q_norm) {
        float squares = 0.0f;
        for (int d = lane; d < head_dim; d += LANES) {
            const float v = q[q_base + d];
            squares = fmaf(v, v, squares);
        }
        squares = row_reduce_sum<LANES>(squares);
        inverse = 1.0f / sqrtf(squares / (float)head_dim + eps);
    }

    // Rotate once into registers: MAX_ITEMS covers head_dim / LANES elements.
    const long long table = (long long)positions[item] * half;
    float rotated[MAX_ITEMS];
#pragma unroll
    for (int t = 0; t < MAX_ITEMS; ++t) rotated[t] = 0.0f;
    for (int d = lane, t = 0; d < head_dim; d += LANES, ++t) {
        const bool low = d < half;
        const int pair = low ? d : d - half;
        const float w_low = do_q_norm ? q_weight[pair] + gemma_offset : 1.0f;
        const float w_high = do_q_norm ? q_weight[half + pair] + gemma_offset : 1.0f;
        const float first = q[q_base + pair] * inverse * w_low;
        const float second = q[q_base + half + pair] * inverse * w_high;
        const float c = cosine[table + pair], s = sine[table + pair];
        rotated[t] = low ? (first * c - second * s) : (second * c + first * s);
    }

    float acc[MAX_ITEMS];
#pragma unroll
    for (int t = 0; t < MAX_ITEMS; ++t) acc[t] = 0.0f;
    float running_max = -INFINITY, denominator = 0.0f;
    const long long kv_row = ((long long)item * kv_heads + kv_head) * cache_length;

    for (int token = 0; token <= context_lengths[item]; ++token) {
        const long long cache_offset = (kv_row + token) * head_dim;
        float dot = 0.0f;
        for (int d = lane, t = 0; d < head_dim; d += LANES, ++t)
            dot = fmaf(rotated[t], key_cache[cache_offset + d], dot);
        dot = row_reduce_sum<LANES>(dot) * multiplier;

        const float next_max = fmaxf(running_max, dot);
        const float old_weight =
            (running_max == -INFINITY) ? 0.0f : __expf(running_max - next_max);
        const float new_weight = __expf(dot - next_max);
        denominator = denominator * old_weight + new_weight;
        for (int d = lane, t = 0; d < head_dim; d += LANES, ++t)
            acc[t] = acc[t] * old_weight + value_cache[cache_offset + d] * new_weight;
        running_max = next_max;
    }

    const float inv = denominator > 0.0f ? 1.0f / denominator : 0.0f;
    for (int d = lane, t = 0; d < head_dim; d += LANES, ++t)
        out[q_base + d] = acc[t] * inv;
}

// ===========================================================================
// 3. cascade_attention_multi
// ===========================================================================

/// Baseline: one thread per output row, scalar head_dim walk.
__global__ void k_cascade_scalar(const float *__restrict__ q,
                                 const float *const *__restrict__ prefix_k,
                                 const float *const *__restrict__ prefix_v,
                                 const long long *__restrict__ prefix_lengths, int levels,
                                 const float *__restrict__ key_cache,
                                 const float *__restrict__ value_cache,
                                 const int *__restrict__ block_table,
                                 const int *__restrict__ context_lens,
                                 float *__restrict__ out, int batch, int query_heads,
                                 int kv_heads, int head_dim, int page_size, int max_blocks,
                                 float multiplier) {
    const long long row = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= (long long)batch * query_heads) return;
    const int request = row / query_heads;
    const int head = row % query_heads;
    const int kv_head = head / (query_heads / kv_heads);
    const long long q_base = row * head_dim;

    float running_max = -INFINITY, denominator = 0.0f;
    for (int d = 0; d < head_dim; ++d) out[q_base + d] = 0.0f;

    auto consume = [&](const float *key, const float *value) {
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) score += q[q_base + d] * key[d];
        score *= multiplier;
        const float next_max = fmaxf(running_max, score);
        const float old_weight =
            (running_max == -INFINITY) ? 0.0f : __expf(running_max - next_max);
        const float new_weight = __expf(score - next_max);
        denominator = denominator * old_weight + new_weight;
        for (int d = 0; d < head_dim; ++d)
            out[q_base + d] = out[q_base + d] * old_weight + value[d] * new_weight;
        running_max = next_max;
    };

    for (int level = 0; level < levels; ++level)
        for (long long position = 0; position < prefix_lengths[level]; ++position) {
            const long long offset = (position * kv_heads + kv_head) * head_dim;
            consume(prefix_k[level] + offset, prefix_v[level] + offset);
        }
    const int context = context_lens[request];
    for (int position = 0; position < context; ++position) {
        const int block = block_table[(long long)request * max_blocks + position / page_size];
        const long long offset =
            (((long long)block * page_size + position % page_size) * kv_heads + kv_head) *
            head_dim;
        consume(key_cache + offset, value_cache + offset);
    }
    const float inv = denominator > 0.0f ? 1.0f / denominator : 0.0f;
    for (int d = 0; d < head_dim; ++d) out[q_base + d] *= inv;
}

/// Candidate: lane group per row, query in registers, one online softmax across
/// the prefix levels and the paged tail in contract order.
template <int LANES, int MAX_ITEMS>
__global__ void k_cascade(const float *__restrict__ q,
                          const float *const *__restrict__ prefix_k,
                          const float *const *__restrict__ prefix_v,
                          const long long *__restrict__ prefix_lengths, int levels,
                          const float *__restrict__ key_cache,
                          const float *__restrict__ value_cache,
                          const int *__restrict__ block_table,
                          const int *__restrict__ context_lens, float *__restrict__ out,
                          int batch, int query_heads, int kv_heads, int head_dim,
                          int page_size, int max_blocks, float multiplier) {
    const long long row = (long long)blockIdx.x * (kBlock / LANES) + threadIdx.x / LANES;
    if (row >= (long long)batch * query_heads) return;
    const int lane = threadIdx.x % LANES;
    const int request = row / query_heads;
    const int head = row % query_heads;
    const int kv_head = head / (query_heads / kv_heads);
    const long long q_base = row * head_dim;

    float q_val[MAX_ITEMS], acc[MAX_ITEMS];
#pragma unroll
    for (int t = 0; t < MAX_ITEMS; ++t) { q_val[t] = 0.0f; acc[t] = 0.0f; }
    for (int d = lane, t = 0; d < head_dim; d += LANES, ++t) q_val[t] = q[q_base + d];

    float running_max = -INFINITY, denominator = 0.0f;
    auto consume = [&](const float *key, const float *value) {
        float dot = 0.0f;
        for (int d = lane, t = 0; d < head_dim; d += LANES, ++t)
            dot = fmaf(q_val[t], key[d], dot);
        dot = row_reduce_sum<LANES>(dot) * multiplier;
        const float next_max = fmaxf(running_max, dot);
        const float old_weight =
            (running_max == -INFINITY) ? 0.0f : __expf(running_max - next_max);
        const float new_weight = __expf(dot - next_max);
        denominator = denominator * old_weight + new_weight;
        for (int d = lane, t = 0; d < head_dim; d += LANES, ++t)
            acc[t] = acc[t] * old_weight + value[d] * new_weight;
        running_max = next_max;
    };

    // Contract order: every prefix level in sequence, then the paged tail.
    for (int level = 0; level < levels; ++level)
        for (long long position = 0; position < prefix_lengths[level]; ++position) {
            const long long offset = (position * kv_heads + kv_head) * head_dim;
            consume(prefix_k[level] + offset, prefix_v[level] + offset);
        }
    const int context = context_lens[request];
    for (int position = 0; position < context; ++position) {
        const int block =
            block_table[(long long)request * max_blocks + position / page_size];
        const long long offset =
            (((long long)block * page_size + position % page_size) * kv_heads + kv_head) *
            head_dim;
        consume(key_cache + offset, value_cache + offset);
    }

    const float inv = denominator > 0.0f ? 1.0f / denominator : 0.0f;
    for (int d = lane, t = 0; d < head_dim; d += LANES, ++t) out[q_base + d] = acc[t] * inv;
}

// ===========================================================================
// fp64 oracles
// ===========================================================================

static std::vector<double> oracle_swin(const std::vector<float> &qkv,
                                       const std::vector<float> &bias,
                                       const std::vector<float> &mask, bool has_mask,
                                       int windows, int tokens, int heads,
                                       int windows_per_image) {
    const int D = kSwinHeadDim;
    std::vector<double> out((size_t)windows * tokens * heads * D, 0.0);
    const double scale = 0.1767766952966369;
    for (int w = 0; w < windows; ++w) {
        const int mask_window = windows_per_image > 0 ? w % windows_per_image : 0;
        for (int h = 0; h < heads; ++h)
            for (int qp = 0; qp < tokens; ++qp) {
                auto at = [&](int pos, int which) {
                    return (size_t)((((size_t)w * tokens + pos) * 3 + which) * heads + h) * D;
                };
                std::vector<double> acc(D, 0.0);
                double maximum = -std::numeric_limits<double>::infinity(), denominator = 0.0;
                for (int kp = 0; kp < tokens; ++kp) {
                    double score = 0.0;
                    for (int d = 0; d < D; ++d)
                        score += (double)qkv[at(qp, 0) + d] * qkv[at(kp, 1) + d];
                    score = score * scale +
                            bias[((size_t)h * tokens + qp) * tokens + kp];
                    if (has_mask)
                        score += mask[((size_t)mask_window * tokens + qp) * tokens + kp];
                    const double next_max = std::max(maximum, score);
                    const double ow = std::isinf(maximum) ? 0.0 : std::exp(maximum - next_max);
                    const double nw = std::exp(score - next_max);
                    denominator = denominator * ow + nw;
                    for (int d = 0; d < D; ++d)
                        acc[d] = acc[d] * ow + (double)qkv[at(kp, 2) + d] * nw;
                    maximum = next_max;
                }
                const size_t o = (size_t)(((size_t)w * tokens + qp) * heads + h) * D;
                for (int d = 0; d < D; ++d) out[o + d] = acc[d] / denominator;
            }
    }
    return out;
}

struct DecodeOracle {
    std::vector<double> out, key_cache, value_cache;
};

static DecodeOracle oracle_decode_cache(
    const std::vector<float> &q, const std::vector<float> &new_k,
    const std::vector<float> &new_v, const std::vector<float> &cosine,
    const std::vector<float> &sine, const std::vector<int32_t> &positions,
    const std::vector<int32_t> &context_lengths, const std::vector<float> &qw,
    const std::vector<float> &kw, std::vector<float> key_cache,
    std::vector<float> value_cache, int batch, int query_heads, int kv_heads,
    int cache_length, int head_dim, float eps, bool do_q_norm, bool do_k_norm,
    float gemma_offset, float scale) {
    const int half = head_dim / 2;
    // Stage 1: append.
    for (int item = 0; item < batch; ++item) {
        const size_t table = (size_t)positions[item] * half;
        for (int h = 0; h < kv_heads; ++h) {
            const size_t src = ((size_t)item * kv_heads + h) * head_dim;
            double inverse = 1.0;
            if (do_k_norm) {
                double squares = 0.0;
                for (int d = 0; d < head_dim; ++d)
                    squares += (double)new_k[src + d] * new_k[src + d];
                inverse = 1.0 / std::sqrt(squares / head_dim + (double)eps);
            }
            const size_t dst =
                (((size_t)item * kv_heads + h) * cache_length + context_lengths[item]) *
                head_dim;
            for (int d = 0; d < half; ++d) {
                const double w0 = do_k_norm ? kw[d] + gemma_offset : 1.0;
                const double w1 = do_k_norm ? kw[half + d] + gemma_offset : 1.0;
                const double first = new_k[src + d] * inverse * w0;
                const double second = new_k[src + half + d] * inverse * w1;
                key_cache[dst + d] =
                    (float)(first * cosine[table + d] - second * sine[table + d]);
                key_cache[dst + half + d] =
                    (float)(second * cosine[table + d] + first * sine[table + d]);
            }
            for (int d = 0; d < head_dim; ++d) value_cache[dst + d] = new_v[src + d];
        }
    }
    // Stage 2: attend over [0, context] inclusive.
    const double multiplier =
        scale == 0.0f ? 1.0 / std::sqrt((double)head_dim) : (double)scale;
    const int group = query_heads / kv_heads;
    std::vector<double> out((size_t)batch * query_heads * head_dim, 0.0);
    for (int item = 0; item < batch; ++item)
        for (int h = 0; h < query_heads; ++h) {
            const int kv_head = h / group;
            const size_t q_base = ((size_t)item * query_heads + h) * head_dim;
            double inverse = 1.0;
            if (do_q_norm) {
                double squares = 0.0;
                for (int d = 0; d < head_dim; ++d)
                    squares += (double)q[q_base + d] * q[q_base + d];
                inverse = 1.0 / std::sqrt(squares / head_dim + (double)eps);
            }
            const size_t table = (size_t)positions[item] * half;
            std::vector<double> rotated(head_dim);
            for (int d = 0; d < half; ++d) {
                const double w0 = do_q_norm ? qw[d] + gemma_offset : 1.0;
                const double w1 = do_q_norm ? qw[half + d] + gemma_offset : 1.0;
                const double first = q[q_base + d] * inverse * w0;
                const double second = q[q_base + half + d] * inverse * w1;
                rotated[d] = first * cosine[table + d] - second * sine[table + d];
                rotated[half + d] = second * cosine[table + d] + first * sine[table + d];
            }
            std::vector<double> acc(head_dim, 0.0);
            double maximum = -std::numeric_limits<double>::infinity(), denominator = 0.0;
            for (int token = 0; token <= context_lengths[item]; ++token) {
                const size_t co =
                    (((size_t)item * kv_heads + kv_head) * cache_length + token) * head_dim;
                double score = 0.0;
                for (int d = 0; d < head_dim; ++d) score += rotated[d] * key_cache[co + d];
                score *= multiplier;
                const double next_max = std::max(maximum, score);
                const double ow = std::isinf(maximum) ? 0.0 : std::exp(maximum - next_max);
                const double nw = std::exp(score - next_max);
                denominator = denominator * ow + nw;
                for (int d = 0; d < head_dim; ++d)
                    acc[d] = acc[d] * ow + (double)value_cache[co + d] * nw;
                maximum = next_max;
            }
            const double inv = denominator > 0.0 ? 1.0 / denominator : 0.0;
            for (int d = 0; d < head_dim; ++d) out[q_base + d] = acc[d] * inv;
        }
    DecodeOracle result;
    result.out = std::move(out);
    result.key_cache.assign(key_cache.begin(), key_cache.end());
    result.value_cache.assign(value_cache.begin(), value_cache.end());
    return result;
}

static std::vector<double> oracle_cascade(
    const std::vector<float> &q, const std::vector<std::vector<float>> &prefix_k,
    const std::vector<std::vector<float>> &prefix_v,
    const std::vector<long long> &prefix_lengths, const std::vector<float> &key_cache,
    const std::vector<float> &value_cache, const std::vector<int32_t> &block_table,
    const std::vector<int32_t> &context_lens, int batch, int query_heads, int kv_heads,
    int head_dim, int page_size, int max_blocks, float scale) {
    const double multiplier =
        scale == 0.0f ? 1.0 / std::sqrt((double)head_dim) : (double)scale;
    const int group = query_heads / kv_heads;
    std::vector<double> out((size_t)batch * query_heads * head_dim, 0.0);
    for (int request = 0; request < batch; ++request)
        for (int h = 0; h < query_heads; ++h) {
            const int kv_head = h / group;
            const size_t q_base = ((size_t)request * query_heads + h) * head_dim;
            std::vector<double> acc(head_dim, 0.0);
            double maximum = -std::numeric_limits<double>::infinity(), denominator = 0.0;
            auto consume = [&](const float *key, const float *value) {
                double score = 0.0;
                for (int d = 0; d < head_dim; ++d) score += (double)q[q_base + d] * key[d];
                score *= multiplier;
                const double next_max = std::max(maximum, score);
                const double ow = std::isinf(maximum) ? 0.0 : std::exp(maximum - next_max);
                const double nw = std::exp(score - next_max);
                denominator = denominator * ow + nw;
                for (int d = 0; d < head_dim; ++d) acc[d] = acc[d] * ow + (double)value[d] * nw;
                maximum = next_max;
            };
            for (size_t level = 0; level < prefix_lengths.size(); ++level)
                for (long long position = 0; position < prefix_lengths[level]; ++position) {
                    const size_t offset = ((size_t)position * kv_heads + kv_head) * head_dim;
                    consume(prefix_k[level].data() + offset, prefix_v[level].data() + offset);
                }
            for (int position = 0; position < context_lens[request]; ++position) {
                const int block =
                    block_table[(size_t)request * max_blocks + position / page_size];
                const size_t offset =
                    (((size_t)block * page_size + position % page_size) * kv_heads +
                     kv_head) *
                    head_dim;
                consume(key_cache.data() + offset, value_cache.data() + offset);
            }
            const double inv = denominator > 0.0 ? 1.0 / denominator : 0.0;
            for (int d = 0; d < head_dim; ++d) out[q_base + d] = acc[d] * inv;
        }
    return out;
}

// ===========================================================================
// Harness
// ===========================================================================

static int grid_rows(long long rows, int lanes) {
    const int per_block = kBlock / lanes;
    return (int)((rows + per_block - 1) / per_block);
}

// --- swin --------------------------------------------------------------------
static bool run_swin(int windows, int tokens, int heads, int windows_per_image,
                     bool has_mask, const char *name, bool bench, qc::Bench *b0,
                     qc::Bench *b1) {
    qc::Rng rng(22001 + windows + tokens * 7);
    const int D = kSwinHeadDim;
    const size_t qkv_count = (size_t)windows * tokens * 3 * heads * D;
    const size_t out_count = (size_t)windows * tokens * heads * D;
    auto qkv = rng.normals(qkv_count, 0.5f);
    auto bias = rng.normals((size_t)heads * tokens * tokens, 0.3f);
    std::vector<float> mask;
    if (has_mask) {
        mask.assign((size_t)windows_per_image * tokens * tokens, 0.0f);
        // Swin's shifted-window mask is 0 or a large negative constant.
        for (size_t i = 0; i < mask.size(); ++i)
            if ((i / 3) % 5 == 0) mask[i] = -100.0f;
    }
    auto ref = oracle_swin(qkv, bias, mask, has_mask, windows, tokens, heads,
                           windows_per_image);

    auto dqkv = qc::dnew(qkv);
    auto dbias = qc::dnew(bias);
    float *dmask = has_mask ? qc::dnew(mask) : nullptr;
    auto dout = qc::dzero<float>(out_count);

    const long long rows = (long long)windows * heads * tokens;
    auto launch_global = [&] {
        k_swin_global<kLanes><<<grid_rows(rows, kLanes), kBlock>>>(
            dqkv, dbias, dmask, dout, windows, tokens, heads, windows_per_image);
    };
    auto launch_lds = [&] {
        k_swin_lds<64><<<windows * heads, kBlock>>>(dqkv, dbias, dmask, dout, windows,
                                                    tokens, heads, windows_per_image);
    };

    char label[160];
    bool ok = true;
    for (int which = 0; which < 2; ++which) {
        QC_CHECK(hipMemset(dout, 0, out_count * sizeof(float)));
        which == 0 ? launch_global() : launch_lds();
        QC_SYNC();
        std::snprintf(label, sizeof(label), "swin_attention_d32 %s %s", name,
                      which == 0 ? "global" : "lds");
        ok &= qc::compare(qc::d2h(dout, out_count), ref, qc::Tol::fp32()).report(label);
    }
    if (bench && ok) {
        *b0 = qc::bench(launch_global, 15, 50);
        *b1 = qc::bench(launch_lds, 15, 50);
    }
    qc::dfree(dqkv, dbias, dout);
    if (dmask) qc::dfree(dmask);
    return ok;
}

// --- decode_cache_attention --------------------------------------------------
template <int MAX_ITEMS>
static bool run_decode(int batch, int query_heads, int kv_heads, int cache_length,
                       int head_dim, bool do_q_norm, bool do_k_norm, bool gemma,
                       float scale, const char *name, bool bench, qc::Bench *b0,
                       qc::Bench *b1) {
    qc::Rng rng(23001 + batch + head_dim);
    const int half = head_dim / 2, max_position = 4096;
    const float eps = 1e-6f;
    std::vector<float> cosine((size_t)max_position * half), sine((size_t)max_position * half);
    for (int p = 0; p < max_position; ++p)
        for (int i = 0; i < half; ++i) {
            const double angle = (double)p * std::pow(10000.0, -2.0 * i / (double)head_dim);
            cosine[(size_t)p * half + i] = (float)std::cos(angle);
            sine[(size_t)p * half + i] = (float)std::sin(angle);
        }
    auto q = rng.normals((size_t)batch * query_heads * head_dim, 0.6f);
    auto new_k = rng.normals((size_t)batch * kv_heads * head_dim, 0.6f);
    auto new_v = rng.normals((size_t)batch * kv_heads * head_dim, 0.6f);
    auto qw = rng.normals(head_dim, 0.1f);
    auto kw = rng.normals(head_dim, 0.1f);
    for (auto &v : qw) v += 1.0f;
    for (auto &v : kw) v += 1.0f;
    auto positions = rng.integers(batch, 0, max_position - 1);
    std::vector<int32_t> context_lengths(batch);
    for (int i = 0; i < batch; ++i) context_lengths[i] = (i * 37 + 5) % (cache_length - 1);
    const size_t cache_count = (size_t)batch * kv_heads * cache_length * head_dim;
    auto key_cache = rng.normals(cache_count, 0.4f);
    auto value_cache = rng.normals(cache_count, 0.4f);

    auto ref = oracle_decode_cache(q, new_k, new_v, cosine, sine, positions,
                                   context_lengths, qw, kw, key_cache, value_cache, batch,
                                   query_heads, kv_heads, cache_length, head_dim, eps,
                                   do_q_norm, do_k_norm, gemma ? 1.0f : 0.0f, scale);
    const float multiplier =
        scale == 0.0f ? 1.0f / std::sqrt((float)head_dim) : scale;

    auto dq = qc::dnew(q);
    auto dnk = qc::dnew(new_k);
    auto dnv = qc::dnew(new_v);
    auto dc = qc::dnew(cosine);
    auto ds = qc::dnew(sine);
    auto dqw = qc::dnew(qw);
    auto dkw = qc::dnew(kw);
    auto dpos = qc::dnew(positions);
    auto dctx = qc::dnew(context_lengths);
    auto dkc = qc::dnew(key_cache);
    auto dvc = qc::dnew(value_cache);
    auto dout = qc::dzero<float>((size_t)batch * query_heads * head_dim);

    const long long kv_rows = (long long)batch * kv_heads;
    const long long q_rows = (long long)batch * query_heads;
    auto append = [&] {
        k_decode_cache_append<kLanes><<<grid_rows(kv_rows, kLanes), kBlock>>>(
            dnk, dnv, dc, ds, dpos, dctx, dkw, dkc, dvc, batch, kv_heads, cache_length,
            head_dim, eps, do_k_norm, gemma ? 1.0f : 0.0f);
    };
    auto attend_fast = [&] {
        k_decode_cache_attend<kLanes, MAX_ITEMS><<<grid_rows(q_rows, kLanes), kBlock>>>(
            dq, dc, ds, dpos, dctx, dqw, dkc, dvc, dout, batch, query_heads, kv_heads,
            cache_length, head_dim, eps, do_q_norm, gemma ? 1.0f : 0.0f, multiplier);
    };
    auto attend_scalar = [&] {
        k_decode_cache_attend_scalar<<<(int)((q_rows + 255) / 256), 256>>>(
            dq, dc, ds, dpos, dctx, dqw, dkc, dvc, dout, batch, query_heads, kv_heads,
            cache_length, head_dim, eps, do_q_norm, gemma ? 1.0f : 0.0f, multiplier);
    };

    char label[160];
    bool ok = true;
    // The append mutates the cache, so restore it before each independent run.
    for (int which = 0; which < 2; ++which) {
        QC_CHECK(hipMemcpy(dkc, key_cache.data(), cache_count * sizeof(float),
                           hipMemcpyHostToDevice));
        QC_CHECK(hipMemcpy(dvc, value_cache.data(), cache_count * sizeof(float),
                           hipMemcpyHostToDevice));
        QC_CHECK(hipMemset(dout, 0, (size_t)batch * query_heads * head_dim * sizeof(float)));
        append();
        QC_SYNC();
        which == 0 ? attend_fast() : attend_scalar();
        QC_SYNC();
        std::snprintf(label, sizeof(label), "decode_cache_attention %s %s", name,
                      which == 0 ? "wavefront" : "scalar");
        ok &= qc::compare(qc::d2h(dout, (size_t)batch * query_heads * head_dim), ref.out,
                          qc::Tol::fp32().with_elementwise(1e-5, 1e-5))
                  .report(label);
        if (which == 0) {
            std::snprintf(label, sizeof(label), "decode_cache_attention %s key_cache", name);
            ok &= qc::compare(qc::d2h(dkc, cache_count), ref.key_cache, qc::Tol::fp32())
                      .report(label);
        }
    }
    if (bench && ok) {
        *b0 = qc::bench([&] { append(); attend_scalar(); }, 10, 30);
        *b1 = qc::bench([&] { append(); attend_fast(); }, 10, 30);
    }
    qc::dfree(dq, dnk, dnv, dc, ds, dqw, dkw, dkc, dvc, dout);
    qc::dfree(dpos, dctx);
    return ok;
}

// --- cascade -----------------------------------------------------------------
template <int MAX_ITEMS>
static bool run_cascade(int batch, int query_heads, int kv_heads, int head_dim,
                        int page_size, int max_blocks, int levels, int prefix_len,
                        const char *name, bool bench, qc::Bench *b0, qc::Bench *b1) {
    qc::Rng rng(24001 + batch + head_dim + levels);
    const int cache_blocks = batch * max_blocks;
    auto q = rng.normals((size_t)batch * query_heads * head_dim, 0.6f);
    const size_t cache_count = (size_t)cache_blocks * page_size * kv_heads * head_dim;
    auto key_cache = rng.normals(cache_count, 0.5f);
    auto value_cache = rng.normals(cache_count, 0.5f);
    std::vector<int32_t> block_table((size_t)batch * max_blocks);
    for (size_t i = 0; i < block_table.size(); ++i) block_table[i] = (int)(i % cache_blocks);
    std::vector<int32_t> context_lens(batch);
    for (int i = 0; i < batch; ++i)
        context_lens[i] = (i * 29 + 7) % (max_blocks * page_size);

    std::vector<std::vector<float>> prefix_k(levels), prefix_v(levels);
    std::vector<long long> prefix_lengths(levels);
    for (int l = 0; l < levels; ++l) {
        prefix_lengths[l] = prefix_len + l * 13;
        prefix_k[l] = rng.normals((size_t)prefix_lengths[l] * kv_heads * head_dim, 0.5f);
        prefix_v[l] = rng.normals((size_t)prefix_lengths[l] * kv_heads * head_dim, 0.5f);
    }
    auto ref = oracle_cascade(q, prefix_k, prefix_v, prefix_lengths, key_cache, value_cache,
                              block_table, context_lens, batch, query_heads, kv_heads,
                              head_dim, page_size, max_blocks, 0.0f);
    const float multiplier = 1.0f / std::sqrt((float)head_dim);

    auto dq = qc::dnew(q);
    auto dkc = qc::dnew(key_cache);
    auto dvc = qc::dnew(value_cache);
    auto dbt = qc::dnew(block_table);
    auto dctx = qc::dnew(context_lens);
    auto dout = qc::dzero<float>((size_t)batch * query_heads * head_dim);
    // Prefix levels are separate allocations reached through a device pointer array.
    std::vector<float *> hk(levels), hv(levels);
    for (int l = 0; l < levels; ++l) {
        hk[l] = qc::dnew(prefix_k[l]);
        hv[l] = qc::dnew(prefix_v[l]);
    }
    float **dpk = nullptr, **dpv = nullptr;
    long long *dpl = nullptr;
    if (levels > 0) {
        QC_CHECK(hipMalloc(&dpk, levels * sizeof(float *)));
        QC_CHECK(hipMalloc(&dpv, levels * sizeof(float *)));
        QC_CHECK(hipMemcpy(dpk, hk.data(), levels * sizeof(float *), hipMemcpyHostToDevice));
        QC_CHECK(hipMemcpy(dpv, hv.data(), levels * sizeof(float *), hipMemcpyHostToDevice));
        dpl = qc::dnew(prefix_lengths);
    }

    const long long rows = (long long)batch * query_heads;
    auto fast = [&] {
        k_cascade<kLanes, MAX_ITEMS><<<grid_rows(rows, kLanes), kBlock>>>(
            dq, (const float *const *)dpk, (const float *const *)dpv, dpl, levels, dkc,
            dvc, dbt, dctx, dout, batch, query_heads, kv_heads, head_dim, page_size,
            max_blocks, multiplier);
    };
    auto scalar = [&] {
        k_cascade_scalar<<<(int)((rows + 255) / 256), 256>>>(
            dq, (const float *const *)dpk, (const float *const *)dpv, dpl, levels, dkc,
            dvc, dbt, dctx, dout, batch, query_heads, kv_heads, head_dim, page_size,
            max_blocks, multiplier);
    };

    char label[160];
    bool ok = true;
    for (int which = 0; which < 2; ++which) {
        QC_CHECK(hipMemset(dout, 0, (size_t)batch * query_heads * head_dim * sizeof(float)));
        which == 0 ? fast() : scalar();
        QC_SYNC();
        std::snprintf(label, sizeof(label), "cascade_attention_multi %s %s", name,
                      which == 0 ? "wavefront" : "scalar");
        ok &= qc::compare(qc::d2h(dout, (size_t)batch * query_heads * head_dim), ref,
                          qc::Tol::fp32().with_elementwise(1e-5, 1e-5))
                  .report(label);
    }
    if (bench && ok) {
        *b0 = qc::bench(scalar, 10, 30);
        *b1 = qc::bench(fast, 10, 30);
    }
    qc::dfree(dq, dkc, dvc, dout);
    qc::dfree(dbt, dctx);
    for (int l = 0; l < levels; ++l) qc::dfree(hk[l], hv[l]);
    if (dpk) { QC_CHECK(hipFree(dpk)); QC_CHECK(hipFree(dpv)); qc::dfree(dpl); }
    return ok;
}

int main(int argc, char **argv) {
    const bool do_bench = qc::bench_requested(argc, argv);
    qc::print_environment("attn_composites (CDNA3 gfx942)");
    bool ok = true;
    qc::Bench b0, b1;

    std::printf("correctness (fp64 oracles = QuixiCore-CPU composite references):\n");

    // swin_attention_d32 -- with and without the shifted-window mask, and a
    // multi-image case where mask_window = window %% windows_per_image matters.
    ok &= run_swin(4, 49, 4, 0, false, "W4 T49 H4 nomask", false, nullptr, nullptr);
    ok &= run_swin(8, 49, 4, 4, true, "W8 T49 H4 mask 2 images", false, nullptr, nullptr);
    ok &= run_swin(4, 64, 8, 2, true, "W4 T64 H8 mask", false, nullptr, nullptr);
    ok &= run_swin(3, 16, 2, 0, false, "W3 T16 H2 tiny", false, nullptr, nullptr);

    // decode_cache_attention -- norm on/off, GQA, Gemma weights, explicit scale.
    ok &= run_decode<4>(8, 8, 2, 256, 128, true, true, false, 0.0f, "B8 Hq8 Hk2 D128 norm", false, nullptr, nullptr);
    ok &= run_decode<4>(8, 8, 2, 256, 128, false, false, false, 0.0f, "B8 no norm", false, nullptr, nullptr);
    ok &= run_decode<4>(4, 8, 8, 128, 128, true, true, true, 0.0f, "B4 MHA gemma", false, nullptr, nullptr);
    ok &= run_decode<2>(5, 4, 1, 97, 64, true, false, false, 0.0625f, "B5 ragged explicit scale", false, nullptr, nullptr);

    // cascade_attention_multi -- zero levels (paged only), one level, three levels.
    ok &= run_cascade<4>(4, 8, 2, 128, 16, 8, 0, 0, "B4 levels=0 paged only", false, nullptr, nullptr);
    ok &= run_cascade<4>(4, 8, 2, 128, 16, 8, 1, 32, "B4 levels=1", false, nullptr, nullptr);
    ok &= run_cascade<4>(3, 4, 1, 128, 16, 4, 3, 17, "B3 levels=3 ragged", false, nullptr, nullptr);
    ok &= run_cascade<2>(4, 8, 2, 64, 32, 8, 2, 40, "B4 D64 page32 levels=2", false, nullptr, nullptr);

    if (do_bench && ok) {
        std::printf("\nA/B (HIP-event median):\n");
        ok &= run_swin(1024, 49, 8, 64, true, "W1024 T49 H8", true, &b0, &b1);
        b0.report("swin_attention_d32 global K/V");
        b1.report("swin_attention_d32 LDS K/V");
        qc::report_ab("swin_attention_d32 W1024 T49 H8", b0, b1);

        ok &= run_decode<4>(256, 32, 8, 4096, 128, true, true, false, 0.0f, "B256 ctx4096", true, &b0, &b1);
        b0.report("decode_cache_attention scalar");
        b1.report("decode_cache_attention wavefront");
        qc::report_ab("decode_cache_attention B256 Hq32 D128", b0, b1);

        ok &= run_cascade<4>(128, 32, 8, 128, 16, 64, 2, 128, "B128 levels=2", true, &b0, &b1);
        b0.report("cascade_attention_multi scalar");
        b1.report("cascade_attention_multi wavefront");
        qc::report_ab("cascade_attention_multi B128 Hq32 D128", b0, b1);
    }

    return qc::finish(ok);
}
