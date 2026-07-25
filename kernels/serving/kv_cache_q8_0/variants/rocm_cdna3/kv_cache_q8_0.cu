/**
 * @file
 * @brief CDNA3 (gfx942) Q8_0 paged KV-cache codec: scatter, gather, block copy,
 *        and decode attention directly against the packed cache.
 *
 * Semantic source is ../QuixiCore-CPU/kernels/attention/attention_q8_kv.cpp.
 * Operations: kv_cache_scatter_q8_0 (:102), kv_cache_gather_q8_0 (:235),
 * kv_cache_copy_blocks_q8_0 (:351), paged_attention_q8_0 (:417).
 *
 * ## Canonical layout -- two separate planes, not interleaved blocks
 *
 * Q8_0 here is NOT the GGUF interleaved-block layout. Codes and scales live in
 * two independent arrays:
 *
 *   codes  int8   [cache_slots, heads, head_dim]      slot = block*page_size + offset
 *   scales uint16 [cache_slots, heads, head_dim/32]   raw fp16 BITS, group of 32
 *
 * A port that packs a 2-byte scale ahead of each 32-byte code run -- the actual
 * GGUF Q8_0 block -- produces plausible-looking numbers and a cache no other
 * backend can read. head_dim must be a multiple of 32.
 *
 * ## Quantization rule
 *
 *   scale = amax(group) / 127                     stored as fp16 bits
 *   code  = clamp(copysign(floor(|v/scale| + 0.5), v), -127, 127)
 *
 * `floor(|x| + 0.5)` with the sign reapplied is round-half-away-from-zero, which
 * differs from rintf()'s round-half-to-even on exact .5 ties. Matched exactly.
 * A group whose amax is 0 stores scale 0 and codes 0 (the inverse is forced to
 * 0 rather than dividing by zero). Non-finite input is rejected.
 *
 * ## Scatter rewrites the whole cache
 *
 * The reference zero-fills every code and scale in the cache before writing the
 * requested slots. It is a full rewrite, not an incremental append, and slots
 * carrying -1 are skipped (leaving them zeroed). Reproduced here as an explicit
 * zero-fill kernel; skipping it would leave stale data in untouched slots and
 * pass any test that only reads back the slots it wrote.
 *
 * ## Decode attention softmax is TILED, not per-key
 *
 * paged_attention_q8_0 accumulates over 32-key tiles: it computes the whole
 * tile's scores, takes that tile's max, rescales the accumulator once, then adds
 * all 32 weighted values. A per-key online softmax gives a numerically different
 * (though equally valid) answer, so the tiling is reproduced to keep the two
 * backends comparable. Sliding window: first = window > 0 ? max(0, context -
 * window) : 0. A block id below zero is skipped entirely -- sparse cache blocks
 * contribute nothing rather than contributing zeros.
 *
 * The query stays fp32; only K and V are quantized. Scores are
 * sum_groups(scale_g * sum_lanes(q * code)).
 *
 *   hipcc -std=c++17 -O3 --offload-arch=gfx942 kv_cache_q8_0.cu -o kv_cache_q8_0.out
 */
#include "../../../../common/cdna3_harness.cuh"

static constexpr int kQ8Group = 32;
static constexpr int kBlock = 256;
static constexpr int kLanes = 32;   // see kernels/attention/rope_variants
static constexpr int kScoreTile = 32;

template <int LANES>
__device__ __forceinline__ float row_reduce_sum(float v) {
#pragma unroll
    for (int offset = LANES / 2; offset > 0; offset >>= 1)
        v += __shfl_xor(v, offset, LANES);
    return v;
}
template <int LANES>
__device__ __forceinline__ float row_reduce_max(float v) {
#pragma unroll
    for (int offset = LANES / 2; offset > 0; offset >>= 1)
        v = fmaxf(v, __shfl_xor(v, offset, LANES));
    return v;
}

/// Round-half-away-from-zero, matching the reference's copysign(floor(|x|+0.5)).
__device__ __forceinline__ int8_t encode_q8(float value, float inverse) {
    const float rounded = copysignf(floorf(fabsf(value * inverse) + 0.5f), value);
    return (int8_t)fminf(fmaxf(rounded, -127.0f), 127.0f);
}

__device__ __forceinline__ long long code_row(long long slot, int head, int heads,
                                              int head_dim) {
    return (slot * heads + head) * head_dim;
}
__device__ __forceinline__ long long scale_row(long long slot, int head, int heads,
                                               int groups) {
    return (slot * heads + head) * groups;
}

// ===========================================================================
// scatter
// ===========================================================================

/// The reference zero-fills the entire cache before scattering. Untouched slots
/// must read back as zero, so this cannot be skipped as an "optimization".
__global__ void k_q8_zero(int8_t *__restrict__ key_codes,
                          uint16_t *__restrict__ key_scales,
                          int8_t *__restrict__ value_codes,
                          uint16_t *__restrict__ value_scales, long long code_count,
                          long long scale_count) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < code_count) {
        key_codes[i] = 0;
        value_codes[i] = 0;
    }
    if (i < scale_count) {
        key_scales[i] = 0;
        value_scales[i] = 0;
    }
}

/// One lane group per (token, head); each lane owns whole groups of 32.
template <int LANES>
__global__ void k_q8_scatter(const float *__restrict__ key,
                             const float *__restrict__ value,
                             const int *__restrict__ slots,
                             int8_t *__restrict__ key_codes,
                             uint16_t *__restrict__ key_scales,
                             int8_t *__restrict__ value_codes,
                             uint16_t *__restrict__ value_scales, int count, int heads,
                             int head_dim, int *__restrict__ invalid) {
    const long long task = (long long)blockIdx.x * (kBlock / LANES) + threadIdx.x / LANES;
    if (task >= (long long)count * heads) return;
    const int lane = threadIdx.x % LANES;
    const int token = task / heads, head = task % heads;
    const int slot = slots[token];
    if (slot < 0) return;  // skipped slots stay zeroed

    const int groups = head_dim / kQ8Group;
    const long long source = ((long long)token * heads + head) * head_dim;
    const long long code_base = code_row(slot, head, heads, head_dim);
    const long long scale_base = scale_row(slot, head, heads, groups);

    for (int group = lane; group < groups; group += LANES) {
        const long long g0 = (long long)group * kQ8Group;
        float key_amax = 0.0f, value_amax = 0.0f;
        bool bad = false;
        for (int l = 0; l < kQ8Group; ++l) {
            const float kv = key[source + g0 + l];
            const float vv = value[source + g0 + l];
            if (!isfinite(kv) || !isfinite(vv)) bad = true;
            key_amax = fmaxf(key_amax, fabsf(kv));
            value_amax = fmaxf(value_amax, fabsf(vv));
        }
        if (bad) {
            atomicExch(invalid, 1);
            continue;
        }
        const float key_scale = key_amax / 127.0f;
        const float value_scale = value_amax / 127.0f;
        const float key_inverse = key_scale > 0.0f ? 1.0f / key_scale : 0.0f;
        const float value_inverse = value_scale > 0.0f ? 1.0f / value_scale : 0.0f;
        key_scales[scale_base + group] = __half_as_ushort(__float2half(key_scale));
        value_scales[scale_base + group] = __half_as_ushort(__float2half(value_scale));
        for (int l = 0; l < kQ8Group; ++l) {
            key_codes[code_base + g0 + l] = encode_q8(key[source + g0 + l], key_inverse);
            value_codes[code_base + g0 + l] =
                encode_q8(value[source + g0 + l], value_inverse);
        }
    }
}

// ===========================================================================
// gather
// ===========================================================================

template <int LANES>
__global__ void k_q8_gather(const int8_t *__restrict__ key_codes,
                            const uint16_t *__restrict__ key_scales,
                            const int8_t *__restrict__ value_codes,
                            const uint16_t *__restrict__ value_scales,
                            const int *__restrict__ block_table,
                            const int *__restrict__ cumulative_lengths,
                            float *__restrict__ key_out, float *__restrict__ value_out,
                            int num_tokens, int sequences, int heads, int head_dim,
                            int page_size, int max_blocks) {
    const long long task = (long long)blockIdx.x * (kBlock / LANES) + threadIdx.x / LANES;
    if (task >= (long long)num_tokens * heads) return;
    const int lane = threadIdx.x % LANES;
    const int token = task / heads, head = task % heads;

    // Locate the sequence owning this token (upper_bound on the prefix sums).
    int sequence = 0;
    for (int s = 1; s <= sequences; ++s) {
        if (cumulative_lengths[s] > token) { sequence = s - 1; break; }
    }
    const int local = token - cumulative_lengths[sequence];
    const int block = block_table[(long long)sequence * max_blocks + local / page_size];
    const long long output_base = ((long long)token * heads + head) * head_dim;

    if (block < 0) {  // hole in the block table reads back as zeros
        for (int d = lane; d < head_dim; d += LANES) {
            key_out[output_base + d] = 0.0f;
            value_out[output_base + d] = 0.0f;
        }
        return;
    }
    const int groups = head_dim / kQ8Group;
    const long long slot = (long long)block * page_size + local % page_size;
    const long long codes = code_row(slot, head, heads, head_dim);
    const long long scales = scale_row(slot, head, heads, groups);
    for (int d = lane; d < head_dim; d += LANES) {
        const int group = d / kQ8Group;
        key_out[output_base + d] =
            key_codes[codes + d] * __half2float(__ushort_as_half(key_scales[scales + group]));
        value_out[output_base + d] =
            value_codes[codes + d] *
            __half2float(__ushort_as_half(value_scales[scales + group]));
    }
}

// ===========================================================================
// copy_blocks -- full cache copy, then per-pair block overwrite
// ===========================================================================

__global__ void k_q8_copy_all(const int8_t *__restrict__ kc, const uint16_t *__restrict__ ks,
                              const int8_t *__restrict__ vc, const uint16_t *__restrict__ vs,
                              int8_t *__restrict__ kco, uint16_t *__restrict__ kso,
                              int8_t *__restrict__ vco, uint16_t *__restrict__ vso,
                              long long code_count, long long scale_count) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < code_count) { kco[i] = kc[i]; vco[i] = vc[i]; }
    if (i < scale_count) { kso[i] = ks[i]; vso[i] = vs[i]; }
}

__global__ void k_q8_copy_pairs(const int8_t *__restrict__ kc,
                                const uint16_t *__restrict__ ks,
                                const int8_t *__restrict__ vc,
                                const uint16_t *__restrict__ vs,
                                int8_t *__restrict__ kco, uint16_t *__restrict__ kso,
                                int8_t *__restrict__ vco, uint16_t *__restrict__ vso,
                                const long long *__restrict__ block_pairs, int pair_count,
                                long long codes_per_block, long long scales_per_block) {
    const int pair = blockIdx.y;
    if (pair >= pair_count) return;
    const long long source = block_pairs[2 * pair];
    const long long destination = block_pairs[2 * pair + 1];
    if (source < 0 || destination < 0) return;
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < codes_per_block) {
        kco[destination * codes_per_block + i] = kc[source * codes_per_block + i];
        vco[destination * codes_per_block + i] = vc[source * codes_per_block + i];
    }
    if (i < scales_per_block) {
        kso[destination * scales_per_block + i] = ks[source * scales_per_block + i];
        vso[destination * scales_per_block + i] = vs[source * scales_per_block + i];
    }
}

// ===========================================================================
// paged attention against the packed cache
// ===========================================================================

/// Baseline: one thread per output row, scalar over head_dim -- the shape a
/// direct transliteration of the reference produces.
__global__ void k_q8_paged_scalar(const float *__restrict__ q,
                                  const int8_t *__restrict__ key_codes,
                                  const uint16_t *__restrict__ key_scales,
                                  const int8_t *__restrict__ value_codes,
                                  const uint16_t *__restrict__ value_scales,
                                  const int *__restrict__ block_table,
                                  const int *__restrict__ context_lens,
                                  float *__restrict__ out, int batch, int query_heads,
                                  int kv_heads, int head_dim, int page_size,
                                  int max_blocks, float score_scale, int window) {
    const long long item = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= (long long)batch * query_heads) return;
    const int request = item / query_heads, head = item % query_heads;
    const int kv_head = head / (query_heads / kv_heads);
    const int context = context_lens[request];
    const int first = window > 0 ? max(0, context - window) : 0;
    const int groups = head_dim / kQ8Group;
    const long long q_base = item * head_dim;

    for (int d = 0; d < head_dim; ++d) out[q_base + d] = 0.0f;
    float maximum = -INFINITY;
    double denominator = 0.0;

    for (int tile = first; tile < context; tile += kScoreTile) {
        const int tile_end = min(context, tile + kScoreTile);
        float scores[kScoreTile];
        long long slots[kScoreTile];
        float tile_maximum = -INFINITY;
        int valid_rows = 0;
        for (int position = tile; position < tile_end; ++position) {
            const int block =
                block_table[(long long)request * max_blocks + position / page_size];
            if (block < 0) continue;
            const long long slot = (long long)block * page_size + position % page_size;
            const long long codes = code_row(slot, kv_head, kv_heads, head_dim);
            const long long scales = scale_row(slot, kv_head, kv_heads, groups);
            double dot = 0.0;
            for (int group = 0; group < groups; ++group) {
                const float gs =
                    __half2float(__ushort_as_half(key_scales[scales + group]));
                double group_dot = 0.0;
                for (int l = 0; l < kQ8Group; ++l)
                    group_dot += q[q_base + group * kQ8Group + l] *
                                 key_codes[codes + group * kQ8Group + l];
                dot += gs * group_dot;
            }
            scores[valid_rows] = (float)(dot * score_scale);
            slots[valid_rows] = slot;
            tile_maximum = fmaxf(tile_maximum, scores[valid_rows]);
            ++valid_rows;
        }
        if (valid_rows == 0) continue;
        const float next_maximum = fmaxf(maximum, tile_maximum);
        const double old_weight = denominator > 0.0 ? exp((double)(maximum - next_maximum)) : 0.0;
        denominator *= old_weight;
        if (old_weight != 1.0)
            for (int d = 0; d < head_dim; ++d) out[q_base + d] *= (float)old_weight;
        for (int index = 0; index < valid_rows; ++index) {
            const double weight = exp((double)(scores[index] - next_maximum));
            denominator += weight;
            const long long codes = code_row(slots[index], kv_head, kv_heads, head_dim);
            const long long scales = scale_row(slots[index], kv_head, kv_heads, groups);
            for (int group = 0; group < groups; ++group) {
                const float ws =
                    (float)(weight *
                            __half2float(__ushort_as_half(value_scales[scales + group])));
                for (int l = 0; l < kQ8Group; ++l)
                    out[q_base + group * kQ8Group + l] +=
                        ws * value_codes[codes + group * kQ8Group + l];
            }
        }
        maximum = next_maximum;
    }
    if (denominator > 0.0) {
        const float inverse = (float)(1.0 / denominator);
        for (int d = 0; d < head_dim; ++d) out[q_base + d] *= inverse;
    }
}

/// Candidate: one lane group per output row. The query is held in registers,
/// each lane owns head_dim/LANES dims, and the tile's scores are produced by a
/// lane-parallel dot product plus a wavefront reduction. The 32-key tiling and
/// its single rescale per tile are preserved exactly.
template <int LANES, int MAX_ITEMS>
__global__ void k_q8_paged(const float *__restrict__ q,
                           const int8_t *__restrict__ key_codes,
                           const uint16_t *__restrict__ key_scales,
                           const int8_t *__restrict__ value_codes,
                           const uint16_t *__restrict__ value_scales,
                           const int *__restrict__ block_table,
                           const int *__restrict__ context_lens, float *__restrict__ out,
                           int batch, int query_heads, int kv_heads, int head_dim,
                           int page_size, int max_blocks, float score_scale, int window) {
    const long long item = (long long)blockIdx.x * (kBlock / LANES) + threadIdx.x / LANES;
    if (item >= (long long)batch * query_heads) return;
    const int lane = threadIdx.x % LANES;
    const int request = item / query_heads, head = item % query_heads;
    const int kv_head = head / (query_heads / kv_heads);
    const int context = context_lens[request];
    const int first = window > 0 ? max(0, context - window) : 0;
    const int groups = head_dim / kQ8Group;
    const long long q_base = item * head_dim;

    float q_val[MAX_ITEMS], acc[MAX_ITEMS];
#pragma unroll
    for (int t = 0; t < MAX_ITEMS; ++t) { q_val[t] = 0.0f; acc[t] = 0.0f; }
    for (int d = lane, t = 0; d < head_dim; d += LANES, ++t) q_val[t] = q[q_base + d];

    float maximum = -INFINITY;
    float denominator = 0.0f;

    for (int tile = first; tile < context; tile += kScoreTile) {
        const int tile_end = min(context, tile + kScoreTile);
        float scores[kScoreTile];
        long long slots[kScoreTile];
        float tile_maximum = -INFINITY;
        int valid_rows = 0;

        for (int position = tile; position < tile_end; ++position) {
            const int block =
                block_table[(long long)request * max_blocks + position / page_size];
            if (block < 0) continue;
            const long long slot = (long long)block * page_size + position % page_size;
            const long long codes = code_row(slot, kv_head, kv_heads, head_dim);
            const long long scales = scale_row(slot, kv_head, kv_heads, groups);
            float dot = 0.0f;
            for (int d = lane, t = 0; d < head_dim; d += LANES, ++t) {
                const float gs =
                    __half2float(__ushort_as_half(key_scales[scales + d / kQ8Group]));
                dot = fmaf(q_val[t] * gs, (float)key_codes[codes + d], dot);
            }
            dot = row_reduce_sum<LANES>(dot);
            scores[valid_rows] = dot * score_scale;
            slots[valid_rows] = slot;
            tile_maximum = fmaxf(tile_maximum, scores[valid_rows]);
            ++valid_rows;
        }
        if (valid_rows == 0) continue;

        const float next_maximum = fmaxf(maximum, tile_maximum);
        const float old_weight = denominator > 0.0f ? __expf(maximum - next_maximum) : 0.0f;
        denominator *= old_weight;
        if (old_weight != 1.0f)
#pragma unroll
            for (int t = 0; t < MAX_ITEMS; ++t) acc[t] *= old_weight;

        for (int index = 0; index < valid_rows; ++index) {
            const float weight = __expf(scores[index] - next_maximum);
            denominator += weight;
            const long long codes = code_row(slots[index], kv_head, kv_heads, head_dim);
            const long long scales = scale_row(slots[index], kv_head, kv_heads, groups);
            for (int d = lane, t = 0; d < head_dim; d += LANES, ++t) {
                const float ws =
                    weight *
                    __half2float(__ushort_as_half(value_scales[scales + d / kQ8Group]));
                acc[t] = fmaf(ws, (float)value_codes[codes + d], acc[t]);
            }
        }
        maximum = next_maximum;
    }

    const float inverse = denominator > 0.0f ? 1.0f / denominator : 0.0f;
    for (int d = lane, t = 0; d < head_dim; d += LANES, ++t)
        out[q_base + d] = acc[t] * inverse;
}

// ===========================================================================
// Host reference (fp64 where the reference uses double)
// ===========================================================================

static uint16_t h_f32_to_f16_bits(float v) {
    __half h = __float2half(v);
    uint16_t bits;
    std::memcpy(&bits, &h, 2);
    return bits;
}
static float h_f16_bits_to_f32(uint16_t bits) {
    __half h;
    std::memcpy(&h, &bits, 2);
    return __half2float(h);
}
static int8_t h_encode_q8(float value, float inverse) {
    const float rounded = std::copysign(std::floor(std::fabs(value * inverse) + 0.5f), value);
    return (int8_t)std::min(std::max(rounded, -127.0f), 127.0f);
}

struct Q8Cache {
    std::vector<int8_t> key_codes, value_codes;
    std::vector<uint16_t> key_scales, value_scales;
};

static Q8Cache oracle_scatter(const std::vector<float> &key, const std::vector<float> &value,
                              const std::vector<int32_t> &slots, long long cache_slots,
                              int count, int heads, int head_dim) {
    const int groups = head_dim / kQ8Group;
    Q8Cache c;
    c.key_codes.assign((size_t)cache_slots * heads * head_dim, 0);
    c.value_codes.assign(c.key_codes.size(), 0);
    c.key_scales.assign((size_t)cache_slots * heads * groups, 0);
    c.value_scales.assign(c.key_scales.size(), 0);
    for (int token = 0; token < count; ++token) {
        const int slot = slots[token];
        if (slot < 0) continue;
        for (int head = 0; head < heads; ++head) {
            const size_t src = ((size_t)token * heads + head) * head_dim;
            const size_t cb = ((size_t)slot * heads + head) * head_dim;
            const size_t sb = ((size_t)slot * heads + head) * groups;
            for (int g = 0; g < groups; ++g) {
                float ka = 0.0f, va = 0.0f;
                for (int l = 0; l < kQ8Group; ++l) {
                    ka = std::max(ka, std::fabs(key[src + g * kQ8Group + l]));
                    va = std::max(va, std::fabs(value[src + g * kQ8Group + l]));
                }
                const float ks = ka / 127.0f, vs = va / 127.0f;
                const float ki = ks > 0.0f ? 1.0f / ks : 0.0f;
                const float vi = vs > 0.0f ? 1.0f / vs : 0.0f;
                c.key_scales[sb + g] = h_f32_to_f16_bits(ks);
                c.value_scales[sb + g] = h_f32_to_f16_bits(vs);
                for (int l = 0; l < kQ8Group; ++l) {
                    c.key_codes[cb + g * kQ8Group + l] =
                        h_encode_q8(key[src + g * kQ8Group + l], ki);
                    c.value_codes[cb + g * kQ8Group + l] =
                        h_encode_q8(value[src + g * kQ8Group + l], vi);
                }
            }
        }
    }
    return c;
}

static void oracle_gather(const Q8Cache &c, const std::vector<int32_t> &block_table,
                          const std::vector<int32_t> &cumulative, int num_tokens,
                          int sequences, int heads, int head_dim, int page_size,
                          int max_blocks, std::vector<double> &key_out,
                          std::vector<double> &value_out) {
    const int groups = head_dim / kQ8Group;
    key_out.assign((size_t)num_tokens * heads * head_dim, 0.0);
    value_out.assign(key_out.size(), 0.0);
    for (int token = 0; token < num_tokens; ++token) {
        int sequence = 0;
        for (int s = 1; s <= sequences; ++s)
            if (cumulative[s] > token) { sequence = s - 1; break; }
        const int local = token - cumulative[sequence];
        const int block = block_table[(size_t)sequence * max_blocks + local / page_size];
        for (int head = 0; head < heads; ++head) {
            const size_t ob = ((size_t)token * heads + head) * head_dim;
            if (block < 0) continue;  // already zero
            const size_t slot = (size_t)block * page_size + local % page_size;
            const size_t cb = (slot * heads + head) * head_dim;
            const size_t sb = (slot * heads + head) * groups;
            for (int d = 0; d < head_dim; ++d) {
                key_out[ob + d] =
                    c.key_codes[cb + d] * h_f16_bits_to_f32(c.key_scales[sb + d / kQ8Group]);
                value_out[ob + d] = c.value_codes[cb + d] *
                                    h_f16_bits_to_f32(c.value_scales[sb + d / kQ8Group]);
            }
        }
    }
}

static std::vector<double> oracle_paged(const std::vector<float> &q, const Q8Cache &c,
                                        const std::vector<int32_t> &block_table,
                                        const std::vector<int32_t> &context_lens, int batch,
                                        int query_heads, int kv_heads, int head_dim,
                                        int page_size, int max_blocks, float scale,
                                        int window) {
    const float score_scale = scale > 0.0f ? scale : 1.0f / std::sqrt((float)head_dim);
    const int groups = head_dim / kQ8Group;
    std::vector<double> out((size_t)batch * query_heads * head_dim, 0.0);
    for (int request = 0; request < batch; ++request)
        for (int head = 0; head < query_heads; ++head) {
            const int kv_head = head / (query_heads / kv_heads);
            const int context = context_lens[request];
            const int first = window > 0 ? std::max(0, context - window) : 0;
            const size_t q_base = ((size_t)request * query_heads + head) * head_dim;
            std::vector<double> acc(head_dim, 0.0);
            float maximum = -std::numeric_limits<float>::infinity();
            double denominator = 0.0;
            for (int tile = first; tile < context; tile += kScoreTile) {
                const int tile_end = std::min(context, tile + kScoreTile);
                std::vector<float> scores;
                std::vector<size_t> slots;
                float tile_maximum = -std::numeric_limits<float>::infinity();
                for (int position = tile; position < tile_end; ++position) {
                    const int block =
                        block_table[(size_t)request * max_blocks + position / page_size];
                    if (block < 0) continue;
                    const size_t slot = (size_t)block * page_size + position % page_size;
                    const size_t cb = (slot * kv_heads + kv_head) * head_dim;
                    const size_t sb = (slot * kv_heads + kv_head) * groups;
                    double dot = 0.0;
                    for (int g = 0; g < groups; ++g) {
                        const float gs = h_f16_bits_to_f32(c.key_scales[sb + g]);
                        double gd = 0.0;
                        for (int l = 0; l < kQ8Group; ++l)
                            gd += q[q_base + g * kQ8Group + l] * c.key_codes[cb + g * kQ8Group + l];
                        dot += gs * gd;
                    }
                    scores.push_back((float)(dot * score_scale));
                    slots.push_back(slot);
                    tile_maximum = std::max(tile_maximum, scores.back());
                }
                if (scores.empty()) continue;
                const float next_maximum = std::max(maximum, tile_maximum);
                const double old_weight =
                    denominator > 0.0 ? std::exp((double)(maximum - next_maximum)) : 0.0;
                denominator *= old_weight;
                if (old_weight != 1.0)
                    for (int d = 0; d < head_dim; ++d) acc[d] *= old_weight;
                for (size_t index = 0; index < scores.size(); ++index) {
                    const double weight = std::exp((double)(scores[index] - next_maximum));
                    denominator += weight;
                    const size_t cb = (slots[index] * kv_heads + kv_head) * head_dim;
                    const size_t sb = (slots[index] * kv_heads + kv_head) * groups;
                    for (int d = 0; d < head_dim; ++d)
                        acc[d] += weight *
                                  h_f16_bits_to_f32(c.value_scales[sb + d / kQ8Group]) *
                                  c.value_codes[cb + d];
                }
                maximum = next_maximum;
            }
            const double inverse = denominator > 0.0 ? 1.0 / denominator : 0.0;
            for (int d = 0; d < head_dim; ++d) out[q_base + d] = acc[d] * inverse;
        }
    return out;
}

// ===========================================================================
// Harness
// ===========================================================================

static int grid_rows(long long rows, int lanes) {
    return (int)((rows + (kBlock / lanes) - 1) / (kBlock / lanes));
}

static bool run_codec(int cache_blocks, int page_size, int count, int heads, int head_dim,
                      bool with_holes, const char *name) {
    qc::Rng rng(31001 + count + head_dim);
    const long long cache_slots = (long long)cache_blocks * page_size;
    const int groups = head_dim / kQ8Group;
    auto key = rng.normals((size_t)count * heads * head_dim, 0.7f);
    auto value = rng.normals((size_t)count * heads * head_dim, 0.7f);
    // Include an all-zero group: amax == 0 must store scale 0 and codes 0 rather
    // than dividing by zero.
    for (int l = 0; l < kQ8Group; ++l) key[l] = 0.0f;

    std::vector<int32_t> slots(count);
    for (int i = 0; i < count; ++i)
        slots[i] = (with_holes && (i % 5 == 3)) ? -1 : (int)((i * 7 + 3) % cache_slots);

    auto ref = oracle_scatter(key, value, slots, cache_slots, count, heads, head_dim);

    auto dkey = qc::dnew(key);
    auto dvalue = qc::dnew(value);
    auto dslots = qc::dnew(slots);
    auto dkc = qc::dzero<int8_t>((size_t)cache_slots * heads * head_dim);
    auto dvc = qc::dzero<int8_t>((size_t)cache_slots * heads * head_dim);
    auto dks = qc::dzero<uint16_t>((size_t)cache_slots * heads * groups);
    auto dvs = qc::dzero<uint16_t>((size_t)cache_slots * heads * groups);
    auto dinvalid = qc::dzero<int>(1);
    const long long code_count = (long long)cache_slots * heads * head_dim;
    const long long scale_count = (long long)cache_slots * heads * groups;

    // Pre-dirty the cache so a missing zero-fill would be caught.
    QC_CHECK(hipMemset(dkc, 0x7f, code_count));
    QC_CHECK(hipMemset(dvc, 0x7f, code_count));

    k_q8_zero<<<(int)((code_count + 255) / 256), 256>>>(dkc, dks, dvc, dvs, code_count,
                                                        scale_count);
    k_q8_scatter<kLanes><<<grid_rows((long long)count * heads, kLanes), kBlock>>>(
        dkey, dvalue, dslots, dkc, dks, dvc, dvs, count, heads, head_dim, dinvalid);
    QC_SYNC();

    char label[160];
    bool ok = true;
    std::snprintf(label, sizeof(label), "kv_cache_scatter_q8_0 %s codes", name);
    {
        auto got = qc::d2h(dkc, (size_t)code_count);
        std::vector<double> refd(ref.key_codes.begin(), ref.key_codes.end());
        ok &= qc::compare(got, refd, qc::Tol::exact()).report(label);
    }
    std::snprintf(label, sizeof(label), "kv_cache_scatter_q8_0 %s scales", name);
    {
        auto got = qc::d2h(dks, (size_t)scale_count);
        std::vector<double> refd(ref.key_scales.begin(), ref.key_scales.end());
        ok &= qc::compare(got, refd, qc::Tol::exact()).report(label);
    }
    std::snprintf(label, sizeof(label), "kv_cache_scatter_q8_0 %s value codes", name);
    {
        auto got = qc::d2h(dvc, (size_t)code_count);
        std::vector<double> refd(ref.value_codes.begin(), ref.value_codes.end());
        ok &= qc::compare(got, refd, qc::Tol::exact()).report(label);
    }

    // --- gather round-trip -------------------------------------------------
    const int sequences = 2, max_blocks = cache_blocks / sequences > 0 ? cache_blocks / sequences : 1;
    const int num_tokens = std::min(count, sequences * max_blocks * page_size);
    std::vector<int32_t> cumulative(sequences + 1, 0);
    for (int s = 0; s < sequences; ++s)
        cumulative[s + 1] = cumulative[s] + num_tokens / sequences;
    cumulative[sequences] = num_tokens;
    std::vector<int32_t> block_table((size_t)sequences * max_blocks);
    for (size_t i = 0; i < block_table.size(); ++i)
        block_table[i] = (with_holes && i % 7 == 2) ? -1 : (int)(i % cache_blocks);

    std::vector<double> kref, vref;
    oracle_gather(ref, block_table, cumulative, num_tokens, sequences, heads, head_dim,
                  page_size, max_blocks, kref, vref);

    auto dbt = qc::dnew(block_table);
    auto dcum = qc::dnew(cumulative);
    auto dko = qc::dzero<float>((size_t)num_tokens * heads * head_dim);
    auto dvo = qc::dzero<float>((size_t)num_tokens * heads * head_dim);
    k_q8_gather<kLanes><<<grid_rows((long long)num_tokens * heads, kLanes), kBlock>>>(
        dkc, dks, dvc, dvs, dbt, dcum, dko, dvo, num_tokens, sequences, heads, head_dim,
        page_size, max_blocks);
    QC_SYNC();
    std::snprintf(label, sizeof(label), "kv_cache_gather_q8_0 %s key", name);
    ok &= qc::compare(qc::d2h(dko, (size_t)num_tokens * heads * head_dim), kref,
                      qc::Tol::fp32())
              .report(label);
    std::snprintf(label, sizeof(label), "kv_cache_gather_q8_0 %s value", name);
    ok &= qc::compare(qc::d2h(dvo, (size_t)num_tokens * heads * head_dim), vref,
                      qc::Tol::fp32())
              .report(label);

    // --- copy_blocks -------------------------------------------------------
    const int pair_count = 3;
    std::vector<long long> pairs = {0, 1, 2, 3, -1, 4};
    pairs.resize((size_t)pair_count * 2);
    const long long codes_per_block = (long long)page_size * heads * head_dim;
    const long long scales_per_block = codes_per_block / kQ8Group;
    Q8Cache copied = ref;
    for (int p = 0; p < pair_count; ++p) {
        const long long s = pairs[2 * p], d = pairs[2 * p + 1];
        if (s < 0 || d < 0) continue;
        std::copy_n(ref.key_codes.begin() + s * codes_per_block, codes_per_block,
                    copied.key_codes.begin() + d * codes_per_block);
        std::copy_n(ref.value_codes.begin() + s * codes_per_block, codes_per_block,
                    copied.value_codes.begin() + d * codes_per_block);
        std::copy_n(ref.key_scales.begin() + s * scales_per_block, scales_per_block,
                    copied.key_scales.begin() + d * scales_per_block);
        std::copy_n(ref.value_scales.begin() + s * scales_per_block, scales_per_block,
                    copied.value_scales.begin() + d * scales_per_block);
    }
    auto dpairs = qc::dnew(pairs);
    auto dkc2 = qc::dzero<int8_t>((size_t)code_count);
    auto dvc2 = qc::dzero<int8_t>((size_t)code_count);
    auto dks2 = qc::dzero<uint16_t>((size_t)scale_count);
    auto dvs2 = qc::dzero<uint16_t>((size_t)scale_count);
    k_q8_copy_all<<<(int)((code_count + 255) / 256), 256>>>(dkc, dks, dvc, dvs, dkc2, dks2,
                                                            dvc2, dvs2, code_count,
                                                            scale_count);
    {
        dim3 grid((unsigned)((codes_per_block + 255) / 256), (unsigned)pair_count);
        k_q8_copy_pairs<<<grid, 256>>>(dkc, dks, dvc, dvs, dkc2, dks2, dvc2, dvs2, dpairs,
                                       pair_count, codes_per_block, scales_per_block);
    }
    QC_SYNC();
    std::snprintf(label, sizeof(label), "kv_cache_copy_blocks_q8_0 %s", name);
    {
        auto got = qc::d2h(dkc2, (size_t)code_count);
        std::vector<double> refd(copied.key_codes.begin(), copied.key_codes.end());
        ok &= qc::compare(got, refd, qc::Tol::exact()).report(label);
    }

    qc::dfree(dkey, dvalue, dko, dvo);
    qc::dfree(dslots, dbt, dcum, dinvalid);
    qc::dfree(dkc, dvc, dkc2, dvc2);
    qc::dfree(dks, dvs, dks2, dvs2);
    qc::dfree(dpairs);
    return ok;
}

template <int MAX_ITEMS>
static bool run_paged(int cache_blocks, int page_size, int batch, int query_heads,
                      int kv_heads, int head_dim, int max_blocks, float scale, int window,
                      bool with_holes, const char *name, bool bench, qc::Bench *b0,
                      qc::Bench *b1) {
    qc::Rng rng(32001 + batch + head_dim + window);
    const long long cache_slots = (long long)cache_blocks * page_size;
    const int groups = head_dim / kQ8Group;
    const int fill_tokens = (int)cache_slots;
    auto key = rng.normals((size_t)fill_tokens * kv_heads * head_dim, 0.7f);
    auto value = rng.normals((size_t)fill_tokens * kv_heads * head_dim, 0.7f);
    std::vector<int32_t> fill_slots(fill_tokens);
    for (int i = 0; i < fill_tokens; ++i) fill_slots[i] = i;
    auto cache = oracle_scatter(key, value, fill_slots, cache_slots, fill_tokens, kv_heads,
                                head_dim);

    auto q = rng.normals((size_t)batch * query_heads * head_dim, 0.6f);
    std::vector<int32_t> block_table((size_t)batch * max_blocks);
    for (size_t i = 0; i < block_table.size(); ++i)
        block_table[i] = (with_holes && i % 11 == 5) ? -1 : (int)(i % cache_blocks);
    std::vector<int32_t> context_lens(batch);
    for (int i = 0; i < batch; ++i)
        context_lens[i] = (i * 37 + 11) % (max_blocks * page_size) + 1;

    auto ref = oracle_paged(q, cache, block_table, context_lens, batch, query_heads,
                            kv_heads, head_dim, page_size, max_blocks, scale, window);
    const float score_scale = scale > 0.0f ? scale : 1.0f / std::sqrt((float)head_dim);

    auto dq = qc::dnew(q);
    auto dkc = qc::dnew(cache.key_codes);
    auto dvc = qc::dnew(cache.value_codes);
    auto dks = qc::dnew(cache.key_scales);
    auto dvs = qc::dnew(cache.value_scales);
    auto dbt = qc::dnew(block_table);
    auto dctx = qc::dnew(context_lens);
    auto dout = qc::dzero<float>((size_t)batch * query_heads * head_dim);
    const long long rows = (long long)batch * query_heads;

    auto fast = [&] {
        k_q8_paged<kLanes, MAX_ITEMS><<<grid_rows(rows, kLanes), kBlock>>>(
            dq, dkc, dks, dvc, dvs, dbt, dctx, dout, batch, query_heads, kv_heads, head_dim,
            page_size, max_blocks, score_scale, window);
    };
    auto scalar = [&] {
        k_q8_paged_scalar<<<(int)((rows + 127) / 128), 128>>>(
            dq, dkc, dks, dvc, dvs, dbt, dctx, dout, batch, query_heads, kv_heads, head_dim,
            page_size, max_blocks, score_scale, window);
    };

    char label[160];
    bool ok = true;
    for (int which = 0; which < 2; ++which) {
        QC_CHECK(hipMemset(dout, 0, (size_t)batch * query_heads * head_dim * sizeof(float)));
        which == 0 ? fast() : scalar();
        QC_SYNC();
        std::snprintf(label, sizeof(label), "paged_attention_q8_0 %s %s", name,
                      which == 0 ? "wavefront" : "scalar");
        ok &= qc::compare(qc::d2h(dout, (size_t)batch * query_heads * head_dim), ref,
                          qc::Tol::fp32().with_elementwise(1e-4, 1e-5))
                  .report(label);
    }
    if (bench && ok) {
        *b0 = qc::bench(scalar, 5, 20);
        *b1 = qc::bench(fast, 10, 30);
    }
    qc::dfree(dq, dout);
    qc::dfree(dkc, dvc);
    qc::dfree(dks, dvs);
    qc::dfree(dbt, dctx);
    return ok;
}

int main(int argc, char **argv) {
    const bool do_bench = qc::bench_requested(argc, argv);
    qc::print_environment("kv_cache_q8_0 (CDNA3 gfx942)");
    bool ok = true;
    qc::Bench b0, b1;

    std::printf("correctness (fp64/bit-exact oracles = quixicore_cpu Q8_0 KV codec):\n");
    ok &= run_codec(8, 16, 96, 4, 128, false, "B8 P16 T96 H4 D128");
    ok &= run_codec(8, 16, 96, 4, 128, true, "B8 P16 T96 H4 D128 holes");
    ok &= run_codec(4, 32, 64, 2, 64, false, "B4 P32 T64 H2 D64");
    ok &= run_codec(6, 16, 50, 3, 256, true, "B6 P16 T50 H3 D256 holes");

    ok &= run_paged<4>(16, 16, 8, 8, 2, 128, 8, 0.0f, 0, false, "B8 Hq8 Hk2 D128", false, nullptr, nullptr);
    ok &= run_paged<4>(16, 16, 8, 8, 2, 128, 8, 0.0f, 64, false, "window=64", false, nullptr, nullptr);
    ok &= run_paged<4>(16, 16, 6, 4, 4, 128, 8, 0.125f, 0, true, "explicit scale + holes", false, nullptr, nullptr);
    ok &= run_paged<2>(8, 32, 5, 4, 1, 64, 4, 0.0f, 0, false, "D64 page32 MQA", false, nullptr, nullptr);

    if (do_bench && ok) {
        std::printf("\nA/B scalar transliteration vs lane-group wavefront "
                    "(HIP-event median):\n");
        ok &= run_paged<4>(512, 16, 128, 32, 8, 128, 32, 0.0f, 0, false,
                           "B128 Hq32 Hk8 D128 ctx512", true, &b0, &b1);
        b0.report("paged_attention_q8_0 scalar");
        b1.report("paged_attention_q8_0 wavefront");
        qc::report_ab("paged_attention_q8_0 B128 Hq32 D128", b0, b1);
    }

    return qc::finish(ok);
}
