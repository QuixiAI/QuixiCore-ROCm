/**
 * @file
 * @brief CDNA3 (gfx942) rotary-position-embedding variants: the eight RoPE
 *        operations the Metal and CPU manifests publish that ROCm lacked.
 *
 * Grouped into one operation directory the way this repo already groups
 * `kernels/serving` and `kernels/linear_attention`: these share their row
 * geometry and their optimization lever, and splitting them into eight
 * near-identical directories would obscure that. Each still carries its own
 * correctness cases and its own focused A/B (see README.cdna3.md).
 *
 * Semantic sources, all in ../QuixiCore-CPU/kernels/attention/:
 *   rotary_positioned          rotary_positioned_ref.cpp:88
 *   mrope                      rotary_positioned_ref.cpp:143
 *   rope_table                 attention_extended_ref.cpp:409
 *   rope_interleaved_to_split  attention_ref.cpp:137
 *   rope_backward              ../utils/tensor_ops_ref.cpp:515
 *   qk_norm_rope_positioned    attention_extended_ref.cpp:453
 *   qk_norm_rope_split         attention_extended_ref.cpp:554
 *   rope_q_norm                attention_serving_ref.cpp:317
 *
 * Two layout conventions run through all of them and must not be conflated:
 *
 *   split        pair p couples elements (p, half + p)      <- the common case
 *   interleaved  pair p couples elements (2p, 2p + 1)
 *
 * `mrope` and `rope_q_norm` are split-only by contract. `rope_interleaved_to_split`
 * reads interleaved and writes split -- it is a layout conversion fused into the
 * rotation, not a rotation with a flag.
 *
 * Rotation, given cos/sin for pair p:
 *   y[first]  = x[first] * cos - x[second] * sin
 *   y[second] = x[second] * cos + x[first] * sin
 * `rope_backward` applies the transpose (rotation by -theta), which is why its
 * signs differ rather than being a copy of the forward with swapped operands.
 *
 * A rotary_dim smaller than head_dim leaves the tail [rotary_dim, head_dim)
 * unrotated -- copied through for the plain variants, and norm-scaled but
 * unrotated for the qk_norm variants. Dropping the tail is a silent accuracy
 * bug on partial-RoPE models, so every variant that accepts rotary_dim has a
 * partial-rotary test case.
 *
 * Candidate: one full 64-lane wavefront per (row) with lanes striding the pair
 * index. Baseline for the A/B: the same kernel at 32 lanes per row, which is the
 * shape a direct CUDA port lands in -- widening to the native wavefront is the
 * lever that measured +30-71% on this repo's norm kernels.
 *
 *   hipcc -std=c++17 -O3 --offload-arch=gfx942 rope_variants.cu -o rope_variants.out
 */
#include "../../../../common/cdna3_harness.cuh"

using qc::StorageBf16;
using qc::StorageF32;
using qc::StorageFp16;

// ===========================================================================
// Device kernels. LANES is the A/B knob: 32 (ported shape) vs 64 (wavefront).
// ===========================================================================

/// Non-interleaved (split) pair indices, or interleaved ones.
__device__ __forceinline__ void pair_indices(bool interleaved, int pair, int pairs,
                                             int &first, int &second) {
    first = interleaved ? 2 * pair : pair;
    second = interleaved ? 2 * pair + 1 : pairs + pair;
}

// --- 1. rotary_positioned: [B, H, T, D], table lookup by token position ------
template <typename S, int LANES>
__global__ void k_rotary_positioned(const typename S::storage *__restrict__ x,
                                    const float *__restrict__ cosine,
                                    const float *__restrict__ sine,
                                    const int *__restrict__ positions,
                                    typename S::storage *__restrict__ y, int B, int H,
                                    int T, int D, int rotary_dim,
                                    int positions_per_batch) {
    const int row = blockIdx.x * (blockDim.x / LANES) + (threadIdx.x / LANES);
    const long long rows = (long long)B * H * T;
    if (row >= rows) return;
    const int lane = threadIdx.x % LANES;

    const int token = row % T;
    const int item = row / (H * T);
    const int position_index = (positions_per_batch ? item * T : 0) + token;
    const int pairs = rotary_dim / 2;
    const long long table = (long long)positions[position_index] * pairs;
    const long long base = (long long)row * D;

    for (int pair = lane; pair < pairs; pair += LANES) {
        int first, second;
        pair_indices(false, pair, pairs, first, second);
        const float a = S::to_float(x[base + first]);
        const float b = S::to_float(x[base + second]);
        const float c = cosine[table + pair], s = sine[table + pair];
        y[base + first] = S::from_float(a * c - b * s);
        y[base + second] = S::from_float(b * c + a * s);
    }
    // Unrotated tail must be copied, not left undefined.
    for (int d = rotary_dim + lane; d < D; d += LANES) y[base + d] = x[base + d];
}

/// Interleaved-layout twin of the above (same table, different pair coupling).
template <typename S, int LANES>
__global__ void k_rotary_positioned_interleaved(
    const typename S::storage *__restrict__ x, const float *__restrict__ cosine,
    const float *__restrict__ sine, const int *__restrict__ positions,
    typename S::storage *__restrict__ y, int B, int H, int T, int D, int rotary_dim,
    int positions_per_batch) {
    const int row = blockIdx.x * (blockDim.x / LANES) + (threadIdx.x / LANES);
    const long long rows = (long long)B * H * T;
    if (row >= rows) return;
    const int lane = threadIdx.x % LANES;

    const int token = row % T;
    const int item = row / (H * T);
    const int position_index = (positions_per_batch ? item * T : 0) + token;
    const int pairs = rotary_dim / 2;
    const long long table = (long long)positions[position_index] * pairs;
    const long long base = (long long)row * D;

    for (int pair = lane; pair < pairs; pair += LANES) {
        int first, second;
        pair_indices(true, pair, pairs, first, second);
        const float a = S::to_float(x[base + first]);
        const float b = S::to_float(x[base + second]);
        const float c = cosine[table + pair], s = sine[table + pair];
        y[base + first] = S::from_float(a * c - b * s);
        y[base + second] = S::from_float(b * c + a * s);
    }
    for (int d = rotary_dim + lane; d < D; d += LANES) y[base + d] = x[base + d];
}

// --- 2. mrope: three position axes, split layout only ------------------------
// section_interleaved picks axis = pair % 3; otherwise the three `sections`
// counts partition the pair range into contiguous x/y/t spans.
template <typename S, int LANES>
__global__ void k_mrope(const typename S::storage *__restrict__ x,
                        const float *__restrict__ cosine,
                        const float *__restrict__ sine,
                        const int *__restrict__ positions,
                        const int *__restrict__ sections,
                        typename S::storage *__restrict__ y, int B, int H, int T, int D,
                        int rotary_dim, int section_interleaved,
                        int positions_per_batch) {
    const int row = blockIdx.x * (blockDim.x / LANES) + (threadIdx.x / LANES);
    const long long rows = (long long)B * H * T;
    if (row >= rows) return;
    const int lane = threadIdx.x % LANES;

    const int token = row % T;
    const int item = row / (H * T);
    const int *item_positions = positions + (positions_per_batch ? item * 3 * T : 0);
    const int selected[3] = {item_positions[token], item_positions[T + token],
                             item_positions[2 * T + token]};
    const int pairs = rotary_dim / 2;
    const int s0 = section_interleaved ? 0 : sections[0];
    const int s1 = section_interleaved ? 0 : sections[1];
    const long long base = (long long)row * D;

    for (int pair = lane; pair < pairs; pair += LANES) {
        int axis;
        if (section_interleaved) {
            axis = pair % 3;
        } else {
            axis = pair < s0 ? 0 : (pair < s0 + s1 ? 1 : 2);
        }
        const long long table = (long long)selected[axis] * pairs + pair;
        const float a = S::to_float(x[base + pair]);
        const float b = S::to_float(x[base + pairs + pair]);
        const float c = cosine[table], s = sine[table];
        y[base + pair] = S::from_float(a * c - b * s);
        y[base + pairs + pair] = S::from_float(b * c + a * s);
    }
    for (int d = rotary_dim + lane; d < D; d += LANES) y[base + d] = x[base + d];
}

// --- 3. rope_table: [T, H, D], full head_dim rotary, shared cos/sin table -----
template <typename S, int LANES>
__global__ void k_rope_table(const typename S::storage *__restrict__ x,
                             const float *__restrict__ cosine,
                             const float *__restrict__ sine,
                             const int *__restrict__ positions,
                             typename S::storage *__restrict__ y, int T, int H, int D,
                             int interleaved) {
    const int row = blockIdx.x * (blockDim.x / LANES) + (threadIdx.x / LANES);
    if (row >= T * H) return;
    const int lane = threadIdx.x % LANES;
    const int token = row / H;
    const int half = D / 2;
    const long long table = (long long)positions[token] * half;
    const long long base = (long long)row * D;

    for (int i = lane; i < half; i += LANES) {
        int first, second;
        pair_indices(interleaved != 0, i, half, first, second);
        const float a = S::to_float(x[base + first]);
        const float b = S::to_float(x[base + second]);
        const float c = cosine[table + i], s = sine[table + i];
        y[base + first] = S::from_float(a * c - b * s);
        y[base + second] = S::from_float(b * c + a * s);
    }
}

// --- 4. rope_interleaved_to_split: reads interleaved, writes split ------------
// Angles are derived from `base_theta` and `pos0 + token` rather than a table.
// The reference computes them in double; so does this, because the frequency
// spread across pairs is exactly where float loses the low-order bits.
template <typename S, int LANES>
__global__ void k_rope_interleaved_to_split(const typename S::storage *__restrict__ x,
                                            typename S::storage *__restrict__ y, int T,
                                            int H, int D, float base_theta,
                                            long long pos0) {
    const int row = blockIdx.x * (blockDim.x / LANES) + (threadIdx.x / LANES);
    if (row >= T * H) return;
    const int lane = threadIdx.x % LANES;
    const int token = row / H;
    const int half = D / 2;
    const long long base = (long long)row * D;
    const double position = (double)(pos0 + token);

    // pow() per element made this kernel transcendental-bound (1.2 TB/s against
    // 4.0 TB/s for the table-driven variants). base^e == exp2(e * log2(base)),
    // and log2(base) is loop-invariant, so this trades a pow for one exp2; one
    // sincos then replaces a separate cos and sin. Still evaluated in double:
    // the frequency spread across pairs is where float loses the low bits.
    const double log2_base = log2((double)base_theta);
    for (int pair = lane; pair < half; pair += LANES) {
        const double frequency = exp2(-2.0 * (double)pair / (double)D * log2_base);
        const double angle = position * frequency;
        double sd, cd;
        sincos(angle, &sd, &cd);
        const float c = (float)cd, s = (float)sd;
        const float a = S::to_float(x[base + 2 * pair]);       // interleaved in
        const float b = S::to_float(x[base + 2 * pair + 1]);
        y[base + pair] = S::from_float(a * c - b * s);         // split out
        y[base + half + pair] = S::from_float(b * c + a * s);
    }
}

// --- 5. rope_backward: transpose of the split rotation -----------------------
template <typename S, int LANES>
__global__ void k_rope_backward(const typename S::storage *__restrict__ grad_out,
                                typename S::storage *__restrict__ grad_in, int T, int H,
                                int D, float base_theta, long long pos0) {
    const int row = blockIdx.x * (blockDim.x / LANES) + (threadIdx.x / LANES);
    if (row >= T * H) return;
    const int lane = threadIdx.x % LANES;
    const int token = row / H;
    const int half = D / 2;
    const long long base = (long long)row * D;
    const double position = (double)(pos0 + token);

    // Same pow -> exp2 + sincos change as the forward conversion above.
    const double log2_base = log2((double)base_theta);
    for (int i = lane; i < half; i += LANES) {
        const double theta = position * exp2(-2.0 * (double)i / (double)D * log2_base);
        double sd, cd;
        sincos(theta, &sd, &cd);
        const float c = (float)cd, s = (float)sd;
        const float a = S::to_float(grad_out[base + i]);
        const float b = S::to_float(grad_out[base + half + i]);
        // Rotation by -theta: note the sign pattern differs from the forward.
        grad_in[base + i] = S::from_float(a * c + b * s);
        grad_in[base + half + i] = S::from_float(-a * s + b * c);
    }
}

// --- 6/7. qk_norm_rope_positioned and _split ---------------------------------
// Packed QKV [T, Hq+Hk+Hv, D]. Q and K heads are RMS-normalized over the FULL
// head_dim (not rotary_dim), scaled by their per-dim weight, then rotated over
// rotary_dim. V heads are copied through untouched. `weight_offset` carries
// Gemma's stored "(1+w)" convention without materializing an adjusted weight.
template <typename S, int LANES>
__global__ void k_qk_norm_rope(const typename S::storage *__restrict__ qkv,
                               const float *__restrict__ q_weight,
                               const float *__restrict__ k_weight,
                               const float *__restrict__ cosine,
                               const float *__restrict__ sine,
                               const int *__restrict__ positions,
                               typename S::storage *__restrict__ y,
                               typename S::storage *__restrict__ q_out,
                               typename S::storage *__restrict__ k_out,
                               typename S::storage *__restrict__ v_out, int T, int Hq,
                               int Hk, int Hv, int D, int rotary_dim, float eps,
                               int interleaved, float weight_offset, int split_outputs) {
    const int total_heads = Hq + Hk + Hv;
    const int row = blockIdx.x * (blockDim.x / LANES) + (threadIdx.x / LANES);
    if (row >= T * total_heads) return;
    const int lane = threadIdx.x % LANES;
    const int token = row / total_heads;
    const int head = row % total_heads;
    const long long base = (long long)row * D;

    // Where this head's output goes: one packed buffer, or three split ones.
    typename S::storage *dst;
    long long dst_base;
    if (split_outputs) {
        if (head < Hq) {
            dst = q_out;
            dst_base = ((long long)token * Hq + head) * D;
        } else if (head < Hq + Hk) {
            dst = k_out;
            dst_base = ((long long)token * Hk + (head - Hq)) * D;
        } else {
            dst = v_out;
            dst_base = ((long long)token * Hv + (head - Hq - Hk)) * D;
        }
    } else {
        dst = y;
        dst_base = base;
    }

    if (head >= Hq + Hk) {  // value head: copy through
        for (int d = lane; d < D; d += LANES) dst[dst_base + d] = qkv[base + d];
        return;
    }

    // RMS over the full head_dim, reduced across the row's lanes.
    float squares = 0.0f;
    for (int d = lane; d < D; d += LANES) {
        const float v = S::to_float(qkv[base + d]);
        squares = fmaf(v, v, squares);
    }
#pragma unroll
    for (int offset = LANES / 2; offset > 0; offset >>= 1)
        squares += __shfl_xor(squares, offset, LANES);
    const float inverse = 1.0f / sqrtf(squares / (float)D + eps);

    const float *weight = head < Hq ? q_weight : k_weight;
    const int pairs = rotary_dim / 2;
    const long long table = (long long)positions[token] * pairs;

    for (int pair = lane; pair < pairs; pair += LANES) {
        int first, second;
        pair_indices(interleaved != 0, pair, pairs, first, second);
        // Norm and weight are applied BEFORE the rotation.
        const float a =
            S::to_float(qkv[base + first]) * inverse * (weight[first] + weight_offset);
        const float b =
            S::to_float(qkv[base + second]) * inverse * (weight[second] + weight_offset);
        const float c = cosine[table + pair], s = sine[table + pair];
        dst[dst_base + first] = S::from_float(a * c - b * s);
        dst[dst_base + second] = S::from_float(b * c + a * s);
    }
    // Unrotated tail is still normalized and weighted.
    for (int d = rotary_dim + lane; d < D; d += LANES)
        dst[dst_base + d] = S::from_float(S::to_float(qkv[base + d]) * inverse *
                                          (weight[d] + weight_offset));
}

/// mrope flavour of the above (3-axis table, split layout, no interleave).
template <typename S, int LANES>
__global__ void k_qk_norm_rope_mrope(
    const typename S::storage *__restrict__ qkv, const float *__restrict__ q_weight,
    const float *__restrict__ k_weight, const float *__restrict__ cosine,
    const float *__restrict__ sine, const int *__restrict__ positions,
    const int *__restrict__ sections, typename S::storage *__restrict__ y, int T, int Hq,
    int Hk, int Hv, int D, int rotary_dim, float eps, float weight_offset,
    int section_interleaved) {
    const int total_heads = Hq + Hk + Hv;
    const int row = blockIdx.x * (blockDim.x / LANES) + (threadIdx.x / LANES);
    if (row >= T * total_heads) return;
    const int lane = threadIdx.x % LANES;
    const int token = row / total_heads;
    const int head = row % total_heads;
    const long long base = (long long)row * D;

    if (head >= Hq + Hk) {
        for (int d = lane; d < D; d += LANES) y[base + d] = qkv[base + d];
        return;
    }

    float squares = 0.0f;
    for (int d = lane; d < D; d += LANES) {
        const float v = S::to_float(qkv[base + d]);
        squares = fmaf(v, v, squares);
    }
#pragma unroll
    for (int offset = LANES / 2; offset > 0; offset >>= 1)
        squares += __shfl_xor(squares, offset, LANES);
    const float inverse = 1.0f / sqrtf(squares / (float)D + eps);

    const float *weight = head < Hq ? q_weight : k_weight;
    const int pairs = rotary_dim / 2;
    const int selected[3] = {positions[token], positions[T + token],
                             positions[2 * T + token]};
    const int s0 = section_interleaved ? 0 : sections[0];
    const int s1 = section_interleaved ? 0 : sections[1];

    for (int pair = lane; pair < pairs; pair += LANES) {
        const int axis = section_interleaved ? (pair % 3)
                                             : (pair < s0 ? 0 : (pair < s0 + s1 ? 1 : 2));
        const long long table = (long long)selected[axis] * pairs + pair;
        const float a =
            S::to_float(qkv[base + pair]) * inverse * (weight[pair] + weight_offset);
        const float b = S::to_float(qkv[base + pairs + pair]) * inverse *
                        (weight[pairs + pair] + weight_offset);
        const float c = cosine[table], s = sine[table];
        y[base + pair] = S::from_float(a * c - b * s);
        y[base + pairs + pair] = S::from_float(b * c + a * s);
    }
    for (int d = rotary_dim + lane; d < D; d += LANES)
        y[base + d] = S::from_float(S::to_float(qkv[base + d]) * inverse *
                                    (weight[d] + weight_offset));
}

// --- 8. rope_q_norm: [T, H, D], optional norm, split layout, full rotary -----
template <typename S, int LANES>
__global__ void k_rope_q_norm(const typename S::storage *__restrict__ q,
                              const float *__restrict__ cosine,
                              const float *__restrict__ sine,
                              const int *__restrict__ positions,
                              const float *__restrict__ norm_weight,
                              typename S::storage *__restrict__ out, int T, int H, int D,
                              int do_norm, float gemma_offset, float eps) {
    const int row = blockIdx.x * (blockDim.x / LANES) + (threadIdx.x / LANES);
    if (row >= T * H) return;
    const int lane = threadIdx.x % LANES;
    const int token = row / H;
    const int half = D / 2;
    const long long base = (long long)row * D;

    float inverse = 1.0f;
    if (do_norm) {
        float squares = 0.0f;
        for (int d = lane; d < D; d += LANES) {
            const float v = S::to_float(q[base + d]);
            squares = fmaf(v, v, squares);
        }
#pragma unroll
        for (int offset = LANES / 2; offset > 0; offset >>= 1)
            squares += __shfl_xor(squares, offset, LANES);
        inverse = 1.0f / sqrtf(squares / (float)D + eps);
    }

    const long long table = (long long)positions[token] * half;
    for (int d = lane; d < half; d += LANES) {
        const float w0 = do_norm ? norm_weight[d] + gemma_offset : 1.0f;
        const float w1 = do_norm ? norm_weight[d + half] + gemma_offset : 1.0f;
        const float a = S::to_float(q[base + d]) * inverse * w0;
        const float b = S::to_float(q[base + half + d]) * inverse * w1;
        const float c = cosine[table + d], s = sine[table + d];
        out[base + d] = S::from_float(a * c - b * s);
        out[base + half + d] = S::from_float(b * c + a * s);
    }
}

// ===========================================================================
// fp64 oracles
// ===========================================================================

static void build_table(std::vector<float> &cosine, std::vector<float> &sine,
                        int max_position, int pairs, double base_theta) {
    cosine.resize((size_t)max_position * pairs);
    sine.resize((size_t)max_position * pairs);
    for (int p = 0; p < max_position; ++p)
        for (int i = 0; i < pairs; ++i) {
            const double angle = (double)p * std::pow(base_theta, -2.0 * i / (2.0 * pairs));
            cosine[(size_t)p * pairs + i] = (float)std::cos(angle);
            sine[(size_t)p * pairs + i] = (float)std::sin(angle);
        }
}

static std::vector<double> oracle_rotary_positioned(
    const std::vector<float> &x, const std::vector<float> &cosine,
    const std::vector<float> &sine, const std::vector<int32_t> &positions, int B, int H,
    int T, int D, int rotary_dim, bool interleaved, bool positions_per_batch) {
    std::vector<double> y((size_t)B * H * T * D);
    const int pairs = rotary_dim / 2;
    for (long long row = 0; row < (long long)B * H * T; ++row) {
        const int token = row % T;
        const int item = row / (H * T);
        const long long table =
            (long long)positions[(positions_per_batch ? item * T : 0) + token] * pairs;
        const long long base = row * D;
        for (int pair = 0; pair < pairs; ++pair) {
            const int first = interleaved ? 2 * pair : pair;
            const int second = interleaved ? 2 * pair + 1 : pairs + pair;
            const double a = x[base + first], b = x[base + second];
            const double c = cosine[table + pair], s = sine[table + pair];
            y[base + first] = a * c - b * s;
            y[base + second] = b * c + a * s;
        }
        for (int d = rotary_dim; d < D; ++d) y[base + d] = x[base + d];
    }
    return y;
}

static std::vector<double> oracle_mrope(const std::vector<float> &x,
                                        const std::vector<float> &cosine,
                                        const std::vector<float> &sine,
                                        const std::vector<int32_t> &positions,
                                        const std::vector<int32_t> &sections, int B, int H,
                                        int T, int D, int rotary_dim,
                                        bool section_interleaved,
                                        bool positions_per_batch) {
    std::vector<double> y((size_t)B * H * T * D);
    const int pairs = rotary_dim / 2;
    for (long long row = 0; row < (long long)B * H * T; ++row) {
        const int token = row % T;
        const int item = row / (H * T);
        const int32_t *item_positions =
            positions.data() + (positions_per_batch ? (size_t)item * 3 * T : 0);
        const int selected[3] = {item_positions[token], item_positions[T + token],
                                 item_positions[2 * T + token]};
        const long long base = row * D;
        for (int pair = 0; pair < pairs; ++pair) {
            int axis;
            if (section_interleaved) axis = pair % 3;
            else axis = pair < sections[0] ? 0
                                           : (pair < sections[0] + sections[1] ? 1 : 2);
            const long long table = (long long)selected[axis] * pairs + pair;
            const double a = x[base + pair], b = x[base + pairs + pair];
            const double c = cosine[table], s = sine[table];
            y[base + pair] = a * c - b * s;
            y[base + pairs + pair] = b * c + a * s;
        }
        for (int d = rotary_dim; d < D; ++d) y[base + d] = x[base + d];
    }
    return y;
}

static std::vector<double> oracle_rope_table(const std::vector<float> &x,
                                             const std::vector<float> &cosine,
                                             const std::vector<float> &sine,
                                             const std::vector<int32_t> &positions, int T,
                                             int H, int D, bool interleaved) {
    std::vector<double> y((size_t)T * H * D);
    const int half = D / 2;
    for (long long row = 0; row < (long long)T * H; ++row) {
        const int token = row / H;
        const long long table = (long long)positions[token] * half;
        const long long base = row * D;
        for (int i = 0; i < half; ++i) {
            const int first = interleaved ? 2 * i : i;
            const int second = interleaved ? 2 * i + 1 : half + i;
            const double a = x[base + first], b = x[base + second];
            const double c = cosine[table + i], s = sine[table + i];
            y[base + first] = a * c - b * s;
            y[base + second] = b * c + a * s;
        }
    }
    return y;
}

static std::vector<double> oracle_interleaved_to_split(const std::vector<float> &x, int T,
                                                       int H, int D, double base_theta,
                                                       long long pos0) {
    std::vector<double> y((size_t)T * H * D);
    const int half = D / 2;
    for (long long row = 0; row < (long long)T * H; ++row) {
        const int token = row / H;
        const long long base = row * D;
        const double position = (double)(pos0 + token);
        for (int pair = 0; pair < half; ++pair) {
            const double frequency = std::pow(base_theta, -2.0 * pair / (double)D);
            const double angle = position * frequency;
            const float c = (float)std::cos(angle), s = (float)std::sin(angle);
            const double a = x[base + 2 * pair], b = x[base + 2 * pair + 1];
            y[base + pair] = a * c - b * s;
            y[base + half + pair] = b * c + a * s;
        }
    }
    return y;
}

static std::vector<double> oracle_rope_backward(const std::vector<float> &grad_out, int T,
                                                int H, int D, double base_theta,
                                                long long pos0) {
    std::vector<double> gi((size_t)T * H * D);
    const int half = D / 2;
    for (long long row = 0; row < (long long)T * H; ++row) {
        const int token = row / H;
        const long long base = row * D;
        const double position = (double)(pos0 + token);
        for (int i = 0; i < half; ++i) {
            const double theta = position * std::pow(base_theta, -2.0 * i / (double)D);
            const float c = (float)std::cos(theta), s = (float)std::sin(theta);
            const double a = grad_out[base + i], b = grad_out[base + half + i];
            gi[base + i] = a * c + b * s;
            gi[base + half + i] = -a * s + b * c;
        }
    }
    return gi;
}

static std::vector<double> oracle_qk_norm_rope(
    const std::vector<float> &qkv, const std::vector<float> &qw,
    const std::vector<float> &kw, const std::vector<float> &cosine,
    const std::vector<float> &sine, const std::vector<int32_t> &positions, int T, int Hq,
    int Hk, int Hv, int D, int rotary_dim, float eps, bool interleaved,
    float weight_offset) {
    const int total_heads = Hq + Hk + Hv;
    std::vector<double> y((size_t)T * total_heads * D);
    const int pairs = rotary_dim / 2;
    for (int token = 0; token < T; ++token) {
        for (int head = 0; head < total_heads; ++head) {
            const long long base = ((long long)token * total_heads + head) * D;
            if (head >= Hq + Hk) {
                for (int d = 0; d < D; ++d) y[base + d] = qkv[base + d];
                continue;
            }
            double squares = 0.0;
            for (int d = 0; d < D; ++d) squares += (double)qkv[base + d] * qkv[base + d];
            const double inverse = 1.0 / std::sqrt(squares / D + (double)eps);
            const std::vector<float> &w = head < Hq ? qw : kw;
            const long long table = (long long)positions[token] * pairs;
            for (int pair = 0; pair < pairs; ++pair) {
                const int first = interleaved ? 2 * pair : pair;
                const int second = interleaved ? 2 * pair + 1 : pairs + pair;
                const double a = qkv[base + first] * inverse * (w[first] + weight_offset);
                const double b =
                    qkv[base + second] * inverse * (w[second] + weight_offset);
                const double c = cosine[table + pair], s = sine[table + pair];
                y[base + first] = a * c - b * s;
                y[base + second] = b * c + a * s;
            }
            for (int d = rotary_dim; d < D; ++d)
                y[base + d] = qkv[base + d] * inverse * (w[d] + weight_offset);
        }
    }
    return y;
}

static std::vector<double> oracle_rope_q_norm(const std::vector<float> &q,
                                              const std::vector<float> &cosine,
                                              const std::vector<float> &sine,
                                              const std::vector<int32_t> &positions,
                                              const std::vector<float> &norm_weight, int T,
                                              int H, int D, bool do_norm,
                                              float gemma_offset, float eps) {
    std::vector<double> out((size_t)T * H * D);
    const int half = D / 2;
    for (long long row = 0; row < (long long)T * H; ++row) {
        const int token = row / H;
        const long long base = row * D;
        double inverse = 1.0;
        if (do_norm) {
            double squares = 0.0;
            for (int d = 0; d < D; ++d) squares += (double)q[base + d] * q[base + d];
            inverse = 1.0 / std::sqrt(squares / D + (double)eps);
        }
        const long long table = (long long)positions[token] * half;
        for (int d = 0; d < half; ++d) {
            const double w0 = do_norm ? norm_weight[d] + gemma_offset : 1.0;
            const double w1 = do_norm ? norm_weight[d + half] + gemma_offset : 1.0;
            const double a = q[base + d] * inverse * w0;
            const double b = q[base + half + d] * inverse * w1;
            const double c = cosine[table + d], s = sine[table + d];
            out[base + d] = a * c - b * s;
            out[base + half + d] = b * c + a * s;
        }
    }
    return out;
}

// ===========================================================================
// Harness
// ===========================================================================

static constexpr int kBlock = 256;

// Selected configuration: 32 lanes per row.
//
// Widening to the full 64-lane wavefront is this repo's standard first lever for
// row kernels (+30-71% on norms), and it was measured here and REJECTED: it lost
// on all eight operations, 0.82x-0.96x, reproduced across two independent runs.
//
// The norm result does not transfer because it came from a shape where half the
// wavefront sat idle. These kernels pack `kBlock / LANES` rows per block, so a
// 32-lane row already fills the wavefront with two rows. Halving the lanes per
// row then doubles the pairs each lane owns, which doubles the independent loads
// in flight -- and with no cross-lane reduction to amortize, that extra ILP is
// worth more than the wider row. See README.cdna3.md.
static constexpr int kRopeLanes = 32;

template <int LANES>
static int rows_grid(long long rows) {
    const int rows_per_block = kBlock / LANES;
    return (int)((rows + rows_per_block - 1) / rows_per_block);
}

/// Runs one operation at 32 and 64 lanes/row, checks both, optionally times both.
struct OpResult {
    bool ok = true;
    qc::Bench baseline, candidate;
    bool timed = false;
};

template <typename S, typename LaunchFn>
static bool check_launch(LaunchFn &&launch, typename S::storage *dout, size_t count,
                         const std::vector<double> &ref, const char *label) {
    QC_CHECK(hipMemset(dout, 0, count * sizeof(typename S::storage)));
    launch();
    QC_SYNC();
    return qc::compare(qc::d2h(dout, count), ref, qc::tol_for<S>()).report(label);
}

// --- per-operation drivers ---------------------------------------------------

template <typename S>
static bool op_rotary_positioned(int B, int H, int T, int D, int rotary_dim,
                                 bool interleaved, bool per_batch, const char *name,
                                 bool bench, OpResult *out) {
    qc::Rng rng(11001 + D + T);
    const size_t count = (size_t)B * H * T * D;
    const int max_position = 512;
    const int pairs = rotary_dim / 2;
    std::vector<float> cosine, sine;
    build_table(cosine, sine, max_position, pairs, 10000.0);
    auto xf = rng.normals(count, 0.8f);
    const int position_count = per_batch ? B * T : T;
    auto positions = rng.integers(position_count, 0, max_position - 1);

    auto xs = qc::to_storage<typename S::storage>(xf);
    std::vector<float> xr(count);
    for (size_t i = 0; i < count; ++i) xr[i] = (float)qc::to_double(xs[i]);
    auto ref = oracle_rotary_positioned(xr, cosine, sine, positions, B, H, T, D,
                                        rotary_dim, interleaved, per_batch);

    auto dx = qc::dnew(xs);
    auto dc = qc::dnew(cosine);
    auto ds = qc::dnew(sine);
    auto dp = qc::dnew(positions);
    auto dy = qc::dzero<typename S::storage>(count);
    const long long rows = (long long)B * H * T;

    auto make = [&](int lanes) {
        return [&, lanes] {
            if (interleaved) {
                if (lanes == 32)
                    k_rotary_positioned_interleaved<S, 32><<<rows_grid<32>(rows), kBlock>>>(
                        dx, dc, ds, dp, dy, B, H, T, D, rotary_dim, per_batch);
                else
                    k_rotary_positioned_interleaved<S, 64><<<rows_grid<64>(rows), kBlock>>>(
                        dx, dc, ds, dp, dy, B, H, T, D, rotary_dim, per_batch);
            } else {
                if (lanes == 32)
                    k_rotary_positioned<S, 32><<<rows_grid<32>(rows), kBlock>>>(
                        dx, dc, ds, dp, dy, B, H, T, D, rotary_dim, per_batch);
                else
                    k_rotary_positioned<S, 64><<<rows_grid<64>(rows), kBlock>>>(
                        dx, dc, ds, dp, dy, B, H, T, D, rotary_dim, per_batch);
            }
        };
    };
    char label[192];
    bool ok = true;
    std::snprintf(label, sizeof(label), "rotary_positioned %s [%s] w32", name, S::name);
    ok &= check_launch<S>(make(32), dy, count, ref, label);
    std::snprintf(label, sizeof(label), "rotary_positioned %s [%s] w64", name, S::name);
    ok &= check_launch<S>(make(64), dy, count, ref, label);
    if (bench && ok && out) {
        out->baseline = qc::bench(make(32), 25, 50);
        out->candidate = qc::bench(make(64), 25, 50);
        out->timed = true;
    }
    qc::dfree(dx, dy);
    qc::dfree(dc, ds);
    qc::dfree(dp);
    return ok;
}

template <typename S>
static bool op_mrope(int B, int H, int T, int D, int rotary_dim, bool section_interleaved,
                     bool per_batch, const char *name, bool bench, OpResult *out) {
    qc::Rng rng(11002 + D + T);
    const size_t count = (size_t)B * H * T * D;
    const int max_position = 512;
    const int pairs = rotary_dim / 2;
    std::vector<float> cosine, sine;
    build_table(cosine, sine, max_position, pairs, 10000.0);
    auto xf = rng.normals(count, 0.8f);
    auto positions = rng.integers((per_batch ? B : 1) * 3 * T, 0, max_position - 1);
    // Contiguous x/y/t spans that sum to `pairs`, as the reference requires.
    std::vector<int32_t> sections(3);
    sections[0] = (pairs + 2) / 3;
    sections[1] = (pairs + 1) / 3;
    sections[2] = pairs - sections[0] - sections[1];

    auto xs = qc::to_storage<typename S::storage>(xf);
    std::vector<float> xr(count);
    for (size_t i = 0; i < count; ++i) xr[i] = (float)qc::to_double(xs[i]);
    auto ref = oracle_mrope(xr, cosine, sine, positions, sections, B, H, T, D, rotary_dim,
                            section_interleaved, per_batch);

    auto dx = qc::dnew(xs);
    auto dc = qc::dnew(cosine);
    auto ds = qc::dnew(sine);
    auto dp = qc::dnew(positions);
    auto dsec = qc::dnew(sections);
    auto dy = qc::dzero<typename S::storage>(count);
    const long long rows = (long long)B * H * T;

    auto make = [&](int lanes) {
        return [&, lanes] {
            if (lanes == 32)
                k_mrope<S, 32><<<rows_grid<32>(rows), kBlock>>>(
                    dx, dc, ds, dp, dsec, dy, B, H, T, D, rotary_dim, section_interleaved,
                    per_batch);
            else
                k_mrope<S, 64><<<rows_grid<64>(rows), kBlock>>>(
                    dx, dc, ds, dp, dsec, dy, B, H, T, D, rotary_dim, section_interleaved,
                    per_batch);
        };
    };
    char label[192];
    bool ok = true;
    std::snprintf(label, sizeof(label), "mrope %s [%s] w32", name, S::name);
    ok &= check_launch<S>(make(32), dy, count, ref, label);
    std::snprintf(label, sizeof(label), "mrope %s [%s] w64", name, S::name);
    ok &= check_launch<S>(make(64), dy, count, ref, label);
    if (bench && ok && out) {
        out->baseline = qc::bench(make(32), 25, 50);
        out->candidate = qc::bench(make(64), 25, 50);
        out->timed = true;
    }
    qc::dfree(dx, dy);
    qc::dfree(dc, ds);
    qc::dfree(dp, dsec);
    return ok;
}

template <typename S>
static bool op_rope_table(int T, int H, int D, bool interleaved, const char *name,
                          bool bench, OpResult *out) {
    qc::Rng rng(11003 + D + T);
    const size_t count = (size_t)T * H * D;
    const int max_position = 1024, half = D / 2;
    std::vector<float> cosine, sine;
    build_table(cosine, sine, max_position, half, 10000.0);
    auto xf = rng.normals(count, 0.8f);
    auto positions = rng.integers(T, 0, max_position - 1);

    auto xs = qc::to_storage<typename S::storage>(xf);
    std::vector<float> xr(count);
    for (size_t i = 0; i < count; ++i) xr[i] = (float)qc::to_double(xs[i]);
    auto ref = oracle_rope_table(xr, cosine, sine, positions, T, H, D, interleaved);

    auto dx = qc::dnew(xs);
    auto dc = qc::dnew(cosine);
    auto ds = qc::dnew(sine);
    auto dp = qc::dnew(positions);
    auto dy = qc::dzero<typename S::storage>(count);

    auto make = [&](int lanes) {
        return [&, lanes] {
            if (lanes == 32)
                k_rope_table<S, 32><<<rows_grid<32>((long long)T * H), kBlock>>>(
                    dx, dc, ds, dp, dy, T, H, D, interleaved);
            else
                k_rope_table<S, 64><<<rows_grid<64>((long long)T * H), kBlock>>>(
                    dx, dc, ds, dp, dy, T, H, D, interleaved);
        };
    };
    char label[192];
    bool ok = true;
    std::snprintf(label, sizeof(label), "rope_table %s [%s] w32", name, S::name);
    ok &= check_launch<S>(make(32), dy, count, ref, label);
    std::snprintf(label, sizeof(label), "rope_table %s [%s] w64", name, S::name);
    ok &= check_launch<S>(make(64), dy, count, ref, label);
    if (bench && ok && out) {
        out->baseline = qc::bench(make(32), 25, 50);
        out->candidate = qc::bench(make(64), 25, 50);
        out->timed = true;
    }
    qc::dfree(dx, dy);
    qc::dfree(dc, ds);
    qc::dfree(dp);
    return ok;
}

template <typename S>
static bool op_interleaved_to_split(int T, int H, int D, long long pos0, const char *name,
                                    bool bench, OpResult *out) {
    qc::Rng rng(11004 + D + T);
    const size_t count = (size_t)T * H * D;
    const double base_theta = 10000.0;
    auto xf = rng.normals(count, 0.8f);
    auto xs = qc::to_storage<typename S::storage>(xf);
    std::vector<float> xr(count);
    for (size_t i = 0; i < count; ++i) xr[i] = (float)qc::to_double(xs[i]);
    auto ref = oracle_interleaved_to_split(xr, T, H, D, base_theta, pos0);

    auto dx = qc::dnew(xs);
    auto dy = qc::dzero<typename S::storage>(count);
    auto make = [&](int lanes) {
        return [&, lanes] {
            if (lanes == 32)
                k_rope_interleaved_to_split<S, 32>
                    <<<rows_grid<32>((long long)T * H), kBlock>>>(dx, dy, T, H, D,
                                                                  (float)base_theta, pos0);
            else
                k_rope_interleaved_to_split<S, 64>
                    <<<rows_grid<64>((long long)T * H), kBlock>>>(dx, dy, T, H, D,
                                                                  (float)base_theta, pos0);
        };
    };
    char label[192];
    bool ok = true;
    std::snprintf(label, sizeof(label), "rope_interleaved_to_split %s [%s] w32", name,
                  S::name);
    ok &= check_launch<S>(make(32), dy, count, ref, label);
    std::snprintf(label, sizeof(label), "rope_interleaved_to_split %s [%s] w64", name,
                  S::name);
    ok &= check_launch<S>(make(64), dy, count, ref, label);
    if (bench && ok && out) {
        out->baseline = qc::bench(make(32), 25, 50);
        out->candidate = qc::bench(make(64), 25, 50);
        out->timed = true;
    }
    qc::dfree(dx, dy);
    return ok;
}

template <typename S>
static bool op_rope_backward(int T, int H, int D, long long pos0, const char *name,
                             bool bench, OpResult *out) {
    qc::Rng rng(11005 + D + T);
    const size_t count = (size_t)T * H * D;
    const double base_theta = 10000.0;
    auto gf = rng.normals(count, 0.8f);
    auto gs = qc::to_storage<typename S::storage>(gf);
    std::vector<float> gr(count);
    for (size_t i = 0; i < count; ++i) gr[i] = (float)qc::to_double(gs[i]);
    auto ref = oracle_rope_backward(gr, T, H, D, base_theta, pos0);

    auto dg = qc::dnew(gs);
    auto dy = qc::dzero<typename S::storage>(count);
    auto make = [&](int lanes) {
        return [&, lanes] {
            if (lanes == 32)
                k_rope_backward<S, 32><<<rows_grid<32>((long long)T * H), kBlock>>>(
                    dg, dy, T, H, D, (float)base_theta, pos0);
            else
                k_rope_backward<S, 64><<<rows_grid<64>((long long)T * H), kBlock>>>(
                    dg, dy, T, H, D, (float)base_theta, pos0);
        };
    };
    char label[192];
    bool ok = true;
    std::snprintf(label, sizeof(label), "rope_backward %s [%s] w32", name, S::name);
    ok &= check_launch<S>(make(32), dy, count, ref, label);
    std::snprintf(label, sizeof(label), "rope_backward %s [%s] w64", name, S::name);
    ok &= check_launch<S>(make(64), dy, count, ref, label);
    if (bench && ok && out) {
        out->baseline = qc::bench(make(32), 25, 50);
        out->candidate = qc::bench(make(64), 25, 50);
        out->timed = true;
    }
    qc::dfree(dg, dy);
    return ok;
}

template <typename S>
static bool op_qk_norm_rope(int T, int Hq, int Hk, int Hv, int D, int rotary_dim,
                            bool interleaved, float weight_offset, bool split_outputs,
                            const char *name, bool bench, OpResult *out) {
    qc::Rng rng(11006 + D + T + Hq);
    const int total_heads = Hq + Hk + Hv;
    const size_t count = (size_t)T * total_heads * D;
    const int max_position = 512, pairs = rotary_dim / 2;
    const float eps = 1e-6f;
    std::vector<float> cosine, sine;
    build_table(cosine, sine, max_position, pairs, 10000.0);
    auto qkvf = rng.normals(count, 0.8f);
    auto qw = rng.normals(D, 0.1f);
    auto kw = rng.normals(D, 0.1f);
    for (auto &v : qw) v += 1.0f;
    for (auto &v : kw) v += 1.0f;
    auto positions = rng.integers(T, 0, max_position - 1);

    auto qkvs = qc::to_storage<typename S::storage>(qkvf);
    std::vector<float> qkvr(count);
    for (size_t i = 0; i < count; ++i) qkvr[i] = (float)qc::to_double(qkvs[i]);
    auto ref = oracle_qk_norm_rope(qkvr, qw, kw, cosine, sine, positions, T, Hq, Hk, Hv, D,
                                   rotary_dim, eps, interleaved, weight_offset);

    auto dqkv = qc::dnew(qkvs);
    auto dqw = qc::dnew(qw);
    auto dkw = qc::dnew(kw);
    auto dc = qc::dnew(cosine);
    auto ds = qc::dnew(sine);
    auto dp = qc::dnew(positions);
    auto dy = qc::dzero<typename S::storage>(count);
    // Split outputs write three buffers; they are re-packed for comparison.
    auto dq = qc::dzero<typename S::storage>((size_t)T * Hq * D);
    auto dk = qc::dzero<typename S::storage>((size_t)T * Hk * D);
    auto dv = qc::dzero<typename S::storage>((size_t)T * Hv * D);
    const long long rows = (long long)T * total_heads;

    auto make = [&](int lanes) {
        return [&, lanes] {
            if (lanes == 32)
                k_qk_norm_rope<S, 32><<<rows_grid<32>(rows), kBlock>>>(
                    dqkv, dqw, dkw, dc, ds, dp, dy, dq, dk, dv, T, Hq, Hk, Hv, D,
                    rotary_dim, eps, interleaved, weight_offset, split_outputs);
            else
                k_qk_norm_rope<S, 64><<<rows_grid<64>(rows), kBlock>>>(
                    dqkv, dqw, dkw, dc, ds, dp, dy, dq, dk, dv, T, Hq, Hk, Hv, D,
                    rotary_dim, eps, interleaved, weight_offset, split_outputs);
        };
    };

    char label[192];
    bool ok = true;
    const char *op = split_outputs ? "qk_norm_rope_split" : "qk_norm_rope_positioned";
    for (int lanes : {32, 64}) {
        QC_CHECK(hipMemset(dy, 0, count * sizeof(typename S::storage)));
        make(lanes)();
        QC_SYNC();
        std::vector<typename S::storage> got;
        if (split_outputs) {
            // Re-pack q/k/v back into the reference's [T, total_heads, D] order.
            auto hq = qc::d2h(dq, (size_t)T * Hq * D);
            auto hk = qc::d2h(dk, (size_t)T * Hk * D);
            auto hv = qc::d2h(dv, (size_t)T * Hv * D);
            got.resize(count);
            for (int t = 0; t < T; ++t) {
                std::copy_n(hq.begin() + (size_t)t * Hq * D, (size_t)Hq * D,
                            got.begin() + (size_t)t * total_heads * D);
                std::copy_n(hk.begin() + (size_t)t * Hk * D, (size_t)Hk * D,
                            got.begin() + (size_t)t * total_heads * D + (size_t)Hq * D);
                std::copy_n(hv.begin() + (size_t)t * Hv * D, (size_t)Hv * D,
                            got.begin() + (size_t)t * total_heads * D +
                                (size_t)(Hq + Hk) * D);
            }
        } else {
            got = qc::d2h(dy, count);
        }
        std::snprintf(label, sizeof(label), "%s %s [%s] w%d", op, name, S::name, lanes);
        ok &= qc::compare(got, ref, qc::tol_for<S>()).report(label);
    }
    if (bench && ok && out) {
        out->baseline = qc::bench(make(32), 25, 50);
        out->candidate = qc::bench(make(64), 25, 50);
        out->timed = true;
    }
    qc::dfree(dqkv, dy, dq, dk, dv);
    qc::dfree(dqw, dkw, dc, ds);
    qc::dfree(dp);
    return ok;
}

template <typename S>
static bool op_qk_norm_rope_mrope(int T, int Hq, int Hk, int Hv, int D, int rotary_dim,
                                  bool section_interleaved, const char *name) {
    qc::Rng rng(11007 + D + T);
    const int total_heads = Hq + Hk + Hv;
    const size_t count = (size_t)T * total_heads * D;
    const int max_position = 512, pairs = rotary_dim / 2;
    const float eps = 1e-6f, weight_offset = 0.0f;
    std::vector<float> cosine, sine;
    build_table(cosine, sine, max_position, pairs, 10000.0);
    auto qkvf = rng.normals(count, 0.8f);
    auto qw = rng.normals(D, 0.1f);
    auto kw = rng.normals(D, 0.1f);
    for (auto &v : qw) v += 1.0f;
    for (auto &v : kw) v += 1.0f;
    auto positions = rng.integers(3 * T, 0, max_position - 1);
    std::vector<int32_t> sections(3);
    sections[0] = (pairs + 2) / 3;
    sections[1] = (pairs + 1) / 3;
    sections[2] = pairs - sections[0] - sections[1];

    auto qkvs = qc::to_storage<typename S::storage>(qkvf);
    std::vector<float> qkvr(count);
    for (size_t i = 0; i < count; ++i) qkvr[i] = (float)qc::to_double(qkvs[i]);

    // Oracle: same as oracle_qk_norm_rope but with the 3-axis table selection.
    std::vector<double> ref(count);
    for (int token = 0; token < T; ++token)
        for (int head = 0; head < total_heads; ++head) {
            const long long base = ((long long)token * total_heads + head) * D;
            if (head >= Hq + Hk) {
                for (int d = 0; d < D; ++d) ref[base + d] = qkvr[base + d];
                continue;
            }
            double squares = 0.0;
            for (int d = 0; d < D; ++d) squares += (double)qkvr[base + d] * qkvr[base + d];
            const double inverse = 1.0 / std::sqrt(squares / D + (double)eps);
            const std::vector<float> &w = head < Hq ? qw : kw;
            const int selected[3] = {positions[token], positions[T + token],
                                     positions[2 * T + token]};
            for (int pair = 0; pair < pairs; ++pair) {
                const int axis =
                    section_interleaved
                        ? pair % 3
                        : (pair < sections[0] ? 0
                                              : (pair < sections[0] + sections[1] ? 1 : 2));
                const long long table = (long long)selected[axis] * pairs + pair;
                const double a = qkvr[base + pair] * inverse * (w[pair] + weight_offset);
                const double b =
                    qkvr[base + pairs + pair] * inverse * (w[pairs + pair] + weight_offset);
                ref[base + pair] = a * cosine[table] - b * sine[table];
                ref[base + pairs + pair] = b * cosine[table] + a * sine[table];
            }
            for (int d = rotary_dim; d < D; ++d)
                ref[base + d] = qkvr[base + d] * inverse * (w[d] + weight_offset);
        }

    auto dqkv = qc::dnew(qkvs);
    auto dqw = qc::dnew(qw);
    auto dkw = qc::dnew(kw);
    auto dc = qc::dnew(cosine);
    auto ds = qc::dnew(sine);
    auto dp = qc::dnew(positions);
    auto dsec = qc::dnew(sections);
    auto dy = qc::dzero<typename S::storage>(count);
    const long long rows = (long long)T * total_heads;

    char label[192];
    bool ok = true;
    for (int lanes : {32, 64}) {
        QC_CHECK(hipMemset(dy, 0, count * sizeof(typename S::storage)));
        if (lanes == 32)
            k_qk_norm_rope_mrope<S, 32><<<rows_grid<32>(rows), kBlock>>>(
                dqkv, dqw, dkw, dc, ds, dp, dsec, dy, T, Hq, Hk, Hv, D, rotary_dim, eps,
                weight_offset, section_interleaved);
        else
            k_qk_norm_rope_mrope<S, 64><<<rows_grid<64>(rows), kBlock>>>(
                dqkv, dqw, dkw, dc, ds, dp, dsec, dy, T, Hq, Hk, Hv, D, rotary_dim, eps,
                weight_offset, section_interleaved);
        QC_SYNC();
        std::snprintf(label, sizeof(label), "qk_norm_rope_positioned mrope %s [%s] w%d",
                      name, S::name, lanes);
        ok &= qc::compare(qc::d2h(dy, count), ref, qc::tol_for<S>()).report(label);
    }
    qc::dfree(dqkv, dy);
    qc::dfree(dqw, dkw, dc, ds);
    qc::dfree(dp, dsec);
    return ok;
}

template <typename S>
static bool op_rope_q_norm(int T, int H, int D, bool do_norm, bool gemma, const char *name,
                           bool bench, OpResult *out) {
    qc::Rng rng(11008 + D + T);
    const size_t count = (size_t)T * H * D;
    const int max_position = 1024, half = D / 2;
    const float eps = 1e-6f;
    std::vector<float> cosine, sine;
    build_table(cosine, sine, max_position, half, 10000.0);
    auto qf = rng.normals(count, 0.8f);
    auto nw = rng.normals(D, 0.1f);
    if (!gemma) for (auto &v : nw) v += 1.0f;
    auto positions = rng.integers(T, 0, max_position - 1);

    auto qs = qc::to_storage<typename S::storage>(qf);
    std::vector<float> qr(count);
    for (size_t i = 0; i < count; ++i) qr[i] = (float)qc::to_double(qs[i]);
    auto ref = oracle_rope_q_norm(qr, cosine, sine, positions, nw, T, H, D, do_norm,
                                  gemma ? 1.0f : 0.0f, eps);

    auto dq = qc::dnew(qs);
    auto dc = qc::dnew(cosine);
    auto ds = qc::dnew(sine);
    auto dp = qc::dnew(positions);
    auto dnw = qc::dnew(nw);
    auto dy = qc::dzero<typename S::storage>(count);

    auto make = [&](int lanes) {
        return [&, lanes] {
            if (lanes == 32)
                k_rope_q_norm<S, 32><<<rows_grid<32>((long long)T * H), kBlock>>>(
                    dq, dc, ds, dp, dnw, dy, T, H, D, do_norm, gemma ? 1.0f : 0.0f, eps);
            else
                k_rope_q_norm<S, 64><<<rows_grid<64>((long long)T * H), kBlock>>>(
                    dq, dc, ds, dp, dnw, dy, T, H, D, do_norm, gemma ? 1.0f : 0.0f, eps);
        };
    };
    char label[192];
    bool ok = true;
    std::snprintf(label, sizeof(label), "rope_q_norm %s [%s] w32", name, S::name);
    ok &= check_launch<S>(make(32), dy, count, ref, label);
    std::snprintf(label, sizeof(label), "rope_q_norm %s [%s] w64", name, S::name);
    ok &= check_launch<S>(make(64), dy, count, ref, label);
    if (bench && ok && out) {
        out->baseline = qc::bench(make(32), 25, 50);
        out->candidate = qc::bench(make(64), 25, 50);
        out->timed = true;
    }
    qc::dfree(dq, dy);
    qc::dfree(dc, ds, dnw);
    qc::dfree(dp);
    return ok;
}

static void report_ab(const char *op, const char *shape, const OpResult &r, double bytes) {
    if (!r.timed) return;
    char label[160];
    std::snprintf(label, sizeof(label), "%s %s w32", op, shape);
    r.baseline.report_bandwidth(label, bytes);
    std::snprintf(label, sizeof(label), "%s %s w64", op, shape);
    r.candidate.report_bandwidth(label, bytes);
    std::snprintf(label, sizeof(label), "%s %s", op, shape);
    qc::report_ab(label, r.baseline, r.candidate);
}

int main(int argc, char **argv) {
    const bool do_bench = qc::bench_requested(argc, argv);
    qc::print_environment("rope_variants (CDNA3 gfx942)");
    bool ok = true;
    OpResult r;

    std::printf("correctness (fp64 oracles = QuixiCore-CPU rotary references):\n");

    // 1. rotary_positioned -- split/interleaved, partial rotary, per-batch positions
    ok &= op_rotary_positioned<StorageF32>(2, 8, 37, 128, 128, false, false, "split", false, nullptr);
    ok &= op_rotary_positioned<StorageF32>(2, 8, 37, 128, 128, true, false, "interleaved", false, nullptr);
    ok &= op_rotary_positioned<StorageF32>(2, 8, 37, 128, 64, false, false, "partial rd=64", false, nullptr);
    ok &= op_rotary_positioned<StorageF32>(3, 4, 16, 64, 64, false, true, "per-batch pos", false, nullptr);
    ok &= op_rotary_positioned<StorageBf16>(2, 4, 16, 128, 128, false, false, "split", false, nullptr);
    ok &= op_rotary_positioned<StorageFp16>(2, 4, 16, 128, 96, false, false, "partial rd=96", false, nullptr);

    // 2. mrope -- sectioned and section-interleaved, partial rotary
    ok &= op_mrope<StorageF32>(2, 8, 37, 128, 128, false, false, "sections", false, nullptr);
    ok &= op_mrope<StorageF32>(2, 8, 37, 128, 128, true, false, "interleaved sections", false, nullptr);
    ok &= op_mrope<StorageF32>(2, 4, 16, 128, 64, false, true, "partial + per-batch", false, nullptr);
    ok &= op_mrope<StorageBf16>(2, 4, 16, 128, 128, false, false, "sections", false, nullptr);

    // 3. rope_table
    ok &= op_rope_table<StorageF32>(101, 8, 128, false, "split", false, nullptr);
    ok &= op_rope_table<StorageF32>(101, 8, 128, true, "interleaved", false, nullptr);
    ok &= op_rope_table<StorageBf16>(64, 4, 64, false, "split", false, nullptr);

    // 4. rope_interleaved_to_split -- nonzero pos0 exercises the position offset
    ok &= op_interleaved_to_split<StorageF32>(101, 8, 128, 0, "pos0=0", false, nullptr);
    ok &= op_interleaved_to_split<StorageF32>(64, 8, 64, 1024, "pos0=1024", false, nullptr);
    ok &= op_interleaved_to_split<StorageBf16>(64, 4, 128, 7, "pos0=7", false, nullptr);

    // 5. rope_backward
    ok &= op_rope_backward<StorageF32>(101, 8, 128, 0, "pos0=0", false, nullptr);
    ok &= op_rope_backward<StorageF32>(64, 8, 64, 512, "pos0=512", false, nullptr);
    ok &= op_rope_backward<StorageBf16>(64, 4, 128, 3, "pos0=3", false, nullptr);

    // 6. qk_norm_rope_positioned -- GQA, partial rotary, Gemma offset, mrope
    ok &= op_qk_norm_rope<StorageF32>(37, 8, 2, 2, 128, 128, false, 0.0f, false, "GQA", false, nullptr);
    ok &= op_qk_norm_rope<StorageF32>(37, 8, 2, 2, 128, 128, true, 0.0f, false, "interleaved", false, nullptr);
    ok &= op_qk_norm_rope<StorageF32>(16, 4, 4, 0, 128, 64, false, 0.0f, false, "partial, no V", false, nullptr);
    ok &= op_qk_norm_rope<StorageF32>(16, 4, 2, 2, 64, 64, false, 1.0f, false, "gemma offset", false, nullptr);
    ok &= op_qk_norm_rope_mrope<StorageF32>(16, 4, 2, 2, 128, 128, false, "sections");
    ok &= op_qk_norm_rope_mrope<StorageF32>(16, 4, 2, 2, 128, 128, true, "interleaved sections");
    ok &= op_qk_norm_rope<StorageBf16>(16, 4, 2, 2, 128, 128, false, 0.0f, false, "GQA", false, nullptr);

    // 7. qk_norm_rope_split -- same math, three output buffers
    ok &= op_qk_norm_rope<StorageF32>(37, 8, 2, 2, 128, 128, false, 0.0f, true, "GQA", false, nullptr);
    ok &= op_qk_norm_rope<StorageF32>(16, 4, 2, 2, 64, 64, false, 1.0f, true, "gemma weight", false, nullptr);
    ok &= op_qk_norm_rope<StorageBf16>(16, 4, 2, 2, 128, 128, false, 0.0f, true, "GQA", false, nullptr);

    // 8. rope_q_norm -- norm on/off, Gemma weight convention
    ok &= op_rope_q_norm<StorageF32>(101, 8, 128, true, false, "norm", false, nullptr);
    ok &= op_rope_q_norm<StorageF32>(101, 8, 128, false, false, "no norm", false, nullptr);
    ok &= op_rope_q_norm<StorageF32>(64, 4, 64, true, true, "gemma weight", false, nullptr);
    ok &= op_rope_q_norm<StorageBf16>(64, 4, 128, true, false, "norm", false, nullptr);

    if (do_bench && ok) {
        std::printf("\nA/B 32 lanes/row (ported shape) vs 64 lanes/row (full wavefront)\n"
                    "(HIP-event median, warmup 10 / iters 50, fp32):\n");
        const int B = 8, H = 32, T = 2048, D = 128;
        const double bytes = 2.0 * (double)B * H * T * D * sizeof(float);
        const double bytes_th = 2.0 * (double)(T * 8) * H * D * sizeof(float);

        ok &= op_rotary_positioned<StorageF32>(B, H, T, D, D, false, false, "B8H32T2048D128", true, &r);
        report_ab("rotary_positioned", "B8H32T2048D128", r, bytes);
        ok &= op_mrope<StorageF32>(B, H, T, D, D, false, false, "B8H32T2048D128", true, &r);
        report_ab("mrope", "B8H32T2048D128", r, bytes);
        ok &= op_rope_table<StorageF32>(T * 8, H, D, false, "T16384H32D128", true, &r);
        report_ab("rope_table", "T16384H32D128", r, bytes_th);
        ok &= op_interleaved_to_split<StorageF32>(T * 8, H, D, 0, "T16384H32D128", true, &r);
        report_ab("rope_interleaved_to_split", "T16384H32D128", r, bytes_th);
        ok &= op_rope_backward<StorageF32>(T * 8, H, D, 0, "T16384H32D128", true, &r);
        report_ab("rope_backward", "T16384H32D128", r, bytes_th);
        ok &= op_qk_norm_rope<StorageF32>(T * 4, 32, 8, 8, D, D, false, 0.0f, false, "T8192Hq32Hk8Hv8D128", true, &r);
        report_ab("qk_norm_rope_positioned", "T8192Hq32Hk8Hv8D128", r,
                  2.0 * (double)(T * 4) * 48 * D * sizeof(float));
        ok &= op_qk_norm_rope<StorageF32>(T * 4, 32, 8, 8, D, D, false, 0.0f, true, "T8192Hq32Hk8Hv8D128", true, &r);
        report_ab("qk_norm_rope_split", "T8192Hq32Hk8Hv8D128", r,
                  2.0 * (double)(T * 4) * 48 * D * sizeof(float));
        ok &= op_rope_q_norm<StorageF32>(T * 8, H, D, true, false, "T16384H32D128", true, &r);
        report_ab("rope_q_norm", "T16384H32D128", r, bytes_th);
    }

    return qc::finish(ok);
}
