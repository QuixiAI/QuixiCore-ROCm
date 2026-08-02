#include "hip/hip_runtime.h"
/**
 * @file
 * @brief CDNA3 (gfx942) decode attention directly against an MXFP8 paged cache.
 *
 * Semantic source: ../QuixiCore-CPU/kernels/attention/attention_mxfp8.cpp
 * (paged_attention_mxfp8). Cache layout and codec live in mxfp8_common.cuh.
 *
 * ## The softmax is ONLINE and tiled at 16, not 32
 *
 * The sibling q8_0 kernel tiles at 32. This one tiles at 16 and the tile size is
 * observable: each tile takes its own max, rescales the running accumulator
 * once, then adds all of that tile's weighted values. Changing the tile changes
 * the floating-point answer (it is still a valid softmax, just a different one),
 * so 16 is reproduced exactly.
 *
 *   next_max   = max(running_max, tile_max)
 *   old_weight = denominator > 0 ? exp(running_max - next_max) : 0
 *   denominator *= old_weight;  output *= old_weight   (skipped when == 1)
 *   for each key in tile: w = exp(score - next_max); denom += w; output += w*V
 *
 * Note `old_weight` is 0 -- not 1 -- on the first tile, which zeroes an
 * already-zero accumulator. Reproduced rather than "fixed".
 *
 * ## Dot product grouping is part of the contract
 *
 * The reference accumulates each 32-element MX group into a double `group_dot`,
 * then does `dot += group_scale * group_dot`. Folding the scale into each
 * element instead is algebraically identical and numerically different. The
 * grouping is kept: lanes stride the head dimension holding one partial per
 * group, the partials are wave-reduced, and only then are the group scales
 * applied.
 *
 * Lane-strided reduction does reorder the summation within a group relative to
 * the CPU's sequential loop, so this is checked against a host replica with a
 * tolerance, not bit-exactly -- unlike the scatter/gather codec, which is
 * byte-exact.
 *
 * ## Shape rules from the reference
 *
 *   head_dim is 64 or 128 only; query_heads % kv_heads == 0 (GQA);
 *   scale <= 0 means 1/sqrt(head_dim); window > 0 limits to the last `window`
 *   positions; row = block_table[req][pos/page_size]*page_size + pos%page_size.
 *
 * Build: make      Test: make test
 */
#include "mxfp8_common.cuh"
#include <cstdio>
#include <vector>
#include <random>
#include <algorithm>
#include <limits>

namespace mxkv {

constexpr int kScoreTile = 16;   // matches the reference; see header
constexpr int kMaxGroups = 4;    // head_dim <= 128 => head_dim/32 <= 4

__device__ __forceinline__ float wave_sum(float v) {
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) v += __shfl_xor(v, off);
    return v;
}

// One 64-lane wavefront per (request, query head).
__global__ void k_mx_paged_attention(const float *__restrict__ q,
                                     const uint8_t *__restrict__ key_cache,
                                     const uint8_t *__restrict__ value_cache,
                                     const int *__restrict__ block_table,
                                     const int *__restrict__ context_lens,
                                     float *__restrict__ out,
                                     int batch, int query_heads, int kv_heads,
                                     int head_dim, int page_size, int max_blocks,
                                     float scale, int window) {
    const int item = blockIdx.x;
    if (item >= batch * query_heads) return;
    const int lane = threadIdx.x;
    const int request = item / query_heads;
    const int qhead = item % query_heads;
    const int kvhead = qhead / (query_heads / kv_heads);
    const int context = context_lens[request];
    const int start = window > 0 ? max(0, context - window) : 0;
    const int groups = head_dim / kMxGroup;
    const float score_scale = scale > 0.0f ? scale : rsqrtf((float)head_dim);

    const float *query = q + (long long)item * head_dim;
    float *output = out + (long long)item * head_dim;

    // Each lane owns head_dim/64 output elements (1 for 64, 2 for 128).
    float acc[2] = {0.0f, 0.0f};
    const int per_lane = head_dim / 64;

    float running_max = -INFINITY;
    double denominator = 0.0;

    __shared__ float s_scores[kScoreTile];
    __shared__ int s_rows[kScoreTile];

    for (int tile = start; tile < context; tile += kScoreTile) {
        const int tile_end = min(context, tile + kScoreTile);
        const int valid = tile_end - tile;

        // ---- scores for this tile ----
        for (int idx = 0; idx < valid; ++idx) {
            const int position = tile + idx;
            const int physical = block_table[(long long)request * max_blocks + position / page_size];
            const long long row = (long long)physical * page_size + position % page_size;

            double partial[kMaxGroups];
            #pragma unroll
            for (int g = 0; g < kMaxGroups; ++g) partial[g] = 0.0;

            for (int d = lane; d < head_dim; d += 64) {
                const int g = d / kMxGroup;
                const long long base = mx_group_base(row, kvhead, g, kv_heads, groups);
                partial[g] += (double)query[d] * (double)e4m3fn_decode(key_cache[base + 1 + (d % kMxGroup)]);
            }
            double dot = 0.0;
            for (int g = 0; g < groups; ++g) {
                const float summed = wave_sum((float)partial[g]);
                const long long base = mx_group_base(row, kvhead, g, kv_heads, groups);
                dot += (double)e8m0_decode_pow2(key_cache[base]) * (double)summed;
            }
            if (lane == 0) {
                s_scores[idx] = (float)(dot * score_scale);
                s_rows[idx] = (int)row;
            }
        }
        __syncthreads();

        float tile_max = -INFINITY;
        for (int idx = 0; idx < valid; ++idx) tile_max = fmaxf(tile_max, s_scores[idx]);
        const float next_max = fmaxf(running_max, tile_max);
        const double old_weight = denominator > 0.0 ? exp((double)running_max - (double)next_max) : 0.0;
        denominator *= old_weight;
        if (old_weight != 1.0) {
            #pragma unroll
            for (int r = 0; r < 2; ++r) if (r < per_lane) acc[r] *= (float)old_weight;
        }

        for (int idx = 0; idx < valid; ++idx) {
            const double weight = exp((double)s_scores[idx] - (double)next_max);
            denominator += weight;
            const long long row = s_rows[idx];
            #pragma unroll
            for (int r = 0; r < 2; ++r) {
                if (r >= per_lane) continue;
                const int d = lane + r * 64;
                const int g = d / kMxGroup;
                const long long base = mx_group_base(row, kvhead, g, kv_heads, groups);
                const float scaled_weight = (float)(weight * (double)e8m0_decode_pow2(value_cache[base]));
                acc[r] += scaled_weight * e4m3fn_decode(value_cache[base + 1 + (d % kMxGroup)]);
            }
        }
        running_max = next_max;
        __syncthreads();
    }

    if (denominator > 0.0) {
        const float inverse = (float)(1.0 / denominator);
        #pragma unroll
        for (int r = 0; r < 2; ++r) if (r < per_lane) acc[r] *= inverse;
    }
    #pragma unroll
    for (int r = 0; r < 2; ++r) if (r < per_lane) output[lane + r * 64] = acc[r];
}

}  // namespace mxkv

// ===========================================================================
// self-checking harness
// ===========================================================================
using namespace mxkv;

#define CK(x) do { hipError_t e=(x); if(e){printf("HIP %s @%d\n",hipGetErrorString(e),__LINE__);exit(1);} } while(0)
static int g_fail = 0;
static void report(const char *n, bool ok, const char *d = "") {
    printf("%-44s %s %s\n", n, ok ? "PASS" : "FAIL", d);
    if (!ok) ++g_fail;
}

int main() {
    const int batch = 5, query_heads = 8, kv_heads = 2, head_dim = 128;
    const int page_size = 16, cache_blocks = 64, max_blocks = 8;
    const int groups = head_dim / kMxGroup;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> uf(-1.5f, 1.5f);

    std::vector<float> q((size_t)batch * query_heads * head_dim);
    for (auto &v : q) v = uf(rng);
    std::vector<int> ctx(batch), bt((size_t)batch * max_blocks);
    for (int b = 0; b < batch; ++b) ctx[b] = 20 + b * 17;             // spans tiles unevenly
    for (size_t i = 0; i < bt.size(); ++i) bt[i] = (int)(i * 3 % cache_blocks);

    const size_t cache_bytes = (size_t)cache_blocks * page_size * kv_heads * groups * kMxBlockBytes;
    std::vector<uint8_t> kc(cache_bytes), vc(cache_bytes);
    std::uniform_int_distribution<int> byte(0, 254);
    for (size_t i = 0; i < cache_bytes; ++i) {
        // keep scale bytes in a sane exponent range so scores don't overflow
        const bool is_scale = (i % kMxBlockBytes) == 0;
        // 0x7f / 0xff are E4M3FN NaN. The encoder only emits them for NaN
        // input, so a cache containing them is not reachable in practice --
        // and a NaN here poisons every comparison below (see the finite check).
        auto safe_code = [&]{ uint8_t c; do { c = (uint8_t)byte(rng); }
                              while ((c & 0x7f) == 0x7f); return c; };
        kc[i] = is_scale ? (uint8_t)(120 + (i % 12)) : safe_code();
        vc[i] = is_scale ? (uint8_t)(120 + (i % 9)) : safe_code();
    }

    float *dq, *dout; uint8_t *dkc, *dvc; int *dbt, *dctx;
    CK(hipMalloc(&dq, q.size()*4));       CK(hipMemcpy(dq, q.data(), q.size()*4, hipMemcpyHostToDevice));
    CK(hipMalloc(&dkc, cache_bytes));     CK(hipMemcpy(dkc, kc.data(), cache_bytes, hipMemcpyHostToDevice));
    CK(hipMalloc(&dvc, cache_bytes));     CK(hipMemcpy(dvc, vc.data(), cache_bytes, hipMemcpyHostToDevice));
    CK(hipMalloc(&dbt, bt.size()*4));     CK(hipMemcpy(dbt, bt.data(), bt.size()*4, hipMemcpyHostToDevice));
    CK(hipMalloc(&dctx, batch*4));        CK(hipMemcpy(dctx, ctx.data(), batch*4, hipMemcpyHostToDevice));
    CK(hipMalloc(&dout, q.size()*4));

    for (int window : {0, 24}) {
        k_mx_paged_attention<<<batch*query_heads, 64>>>(dq, dkc, dvc, dbt, dctx, dout,
            batch, query_heads, kv_heads, head_dim, page_size, max_blocks, 0.0f, window);
        CK(hipDeviceSynchronize());
        if (hipGetLastError() != hipSuccess) { printf("KERNEL ERR\n"); return 1; }
        std::vector<float> got(q.size());
        CK(hipMemcpy(got.data(), dout, q.size()*4, hipMemcpyDeviceToHost));

        // ---- host replica, mirroring the reference's tiling and grouping ----
        // A non-finite result must fail loudly. std::max(worst, nan) returns
        // worst, so without this an all-NaN output reports worst == 0 and PASSES.
        size_t nonfinite = 0;
        for (size_t i = 0; i < got.size(); ++i)
            if (!std::isfinite(got[i])) ++nonfinite;

        double worst = 0.0;
        const float score_scale = 1.0f / std::sqrt((float)head_dim);
        for (int item = 0; item < batch*query_heads; ++item) {
            const int request = item / query_heads, qhead = item % query_heads;
            const int kvhead = qhead / (query_heads / kv_heads);
            const int context = ctx[request];
            const int start = window > 0 ? std::max(0, context - window) : 0;
            std::vector<float> output(head_dim, 0.0f);
            float maximum = -std::numeric_limits<float>::infinity();
            double denominator = 0.0;
            for (int tile = start; tile < context; tile += kScoreTile) {
                const int tile_end = std::min(context, tile + kScoreTile);
                float tmax = -std::numeric_limits<float>::infinity();
                std::vector<float> sc; std::vector<long long> rws;
                for (int pos = tile; pos < tile_end; ++pos) {
                    const int phys = bt[(size_t)request*max_blocks + pos/page_size];
                    const long long row = (long long)phys*page_size + pos%page_size;
                    double dot = 0.0;
                    for (int g = 0; g < groups; ++g) {
                        const long long base = mx_group_base(row, kvhead, g, kv_heads, groups);
                        double gd = 0.0;
                        for (int d = 0; d < kMxGroup; ++d)
                            gd += (double)q[(size_t)item*head_dim + g*kMxGroup + d] *
                                  (double)e4m3fn_decode(kc[base+1+d]);
                        dot += (double)e8m0_decode_pow2(kc[base]) * gd;
                    }
                    sc.push_back((float)(dot*score_scale)); rws.push_back(row);
                    tmax = std::max(tmax, sc.back());
                }
                const float nmax = std::max(maximum, tmax);
                const double ow = denominator > 0.0 ? std::exp((double)maximum-(double)nmax) : 0.0;
                denominator *= ow;
                if (ow != 1.0) for (int d = 0; d < head_dim; ++d) output[d] *= (float)ow;
                for (size_t k = 0; k < sc.size(); ++k) {
                    const double w = std::exp((double)sc[k]-(double)nmax);
                    denominator += w;
                    for (int g = 0; g < groups; ++g) {
                        const long long base = mx_group_base(rws[k], kvhead, g, kv_heads, groups);
                        const float sw = (float)(w * (double)e8m0_decode_pow2(vc[base]));
                        for (int d = 0; d < kMxGroup; ++d)
                            output[g*kMxGroup+d] += sw * e4m3fn_decode(vc[base+1+d]);
                    }
                }
                maximum = nmax;
            }
            if (denominator > 0.0) { const float inv = (float)(1.0/denominator);
                for (int d = 0; d < head_dim; ++d) output[d] *= inv; }
            for (int d = 0; d < head_dim; ++d) {
                const double ref = output[d];
                const double rel = std::fabs((double)got[(size_t)item*head_dim+d]-ref)
                                   / std::max(1e-3, std::fabs(ref));
                worst = std::max(worst, rel);
            }
        }
        char detail[96];
        snprintf(detail, sizeof detail, "(window=%d, worst rel %.3e, %zu non-finite)",
                 window, worst, nonfinite);
        // lane-strided reduction reorders the within-group sum vs the CPU loop
        report("paged_attention_mxfp8 vs host replica",
               nonfinite == 0 && worst < 2e-3, detail);
    }

    printf("\n%s (%d failures)\n", g_fail ? "FAILED" : "ALL PASS", g_fail);
    return g_fail ? 1 : 0;
}
