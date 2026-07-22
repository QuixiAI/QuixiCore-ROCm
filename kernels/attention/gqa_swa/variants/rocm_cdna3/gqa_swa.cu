/**
 * @file
 * @brief CDNA3 (gfx942) symmetric sliding-window GQA flash attention.
 *
 * Shape-named port of embeddinggemma.c `mfma_attention_f16_kernel`
 * (src/engine_rocm.hip), which the source itself notes was "adapted from
 * QuixiCore's BQ=BK=16 GQA kernel." The specialization this adds over the
 * repo's `gqa`/`gqa_causal` forward is a **symmetric (bidirectional, centered)
 * sliding window**: an encoder-style query at position q attends keys in
 * [q - window/2, q + window/2], with no causal mask. window == 0 recovers full
 * bidirectional attention. GQA with a single shared KV head (H_KV = 1); one
 * wave64 per (16-query block, head) reuses each K/V tile across 16 queries and
 * keeps softmax online. QK^T and P@V both run on v_mfma_f32_16x16x16_f16.
 *
 * Layout: Q/O [T, H*D] (head-major within a token), K/V [T, D] (shared KV head).
 * Q is expected pre-scaled by 1/sqrt(D) (Gemma folds the attn scale into Q).
 */
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <random>

#define CK(x) do { \
    hipError_t e = (x); \
    if (e != hipSuccess) { \
        printf("HIP %s @%d\n", hipGetErrorString(e), __LINE__); \
        exit(1); \
    } \
} while (0)

#ifndef HEAD_DIM
#define HEAD_DIM 256
#endif
#ifndef N_HEAD
#define N_HEAD 3
#endif
#define N_EMBD (N_HEAD * HEAD_DIM)

typedef __attribute__((__vector_size__(4 * sizeof(__fp16)))) __fp16 half4_t;
typedef __attribute__((__vector_size__(4 * sizeof(float)))) float float4_t;

__device__ __forceinline__ float4_t mfma_m16n16k16(
    half4_t a, half4_t b, float4_t acc) {
    return __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, acc, 0, 0, 0);
}

// Symmetric sliding-window GQA (H_KV=1). One wave64 per (query block, head).
__global__ void gqa_swa_kernel(const __half *q_half, const __half *k_half,
                               const __half *v_half, float *output,
                               uint32_t seq_tokens, uint32_t window) {
    constexpr uint32_t query_tile = 16;
    constexpr uint32_t key_tile = 16;
    constexpr int dim_tiles = HEAD_DIM / 16;
    __shared__ float scores[query_tile][key_tile];
    __shared__ __half probabilities[query_tile][key_tile];
    __shared__ float running_max[query_tile];
    __shared__ float running_sum[query_tile];
    __shared__ float correction[query_tile];

    const uint32_t head = blockIdx.y;
    if (head >= N_HEAD) return;
    const int lane = threadIdx.x & 63;
    const int lane_group = lane >> 4;
    const int lane_item = lane & 15;
    const uint32_t query_start = blockIdx.x * query_tile;
    const uint32_t sequence_start = 0;
    const uint32_t sequence_stop = seq_tokens;
    if (query_start >= sequence_stop) return;
    const uint32_t query_stop = min(sequence_stop, query_start + query_tile);
    const __fp16 *q = reinterpret_cast<const __fp16 *>(q_half);
    const __fp16 *k = reinterpret_cast<const __fp16 *>(k_half);
    const __fp16 *v = reinterpret_cast<const __fp16 *>(v_half);

    half4_t query_fragments[dim_tiles];
#pragma unroll
    for (int dim_tile = 0; dim_tile < dim_tiles; dim_tile++) {
        const uint32_t query_index = query_start + lane_item;
        const uint32_t dim = dim_tile * 16 + lane_group * 4;
        if (query_index < query_stop) {
            query_fragments[dim_tile] = *reinterpret_cast<const half4_t *>(
                q + static_cast<size_t>(query_index) * N_EMBD + head * HEAD_DIM + dim);
        } else {
            query_fragments[dim_tile] = half4_t{0, 0, 0, 0};
        }
    }
    float4_t output_accumulators[dim_tiles];
#pragma unroll
    for (int dim_tile = 0; dim_tile < dim_tiles; dim_tile++)
        output_accumulators[dim_tile] = float4_t{0, 0, 0, 0};
    if (lane < static_cast<int>(query_tile)) {
        running_max[lane] = -__int_as_float(0x7f800000);
        running_sum[lane] = 0.0f;
    }
    __syncthreads();

    uint32_t union_first = sequence_start;
    uint32_t union_last = sequence_stop;
    if (window != 0) {
        const uint32_t half_window = window / 2;
        union_first = query_start > sequence_start + half_window
            ? query_start - half_window : sequence_start;
        union_last = min(sequence_stop, query_stop + half_window);
    }

    for (uint32_t key_start = union_first; key_start < union_last; key_start += key_tile) {
        float4_t score = {0, 0, 0, 0};
#pragma unroll
        for (int dim_tile = 0; dim_tile < dim_tiles; dim_tile++) {
            const uint32_t key_index = key_start + lane_item;
            const uint32_t dim = dim_tile * 16 + lane_group * 4;
            half4_t key_fragment = {0, 0, 0, 0};
            if (key_index < union_last) {
                key_fragment = *reinterpret_cast<const half4_t *>(
                    k + static_cast<size_t>(key_index) * HEAD_DIM + dim);
            }
            score = mfma_m16n16k16(query_fragments[dim_tile], key_fragment, score);
        }
#pragma unroll
        for (int item = 0; item < 4; item++)
            scores[4 * lane_group + item][lane_item] = score[item];
        __syncthreads();

        if (lane < static_cast<int>(query_tile)) {
            const uint32_t query_index = query_start + lane;
            if (query_index < query_stop) {
                uint32_t first = sequence_start;
                uint32_t last = sequence_stop;
                if (window != 0) {
                    const uint32_t half_window = window / 2;
                    first = query_index > sequence_start + half_window
                        ? query_index - half_window : sequence_start;
                    last = min(sequence_stop, query_index + half_window + 1);
                }
                const float previous_max = running_max[lane];
                float next_max = previous_max;
#pragma unroll
                for (uint32_t item = 0; item < key_tile; item++) {
                    const uint32_t key_index = key_start + item;
                    if (key_index < union_last && key_index >= first && key_index < last)
                        next_max = fmaxf(next_max, scores[lane][item]);
                }
                const float alpha = __expf(previous_max - next_max);
                float next_sum = running_sum[lane] * alpha;
#pragma unroll
                for (uint32_t item = 0; item < key_tile; item++) {
                    const uint32_t key_index = key_start + item;
                    const bool active = key_index < union_last &&
                        key_index >= first && key_index < last;
                    const float probability = active
                        ? __expf(scores[lane][item] - next_max) : 0.0f;
                    probabilities[lane][item] = __float2half(probability);
                    next_sum += probability;
                }
                running_max[lane] = next_max;
                running_sum[lane] = next_sum;
                correction[lane] = alpha;
            } else {
#pragma unroll
                for (uint32_t item = 0; item < key_tile; item++)
                    probabilities[lane][item] = __float2half(0.0f);
                correction[lane] = 1.0f;
            }
        }
        __syncthreads();

#pragma unroll
        for (int dim_tile = 0; dim_tile < dim_tiles; dim_tile++)
#pragma unroll
            for (int item = 0; item < 4; item++)
                output_accumulators[dim_tile][item] *= correction[4 * lane_group + item];
        const half4_t probability_fragment =
            *reinterpret_cast<const half4_t *>(&probabilities[lane_item][lane_group * 4]);
#pragma unroll
        for (int dim_tile = 0; dim_tile < dim_tiles; dim_tile++) {
            const uint32_t dim = dim_tile * 16 + lane_item;
            half4_t value_fragment = {0, 0, 0, 0};
#pragma unroll
            for (int item = 0; item < 4; item++) {
                const uint32_t key_index = key_start + lane_group * 4 + item;
                if (key_index < union_last)
                    value_fragment[item] = v[static_cast<size_t>(key_index) * HEAD_DIM + dim];
            }
            output_accumulators[dim_tile] = mfma_m16n16k16(
                probability_fragment, value_fragment, output_accumulators[dim_tile]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int dim_tile = 0; dim_tile < dim_tiles; dim_tile++) {
        const uint32_t dim = dim_tile * 16 + lane_item;
#pragma unroll
        for (int item = 0; item < 4; item++) {
            const uint32_t query_index = query_start + 4 * lane_group + item;
            if (query_index < query_stop) {
                const size_t output_index =
                    static_cast<size_t>(query_index) * N_EMBD + head * HEAD_DIM + dim;
                output[output_index] =
                    output_accumulators[dim_tile][item] / running_sum[4 * lane_group + item];
            }
        }
    }
}

template <typename T>
static T* dnew(const std::vector<T>& h) {
    T* d = nullptr;
    CK(hipMalloc(&d, h.size() * sizeof(T)));
    CK(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice));
    return d;
}
template <typename T>
static std::vector<T> d2h(const T* d, size_t n) {
    std::vector<T> h(n);
    CK(hipMemcpy(h.data(), d, n * sizeof(T), hipMemcpyDeviceToHost));
    return h;
}
template <typename FN>
static float bench_ms(FN&& fn, int warmup = 10, int iters = 50) {
    for (int i = 0; i < warmup; ++i) fn();
    CK(hipDeviceSynchronize());
    std::vector<float> t(iters);
    hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    for (int i = 0; i < iters; ++i) {
        CK(hipEventRecord(a)); fn(); CK(hipEventRecord(b));
        CK(hipEventSynchronize(b)); CK(hipEventElapsedTime(&t[i], a, b));
    }
    CK(hipEventDestroy(a)); CK(hipEventDestroy(b));
    std::sort(t.begin(), t.end());
    return t[iters / 2];
}

// fp64 host reference: symmetric-window scaled attention on the fp16 inputs.
static std::vector<float> host_ref(const std::vector<__half>& Q,
                                   const std::vector<__half>& K,
                                   const std::vector<__half>& V,
                                   int T, uint32_t window) {
    std::vector<float> out(static_cast<size_t>(T) * N_EMBD, 0.0f);
    for (int h = 0; h < N_HEAD; ++h) {
        for (int qi = 0; qi < T; ++qi) {
            int first = 0, last = T;
            if (window != 0) {
                const int hw = (int)(window / 2);
                first = std::max(0, qi - hw);
                last = std::min(T, qi + hw + 1);
            }
            double m = -1e30;
            std::vector<double> s(last - first);
            for (int ki = first; ki < last; ++ki) {
                double dot = 0.0;
                for (int d = 0; d < HEAD_DIM; ++d)
                    dot += (double)__half2float(Q[(size_t)qi * N_EMBD + h * HEAD_DIM + d]) *
                           (double)__half2float(K[(size_t)ki * HEAD_DIM + d]);
                s[ki - first] = dot;
                m = std::max(m, dot);
            }
            double denom = 0.0;
            for (double& x : s) { x = std::exp(x - m); denom += x; }
            for (int d = 0; d < HEAD_DIM; ++d) {
                double acc = 0.0;
                for (int ki = first; ki < last; ++ki)
                    acc += s[ki - first] * (double)__half2float(V[(size_t)ki * HEAD_DIM + d]);
                out[(size_t)qi * N_EMBD + h * HEAD_DIM + d] = (float)(acc / denom);
            }
        }
    }
    return out;
}

static int check(const char* name, const std::vector<float>& got,
                 const std::vector<float>& ref, double tol) {
    double sad = 0.0, sref = 0.0, maxe = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double d = std::abs((double)got[i] - (double)ref[i]);
        sad += d; sref += std::abs((double)ref[i]); maxe = std::max(maxe, d);
    }
    const double rel = sad / std::max(sref, 1e-30);
    const bool ok = rel < tol;
    printf("%-26s rel %.3e max %.3e %s\n", name, rel, maxe, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

static int run_case(int T, uint32_t window, bool bench) {
    std::mt19937 rng(19);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const float scale = 1.0f / std::sqrt((float)HEAD_DIM);
    std::vector<__half> Q((size_t)T * N_EMBD), K((size_t)T * HEAD_DIM), V((size_t)T * HEAD_DIM);
    for (auto& v : Q) v = __float2half(nd(rng) * scale);  // fold attn scale into Q
    for (auto& v : K) v = __float2half(nd(rng));
    for (auto& v : V) v = __float2half(nd(rng));
    auto ref = host_ref(Q, K, V, T, window);

    auto dQ = dnew(Q); auto dK = dnew(K); auto dV = dnew(V);
    float* dO = nullptr;
    CK(hipMalloc(&dO, ref.size() * sizeof(float)));
    dim3 grid((T + 15) / 16, N_HEAD);
    auto run = [&] { gqa_swa_kernel<<<grid, 64>>>(dQ, dK, dV, dO, (uint32_t)T, window); };
    run();
    CK(hipDeviceSynchronize()); CK(hipGetLastError());
    char name[64];
    std::snprintf(name, sizeof(name), "gqa_swa T=%d w=%u", T, window);
    int rc = check(name, d2h(dO, ref.size()), ref, 1.5e-2);
    if (bench && rc == 0) {
        const float t = bench_ms(run);
        printf("  %s  %.4f ms\n", name, t);
    }
    CK(hipFree(dQ)); CK(hipFree(dK)); CK(hipFree(dV)); CK(hipFree(dO));
    return rc;
}

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    int rc = 0;
    // Correctness: full vs symmetric window; ragged and aligned T.
    rc |= run_case(37, 0, false);      // full attention, ragged
    rc |= run_case(37, 16, false);     // narrow symmetric window
    rc |= run_case(512, 0, false);     // full
    rc |= run_case(512, 128, false);   // symmetric window band
    rc |= run_case(512, 256, false);
    if (bench) {
        printf("gqa_swa bench (full window=0 vs symmetric window):\n");
        rc |= run_case(2048, 0, true);     // full O(T^2)
        rc |= run_case(2048, 256, true);   // banded
        rc |= run_case(2048, 512, true);
    }
    printf(rc == 0 ? "ALL PASS\n" : "FAIL\n");
    return rc;
}
