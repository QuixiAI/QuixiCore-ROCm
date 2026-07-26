#include "hip/hip_runtime.h"
/**
 * @file
 * @brief Harness for the MoE family (tm_moe_kernels.cuh): runs the FULL
 * pipeline route -> histogram -> scan -> scatter -> pad -> gather ->
 * swiglu grouped GEMM -> rect grouped GEMM (down-proj) -> finalize on random
 * data and compares the end-to-end MoE MLP output against a dense fp64 CPU
 * reference. Also checks every intermediate invariant (counts, offsets,
 * segment membership, inverse maps, pad sentinels, gathered rows).
 *
 * Build:
 *   nvcc -std=c++17 -O2 -gencode arch=compute_86,code=sm_86 \
 *        moe_test.cu -o moe_test.out
 * Run: CUDA_VISIBLE_DEVICES=6 ./moe_test.out
 */
#include "tm_moe_kernels.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <numeric>

using namespace tmoe;

static int g_fail = 0;
#define CUCHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    printf("CUDA error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

template <typename T> T* dnew(const std::vector<T>& h) {
    T* d; CUCHECK(hipMalloc(&d, h.size() * sizeof(T)));
    CUCHECK(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice));
    return d;
}
template <typename T> T* dzero(size_t n) {
    T* d; CUCHECK(hipMalloc(&d, n * sizeof(T)));
    CUCHECK(hipMemset(d, 0, n * sizeof(T)));
    return d;
}
template <typename T> std::vector<T> d2h(const T* d, size_t n) {
    std::vector<T> h(n);
    CUCHECK(hipMemcpy(h.data(), d, n * sizeof(T), hipMemcpyDeviceToHost));
    return h;
}
static void report(const char* name, bool ok, double err = -1.0) {
    if (err >= 0) printf("%-40s %s  (err %.3e)\n", name, ok ? "PASS" : "FAIL", err);
    else printf("%-40s %s\n", name, ok ? "PASS" : "FAIL");
    if (!ok) ++g_fail;
}

static std::mt19937 g_rng(2025);
static std::vector<float> randv(size_t n, float lo = -1.0f, float hi = 1.0f) {
    std::uniform_real_distribution<float> d(lo, hi);
    std::vector<float> v(n);
    for (auto& x : v) x = d(g_rng);
    return v;
}

struct BenchStats {
    float median_ms;
    float min_ms;
    float max_ms;
    int warmups;
    int iters;
    int repeats;
};

template <typename F>
BenchStats bench_kernel(F&& launch, int warmups = 5, int iters = 20,
                        int repeats = 10) {
    hipEvent_t start, stop;
    CUCHECK(hipEventCreate(&start));
    CUCHECK(hipEventCreate(&stop));
    for (int i = 0; i < warmups; ++i)
        for (int r = 0; r < repeats; ++r) launch();
    CUCHECK(hipDeviceSynchronize());
    std::vector<float> samples(iters);
    for (int i = 0; i < iters; ++i) {
        CUCHECK(hipEventRecord(start));
        for (int r = 0; r < repeats; ++r) launch();
        CUCHECK(hipEventRecord(stop));
        CUCHECK(hipEventSynchronize(stop));
        CUCHECK(hipEventElapsedTime(&samples[i], start, stop));
        samples[i] /= float(repeats);
    }
    CUCHECK(hipEventDestroy(start));
    CUCHECK(hipEventDestroy(stop));
    std::sort(samples.begin(), samples.end());
    return {samples[samples.size() / 2], samples.front(), samples.back(),
            warmups, iters, repeats};
}

static void report_bench(const char* base_name, const BenchStats& base,
                         const char* cand_name, const BenchStats& cand,
                         double work_items) {
    const double base_rate = work_items / (double(base.median_ms) * 1e-3);
    const double cand_rate = work_items / (double(cand.median_ms) * 1e-3);
    printf("  %-34s %8.4f ms  %.2f Gitem/s (min %.4f max %.4f spread %.2fx, w%d/i%d/r%d)\n",
           base_name, base.median_ms, base_rate / 1e9, base.min_ms, base.max_ms,
           base.max_ms / std::max(base.min_ms, 1e-9f), base.warmups, base.iters,
           base.repeats);
    printf("  %-34s %8.4f ms  %.2f Gitem/s (min %.4f max %.4f spread %.2fx, w%d/i%d/r%d)\n",
           cand_name, cand.median_ms, cand_rate / 1e9, cand.min_ms, cand.max_ms,
           cand.max_ms / std::max(cand.min_ms, 1e-9f), cand.warmups, cand.iters,
           cand.repeats);
    printf("  speedup %.2fx  keep=%s\n", base.median_ms / cand.median_ms,
           cand.median_ms <= base.median_ms ? cand_name : base_name);
}

__device__ __forceinline__ float bench_score(float v, int scoring) {
    if (scoring == MOE_SCORE_SOFTMAX) return expf(v);
    if (scoring == MOE_SCORE_SIGMOID) return 1.0f / (1.0f + expf(-v));
    return sqrtf(log1pf(expf(-fabsf(v))) + fmaxf(v, 0.0f));
}

__global__ void moe_route_grouped_scalar(const float* __restrict__ logits,
                                         const float* __restrict__ bias,
                                         int* __restrict__ topk_ids,
                                         float* __restrict__ topk_weights,
                                         int tokens, int E, int K, int groups,
                                         int top_groups, int renormalize,
                                         float routed_scale, int scoring) {
    constexpr int MAX_E = 128;
    constexpr int MAX_G = 64;
    constexpr int MAX_K = MOE_MAX_K;
    const int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= tokens || E > MAX_E || groups > MAX_G || K > MAX_K) return;
    float score[MAX_E];
    float select[MAX_E];
    const float* row = logits + (long)token * E;
    if (scoring == MOE_SCORE_SOFTMAX) {
        float m = MOE_NEG_INF;
        for (int e = 0; e < E; ++e) m = fmaxf(m, row[e]);
        float den = 0.0f;
        for (int e = 0; e < E; ++e) {
            score[e] = expf(row[e] - m);
            den += score[e];
        }
        for (int e = 0; e < E; ++e) score[e] /= den;
    } else {
        for (int e = 0; e < E; ++e) score[e] = bench_score(row[e], scoring);
    }
    for (int e = 0; e < E; ++e) select[e] = score[e] + (bias ? bias[e] : 0.0f);

    float group_score[MAX_G];
    bool keep[MAX_G];
    const int gsz = E / groups;
    for (int g = 0; g < groups; ++g) {
        keep[g] = false;
        float first = MOE_NEG_INF;
        float second = MOE_NEG_INF;
        for (int i = 0; i < gsz; ++i) {
            const float v = select[g * gsz + i];
            if (v > first) { second = first; first = v; }
            else if (v > second) { second = v; }
        }
        group_score[g] = first + (gsz > 1 ? second : 0.0f);
    }
    for (int r = 0; r < top_groups; ++r) {
        int best = -1;
        for (int g = 0; g < groups; ++g) {
            if (keep[g]) continue;
            if (best < 0 || group_score[g] > group_score[best]) best = g;
        }
        keep[best] = true;
    }

    int ids[MAX_K];
    float vals[MAX_K];
    for (int k = 0; k < K; ++k) { ids[k] = -1; vals[k] = MOE_NEG_INF; }
    for (int e = 0; e < E; ++e) {
        if (!keep[e / gsz]) continue;
        const float v = select[e];
        for (int k = 0; k < K; ++k) {
            if (ids[k] < 0 || v > vals[k] || (v == vals[k] && e < ids[k])) {
                for (int j = K - 1; j > k; --j) { ids[j] = ids[j - 1]; vals[j] = vals[j - 1]; }
                ids[k] = e;
                vals[k] = v;
                break;
            }
        }
    }
    float den = 0.0f;
    for (int k = 0; k < K; ++k) den += score[ids[k]];
    const float scale = renormalize ? routed_scale / den : routed_scale;
    for (int k = 0; k < K; ++k) {
        topk_ids[(long)token * K + k] = ids[k];
        topk_weights[(long)token * K + k] = score[ids[k]] * scale;
    }
}

__global__ void moe_gather_backward_scalar(const float* __restrict__ grad_gathered,
                                           const int* __restrict__ gather_rows,
                                           float* __restrict__ grad_input,
                                           int gathered_rows, int dim) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= gathered_rows) return;
    const int src = gather_rows[r];
    if (src < 0) return;
    for (int i = 0; i < dim; ++i)
        atomicAdd(&grad_input[(long)src * dim + i], grad_gathered[(long)r * dim + i]);
}

__global__ void moe_finalize_backward_scalar(const float* __restrict__ grad_out,
                                             const float* __restrict__ expert_out,
                                             const int* __restrict__ inverse,
                                             const float* __restrict__ expert_weights,
                                             float* __restrict__ grad_expert_out,
                                             float* __restrict__ grad_expert_weights,
                                             int tokens, int top_k, int dim) {
    const int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= tokens) return;
    for (int k = 0; k < top_k; ++k) {
        const long route = (long)token * top_k + k;
        const int p = inverse[route];
        if (p < 0) continue;
        const float w = expert_weights[route];
        float wg = 0.0f;
        for (int i = 0; i < dim; ++i) {
            const float g = grad_out[(long)token * dim + i];
            atomicAdd(&grad_expert_out[(long)p * dim + i], w * g);
            wg += g * expert_out[(long)p * dim + i];
        }
        grad_expert_weights[route] = wg;
    }
}

__global__ void moe_grouped_gemm_backward_input_scalar(
    const float* __restrict__ grad_out, const float* __restrict__ W,
    const int* __restrict__ expert_ids, float* __restrict__ grad_input,
    int rows, int K_in, int N_out) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const int e = expert_ids[r];
    if (e < 0) return;
    const float* We = W + (long)e * K_in * N_out;
    const float* grow = grad_out + (long)r * N_out;
    for (int i = 0; i < K_in; ++i) {
        float acc = 0.0f;
        for (int o = 0; o < N_out; ++o) acc += grow[o] * We[(long)i * N_out + o];
        grad_input[(long)r * K_in + i] = acc;
    }
}

__global__ void moe_grouped_gemm_backward_weight_scalar(
    const float* __restrict__ x, const float* __restrict__ grad_out,
    const int* __restrict__ expert_ids, float* __restrict__ grad_weights,
    int rows, int K_in, int N_out) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const int e = expert_ids[r];
    if (e < 0) return;
    float* ge = grad_weights + (long)e * K_in * N_out;
    const float* xr = x + (long)r * K_in;
    const float* gr = grad_out + (long)r * N_out;
    for (long t = 0; t < (long)K_in * N_out; ++t) {
        const int i = int(t / N_out);
        const int o = int(t % N_out);
        atomicAdd(&ge[t], xr[i] * gr[o]);
    }
}

void run_bench() {
    printf("== Phase 4 dense MoE A/B\n");
    {
        const int tokens = 16384, E = 128, K = 4, groups = 16, top_groups = 4;
        const float routed_scale = 2.5f;
        auto logits = randv((size_t)tokens * E, -3.0f, 3.0f);
        auto bias = randv(E, -0.2f, 0.2f);
        auto* dlog = dnew(logits);
        auto* dbias = dnew(bias);
        int* ids0 = dzero<int>((size_t)tokens * K);
        int* ids1 = dzero<int>((size_t)tokens * K);
        float* w0 = dzero<float>((size_t)tokens * K);
        float* w1 = dzero<float>((size_t)tokens * K);
        auto base = [&] {
            moe_route_grouped_scalar<<<(tokens + 127) / 128, 128>>>(
                dlog, dbias, ids0, w0, tokens, E, K, groups, top_groups, 1,
                routed_scale, MOE_SCORE_SIGMOID);
        };
        auto cand = [&] {
            moe_route_grouped<float><<<tokens, 32, 2 * E * sizeof(float)>>>(
                dlog, dbias, ids1, w1, E, K, groups, top_groups, 1,
                routed_scale, MOE_SCORE_SIGMOID);
        };
        base(); cand(); CUCHECK(hipDeviceSynchronize());
        printf("== moe_route_grouped fp32 tokens=%d E=%d K=%d groups=%d top_groups=%d\n",
               tokens, E, K, groups, top_groups);
        report_bench("scalar one-thread/token", bench_kernel(base, 5, 20, 20),
                     "warp grouped route", bench_kernel(cand, 5, 20, 100),
                     double(tokens) * E);
    }

    {
        const int tokens = 4096, gathered_rows = 16384, dim = 1024;
        auto grad = randv((size_t)gathered_rows * dim, -1.0f, 1.0f);
        std::vector<int> rows(gathered_rows);
        for (int r = 0; r < gathered_rows; ++r) rows[r] = (r % 11 == 0) ? -1 : (r * 17) % tokens;
        auto* dgrad = dnew(grad);
        auto* drows = dnew(rows);
        float* out0 = dzero<float>((size_t)tokens * dim);
        float* out1 = dzero<float>((size_t)tokens * dim);
        auto base = [&] {
            CUCHECK(hipMemset(out0, 0, (size_t)tokens * dim * sizeof(float)));
            moe_gather_backward_scalar<<<(gathered_rows + 255) / 256, 256>>>(
                dgrad, drows, out0, gathered_rows, dim);
        };
        auto cand = [&] {
            CUCHECK(hipMemset(out1, 0, (size_t)tokens * dim * sizeof(float)));
            moe_gather_backward<float><<<gathered_rows, 128>>>(
                dgrad, drows, out1, gathered_rows, dim);
        };
        base(); cand(); CUCHECK(hipDeviceSynchronize());
        printf("== moe_gather_backward fp32 gathered_rows=%d tokens=%d dim=%d\n",
               gathered_rows, tokens, dim);
        report_bench("scalar one-thread/row", bench_kernel(base, 5, 20, 10),
                     "row-parallel atomics", bench_kernel(cand, 5, 20, 100),
                     double(gathered_rows) * dim);
    }

    {
        const int tokens = 8192, top_k = 4, dim = 1024, routes = tokens * top_k;
        auto grad = randv((size_t)tokens * dim, -1.0f, 1.0f);
        auto expert = randv((size_t)routes * dim, -1.0f, 1.0f);
        auto weights = randv(routes, 0.1f, 1.0f);
        std::vector<int> inv(routes);
        for (int r = 0; r < routes; ++r) inv[r] = (r * 37) % routes;
        auto* dgrad = dnew(grad);
        auto* dexpert = dnew(expert);
        auto* dinv = dnew(inv);
        auto* dweights = dnew(weights);
        float* geo0 = dzero<float>((size_t)routes * dim);
        float* geo1 = dzero<float>((size_t)routes * dim);
        float* gew0 = dzero<float>(routes);
        float* gew1 = dzero<float>(routes);
        auto base = [&] {
            CUCHECK(hipMemset(geo0, 0, (size_t)routes * dim * sizeof(float)));
            moe_finalize_backward_scalar<<<(tokens + 127) / 128, 128>>>(
                dgrad, dexpert, dinv, dweights, geo0, gew0, tokens, top_k, dim);
        };
        auto cand = [&] {
            CUCHECK(hipMemset(geo1, 0, (size_t)routes * dim * sizeof(float)));
            moe_finalize_backward<float><<<tokens, 32>>>(
                dgrad, dexpert, dinv, dweights, geo1, gew1, top_k, dim);
        };
        base(); cand(); CUCHECK(hipDeviceSynchronize());
        printf("== moe_finalize_backward fp32 tokens=%d K=%d dim=%d\n",
               tokens, top_k, dim);
        report_bench("scalar one-thread/token", bench_kernel(base, 5, 20, 10),
                     "warp token backward", bench_kernel(cand, 5, 20, 50),
                     double(tokens) * top_k * dim);
    }

    {
        const int rows = 2048, experts = 16, kin = 256, nout = 256;
        auto grad = randv((size_t)rows * nout, -1.0f, 1.0f);
        auto weights = randv((size_t)experts * kin * nout, -1.0f, 1.0f);
        std::vector<int> eid(rows);
        for (int r = 0; r < rows; ++r) eid[r] = (r % 13 == 0) ? -1 : r % experts;
        auto* dgrad = dnew(grad);
        auto* dw = dnew(weights);
        auto* deid = dnew(eid);
        float* out0 = dzero<float>((size_t)rows * kin);
        float* out1 = dzero<float>((size_t)rows * kin);
        auto base = [&] {
            moe_grouped_gemm_backward_input_scalar<<<(rows + 127) / 128, 128>>>(
                dgrad, dw, deid, out0, rows, kin, nout);
        };
        auto cand = [&] {
            moe_grouped_gemm_backward_input<float><<<rows, 128>>>(
                dgrad, dw, deid, out1, rows, kin, nout);
        };
        base(); cand(); CUCHECK(hipDeviceSynchronize());
        printf("== moe_grouped_gemm_backward_input fp32 rows=%d experts=%d K=%d N=%d\n",
               rows, experts, kin, nout);
        report_bench("scalar one-thread/row", bench_kernel(base, 5, 20, 50),
                     "row-parallel input grad", bench_kernel(cand, 5, 20, 20),
                     2.0 * rows * kin * nout);
    }

    {
        const int rows = 2048, experts = 16, kin = 128, nout = 128;
        auto x = randv((size_t)rows * kin, -1.0f, 1.0f);
        auto grad = randv((size_t)rows * nout, -1.0f, 1.0f);
        std::vector<int> eid(rows);
        for (int r = 0; r < rows; ++r) eid[r] = (r % 17 == 0) ? -1 : r % experts;
        auto* dx = dnew(x);
        auto* dgrad = dnew(grad);
        auto* deid = dnew(eid);
        float* out0 = dzero<float>((size_t)experts * kin * nout);
        float* out1 = dzero<float>((size_t)experts * kin * nout);
        auto base = [&] {
            CUCHECK(hipMemset(out0, 0, (size_t)experts * kin * nout * sizeof(float)));
            moe_grouped_gemm_backward_weight_scalar<<<(rows + 127) / 128, 128>>>(
                dx, dgrad, deid, out0, rows, kin, nout);
        };
        auto cand = [&] {
            CUCHECK(hipMemset(out1, 0, (size_t)experts * kin * nout * sizeof(float)));
            moe_grouped_gemm_backward_weight<float><<<rows, 256>>>(
                dx, dgrad, deid, out1, rows, kin, nout);
        };
        base(); cand(); CUCHECK(hipDeviceSynchronize());
        printf("== moe_grouped_gemm_backward_weight fp32 rows=%d experts=%d K=%d N=%d\n",
               rows, experts, kin, nout);
        report_bench("scalar one-thread/row", bench_kernel(base, 5, 20, 10),
                     "row-parallel weight grad", bench_kernel(cand, 5, 20, 10),
                     2.0 * rows * kin * nout);
    }
}

int main(int argc, char** argv) {
    const int T = 200, E = 11, K = 4, H = 64, INTER = 96;
    const int TK = T * K;
    const int total_pad_max = ((TK + 31 * E) + 31) / 32 * 32;
    const int max_tiles = total_pad_max / 32;

    auto logits = randv((size_t)T * E, -3, 3);
    auto x = randv((size_t)T * H);
    auto w1 = randv((size_t)E * H * 2 * INTER, -0.3f, 0.3f);   // (E, H, 2*inter) [gate|up]
    auto w2 = randv((size_t)E * INTER * H, -0.3f, 0.3f);       // (E, inter, H)

    // ---------------- CPU fp64 reference ----------------
    std::vector<int> ref_ids(TK);
    std::vector<double> ref_w(TK);
    for (int t = 0; t < T; ++t) {
        std::vector<int> order(E);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
            const float la = logits[(size_t)t * E + a], lb = logits[(size_t)t * E + b];
            return la > lb || (la == lb && a < b);       // smaller-id ties
        });
        double m = -1e300;
        for (int k = 0; k < K; ++k) m = std::max(m, (double)logits[(size_t)t * E + order[k]]);
        double s = 0;
        for (int k = 0; k < K; ++k) s += std::exp((double)logits[(size_t)t * E + order[k]] - m);
        for (int k = 0; k < K; ++k) {
            ref_ids[t * K + k] = order[k];
            ref_w[t * K + k] = std::exp((double)logits[(size_t)t * E + order[k]] - m) / s;
        }
    }
    // dense MoE MLP: out[t] = sum_k w * ((silu(x@W1g)*(x@W1u)) @ W2)[e]
    std::vector<double> ref_out((size_t)T * H, 0.0);
    for (int t = 0; t < T; ++t) {
        for (int k = 0; k < K; ++k) {
            const int e = ref_ids[t * K + k];
            const double wgt = ref_w[t * K + k];
            std::vector<double> inter(INTER);
            for (int c = 0; c < INTER; ++c) {
                double g = 0, u = 0;
                for (int i = 0; i < H; ++i) {
                    const double a = x[(size_t)t * H + i];
                    g += a * w1[(size_t)e * H * 2 * INTER + (size_t)i * 2 * INTER + c];
                    u += a * w1[(size_t)e * H * 2 * INTER + (size_t)i * 2 * INTER + INTER + c];
                }
                inter[c] = (g / (1.0 + std::exp(-g))) * u;
            }
            for (int h = 0; h < H; ++h) {
                double acc = 0;
                for (int c = 0; c < INTER; ++c)
                    acc += inter[c] * w2[(size_t)e * INTER * H + (size_t)c * H + h];
                ref_out[(size_t)t * H + h] += wgt * acc;
            }
        }
    }

    // ---------------- GPU pipeline ----------------
    float *dlog = dnew(logits), *dx = dnew(x), *dw1 = dnew(w1), *dw2 = dnew(w2);
    int *dids = dzero<int>(TK);
    float *dwts = dzero<float>(TK);
    moe_route_topk<float><<<T, 32>>>(dlog, dids, dwts, E, K);
    CUCHECK(hipDeviceSynchronize());
    auto hids = d2h(dids, TK);
    auto hwts = d2h(dwts, TK);
    {
        long mm = 0;
        double werr = 0;
        for (int i = 0; i < TK; ++i) {
            mm += hids[i] != ref_ids[i];
            werr = std::max(werr, std::abs((double)hwts[i] - ref_w[i]));
        }
        report("moe_route_topk ids (exact)", mm == 0);
        report("moe_route_topk weights", werr < 1e-6, werr);
    }

    int *dcnt = dzero<int>(E), *doff = dzero<int>(E + 1), *dcur = dzero<int>(E);
    moe_histogram<<<(TK + 255) / 256, 256>>>(dids, dcnt, TK);
    moe_scan_offsets<<<1, MOE_SCAN_NT>>>(dcnt, doff, dcur, E);
    CUCHECK(hipDeviceSynchronize());
    auto hoff = d2h(doff, E + 1);
    {
        std::vector<int> cnt(E, 0);
        for (int i = 0; i < TK; ++i) cnt[ref_ids[i]]++;
        bool ok = true;
        int run = 0;
        for (int e = 0; e < E; ++e) { ok &= hoff[e] == run; run += cnt[e]; }
        ok &= hoff[E] == TK;
        report("moe_histogram + scan_offsets", ok);
    }

    int *dsri = dzero<int>(TK), *dinv = dzero<int>(TK);
    moe_scatter<<<(TK + 255) / 256, 256>>>(dids, dcur, dsri, dinv, TK);
    CUCHECK(hipDeviceSynchronize());
    auto hsri = d2h(dsri, TK);
    auto hinv = d2h(dinv, TK);
    {
        bool ok = true;
        for (int p = 0; p < TK; ++p) {                    // segment membership
            int lo = 0;
            while (lo + 1 < E + 1 && hoff[lo + 1] <= p) ++lo;
            ok &= ref_ids[hsri[p]] == lo;
        }
        for (int r = 0; r < TK; ++r) ok &= hsri[hinv[r]] == r;   // inverse map
        report("moe_scatter (segments + inverse)", ok);
    }

    int *dofp = dzero<int>(E + 1), *deot = dzero<int>(max_tiles),
        *dgix = dzero<int>(total_pad_max), *dinvp = dzero<int>(TK);
    moe_pad_offsets<<<1, MOE_SCAN_NT>>>(doff, dofp, deot, dgix, E, max_tiles, total_pad_max);
    moe_pad_scatter<<<(TK + 255) / 256, 256>>>(dsri, doff, dofp, dgix, dinvp, TK, E, K);
    CUCHECK(hipDeviceSynchronize());
    auto hofp = d2h(dofp, E + 1);
    auto heot = d2h(deot, max_tiles);
    auto hgix = d2h(dgix, total_pad_max);
    auto hinvp = d2h(dinvp, TK);
    const int total_pad = hofp[E];
    {
        bool ok = true;
        int run = 0;
        for (int e = 0; e < E; ++e) {                     // ceil32 exclusive scan
            ok &= hofp[e] == run;
            run += (hoff[e + 1] - hoff[e] + 31) / 32 * 32;
        }
        ok &= total_pad == run && total_pad % 32 == 0;
        for (int t = 0; t < max_tiles; ++t) {             // expert_of_tile
            const int pos = t * 32;
            if (pos >= total_pad) { ok &= heot[t] == -1; continue; }
            int lo = 0;
            while (lo + 1 < E && hofp[lo + 1] <= pos) ++lo;
            ok &= heot[t] == lo;
        }
        for (int r = 0; r < TK; ++r) {                    // inv_pad consistency
            const int pp = hinvp[r];
            ok &= pp >= 0 && pp < total_pad && hgix[pp] == r / K;
        }
        for (int p = 0; p < total_pad_max; ++p)           // pad rows stay -1
            if (hgix[p] >= 0) ok &= hgix[p] < T;
        report("moe_pad_offsets + pad_scatter", ok);
    }

    float *dperm = dzero<float>((size_t)total_pad_max * H);
    moe_gather<float><<<total_pad_max, 128>>>(dx, dgix, dperm, H);
    CUCHECK(hipDeviceSynchronize());
    {
        auto hperm = d2h(dperm, (size_t)total_pad_max * H);
        bool ok = true;
        for (int p = 0; p < total_pad_max; ++p) {
            for (int i = 0; i < H; i += 17) {
                const float want = hgix[p] >= 0 ? x[(size_t)hgix[p] * H + i] : 0.0f;
                ok &= hperm[(size_t)p * H + i] == want;
            }
        }
        report("moe_gather (rows + zero pads)", ok);
    }

    // grouped GEMMs over the padded schedule
    float *dinter = dzero<float>((size_t)total_pad_max * INTER);
    float *ddown = dzero<float>((size_t)total_pad_max * H);
    {
        dim3 g1{unsigned(INTER / 32), unsigned(total_pad / 32)};
        moe_grouped_gemm_swiglu<float><<<g1, 256>>>(dinter, dperm, dw1, deot,
                                                    total_pad, H, INTER);
        dim3 g2{unsigned(H / 32), unsigned(total_pad / 32)};
        moe_grouped_gemm_rect<float><<<g2, 256>>>(ddown, dinter, dw2, deot,
                                                  total_pad, INTER, H);
        CUCHECK(hipDeviceSynchronize());
    }

    float *dout = dzero<float>((size_t)T * H);
    moe_finalize<float><<<T, 32>>>(ddown, dinvp, dwts, dout, K, H);
    CUCHECK(hipDeviceSynchronize());
    {
        auto hout = d2h(dout, (size_t)T * H);
        double err = 0;
        for (size_t i = 0; i < hout.size(); ++i) {
            const double s = std::max(1.0, std::abs(ref_out[i]));
            err = std::max(err, std::abs((double)hout[i] - ref_out[i]) / s);
        }
        report("END-TO-END MoE MLP vs dense fp64", err < 5e-5, err);
    }

    // square grouped gemm parity (rect with K=N=H) on a small identity-ish case
    {
        std::vector<float> wsq((size_t)E * H * H);
        for (auto& v : wsq) v = std::uniform_real_distribution<float>(-0.3f, 0.3f)(g_rng);
        float* dwsq = dnew(wsq);
        float* dsq = dzero<float>((size_t)total_pad_max * H);
        dim3 g{unsigned(H / 32), unsigned(total_pad / 32)};
        moe_grouped_gemm_rect<float><<<g, 256>>>(dsq, dperm, dwsq, deot, total_pad, H, H);
        CUCHECK(hipDeviceSynchronize());
        auto hsq = d2h(dsq, (size_t)total_pad_max * H);
        auto hperm = d2h(dperm, (size_t)total_pad_max * H);
        double err = 0;
        for (int p = 0; p < total_pad; p += 7) {
            const int e = heot[p / 32];
            for (int c = 0; c < H; c += 13) {
                double acc = 0;
                for (int k = 0; k < H; ++k)
                    acc += (double)hperm[(size_t)p * H + k] * wsq[(size_t)e * H * H + (size_t)k * H + c];
                const double s = std::max(1.0, std::abs(acc));
                err = std::max(err, std::abs((double)hsq[(size_t)p * H + c] - acc) / s);
            }
        }
        report("moe_grouped_gemm square (spot fp64)", err < 5e-6, err);
    }

    // ---------------- Phase 4: MoE completeness ----------------
    // moe_route_grouped vs a host replica of the CPU reference (sigmoid scoring,
    // selection bias, group top-two ranking, renormalize x routed_scale).
    {
        const int T4 = 24, E4 = 16, K4 = 4, G4 = 4, TG4 = 2;
        const float rscale = 2.5f;
        std::vector<float> lg((size_t)T4 * E4), bs(E4);
        for (auto& v : lg) v = randv(1,-3.f, 3.f)[0];
        for (auto& v : bs) v = randv(1,-0.2f, 0.2f)[0];
        float *dlg = dnew(lg), *dbs = dnew(bs);
        int* did4 = dzero<int>((size_t)T4 * K4);
        float* dwt4 = dzero<float>((size_t)T4 * K4);
        moe_route_grouped<float><<<T4, 32, 2 * E4 * sizeof(float)>>>(
            dlg, dbs, did4, dwt4, E4, K4, G4, TG4, 1, rscale, MOE_SCORE_SIGMOID);
        CUCHECK(hipDeviceSynchronize());
        auto gid = d2h(did4, (size_t)T4 * K4);
        auto gwt = d2h(dwt4, (size_t)T4 * K4);

        long mism = 0; double werr = 0;
        const int gsz = E4 / G4;
        for (int t = 0; t < T4; ++t) {
            std::vector<double> sc(E4), sel(E4);
            for (int e = 0; e < E4; ++e) {
                sc[e] = 1.0 / (1.0 + std::exp(-(double)lg[(size_t)t * E4 + e]));
                sel[e] = sc[e] + bs[e];
            }
            std::vector<int> gord(G4);
            for (int g = 0; g < G4; ++g) gord[g] = g;
            auto top2 = [&](int g) {
                double f = -1e300, s = -1e300;
                for (int i = 0; i < gsz; ++i) {
                    double v = sel[(size_t)g * gsz + i];
                    if (v > f) { s = f; f = v; } else if (v > s) s = v;
                }
                return f + (gsz > 1 ? s : 0.0);
            };
            std::stable_sort(gord.begin(), gord.end(), [&](int a, int b) {
                double la = top2(a), lb = top2(b);
                return la == lb ? a < b : la > lb; });
            std::vector<int> cand;
            for (int r = 0; r < TG4; ++r)
                for (int i = 0; i < gsz; ++i) cand.push_back(gord[r] * gsz + i);
            std::partial_sort(cand.begin(), cand.begin() + K4, cand.end(), [&](int a, int b) {
                return sel[a] == sel[b] ? a < b : sel[a] > sel[b]; });
            double den = 0;
            for (int k = 0; k < K4; ++k) den += sc[cand[k]];
            for (int k = 0; k < K4; ++k) {
                mism += gid[(size_t)t * K4 + k] != cand[k];
                werr = std::max(werr, std::abs((double)gwt[(size_t)t * K4 + k]
                                               - sc[cand[k]] * rscale / den));
            }
        }
        printf("%-34s %s (%ld id mism, w err %.3e)\n", "moe_route_grouped",
               (mism == 0 && werr < 1e-5) ? "PASS" : "FAIL", mism, werr);
        g_fail += !(mism == 0 && werr < 1e-5);
    }

    // moe_gather_backward: scatter-add with repeats and -1 padding.
    {
        const int TOK = 16, GR = 40, D = 32;
        std::vector<float> gg((size_t)GR * D);
        for (auto& v : gg) v = randv(1,-1.f, 1.f)[0];
        std::vector<int> rows(GR);
        for (int r = 0; r < GR; ++r) rows[r] = (r % 7 == 0) ? -1 : (r * 5) % TOK;
        float* dgg = dnew(gg); int* drow = dnew(rows);
        float* dgi = dzero<float>((size_t)TOK * D);
        moe_gather_backward<float><<<GR, 128>>>(dgg, drow, dgi, GR, D);
        CUCHECK(hipDeviceSynchronize());
        auto got = d2h(dgi, (size_t)TOK * D);
        std::vector<double> ref((size_t)TOK * D, 0.0);
        for (int r = 0; r < GR; ++r)
            if (rows[r] >= 0)
                for (int i = 0; i < D; ++i) ref[(size_t)rows[r] * D + i] += gg[(size_t)r * D + i];
        double err = 0;
        for (size_t i = 0; i < ref.size(); ++i)
            err = std::max(err, std::abs((double)got[i] - ref[i]) / std::max(1.0, std::abs(ref[i])));
        report("moe_gather_backward (fp64)", err < 1e-6, err);
    }

    // moe_finalize_backward: adjoint of the weighted combine.
    {
        const int TOK = 8, KK = 2, D = 16, R = TOK * KK;
        std::vector<float> go((size_t)TOK * D), eo((size_t)R * D), ew(R);
        for (auto& v : go) v = randv(1,-1.f, 1.f)[0];
        for (auto& v : eo) v = randv(1,-1.f, 1.f)[0];
        for (auto& v : ew) v = randv(1,0.1f, 1.f)[0];
        std::vector<int> inv(R);
        for (int i = 0; i < R; ++i) inv[i] = (i * 3 + 1) % R;   // a permutation for R=16
        float* dgo = dnew(go); float* deo = dnew(eo); float* dew = dnew(ew);
        int* dinv4 = dnew(inv);
        float* dgeo = dzero<float>((size_t)R * D);
        float* dgew = dzero<float>(R);
        moe_finalize_backward<float><<<TOK, 32>>>(dgo, deo, dinv4, dew, dgeo, dgew, KK, D);
        CUCHECK(hipDeviceSynchronize());
        auto geo = d2h(dgeo, (size_t)R * D); auto gew = d2h(dgew, R);
        std::vector<double> reo((size_t)R * D, 0.0), rew(R, 0.0);
        for (int t = 0; t < TOK; ++t)
            for (int k = 0; k < KK; ++k) {
                const int rt = t * KK + k, p = inv[rt];
                double wg = 0;
                for (int i = 0; i < D; ++i) {
                    const double g = go[(size_t)t * D + i];
                    reo[(size_t)p * D + i] += ew[rt] * g;
                    wg += g * eo[(size_t)p * D + i];
                }
                rew[rt] = wg;
            }
        double err = 0;
        for (size_t i = 0; i < reo.size(); ++i)
            err = std::max(err, std::abs((double)geo[i] - reo[i]) / std::max(1.0, std::abs(reo[i])));
        for (int i = 0; i < R; ++i)
            err = std::max(err, std::abs((double)gew[i] - rew[i]) / std::max(1.0, std::abs(rew[i])));
        report("moe_finalize_backward (fp64)", err < 1e-6, err);
    }

    // moe_grouped_gemm backward wrt input and wrt weights.
    {
        const int R = 20, EE = 3, KI = 24, NO = 12;
        std::vector<float> xx((size_t)R * KI), go((size_t)R * NO),
                           ww((size_t)EE * KI * NO);
        for (auto& v : xx) v = randv(1,-1.f, 1.f)[0];
        for (auto& v : go) v = randv(1,-1.f, 1.f)[0];
        for (auto& v : ww) v = randv(1,-1.f, 1.f)[0];
        std::vector<int> eid(R);
        for (int r = 0; r < R; ++r) eid[r] = (r == R - 1) ? -1 : r % EE;   // last row padded
        float* dx4 = dnew(xx); float* dgo = dnew(go); float* dw4 = dnew(ww);
        int* deid = dnew(eid);
        float* dgi = dzero<float>((size_t)R * KI);
        float* dgw = dzero<float>((size_t)EE * KI * NO);
        moe_grouped_gemm_backward_input<float><<<R, 128>>>(dgo, dw4, deid, dgi, R, KI, NO);
        moe_grouped_gemm_backward_weight<float><<<R, 256>>>(dx4, dgo, deid, dgw, R, KI, NO);
        CUCHECK(hipDeviceSynchronize());
        auto gi = d2h(dgi, (size_t)R * KI); auto gw = d2h(dgw, (size_t)EE * KI * NO);
        double ei = 0, ew2 = 0;
        for (int r = 0; r < R; ++r) {
            if (eid[r] < 0) continue;
            for (int i = 0; i < KI; ++i) {
                double acc = 0;
                for (int o = 0; o < NO; ++o)
                    acc += (double)go[(size_t)r * NO + o]
                         * ww[((size_t)eid[r] * KI + i) * NO + o];
                ei = std::max(ei, std::abs((double)gi[(size_t)r * KI + i] - acc)
                                  / std::max(1.0, std::abs(acc)));
            }
        }
        std::vector<double> rw((size_t)EE * KI * NO, 0.0);
        for (int r = 0; r < R; ++r) {
            if (eid[r] < 0) continue;
            for (int i = 0; i < KI; ++i)
                for (int o = 0; o < NO; ++o)
                    rw[((size_t)eid[r] * KI + i) * NO + o] +=
                        (double)xx[(size_t)r * KI + i] * go[(size_t)r * NO + o];
        }
        for (size_t i = 0; i < rw.size(); ++i)
            ew2 = std::max(ew2, std::abs((double)gw[i] - rw[i]) / std::max(1.0, std::abs(rw[i])));
        report("moe_grouped_gemm_backward_input (fp64)", ei < 1e-6, ei);
        report("moe_grouped_gemm_backward_weight (fp64)", ew2 < 1e-6, ew2);
    }

    printf("\n%s (%d failures)\n", g_fail ? "FAILED" : "ALL PASS", g_fail);
    if (g_fail == 0 && argc > 1 && std::strcmp(argv[1], "--bench") == 0) run_bench();
    return g_fail ? 1 : 0;
}
