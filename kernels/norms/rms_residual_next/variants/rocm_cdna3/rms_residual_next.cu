/**
 * @file
 * @brief CDNA3 (gfx942) fused residual-add + double RMSNorm with the residual
 *        stream held in registers between the two norms.
 *
 * Shape/format-named port of embeddinggemma.c `rms_residual_next_f32_cached` /
 * `rms_residual_next_f16_cached` (register-cached) and `rms_residual_next_f16`
 * (general n_cols) in src/engine_rocm.hip; math matches the CPU oracle
 * `ei_rms_norm_residual_inplace` + `ei_rms_norm` in src/kernels.c.
 *
 * One launch (one 32-lane warp per token row) does, over n_cols = EI_N_EMBD:
 *   1. projected_inv = rsqrt( mean(input^2) + eps )
 *   2. residual[c]  += input[c] * post_weight[c] * projected_inv   (updated stream)
 *   3. residual_inv  = rsqrt( mean(residual^2) + eps )
 *   4. next_out[c]   = residual[c] * next_weight[c] * residual_inv
 * collapsing the ~4 norm launches/layer to 2. The cached variant keeps the
 * EI_N_EMBD/32 = 24 residual values per lane in registers between steps 2 and 4
 * so the residual stream is not re-read from device memory.
 *
 * Caveat: the register-cache count (items_per_lane) is shape-locked to
 * EI_N_EMBD/32 = 24. Self-parity vs the separate-launch route is ~0.999997
 * (accumulation order + f16 output rounding), so validation is against the fp64
 * oracle at a tolerance, not bitwise.
 *
 * RMSNorm form matches the model: out = x * weight * rsqrt(mean(x^2)+eps); any
 * (1+gamma) offset is folded into the stored weight upstream.
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

// EmbeddingGemma-300M hidden width (src/model.h EI_N_EMBD); the register cache
// depth is shape-locked to EI_N_EMBD / 32.
constexpr int kNEmbd = 768;
constexpr int kThreads = 256;
constexpr int kWarpsPerBlock = kThreads / 32;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down(value, offset, 32);
    }
    return value;
}

// ---- Candidate (fused, register-cached) ----------------------------------

__global__ void rms_residual_next_f32_cached_kernel(
    const float *input, const float *post_weight, float *residual,
    const float *next_weight, float *next_output, uint32_t n_rows, float eps) {
    constexpr int items_per_lane = kNEmbd / 32;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const uint32_t row = blockIdx.x * kWarpsPerBlock + warp;
    if (row >= n_rows) return;
    const size_t base = static_cast<size_t>(row) * kNEmbd;
    float projected_ss = 0.0f;
#pragma unroll
    for (int item = 0; item < items_per_lane; item++) {
        const int col = lane + item * 32;
        const float value = input[base + col];
        projected_ss = fmaf(value, value, projected_ss);
    }
    projected_ss = warp_sum(projected_ss);
    projected_ss = __shfl(projected_ss, 0, 32);
    const float projected_inv = rsqrtf(projected_ss / static_cast<float>(kNEmbd) + eps);

    float residual_values[items_per_lane];
    float residual_ss = 0.0f;
#pragma unroll
    for (int item = 0; item < items_per_lane; item++) {
        const int col = lane + item * 32;
        const float value = residual[base + col] +
            input[base + col] * post_weight[col] * projected_inv;
        residual_values[item] = value;
        residual[base + col] = value;
        residual_ss = fmaf(value, value, residual_ss);
    }
    residual_ss = warp_sum(residual_ss);
    residual_ss = __shfl(residual_ss, 0, 32);
    const float residual_inv = rsqrtf(residual_ss / static_cast<float>(kNEmbd) + eps);
#pragma unroll
    for (int item = 0; item < items_per_lane; item++) {
        const int col = lane + item * 32;
        next_output[base + col] = residual_values[item] * next_weight[col] * residual_inv;
    }
}

__global__ void rms_residual_next_f16_cached_kernel(
    const float *input, const float *post_weight, float *residual,
    const float *next_weight, __half *next_output, uint32_t n_rows, float eps) {
    constexpr int items_per_lane = kNEmbd / 32;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const uint32_t row = blockIdx.x * kWarpsPerBlock + warp;
    if (row >= n_rows) return;
    const size_t base = static_cast<size_t>(row) * kNEmbd;
    float projected_ss = 0.0f;
#pragma unroll
    for (int item = 0; item < items_per_lane; item++) {
        const int col = lane + item * 32;
        const float value = input[base + col];
        projected_ss = fmaf(value, value, projected_ss);
    }
    projected_ss = warp_sum(projected_ss);
    projected_ss = __shfl(projected_ss, 0, 32);
    const float projected_inv = rsqrtf(projected_ss / static_cast<float>(kNEmbd) + eps);

    float residual_values[items_per_lane];
    float residual_ss = 0.0f;
#pragma unroll
    for (int item = 0; item < items_per_lane; item++) {
        const int col = lane + item * 32;
        const float value = residual[base + col] +
            input[base + col] * post_weight[col] * projected_inv;
        residual_values[item] = value;
        residual[base + col] = value;
        residual_ss = fmaf(value, value, residual_ss);
    }
    residual_ss = warp_sum(residual_ss);
    residual_ss = __shfl(residual_ss, 0, 32);
    const float residual_inv = rsqrtf(residual_ss / static_cast<float>(kNEmbd) + eps);
#pragma unroll
    for (int item = 0; item < items_per_lane; item++) {
        const int col = lane + item * 32;
        next_output[base + col] = __float2half(
            residual_values[item] * next_weight[col] * residual_inv);
    }
}

// ---- Baseline pieces (unfused = separate norm launches) ------------------
// Step 1+2: projected RMS over input, then residual += input*post_weight*inv.
__global__ void rms_residual_add_kernel(
    const float *input, const float *post_weight, float *residual,
    uint32_t n_rows, uint32_t n_cols, float eps) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const uint32_t row = blockIdx.x * kWarpsPerBlock + warp;
    if (row >= n_rows) return;
    const size_t base = static_cast<size_t>(row) * n_cols;
    float ss = 0.0f;
    for (uint32_t col = lane; col < n_cols; col += 32) {
        const float value = input[base + col];
        ss = fmaf(value, value, ss);
    }
    ss = warp_sum(ss);
    ss = __shfl(ss, 0, 32);
    const float inv = rsqrtf(ss / static_cast<float>(n_cols) + eps);
    for (uint32_t col = lane; col < n_cols; col += 32) {
        residual[base + col] += input[base + col] * post_weight[col] * inv;
    }
}

// Step 3+4 (f32 out): RMS over residual, next_output = residual*next_weight*inv.
__global__ void rms_norm_to_f32_kernel(
    const float *residual, const float *next_weight, float *next_output,
    uint32_t n_rows, uint32_t n_cols, float eps) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const uint32_t row = blockIdx.x * kWarpsPerBlock + warp;
    if (row >= n_rows) return;
    const size_t base = static_cast<size_t>(row) * n_cols;
    float ss = 0.0f;
    for (uint32_t col = lane; col < n_cols; col += 32) {
        const float value = residual[base + col];
        ss = fmaf(value, value, ss);
    }
    ss = warp_sum(ss);
    ss = __shfl(ss, 0, 32);
    const float inv = rsqrtf(ss / static_cast<float>(n_cols) + eps);
    for (uint32_t col = lane; col < n_cols; col += 32) {
        next_output[base + col] = residual[base + col] * next_weight[col] * inv;
    }
}

// Step 3+4 (f16 out).
__global__ void rms_norm_to_f16_kernel(
    const float *residual, const float *next_weight, __half *next_output,
    uint32_t n_rows, uint32_t n_cols, float eps) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const uint32_t row = blockIdx.x * kWarpsPerBlock + warp;
    if (row >= n_rows) return;
    const size_t base = static_cast<size_t>(row) * n_cols;
    float ss = 0.0f;
    for (uint32_t col = lane; col < n_cols; col += 32) {
        const float value = residual[base + col];
        ss = fmaf(value, value, ss);
    }
    ss = warp_sum(ss);
    ss = __shfl(ss, 0, 32);
    const float inv = rsqrtf(ss / static_cast<float>(n_cols) + eps);
    for (uint32_t col = lane; col < n_cols; col += 32) {
        next_output[base + col] = __float2half(
            residual[base + col] * next_weight[col] * inv);
    }
}

// ---- Host harness ---------------------------------------------------------

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
    hipEvent_t a, b;
    CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    for (int i = 0; i < iters; ++i) {
        CK(hipEventRecord(a)); fn(); CK(hipEventRecord(b));
        CK(hipEventSynchronize(b)); CK(hipEventElapsedTime(&t[i], a, b));
    }
    CK(hipEventDestroy(a)); CK(hipEventDestroy(b));
    std::sort(t.begin(), t.end());
    return t[iters / 2];
}

static int row_blocks(uint32_t rows) {
    return static_cast<int>((rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
}

// fp64 oracle for the whole fused pipeline; fills the updated residual and the
// next projection input.
static void host_ref(const std::vector<float>& input,
                     const std::vector<float>& post_weight,
                     const std::vector<float>& residual_in,
                     const std::vector<float>& next_weight,
                     int M, int N, float eps,
                     std::vector<float>& residual_out,
                     std::vector<float>& next_out) {
    residual_out.assign(static_cast<size_t>(M) * N, 0.0f);
    next_out.assign(static_cast<size_t>(M) * N, 0.0f);
    for (int m = 0; m < M; ++m) {
        const size_t base = static_cast<size_t>(m) * N;
        double pss = 0.0;
        for (int c = 0; c < N; ++c) pss += (double)input[base + c] * input[base + c];
        const double pinv = 1.0 / std::sqrt(pss / (double)N + (double)eps);
        double rss = 0.0;
        std::vector<double> rv(N);
        for (int c = 0; c < N; ++c) {
            const double v = (double)residual_in[base + c] +
                (double)input[base + c] * (double)post_weight[c] * pinv;
            rv[c] = v;
            residual_out[base + c] = (float)v;
            rss += v * v;
        }
        const double rinv = 1.0 / std::sqrt(rss / (double)N + (double)eps);
        for (int c = 0; c < N; ++c)
            next_out[base + c] = (float)(rv[c] * (double)next_weight[c] * rinv);
    }
}

template <typename T>
static int check(const char* name, const std::vector<T>& got,
                 const std::vector<float>& ref, double tol) {
    double sad = 0.0, sref = 0.0, maxe = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double g = (double)(float)got[i];
        const double d = std::abs(g - (double)ref[i]);
        sad += d; sref += std::abs((double)ref[i]); maxe = std::max(maxe, d);
    }
    const double rel = sad / std::max(sref, 1e-30);
    const bool ok = rel < tol;
    printf("%-30s rel %.3e max %.3e %s\n", name, rel, maxe, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// cosine similarity (self-parity metric used across the sibling ports).
template <typename T>
static double cosine(const std::vector<T>& a, const std::vector<float>& b) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < b.size(); ++i) {
        const double x = (double)(float)a[i], y = (double)b[i];
        dot += x * y; na += x * x; nb += y * y;
    }
    return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
}

static int run_case(int M, bool bench) {
    const int N = kNEmbd;
    const float eps = 1e-6f;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> input(static_cast<size_t>(M) * N);
    std::vector<float> residual(static_cast<size_t>(M) * N);
    std::vector<float> post_weight(N), next_weight(N);
    for (auto& v : input) v = nd(rng) * 0.5f;
    for (auto& v : residual) v = nd(rng) * 0.5f;
    for (auto& v : post_weight) v = 1.0f + nd(rng) * 0.1f;   // (1+gamma)-style
    for (auto& v : next_weight) v = 1.0f + nd(rng) * 0.1f;

    std::vector<float> ref_res, ref_next;
    host_ref(input, post_weight, residual, next_weight, M, N, eps, ref_res, ref_next);

    auto dInput = dnew(input);
    auto dPost = dnew(post_weight);
    auto dNextW = dnew(next_weight);

    // f32 fused candidate.
    auto dResF32 = dnew(residual);
    float *dNextF32; CK(hipMalloc(&dNextF32, ref_next.size() * sizeof(float)));
    auto runFusedF32 = [&] {
        rms_residual_next_f32_cached_kernel<<<row_blocks(M), kThreads>>>(
            dInput, dPost, dResF32, dNextW, dNextF32, M, eps);
    };
    // f16 fused candidate.
    auto dResF16 = dnew(residual);
    __half *dNextF16; CK(hipMalloc(&dNextF16, ref_next.size() * sizeof(__half)));
    auto runFusedF16 = [&] {
        rms_residual_next_f16_cached_kernel<<<row_blocks(M), kThreads>>>(
            dInput, dPost, dResF16, dNextW, dNextF16, M, eps);
    };
    // Unfused baselines: two launches each (residual add, then next norm). The
    // residual buffers start as a fresh copy of `residual` (dnew), so the FIRST
    // invocation below is the correctness reference; the timed A/B is pure
    // kernel launches on both sides (no host<->device copies in the timed
    // region) so the measurement reflects the extra launch + residual DRAM
    // round-trip the fusion removes, not bookkeeping.
    auto dResUf32 = dnew(residual);
    float *dNextUf32; CK(hipMalloc(&dNextUf32, ref_next.size() * sizeof(float)));
    auto runUnfusedF32 = [&] {
        rms_residual_add_kernel<<<row_blocks(M), kThreads>>>(
            dInput, dPost, dResUf32, M, N, eps);
        rms_norm_to_f32_kernel<<<row_blocks(M), kThreads>>>(
            dResUf32, dNextW, dNextUf32, M, N, eps);
    };
    auto dResUf16 = dnew(residual);
    __half *dNextUf16; CK(hipMalloc(&dNextUf16, ref_next.size() * sizeof(__half)));
    auto runUnfusedF16 = [&] {
        rms_residual_add_kernel<<<row_blocks(M), kThreads>>>(
            dInput, dPost, dResUf16, M, N, eps);
        rms_norm_to_f16_kernel<<<row_blocks(M), kThreads>>>(
            dResUf16, dNextW, dNextUf16, M, N, eps);
    };

    // Correctness first (fresh buffers, single invocation each), before any
    // repeated bench launches drift the in-place residual.
    runFusedF32(); runFusedF16(); runUnfusedF32(); runUnfusedF16();
    CK(hipDeviceSynchronize()); CK(hipGetLastError());

    auto hResF32 = d2h(dResF32, ref_res.size());
    auto hNextF32 = d2h(dNextF32, ref_next.size());
    auto hNextF16 = d2h(dNextF16, ref_next.size());
    auto hResUf32 = d2h(dResUf32, ref_res.size());
    auto hNextUf32 = d2h(dNextUf32, ref_next.size());
    auto hNextUf16 = d2h(dNextUf16, ref_next.size());

    int rc = 0;
    printf("--- M=%d ---\n", M);
    rc |= check("fused_f32 residual", hResF32, ref_res, 1e-4);
    rc |= check("fused_f32 next", hNextF32, ref_next, 1e-4);
    rc |= check("fused_f16 next", hNextF16, ref_next, 3e-3);
    rc |= check("unfused_f32 residual (base)", hResUf32, ref_res, 1e-4);
    rc |= check("unfused_f32 next (base)", hNextUf32, ref_next, 1e-4);
    rc |= check("unfused_f16 next (base)", hNextUf16, ref_next, 3e-3);
    printf("%-30s cos %.7f\n", "fused_f32 vs unfused_f32 next",
           cosine(hNextF32, std::vector<float>(hNextUf32.begin(), hNextUf32.end())));
    printf("%-30s cos %.7f\n", "fused_f32 vs oracle next", cosine(hNextF32, ref_next));

    if (bench && rc == 0) {
        const float tf32 = bench_ms(runFusedF32);
        const float tu32 = bench_ms(runUnfusedF32);
        const float tf16 = bench_ms(runFusedF16);
        const float tu16 = bench_ms(runUnfusedF16);
        printf("rms_residual_next bench M=%d N=%d\n", M, N);
        printf("  f32: unfused(base) %.5f ms  fused(cand) %.5f ms  (%.3fx)\n",
               tu32, tf32, tu32 / tf32);
        printf("  f16: unfused(base) %.5f ms  fused(cand) %.5f ms  (%.3fx)\n",
               tu16, tf16, tu16 / tf16);
    }
    CK(hipFree(dInput)); CK(hipFree(dPost)); CK(hipFree(dNextW));
    CK(hipFree(dResF32)); CK(hipFree(dNextF32));
    CK(hipFree(dResF16)); CK(hipFree(dNextF16));
    CK(hipFree(dResUf32)); CK(hipFree(dNextUf32));
    CK(hipFree(dResUf16)); CK(hipFree(dNextUf16));
    return rc;
}

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    int rc = 0;
    rc |= run_case(64, false);
    if (bench) {
        rc |= run_case(64, true);
        rc |= run_case(256, true);
        rc |= run_case(512, true);
        rc |= run_case(1024, true);
        rc |= run_case(2048, true);
    }
    printf(rc == 0 ? "ALL PASS\n" : "FAIL\n");
    return rc;
}
