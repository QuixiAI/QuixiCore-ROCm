/**
 * @file
 * @brief CDNA3 (gfx942) sentence-embedding pooling head: masked mean-pool of
 *        per-token RMSNorm, then L2-normalize. One wave64 owns a sequence.
 *
 * Shape-named port of embeddinggemma.c `pool_kernel` (src/engine_rocm.hip,
 * ~L2463); the fp64 oracle is the CPU reference `ei_mean_pool_rms_l2`
 * (src/kernels.c, ~L417). Named by the D-vector shape (D in {256,512,768,1024}),
 * never by model. Semantics matched exactly (order matters):
 *
 *   for each token t in [start, stop):                       // per sequence
 *       ss   = sum_d x[t][d]^2
 *       inv  = 1 / sqrt(ss / D + eps)                         // RMSNorm scale
 *       m[d] += x[t][d] * inv * w[d]                          // learned gain w
 *   m[d] *= 1 / n_tokens                                      // masked mean-pool
 *   l2   = sum_d m[d]^2
 *   y[d] = m[d] * (l2 == 0 ? 1 : rsqrt(l2))                   // L2-normalize
 *
 * The learned RMS gain `w` is applied multiplicatively exactly as the reference
 * `ei_norm_scale` (out = x*scale*w). Gemma's stored "(1+w)" gain is baked into
 * the exported weight upstream at load -- both the CPU reference and the source
 * HIP kernel use plain `w`, so this kernel carries no (1+w). eps is the model
 * rms_eps (embeddinggemma default 1e-6).
 *
 * This is the plain mean-pool -> RMS -> L2 head only. It deliberately does NOT
 * port embeddinggemma's `final_singleton_pool` fused-singleton epilogue, which
 * fails parity on GEMM shapes (cosine 0.770).
 *
 * Layout: input X [total_tokens, D] (token-major), offsets[batch+1] give each
 * sequence's [start, stop) token range, weight [D], output Y [batch, D].
 *
 * Candidate: one wave64 owns a sequence's D-vector, register-blocked ITEMS=D/64
 * per lane; each per-token sum-of-squares is a single wave shfl_xor butterfly
 * (no LDS, no block sync), and the normalized rows stay in registers across the
 * token loop. Baseline (A/B): the same math composed as two passes through
 * global memory -- stage 1 RMS-normalizes every token row into a
 * [total_tokens, D] temp, stage 2 mean-pools + L2 over the temp. The fusion
 * removes the temp round-trip (~3x -> ~1x token-matrix traffic).
 */
#include <hip/hip_runtime.h>
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

constexpr int kWave = 64;  // gfx942 wavefront
constexpr int kThreads = 256;
constexpr int kWavesPerBlock = kThreads / kWave;

// Wave64 all-reduce (every lane returns the full 64-lane sum).
__device__ __forceinline__ float wave_reduce_sum(float v) {
#pragma unroll
    for (int offset = kWave / 2; offset > 0; offset >>= 1)
        v += __shfl_xor(v, offset, kWave);
    return v;
}

// ---- candidate: fused mean-pool -> per-token RMSNorm -> L2, one wave/seq -----
template <int D>
__global__ void pool_mean_rms_l2(const float *__restrict__ input,
                                 const float *__restrict__ weight,
                                 float *__restrict__ output,
                                 const uint32_t *__restrict__ offsets,
                                 uint32_t batch_size, float eps) {
    constexpr int ITEMS = D / kWave;
    static_assert(D % kWave == 0, "D must be a multiple of the wavefront (64)");
    const int wave = threadIdx.x >> 6;
    const int lane = threadIdx.x & 63;
    const uint32_t sequence = blockIdx.x * kWavesPerBlock + wave;
    if (sequence >= batch_size) return;
    const uint32_t start = offsets[sequence];
    const uint32_t stop = offsets[sequence + 1];

    float pooled[ITEMS];
#pragma unroll
    for (int item = 0; item < ITEMS; item++) pooled[item] = 0.0f;

    for (uint32_t token = start; token < stop; token++) {
        const size_t base = static_cast<size_t>(token) * D;
        float values[ITEMS];
        float ss = 0.0f;
#pragma unroll
        for (int item = 0; item < ITEMS; item++) {
            const int dim = lane + item * kWave;
            const float v = input[base + dim];
            values[item] = v;
            ss = fmaf(v, v, ss);
        }
        ss = wave_reduce_sum(ss);
        const float inv = rsqrtf(ss / static_cast<float>(D) + eps);
#pragma unroll
        for (int item = 0; item < ITEMS; item++) {
            const int dim = lane + item * kWave;
            pooled[item] = fmaf(values[item] * weight[dim], inv, pooled[item]);
        }
    }

    const float inv_tokens = 1.0f / static_cast<float>(stop - start);
    float l2 = 0.0f;
#pragma unroll
    for (int item = 0; item < ITEMS; item++) {
        pooled[item] *= inv_tokens;
        l2 = fmaf(pooled[item], pooled[item], l2);
    }
    l2 = wave_reduce_sum(l2);
    const float inv_l2 = l2 == 0.0f ? 1.0f : rsqrtf(l2);
    const size_t out_base = static_cast<size_t>(sequence) * D;
#pragma unroll
    for (int item = 0; item < ITEMS; item++) {
        const int dim = lane + item * kWave;
        output[out_base + dim] = pooled[item] * inv_l2;
    }
}

// ---- baseline: compose the ops (RMS rows -> global temp, then mean + L2) -----
// Stage 1: RMS-normalize every token row into a temp [total_tokens, D] buffer.
template <int D>
__global__ void rms_norm_rows(const float *__restrict__ input,
                              const float *__restrict__ weight,
                              float *__restrict__ out, uint32_t total_tokens,
                              float eps) {
    constexpr int ITEMS = D / kWave;
    const int wave = threadIdx.x >> 6;
    const int lane = threadIdx.x & 63;
    const uint32_t token = blockIdx.x * kWavesPerBlock + wave;
    if (token >= total_tokens) return;
    const size_t base = static_cast<size_t>(token) * D;
    float ss = 0.0f;
#pragma unroll
    for (int item = 0; item < ITEMS; item++) {
        const float v = input[base + lane + item * kWave];
        ss = fmaf(v, v, ss);
    }
    ss = wave_reduce_sum(ss);
    const float inv = rsqrtf(ss / static_cast<float>(D) + eps);
#pragma unroll
    for (int item = 0; item < ITEMS; item++) {
        const int dim = lane + item * kWave;
        out[base + dim] = input[base + dim] * weight[dim] * inv;
    }
}
// Stage 2: mean over each sequence's normalized rows, then L2-normalize.
template <int D>
__global__ void mean_l2_reduce(const float *__restrict__ normed,
                               float *__restrict__ output,
                               const uint32_t *__restrict__ offsets,
                               uint32_t batch_size) {
    constexpr int ITEMS = D / kWave;
    const int wave = threadIdx.x >> 6;
    const int lane = threadIdx.x & 63;
    const uint32_t sequence = blockIdx.x * kWavesPerBlock + wave;
    if (sequence >= batch_size) return;
    const uint32_t start = offsets[sequence];
    const uint32_t stop = offsets[sequence + 1];
    float pooled[ITEMS];
#pragma unroll
    for (int item = 0; item < ITEMS; item++) pooled[item] = 0.0f;
    for (uint32_t token = start; token < stop; token++) {
        const size_t base = static_cast<size_t>(token) * D;
#pragma unroll
        for (int item = 0; item < ITEMS; item++)
            pooled[item] += normed[base + lane + item * kWave];
    }
    const float inv_tokens = 1.0f / static_cast<float>(stop - start);
    float l2 = 0.0f;
#pragma unroll
    for (int item = 0; item < ITEMS; item++) {
        pooled[item] *= inv_tokens;
        l2 = fmaf(pooled[item], pooled[item], l2);
    }
    l2 = wave_reduce_sum(l2);
    const float inv_l2 = l2 == 0.0f ? 1.0f : rsqrtf(l2);
    const size_t out_base = static_cast<size_t>(sequence) * D;
#pragma unroll
    for (int item = 0; item < ITEMS; item++) {
        const int dim = lane + item * kWave;
        output[out_base + dim] = pooled[item] * inv_l2;
    }
}

// ============================== harness ==============================
template <typename T>
static T *dnew(const std::vector<T> &h) {
    T *d = nullptr;
    CK(hipMalloc(&d, h.size() * sizeof(T)));
    CK(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice));
    return d;
}
template <typename T>
static std::vector<T> d2h(const T *d, size_t n) {
    std::vector<T> h(n);
    CK(hipMemcpy(h.data(), d, n * sizeof(T), hipMemcpyDeviceToHost));
    return h;
}
template <typename FN>
static float bench_ms(FN &&fn, int warmup = 10, int iters = 50) {
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
static int row_blocks(size_t rows) {
    return static_cast<int>((rows + kWavesPerBlock - 1) / kWavesPerBlock);
}
static float frand(std::mt19937 &r) {
    std::normal_distribution<float> nd(0.f, 1.f);
    return nd(r);
}

// fp64 oracle mirroring ei_mean_pool_rms_l2 (per-token RMS -> mean -> L2).
static void oracle(const std::vector<float> &x, const std::vector<float> &w,
                   int D, uint32_t start, uint32_t stop, float eps,
                   std::vector<double> &out) {
    out.assign(D, 0.0);
    for (uint32_t t = start; t < stop; t++) {
        double ss = 0.0;
        for (int d = 0; d < D; d++) { double v = x[(size_t)t * D + d]; ss += v * v; }
        double scale = 1.0 / std::sqrt(ss / (double)D + (double)eps);
        for (int d = 0; d < D; d++)
            out[d] += (double)x[(size_t)t * D + d] * scale * (double)w[d];
    }
    double inv_tokens = 1.0 / (double)(stop - start);
    double ss2 = 0.0;
    for (int d = 0; d < D; d++) { out[d] *= inv_tokens; ss2 += out[d] * out[d]; }
    if (ss2 != 0.0) {
        double inv = 1.0 / std::sqrt(ss2);
        for (int d = 0; d < D; d++) out[d] *= inv;
    }
}

template <int D>
static bool correctness() {
    std::mt19937 rng(1234 + D);
    const float eps = 1e-6f;  // model rms_eps
    // One packed batch, mixed token counts incl. singleton and long sequences.
    const int token_counts[] = {1, 2, 4, 7, 37, 128, 300, 512};
    const int nseq = sizeof(token_counts) / sizeof(int);
    std::vector<uint32_t> off(nseq + 1, 0);
    for (int s = 0; s < nseq; s++) off[s + 1] = off[s] + token_counts[s];
    const uint32_t total = off[nseq];
    std::vector<float> x((size_t)total * D), w(D);
    for (auto &v : x) v = frand(rng) * 0.7f;
    for (auto &v : w) v = 1.0f + frand(rng) * 0.1f;  // Gemma learned gain ~ 1

    auto dx = dnew(x), dw = dnew(w);
    auto doff = dnew(off);
    float *dout = nullptr;
    CK(hipMalloc(&dout, (size_t)nseq * D * sizeof(float)));
    pool_mean_rms_l2<D><<<row_blocks(nseq), kThreads>>>(dx, dw, dout, doff, nseq, eps);
    CK(hipDeviceSynchronize()); CK(hipGetLastError());
    auto got = d2h(dout, (size_t)nseq * D);

    double gsum = 0, rsum = 0, gmax = 0, dotp = 0, ng = 0, no = 0;
    std::vector<double> ref;
    for (int s = 0; s < nseq; s++) {
        oracle(x, w, D, off[s], off[s + 1], eps, ref);
        for (int d = 0; d < D; d++) {
            double g = got[(size_t)s * D + d], o = ref[d];
            gsum += std::abs(g - o); rsum += std::abs(o);
            gmax = std::max(gmax, std::abs(g - o));
            dotp += g * o; ng += g * g; no += o * o;
        }
    }
    double rel = gsum / std::max(rsum, 1e-30);
    double cosine = dotp / std::max(std::sqrt(ng * no), 1e-30);
    bool pass = rel < 0.02 && cosine > 0.9999;
    printf("  D=%-4d correctness (n_tokens 1..512): rel %.3e max %.3e cosine %.8f  %s\n",
           D, rel, gmax, cosine, pass ? "PASS" : "FAIL");
    CK(hipFree(dx)); CK(hipFree(dw)); CK(hipFree(dout)); CK(hipFree(doff));
    return pass;
}

template <int D>
static bool perf(int B, int T) {
    std::mt19937 rng(999 + D);
    const float eps = 1e-6f;
    std::vector<uint32_t> off(B + 1, 0);
    for (int s = 0; s < B; s++) off[s + 1] = off[s] + T;
    const uint32_t total = off[B];
    std::vector<float> x((size_t)total * D), w(D);
    for (auto &v : x) v = frand(rng) * 0.7f;
    for (auto &v : w) v = 1.0f + frand(rng) * 0.1f;

    auto dx = dnew(x), dw = dnew(w);
    auto doff = dnew(off);
    float *dout = nullptr, *dtmp = nullptr;
    CK(hipMalloc(&dout, (size_t)B * D * sizeof(float)));
    CK(hipMalloc(&dtmp, (size_t)total * D * sizeof(float)));  // naive stage-1 temp

    auto fused = [&] {
        pool_mean_rms_l2<D><<<row_blocks(B), kThreads>>>(dx, dw, dout, doff, B, eps);
    };
    auto naive = [&] {
        rms_norm_rows<D><<<row_blocks(total), kThreads>>>(dx, dw, dtmp, total, eps);
        mean_l2_reduce<D><<<row_blocks(B), kThreads>>>(dtmp, dout, doff, B);
    };
    // Verify the two paths agree before timing.
    fused(); auto a = d2h(dout, (size_t)B * D);
    naive(); auto b = d2h(dout, (size_t)B * D);
    CK(hipDeviceSynchronize()); CK(hipGetLastError());
    double d = 0, r = 0;
    for (size_t i = 0; i < a.size(); i++) { d += std::abs(a[i] - b[i]); r += std::abs(a[i]); }
    if (d / std::max(r, 1e-30) > 1e-4) { printf("  D=%d fused/naive disagree\n", D); return false; }

    const float mf = bench_ms(fused), mn = bench_ms(naive);
    const double token_bytes = (double)total * D * sizeof(float);
    printf("  D=%-4d B=%d T=%d: fused %.4f ms (%.0f GB/s read) | naive %.4f ms | %.2fx\n",
           D, B, T, mf, token_bytes / (mf * 1e-3) / 1e9, mn, mn / mf);
    CK(hipFree(dx)); CK(hipFree(dw)); CK(hipFree(dout)); CK(hipFree(doff)); CK(hipFree(dtmp));
    return true;
}

int main(int argc, char **argv) {
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    bool ok = true;
    printf("pool_mean_rms_l2 correctness (fp64 oracle = ei_mean_pool_rms_l2):\n");
    ok &= correctness<256>();
    ok &= correctness<512>();
    ok &= correctness<768>();
    ok &= correctness<1024>();
    if (bench && ok) {
        printf("pool_mean_rms_l2 A/B fused vs naive-composed "
               "(HIP-event median, warmup 10 / iters 50):\n");
        // Representative embedding-server batch: many short sequences.
        ok &= perf<256>(2048, 64);
        ok &= perf<512>(2048, 64);
        ok &= perf<768>(2048, 64);
        ok &= perf<1024>(2048, 64);
        // Long-sequence stress (per-token reduction count grows).
        ok &= perf<768>(512, 256);
        ok &= perf<768>(8192, 16);
    }
    printf("%s\n", ok ? "ALL PASS" : "FAIL");
    return ok ? 0 : 1;
}
