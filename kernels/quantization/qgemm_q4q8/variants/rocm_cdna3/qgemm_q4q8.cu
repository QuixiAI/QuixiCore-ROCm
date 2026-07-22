/**
 * @file
 * @brief CDNA3 (gfx942) Q4_0-weight x Q8_0-activation integer GEMM.
 *
 * Shape/format-named port of the embeddinggemma.c `q4_q8_projection` path
 * (src/engine_rocm.hip: q4_q8_dot / q4_q8_projection_kernel). This is the
 * llama.cpp Q4_0 x Q8_0 route: 4-bit packed weights with a -8 zero point and a
 * per-block fp16 scale, dotted against int8 activations that carry a per-block
 * fp16 scale plus a precomputed int16 code sum. The integer dot uses the gfx942
 * `v_dot4` (sdot4) instruction; the -8 weight zero point is corrected once per
 * block via `-8 * q8.sum`.
 *
 * Distinct from `qgemm` (Q4_0 -> fp16 MFMA, fp16 activations) and from
 * `qgemm_int` (W8A8 / W2A8, 8-bit or BitNet weights). Contract, token-major:
 *   Wq  : (N, K) Q4_0 blocks   (row-major over output features N)
 *   Xq  : (M, K) activation_q8 blocks (token-major)
 *   Y   : (M, N) f32           Y[m,n] = sum_k dequant(Wq[n])_k * dequant(Xq[m])_k
 * K divisible by 32 (the Q4_0/Q8_0 block width).
 */
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CK(x) do { \
    hipError_t e = (x); \
    if (e != hipSuccess) { \
        printf("HIP %s @%d\n", hipGetErrorString(e), __LINE__); \
        exit(1); \
    } \
} while (0)

constexpr int kQK = 32;

struct __align__(2) block_q4_0 {
    __half d;
    uint8_t qs[16];
};

struct __align__(4) activation_q8 {
    __half d;
    int16_t sum;
    int8_t qs[32];
};

static_assert(sizeof(block_q4_0) == 18, "q4_0 block layout mismatch");
static_assert(sizeof(activation_q8) == 36, "activation q8 block layout mismatch");

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down(value, offset, 32);
    }
    return value;
}

__device__ __forceinline__ uint32_t load_u32_unaligned(const uint8_t *bytes) {
    return static_cast<uint32_t>(bytes[0]) |
           (static_cast<uint32_t>(bytes[1]) << 8) |
           (static_cast<uint32_t>(bytes[2]) << 16) |
           (static_cast<uint32_t>(bytes[3]) << 24);
}

__device__ __forceinline__ int sdot4(int a, int b, int accumulator) {
    return __builtin_amdgcn_sdot4(a, b, accumulator, false);
}

// One 32-lane wavefront-reduced Q4_0 x Q8_0 dot over a full row (embeddinggemma
// q4_q8_dot). Lane strides over blocks by 32; -8 zero point corrected per block.
__device__ __forceinline__ float q4_q8_dot(const block_q4_0 *weights,
                                           const activation_q8 *input,
                                           int n_cols, int lane) {
    const int blocks_per_row = n_cols / kQK;
    float sum = 0.0f;
    for (int block = lane; block < blocks_per_row; block += 32) {
        const block_q4_0 &q4 = weights[block];
        const activation_q8 &q8 = input[block];
        int dot = 0;
#pragma unroll
        for (int group = 0; group < 4; group++) {
            const uint32_t packed = load_u32_unaligned(q4.qs + group * 4);
            const int low = static_cast<int>(packed & 0x0f0f0f0fu);
            const int high = static_cast<int>((packed >> 4) & 0x0f0f0f0fu);
            const int q8_low = *reinterpret_cast<const int *>(q8.qs + group * 4);
            const int q8_high = *reinterpret_cast<const int *>(q8.qs + 16 + group * 4);
            dot = sdot4(low, q8_low, dot);
            dot = sdot4(high, q8_high, dot);
        }
        dot -= 8 * static_cast<int>(q8.sum);
        sum = fmaf(__half2float(q4.d) * __half2float(q8.d),
                   static_cast<float>(dot), sum);
    }
    return warp_sum(sum);
}

// Candidate: sdot4 integer path, one wavefront per (token, output row).
__global__ void q4q8_sdot4_kernel(const block_q4_0 *weights,
                                  const activation_q8 *input, float *output,
                                  uint32_t n_rows, uint32_t n_cols,
                                  uint32_t n_tokens) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const size_t task = static_cast<size_t>(blockIdx.x) * (blockDim.x >> 5) + warp;
    const size_t task_count = static_cast<size_t>(n_tokens) * n_rows;
    if (task >= task_count) return;
    const uint32_t token = static_cast<uint32_t>(task / n_rows);
    const uint32_t row = static_cast<uint32_t>(task -
        static_cast<size_t>(token) * n_rows);
    const uint32_t blocks_per_row = n_cols / kQK;
    const float sum = q4_q8_dot(
        weights + static_cast<size_t>(row) * blocks_per_row,
        input + static_cast<size_t>(token) * blocks_per_row,
        static_cast<int>(n_cols), lane);
    if (lane == 0) output[static_cast<size_t>(token) * n_rows + row] = sum;
}

// Baseline: float-dequant scalar dot (pre-sdot4 route) for the A/B comparison.
__global__ void q4q8_fdequant_kernel(const block_q4_0 *weights,
                                     const activation_q8 *input, float *output,
                                     uint32_t n_rows, uint32_t n_cols,
                                     uint32_t n_tokens) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const size_t task = static_cast<size_t>(blockIdx.x) * (blockDim.x >> 5) + warp;
    const size_t task_count = static_cast<size_t>(n_tokens) * n_rows;
    if (task >= task_count) return;
    const uint32_t token = static_cast<uint32_t>(task / n_rows);
    const uint32_t row = static_cast<uint32_t>(task -
        static_cast<size_t>(token) * n_rows);
    const uint32_t blocks_per_row = n_cols / kQK;
    const block_q4_0 *w = weights + static_cast<size_t>(row) * blocks_per_row;
    const activation_q8 *x = input + static_cast<size_t>(token) * blocks_per_row;
    float sum = 0.0f;
    for (uint32_t block = lane; block < blocks_per_row; block += 32) {
        const float wd = __half2float(w[block].d);
        const float xd = __half2float(x[block].d);
        float bsum = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; i++) {
            const uint8_t code = w[block].qs[i];
            const int lo = (code & 0x0f) - 8;
            const int hi = (code >> 4) - 8;
            bsum += static_cast<float>(lo) * static_cast<float>(x[block].qs[i]);
            bsum += static_cast<float>(hi) * static_cast<float>(x[block].qs[i + 16]);
        }
        sum = fmaf(wd * xd, bsum, sum);
    }
    sum = warp_sum(sum);
    if (lane == 0) output[static_cast<size_t>(token) * n_rows + row] = sum;
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
    hipEvent_t a, b;
    CK(hipEventCreate(&a));
    CK(hipEventCreate(&b));
    for (int i = 0; i < iters; ++i) {
        CK(hipEventRecord(a));
        fn();
        CK(hipEventRecord(b));
        CK(hipEventSynchronize(b));
        CK(hipEventElapsedTime(&t[i], a, b));
    }
    CK(hipEventDestroy(a));
    CK(hipEventDestroy(b));
    std::sort(t.begin(), t.end());
    return t[iters / 2];
}

// Quantize a float row-major (rows, K) buffer into Q4_0 blocks (row-major).
static std::vector<block_q4_0> quantize_q4_0(const std::vector<float>& src,
                                             int rows, int K) {
    const int bpr = K / kQK;
    std::vector<block_q4_0> out(static_cast<size_t>(rows) * bpr);
    for (int r = 0; r < rows; ++r) {
        for (int b = 0; b < bpr; ++b) {
            const float* base = src.data() + static_cast<size_t>(r) * K + b * kQK;
            float amax = 0.0f, vmax = 0.0f;
            for (int i = 0; i < kQK; ++i) {
                if (std::fabs(base[i]) > amax) { amax = std::fabs(base[i]); vmax = base[i]; }
            }
            const float d = vmax / -8.0f;
            const float id = d ? 1.0f / d : 0.0f;
            block_q4_0& blk = out[static_cast<size_t>(r) * bpr + b];
            blk.d = __float2half(d);
            for (int i = 0; i < 16; ++i) {
                const int x0 = std::min(15, (int)std::lround(base[i] * id + 8.0f));
                const int x1 = std::min(15, (int)std::lround(base[i + 16] * id + 8.0f));
                blk.qs[i] = (uint8_t)(std::max(0, x0) | (std::max(0, x1) << 4));
            }
        }
    }
    return out;
}

// Quantize a float row-major (rows, K) buffer into activation_q8 blocks.
static std::vector<activation_q8> quantize_q8_act(const std::vector<float>& src,
                                                  int rows, int K) {
    const int bpr = K / kQK;
    std::vector<activation_q8> out(static_cast<size_t>(rows) * bpr);
    for (int r = 0; r < rows; ++r) {
        for (int b = 0; b < bpr; ++b) {
            const float* base = src.data() + static_cast<size_t>(r) * K + b * kQK;
            float amax = 0.0f;
            for (int i = 0; i < kQK; ++i) amax = std::max(amax, std::fabs(base[i]));
            const float scale = amax / 127.0f;
            const float inv = scale ? 1.0f / scale : 0.0f;
            activation_q8& blk = out[static_cast<size_t>(r) * bpr + b];
            blk.d = __float2half(scale);
            int sum = 0;
            for (int i = 0; i < kQK; ++i) {
                int q = (int)std::lround(base[i] * inv);
                q = std::max(-128, std::min(127, q));
                blk.qs[i] = (int8_t)q;
                sum += q;
            }
            blk.sum = (int16_t)sum;
        }
    }
    return out;
}

// Host reference: dequantize both operands per block with the fp16-rounded
// scales and accumulate in fp64, matching the device per-block scaling order.
static std::vector<float> host_ref(const std::vector<block_q4_0>& W,
                                   const std::vector<activation_q8>& X,
                                   int N, int K, int M) {
    const int bpr = K / kQK;
    std::vector<float> Y(static_cast<size_t>(M) * N, 0.0f);
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int b = 0; b < bpr; ++b) {
                const block_q4_0& w = W[static_cast<size_t>(n) * bpr + b];
                const activation_q8& x = X[static_cast<size_t>(m) * bpr + b];
                int dot = 0, xsum = 0;
                for (int i = 0; i < 16; ++i) {
                    const uint8_t code = w.qs[i];
                    dot += ((code & 0x0f) - 8) * (int)x.qs[i];
                    dot += ((code >> 4) - 8) * (int)x.qs[i + 16];
                }
                (void)xsum;
                acc += (double)__half2float(w.d) * (double)__half2float(x.d) * (double)dot;
            }
            Y[static_cast<size_t>(m) * N + n] = (float)acc;
        }
    }
    return Y;
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
    printf("%-22s rel %.3e max %.3e %s\n", name, rel, maxe, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

static int run_case(int N, int K, int M, bool bench) {
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> Wf(static_cast<size_t>(N) * K), Xf(static_cast<size_t>(M) * K);
    for (auto& v : Wf) v = nd(rng) * 0.1f;
    for (auto& v : Xf) v = nd(rng);
    auto W = quantize_q4_0(Wf, N, K);
    auto X = quantize_q8_act(Xf, M, K);
    auto ref = host_ref(W, X, N, K, M);

    auto dW = dnew(W);
    auto dX = dnew(X);
    float *dYs = nullptr, *dYf = nullptr;
    CK(hipMalloc(&dYs, ref.size() * sizeof(float)));
    CK(hipMalloc(&dYf, ref.size() * sizeof(float)));

    const int threads = 256;
    const int warps = threads / 32;
    const size_t tasks = static_cast<size_t>(M) * N;
    const int blocks = (int)((tasks + warps - 1) / warps);
    auto runS = [&] { q4q8_sdot4_kernel<<<blocks, threads>>>(dW, dX, dYs, N, K, M); };
    auto runF = [&] { q4q8_fdequant_kernel<<<blocks, threads>>>(dW, dX, dYf, N, K, M); };
    runS(); runF();
    CK(hipDeviceSynchronize());
    CK(hipGetLastError());

    int rc = 0;
    rc |= check("q4q8_sdot4", d2h(dYs, ref.size()), ref, 2e-3);
    rc |= check("q4q8_fdequant(base)", d2h(dYf, ref.size()), ref, 2e-3);

    if (bench && rc == 0) {
        const float ts = bench_ms(runS);
        const float tf = bench_ms(runF);
        const double ops = 2.0 * (double)N * M * K;
        printf("q4q8 bench N=%d K=%d M=%d\n", N, K, M);
        printf("  fdequant(base) %.4f ms  %.2f TOPS\n", tf, ops / (tf * 1e-3) / 1e12);
        printf("  sdot4(cand)    %.4f ms  %.2f TOPS  (%.2fx)\n",
               ts, ops / (ts * 1e-3) / 1e12, tf / ts);
    }
    CK(hipFree(dW)); CK(hipFree(dX)); CK(hipFree(dYs)); CK(hipFree(dYf));
    return rc;
}

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    int rc = 0;
    // Correctness: small ragged + embeddinggemma FFN/QKV shapes.
    rc |= run_case(48, 768, 17, false);
    rc |= run_case(1152, 768, 64, false);   // FFN up/gate proj, 64-token prefill
    rc |= run_case(768, 768, 1, false);     // decode singleton
    if (bench) {
        rc |= run_case(1152, 768, 64, true);   // n_ff x n_embd prefill
        rc |= run_case(768, 1152, 64, true);   // down-proj shape
        rc |= run_case(768, 768, 1, true);     // decode
    }
    printf(rc == 0 ? "ALL PASS\n" : "FAIL\n");
    return rc;
}
