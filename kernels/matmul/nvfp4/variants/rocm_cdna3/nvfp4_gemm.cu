/**
 * @file
 * @brief Correctness-first dense NVFP4 GEMM for CDNA3.
 */
#include <cstring>
#include "../../../../quantization/qgemm/variants/rocm_cdna3/quant_formats.cuh"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define HC(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    printf("HIP %s @%d\n", hipGetErrorString(e), __LINE__); exit(1); } } while (0)

__global__ void dequant_nvfp4(const uint8_t* Q, float* F, int rows, int K) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * K) return;
    const int r = idx / K;
    const int k = idx % K;
    const int bpr = K / tmq::nvfp4::block_k;
    const uint8_t* base = Q + (size_t(r) * bpr + k / tmq::nvfp4::block_k) * tmq::nvfp4::block_bytes;
    F[idx] = tmq::nvfp4::dequant(base, k % tmq::nvfp4::block_k);
}

__global__ void fp32_gemm_abt(const float* A, const float* B, float* C, int M, int N, int K) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) acc += A[size_t(m) * K + k] * B[size_t(n) * K + k];
    C[size_t(m) * N + n] = acc;
}

__global__ void nvfp4_gemm_fused(const uint8_t* A, const uint8_t* B, float* C, int M, int N, int K) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) return;
    const int bpr = K / tmq::nvfp4::block_k;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        const uint8_t* ab = A + (size_t(m) * bpr + k / tmq::nvfp4::block_k) * tmq::nvfp4::block_bytes;
        const uint8_t* bb = B + (size_t(n) * bpr + k / tmq::nvfp4::block_k) * tmq::nvfp4::block_bytes;
        acc += tmq::nvfp4::dequant(ab, k % tmq::nvfp4::block_k) *
               tmq::nvfp4::dequant(bb, k % tmq::nvfp4::block_k);
    }
    C[size_t(m) * N + n] = acc;
}

template <typename T>
static T* dnew(const std::vector<T>& h) {
    T* d = nullptr;
    HC(hipMalloc(&d, h.size() * sizeof(T)));
    HC(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice));
    return d;
}

template <typename T>
static std::vector<T> d2h(const T* d, size_t n) {
    std::vector<T> h(n);
    HC(hipMemcpy(h.data(), d, n * sizeof(T), hipMemcpyDeviceToHost));
    return h;
}

template <typename FN>
static float bench_ms(FN&& fn, int warmup = 5, int iters = 30) {
    for (int i = 0; i < warmup; i++) fn();
    HC(hipDeviceSynchronize());
    std::vector<float> t(iters);
    hipEvent_t a, b;
    HC(hipEventCreate(&a));
    HC(hipEventCreate(&b));
    for (int i = 0; i < iters; i++) {
        HC(hipEventRecord(a)); fn(); HC(hipEventRecord(b)); HC(hipEventSynchronize(b));
        HC(hipEventElapsedTime(&t[i], a, b));
    }
    HC(hipEventDestroy(a));
    HC(hipEventDestroy(b));
    std::sort(t.begin(), t.end());
    return t[iters / 2];
}

static float e2m1_host(unsigned nib) {
    const float mag = (nib & 0x6) ? std::ldexp(1.0f + float(nib & 1) * 0.5f, int((nib >> 1) & 0x3) - 1)
                                  : ((nib & 1) ? 0.5f : 0.0f);
    return (nib & 0x8) ? -mag : mag;
}

static float nvfp4_host(const uint8_t* base, int col) {
    const uint8_t* qs = base + 1;
    const unsigned nib = (col < 8) ? (qs[col] & 0x0F) : (qs[col - 8] >> 4);
    return tmq::e4m3_decode_host(base[0]) * e2m1_host(nib);
}

static std::vector<uint8_t> random_nvfp4(int rows, int K, std::mt19937& rng) {
    const int bpr = K / tmq::nvfp4::block_k;
    std::vector<uint8_t> q(size_t(rows) * bpr * tmq::nvfp4::block_bytes);
    std::uniform_int_distribution<int> scale_pick(1, 4);
    std::uniform_int_distribution<int> nib(0, 15);
    for (int r = 0; r < rows; ++r) {
        for (int b = 0; b < bpr; ++b) {
            uint8_t* base = q.data() + (size_t(r) * bpr + b) * tmq::nvfp4::block_bytes;
            base[0] = tmq::e4m3_encode(0.125f * float(scale_pick(rng)));
            for (int i = 0; i < 8; ++i) base[1 + i] = uint8_t(nib(rng) | (nib(rng) << 4));
        }
    }
    return q;
}

static std::vector<float> host_ref(const std::vector<uint8_t>& A,
                                   const std::vector<uint8_t>& B,
                                   int M, int N, int K) {
    const int bpr = K / tmq::nvfp4::block_k;
    std::vector<float> C(size_t(M) * N);
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k) {
                const uint8_t* ab = A.data() + (size_t(m) * bpr + k / tmq::nvfp4::block_k) * tmq::nvfp4::block_bytes;
                const uint8_t* bb = B.data() + (size_t(n) * bpr + k / tmq::nvfp4::block_k) * tmq::nvfp4::block_bytes;
                acc += double(nvfp4_host(ab, k % tmq::nvfp4::block_k)) *
                       double(nvfp4_host(bb, k % tmq::nvfp4::block_k));
            }
            C[size_t(m) * N + n] = float(acc);
        }
    }
    return C;
}

static int check_shape(int M, int N, int K, bool timing) {
    std::mt19937 rng(29);
    auto Ah = random_nvfp4(M, K, rng);
    auto Bh = random_nvfp4(N, K, rng);
    uint8_t* dA = dnew(Ah);
    uint8_t* dB = dnew(Bh);
    float *dC = nullptr, *dAf = nullptr, *dBf = nullptr, *dCb = nullptr;
    HC(hipMalloc(&dC, size_t(M) * N * sizeof(float)));
    HC(hipMalloc(&dAf, size_t(M) * K * sizeof(float)));
    HC(hipMalloc(&dBf, size_t(N) * K * sizeof(float)));
    HC(hipMalloc(&dCb, size_t(M) * N * sizeof(float)));
    dim3 block(16, 16), grid((N + 15) / 16, (M + 15) / 16);
    auto fused = [&] { nvfp4_gemm_fused<<<grid, block>>>(dA, dB, dC, M, N, K); };
    auto explicit_dequant = [&] {
        dequant_nvfp4<<<(M * K + 255) / 256, 256>>>(dA, dAf, M, K);
        dequant_nvfp4<<<(N * K + 255) / 256, 256>>>(dB, dBf, N, K);
        fp32_gemm_abt<<<grid, block>>>(dAf, dBf, dCb, M, N, K);
    };
    fused();
    explicit_dequant();
    HC(hipDeviceSynchronize());
    HC(hipGetLastError());
    auto got = d2h(dC, size_t(M) * N);
    auto got_explicit = d2h(dCb, size_t(M) * N);
    auto ref = host_ref(Ah, Bh, M, N, K);
    double worst = 0.0, worst_explicit = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double s = std::max(1.0, std::abs(double(ref[i])));
        worst = std::max(worst, std::abs(double(got[i]) - ref[i]) / s);
        worst_explicit = std::max(worst_explicit, std::abs(double(got_explicit[i]) - ref[i]) / s);
    }
    const bool ok = worst < 1e-5 && worst_explicit < 1e-5;
    printf("nvfp4_gemm M=%d N=%d K=%d %s (fused rel %.3e explicit rel %.3e)\n",
           M, N, K, ok ? "PASS" : "FAIL", worst, worst_explicit);
    if (timing && ok) {
        const float tf = bench_ms(fused);
        const float te = bench_ms(explicit_dequant);
        const double ops = 2.0 * M * N * K;
        printf("fused          : %.3f ms  %.2f TFLOP/s\n", tf, ops / (tf * 1e-3) / 1e12);
        printf("explicit+fp32  : %.3f ms  %.2f TFLOP/s  keep=%s\n",
               te, ops / (te * 1e-3) / 1e12, tf <= te ? "fused" : "explicit+fp32");
    }
    HC(hipFree(dA)); HC(hipFree(dB)); HC(hipFree(dC)); HC(hipFree(dAf)); HC(hipFree(dBf)); HC(hipFree(dCb));
    return ok ? 0 : 1;
}

int main(int argc, char** argv) {
    const int M = argc > 1 ? atoi(argv[1]) : 64;
    const int N = argc > 2 ? atoi(argv[2]) : 64;
    const int K = argc > 3 ? atoi(argv[3]) : 256;
    if (K % tmq::nvfp4::block_k != 0) {
        printf("K must be divisible by %d\n", tmq::nvfp4::block_k);
        return 2;
    }
    return check_shape(M, N, K, true);
}
