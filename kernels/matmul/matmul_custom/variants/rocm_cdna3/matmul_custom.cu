/**
 * @file
 * @brief Correctness-first CDNA3 port of Metal matmul_custom.
 *
 * Contract: C = A(N,K) @ B(K,M), dtype f32 or bf16, with the Metal fast-path
 * shape constraints N%32, M%32, K%16. This implementation supports tails too,
 * but the harness validates the contract-aligned shapes.
 */
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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

template <typename T> __device__ __forceinline__ float tf(T x);
template <> __device__ __forceinline__ float tf<float>(float x) { return x; }
template <> __device__ __forceinline__ float tf<__hip_bfloat16>(__hip_bfloat16 x) { return __bfloat162float(x); }
template <typename T> __device__ __forceinline__ T ft(float x);
template <> __device__ __forceinline__ float ft<float>(float x) { return x; }
template <> __device__ __forceinline__ __hip_bfloat16 ft<__hip_bfloat16>(float x) { return __float2bfloat16(x); }

template <typename T>
__global__ void matmul_custom_direct(T* __restrict__ C,
                                     const T* __restrict__ A,
                                     const T* __restrict__ B,
                                     int N,
                                     int K,
                                     int M) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= N || col >= M) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) acc += tf(A[size_t(row) * K + k]) * tf(B[size_t(k) * M + col]);
    C[size_t(row) * M + col] = ft<T>(acc);
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
static float bench_ms(FN&& fn, int warmup = 5, int iters = 30) {
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

static std::vector<__hip_bfloat16> bf16v(const std::vector<float>& x) {
    std::vector<__hip_bfloat16> y(x.size());
    for (size_t i = 0; i < x.size(); ++i) y[i] = __float2bfloat16(x[i]);
    return y;
}

static float hbf(__hip_bfloat16 x) { return __bfloat162float(x); }

template <typename T>
static int run_one(const char* name, const std::vector<float>& Af, const std::vector<float>& Bf,
                   int N, int K, int M, bool bench);

template <>
int run_one<float>(const char* name, const std::vector<float>& Af, const std::vector<float>& Bf,
                   int N, int K, int M, bool bench) {
    auto dA = dnew(Af);
    auto dB = dnew(Bf);
    float* dC = nullptr;
    CK(hipMalloc(&dC, size_t(N) * M * sizeof(float)));
    dim3 block(16, 16), grid((M + 15) / 16, (N + 15) / 16);
    auto launch = [&] { matmul_custom_direct<float><<<grid, block>>>(dC, dA, dB, N, K, M); };
    launch();
    CK(hipDeviceSynchronize());
    CK(hipGetLastError());
    auto C = d2h(dC, size_t(N) * M);
    double sad = 0.0, sref = 0.0, maxe = 0.0;
    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < M; ++c) {
            double ref = 0.0;
            for (int k = 0; k < K; ++k) ref += double(Af[size_t(r) * K + k]) * Bf[size_t(k) * M + c];
            const double d = std::abs(C[size_t(r) * M + c] - ref);
            sad += d;
            sref += std::abs(ref);
            maxe = std::max(maxe, d);
        }
    }
    const double rel = sad / std::max(sref, 1e-30);
    const bool ok = rel < 2e-6;
    printf("%s f32 rel %.3e max %.3e %s\n", name, rel, maxe, ok ? "PASS" : "FAIL");
    if (bench && ok) {
        const float ms = bench_ms(launch);
        const double ops = 2.0 * N * M * K;
        printf("%s f32 bench N=%d K=%d M=%d %.4f ms %.2f TFLOP/s\n", name, N, K, M, ms,
               ops / (ms * 1e-3) / 1e12);
    }
    CK(hipFree(dA));
    CK(hipFree(dB));
    CK(hipFree(dC));
    return ok ? 0 : 1;
}

template <>
int run_one<__hip_bfloat16>(const char* name, const std::vector<float>& Af, const std::vector<float>& Bf,
                            int N, int K, int M, bool bench) {
    auto Ab = bf16v(Af);
    auto Bb = bf16v(Bf);
    auto dA = dnew(Ab);
    auto dB = dnew(Bb);
    __hip_bfloat16* dC = nullptr;
    CK(hipMalloc(&dC, size_t(N) * M * sizeof(__hip_bfloat16)));
    dim3 block(16, 16), grid((M + 15) / 16, (N + 15) / 16);
    auto launch = [&] { matmul_custom_direct<__hip_bfloat16><<<grid, block>>>(dC, dA, dB, N, K, M); };
    launch();
    CK(hipDeviceSynchronize());
    CK(hipGetLastError());
    auto C = d2h(dC, size_t(N) * M);
    double sad = 0.0, sref = 0.0, maxe = 0.0;
    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < M; ++c) {
            double ref = 0.0;
            for (int k = 0; k < K; ++k) ref += double(hbf(Ab[size_t(r) * K + k])) * hbf(Bb[size_t(k) * M + c]);
            const double d = std::abs(double(hbf(C[size_t(r) * M + c])) - ref);
            sad += d;
            sref += std::abs(ref);
            maxe = std::max(maxe, d);
        }
    }
    const double rel = sad / std::max(sref, 1e-30);
    const bool ok = rel < 6e-3;
    printf("%s bf16 rel %.3e max %.3e %s\n", name, rel, maxe, ok ? "PASS" : "FAIL");
    if (bench && ok) {
        const float ms = bench_ms(launch);
        const double ops = 2.0 * N * M * K;
        printf("%s bf16 bench N=%d K=%d M=%d %.4f ms %.2f TFLOP/s\n", name, N, K, M, ms,
               ops / (ms * 1e-3) / 1e12);
    }
    CK(hipFree(dA));
    CK(hipFree(dB));
    CK(hipFree(dC));
    return ok ? 0 : 1;
}

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    const int N = bench ? 512 : 64;
    const int K = bench ? 512 : 128;
    const int M = bench ? 512 : 96;
    std::mt19937 rng(41);
    std::normal_distribution<float> nd(0.0f, 0.12f);
    std::vector<float> A(size_t(N) * K), B(size_t(K) * M);
    for (auto& x : A) x = nd(rng);
    for (auto& x : B) x = nd(rng);
    int rc = 0;
    rc |= run_one<float>("matmul_custom", A, B, N, K, M, bench);
    rc |= run_one<__hip_bfloat16>("matmul_custom", A, B, N, K, M, bench);
    return rc;
}
