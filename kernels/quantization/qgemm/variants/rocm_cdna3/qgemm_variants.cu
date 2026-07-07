/**
 * @file
 * @brief Harness for CDNA3 qgemm_actorder and qgemm_blockscale.
 */
#include "qgemm_variants_kernels.cuh"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

using namespace tmq;

static int g_fail = 0;

#define HC(x) do { \
    hipError_t e = (x); \
    if (e != hipSuccess) { \
        printf("HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

template <typename T>
static T* dnew(const std::vector<T>& h) {
    T* d = nullptr;
    HC(hipMalloc(&d, h.size() * sizeof(T)));
    HC(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice));
    return d;
}

template <typename T>
static T* dzero(size_t n) {
    T* d = nullptr;
    HC(hipMalloc(&d, n * sizeof(T)));
    HC(hipMemset(d, 0, n * sizeof(T)));
    return d;
}

template <typename T>
static std::vector<T> d2h(const T* d, size_t n) {
    std::vector<T> h(n);
    HC(hipMemcpy(h.data(), d, n * sizeof(T), hipMemcpyDeviceToHost));
    return h;
}

static void report(const char* name, double err, double tol) {
    const bool ok = err <= tol;
    printf("%-42s %s  (rel err %.3e, tol %.1e)\n", name, ok ? "PASS" : "FAIL", err, tol);
    if (!ok) ++g_fail;
}

template <typename FN>
static float bench_ms(FN&& fn, int warmup = 10, int iters = 50) {
    for (int i = 0; i < warmup; i++) fn();
    HC(hipDeviceSynchronize());
    std::vector<float> t(iters);
    hipEvent_t a, b;
    HC(hipEventCreate(&a));
    HC(hipEventCreate(&b));
    for (int i = 0; i < iters; i++) {
        HC(hipEventRecord(a));
        fn();
        HC(hipEventRecord(b));
        HC(hipEventSynchronize(b));
        HC(hipEventElapsedTime(&t[i], a, b));
    }
    HC(hipEventDestroy(a));
    HC(hipEventDestroy(b));
    std::sort(t.begin(), t.end());
    return t[iters / 2];
}

static std::mt19937 g_rng(31);

template<typename FMT>
__global__ void dequant_to_fp16_variant(__half* out, const uint8_t* Wq, int N, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * K) return;
    int row = idx / K;
    int col = idx % K;
    const uint8_t* base = Wq + (size_t(row) * (K / FMT::block_k) + col / FMT::block_k) * FMT::block_bytes;
    out[idx] = __float2half(FMT::dequant(base, col % FMT::block_k));
}

int main() {
    const int M = 64;
    const int N = 128;
    const int K = 256;
    std::normal_distribution<float> nd(0.0f, 0.5f);

    std::vector<__half> X(size_t(M) * K);
    for (auto& x : X) x = __float2half(nd(g_rng));
    __half* dX = dnew(X);

    {
        const int BK = q4_0::block_k;
        const int BB = q4_0::block_bytes;
        const int bpr = K / BK;
        std::vector<uint8_t> Wq(size_t(N) * bpr * BB);
        std::uniform_int_distribution<int> ud(0, 255);
        for (auto& b : Wq) b = uint8_t(ud(g_rng));
        for (int n = 0; n < N; ++n) {
            for (int b = 0; b < bpr; ++b) {
                const __half s = __float2half(nd(g_rng) * 0.1f);
                memcpy(Wq.data() + (size_t(n) * bpr + b) * BB, &s, 2);
            }
        }
        uint8_t* dWq = dnew(Wq);
        __half* dWd = dzero<__half>(size_t(N) * K);
        dequant_to_fp16_variant<q4_0><<<(N * K + 255) / 256, 256>>>(dWd, dWq, N, K);
        HC(hipDeviceSynchronize());
        HC(hipGetLastError());
        auto Wd = d2h(dWd, size_t(N) * K);

        std::vector<int> perm(K);
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), g_rng);
        int* dperm = dnew(perm);

        float* dY = dzero<float>(size_t(M) * N);
        dim3 grid(unsigned(N / 16), unsigned(M / 16));
        auto launch = [&] {
            qgemm_actorder<q4_0><<<grid, 64>>>(dY, dX, dWq, dperm, M, N, K);
        };
        launch();
        HC(hipDeviceSynchronize());
        HC(hipGetLastError());
        auto Y = d2h(dY, size_t(M) * N);

        double worst = 0.0;
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                double acc = 0.0;
                for (int i = 0; i < K; ++i) {
                    acc += double(__half2float(Wd[size_t(n) * K + i])) *
                           double(__half2float(X[size_t(m) * K + perm[i]]));
                }
                const double s = std::max(1.0, std::abs(acc));
                worst = std::max(worst, std::abs(double(Y[size_t(m) * N + n]) - acc) / s);
            }
        }
        report("qgemm_actorder q4_0 (fp64 gathered)", worst, 5e-3);
        const float t = bench_ms(launch);
        printf("qgemm_actorder q4_0: %.3f ms  %.2f TFLOP/s\n",
               t, 2.0 * M * N * K / 1e12 / (t / 1e3));

        HC(hipFree(dWq));
        HC(hipFree(dWd));
        HC(hipFree(dperm));
        HC(hipFree(dY));
    }

    {
        const int BK = fp8_raw::block_k;
        const int BB = fp8_raw::block_bytes;
        const int bpr = K / BK;
        std::vector<uint8_t> Wq(size_t(N) * bpr * BB);
        std::uniform_int_distribution<int> ud(0, 255);
        for (auto& b : Wq) {
            uint8_t v = uint8_t(ud(g_rng));
            if ((v & 0x7F) == 0x7F) v &= 0xFE;
            b = v;
        }
        const int NT = N / 128;
        const int KT = K / 128;
        std::vector<__half> sc2d(size_t(NT) * KT);
        for (auto& s : sc2d) s = __float2half(0.02f + 0.1f * std::abs(nd(g_rng)));

        uint8_t* dWq = dnew(Wq);
        __half* dsc = dnew(sc2d);
        __half* dWd = dzero<__half>(size_t(N) * K);
        dequant_to_fp16_variant<fp8_raw><<<(N * K + 255) / 256, 256>>>(dWd, dWq, N, K);
        HC(hipDeviceSynchronize());
        HC(hipGetLastError());
        auto Wd = d2h(dWd, size_t(N) * K);

        float* dY = dzero<float>(size_t(M) * N);
        dim3 grid(unsigned(N / 16), unsigned(M / 16));
        auto launch = [&] {
            qgemm_blockscale<fp8_raw><<<grid, 64>>>(dY, dX, dWq, dsc, M, N, K);
        };
        launch();
        HC(hipDeviceSynchronize());
        HC(hipGetLastError());
        auto Y = d2h(dY, size_t(M) * N);

        double worst = 0.0;
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                double acc = 0.0;
                for (int i = 0; i < K; ++i) {
                    const __half ws = __float2half(__half2float(Wd[size_t(n) * K + i]) *
                                                   __half2float(sc2d[size_t(n / 128) * KT + i / 128]));
                    acc += double(__half2float(ws)) * double(__half2float(X[size_t(m) * K + i]));
                }
                const double s = std::max(1.0, std::abs(acc));
                worst = std::max(worst, std::abs(double(Y[size_t(m) * N + n]) - acc) / s);
            }
        }
        report("qgemm_blockscale fp8_raw (fp64 tiled)", worst, 5e-3);
        const float t = bench_ms(launch);
        printf("qgemm_blockscale fp8_raw: %.3f ms  %.2f TFLOP/s\n",
               t, 2.0 * M * N * K / 1e12 / (t / 1e3));

        HC(hipFree(dWq));
        HC(hipFree(dsc));
        HC(hipFree(dWd));
        HC(hipFree(dY));
    }

    HC(hipFree(dX));
    printf("\n%s (%d failures)\n", g_fail ? "FAILED" : "ALL PASS", g_fail);
    return g_fail ? 1 : 0;
}
