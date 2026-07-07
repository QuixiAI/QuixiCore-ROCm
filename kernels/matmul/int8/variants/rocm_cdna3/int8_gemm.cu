/**
 * @file
 * @brief CDNA3 dense int8 GEMM: C[M,N] = A[M,K] @ B[N,K]^T, int32 output.
 */
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define HC(x) do { \
    hipError_t e = (x); \
    if (e != hipSuccess) { \
        printf("HIP %s @%d\n", hipGetErrorString(e), __LINE__); \
        exit(1); \
    } \
} while (0)

__global__ void int8_gemm_scalar(const int8_t* A, const int8_t* B, int32_t* C,
                                 int M, int N, int K) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) return;
    int acc = 0;
    for (int k = 0; k < K; ++k)
        acc += int(A[size_t(m) * K + k]) * int(B[size_t(n) * K + k]);
    C[size_t(m) * N + n] = acc;
}

__global__ void int8_gemm_sdot4(const int8_t* A, const int8_t* B, int32_t* C,
                                int M, int N, int K) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) return;
    int acc = 0;
    for (int k = 0; k < K; k += 4) {
        const uint32_t av = *reinterpret_cast<const uint32_t*>(A + size_t(m) * K + k);
        const uint32_t bv = *reinterpret_cast<const uint32_t*>(B + size_t(n) * K + k);
        acc = __builtin_amdgcn_sdot4(int(av), int(bv), acc, false);
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

static std::vector<int32_t> reference(const std::vector<int8_t>& A,
                                      const std::vector<int8_t>& B,
                                      int M, int N, int K) {
    std::vector<int32_t> C(size_t(M) * N);
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            int acc = 0;
            for (int k = 0; k < K; ++k)
                acc += int(A[size_t(m) * K + k]) * int(B[size_t(n) * K + k]);
            C[size_t(m) * N + n] = acc;
        }
    }
    return C;
}

int main(int argc, char** argv) {
    const int M = argc > 1 ? atoi(argv[1]) : 256;
    const int N = argc > 2 ? atoi(argv[2]) : 256;
    const int K = argc > 3 ? atoi(argv[3]) : 512;
    if (K % 4 != 0) {
        printf("K must be divisible by 4 for sdot4 path\n");
        return 2;
    }

    std::mt19937 rng(17);
    std::uniform_int_distribution<int> dist(-8, 8);
    std::vector<int8_t> A(size_t(M) * K), B(size_t(N) * K);
    for (auto& x : A) x = int8_t(dist(rng));
    for (auto& x : B) x = int8_t(dist(rng));
    auto ref = reference(A, B, M, N, K);

    int8_t* dA = dnew(A);
    int8_t* dB = dnew(B);
    int32_t* dC0 = nullptr;
    int32_t* dC1 = nullptr;
    HC(hipMalloc(&dC0, ref.size() * sizeof(int32_t)));
    HC(hipMalloc(&dC1, ref.size() * sizeof(int32_t)));

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    auto scalar = [&] { int8_gemm_scalar<<<grid, block>>>(dA, dB, dC0, M, N, K); };
    auto sdot4 = [&] { int8_gemm_sdot4<<<grid, block>>>(dA, dB, dC1, M, N, K); };

    scalar();
    sdot4();
    HC(hipDeviceSynchronize());
    HC(hipGetLastError());
    auto c0 = d2h(dC0, ref.size());
    auto c1 = d2h(dC1, ref.size());
    int fail = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (c0[i] != ref[i] || c1[i] != ref[i]) {
            if (fail < 5) printf("mismatch %zu ref=%d scalar=%d sdot4=%d\n", i, ref[i], c0[i], c1[i]);
            ++fail;
        }
    }
    printf("int8_gemm exactness: %s (%d mismatches)\n", fail ? "FAIL" : "PASS", fail);
    if (fail) return 1;

    const float ts = bench_ms(scalar);
    const float td = bench_ms(sdot4);
    const double ops = 2.0 * M * N * K;
    printf("== int8_gemm M=%d N=%d K=%d\n", M, N, K);
    printf("scalar: %.3f ms  %.2f TOPS\n", ts, ops / (ts * 1e-3) / 1e12);
    printf("sdot4 : %.3f ms  %.2f TOPS  keep=%s\n", td, ops / (td * 1e-3) / 1e12,
           td <= ts ? "sdot4" : "scalar");

    HC(hipFree(dA));
    HC(hipFree(dB));
    HC(hipFree(dC0));
    HC(hipFree(dC1));
    return 0;
}
