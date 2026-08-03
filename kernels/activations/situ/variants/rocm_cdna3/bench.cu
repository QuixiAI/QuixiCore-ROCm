#include "hip/hip_runtime.h"
/**
 * @file
 * @brief SITU perf: achieved bandwidth against the MI300X roofline.
 *
 * This kernel is memory bound by construction -- it reads 2d and writes d per
 * row with a handful of transcendentals in between -- so the number that
 * matters is bytes/s, not FLOP/s. Reported as a fraction of 5.3 TB/s (MI300X
 * HBM3 peak) so a regression is visible without knowing the absolute figure.
 *
 * Shapes are the ones Kimi K3 actually runs: the dense MLP at 33792 and the
 * routed experts at 3072, each with the released beta/linear_beta.
 *
 *   make bench
 */
#include "tm_situ_kernels.cuh"

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace quixicore::activations;

#define CK(x)                                                       \
    do {                                                            \
        hipError_t e = (x);                                         \
        if (e != hipSuccess) {                                      \
            printf("HIP %s @%d\n", hipGetErrorString(e), __LINE__); \
            exit(1);                                                \
        }                                                           \
    } while (0)

static constexpr double kPeakBytesPerSec = 5.3e12;  // MI300X HBM3

template <typename T>
static void bench_dense(const char* label, int rows, int d) {
    const float beta = 4.0f, linear_beta = 25.0f;
    T *in, *out;
    CK(hipMalloc(&in, (size_t)rows * 2 * d * sizeof(T)));
    CK(hipMalloc(&out, (size_t)rows * d * sizeof(T)));
    CK(hipMemset(in, 0, (size_t)rows * 2 * d * sizeof(T)));

    for (int i = 0; i < 20; ++i)
        situ_and_mul<T><<<rows, 256>>>(out, in, d, beta, linear_beta);
    CK(hipDeviceSynchronize());

    hipEvent_t t0, t1;
    CK(hipEventCreate(&t0));
    CK(hipEventCreate(&t1));
    const int iters = 200;
    CK(hipEventRecord(t0));
    for (int i = 0; i < iters; ++i)
        situ_and_mul<T><<<rows, 256>>>(out, in, d, beta, linear_beta);
    CK(hipEventRecord(t1));
    CK(hipEventSynchronize(t1));
    float ms = 0.0f;
    CK(hipEventElapsedTime(&ms, t0, t1));

    const double bytes = (double)rows * 3.0 * d * sizeof(T);
    const double bw = bytes * iters / (ms * 1e-3);
    printf("  %-34s %8.3f us  %7.2f GB/s  %5.1f%% of peak\n", label,
           ms * 1000.0 / iters, bw / 1e9, 100.0 * bw / kPeakBytesPerSec);
    CK(hipFree(in));
    CK(hipFree(out));
    CK(hipEventDestroy(t0));
    CK(hipEventDestroy(t1));
}

int main() {
    printf("SITU bandwidth (MI300X peak %.1f TB/s)\n", kPeakBytesPerSec / 1e12);
    printf("-- dense MLP width (K3 intermediate_size 33792) --\n");
    bench_dense<__hip_bfloat16>("bf16 rows=1     d=33792", 1, 33792);
    bench_dense<__hip_bfloat16>("bf16 rows=512   d=33792", 512, 33792);
    printf("-- routed expert width (K3 moe_intermediate_size 3072) --\n");
    bench_dense<__hip_bfloat16>("bf16 rows=4096  d=3072", 4096, 3072);
    bench_dense<__hip_bfloat16>("bf16 rows=16384 d=3072", 16384, 3072);
    return 0;
}
