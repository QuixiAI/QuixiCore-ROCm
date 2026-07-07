/**
 * @file
 * @brief Focused CDNA3 performance harness for the elementwise/norm family.
 *
 * Times the row-reduction kernels (RMSNorm/LayerNorm/softmax fwd) and a couple
 * of flat elementwise kernels (GELU, add) with HIP events, reporting median
 * GB/s over the perf.md norm/softmax shape set. Also runs one CDNA3-specific
 * A/B: the faithful port launches 32-thread blocks (half a 64-wide wavefront)
 * per row; the candidate uses a full 64-lane wavefront per row. Correctness of
 * the family is covered by elementwise_test.out (fp64 oracle); this file only
 * measures.
 *
 * Build: hipcc -std=c++17 -O3 --offload-arch=gfx942 bench.cu -o bench.out
 * Run:   HIP_VISIBLE_DEVICES=0 ./bench.out
 */
#include "tm_elementwise_kernels.cuh"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <random>

#define HIPCHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    printf("HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

using namespace tme;

// A/B is the SHIPPED kernels (tm_elementwise_kernels.cuh) at 32 vs 64 threads:
// they are blockDim.x-aware (rowreduce_* over blockDim.x), so <<<M,32>>> is the
// half-wavefront baseline and <<<M,64>>> is the landed full-wavefront version.

// ---- timing ----------------------------------------------------------------
template <typename F>
static double time_ms(F launch, int warmup = 20, int iters = 100) {
    for (int i = 0; i < warmup; ++i) launch();
    HIPCHECK(hipDeviceSynchronize());
    std::vector<float> samples(iters);
    hipEvent_t a, b; HIPCHECK(hipEventCreate(&a)); HIPCHECK(hipEventCreate(&b));
    for (int i = 0; i < iters; ++i) {
        HIPCHECK(hipEventRecord(a));
        launch();
        HIPCHECK(hipEventRecord(b));
        HIPCHECK(hipEventSynchronize(b));
        HIPCHECK(hipEventElapsedTime(&samples[i], a, b));
    }
    hipEventDestroy(a); hipEventDestroy(b);
    std::sort(samples.begin(), samples.end());
    return samples[iters / 2];   // median
}

static std::mt19937 g_rng(7);
static std::vector<float> randv(size_t n) {
    std::uniform_real_distribution<float> d(-2.f, 2.f);
    std::vector<float> v(n); for (auto& x : v) x = d(g_rng); return v;
}
static float* dnew(const std::vector<float>& h) {
    float* d; HIPCHECK(hipMalloc(&d, h.size() * sizeof(float)));
    HIPCHECK(hipMemcpy(d, h.data(), h.size() * sizeof(float), hipMemcpyHostToDevice));
    return d;
}

int main() {
    hipDeviceProp_t p; HIPCHECK(hipGetDeviceProperties(&p, 0));
    printf("# device: %s  %s  warpSize=%d\n", p.name, p.gcnArchName, p.warpSize);
    printf("# bytes model = (2*M*D)*4 (read x + write o; w/b cached)\n\n");

    const int rows_set[] = {4096, 16384, 65536};
    const int hid_set[]  = {768, 1024, 2048, 4096, 8192};

    printf("%-10s %-7s | %-22s | %-22s | %-22s | %-22s\n",
           "rows", "hidden", "rms32 (ms / GB/s)", "rms64 (ms / GB/s)",
           "ln32 (ms / GB/s)", "ln64 (ms / GB/s)");
    for (int M : rows_set) for (int D : hid_set) {
        auto x = randv((size_t)M * D), w = randv(D), b = randv(D);
        float* dx = dnew(x); float* dw = dnew(w); float* db = dnew(b);
        float* o; HIPCHECK(hipMalloc(&o, (size_t)M * D * sizeof(float)));
        const double bytes = 2.0 * (double)M * D * 4.0;
        auto gbps = [&](double ms){ return bytes / (ms * 1e-3) / 1e9; };

        double r32 = time_ms([&]{ rms_norm_fwd<float><<<M, 32>>>(dx, dw, o, D, 1e-5f); });
        double r64 = time_ms([&]{ rms_norm_fwd<float><<<M, 64>>>(dx, dw, o, D, 1e-5f); });
        double l32 = time_ms([&]{ layernorm_fwd<float><<<M, 32>>>(dx, dw, db, o, D, 1e-5f); });
        double l64 = time_ms([&]{ layernorm_fwd<float><<<M, 64>>>(dx, dw, db, o, D, 1e-5f); });

        printf("%-10d %-7d | %8.4f / %7.1f | %8.4f / %7.1f | %8.4f / %7.1f | %8.4f / %7.1f\n",
               M, D, r32, gbps(r32), r64, gbps(r64), l32, gbps(l32), l64, gbps(l64));
        hipFree(dx); hipFree(dw); hipFree(db); hipFree(o);
    }

    // flat elementwise references (GELU fwd, add) at a large tensor
    {
        const long n = 64L * 1024 * 1024;
        auto x = randv(n), y = randv(n);
        float* dx = dnew(x); float* dy = dnew(y);
        float* o; HIPCHECK(hipMalloc(&o, n * sizeof(float)));
        const int TPB = 256; const unsigned grid = (n + TPB - 1) / TPB;
        double tg = time_ms([&]{ gelu_fwd<float><<<grid, TPB>>>(dx, o, n); });
        double ta = time_ms([&]{ add_ew<float><<<grid, TPB>>>(dx, dy, o, n); });
        printf("\ngelu_fwd  n=%ld : %.4f ms  %.1f GB/s (2*n*4)\n", n, tg, 2.0*n*4/(tg*1e-3)/1e9);
        printf("add_ew    n=%ld : %.4f ms  %.1f GB/s (3*n*4)\n", n, ta, 3.0*n*4/(ta*1e-3)/1e9);
        hipFree(dx); hipFree(dy); hipFree(o);
    }
    return 0;
}
