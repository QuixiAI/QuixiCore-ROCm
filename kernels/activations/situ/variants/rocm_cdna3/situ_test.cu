#include "hip/hip_runtime.h"
/**
 * @file
 * @brief fp64-oracle correctness for the SITU kernels.
 *
 * The oracle is computed in double on the host from the *same* input bytes the
 * GPU reads, so a mismatch is the kernel's and not the generator's.
 *
 * Tolerances are *relative*, because the output range depends on the betas:
 * with Kimi K3's beta=4 / linear_beta=25 the product reaches ~16, where bf16's
 * ~8 significand bits are worth ~3e-2 of absolute error all on their own. An
 * absolute bound would either pass everything at small beta or fail a correct
 * kernel at large beta. The bound per dtype is roughly one ulp of that dtype:
 * 2^-8 for bf16, 2^-11 for fp16, and fp32 compared tightly since the kernel
 * accumulates there.
 *
 * The masked kernel is checked for two things the dense one cannot express:
 * that a zero-token expert is skipped entirely, and that rows past an expert's
 * token count keep whatever was in `out` beforehand. Both are how the MoE path
 * avoids paying for padding, and both fail silently if the guard is wrong --
 * the model still generates, just from garbage rows.
 *
 *   make test
 */
#include "tm_situ_kernels.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace quixicore::activations;

#define CK(x)                                                            \
    do {                                                                 \
        hipError_t e = (x);                                              \
        if (e != hipSuccess) {                                           \
            printf("HIP %s @%d\n", hipGetErrorString(e), __LINE__);      \
            exit(1);                                                     \
        }                                                                \
    } while (0)

static int g_fail = 0;

static double situ_ref(double g, double u, double beta, double linear_beta) {
    const double gate_out = beta * std::tanh(g / beta) * (1.0 / (1.0 + std::exp(-g)));
    const double up_out =
        linear_beta > 0.0 ? linear_beta * std::tanh(u / linear_beta) : u;
    return gate_out * up_out;
}

/// Deterministic, spread over roughly [-4, 4] so both tanh tails are exercised.
static float sample(int i) {
    return 4.0f * std::sin(0.7f * (float)i + 0.3f) * std::cos(0.11f * (float)i);
}

template <typename T>
static void run_dense(const char* label, double tol, float beta,
                      float linear_beta) {
    const int rows = 37, d = 1024;
    std::vector<float> host(rows * 2 * d);
    for (size_t i = 0; i < host.size(); ++i) host[i] = sample((int)i);

    std::vector<T> in(host.size());
    for (size_t i = 0; i < host.size(); ++i) in[i] = situ_ft<T>(host[i]);
    // Read back what the GPU will actually see, post-narrowing.
    std::vector<float> seen(host.size());
    for (size_t i = 0; i < host.size(); ++i) seen[i] = situ_tf<T>(in[i]);

    T *d_in, *d_out;
    CK(hipMalloc(&d_in, in.size() * sizeof(T)));
    CK(hipMalloc(&d_out, (size_t)rows * d * sizeof(T)));
    CK(hipMemcpy(d_in, in.data(), in.size() * sizeof(T), hipMemcpyHostToDevice));
    situ_and_mul<T><<<rows, 256>>>(d_out, d_in, d, beta, linear_beta);
    CK(hipDeviceSynchronize());
    std::vector<T> out((size_t)rows * d);
    CK(hipMemcpy(out.data(), d_out, out.size() * sizeof(T), hipMemcpyDeviceToHost));

    double worst = 0.0;
    for (int r = 0; r < rows; ++r) {
        for (int i = 0; i < d; ++i) {
            const double want = situ_ref(seen[(size_t)r * 2 * d + i],
                                         seen[(size_t)r * 2 * d + d + i], beta,
                                         linear_beta);
            const double got = situ_tf<T>(out[(size_t)r * d + i]);
            worst = std::fmax(worst,
                              std::fabs(got - want) / std::fmax(std::fabs(want), 1e-3));
        }
    }
    const bool ok = worst <= tol;
    if (!ok) g_fail = 1;
    printf("  %-28s beta=%-5g lb=%-6g worst_rel=%.3e  %s\n", label, beta, linear_beta,
           worst, ok ? "ok" : "FAIL");
    CK(hipFree(d_in));
    CK(hipFree(d_out));
}

template <typename T>
static void run_masked(const char* label, double tol) {
    const int E = 5, Tmax = 7, d = 512;
    const float beta = 1.5f, linear_beta = 3.0f;
    const int counts[E] = {0, 3, 7, 1, 5};  // includes an empty expert

    std::vector<float> host((size_t)E * Tmax * 2 * d);
    for (size_t i = 0; i < host.size(); ++i) host[i] = sample((int)i);
    std::vector<T> in(host.size());
    for (size_t i = 0; i < host.size(); ++i) in[i] = situ_ft<T>(host[i]);
    std::vector<float> seen(host.size());
    for (size_t i = 0; i < host.size(); ++i) seen[i] = situ_tf<T>(in[i]);

    const size_t out_n = (size_t)E * Tmax * d;
    std::vector<T> out_init(out_n, situ_ft<T>(-7.0f));  // sentinel

    T *d_in, *d_out;
    int* d_cnt;
    CK(hipMalloc(&d_in, in.size() * sizeof(T)));
    CK(hipMalloc(&d_out, out_n * sizeof(T)));
    CK(hipMalloc(&d_cnt, E * sizeof(int)));
    CK(hipMemcpy(d_in, in.data(), in.size() * sizeof(T), hipMemcpyHostToDevice));
    CK(hipMemcpy(d_out, out_init.data(), out_n * sizeof(T), hipMemcpyHostToDevice));
    CK(hipMemcpy(d_cnt, counts, sizeof(counts), hipMemcpyHostToDevice));

    constexpr int block = 256;
    dim3 grid((d + block - 1) / block, E);
    masked_situ_and_mul<T><<<grid, block>>>(d_out, d_in, d_cnt, Tmax, d, beta,
                                            linear_beta);
    CK(hipDeviceSynchronize());
    std::vector<T> out(out_n);
    CK(hipMemcpy(out.data(), d_out, out_n * sizeof(T), hipMemcpyDeviceToHost));

    double worst = 0.0;
    long long touched_dead = 0;
    for (int e = 0; e < E; ++e) {
        for (int t = 0; t < Tmax; ++t) {
            for (int i = 0; i < d; ++i) {
                const size_t row = ((size_t)e * Tmax + t);
                const double got = situ_tf<T>(out[row * d + i]);
                if (t >= counts[e]) {
                    if (got != -7.0) ++touched_dead;
                    continue;
                }
                const double want =
                    situ_ref(seen[row * 2 * d + i], seen[row * 2 * d + d + i],
                             beta, linear_beta);
                worst = std::fmax(worst,
                                  std::fabs(got - want) / std::fmax(std::fabs(want), 1e-3));
            }
        }
    }
    const bool ok = worst <= tol && touched_dead == 0;
    if (!ok) g_fail = 1;
    printf("  %-28s worst_rel=%.3e  dead-rows-written=%lld  %s\n", label, worst,
           touched_dead, ok ? "ok" : "FAIL");
    CK(hipFree(d_in));
    CK(hipFree(d_out));
    CK(hipFree(d_cnt));
}

int main() {
    printf("SITU (SituGLU) vs fp64 oracle\n");
    run_dense<float>("dense fp32", 1e-6, 1.0f, -1.0f);
    run_dense<float>("dense fp32", 1e-6, 4.0f, 25.0f);  // Kimi K3's values
    run_dense<__hip_bfloat16>("dense bf16", 8e-3, 4.0f, 25.0f);
    run_dense<__half>("dense fp16", 1e-3, 4.0f, 25.0f);
    run_masked<float>("masked fp32", 1e-6);
    run_masked<__hip_bfloat16>("masked bf16", 8e-3);
    printf("\n%s\n", g_fail ? "FAILED" : "ALL PASS");
    return g_fail;
}
