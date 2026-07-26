#include "hip/hip_runtime.h"
/**
 * @file
 * @brief CDNA3 (gfx942) TurboQuant query transform.
 *
 * Semantic source: ../QuixiCore-CPU/kernels/attention/attention_turboquant.cpp
 * (turboquant_query_transform :223, fwht :42).
 *
 *   transformed[d] = q[d] * signs[d]
 *   fwht(transformed, head_size)            <- UNNORMALIZED, in place
 *   transformed[d] *= 1 / sqrt(head_size)
 *
 * head_size is 64, 128 or 256 only. `signs` is one vector of length head_size
 * shared by every row, and non-finite entries in it reject the call.
 *
 * ## The FWHT is unnormalized and the scale is applied ONCE, afterwards
 *
 * The butterfly is the plain a+b / a-b recurrence with no per-stage 1/sqrt(2).
 * Folding the normalization into each stage is the usual "orthonormal" variant
 * and gives a different answer for the same input -- the reference normalizes
 * exactly once at the end, so that is what happens here.
 *
 * ## This one IS bit-exact
 *
 * Every butterfly within a stage reads a disjoint pair and writes back to those
 * same two slots, so the result does not depend on the order threads execute
 * them. Unlike the reduction-based kernels in this tree, the harness holds it to
 * exact equality with the host -- a tolerance here would hide a real bug.
 *
 * Build: make      Test: make test
 */
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

namespace tq {

// One block per (row, head); blockDim == head_size. Shared holds the vector
// being transformed in place.
template <int HEAD_SIZE>
__global__ void k_turboquant_query_transform(const float *__restrict__ q,
                                             const float *__restrict__ signs,
                                             float *__restrict__ transformed,
                                             long long items) {
    __shared__ float buf[HEAD_SIZE];
    const long long item = blockIdx.x;
    if (item >= items) return;
    const int d = threadIdx.x;
    const long long base = item * HEAD_SIZE;

    buf[d] = q[base + d] * signs[d];
    __syncthreads();

    // Unnormalized FWHT. Stage `width` pairs index i with i+width inside each
    // 2*width-wide span; HEAD_SIZE/2 butterflies per stage.
    #pragma unroll
    for (int width = 1; width < HEAD_SIZE; width *= 2) {
        float a = 0.0f, b = 0.0f;
        int ia = 0, ib = 0;
        const bool active = d < HEAD_SIZE / 2;
        if (active) {
            const int span = (d / width) * 2 * width;
            const int item_in = d % width;
            ia = span + item_in;
            ib = span + width + item_in;
            a = buf[ia];
            b = buf[ib];
        }
        __syncthreads();
        if (active) {
            buf[ia] = a + b;
            buf[ib] = a - b;
        }
        __syncthreads();
    }

    transformed[base + d] = buf[d] * rsqrtf((float)HEAD_SIZE);
}

}  // namespace tq

// ===========================================================================
using namespace tq;
#define CK(x) do { hipError_t e=(x); if(e){printf("HIP %s @%d\n",hipGetErrorString(e),__LINE__);exit(1);} } while(0)
static int g_fail = 0;
static void report(const char *n, bool ok, const char *d = "") {
    printf("%-46s %s %s\n", n, ok ? "PASS" : "FAIL", d);
    if (!ok) ++g_fail;
}

static void host_fwht(float *v, int n) {
    for (int width = 1; width < n; width *= 2)
        for (int base = 0; base < n; base += 2 * width)
            for (int i = 0; i < width; ++i) {
                const float a = v[base + i], b = v[base + width + i];
                v[base + i] = a + b;
                v[base + width + i] = a - b;
            }
}

template <int HEAD_SIZE>
static void run_case(long long rows, long long heads) {
    const long long items = rows * heads;
    std::mt19937 rng(900 + HEAD_SIZE);
    std::uniform_real_distribution<float> uf(-2.0f, 2.0f);
    std::vector<float> q((size_t)items * HEAD_SIZE), signs(HEAD_SIZE);
    for (auto &v : q) v = uf(rng);
    for (auto &v : signs) v = (uf(rng) < 0.0f) ? -1.0f : 1.0f;   // sign vector

    float *dq, *ds, *dt;
    CK(hipMalloc(&dq, q.size()*4));    CK(hipMemcpy(dq, q.data(), q.size()*4, hipMemcpyHostToDevice));
    CK(hipMalloc(&ds, HEAD_SIZE*4));   CK(hipMemcpy(ds, signs.data(), HEAD_SIZE*4, hipMemcpyHostToDevice));
    CK(hipMalloc(&dt, q.size()*4));
    k_turboquant_query_transform<HEAD_SIZE><<<items, HEAD_SIZE>>>(dq, ds, dt, items);
    CK(hipDeviceSynchronize());
    if (hipGetLastError() != hipSuccess) { printf("KERNEL ERR\n"); exit(1); }
    std::vector<float> got(q.size());
    CK(hipMemcpy(got.data(), dt, q.size()*4, hipMemcpyDeviceToHost));

    const float normalization = 1.0f / std::sqrt((float)HEAD_SIZE);
    size_t mismatch = 0, nonfinite = 0;
    double worst = 0.0;
    for (long long it = 0; it < items; ++it) {
        std::vector<float> ref(HEAD_SIZE);
        for (int d = 0; d < HEAD_SIZE; ++d) ref[d] = q[(size_t)it*HEAD_SIZE+d] * signs[d];
        host_fwht(ref.data(), HEAD_SIZE);
        for (int d = 0; d < HEAD_SIZE; ++d) {
            ref[d] *= normalization;
            const float g = got[(size_t)it*HEAD_SIZE+d];
            if (!std::isfinite(g)) ++nonfinite;
            if (g != ref[d]) ++mismatch;
            worst = std::max(worst, std::fabs((double)g - ref[d]));
        }
    }
    char d[128], name[64];
    snprintf(name, sizeof name, "turboquant_query_transform hs=%d (exact)", HEAD_SIZE);
    snprintf(d, sizeof d, "(%zu of %zu differ, worst abs %.3g, %zu non-finite)",
             mismatch, q.size(), worst, nonfinite);
    report(name, mismatch == 0 && nonfinite == 0, d);
    hipFree(dq); hipFree(ds); hipFree(dt);
}

int main() {
    run_case<64>(37, 3);
    run_case<128>(21, 4);
    run_case<256>(11, 2);
    printf("\n%s (%d failures)\n", g_fail ? "FAILED" : "ALL PASS", g_fail);
    return g_fail ? 1 : 0;
}
