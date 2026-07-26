#include "hip/hip_runtime.h"
/**
 * @file
 * @brief CDNA3 (gfx942) running KV-cache scale update.
 *
 * Semantic source: ../QuixiCore-CPU/kernels/serving/kv_cache_extended_ref.cpp
 * (kv_cache_scale_update) and its helper kv_cache_scales in serving_ref.cpp.
 *
 *   amax over key and value; any non-finite input rejects the whole call
 *   key_scale   = amax(key)   / 240
 *   value_scale = amax(value) / 240
 *   *new_scale  = max(old_scale, computed)          -- monotone, never shrinks
 *
 * The divisor is **240**, not 448. The MXFP8 codec next door divides by 448
 * (the largest finite E4M3FN magnitude); this path uses the 240 convention and
 * the two are not interchangeable. Copying the neighbouring constant produces
 * scales that are ~1.87x too small and a cache that quietly clips.
 *
 * The update is a max against the incoming scale, so it only ever grows -- a
 * caller feeding a quieter batch must not see the scale drop.
 *
 * Reduction: block-local wave reduce, then an atomicMax on the float's bit
 * pattern. That ordering trick is valid only because the values are magnitudes
 * (non-negative), where IEEE-754 bit order matches numeric order.
 *
 * Build: make      Test: make test
 */
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

namespace kvscale {

constexpr float kScaleDivisor = 240.0f;

__device__ __forceinline__ float wave_max(float v) {
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor(v, off));
    return v;
}

// atomicMax on non-negative floats via their bit pattern (IEEE order == numeric
// order for non-negatives). Magnitudes only -- do not reuse for signed values.
__device__ __forceinline__ void atomic_max_positive(float *address, float value) {
    atomicMax((unsigned int *)address, __float_as_uint(value));
}

__global__ void k_amax(const float *__restrict__ key, const float *__restrict__ value,
                       float *__restrict__ key_amax, float *__restrict__ value_amax,
                       int *__restrict__ invalid, long long count) {
    float kmax = 0.0f, vmax = 0.0f;
    bool bad = false;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < count; i += (long long)gridDim.x * blockDim.x) {
        const float k = key[i], v = value[i];
        if (!isfinite(k) || !isfinite(v)) bad = true;
        kmax = fmaxf(kmax, fabsf(k));
        vmax = fmaxf(vmax, fabsf(v));
    }
    if (bad) atomicExch(invalid, 1);
    kmax = wave_max(kmax);
    vmax = wave_max(vmax);
    if ((threadIdx.x & 63) == 0) {
        atomic_max_positive(key_amax, kmax);
        atomic_max_positive(value_amax, vmax);
    }
}

__global__ void k_finalize(const float *__restrict__ key_amax,
                           const float *__restrict__ value_amax,
                           float old_key_scale, float old_value_scale,
                           float *__restrict__ new_key_scale,
                           float *__restrict__ new_value_scale) {
    *new_key_scale = fmaxf(old_key_scale, *key_amax / kScaleDivisor);
    *new_value_scale = fmaxf(old_value_scale, *value_amax / kScaleDivisor);
}

}  // namespace kvscale

// ===========================================================================
using namespace kvscale;
#define CK(x) do { hipError_t e=(x); if(e){printf("HIP %s @%d\n",hipGetErrorString(e),__LINE__);exit(1);} } while(0)
static int g_fail = 0;
static void report(const char *n, bool ok, const char *d = "") {
    printf("%-46s %s %s\n", n, ok ? "PASS" : "FAIL", d);
    if (!ok) ++g_fail;
}

static void run_case(const char *name, long long count, float old_k, float old_v,
                     bool inject_nan, bool expect_invalid) {
    std::mt19937 rng(4242);
    std::uniform_real_distribution<float> uf(-9.0f, 9.0f);
    std::vector<float> key(count), value(count);
    for (auto &v : key) v = uf(rng);
    for (auto &v : value) v = uf(rng);
    if (inject_nan) key[count / 2] = NAN;

    float *dk, *dv, *dka, *dva, *dnk, *dnv; int *dinv;
    CK(hipMalloc(&dk, count*4));  CK(hipMemcpy(dk, key.data(), count*4, hipMemcpyHostToDevice));
    CK(hipMalloc(&dv, count*4));  CK(hipMemcpy(dv, value.data(), count*4, hipMemcpyHostToDevice));
    CK(hipMalloc(&dka, 4));       CK(hipMemset(dka, 0, 4));
    CK(hipMalloc(&dva, 4));       CK(hipMemset(dva, 0, 4));
    CK(hipMalloc(&dnk, 4));       CK(hipMalloc(&dnv, 4));
    CK(hipMalloc(&dinv, 4));      CK(hipMemset(dinv, 0, 4));

    const int block = 256;
    const int grid = (int)std::min<long long>(1024, (count + block - 1) / block);
    k_amax<<<grid, block>>>(dk, dv, dka, dva, dinv, count);
    k_finalize<<<1, 1>>>(dka, dva, old_k, old_v, dnk, dnv);
    CK(hipDeviceSynchronize());
    if (hipGetLastError() != hipSuccess) { printf("KERNEL ERR\n"); exit(1); }

    float nk, nv; int inv;
    CK(hipMemcpy(&nk, dnk, 4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(&nv, dnv, 4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(&inv, dinv, 4, hipMemcpyDeviceToHost));

    if (expect_invalid) {
        report(name, inv == 1, "(non-finite input flagged)");
    } else {
        float kmax = 0.0f, vmax = 0.0f;
        for (long long i = 0; i < count; ++i) {
            kmax = std::max(kmax, std::fabs(key[i]));
            vmax = std::max(vmax, std::fabs(value[i]));
        }
        const float want_k = std::max(old_k, kmax / kScaleDivisor);
        const float want_v = std::max(old_v, vmax / kScaleDivisor);
        const bool finite = std::isfinite(nk) && std::isfinite(nv);
        char d[128];
        snprintf(d, sizeof d, "(k %.9g want %.9g | v %.9g want %.9g)", nk, want_k, nv, want_v);
        report(name, finite && nk == want_k && nv == want_v && inv == 0, d);
    }
    hipFree(dk); hipFree(dv); hipFree(dka); hipFree(dva);
    hipFree(dnk); hipFree(dnv); hipFree(dinv);
}

int main() {
    // exact: the reduction is a max, so it is order-independent and must match
    // the host bit-for-bit -- unlike a summation.
    run_case("kv_cache_scale_update (old=0, exact)", 100000, 0.0f, 0.0f, false, false);
    run_case("kv_cache_scale_update (old dominates)", 50000, 1e3f, 1e3f, false, false);
    run_case("kv_cache_scale_update (single element)", 1, 0.0f, 0.0f, false, false);
    run_case("kv_cache_scale_update (non-finite rejected)", 4096, 0.0f, 0.0f, true, true);
    printf("\n%s (%d failures)\n", g_fail ? "FAILED" : "ALL PASS", g_fail);
    return g_fail ? 1 : 0;
}
