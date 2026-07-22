/**
 * @file
 * @brief CDNA3 runtime quantization variants missing from the qgemv quant_rt port.
 */
#include "../../../qgemm/variants/rocm_cdna3/quant_formats.cuh"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
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

template <typename T> __device__ __forceinline__ float to_f(T x);
template <> __device__ __forceinline__ float to_f<float>(float x) { return x; }
template <> __device__ __forceinline__ float to_f<__half>(__half x) { return __half2float(x); }
template <> __device__ __forceinline__ float to_f<__hip_bfloat16>(__hip_bfloat16 x) {
    return __bfloat162float(x);
}

__device__ __forceinline__ float block_max(float v) {
    __shared__ float smem[256];
    const int tid = threadIdx.x;
    smem[tid] = v;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    return smem[0];
}

__device__ __forceinline__ float block_min(float v) {
    __shared__ float smem[256];
    const int tid = threadIdx.x;
    smem[tid] = v;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fminf(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    return smem[0];
}

__device__ __forceinline__ int8_t i8_enc(float x) {
    return int8_t(int(fminf(127.0f, fmaxf(-127.0f, rintf(x)))));
}

template <typename T>
__global__ void quantize_per_group_fp8_kernel(const T* __restrict__ x,
                                              uint8_t* __restrict__ codes,
                                              float* __restrict__ scale,
                                              int rows,
                                              int D,
                                              int G,
                                              int ue8m0) {
    const int g = blockIdx.x;
    const int row = blockIdx.y;
    if (row >= rows) return;
    const int ng = D / G;
    const long gbase = long(row) * D + long(g) * G;
    float amax = 0.0f;
    for (int i = threadIdx.x; i < G; i += blockDim.x) amax = fmaxf(amax, fabsf(to_f(x[gbase + i])));
    amax = block_max(amax);
    float s = amax / 448.0f;
    if (ue8m0 != 0 && amax > 0.0f) s = exp2f(ceilf(log2f(fmaxf(amax, 1e-10f) / 448.0f)));
    const float inv = s > 0.0f ? 1.0f / s : 0.0f;
    for (int i = threadIdx.x; i < G; i += blockDim.x) codes[gbase + i] = tmq::e4m3_encode(to_f(x[gbase + i]) * inv);
    if (threadIdx.x == 0) scale[long(row) * ng + g] = s;
}

template <typename T>
__global__ void quantize_per_group_int8_kernel(const T* __restrict__ x,
                                               int8_t* __restrict__ codes,
                                               float* __restrict__ scale,
                                               int rows,
                                               int D,
                                               int G) {
    const int g = blockIdx.x;
    const int row = blockIdx.y;
    if (row >= rows) return;
    const int ng = D / G;
    const long gbase = long(row) * D + long(g) * G;
    float amax = 0.0f;
    for (int i = threadIdx.x; i < G; i += blockDim.x) amax = fmaxf(amax, fabsf(to_f(x[gbase + i])));
    amax = block_max(amax);
    const float s = amax / 127.0f;
    const float inv = s > 0.0f ? 1.0f / s : 0.0f;
    for (int i = threadIdx.x; i < G; i += blockDim.x) codes[gbase + i] = i8_enc(to_f(x[gbase + i]) * inv);
    if (threadIdx.x == 0) scale[long(row) * ng + g] = s;
}

template <typename T>
__global__ void quantize_per_token_int8_azp_kernel(const T* __restrict__ x,
                                                   int8_t* __restrict__ codes,
                                                   float* __restrict__ scale,
                                                   int* __restrict__ azp_out,
                                                   int rows,
                                                   int D) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const long base = long(row) * D;
    float mn = 3.4028234663852886e38f;
    float mx = -3.4028234663852886e38f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        const float v = to_f(x[base + i]);
        mn = fminf(mn, v);
        mx = fmaxf(mx, v);
    }
    mn = block_min(mn);
    mx = block_max(mx);
    const float range = mx - mn;
    const float s = range > 0.0f ? range / 255.0f : fmaxf(fabsf(mn) / 127.0f, 1e-7f);
    const float inv = 1.0f / s;
    const int azp = int(rintf(-128.0f - mn * inv));
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        const int q = int(rintf(to_f(x[base + i]) * inv)) + azp;
        codes[base + i] = int8_t(max(-128, min(127, q)));
    }
    if (threadIdx.x == 0) {
        scale[row] = s;
        azp_out[row] = azp;
    }
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
static float bench_ms(FN&& fn, int warmup = 20, int iters = 100) {
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

static int run_case(bool bench) {
    const int rows = bench ? 4096 : 67;
    const int D = 512;
    const int G = 128;
    const int ng = D / G;
    const size_t n = size_t(rows) * D;
    std::mt19937 rng(31);
    std::normal_distribution<float> nd(0.0f, 2.0f);
    std::vector<float> x(n);
    for (auto& v : x) v = nd(rng);
    x[D + 7] = 0.0f; // keep a small deterministic edge in the distribution.

    auto dx = dnew(x);
    uint8_t* dfp8 = nullptr;
    int8_t *di8 = nullptr, *dazc = nullptr;
    float *dsfp8 = nullptr, *dsi8 = nullptr, *dsaz = nullptr;
    int* dazp = nullptr;
    CK(hipMalloc(&dfp8, n));
    CK(hipMalloc(&di8, n));
    CK(hipMalloc(&dazc, n));
    CK(hipMalloc(&dsfp8, size_t(rows) * ng * sizeof(float)));
    CK(hipMalloc(&dsi8, size_t(rows) * ng * sizeof(float)));
    CK(hipMalloc(&dsaz, rows * sizeof(float)));
    CK(hipMalloc(&dazp, rows * sizeof(int)));
    dim3 ggrid(ng, rows);
    auto run_fp8 = [&] { quantize_per_group_fp8_kernel<float><<<ggrid, 128>>>(dx, dfp8, dsfp8, rows, D, G, 1); };
    auto run_i8 = [&] { quantize_per_group_int8_kernel<float><<<ggrid, 128>>>(dx, di8, dsi8, rows, D, G); };
    auto run_azp = [&] { quantize_per_token_int8_azp_kernel<float><<<rows, 256>>>(dx, dazc, dsaz, dazp, rows, D); };
    run_fp8();
    run_i8();
    run_azp();
    CK(hipDeviceSynchronize());
    CK(hipGetLastError());
    auto cfp8 = d2h(dfp8, n);
    auto ci8 = d2h(di8, n);
    auto caz = d2h(dazc, n);
    auto sfp8 = d2h(dsfp8, size_t(rows) * ng);
    auto si8 = d2h(dsi8, size_t(rows) * ng);
    auto saz = d2h(dsaz, rows);
    auto az = d2h(dazp, rows);

    int bad = 0;
    int fp8_bad = 0;
    double i8_halfsteps = 0.0, azp_rel = 0.0, azp_den = 0.0;
    for (int r = 0; r < rows; ++r) {
        for (int g = 0; g < ng; ++g) {
            float amax = 0.0f;
            for (int i = 0; i < G; ++i) amax = std::max(amax, std::abs(x[size_t(r) * D + g * G + i]));
            const float s8 = amax > 0.0f ? std::exp2(std::ceil(std::log2(std::max(amax, 1e-10f) / 448.0f))) : 0.0f;
            const float si = amax / 127.0f;
            if (std::abs(sfp8[size_t(r) * ng + g] - s8) > 1e-6f) ++bad;
            if (std::abs(si8[size_t(r) * ng + g] - si) > 1e-6f) ++bad;
            for (int i = 0; i < G; ++i) {
                const size_t idx = size_t(r) * D + g * G + i;
                const uint8_t fp8_ref = tmq::e4m3_encode(x[idx] / std::max(sfp8[size_t(r) * ng + g], 1e-20f));
                const float i8_dec = float(ci8[idx]) * si8[size_t(r) * ng + g];
                fp8_bad += cfp8[idx] != fp8_ref;
                i8_halfsteps = std::max(i8_halfsteps, double(std::abs(i8_dec - x[idx]) / std::max(si8[size_t(r) * ng + g], 1e-20f)));
            }
        }
        float mn = x[size_t(r) * D], mx = mn;
        for (int i = 1; i < D; ++i) {
            mn = std::min(mn, x[size_t(r) * D + i]);
            mx = std::max(mx, x[size_t(r) * D + i]);
        }
        const float s = (mx - mn) > 0.0f ? (mx - mn) / 255.0f : std::max(std::abs(mn) / 127.0f, 1e-7f);
        const int zp = int(std::rint(-128.0f - mn / s));
        if (std::abs(saz[r] - s) > 1e-6f || az[r] != zp) ++bad;
        for (int i = 0; i < D; ++i) {
            const size_t idx = size_t(r) * D + i;
            const float dec = saz[r] * (float(caz[idx]) - float(az[r]));
            azp_rel += std::abs(dec - x[idx]);
            azp_den += std::abs(x[idx]);
        }
    }
    const double azp = azp_rel / std::max(azp_den, 1e-30);
    const bool ok = bad == 0 && fp8_bad == 0 && i8_halfsteps <= 0.5001 && azp < 0.01;
    printf("quantize_per_group_fp8 ue8m0 code_bad=%d scale_bad=%d %s\n", fp8_bad, bad, (fp8_bad == 0 && bad == 0) ? "PASS" : "FAIL");
    printf("quantize_per_group_int8 max %.3f %s\n", i8_halfsteps, i8_halfsteps <= 0.5001 ? "PASS" : "FAIL");
    printf("quantize_per_token_int8_azp rel %.3e %s\n", azp, azp < 0.01 ? "PASS" : "FAIL");
    if (bench && ok) {
        const float tf = bench_ms(run_fp8);
        const float ti = bench_ms(run_i8);
        const float ta = bench_ms(run_azp);
        const double gb = double(n) * (sizeof(float) + 1.0) / 1e9;
        printf("quant_rt_group bench rows=%d D=%d G=%d\n", rows, D, G);
        printf("  group_fp8_ue8m0 %.4f ms %.1f GB/s\n", tf, gb / (tf * 1e-3));
        printf("  group_int8      %.4f ms %.1f GB/s\n", ti, gb / (ti * 1e-3));
        printf("  token_int8_azp  %.4f ms %.1f GB/s\n", ta, gb / (ta * 1e-3));
    }
    CK(hipFree(dx));
    CK(hipFree(dfp8));
    CK(hipFree(di8));
    CK(hipFree(dazc));
    CK(hipFree(dsfp8));
    CK(hipFree(dsi8));
    CK(hipFree(dsaz));
    CK(hipFree(dazp));
    return ok ? 0 : 1;
}

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    return run_case(bench);
}
