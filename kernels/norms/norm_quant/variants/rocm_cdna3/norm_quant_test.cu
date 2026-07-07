/**
 * @file
 * @brief Correctness and focused timing harness for CDNA3 norm-quant kernels.
 *
 * Build:
 *   hipcc -std=c++17 -O3 --offload-arch=gfx942 norm_quant_test.cu -o norm_quant_test.out
 */
#include "tm_norm_quant_kernels.cuh"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>

using namespace tmnq;

static int g_fail = 0;

#define CK(x) do { \
    hipError_t e = (x); \
    if (e != hipSuccess) { \
        printf("HIP %s @%d\n", hipGetErrorString(e), __LINE__); \
        exit(1); \
    } \
} while (0)

template <typename T>
static T* dnew(const std::vector<T>& h) {
    T* d = nullptr;
    CK(hipMalloc(&d, h.size() * sizeof(T)));
    CK(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice));
    return d;
}

template <typename T>
static T* dz(size_t n) {
    T* d = nullptr;
    CK(hipMalloc(&d, n * sizeof(T)));
    CK(hipMemset(d, 0, n * sizeof(T)));
    return d;
}

template <typename T>
static std::vector<T> d2h(const T* d, size_t n) {
    std::vector<T> h(n);
    CK(hipMemcpy(h.data(), d, n * sizeof(T), hipMemcpyDeviceToHost));
    return h;
}

static void report(const char* nm, double e, double tol) {
    const bool ok = e <= tol;
    printf("%-44s %s (rel %.3e, tol %.1e)\n", nm, ok ? "PASS" : "FAIL", e, tol);
    if (!ok) ++g_fail;
}

static std::mt19937 rng(5);

static std::vector<float> rv(size_t n, float lo, float hi) {
    std::uniform_real_distribution<float> d(lo, hi);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}

static double e4dec(uint8_t v) {
    float m;
    if (v & 0x78) {
        const int e = (v >> 3) & 0xF;
        const int mm = v & 7;
        m = std::ldexp(1.0f + mm / 8.0f, e - 7);
    } else {
        m = float(v & 7) * 0.001953125f;
    }
    return (v & 0x80) ? -m : m;
}

template <typename FN>
static float bench_ms(FN&& fn, int warmup = 10, int iters = 50) {
    for (int i = 0; i < warmup; i++) fn();
    CK(hipDeviceSynchronize());
    std::vector<float> t(iters);
    hipEvent_t a, b;
    CK(hipEventCreate(&a));
    CK(hipEventCreate(&b));
    for (int i = 0; i < iters; i++) {
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

static void correctness() {
    const int M = 40;
    const int D = 768;
    const float eps = 1e-5f;
    auto x = rv(size_t(M) * D, -2.0f, 2.0f);
    auto res = rv(size_t(M) * D, -2.0f, 2.0f);
    auto w = rv(D, -1.0f, 1.0f);

    std::vector<__half> xh(size_t(M) * D), rh(size_t(M) * D), wh(D);
    for (size_t i = 0; i < xh.size(); ++i) {
        xh[i] = __float2half(x[i]);
        rh[i] = __float2half(res[i]);
    }
    for (int i = 0; i < D; ++i) wh[i] = __float2half(w[i]);

    auto dx = dnew(xh);
    auto dr = dnew(rh);
    auto dw = dnew(wh);

    auto rmsref = [&](bool resid, std::vector<double>& out) {
        out.resize(size_t(M) * D);
        for (int r = 0; r < M; ++r) {
            double ss = 0.0;
            std::vector<double> v(D);
            for (int j = 0; j < D; ++j) {
                v[j] = double(x[size_t(r) * D + j]) + (resid ? double(res[size_t(r) * D + j]) : 0.0);
                ss += v[j] * v[j];
            }
            const double inv = 1.0 / std::sqrt(ss / D + eps);
            for (int j = 0; j < D; ++j) out[size_t(r) * D + j] = v[j] * inv * w[j];
        }
    };

    for (int fp8 = 1; fp8 >= 0; --fp8) {
        std::vector<double> ref;
        rmsref(false, ref);
        uint8_t* dc = dz<uint8_t>(size_t(M) * D);
        float* dsc = dz<float>(M);
        if (fp8) {
            rms_norm_quant<__half, true, true, false><<<M, 256>>>(
                dc, dsc, nullptr, dx, nullptr, dw, D, eps, 0.0f);
        } else {
            rms_norm_quant<__half, false, true, false><<<M, 256>>>(
                dc, dsc, nullptr, dx, nullptr, dw, D, eps, 0.0f);
        }
        CK(hipDeviceSynchronize());
        CK(hipGetLastError());
        auto cd = d2h(dc, size_t(M) * D);
        auto sc = d2h(dsc, M);
        double gs = 0.0;
        double rs = 0.0;
        for (int r = 0; r < M; ++r) {
            for (int j = 0; j < D; ++j) {
                const double got = fp8 ? e4dec(cd[size_t(r) * D + j]) * sc[r]
                                       : double(int8_t(cd[size_t(r) * D + j])) * sc[r];
                gs += std::abs(got - ref[size_t(r) * D + j]);
                rs += std::abs(ref[size_t(r) * D + j]);
            }
        }
        report(fp8 ? "rms_norm_quant fp8 dyn (rt)" : "rms_norm_quant int8 dyn (rt)",
               gs / std::max(rs, 1e-30), fp8 ? 6e-2 : 1.5e-2);
        CK(hipFree(dc));
        CK(hipFree(dsc));
    }

    {
        std::vector<double> ref;
        rmsref(true, ref);
        uint8_t* dc = dz<uint8_t>(size_t(M) * D);
        float* dsc = dz<float>(M);
        __half* dro = dz<__half>(size_t(M) * D);
        rms_norm_quant<__half, false, true, true><<<M, 256>>>(
            dc, dsc, dro, dx, dr, dw, D, eps, 0.0f);
        CK(hipDeviceSynchronize());
        CK(hipGetLastError());
        auto cd = d2h(dc, size_t(M) * D);
        auto sc = d2h(dsc, M);
        auto ro = d2h(dro, size_t(M) * D);
        double gs = 0.0;
        double rs = 0.0;
        double roerr = 0.0;
        for (int r = 0; r < M; ++r) {
            for (int j = 0; j < D; ++j) {
                const double got = double(int8_t(cd[size_t(r) * D + j])) * sc[r];
                gs += std::abs(got - ref[size_t(r) * D + j]);
                rs += std::abs(ref[size_t(r) * D + j]);
                roerr = std::max(roerr, double(std::abs(__half2float(ro[size_t(r) * D + j]) -
                                                       (x[size_t(r) * D + j] + res[size_t(r) * D + j]))));
            }
        }
        report("rms_norm_add_quant int8 dyn (rt)", gs / std::max(rs, 1e-30), 1.5e-2);
        report("res_out (x+residual)", roerr, 2e-2);
        CK(hipFree(dc));
        CK(hipFree(dsc));
        CK(hipFree(dro));
    }

    {
        const float inv_static = 64.0f;
        std::vector<double> ref;
        rmsref(false, ref);
        uint8_t* dc = dz<uint8_t>(size_t(M) * D);
        rms_norm_quant<__half, false, false, false><<<M, 256>>>(
            dc, nullptr, nullptr, dx, nullptr, dw, D, eps, inv_static);
        CK(hipDeviceSynchronize());
        CK(hipGetLastError());
        auto cd = d2h(dc, size_t(M) * D);
        double gs = 0.0;
        double rs = 0.0;
        for (int r = 0; r < M; ++r) {
            for (int j = 0; j < D; ++j) {
                const double got = double(int8_t(cd[size_t(r) * D + j])) / inv_static;
                gs += std::abs(got - ref[size_t(r) * D + j]);
                rs += std::abs(ref[size_t(r) * D + j]);
            }
        }
        report("rms_norm_quant int8 static (rt)", gs / std::max(rs, 1e-30), 1.5e-2);
        CK(hipFree(dc));
    }

    {
        int8_t* dc = dz<int8_t>(size_t(M) * D);
        float* dsc = dz<float>(M);
        int* daz = dz<int>(M);
        azp_int8_quant<__half, true><<<M, 32>>>(dc, dsc, daz, dx, D, 0.0f, 0);
        CK(hipDeviceSynchronize());
        CK(hipGetLastError());
        auto cd = d2h(dc, size_t(M) * D);
        auto sc = d2h(dsc, M);
        auto az = d2h(daz, M);
        double gs = 0.0;
        double rs = 0.0;
        for (int r = 0; r < M; ++r) {
            for (int j = 0; j < D; ++j) {
                const double got = (double(cd[size_t(r) * D + j]) - az[r]) * sc[r];
                gs += std::abs(got - x[size_t(r) * D + j]);
                rs += std::abs(x[size_t(r) * D + j]);
            }
        }
        report("azp_int8_quant dyn (rt)", gs / std::max(rs, 1e-30), 1.2e-2);
        CK(hipFree(dc));
        CK(hipFree(dsc));
        CK(hipFree(daz));
    }

    {
        const float scale = 0.02f;
        const int zp = -3;
        int8_t* dc = dz<int8_t>(size_t(M) * D);
        azp_int8_quant<__half, false><<<M, 32>>>(dc, nullptr, nullptr, dx, D, scale, zp);
        CK(hipDeviceSynchronize());
        CK(hipGetLastError());
        auto cd = d2h(dc, size_t(M) * D);
        double gs = 0.0;
        double rs = 0.0;
        for (int r = 0; r < M; ++r) {
            for (int j = 0; j < D; ++j) {
                const double got = (double(cd[size_t(r) * D + j]) - zp) * scale;
                gs += std::abs(got - x[size_t(r) * D + j]);
                rs += std::abs(x[size_t(r) * D + j]);
            }
        }
        report("azp_int8_quant static (rt)", gs / std::max(rs, 1e-30), 1.8e-2);
        CK(hipFree(dc));
    }

    {
        const int GS = 128;
        const int NG = D / GS;
        int8_t* dc = dz<int8_t>(size_t(M) * D);
        float* dsc = dz<float>(size_t(M) * NG);
        dim3 grid(NG, M);
        per_token_group_int8_quant<__half><<<grid, 32>>>(dc, dsc, dx, D, GS, NG, 1e-6f);
        CK(hipDeviceSynchronize());
        CK(hipGetLastError());
        auto cd = d2h(dc, size_t(M) * D);
        auto sc = d2h(dsc, size_t(M) * NG);
        double gs = 0.0;
        double rs = 0.0;
        for (int r = 0; r < M; ++r) {
            for (int j = 0; j < D; ++j) {
                const double got = double(cd[size_t(r) * D + j]) * sc[size_t(r) * NG + j / GS];
                gs += std::abs(got - x[size_t(r) * D + j]);
                rs += std::abs(x[size_t(r) * D + j]);
            }
        }
        report("per_token_group_int8 (rt)", gs / std::max(rs, 1e-30), 1.5e-2);
        CK(hipFree(dc));
        CK(hipFree(dsc));
    }

    CK(hipFree(dx));
    CK(hipFree(dr));
    CK(hipFree(dw));
}

static void perf_run() {
    const int M = 16384;
    const int D = 4096;
    const float eps = 1e-5f;
    auto x = rv(size_t(M) * D, -2.0f, 2.0f);
    auto w = rv(D, -1.0f, 1.0f);
    std::vector<__half> xh(size_t(M) * D), wh(D);
    for (size_t i = 0; i < xh.size(); ++i) xh[i] = __float2half(x[i]);
    for (int i = 0; i < D; ++i) wh[i] = __float2half(w[i]);
    auto dx = dnew(xh);
    auto dw = dnew(wh);
    uint8_t* dc = dz<uint8_t>(size_t(M) * D);
    float* dsc = dz<float>(M);

    auto launch128 = [&] {
        rms_norm_quant<__half, false, true, false><<<M, 128>>>(
            dc, dsc, nullptr, dx, nullptr, dw, D, eps, 0.0f);
    };
    auto launch256 = [&] {
        rms_norm_quant<__half, false, true, false><<<M, 256>>>(
            dc, dsc, nullptr, dx, nullptr, dw, D, eps, 0.0f);
    };

    launch256();
    CK(hipDeviceSynchronize());
    CK(hipGetLastError());

    const float t128 = bench_ms(launch128, 5, 30);
    const float t256 = bench_ms(launch256, 5, 30);
    const double bytes = double(M) * D * (2.0 + 1.0) + double(D) * 2.0 + double(M) * 4.0;
    printf("\n== norm_quant perf: rms_norm_quant int8 dyn M=%d D=%d dtype=fp16\n", M, D);
    printf("block128: %.3f ms  %.1f GB/s\n", t128, bytes / (t128 * 1e-3) / 1e9);
    printf("block256: %.3f ms  %.1f GB/s  keep=%s\n", t256, bytes / (t256 * 1e-3) / 1e9,
           t256 <= t128 ? "block256" : "block128");

    CK(hipFree(dx));
    CK(hipFree(dw));
    CK(hipFree(dc));
    CK(hipFree(dsc));
}

int main() {
    correctness();
    printf("\n%s (%d failures)\n", g_fail ? "FAILED" : "ALL PASS", g_fail);
    if (g_fail) return 1;
    perf_run();
    return 0;
}
