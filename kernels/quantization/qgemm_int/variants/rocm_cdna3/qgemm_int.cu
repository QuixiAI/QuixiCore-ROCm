/**
 * @file
 * @brief CDNA3 port of Metal qgemm_int public kernels.
 *
 * Outputs follow the Metal contract: D is (N, M) half, W is row-major (N, K),
 * activations are token-major (M, K), and K is divisible by 4 for W8A8.
 */
#include "../../../qgemm/variants/rocm_cdna3/quant_formats.cuh"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

__device__ __forceinline__ int block_sum_i32(int v) {
    __shared__ int smem[256];
    const int tid = threadIdx.x;
    smem[tid] = v;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    return smem[0];
}

__device__ __forceinline__ float block_sum_f32(float v) {
    __shared__ float smem[256];
    const int tid = threadIdx.x;
    smem[tid] = v;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    return smem[0];
}

__global__ void qgemm_w8a8_kernel(__half* __restrict__ D,
                                  const int8_t* __restrict__ Wq,
                                  const int8_t* __restrict__ Xq,
                                  const __half* __restrict__ w_scale,
                                  const __half* __restrict__ a_scale,
                                  int N,
                                  int K,
                                  int M) {
    const int n = blockIdx.x;
    const int m = blockIdx.y;
    int acc = 0;
    for (int k = threadIdx.x * 4; k < K; k += blockDim.x * 4) {
        const uint32_t wv = *reinterpret_cast<const uint32_t*>(Wq + size_t(n) * K + k);
        const uint32_t xv = *reinterpret_cast<const uint32_t*>(Xq + size_t(m) * K + k);
        acc = __builtin_amdgcn_sdot4(int(wv), int(xv), acc, false);
    }
    acc = block_sum_i32(acc);
    if (threadIdx.x == 0) {
        const float y = float(acc) * __half2float(w_scale[n]) * __half2float(a_scale[m]);
        D[size_t(n) * M + m] = __float2half(y);
    }
}

__global__ void qgemm_w8a8_azp_kernel(__half* __restrict__ D,
                                      const int8_t* __restrict__ Wq,
                                      const int8_t* __restrict__ Xq,
                                      const __half* __restrict__ w_scale,
                                      const float* __restrict__ a_scale,
                                      const int* __restrict__ w_rowsum,
                                      const int* __restrict__ azp,
                                      int N,
                                      int K,
                                      int M) {
    const int n = blockIdx.x;
    const int m = blockIdx.y;
    int acc = 0;
    for (int k = threadIdx.x * 4; k < K; k += blockDim.x * 4) {
        const uint32_t wv = *reinterpret_cast<const uint32_t*>(Wq + size_t(n) * K + k);
        const uint32_t xv = *reinterpret_cast<const uint32_t*>(Xq + size_t(m) * K + k);
        acc = __builtin_amdgcn_sdot4(int(wv), int(xv), acc, false);
    }
    acc = block_sum_i32(acc);
    if (threadIdx.x == 0) {
        const float y = float(acc - azp[m] * w_rowsum[n]) * __half2float(w_scale[n]) * a_scale[m];
        D[size_t(n) * M + m] = __float2half(y);
    }
}

__global__ void qgemm_w2a8_kernel(__half* __restrict__ D,
                                  const uint8_t* __restrict__ Wq,
                                  const int8_t* __restrict__ Xq,
                                  const __half* __restrict__ a_scale,
                                  int N,
                                  int K,
                                  int M) {
    const int n = blockIdx.x;
    const int m = blockIdx.y;
    const int bpr = K / tmq::bitnet::block_k;
    const uint8_t* wrow = Wq + size_t(n) * bpr * tmq::bitnet::block_bytes;
    float acc = 0.0f;
    for (int b = threadIdx.x; b < bpr; b += blockDim.x) {
        const uint8_t* base = wrow + size_t(b) * tmq::bitnet::block_bytes;
        const int8_t* x = Xq + size_t(m) * K + b * tmq::bitnet::block_k;
        int isum = 0;
        #pragma unroll
        for (int c = 0; c < tmq::bitnet::block_k; ++c) {
            isum += tmq::bitnet::code(base, c) * int(x[c]);
        }
        acc += float(isum) * tmq::bitnet::gscale(base);
    }
    acc = block_sum_f32(acc);
    if (threadIdx.x == 0) D[size_t(n) * M + m] = __float2half(acc * __half2float(a_scale[m]));
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
static float bench_ms(FN&& fn, int warmup = 10, int iters = 50) {
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

static void put_half(std::vector<uint8_t>& v, size_t off, float x) {
    __half h = __float2half(x);
    std::memcpy(v.data() + off, &h, sizeof(h));
}

static std::vector<uint8_t> pack_bitnet(const std::vector<int>& code, const std::vector<float>& scale,
                                        int N, int K) {
    const int bpr = K / tmq::bitnet::block_k;
    std::vector<uint8_t> out(size_t(N) * bpr * tmq::bitnet::block_bytes, 0);
    for (int n = 0; n < N; ++n) {
        for (int b = 0; b < bpr; ++b) {
            uint8_t* base = out.data() + (size_t(n) * bpr + b) * tmq::bitnet::block_bytes;
            put_half(out, (size_t(n) * bpr + b) * tmq::bitnet::block_bytes, scale[size_t(n) * bpr + b]);
            for (int c = 0; c < tmq::bitnet::block_k; ++c) {
                const int q = code[size_t(n) * K + b * tmq::bitnet::block_k + c] + 1;
                base[2 + (c >> 2)] |= uint8_t((q & 3) << ((c & 3) * 2));
            }
        }
    }
    return out;
}

static int check_half(const char* name, const std::vector<__half>& got,
                      const std::vector<float>& ref, double tol) {
    double sad = 0.0, sref = 0.0, maxe = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double g = __half2float(got[i]);
        const double d = std::abs(g - ref[i]);
        sad += d;
        sref += std::abs(ref[i]);
        maxe = std::max(maxe, d);
    }
    const double rel = sad / std::max(sref, 1e-30);
    const bool ok = rel < tol;
    printf("%-16s rel %.3e max %.3e %s\n", name, rel, maxe, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

static int run_case(bool bench) {
    const int N = bench ? 512 : 48;
    const int M = bench ? 64 : 17;
    const int K = bench ? 1024 : 128;
    std::mt19937 rng(21);
    std::uniform_int_distribution<int> qd(-16, 16);
    std::uniform_int_distribution<int> bd(-1, 1);
    std::uniform_real_distribution<float> sd(0.002f, 0.04f);
    std::vector<int8_t> W(size_t(N) * K), X(size_t(M) * K);
    std::vector<__half> ws(N), as_h(M);
    std::vector<float> as_f(M);
    std::vector<int> rowsum(N), azp(M), W2c(size_t(N) * K);
    std::vector<float> bscale(size_t(N) * (K / tmq::bitnet::block_k));
    for (auto& x : W) x = int8_t(qd(rng));
    for (auto& x : X) x = int8_t(qd(rng));
    for (int n = 0; n < N; ++n) {
        ws[n] = __float2half(sd(rng));
        int rs = 0;
        for (int k = 0; k < K; ++k) rs += int(W[size_t(n) * K + k]);
        rowsum[n] = rs;
    }
    for (int m = 0; m < M; ++m) {
        as_f[m] = sd(rng);
        as_h[m] = __float2half(as_f[m]);
        azp[m] = (m % 7) - 3;
    }
    for (size_t i = 0; i < W2c.size(); ++i) W2c[i] = bd(rng);
    for (auto& s : bscale) s = sd(rng) * 8.0f;
    auto W2 = pack_bitnet(W2c, bscale, N, K);

    std::vector<float> ref8(size_t(N) * M), ref8a(size_t(N) * M), ref2(size_t(N) * M);
    for (int n = 0; n < N; ++n) {
        for (int m = 0; m < M; ++m) {
            int acc = 0;
            for (int k = 0; k < K; ++k) acc += int(W[size_t(n) * K + k]) * int(X[size_t(m) * K + k]);
            ref8[size_t(n) * M + m] = float(acc) * __half2float(ws[n]) * __half2float(as_h[m]);
            ref8a[size_t(n) * M + m] = float(acc - azp[m] * rowsum[n]) * __half2float(ws[n]) * as_f[m];
            double s = 0.0;
            for (int k = 0; k < K; ++k) {
                const int b = k / tmq::bitnet::block_k;
                s += double(W2c[size_t(n) * K + k]) * int(X[size_t(m) * K + k]) *
                     bscale[size_t(n) * (K / tmq::bitnet::block_k) + b];
            }
            ref2[size_t(n) * M + m] = float(s * __half2float(as_h[m]));
        }
    }

    auto dW = dnew(W);
    auto dX = dnew(X);
    auto dws = dnew(ws);
    auto das_h = dnew(as_h);
    auto das_f = dnew(as_f);
    auto drs = dnew(rowsum);
    auto dazp = dnew(azp);
    auto dW2 = dnew(W2);
    __half *d8 = nullptr, *d8a = nullptr, *d2 = nullptr;
    CK(hipMalloc(&d8, ref8.size() * sizeof(__half)));
    CK(hipMalloc(&d8a, ref8.size() * sizeof(__half)));
    CK(hipMalloc(&d2, ref8.size() * sizeof(__half)));
    dim3 grid(N, M);
    auto run8 = [&] { qgemm_w8a8_kernel<<<grid, 128>>>(d8, dW, dX, dws, das_h, N, K, M); };
    auto run8a = [&] { qgemm_w8a8_azp_kernel<<<grid, 128>>>(d8a, dW, dX, dws, das_f, drs, dazp, N, K, M); };
    auto run2 = [&] { qgemm_w2a8_kernel<<<grid, 128>>>(d2, dW2, dX, das_h, N, K, M); };
    run8();
    run8a();
    run2();
    CK(hipDeviceSynchronize());
    CK(hipGetLastError());
    int rc = 0;
    rc |= check_half("qgemm_w8a8", d2h(d8, ref8.size()), ref8, 2e-3);
    rc |= check_half("qgemm_w8a8_azp", d2h(d8a, ref8.size()), ref8a, 2e-3);
    rc |= check_half("qgemm_w2a8", d2h(d2, ref2.size()), ref2, 3e-3);
    if (bench && rc == 0) {
        const float t8 = bench_ms(run8);
        const float t8a = bench_ms(run8a);
        const float t2 = bench_ms(run2);
        const double ops = 2.0 * N * M * K;
        printf("qgemm_int bench N=%d M=%d K=%d\n", N, M, K);
        printf("  w8a8    %.4f ms %.2f TOPS\n", t8, ops / (t8 * 1e-3) / 1e12);
        printf("  w8a8azp %.4f ms %.2f TOPS\n", t8a, ops / (t8a * 1e-3) / 1e12);
        printf("  w2a8    %.4f ms %.2f TOPS-equivalent\n", t2, ops / (t2 * 1e-3) / 1e12);
    }
    CK(hipFree(dW));
    CK(hipFree(dX));
    CK(hipFree(dws));
    CK(hipFree(das_h));
    CK(hipFree(das_f));
    CK(hipFree(drs));
    CK(hipFree(dazp));
    CK(hipFree(dW2));
    CK(hipFree(d8));
    CK(hipFree(d8a));
    CK(hipFree(d2));
    return rc;
}

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    return run_case(bench);
}
