/**
 * @file
 * @brief CDNA3 port of Metal marginal layout/bit utilities.
 */
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
template <> __device__ __forceinline__ float to_f<__hip_bfloat16>(__hip_bfloat16 x) { return __bfloat162float(x); }
template <typename T> __device__ __forceinline__ T from_f(float x);
template <> __device__ __forceinline__ float from_f<float>(float x) { return x; }
template <> __device__ __forceinline__ __half from_f<__half>(float x) { return __float2half(x); }
template <> __device__ __forceinline__ __hip_bfloat16 from_f<__hip_bfloat16>(float x) { return __float2bfloat16(x); }

template <typename T>
__global__ void tau_tail_kernel(const T* __restrict__ qkv,
                                const T* __restrict__ tok_qv_lin,
                                const T* __restrict__ tau_pos_table,
                                const int* __restrict__ positions,
                                T* __restrict__ out,
                                int total,
                                int Tn,
                                int n_heads,
                                int head_dim,
                                int q_dim) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int row_width = 3 * q_dim;
    const int tok = idx / row_width;
    const int rem = idx - tok * row_width;
    float y = to_f(qkv[idx]);
    if (tok < Tn && rem < q_dim) {
        const int head = rem / head_dim;
        const float tq = tanhf(to_f(tok_qv_lin[tok * 2 * n_heads + head]));
        const float tp = to_f(tau_pos_table[positions[tok] * n_heads + head]);
        y *= tq + tp;
    } else if (tok < Tn && rem >= 2 * q_dim) {
        const int vrem = rem - 2 * q_dim;
        const int head = vrem / head_dim;
        const float tv = tanhf(to_f(tok_qv_lin[tok * 2 * n_heads + n_heads + head]));
        const float tp = to_f(tau_pos_table[positions[tok] * n_heads + head]);
        y *= tv + tp;
    }
    out[idx] = from_f<T>(y);
}

__global__ void packbits_uint8_kernel(const uint8_t* __restrict__ input,
                                      uint8_t* __restrict__ output,
                                      int num_elements,
                                      int bit_order_big) {
    const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int base = out_idx * 8;
    if (base >= num_elements) return;
    uint8_t packed = 0;
    for (int bit = 0; bit < 8; ++bit) {
        const int idx = base + bit;
        if (idx < num_elements && input[idx] != 0) {
            const int shift = bit_order_big ? 7 - bit : bit;
            packed |= uint8_t(1u << unsigned(shift));
        }
    }
    output[out_idx] = packed;
}

__global__ void segment_packbits_uint8_kernel(const uint8_t* __restrict__ input,
                                              const int* __restrict__ input_indptr,
                                              const int* __restrict__ output_indptr,
                                              uint8_t* __restrict__ output,
                                              int num_segments,
                                              int total_output_bytes,
                                              int bit_order_big) {
    const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= total_output_bytes) return;
    int lo = 0, hi = num_segments;
    while (lo + 1 < hi) {
        const int mid = lo + ((hi - lo) >> 1);
        if (output_indptr[mid] <= out_idx) lo = mid;
        else hi = mid;
    }
    const int input_start = input_indptr[lo];
    const int input_end = input_indptr[lo + 1];
    const int base = input_start + (out_idx - output_indptr[lo]) * 8;
    uint8_t packed = 0;
    for (int bit = 0; bit < 8; ++bit) {
        const int idx = base + bit;
        if (idx < input_end && input[idx] != 0) {
            const int shift = bit_order_big ? 7 - bit : bit;
            packed |= uint8_t(1u << unsigned(shift));
        }
    }
    output[out_idx] = packed;
}

__global__ void permute_cols_16bit_kernel(const uint16_t* __restrict__ input,
                                          const int* __restrict__ perm,
                                          uint16_t* __restrict__ output,
                                          int rows,
                                          int cols) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= rows || col >= cols) return;
    output[size_t(row) * cols + col] = input[size_t(row) * cols + perm[col]];
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

static std::vector<uint8_t> ref_packbits(const std::vector<uint8_t>& x, bool big) {
    std::vector<uint8_t> out((x.size() + 7) / 8);
    for (size_t b = 0; b < out.size(); ++b) {
        uint8_t p = 0;
        for (int bit = 0; bit < 8; ++bit) {
            const size_t idx = b * 8 + bit;
            if (idx < x.size() && x[idx]) p |= uint8_t(1u << unsigned(big ? 7 - bit : bit));
        }
        out[b] = p;
    }
    return out;
}

static int run_case(bool bench) {
    int rc = 0;
    std::mt19937 rng(51);
    std::normal_distribution<float> nd(0.0f, 0.5f);

    {
        const int Tn = bench ? 4096 : 13;
        const int n_heads = 4;
        const int head_dim = 16;
        const int q_dim = n_heads * head_dim;
        const int width = 3 * q_dim;
        const int max_pos = 128;
        std::vector<float> qkv(size_t(Tn) * width), lin(size_t(Tn) * 2 * n_heads),
            tau(size_t(max_pos) * n_heads);
        std::vector<int> pos(Tn);
        for (auto& x : qkv) x = nd(rng);
        for (auto& x : lin) x = nd(rng);
        for (auto& x : tau) x = 0.2f + 0.05f * nd(rng);
        for (int i = 0; i < Tn; ++i) pos[i] = i % max_pos;
        auto dq = dnew(qkv);
        auto dl = dnew(lin);
        auto dt = dnew(tau);
        auto dp = dnew(pos);
        float* dout = nullptr;
        CK(hipMalloc(&dout, qkv.size() * sizeof(float)));
        auto launch = [&] {
            tau_tail_kernel<float><<<(int(qkv.size()) + 255) / 256, 256>>>(
                dq, dl, dt, dp, dout, int(qkv.size()), Tn, n_heads, head_dim, q_dim);
        };
        launch();
        CK(hipDeviceSynchronize());
        CK(hipGetLastError());
        auto got = d2h(dout, qkv.size());
        double maxe = 0.0;
        for (int t = 0; t < Tn; ++t) {
            for (int c = 0; c < width; ++c) {
                float ref = qkv[size_t(t) * width + c];
                if (c < q_dim) {
                    const int h = c / head_dim;
                    ref *= std::tanh(lin[size_t(t) * 2 * n_heads + h]) + tau[size_t(pos[t]) * n_heads + h];
                } else if (c >= 2 * q_dim) {
                    const int h = (c - 2 * q_dim) / head_dim;
                    ref *= std::tanh(lin[size_t(t) * 2 * n_heads + n_heads + h]) + tau[size_t(pos[t]) * n_heads + h];
                }
                maxe = std::max(maxe, double(std::abs(got[size_t(t) * width + c] - ref)));
            }
        }
        const bool ok = maxe < 1e-6;
        printf("tau_tail max %.3e %s\n", maxe, ok ? "PASS" : "FAIL");
        rc |= ok ? 0 : 1;
        if (bench && ok) {
            const float ms = bench_ms(launch);
            const double gb = double(qkv.size()) * 2.0 * sizeof(float) / 1e9;
            printf("tau_tail bench %.4f ms %.1f GB/s\n", ms, gb / (ms * 1e-3));
        }
        CK(hipFree(dq));
        CK(hipFree(dl));
        CK(hipFree(dt));
        CK(hipFree(dp));
        CK(hipFree(dout));
    }

    {
        const int n = bench ? (1 << 22) + 3 : 1237;
        std::uniform_int_distribution<int> bd(0, 1);
        std::vector<uint8_t> x(n);
        for (auto& v : x) v = uint8_t(bd(rng));
        auto ref_big = ref_packbits(x, true);
        auto ref_lit = ref_packbits(x, false);
        auto dx = dnew(x);
        uint8_t *db = nullptr, *dl = nullptr;
        CK(hipMalloc(&db, ref_big.size()));
        CK(hipMalloc(&dl, ref_lit.size()));
        auto run_big = [&] { packbits_uint8_kernel<<<(int(ref_big.size()) + 255) / 256, 256>>>(dx, db, n, 1); };
        auto run_lit = [&] { packbits_uint8_kernel<<<(int(ref_lit.size()) + 255) / 256, 256>>>(dx, dl, n, 0); };
        run_big();
        run_lit();
        CK(hipDeviceSynchronize());
        CK(hipGetLastError());
        auto gb = d2h(db, ref_big.size());
        auto gl = d2h(dl, ref_lit.size());
        const bool ok = gb == ref_big && gl == ref_lit;
        printf("packbits big/little %s\n", ok ? "PASS" : "FAIL");
        rc |= ok ? 0 : 1;
        if (bench && ok) {
            const float ms = bench_ms(run_big);
            printf("packbits bench n=%d %.4f ms %.1f GB/s-input\n", n, ms, double(n) / (ms * 1e-3) / 1e9);
        }
        CK(hipFree(dx));
        CK(hipFree(db));
        CK(hipFree(dl));
    }

    {
        std::vector<int> inptr{0, 3, 20, 21, 55, 123};
        std::vector<int> outptr(inptr.size());
        for (size_t i = 1; i < inptr.size(); ++i) outptr[i] = outptr[i - 1] + (inptr[i] - inptr[i - 1] + 7) / 8;
        std::vector<uint8_t> x(inptr.back());
        for (size_t i = 0; i < x.size(); ++i) x[i] = uint8_t((i * 17) & 1);
        std::vector<uint8_t> ref(outptr.back());
        for (size_t s = 0; s + 1 < inptr.size(); ++s) {
            std::vector<uint8_t> seg(x.begin() + inptr[s], x.begin() + inptr[s + 1]);
            auto p = ref_packbits(seg, true);
            std::copy(p.begin(), p.end(), ref.begin() + outptr[s]);
        }
        auto dx = dnew(x);
        auto di = dnew(inptr);
        auto dof = dnew(outptr);
        uint8_t* dout = nullptr;
        CK(hipMalloc(&dout, ref.size()));
        segment_packbits_uint8_kernel<<<(int(ref.size()) + 127) / 128, 128>>>(
            dx, di, dof, dout, int(inptr.size() - 1), int(ref.size()), 1);
        CK(hipDeviceSynchronize());
        CK(hipGetLastError());
        auto got = d2h(dout, ref.size());
        const bool ok = got == ref;
        printf("segment_packbits %s\n", ok ? "PASS" : "FAIL");
        rc |= ok ? 0 : 1;
        CK(hipFree(dx));
        CK(hipFree(di));
        CK(hipFree(dof));
        CK(hipFree(dout));
    }

    {
        const int rows = 33, cols = 48;
        std::vector<uint16_t> x(size_t(rows) * cols), ref(size_t(rows) * cols);
        std::vector<int> perm(cols);
        for (size_t i = 0; i < x.size(); ++i) x[i] = uint16_t(i * 13);
        for (int c = 0; c < cols; ++c) perm[c] = (cols - 1 - c * 5 % cols + cols) % cols;
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c) ref[size_t(r) * cols + c] = x[size_t(r) * cols + perm[c]];
        auto dx = dnew(x);
        auto dp = dnew(perm);
        uint16_t* dout = nullptr;
        CK(hipMalloc(&dout, ref.size() * sizeof(uint16_t)));
        dim3 block(16, 16), grid((cols + 15) / 16, (rows + 15) / 16);
        permute_cols_16bit_kernel<<<grid, block>>>(dx, dp, dout, rows, cols);
        CK(hipDeviceSynchronize());
        CK(hipGetLastError());
        auto got = d2h(dout, ref.size());
        const bool ok = got == ref;
        printf("permute_cols_16bit %s\n", ok ? "PASS" : "FAIL");
        rc |= ok ? 0 : 1;
        CK(hipFree(dx));
        CK(hipFree(dp));
        CK(hipFree(dout));
    }
    return rc;
}

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    return run_case(bench);
}
