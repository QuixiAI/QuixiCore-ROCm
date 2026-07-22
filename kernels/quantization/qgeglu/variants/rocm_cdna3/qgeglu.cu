/**
 * @file
 * @brief CDNA3 (gfx942) quantized gated GeGLU: fused up & gate Q4_0 MFMA
 *        projections with a gelu(gate)*up epilogue.
 *
 * Shape/format-named port of embeddinggemma.c `q4_mfma_up_gate_gelu_f16_kernel`
 * (src/engine_rocm.hip). Two weight-only Q4_0 projections of the same fp16
 * activation are accumulated in one wavefront using v_mfma_f32_16x16x16_f16
 * (Q4_0 dequantized to fp16, same primitive as `qgemm`/`qflux`), then the
 * gated GeGLU epilogue `gelu_tanh(gate) * up` is applied in registers before a
 * single fp16 store. This fuses what `qflux` (single projection + gelu + bias)
 * and `flux_gate` (dense GEMM * precomputed gate) do not: a dual quantized
 * projection with the gate activation applied to one of the two products.
 *
 * Contract, token-major, tile 16x16 per wavefront:
 *   Wup, Wgate : (N, K) Q4_0 blocks   (row-major over hidden features N)
 *   X          : (M, K) fp16          (token-major activations)
 *   Y          : (M, N) fp16          Y[m,n] = gelu_tanh(gate[m,n]) * up[m,n]
 * K, N, M all multiples of 16 for the fast path (edge rows/tokens masked).
 */
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <random>

#define CK(x) do { \
    hipError_t e = (x); \
    if (e != hipSuccess) { \
        printf("HIP %s @%d\n", hipGetErrorString(e), __LINE__); \
        exit(1); \
    } \
} while (0)

constexpr int kQK = 32;

struct __align__(2) block_q4_0 {
    __half d;
    uint8_t qs[16];
};

typedef __attribute__((__vector_size__(4 * sizeof(__fp16)))) __fp16 half4_t;
typedef __attribute__((__vector_size__(4 * sizeof(float)))) float float4_t;

__device__ __forceinline__ float gelu_tanh(float x) {
    const float c = 0.7978845608028654f;   // sqrt(2/pi)
    return 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x * x * x)));
}

__device__ __forceinline__ half4_t load_input_mfma_fragment(
    const __half *input, int n_tokens, int n_cols, int token_start, int col_start) {
    const int lane = threadIdx.x & 63;
    const int token = token_start + (lane & 15);
    const int col = col_start + (lane >> 4) * 4;
    if (token < n_tokens) {
        return *reinterpret_cast<const half4_t *>(
            input + static_cast<size_t>(token) * n_cols + col);
    }
    return half4_t{0, 0, 0, 0};
}

__device__ __forceinline__ half4_t load_q4_mfma_fragment(
    const block_q4_0 *weights, int n_rows, int n_cols, int row_start, int col_start) {
    const int lane = threadIdx.x & 63;
    const int row = row_start + (lane & 15);
    const int col = col_start + (lane >> 4) * 4;
    if (row >= n_rows) return half4_t{0, 0, 0, 0};
    const int blocks_per_row = n_cols / kQK;
    const block_q4_0 &packed = weights[
        static_cast<size_t>(row) * blocks_per_row + col / kQK];
    const int block_col = col % kQK;
    const float scale = __half2float(packed.d);
    half4_t fragment;
#pragma unroll
    for (int item = 0; item < 4; item++) {
        const int quant_col = block_col + item;
        const uint8_t code = packed.qs[quant_col < 16 ? quant_col : quant_col - 16];
        const int quant = quant_col < 16 ? (code & 0x0f) - 8 : (code >> 4) - 8;
        fragment[item] = static_cast<__fp16>(scale * static_cast<float>(quant));
    }
    return fragment;
}

__device__ __forceinline__ float4_t mfma_m16n16k16(
    half4_t input, half4_t weights, float4_t accumulator) {
    return __builtin_amdgcn_mfma_f32_16x16x16f16(input, weights, accumulator, 0, 0, 0);
}

// Candidate: fused dual Q4_0 MFMA projection + gelu(gate)*up epilogue.
__global__ void q4_geglu_fused_kernel(
    const block_q4_0 *up_weights, const block_q4_0 *gate_weights,
    const __half *input, __half *output, int n_tokens, int n_rows, int n_cols) {
    const int row_start = static_cast<int>(blockIdx.x) * 16;
    const int token_start = static_cast<int>(blockIdx.y) * 16;
    float4_t up_acc = {0, 0, 0, 0};
    float4_t gate_acc = {0, 0, 0, 0};
    for (int col_start = 0; col_start < n_cols; col_start += 16) {
        const half4_t in = load_input_mfma_fragment(input, n_tokens, n_cols, token_start, col_start);
        const half4_t uw = load_q4_mfma_fragment(up_weights, n_rows, n_cols, row_start, col_start);
        const half4_t gw = load_q4_mfma_fragment(gate_weights, n_rows, n_cols, row_start, col_start);
        up_acc = mfma_m16n16k16(in, uw, up_acc);
        gate_acc = mfma_m16n16k16(in, gw, gate_acc);
    }
    const int lane = threadIdx.x & 63;
    const int row = row_start + (lane & 15);
    const int token = token_start + (lane >> 4) * 4;
#pragma unroll
    for (int item = 0; item < 4; item++) {
        if (row < n_rows && token + item < n_tokens) {
            output[static_cast<size_t>(token + item) * n_rows + row] =
                __float2half(gelu_tanh(gate_acc[item]) * up_acc[item]);
        }
    }
}

// Baseline piece: single Q4_0 MFMA projection to fp32 (unfused route).
__global__ void q4_mfma_proj_kernel(
    const block_q4_0 *weights, const __half *input, float *output,
    int n_tokens, int n_rows, int n_cols) {
    const int row_start = static_cast<int>(blockIdx.x) * 16;
    const int token_start = static_cast<int>(blockIdx.y) * 16;
    float4_t acc = {0, 0, 0, 0};
    for (int col_start = 0; col_start < n_cols; col_start += 16) {
        const half4_t in = load_input_mfma_fragment(input, n_tokens, n_cols, token_start, col_start);
        const half4_t w = load_q4_mfma_fragment(weights, n_rows, n_cols, row_start, col_start);
        acc = mfma_m16n16k16(in, w, acc);
    }
    const int lane = threadIdx.x & 63;
    const int row = row_start + (lane & 15);
    const int token = token_start + (lane >> 4) * 4;
#pragma unroll
    for (int item = 0; item < 4; item++) {
        if (row < n_rows && token + item < n_tokens) {
            output[static_cast<size_t>(token + item) * n_rows + row] = acc[item];
        }
    }
}

// Baseline piece: elementwise gelu(gate)*up over fp32 buffers -> fp16.
__global__ void gelu_mul_kernel(const float *up, const float *gate,
                                __half *output, size_t count) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    output[i] = __float2half(gelu_tanh(gate[i]) * up[i]);
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
    CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    for (int i = 0; i < iters; ++i) {
        CK(hipEventRecord(a)); fn(); CK(hipEventRecord(b));
        CK(hipEventSynchronize(b)); CK(hipEventElapsedTime(&t[i], a, b));
    }
    CK(hipEventDestroy(a)); CK(hipEventDestroy(b));
    std::sort(t.begin(), t.end());
    return t[iters / 2];
}

static std::vector<block_q4_0> quantize_q4_0(const std::vector<float>& src, int rows, int K) {
    const int bpr = K / kQK;
    std::vector<block_q4_0> out(static_cast<size_t>(rows) * bpr);
    for (int r = 0; r < rows; ++r) for (int b = 0; b < bpr; ++b) {
        const float* base = src.data() + static_cast<size_t>(r) * K + b * kQK;
        float amax = 0.0f, vmax = 0.0f;
        for (int i = 0; i < kQK; ++i)
            if (std::fabs(base[i]) > amax) { amax = std::fabs(base[i]); vmax = base[i]; }
        const float d = vmax / -8.0f;
        const float id = d ? 1.0f / d : 0.0f;
        block_q4_0& blk = out[static_cast<size_t>(r) * bpr + b];
        blk.d = __float2half(d);
        for (int i = 0; i < 16; ++i) {
            const int x0 = std::min(15, (int)std::lround(base[i] * id + 8.0f));
            const int x1 = std::min(15, (int)std::lround(base[i + 16] * id + 8.0f));
            blk.qs[i] = (uint8_t)(std::max(0, x0) | (std::max(0, x1) << 4));
        }
    }
    return out;
}

// Host reference in fp64 using the fp16-rounded weight scale and fp16 inputs.
static std::vector<float> host_ref(const std::vector<block_q4_0>& Wup,
                                   const std::vector<block_q4_0>& Wgate,
                                   const std::vector<__half>& X,
                                   int N, int K, int M) {
    const int bpr = K / kQK;
    std::vector<float> Y(static_cast<size_t>(M) * N);
    for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n) {
        double up = 0.0, gate = 0.0;
        for (int b = 0; b < bpr; ++b) {
            const block_q4_0& wu = Wup[static_cast<size_t>(n) * bpr + b];
            const block_q4_0& wg = Wgate[static_cast<size_t>(n) * bpr + b];
            const double du = __half2float(wu.d), dg = __half2float(wg.d);
            for (int i = 0; i < 16; ++i) {
                const double xlo = __half2float(X[static_cast<size_t>(m) * K + b * kQK + i]);
                const double xhi = __half2float(X[static_cast<size_t>(m) * K + b * kQK + i + 16]);
                up += du * (double)((wu.qs[i] & 0x0f) - 8) * xlo;
                up += du * (double)((wu.qs[i] >> 4) - 8) * xhi;
                gate += dg * (double)((wg.qs[i] & 0x0f) - 8) * xlo;
                gate += dg * (double)((wg.qs[i] >> 4) - 8) * xhi;
            }
        }
        const double c = 0.7978845608028654;
        const double g = 0.5 * gate * (1.0 + std::tanh(c * (gate + 0.044715 * gate * gate * gate)));
        Y[static_cast<size_t>(m) * N + n] = (float)(g * up);
    }
    return Y;
}

static int check(const char* name, const std::vector<__half>& got,
                 const std::vector<float>& ref, double tol) {
    double sad = 0.0, sref = 0.0, maxe = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double d = std::abs((double)__half2float(got[i]) - (double)ref[i]);
        sad += d; sref += std::abs((double)ref[i]); maxe = std::max(maxe, d);
    }
    const double rel = sad / std::max(sref, 1e-30);
    const bool ok = rel < tol;
    printf("%-22s rel %.3e max %.3e %s\n", name, rel, maxe, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

static int run_case(int N, int K, int M, bool bench) {
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> Wuf(static_cast<size_t>(N) * K), Wgf(static_cast<size_t>(N) * K);
    std::vector<float> Xf(static_cast<size_t>(M) * K);
    std::vector<__half> X(static_cast<size_t>(M) * K);
    for (auto& v : Wuf) v = nd(rng) * 0.1f;
    for (auto& v : Wgf) v = nd(rng) * 0.1f;
    for (size_t i = 0; i < Xf.size(); ++i) { Xf[i] = nd(rng) * 0.5f; X[i] = __float2half(Xf[i]); }
    auto Wup = quantize_q4_0(Wuf, N, K);
    auto Wgate = quantize_q4_0(Wgf, N, K);
    auto ref = host_ref(Wup, Wgate, X, N, K, M);

    auto dWu = dnew(Wup); auto dWg = dnew(Wgate); auto dX = dnew(X);
    __half *dYf = nullptr, *dYb = nullptr;
    float *dUp = nullptr, *dGate = nullptr;
    CK(hipMalloc(&dYf, ref.size() * sizeof(__half)));
    CK(hipMalloc(&dYb, ref.size() * sizeof(__half)));
    CK(hipMalloc(&dUp, ref.size() * sizeof(float)));
    CK(hipMalloc(&dGate, ref.size() * sizeof(float)));

    dim3 grid((N + 15) / 16, (M + 15) / 16);
    auto runFused = [&] {
        q4_geglu_fused_kernel<<<grid, 64>>>(dWu, dWg, dX, dYf, M, N, K);
    };
    auto runUnfused = [&] {
        q4_mfma_proj_kernel<<<grid, 64>>>(dWu, dX, dUp, M, N, K);
        q4_mfma_proj_kernel<<<grid, 64>>>(dWg, dX, dGate, M, N, K);
        const size_t count = ref.size();
        gelu_mul_kernel<<<(count + 255) / 256, 256>>>(dUp, dGate, dYb, count);
    };
    runFused(); runUnfused();
    CK(hipDeviceSynchronize()); CK(hipGetLastError());

    int rc = 0;
    rc |= check("qgeglu_fused", d2h(dYf, ref.size()), ref, 1e-2);
    rc |= check("qgeglu_unfused(base)", d2h(dYb, ref.size()), ref, 1e-2);

    if (bench && rc == 0) {
        const float tf = bench_ms(runFused);
        const float tb = bench_ms(runUnfused);
        const double ops = 2.0 * 2.0 * (double)N * M * K;  // two projections
        printf("qgeglu bench N=%d K=%d M=%d\n", N, K, M);
        printf("  unfused(base) %.4f ms  %.2f TFLOP/s\n", tb, ops / (tb * 1e-3) / 1e12);
        printf("  fused(cand)   %.4f ms  %.2f TFLOP/s  (%.2fx)\n",
               tf, ops / (tf * 1e-3) / 1e12, tb / tf);
    }
    CK(hipFree(dWu)); CK(hipFree(dWg)); CK(hipFree(dX));
    CK(hipFree(dYf)); CK(hipFree(dYb)); CK(hipFree(dUp)); CK(hipFree(dGate));
    return rc;
}

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    int rc = 0;
    rc |= run_case(1152, 768, 16, false);   // n_ff x n_embd, one tile of tokens
    rc |= run_case(1152, 768, 64, false);
    if (bench) {
        rc |= run_case(1152, 768, 64, true);
        rc |= run_case(1152, 768, 256, true);
    }
    printf(rc == 0 ? "ALL PASS\n" : "FAIL\n");
    return rc;
}
