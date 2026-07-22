/**
 * @file
 * @brief CDNA3 (gfx942) fused combined Q/K/V projection: three weight-only
 *        Q4_0 MFMA projections of one shared fp16 activation fragment, emitted
 *        from a single launch over the combined output-row space.
 *
 * Shape/format-named port of embeddinggemma.c `q4_mfma_qkv_projection_kernel`
 * (and its `_wide` sibling / `qkv_mfma_target` host dispatch, src/engine_rocm.hip).
 * The Q (N=768), K (N=256) and V (N=256) projections all read the same
 * (M,768) fp16 activation over the same K=768 contraction; the fusion routes a
 * combined 16-row output tile to whichever of the three Q4_0 weight matrices it
 * lands in (heterogeneous output-row counts, one shared activation fragment) and
 * accumulates with v_mfma_f32_16x16x16_f16 (Q4_0 dequantized to fp16, the same
 * primitive as qgemm/qflux/qgeglu). This is the exact structural analog of the
 * qgeglu up+gate fusion applied to the attention projection.
 *
 * This is the COMBINED PROJECTION ONLY. Norm and RoPE are deliberately NOT
 * folded in: the QKV+norm+RoPE mega-fusion was measured at 0.997-1.008x on
 * CDNA3 (coarse fusion collapses occupancy). See README.cdna3.md.
 *
 * Contract, token-major, tile 16x16 per wavefront:
 *   Wq (Nq=768, K=768) Q4_0 ; Wk,Wv (Nkv=256, K=768) Q4_0   (row-major over N)
 *   X  (M, K=768) fp16                                       (token-major)
 *   Q  (M, 768) f32 ; K (M, 256) f32 ; V (M, 256) f32        (token-major)
 * K, N, M multiples of 16 for the fast path (edge rows/tokens masked).
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
// EmbeddingGemma-300M attention-projection shape (src/model.h):
//   EI_N_EMBD = 768 (contraction K, and Q output rows), EI_HEAD_DIM = 256.
constexpr int kNEmbd = 768;     // K (contraction) and Q output-row count
constexpr int kHeadDim = 256;   // K and V output-row count

struct __align__(2) block_q4_0 {
    __half d;
    uint8_t qs[16];
};

typedef __attribute__((__vector_size__(4 * sizeof(__fp16)))) __fp16 half4_t;
typedef __attribute__((__vector_size__(4 * sizeof(float)))) float float4_t;

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

// Route one combined 16-row tile to the Q, K or V weight/output span.
// Faithful port of engine_rocm.hip `qkv_mfma_target`.
__device__ __forceinline__ void qkv_target(
    int combined_row_start,
    const block_q4_0 *q_weights, const block_q4_0 *k_weights, const block_q4_0 *v_weights,
    float *q_output, float *k_output, float *v_output,
    const block_q4_0 *&weights, float *&output, int &row_start, int &output_rows) {
    if (combined_row_start < kNEmbd) {
        weights = q_weights;
        output = q_output;
        row_start = combined_row_start;
        output_rows = kNEmbd;
    } else if (combined_row_start < kNEmbd + kHeadDim) {
        weights = k_weights;
        output = k_output;
        row_start = combined_row_start - kNEmbd;
        output_rows = kHeadDim;
    } else {
        weights = v_weights;
        output = v_output;
        row_start = combined_row_start - kNEmbd - kHeadDim;
        output_rows = kHeadDim;
    }
}

// Candidate: fused combined Q/K/V projection in one launch over the combined
// row space (Nq + Nk + Nv rows), one shared X fragment, MFMA per tile.
__global__ void q4_qkv_fused_kernel(
    const block_q4_0 *q_weights, const block_q4_0 *k_weights,
    const block_q4_0 *v_weights, const __half *input,
    float *q_output, float *k_output, float *v_output, int n_tokens) {
    constexpr int combined_rows = kNEmbd + 2 * kHeadDim;
    const int combined_row_start = static_cast<int>(blockIdx.x) * 16;
    const int token_start = static_cast<int>(blockIdx.y) * 16;
    const block_q4_0 *weights;
    float *output;
    int row_start;
    int output_rows;
    qkv_target(combined_row_start, q_weights, k_weights, v_weights,
               q_output, k_output, v_output, weights, output, row_start, output_rows);
    float4_t accumulator = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int col_start = 0; col_start < kNEmbd; col_start += 16) {
        const half4_t in = load_input_mfma_fragment(
            input, n_tokens, kNEmbd, token_start, col_start);
        const half4_t w = load_q4_mfma_fragment(
            weights, output_rows, kNEmbd, row_start, col_start);
        accumulator = mfma_m16n16k16(in, w, accumulator);
    }
    const int lane = threadIdx.x & 63;
    const int row = row_start + (lane & 15);
    const int token = token_start + (lane >> 4) * 4;
#pragma unroll
    for (int item = 0; item < 4; item++) {
        if (combined_row_start < combined_rows && row < output_rows &&
            token + item < n_tokens) {
            output[static_cast<size_t>(token + item) * output_rows + row] =
                accumulator[item];
        }
    }
}

// Baseline piece: one Q4_0 weight-only MFMA projection to fp32 (unfused route
// = three separate launches, one per Q/K/V).
__global__ void q4_mfma_proj_kernel(
    const block_q4_0 *weights, const __half *input, float *output,
    int n_tokens, int n_rows, int n_cols) {
    const int row_start = static_cast<int>(blockIdx.x) * 16;
    const int token_start = static_cast<int>(blockIdx.y) * 16;
    float4_t acc = {0.0f, 0.0f, 0.0f, 0.0f};
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

// fp64 host reference for one Q4_0 weight-only projection (fp16-rounded weight
// scale and fp16 inputs, matching the device dequant+MFMA operands).
static std::vector<float> host_ref_proj(const std::vector<block_q4_0>& W,
                                        const std::vector<__half>& X,
                                        int N, int K, int M) {
    const int bpr = K / kQK;
    std::vector<float> Y(static_cast<size_t>(M) * N);
    for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n) {
        double acc = 0.0;
        for (int b = 0; b < bpr; ++b) {
            const block_q4_0& w = W[static_cast<size_t>(n) * bpr + b];
            const double d = __half2float(w.d);
            for (int i = 0; i < 16; ++i) {
                const double xlo = __half2float(X[static_cast<size_t>(m) * K + b * kQK + i]);
                const double xhi = __half2float(X[static_cast<size_t>(m) * K + b * kQK + i + 16]);
                acc += d * (double)((w.qs[i] & 0x0f) - 8) * xlo;
                acc += d * (double)((w.qs[i] >> 4) - 8) * xhi;
            }
        }
        Y[static_cast<size_t>(m) * N + n] = (float)acc;
    }
    return Y;
}

static int check(const char* name, const std::vector<float>& got,
                 const std::vector<float>& ref, double tol) {
    double sad = 0.0, sref = 0.0, maxe = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double d = std::abs((double)got[i] - (double)ref[i]);
        sad += d; sref += std::abs((double)ref[i]); maxe = std::max(maxe, d);
    }
    const double rel = sad / std::max(sref, 1e-30);
    const bool ok = rel < tol;
    printf("%-26s rel %.3e max %.3e %s\n", name, rel, maxe, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// Max abs difference between two device output buffers (fused vs unfused).
static double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        m = std::max(m, std::abs((double)a[i] - (double)b[i]));
    return m;
}

static int run_case(int M, bool bench) {
    const int Nq = kNEmbd, Nkv = kHeadDim, K = kNEmbd;
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> Wqf(static_cast<size_t>(Nq) * K);
    std::vector<float> Wkf(static_cast<size_t>(Nkv) * K);
    std::vector<float> Wvf(static_cast<size_t>(Nkv) * K);
    std::vector<__half> X(static_cast<size_t>(M) * K);
    for (auto& v : Wqf) v = nd(rng) * 0.1f;
    for (auto& v : Wkf) v = nd(rng) * 0.1f;
    for (auto& v : Wvf) v = nd(rng) * 0.1f;
    for (auto& v : X) v = __float2half(nd(rng) * 0.5f);
    auto Wq = quantize_q4_0(Wqf, Nq, K);
    auto Wk = quantize_q4_0(Wkf, Nkv, K);
    auto Wv = quantize_q4_0(Wvf, Nkv, K);
    auto refQ = host_ref_proj(Wq, X, Nq, K, M);
    auto refK = host_ref_proj(Wk, X, Nkv, K, M);
    auto refV = host_ref_proj(Wv, X, Nkv, K, M);

    auto dWq = dnew(Wq); auto dWk = dnew(Wk); auto dWv = dnew(Wv); auto dX = dnew(X);
    float *dQf, *dKf, *dVf, *dQb, *dKb, *dVb;
    CK(hipMalloc(&dQf, refQ.size() * sizeof(float)));
    CK(hipMalloc(&dKf, refK.size() * sizeof(float)));
    CK(hipMalloc(&dVf, refV.size() * sizeof(float)));
    CK(hipMalloc(&dQb, refQ.size() * sizeof(float)));
    CK(hipMalloc(&dKb, refK.size() * sizeof(float)));
    CK(hipMalloc(&dVb, refV.size() * sizeof(float)));

    const int tok16 = (M + 15) / 16;
    dim3 fused_grid((Nq + 2 * Nkv) / 16, tok16);
    auto runFused = [&] {
        q4_qkv_fused_kernel<<<fused_grid, 64>>>(dWq, dWk, dWv, dX, dQf, dKf, dVf, M);
    };
    auto runUnfused = [&] {
        q4_mfma_proj_kernel<<<dim3((Nq + 15) / 16, tok16), 64>>>(dWq, dX, dQb, M, Nq, K);
        q4_mfma_proj_kernel<<<dim3((Nkv + 15) / 16, tok16), 64>>>(dWk, dX, dKb, M, Nkv, K);
        q4_mfma_proj_kernel<<<dim3((Nkv + 15) / 16, tok16), 64>>>(dWv, dX, dVb, M, Nkv, K);
    };
    runFused(); runUnfused();
    CK(hipDeviceSynchronize()); CK(hipGetLastError());

    auto hQf = d2h(dQf, refQ.size()), hKf = d2h(dKf, refK.size()), hVf = d2h(dVf, refV.size());
    auto hQb = d2h(dQb, refQ.size()), hKb = d2h(dKb, refK.size()), hVb = d2h(dVb, refV.size());

    int rc = 0;
    printf("--- M=%d ---\n", M);
    rc |= check("qkv_fused Q", hQf, refQ, 1e-2);
    rc |= check("qkv_fused K", hKf, refK, 1e-2);
    rc |= check("qkv_fused V", hVf, refV, 1e-2);
    rc |= check("qkv_unfused Q (base)", hQb, refQ, 1e-2);
    rc |= check("qkv_unfused K (base)", hKb, refK, 1e-2);
    rc |= check("qkv_unfused V (base)", hVb, refV, 1e-2);
    const double pd = std::max({max_abs_diff(hQf, hQb), max_abs_diff(hKf, hKb),
                                max_abs_diff(hVf, hVb)});
    printf("%-26s max %.3e %s\n", "fused-vs-unfused parity", pd,
           pd == 0.0 ? "BIT-IDENTICAL" : "");

    if (bench && rc == 0) {
        const float tf = bench_ms(runFused);
        const float tb = bench_ms(runUnfused);
        const double ops = 2.0 * (double)M * K * (Nq + 2.0 * Nkv);
        printf("qkv bench M=%d (Nq=%d Nkv=%d K=%d)\n", M, Nq, Nkv, K);
        printf("  unfused(base) %.4f ms  %.2f TFLOP/s\n", tb, ops / (tb * 1e-3) / 1e12);
        printf("  fused(cand)   %.4f ms  %.2f TFLOP/s  (%.3fx)\n",
               tf, ops / (tf * 1e-3) / 1e12, tb / tf);
    }
    CK(hipFree(dWq)); CK(hipFree(dWk)); CK(hipFree(dWv)); CK(hipFree(dX));
    CK(hipFree(dQf)); CK(hipFree(dKf)); CK(hipFree(dVf));
    CK(hipFree(dQb)); CK(hipFree(dKb)); CK(hipFree(dVb));
    return rc;
}

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    int rc = 0;
    rc |= run_case(16, false);
    rc |= run_case(64, false);
    if (bench) {
        // Sweep the native-MFMA band (~32-368 tokens) plus above-band points,
        // reported honestly.
        rc |= run_case(32, true);
        rc |= run_case(64, true);
        rc |= run_case(128, true);
        rc |= run_case(256, true);
        rc |= run_case(368, true);
        rc |= run_case(512, true);
        rc |= run_case(1024, true);
    }
    printf(rc == 0 ? "ALL PASS\n" : "FAIL\n");
    return rc;
}
