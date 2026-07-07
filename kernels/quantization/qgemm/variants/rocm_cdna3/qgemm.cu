#include "hip/hip_runtime.h"
// Weight-only quantized GEMM, CUDA/SM86 port of ThunderMittens kernels/qgemm.
// Torch-linear semantics: Y(M,N) = X(M,K) @ dequant(Wq(N,K))^T, fp16 X, fp32 accum.
//
// Two paths, mirroring TM:
//  - fragment path (Marlin zero-shuffle): FMT::dequant straight into the mma B
//    fragment (load_wfrag<FMT>), no shared staging, no barriers. Used for
//    block_k <= 64 formats (all the bit-arithmetic ones incl. mxfp8/mxfp4/nvfp4).
//  - full-dequant route for the branchy 256-superblock k/i-quants: dequant the
//    whole W to fp16 once (dequant_all), then run the SAME kernel with the
//    fp16_raw passthrough format. (TM measured dequant-then-GEMM 2-2.3x faster
//    than in-GEMM branchy dequant for these formats at M>=64.)
//
// Correctness-first: W fragments read straight from global (L2-cached; the
// cp.async ring + wider tiles are the perf pass). One warp per 16x16 output tile.
//
// Build:
//   /usr/local/cuda/bin/nvcc qgemm.cu -std=c++20 -O2 -DKITTENS_SM86 \
//     -gencode arch=compute_86,code=sm_86 -o qgemm.out
#include "tm_qmm_mfma.cuh"
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>

using namespace tmq;

// ---- qgemm: one 64-wide wavefront per 16x16 output tile (CDNA3 MFMA) ----
// Y(M,N) = X(M,K) @ W(N,K)^T. A=X, B=W^T; one v_mfma_f32_16x16x16_f16 per K=16.
// Lane l owns output column n0+l%16, rows m0 + 4*(l/16) + {0..3}.
template<typename FMT>
__global__ void qgemm(float* Y, const __half* X, const uint8_t* Wq, int M, int N, int K) {
    const int n0 = blockIdx.x * 16;
    const int m0 = blockIdx.y * 16;
    const int bpr = K / FMT::block_k;

    float4_t acc = {0, 0, 0, 0};
    for (int k0 = 0; k0 < K; k0 += 16) {
        half4_t a = load_xfrag(X, K, m0, k0);
        half4_t b = load_wfrag<FMT>(Wq, bpr, n0, k0);
        acc = mma_16x16x16(a, b, acc);
    }
    const int l = threadIdx.x & 63;
    const int n = n0 + (l & 15);
    const int mrow = m0 + (l >> 4) * 4;
    #pragma unroll
    for (int v = 0; v < 4; v++)
        if (mrow + v < M) Y[size_t(mrow + v) * N + n] = acc[v];
}

// wide N-tile variant (perf pass for M>=~256): NT 16-wide N-tiles per wavefront;
// the X fragment is loaded once per k-step and reused across NT W-fragments, so
// X traffic drops NT-fold and the MFMA:load ratio rises NT-fold. Bitwise-identical
// to qgemm (same fragment math). Occupancy-limited at tiny M (see qgemm_pick_nt).
template<typename FMT, int NT>
__global__ void qgemm_wide(float* Y, const __half* X, const uint8_t* Wq, int M, int N, int K) {
    const int n0 = blockIdx.x * (16 * NT);
    const int m0 = blockIdx.y * 16;
    const int bpr = K / FMT::block_k;
    float4_t acc[NT];
    #pragma unroll
    for (int nt = 0; nt < NT; nt++) acc[nt] = float4_t{0, 0, 0, 0};
    for (int k0 = 0; k0 < K; k0 += 16) {
        half4_t a = load_xfrag(X, K, m0, k0);
        #pragma unroll
        for (int nt = 0; nt < NT; nt++) {
            half4_t b = load_wfrag<FMT>(Wq, bpr, n0 + nt * 16, k0);
            acc[nt] = mma_16x16x16(a, b, acc[nt]);
        }
    }
    const int l = threadIdx.x & 63, ln = l & 15, mrow = m0 + (l >> 4) * 4;
    #pragma unroll
    for (int nt = 0; nt < NT; nt++) { const int n = n0 + nt * 16 + ln;
        #pragma unroll
        for (int v = 0; v < 4; v++) if (mrow + v < M) Y[size_t(mrow + v) * N + n] = acc[nt][v];
    }
}

// Pick NT (N-tiles/wavefront) so the grid still fills the device (~2 waves over
// 304 CUs); wider tiles amortize the X load but shrink the grid, so cap by
// occupancy and require 16*NT | N. NT=1 => identical to qgemm (decode fallback).
static inline int qgemm_pick_nt(int M, int N) {
    const long tilesM = (M + 15) / 16;
    for (int nt : {4, 2}) if (N % (16 * nt) == 0 && (long)(N / (16 * nt)) * tilesM >= 608) return nt;
    return 1;
}

// full-dequant kernel (route for 256-superblock formats)
template<typename FMT>
__global__ void dequant_to_fp16(half* out, const uint8_t* Wq, int N, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * K) return;
    int row = idx / K, col = idx % K;
    const uint8_t* base = Wq + (size_t(row) * (K / FMT::block_k) + col / FMT::block_k) * FMT::block_bytes;
    out[idx] = __float2half(FMT::dequant(base, col % FMT::block_k));
}

// ---- ksplit variant (perf pass): at decode shapes the warp-per-tile grid is
// only (N/16)*(M/16) warps — half the SMs idle at M=64,N=512. Slice K across
// blockIdx.z and atomicAdd fp32 partials into a zeroed Y. Same fragment math.
// (Also consolidated in tm_kernels.cuh as tmq::qgemm_ksplit.) ----
template<typename FMT>
__global__ void qgemm_ksplit(float* Y, const __half* X, const uint8_t* Wq,
                             int M, int N, int K, int k_chunk) {
    const int n0 = blockIdx.x * 16;
    const int m0 = blockIdx.y * 16;
    const int k_beg = blockIdx.z * k_chunk;
    const int k_end = min(K, k_beg + k_chunk);
    const int bpr = K / FMT::block_k;

    float4_t acc = {0, 0, 0, 0};
    for (int k0 = k_beg; k0 < k_end; k0 += 16) {
        half4_t a = load_xfrag(X, K, m0, k0);
        half4_t b = load_wfrag<FMT>(Wq, bpr, n0, k0);
        acc = mma_16x16x16(a, b, acc);
    }
    const int l = threadIdx.x & 63;
    const int n = n0 + (l & 15);
    const int mrow = m0 + (l >> 4) * 4;
    #pragma unroll
    for (int v = 0; v < 4; v++)
        if (mrow + v < M) atomicAdd(&Y[size_t(mrow + v) * N + n], acc[v]);
}

static inline int qgemm_pick_kchunk(int M, int N, int K, int block_k) {
    const long tiles = long((N + 15) / 16) * ((M + 15) / 16);
    const int target = 1664;                      // ~82 SMs x 20 warps
    int splits = int((target + tiles - 1) / tiles);
    const int align = (block_k > 16) ? block_k : 16;
    int chunk = ((K / (splits > 0 ? splits : 1)) + align - 1) / align * align;
    if (chunk < align) chunk = align;
    return chunk;
}

// ---- harness ----
static std::vector<uint8_t> read_file(const std::string& p) {
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) { fprintf(stderr, "missing %s\n", p.c_str()); exit(2); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> v(n);
    if (fread(v.data(), 1, n, f) != size_t(n)) exit(2);
    fclose(f);
    return v;
}

template<typename FMT>
int run(const std::string& dir, int N, int K) {
    constexpr bool SUPERBLOCK = FMT::block_k > 64;   // route k/i-quants via full dequant
    const int M = 64;
    auto Wq_h = read_file(dir + "/Wq.bin");
    auto X_h  = read_file(dir + "/X2.bin");     // fp16 (M,K)
    auto Y_h  = read_file(dir + "/Y_ref.bin");  // fp32 (M,N) = X @ Wdeq^T

    uint8_t* dWq; half* dX; float* dY; half* dWf = nullptr;
    hipMalloc(&dWq, Wq_h.size());
    hipMalloc(&dX, sizeof(half) * M * K);
    hipMalloc(&dY, sizeof(float) * size_t(M) * N);
    hipMemcpy(dWq, Wq_h.data(), Wq_h.size(), hipMemcpyHostToDevice);
    hipMemcpy(dX, X_h.data(), X_h.size(), hipMemcpyHostToDevice);

    dim3 grid(N / 16, (M + 15) / 16);
    hipEvent_t t0, t1; hipEventCreate(&t0); hipEventCreate(&t1);
    auto launch = [&] {
        if constexpr (SUPERBLOCK) {
            dequant_to_fp16<FMT><<<(N * K + 255) / 256, 256>>>(dWf, dWq, N, K);
            qgemm<fp16_raw><<<grid, 64>>>(dY, dX, reinterpret_cast<const uint8_t*>(dWf), M, N, K);
        } else {
            qgemm<FMT><<<grid, 64>>>(dY, dX, dWq, M, N, K);
        }
    };
    if (SUPERBLOCK) hipMalloc(&dWf, sizeof(half) * size_t(N) * K);
    launch();
    hipDeviceSynchronize();
    if (hipGetLastError() != hipSuccess) { printf("KERNEL ERROR\n"); return 1; }
    int iters = 50;
    hipEventRecord(t0);
    for (int i = 0; i < iters; i++) launch();
    hipEventRecord(t1); hipEventSynchronize(t1);
    float ms; hipEventElapsedTime(&ms, t0, t1); ms /= iters;

    std::vector<float> got(size_t(M) * N);
    hipMemcpy(got.data(), dY, sizeof(float) * got.size(), hipMemcpyDeviceToHost);
    const float* ref = reinterpret_cast<const float*>(Y_h.data());
    double gsum = 0, rsum = 0, gmax = 0;
    for (size_t i = 0; i < got.size(); i++) {
        double d = std::abs(double(got[i]) - double(ref[i]));
        gmax = std::max(gmax, d); gsum += d; rsum += std::abs(double(ref[i]));
    }
    double rel = gsum / std::max(rsum, 1e-30);
    double tflop = 2.0 * M * N * K / 1e12;
    printf("qgemm%s: rel %.4f%% max %.4g | %.3f ms  %.2f TFLOP/s  (%s)\n",
           SUPERBLOCK ? "[dequant-route]" : "", 100 * rel, gmax, ms, tflop / (ms / 1e3),
           rel < 0.02 ? "PASS" : "FAIL");
    int rc = rel < 0.02 ? 0 : 1;

    // ---- ksplit variant: K sliced across blockIdx.z + fp32 atomic combine ----
    {
        const int chunk = qgemm_pick_kchunk(M, N, K, SUPERBLOCK ? 16 : FMT::block_k);
        const int splits = (K + chunk - 1) / chunk;
        dim3 gridz(N / 16, (M + 15) / 16, splits);
        auto launch2 = [&] {
            hipMemsetAsync(dY, 0, sizeof(float) * size_t(M) * N);
            if constexpr (SUPERBLOCK) {
                dequant_to_fp16<FMT><<<(N * K + 255) / 256, 256>>>(dWf, dWq, N, K);
                qgemm_ksplit<fp16_raw><<<gridz, 64>>>(dY, dX,
                    reinterpret_cast<const uint8_t*>(dWf), M, N, K, chunk);
            } else {
                qgemm_ksplit<FMT><<<gridz, 64>>>(dY, dX, dWq, M, N, K, chunk);
            }
        };
        launch2();
        hipDeviceSynchronize();
        if (hipGetLastError() != hipSuccess) { printf("KSPLIT KERNEL ERROR\n"); return 1; }
        hipEventRecord(t0);
        for (int i = 0; i < iters; i++) launch2();
        hipEventRecord(t1); hipEventSynchronize(t1);
        hipEventElapsedTime(&ms, t0, t1); ms /= iters;
        hipMemcpy(got.data(), dY, sizeof(float) * got.size(), hipMemcpyDeviceToHost);
        gsum = 0; rsum = 0; gmax = 0;
        for (size_t i = 0; i < got.size(); i++) {
            double d = std::abs(double(got[i]) - double(ref[i]));
            gmax = std::max(gmax, d); gsum += d; rsum += std::abs(double(ref[i]));
        }
        rel = gsum / std::max(rsum, 1e-30);
        printf("qgemm-ksplit(x%d)%s: rel %.4f%% max %.4g | %.3f ms  %.2f TFLOP/s  (%s)\n",
               splits, SUPERBLOCK ? "[dequant-route]" : "", 100 * rel, gmax, ms,
               tflop / (ms / 1e3), rel < 0.02 ? "PASS" : "FAIL");
        rc |= !(rel < 0.02);
    }

    // ---- wide N-tile variant: golden-validate at NT=4 (the golden dir is the
    // decode M=64 shape, occupancy-unfavorable for wide, so qgemm_pick_nt(M,N)
    // would pick NT=%d in production; forced NT=4 here purely to check correctness
    // vs golden — the prefill/large-M A/B is in qgemm_bench.cu). ----
    if (N % (16 * 4) == 0) {
        dim3 gridw(N / (16 * 4), (M + 15) / 16);
        auto launchw = [&] {
            if constexpr (SUPERBLOCK) {
                dequant_to_fp16<FMT><<<(N * K + 255) / 256, 256>>>(dWf, dWq, N, K);
                qgemm_wide<fp16_raw, 4><<<gridw, 64>>>(dY, dX, reinterpret_cast<const uint8_t*>(dWf), M, N, K);
            } else {
                qgemm_wide<FMT, 4><<<gridw, 64>>>(dY, dX, dWq, M, N, K);
            }
        };
        launchw();
        hipDeviceSynchronize();
        if (hipGetLastError() != hipSuccess) { printf("WIDE KERNEL ERROR\n"); return 1; }
        hipMemcpy(got.data(), dY, sizeof(float) * got.size(), hipMemcpyDeviceToHost);
        gsum = 0; rsum = 0; gmax = 0;
        for (size_t i = 0; i < got.size(); i++) {
            double d = std::abs(double(got[i]) - double(ref[i]));
            gmax = std::max(gmax, d); gsum += d; rsum += std::abs(double(ref[i]));
        }
        rel = gsum / std::max(rsum, 1e-30);
        printf("qgemm-wide(NT=4)%s: rel %.4f%% max %.4g  (%s)  [pick_nt=%d]\n",
               SUPERBLOCK ? "[dequant-route]" : "", 100 * rel, gmax,
               rel < 0.02 ? "PASS" : "FAIL", qgemm_pick_nt(M, N));
        rc |= !(rel < 0.02);
    }
    return rc;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <golden_dir>\n", argv[0]); return 2; }
    std::string dir = argv[1];
    // Checked-in golden omits meta.txt: infer format from dir basename, dims
    // default to gen_golden.py (N=512, K=4096). Override: <dir> [N K].
    char fmt[64]; int N = 512, K = 4096;
    FILE* f = fopen((dir + "/meta.txt").c_str(), "r");
    if (f) {
        if (fscanf(f, "%63s %d %d", fmt, &N, &K) != 3) return 2;
        fclose(f);
    } else {
        std::string base = dir;
        while (!base.empty() && base.back() == '/') base.pop_back();
        auto slash = base.find_last_of('/');
        snprintf(fmt, sizeof(fmt), "%s", (slash == std::string::npos ? base : base.substr(slash + 1)).c_str());
        if (argc >= 4) { N = atoi(argv[2]); K = atoi(argv[3]); }
    }
    printf("== qgemm %s  N=%d K=%d M=64\n", fmt, N, K);
    std::string s(fmt);
    if (s == "q8_0")       return run<q8_0>(dir, N, K);
    if (s == "q4_0")       return run<q4_0>(dir, N, K);
    if (s == "q4_1")       return run<q4_1>(dir, N, K);
    if (s == "q5_0")       return run<q5_0>(dir, N, K);
    if (s == "q5_1")       return run<q5_1>(dir, N, K);
    if (s == "kU4B8")      return run<kU4B8>(dir, N, K);
    if (s == "kU4")        return run<kU4>(dir, N, K);
    if (s == "hqq")        return run<hqq>(dir, N, K);
    if (s == "fp8_e4m3")   return run<fp8_e4m3>(dir, N, K);
    if (s == "e5m2")       return run<e5m2>(dir, N, K);
    if (s == "fp8_block")  return run<fp8_block>(dir, N, K);
    if (s == "fp4_e2m1")   return run<fp4_e2m1>(dir, N, K);
    if (s == "mxfp8")      return run<mxfp8>(dir, N, K);
    if (s == "mxfp4")      return run<mxfp4>(dir, N, K);
    if (s == "nvfp4")      return run<nvfp4>(dir, N, K);
    if (s == "mxfp6_e3m2") return run<mxfp6_e3m2>(dir, N, K);
    if (s == "mxfp6_e2m3") return run<mxfp6_e2m3>(dir, N, K);
    if (s == "bitnet")     return run<bitnet>(dir, N, K);
    if (s == "q2_K")       return run<q2_K>(dir, N, K);
    if (s == "q3_K")       return run<q3_K>(dir, N, K);
    if (s == "q4_K")       return run<q4_K>(dir, N, K);
    if (s == "q5_K")       return run<q5_K>(dir, N, K);
    if (s == "q6_K")       return run<q6_K>(dir, N, K);
    if (s == "iq4_nl")     return run<iq4_nl>(dir, N, K);
    if (s == "iq4_xs")     return run<iq4_xs>(dir, N, K);
    if (s == "iq2_xxs")    return run<iq2_xxs>(dir, N, K);
    if (s == "iq2_xs")     return run<iq2_xs>(dir, N, K);
    if (s == "iq3_xxs")    return run<iq3_xxs>(dir, N, K);
    if (s == "iq1_s")      return run<iq1_s>(dir, N, K);
    fprintf(stderr, "unknown format %s\n", fmt);
    return 2;
}
