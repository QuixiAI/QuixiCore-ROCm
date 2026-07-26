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
#include "qgemm_kernels.cuh"
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>

using namespace tmq;


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
int run(const std::string& dir, int N, int K, int M) {
    constexpr bool SUPERBLOCK = FMT::block_k > 64;   // route k/i-quants via full dequant
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

    // ---- in-GEMM route for the 256-superblock k-quants: decode straight into
    // the MFMA B fragment via dequant4<FMT> (one sub-scale unpack per fragment).
    // The dequant route above materializes all of W as fp16, which costs N*K*2
    // bytes of extra footprint — 2 TB for a 262 GB Q2_K checkpoint — so serving
    // needs this path even where the dequant route is faster on a 512x4096 tile.
    if constexpr (SUPERBLOCK) {
        auto launch_ig = [&] { qgemm<FMT><<<grid, 64>>>(dY, dX, dWq, M, N, K); };
        launch_ig();
        hipDeviceSynchronize();
        if (hipGetLastError() != hipSuccess) { printf("IN-GEMM KERNEL ERROR\n"); return 1; }
        hipEventRecord(t0);
        for (int i = 0; i < iters; i++) launch_ig();
        hipEventRecord(t1); hipEventSynchronize(t1);
        float ms_ig; hipEventElapsedTime(&ms_ig, t0, t1); ms_ig /= iters;

        std::vector<float> got_ig(size_t(M) * N);
        hipMemcpy(got_ig.data(), dY, sizeof(float) * got_ig.size(), hipMemcpyDeviceToHost);
        double gsum_ig = 0, gmax_ig = 0, exact = 0;
        for (size_t i = 0; i < got_ig.size(); i++) {
            double d = std::abs(double(got_ig[i]) - double(ref[i]));
            gmax_ig = std::max(gmax_ig, d); gsum_ig += d;
            if (got_ig[i] != got[i]) exact += 1;
        }
        double rel_ig = gsum_ig / std::max(rsum, 1e-30);
        // fp16 upper bound: same kernel, W already dequantized (dWf still holds it).
        // ms_ig/ms_f16 is the true in-GEMM decode overhead.
        auto launch_f16 = [&] {
            qgemm<fp16_raw><<<grid, 64>>>(dY, dX, reinterpret_cast<const uint8_t*>(dWf), M, N, K);
        };
        launch_f16();
        hipDeviceSynchronize();
        hipEventRecord(t0);
        for (int i = 0; i < iters; i++) launch_f16();
        hipEventRecord(t1); hipEventSynchronize(t1);
        float ms_f16; hipEventElapsedTime(&ms_f16, t0, t1); ms_f16 /= iters;

        printf("qgemm[in-gemm]:  rel %.4f%% max %.4g | %.3f ms  %.2f TFLOP/s  "
               "(%s)  vs-dequant-route %.2fx  decode-overhead %.2fx  bitdiff %.0f/%zu\n",
               100 * rel_ig, gmax_ig, ms_ig, tflop / (ms_ig / 1e3),
               rel_ig < 0.02 ? "PASS" : "FAIL", ms / ms_ig, ms_ig / ms_f16,
               exact, got_ig.size());
        if (rel_ig >= 0.02) rc = 1;
    }

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

        // in-GEMM ksplit: the decode-shape serving path (K-slice restores
        // occupancy; the fragment decodes straight from the packed weights).
        if constexpr (SUPERBLOCK) {
            const int chunk_ig = qgemm_pick_kchunk(M, N, K, FMT::block_k);
            const int splits_ig = (K + chunk_ig - 1) / chunk_ig;
            dim3 gridz_ig(N / 16, (M + 15) / 16, splits_ig);
            auto launch_igk = [&] {
                hipMemsetAsync(dY, 0, sizeof(float) * size_t(M) * N);
                qgemm_ksplit<FMT><<<gridz_ig, 64>>>(dY, dX, dWq, M, N, K, chunk_ig);
            };
            launch_igk();
            hipDeviceSynchronize();
            if (hipGetLastError() != hipSuccess) { printf("IN-GEMM KSPLIT ERROR\n"); return 1; }
            hipEventRecord(t0);
            for (int i = 0; i < iters; i++) launch_igk();
            hipEventRecord(t1); hipEventSynchronize(t1);
            float ms_igk; hipEventElapsedTime(&ms_igk, t0, t1); ms_igk /= iters;
            hipMemcpy(got.data(), dY, sizeof(float) * got.size(), hipMemcpyDeviceToHost);
            double gs = 0, gm = 0;
            for (size_t i = 0; i < got.size(); i++) {
                double d = std::abs(double(got[i]) - double(ref[i]));
                gm = std::max(gm, d); gs += d;
            }
            double rel_igk = gs / std::max(rsum, 1e-30);
            printf("qgemm-ksplit(x%d)[in-gemm]: rel %.4f%% max %.4g | %.3f ms  "
                   "%.2f TFLOP/s  (%s)\n",
                   splits_ig, 100 * rel_igk, gm, ms_igk, tflop / (ms_igk / 1e3),
                   rel_igk < 0.02 ? "PASS" : "FAIL");
            rc |= !(rel_igk < 0.02);
        }
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
        if (M > 64) {
            hipEventRecord(t0);
            for (int i = 0; i < iters; i++) launchw();
            hipEventRecord(t1); hipEventSynchronize(t1);
            hipEventElapsedTime(&ms, t0, t1); ms /= iters;
            printf("qgemm-wide(NT=4)%s: %.3f ms  %.2f TFLOP/s\n",
                   SUPERBLOCK ? "[dequant-route]" : "", ms, tflop / (ms / 1e3));
        }

        dim3 gridc(N / (16 * 4), (M + 16 * 4 - 1) / (16 * 4));
        auto launchc = [&] {
            if constexpr (SUPERBLOCK) {
                dequant_to_fp16<FMT><<<(N * K + 255) / 256, 256>>>(dWf, dWq, N, K);
                qgemm_cta_lds<fp16_raw, 4, 4><<<gridc, 64 * 4>>>(
                    dY, dX, reinterpret_cast<const uint8_t*>(dWf), M, N, K);
            } else {
                qgemm_cta_lds<FMT, 4, 4><<<gridc, 64 * 4>>>(dY, dX, dWq, M, N, K);
            }
        };
        launchc();
        hipDeviceSynchronize();
        if (hipGetLastError() != hipSuccess) { printf("CTA-LDS KERNEL ERROR\n"); return 1; }
        hipMemcpy(got.data(), dY, sizeof(float) * got.size(), hipMemcpyDeviceToHost);
        gsum = 0; rsum = 0; gmax = 0;
        for (size_t i = 0; i < got.size(); i++) {
            double d = std::abs(double(got[i]) - double(ref[i]));
            gmax = std::max(gmax, d); gsum += d; rsum += std::abs(double(ref[i]));
        }
        rel = gsum / std::max(rsum, 1e-30);
        printf("qgemm-ctaLDS(4x4)%s: rel %.4f%% max %.4g  (%s)\n",
               SUPERBLOCK ? "[dequant-route]" : "", 100 * rel, gmax,
               rel < 0.02 ? "PASS" : "FAIL");
        rc |= !(rel < 0.02);
        if (M > 64) {
            hipEventRecord(t0);
            for (int i = 0; i < iters; i++) launchc();
            hipEventRecord(t1); hipEventSynchronize(t1);
            hipEventElapsedTime(&ms, t0, t1); ms /= iters;
            printf("qgemm-ctaLDS(4x4)%s: %.3f ms  %.2f TFLOP/s\n",
                   SUPERBLOCK ? "[dequant-route]" : "", ms, tflop / (ms / 1e3));
        }
    }
    return rc;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <golden_dir>\n", argv[0]); return 2; }
    std::string dir = argv[1];
    // Checked-in golden omits meta.txt: infer format from dir basename, dims
    // default to gen_golden.py (N=512, K=4096, M=64). Override: <dir> [N K M].
    char fmt[64]; int N = 512, K = 4096, M = 64;
    FILE* f = fopen((dir + "/meta.txt").c_str(), "r");
    if (f) {
        int got = fscanf(f, "%63s %d %d %d", fmt, &N, &K, &M);
        if (got != 3 && got != 4) return 2;
        if (got == 3) M = 64;
        fclose(f);
    } else {
        std::string base = dir;
        while (!base.empty() && base.back() == '/') base.pop_back();
        auto slash = base.find_last_of('/');
        snprintf(fmt, sizeof(fmt), "%s", (slash == std::string::npos ? base : base.substr(slash + 1)).c_str());
        if (argc >= 4) { N = atoi(argv[2]); K = atoi(argv[3]); }
        if (argc >= 5) M = atoi(argv[4]);
    }
    if (argc >= 5) M = atoi(argv[4]);
    printf("== qgemm %s  N=%d K=%d M=%d\n", fmt, N, K, M);
    std::string s(fmt);
    if (s == "q8_0")       return run<q8_0>(dir, N, K, M);
    if (s == "q4_0")       return run<q4_0>(dir, N, K, M);
    if (s == "q4_1")       return run<q4_1>(dir, N, K, M);
    if (s == "q5_0")       return run<q5_0>(dir, N, K, M);
    if (s == "q5_1")       return run<q5_1>(dir, N, K, M);
    if (s == "kU4B8")      return run<kU4B8>(dir, N, K, M);
    if (s == "kU4")        return run<kU4>(dir, N, K, M);
    if (s == "hqq")        return run<hqq>(dir, N, K, M);
    if (s == "fp8_e4m3")   return run<fp8_e4m3>(dir, N, K, M);
    if (s == "e5m2")       return run<e5m2>(dir, N, K, M);
    if (s == "fp8_block")  return run<fp8_block>(dir, N, K, M);
    if (s == "fp4_e2m1")   return run<fp4_e2m1>(dir, N, K, M);
    if (s == "mxfp8")      return run<mxfp8>(dir, N, K, M);
    if (s == "mxfp4")      return run<mxfp4>(dir, N, K, M);
    if (s == "nvfp4")      return run<nvfp4>(dir, N, K, M);
    if (s == "mxfp6_e3m2") return run<mxfp6_e3m2>(dir, N, K, M);
    if (s == "mxfp6_e2m3") return run<mxfp6_e2m3>(dir, N, K, M);
    if (s == "bitnet")     return run<bitnet>(dir, N, K, M);
    if (s == "q2_K")       return run<q2_K>(dir, N, K, M);
    if (s == "q3_K")       return run<q3_K>(dir, N, K, M);
    if (s == "q4_K")       return run<q4_K>(dir, N, K, M);
    if (s == "q5_K")       return run<q5_K>(dir, N, K, M);
    if (s == "q6_K")       return run<q6_K>(dir, N, K, M);
    if (s == "iq4_nl")     return run<iq4_nl>(dir, N, K, M);
    if (s == "iq4_xs")     return run<iq4_xs>(dir, N, K, M);
    if (s == "iq2_xxs")    return run<iq2_xxs>(dir, N, K, M);
    if (s == "iq2_xs")     return run<iq2_xs>(dir, N, K, M);
    if (s == "iq3_xxs")    return run<iq3_xxs>(dir, N, K, M);
    if (s == "iq1_s")      return run<iq1_s>(dir, N, K, M);
    fprintf(stderr, "unknown format %s\n", fmt);
    return 2;
}
