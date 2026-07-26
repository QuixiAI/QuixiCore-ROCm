#include "hip/hip_runtime.h"
/**
 * @file
 * @brief In-GEMM k-quant A/B at real serving shapes. The golden harness runs a
 * single 512x4096 tile at M=64, which is occupancy-bound (128 wavefronts on 304
 * CUs) and therefore says little about the decode cost. This sweeps the GLM-5.2
 * projection shapes across decode -> prefill M, comparing the three in-GEMM
 * routes (register fragment, K-split, multi-wave CTA/LDS) against the fp16
 * upper bound on the same tile.
 *
 * Timing only — correctness for every format is the golden harness's job
 * (`make test`), so weights here are random bytes. No k-quant decoder branches
 * on weight VALUES (only on column position), so random bytes are timing-valid.
 *
 * Modes:
 *   ./qgemm_q2k_bench.out                 q2_K route sweep over GLM-5.2 shapes
 *   ./qgemm_q2k_bench.out M N K           q2_K, one shape
 *   ./qgemm_q2k_bench.out --formats       q8_0 vs q6_K/q5_K/q4_K at the
 *                                         attention shapes, with bytes/weight
 */
#include "qgemm_kernels.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <cmath>

using namespace tmq;

#define CK(x) do { hipError_t e=(x); if(e){printf("HIP %s @%d\n",hipGetErrorString(e),__LINE__);exit(1);} } while(0)

template<class L> static double med(L fn, int warm = 10, int it = 50) {
    for (int i = 0; i < warm; i++) fn();
    CK(hipDeviceSynchronize());
    hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
    hipEventRecord(a);
    for (int i = 0; i < it; i++) fn();
    hipEventRecord(b); CK(hipEventSynchronize(b));
    float ms; hipEventElapsedTime(&ms, a, b);
    hipEventDestroy(a); hipEventDestroy(b);
    return ms / it;
}

static std::vector<uint8_t> rand_w(size_t n, std::mt19937& rng) {
    std::vector<uint8_t> v(n);
    std::uniform_int_distribution<int> byte(0, 255);
    for (auto& b : v) b = (uint8_t)byte(rng);
    return v;
}

// ---- q2_K route sweep: fragment vs ksplit vs ctaLDS, all vs the fp16 bound ----
static void run(int M, int N, int K) {
    const int bpr = K / q2_K::block_k;
    const size_t wbytes = (size_t)N * bpr * q2_K::block_bytes;
    std::mt19937 rng(7);
    auto Wh = rand_w(wbytes, rng);
    std::vector<__half> Xh((size_t)M * K);
    std::uniform_real_distribution<float> uf(-1, 1);
    for (auto& x : Xh) x = __float2half(uf(rng));

    uint8_t* dW; __half* dX; float* dY; __half* dWf;
    CK(hipMalloc(&dW, wbytes)); CK(hipMemcpy(dW, Wh.data(), wbytes, hipMemcpyHostToDevice));
    CK(hipMalloc(&dX, sizeof(__half) * Xh.size()));
    CK(hipMemcpy(dX, Xh.data(), sizeof(__half) * Xh.size(), hipMemcpyHostToDevice));
    CK(hipMalloc(&dY, sizeof(float) * (size_t)M * N));
    CK(hipMalloc(&dWf, sizeof(__half) * (size_t)N * K));

    dim3 gB(N / 16, (M + 15) / 16);
    dequant_to_fp16<q2_K><<<(int)(((size_t)N * K + 255) / 256), 256>>>(dWf, dW, N, K);
    CK(hipDeviceSynchronize());
    const double t_f16 = med([&]{ qgemm<fp16_raw><<<gB, 64>>>(
        dY, dX, reinterpret_cast<const uint8_t*>(dWf), M, N, K); });
    const double t_base = med([&]{ qgemm<q2_K><<<gB, 64>>>(dY, dX, dW, M, N, K); });

    const int chunk = qgemm_pick_kchunk(M, N, K, q2_K::block_k);
    const int splits = (K + chunk - 1) / chunk;
    dim3 gZ(N / 16, (M + 15) / 16, splits);
    const double t_ks = med([&]{
        CK(hipMemsetAsync(dY, 0, sizeof(float) * (size_t)M * N));
        qgemm_ksplit<q2_K><<<gZ, 64>>>(dY, dX, dW, M, N, K, chunk); });

    double t_cta = 0;
    const int MT = 4, NT = 4;
    const bool can_cta = (M % (16 * MT) == 0) && (N % (16 * NT) == 0);
    if (can_cta) {
        dim3 gC(N / (16 * NT), M / (16 * MT));
        t_cta = med([&]{ qgemm_cta_lds<q2_K, MT, NT><<<gC, 64 * MT>>>(dY, dX, dW, M, N, K); });
    }

    const double tflop = 2.0 * M * N * K / 1e12;
    auto tf = [&](double ms) { return tflop / (ms / 1e3); };
    printf("M=%-5d N=%-6d K=%-6d | fp16 %7.2f | frag %7.2f (%.2fx) | ksplit x%-2d %7.2f (%.2fx) | ",
           M, N, K, tf(t_f16), tf(t_base), t_base / t_f16, splits, tf(t_ks), t_ks / t_f16);
    if (can_cta) printf("ctaLDS %7.2f (%.2fx)\n", tf(t_cta), t_cta / t_f16);
    else         printf("ctaLDS      -\n");
    hipFree(dW); hipFree(dX); hipFree(dY); hipFree(dWf);
}

// ---- format comparison at fixed shape: what a weight-format swap actually buys.
// Decode is weight-bandwidth bound, so the figure of merit is GB/s of PACKED
// weight moved and the resulting ms, not TFLOP/s. ----
template<typename FMT>
static void run_fmt(const char* name, int M, int N, int K) {
    if (K % FMT::block_k) { printf("  %-6s  (K not a multiple of block_k)\n", name); return; }
    const int bpr = K / FMT::block_k;
    const size_t wbytes = (size_t)N * bpr * FMT::block_bytes;
    std::mt19937 rng(11);
    auto Wh = rand_w(wbytes, rng);
    std::vector<__half> Xh((size_t)M * K);
    std::uniform_real_distribution<float> uf(-1, 1);
    for (auto& x : Xh) x = __float2half(uf(rng));

    uint8_t* dW; __half* dX; float* dY;
    CK(hipMalloc(&dW, wbytes)); CK(hipMemcpy(dW, Wh.data(), wbytes, hipMemcpyHostToDevice));
    CK(hipMalloc(&dX, sizeof(__half) * Xh.size()));
    CK(hipMemcpy(dX, Xh.data(), sizeof(__half) * Xh.size(), hipMemcpyHostToDevice));
    CK(hipMalloc(&dY, sizeof(float) * (size_t)M * N));

    // decode regime: K-split fragment path (the M<128 winner)
    const int chunk = qgemm_pick_kchunk(M, N, K, FMT::block_k);
    const int splits = (K + chunk - 1) / chunk;
    dim3 gZ(N / 16, (M + 15) / 16, splits);
    const double t = med([&]{
        CK(hipMemsetAsync(dY, 0, sizeof(float) * (size_t)M * N));
        qgemm_ksplit<FMT><<<gZ, 64>>>(dY, dX, dW, M, N, K, chunk); });

    const double bpw = 8.0 * FMT::block_bytes / FMT::block_k;
    printf("  %-6s  %5.2f bit/w  %8.3f MB  %7.4f ms  %7.1f GB/s  %6.2f TFLOP/s\n",
           name, bpw, wbytes / 1e6, t, wbytes / 1e6 / t, 2.0 * M * N * K / 1e12 / (t / 1e3));
    hipFree(dW); hipFree(dX); hipFree(dY);
}

int main(int argc, char** argv) {
    if (argc > 1 && !strcmp(argv[1], "--formats")) {
        printf("== weight-format A/B at GLM-5.2 attention shapes (ksplit, decode regime)\n");
        printf("   figure of merit at decode is ms / GB/s of packed weight, not TFLOP/s\n");
        const int shapes[][2] = {{6144, 16384}, {16384, 2048}, {6144, 2048}};
        const char* nm[] = {"o_proj (6144x16384)", "q_b (16384x2048)", "kv/gate (6144x2048)"};
        for (int s = 0; s < 3; s++) {
            for (int M : {16, 64}) {
                printf("\n%s  M=%d\n", nm[s], M);
                run_fmt<q8_0>("q8_0", M, shapes[s][0], shapes[s][1]);
                run_fmt<q6_K>("q6_K", M, shapes[s][0], shapes[s][1]);
                run_fmt<q5_K>("q5_K", M, shapes[s][0], shapes[s][1]);
                run_fmt<q4_K>("q4_K", M, shapes[s][0], shapes[s][1]);
                run_fmt<q4_0>("q4_0", M, shapes[s][0], shapes[s][1]);
                run_fmt<iq4_nl>("iq4nl", M, shapes[s][0], shapes[s][1]);
                run_fmt<mxfp4>("mxfp4", M, shapes[s][0], shapes[s][1]);
                run_fmt<nvfp4>("nvfp4", M, shapes[s][0], shapes[s][1]);
            }
        }
        return 0;
    }
    printf("== q2_K in-GEMM A/B  (TFLOP/s; (x) = time vs fp16 upper bound on same tile)\n");
    if (argc > 3) { run(atoi(argv[1]), atoi(argv[2]), atoi(argv[3])); return 0; }
    const int shapes[][2] = {{2048, 6144}, {6144, 2048}, {6144, 16384}, {16384, 2048}};
    for (auto& s : shapes) {
        for (int M : {64, 256, 1024, 4096}) run(M, s[0], s[1]);
        printf("\n");
    }
    return 0;
}
