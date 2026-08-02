#include "hip/hip_runtime.h"
/**
 * @file
 * @brief Harness for the IQ2_XXS tiled GEMM.
 *
 * Dequant is checked BITWISE against a host decode: a weight is a grid byte
 * times a sign times a scale, all exactly representable, so there is nothing to
 * tolerate. The GEMM is checked against an fp64 accumulation of the same
 * operands, judged as |err| / sum|terms| rather than |err| / |result| -- a row
 * whose terms cancel makes the latter explode while saying nothing about the
 * kernel.
 *
 * Activations are quantized on the host and handed over already in q8_1 form,
 * so this measures the tile kernel alone. Two q8_1 details have to be mirrored
 * exactly or the reference drifts far enough to hide real errors: the quants
 * are formed with the fp32 scale while the stored and later-applied scale is
 * fp16, and roundf breaks ties away from zero.
 *
 * Build: make iq2xxs_mmq_test.out
 * Run:   HIP_VISIBLE_DEVICES=0 ./iq2xxs_mmq_test.out [--bench]
 */
#include "iq2xxs_mmq_kernels.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace qciq2;
static int g_fail = 0;
#define CK(x)                                                       \
    do {                                                            \
        hipError_t e = (x);                                         \
        if (e) {                                                    \
            printf("HIP %s @%d\n", hipGetErrorString(e), __LINE__); \
            exit(1);                                                \
        }                                                           \
    } while (0)

template <typename T>
static T* dnew(const std::vector<T>& h) {
    T* d;
    CK(hipMalloc(&d, h.size() * sizeof(T)));
    CK(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice));
    return d;
}
static void rep(const char* nm, bool ok, const char* det = "") {
    printf("%-46s %s %s\n", nm, ok ? "PASS" : "FAIL", det);
    if (!ok) ++g_fail;
}

// Host mirrors of the device tables, so the reference is independent.
static uint64_t h_grid[256];
static uint8_t h_ksigns[128];

static void fetch_tables() {
    CK(hipMemcpyFromSymbol(h_grid, HIP_SYMBOL(tmq::iq2xxs_grid), sizeof(h_grid)));
    CK(hipMemcpyFromSymbol(h_ksigns, HIP_SYMBOL(tmq::ksigns_iq2xs),
                           sizeof(h_ksigns)));
}

/// Decode one superblock to 256 exact weights.
static void host_block(const block_iq2_xxs& b, float* out) {
    const float d = __half2float(b.d);
    for (int g = 0; g < 8; ++g) {
        uint32_t aux_g, aux_s;
        memcpy(&aux_g, (const uint8_t*)b.qs + 8 * g + 0, 4);
        memcpy(&aux_s, (const uint8_t*)b.qs + 8 * g + 4, 4);
        const uint8_t* aux8 = (const uint8_t*)&aux_g;
        const float sc = d * (0.5f + (aux_s >> 28)) * 0.25f;
        for (int e = 0; e < 4; ++e) {
            const uint8_t* grid = (const uint8_t*)&h_grid[aux8[e]];
            const uint8_t signs = h_ksigns[(aux_s >> (7 * e)) & 127];
            for (int j = 0; j < 8; ++j)
                out[32 * g + 8 * e + j] =
                    sc * (float)grid[j] * ((signs >> j) & 1 ? -1.0f : 1.0f);
        }
    }
}

static std::vector<block_iq2_xxs> rand_blocks(size_t n, std::mt19937& rng) {
    std::uniform_int_distribution<int> qd(0, 65535);
    std::vector<block_iq2_xxs> v(n);
    for (auto& b : v) {
        b.d = __float2half(0.08f);  // a random fp16 would give NaN and hide all
        for (int i = 0; i < 32; ++i) b.qs[i] = (uint16_t)qd(rng);
    }
    return v;
}

static void test_dequant() {
    std::mt19937 rng(7);
    const size_t NB = 512;
    auto blocks = rand_blocks(NB, rng);
    std::vector<float> y(NB * QK_K_IQ2, -12345.0f);
    auto* db = dnew(blocks);
    auto* dy = dnew(y);
    dequant_iq2_xxs<float><<<(NB + 63) / 64, 64>>>(db, dy, (long)NB);
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(y.data(), dy, y.size() * sizeof(float), hipMemcpyDeviceToHost));

    long mm = 0;
    std::vector<float> want(QK_K_IQ2);
    for (size_t b = 0; b < NB; ++b) {
        host_block(blocks[b], want.data());
        for (int j = 0; j < QK_K_IQ2; ++j)
            if (memcmp(&y[b * QK_K_IQ2 + j], &want[j], 4) != 0) ++mm;
    }
    char det[64];
    snprintf(det, sizeof(det), "(%ld/%zu)", mm, NB * QK_K_IQ2);
    rep("dequant_iq2_xxs bitwise", mm == 0, det);
    CK(hipFree(db));
    CK(hipFree(dy));
}

static constexpr int MX = 8, MY = 32, NW = 8;

static void run_gemm(int nrows, int ncols, int ncols_y, const char* what) {
    std::mt19937 rng(19);
    const int nsb = ncols / QK_K_IQ2;
    auto w = rand_blocks((size_t)nrows * nsb, rng);

    const int padded = (ncols + 511) / 512 * 512;
    const int nq8 = padded / QK8_1_IQ2;
    std::vector<block_q8_1> yb((size_t)ncols_y * nq8);
    std::vector<float> xf((size_t)ncols_y * padded, 0.0f);
    std::uniform_real_distribution<float> xd(-1.f, 1.f);
    for (int c = 0; c < ncols_y; ++c)
        for (int k = 0; k < ncols; ++k) xf[(size_t)c * padded + k] = xd(rng);
    for (int c = 0; c < ncols_y; ++c) {
        for (int b = 0; b < nq8; ++b) {
            const float* src = &xf[(size_t)c * padded + b * QK8_1_IQ2];
            float amax = 0.f, sum = 0.f;
            for (int j = 0; j < QK8_1_IQ2; ++j) {
                amax = fmaxf(amax, fabsf(src[j]));
                sum += src[j];
            }
            const float d = amax / 127.0f;
            block_q8_1& blk = yb[(size_t)c * nq8 + b];
            for (int j = 0; j < QK8_1_IQ2; ++j)
                blk.qs[j] = amax == 0.f ? 0 : (int8_t)roundf(src[j] / d);
            blk.ds = __floats2half2_rn(d, sum);
        }
    }

    std::vector<float> out((size_t)ncols_y * nrows, -1.f);
    auto* dw = dnew(w);
    auto* dy = dnew(yb);
    auto* dd = dnew(out);

    const dim3 grid((nrows + MY - 1) / MY, (ncols_y + MX - 1) / MX, 1);
    const dim3 blk(IQ2_WARP, NW, 1);
    if (nrows % MY == 0)
        iq2xxs_mmq_q8_1<MX, MY, NW, false>
            <<<grid, blk>>>(dw, dy, dd, ncols, nrows, ncols_y, padded, nrows);
    else
        iq2xxs_mmq_q8_1<MX, MY, NW, true>
            <<<grid, blk>>>(dw, dy, dd, ncols, nrows, ncols_y, padded, nrows);
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(out.data(), dd, out.size() * sizeof(float),
                 hipMemcpyDeviceToHost));

    double worst = 0;
    std::vector<float> wv(QK_K_IQ2);
    for (int c = 0; c < ncols_y; ++c) {
        for (int r = 0; r < nrows; ++r) {
            double want = 0, mag = 0;
            for (int b = 0; b < nsb; ++b) {
                host_block(w[(size_t)r * nsb + b], wv.data());
                for (int j = 0; j < QK_K_IQ2; ++j) {
                    const int e = b * QK_K_IQ2 + j;
                    const block_q8_1& q = yb[(size_t)c * nq8 + e / QK8_1_IQ2];
                    const double t = (double)wv[j] *
                                     (double)q.qs[e % QK8_1_IQ2] *
                                     (double)__low2float(q.ds);
                    want += t;
                    mag += fabs(t);
                }
            }
            worst = std::max(worst, fabs(out[(size_t)c * nrows + r] - want) /
                                        std::max(1.0, mag));
        }
    }
    char nm[96], det[64];
    snprintf(nm, sizeof(nm), "iq2_xxs mmq %dx%d x %d cols %s", nrows, ncols,
             ncols_y, what);
    snprintf(det, sizeof(det), "(err/|terms| %.2e)", worst);
    rep(nm, worst < 5e-6, det);

    CK(hipFree(dw));
    CK(hipFree(dy));
    CK(hipFree(dd));
}

static void bench() {
    const int nrows = 2048, ncols = 4096, ncols_y = 512, reps = 50;
    std::mt19937 rng(5);
    auto w = rand_blocks((size_t)nrows * (ncols / QK_K_IQ2), rng);
    const int nq8 = ncols / QK8_1_IQ2;
    std::vector<block_q8_1> yb((size_t)ncols_y * nq8);
    for (auto& b : yb) {
        memset(b.qs, 1, QK8_1_IQ2);
        b.ds = __floats2half2_rn(0.05f, 0.f);
    }
    std::vector<float> out((size_t)ncols_y * nrows, 0.f);
    auto* dw = dnew(w);
    auto* dy = dnew(yb);
    auto* dd = dnew(out);

    const dim3 grid((nrows + MY - 1) / MY, (ncols_y + MX - 1) / MX, 1);
    const dim3 blk(IQ2_WARP, NW, 1);
    hipEvent_t a, b;
    CK(hipEventCreate(&a));
    CK(hipEventCreate(&b));
    for (int i = 0; i < 10; ++i)
        iq2xxs_mmq_q8_1<MX, MY, NW, false>
            <<<grid, blk>>>(dw, dy, dd, ncols, nrows, ncols_y, ncols, nrows);
    CK(hipDeviceSynchronize());
    CK(hipEventRecord(a));
    for (int i = 0; i < reps; ++i)
        iq2xxs_mmq_q8_1<MX, MY, NW, false>
            <<<grid, blk>>>(dw, dy, dd, ncols, nrows, ncols_y, ncols, nrows);
    CK(hipEventRecord(b));
    CK(hipEventSynchronize(b));
    float ms = 0;
    CK(hipEventElapsedTime(&ms, a, b));
    const double flop = 2.0 * nrows * ncols * ncols_y;
    printf("iq2xxs_mmq_q8_1 %dx%dx%d: %.4f ms, %.1f TFLOP/s\n", nrows, ncols,
           ncols_y, ms / reps, flop / (ms / reps) / 1e9);
}

int main(int argc, char** argv) {
    fetch_tables();
    if (argc > 1 && strcmp(argv[1], "--bench") == 0) {
        bench();
        return 0;
    }
    test_dequant();
    run_gemm(32, 512, 8, "one tile");
    run_gemm(64, 512, 8, "two row tiles");
    run_gemm(32, 1024, 16, "deep K, two col tiles");
    run_gemm(40, 512, 8, "rows not a multiple of the tile");
    run_gemm(32, 512, 5, "cols not a multiple of the tile");
    printf("%s\n", g_fail ? "FAILED" : "ALL PASS");
    return g_fail ? 1 : 0;
}
