#include "hip/hip_runtime.h"
/**
 * @file
 * @brief Harness for the MXFP4 tiled GEMM.
 *
 * Checked against an fp64 host accumulation of the same quantized operands.
 * The activations are quantized on the host and handed to the kernel already
 * in q8_1 form, so this measures the tile kernel alone -- feeding it raw floats
 * and letting a device quantizer intervene would fold that kernel's rounding
 * into the result and mask a real error here.
 *
 * The judgement is |err| over the accumulated magnitude, not over the result:
 * a row whose terms cancel makes a relative-to-result metric explode while
 * saying nothing about the kernel.
 *
 * Build: make mxfp4_mmq_test.out
 * Run:   HIP_VISIBLE_DEVICES=0 ./mxfp4_mmq_test.out [--bench]
 */
#include "mxfp4_mmq_kernels.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace qcmxfp4;
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

static const int8_t KV[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
static float host_e8m0(uint8_t x) {
    const uint32_t bits = (x == 0) ? 0x00400000u : ((uint32_t)x << 23);
    float r;
    memcpy(&r, &bits, 4);
    return r;
}
static void host_block(const block_mxfp4& b, float* out) {
    const float d = host_e8m0(b.e) * 0.5f;
    for (int j = 0; j < 16; ++j) {
        out[j] = d * (float)KV[b.qs[j] & 0xF];
        out[j + 16] = d * (float)KV[b.qs[j] >> 4];
    }
}

// MMQ_X is fixed at 8 to match the tile the serving path uses for MoE, where
// each tile column is a routed row rather than a whole request.
static constexpr int MX = 8, MY = 128, NW = 8;

static void run(int nrows, int ncols, int ncols_y, const char* what) {
    std::mt19937 rng(37);
    const int nblocks = ncols / QK_MXFP4;
    std::uniform_int_distribution<int> qd(0, 255), ed(118, 134);
    std::vector<block_mxfp4> w((size_t)nrows * nblocks);
    for (auto& b : w) {
        b.e = (uint8_t)ed(rng);
        for (int i = 0; i < 16; ++i) b.qs[i] = (uint8_t)qd(rng);
    }

    // Activations, quantized on the host exactly as q8_1 specifies: one fp16
    // scale per 32 values, quants formed with the fp32 scale. Dividing by the
    // rounded scale instead leaves a systematic 2^-11 bias.
    const int padded = (ncols + 511) / 512 * 512;
    const int ny = padded / QK8_1;
    std::vector<block_q8_1> yb((size_t)ncols_y * ny);
    std::vector<float> xf((size_t)ncols_y * padded, 0.0f);
    std::uniform_real_distribution<float> xd(-1.f, 1.f);
    for (int c = 0; c < ncols_y; ++c)
        for (int k = 0; k < ncols; ++k) xf[(size_t)c * padded + k] = xd(rng);
    for (int c = 0; c < ncols_y; ++c) {
        for (int b = 0; b < ny; ++b) {
            const float* src = &xf[(size_t)c * padded + b * QK8_1];
            float amax = 0.f, sum = 0.f;
            for (int j = 0; j < QK8_1; ++j) {
                amax = fmaxf(amax, fabsf(src[j]));
                sum += src[j];
            }
            const float d = amax / 127.0f;
            block_q8_1& blk = yb[(size_t)c * ny + b];
            for (int j = 0; j < QK8_1; ++j)
                blk.qs[j] = amax == 0.f ? 0 : (int8_t)roundf(src[j] / d);
            blk.ds = __floats2half2_rn(d, sum);
        }
    }

    std::vector<float> out((size_t)ncols_y * nrows, -1.f);
    auto* dw = dnew(w);
    auto* dy = dnew(yb);
    auto* dd = dnew(out);

    const dim3 grid((nrows + MY - 1) / MY, (ncols_y + MX - 1) / MX, 1);
    const dim3 blk(MMQ_WARP, NW, 1);
    if (nrows % MY == 0)
        mxfp4_mmq_q8_1<MX, MY, NW, false><<<grid, blk>>>(
            dw, dy, dd, ncols, nrows, ncols_y, padded, nrows);
    else
        mxfp4_mmq_q8_1<MX, MY, NW, true><<<grid, blk>>>(
            dw, dy, dd, ncols, nrows, ncols_y, padded, nrows);
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(out.data(), dd, out.size() * sizeof(float),
                 hipMemcpyDeviceToHost));

    double worst = 0;
    std::vector<float> wv(QK_MXFP4);
    for (int c = 0; c < ncols_y; ++c) {
        for (int r = 0; r < nrows; ++r) {
            double want = 0, mag = 0;
            for (int b = 0; b < nblocks; ++b) {
                host_block(w[(size_t)r * nblocks + b], wv.data());
                const block_q8_1& q = yb[(size_t)c * ny + b];
                const double d8 = (double)__low2float(q.ds);
                for (int j = 0; j < QK_MXFP4; ++j) {
                    const double t = (double)wv[j] * (double)q.qs[j] * d8;
                    want += t;
                    mag += fabs(t);
                }
            }
            worst = std::max(
                worst, fabs(out[(size_t)c * nrows + r] - want) / std::max(1.0, mag));
        }
    }
    char nm[80], det[64];
    snprintf(nm, sizeof(nm), "mxfp4 MMQ %dx%d x %d cols %s", nrows, ncols,
             ncols_y, what);
    snprintf(det, sizeof(det), "(err/|terms| %.2e)", worst);
    const bool ok = worst < 5e-6;
    printf("%-56s %s %s\n", nm, ok ? "PASS" : "FAIL", det);
    if (!ok) ++g_fail;

    CK(hipFree(dw)); CK(hipFree(dy)); CK(hipFree(dd));
}

static void bench() {
    const int nrows = 2048, ncols = 4096, ncols_y = 512, reps = 50;
    const int nblocks = ncols / QK_MXFP4, padded = ncols, ny = padded / QK8_1;
    std::vector<block_mxfp4> w((size_t)nrows * nblocks);
    for (auto& b : w) { b.e = 127; memset(b.qs, 0x21, 16); }
    std::vector<block_q8_1> yb((size_t)ncols_y * ny);
    for (auto& b : yb) { memset(b.qs, 1, QK8_1); b.ds = __floats2half2_rn(0.05f, 0.f); }
    std::vector<float> out((size_t)ncols_y * nrows, 0.f);
    auto* dw = dnew(w);
    auto* dy = dnew(yb);
    auto* dd = dnew(out);

    const dim3 grid((nrows + MY - 1) / MY, (ncols_y + MX - 1) / MX, 1);
    const dim3 blk(MMQ_WARP, NW, 1);
    hipEvent_t a, b;
    CK(hipEventCreate(&a));
    CK(hipEventCreate(&b));
    for (int i = 0; i < 10; ++i)
        mxfp4_mmq_q8_1<MX, MY, NW, false><<<grid, blk>>>(
            dw, dy, dd, ncols, nrows, ncols_y, padded, nrows);
    CK(hipDeviceSynchronize());
    CK(hipEventRecord(a));
    for (int i = 0; i < reps; ++i)
        mxfp4_mmq_q8_1<MX, MY, NW, false><<<grid, blk>>>(
            dw, dy, dd, ncols, nrows, ncols_y, padded, nrows);
    CK(hipEventRecord(b));
    CK(hipEventSynchronize(b));
    float ms = 0;
    CK(hipEventElapsedTime(&ms, a, b));
    const double flop = 2.0 * nrows * ncols * ncols_y;
    printf("mxfp4_mmq_q8_1 %dx%dx%d: %.4f ms, %.1f TFLOP/s\n", nrows, ncols,
           ncols_y, ms / reps, flop / (ms / reps) / 1e9);
}

int main(int argc, char** argv) {
    if (argc > 1 && strcmp(argv[1], "--bench") == 0) {
        bench();
        return 0;
    }
    run(128, 512, 8, "one tile");
    run(256, 512, 8, "two row tiles");
    run(128, 2048, 32, "deep K, many cols");
    run(130, 512, 8, "rows not a multiple of the tile");
    run(128, 512, 5, "cols not a multiple of the tile");
    printf("%s\n", g_fail ? "FAILED" : "ALL PASS");
    return g_fail ? 1 : 0;
}
