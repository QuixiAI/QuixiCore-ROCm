#include "hip/hip_runtime.h"
/**
 * @file
 * @brief Harness for the MXFP4 GGUF kernels.
 *
 * The reference is a host decode of the block layout, which is exact: the
 * dequantized value of a nibble is a table entry times a power of two, so both
 * paths must agree BITWISE, not within a tolerance. The dot products are
 * checked against an fp64 accumulation instead, since those do add rounding.
 *
 * The element-position checks matter more than they look. The low nibbles
 * supply values [0,16) and the high nibbles [16,32); swapping that still gives
 * a plausible distribution and a nearly identical norm, so a norm-only test
 * would pass a wrong kernel.
 *
 * Build: make mxfp4_gguf_test.out
 * Run:   HIP_VISIBLE_DEVICES=0 ./mxfp4_gguf_test.out [--bench]
 */
#include "mxfp4_gguf_kernels.cuh"

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
template <typename T>
static std::vector<T> d2h(const T* d, size_t n) {
    std::vector<T> h(n);
    CK(hipMemcpy(h.data(), d, n * sizeof(T), hipMemcpyDeviceToHost));
    return h;
}
static void rep(const char* nm, bool ok, const char* det = "") {
    printf("%-38s %s %s\n", nm, ok ? "PASS" : "FAIL", det);
    if (!ok) ++g_fail;
}

static const int8_t KV[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
static float host_e8m0(uint8_t x) {
    const uint32_t bits = (x == 0) ? 0x00400000u : ((uint32_t)x << 23);
    float r;
    memcpy(&r, &bits, 4);
    return r;
}
// Host mirror of the block layout: low nibbles -> [0,16), high -> [16,32).
static void host_block(const block_mxfp4& b, float* out) {
    const float d = host_e8m0(b.e) * 0.5f;
    for (int j = 0; j < 16; ++j) {
        out[j] = d * (float)KV[b.qs[j] & 0xF];
        out[j + 16] = d * (float)KV[b.qs[j] >> 4];
    }
}

static std::vector<block_mxfp4> rand_blocks(size_t n, std::mt19937& rng) {
    std::uniform_int_distribution<int> qd(0, 255);
    std::uniform_int_distribution<int> ed(118, 134);  // sane exponents
    std::vector<block_mxfp4> v(n);
    for (auto& b : v) {
        b.e = (uint8_t)ed(rng);
        for (int i = 0; i < 16; ++i) b.qs[i] = (uint8_t)qd(rng);
    }
    return v;
}

// ------------------------------------------------------------------ dequant
static void test_dequant() {
    std::mt19937 rng(11);
    const size_t NB = 4096;
    auto blocks = rand_blocks(NB, rng);
    std::vector<float> y(NB * QK_MXFP4, -12345.0f);
    auto* db = dnew(blocks);
    auto* dy = dnew(y);
    dequant_mxfp4<float><<<(NB + 255) / 256, 256>>>(db, dy, (long)NB);
    CK(hipDeviceSynchronize());
    y = d2h(dy, y.size());

    long mm = 0;
    std::vector<float> want(QK_MXFP4);
    for (size_t b = 0; b < NB; ++b) {
        host_block(blocks[b], want.data());
        for (int j = 0; j < QK_MXFP4; ++j)
            if (memcmp(&y[b * QK_MXFP4 + j], &want[j], 4) != 0) ++mm;
    }
    char det[64];
    snprintf(det, sizeof(det), "(%ld/%zu)", mm, NB * QK_MXFP4);
    rep("dequant_mxfp4 bitwise", mm == 0, det);

    // Explicitly pin the half ordering: a swapped kernel would fail here even
    // if the multiset of values were right.
    block_mxfp4 probe{};
    probe.e = 127;                       // scale 1.0
    for (int i = 0; i < 16; ++i) probe.qs[i] = (uint8_t)((i & 0xF) | (0x1 << 4));
    std::vector<block_mxfp4> one{probe};
    std::vector<float> yo(QK_MXFP4, 0.f);
    auto* d1 = dnew(one);
    auto* dyo = dnew(yo);
    dequant_mxfp4<float><<<1, 1>>>(d1, dyo, 1L);
    CK(hipDeviceSynchronize());
    yo = d2h(dyo, yo.size());
    bool order_ok = true;
    for (int j = 0; j < 16; ++j) {
        order_ok &= (yo[j] == 0.5f * (float)KV[j]);        // low nibble -> first
        order_ok &= (yo[j + 16] == 0.5f * (float)KV[1]);   // high nibble -> second
    }
    rep("dequant nibble-half ordering", order_ok);
    CK(hipFree(db)); CK(hipFree(dy)); CK(hipFree(d1)); CK(hipFree(dyo));
}

// --------------------------------------------------------------------- gemv
static void run_gemv(int nrows, int ncols, bool moe, int experts, int topk) {
    std::mt19937 rng(23);
    const int nblocks = ncols / QK_MXFP4;
    const int nmat = moe ? experts : 1;
    auto blocks = rand_blocks((size_t)nmat * nrows * nblocks, rng);

    const int tokens = moe ? topk : 1;
    std::vector<int8_t> xq((size_t)tokens * ncols);
    std::uniform_int_distribution<int> xd(-127, 127);
    for (auto& v : xq) v = (int8_t)xd(rng);
    std::vector<float> xs((size_t)tokens * nblocks);
    std::uniform_real_distribution<float> sd(0.01f, 0.1f);
    for (auto& v : xs) v = sd(rng);

    std::vector<int> eid(tokens);
    for (int t = 0; t < tokens; ++t) eid[t] = (t == 1 && moe) ? -1 : t % experts;

    const int out_rows = moe ? tokens * nrows : nrows;
    std::vector<float> y(out_rows, -1.f);
    auto* dw = dnew(blocks);
    auto* dxq = dnew(xq);
    auto* dxs = dnew(xs);
    auto* dy = dnew(y);
    int* de = nullptr;
    if (moe) de = dnew(eid);

    mxfp4_gemv_q8_1<float><<<out_rows, 64>>>(
        dw, dxq, dxs, dy, ncols, out_rows, de, moe ? nrows : 0,
        (long)nrows * nblocks);
    CK(hipDeviceSynchronize());
    y = d2h(dy, y.size());

    double worst = 0;
    long padbad = 0;
    std::vector<float> wv(QK_MXFP4);
    for (int r = 0; r < out_rows; ++r) {
        const int tok = moe ? r / nrows : 0;
        const int orow = moe ? r % nrows : r;
        const int e = moe ? eid[tok] : 0;
        if (moe && e < 0) {
            if (y[r] != 0.0f) ++padbad;
            continue;
        }
        double want = 0, mag = 0;
        const block_mxfp4* w =
            blocks.data() + (size_t)e * nrows * nblocks + (size_t)orow * nblocks;
        for (int b = 0; b < nblocks; ++b) {
            host_block(w[b], wv.data());
            for (int j = 0; j < QK_MXFP4; ++j) {
                const double t = (double)wv[j] *
                                 (double)xq[(size_t)tok * ncols + b * QK_MXFP4 + j] *
                                 (double)xs[(size_t)tok * nblocks + b];
                want += t;
                mag += std::fabs(t);
            }
        }
        // Judge against the ACCUMULATED magnitude, not the (possibly
        // cancelling) result. Dividing by |want| makes the metric explode on
        // rows that happen to cancel, which says nothing about the kernel; the
        // fp32 bound for a length-N dot scales with sum|terms|.
        worst = std::max(worst, std::fabs(y[r] - want) / std::max(1.0, mag));
    }
    char nm[64], det[64];
    snprintf(nm, sizeof(nm), "%s ncols=%d nrows=%d", moe ? "mxfp4 MoE gemv" : "mxfp4 gemv",
             ncols, nrows);
    snprintf(det, sizeof(det), "(err/|terms| %.2e%s)", worst,
             moe ? (padbad ? ", PAD BAD" : ", pad ok") : "");
    // sqrt(512) * 2^-24 ~ 1.3e-6; allow a small factor over that.
    rep(nm, worst < 5e-6 && padbad == 0, det);
    CK(hipFree(dw)); CK(hipFree(dxq)); CK(hipFree(dxs)); CK(hipFree(dy));
    if (de) CK(hipFree(de));
}

// -------------------------------------------------------------------- bench
static void bench() {
    const int nrows = 4096, ncols = 4096, nblocks = ncols / QK_MXFP4, reps = 200;
    std::mt19937 rng(3);
    auto blocks = rand_blocks((size_t)nrows * nblocks, rng);
    std::vector<int8_t> xq(ncols, 1);
    std::vector<float> xs(nblocks, 0.05f), y(nrows, 0.f);
    auto* dw = dnew(blocks);
    auto* dxq = dnew(xq);
    auto* dxs = dnew(xs);
    auto* dy = dnew(y);
    hipEvent_t a, b;
    CK(hipEventCreate(&a));
    CK(hipEventCreate(&b));
    for (int i = 0; i < 20; ++i)
        mxfp4_gemv_q8_1<float><<<nrows, 64>>>(dw, dxq, dxs, dy, ncols, nrows,
                                              nullptr, 0, 0);
    CK(hipDeviceSynchronize());
    CK(hipEventRecord(a));
    for (int i = 0; i < reps; ++i)
        mxfp4_gemv_q8_1<float><<<nrows, 64>>>(dw, dxq, dxs, dy, ncols, nrows,
                                              nullptr, 0, 0);
    CK(hipEventRecord(b));
    CK(hipEventSynchronize(b));
    float ms = 0;
    CK(hipEventElapsedTime(&ms, a, b));
    const double bytes = (double)blocks.size() * sizeof(block_mxfp4);
    printf("mxfp4_gemv_q8_1 %dx%d: %.4f ms, %.1f GB/s (weight-bound)\n", nrows,
           ncols, ms / reps, bytes / (ms / reps) / 1e6);
}

int main(int argc, char** argv) {
    if (argc > 1 && strcmp(argv[1], "--bench") == 0) {
        bench();
        return 0;
    }
    test_dequant();
    run_gemv(64, 256, false, 1, 1);
    run_gemv(512, 2048, false, 1, 1);
    run_gemv(128, 512, false, 1, 1);   // control: same shape, no MoE indexing
    run_gemv(128, 512, true, 8, 6);
    printf("%s\n", g_fail ? "FAILED" : "ALL PASS");
    return g_fail ? 1 : 0;
}
