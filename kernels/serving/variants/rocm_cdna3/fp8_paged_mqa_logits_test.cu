#include "hip/hip_runtime.h"
/**
 * @file
 * @brief Harness for the PAGED DSA indexer MQA logits kernel
 * (fp8_paged_mqa_logits_kernel.cuh): keys fetched through a block table from
 * a block-strided "packed slab" cache with 64-bit addressing.
 *
 * Why it exists: AITER's paged kernel reads the cache with 32-bit buffer-load
 * offsets and returns garbage past a 2 GiB byte offset (GLM-5.2/MI300X,
 * 2026-09). Contract pinned three ways, no external reference:
 *   1. bitwise equality with the contiguous kernel run on a gathered copy of
 *      the same rows (same MFMA body, same epilogue, same association);
 *   2. an fp64 oracle for general inputs;
 *   3. a case whose blocks sit past the 2 GiB byte offset on the real
 *      6,081,792-byte block stride of the served GLM-5.2 slab.
 *
 * Build: make fp8_paged_mqa_logits_test.out
 * Run:   HIP_VISIBLE_DEVICES=0 ./fp8_paged_mqa_logits_test.out [--bench]
 */
#include "fp8_mqa_logits_kernel.cuh"
#include "fp8_paged_mqa_logits_kernel.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace qcrocm;
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
static void rep(const char* nm, bool ok, const char* detail = "") {
    printf("%-46s %s %s\n", nm, ok ? "PASS" : "FAIL", detail);
    if (!ok) ++g_fail;
}
static unsigned char f2fp8(float f) {  // e4m3fnuz, RNE (see fp8_mqa_logits_test.cu)
    unsigned int u; memcpy(&u, &f, 4);
    const unsigned int sign = (u >> 31) & 1u; const int be = int((u >> 23) & 0xFFu);
    unsigned int man = u & 0x7FFFFFu;
    if (be == 0xFF) return 0x80;
    if ((u & 0x7FFFFFFFu) == 0) return 0;
    int e = be - 127 + 8; unsigned int m;
    if (e <= 0) {
        const int sh = 20 - e + 1; if (sh > 31) return 0;
        const unsigned int full = man | 0x800000u; m = full >> sh;
        const unsigned int rem = full & ((1u << sh) - 1), half = 1u << (sh - 1);
        if (rem > half || (rem == half && (m & 1u))) ++m;
        return (unsigned char)((sign << 7) | (m >= 8u ? 8u : m));
    }
    m = man >> 20; const unsigned int rem = man & 0xFFFFFu;
    if (rem > 0x80000u || (rem == 0x80000u && (m & 1u))) { if (++m == 8u) { m = 0; ++e; } }
    if (e >= 16) return (unsigned char)((sign << 7) | 0x7Fu);
    return (unsigned char)((sign << 7) | (e << 3) | m);
}
static float fp82f(unsigned char b) {
    const int s = b >> 7, e = (b >> 3) & 0xF, m = b & 7;
    if (b == 0x80) return NAN;
    float v = e == 0 ? std::ldexp((float)m, -2 - 8) : std::ldexp(1.0f + (float)m / 8.0f, e - 8);
    return s ? -v : v;
}

static const int D = 128, BS = 64, TILE = 16;
static const long STRIDE = 6081792;  // bytes per packed block on the GLM-5.2 slab

// SHUFFLE tile layout of one block: [BS/16][D/16][16 rows][16 bytes], scales
// (fp32, one per row) after the BS*D value bytes.
static size_t val_off(long blk, int off, int d) {
    return (size_t)blk * STRIDE + (size_t)(off / TILE) * (TILE * D) +
           (size_t)(d / TILE) * (TILE * TILE) + (size_t)(off % TILE) * TILE + (d % TILE);
}
static size_t scale_off(long blk, int off) {
    return (size_t)blk * STRIDE + (size_t)BS * D + (size_t)off * 4;
}

struct Case {
    int H, rows, ctx, first_block;
};

// Runs one case: rows query rows over one request occupying blocks
// [first_block, ...) with `ctx` keys; row r attends the prefix of length
// ctx - (rows-1-r). Returns worst relative error vs fp64 and whether the paged
// output is bitwise equal to the contiguous kernel on the gathered rows.
template <int H>
static void run_case(const Case& c, double& worst, bool& bitwise, long& nan_count) {
    std::mt19937 rng(11 + c.first_block);
    std::uniform_real_distribution<float> ud(-3.f, 3.f);
    const int nblk = (c.ctx + BS - 1) / BS;
    const long nb = c.first_block + nblk;
    std::vector<unsigned char> slab((size_t)nb * STRIDE, 0);
    std::vector<unsigned char> gathered((size_t)c.ctx * D);
    std::vector<float> gscale(c.ctx);
    for (int p = 0; p < c.ctx; ++p) {
        const long blk = c.first_block + p / BS; const int off = p % BS;
        for (int d = 0; d < D; ++d) {
            const unsigned char v = f2fp8(ud(rng));
            gathered[(size_t)p * D + d] = v;
            slab[val_off(blk, off, d)] = v;
        }
        const float s = std::fabs(ud(rng)) * 0.2f + 0.05f;
        gscale[p] = s;
        memcpy(&slab[scale_off(blk, off)], &s, 4);
    }
    std::vector<unsigned char> q((size_t)c.rows * H * D);
    for (auto& v : q) v = f2fp8(ud(rng));
    std::vector<float> w((size_t)c.rows * H);
    for (auto& v : w) v = ud(rng);
    std::vector<int> seq(c.rows), bt((size_t)c.rows * 128, 0);
    for (int r = 0; r < c.rows; ++r) {
        seq[r] = c.ctx - (c.rows - 1 - r);
        for (int b = 0; b < nblk; ++b) bt[(size_t)r * 128 + b] = c.first_block + b;
    }
    const int L = ((c.ctx + 63) / 64) * 64;
    std::vector<float> out((size_t)c.rows * L, NAN), ref((size_t)c.rows * c.ctx, -INFINITY);
    // --- paged kernel on the slab ---
    unsigned char* dslab; CK(hipMalloc(&dslab, slab.size()));
    CK(hipMemcpy(dslab, slab.data(), slab.size(), hipMemcpyHostToDevice));
    auto *dq = dnew(q); auto *dw = dnew(w); auto *dseq = dnew(seq); auto *dbt = dnew(bt);
    auto *dout = dnew(out);
    // 5 splits of 1024 keys: exercises multi-split rows and empty splits.
    fp8_paged_mqa_logits<H, D, BS, 4, TILE, TILE><<<dim3(c.rows, 5), 256>>>(
        dq, dslab, reinterpret_cast<const float*>(dslab + (size_t)BS * D), dw, dseq, dbt,
        dout, (long)H * D, H, L, STRIDE, STRIDE / 4, 128, BS, L, 1024);
    CK(hipDeviceSynchronize());
    out = d2h(dout, out.size());
    // --- contiguous kernel on the gathered rows (bitwise reference) ---
    std::vector<int> cs(c.rows, 0), ce(seq);
    auto *dg = dnew(gathered); auto *dgs = dnew(gscale); auto *dcs = dnew(cs); auto *dce = dnew(ce);
    auto *dref = dnew(ref);
    fp8_mqa_logits<H, D, BS, 4><<<c.rows, 256>>>(dq, dg, dgs, dw, dcs, dce, dref,
                                                (long)H * D, H, c.ctx, c.ctx);
    CK(hipDeviceSynchronize());
    ref = d2h(dref, ref.size());
    worst = 0; bitwise = true; nan_count = 0;
    for (int r = 0; r < c.rows; ++r)
        for (int n = 0; n < seq[r]; ++n) {
            const float got = out[(size_t)r * L + n];
            const float ctg = ref[(size_t)r * c.ctx + n];
            if (!std::isfinite(got)) { ++nan_count; continue; }
            if (memcmp(&got, &ctg, 4) != 0) bitwise = false;
            double want = 0;
            for (int h = 0; h < H; ++h) {
                double dot = 0;
                for (int d = 0; d < D; ++d)
                    dot += (double)fp82f(q[((size_t)r * H + h) * D + d]) *
                           (double)fp82f(gathered[(size_t)n * D + d]);
                want += (double)w[(size_t)r * H + h] * std::max(dot * (double)gscale[n], 0.0);
            }
            worst = std::max(worst, std::fabs(got - want) / std::max(1.0, std::fabs(want)));
        }
    CK(hipFree(dslab)); CK(hipFree(dq)); CK(hipFree(dw)); CK(hipFree(dseq)); CK(hipFree(dbt));
    CK(hipFree(dout)); CK(hipFree(dg)); CK(hipFree(dgs)); CK(hipFree(dcs)); CK(hipFree(dce));
    CK(hipFree(dref));
}

template <int H>
static void test_case(const Case& c, const char* label) {
    double worst; bool bitwise; long nans;
    run_case<H>(c, worst, bitwise, nans);
    char nm[96], det[96];
    snprintf(nm, sizeof(nm), "H=%d %s", H, label);
    snprintf(det, sizeof(det), "(rel %.2e, bitwise=%s, nan=%ld, top block @ %.2f GiB)",
             worst, bitwise ? "yes" : "NO", nans,
             (double)(c.first_block + (c.ctx + BS - 1) / BS) * STRIDE / (1 << 30));
    rep(nm, worst < 1e-4 && bitwise && nans == 0, det);
}

static void bench() {
    // Decode-shaped: 64 rows over a 32K context each (32 requests x 2 tokens).
    const int H = 32, rows = 64, ctx = 32768, first_block = 8;
    const int nblk = ctx / BS; const long nb = first_block + nblk;
    unsigned char* dslab; CK(hipMalloc(&dslab, (size_t)nb * STRIDE));
    CK(hipMemset(dslab, 0x38, (size_t)nb * STRIDE));
    std::vector<unsigned char> q((size_t)rows * H * D, 0x38);
    std::vector<float> w((size_t)rows * H, 0.5f);
    std::vector<int> seq(rows, ctx), bt((size_t)rows * (nblk + 1));
    for (int r = 0; r < rows; ++r) for (int b = 0; b < nblk; ++b) bt[(size_t)r * (nblk + 1) + b] = first_block + b;
    auto *dq = dnew(q); auto *dw = dnew(w); auto *dseq = dnew(seq); auto *dbt = dnew(bt);
    float* dout; CK(hipMalloc(&dout, (size_t)rows * ctx * 4));
    auto launch = [&](int split_len) {
        const int splits = (ctx + split_len - 1) / split_len;
        fp8_paged_mqa_logits<32, D, BS, 4, TILE, TILE><<<dim3(rows, splits), 256>>>(
            dq, dslab, reinterpret_cast<const float*>(dslab + (size_t)BS * D), dw, dseq, dbt,
            dout, (long)H * D, H, ctx, STRIDE, STRIDE / 4, nblk + 1, BS, ctx, split_len);
    };
    hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    // Graph-captured decode must size the grid for max_model_len, so most
    // splits are empty at typical contexts: measure that overhead too.
    auto launch_wide = [&](int split_len, int max_len_grid) {
        const int splits = (max_len_grid + split_len - 1) / split_len;
        fp8_paged_mqa_logits<32, D, BS, 4, TILE, TILE><<<dim3(rows, splits), 256>>>(
            dq, dslab, reinterpret_cast<const float*>(dslab + (size_t)BS * D), dw, dseq, dbt,
            dout, (long)H * D, H, ctx, STRIDE, STRIDE / 4, nblk + 1, BS, ctx, split_len);
    };
    for (int wide : {262144, 1048576}) for (int split_len : {1024, 2048, 4096}) {
        for (int i = 0; i < 3; ++i) launch_wide(split_len, wide);
        CK(hipDeviceSynchronize());
        CK(hipEventRecord(a));
        for (int i = 0; i < 10; ++i) launch_wide(split_len, wide);
        CK(hipEventRecord(b)); CK(hipEventSynchronize(b));
        float ms = 0; CK(hipEventElapsedTime(&ms, a, b)); ms /= 10;
        printf("  grid for max_len %d, split=%d (%d CTAs, %.0f%% empty): %.3f ms\n", wide, split_len,
               rows * ((wide + split_len - 1) / split_len),
               100.0 * (1.0 - (double)ctx / wide), ms);
    }
    for (int split_len : {ctx, 8192, 4096, 2048, 1024}) {
        for (int i = 0; i < 5; ++i) launch(split_len);
        CK(hipDeviceSynchronize());
        const int reps = 30;
        CK(hipEventRecord(a));
        for (int i = 0; i < reps; ++i) launch(split_len);
        CK(hipEventRecord(b)); CK(hipEventSynchronize(b));
        float ms = 0; CK(hipEventElapsedTime(&ms, a, b)); ms /= reps;
        const double flop = 2.0 * rows * ctx * H * D;
        const double bytes = (double)rows * ctx * (D + 4) + (double)rows * ctx * 4;
        printf("fp8_paged_mqa_logits H=32 rows=%d ctx=%d split=%d (%d CTAs): %.3f ms, %.1f TFLOP/s, %.0f GB/s effective\n",
               rows, ctx, split_len, rows * ((ctx + split_len - 1) / split_len), ms,
               flop / ms / 1e9, bytes / ms / 1e6);
    }
}

int main(int argc, char** argv) {
    if (argc > 1 && strcmp(argv[1], "--bench") == 0) { bench(); return 0; }
    test_case<32>({32, 4, 700, 2}, "low blocks, ctx 700");
    test_case<64>({64, 4, 700, 2}, "low blocks, ctx 700");
    test_case<32>({32, 6, 4562, 3}, "low blocks, ctx 4562, 6 rows");
    // Past the 2 GiB byte offset (block 353 = 2.00 GiB at the real stride).
    test_case<32>({32, 4, 1500, 354}, "PAST 2 GiB, ctx 1500");
    test_case<64>({64, 4, 3000, 400}, "PAST 2 GiB, ctx 3000");
    printf("%s\n", g_fail ? "FAILED" : "ALL PASS");
    return g_fail ? 1 : 0;
}
