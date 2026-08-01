#include "hip/hip_runtime.h"
/**
 * @file
 * @brief Harness for the DSA indexer MQA logits kernel
 * (fp8_mqa_logits_kernel.cuh) and its MFMA primitive (mfma_fp8_dot.cuh).
 *
 * This kernel replaces a Triton kernel in SlimServe and its contract is BITWISE
 * equality with it, which constrains not just the arithmetic but the order the
 * 32 or 64 per-head terms are summed in. Triton is not linkable here (the
 * harness is torch-free), so the contract is pinned three ways that need no
 * external reference:
 *
 *   1. exact arithmetic -- with inputs whose products and partial sums are all
 *      representable, association cannot matter, so a bitwise match against a
 *      host sum proves every index, mask, weight and scale is right;
 *   2. fp64 oracle for general inputs, which catches magnitude errors;
 *   3. the REDUCTION TREE itself, probed directly (see test_tree). This is the
 *      one that locks in the bitwise contract: it asserts the specific
 *      association Triton uses, so a refactor that changes the summation order
 *      fails here rather than silently diverging in production.
 *
 * Build: make fp8_mqa_logits_test.out
 * Run:   HIP_VISIBLE_DEVICES=0 ./fp8_mqa_logits_test.out [--bench]
 */
#include "fp8_mqa_logits_kernel.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
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
    printf("%-40s %s %s\n", nm, ok ? "PASS" : "FAIL", detail);
    if (!ok) ++g_fail;
}

// e4m3fnuz encode, round-to-nearest-even (see sparse_indexer_test.cu).
static unsigned char f2fp8(float f) {
    unsigned int u;
    memcpy(&u, &f, 4);
    const unsigned int sign = (u >> 31) & 1u;
    const int be = int((u >> 23) & 0xFFu);
    unsigned int man = u & 0x7FFFFFu;
    if (be == 0xFF) return 0x80;
    if ((u & 0x7FFFFFFFu) == 0) return 0;
    int e = be - 127 + 8;
    unsigned int m;
    if (e <= 0) {
        const int sh = 20 - e + 1;
        if (sh > 31) return 0;
        const unsigned int full = man | 0x800000u;
        m = full >> sh;
        const unsigned int rem = full & ((1u << sh) - 1), half = 1u << (sh - 1);
        if (rem > half || (rem == half && (m & 1u))) ++m;
        return (unsigned char)((sign << 7) | (m >= 8u ? 8u : m));
    }
    m = man >> 20;
    const unsigned int rem = man & 0xFFFFFu;
    if (rem > 0x80000u || (rem == 0x80000u && (m & 1u))) {
        if (++m == 8u) { m = 0; ++e; }
    }
    if (e >= 16) return (unsigned char)((sign << 7) | 0x7Fu);
    return (unsigned char)((sign << 7) | (e << 3) | m);
}
static float fp82f(unsigned char b) {
    const int s = b >> 7, e = (b >> 3) & 0xF, m = b & 7;
    if (b == 0x80) return NAN;
    float v = e == 0 ? std::ldexp((float)m, -2 - 8)
                     : std::ldexp(1.0f + (float)m / 8.0f, e - 8);
    return s ? -v : v;
}

static const int D = 128, BK = 64;

template <int H>
static void launch(const std::vector<unsigned char>& q,
                   const std::vector<unsigned char>& kv,
                   const std::vector<float>& ks, const std::vector<float>& w,
                   const std::vector<int>& cs, const std::vector<int>& ce,
                   std::vector<float>& out, int M, int N) {
    auto* dq = dnew(q);
    auto* dk = dnew(kv);
    auto *dks = dnew(ks), *dw = dnew(w);
    auto *dcs = dnew(cs), *dce = dnew(ce);
    auto* dout = dnew(out);
    fp8_mqa_logits<H, D, BK, 4><<<M, 256>>>(dq, dk, dks, dw, dcs, dce, dout,
                                            (long)H * D, H, N, N);
    CK(hipDeviceSynchronize());
    out = d2h(dout, out.size());
    CK(hipFree(dq)); CK(hipFree(dk)); CK(hipFree(dks));
    CK(hipFree(dw)); CK(hipFree(dcs)); CK(hipFree(dce)); CK(hipFree(dout));
}

// ------------------------------------------------------------------- exact
template <int H>
static void test_exact() {
    const int M = 8, N = 128;
    std::mt19937 rng(21);
    std::uniform_int_distribution<int> sd(0, 2);
    std::vector<unsigned char> q((size_t)M * H * D), kv((size_t)N * D);
    // Small integers: every product and partial sum is exact in fp32, so the
    // summation order cannot change the answer.
    for (auto& v : q) v = f2fp8((float)sd(rng));
    for (auto& v : kv) v = f2fp8((float)sd(rng));
    std::vector<float> ks(N, 1.0f), w((size_t)M * H);
    for (auto& v : w) v = (float)sd(rng);
    std::vector<int> cs(M, 0), ce(M, N);
    std::vector<float> out((size_t)M * N, -INFINITY);
    launch<H>(q, kv, ks, w, cs, ce, out, M, N);

    long mm = 0;
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float want = 0.0f;
            for (int h = 0; h < H; ++h) {
                float dot = 0.0f;
                for (int d = 0; d < D; ++d)
                    dot += fp82f(q[((size_t)m * H + h) * D + d]) *
                           fp82f(kv[(size_t)n * D + d]);
                want += w[(size_t)m * H + h] * fmaxf(dot, 0.0f);
            }
            if (memcmp(&out[(size_t)m * N + n], &want, 4) != 0) ++mm;
        }
    char nm[64];
    snprintf(nm, sizeof(nm), "H=%d exact-arithmetic bitwise", H);
    rep(nm, mm == 0);
}

// -------------------------------------------------------------- fp64 oracle
template <int H>
static void test_oracle() {
    const int M = 16, N = 200;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> ud(-3.f, 3.f);
    std::vector<unsigned char> q((size_t)M * H * D), kv((size_t)N * D);
    for (auto& v : q) v = f2fp8(ud(rng));
    for (auto& v : kv) v = f2fp8(ud(rng));
    std::vector<float> ks(N), w((size_t)M * H);
    for (auto& v : ks) v = std::fabs(ud(rng)) * 0.2f + 0.05f;
    for (auto& v : w) v = ud(rng);
    std::vector<int> cs(M), ce(M);
    for (int m = 0; m < M; ++m) { cs[m] = m * 3 % 40; ce[m] = N - (m % 7); }
    std::vector<float> out((size_t)M * N, -INFINITY);
    launch<H>(q, kv, ks, w, cs, ce, out, M, N);

    double worst = 0;
    long masked_bad = 0;
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            const float got = out[(size_t)m * N + n];
            if (n < cs[m] || n >= ce[m]) {          // outside the window
                if (std::isfinite(got)) ++masked_bad;
                continue;
            }
            double want = 0;
            for (int h = 0; h < H; ++h) {
                double dot = 0;
                for (int d = 0; d < D; ++d)
                    dot += (double)fp82f(q[((size_t)m * H + h) * D + d]) *
                           (double)fp82f(kv[(size_t)n * D + d]);
                want += (double)w[(size_t)m * H + h] *
                        std::max(dot * (double)ks[n], 0.0);
            }
            const double den = std::max(1.0, std::fabs(want));
            worst = std::max(worst, std::fabs(got - want) / den);
        }
    char nm[64], det[64];
    snprintf(nm, sizeof(nm), "H=%d fp64 oracle + window mask", H);
    snprintf(det, sizeof(det), "(rel %.2e, %ld unmasked)", worst, masked_bad);
    // fp32 accumulation over D=128 then H heads against an fp64
    // oracle: 1e-5 relative is the expected floor, not a defect.
    rep(nm, worst < 1e-4 && masked_bad == 0, det);
}

// ------------------------------------------------------------ reduction tree
// Pins the association, which is what the bitwise contract rests on.
//
// Force every dot to 1 (a single 1.0 in dim 0 of both q and k) and set the
// scales to 1, so logits[n] is just the sum of the per-head weights. Then put
// 1.0 at one head and 2^-24 at two others: 1 + 2^-24 rounds back to 1, while
// 1 + 2^-23 is exact, so the result reveals whether those two heads were summed
// with each other BEFORE meeting the large one. Sweeping pairs recovers the
// tree.
//
// The expected tree, measured against Triton on gfx942:
//   head = bitrev4(lane % 16) + (H/2)*pair + 16*wave
// so two heads merge early exactly when their indices agree on the bits that
// are consumed late. Equivalent formulation used below: build the merge order
// from that mapping and require the observed merges to match.
template <int H>
static void test_tree() {
    const int M = H * 2, N = 16;
    auto bitrev4 = [](int x) {
        return ((x & 1) << 3) | ((x & 2) << 1) | ((x & 4) >> 1) | ((x & 8) >> 3);
    };
    // Coordinates of a head, ordered by the stage that consumes them:
    //   c[0]      the intra-lane pair          (earliest)
    //   c[1..4]   lane bits 0..3, where bitrev4(lane) == h % 16
    //   c[5]      the wave                     (latest, H=64 only)
    auto coords = [&](int h, int* c) {
        c[0] = h / (H / 2);
        int lo = 0;
        for (int l = 0; l < 16; ++l)
            if (bitrev4(l) == (h % 16)) lo = l;
        for (int b = 0; b < 4; ++b) c[1 + b] = (lo >> b) & 1;
        c[5] = (h % (H / 2)) / 16;
    };
    // A binary reduction consumes coordinates in stage order, so two heads meet
    // at the LAST stage where they still differ.
    auto meet = [&](int a, int b) {
        int ca[6], cb[6];
        coords(a, ca);
        coords(b, cb);
        int m = -1;
        for (int k = 0; k < 6; ++k)
            if (ca[k] != cb[k]) m = k;
        return m;
    };

    std::vector<unsigned char> q((size_t)M * H * D, 0), kv((size_t)N * D, 0);
    for (int m = 0; m < M; ++m)
        for (int h = 0; h < H; ++h) q[((size_t)m * H + h) * D] = f2fp8(1.0f);
    for (int n = 0; n < N; ++n) kv[(size_t)n * D] = f2fp8(1.0f);
    std::vector<float> ks(N, 1.0f);
    std::vector<int> cs(M, 0), ce(M, N);

    const float BIG = 1.0f, EPS = std::ldexp(1.0f, -24);
    long checked = 0, bad = 0;
    for (int i = 1; i < H; ++i) {
        std::vector<float> w((size_t)M * H, 0.0f);
        std::vector<int> js;
        for (int j = i + 1; j < H && (int)js.size() < M; ++j) {
            const int r = js.size();
            w[(size_t)r * H + 0] = BIG;
            w[(size_t)r * H + i] = EPS;
            w[(size_t)r * H + j] = EPS;
            js.push_back(j);
        }
        if (js.empty()) continue;
        std::vector<float> out((size_t)M * N, -INFINITY);
        launch<H>(q, kv, ks, w, cs, ce, out, M, N);
        for (size_t r = 0; r < js.size(); ++r) {
            const bool merged = out[r * N] != BIG;
            const bool want =
                meet(i, js[r]) < std::min(meet(i, 0), meet(js[r], 0));
            ++checked;
            if (merged != want) ++bad;
        }
    }
    char nm[64], det[64];
    snprintf(nm, sizeof(nm), "H=%d reduction tree matches spec", H);
    snprintf(det, sizeof(det), "(%ld/%ld pairs)", checked - bad, checked);
    rep(nm, bad == 0, det);
}

// -------------------------------------------------------------------- bench
static void bench() {
    const int H = 32, M = 512, N = 2048, reps = 50;
    std::vector<unsigned char> q((size_t)M * H * D, 0x38), kv((size_t)N * D, 0x38);
    std::vector<float> ks(N, 1.0f), w((size_t)M * H, 0.5f);
    std::vector<int> cs(M, 0), ce(M, N);
    auto* dq = dnew(q); auto* dk = dnew(kv);
    auto *dks = dnew(ks), *dw = dnew(w);
    auto *dcs = dnew(cs), *dce = dnew(ce);
    std::vector<float> out((size_t)M * N, -INFINITY);
    auto* dout = dnew(out);
    hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    for (int i = 0; i < 5; ++i)
        fp8_mqa_logits<32, D, BK, 4><<<M, 256>>>(dq, dk, dks, dw, dcs, dce,
                                                 dout, (long)H * D, H, N, N);
    CK(hipDeviceSynchronize());
    CK(hipEventRecord(a));
    for (int i = 0; i < reps; ++i)
        fp8_mqa_logits<32, D, BK, 4><<<M, 256>>>(dq, dk, dks, dw, dcs, dce,
                                                 dout, (long)H * D, H, N, N);
    CK(hipEventRecord(b)); CK(hipEventSynchronize(b));
    float ms = 0; CK(hipEventElapsedTime(&ms, a, b));
    const double flop = 2.0 * M * N * H * D;
    printf("fp8_mqa_logits H=32 M=%d N=%d: %.3f ms, %.1f TFLOP/s\n", M, N,
           ms / reps, flop / (ms / reps) / 1e9);
}

int main(int argc, char** argv) {
    if (argc > 1 && strcmp(argv[1], "--bench") == 0) { bench(); return 0; }
    test_exact<32>();
    test_exact<64>();
    test_oracle<32>();
    test_oracle<64>();
    test_tree<32>();
    test_tree<64>();
    printf("%s\n", g_fail ? "FAILED" : "ALL PASS");
    return g_fail ? 1 : 0;
}
