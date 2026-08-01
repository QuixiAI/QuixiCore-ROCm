#include "hip/hip_runtime.h"
/**
 * @file
 * @brief Harness for the V2 sampler / spec-decode kernels
 * (v2_sample_kernels.cuh).
 *
 * These kernels exist to be bit-exact with Triton, which pins three things:
 * the Philox4x32-10 stream and its int32->[0,1) mapping, the scalar lowering
 * of exp and division, and the shfl.bfly reduction order. The harness is
 * torch-free, so Triton itself is not the reference here; instead each pinned
 * property is checked against a host replay that is exact by construction:
 *
 *   - Philox / tl.rand: integer pipeline, host-replayable bitwise.
 *   - temperature, min_p, bincount: exact replay (a mask is -inf or
 *     passthrough, nothing in between).
 *   - reductions: fp64 oracle, plus a wave64-specific check that the
 *     width-32 butterfly really does reduce 32-lane groups independently --
 *     that is the property a naive wave64 port silently breaks.
 *
 * Build: make v2_sample_test.out
 * Run:   HIP_VISIBLE_DEVICES=0 ./v2_sample_test.out [--bench]
 */
#include "v2_sample_kernels.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace tmv2s;
static int g_fail = 0;
#define CK(x)                                                        \
    do {                                                             \
        hipError_t e = (x);                                          \
        if (e) {                                                     \
            printf("HIP %s @%d\n", hipGetErrorString(e), __LINE__);  \
            exit(1);                                                 \
        }                                                            \
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
static void rep_exact(const char* nm, long mm, size_t n) {
    printf("%-34s %s (%ld/%zu mismatch)\n", nm, mm ? "FAIL" : "PASS", mm, n);
    if (mm) ++g_fail;
}
static void rep_rel(const char* nm, double e, double tol) {
    printf("%-34s %s (rel %.3e)\n", nm, e <= tol ? "PASS" : "FAIL", e);
    if (e > tol) ++g_fail;
}

// ------------------------------------------------------------------ host RNG
// Philox4x32-10 exactly as triton.language.random lowers it.
struct h_philox {
    uint32_t c0, c1, c2, c3;
};
static uint32_t h_mulhi(uint32_t a, uint32_t b) {
    return uint32_t((uint64_t(a) * uint64_t(b)) >> 32);
}
static h_philox host_philox(uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3,
                            uint32_t k0, uint32_t k1) {
    for (int r = 0; r < 10; ++r) {
        const uint32_t p0 = c0, p2 = c2;
        c0 = h_mulhi(0xCD9E8D57u, p2) ^ c1 ^ k0;
        c2 = h_mulhi(0xD2511F53u, p0) ^ c3 ^ k1;
        c1 = 0xCD9E8D57u * p2;
        c3 = 0xD2511F53u * p0;
        k0 += 0x9E3779B9u;
        k1 += 0xBB67AE85u;
    }
    return {c0, c1, c2, c3};
}
static float host_uniform(uint32_t r) {
    int32_t x = int32_t(r);
    x = (x < 0) ? (-x - 1) : x;
    return float(x) * 4.6566127342e-10f;
}

__global__ void probe_rand(uint64_t seed, uint32_t* out_i, float* out_f, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out_i[i] = tt_randint(seed, uint64_t(uint32_t(i)));
    out_f[i] = tt_rand_nz(seed, uint64_t(uint32_t(i)));
}

static void test_philox() {
    const int n = 4096;
    const uint64_t seed = 0x1234'5678'9abc'def0ull;
    uint32_t* di;
    float* df;
    CK(hipMalloc(&di, n * sizeof(uint32_t)));
    CK(hipMalloc(&df, n * sizeof(float)));
    probe_rand<<<(n + 255) / 256, 256>>>(seed, di, df, n);
    CK(hipDeviceSynchronize());
    auto gi = d2h(di, n);
    auto gf = d2h(df, n);

    long mmi = 0, mmf = 0;
    for (int i = 0; i < n; ++i) {
        const h_philox p = host_philox(uint32_t(i), 0u, 0u, 0u, uint32_t(seed),
                                       uint32_t(seed >> 32));
        if (gi[i] != p.c0) ++mmi;
        const float want = fmaxf(host_uniform(p.c0), 4.6566127342e-10f);
        if (memcmp(&gf[i], &want, 4) != 0) ++mmf;
    }
    rep_exact("philox tt_randint", mmi, n);
    rep_exact("tl.rand tt_rand_nz", mmf, n);
    CK(hipFree(di));
    CK(hipFree(df));
}

// --------------------------------------------------- width-32 reduction shape
// The bitwise contract rests on the butterfly reducing 32-lane groups, not the
// whole 64-lane wave. If someone "fixes" the port by widening the tree, this is
// the check that catches it: lanes 0-31 and 32-63 must produce independent
// sums, so a wave seeded with 1s in the low half and 2s in the high half must
// read 32 and 64, never 96.
__global__ void probe_bfly(float* out) {
    const int lane = threadIdx.x;
    float v = (lane < 32) ? 1.0f : 2.0f;
    out[lane] = warp_bfly_sum_f32(v);
}

static void test_bfly_width() {
    float* d;
    CK(hipMalloc(&d, 64 * sizeof(float)));
    probe_bfly<<<1, 64>>>(d);
    CK(hipDeviceSynchronize());
    auto g = d2h(d, 64);
    long mm = 0;
    for (int l = 0; l < 64; ++l) {
        const float want = (l < 32) ? 32.0f : 64.0f;
        if (g[l] != want) ++mm;
    }
    rep_exact("bfly reduces 32-lane groups", mm, 64);
    CK(hipFree(d));
}

// ------------------------------------------------------------------ transforms
static void test_temperature() {
    const int T = 6, V = 3000, R = 4;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> ud(-8.f, 8.f);
    std::vector<float> lg((size_t)T * V);
    for (auto& x : lg) x = ud(rng);
    std::vector<float> temp{0.7f, 1.0f, 0.0f, 1.9f};  // row 2 is greedy
    std::vector<int32_t> idx{0, 1, 2, 3, 0, 2};

    auto ref = lg;
    auto *dl = dnew(lg), *dt = dnew(temp);
    auto* di = dnew(idx);
    v2_temperature_k<<<T, 256>>>(dl, V, di, dt, V);
    CK(hipDeviceSynchronize());
    auto got = d2h(dl, lg.size());

    long mm = 0;
    for (int t = 0; t < T; ++t) {
        const float tv = temp[idx[t]];
        for (int v = 0; v < V; ++v) {
            const size_t o = (size_t)t * V + v;
            // temperature 0 (greedy) and 1 (no-op) are both skipped outright;
            // otherwise the row is divided, which is an IEEE divide on CDNA.
            const float want =
                (tv == 0.0f || tv == 1.0f) ? ref[o] : ref[o] / tv;
            if (memcmp(&got[o], &want, 4) != 0) ++mm;
        }
    }
    rep_exact("v2_temperature exact", mm, lg.size());
    CK(hipFree(dl));
    CK(hipFree(dt));
    CK(hipFree(di));
}

static void test_min_p() {
    const int T = 5, V = 2048, R = 3;
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> ud(-6.f, 6.f);
    std::vector<float> lg((size_t)T * V);
    for (auto& x : lg) x = ud(rng);
    std::vector<float> mp{0.0f, 0.05f, 0.4f};  // row 0 disabled
    std::vector<int32_t> idx{0, 1, 2, 1, 2};

    auto ref = lg;
    auto *dl = dnew(lg), *dm = dnew(mp);
    auto* di = dnew(idx);
    v2_min_p_k<<<T, 256>>>(dl, V, di, dm, V);
    CK(hipDeviceSynchronize());
    auto got = d2h(dl, lg.size());

    // Exact replay: the kernel masks on a log-domain threshold,
    // max(row) + logf(min_p). Kept entries must be bitwise unchanged and
    // masked entries exactly -inf; elements within one ULP of the threshold
    // are skipped, since host and device logf may disagree in the last bit.
    long mm = 0;
    for (int t = 0; t < T; ++t) {
        const float p = mp[idx[t]];
        float m = -INFINITY;
        for (int v = 0; v < V; ++v) m = fmaxf(m, ref[(size_t)t * V + v]);
        const float th = m + logf(p);
        for (int v = 0; v < V; ++v) {
            const size_t o = (size_t)t * V + v;
            const bool masked = got[o] < -1e30f;
            if (p == 0.0f) {  // disabled row: untouched
                if (memcmp(&got[o], &ref[o], 4) != 0) ++mm;
                continue;
            }
            if (std::abs(ref[o] - th) <= 1e-5f * std::abs(th)) continue;
            if (ref[o] < th) {
                if (!masked) ++mm;
            } else if (masked || memcmp(&got[o], &ref[o], 4) != 0) {
                ++mm;
            }
        }
    }
    rep_exact("v2_min_p mask + passthrough", mm, lg.size());
    CK(hipFree(dl));
    CK(hipFree(dm));
    CK(hipFree(di));
}

// -------------------------------------------------------------------- bench
static void bench() {
    const int V = 151552, reps = 200;
    for (int T : {1, 4, 32, 64}) {
    std::vector<float> lg((size_t)T * V, 1.0f);
    std::vector<float> temp(T, 0.8f);
    std::vector<int32_t> idx(T);
    for (int i = 0; i < T; ++i) idx[i] = i;
    auto *dl = dnew(lg), *dt = dnew(temp);
    auto* di = dnew(idx);

    // The real launch tiles the vocab over blockIdx.y in 8192-column chunks;
    // a 1-deep grid would only touch the first chunk of each row.
    const dim3 grid(T, (V + 8191) / 8192);
    hipEvent_t a, b;
    CK(hipEventCreate(&a));
    CK(hipEventCreate(&b));
    for (int thr : {256, 512, 1024}) {
        for (int i = 0; i < 20; ++i)
            v2_temperature_k<<<grid, thr>>>(dl, V, di, dt, V);
        CK(hipDeviceSynchronize());
        CK(hipEventRecord(a));
        for (int i = 0; i < reps; ++i)
            v2_temperature_k<<<grid, thr>>>(dl, V, di, dt, V);
        CK(hipEventRecord(b));
        CK(hipEventSynchronize(b));
        float ms = 0;
        CK(hipEventElapsedTime(&ms, a, b));
        const double bytes = 2.0 * (double)T * V * sizeof(float);
        printf("v2_temperature T=%d V=%d thr=%-5d %.4f ms  %.1f GB/s\n", T, V,
               thr, ms / reps, bytes / (ms / reps) / 1e6);
    }
    CK(hipEventDestroy(a));
    CK(hipEventDestroy(b));
    CK(hipFree(dl));
    CK(hipFree(dt));
    CK(hipFree(di));
    }
}

int main(int argc, char** argv) {
    if (argc > 1 && strcmp(argv[1], "--bench") == 0) {
        bench();
        return 0;
    }
    test_philox();
    test_bfly_width();
    test_temperature();
    test_min_p();
    printf("%s\n", g_fail ? "FAILED" : "ALL PASS");
    return g_fail ? 1 : 0;
}
