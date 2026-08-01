#include "hip/hip_runtime.h"
/**
 * @file
 * @brief Harness for the ROCm sparse-MLA indexer kernels
 * (sparse_indexer_kernels.cuh): request-index conversion, sparse seqlen, and
 * the per-token fp8 K-quant + paged scatter.
 *
 * These have no CUDA counterpart -- the ROCm sparse backend is their only
 * consumer -- so the Triton kernel they replace is the reference, and the bar
 * is bitwise equality with it.
 *
 * Scope note: the two index kernels are checked completely here (exact integer
 * output, host-replayable). For the K-quant kernel this harness checks
 * everything the kernel itself decides -- the amax reduction, the scale, the
 * shuffled tile layout, and the negative-slot skip -- using a self-contained
 * e4m3fnuz conversion, because the harness must stay torch-free. Production
 * instantiates the same kernel with c10::Float8_e4m3fnuz; that the two
 * conversions agree is established in SlimServe's differential test against
 * Triton and is deliberately out of scope here.
 *
 * Build: make sparse_indexer_test.out
 * Run:   HIP_VISIBLE_DEVICES=0 ./sparse_indexer_test.out
 */
#include "sparse_indexer_kernels.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
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
static void rep(const char* nm, long mm, size_t n) {
    printf("%-36s %s (%ld/%zu mismatch)\n", nm, mm ? "FAIL" : "PASS", mm, n);
    if (mm) ++g_fail;
}

static std::mt19937 rng(5150);

// --------------------------------------------------------- index conversion
static void test_convert() {
    const int num_tokens = 11, num_reqs = 4, maxblk = 96, blk = 64, topk = 512;
    std::vector<int> req_id(num_tokens);
    for (int i = 0; i < num_tokens; ++i) req_id[i] = i % num_reqs;

    std::vector<int> bt((size_t)num_reqs * maxblk);
    std::uniform_int_distribution<int> bd(0, 900);
    for (auto& b : bt) b = bd(rng);

    std::vector<int> tok((size_t)num_tokens * topk);
    std::uniform_int_distribution<int> td(-1, blk * maxblk - 1);
    for (auto& t : tok) t = td(rng);
    for (size_t i = 0; i < tok.size(); i += 7) tok[i] = -1;   // padding
    tok[3] = blk * maxblk + 100;                              // OOB block id

    std::vector<int> cu(num_tokens + 1, 0);
    std::uniform_int_distribution<int> ld(1, topk);
    for (int i = 0; i < num_tokens; ++i) cu[i + 1] = cu[i] + ld(rng);
    std::vector<int> out(cu[num_tokens], -999);

    auto *dr = dnew(req_id), *dbt = dnew(bt), *dtk = dnew(tok), *dcu = dnew(cu),
         *dout = dnew(out);
    convert_req_index_to_global_index<<<num_tokens, 256>>>(
        dr, dbt, dtk, dcu, dout, maxblk, blk, topk, maxblk, 1, topk, 1);
    CK(hipDeviceSynchronize());
    auto got = d2h(dout, out.size());

    std::vector<int> want(out.size(), -999);
    for (int t = 0; t < num_tokens; ++t) {
        for (int c = 0; c < topk; ++c) {
            if (cu[t] + c >= cu[t + 1]) break;
            const int tk = tok[(size_t)t * topk + c];
            const int bid = tk / blk, off = tk % blk;
            const bool ok = bid < maxblk && bid >= 0;
            want[cu[t] + c] =
                (tk >= 0 && ok) ? bt[(size_t)req_id[t] * maxblk + bid] * blk + off
                                : 0;
        }
    }
    long mm = 0;
    for (size_t i = 0; i < want.size(); ++i) mm += (got[i] != want[i]);
    rep("convert_req_index_to_global", mm, want.size());
    for (int* p : {dr, dbt, dtk, dcu, dout}) CK(hipFree(p));
}

// ------------------------------------------------------------ sparse seqlen
static void test_seqlen() {
    const int nseq = 5, topk = 2048;
    const int qlen[nseq] = {3, 1, 1, 7, 1};
    const int slen[nseq] = {120, 0, 45, 3000, 7};  // one zero-length sequence
    std::vector<int> cu(nseq + 1, 0);
    for (int i = 0; i < nseq; ++i) cu[i + 1] = cu[i] + qlen[i];
    const int ntok = cu[nseq];

    std::vector<int> seq(slen, slen + nseq), out(ntok, 0);
    auto *ds = dnew(seq), *dcu = dnew(cu), *dout = dnew(out);
    generate_sparse_seqlen<<<nseq, 256>>>(ds, dcu, dout, topk);
    CK(hipDeviceSynchronize());
    auto got = d2h(dout, ntok);

    std::vector<int> want(ntok, 0);
    for (int s = 0; s < nseq; ++s) {
        if (slen[s] == 0) continue;  // left at the zero-initialized value
        const int ctx = slen[s] - qlen[s];
        for (int o = 0; o < qlen[s]; ++o)
            want[cu[s] + o] = (ctx + o + 1 < topk) ? (ctx + o + 1) : topk;
    }
    long mm = 0;
    for (int i = 0; i < ntok; ++i) mm += (got[i] != want[i]);
    rep("generate_sparse_seqlen", mm, ntok);
    for (int* p : {ds, dcu, dout}) CK(hipFree(p));
}

// ------------------------------------------------------------------ K-quant
// Standalone e4m3fnuz: 1 sign, 4 exponent (bias 8), 3 mantissa, no inf, 0x80 is
// NaN. Round-to-nearest-even, matching the conversion the kernel is
// instantiated with in production.
struct qc_fp8 {
    unsigned char bits;
    __host__ __device__ qc_fp8() : bits(0) {}
    __host__ __device__ explicit qc_fp8(float f) { bits = from_float(f); }
    __host__ __device__ static unsigned char from_float(float f) {
        unsigned int u;
        __builtin_memcpy(&u, &f, 4);
        const unsigned int sign = (u >> 31) & 1u;
        int exp = int((u >> 23) & 0xFFu) - 127;
        unsigned int man = u & 0x7FFFFFu;
        if (((u >> 23) & 0xFFu) == 0xFFu) return 0x80;  // NaN/Inf -> NaN
        if ((u & 0x7FFFFFFFu) == 0) return 0;           // +-0 -> 0
        int e = exp + 8;                                 // fnuz bias
        unsigned int m;
        if (e <= 0) {                                    // subnormal
            const int shift = 20 - e + 1;
            if (shift > 31) return 0;
            const unsigned int full = man | 0x800000u;
            m = full >> shift;
            const unsigned int rem = full & ((1u << shift) - 1);
            const unsigned int half = 1u << (shift - 1);
            if (rem > half || (rem == half && (m & 1u))) ++m;
            if (m >= 8u) return (unsigned char)((sign << 7) | 8u);
            return (unsigned char)((sign << 7) | m);
        }
        m = man >> 20;
        const unsigned int rem = man & 0xFFFFFu;
        if (rem > 0x80000u || (rem == 0x80000u && (m & 1u))) {
            ++m;
            if (m == 8u) {
                m = 0;
                ++e;
            }
        }
        if (e >= 16) return (unsigned char)((sign << 7) | 0x7Fu);  // saturate
        return (unsigned char)((sign << 7) | (e << 3) | m);
    }
};

static void test_kquant() {
    const int num_tokens = 37, head_dim = 128, block_size = 64, nblocks = 12;
    const int btile = 16, htile = 16;
    const float fp8_max = 224.0f;  // e4m3fnuz

    std::vector<float> k((size_t)num_tokens * head_dim);
    std::uniform_real_distribution<float> kd(-3.f, 3.f);
    for (auto& v : k) v = kd(rng);
    for (int i = 0; i < head_dim; ++i) k[i] = 1e-7f;  // exercises the amax clamp

    // Unique slots: two tokens sharing a slot race, which is not a property of
    // this kernel.
    std::vector<long> all(nblocks * block_size);
    std::iota(all.begin(), all.end(), 0L);
    std::shuffle(all.begin(), all.end(), rng);
    std::vector<long> slots(all.begin(), all.begin() + num_tokens);
    for (size_t i = 1; i < slots.size(); i += 5) slots[i] = -1;  // padding

    std::vector<unsigned char> cache((size_t)nblocks * block_size * head_dim, 0);
    std::vector<float> scales((size_t)nblocks * block_size, 0.f);
    auto* dk = dnew(k);
    auto* dc = dnew(cache);
    auto* dsc = dnew(scales);
    auto* dsl = dnew(slots);

    indexer_k_quant_and_cache<float, qc_fp8><<<num_tokens, 128>>>(
        dk, reinterpret_cast<qc_fp8*>(dc), dsc, dsl, block_size,
        (long)block_size * head_dim, block_size, head_dim, btile, htile,
        fp8_max, 1);
    CK(hipDeviceSynchronize());
    auto gotc = d2h(dc, cache.size());
    auto gots = d2h(dsc, scales.size());

    long mmv = 0, mms = 0;
    std::vector<unsigned char> wantc(cache.size(), 0);
    std::vector<float> wants(scales.size(), 0.f);
    for (int t = 0; t < num_tokens; ++t) {
        if (slots[t] < 0) continue;
        float amax = 0.f;
        for (int i = 0; i < head_dim; ++i)
            amax = std::max(amax, std::fabs(k[(size_t)t * head_dim + i]));
        const float scale = std::max(1e-4f, amax) / fp8_max;
        const long bid = slots[t] / block_size;
        const int boff = (int)(slots[t] % block_size);
        const long base = bid * block_size * head_dim +
                          (long)(boff / btile) * btile * head_dim +
                          (long)(boff % btile) * htile;
        for (int i = 0; i < head_dim; ++i) {
            const int to = (i / htile) * btile * htile + (i % htile);
            wantc[base + to] =
                qc_fp8::from_float(k[(size_t)t * head_dim + i] / scale);
        }
        wants[bid * block_size + boff] = scale;
    }
    for (size_t i = 0; i < wantc.size(); ++i) mmv += (gotc[i] != wantc[i]);
    for (size_t i = 0; i < wants.size(); ++i)
        mms += (memcmp(&gots[i], &wants[i], 4) != 0);
    rep("indexer_k_quant values (shuffled)", mmv, wantc.size());
    rep("indexer_k_quant scales", mms, wants.size());
    CK(hipFree(dk));
    CK(hipFree(dc));
    CK(hipFree(dsc));
    CK(hipFree(dsl));
}


// -------------------------------------------------------------------- bench
static void bench() {
    // K-quant is the hot one: one launch per DSA layer per forward, 78 a step
    // on GLM-5.2. One block per token, one lane per head-dim element, so the
    // only free knob is how many lanes cooperate on a 128-wide row.
    const int head_dim = 128, block_size = 64, nblocks = 4096, reps = 500;
    const float fp8_max = 224.0f;
    for (int num_tokens : {1, 32, 512, 8192}) {
        std::vector<float> k((size_t)num_tokens * head_dim, 0.5f);
        std::vector<long> slots(num_tokens);
        std::iota(slots.begin(), slots.end(), 0L);
        std::vector<unsigned char> cache((size_t)nblocks * block_size * head_dim, 0);
        std::vector<float> scales((size_t)nblocks * block_size, 0.f);
        auto* dk = dnew(k); auto* dc = dnew(cache);
        auto* dsc = dnew(scales); auto* dsl = dnew(slots);
        hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
        for (int thr : {64, 128, 256}) {
            for (int i = 0; i < 20; ++i)
                indexer_k_quant_and_cache<float, qc_fp8><<<num_tokens, thr>>>(
                    dk, reinterpret_cast<qc_fp8*>(dc), dsc, dsl, block_size,
                    (long)block_size * head_dim, block_size, head_dim, 16, 16,
                    fp8_max, 1);
            CK(hipDeviceSynchronize());
            CK(hipEventRecord(a));
            for (int i = 0; i < reps; ++i)
                indexer_k_quant_and_cache<float, qc_fp8><<<num_tokens, thr>>>(
                    dk, reinterpret_cast<qc_fp8*>(dc), dsc, dsl, block_size,
                    (long)block_size * head_dim, block_size, head_dim, 16, 16,
                    fp8_max, 1);
            CK(hipEventRecord(b)); CK(hipEventSynchronize(b));
            float ms = 0; CK(hipEventElapsedTime(&ms, a, b));
            printf("indexer_k_quant tokens=%-5d thr=%-4d %.2f us/launch\n",
                   num_tokens, thr, ms * 1000.0f / reps);
        }
        CK(hipEventDestroy(a)); CK(hipEventDestroy(b));
        CK(hipFree(dk)); CK(hipFree(dc)); CK(hipFree(dsc)); CK(hipFree(dsl));
    }
}

int main(int argc, char** argv) {
    if (argc > 1 && strcmp(argv[1], "--bench") == 0) { bench(); return 0; }
    test_convert();
    test_seqlen();
    test_kquant();
    printf("%s\n", g_fail ? "FAILED" : "ALL PASS");
    return g_fail ? 1 : 0;
}
