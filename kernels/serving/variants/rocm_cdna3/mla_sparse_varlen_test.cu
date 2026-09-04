#include "hip/hip_runtime.h"
/**
 * @file
 * @brief Harness for the sparse (DSA top-k) MLA decode kernels on the packed
 * cross-layer KV slab (mla_sparse_varlen_kernels.cuh, namespace qc_rocm_mla):
 * mla_sparse_decode_partition + mla_decode_reduce.
 *
 * Each query row attends a varlen span of global page-size-1 KV ids
 * (kv_indices[kv_indptr[t]:kv_indptr[t+1]], -1 padded) fetched from a
 * BLOCK-STRIDED bf16 cache view (rows contiguous inside a block, blocks
 * cache_block_stride elements apart). Checked against an fp64 softmax over the
 * same live rows, with 1 and 4 splits, on a genuinely strided slab.
 *
 * Build: make mla_sparse_varlen_test.out
 * Run:   HIP_VISIBLE_DEVICES=0 ./mla_sparse_varlen_test.out [--bench]
 */
#include "mla_sparse_varlen_kernels.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace qc_rocm_mla;
typedef __hip_bfloat16 bf16;
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
    T* d; CK(hipMalloc(&d, h.size() * sizeof(T)));
    CK(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice));
    return d;
}
template <typename T>
static std::vector<T> d2h(const T* d, size_t n) {
    std::vector<T> h(n); CK(hipMemcpy(h.data(), d, n * sizeof(T), hipMemcpyDeviceToHost)); return h;
}
static void rep(const char* nm, bool ok, const char* detail = "") {
    printf("%-46s %s %s\n", nm, ok ? "PASS" : "FAIL", detail);
    if (!ok) ++g_fail;
}
static double b2d(bf16 x) { return (double)__bfloat162float(x); }

// A strided slab: each block holds `layer_off` rows of padding then
// block_size rows of this layer, row_width entries per block.
struct Slab {
    long num_blocks, block_size, layer_off, row_width;  // row_width in entries
    std::vector<bf16> data;
    long stride() const { return row_width * kEntry; }  // elements between blocks
    size_t row_index(long gidx) const {  // element index of row gidx's first entry
        const long blk = gidx / block_size, off = gidx % block_size;
        return (size_t)blk * stride() + (size_t)(layer_off + off) * kEntry;
    }
};

static void run(int T, int H, int topk, long num_blocks, long block_size, int num_splits,
                double& worst, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> ud(-1.f, 1.f);
    Slab s{num_blocks, block_size, 3, block_size + 5};
    s.data.resize((size_t)num_blocks * s.stride());
    for (auto& v : s.data) v = __float2bfloat16(ud(rng));
    std::vector<bf16> q((size_t)T * H * kEntry);
    for (auto& v : q) v = __float2bfloat16(ud(rng));
    const long total = num_blocks * block_size;
    std::vector<int> indptr(T + 1, 0), indices;
    for (int t = 0; t < T; ++t) {
        const int live = 1 + (int)(rng() % topk);
        for (int i = 0; i < topk; ++i)
            indices.push_back(i < live ? (int)(rng() % total) : -1);
        indptr[t + 1] = indptr[t] + topk;
    }
    const float scale = 0.042f;
    auto *dq = dnew(q); auto *dslab = dnew(s.data); auto *dptr = dnew(indptr); auto *didx = dnew(indices);
    float *dpo, *dpm, *dps; bf16* dout;
    CK(hipMalloc(&dpo, (size_t)T * H * num_splits * kLatent * 4));
    CK(hipMalloc(&dpm, (size_t)T * H * num_splits * 4));
    CK(hipMalloc(&dps, (size_t)T * H * num_splits * 4));
    CK(hipMalloc(&dout, (size_t)T * H * kLatent * 2));
    // the layer's view starts layer_off rows into each block
    const bf16* view = dslab + (size_t)s.layer_off * kEntry;
    mla_sparse_decode_partition<bf16><<<dim3(H, T, num_splits), kWave>>>(
        dq, view, dptr, didx, dpo, dpm, dps, H, (int)block_size, s.stride(), num_splits, scale);
    mla_decode_reduce<bf16><<<dim3(H, T), kWave>>>(dpo, dpm, dps, dout, H, num_splits);
    CK(hipDeviceSynchronize());
    auto out = d2h(dout, (size_t)T * H * kLatent);
    worst = 0;
    for (int t = 0; t < T; ++t) for (int h = 0; h < H; ++h) {
        const bf16* qr = &q[((size_t)t * H + h) * kEntry];
        std::vector<double> sc; std::vector<long> rows;
        double m = -1e300;
        for (int i = indptr[t]; i < indptr[t + 1]; ++i) {
            const int g = indices[i]; if (g < 0) continue;
            const size_t base = s.row_index(g);
            double dot = 0;
            for (int d = 0; d < kEntry; ++d) dot += b2d(qr[d]) * b2d(s.data[base + d]);
            sc.push_back(dot * scale); rows.push_back((long)base); m = std::max(m, dot * scale);
        }
        std::vector<double> acc(kLatent, 0.0); double den = 0;
        for (size_t i = 0; i < sc.size(); ++i) {
            const double p = std::exp(sc[i] - m); den += p;
            for (int d = 0; d < kLatent; ++d) acc[d] += p * b2d(s.data[rows[i] + d]);
        }
        for (int d = 0; d < kLatent; ++d) {
            const double want = acc[d] / den;
            const double got = b2d(out[((size_t)t * H + h) * kLatent + d]);
            worst = std::max(worst, std::fabs(got - want));
        }
    }
    CK(hipFree(dq)); CK(hipFree(dslab)); CK(hipFree(dptr)); CK(hipFree(didx));
    CK(hipFree(dpo)); CK(hipFree(dpm)); CK(hipFree(dps)); CK(hipFree(dout));
}

static void bench() {
    const int T = 32, H = 16, topk = 2048, num_splits = 8;
    const long num_blocks = 4096, block_size = 64;  // 262K rows, GLM decode shape
    Slab s{num_blocks, block_size, 3, block_size + 5};
    bf16* dslab; CK(hipMalloc(&dslab, (size_t)num_blocks * s.stride() * 2));
    CK(hipMemset(dslab, 0x3c, (size_t)num_blocks * s.stride() * 2));
    std::vector<bf16> q((size_t)T * H * kEntry, __float2bfloat16(0.1f));
    std::mt19937 rng(3);
    std::vector<int> indptr(T + 1), indices((size_t)T * topk);
    for (int t = 0; t <= T; ++t) indptr[t] = t * topk;
    for (auto& v : indices) v = (int)(rng() % (num_blocks * block_size));
    auto *dq = dnew(q); auto *dptr = dnew(indptr); auto *didx = dnew(indices);
    float *dpo, *dpm, *dps; bf16* dout;
    CK(hipMalloc(&dpo, (size_t)T * H * num_splits * kLatent * 4));
    CK(hipMalloc(&dpm, (size_t)T * H * num_splits * 4)); CK(hipMalloc(&dps, (size_t)T * H * num_splits * 4));
    CK(hipMalloc(&dout, (size_t)T * H * kLatent * 2));
    const bf16* view = dslab + (size_t)s.layer_off * kEntry;
    auto launch = [&]() {
        mla_sparse_decode_partition<bf16><<<dim3(H, T, num_splits), kWave>>>(
            dq, view, dptr, didx, dpo, dpm, dps, H, (int)block_size, s.stride(), num_splits, 0.042f);
        mla_decode_reduce<bf16><<<dim3(H, T), kWave>>>(dpo, dpm, dps, dout, H, num_splits);
    };
    hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    for (int i = 0; i < 5; ++i) launch();
    CK(hipDeviceSynchronize());
    const int reps = 30;
    CK(hipEventRecord(a));
    for (int i = 0; i < reps; ++i) launch();
    CK(hipEventRecord(b)); CK(hipEventSynchronize(b));
    float ms = 0; CK(hipEventElapsedTime(&ms, a, b)); ms /= reps;
    const double bytes = (double)T * H * topk * kEntry * 2;  // KV rows read
    printf("mla_sparse_decode T=%d H=%d topk=%d splits=%d: %.3f ms, %.0f GB/s KV read\n",
           T, H, topk, num_splits, ms, bytes / ms / 1e6);
}

int main(int argc, char** argv) {
    if (argc > 1 && strcmp(argv[1], "--bench") == 0) { bench(); return 0; }
    double worst;
    run(5, 12, 96, 24, 64, 1, worst, 1);
    { char d[64]; snprintf(d, sizeof(d), "(max abs %.4f)", worst); rep("sparse decode strided, 1 split", worst < 2e-2, d); }
    run(5, 12, 96, 24, 256, 4, worst, 2);
    { char d[64]; snprintf(d, sizeof(d), "(max abs %.4f)", worst); rep("sparse decode strided bs256, 4 splits", worst < 2e-2, d); }
    run(3, 16, 2048, 160, 64, 8, worst, 3);
    { char d[64]; snprintf(d, sizeof(d), "(max abs %.4f)", worst); rep("sparse decode topk 2048, 8 splits", worst < 2e-2, d); }
    printf("%s\n", g_fail ? "FAILED" : "ALL PASS");
    return g_fail ? 1 : 0;
}
