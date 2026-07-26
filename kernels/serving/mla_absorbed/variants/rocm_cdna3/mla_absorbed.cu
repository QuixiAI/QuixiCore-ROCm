#include "hip/hip_runtime.h"
/**
 * @file
 * @brief CDNA3 (gfx942) quantized MLA decode against an absorbed kv_b, dense
 *        and sparse (DSA) variants.
 *
 * Semantic source: ../QuixiCore-CPU/kernels/attention/mla_absorb_ref.cpp
 * (quantized_mla_decode_absorbed, quantized_mla_decode_absorbed_sparse).
 *
 * This is the GLM-5.2 / DeepSeek-V3.2 decode shape: the KV cache stores only the
 * low-rank latent plus a RoPE tail, and kv_b is "absorbed" so the query is
 * projected into latent space once, scored against the raw cache, and projected
 * back out at the end. Nothing ever materializes full K/V.
 *
 *   query_latent[l] = sum_d q[d] * W[head*(nope+value) + d, l]      d < nope_dim
 *   score           = <query_latent, latent[row]> + <q[nope..], rope[row]>
 *   mixed_latent    = softmax-weighted sum of latent[row]
 *   out[v]          = sum_l W[head*(nope+value) + nope + v, l] * mixed_latent[l]
 *
 * W is GGUF row-packed: row_bytes = (latent_dim/block_k)*block_bytes, and
 * element (row,col) lives at row*row_bytes + (col/block_k)*block_bytes, index
 * col%block_k. Same layout the qgemm kernels use.
 *
 * ## The softmax is PER-KEY and asymmetric -- not the tiled form
 *
 * The sibling mxfp8 paged attention tiles at 16. This one updates on every key,
 * and the two branches are NOT symmetric:
 *
 *   if (score > maximum):  old = exp(maximum - score)
 *                          denominator = denominator*old + 1
 *                          mixed = mixed*old + latent          <- weight is 1
 *                          maximum = score
 *   else:                  w = exp(score - maximum)
 *                          denominator += w
 *                          mixed += latent*w
 *
 * The new-maximum branch adds the latent with an implicit weight of exactly 1
 * rather than exp(0), and rescales in float while the running denominator is
 * double. Collapsing the two branches into one "general" update changes the
 * result. Reproduced as written.
 *
 * ## Sparse variant
 *
 * `token_indices` is [batch, max_topk] holding logical positions, resolved
 * through the same page table as the dense path; `topk_lengths[request]` gives
 * the count. Dense is the same loop with position == selected.
 *
 * Build: make      Test: make test
 */
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <limits>

namespace mlaq {

// ---- GGUF q8_0: {half d; int8 qs[32];} = 34 bytes, 32 weights ----
struct q8_0 {
    static constexpr int block_k = 32, block_bytes = 34;
    __host__ __device__ static float dequant(const uint8_t *base, int col) {
        // typed access, not memcpy: HIP shadows memcpy with a device-only
        // overload. Blocks are 34 bytes so the half is 2-byte aligned.
        return __half2float(*reinterpret_cast<const __half *>(base)) *
               (float)((const int8_t *)(base + 2))[col];
    }
};

template <typename FMT>
__host__ __device__ __forceinline__ float packed_at(const uint8_t *bytes, long long row,
                                                    long long column, long long row_bytes) {
    const long long block = column / FMT::block_k;
    return FMT::dequant(bytes + (size_t)row * row_bytes + (size_t)block * FMT::block_bytes,
                        (int)(column % FMT::block_k));
}

__device__ __forceinline__ float wave_sum(float v) {
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) v += __shfl_xor(v, off);
    return v;
}

// One block per (request, head). Shared holds query_latent and mixed_latent.
template <typename FMT, bool Sparse, int BLOCK>
__global__ void k_mla_absorbed(const uint8_t *__restrict__ packed_kv_b,
                               const float *__restrict__ q,
                               const float *__restrict__ latent_cache,
                               const float *__restrict__ rope_cache,
                               const int *__restrict__ block_table,
                               const int *__restrict__ lengths_or_indices,
                               const int *__restrict__ topk_lengths,
                               float *__restrict__ out,
                               int batch, int heads, int latent_dim, int nope_dim,
                               int rope_dim, int value_dim, int page_size,
                               int max_blocks, int max_topk, float scale) {
    // One 64-lane wavefront per block: wave_sum's xor reduction leaves the full
    // sum in every lane, so the score needs no shared staging and every lane
    // runs the identical online-softmax update in lockstep.
    extern __shared__ float smem[];
    float *query_latent = smem;                 // [latent_dim]
    float *mixed_latent = smem + latent_dim;    // [latent_dim]

    const int item = blockIdx.x;
    if (item >= batch * heads) return;
    const int tid = threadIdx.x;
    const int request = item / heads, head = item % heads;
    const int count = Sparse ? topk_lengths[request] : lengths_or_indices[request];

    const int query_dim = nope_dim + rope_dim;
    const long long row_base = (long long)head * (nope_dim + value_dim);
    const long long row_bytes = (long long)(latent_dim / FMT::block_k) * FMT::block_bytes;
    const float score_scale = scale == 0.0f ? rsqrtf((float)query_dim) : scale;
    const float *query = q + (long long)item * query_dim;

    // ---- absorb: query_latent[l] = sum_d q[d] * W[row_base + d, l] ----
    for (int l = tid; l < latent_dim; l += BLOCK) {
        float acc = 0.0f;
        for (int d = 0; d < nope_dim; ++d)
            acc += query[d] * packed_at<FMT>(packed_kv_b, row_base + d, l, row_bytes);
        query_latent[l] = acc;
        mixed_latent[l] = 0.0f;
    }
    __syncthreads();

    double maximum = -INFINITY;
    double denominator = 0.0;

    for (int selected = 0; selected < count; ++selected) {
        const int position = Sparse ? lengths_or_indices[(long long)request * max_topk + selected]
                                    : selected;
        const int block = block_table[(long long)request * max_blocks + position / page_size];
        const long long cache_row = (long long)block * page_size + position % page_size;
        const float *latent = latent_cache + cache_row * latent_dim;

        // score = <query_latent, latent> + <q[nope..], rope>
        // The reference accumulates in double; per-lane partials do too, and the
        // cross-lane reduction is float, so this is close but not bit-exact.
        double partial = 0.0;
        for (int l = tid; l < latent_dim; l += BLOCK)
            partial += (double)query_latent[l] * (double)latent[l];
        if (rope_dim != 0) {
            const float *rope = rope_cache + cache_row * rope_dim;
            for (int r = tid; r < rope_dim; r += BLOCK)
                partial += (double)query[nope_dim + r] * (double)rope[r];
        }
        const double score = (double)wave_sum((float)partial) * (double)score_scale;

        // asymmetric online update -- see header
        if (score > maximum) {
            const double old_weight = exp(maximum - score);
            denominator = denominator * old_weight + 1.0;
            for (int l = tid; l < latent_dim; l += BLOCK)
                mixed_latent[l] = (float)(mixed_latent[l] * old_weight + latent[l]);
            maximum = score;
        } else {
            const double weight = exp(score - maximum);
            denominator += weight;
            for (int l = tid; l < latent_dim; l += BLOCK)
                mixed_latent[l] += (float)(latent[l] * weight);
        }
        __syncthreads();
    }

    if (denominator > 0.0) {
        const float inverse = (float)(1.0 / denominator);
        for (int l = tid; l < latent_dim; l += BLOCK) mixed_latent[l] *= inverse;
    }
    __syncthreads();

    // ---- unabsorb: out[v] = sum_l W[row_base + nope + v, l] * mixed_latent[l] ----
    float *destination = out + (long long)item * value_dim;
    for (int v = tid; v < value_dim; v += BLOCK) {
        float sum = 0.0f;
        const long long weight_row = row_base + nope_dim + v;
        for (int l = 0; l < latent_dim; ++l)
            sum += packed_at<FMT>(packed_kv_b, weight_row, l, row_bytes) * mixed_latent[l];
        destination[v] = sum;
    }
}

}  // namespace mlaq

// ===========================================================================
// self-checking harness (fp64 host replica; not bit-exact by design)
// ===========================================================================
using namespace mlaq;

#define CK(x) do { hipError_t e=(x); if(e){printf("HIP %s @%d\n",hipGetErrorString(e),__LINE__);exit(1);} } while(0)
static int g_fail = 0;
static void report(const char *n, bool ok, const char *d = "") {
    printf("%-46s %s %s\n", n, ok ? "PASS" : "FAIL", d);
    if (!ok) ++g_fail;
}

// GLM-5.2 geometry: latent 512, rope 64, and a reduced head count for the test.
int main() {
    const int batch = 3, heads = 4, latent_dim = 512, nope_dim = 64;
    const int rope_dim = 64, value_dim = 64, page_size = 16;
    const int cache_blocks = 32, max_blocks = 6, max_topk = 24;
    const int query_dim = nope_dim + rope_dim;
    const int weight_rows = heads * (nope_dim + value_dim);
    const long long row_bytes = (long long)(latent_dim / q8_0::block_k) * q8_0::block_bytes;

    std::mt19937 rng(31337);
    std::uniform_real_distribution<float> uf(-1.0f, 1.0f);

    std::vector<uint8_t> W((size_t)weight_rows * row_bytes);
    for (size_t r = 0; r < (size_t)weight_rows; ++r)
        for (int b = 0; b < latent_dim / q8_0::block_k; ++b) {
            uint8_t *blk = W.data() + r * row_bytes + (size_t)b * q8_0::block_bytes;
            *reinterpret_cast<__half *>(blk) = __float2half(0.01f + 0.03f * std::fabs(uf(rng)));
            for (int i = 0; i < q8_0::block_k; ++i)
                ((int8_t *)(blk + 2))[i] = (int8_t)(int)(uf(rng) * 127.0f);
        }
    std::vector<float> q((size_t)batch * heads * query_dim);
    for (auto &v : q) v = uf(rng);
    std::vector<float> latent((size_t)cache_blocks * page_size * latent_dim);
    for (auto &v : latent) v = uf(rng) * 0.1f;
    std::vector<float> rope((size_t)cache_blocks * page_size * rope_dim);
    for (auto &v : rope) v = uf(rng) * 0.1f;
    std::vector<int> bt((size_t)batch * max_blocks);
    for (size_t i = 0; i < bt.size(); ++i) bt[i] = (int)(i * 5 % cache_blocks);
    std::vector<int> ctx(batch), topk_len(batch), tok((size_t)batch * max_topk);
    for (int b = 0; b < batch; ++b) {
        ctx[b] = 20 + b * 9;
        topk_len[b] = 7 + b * 5;
        for (int k = 0; k < max_topk; ++k) tok[(size_t)b * max_topk + k] = (k * 7 + b) % (max_blocks * page_size);
    }

    uint8_t *dW; float *dq, *dlat, *drope, *dout;
    int *dbt, *dctx, *dtok, *dtl;
    CK(hipMalloc(&dW, W.size()));            CK(hipMemcpy(dW, W.data(), W.size(), hipMemcpyHostToDevice));
    CK(hipMalloc(&dq, q.size()*4));          CK(hipMemcpy(dq, q.data(), q.size()*4, hipMemcpyHostToDevice));
    CK(hipMalloc(&dlat, latent.size()*4));   CK(hipMemcpy(dlat, latent.data(), latent.size()*4, hipMemcpyHostToDevice));
    CK(hipMalloc(&drope, rope.size()*4));    CK(hipMemcpy(drope, rope.data(), rope.size()*4, hipMemcpyHostToDevice));
    CK(hipMalloc(&dbt, bt.size()*4));        CK(hipMemcpy(dbt, bt.data(), bt.size()*4, hipMemcpyHostToDevice));
    CK(hipMalloc(&dctx, batch*4));           CK(hipMemcpy(dctx, ctx.data(), batch*4, hipMemcpyHostToDevice));
    CK(hipMalloc(&dtok, tok.size()*4));      CK(hipMemcpy(dtok, tok.data(), tok.size()*4, hipMemcpyHostToDevice));
    CK(hipMalloc(&dtl, batch*4));            CK(hipMemcpy(dtl, topk_len.data(), batch*4, hipMemcpyHostToDevice));
    CK(hipMalloc(&dout, (size_t)batch*heads*value_dim*4));

    const size_t shmem = (size_t)2 * latent_dim * sizeof(float);

    // host replica, mirroring the reference exactly
    auto replica = [&](bool sparse, std::vector<float> &ref) {
        ref.assign((size_t)batch * heads * value_dim, 0.0f);
        const float score_scale = 1.0f / std::sqrt((float)query_dim);
        for (int item = 0; item < batch * heads; ++item) {
            const int request = item / heads, head = item % heads;
            const int count = sparse ? topk_len[request] : ctx[request];
            const long long row_base = (long long)head * (nope_dim + value_dim);
            std::vector<float> ql(latent_dim, 0.0f), mixed(latent_dim, 0.0f);
            for (int l = 0; l < latent_dim; ++l) {
                float acc = 0.0f;
                for (int d = 0; d < nope_dim; ++d)
                    acc += q[(size_t)item*query_dim+d] * packed_at<q8_0>(W.data(), row_base+d, l, row_bytes);
                ql[l] = acc;
            }
            double maximum = -std::numeric_limits<double>::infinity(), denom = 0.0;
            for (int sel = 0; sel < count; ++sel) {
                const int pos = sparse ? tok[(size_t)request*max_topk+sel] : sel;
                const int blk = bt[(size_t)request*max_blocks + pos/page_size];
                const long long crow = (long long)blk*page_size + pos%page_size;
                const float *lat = latent.data() + crow*latent_dim;
                double score = 0.0;
                for (int l = 0; l < latent_dim; ++l) score += (double)ql[l]*lat[l];
                for (int r = 0; r < rope_dim; ++r)
                    score += (double)q[(size_t)item*query_dim+nope_dim+r] * rope[crow*rope_dim+r];
                score *= score_scale;
                if (score > maximum) {
                    const double ow = std::exp(maximum - score);
                    denom = denom*ow + 1.0;
                    for (int l = 0; l < latent_dim; ++l) mixed[l] = (float)(mixed[l]*ow + lat[l]);
                    maximum = score;
                } else {
                    const double w = std::exp(score - maximum);
                    denom += w;
                    for (int l = 0; l < latent_dim; ++l) mixed[l] += (float)(lat[l]*w);
                }
            }
            if (denom > 0.0) { const float inv = (float)(1.0/denom);
                for (auto &v : mixed) v *= inv; }
            for (int v = 0; v < value_dim; ++v) {
                float sum = 0.0f;
                for (int l = 0; l < latent_dim; ++l)
                    sum += packed_at<q8_0>(W.data(), row_base+nope_dim+v, l, row_bytes) * mixed[l];
                ref[(size_t)item*value_dim+v] = sum;
            }
        }
    };

    auto check = [&](const char *name, bool sparse) {
        CK(hipMemset(dout, 0, (size_t)batch*heads*value_dim*4));
        if (sparse)
            k_mla_absorbed<q8_0, true, 64><<<batch*heads, 64, shmem>>>(
                dW, dq, dlat, drope, dbt, dtok, dtl, dout, batch, heads, latent_dim,
                nope_dim, rope_dim, value_dim, page_size, max_blocks, max_topk, 0.0f);
        else
            k_mla_absorbed<q8_0, false, 64><<<batch*heads, 64, shmem>>>(
                dW, dq, dlat, drope, dbt, dctx, nullptr, dout, batch, heads, latent_dim,
                nope_dim, rope_dim, value_dim, page_size, max_blocks, max_topk, 0.0f);
        CK(hipDeviceSynchronize());
        if (hipGetLastError() != hipSuccess) { printf("KERNEL ERR (%s)\n", name); exit(1); }
        std::vector<float> got((size_t)batch*heads*value_dim);
        CK(hipMemcpy(got.data(), dout, got.size()*4, hipMemcpyDeviceToHost));
        std::vector<float> ref; replica(sparse, ref);
        // Non-finite must fail loudly: std::max(worst, nan) silently returns worst.
        size_t nonfinite = 0;
        for (float v : got) if (!std::isfinite(v)) ++nonfinite;
        double worst = 0.0;
        for (size_t i = 0; i < got.size(); ++i)
            worst = std::max(worst, std::fabs((double)got[i]-ref[i]) / std::max(1e-3, std::fabs((double)ref[i])));
        char d[96]; snprintf(d, sizeof d, "(worst rel %.3e, %zu non-finite)", worst, nonfinite);
        report(name, nonfinite == 0 && worst < 2e-3, d);
    };

    check("quantized_mla_decode_absorbed (q8_0)", false);
    check("quantized_mla_decode_absorbed_sparse (q8_0)", true);

    printf("\n%s (%d failures)\n", g_fail ? "FAILED" : "ALL PASS", g_fail);
    return g_fail ? 1 : 0;
}
