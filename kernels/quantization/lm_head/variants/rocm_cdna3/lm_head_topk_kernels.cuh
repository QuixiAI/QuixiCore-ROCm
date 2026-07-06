#include "hip/hip_runtime.h"
#pragma once
// Minimal scalar lm_head top-k/top-p kernels extracted from
// QuixiCore-CUDA/kernels/quant/tm_kernels.cuh (lines ~545-725), WITHOUT the
// mma-based qgemm kernels (not needed by lm_head_topkp). CDNA3: no tensor core.
#include "quant_formats.cuh"
#include "quant_formats_tables.cuh"   // dequant8
#include "tm_rng.cuh"
#include "tm_warp.cuh"                 // masked_topk / masked_topk_local
#include <hip/hip_fp16.h>
#ifndef LMH_NEG_INF
#define LMH_NEG_INF (-3.4028234663852886e38f)
#endif
namespace tmq {

__device__ __forceinline__ void warp_argmax(float& best, int& bi) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_xor(best, off);
        const int   oi = __shfl_xor(bi, off);
        if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
    }
}

#define LMH_MAX_K 64
#define LMH_MAX_PER_LANE 64

// USE_LSE=1 also emits the per-tile tempered logsumexp (top-p path).
template<int USE_LSE>
__global__ void lm_head_topk_partials(const half* h, const half* W, float* part_val,
                                      int* part_id, const float* bias, int V, int K,
                                      int TILE_V, int num_vtiles, int topk, int use_bias,
                                      float invtemp, float* part_lse) {
    const int vtile = blockIdx.x, t = blockIdx.y, lane = threadIdx.x;
    const int v0 = vtile * TILE_V, v1 = min(v0 + TILE_V, V);
    const half* hrow = h + size_t(t) * K;
    float mine_val[LMH_MAX_PER_LANE];
    int   mine_id[LMH_MAX_PER_LANE];
    bool  used[LMH_MAX_PER_LANE];
    int   nmine = 0;
    float lmax = LMH_NEG_INF, lsum = 0.0f;
    for (int v = v0 + lane; v < v1; v += 32) {
        const half* wrow = W + size_t(v) * K;
        float acc = 0.0f;
        for (int j = 0; j < K / 2; j++) {
            half2 w2 = reinterpret_cast<const half2*>(wrow)[j];
            half2 h2 = reinterpret_cast<const half2*>(hrow)[j];
            acc += __half2float(w2.x) * __half2float(h2.x) + __half2float(w2.y) * __half2float(h2.y);
        }
        float ls = acc;
        if (use_bias) ls += bias[v];
        if (USE_LSE) {
            const float tl = ls * invtemp;
            if (tl > lmax) { lsum = lsum * expf(lmax - tl) + 1.0f; lmax = tl; }
            else           { lsum += expf(tl - lmax); }
        }
        mine_val[nmine] = ls;
        mine_id[nmine] = v;
        ++nmine;
    }
    if (USE_LSE) {
        const float gmax = tms::warp_max_f(lmax);
        const float gsum = tms::warp_sum_f((lmax == LMH_NEG_INF) ? 0.0f : lsum * expf(lmax - gmax));
        if (lane == 0)
            part_lse[size_t(t) * num_vtiles + vtile] = (gsum > 0.0f) ? (gmax + logf(gsum)) : LMH_NEG_INF;
    }
    const size_t pbase = (size_t(t) * num_vtiles + vtile) * topk;
    tms::masked_topk_local(mine_val, mine_id, used, nmine, topk, LMH_NEG_INF,
        [&](int kk, float gbest, int gid) {
            if (lane == 0) {
                part_val[pbase + kk] = gbest;
                part_id[pbase + kk] = (gbest == LMH_NEG_INF) ? -1 : gid;
            }
        });
}

template<typename FMT, int USE_LSE>
__global__ void lm_head_topk_partials_q(const half* h, const uint8_t* Wq, float* part_val,
                                        int* part_id, const float* bias, int V, int K,
                                        int TILE_V, int num_vtiles, int topk, int use_bias,
                                        float invtemp, float* part_lse) {
    const int vtile = blockIdx.x, t = blockIdx.y, lane = threadIdx.x;
    const int v0 = vtile * TILE_V, v1 = min(v0 + TILE_V, V);
    const int bpr = K / FMT::block_k;
    const half* hrow = h + size_t(t) * K;
    float mine_val[LMH_MAX_PER_LANE];
    int   mine_id[LMH_MAX_PER_LANE];
    bool  used[LMH_MAX_PER_LANE];
    int   nmine = 0;
    float lmax = LMH_NEG_INF, lsum = 0.0f;
    for (int v = v0 + lane; v < v1; v += 32) {
        const uint8_t* row_base = Wq + size_t(v) * bpr * FMT::block_bytes;
        float acc = 0.0f;
        for (int kb = 0; kb < bpr; kb++) {
            const uint8_t* base = row_base + size_t(kb) * FMT::block_bytes;
            const half* xp = hrow + kb * FMT::block_k;
            for (int c0 = 0; c0 < FMT::block_k; c0 += 8) {
                float w[8];
                dequant8<FMT>(base, c0, w);
                #pragma unroll
                for (int i = 0; i < 8; i++) acc += w[i] * __half2float(xp[c0 + i]);
            }
        }
        float ls = acc;
        if (use_bias) ls += bias[v];
        if (USE_LSE) {
            const float tl = ls * invtemp;
            if (tl > lmax) { lsum = lsum * expf(lmax - tl) + 1.0f; lmax = tl; }
            else           { lsum += expf(tl - lmax); }
        }
        mine_val[nmine] = ls;
        mine_id[nmine] = v;
        ++nmine;
    }
    if (USE_LSE) {
        const float gmax = tms::warp_max_f(lmax);
        const float gsum = tms::warp_sum_f((lmax == LMH_NEG_INF) ? 0.0f : lsum * expf(lmax - gmax));
        if (lane == 0)
            part_lse[size_t(t) * num_vtiles + vtile] = (gsum > 0.0f) ? (gmax + logf(gsum)) : LMH_NEG_INF;
    }
    const size_t pbase = (size_t(t) * num_vtiles + vtile) * topk;
    tms::masked_topk_local(mine_val, mine_id, used, nmine, topk, LMH_NEG_INF,
        [&](int kk, float gbest, int gid) {
            if (lane == 0) {
                part_val[pbase + kk] = gbest;
                part_id[pbase + kk] = (gbest == LMH_NEG_INF) ? -1 : gid;
            }
        });
}

// top-k reduce: global merge of the per-tile partial winners, then Gumbel-max
// among the k winners (tempered; noise by global vocab id).
__global__ void lm_head_topk_reduce(const float* part_val, const int* part_id,
                                    int* out_idx, int num_vtiles, int topk,
                                    unsigned seed, float invtemp) {
    const int t = blockIdx.x, lane = threadIdx.x;
    const int ncand = num_vtiles * topk;
    const size_t base = size_t(t) * ncand;
    int   chosen_id[LMH_MAX_K];
    float chosen_val[LMH_MAX_K];
    auto cand = [&](int idx, int& id, float& v, bool& valid) {
        id = part_id[base + idx];
        v = part_val[base + idx];
        valid = id >= 0;
    };
    tms::masked_topk(cand, ncand, topk, lane, LMH_NEG_INF, chosen_id, chosen_val);
    float best = LMH_NEG_INF;
    int bi = chosen_id[0];
    for (int kk = 0; kk < topk; ++kk) {
        if (chosen_id[kk] < 0) continue;
        const float g = rng_gumbel(seed, unsigned(t), unsigned(chosen_id[kk]));
        const float p = chosen_val[kk] * invtemp + g;
        if (p > best || (p == best && chosen_id[kk] < bi)) { best = p; bi = chosen_id[kk]; }
    }
    if (lane == 0) out_idx[t] = bi;
}

// top-p reduce: nucleus over the merged over-selected pool with the TRUE
// full-vocab normalizer from the per-tile lses; 32-step bisection of the
// tempered-logit threshold, then Gumbel-max over {ls >= L}.
__global__ void lm_head_topp_reduce(const float* part_val, const int* part_id,
                                    int* out_idx, int num_vtiles, int topk, float p,
                                    unsigned seed, float invtemp, const float* part_lse) {
    const int t = blockIdx.x, lane = threadIdx.x;
    const int ncand = num_vtiles * topk;
    const size_t base = size_t(t) * ncand;
    float mx = LMH_NEG_INF;
    for (int j = lane; j < ncand; j += 32)
        if (part_id[base + j] >= 0) mx = fmaxf(mx, part_val[base + j] * invtemp);
    mx = tms::warp_max_f(mx);
    const size_t lbase = size_t(t) * num_vtiles;
    float Z = 0.0f;
    for (int vt = lane; vt < num_vtiles; vt += 32) {
        const float pl = part_lse[lbase + vt];
        if (pl > LMH_NEG_INF) Z += expf(pl - mx);
    }
    Z = tms::warp_sum_f(Z);
    float lo = mx - 40.0f, hi = mx;             // largest L with mass{ls>=L} >= p
    for (int it = 0; it < 32; ++it) {
        const float mid = 0.5f * (lo + hi);
        float sm = 0.0f;
        for (int j = lane; j < ncand; j += 32) {
            const float ls = part_val[base + j] * invtemp;
            if (part_id[base + j] >= 0 && ls >= mid) sm += expf(ls - mx);
        }
        sm = tms::warp_sum_f(sm) / Z;
        if (sm >= p) lo = mid; else hi = mid;
    }
    const float L = lo;
    float best = LMH_NEG_INF;
    int bi = -1;
    for (int j = lane; j < ncand; j += 32) {
        const int id = part_id[base + j];
        const float ls = part_val[base + j] * invtemp;
        if (id < 0 || ls < L) continue;
        const float pert = ls + rng_gumbel(seed, unsigned(t), unsigned(id));
        if (pert > best || (pert == best && id < bi)) { best = pert; bi = id; }
    }
    float gbest = best;
    int gid = (bi < 0) ? 0x7fffffff : bi;
    warp_argmax(gbest, gid);
    if (lane == 0) out_idx[t] = (gbest == LMH_NEG_INF) ? -1 : gid;
}

}  // namespace tmq
