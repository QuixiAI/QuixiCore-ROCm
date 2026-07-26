#include "hip/hip_runtime.h"
/**
 * @file
 * @brief CUDA/SM86 port of ThunderMittens' MoE family (moe.metal): routing
 * top-k, permute pipeline (histogram/scan/scatter), 32-row padded schedule
 * (vLLM moe_align pattern), gather, grouped expert GEMMs (square, rect, fused
 * SwiGLU) and the atomic-free weighted finalize.
 *
 * Faithful math/layout port. The grouped GEMMs here are scalar-FMA fp32
 * kernels (one 256-thread block per 32x32 output tile, expert_of_tile<0
 * early-exit) — correctness-first; the perf pass routes them through the
 * 44-TFLOP mma GEMM. Reuses tm_warp.cuh masked_topk (smaller-id ties) and
 * block_exclusive_scan_i32 — the same helpers the samplers/varlen kernels
 * were validated with.
 *
 * Layout contract (identical to TM):
 *   topk_ids/weights (T, K); routing row r in [0,T*K) = (token r/K, slot r%K)
 *   offsets (E+1) exclusive; sorted_row_idx (TK); inv_idx (TK)
 *   off_pad (E+1) 32-padded; expert_of_tile (max_tiles) -1 sentinel;
 *   gather_idx (total_pad_max) -1 -> zero row; inv_pad (TK)
 *   W (E, H, H) / W1 (E, H, 2*inter) [gate|up] / W2 (E, inter, H)
 */
#pragma once
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <cstdint>
#include "tm_warp.cuh"

namespace tmoe {

using tms::masked_topk;
using tms::block_exclusive_scan_i32;

#define MOE_NEG_INF (-3.4028234663852886e38f)
#define MOE_MAX_K 16
#define MOE_SCAN_NT 256

template <typename T> __device__ __forceinline__ float mtf(T v);
template <> __device__ __forceinline__ float mtf<float>(float v)                 { return v; }
template <> __device__ __forceinline__ float mtf<__half>(__half v)               { return __half2float(v); }
template <> __device__ __forceinline__ float mtf<__hip_bfloat16>(__hip_bfloat16 v) { return __bfloat162float(v); }
template <typename T> __device__ __forceinline__ T mft(float v);
template <> __device__ __forceinline__ float          mft<float>(float v)         { return v; }
template <> __device__ __forceinline__ __half         mft<__half>(float v)        { return __float2half(v); }
template <> __device__ __forceinline__ __hip_bfloat16  mft<__hip_bfloat16>(float v) { return __float2bfloat16(v); }

// Top-k experts per token (K masked-argmax rounds) + renormalized softmax
// weights over the selected logits (the Mixtral rule). One warp per token.
template <typename T>
__global__ void moe_route_topk(const T* __restrict__ logits, int* __restrict__ topk_ids,
                               float* __restrict__ topk_weights, int E, int K) {
    const long base = (long)blockIdx.x * E;
    const int lane = threadIdx.x;
    int chosen_id[MOE_MAX_K];
    float chosen_logit[MOE_MAX_K];
    auto cand = [&](int idx, int& id, float& v, bool& valid) {
        id = idx; v = mtf(logits[base + idx]); valid = true;
    };
    masked_topk(cand, E, K, lane, MOE_NEG_INF, chosen_id, chosen_logit);

    float m = MOE_NEG_INF;
    for (int k = 0; k < K; ++k) m = fmaxf(m, chosen_logit[k]);
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) sum += expf(chosen_logit[k] - m);
    const float inv = 1.0f / sum;
    if (lane == 0) {
        const long ob = (long)blockIdx.x * K;
        for (int k = 0; k < K; ++k) {
            topk_ids[ob + k] = chosen_id[k];
            topk_weights[ob + k] = expf(chosen_logit[k] - m) * inv;
        }
    }
}

__global__ void moe_histogram(const int* __restrict__ topk_ids, int* __restrict__ counts,
                              int TK) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= TK) return;
    atomicAdd(counts + topk_ids[tid], 1);
}

// Exclusive prefix sum of counts -> offsets (E+1) + scatter cursor seed.
// One block of MOE_SCAN_NT threads; running prefix across tiles supports any E.
__global__ void moe_scan_offsets(const int* __restrict__ counts, int* __restrict__ offsets,
                                 int* __restrict__ cursor, int E) {
    __shared__ int sg_sums[MOE_SCAN_NT / 32];
    __shared__ int running;
    const int tid = threadIdx.x;
    if (tid == 0) running = 0;
    __syncthreads();
    for (int b = 0; b < E; b += MOE_SCAN_NT) {
        const int e = b + tid;
        const int v = (e < E) ? counts[e] : 0;
        int total;
        const int excl = block_exclusive_scan_i32(v, tid, MOE_SCAN_NT, sg_sums, total);
        if (e < E) {
            offsets[e] = running + excl;
            cursor[e] = running + excl;
        }
        __syncthreads();
        if (tid == 0) running += total;
        __syncthreads();
    }
    if (tid == 0) offsets[E] = running;
}

__global__ void moe_scatter(const int* __restrict__ topk_ids, int* __restrict__ cursor,
                            int* __restrict__ sorted_row_idx, int* __restrict__ inv_idx,
                            int TK) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= TK) return;
    const int pos = atomicAdd(cursor + topk_ids[tid], 1);
    sorted_row_idx[pos] = tid;
    inv_idx[tid] = pos;
}

// Padded schedule: off_pad = exclusive scan of ceil32(counts); expert_of_tile
// via binary search over off_pad (-1 beyond the real total); gather_idx -> -1.
// Single block.
__global__ void moe_pad_offsets(const int* __restrict__ offsets, int* __restrict__ off_pad,
                                int* __restrict__ expert_of_tile, int* __restrict__ gather_idx,
                                int E, int max_tiles, int total_pad_max) {
    __shared__ int sg_sums[MOE_SCAN_NT / 32];
    __shared__ int running;
    const int tid = threadIdx.x;
    if (tid == 0) running = 0;
    __syncthreads();
    for (int b = 0; b < E; b += MOE_SCAN_NT) {
        const int e = b + tid;
        const int count = (e < E) ? (offsets[e + 1] - offsets[e]) : 0;
        const int padded = ((count + 31) / 32) * 32;
        int total;
        const int excl = block_exclusive_scan_i32(padded, tid, MOE_SCAN_NT, sg_sums, total);
        if (e < E) off_pad[e] = running + excl;
        __syncthreads();
        if (tid == 0) running += total;
        __syncthreads();
    }
    if (tid == 0) off_pad[E] = running;
    __syncthreads();

    const int total_pad = off_pad[E];
    for (int t = tid; t < max_tiles; t += MOE_SCAN_NT) {
        const int pos = t * 32;
        if (pos >= total_pad) { expert_of_tile[t] = -1; continue; }
        int lo = 0, hi = E;              // largest e with off_pad[e] <= pos
        while (hi - lo > 1) {
            const int mid = (lo + hi) / 2;
            if (off_pad[mid] <= pos) lo = mid; else hi = mid;
        }
        expert_of_tile[t] = lo;
    }
    for (int p = tid; p < total_pad_max; p += MOE_SCAN_NT) gather_idx[p] = -1;
}

// Compact position p -> padded position; record gather (padpos -> token) and
// inv_pad (routing row -> padpos, what finalize reads).
__global__ void moe_pad_scatter(const int* __restrict__ sorted_row_idx,
                                const int* __restrict__ offsets, const int* __restrict__ off_pad,
                                int* __restrict__ gather_idx, int* __restrict__ inv_pad,
                                int TK, int E, int K) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= TK) return;
    int lo = 0, hi = E;                  // expert whose compact segment contains p
    while (hi - lo > 1) {
        const int mid = (lo + hi) / 2;
        if (offsets[mid] <= p) lo = mid; else hi = mid;
    }
    const int padpos = off_pad[lo] + (p - offsets[lo]);
    const int r = sorted_row_idx[p];
    gather_idx[padpos] = r / K;
    inv_pad[r] = padpos;
}

// permuted_input[p, :] = x[gather_idx[p], :] (zeros for pad rows).
// One 128-thread block per padded row.
template <typename T>
__global__ void moe_gather(const T* __restrict__ x, const int* __restrict__ gather_idx,
                           T* __restrict__ out, int H) {
    const long p = blockIdx.x;
    const int src = gather_idx[p];
    T* dst = out + p * H;
    if (src < 0) {
        for (int i = threadIdx.x; i < H; i += blockDim.x) dst[i] = mft<T>(0.0f);
        return;
    }
    const T* row = x + (long)src * H;
    for (int i = threadIdx.x; i < H; i += blockDim.x) dst[i] = row[i];
}

// ---- grouped (segmented) expert GEMMs, one 256-thread block per 32x32 output
// tile; every tile belongs to exactly one expert (32-row padding), tiles with
// expert_of_tile < 0 exit (their rows are never read downstream). ----
template <typename T>
__global__ void moe_grouped_gemm_rect(T* __restrict__ out, const T* __restrict__ A,
                                      const T* __restrict__ W,
                                      const int* __restrict__ expert_of_tile,
                                      int total_rows, int K_dim, int N_out) {
    const int OY = blockIdx.y, OX = blockIdx.x;
    const int e = expert_of_tile[OY];
    if (e < 0) return;
    const T* We = W + (long)e * K_dim * N_out;
    // thread owns 4 of the tile's 1024 elements
    for (int t = threadIdx.x; t < 32 * 32; t += blockDim.x) {
        const int r = OY * 32 + t / 32;
        const int c = OX * 32 + t % 32;
        if (r >= total_rows || c >= N_out) continue;
        float acc = 0.0f;
        const T* arow = A + (long)r * K_dim;
        for (int k = 0; k < K_dim; ++k)
            acc += mtf(arow[k]) * mtf(We[(long)k * N_out + c]);
        out[(long)r * N_out + c] = mft<T>(acc);
    }
}

// TM's square moe_grouped_gemm (W (E,H,H)) is the K_dim==N_out==H case of
// rect; hosts dispatch moe_grouped_gemm_rect for both.

// Fused SiLU-GLU GEMM1: out = silu(A @ W1_gate) * (A @ W1_up); W1[e] is
// (H, 2*inter) laid out [gate | up].
template <typename T>
__global__ void moe_grouped_gemm_swiglu(T* __restrict__ out, const T* __restrict__ A,
                                        const T* __restrict__ W1,
                                        const int* __restrict__ expert_of_tile,
                                        int total_rows, int H, int inter) {
    const int OY = blockIdx.y, OX = blockIdx.x;
    const int e = expert_of_tile[OY];
    if (e < 0) return;
    const T* We = W1 + (long)e * H * (2 * inter);
    for (int t = threadIdx.x; t < 32 * 32; t += blockDim.x) {
        const int r = OY * 32 + t / 32;
        const int c = OX * 32 + t % 32;
        if (r >= total_rows || c >= inter) continue;
        float gate = 0.0f, up = 0.0f;
        const T* arow = A + (long)r * H;
        for (int k = 0; k < H; ++k) {
            const float a = mtf(arow[k]);
            gate += a * mtf(We[(long)k * (2 * inter) + c]);
            up   += a * mtf(We[(long)k * (2 * inter) + inter + c]);
        }
        const float s = gate / (1.0f + expf(-gate));   // silu
        out[(long)r * inter + c] = mft<T>(s * up);
    }
}

// Per-token weighted k-way reduce of the (permuted-order) expert outputs via
// the inverse map — no atomics. One warp per token. inv is inv_idx (compact)
// or inv_pad (padded schedule); expert_out rows indexed by it either way.
template <typename T>
__global__ void moe_finalize(const T* __restrict__ expert_out, const int* __restrict__ inv,
                             const float* __restrict__ topk_weights, T* __restrict__ out,
                             int K, int Hdim) {
    const int token = blockIdx.x;
    const long wbase = (long)token * K;
    const long obase = (long)token * Hdim;
    for (int h = threadIdx.x; h < Hdim; h += 32) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            const int pos = inv[token * K + k];
            acc += topk_weights[wbase + k] * mtf(expert_out[(long)pos * Hdim + h]);
        }
        out[obase + h] = mft<T>(acc);
    }
}

// ================= Phase 4: MoE completeness (Metal/CPU parity) =================
// These take ROW-SHAPED expert ids (one per padded row, -1 = padding) rather
// than the per-tile expert_of_tile the forward grouped GEMMs use, matching the
// CPU sibling contract. Accumulating kernels (gather_backward,
// finalize_backward, grouped_gemm_backward_weight) require their destination
// zeroed by the caller, exactly as the CPU reference fills before accumulating.

enum MoeScoring { MOE_SCORE_SOFTMAX = 0, MOE_SCORE_SIGMOID = 1, MOE_SCORE_SQRT_SOFTPLUS = 2 };
#define MOE_MAX_GROUPS 64

// Grouped top-k routing ("noaux_tc"): score the logits, add a per-expert
// selection bias, rank expert GROUPS by the sum of their top two biased scores,
// keep the best `top_groups`, then top-k experts within that candidate set.
// Emitted weights are the UNBIASED scores of the winners, optionally
// renormalized over the selection, times routed_scale. Ties break to the lower
// id at both levels. One 32-lane block per token; smem holds scores+selection.
template <typename T>
__global__ void moe_route_grouped(const T* __restrict__ logits,
                                  const float* __restrict__ bias,
                                  int* __restrict__ topk_ids,
                                  float* __restrict__ topk_weights,
                                  int E, int K, int groups, int top_groups,
                                  int renormalize, float routed_scale, int scoring) {
    extern __shared__ float smem[];
    float* score = smem;            // [E] unbiased score
    float* select = smem + E;       // [E] score + bias
    __shared__ unsigned char gkeep[MOE_MAX_GROUPS];
    const int lane = threadIdx.x;
    const long base = (long)blockIdx.x * E;

    if (scoring == MOE_SCORE_SOFTMAX) {
        float m = MOE_NEG_INF;
        for (int e = lane; e < E; e += 32) m = fmaxf(m, mtf(logits[base + e]));
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor(m, off));
        float s = 0.0f;
        for (int e = lane; e < E; e += 32) { const float v = expf(mtf(logits[base + e]) - m);
                                             score[e] = v; s += v; }
        s = tms::warp_sum_f(s);
        const float inv = 1.0f / s;
        for (int e = lane; e < E; e += 32) score[e] *= inv;
    } else {
        for (int e = lane; e < E; e += 32) {
            const float v = mtf(logits[base + e]);
            score[e] = (scoring == MOE_SCORE_SIGMOID) ? 1.0f / (1.0f + expf(-v))
                                                      : sqrtf(log1pf(expf(-fabsf(v))) + fmaxf(v, 0.0f));
        }
    }
    for (int e = lane; e < E; e += 32) select[e] = score[e] + (bias ? bias[e] : 0.0f);
    __syncthreads();

    // Group ranking on one lane: `groups` is small (<= MOE_MAX_GROUPS) and the
    // top-two-sum + stable tie rule is cheapest done serially.
    if (lane == 0) {
        const int gsz = E / groups;
        float gs[MOE_MAX_GROUPS];
        for (int g = 0; g < groups; ++g) {
            float f = MOE_NEG_INF, sd = MOE_NEG_INF;
            for (int i = 0; i < gsz; ++i) {
                const float v = select[g * gsz + i];
                if (v > f) { sd = f; f = v; } else if (v > sd) { sd = v; }
            }
            gs[g] = f + (gsz > 1 ? sd : 0.0f);
        }
        for (int g = 0; g < groups; ++g) gkeep[g] = 0;
        for (int r = 0; r < top_groups; ++r) {
            int best = -1;
            for (int g = 0; g < groups; ++g) {
                if (gkeep[g]) continue;
                if (best < 0 || gs[g] > gs[best]) best = g;   // ties -> lower id
            }
            gkeep[best] = 1;
        }
    }
    __syncthreads();

    const int gsz = E / groups;
    int chosen_id[MOE_MAX_K];
    float chosen_val[MOE_MAX_K];
    auto cand = [&](int idx, int& id, float& v, bool& valid) {
        id = idx; v = select[idx]; valid = gkeep[idx / gsz] != 0;
    };
    masked_topk(cand, E, K, lane, MOE_NEG_INF, chosen_id, chosen_val);

    if (lane == 0) {
        float denom = 0.0f;
        for (int k = 0; k < K; ++k) denom += score[chosen_id[k]];
        const float f = renormalize ? routed_scale / denom : routed_scale;
        const long ob = (long)blockIdx.x * K;
        for (int k = 0; k < K; ++k) {
            topk_ids[ob + k] = chosen_id[k];
            topk_weights[ob + k] = score[chosen_id[k]] * f;
        }
    }
}

// Adjoint of moe_gather: scatter-add each gathered row back to its source
// token. gather_rows[r] < 0 marks a padded row and contributes nothing.
// Rows can repeat (a token feeds several experts), so this accumulates.
template <typename T>
__global__ void moe_gather_backward(const T* __restrict__ grad_gathered,
                                    const int* __restrict__ gather_rows,
                                    float* __restrict__ grad_input,
                                    int gathered_rows, int dim) {
    const int r = blockIdx.x;
    if (r >= gathered_rows) return;
    const int src = gather_rows[r];
    if (src < 0) return;
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        atomicAdd(&grad_input[(long)src * dim + i], mtf(grad_gathered[(long)r * dim + i]));
}

// Adjoint of moe_finalize. grad_expert_out[inverse[route]] += w[route]*grad_out,
// and grad_expert_weights[route] = <grad_out[token], expert_out[inverse[route]]>.
template <typename T>
__global__ void moe_finalize_backward(const T* __restrict__ grad_out,
                                      const T* __restrict__ expert_out,
                                      const int* __restrict__ inverse,
                                      const float* __restrict__ expert_weights,
                                      float* __restrict__ grad_expert_out,
                                      float* __restrict__ grad_expert_weights,
                                      int top_k, int dim) {
    const int token = blockIdx.x;
    const int lane = threadIdx.x;
    for (int k = 0; k < top_k; ++k) {
        const long route = (long)token * top_k + k;
        const int p = inverse[route];
        if (p < 0) continue;
        const float w = expert_weights[route];
        float wg = 0.0f;
        for (int i = lane; i < dim; i += blockDim.x) {
            const float g = mtf(grad_out[(long)token * dim + i]);
            atomicAdd(&grad_expert_out[(long)p * dim + i], w * g);
            wg += g * mtf(expert_out[(long)p * dim + i]);
        }
        wg = tms::warp_sum_f(wg);
        if (lane == 0) grad_expert_weights[route] = wg;
    }
}

// Adjoint w.r.t. the input of the row-indexed expert GEMM. W is [E, K_in, N_out]
// (input-major, matching moe_grouped_gemm_rect), so this contracts over N_out.
template <typename T>
__global__ void moe_grouped_gemm_backward_input(const T* __restrict__ grad_out,
                                                const T* __restrict__ W,
                                                const int* __restrict__ expert_ids,
                                                float* __restrict__ grad_input,
                                                int rows, int K_in, int N_out) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    const int e = expert_ids[r];
    if (e < 0) return;
    const T* We = W + (long)e * K_in * N_out;
    const T* grow = grad_out + (long)r * N_out;
    for (int i = threadIdx.x; i < K_in; i += blockDim.x) {
        float acc = 0.0f;
        for (int o = 0; o < N_out; ++o) acc += mtf(grow[o]) * mtf(We[(long)i * N_out + o]);
        grad_input[(long)r * K_in + i] = acc;
    }
}

// Adjoint w.r.t. the weights: accumulate the outer product x[r] (x) grad_out[r]
// into the expert that row r was routed to. Rows sharing an expert collide, so
// this accumulates; caller zeroes grad_weights.
template <typename T>
__global__ void moe_grouped_gemm_backward_weight(const T* __restrict__ x,
                                                 const T* __restrict__ grad_out,
                                                 const int* __restrict__ expert_ids,
                                                 float* __restrict__ grad_weights,
                                                 int rows, int K_in, int N_out) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    const int e = expert_ids[r];
    if (e < 0) return;
    float* ge = grad_weights + (long)e * K_in * N_out;
    const T* xr = x + (long)r * K_in;
    const T* gr = grad_out + (long)r * N_out;
    for (long t = threadIdx.x; t < (long)K_in * N_out; t += blockDim.x) {
        const int i = (int)(t / N_out), o = (int)(t % N_out);
        atomicAdd(&ge[t], mtf(xr[i]) * mtf(gr[o]));
    }
}

} // namespace tmoe
