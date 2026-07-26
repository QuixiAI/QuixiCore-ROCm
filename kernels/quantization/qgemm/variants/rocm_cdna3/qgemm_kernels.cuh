/**
 * @file
 * @brief qgemm kernel set (CDNA3 MFMA), split out of qgemm.cu so the golden
 * harness and the shape benches can share one definition of the kernels.
 */
#pragma once
#include "tm_qmm_mfma.cuh"
#include <hip/hip_fp16.h>

using namespace tmq;

// ---- qgemm: one 64-wide wavefront per 16x16 output tile (CDNA3 MFMA) ----
// Y(M,N) = X(M,K) @ W(N,K)^T. A=X, B=W^T; one v_mfma_f32_16x16x16_f16 per K=16.
// Lane l owns output column n0+l%16, rows m0 + 4*(l/16) + {0..3}.
template<typename FMT>
__global__ void qgemm(float* Y, const __half* X, const uint8_t* Wq, int M, int N, int K) {
    const int n0 = blockIdx.x * 16;
    const int m0 = blockIdx.y * 16;
    const int bpr = K / FMT::block_k;

    float4_t acc = {0, 0, 0, 0};
    for (int k0 = 0; k0 < K; k0 += 16) {
        half4_t a = load_xfrag(X, K, m0, k0);
        half4_t b = load_wfrag<FMT>(Wq, bpr, n0, k0);
        acc = mma_16x16x16(a, b, acc);
    }
    const int l = threadIdx.x & 63;
    const int n = n0 + (l & 15);
    const int mrow = m0 + (l >> 4) * 4;
    #pragma unroll
    for (int v = 0; v < 4; v++)
        if (mrow + v < M) Y[size_t(mrow + v) * N + n] = acc[v];
}

// wide N-tile variant (perf pass for M>=~256): NT 16-wide N-tiles per wavefront;
// the X fragment is loaded once per k-step and reused across NT W-fragments, so
// X traffic drops NT-fold and the MFMA:load ratio rises NT-fold. Bitwise-identical
// to qgemm (same fragment math). Occupancy-limited at tiny M (see qgemm_pick_nt).
template<typename FMT, int NT>
__global__ void qgemm_wide(float* Y, const __half* X, const uint8_t* Wq, int M, int N, int K) {
    const int n0 = blockIdx.x * (16 * NT);
    const int m0 = blockIdx.y * 16;
    const int bpr = K / FMT::block_k;
    float4_t acc[NT];
    #pragma unroll
    for (int nt = 0; nt < NT; nt++) acc[nt] = float4_t{0, 0, 0, 0};
    for (int k0 = 0; k0 < K; k0 += 16) {
        half4_t a = load_xfrag(X, K, m0, k0);
        #pragma unroll
        for (int nt = 0; nt < NT; nt++) {
            half4_t b = load_wfrag<FMT>(Wq, bpr, n0 + nt * 16, k0);
            acc[nt] = mma_16x16x16(a, b, acc[nt]);
        }
    }
    const int l = threadIdx.x & 63, ln = l & 15, mrow = m0 + (l >> 4) * 4;
    #pragma unroll
    for (int nt = 0; nt < NT; nt++) { const int n = n0 + nt * 16 + ln;
        #pragma unroll
        for (int v = 0; v < 4; v++) if (mrow + v < M) Y[size_t(mrow + v) * N + n] = acc[nt][v];
    }
}

template<typename FMT, int MT, int NT>
__device__ __forceinline__ void stage_qgemm_cta_tile(
        __half* sX, __half* sW, const __half* X, const uint8_t* Wq,
        int M, int K, int bpr, int m0, int n0, int k0) {
    const int tid = threadIdx.x;
    for (int idx = tid; idx < MT * 16 * 16; idx += blockDim.x) {
        const int mt = idx / (16 * 16), r = idx - mt * 16 * 16;
        const int m = r / 16, k = r & 15, gm = m0 + mt * 16 + m;
        sX[idx] = (gm < M) ? X[size_t(gm) * K + k0 + k] : __float2half(0.0f);
    }
    // W: one 8-wide K span per thread (Metal's dequant_into_shared). dequant8
    // unpacks the block/sub-block scale ONCE per span instead of once per
    // element — 16x fewer scale decodes per 16-wide row segment — and the 8
    // packed weights a span needs are contiguous in the block, so the global
    // read is one short burst instead of 8 scattered byte reads. An 8-aligned
    // span never straddles a quant block (every block_k is a multiple of 8).
    // Bit-identical to the per-element fill.
    constexpr int SPANS = 16 / 8;
    for (int idx = tid; idx < NT * 16 * SPANS; idx += blockDim.x) {
        const int span = idx % SPANS, rowi = idx / SPANS;
        const int nt = rowi / 16, n = rowi - nt * 16;
        const int k = k0 + span * 8;
        const int kb = k / FMT::block_k, cin = k % FMT::block_k;
        const uint8_t* base = Wq + (size_t(n0 + nt * 16 + n) * bpr + kb) * FMT::block_bytes;
        float w[8];
        dequant8<FMT>(base, cin, w);
        __half* dst = sW + nt * 16 * 16 + n * 16 + span * 8;
        #pragma unroll
        for (int i = 0; i < 8; i++) dst[i] = __float2half(w[i]);
    }
}

__device__ __forceinline__ half4_t load_xfrag_cta_smem(const __half* sX, int mt) {
    const int l = threadIdx.x & 63;
    const int m = l & 15;
    const int k = (l >> 4) * 4;
    return *reinterpret_cast<const half4_t*>(sX + mt * 16 * 16 + m * 16 + k);
}

__device__ __forceinline__ half4_t load_wfrag_cta_smem(const __half* sW, int nt) {
    const int l = threadIdx.x & 63;
    const int n = l & 15;
    const int k = (l >> 4) * 4;
    return *reinterpret_cast<const half4_t*>(sW + nt * 16 * 16 + n * 16 + k);
}

// Multi-wave CTA candidate: MT wavefronts own MT 16-row output tiles and share
// the same NT 16-column W tile through LDS. Useful only at prefill-scale M,
// where the larger CTA grid still fills the device and W/dequant reuse amortizes
// the LDS/barrier cost.
template<typename FMT, int MT, int NT>
__global__ void qgemm_cta_lds(float* Y, const __half* X, const uint8_t* Wq, int M, int N, int K) {
    const int n0 = blockIdx.x * (16 * NT);
    const int m0 = blockIdx.y * (16 * MT);
    const int wave = threadIdx.x >> 6;
    const int bpr = K / FMT::block_k;
    __shared__ __half sX[2][MT * 16 * 16];
    __shared__ __half sW[2][NT * 16 * 16];
    float4_t acc[NT];
    #pragma unroll
    for (int nt = 0; nt < NT; nt++) acc[nt] = float4_t{0, 0, 0, 0};

    stage_qgemm_cta_tile<FMT, MT, NT>(sX[0], sW[0], X, Wq, M, K, bpr, m0, n0, 0);
    __syncthreads();
    for (int k0 = 0; k0 < K; k0 += 16) {
        const int buf = (k0 / 16) & 1, nxt = buf ^ 1;
        half4_t a = load_xfrag_cta_smem(sX[buf], wave);
        half4_t b[NT];
        #pragma unroll
        for (int nt = 0; nt < NT; nt++) b[nt] = load_wfrag_cta_smem(sW[buf], nt);
        if (k0 + 16 < K) {
            stage_qgemm_cta_tile<FMT, MT, NT>(sX[nxt], sW[nxt], X, Wq, M, K, bpr, m0, n0, k0 + 16);
        }
        #pragma unroll
        for (int nt = 0; nt < NT; nt++) acc[nt] = mma_16x16x16(a, b[nt], acc[nt]);
        __syncthreads();
    }
    const int l = threadIdx.x & 63, ln = l & 15;
    const int mrow = m0 + wave * 16 + (l >> 4) * 4;
    #pragma unroll
    for (int nt = 0; nt < NT; nt++) {
        const int n = n0 + nt * 16 + ln;
        #pragma unroll
        for (int v = 0; v < 4; v++) if (mrow + v < M) Y[size_t(mrow + v) * N + n] = acc[nt][v];
    }
}

// Pick NT (N-tiles/wavefront) so the grid still fills the device (~2 waves over
// 304 CUs); wider tiles amortize the X load but shrink the grid, so cap by
// occupancy and require 16*NT | N. NT=1 => identical to qgemm (decode fallback).
static inline int qgemm_pick_nt(int M, int N) {
    const long tilesM = (M + 15) / 16;
    for (int nt : {4, 2}) if (N % (16 * nt) == 0 && (long)(N / (16 * nt)) * tilesM >= 608) return nt;
    return 1;
}

// full-dequant kernel (route for 256-superblock formats)
template<typename FMT>
__global__ void dequant_to_fp16(half* out, const uint8_t* Wq, int N, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * K) return;
    int row = idx / K, col = idx % K;
    const uint8_t* base = Wq + (size_t(row) * (K / FMT::block_k) + col / FMT::block_k) * FMT::block_bytes;
    out[idx] = __float2half(FMT::dequant(base, col % FMT::block_k));
}

// ---- ksplit variant (perf pass): at decode shapes the warp-per-tile grid is
// only (N/16)*(M/16) warps — half the SMs idle at M=64,N=512. Slice K across
// blockIdx.z and atomicAdd fp32 partials into a zeroed Y. Same fragment math.
// (Also consolidated in tm_kernels.cuh as tmq::qgemm_ksplit.) ----
template<typename FMT>
__global__ void qgemm_ksplit(float* Y, const __half* X, const uint8_t* Wq,
                             int M, int N, int K, int k_chunk) {
    const int n0 = blockIdx.x * 16;
    const int m0 = blockIdx.y * 16;
    const int k_beg = blockIdx.z * k_chunk;
    const int k_end = min(K, k_beg + k_chunk);
    const int bpr = K / FMT::block_k;

    float4_t acc = {0, 0, 0, 0};
    for (int k0 = k_beg; k0 < k_end; k0 += 16) {
        half4_t a = load_xfrag(X, K, m0, k0);
        half4_t b = load_wfrag<FMT>(Wq, bpr, n0, k0);
        acc = mma_16x16x16(a, b, acc);
    }
    const int l = threadIdx.x & 63;
    const int n = n0 + (l & 15);
    const int mrow = m0 + (l >> 4) * 4;
    #pragma unroll
    for (int v = 0; v < 4; v++)
        if (mrow + v < M) atomicAdd(&Y[size_t(mrow + v) * N + n], acc[v]);
}

static inline int qgemm_pick_kchunk(int M, int N, int K, int block_k) {
    const long tiles = long((N + 15) / 16) * ((M + 15) / 16);
    const int target = 1664;                      // ~82 SMs x 20 warps
    int splits = int((target + tiles - 1) / tiles);
    const int align = (block_k > 16) ? block_k : 16;
    int chunk = ((K / (splits > 0 ? splits : 1)) + align - 1) / align * align;
    if (chunk < align) chunk = align;
    return chunk;
}

