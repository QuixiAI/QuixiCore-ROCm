#pragma once
#include "kittens.cuh"
/**
 * CDNA3 (gfx942) GQA forward attention — MFMA-tiled flash kernel.
 *
 * Processes a BQ=16 query block per 64-lane wavefront, reusing each K/V block
 * across all 16 queries and using bf16 MFMA (v_mfma_f32_16x16x16_bf16) for QK^T
 * and P@V. The softmax reduces over an LDS transpose of S (lane-owns-full-row),
 * so there is no distributed-layout reduction. ~13-16x faster than the naive
 * one-wavefront-per-query kernel (see attn_bench.cu); validated to <0.02 rel
 * error vs an fp32 host reference (harness.cpp) and PyTorch SDPA (verify_sdpa.py).
 * The naive kernel is kept as the correctness oracle in attn_bench.cu / attn_mfma.cu.
 *
 * Layout [B,N,H,D] (Q/O), [B,N,H_KV,D] (K/V). GQA kv=h/(H/H_KV). scale=1/sqrt(D).
 * ATTN_CAUSAL restricts key j<=query i. D and N are multiples of 16.
 */
#ifndef ATTN_B
#define ATTN_B 16
#endif
#ifndef ATTN_H
#define ATTN_H 64
#endif
#ifndef ATTN_H_KV
#define ATTN_H_KV 8
#endif
#ifndef ATTN_N
#define ATTN_N 2048
#endif
#ifndef ATTN_D
#define ATTN_D 128
#endif
#ifndef ATTN_CAUSAL
#define ATTN_CAUSAL 0
#endif

using namespace kittens;
using _gl_QKVO = gl<bf16, -1, -1, -1, -1>;

template<int D> struct attn_globals {
    _gl_QKVO Qg, Kg, Vg, Og;
    gl<float, -1, -1, -1, -1> L_vec;
    hipStream_t stream;
    dim3 grid()  { return dim3(ATTN_N / 16, ATTN_H, ATTN_B); } // one wavefront per 16-query block
    dim3 block() { return dim3(64); }
    size_t dynamic_shared_memory() { return 0; }
};

typedef __attribute__((__vector_size__(4 * sizeof(short)))) short attn_s4;
typedef __attribute__((__vector_size__(4 * sizeof(float)))) float attn_f4;
__device__ __forceinline__ attn_f4 attn_mma(attn_s4 a, attn_s4 b, attn_f4 c) {
    return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, c, 0, 0, 0);
}
__device__ __forceinline__ short attn_bfb(bf16 x) { short s; __builtin_memcpy(&s, &x, 2); return s; }

template<int D>
__global__ void attend_ker(const attn_globals<D> g) {
    constexpr int BQ = 16, BK = 16;
    const bf16* Q = &g.Qg[{0, 0, 0, 0}];
    const bf16* K = &g.Kg[{0, 0, 0, 0}];
    const bf16* V = &g.Vg[{0, 0, 0, 0}];
    bf16* O = &g.Og[{0, 0, 0, 0}];
    float* Lv = &g.L_vec[{0, 0, 0, 0}];
    const int N = ATTN_N;
    const int qb = blockIdx.x, h = blockIdx.y, b = blockIdx.z, l = threadIdx.x;
    const int hk = h / (ATTN_H / ATTN_H_KV), q0 = qb * BQ, DT = D / 16;
    const float scale = 1.0f / sqrtf((float)D);
    const int lo = l >> 4, li = l & 15;
    auto Qi = [&](int qi, int d) { return (size_t)(((b * N + q0 + qi) * ATTN_H + h) * D + d); };
    auto Ki = [&](int kj, int d) { return (size_t)(((b * N + kj) * ATTN_H_KV + hk) * D + d); };

    attn_s4 Qf[ATTN_D / 16];
    #pragma unroll
    for (int s = 0; s < DT; s++) { const int d0 = s * 16 + lo * 4; attn_s4 f;
        #pragma unroll
        for (int v = 0; v < 4; v++) ((short*)&f)[v] = attn_bfb(Q[Qi(li, d0 + v)]); Qf[s] = f; }
    attn_f4 Oacc[ATTN_D / 16];
    #pragma unroll
    for (int s = 0; s < DT; s++) Oacc[s] = attn_f4{0, 0, 0, 0};
    __shared__ float sS[BQ][BK];
    __shared__ bf16  sP[BQ][BK];
    __shared__ float sm[BQ], sl[BQ], sc[BQ];
    if (l < BQ) { sm[l] = -3.0e38f; sl[l] = 0.0f; }
    __syncthreads();

    const int kmax = ATTN_CAUSAL ? (q0 + BQ) : N;
    for (int k0 = 0; k0 < kmax; k0 += BK) {
        attn_f4 S = attn_f4{0, 0, 0, 0};
        #pragma unroll
        for (int s = 0; s < DT; s++) { const int d0 = s * 16 + lo * 4; attn_s4 Kf;
            #pragma unroll
            for (int v = 0; v < 4; v++) ((short*)&Kf)[v] = attn_bfb(K[Ki(k0 + li, d0 + v)]);
            S = attn_mma(Qf[s], Kf, S); }
        #pragma unroll
        for (int v = 0; v < 4; v++) sS[4 * lo + v][li] = S[v] * scale;
        __syncthreads();
        if (l < BQ) { const int qi = l; float mp = sm[qi], mx = mp;
            #pragma unroll
            for (int j = 0; j < BK; j++) { int kk = k0 + j; if (!ATTN_CAUSAL || kk <= q0 + qi) mx = fmaxf(mx, sS[qi][j]); }
            const float corr = __expf(mp - mx); float ls = sl[qi] * corr;
            #pragma unroll
            for (int j = 0; j < BK; j++) { int kk = k0 + j; float p = (!ATTN_CAUSAL || kk <= q0 + qi) ? __expf(sS[qi][j] - mx) : 0.0f;
                sP[qi][j] = __float2bfloat16(p); ls += p; }
            sm[qi] = mx; sl[qi] = ls; sc[qi] = corr; }
        __syncthreads();
        #pragma unroll
        for (int s = 0; s < DT; s++) { attn_f4 o = Oacc[s];
            #pragma unroll
            for (int v = 0; v < 4; v++) o[v] *= sc[4 * lo + v]; Oacc[s] = o; }
        attn_s4 Pf;
        #pragma unroll
        for (int v = 0; v < 4; v++) ((short*)&Pf)[v] = attn_bfb(sP[li][lo * 4 + v]);
        #pragma unroll
        for (int s = 0; s < DT; s++) { const int d = s * 16 + li; attn_s4 Vf;
            #pragma unroll
            for (int v = 0; v < 4; v++) ((short*)&Vf)[v] = attn_bfb(V[Ki(k0 + lo * 4 + v, d)]);
            Oacc[s] = attn_mma(Pf, Vf, Oacc[s]); }
        __syncthreads();
    }
    #pragma unroll
    for (int s = 0; s < DT; s++) { const int d = s * 16 + li;
        #pragma unroll
        for (int v = 0; v < 4; v++) { const int qi = 4 * lo + v; const float inv = sl[qi] > 0.0f ? 1.0f / sl[qi] : 0.0f;
            O[Qi(qi, d)] = __float2bfloat16(Oacc[s][v] * inv); } }
    if (l < BQ) { const int qi = l; Lv[(size_t)(b * ATTN_H + h) * N + q0 + qi] = sm[qi] + __logf(sl[qi] > 0.0f ? sl[qi] : 1.0f); }
}

template<int D>
void dispatch_micro(attn_globals<D> g) {
    attend_ker<D><<<g.grid(), g.block(), 0, g.stream>>>(g);
}
