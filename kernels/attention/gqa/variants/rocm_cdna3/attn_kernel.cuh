#pragma once
#include "kittens.cuh"
/**
 * CDNA3 (gfx942) GQA forward attention — correctness-first native kernel.
 *
 * The HipKittens CDNA4 GQA kernel is numerically wrong on CDNA3 (it is built on
 * CDNA4 register-tile geometry / packed-shape reinterpret_casts; confirmed vs a
 * standalone host ref AND PyTorch SDPA). Rather than fight that microarchitecture-
 * specific code, this is a clean, correct CDNA3 flash-attention forward using the
 * same online-softmax proven in the serving paged_attention port: one 64-wide
 * wavefront per (query, head, batch); the 64 lanes split the head dim D
 * (EPL = D/64 elements per lane); QK^T is a wavefront reduction; softmax state
 * (m, l) and the O accumulator are kept online across the KV sequence.
 *
 * Layout [B,N,H,D] for Q/O and [B,N,H_KV,D] for K/V (matches the torch test and
 * the standalone harness). GQA head map: kv = h / (H/H_KV). Scale = 1/sqrt(D).
 * Correctness gate: harness.cpp (host fp32 ref) and test_python.py vs SDPA.
 * Perf note: this is O(N) per query with no K/V reuse across queries — correct
 * but bandwidth-bound; an MFMA-tiled version (reusing tm_qmm_mfma.cuh) is the
 * perf follow-up.
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
    dim3 grid()  { return dim3(ATTN_N, ATTN_H, ATTN_B); } // one wavefront per (query, head, batch)
    dim3 block() { return dim3(64); }
    size_t dynamic_shared_memory() { return 0; }
};

template<int D>
__global__ void attend_ker(const attn_globals<D> g) {
    constexpr int WF = 64;
    constexpr int EPL = D / WF;                 // elements of the head dim per lane (D multiple of 64)
    const int q = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
    const int lane = threadIdx.x;
    const int hk = h / (ATTN_H / ATTN_H_KV);    // GQA: query head -> kv head
    const int N = ATTN_N;
    const float scale = 1.0f / sqrtf((float)D);

    float qv[EPL], acc[EPL];
    #pragma unroll
    for (int e = 0; e < EPL; e++) { qv[e] = __bfloat162float(g.Qg[{b, q, h, lane + e * WF}]); acc[e] = 0.0f; }

    float m = -3.4028234663852886e38f, l = 0.0f;
    const int jmax = ATTN_CAUSAL ? (q + 1) : N;   // causal: query q attends to keys 0..q
    for (int j = 0; j < jmax; j++) {
        float part = 0.0f;
        #pragma unroll
        for (int e = 0; e < EPL; e++) part += qv[e] * __bfloat162float(g.Kg[{b, j, hk, lane + e * WF}]);
        // QK^T: reduce the D-partial across the 64-lane wavefront
        #pragma unroll
        for (int o = 32; o > 0; o >>= 1) part += __shfl_xor(part, o);
        const float s = part * scale;
        const float mn = fmaxf(m, s);
        const float corr = __expf(m - mn);
        const float p = __expf(s - mn);
        l = l * corr + p;
        #pragma unroll
        for (int e = 0; e < EPL; e++)
            acc[e] = acc[e] * corr + p * __bfloat162float(g.Vg[{b, j, hk, lane + e * WF}]);
        m = mn;
    }
    const float inv = (l > 0.0f) ? 1.0f / l : 0.0f;
    #pragma unroll
    for (int e = 0; e < EPL; e++) g.Og[{b, q, h, lane + e * WF}] = __float2bfloat16(acc[e] * inv);
    if (lane == 0) g.L_vec[{b, h, 0, q}] = m + __logf(l > 0.0f ? l : 1.0f);
}

template<int D>
void dispatch_micro(attn_globals<D> g) {
    attend_ker<D><<<g.grid(), g.block(), 0, g.stream>>>(g);
}
