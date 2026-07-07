#pragma once
/**
 * @file
 * @brief CDNA3 (gfx942) MFMA-tiled GQA flash-attention forward + standalone fp64
 * oracle. Optimized replacement for the naive one-wavefront-per-query kernel
 * (attn_kernel.cuh): processes a BQ=16 query block per 64-lane wavefront, reusing
 * each K/V block across all 16 queries and using bf16 MFMA
 * (v_mfma_f32_16x16x16_bf16) for QK^T and P@V. The softmax reduces over an LDS
 * transpose of S (lane-owns-full-row), so no distributed-layout reduction.
 *
 * Layout [B,N,H,D] (Q/O), [B,N,H_KV,D] (K/V). GQA kv=h/(H/H_KV). scale=1/sqrt(D).
 * ATTN_CAUSAL restricts key j<=query i. D and N multiples of 16.
 *   hipcc -std=c++17 -O3 --offload-arch=gfx942 attn_mfma.cu -o attn_mfma.out
 */
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <random>
#include <cmath>
#ifndef ATTN_B
#define ATTN_B 1
#endif
#ifndef ATTN_H
#define ATTN_H 8
#endif
#ifndef ATTN_H_KV
#define ATTN_H_KV 2
#endif
#ifndef ATTN_N
#define ATTN_N 256
#endif
#ifndef ATTN_D
#define ATTN_D 128
#endif
#ifndef ATTN_CAUSAL
#define ATTN_CAUSAL 0
#endif
#define CK(x) do{hipError_t e=(x); if(e!=hipSuccess){printf("HIP %s @%d\n",hipGetErrorString(e),__LINE__);exit(1);} }while(0)
using bf16 = __hip_bfloat16;
typedef __attribute__((__vector_size__(4*sizeof(short)))) short short4_t;
typedef __attribute__((__vector_size__(4*sizeof(float))))  float  float4_t;
__device__ __forceinline__ float4_t mma_bf(short4_t a, short4_t b, float4_t c){
    return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, c, 0, 0, 0);
}
__device__ __forceinline__ short bfb(bf16 x){ short s; __builtin_memcpy(&s,&x,2); return s; }

// One 64-lane wavefront per (query-block qb, head h, batch b). BQ=BK=16.
template<int D>
__global__ void attend_ker(const bf16* Q,const bf16* K,const bf16* V,bf16* O,float* Lv,int N){
    constexpr int BQ=16, BK=16;
    const int qb=blockIdx.x, h=blockIdx.y, b=blockIdx.z, l=threadIdx.x;
    const int hk=h/(ATTN_H/ATTN_H_KV), q0=qb*BQ, DT=D/16;
    const float scale=1.0f/sqrtf((float)D);
    const int lo=l>>4, li=l&15;                                // (l/16, l%16)
    auto Qi=[&](int qi,int d){return (size_t)(((b*N+q0+qi)*ATTN_H+h)*D+d);};
    auto Ki=[&](int kj,int d){return (size_t)(((b*N+kj)*ATTN_H_KV+hk)*D+d);};
    // preload Q fragments: for S=Q@K^T, a[v]=Q[qi=li][k=s*16+lo*4+v]
    short4_t Qf[ATTN_D/16];
    #pragma unroll
    for(int s=0;s<DT;s++){ const int d0=s*16+lo*4; short4_t f;
        #pragma unroll
        for(int v=0;v<4;v++) ((short*)&f)[v]=bfb(Q[Qi(li,d0+v)]); Qf[s]=f; }
    float4_t Oacc[ATTN_D/16];
    #pragma unroll
    for(int s=0;s<DT;s++) Oacc[s]=float4_t{0,0,0,0};
    __shared__ float sS[BQ][BK];
    __shared__ bf16  sP[BQ][BK];
    __shared__ float sm[BQ], sl[BQ], sc[BQ];
    if(l<BQ){ sm[l]=-3.0e38f; sl[l]=0.0f; }
    __syncthreads();

    const int kmax = ATTN_CAUSAL ? (q0+BQ) : N;                // blocks fully past the last query are skipped
    for(int k0=0;k0<kmax;k0+=BK){
        // S[16x16] = Q @ K^T, accumulate over DT head-dim steps
        float4_t S=float4_t{0,0,0,0};
        #pragma unroll
        for(int s=0;s<DT;s++){ const int d0=s*16+lo*4; short4_t Kf;
            #pragma unroll
            for(int v=0;v<4;v++) ((short*)&Kf)[v]=bfb(K[Ki(k0+li,d0+v)]);
            S=mma_bf(Qf[s],Kf,S); }
        // S layout: lane l holds S[qi=4*lo+v][kj=li]. Write scaled S to LDS.
        #pragma unroll
        for(int v=0;v<4;v++) sS[4*lo+v][li]=S[v]*scale;
        __syncthreads();
        // softmax: lanes 0..15 own a query row (online running max/sum)
        if(l<BQ){ const int qi=l;
            float mp=sm[qi], mx=mp;
            #pragma unroll
            for(int j=0;j<BK;j++){ int kk=k0+j; if(!ATTN_CAUSAL || kk<=q0+qi) mx=fmaxf(mx,sS[qi][j]); }
            const float corr=__expf(mp-mx); float ls=sl[qi]*corr;
            #pragma unroll
            for(int j=0;j<BK;j++){ int kk=k0+j; float p=(!ATTN_CAUSAL||kk<=q0+qi)?__expf(sS[qi][j]-mx):0.0f;
                sP[qi][j]=__float2bfloat16(p); ls+=p; }
            sm[qi]=mx; sl[qi]=ls; sc[qi]=corr; }
        __syncthreads();
        // rescale O accumulator by the per-row correction (row qi=4*lo+v)
        #pragma unroll
        for(int s=0;s<DT;s++){ float4_t o=Oacc[s];
            #pragma unroll
            for(int v=0;v<4;v++) o[v]*=sc[4*lo+v]; Oacc[s]=o; }
        // O += P @ V : a[v]=P[qi=li][kj=lo*4+v], b[v]=V[kj=lo*4+v][d=s*16+li]
        short4_t Pf;
        #pragma unroll
        for(int v=0;v<4;v++) ((short*)&Pf)[v]=bfb(sP[li][lo*4+v]);
        #pragma unroll
        for(int s=0;s<DT;s++){ const int d=s*16+li; short4_t Vf;
            #pragma unroll
            for(int v=0;v<4;v++) ((short*)&Vf)[v]=bfb(V[Ki(k0+lo*4+v,d)]);
            Oacc[s]=mma_bf(Pf,Vf,Oacc[s]); }
        __syncthreads();
    }
    // epilogue: O[qi][d] /= l_qi. Oacc[s]: O[qi=4*lo+v][d=s*16+li]
    #pragma unroll
    for(int s=0;s<DT;s++){ const int d=s*16+li;
        #pragma unroll
        for(int v=0;v<4;v++){ const int qi=4*lo+v; const float inv=sl[qi]>0.0f?1.0f/sl[qi]:0.0f;
            O[Qi(qi,d)]=__float2bfloat16(Oacc[s][v]*inv); } }
    if(l<BQ){ const int qi=l; Lv[(size_t)(b*ATTN_H+h)*N+q0+qi]=sm[qi]+__logf(sl[qi]>0.0f?sl[qi]:1.0f); }
}

