/**
 * @file
 * @brief qgemm perf A/B: baseline (1 N-tile/wavefront) vs qgemm_wide<NT>
 * (NT N-tiles/wavefront, X fragment loaded once and reused across NT
 * W-fragments -> X traffic /NT, MFMA:load ratio *NT), an LDS-staged
 * qgemm_wide_lds<NT> candidate, and a multi-wave CTA LDS candidate that reuses
 * W across several M-subtiles. Uses fp16_raw to isolate tiling/coalescing wins
 * from dequant (the wins are orthogonal to the quant format).
 * Correctness: wide vs base compared bitwise here; qgemm.cu (golden) covers formats.
 *   hipcc -std=c++17 -O3 --offload-arch=gfx942 qgemm_bench.cu -o qgemm_bench.out
 */
#include "tm_qmm_mfma.cuh"
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
using namespace tmq;
#define CK(x) do{hipError_t e=(x); if(e!=hipSuccess){printf("HIP %s @%d\n",hipGetErrorString(e),__LINE__);exit(1);} }while(0)

// baseline: 1 wavefront per 16x16 output tile (== shipped qgemm)
template<typename FMT>
__global__ void qgemm_base(float* Y,const __half* X,const uint8_t* Wq,int M,int N,int K){
    const int n0=blockIdx.x*16, m0=blockIdx.y*16, bpr=K/FMT::block_k;
    float4_t acc={0,0,0,0};
    for(int k0=0;k0<K;k0+=16){ half4_t a=load_xfrag(X,K,m0,k0); half4_t b=load_wfrag<FMT>(Wq,bpr,n0,k0);
        acc=mma_16x16x16(a,b,acc); }
    const int l=threadIdx.x&63, n=n0+(l&15), mrow=m0+(l>>4)*4;
    #pragma unroll
    for(int v=0;v<4;v++) if(mrow+v<M) Y[size_t(mrow+v)*N+n]=acc[v];
}
// wide: NT N-tiles per wavefront; X fragment loaded once, reused across NT.
template<typename FMT,int NT>
__global__ void qgemm_wide(float* Y,const __half* X,const uint8_t* Wq,int M,int N,int K){
    const int n0=blockIdx.x*(16*NT), m0=blockIdx.y*16, bpr=K/FMT::block_k;
    float4_t acc[NT];
    #pragma unroll
    for(int nt=0;nt<NT;nt++) acc[nt]=float4_t{0,0,0,0};
    for(int k0=0;k0<K;k0+=16){ half4_t a=load_xfrag(X,K,m0,k0);
        #pragma unroll
        for(int nt=0;nt<NT;nt++){ half4_t b=load_wfrag<FMT>(Wq,bpr,n0+nt*16,k0);
            acc[nt]=mma_16x16x16(a,b,acc[nt]); } }
    const int l=threadIdx.x&63, ln=l&15, mrow=m0+(l>>4)*4;
    #pragma unroll
    for(int nt=0;nt<NT;nt++){ int n=n0+nt*16+ln;
        #pragma unroll
        for(int v=0;v<4;v++) if(mrow+v<M) Y[size_t(mrow+v)*N+n]=acc[nt][v]; }
}
// 2D register tile: MT M-tiles x NT N-tiles per wavefront; reuse X across N AND W
// across M -> MT*NT MFMAs per (MT+NT) fragment loads (better ratio than 1D wide).
template<typename FMT,int MT,int NT>
__global__ void qgemm_wide2d(float* Y,const __half* X,const uint8_t* Wq,int M,int N,int K){
    const int n0=blockIdx.x*(16*NT), m0=blockIdx.y*(16*MT), bpr=K/FMT::block_k;
    float4_t acc[MT][NT];
    #pragma unroll
    for(int mt=0;mt<MT;mt++) for(int nt=0;nt<NT;nt++) acc[mt][nt]=float4_t{0,0,0,0};
    for(int k0=0;k0<K;k0+=16){
        half4_t a[MT], b[NT];
        #pragma unroll
        for(int mt=0;mt<MT;mt++) a[mt]=load_xfrag(X,K,m0+mt*16,k0);
        #pragma unroll
        for(int nt=0;nt<NT;nt++) b[nt]=load_wfrag<FMT>(Wq,bpr,n0+nt*16,k0);
        #pragma unroll
        for(int mt=0;mt<MT;mt++) for(int nt=0;nt<NT;nt++) acc[mt][nt]=mma_16x16x16(a[mt],b[nt],acc[mt][nt]);
    }
    const int l=threadIdx.x&63, ln=l&15, mrb=(l>>4)*4;
    #pragma unroll
    for(int mt=0;mt<MT;mt++) for(int nt=0;nt<NT;nt++){ int n=n0+nt*16+ln, mrow=m0+mt*16+mrb;
        #pragma unroll
        for(int v=0;v<4;v++) if(mrow+v<M) Y[size_t(mrow+v)*N+n]=acc[mt][nt][v]; }
}
template<int NT>
__device__ __forceinline__ void stage_raw_wide_tile(__half* sX,__half* sW,const __half* X,const __half* W,
                                                    int M,int K,int m0,int n0,int k0){
    const int tid=threadIdx.x&63;
    for(int idx=tid;idx<16*16;idx+=64){
        const int m=idx/16, k=idx&15;
        sX[idx]=(m0+m<M) ? X[size_t(m0+m)*K + k0+k] : __float2half(0.0f);
    }
    for(int idx=tid;idx<NT*16*16;idx+=64){
        const int nt=idx/(16*16), r=idx-nt*16*16, n=r/16, k=r&15;
        sW[idx]=W[size_t(n0+nt*16+n)*K + k0+k];
    }
}
__device__ __forceinline__ half4_t load_xfrag_smem(const __half* sX){
    const int l=threadIdx.x&63, m=l&15, k=(l>>4)*4;
    return *reinterpret_cast<const half4_t*>(sX + m*16 + k);
}
__device__ __forceinline__ half4_t load_wfrag_smem(const __half* sW,int nt){
    const int l=threadIdx.x&63, n=l&15, k=(l>>4)*4;
    return *reinterpret_cast<const half4_t*>(sW + nt*16*16 + n*16 + k);
}
// LDS-staged wide candidate for fp16_raw: coalesced global loads fill X and W
// tiles, then lanes read the same MFMA fragments from LDS. The ping-pong buffers
// let the next K tile's global loads issue before the current tile's MFMA work.
template<int NT>
__global__ void qgemm_wide_lds(float* Y,const __half* X,const uint8_t* Wq,int M,int N,int K){
    const int n0=blockIdx.x*(16*NT), m0=blockIdx.y*16;
    const __half* W=reinterpret_cast<const __half*>(Wq);
    __shared__ __half sX[2][16*16];
    __shared__ __half sW[2][NT*16*16];
    float4_t acc[NT];
    #pragma unroll
    for(int nt=0;nt<NT;nt++) acc[nt]=float4_t{0,0,0,0};

    stage_raw_wide_tile<NT>(sX[0],sW[0],X,W,M,K,m0,n0,0);
    __syncthreads();
    for(int k0=0;k0<K;k0+=16){
        const int buf=(k0/16)&1, nxt=buf^1;
        half4_t a=load_xfrag_smem(sX[buf]);
        half4_t b[NT];
        #pragma unroll
        for(int nt=0;nt<NT;nt++) b[nt]=load_wfrag_smem(sW[buf],nt);
        if(k0+16<K) stage_raw_wide_tile<NT>(sX[nxt],sW[nxt],X,W,M,K,m0,n0,k0+16);
        #pragma unroll
        for(int nt=0;nt<NT;nt++) acc[nt]=mma_16x16x16(a,b[nt],acc[nt]);
        __syncthreads();
    }
    const int l=threadIdx.x&63, ln=l&15, mrow=m0+(l>>4)*4;
    #pragma unroll
    for(int nt=0;nt<NT;nt++){ int n=n0+nt*16+ln;
        #pragma unroll
        for(int v=0;v<4;v++) if(mrow+v<M) Y[size_t(mrow+v)*N+n]=acc[nt][v]; }
}
template<int MT,int NT>
__device__ __forceinline__ void stage_raw_cta_tile(__half* sX,__half* sW,const __half* X,const __half* W,
                                                   int M,int K,int m0,int n0,int k0){
    const int tid=threadIdx.x;
    for(int idx=tid;idx<MT*16*16;idx+=blockDim.x){
        const int mt=idx/(16*16), r=idx-mt*16*16, m=r/16, k=r&15;
        const int gm=m0+mt*16+m;
        sX[idx]=(gm<M) ? X[size_t(gm)*K + k0+k] : __float2half(0.0f);
    }
    for(int idx=tid;idx<NT*16*16;idx+=blockDim.x){
        const int nt=idx/(16*16), r=idx-nt*16*16, n=r/16, k=r&15;
        sW[idx]=W[size_t(n0+nt*16+n)*K + k0+k];
    }
}
__device__ __forceinline__ half4_t load_xfrag_cta_smem(const __half* sX,int mt){
    const int l=threadIdx.x&63, m=l&15, k=(l>>4)*4;
    return *reinterpret_cast<const half4_t*>(sX + mt*16*16 + m*16 + k);
}
// Multi-wave CTA candidate for fp16_raw: MT wavefronts cooperate in one block,
// each owning one 16-row M tile while all waves share the NT 16-column W tiles.
// This tests whether LDS staging pays once W global traffic is amortized over MT.
template<int MT,int NT>
__global__ void qgemm_cta_lds(float* Y,const __half* X,const uint8_t* Wq,int M,int N,int K){
    const int n0=blockIdx.x*(16*NT), m0=blockIdx.y*(16*MT);
    const int wave=threadIdx.x>>6;
    const __half* W=reinterpret_cast<const __half*>(Wq);
    __shared__ __half sX[2][MT*16*16];
    __shared__ __half sW[2][NT*16*16];
    float4_t acc[NT];
    #pragma unroll
    for(int nt=0;nt<NT;nt++) acc[nt]=float4_t{0,0,0,0};

    stage_raw_cta_tile<MT,NT>(sX[0],sW[0],X,W,M,K,m0,n0,0);
    __syncthreads();
    for(int k0=0;k0<K;k0+=16){
        const int buf=(k0/16)&1, nxt=buf^1;
        half4_t a=load_xfrag_cta_smem(sX[buf],wave);
        half4_t b[NT];
        #pragma unroll
        for(int nt=0;nt<NT;nt++) b[nt]=load_wfrag_smem(sW[buf],nt);
        if(k0+16<K) stage_raw_cta_tile<MT,NT>(sX[nxt],sW[nxt],X,W,M,K,m0,n0,k0+16);
        #pragma unroll
        for(int nt=0;nt<NT;nt++) acc[nt]=mma_16x16x16(a,b[nt],acc[nt]);
        __syncthreads();
    }
    const int l=threadIdx.x&63, ln=l&15, mrow=m0+wave*16+(l>>4)*4;
    #pragma unroll
    for(int nt=0;nt<NT;nt++){ int n=n0+nt*16+ln;
        #pragma unroll
        for(int v=0;v<4;v++) if(mrow+v<M) Y[size_t(mrow+v)*N+n]=acc[nt][v]; }
}
static uint16_t f2h(float f){__half h=__float2half(f);uint16_t u;__builtin_memcpy(&u,&h,2);return u;}
template<class L> static double med(L fn,int w=10,int it=50){ for(int i=0;i<w;i++)fn(); hipDeviceSynchronize();
    std::vector<float> t(it); hipEvent_t a,b; hipEventCreate(&a);hipEventCreate(&b);
    for(int i=0;i<it;i++){hipEventRecord(a);fn();hipEventRecord(b);hipEventSynchronize(b);hipEventElapsedTime(&t[i],a,b);}
    std::sort(t.begin(),t.end()); return t[it/2]; }
int main(int argc,char**argv){
    int M=argc>1?atoi(argv[1]):64, N=argc>2?atoi(argv[2]):4096, K=argc>3?atoi(argv[3]):4096;
    const int NT=4;                                   // 16*NT must divide N
    printf("== qgemm A/B (fp16_raw)  M=%d N=%d K=%d  NT=%d\n",M,N,K,NT);
    std::mt19937 rng(2); std::normal_distribution<float> nd(0,0.3f);
    std::vector<__half> X(M*K); for(auto&x:X)x=__float2half(nd(rng));
    // fp16_raw W is effectively N*K row-major fp16 (block_k=16, 16 halfs/block)
    std::vector<__half> W(N*K); for(auto&x:W)x=__float2half(nd(rng));
    __half *dX; uint8_t *dW; float *dY0,*dY1;
    CK(hipMalloc(&dX,X.size()*2)); CK(hipMemcpy(dX,X.data(),X.size()*2,hipMemcpyHostToDevice));
    CK(hipMalloc(&dW,W.size()*2)); CK(hipMemcpy(dW,W.data(),W.size()*2,hipMemcpyHostToDevice));
    CK(hipMalloc(&dY0,(size_t)M*N*4)); CK(hipMalloc(&dY1,(size_t)M*N*4));
    dim3 gB(N/16,(M+15)/16), gW(N/(16*NT),(M+15)/16);
    qgemm_base<fp16_raw><<<gB,64>>>(dY0,dX,dW,M,N,K);
    qgemm_wide<fp16_raw,NT><<<gW,64>>>(dY1,dX,dW,M,N,K);
    CK(hipDeviceSynchronize()); CK(hipGetLastError());
    std::vector<float> y0(M*N),y1(M*N);
    CK(hipMemcpy(y0.data(),dY0,y0.size()*4,hipMemcpyDeviceToHost)); CK(hipMemcpy(y1.data(),dY1,y1.size()*4,hipMemcpyDeviceToHost));
    double mx=0; for(size_t k=0;k<y0.size();k++) mx=std::max(mx,(double)std::abs(y0[k]-y1[k]));
    // 2D tile 2x2 (needs 32|M and 32|N; M may be <32 at decode -> guard)
    const int MT=2, NT2=2; bool can2d = (N%(16*NT2)==0) && (M%(16*MT)==0);
    dim3 g2((unsigned)(N/(16*NT2)),(unsigned)((M+16*MT-1)/(16*MT)));
    float* dY2=nullptr; double mx2=0;
    if(can2d){ CK(hipMalloc(&dY2,(size_t)M*N*4)); qgemm_wide2d<fp16_raw,MT,NT2><<<g2,64>>>(dY2,dX,dW,M,N,K);
        CK(hipDeviceSynchronize()); std::vector<float> y2(M*N); CK(hipMemcpy(y2.data(),dY2,y2.size()*4,hipMemcpyDeviceToHost));
        for(size_t k=0;k<y0.size();k++) mx2=std::max(mx2,(double)std::abs(y0[k]-y2[k])); }
    float* dY3=nullptr; double mx3=0; bool canlds = (N%(16*NT)==0);
    if(canlds){ CK(hipMalloc(&dY3,(size_t)M*N*4)); qgemm_wide_lds<NT><<<gW,64>>>(dY3,dX,dW,M,N,K);
        CK(hipDeviceSynchronize()); std::vector<float> y3(M*N); CK(hipMemcpy(y3.data(),dY3,y3.size()*4,hipMemcpyDeviceToHost));
        for(size_t k=0;k<y0.size();k++) mx3=std::max(mx3,(double)std::abs(y0[k]-y3[k])); }
    const int MTCTA=4, NTCTA=4; bool cancta = (N%(16*NTCTA)==0);
    dim3 gC((unsigned)(N/(16*NTCTA)),(unsigned)((M+16*MTCTA-1)/(16*MTCTA)));
    float* dY4=nullptr; double mx4=0;
    if(cancta){ CK(hipMalloc(&dY4,(size_t)M*N*4)); qgemm_cta_lds<MTCTA,NTCTA><<<gC,64*MTCTA>>>(dY4,dX,dW,M,N,K);
        CK(hipDeviceSynchronize()); std::vector<float> y4(M*N); CK(hipMemcpy(y4.data(),dY4,y4.size()*4,hipMemcpyDeviceToHost));
        for(size_t k=0;k<y0.size();k++) mx4=std::max(mx4,(double)std::abs(y0[k]-y4[k])); }
    double tb=med([&]{qgemm_base<fp16_raw><<<gB,64>>>(dY0,dX,dW,M,N,K);});
    double tw=med([&]{qgemm_wide<fp16_raw,NT><<<gW,64>>>(dY1,dX,dW,M,N,K);});
    double flop=2.0*M*N*(double)K;
    printf("base   (NT=1)   : %.3f ms  %.1f TFLOP/s\n",tb,flop/(tb*1e-3)/1e12);
    printf("wide   (NT=%d)   : %.3f ms  %.1f TFLOP/s  (%.2fx, diff %.3g)\n",NT,tw,flop/(tw*1e-3)/1e12,tb/tw,mx);
    if(can2d){ double t2=med([&]{qgemm_wide2d<fp16_raw,MT,NT2><<<g2,64>>>(dY2,dX,dW,M,N,K);});
        printf("wide2d (%dx%d)  : %.3f ms  %.1f TFLOP/s  (%.2fx, diff %.3g)\n",MT,NT2,t2,flop/(t2*1e-3)/1e12,tb/t2,mx2); }
    if(canlds){ double tl=med([&]{qgemm_wide_lds<NT><<<gW,64>>>(dY3,dX,dW,M,N,K);});
        printf("wideLDS(NT=%d)  : %.3f ms  %.1f TFLOP/s  (%.2fx vs base, %.2fx vs wide, diff %.3g)\n",
               NT,tl,flop/(tl*1e-3)/1e12,tb/tl,tw/tl,mx3); }
    if(cancta){ double tc=med([&]{qgemm_cta_lds<MTCTA,NTCTA><<<gC,64*MTCTA>>>(dY4,dX,dW,M,N,K);});
        printf("ctaLDS (%dx%d)  : %.3f ms  %.1f TFLOP/s  (%.2fx vs base, %.2fx vs wide, diff %.3g)\n",
               MTCTA,NTCTA,tc,flop/(tc*1e-3)/1e12,tb/tc,tw/tc,mx4); }
    return 0;
}
