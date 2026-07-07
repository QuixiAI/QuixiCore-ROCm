/**
 * @file
 * @brief qgemm perf A/B: baseline (1 N-tile/wavefront) vs qgemm_wide<NT>
 * (NT N-tiles/wavefront, X fragment loaded once and reused across NT
 * W-fragments -> X traffic /NT, MFMA:load ratio *NT). Uses fp16_raw to isolate
 * the tiling win from dequant (the win is orthogonal to the quant format).
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
    double tb=med([&]{qgemm_base<fp16_raw><<<gB,64>>>(dY0,dX,dW,M,N,K);});
    double tw=med([&]{qgemm_wide<fp16_raw,NT><<<gW,64>>>(dY1,dX,dW,M,N,K);});
    double flop=2.0*M*N*(double)K;
    printf("base (NT=1): %.3f ms  %.1f TFLOP/s\n",tb,flop/(tb*1e-3)/1e12);
    printf("wide (NT=%d): %.3f ms  %.1f TFLOP/s\n",NT,tw,flop/(tw*1e-3)/1e12);
    printf("speedup: %.2fx  (base vs wide max abs diff %.3g)\n",tb/tw,mx);
    return 0;
}
