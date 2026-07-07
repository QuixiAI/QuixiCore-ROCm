/**
 * @file
 * @brief flux GEMM perf A/B: baseline (column-strided B load) vs LDS-staged B
 * (coalesced load into LDS, read in MFMA fragment layout). Y=A@B, one 64-lane
 * wavefront per 16x16 output tile. Correctness of both is spot-checked vs the
 * other; flux.cu (fp64 oracle) covers full correctness.
 *   hipcc -std=c++17 -O3 --offload-arch=gfx942 flux_bench.cu -o flux_bench.out
 */
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#define CK(x) do{hipError_t e=(x); if(e!=hipSuccess){printf("HIP %s @%d\n",hipGetErrorString(e),__LINE__);exit(1);} }while(0)
typedef __attribute__((__vector_size__(4*sizeof(__fp16)))) __fp16 half4_t;
typedef __attribute__((__vector_size__(4*sizeof(float))))  float  float4_t;

// baseline: B loaded column-strided (4 scalar loads/lane/k-step)
__global__ void flux_base(float* Y,const __half* A,const __half* B,int M,int N,int K){
    const int n0=blockIdx.x*16,m0=blockIdx.y*16,l=threadIdx.x&63;
    const int n=n0+(l&15),mlo=m0+(l>>4)*4;
    float4_t acc={0,0,0,0};
    for(int k0=0;k0<K;k0+=16){ const int mrow=m0+(l&15),kk=k0+(l>>4)*4;
        half4_t a=*reinterpret_cast<const half4_t*>(A+(size_t)mrow*K+kk);
        half4_t b;
        #pragma unroll
        for(int v=0;v<4;v++) b[v]=(__fp16)__half2float(B[(size_t)(kk+v)*N+n]);
        acc=__builtin_amdgcn_mfma_f32_16x16x16f16(a,b,acc,0,0,0); }
    #pragma unroll
    for(int v=0;v<4;v++){ int m=mlo+v; if(m<M) Y[(size_t)m*N+n]=acc[v]; }
}
// LDS-staged B: coalesced global load of a [16 K x 16 N] B tile into LDS, then
// read in the MFMA b-fragment layout (b[v]=B[k=4*(l/16)+v][n=l%16]).
__global__ void flux_lds(float* Y,const __half* A,const __half* B,int M,int N,int K){
    const int n0=blockIdx.x*16,m0=blockIdx.y*16,l=threadIdx.x&63;
    const int n=n0+(l&15),mlo=m0+(l>>4)*4;
    __shared__ __fp16 sB[16][16];
    float4_t acc={0,0,0,0};
    for(int k0=0;k0<K;k0+=16){ const int mrow=m0+(l&15),kk=k0+(l>>4)*4;
        half4_t a=*reinterpret_cast<const half4_t*>(A+(size_t)mrow*K+kk);
        // coalesced: 64 lanes load 256 elems (4/lane) of B[k0..+16][n0..+16]
        #pragma unroll
        for(int i=0;i<4;i++){ int f=l*4+i, kr=f>>4, nc=f&15; sB[kr][nc]=(__fp16)__half2float(B[(size_t)(k0+kr)*N+n0+nc]); }
        __syncthreads();
        half4_t b;
        #pragma unroll
        for(int v=0;v<4;v++) b[v]=sB[(l>>4)*4+v][l&15];
        acc=__builtin_amdgcn_mfma_f32_16x16x16f16(a,b,acc,0,0,0);
        __syncthreads(); }
    #pragma unroll
    for(int v=0;v<4;v++){ int m=mlo+v; if(m<M) Y[(size_t)m*N+n]=acc[v]; }
}
static uint16_t f2h(float f){__half h=__float2half(f);uint16_t u;__builtin_memcpy(&u,&h,2);return u;}
template<class L> static double med(L fn,int w=10,int it=50){ for(int i=0;i<w;i++)fn(); hipDeviceSynchronize();
    std::vector<float> t(it); hipEvent_t a,b; hipEventCreate(&a);hipEventCreate(&b);
    for(int i=0;i<it;i++){hipEventRecord(a);fn();hipEventRecord(b);hipEventSynchronize(b);hipEventElapsedTime(&t[i],a,b);}
    std::sort(t.begin(),t.end()); return t[it/2]; }
int main(int argc,char**argv){
    int M=argc>1?atoi(argv[1]):2048, N=argc>2?atoi(argv[2]):2048, K=argc>3?atoi(argv[3]):2048;
    printf("== flux GEMM A/B  M=%d N=%d K=%d\n",M,N,K);
    std::mt19937 rng(1); std::normal_distribution<float> nd(0,0.3f);
    std::vector<__half> A(M*K),B(K*N); for(auto&x:A)x=__float2half(nd(rng)); for(auto&x:B)x=__float2half(nd(rng));
    __half *dA,*dB; float *dY0,*dY1;
    CK(hipMalloc(&dA,A.size()*2));CK(hipMemcpy(dA,A.data(),A.size()*2,hipMemcpyHostToDevice));
    CK(hipMalloc(&dB,B.size()*2));CK(hipMemcpy(dB,B.data(),B.size()*2,hipMemcpyHostToDevice));
    CK(hipMalloc(&dY0,(size_t)M*N*4));CK(hipMalloc(&dY1,(size_t)M*N*4));
    dim3 grid(N/16,(M+15)/16);
    flux_base<<<grid,64>>>(dY0,dA,dB,M,N,K); flux_lds<<<grid,64>>>(dY1,dA,dB,M,N,K);
    CK(hipDeviceSynchronize());CK(hipGetLastError());
    std::vector<float> y0(M*N),y1(M*N);
    CK(hipMemcpy(y0.data(),dY0,y0.size()*4,hipMemcpyDeviceToHost));CK(hipMemcpy(y1.data(),dY1,y1.size()*4,hipMemcpyDeviceToHost));
    double mx=0; for(size_t k=0;k<y0.size();k++) mx=std::max(mx,(double)std::abs(y0[k]-y1[k]));
    double tb=med([&]{flux_base<<<grid,64>>>(dY0,dA,dB,M,N,K);});
    double tl=med([&]{flux_lds<<<grid,64>>>(dY1,dA,dB,M,N,K);});
    double flop=2.0*M*N*(double)K;
    printf("base(strided-B): %.3f ms  %.1f TFLOP/s\n",tb,flop/(tb*1e-3)/1e12);
    printf("lds  (coalesced): %.3f ms  %.1f TFLOP/s\n",tl,flop/(tl*1e-3)/1e12);
    printf("speedup: %.2fx  (base vs lds max abs diff %.3g)\n",tb/tl,mx);
    return 0;
}
