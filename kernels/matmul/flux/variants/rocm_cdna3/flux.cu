/**
 * @file
 * @brief CDNA3 (gfx942) flux kernels — dense bf16 matmul with fused epilogues,
 * ported from QuixiCore-CUDA/kernels/flux (ThunderKittens). MFMA
 * v_mfma_f32_16x16x16_f16 on a 64-wide wavefront + standalone fp64 oracle.
 *
 *   flux_gelu : Y = gelu_tanh(A[M,K] @ B[K,N] + bias[N])
 *   flux_gate : Y = (A[M,K] @ B[K,N]) * gate[M,N]     (elementwise gate)
 *
 * One wavefront per 16x16 output tile. A,B,gate row-major; bias per output col.
 *   hipcc -std=c++17 -O3 --offload-arch=gfx942 flux.cu -o flux.out
 */
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <random>
#include <cmath>
#define CK(x) do{hipError_t e=(x); if(e!=hipSuccess){printf("HIP err %s @%d\n",hipGetErrorString(e),__LINE__);exit(1);} }while(0)

typedef __attribute__((__vector_size__(4*sizeof(__fp16)))) __fp16 half4_t;
typedef __attribute__((__vector_size__(4*sizeof(float))))  float  float4_t;

__device__ __forceinline__ float gelu_tanh(float x){
    const float k=0.7978845608028654f, a=0.044715f;
    return 0.5f*x*(1.0f+tanhf(k*(x+a*x*x*x)));
}

// Y = A[M,K] @ B[K,N]; A,B row-major, half. MFMA layout: lane l, out col n=n0+l%16,
// rows m0+4*(l/16)+{0..3}. a = 4 contiguous K of one A row (half4 reinterpret);
// b = 4 K of one B column (strided by N -> individual loads).
template<int EPILOGUE>   // 0 = gelu+bias, 1 = gate
__global__ void flux(float* Y, const __half* A, const __half* B,
                     const float* bias, const __half* gate, int M, int N, int K){
    const int n0=blockIdx.x*16, m0=blockIdx.y*16, l=threadIdx.x&63;
    const int n=n0+(l&15), mlo=m0+(l>>4)*4;
    float4_t acc={0,0,0,0};
    for(int k0=0;k0<K;k0+=16){
        const int mrow=m0+(l&15), kk=k0+(l>>4)*4;
        half4_t a=*reinterpret_cast<const half4_t*>(A+(size_t)mrow*K+kk);
        half4_t b;
        #pragma unroll
        for(int v=0;v<4;v++) b[v]=(__fp16)__half2float(B[(size_t)(kk+v)*N+n]);
        acc=__builtin_amdgcn_mfma_f32_16x16x16f16(a,b,acc,0,0,0);
    }
    #pragma unroll
    for(int v=0;v<4;v++){ const int m=mlo+v; if(m>=M) continue;
        float y=acc[v];
        if(EPILOGUE==0) y=gelu_tanh(y+bias[n]);
        else            y=y*__half2float(gate[(size_t)m*N+n]);
        Y[(size_t)m*N+n]=y;
    }
}

static uint16_t f2h(float f){ __half h=__float2half(f); uint16_t u; std::memcpy(&u,&h,2); return u; }
int main(){
    const int M=64,N=128,K=256;
    std::mt19937 rng(3); std::normal_distribution<float> nd(0,0.5f);
    auto genh=[&](int n){std::vector<__half> v(n);for(auto&x:v)x=__float2half(nd(rng));return v;};
    auto A=genh(M*K),B=genh(K*N),G=genh(M*N); std::vector<float> bias(N); for(auto&x:bias)x=nd(rng);
    auto af=[&](int m,int k){return (double)__half2float(A[m*K+k]);};
    auto bf=[&](int k,int n){return (double)__half2float(B[k*N+n]);};
    // fp64 refs
    std::vector<double> rge(M*N),rga(M*N);
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ double s=0; for(int k=0;k<K;k++) s+=af(m,k)*bf(k,n);
        double x=s+bias[n]; const double kk=0.7978845608028654,a=0.044715;
        rge[m*N+n]=0.5*x*(1.0+std::tanh(kk*(x+a*x*x*x)));
        rga[m*N+n]=s*(double)__half2float(G[m*N+n]); }
    __half *dA,*dB,*dG; float *dbias,*dYe,*dYg;
    auto up=[&](std::vector<__half>&h){__half*d;CK(hipMalloc(&d,h.size()*2));CK(hipMemcpy(d,h.data(),h.size()*2,hipMemcpyHostToDevice));return d;};
    dA=up(A);dB=up(B);dG=up(G);
    CK(hipMalloc(&dbias,N*4));CK(hipMemcpy(dbias,bias.data(),N*4,hipMemcpyHostToDevice));
    CK(hipMalloc(&dYe,M*N*4));CK(hipMalloc(&dYg,M*N*4));
    dim3 grid(N/16,(M+15)/16);
    flux<0><<<grid,64>>>(dYe,dA,dB,dbias,nullptr,M,N,K);
    flux<1><<<grid,64>>>(dYg,dA,dB,nullptr,dG,M,N,K);
    CK(hipDeviceSynchronize()); CK(hipGetLastError());
    std::vector<float> ye(M*N),yg(M*N);
    CK(hipMemcpy(ye.data(),dYe,M*N*4,hipMemcpyDeviceToHost)); CK(hipMemcpy(yg.data(),dYg,M*N*4,hipMemcpyDeviceToHost));
    auto rep=[&](const char*nm,std::vector<float>&g,std::vector<double>&r){double sa=0,sr=0,mx=0;
        for(size_t k=0;k<r.size();k++){double d=std::abs(g[k]-r[k]);sa+=d;sr+=std::abs(r[k]);mx=std::max(mx,d);}
        double rel=sa/std::max(sr,1e-30);printf("%-10s mean rel %.4f%% max %.4g (%s)\n",nm,100*rel,mx,rel<0.01?"PASS":"FAIL");return rel<0.01;};
    bool ok=true; ok&=rep("flux_gelu",ye,rge); ok&=rep("flux_gate",yg,rga);
    printf("%s\n",ok?"ALL PASS":"FAILED"); return ok?0:1;
}
