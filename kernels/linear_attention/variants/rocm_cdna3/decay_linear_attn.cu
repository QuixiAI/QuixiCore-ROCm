/**
 * @file
 * @brief CDNA3 (gfx942) decay linear attention, port of the kittens
 * QuixiCore-CUDA/kernels/linear_attention/linear_attention.cu. Matches its
 * gentests.py: causal linear attention with per-head exponential decay:
 *   o[i] = sum_{j<=i} (Q_i . K_j) * exp(-slope_h*(i-j)) * V[j]
 * (D=D_VO=128). One 64-wide wavefront per (query,head,batch); lanes own DV;
 * the D-dot is a wavefront reduction. Standalone fp64 oracle.
 * NOTE: complements the already-ported TM linear attention (lin_attn_tm) which
 * covers the non-decay / chunked forms.
 *   hipcc -std=c++17 -O3 --offload-arch=gfx942 decay_linear_attn.cu
 */
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <random>
#include <cmath>
#ifndef B_
#define B_ 1
#endif
#ifndef H_
#define H_ 2
#endif
#ifndef N_
#define N_ 256
#endif
#define D_ 128
#define CK(x) do{hipError_t e=(x); if(e!=hipSuccess){printf("HIP err %s @%d\n",hipGetErrorString(e),__LINE__);exit(1);} }while(0)
using bf16 = __hip_bfloat16;
__device__ __forceinline__ float wsum(float v){ for(int o=32;o>0;o>>=1) v+=__shfl_xor(v,o); return v; }

__global__ void dla_ker(const bf16* Q,const bf16* K,const bf16* V,const float* slopes,bf16* O,int N){
    constexpr int WF=64, EPL=D_/WF;
    const int i=blockIdx.x,h=blockIdx.y,b=blockIdx.z,lane=threadIdx.x;
    const float slope=slopes[h];
    auto idx=[&](int r,int e){return (size_t)(((b*N+r)*H_+h))*D_+lane+e*WF;};
    float qi[EPL],acc[EPL];
    #pragma unroll
    for(int e=0;e<EPL;e++){ qi[e]=__bfloat162float(Q[idx(i,e)]); acc[e]=0.0f; }
    for(int j=0;j<=i;j++){
        float p=0;
        #pragma unroll
        for(int e=0;e<EPL;e++) p+=qi[e]*__bfloat162float(K[idx(j,e)]);
        const float qk=wsum(p), decay=__expf(-slope*(float)(i-j)), w=qk*decay;
        #pragma unroll
        for(int e=0;e<EPL;e++) acc[e]+=w*__bfloat162float(V[idx(j,e)]);
    }
    #pragma unroll
    for(int e=0;e<EPL;e++) O[idx(i,e)]=__float2bfloat16(acc[e]);
}

static uint16_t f2bf(float f){uint32_t x;std::memcpy(&x,&f,4);uint32_t r=x+0x7FFFu+((x>>16)&1u);return uint16_t(r>>16);}
static float bf2f(uint16_t b){uint32_t x=uint32_t(b)<<16;float f;std::memcpy(&f,&x,4);return f;}
int main(){
    const int B=B_,H=H_,N=N_;
    std::mt19937 rng(9); std::normal_distribution<float> nd(0,1); std::uniform_real_distribution<float> ur(0,1);
    auto gen=[&](int n,float s){std::vector<uint16_t> v(n);for(auto&x:v)x=f2bf(nd(rng)*s);return v;};
    auto Q=gen(B*N*H*D_,1),K=gen(B*N*H*D_,1),V=gen(B*N*H*D_,0.2f);
    std::vector<float> slopes(H); for(int h=0;h<H;h++) slopes[h]=0.1f+ur(rng)*0.5f;
    auto xf=[&](const std::vector<uint16_t>&A,int b,int r,int h,int d){return (double)bf2f(A[(((size_t)b*N+r)*H+h)*D_+d]);};
    std::vector<double> ref(B*N*H*D_,0);
    for(int b=0;b<B;b++)for(int h=0;h<H;h++)for(int i=0;i<N;i++)
        for(int j=0;j<=i;j++){ double qk=0; for(int d=0;d<D_;d++)qk+=xf(Q,b,i,h,d)*xf(K,b,j,h,d);
            double w=qk*std::exp(-(double)slopes[h]*(i-j));
            for(int e=0;e<D_;e++) ref[(((size_t)b*N+i)*H+h)*D_+e]+=w*xf(V,b,j,h,e); }
    bf16 *dQ,*dK,*dV,*dO; float* dsl;
    auto up=[&](std::vector<uint16_t>&h){bf16*d;CK(hipMalloc(&d,h.size()*2));CK(hipMemcpy(d,h.data(),h.size()*2,hipMemcpyHostToDevice));return d;};
    dQ=up(Q);dK=up(K);dV=up(V); CK(hipMalloc(&dO,(size_t)B*N*H*D_*2));
    CK(hipMalloc(&dsl,H*4));CK(hipMemcpy(dsl,slopes.data(),H*4,hipMemcpyHostToDevice));
    dla_ker<<<dim3(N,H,B),64>>>(dQ,dK,dV,dsl,dO,N);
    CK(hipDeviceSynchronize()); CK(hipGetLastError());
    std::vector<uint16_t> O(B*N*H*D_); CK(hipMemcpy(O.data(),dO,O.size()*2,hipMemcpyDeviceToHost));
    double sa=0,sr=0,mx=0; for(size_t k=0;k<ref.size();k++){double d=std::abs(bf2f(O[k])-ref[k]);sa+=d;sr+=std::abs(ref[k]);mx=std::max(mx,d);}
    double rel=sa/std::max(sr,1e-30);
    printf("== decay linear attention  B=%d H=%d N=%d D=%d\n",B,H,N,D_);
    printf("o vs fp64 ref: mean rel %.4f%%  max abs %.4g  (%s)\n",100*rel,mx,rel<0.02?"PASS":"FAIL");
    return rel<0.02?0:1;
}
