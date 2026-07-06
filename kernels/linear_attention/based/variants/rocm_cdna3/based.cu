/**
 * @file
 * @brief CDNA3 (gfx942) "Based" 2nd-order Taylor-feature linear attention
 * (causal), ported from QuixiCore-CUDA/kernels/based (ThunderKittens).
 * Output (matching kernels/based gentests.py, D=D_QK=16, DV=64):
 *   o[n] = sum_{m<=n} V[m] * ( 1 + (Q_n.K_m)/sqrt(D) + (Q_n.K_m)^2/(2D) )
 * i.e. the Taylor-2 approx of exp attention: T0 (cumsum V) + T1/sqrt(D) +
 * T2/(2D) with T1=(Q.K), T2=(Q.K)^2.
 *
 * One 64-wide wavefront per (query n, head, batch); the 64 lanes own the DV=64
 * output/value dims (1 each); the D=16 query/key dot is computed per lane.
 * Standalone fp64 oracle. hipcc -std=c++17 -O3 --offload-arch=gfx942 based.cu
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
#define N_ 128
#endif
#define DQK 16
#define DV_ 64
#define CK(x) do{hipError_t e=(x); if(e!=hipSuccess){printf("HIP err %s @%d\n",hipGetErrorString(e),__LINE__);exit(1);} }while(0)
using bf16 = __hip_bfloat16;

// one wavefront per (n, h, b); lane owns value dim = lane (DV=64).
__global__ void based_ker(const bf16* Q,const bf16* K,const bf16* V,bf16* O,int N){
    const int n=blockIdx.x,h=blockIdx.y,b=blockIdx.z,lane=threadIdx.x;
    const float rd=sqrtf((float)DQK), inv1=1.0f/rd, inv2=1.0f/(2.0f*(float)DQK);
    // load Q[n] (DQK=16) — every lane holds the full query vector
    float qreg[DQK];
    #pragma unroll
    for(int d=0;d<DQK;d++) qreg[d]=__bfloat162float(Q[(size_t)(((b*N+n)*H_+h))*DQK+d]);
    float o=0.0f;
    for(int m=0;m<=n;m++){                    // causal
        float s=0.0f;
        #pragma unroll
        for(int d=0;d<DQK;d++) s+=qreg[d]*__bfloat162float(K[(size_t)(((b*N+m)*H_+h))*DQK+d]);
        const float p=1.0f + s*inv1 + s*s*inv2;
        o += p*__bfloat162float(V[(size_t)(((b*N+m)*H_+h))*DV_+lane]);
    }
    O[(size_t)(((b*N+n)*H_+h))*DV_+lane]=__float2bfloat16(o);
}

static uint16_t f2bf(float f){uint32_t x;std::memcpy(&x,&f,4);uint32_t r=x+0x7FFFu+((x>>16)&1u);return uint16_t(r>>16);}
static float bf2f(uint16_t b){uint32_t x=uint32_t(b)<<16;float f;std::memcpy(&f,&x,4);return f;}
int main(){
    const int B=B_,H=H_,N=N_;
    std::mt19937 rng(5); std::normal_distribution<float> nd(0,1);
    auto gq=[&](int n){std::vector<uint16_t> v(n);for(auto&x:v)x=f2bf(nd(rng)/std::sqrt((float)DQK));return v;};
    auto gv=[&](int n){std::vector<uint16_t> v(n);for(auto&x:v)x=f2bf(nd(rng)/(float)DV_);return v;};
    auto Q=gq(B*N*H*DQK),K=gq(B*N*H*DQK),V=gv(B*N*H*DV_);
    auto qf=[&](int b,int n,int h,int d){return (double)bf2f(Q[(((size_t)b*N+n)*H+h)*DQK+d]);};
    auto kf=[&](int b,int m,int h,int d){return (double)bf2f(K[(((size_t)b*N+m)*H+h)*DQK+d]);};
    auto vf=[&](int b,int m,int h,int e){return (double)bf2f(V[(((size_t)b*N+m)*H+h)*DV_+e]);};
    // fp64 reference
    std::vector<double> ref(B*N*H*DV_,0); const double rd=std::sqrt((double)DQK);
    for(int b=0;b<B;b++)for(int h=0;h<H;h++)for(int n=0;n<N;n++)
        for(int m=0;m<=n;m++){ double s=0; for(int d=0;d<DQK;d++)s+=qf(b,n,h,d)*kf(b,m,h,d);
            double p=1.0+s/rd+s*s/(2.0*DQK);
            for(int e=0;e<DV_;e++) ref[(((size_t)b*N+n)*H+h)*DV_+e]+=p*vf(b,m,h,e); }
    bf16 *dQ,*dK,*dV,*dO;
    auto up=[&](std::vector<uint16_t>&h){bf16*d;CK(hipMalloc(&d,h.size()*2));CK(hipMemcpy(d,h.data(),h.size()*2,hipMemcpyHostToDevice));return d;};
    dQ=up(Q);dK=up(K);dV=up(V); CK(hipMalloc(&dO,(size_t)B*N*H*DV_*2));
    based_ker<<<dim3(N,H,B),64>>>(dQ,dK,dV,dO,N);
    CK(hipDeviceSynchronize()); CK(hipGetLastError());
    std::vector<uint16_t> O(B*N*H*DV_); CK(hipMemcpy(O.data(),dO,O.size()*2,hipMemcpyDeviceToHost));
    double sa=0,sr=0,mx=0; for(size_t k=0;k<ref.size();k++){double d=std::abs(bf2f(O[k])-ref[k]);sa+=d;sr+=std::abs(ref[k]);mx=std::max(mx,d);}
    double rel=sa/std::max(sr,1e-30);
    printf("== based (Taylor-2 causal linear attn)  B=%d H=%d N=%d D=%d DV=%d\n",B,H,N,DQK,DV_);
    printf("o vs fp64 ref: mean rel %.4f%%  max abs %.4g  (%s)\n",100*rel,mx,rel<0.02?"PASS":"FAIL");
    return rel<0.02?0:1;
}
