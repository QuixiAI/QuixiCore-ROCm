/**
 * @file
 * @brief CDNA3 (gfx942) Mamba2 SSD (state-space duality) kernel, port of the
 * kittens QuixiCore-CUDA/kernels/mamba2/mamba2.cu. The chunked reference
 * (gentests.py ssd_minimal_discrete: Y_diag intra-chunk + Y_off inter-chunk)
 * computes the SSD quadratic form
 *   Y[t,h,p] = sum_{s<=t} (C[t,h,:] . B[s,h,:]) * exp(Acum[t,h]-Acum[s,h]) * X[s,h,p]
 * with A the per-(head,timestep) log-decay and Acum its causal prefix sum. This
 * evaluates that form directly (identical result, non-chunked). Complements the
 * already-ported selective_scan (the recurrent form).
 * dstate N=64 (C/B), headdim P=64 (X/Y). One wavefront per (t,head,batch);
 * lanes own P; the C.B state dot is a wavefront reduction. fp64 oracle.
 *   hipcc -std=c++17 -O3 --offload-arch=gfx942 mamba2_ssd.cu -o mamba2_ssd.out
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
#define DSTATE 64
#define HEADDIM 64
#define CK(x) do{hipError_t e=(x); if(e!=hipSuccess){printf("HIP err %s @%d\n",hipGetErrorString(e),__LINE__);exit(1);} }while(0)
using bf16 = __hip_bfloat16;
__device__ __forceinline__ float wsum(float v){ for(int o=32;o>0;o>>=1) v+=__shfl_xor(v,o); return v; }

// one wavefront per (t, head, batch); lane owns headdim p=lane (HEADDIM=64=WF).
__global__ void ssd_ker(const bf16* C,const bf16* Bm,const bf16* X,const float* Acum,bf16* Y,int N){
    const int t=blockIdx.x,h=blockIdx.y,b=blockIdx.z,lane=threadIdx.x;
    auto Cs=[&](int r,int n){return (size_t)(((b*N+r)*H_+h))*DSTATE+n;};
    auto Xs=[&](int r){return (size_t)(((b*N+r)*H_+h))*HEADDIM+lane;};
    const float act=Acum[(size_t)(b*H_+h)*N+t];
    float creg[DSTATE];
    #pragma unroll
    for(int n=0;n<DSTATE;n++) creg[n]=__bfloat162float(C[Cs(t,n)]);
    float y=0.0f;
    for(int s=0;s<=t;s++){
        float sc=0.0f;   // C_t . B_s over dstate; every lane computes it (redundant)
        #pragma unroll
        for(int n=0;n<DSTATE;n++) sc+=creg[n]*__bfloat162float(Bm[Cs(s,n)]);
        const float decay=__expf(act-Acum[(size_t)(b*H_+h)*N+s]);
        y += sc*decay*__bfloat162float(X[Xs(s)]);
    }
    Y[Xs(t)]=__float2bfloat16(y);
}

static uint16_t f2bf(float f){uint32_t x;std::memcpy(&x,&f,4);uint32_t r=x+0x7FFFu+((x>>16)&1u);return uint16_t(r>>16);}
static float bf2f(uint16_t b){uint32_t x=uint32_t(b)<<16;float f;std::memcpy(&f,&x,4);return f;}
int main(){
    const int B=B_,H=H_,N=N_;
    std::mt19937 rng(13); std::normal_distribution<float> nd(0,1); std::uniform_real_distribution<float> ur(0,1);
    auto gen=[&](int n,float s){std::vector<uint16_t> v(n);for(auto&x:v)x=f2bf(nd(rng)*s);return v;};
    auto C=gen(B*N*H*DSTATE,1),Bm=gen(B*N*H*DSTATE,1),X=gen(B*N*H*HEADDIM,0.3f);
    // per-(b,h,t) log-decay A in (-0.5,0), prefix-summed causally
    std::vector<float> A(B*H*N),Acum(B*H*N);
    for(int b=0;b<B;b++)for(int h=0;h<H;h++){double c=0;for(int t=0;t<N;t++){A[(b*H+h)*N+t]=-0.5f*ur(rng); c+=A[(b*H+h)*N+t]; Acum[(b*H+h)*N+t]=(float)c;}}
    auto cf=[&](const std::vector<uint16_t>&Ar,int b,int r,int h,int n,int st){return (double)bf2f(Ar[(((size_t)b*N+r)*H+h)*st+n]);};
    std::vector<double> ref(B*N*H*HEADDIM,0);
    for(int b=0;b<B;b++)for(int h=0;h<H;h++)for(int t=0;t<N;t++){ double at=Acum[(b*H+h)*N+t];
        for(int s=0;s<=t;s++){ double sc=0; for(int n=0;n<DSTATE;n++)sc+=cf(C,b,t,h,n,DSTATE)*cf(Bm,b,s,h,n,DSTATE);
            double dec=std::exp(at-(double)Acum[(b*H+h)*N+s]);
            for(int p=0;p<HEADDIM;p++) ref[(((size_t)b*N+t)*H+h)*HEADDIM+p]+=sc*dec*cf(X,b,s,h,p,HEADDIM); } }
    bf16 *dC,*dB,*dX,*dY; float* dAc;
    auto up=[&](std::vector<uint16_t>&h){bf16*d;CK(hipMalloc(&d,h.size()*2));CK(hipMemcpy(d,h.data(),h.size()*2,hipMemcpyHostToDevice));return d;};
    dC=up(C);dB=up(Bm);dX=up(X); CK(hipMalloc(&dY,(size_t)B*N*H*HEADDIM*2));
    CK(hipMalloc(&dAc,Acum.size()*4));CK(hipMemcpy(dAc,Acum.data(),Acum.size()*4,hipMemcpyHostToDevice));
    ssd_ker<<<dim3(N,H,B),64>>>(dC,dB,dX,dAc,dY,N);
    CK(hipDeviceSynchronize()); CK(hipGetLastError());
    std::vector<uint16_t> Y(B*N*H*HEADDIM); CK(hipMemcpy(Y.data(),dY,Y.size()*2,hipMemcpyDeviceToHost));
    double sa=0,sr=0,mx=0; for(size_t k=0;k<ref.size();k++){double d=std::abs(bf2f(Y[k])-ref[k]);sa+=d;sr+=std::abs(ref[k]);mx=std::max(mx,d);}
    double rel=sa/std::max(sr,1e-30);
    printf("== mamba2 SSD  B=%d H=%d N=%d dstate=%d headdim=%d\n",B,H,N,DSTATE,HEADDIM);
    printf("Y vs fp64 ref: mean rel %.4f%%  max abs %.4g  (%s)\n",100*rel,mx,rel<0.02?"PASS":"FAIL");
    return rel<0.02?0:1;
}
