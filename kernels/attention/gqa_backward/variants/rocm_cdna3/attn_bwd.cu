/**
 * @file
 * @brief CDNA3 (gfx942) GQA attention BACKWARD — correctness-first native kernel
 * + standalone fp64 oracle. Same wavefront/online style as the forward
 * (kernels/attention/gqa/variants/rocm_cdna3): one 64-wide wavefront per row,
 * lanes split the head dim (EPL=D/64), dot products via wavefront reduction.
 *
 * Given Q,K,V,dO (and O,L from the forward), computes:
 *   D_i   = sum_d dO_i * O_i
 *   S_ij  = (Q_i . K_j) * scale ;  P_ij = exp(S_ij - L_i)
 *   dV_j += P_ij * dO_i
 *   dP_ij = dO_i . V_j ;  dS_ij = P_ij * (dP_ij - D_i) * scale
 *   dQ_i += dS_ij * K_j ;  dK_j += dS_ij * Q_i
 * Layout [B,N,H,D] (Q/O/dO/dQ), [B,N,H_KV,D] (K/V/dK/dV). GQA kv = h/(H/H_KV).
 * ATTN_CAUSAL restricts i>=j. Build defaults to a small shape for a fast oracle.
 *
 *   hipcc -std=c++17 -O3 --offload-arch=gfx942 attn_bwd.cu -o attn_bwd.out
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
#define H_ 4
#endif
#ifndef HKV_
#define HKV_ 2
#endif
#ifndef N_
#define N_ 128
#endif
#ifndef D_
#define D_ 128
#endif
#ifndef CAUSAL_
#define CAUSAL_ 0
#endif
#define CK(x) do{hipError_t e=(x); if(e!=hipSuccess){printf("HIP err %s @%d\n",hipGetErrorString(e),__LINE__);exit(1);} }while(0)
using bf16 = __hip_bfloat16;

__device__ __forceinline__ float wsum(float v){
    #pragma unroll
    for(int o=32;o>0;o>>=1) v+=__shfl_xor(v,o);
    return v;
}

// dQ: one wavefront per (query i, head h, batch b).
__global__ void bwd_dq(const bf16* Q,const bf16* K,const bf16* V,const bf16* dO,
                       const bf16* O,const float* L,float* dQ,int N,int D){
    const int WF=64, EPL=D/WF;
    const int i=blockIdx.x,h=blockIdx.y,b=blockIdx.z,lane=threadIdx.x;
    const int hk=h/(H_/HKV_); const float scale=1.f/sqrtf((float)D);
    auto Qi=[&](int e){return (size_t)((b*N+i)*H_+h)*D+lane+e*WF;};
    auto Kj=[&](int j,int e){return (size_t)((b*N+j)*HKV_+hk)*D+lane+e*WF;};
    float qv[EPL],dOe[EPL],dq[EPL];
    float dip=0;
    #pragma unroll
    for(int e=0;e<EPL;e++){ qv[e]=__bfloat162float(Q[Qi(e)]); dOe[e]=__bfloat162float(dO[Qi(e)]);
        dip+=dOe[e]*__bfloat162float(O[Qi(e)]); dq[e]=0.f; }
    const float Di=wsum(dip);
    const float Li=L[(size_t)(b*H_+h)*N+i];
    const int jmax=CAUSAL_?(i+1):N;
    for(int j=0;j<jmax;j++){
        float sp=0,dpp=0;
        #pragma unroll
        for(int e=0;e<EPL;e++){ sp+=qv[e]*__bfloat162float(K[Kj(j,e)]); dpp+=dOe[e]*__bfloat162float(V[Kj(j,e)]); }
        const float s=wsum(sp)*scale, p=__expf(s-Li), dp=wsum(dpp);
        const float ds=p*(dp-Di)*scale;
        #pragma unroll
        for(int e=0;e<EPL;e++) dq[e]+=ds*__bfloat162float(K[Kj(j,e)]);
    }
    #pragma unroll
    for(int e=0;e<EPL;e++) dQ[Qi(e)]=dq[e];
}

// dK,dV: one wavefront per (key j, kv head hk, batch b); loops query heads in the
// group and query rows i.
__global__ void bwd_dkv(const bf16* Q,const bf16* K,const bf16* V,const bf16* dO,
                        const bf16* O,const float* L,float* dK,float* dV,int N,int D){
    const int WF=64, EPL=D/WF, GS=H_/HKV_;
    const int j=blockIdx.x,hk=blockIdx.y,b=blockIdx.z,lane=threadIdx.x;
    const float scale=1.f/sqrtf((float)D);
    auto Kj=[&](int e){return (size_t)((b*N+j)*HKV_+hk)*D+lane+e*WF;};
    auto Qi=[&](int i,int h,int e){return (size_t)((b*N+i)*H_+h)*D+lane+e*WF;};
    float kv[EPL],vv[EPL],dk[EPL],dv[EPL];
    #pragma unroll
    for(int e=0;e<EPL;e++){ kv[e]=__bfloat162float(K[Kj(e)]); vv[e]=__bfloat162float(V[Kj(e)]); dk[e]=0; dv[e]=0; }
    for(int g=0;g<GS;g++){ const int h=hk*GS+g;
        const int imin=CAUSAL_?j:0;
        for(int i=imin;i<N;i++){
            const float Li=L[(size_t)(b*H_+h)*N+i];
            float sp=0,dip=0,dpp=0;
            #pragma unroll
            for(int e=0;e<EPL;e++){ float dq=__bfloat162float(dO[Qi(i,h,e)]);
                sp+=__bfloat162float(Q[Qi(i,h,e)])*kv[e]; dip+=dq*__bfloat162float(O[Qi(i,h,e)]); dpp+=dq*vv[e]; }
            const float s=wsum(sp)*scale, p=__expf(s-Li), Di=wsum(dip), dp=wsum(dpp);
            const float ds=p*(dp-Di)*scale;
            #pragma unroll
            for(int e=0;e<EPL;e++){ dv[e]+=p*__bfloat162float(dO[Qi(i,h,e)]); dk[e]+=ds*__bfloat162float(Q[Qi(i,h,e)]); }
        }
    }
    #pragma unroll
    for(int e=0;e<EPL;e++){ dK[Kj(e)]=dk[e]; dV[Kj(e)]=dv[e]; }
}

static uint16_t f2bf(float f){uint32_t x;std::memcpy(&x,&f,4);uint32_t r=x+0x7FFFu+((x>>16)&1u);return uint16_t(r>>16);}
static float bf2f(uint16_t b){uint32_t x=uint32_t(b)<<16;float f;std::memcpy(&f,&x,4);return f;}

int main(){
    const int B=B_,H=H_,HK=HKV_,N=N_,D=D_,GS=H/HK; const double scale=1.0/std::sqrt((double)D);
    printf("== gqa bwd  B=%d H=%d H_KV=%d N=%d D=%d causal=%d\n",B,H,HK,N,D,CAUSAL_);
    std::mt19937 rng(1); std::normal_distribution<float> nd(0,1);
    auto gen=[&](int n){std::vector<uint16_t> v(n);for(auto&x:v)x=f2bf(nd(rng));return v;};
    auto Q=gen(B*N*H*D),K=gen(B*N*HK*D),V=gen(B*N*HK*D),dO=gen(B*N*H*D);
    auto qf=[&](int b,int i,int h,int d){return (double)bf2f(Q[(((size_t)b*N+i)*H+h)*D+d]);};
    auto kf=[&](int b,int j,int h,int d){return (double)bf2f(K[(((size_t)b*N+j)*HK+h)*D+d]);};
    auto vf=[&](int b,int j,int h,int d){return (double)bf2f(V[(((size_t)b*N+j)*HK+h)*D+d]);};
    auto gf=[&](int b,int i,int h,int d){return (double)bf2f(dO[(((size_t)b*N+i)*H+h)*D+d]);};

    // host fp64: O, L, and reference dQ,dK,dV
    std::vector<uint16_t> O(B*N*H*D); std::vector<float> Lh(B*H*N);
    std::vector<double> dQr(B*N*H*D,0),dKr(B*N*HK*D,0),dVr(B*N*HK*D,0);
    std::vector<double> Dvec(B*N*H,0);
    for(int b=0;b<B;b++)for(int h=0;h<H;h++){int hk=h/GS; for(int i=0;i<N;i++){
        int jm=CAUSAL_?(i+1):N; std::vector<double> s(jm); double m=-1e300;
        for(int j=0;j<jm;j++){double d=0;for(int e=0;e<D;e++)d+=qf(b,i,h,e)*kf(b,j,hk,e); s[j]=d*scale; m=std::max(m,s[j]);}
        double Z=0; for(int j=0;j<jm;j++){s[j]=std::exp(s[j]-m);Z+=s[j];}
        double Di=0;
        for(int e=0;e<D;e++){double o=0;for(int j=0;j<jm;j++)o+=s[j]/Z*vf(b,j,hk,e); O[(((size_t)b*N+i)*H+h)*D+e]=f2bf((float)o); Di+=gf(b,i,h,e)*o;}
        Lh[(size_t)(b*H+h)*N+i]=(float)(m+std::log(Z)); Dvec[((size_t)b*N+i)*H+h]=Di;
        for(int j=0;j<jm;j++){double p=s[j]/Z, dp=0; for(int e=0;e<D;e++) dp+=gf(b,i,h,e)*vf(b,j,hk,e);
            double ds=p*(dp-Di)*scale;
            for(int e=0;e<D;e++){ dQr[(((size_t)b*N+i)*H+h)*D+e]+=ds*kf(b,j,hk,e);
                dKr[(((size_t)b*N+j)*HK+hk)*D+e]+=ds*qf(b,i,h,e);
                dVr[(((size_t)b*N+j)*HK+hk)*D+e]+=p*gf(b,i,h,e); }
        }
    }}

    bf16 *dQ_,*dK_,*dV_,*ddO,*dOo; float *dL,*ddQ,*ddK,*ddV;
    auto up=[&](const std::vector<uint16_t>&h){bf16* d;CK(hipMalloc(&d,h.size()*2));CK(hipMemcpy(d,h.data(),h.size()*2,hipMemcpyHostToDevice));return d;};
    dQ_=up(Q);dK_=up(K);dV_=up(V);ddO=up(dO);dOo=up(O);
    CK(hipMalloc(&dL,Lh.size()*4));CK(hipMemcpy(dL,Lh.data(),Lh.size()*4,hipMemcpyHostToDevice));
    CK(hipMalloc(&ddQ,(size_t)B*N*H*D*4));CK(hipMalloc(&ddK,(size_t)B*N*HK*D*4));CK(hipMalloc(&ddV,(size_t)B*N*HK*D*4));
    bwd_dq<<<dim3(N,H,B),64>>>(dQ_,dK_,dV_,ddO,dOo,dL,ddQ,N,D);
    bwd_dkv<<<dim3(N,HK,B),64>>>(dQ_,dK_,dV_,ddO,dOo,dL,ddK,ddV,N,D);
    CK(hipDeviceSynchronize()); CK(hipGetLastError());
    std::vector<float> gQ(B*N*H*D),gK(B*N*HK*D),gV(B*N*HK*D);
    CK(hipMemcpy(gQ.data(),ddQ,gQ.size()*4,hipMemcpyDeviceToHost));
    CK(hipMemcpy(gK.data(),ddK,gK.size()*4,hipMemcpyDeviceToHost));
    CK(hipMemcpy(gV.data(),ddV,gV.size()*4,hipMemcpyDeviceToHost));
    auto rep=[&](const char* nm,std::vector<float>&got,std::vector<double>&ref){
        double sa=0,sr=0,mx=0; for(size_t k=0;k<ref.size();k++){double d=std::abs(got[k]-ref[k]);sa+=d;sr+=std::abs(ref[k]);mx=std::max(mx,d);}
        double rel=sa/std::max(sr,1e-30); printf("%-6s mean rel %.4f%%  max abs %.4g  (%s)\n",nm,100*rel,mx,rel<0.02?"PASS":"FAIL"); return rel<0.02;};
    bool ok=true; ok&=rep("dQ",gQ,dQr); ok&=rep("dK",gK,dKr); ok&=rep("dV",gV,dVr);
    printf("%s\n", ok?"ALL PASS":"FAILED"); return ok?0:1;
}
