/**
 * @file
 * @brief CDNA3 (gfx942) Hedgehog hybrid linear+exact attention, ported from
 * QuixiCore-CUDA/kernels/hedgehog (ThunderKittens). Matches gentests.py:
 *
 *  feature(x,map) = concat( softmax(x@map), softmax(-x@map) ).clamp(min=1e-6)
 *    (x:[.,D_QK], map:[D_QK,64] -> proj[.,64] -> feature[.,128])
 *  Qs=feature(Q,Qmap), Ks=feature(K,Kmap)
 *  Block=64. For query i (block bi=i/64):
 *    window keys = causal j with block(j) in {bi, bi-1} (recent 128) -> EXACT:
 *       a_exp[j] = beta * exp( (Q_i.K_j - rowmax)/sqrt(D_QK) )   (rowmax over window)
 *    older keys = causal j with block(j) <= bi-2 -> LINEAR:
 *       a_lin[j] = alpha * (Qs_i . Ks_j)
 *    a = a_exp + a_lin ; out_i = sum_j a[j] V[j] / (sum_j a[j] + 1e-6)
 *
 * Kernel 1 computes the feature maps; kernel 2 the hybrid attention.
 *   hipcc -std=c++17 -O3 --offload-arch=gfx942 hedgehog.cu -o hedgehog.out
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
#define DQK 128
#define DVO 128
#define FEAT 128     // 64 pos + 64 neg
#define BLK 64
#define CK(x) do{hipError_t e=(x); if(e!=hipSuccess){printf("HIP err %s @%d\n",hipGetErrorString(e),__LINE__);exit(1);} }while(0)
using bf16 = __hip_bfloat16;
__device__ __forceinline__ float wsum(float v){ for(int o=32;o>0;o>>=1) v+=__shfl_xor(v,o); return v; }
__device__ __forceinline__ float wmax(float v){ for(int o=32;o>0;o>>=1) v=fmaxf(v,__shfl_xor(v,o)); return v; }

// feature map: one wavefront per (row, head, batch). 64 lanes = 64 proj dims.
__global__ void feat_ker(const bf16* X,const bf16* map,bf16* Fs,int N){
    const int n=blockIdx.x,h=blockIdx.y,b=blockIdx.z,c=threadIdx.x; // c in 0..63
    float proj=0.0f;
    #pragma unroll
    for(int d=0;d<DQK;d++) proj+=__bfloat162float(X[(size_t)(((b*N+n)*H_+h))*DQK+d])
                                 *__bfloat162float(map[(size_t)(h*DQK+d)*64+c]);
    // softmax(proj) over the 64 lanes
    const float pm=wmax(proj), pe=__expf(proj-pm), pden=wsum(pe);
    // softmax(-proj)
    const float nm=wmax(-proj), ne=__expf(-proj-nm), nden=wsum(ne);
    float pos=pe/pden, neg=ne/nden;
    pos=fmaxf(pos,1e-6f); neg=fmaxf(neg,1e-6f);
    const size_t base=(size_t)(((b*N+n)*H_+h))*FEAT;
    Fs[base+c]=__float2bfloat16(pos);
    Fs[base+64+c]=__float2bfloat16(neg);
}

// hybrid attention: one wavefront per (query i, head, batch); lanes own DVO (EPL=2).
__global__ void attn_ker(const bf16* Q,const bf16* K,const bf16* V,
                         const bf16* Qs,const bf16* Ks,const float* alphas,const float* betas,
                         bf16* O,int N){
    constexpr int WF=64, EPL=DVO/WF, EPLF=FEAT/WF;
    const int i=blockIdx.x,h=blockIdx.y,b=blockIdx.z,lane=threadIdx.x;
    const int bi=i/BLK; const float rd=sqrtf((float)DQK);
    const float al=alphas[h], be=betas[h];
    auto Xi=[&](const bf16* P,int r,int e,int stride){return (size_t)(((b*N+r)*H_+h))*stride+lane+e*WF;};
    // cache Q[i], Qs[i]
    float qi[EPL], qsi[EPLF];
    #pragma unroll
    for(int e=0;e<EPL;e++)  qi[e]=__bfloat162float(Q[Xi(Q,i,e,DQK)]);
    #pragma unroll
    for(int e=0;e<EPLF;e++) qsi[e]=__bfloat162float(Qs[Xi(Qs,i,e,FEAT)]);
    const int wlo = (bi>=1)?(bi-1)*BLK:0;         // window start
    // pass 1: rowmax of raw Q.K over the causal window
    float wm=-3.4e38f;
    for(int j=wlo;j<=i;j++){ float p=0;
        #pragma unroll
        for(int e=0;e<EPL;e++) p+=qi[e]*__bfloat162float(K[Xi(K,j,e,DQK)]);
        wm=fmaxf(wm,wsum(p)); }
    // pass 2: accumulate a[j] and a[j]*V[j] over all causal j
    float den=0.0f, acc[EPL];
    #pragma unroll
    for(int e=0;e<EPL;e++) acc[e]=0.0f;
    for(int j=0;j<=i;j++){
        const int bj=j/BLK; float a;
        if(bj==bi || bj==bi-1){                    // exact window
            float p=0;
            #pragma unroll
            for(int e=0;e<EPL;e++) p+=qi[e]*__bfloat162float(K[Xi(K,j,e,DQK)]);
            a=be*__expf((wsum(p)-wm)/rd);
        } else {                                   // linear (older)
            float p=0;
            #pragma unroll
            for(int e=0;e<EPLF;e++) p+=qsi[e]*__bfloat162float(Ks[Xi(Ks,j,e,FEAT)]);
            a=al*wsum(p);
        }
        den+=a;
        #pragma unroll
        for(int e=0;e<EPL;e++) acc[e]+=a*__bfloat162float(V[Xi(V,j,e,DVO)]);
    }
    const float inv=1.0f/(den+1e-6f);
    #pragma unroll
    for(int e=0;e<EPL;e++) O[Xi(O,i,e,DVO)]=__float2bfloat16(acc[e]*inv);
}

static uint16_t f2bf(float f){uint32_t x;std::memcpy(&x,&f,4);uint32_t r=x+0x7FFFu+((x>>16)&1u);return uint16_t(r>>16);}
static float bf2f(uint16_t b){uint32_t x=uint32_t(b)<<16;float f;std::memcpy(&f,&x,4);return f;}
int main(){
    const int B=B_,H=H_,N=N_;
    std::mt19937 rng(7); std::normal_distribution<float> nd(0,1); std::uniform_real_distribution<float> ur(0,1);
    auto gen=[&](int n,float sc){std::vector<uint16_t> v(n);for(auto&x:v)x=f2bf(nd(rng)*sc);return v;};
    auto Q=gen(B*N*H*DQK,1),K=gen(B*N*H*DQK,1),V=gen(B*N*H*DVO,0.2f);
    auto Qm=gen(H*DQK*64,1),Km=gen(H*DQK*64,1);
    std::vector<float> alphas(H),betas(H); for(int h=0;h<H;h++){alphas[h]=ur(rng)*3;betas[h]=ur(rng);}
    auto xf=[&](const std::vector<uint16_t>&A,int b,int n,int h,int d,int st){return (double)bf2f(A[(((size_t)b*N+n)*H+h)*st+d]);};
    auto mf=[&](const std::vector<uint16_t>&M,int h,int d,int c){return (double)bf2f(M[(size_t)(h*DQK+d)*64+c]);};
    // fp64 reference feature maps
    auto feature=[&](const std::vector<uint16_t>&X,const std::vector<uint16_t>&M,int b,int n,int h,std::vector<double>&f){
        double proj[64]; for(int c=0;c<64;c++){double s=0;for(int d=0;d<DQK;d++)s+=xf(X,b,n,h,d,DQK)*mf(M,h,d,c);proj[c]=s;}
        double pm=-1e300,nm=-1e300; for(int c=0;c<64;c++){pm=std::max(pm,proj[c]);nm=std::max(nm,-proj[c]);}
        double pd=0,ndn=0; for(int c=0;c<64;c++){pd+=std::exp(proj[c]-pm);ndn+=std::exp(-proj[c]-nm);}
        for(int c=0;c<64;c++){ f[c]=std::max(std::exp(proj[c]-pm)/pd,1e-6); f[64+c]=std::max(std::exp(-proj[c]-nm)/ndn,1e-6);} };
    std::vector<double> ref(B*N*H*DVO,0);
    for(int b=0;b<B;b++)for(int h=0;h<H;h++){ const double rd=std::sqrt((double)DQK);
        std::vector<std::vector<double>> Qsr(N,std::vector<double>(FEAT)),Ksr(N,std::vector<double>(FEAT));
        for(int n=0;n<N;n++){ feature(Q,Qm,b,n,h,Qsr[n]); feature(K,Km,b,n,h,Ksr[n]); }
        for(int i=0;i<N;i++){ int bi=i/BLK; int wlo=(bi>=1)?(bi-1)*BLK:0;
            double wm=-1e300; for(int j=wlo;j<=i;j++){double p=0;for(int d=0;d<DQK;d++)p+=xf(Q,b,i,h,d,DQK)*xf(K,b,j,h,d,DQK); wm=std::max(wm,p);}
            std::vector<double> a(i+1); double den=0;
            for(int j=0;j<=i;j++){ int bj=j/BLK; double av;
                if(bj==bi||bj==bi-1){ double p=0;for(int d=0;d<DQK;d++)p+=xf(Q,b,i,h,d,DQK)*xf(K,b,j,h,d,DQK); av=betas[h]*std::exp((p-wm)/rd); }
                else { double p=0;for(int c=0;c<FEAT;c++)p+=Qsr[i][c]*Ksr[j][c]; av=alphas[h]*p; }
                a[j]=av; den+=av; }
            double inv=1.0/(den+1e-6);
            for(int e=0;e<DVO;e++){double o=0;for(int j=0;j<=i;j++)o+=a[j]*xf(V,b,j,h,e,DVO); ref[(((size_t)b*N+i)*H+h)*DVO+e]=o*inv;}
        }
    }
    // device
    bf16 *dQ,*dK,*dV,*dQm,*dKm,*dQs,*dKs,*dO; float *dal,*dbe;
    auto up=[&](std::vector<uint16_t>&h){bf16*d;CK(hipMalloc(&d,h.size()*2));CK(hipMemcpy(d,h.data(),h.size()*2,hipMemcpyHostToDevice));return d;};
    dQ=up(Q);dK=up(K);dV=up(V);dQm=up(Qm);dKm=up(Km);
    CK(hipMalloc(&dQs,(size_t)B*N*H*FEAT*2));CK(hipMalloc(&dKs,(size_t)B*N*H*FEAT*2));CK(hipMalloc(&dO,(size_t)B*N*H*DVO*2));
    CK(hipMalloc(&dal,H*4));CK(hipMemcpy(dal,alphas.data(),H*4,hipMemcpyHostToDevice));
    CK(hipMalloc(&dbe,H*4));CK(hipMemcpy(dbe,betas.data(),H*4,hipMemcpyHostToDevice));
    feat_ker<<<dim3(N,H,B),64>>>(dQ,dQm,dQs,N);
    feat_ker<<<dim3(N,H,B),64>>>(dK,dKm,dKs,N);
    attn_ker<<<dim3(N,H,B),64>>>(dQ,dK,dV,dQs,dKs,dal,dbe,dO,N);
    CK(hipDeviceSynchronize()); CK(hipGetLastError());
    std::vector<uint16_t> O(B*N*H*DVO); CK(hipMemcpy(O.data(),dO,O.size()*2,hipMemcpyDeviceToHost));
    double sa=0,sr=0,mx=0; for(size_t k=0;k<ref.size();k++){double d=std::abs(bf2f(O[k])-ref[k]);sa+=d;sr+=std::abs(ref[k]);mx=std::max(mx,d);}
    double rel=sa/std::max(sr,1e-30);
    printf("== hedgehog (hybrid linear+exact)  B=%d H=%d N=%d D_QK=%d D_VO=%d\n",B,H,N,DQK,DVO);
    printf("out vs fp64 ref: mean rel %.4f%%  max abs %.4g  (%s)\n",100*rel,mx,rel<0.02?"PASS":"FAIL");
    return rel<0.02?0:1;
}
