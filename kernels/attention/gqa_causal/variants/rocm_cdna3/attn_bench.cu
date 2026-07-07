/**
 * @file
 * @brief Perf A/B for the CDNA3 GQA forward attention: naive one-wavefront-per-
 * query (attn_kernel.cuh style) vs the MFMA-tiled flash kernel (attn_mfma.cuh).
 * HIP-event median timing at a realistic shape; reports ms + TFLOP/s + speedup.
 * Correctness is covered by attn_mfma.cu / harness.cpp (fp32 oracle).
 *   hipcc -std=c++17 -O3 --offload-arch=gfx942 attn_bench.cu -o attn_bench.out
 */
#include "attn_mfma.cuh"
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>
#include <cstdio>

// naive: one 64-lane wavefront per (query, head, batch); lanes split D.
template<int D>
__global__ void naive_ker(const bf16* Q,const bf16* K,const bf16* V,bf16* O,int N){
    constexpr int WF=64, EPL=D/WF;
    const int q=blockIdx.x,h=blockIdx.y,b=blockIdx.z,lane=threadIdx.x;
    const int hk=h/(ATTN_H/ATTN_H_KV); const float scale=1.0f/sqrtf((float)D);
    float qv[EPL],acc[EPL];
    #pragma unroll
    for(int e=0;e<EPL;e++){ qv[e]=__bfloat162float(Q[(size_t)(((b*N+q)*ATTN_H+h)*D+lane+e*WF)]); acc[e]=0; }
    float m=-3.0e38f,l=0.0f; const int jm=ATTN_CAUSAL?(q+1):N;
    for(int j=0;j<jm;j++){ float part=0;
        #pragma unroll
        for(int e=0;e<EPL;e++) part+=qv[e]*__bfloat162float(K[(size_t)(((b*N+j)*ATTN_H_KV+hk)*D+lane+e*WF)]);
        for(int o=32;o>0;o>>=1) part+=__shfl_xor(part,o);
        const float s=part*scale, mn=fmaxf(m,s), corr=__expf(m-mn), p=__expf(s-mn); l=l*corr+p;
        #pragma unroll
        for(int e=0;e<EPL;e++) acc[e]=acc[e]*corr+p*__bfloat162float(V[(size_t)(((b*N+j)*ATTN_H_KV+hk)*D+lane+e*WF)]);
        m=mn; }
    const float inv=l>0?1.0f/l:0.0f;
    #pragma unroll
    for(int e=0;e<EPL;e++) O[(size_t)(((b*N+q)*ATTN_H+h)*D+lane+e*WF)]=__float2bfloat16(acc[e]*inv);
}

static uint16_t f2bf(float f){uint32_t x;std::memcpy(&x,&f,4);uint32_t r=x+0x7FFFu+((x>>16)&1u);return uint16_t(r>>16);}
template<typename L> static double med_ms(L launch,int warm=10,int it=50){
    for(int i=0;i<warm;i++) launch(); hipDeviceSynchronize();
    std::vector<float> t(it); hipEvent_t a,b; hipEventCreate(&a); hipEventCreate(&b);
    for(int i=0;i<it;i++){ hipEventRecord(a); launch(); hipEventRecord(b); hipEventSynchronize(b); hipEventElapsedTime(&t[i],a,b); }
    std::sort(t.begin(),t.end()); return t[it/2];
}
int main(){
    const int B=ATTN_B,H=ATTN_H,HK=ATTN_H_KV,N=ATTN_N,D=ATTN_D;
    printf("== attn fwd A/B  B=%d H=%d H_KV=%d N=%d D=%d causal=%d\n",B,H,HK,N,D,ATTN_CAUSAL);
    std::mt19937 rng(0); std::normal_distribution<float> nd(0,1);
    auto gen=[&](int n){std::vector<uint16_t> v(n);for(auto&x:v)x=f2bf(nd(rng));return v;};
    auto Q=gen(B*N*H*D),K=gen(B*N*HK*D),Vv=gen(B*N*HK*D);
    bf16 *dQ,*dK,*dV,*dO; float* dL;
    auto up=[&](std::vector<uint16_t>&h){bf16*d;hipMalloc(&d,h.size()*2);hipMemcpy(d,h.data(),h.size()*2,hipMemcpyHostToDevice);return d;};
    dQ=up(Q);dK=up(K);dV=up(Vv);hipMalloc(&dO,(size_t)B*N*H*D*2);hipMalloc(&dL,(size_t)B*H*N*4);
    dim3 gN(N,H,B), gM(N/16,H,B);
    double tn=med_ms([&]{ naive_ker<ATTN_D><<<gN,64>>>(dQ,dK,dV,dO,N); });
    double tm=med_ms([&]{ attend_ker<ATTN_D><<<gM,64>>>(dQ,dK,dV,dO,dL,N); });
    const double flop = 4.0*B*H*(double)N*N*D / (ATTN_CAUSAL?2.0:1.0);
    printf("naive : %.3f ms  %.1f TFLOP/s\n", tn, flop/(tn*1e-3)/1e12);
    printf("mfma  : %.3f ms  %.1f TFLOP/s\n", tm, flop/(tm*1e-3)/1e12);
    printf("speedup: %.2fx\n", tn/tm);
    return 0;
}
