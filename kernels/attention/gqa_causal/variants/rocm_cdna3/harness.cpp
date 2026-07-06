/**
 * @file
 * @brief Standalone HIP correctness harness for the CDNA3 GQA forward attention
 * kernel (attn_kernel.cuh). No torch / pyutils — avoids the ROCm 7.2 (system) vs
 * 6.4 (torch wheel) co-load conflict. Generates bf16 Q/K/V, runs the kernel, and
 * compares against an fp32 host softmax-attention reference computed from the
 * same bf16-rounded inputs. Layout [B,N,H,D] (Q/O) and [B,N,H_KV,D] (K/V),
 * matching the torch test; GQA head map kv = h / (H/H_KV); scale = 1/sqrt(D).
 *
 * Build (small shapes for a fast host ref):
 *   hipcc -std=c++20 -O3 --offload-arch=gfx942 -DKITTENS_CDNA3 \
 *     -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math -I../../../../include \
 *     -DATTN_B=1 -DATTN_H=8 -DATTN_H_KV=2 -DATTN_N=256 -DATTN_D=128 \
 *     harness.cpp -o harness.out
 */
#include "attn_kernel.cuh"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <random>
#include <cmath>

#define CK(x) do { hipError_t e=(x); if(e!=hipSuccess){printf("HIP err %s @%d\n",hipGetErrorString(e),__LINE__);return 1;} } while(0)

static uint16_t f2bf(float f){ uint32_t x; std::memcpy(&x,&f,4); uint32_t r=x+0x7FFFu+((x>>16)&1u); return uint16_t(r>>16); }
static float    bf2f(uint16_t b){ uint32_t x=uint32_t(b)<<16; float f; std::memcpy(&f,&x,4); return f; }

int main() {
    const int B=ATTN_B, H=ATTN_H, HK=ATTN_H_KV, N=ATTN_N, D=ATTN_D;
    const int GS = H/HK;
    const float scale = 1.0f/std::sqrt(float(D));
    printf("== gqa fwd  B=%d H=%d H_KV=%d N=%d D=%d\n", B,H,HK,N,D);

    std::mt19937 rng(0); std::normal_distribution<float> nd(0.f,1.f);
    auto gen = [&](int n){ std::vector<uint16_t> v(n); for(auto&x:v) x=f2bf(nd(rng)); return v; };
    auto Q = gen(B*N*H*D), K = gen(B*N*HK*D), V = gen(B*N*HK*D);

    uint16_t *dQ,*dK,*dV,*dO; float* dL;
    CK(hipMalloc(&dQ,Q.size()*2)); CK(hipMalloc(&dK,K.size()*2)); CK(hipMalloc(&dV,V.size()*2));
    CK(hipMalloc(&dO,(size_t)B*N*H*D*2)); CK(hipMalloc(&dL,(size_t)B*H*N*4));
    CK(hipMemcpy(dQ,Q.data(),Q.size()*2,hipMemcpyHostToDevice));
    CK(hipMemcpy(dK,K.data(),K.size()*2,hipMemcpyHostToDevice));
    CK(hipMemcpy(dV,V.data(),V.size()*2,hipMemcpyHostToDevice));

    using GL = _gl_QKVO;
    attn_globals<ATTN_D> g {
        GL((bf16*)dQ, B, N, H,  D),
        GL((bf16*)dK, B, N, HK, D),
        GL((bf16*)dV, B, N, HK, D),
        GL((bf16*)dO, B, N, H,  D),
        gl<float,-1,-1,-1,-1>(dL, B, H, 1, N),
        (hipStream_t)0
    };
    dispatch_micro<ATTN_D>(g);
    CK(hipDeviceSynchronize());
    CK(hipGetLastError());

    std::vector<uint16_t> O((size_t)B*N*H*D); CK(hipMemcpy(O.data(),dO,O.size()*2,hipMemcpyDeviceToHost));

    // fp32 host reference from bf16-rounded inputs
    auto qf=[&](int b,int i,int h,int d){ return bf2f(Q[(((size_t)b*N+i)*H+h)*D+d]); };
    auto kf=[&](int b,int j,int h,int d){ return bf2f(K[(((size_t)b*N+j)*HK+h)*D+d]); };
    auto vf=[&](int b,int j,int h,int d){ return bf2f(V[(((size_t)b*N+j)*HK+h)*D+d]); };
    double worst=0, sabs=0, sref=0;
    std::vector<float> sc(N);
    for(int b=0;b<B;b++) for(int h=0;h<H;h++){ const int hk=h/GS;
        for(int i=0;i<N;i++){
            const int jm = ATTN_CAUSAL ? (i+1) : N;
            float m=-1e30f;
            for(int j=0;j<jm;j++){ float s=0; for(int d=0;d<D;d++) s+=qf(b,i,h,d)*kf(b,j,hk,d); s*=scale; sc[j]=s; m=std::max(m,s); }
            float Z=0; for(int j=0;j<jm;j++){ sc[j]=std::exp(sc[j]-m); Z+=sc[j]; }
            for(int d=0;d<D;d++){ float o=0; for(int j=0;j<jm;j++) o+=sc[j]*vf(b,j,hk,d); o/=Z;
                float got=bf2f(O[(((size_t)b*N+i)*H+h)*D+d]);
                worst=std::max(worst,(double)std::abs(got-o)); sabs+=std::abs(got-o); sref+=std::abs(o);
            }
        }
    }
    double rel = sabs/std::max(sref,1e-30);
    bool ok = rel < 0.02;
    printf("gqa fwd vs fp32 host ref: mean rel %.4f%% max abs %.4g  (%s)\n", 100*rel, worst, ok?"PASS":"FAIL");
    return ok?0:1;
}
