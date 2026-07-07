#include "attn_mfma.cuh"
static uint16_t f2bf(float f){uint32_t x;std::memcpy(&x,&f,4);uint32_t r=x+0x7FFFu+((x>>16)&1u);return uint16_t(r>>16);}
static float bf2f(uint16_t b){uint32_t x=uint32_t(b)<<16;float f;std::memcpy(&f,&x,4);return f;}
int main(){
    const int B=ATTN_B,H=ATTN_H,HK=ATTN_H_KV,N=ATTN_N,D=ATTN_D,GS=H/HK;
    const float scale=1.0f/std::sqrt((float)D);
    printf("== gqa fwd MFMA  B=%d H=%d H_KV=%d N=%d D=%d causal=%d\n",B,H,HK,N,D,ATTN_CAUSAL);
    std::mt19937 rng(0); std::normal_distribution<float> nd(0,1);
    auto gen=[&](int n){std::vector<uint16_t> v(n);for(auto&x:v)x=f2bf(nd(rng));return v;};
    auto Q=gen(B*N*H*D),K=gen(B*N*HK*D),Vv=gen(B*N*HK*D);
    auto qf=[&](int b,int i,int h,int d){return (double)bf2f(Q[(((size_t)b*N+i)*H+h)*D+d]);};
    auto kf=[&](int b,int j,int h,int d){return (double)bf2f(K[(((size_t)b*N+j)*HK+h)*D+d]);};
    auto vf=[&](int b,int j,int h,int e){return (double)bf2f(Vv[(((size_t)b*N+j)*HK+h)*D+e]);};
    std::vector<double> ref(B*N*H*D,0);
    for(int b=0;b<B;b++)for(int h=0;h<H;h++){int hk=h/GS; for(int i=0;i<N;i++){
        int jm=ATTN_CAUSAL?(i+1):N; std::vector<double> s(jm); double m=-1e300;
        for(int j=0;j<jm;j++){double d=0;for(int e=0;e<D;e++)d+=qf(b,i,h,e)*kf(b,j,hk,e); s[j]=d*scale; m=std::max(m,s[j]);}
        double Z=0; for(int j=0;j<jm;j++){s[j]=std::exp(s[j]-m);Z+=s[j];}
        for(int e=0;e<D;e++){double o=0;for(int j=0;j<jm;j++)o+=s[j]*vf(b,j,hk,e); ref[(((size_t)b*N+i)*H+h)*D+e]=o/Z;} } }
    bf16 *dQ,*dK,*dV,*dO; float* dL;
    auto up=[&](std::vector<uint16_t>&h){bf16*d;CK(hipMalloc(&d,h.size()*2));CK(hipMemcpy(d,h.data(),h.size()*2,hipMemcpyHostToDevice));return d;};
    dQ=up(Q);dK=up(K);dV=up(Vv);CK(hipMalloc(&dO,(size_t)B*N*H*D*2));CK(hipMalloc(&dL,(size_t)B*H*N*4));
    dim3 grid(N/16,H,B);
    attend_ker<ATTN_D><<<grid,64>>>(dQ,dK,dV,dO,dL,N);
    CK(hipDeviceSynchronize()); CK(hipGetLastError());
    std::vector<uint16_t> Oh(B*N*H*D); CK(hipMemcpy(Oh.data(),dO,Oh.size()*2,hipMemcpyDeviceToHost));
    double sa=0,sr=0,mx=0; for(size_t k=0;k<ref.size();k++){double d=std::abs(bf2f(Oh[k])-ref[k]);sa+=d;sr+=std::abs(ref[k]);mx=std::max(mx,d);}
    double rel=sa/std::max(sr,1e-30);
    printf("gqa fwd MFMA vs fp32 host ref: mean rel %.4f%%  max abs %.4g  (%s)\n",100*rel,mx,rel<0.02?"PASS":"FAIL");
    return rel<0.02?0:1;
}
