/**
 * @file
 * @brief CDNA3 (gfx942) FFT convolution, port of QuixiCore-CUDA/kernels/fftconv
 * (ThunderKittens). Reference (pytorch_ref.py): y = ifft(fft(u)*fft(k)).real[:L]
 * = the circular convolution of u and k. By the convolution theorem this equals
 *   y[n] = Re( sum_m u[m] * k[(n-m) mod N] )
 * computed directly here (mathematically identical to the FFT path). Inputs are
 * complex (cfloat); output is the real part. One thread per (n, head, batch).
 * Perf follow-up: a Cooley-Tukey / rocFFT path for O(N log N).
 *   hipcc -std=c++17 -O3 --offload-arch=gfx942 fftconv.cu -o fftconv.out
 */
#include <hip/hip_runtime.h>
#include <cstdio>
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
#define N_ 1024
#endif
#define CK(x) do{hipError_t e=(x); if(e!=hipSuccess){printf("HIP err %s @%d\n",hipGetErrorString(e),__LINE__);exit(1);} }while(0)

// circular convolution: y_re[n] = Re(sum_m u[m]*k[(n-m)%N]). u:[B,H,N], k:[H,N].
__global__ void fftconv_ker(const float* ur,const float* ui,const float* kr,const float* ki,
                            float* yr,int N){
    const int n=blockIdx.x*blockDim.x+threadIdx.x, h=blockIdx.y, b=blockIdx.z;
    if(n>=N) return;
    const size_t ub=((size_t)b*H_+h)*N, kb=(size_t)h*N;
    double accr=0.0;
    for(int m=0;m<N;m++){ int t=n-m; if(t<0)t+=N;
        // (ur+i ui)*(kr+i ki) real part = ur*kr - ui*ki
        accr += (double)ur[ub+m]*(double)kr[kb+t] - (double)ui[ub+m]*(double)ki[kb+t];
    }
    yr[ub+n]=(float)accr;
}

int main(){
    const int B=B_,H=H_,N=N_;
    std::mt19937 rng(11); std::normal_distribution<float> nd(0,1);
    auto gen=[&](int n){std::vector<float> v(n);for(auto&x:v)x=nd(rng);return v;};
    auto ur=gen(B*H*N),ui=gen(B*H*N),kr=gen(H*N),ki=gen(H*N);
    // fp64 reference (direct circular conv, real part)
    std::vector<double> ref(B*H*N,0);
    for(int b=0;b<B;b++)for(int h=0;h<H;h++)for(int n=0;n<N;n++){ double a=0;
        for(int m=0;m<N;m++){int t=n-m;if(t<0)t+=N; a+=(double)ur[(b*H+h)*N+m]*kr[h*N+t]-(double)ui[(b*H+h)*N+m]*ki[h*N+t];}
        ref[(b*H+h)*N+n]=a; }
    float *dur,*dui,*dkr,*dki,*dyr;
    auto up=[&](std::vector<float>&v){float*d;CK(hipMalloc(&d,v.size()*4));CK(hipMemcpy(d,v.data(),v.size()*4,hipMemcpyHostToDevice));return d;};
    dur=up(ur);dui=up(ui);dkr=up(kr);dki=up(ki); CK(hipMalloc(&dyr,(size_t)B*H*N*4));
    fftconv_ker<<<dim3((N+255)/256,H,B),256>>>(dur,dui,dkr,dki,dyr,N);
    CK(hipDeviceSynchronize()); CK(hipGetLastError());
    std::vector<float> yr(B*H*N); CK(hipMemcpy(yr.data(),dyr,yr.size()*4,hipMemcpyDeviceToHost));
    double sa=0,sr=0,mx=0; for(size_t k=0;k<ref.size();k++){double d=std::abs(yr[k]-ref[k]);sa+=d;sr+=std::abs(ref[k]);mx=std::max(mx,d);}
    double rel=sa/std::max(sr,1e-30);
    printf("== fftconv (circular conv)  B=%d H=%d N=%d\n",B,H,N);
    printf("y.real vs fp64 ref: mean rel %.4f%%  max abs %.4g  (%s)\n",100*rel,mx,rel<1e-4?"PASS":"FAIL");
    return rel<1e-4?0:1;
}
