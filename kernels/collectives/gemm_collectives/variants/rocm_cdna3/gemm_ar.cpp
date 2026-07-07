// CDNA3 gemm_ar: K-parallel tensor-parallel GEMM + RCCL all-reduce. Correct
// semantics for CUDA parallel/gemm_ar (multimem->RCCL). Overlap = perf follow-up.
#include <hip/hip_runtime.h>
#include <rccl.h>
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#define HC(x) do{hipError_t e=(x); if(e!=hipSuccess){printf("HIP %s @%d\n",hipGetErrorString(e),__LINE__);return 1;}}while(0)
#define NC(x) do{ncclResult_t r=(x); if(r!=ncclSuccess){printf("RCCL %s @%d\n",ncclGetErrorString(r),__LINE__);return 1;}}while(0)
__global__ void gemm(const float*A,const float*B,float*C,int M,int N,int K){
    int m=blockIdx.y*blockDim.y+threadIdx.y,n=blockIdx.x*blockDim.x+threadIdx.x;
    if(m>=M||n>=N)return; float s=0; for(int k=0;k<K;k++)s+=A[(size_t)m*K+k]*B[(size_t)k*N+n]; C[(size_t)m*N+n]=s;
}
int main(){
    int nd=0; HC(hipGetDeviceCount(&nd)); int P=(nd>=4)?4:nd;
    const int M=64,N=64,K=128,Ks=K/P;
    printf("== gemm_ar (K-parallel + all-reduce) across %d GPUs M=%d N=%d K=%d\n",P,M,N,K);
    std::mt19937 rng(1); std::normal_distribution<float> nd2(0,1);
    std::vector<float> A(M*K),B(K*N); for(auto&x:A)x=nd2(rng); for(auto&x:B)x=nd2(rng);
    std::vector<double> ref(M*N,0); for(int m=0;m<M;m++)for(int n=0;n<N;n++){double s=0;for(int k=0;k<K;k++)s+=(double)A[m*K+k]*B[k*N+n];ref[m*N+n]=s;}
    std::vector<int> devs(P); for(int i=0;i<P;i++)devs[i]=i;
    std::vector<ncclComm_t> comm(P); NC(ncclCommInitAll(comm.data(),P,devs.data())); printf("[ck] init done\n");
    std::vector<hipStream_t> st(P), sc(P); std::vector<float*> dA(P),dB(P),dY(P);
    for(int r=0;r<P;r++){ HC(hipSetDevice(r)); HC(hipStreamCreate(&st[r])); HC(hipStreamCreate(&sc[r]));
        std::vector<float> aK(M*Ks),bK(Ks*N);
        for(int m=0;m<M;m++)for(int k=0;k<Ks;k++)aK[m*Ks+k]=A[m*K+r*Ks+k];
        for(int k=0;k<Ks;k++)for(int n=0;n<N;n++)bK[k*N+n]=B[(r*Ks+k)*N+n];
        HC(hipMalloc(&dA[r],M*Ks*4));HC(hipMalloc(&dB[r],Ks*N*4));HC(hipMalloc(&dY[r],M*N*4));
        HC(hipMemcpy(dA[r],aK.data(),M*Ks*4,hipMemcpyHostToDevice));HC(hipMemcpy(dB[r],bK.data(),Ks*N*4,hipMemcpyHostToDevice)); }
    for(int r=0;r<P;r++){ HC(hipSetDevice(r)); dim3 b(16,16),g((N+15)/16,(M+15)/16); gemm<<<g,b,0,st[r]>>>(dA[r],dB[r],dY[r],M,N,Ks); }
    for(int r=0;r<P;r++){ HC(hipSetDevice(r)); HC(hipStreamSynchronize(st[r])); }  // finish GEMMs before collective
    for(int r=0;r<P;r++){ HC(hipSetDevice(r)); HC(hipDeviceSynchronize()); } printf("[ck] gemms done\n");
    NC(ncclGroupStart()); for(int r=0;r<P;r++) NC(ncclAllReduce(dY[r],dY[r],M*N,ncclFloat,ncclSum,comm[r],sc[r])); NC(ncclGroupEnd()); printf("[ck] allreduce enqueued\n");
    for(int r=0;r<P;r++){ HC(hipSetDevice(r)); HC(hipStreamSynchronize(sc[r])); }
    HC(hipSetDevice(0)); std::vector<float> y(M*N); HC(hipMemcpy(y.data(),dY[0],M*N*4,hipMemcpyDeviceToHost));
    double sa=0,sr=0; for(int k=0;k<M*N;k++){sa+=std::abs(y[k]-ref[k]);sr+=std::abs(ref[k]);}
    double rel=sa/std::max(sr,1e-30); int fail=rel>=1e-4;
    printf("gemm_ar vs single-GPU ref: rel %.4g (%s)\n",rel,fail?"FAIL":"PASS");
    for(int r=0;r<P;r++) ncclCommDestroy(comm[r]);
    printf("%s\n",fail?"FAILED":"ALL PASS"); return fail;
}
