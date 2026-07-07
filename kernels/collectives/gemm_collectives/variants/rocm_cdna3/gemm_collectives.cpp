/**
 * @file
 * @brief CDNA3 fused collective+GEMM (tensor/sequence parallel), correct ROCm
 * semantics for the CUDA parallel/{ag_gemm,gemm_rs,gemm_ar} kernels. Those fuse
 * NVIDIA multimem/NVLS collectives with a GEMM for compute/comm overlap; gfx942
 * lacks NVLS, so this composes RCCL (the correct collective) with a local GEMM.
 * Overlap (streamed tiles / Iris XGMI) is the perf follow-up; here we validate
 * the parallel math across GPUs vs a single-GPU reference.
 *
 *   gemm_ar : K-parallel. rank r has A[:,Kr], B[Kr,:]; Y = allreduce(A_r@B_r) = A@B
 *   ag_gemm : M-parallel. rank r has A[Mr,:]; Y = allgather(A)@B = A@B (replicated)
 *   gemm_rs : K-parallel + M-scatter. Y[Mr,:] = reduce_scatter_M(A_r@B_r)
 *
 *   hipcc -std=c++17 -O3 -I/opt/rocm/include/rccl gemm_collectives.cpp -lrccl -o gc.out
 */
#include <hip/hip_runtime.h>
#include <rccl.h>
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#define HC(x) do{hipError_t e=(x); if(e!=hipSuccess){printf("HIP %s @%d\n",hipGetErrorString(e),__LINE__);return 1;}}while(0)
#define NC(x) do{ncclResult_t r=(x); if(r!=ncclSuccess){printf("RCCL %s @%d\n",ncclGetErrorString(r),__LINE__);return 1;}}while(0)

// naive GEMM C[M,N] = A[M,K]@B[K,N] (row-major), one thread per output.
__global__ void gemm(const float* A,const float* B,float* C,int M,int N,int K){
    int m=blockIdx.y*blockDim.y+threadIdx.y, n=blockIdx.x*blockDim.x+threadIdx.x;
    if(m>=M||n>=N) return; float s=0; for(int k=0;k<K;k++) s+=A[(size_t)m*K+k]*B[(size_t)k*N+n];
    C[(size_t)m*N+n]=s;
}
static void launch_gemm(const float*A,const float*B,float*C,int M,int N,int K,hipStream_t st){
    dim3 b(16,16), g((N+15)/16,(M+15)/16); gemm<<<g,b,0,st>>>(A,B,C,M,N,K);
}

int main(){ setvbuf(stdout,NULL,_IOLBF,0);
    int nd=0; HC(hipGetDeviceCount(&nd)); int P=(nd>=4)?4:nd;   // use 4 ranks
    const int M=64,N=64,K=128, Ks=K/P, Ms=M/P;
    printf("== fused collective+GEMM across %d GPUs  M=%d N=%d K=%d\n",P,M,N,K);
    std::mt19937 rng(1); std::normal_distribution<float> ndist(0,1);
    std::vector<float> A(M*K),B(K*N); for(auto&x:A)x=ndist(rng); for(auto&x:B)x=ndist(rng);
    std::vector<double> ref(M*N,0);   // single-GPU reference A@B
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){double s=0;for(int k=0;k<K;k++)s+=(double)A[m*K+k]*B[k*N+n]; ref[m*N+n]=s;}

    std::vector<int> devs(P); for(int i=0;i<P;i++)devs[i]=i;
    std::vector<ncclComm_t> comm(P); NC(ncclCommInitAll(comm.data(),P,devs.data()));
    std::vector<hipStream_t> st(P);
    std::vector<float*> dAar(P),dBar(P),dYar(P),dAag(P),dAsend(P),dB(P),dYag(P),dPart(P),dYrs(P);
    for(int r=0;r<P;r++){ HC(hipSetDevice(r)); HC(hipStreamCreate(&st[r]));
        // gemm_ar: A[:,Kr] (M x Ks), B[Kr,:] (Ks x N)
        std::vector<float> aK(M*Ks),bK(Ks*N);
        for(int m=0;m<M;m++)for(int k=0;k<Ks;k++)aK[m*Ks+k]=A[m*K+r*Ks+k];
        for(int k=0;k<Ks;k++)for(int n=0;n<N;n++)bK[k*N+n]=B[(r*Ks+k)*N+n];
        HC(hipMalloc(&dAar[r],M*Ks*4)); HC(hipMalloc(&dBar[r],Ks*N*4)); HC(hipMalloc(&dYar[r],M*N*4));
        HC(hipMemcpy(dAar[r],aK.data(),M*Ks*4,hipMemcpyHostToDevice)); HC(hipMemcpy(dBar[r],bK.data(),Ks*N*4,hipMemcpyHostToDevice));
        // ag_gemm: A[Mr,:] (Ms x K), full B; gathered A buffer (M x K)
        HC(hipMalloc(&dAag[r],(size_t)M*K*4)); HC(hipMalloc(&dB[r],(size_t)K*N*4)); HC(hipMalloc(&dYag[r],(size_t)M*N*4));
        HC(hipMalloc(&dAsend[r],(size_t)Ms*K*4));   // dedicated all-gather send buffer (non-in-place)
        HC(hipMemcpy(dAsend[r], A.data()+(size_t)r*Ms*K, (size_t)Ms*K*4, hipMemcpyHostToDevice));
        HC(hipMemcpy(dB[r],B.data(),(size_t)K*N*4,hipMemcpyHostToDevice));
        HC(hipMalloc(&dPart[r],(size_t)M*N*4)); HC(hipMalloc(&dYrs[r],(size_t)Ms*N*4));
    }
    // ---- gemm_ar: local partial GEMM (K-shard) then all-reduce ----
    for(int r=0;r<P;r++){ HC(hipSetDevice(r)); launch_gemm(dAar[r],dBar[r],dYar[r],M,N,Ks,st[r]); }
    NC(ncclGroupStart()); for(int r=0;r<P;r++) NC(ncclAllReduce(dYar[r],dYar[r],M*N,ncclFloat,ncclSum,comm[r],st[r])); NC(ncclGroupEnd());
    // ---- ag_gemm: all-gather A shards then full GEMM ----
    NC(ncclGroupStart()); for(int r=0;r<P;r++) NC(ncclAllGather(dAsend[r], dAag[r], (size_t)Ms*K, ncclFloat, comm[r], st[r])); NC(ncclGroupEnd());
    for(int r=0;r<P;r++){ HC(hipSetDevice(r)); launch_gemm(dAag[r],dB[r],dYag[r],M,N,K,st[r]); }
    // ---- gemm_rs: local partial GEMM (K-shard, full M) then reduce-scatter over M ----
    for(int r=0;r<P;r++){ HC(hipSetDevice(r)); launch_gemm(dAar[r],dBar[r],dPart[r],M,N,Ks,st[r]); }
    NC(ncclGroupStart()); for(int r=0;r<P;r++) NC(ncclReduceScatter(dPart[r],dYrs[r],(size_t)Ms*N,ncclFloat,ncclSum,comm[r],st[r])); NC(ncclGroupEnd());
    for(int r=0;r<P;r++){ HC(hipSetDevice(r)); HC(hipStreamSynchronize(st[r])); }

    auto chk=[&](const char*nm,std::vector<float>&got,const double*rf,int rows)->int{
        double sa=0,sr=0; for(int k=0;k<rows*N;k++){sa+=std::abs(got[k]-rf[k]);sr+=std::abs(rf[k]);}
        double rel=sa/std::max(sr,1e-30); printf("%-9s rel %.4g (%s)\n",nm,rel,rel<1e-4?"PASS":"FAIL"); return rel<1e-4?0:1; };
    int fail=0;
    { HC(hipSetDevice(0)); std::vector<float> y(M*N); HC(hipMemcpy(y.data(),dYar[0],M*N*4,hipMemcpyDeviceToHost)); fail+=chk("gemm_ar",y,ref.data(),M); }
    { HC(hipSetDevice(0)); std::vector<float> y(M*N); HC(hipMemcpy(y.data(),dYag[0],M*N*4,hipMemcpyDeviceToHost)); fail+=chk("ag_gemm",y,ref.data(),M); }
    { HC(hipSetDevice(1)); std::vector<float> y(Ms*N); HC(hipMemcpy(y.data(),dYrs[1],Ms*N*4,hipMemcpyDeviceToHost));
      std::vector<double> rf(Ms*N); for(int i=0;i<Ms*N;i++) rf[i]=ref[(size_t)1*Ms*N+i]; fail+=chk("gemm_rs",y,rf.data(),Ms); }
    for(int r=0;r<P;r++) ncclCommDestroy(comm[r]);
    printf("%s\n", fail?"FAILED":"ALL PASS"); return fail?1:0;
}
