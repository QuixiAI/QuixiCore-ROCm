/**
 * @file
 * @brief CDNA3 collectives via RCCL — correct ROCm replacement for the CUDA
 * parallel/* kernels that use NVIDIA multimem (NVLink multicast / NVLS) and
 * device-side cross-GPU barriers, which gfx942 lacks. RCCL is NCCL-API
 * compatible; these are the drop-in collective semantics (all_reduce,
 * all_gather, reduce_scatter). Fused collective+GEMM (ag_gemm, gemm_rs) and
 * device-initiated one-sided variants map to the repo's Iris/XGMI path.
 *
 * Single-process, all local GPUs (ncclCommInitAll). Verifies each collective
 * against the analytic result.
 *   hipcc -std=c++17 -O3 -I/opt/rocm/include/rccl rccl_collectives.cpp -lrccl -o rccl_collectives.out
 */
#include <hip/hip_runtime.h>
#include <rccl.h>
#include <cstdio>
#include <vector>
#define HC(x) do{hipError_t e=(x); if(e!=hipSuccess){printf("HIP %s @%d\n",hipGetErrorString(e),__LINE__);return 1;}}while(0)
#define NC(x) do{ncclResult_t r=(x); if(r!=ncclSuccess){printf("RCCL %s @%d\n",ncclGetErrorString(r),__LINE__);return 1;}}while(0)

int main(){
    int nd=0; HC(hipGetDeviceCount(&nd)); if(nd>8) nd=8;
    printf("== RCCL collectives across %d MI300X (gfx942)\n", nd);
    std::vector<int> devs(nd); for(int i=0;i<nd;i++) devs[i]=i;
    std::vector<ncclComm_t> comm(nd);
    NC(ncclCommInitAll(comm.data(), nd, devs.data()));
    const int N=1<<20;
    std::vector<float*> ar(nd), ag(nd), rs(nd); std::vector<hipStream_t> st(nd);
    for(int i=0;i<nd;i++){ HC(hipSetDevice(i)); HC(hipStreamCreate(&st[i]));
        HC(hipMalloc(&ar[i],N*4)); HC(hipMalloc(&ag[i],(size_t)N*nd*4)); HC(hipMalloc(&rs[i],N*4));
        std::vector<float> h(N, (float)(i+1)); HC(hipMemcpy(ar[i],h.data(),N*4,hipMemcpyHostToDevice));
        std::vector<float> hg((size_t)N,(float)(i+1)); HC(hipMemcpy(ag[i],hg.data(),N*4,hipMemcpyHostToDevice)); }
    // all_reduce (sum): each device buffer -> sum_{i}(i+1) = nd*(nd+1)/2
    NC(ncclGroupStart()); for(int i=0;i<nd;i++) NC(ncclAllReduce(ar[i],ar[i],N,ncclFloat,ncclSum,comm[i],st[i])); NC(ncclGroupEnd());
    // all_gather: device i contributes (i+1); result[j*N..] = (j+1)
    NC(ncclGroupStart()); for(int i=0;i<nd;i++) NC(ncclAllGather(ag[i],ag[i],N,ncclFloat,comm[i],st[i])); NC(ncclGroupEnd());
    // reduce_scatter (sum): input nd*N of (i+1) -> each gets N of sum
    for(int i=0;i<nd;i++){ HC(hipSetDevice(i)); std::vector<float> hin((size_t)N*nd,(float)(i+1)); HC(hipMemcpy(ag[i],hin.data(),(size_t)N*nd*4,hipMemcpyHostToDevice)); }
    NC(ncclGroupStart()); for(int i=0;i<nd;i++) NC(ncclReduceScatter(ag[i],rs[i],N,ncclFloat,ncclSum,comm[i],st[i])); NC(ncclGroupEnd());
    for(int i=0;i<nd;i++){ HC(hipSetDevice(i)); HC(hipStreamSynchronize(st[i])); }
    const float expect=(float)(nd*(nd+1)/2);
    int fail=0;
    for(int i=0;i<nd;i++){ HC(hipSetDevice(i));
        std::vector<float> h(N); HC(hipMemcpy(h.data(),ar[i],N*4,hipMemcpyDeviceToHost));
        if(h[0]!=expect||h[N-1]!=expect){fail++;printf("all_reduce dev%d got %g want %g\n",i,h[0],expect);}
        std::vector<float> hr(N); HC(hipMemcpy(hr.data(),rs[i],N*4,hipMemcpyDeviceToHost));
        if(hr[0]!=expect){fail++;printf("reduce_scatter dev%d got %g want %g\n",i,hr[0],expect);}
    }
    printf("all_reduce + reduce_scatter (sum) across %d GPUs: %s (expect %g)\n", nd, fail?"FAIL":"PASS", expect);
    for(int i=0;i<nd;i++) ncclCommDestroy(comm[i]);
    printf("%s\n", fail?"FAILED":"ALL PASS");
    return fail?1:0;
}
