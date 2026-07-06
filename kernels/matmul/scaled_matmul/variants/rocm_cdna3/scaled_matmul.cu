#include "kittens.cuh"
#include <random>
#include <omp.h>
#include <chrono>

using namespace kittens;

#include "./profile_utils.cpp"

constexpr int NUM_WARPS = 8;

using G = kittens::group<NUM_WARPS>;
using c_dtype = float;

template <int M, int N, int K>
__global__ __launch_bounds__(512, 2) void scaled_matmul(const kittens::gl<fp8e4m3, 1, 1, M, K> A, const kittens::gl<fp8e4m3, 1, 1, N, K> B, const kittens::gl<c_dtype, 1, 1, M, N> C,
    const kittens::gl<c_dtype, 1, 1, 1, M> scale_a, const kittens::gl<c_dtype, 1, 1, 1, N> scale_b) {
    constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
    constexpr int BLOCK_M = 256;
    constexpr int BLOCK_N = 256;
    constexpr int BLOCK_K = 128;
    constexpr int REG_MN = 64;
    constexpr int REG_K = 32;
    constexpr int blocks_per_col = N / BLOCK_N;
    constexpr int k_iters = K / BLOCK_K;

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    auto (&As) = al.allocate<st<fp8e4m3, BLOCK_M, BLOCK_K>>();
    auto (&Bs) = al.allocate<st<fp8e4m3, BLOCK_N, BLOCK_K>>();

    using RT_C = rt_fl<REG_MN, REG_MN, col_l>;
    rt<fp8e4m3, REG_MN, REG_K> tiles[8];
    RT_C c_accum[2];
    zero(c_accum[0]);
    zero(c_accum[1]);

    const int global_block_id = blockIdx.x;
    const int output_m = global_block_id / blocks_per_col;
    const int output_n = global_block_id % blocks_per_col;
    const int warp_id = warpid();
    const int warp_row = warp_id / 4;
    const int warp_col = warp_id % 4;

    G::load(As, A, {0, 0, output_m, 0});
    G::load(Bs, B, {0, 0, output_n, 0});
    __builtin_amdgcn_s_barrier();

    if (warp_row == 1) {
        __builtin_amdgcn_s_barrier();
    }

    for (int k_tile = 0; k_tile < k_iters - 1; ++k_tile) {
        constexpr int buffer_size = (BLOCK_M * BLOCK_K) / NUM_THREADS / sizeof(float4) / sizeof(fp8e4m3);
        float4 a_buffer_next[buffer_size];
        float4 b_buffer_next[buffer_size];

        load_global_to_register_buffer<2, false, NUM_THREADS>(a_buffer_next, buffer_size, A, {0, 0, output_m, k_tile + 1}, As);
        load(tiles[1], subtile_inplace<REG_MN, REG_K>(As, {warp_row, 0}));
        load(tiles[2], subtile_inplace<REG_MN, REG_K>(As, {warp_row + 2, 0}));
        load(tiles[0], subtile_inplace<REG_MN, REG_K>(Bs, {warp_col, 0}));
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(c_accum[0], tiles[1], tiles[0], c_accum[0]);
        mma_ABt(c_accum[1], tiles[2], tiles[0], c_accum[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        load(tiles[3], subtile_inplace<REG_MN, REG_K>(Bs, {warp_col, 1}));
        load(tiles[4], subtile_inplace<REG_MN, REG_K>(As, {warp_row, 1}));
        load(tiles[5], subtile_inplace<REG_MN, REG_K>(As, {warp_row + 2, 1}));
        load(tiles[0], subtile_inplace<REG_MN, REG_K>(Bs, {warp_col, 2}));
        load(tiles[1], subtile_inplace<REG_MN, REG_K>(As, {warp_row, 2}));
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(c_accum[0], tiles[4], tiles[3], c_accum[0]);
        mma_ABt(c_accum[1], tiles[5], tiles[3], c_accum[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        load_global_to_register_buffer<2, false, NUM_THREADS>(b_buffer_next, buffer_size, B, {0, 0, output_n, k_tile + 1}, Bs);
        load(tiles[2], subtile_inplace<REG_MN, REG_K>(As, {warp_row + 2, 2}));
        load(tiles[6], subtile_inplace<REG_MN, REG_K>(Bs, {warp_col, 3}));
        load(tiles[7], subtile_inplace<REG_MN, REG_K>(As, {warp_row, 3}));
        load(tiles[5], subtile_inplace<REG_MN, REG_K>(As, {warp_row + 2, 3}));
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_s_setprio(1);
        mma_ABt(c_accum[0], tiles[1], tiles[0], c_accum[0]);
        mma_ABt(c_accum[1], tiles[2], tiles[0], c_accum[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        asm volatile("s_waitcnt lgkmcnt(0)");
        store_register_buffer_to_shared<NUM_THREADS>(As, a_buffer_next);
        store_register_buffer_to_shared<NUM_THREADS>(Bs, b_buffer_next);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_s_setprio(1);
        mma_ABt(c_accum[0], tiles[7], tiles[6], c_accum[0]);
        mma_ABt(c_accum[1], tiles[5], tiles[6], c_accum[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    }

    load(tiles[0], subtile_inplace<REG_MN, REG_K>(Bs, {warp_col, 0}));
    load(tiles[1], subtile_inplace<REG_MN, REG_K>(As, {warp_row, 0}));
    load(tiles[2], subtile_inplace<REG_MN, REG_K>(As, {warp_row + 2, 0}));
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    __builtin_amdgcn_s_setprio(1);
    mma_ABt(c_accum[0], tiles[1], tiles[0], c_accum[0]);
    mma_ABt(c_accum[1], tiles[2], tiles[0], c_accum[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    load(tiles[3], subtile_inplace<REG_MN, REG_K>(Bs, {warp_col, 1}));
    load(tiles[4], subtile_inplace<REG_MN, REG_K>(As, {warp_row, 1}));
    load(tiles[5], subtile_inplace<REG_MN, REG_K>(As, {warp_row + 2, 1}));
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    __builtin_amdgcn_s_setprio(1);
    mma_ABt(c_accum[0], tiles[4], tiles[3], c_accum[0]);
    mma_ABt(c_accum[1], tiles[5], tiles[3], c_accum[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    load(tiles[0], subtile_inplace<REG_MN, REG_K>(Bs, {warp_col, 2}));
    load(tiles[1], subtile_inplace<REG_MN, REG_K>(As, {warp_row, 2}));
    load(tiles[2], subtile_inplace<REG_MN, REG_K>(As, {warp_row + 2, 2}));
    load(tiles[3], subtile_inplace<REG_MN, REG_K>(Bs, {warp_col, 3}));
    load(tiles[4], subtile_inplace<REG_MN, REG_K>(As, {warp_row, 3}));
    load(tiles[5], subtile_inplace<REG_MN, REG_K>(As, {warp_row + 2, 3}));
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    __builtin_amdgcn_s_setprio(1);
    mma_ABt(c_accum[0], tiles[1], tiles[0], c_accum[0]);
    mma_ABt(c_accum[1], tiles[2], tiles[0], c_accum[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    __builtin_amdgcn_s_setprio(1);
    mma_ABt(c_accum[0], tiles[4], tiles[3], c_accum[0]);
    mma_ABt(c_accum[1], tiles[5], tiles[3], c_accum[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    typename RT_C::col_vec scale_a0_rv;
    typename RT_C::col_vec scale_a1_rv;
    typename RT_C::row_vec scale_b_rv;
    load(scale_a0_rv, scale_a, {output_m * 4 + warp_row});
    load(scale_a1_rv, scale_a, {output_m * 4 + warp_row + 2});
    load(scale_b_rv, scale_b, {output_n * 4 + warp_col});
    asm volatile("s_waitcnt vmcnt(0)");

    mul_col(c_accum[0], c_accum[0], scale_b_rv);
    mul_row(c_accum[0], c_accum[0], scale_a0_rv);
    mul_col(c_accum[1], c_accum[1], scale_b_rv);
    mul_row(c_accum[1], c_accum[1], scale_a1_rv);

    if (warp_row == 0) {
        __builtin_amdgcn_s_barrier();
    }
    store(C, c_accum[0], {0, 0, output_m * 4 + warp_row,     output_n * 4 + warp_col});
    store(C, c_accum[1], {0, 0, output_m * 4 + warp_row + 2, output_n * 4 + warp_col});
}

// Rotating buffer configuration (global constant)
constexpr int ROTATING_BUFFER_COUNT = ((((1024*1024)/8192)*512)/8192)/2; // 500 MiB

// Random initialization function
template <int M, int N, int K>
void random_init(std::vector<fp8e4m3>& a_host, std::vector<fp8e4m3>& b_host, std::vector<c_dtype>& scale_a_host,
    std::vector<c_dtype>& scale_b_host, uint32_t seed = 42) {
    std::mt19937 gen(seed); // Seed for reproducibility
    std::normal_distribution<float> dis(-1.0f, 1.0f);
    for (int i = 0; i < M*K; i++) {
        a_host[i] = fp8e4m3(dis(gen));
    }
    for (int i = 0; i < N*K; i++) {
        b_host[i] = fp8e4m3(dis(gen));
    }
    for (int i = 0; i < M; i++) {
        scale_a_host[i] = c_dtype(1.0f + 100.1f * dis(gen));
    }
    for (int i = 0; i < N; i++) {
        scale_b_host[i] = c_dtype(1.0f + 100.1f * dis(gen));
    }
}

template <int M, int N, int K, int CUs>
TimingResult matmul_host(std::vector<fp8e4m3>& a, std::vector<fp8e4m3>& b, std::vector<c_dtype>& c, 
    std::vector<float>& scale_a, std::vector<float>& scale_b,
    int warmup_iters = 3, int timing_iters = 20) {
    constexpr int threads_per_warp = 64;
    constexpr int warps_per_cu = 8;
    constexpr int threads_per_block = threads_per_warp * warps_per_cu;
    constexpr int shared_mem_bytes = 65536;
    
    // Use global rotating buffer configuration
    constexpr int block_count = ROTATING_BUFFER_COUNT;
    
    // Ensure input vectors have correct size
    if (a.size() != M * K) {
        fprintf(stderr, "Error: Input vector 'a' size %zu does not match expected M*K=%d\n", a.size(), M*K);
        return {0, 0, 0, 0, 0};
    }
    if (b.size() != N * K) {
        fprintf(stderr, "Error: Input vector 'b' size %zu does not match expected N*K=%d\n", b.size(), N*K);
        return {0, 0, 0, 0, 0};
    }
    if (scale_a.size() != M) {
        fprintf(stderr, "Error: Input vector 'scale_a' size %zu does not match expected M=%d\n", scale_a.size(), M);
        return {0, 0, 0, 0, 0};
    }
    if (scale_b.size() != N) {
        fprintf(stderr, "Error: Input vector 'scale_b' size %zu does not match expected N=%d\n", scale_b.size(), N);
        return {0, 0, 0, 0, 0};
    }
    
    // Resize output vector
    c.resize(M * N);
    hipFuncSetAttribute((void*)scaled_matmul<M, N, K>, hipFuncAttributeMaxDynamicSharedMemorySize, shared_mem_bytes);
    HipCheckError();
    
    // Allocate device memory (A and B use rotating buffers, C is single buffer)
    fp8e4m3 *d_a, *d_b;
    c_dtype *d_c;
    c_dtype * d_scale_a, *d_scale_b;
    hipMalloc(&d_a, block_count * M*K*sizeof(fp8e4m3));
    hipMalloc(&d_b, block_count * N*K*sizeof(fp8e4m3));
    hipMalloc(&d_c, M*N*sizeof(c_dtype));  // Single buffer (output only, no rotation needed)
    hipMalloc(&d_scale_a, block_count * M * sizeof(c_dtype));
    hipMalloc(&d_scale_b, block_count * N * sizeof(c_dtype));
    HipCheckError();
    
    // Pre-initialize all buffer sections with random data on host
    printf("Initializing %d rotating buffer sections (%zu MB total, A+B only)...\n",
           block_count,
           (block_count * (M*K*sizeof(fp8e4m3) + N*K*sizeof(fp8e4m3) + M*sizeof(c_dtype) + N*sizeof(c_dtype)) + M*N*sizeof(c_dtype)) / (1024*1024));

    for (int block = 0; block < block_count; ++block) {
        // Generate random data with different seed for each buffer
        random_init<M, N, K>(a, b, scale_a, scale_b, 42 + block);
        // Print the maximum value from each of a and b for this block
        fp8e4m3 max_a = a[0];
        fp8e4m3 max_b = b[0];
        #pragma omp parallel for
        for (int i = 1; i < M*K; ++i) {
            if ((float)a[i] > (float)max_a) max_a = a[i];
        }
        #pragma omp parallel for
        for (int i = 1; i < N*K; ++i) {
            if ((float)b[i] > (float)max_b) max_b = b[i];
        }
        printf("Block %d: max(a) = %f, max(b) = %f\n", block, (float)max_a, (float)max_b);

        // Copy to offset position in device memory
        hipMemcpy(d_a + block * M * K, a.data(), M*K*sizeof(fp8e4m3), hipMemcpyHostToDevice);
        hipMemcpy(d_b + block * N * K, b.data(), N*K*sizeof(fp8e4m3), hipMemcpyHostToDevice);
        hipMemcpy(d_scale_a + block * M, scale_a.data(), M * sizeof(c_dtype), hipMemcpyHostToDevice);
        hipMemcpy(d_scale_b + block * N, scale_b.data(), N * sizeof(c_dtype), hipMemcpyHostToDevice);
    }
    HipCheckError();
    printf("Buffer initialization complete.\n");
    
    // Warmup iterations using rotating buffers (A and B only)
    for (int i = 0; i < warmup_iters; i++) {
        int block_idx = i % block_count;
        fp8e4m3* d_a_current = d_a + block_idx * M * K;
        fp8e4m3* d_b_current = d_b + block_idx * N * K;
        c_dtype* d_scale_a_current = d_scale_a + block_idx * M;
        c_dtype* d_scale_b_current = d_scale_b + block_idx * N;

        hipMemset(d_c, 0, M*N*sizeof(c_dtype));

        kittens::gl<fp8e4m3, 1, 1, M, K> A(d_a_current, nullptr, nullptr, nullptr, nullptr);
        kittens::gl<fp8e4m3, 1, 1, N, K> B(d_b_current, nullptr, nullptr, nullptr, nullptr);
        kittens::gl<c_dtype, 1, 1, M, N> C(d_c, nullptr, nullptr, nullptr, nullptr);
        kittens::gl<c_dtype, 1, 1, 1, M> scale_a_gl(d_scale_a_current, nullptr, nullptr, nullptr, nullptr);
        kittens::gl<c_dtype, 1, 1, 1, N> scale_b_gl(d_scale_b_current, nullptr, nullptr, nullptr, nullptr);

        scaled_matmul<M, N, K><<<(M*N/(256*256)), threads_per_block, shared_mem_bytes>>>(A, B, C, scale_a_gl, scale_b_gl);
        HipCheckError();
        hipDeviceSynchronize();
    }
    
    // Create HIP events for precise kernel timing
    hipEvent_t start_event, stop_event;
    hipEventCreate(&start_event);
    hipEventCreate(&stop_event);
    
    // Timed kernel-only loop with rotating buffers (A and B only)
    std::vector<float> times_ms;
    times_ms.reserve(timing_iters);
    for (int r = 0; r < timing_iters; ++r) {
        // Calculate which buffer section to use for inputs (modulo rotation)
        int block_idx = r % block_count;
        fp8e4m3* d_a_current = d_a + block_idx * M * K;
        fp8e4m3* d_b_current = d_b + block_idx * N * K;
        c_dtype* d_scale_a_current = d_scale_a + block_idx * M;
        c_dtype* d_scale_b_current = d_scale_b + block_idx * N;

        // Zero out the single output buffer
        hipMemset(d_c, 0, M*N*sizeof(c_dtype));

        // Create gl wrappers (A and B rotate, C is always the same)
        kittens::gl<fp8e4m3, 1, 1, M, K> A_current(d_a_current, nullptr, nullptr, nullptr, nullptr);
        kittens::gl<fp8e4m3, 1, 1, N, K> B_current(d_b_current, nullptr, nullptr, nullptr, nullptr);
        kittens::gl<c_dtype, 1, 1, M, N> C_current(d_c, nullptr, nullptr, nullptr, nullptr);
        kittens::gl<c_dtype, 1, 1, 1, M> scale_a_current(d_scale_a_current, nullptr, nullptr, nullptr, nullptr);
        kittens::gl<c_dtype, 1, 1, 1, N> scale_b_current(d_scale_b_current, nullptr, nullptr, nullptr, nullptr);

        hipEventRecord(start_event, 0);
        scaled_matmul<M, N, K><<<(M*N/(256*256)), threads_per_block, shared_mem_bytes>>>(A_current, B_current, C_current, scale_a_current, scale_b_current);
        hipEventRecord(stop_event, 0);
        hipEventSynchronize(stop_event);
        float ms = 0.0f;
        hipEventElapsedTime(&ms, start_event, stop_event);
        times_ms.push_back(ms);
        HipCheckError();
    }
    
    // Calculate best and average times
    float sum_ms = 0.f, best_ms = 1e30f;
    for (float t : times_ms) { 
        sum_ms += t; 
        best_ms = std::min(best_ms, t); 
    }
    float avg_ms = sum_ms / times_ms.size();
    
    // Calculate TFLOPS (2*M*N*K operations)
    double total_ops = 2.0 * M * N * K;
    double best_tflops = (total_ops / (best_ms * 1e-3)) / 1e12;
    double avg_tflops = (total_ops / (avg_ms * 1e-3)) / 1e12;
    
    // Cleanup events
    hipEventDestroy(start_event);
    hipEventDestroy(stop_event);
    HipCheckError();

    // Copy result back to host (single C buffer, no offset needed)
    hipMemcpy(c.data(), d_c, M*N*sizeof(c_dtype), hipMemcpyDeviceToHost);
    HipCheckError();
    
    // Free device memory
    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);
    hipFree(d_scale_a);
    hipFree(d_scale_b);
    HipCheckError();
    
    return {best_ms, avg_ms, best_tflops, avg_tflops, timing_iters};
}


int main() {
    // Reduced problem size for faster timing
    constexpr int M = 8192;  // 256 threadblocks needed for 2048x2048
    constexpr int N = 8192;  
    constexpr int K = 8192;  // Smaller K for reasonable timing
    constexpr int CUs = 256; // 256 threadblocks (1 outer iteration)
    
    // Timing parameters to keep total runtime reasonable
    constexpr int warmup_iters = 500;
    constexpr int timing_iters = 100;

    printf("Matrix dimensions: %dx%dx%d, CUs: %d\n", M, N, K, CUs);
    printf("Warmup iterations: %d, Timing iterations: %d\n\n", warmup_iters, timing_iters);

    // Initialize input matrices (will be initialized inside matmul_host)
    std::vector<fp8e4m3> a_host(M*K);
    std::vector<fp8e4m3> b_host(N*K);
    std::vector<c_dtype> scale_a_host(M);
    std::vector<c_dtype> scale_b_host(N);
    std::vector<c_dtype> c_ref(M*N);
    std::vector<c_dtype> c_host(M*N);

    // Compute test result with timing (data will be generated with rotating buffers)
    printf("Running optimized kernel (scaled_matmul)...\n");
    TimingResult host_timing = matmul_host<M, N, K, CUs>(a_host, b_host, c_host, scale_a_host, scale_b_host, warmup_iters, timing_iters);

    // Initialize data for reference computation (match the buffer used in last timing iteration)
    int last_buffer_idx = (timing_iters - 1) % ROTATING_BUFFER_COUNT;
    random_init<M, N, K>(a_host, b_host, scale_a_host, scale_b_host, 42 + last_buffer_idx);

    // Compute reference result with timing
    printf("Running reference kernel (scaled_matmul_ref)...\n");
    // Reference CPU implementation of matmul with OpenMP
    auto matmul_ref = [](const std::vector<fp8e4m3>& a_host, const std::vector<fp8e4m3>& b_host, std::vector<c_dtype>& c_ref,
        std::vector<float>& scale_a, std::vector<float>& scale_b, int timing_iters) -> TimingResult {
        TimingResult result{};
        // Only time the main loop, not the warmup
        double best_time_ms = 1e30, total_time_ms = 0.0;
        int M = a_host.size() / K;
        int N = b_host.size() / K;
        // Timing
        for (int t = 0; t < timing_iters; ++t) {
            auto t0 = std::chrono::high_resolution_clock::now();
            #pragma omp parallel for
            for (int row = 0; row < M; ++row) {
                for (int col = 0; col < N; ++col) {
                    float acc = 0.0f;
                    for (int k = 0; k < K; ++k) {
                        acc += float(a_host[row * K + k]) * scale_a[row] * float(b_host[col * K + k]) * scale_b[col];
                    }
                    c_ref[row * N + col] = c_dtype(acc);
                }
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            if (ms < best_time_ms) best_time_ms = ms;
            total_time_ms += ms;
        }
        double avg_time_ms = total_time_ms / timing_iters;
        double tflops = 2.0 * M * N * K / 1e12;
        result.best_time_ms = best_time_ms;
        result.avg_time_ms = avg_time_ms;
        result.best_tflops = tflops / (best_time_ms / 1000.0);
        result.avg_tflops = tflops / (avg_time_ms / 1000.0);
        return result;
    };

    TimingResult ref_timing = matmul_ref(a_host, b_host, c_ref, scale_a_host, scale_b_host, 1);

    bool success = true;
    // Compare GPU result (c_host) with CPU reference (c_ref)
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            // c_host is row major: [row*N + col]
            // c_ref is row major: [row*N + col]
            float c_val = float(c_host[row * N + col]);
            float c_ref_val = float(c_ref[row * N + col]);
            float diff = std::abs(c_val - c_ref_val);
            float threshold = std::abs(c_ref_val * 0.01f);
            if (diff > threshold) {
                printf("Mismatch at (row=%d, col=%d): c_host = %f, c_ref = %f, diff = %f\n", row, col, c_val, c_ref_val, diff);
                success = false;
                break;
            }
        }
        if (!success) {
            break;
        }
    }
    // Performance comparison and results
    printf("\n=== PERFORMANCE RESULTS ===\n");
    
    printf("Reference kernel (scaled_matmul_ref):\n");
    printf("  Kernel time (best): %.3f ms,  TFLOPS: %.2f\n", ref_timing.best_time_ms, ref_timing.best_tflops);
    printf("  Kernel time (avg ): %.3f ms,  TFLOPS: %.2f\n", ref_timing.avg_time_ms, ref_timing.avg_tflops);
    
    printf("\nOptimized kernel (scaled_matmul):\n");
    printf("  Kernel time (best): %.3f ms,  TFLOPS: %.2f\n", host_timing.best_time_ms, host_timing.best_tflops);
    printf("  Kernel time (avg ): %.3f ms,  TFLOPS: %.2f\n", host_timing.avg_time_ms, host_timing.avg_tflops);
    
    printf("\nSpeedup (best): %.2fx\n", ref_timing.best_time_ms / host_timing.best_time_ms);
    printf("Speedup (avg ): %.2fx\n", ref_timing.avg_time_ms / host_timing.avg_time_ms);
    
    if (success) {
        printf("\nCorrectness: PASSED\n");
    } else {
        printf("\nCorrectness: FAILED\n");
    }

    return 0;
}
