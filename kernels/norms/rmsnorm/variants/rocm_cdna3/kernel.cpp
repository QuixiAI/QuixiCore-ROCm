#include "kittens.cuh"
#include <hip/hip_cooperative_groups.h>
#include "pyutils/pyutils.cuh"


constexpr int B = 16;
constexpr int H = 16;
constexpr int N = 4096;
constexpr int D = 128; 

#define NUM_WORKERS (4) 
#define NUM_THREADS (NUM_WORKERS*kittens::WARP_THREADS)
#define ROWS_PER_WARP (8)
static_assert(ROWS_PER_WARP % 2 == 0 && ROWS_PER_WARP >= 2);

using G = kittens::group<NUM_WORKERS>;
using namespace kittens;


template <int _N> struct rmsnorm_globals{
    using x_gl = gl<bf16 , -1,-1,-1,-1>;
    using o_gl = gl<bf16 , -1,-1 , -1, -1>;
    using gamma_gl = gl<bf16 , -1 , -1 , -1 , -1>;

    x_gl x;
    o_gl o;
    gamma_gl gamma;
    float epsilon;

    static constexpr int n_per_tile = NUM_WORKERS * ROWS_PER_WARP;
    static constexpr int n_tile_size = N / n_per_tile;
    static_assert(N % n_per_tile == 0, "N must be divisible by NUM_WORKERS * ROWS_PER_WARP");

    dim3 grid() { return dim3(n_tile_size, B, 1); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return 0; }
}; 

#define COMPUTE_X_ONLY(x_reg, x_reg_squared, _D, epsilon) \
    do { \
        mul(x_reg_squared, x_reg, x_reg); \
        bf16 _x_var; \
        sum(_x_var, x_reg_squared); \
        float _var_f32 = __bfloat162float(_x_var) / float(_D); \
        float _inv_rms_f32 = rsqrtf(_var_f32 + epsilon); \
        bf16 _inv_rms = __float2bfloat16(_inv_rms_f32); \
        mul(x_reg, x_reg, _inv_rms); \
    } while(0)

template<int _D>
__global__ void rmsnorm_hk(
    const rmsnorm_globals<_D> g 
){
    static constexpr int LOADS_PER_RV  = rv<bf16, _D>::outer_dim;
    static constexpr int LOADS_PER_ROW = 2 * LOADS_PER_RV;
    static constexpr int PIPELINE      = 2 * LOADS_PER_ROW;
    static constexpr int WAIT_X        = PIPELINE - LOADS_PER_RV;
    static constexpr int WAIT_GAMMA    = PIPELINE - LOADS_PER_ROW;
    static constexpr int WAIT_X_LAST   = LOADS_PER_ROW - LOADS_PER_RV;
    static_assert(PIPELINE <= 63, "pipeline depth exceeds vmcnt range");

    auto warpid = kittens::warpid();
    const int batch = blockIdx.y;
    const int seq_start = blockIdx.x * g.n_per_tile + warpid * ROWS_PER_WARP;

    rv<bf16, _D> x_reg_a, x_reg_b, gamma_reg_a, gamma_reg_b, x_reg_squared;

    load(x_reg_a, g.x, {0, batch, seq_start, 0});
    load(gamma_reg_a, g.gamma, {0, batch, seq_start, 0});
    load(x_reg_b, g.x, {0, batch, seq_start + 1, 0});
    load(gamma_reg_b, g.gamma, {0, batch, seq_start + 1, 0});

    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP - 2; r += 2) {
        asm volatile("s_waitcnt vmcnt(%0)" :: "n"(WAIT_X));
        COMPUTE_X_ONLY(x_reg_a, x_reg_squared, _D, g.epsilon);
        asm volatile("s_waitcnt vmcnt(%0)" :: "n"(WAIT_GAMMA));
        mul(x_reg_a, x_reg_a, gamma_reg_a);
        store(g.o, x_reg_a, {0, batch, seq_start + r, 0});
        load(x_reg_a, g.x, {0, batch, seq_start + r + 2, 0});
        load(gamma_reg_a, g.gamma, {0, batch, seq_start + r + 2, 0});

        asm volatile("s_waitcnt vmcnt(%0)" :: "n"(WAIT_X));
        COMPUTE_X_ONLY(x_reg_b, x_reg_squared, _D, g.epsilon);
        asm volatile("s_waitcnt vmcnt(%0)" :: "n"(WAIT_GAMMA));
        mul(x_reg_b, x_reg_b, gamma_reg_b);
        store(g.o, x_reg_b, {0, batch, seq_start + r + 1, 0});
        load(x_reg_b, g.x, {0, batch, seq_start + r + 3, 0});
        load(gamma_reg_b, g.gamma, {0, batch, seq_start + r + 3, 0});
    }

    asm volatile("s_waitcnt vmcnt(%0)" :: "n"(WAIT_X));
    COMPUTE_X_ONLY(x_reg_a, x_reg_squared, _D, g.epsilon);
    asm volatile("s_waitcnt vmcnt(%0)" :: "n"(WAIT_GAMMA));
    mul(x_reg_a, x_reg_a, gamma_reg_a);
    store(g.o, x_reg_a, {0, batch, seq_start + ROWS_PER_WARP - 2, 0});

    asm volatile("s_waitcnt vmcnt(%0)" :: "n"(WAIT_X_LAST));
    COMPUTE_X_ONLY(x_reg_b, x_reg_squared, _D, g.epsilon);
    asm volatile("s_waitcnt vmcnt(0)");
    mul(x_reg_b, x_reg_b, gamma_reg_b);
    store(g.o, x_reg_b, {0, batch, seq_start + ROWS_PER_WARP - 1, 0});
}


template<int _D>
void dispatch_rmsnorm(rmsnorm_globals<_D> g) {  
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)rmsnorm_hk<_D>, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    rmsnorm_hk<_D><<<g.grid(), g.block(), mem_size>>>(g);
    hipDeviceSynchronize();
}

PYBIND11_MODULE(rms_norm_kernel, m) {
    m.doc() = "rms_norm_kernel python module";
    py::bind_function<dispatch_rmsnorm<D>>(m, "dispatch_rmsnorm", 
        &rmsnorm_globals<D>::x, 
        &rmsnorm_globals<D>::o, 
        &rmsnorm_globals<D>::gamma,
        &rmsnorm_globals<D>::epsilon
    );
}