#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

#define NUM_WORKERS (1) 
#define NUM_THREADS (NUM_WORKERS*kittens::WARP_THREADS)

constexpr int ATTN_B = 16;
constexpr int ATTN_H = 16;
constexpr int ATTN_N = 2048;
constexpr int HEAD_DIM = 128;                   
constexpr float rope_embd_fraction = 1.0f;

constexpr int ROPE_DIM = HEAD_DIM;
constexpr int HALF_ROPE_DIM = 64;
constexpr int EXCESS_DIM = HEAD_DIM - ROPE_DIM; 
constexpr int BLOCK_SIZE = 32;

using namespace kittens;

#define tile_1xEXCESS_ROPE_D st<bf16, BLOCK_SIZE, EXCESS_DIM>
#define reg_tile_1xEXCESS_ROPE_D rt<bf16, BLOCK_SIZE, EXCESS_DIM>

template<int _d_model> struct rotary_globals {
    static constexpr int d_model = _d_model;
    using x_gl = gl<bf16, -1, -1, -1, -1>;
    using o_gl = gl<bf16, -1, -1, -1, -1>;
    using sin_gl = gl<bf16, -1, -1, -1, -1>;
    using cos_gl = gl<bf16, -1, -1, -1, -1>;
    x_gl x;
    o_gl o;
    sin_gl sin;
    cos_gl cos;
    hipStream_t stream;

    dim3 grid() { return dim3(ATTN_H, (ATTN_B + NUM_WORKERS - 1) / NUM_WORKERS, ATTN_N / BLOCK_SIZE); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return (0); }
};

template<int D> 
__global__ void tk_fused_rotary(const rotary_globals<D> g) {
    auto warpid = kittens::warpid();
    auto lane = kittens::laneid();
    const int b = blockIdx.y * NUM_WORKERS + kittens::warpid();
    const int h = blockIdx.x;
    const int n = blockIdx.z;

    using x_tile_t = rt<bf16, BLOCK_SIZE, ROPE_DIM, row_l>;
    using trig_tile_t = rt<bf16, BLOCK_SIZE, HALF_ROPE_DIM, row_l>;
    constexpr int half_dim_tiles = HALF_ROPE_DIM / x_tile_t::tile_size_col;
    constexpr int PT = x_tile_t::packed_per_base_tile;
    x_tile_t x_reg;
    trig_tile_t cos_reg, sin_reg;

    load<2>(cos_reg, g.cos, {0, 0, n, 0});
    load<2>(sin_reg, g.sin, {0, 0, n, 0});
    load<2>(x_reg, g.x, {b, h, n, 0});
    asm volatile("s_waitcnt lgkmcnt(0)");

    #pragma unroll
    for (int row_tile = 0; row_tile < x_tile_t::height; ++row_tile) {
        #pragma unroll
        for (int i = 0; i < half_dim_tiles; ++i) {
            #pragma unroll
            for (int j = 0; j < PT; ++j) {
                const auto x1 = x_reg.tiles[row_tile][i].data[j];
                const auto x2 = x_reg.tiles[row_tile][i + half_dim_tiles].data[j];
                const auto cos = cos_reg.tiles[row_tile][i].data[j];
                const auto sin = sin_reg.tiles[row_tile][i].data[j];
                x_reg.tiles[row_tile][i].data[j] = __hsub2(__hmul2(x1, cos), __hmul2(x2, sin));
                x_reg.tiles[row_tile][i + half_dim_tiles].data[j] = __hadd2(__hmul2(x2, cos), __hmul2(x1, sin));
            }
        }
    }
    store(g.o, x_reg, {b, h, n, 0}); 
}

template<int D>
void dispatch_rotary(rotary_globals<D> g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)tk_fused_rotary<D>, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    tk_fused_rotary<D><<<g.grid(), g.block(), mem_size, g.stream>>>(g);
}

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "tk_kernel python module";
    py::bind_function<dispatch_rotary<HEAD_DIM>>(m, "dispatch_rotary", 
        &rotary_globals<HEAD_DIM>::x, 
        &rotary_globals<HEAD_DIM>::o, 
        &rotary_globals<HEAD_DIM>::sin, 
        &rotary_globals<HEAD_DIM>::cos
    );
}
