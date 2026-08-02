/**
 * @file
 * @brief IQ2_XXS x q8_1 tiled GEMM for CDNA3 (gfx942), decoding at tile load.
 *
 * IQ2_XXS is an E8-lattice codebook quant at 2.0625 bpw. The block is 256
 * weights in 66 bytes:
 *
 *   struct { half d; uint16_t qs[32]; }
 *
 * and each group of 32 weights uses four uint16 of that: two carry four 8-bit
 * indices into a 256-entry grid (each entry is 8 packed magnitudes), the other
 * two carry four 7-bit sign-table indices and, in the top 4 bits, the group's
 * sub-scale. A weight is
 *
 *   grid_byte * (sign ? -1 : 1) * d * (0.5 + sub_scale) * 0.25
 *
 * The point of this kernel is WHERE that decode happens.
 *
 * The obvious tile design -- stage packed quants in LDS and decode inside the
 * inner dot, which is what every non-codebook format does -- is wrong here.
 * Decode for q4_0 or mxfp4 is a shift and a table-free arithmetic step, but for
 * IQ2_XXS it is four dependent random reads of a 2 KB grid plus a sign lookup,
 * and the inner dot runs once per (row, column) pair. Packing therefore
 * multiplies the gather by the tile's column count. Measured in vLLM at
 * DeepSeek-V4 expert shapes that design reached only 1.07x the per-row vector
 * kernel at 512 routed tokens, where q4_0-shaped formats reach 2.5-4x: it
 * amortized the weight load and then paid the gather MMQ_X times over.
 *
 * Decoding once per weight at tile-load time, into signed bytes plus one float
 * scale per 32 weights, takes the same shape to 3.4x. The inner loop then
 * reduces to dp4a over a q8_0-shaped tile. quant_tables.cuh says as much in its
 * own header note: these grid indices are data-dependent rather than
 * warp-uniform, so hot kernels should stage them.
 *
 * The cost is tile width -- 512 decoded weights per row is 128 ints against the
 * 32 the packed form needs -- so MMQ_Y stays small to keep LDS in budget.
 */
#pragma once
#include <hip/hip_fp16.h>

#include <cstdint>

#include "quant_tables.cuh"

namespace qciq2 {

using tmq::iq2xxs_grid;
using tmq::ksigns_iq2xs;

#define QK_K_IQ2 256
#define QK8_1_IQ2 32

typedef struct {
    __half d;
    uint16_t qs[QK_K_IQ2 / 8];  // 32 uint16 = 8 groups x 4
} block_iq2_xxs;
static_assert(sizeof(block_iq2_xxs) == 66, "iq2_xxs block must be 66 bytes");

typedef struct {
    __half2 ds;  // (scale, sum); IQ2_XXS has no offset, so only the scale is read
    int8_t qs[QK8_1_IQ2];
} block_q8_1;

// Superblocks consumed per tile iteration, and what that implies downstream.
#define IQ2_SB_PER_ITER 2
#define IQ2_TILE_INTS (IQ2_SB_PER_ITER * QK_K_IQ2 / 4)   // 128 decoded ints
#define IQ2_TILE_GROUPS (IQ2_SB_PER_ITER * 8)            // 16 groups of 32
#define IQ2_WARP 32

// CUDA's byte-wise SIMD helpers have no HIP equivalent, so define them the way
// ggml's ROCm path does. Both are used only for the sign application below.
__device__ __forceinline__ uint32_t vcmpeq4(uint32_t a, uint32_t b) {
    const uint32_t neq = a ^ b;
    return !(neq & 0xff000000) * 0xff000000 | !(neq & 0x00ff0000) * 0x00ff0000 |
           !(neq & 0x0000ff00) * 0x0000ff00 | !(neq & 0x000000ff) * 0x000000ff;
}

/// Per-byte subtract; a plain 32-bit subtract would propagate borrows across
/// byte lanes and corrupt the neighbouring weights.
__device__ __forceinline__ uint32_t vsub4(uint32_t a, uint32_t b) {
    return ((uint32_t)(uint8_t)(((a >> 24) & 0xff) - ((b >> 24) & 0xff)) << 24) |
           ((uint32_t)(uint8_t)(((a >> 16) & 0xff) - ((b >> 16) & 0xff)) << 16) |
           ((uint32_t)(uint8_t)(((a >> 8) & 0xff) - ((b >> 8) & 0xff)) << 8) |
           ((uint32_t)(uint8_t)((a & 0xff) - (b & 0xff)));
}

/**
 * @brief Decode one 32-weight group to eight ints of signed bytes.
 *
 * `aux_g` holds the four grid indices, `aux_s` the four 7-bit sign indices.
 * Signs are applied byte-wise -- xor by the mask then subtract it, which is a
 * conditional two's-complement negate -- so no branch and no per-element work.
 */
__device__ __forceinline__ void iq2xxs_decode_group(uint32_t aux_g,
                                                    uint32_t aux_s, int* out) {
    const uint8_t* aux8 = (const uint8_t*)&aux_g;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const uint32_t* grid = (const uint32_t*)(iq2xxs_grid + aux8[e]);
        const uint8_t signs = ksigns_iq2xs[(aux_s >> (7 * e)) & 127];
        const uint32_t s0 =
            vcmpeq4(((signs & 0xf) * 0x01010101) & 0x08040201, 0x08040201);
        const uint32_t s1 =
            vcmpeq4(((signs >> 4) * 0x01010101) & 0x08040201, 0x08040201);
        out[2 * e + 0] = vsub4(grid[0] ^ s0, s0);
        out[2 * e + 1] = vsub4(grid[1] ^ s1, s1);
    }
}

/// Scale for one 32-weight group: d * (0.5 + sub) * 0.25, sub in aux_s bits 28+.
__device__ __forceinline__ float iq2xxs_group_scale(__half d, uint32_t aux_s) {
    return __half2float(d) * (0.5f + (aux_s >> 28)) * 0.25f;
}

// ------------------------------------------------------------- dequantize
template <typename dst_t>
__global__ void dequant_iq2_xxs(const void* __restrict__ vx,
                                dst_t* __restrict__ y, long nblocks) {
    const long b = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    const block_iq2_xxs* x = (const block_iq2_xxs*)vx + b;
    dst_t* out = y + b * QK_K_IQ2;
#pragma unroll
    for (int g = 0; g < 8; ++g) {
        uint32_t aux_g, aux_s;
        __builtin_memcpy(&aux_g, (const uint8_t*)x->qs + 8 * g + 0, 4);
        __builtin_memcpy(&aux_s, (const uint8_t*)x->qs + 8 * g + 4, 4);
        int dec[8];
        iq2xxs_decode_group(aux_g, aux_s, dec);
        const float sc = iq2xxs_group_scale(x->d, aux_s);
        const int8_t* b8 = (const int8_t*)dec;
#pragma unroll
        for (int j = 0; j < 32; ++j) out[32 * g + j] = (dst_t)(sc * (float)b8[j]);
    }
}

// -------------------------------------------------------------- tiled GEMM
/**
 * @brief dst[col, row] = sum_k W[row, k] * X[col, k], W iq2_xxs, X q8_1.
 *
 * One block computes an MMQ_Y x MMQ_X output tile. `nrows_y` is the padded
 * activation width. NEED_CHECK clamps the loaded row when nrows_x is not a
 * multiple of MMQ_Y, so an overrunning tile reads live memory and is dropped at
 * write-back rather than being predicated in the inner loop.
 */
template <int MMQ_X, int MMQ_Y, int NWARPS, bool NEED_CHECK>
__global__ void __launch_bounds__(IQ2_WARP* NWARPS, 2)
    iq2xxs_mmq_q8_1(const void* __restrict__ vw, const void* __restrict__ vy,
                    float* __restrict__ dst, const int ncols_x,
                    const int nrows_x, const int ncols_y, const int nrows_y,
                    const int nrows_dst) {
    const block_iq2_xxs* x = (const block_iq2_xxs*)vw;
    const block_q8_1* y = (const block_q8_1*)vy;

    const int sb_per_row = ncols_x / QK_K_IQ2;
    const int q8_per_col = nrows_y / QK8_1_IQ2;
    const int row_x_0 = blockIdx.x * MMQ_Y;
    const int col_y_0 = blockIdx.y * MMQ_X;

    __shared__ int tile_x_qs[MMQ_Y * (IQ2_TILE_INTS + 1)];
    __shared__ float tile_x_d[MMQ_Y * IQ2_TILE_GROUPS];
    __shared__ int tile_y_qs[MMQ_X * IQ2_TILE_INTS];
    __shared__ float tile_y_d[MMQ_X * IQ2_TILE_GROUPS];

    float sum[MMQ_Y / IQ2_WARP][MMQ_X / NWARPS] = {{0.0f}};

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int nthreads = IQ2_WARP * NWARPS;
    const int tid = ty * IQ2_WARP + tx;

    for (int ib0 = 0; ib0 < sb_per_row; ib0 += IQ2_SB_PER_ITER) {
        __syncthreads();

        // ---- weights: decode once per weight, here.
        // Thread tx owns group tx/2 and, within it, the pair of grid entries
        // selected by tx%2 -- four decoded ints each, so 32 threads cover 128.
        {
            const int g = tx / 2;
            const int hlf = tx % 2;
            const int sb = min(g / 8, sb_per_row - 1);
            const int ib32 = g % 8;
#pragma unroll
            for (int i0 = 0; i0 < MMQ_Y; i0 += NWARPS) {
                int i = i0 + ty;
                if (NEED_CHECK) i = min(i, nrows_x - row_x_0 - 1);
                const block_iq2_xxs* bxi = x + (row_x_0 + i) * sb_per_row + ib0 + sb;
                // 66-byte block with qs at offset 2: nothing is 4-byte aligned.
                uint32_t aux_g, aux_s;
                __builtin_memcpy(&aux_g, (const uint8_t*)bxi->qs + 8 * ib32 + 0, 4);
                __builtin_memcpy(&aux_s, (const uint8_t*)bxi->qs + 8 * ib32 + 4, 4);
                int dec[8];
                iq2xxs_decode_group(aux_g, aux_s, dec);
                int* d4 = &tile_x_qs[i * (IQ2_TILE_INTS + 1) + g * 8 + hlf * 4];
#pragma unroll
                for (int l = 0; l < 4; ++l) d4[l] = dec[hlf * 4 + l];
                if (hlf == 0)
                    tile_x_d[i * IQ2_TILE_GROUPS + g] =
                        iq2xxs_group_scale(bxi->d, aux_s);
            }
        }

        // ---- activations: 16 q8_1 blocks per column for this iteration.
        for (int t = tid; t < MMQ_X * IQ2_TILE_INTS; t += nthreads) {
            const int j = t / IQ2_TILE_INTS;
            const int e = t % IQ2_TILE_INTS;  // int index within the 512 weights
            const int col = min(col_y_0 + j, ncols_y - 1);
            const int blk = ib0 * (QK_K_IQ2 / QK8_1_IQ2) + e / 8;
            int v = 0;
            if (blk < q8_per_col)
                __builtin_memcpy(&v, y[col * q8_per_col + blk].qs + 4 * (e % 8), 4);
            tile_y_qs[j * IQ2_TILE_INTS + e] = v;
        }
        for (int t = tid; t < MMQ_X * IQ2_TILE_GROUPS; t += nthreads) {
            const int j = t / IQ2_TILE_GROUPS;
            const int g = t % IQ2_TILE_GROUPS;
            const int col = min(col_y_0 + j, ncols_y - 1);
            const int blk = ib0 * (QK_K_IQ2 / QK8_1_IQ2) + g;
            tile_y_d[j * IQ2_TILE_GROUPS + g] =
                blk < q8_per_col ? __low2float(y[col * q8_per_col + blk].ds) : 0.0f;
        }
        __syncthreads();

        // ---- accumulate: eight dp4a per 32-weight group, the q8_0 inner loop.
        for (int g = 0; g < IQ2_TILE_GROUPS; ++g) {
            if (ib0 * QK_K_IQ2 + g * 32 >= ncols_x) break;
#pragma unroll
            for (int j = 0; j < MMQ_X; j += NWARPS) {
                const int jj = ty + j;
                const float d8 = tile_y_d[jj * IQ2_TILE_GROUPS + g];
#pragma unroll
                for (int i = 0; i < MMQ_Y; i += IQ2_WARP) {
                    const int ii = tx + i;
                    int sumi = 0;
#pragma unroll
                    for (int l = 0; l < 8; ++l)
                        sumi = __builtin_amdgcn_sdot4(
                            tile_x_qs[ii * (IQ2_TILE_INTS + 1) + g * 8 + l],
                            tile_y_qs[jj * IQ2_TILE_INTS + g * 8 + l], sumi,
                            false);
                    sum[i / IQ2_WARP][j / NWARPS] +=
                        tile_x_d[ii * IQ2_TILE_GROUPS + g] * d8 * (float)sumi;
                }
            }
        }
    }

#pragma unroll
    for (int j = 0; j < MMQ_X; j += NWARPS) {
        const int col = col_y_0 + ty + j;
        if (col >= ncols_y) return;
#pragma unroll
        for (int i = 0; i < MMQ_Y; i += IQ2_WARP) {
            const int row = row_x_0 + tx + i;
            if (row >= nrows_dst) continue;
            dst[col * nrows_dst + row] = sum[i / IQ2_WARP][j / NWARPS];
        }
    }
}

}  // namespace qciq2
