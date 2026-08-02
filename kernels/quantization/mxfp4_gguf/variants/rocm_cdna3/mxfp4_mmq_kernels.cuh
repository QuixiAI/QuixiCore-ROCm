/**
 * @file
 * @brief MXFP4 x q8_1 tiled GEMM (the "MMQ" path) for CDNA3 (gfx942).
 *
 * The GEMV in mxfp4_gguf_kernels.cuh reloads the weight row for every output
 * column, so its cost is linear in columns; past a few dozen columns that
 * dominates. This kernel stages a tile of weights in LDS once and reuses it
 * across all MMQ_X columns, which is what makes prefill and large-batch decode
 * affordable. Measured on MI300X at DeepSeek-V4-Flash expert shapes the tile
 * path is ~1.5x the GEMV at 32 tokens and ~2.4x at 512, and loses below ~8, so
 * both kernels have to exist and the caller picks between them.
 *
 * MXFP4's block constants are QK 32 / QR 2 / QI 4 -- identical to q4_0 -- so
 * the tile geometry and index arithmetic here are the classic ggml q4_0 MMQ
 * layout unchanged. Exactly two things differ: the scale is an e8m0 byte
 * rather than an fp16, and the nibbles go through the e2m1 table instead of an
 * offset of 8, which also removes the correction term against the activation
 * sum that q4_0 needs.
 *
 * The 17-byte block is the one real hazard: qs sits at an odd offset and
 * blocks are 17 apart, so nothing in the weight stream is 4-byte aligned.
 * Every read of the quants must be a memcpy, never a cast.
 */
#pragma once
#include <hip/hip_fp16.h>

#include <cstdint>

#include "mxfp4_gguf_kernels.cuh"

namespace qcmxfp4 {

#define QI_MXFP4 (QK_MXFP4 / 8)  // 4 int32 of quants per block
#define QK8_1 32
#define QI8_1 (QK8_1 / 4)
// The tile indexing is written against a 32-lane subgroup, as ggml's is; on
// wave64 two subgroups share a wave, which costs nothing here because every
// cross-lane step goes through LDS rather than shuffles.
#define MMQ_WARP 32

typedef struct {
    __half2 ds;  // (scale, sum); MXFP4 has no offset, so only the scale is read
    int8_t qs[QK8_1];
} block_q8_1;

/**
 * @brief dst[col, row] = sum_k W[row, k] * X[col, k], W mxfp4, X q8_1.
 *
 * One block computes an MMQ_Y x MMQ_X output tile with NWARPS subgroups of
 * MMQ_WARP lanes. `nrows_y` is the padded activation width, which is why it is
 * separate from ncols_x. Set NEED_CHECK when nrows_x is not a multiple of
 * MMQ_Y; it clamps the loaded row so out-of-range tiles read real memory and
 * are dropped at write-back rather than being predicated in the inner loop.
 */
template <int MMQ_X, int MMQ_Y, int NWARPS, bool NEED_CHECK>
__global__ void __launch_bounds__(MMQ_WARP* NWARPS, 2)
    mxfp4_mmq_q8_1(const void* __restrict__ vw, const void* __restrict__ vy,
                   float* __restrict__ dst, const int ncols_x,
                   const int nrows_x, const int ncols_y, const int nrows_y,
                   const int nrows_dst) {
    const block_mxfp4* x = (const block_mxfp4*)vw;
    const block_q8_1* y = (const block_q8_1*)vy;

    const int blocks_per_row_x = ncols_x / QK_MXFP4;
    const int blocks_per_col_y = nrows_y / QK8_1;
    const int blocks_per_warp = MMQ_WARP / QI_MXFP4;
    constexpr int VDR = QI_MXFP4;              // ints consumed per k step
    constexpr int SCALES_PER_ROW = MMQ_WARP / QI_MXFP4;
    constexpr int Y_SCALES = MMQ_WARP / QI8_1;

    const int row_x_0 = blockIdx.x * MMQ_Y;
    const int col_y_0 = blockIdx.y * MMQ_X;

    __shared__ int tile_x_qs[MMQ_Y * (MMQ_WARP + 1)];
    __shared__ float tile_x_d[MMQ_Y * SCALES_PER_ROW + MMQ_Y / QI_MXFP4];
    __shared__ int tile_y_qs[MMQ_X * MMQ_WARP];
    __shared__ float tile_y_d[MMQ_X * Y_SCALES];

    float sum[MMQ_Y / MMQ_WARP][MMQ_X / NWARPS] = {{0.0f}};

    const int tx = threadIdx.x;  // walks K within a block of quants
    const int ty = threadIdx.y;  // walks output columns

    for (int ib0 = 0; ib0 < blocks_per_row_x; ib0 += blocks_per_warp) {
        __syncthreads();

        const int kbx = tx / QI_MXFP4;
        const int kqsx = tx % QI_MXFP4;
#pragma unroll
        for (int i0 = 0; i0 < MMQ_Y; i0 += NWARPS) {
            int i = i0 + ty;
            if (NEED_CHECK) i = min(i, nrows_x - row_x_0 - 1);
            const block_mxfp4* bxi =
                x + (row_x_0 + i) * blocks_per_row_x + ib0 + kbx;
            int q4;
            __builtin_memcpy(&q4, bxi->qs + sizeof(int) * kqsx, sizeof(int));
            tile_x_qs[i * (MMQ_WARP + 1) + tx] = q4;
        }

        const int kbxd = tx % SCALES_PER_ROW;
#pragma unroll
        for (int i0 = 0; i0 < MMQ_Y; i0 += NWARPS * QI_MXFP4) {
            int i = i0 + ty * QI_MXFP4 + tx / SCALES_PER_ROW;
            if (NEED_CHECK) i = min(i, nrows_x - row_x_0 - 1);
            const block_mxfp4* bxi =
                x + (row_x_0 + i) * blocks_per_row_x + ib0 + kbxd;
            tile_x_d[i * SCALES_PER_ROW + i / QI_MXFP4 + kbxd] =
                quixicore::quant::e8m0_decode_ggml(bxi->e) * 0.5f;
        }

        // Two spans: the low nibbles of the tile pair with the first half of
        // the activation blocks, the high nibbles with the second.
#pragma unroll
        for (int ir = 0; ir < 2; ++ir) {
            if (ib0 * QK_MXFP4 + ir * (QK_MXFP4 * blocks_per_warp) / 2 >=
                ncols_x)
                break;
            const int kqs = ir * MMQ_WARP + tx;
            const int kbyd = kqs / QI8_1;
            __syncthreads();

#pragma unroll
            for (int i = 0; i < MMQ_X; i += NWARPS) {
                const int col = min(col_y_0 + ty + i, ncols_y - 1);
                const block_q8_1* by =
                    &y[col * blocks_per_col_y + ib0 * (QK_MXFP4 / QK8_1) +
                       kbyd];
                int q8;
                __builtin_memcpy(&q8, by->qs + sizeof(int) * (tx % QI8_1), 4);
                tile_y_qs[(ty + i) * MMQ_WARP + kqs % MMQ_WARP] = q8;
            }

#pragma unroll
            for (int ids0 = 0; ids0 < MMQ_X; ids0 += NWARPS * QI8_1) {
                const int ids =
                    (ids0 + ty * QI8_1 + tx / Y_SCALES) % MMQ_X;
                const int kby = tx % Y_SCALES;
                const int col = min(col_y_0 + ids, ncols_y - 1);
                tile_y_d[ids * Y_SCALES + kby] = __low2float(
                    y[col * blocks_per_col_y + ib0 * (QK_MXFP4 / QK8_1) +
                      ir * Y_SCALES + kby]
                        .ds);
            }
            __syncthreads();

            for (int k = ir * (MMQ_WARP / 2); k < (ir + 1) * (MMQ_WARP / 2);
                 k += VDR) {
                const int kyqs = k % (QI8_1 / 2) + QI8_1 * (k / (QI8_1 / 2));
#pragma unroll
                for (int j = 0; j < MMQ_X; j += NWARPS) {
                    const int jj = ty + j;
                    const float d8 =
                        tile_y_d[jj * Y_SCALES + (2 * k / QI8_1) % Y_SCALES];
#pragma unroll
                    for (int i = 0; i < MMQ_Y; i += MMQ_WARP) {
                        const int ii = tx + i;
                        int sumi = 0;
#pragma unroll
                        for (int l = 0; l < VDR; ++l) {
                            int vlo, vhi;
                            table_lookup_16(
                                tile_x_qs[ii * (MMQ_WARP + 1) + k + l], vlo,
                                vhi);
                            sumi = quixicore::quant::dp4a(
                                vlo,
                                tile_y_qs[jj * MMQ_WARP +
                                          (kyqs + l) % MMQ_WARP],
                                sumi);
                            sumi = quixicore::quant::dp4a(
                                vhi,
                                tile_y_qs[jj * MMQ_WARP +
                                          (kyqs + l + QI_MXFP4) % MMQ_WARP],
                                sumi);
                        }
                        sum[i / MMQ_WARP][j / NWARPS] +=
                            tile_x_d[ii * SCALES_PER_ROW + ii / QI_MXFP4 +
                                     k / QI_MXFP4] *
                            d8 * (float)sumi;
                    }
                }
            }
        }
    }

#pragma unroll
    for (int j = 0; j < MMQ_X; j += NWARPS) {
        const int col = col_y_0 + ty + j;
        if (col >= ncols_y) return;
#pragma unroll
        for (int i = 0; i < MMQ_Y; i += MMQ_WARP) {
            const int row = row_x_0 + tx + i;
            if (row >= nrows_dst) continue;
            dst[col * nrows_dst + row] = sum[i / MMQ_WARP][j / NWARPS];
        }
    }
}

}  // namespace qcmxfp4
