/**
 * @file
 * @brief OCP MXFP4 in the GGUF block layout, native HIP for CDNA3 (gfx942).
 *
 * `GGML_TYPE_MXFP4` (39) is the format DeepSeek-V4-Flash stores its routed
 * experts in. The block is 32 values in 17 bytes:
 *
 *   struct { uint8_t e; uint8_t qs[16]; }
 *     value[j]      = e8m0(e) * e2m1(qs[j] & 0xF)    for j in [0,16)
 *     value[j + 16] = e8m0(e) * e2m1(qs[j] >> 4)
 *
 * Note the halves: the low nibbles supply the FIRST 16 values and the high
 * nibbles the second 16, not an even/odd interleave. Getting that backwards
 * still produces plausible magnitudes, which is why the harness checks element
 * positions rather than just norms.
 *
 * e2m1 is not uniform, so the nibbles cannot be fed to a dot-product
 * instruction directly the way q4_0's offset ints can. They are mapped through
 * a 16-entry table holding 2x the true values -- {0,1,2,3,4,6,8,12} and
 * negatives, all integers -- and the factor of 2 is folded into the scale. That
 * keeps the inner loop on integer dot4 and matches what ggml does. The lookup
 * itself is byte permutes, which is the CDNA way to index a 16-byte table
 * without LDS or branches.
 */
#pragma once
#include <cstdint>

#include "quant_primitives.cuh"

namespace qcmxfp4 {

#define QK_MXFP4 32

typedef struct {
    uint8_t e;                  // E8M0 exponent-only scale
    uint8_t qs[QK_MXFP4 / 2];   // 16 bytes, two e2m1 codes each
} block_mxfp4;
static_assert(sizeof(block_mxfp4) == 17, "mxfp4 block must be 17 bytes");

// 2x the e2m1 values so the table is integral; the 0.5 is folded into d.
__device__ __constant__ static const int8_t kvalues_mxfp4[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

// The e2m1 table as four uint32, for the byte-permute lookup in
// quixicore::quant::table_lookup_16.
__device__ __forceinline__ void table_lookup_16(int q4, int& x, int& y) {
    quixicore::quant::table_lookup_16(
        reinterpret_cast<const uint32_t*>(kvalues_mxfp4), q4, x, y);
}

// --------------------------------------------------------------- dequantize
template <typename dst_t>
__global__ void dequant_mxfp4(const void* __restrict__ vx, dst_t* __restrict__ y,
                              long nblocks) {
    const long b = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    const block_mxfp4* x = (const block_mxfp4*)vx + b;
    const float d = quixicore::quant::e8m0_decode_ggml(x->e) * 0.5f;
    dst_t* out = y + b * QK_MXFP4;
#pragma unroll
    for (int j = 0; j < QK_MXFP4 / 2; ++j) {
        out[j] = (dst_t)(d * (float)kvalues_mxfp4[x->qs[j] & 0xF]);
        out[j + 16] = (dst_t)(d * (float)kvalues_mxfp4[x->qs[j] >> 4]);
    }
}

// ------------------------------------------------------------- vec dot q8_1
// `u` is 8 int32s of int8 activations: the first 4 pair with the low-nibble
// half of the block, the next 4 with the high-nibble half.
__device__ __forceinline__ float vec_dot_mxfp4_q8_1(const block_mxfp4* bq,
                                                    const int* u, float d8) {
    int sumi = 0;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        int vx, vy;
        table_lookup_16(quixicore::quant::load_int_unaligned(bq->qs, l), vx, vy);
        sumi = quixicore::quant::dp4a(vx, u[l + 0], sumi);
        sumi = quixicore::quant::dp4a(vy, u[l + 4], sumi);
    }
    return quixicore::quant::e8m0_decode_ggml(bq->e) * 0.5f * d8 * (float)sumi;
}

// ----------------------------------------------------------------- GEMV/MoE
// Y[row] = sum_k W[row,k] * X[k], W mxfp4, X pre-quantized to int8 with one
// fp32 scale per 32-wide group. One wave per output row.
//
// `expert_ids` is optional: when non-null the kernel is the MoE form and row r
// reads expert expert_ids[r / rows_per_token]; a negative id zeroes the row.
template <typename dst_t>
__global__ void mxfp4_gemv_q8_1(const void* __restrict__ vw,
                                const int8_t* __restrict__ xq,
                                const float* __restrict__ xs,
                                dst_t* __restrict__ y, int ncols, int nrows,
                                const int* __restrict__ expert_ids,
                                int rows_per_token, long expert_stride_blocks) {
    const int row = blockIdx.x;
    if (row >= nrows) return;
    const int lane = threadIdx.x & 63;

    long wbase = 0;
    const int8_t* xq_row = xq;
    const float* xs_row = xs;
    if (expert_ids) {
        const int tok = row / rows_per_token;
        const int e = expert_ids[tok];
        if (e < 0) {
            if (lane == 0) y[row] = (dst_t)0.0f;
            return;
        }
        wbase = (long)e * expert_stride_blocks;
        xq_row = xq + (long)tok * ncols;
        xs_row = xs + (long)tok * (ncols / QK_MXFP4);
    }

    const int nblocks = ncols / QK_MXFP4;
    const int out_row = expert_ids ? (row % rows_per_token) : row;
    const block_mxfp4* w =
        (const block_mxfp4*)vw + wbase + (long)out_row * nblocks;

    float acc = 0.0f;
    for (int b = lane; b < nblocks; b += 64) {
        int u[8];
        const int8_t* xb = xq_row + (long)b * QK_MXFP4;
        __builtin_memcpy(u, xb, 32);
        acc += vec_dot_mxfp4_q8_1(w + b, u, xs_row[b]);
    }
#pragma unroll
    for (int off = 32; off > 0; off >>= 1) acc += __shfl_xor(acc, off, 64);
    if (lane == 0) y[row] = (dst_t)acc;
}

}  // namespace qcmxfp4
