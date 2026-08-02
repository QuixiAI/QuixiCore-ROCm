/**
 * @file
 * @brief Primitives the quantized-kernel families need and HIP does not supply.
 *
 * Two groups, for two different reasons.
 *
 * **Byte-wise integer SIMD.** CUDA exposes `__dp4a` and the `__v*4` family;
 * HIP exposes none of them by name. Kernels ported from CUDA or from ggml have
 * been carrying private copies of the shims, which is how QuixiCore ended up
 * depending on ggml's `ggml-common.h` to compile a kernel that is otherwise
 * self-contained. These are that dependency removed.
 *
 * **E8M0 block-scale decode.** This one is not mere duplication. The tree holds
 * three DIFFERENT behaviours under two names, and they agree everywhere except
 * at code 0:
 *
 *   | convention | code 0 decodes to | who wants it                        |
 *   | ---------- | ----------------- | ----------------------------------- |
 *   | ieee       | +0.0              | quant_formats.cuh (qgemm/qgemv/...) |
 *   | ldexp      | 2^-127            | serving/kv_cache_mxfp8              |
 *   | ggml       | 2^-126 (min norm) | the GGUF families                   |
 *
 * For every other code all three agree on 2^(code-127). Collapsing them into
 * one `e8m0_decode` would silently move numerics in whichever callers lost the
 * argument, so there deliberately is no such function here: a caller has to
 * name the convention it means. The OCP MX spec leaves code 0 to the format
 * that embeds it, which is why the formats disagree in the first place.
 */
#pragma once
#include <cstdint>

namespace quixicore::quant {

// ------------------------------------------------------------- integer SIMD

/// Four-way signed int8 dot product with accumulate, CUDA's `__dp4a`.
__device__ __forceinline__ int dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

/// Per-byte equality: 0xFF in each byte where the operands match, else 0x00.
__device__ __forceinline__ uint32_t vcmpeq4(uint32_t a, uint32_t b) {
    const uint32_t neq = a ^ b;
    return !(neq & 0xff000000) * 0xff000000 | !(neq & 0x00ff0000) * 0x00ff0000 |
           !(neq & 0x0000ff00) * 0x0000ff00 | !(neq & 0x000000ff) * 0x000000ff;
}

/// Per-byte inequality; the complement of vcmpeq4.
__device__ __forceinline__ uint32_t vcmpne4(uint32_t a, uint32_t b) {
    return ~vcmpeq4(a, b);
}

/// Per-byte subtract. The lane isolation is the point: a plain 32-bit subtract
/// propagates borrows into the next byte and corrupts the neighbouring value.
__device__ __forceinline__ uint32_t vsub4(uint32_t a, uint32_t b) {
    return ((uint32_t)(uint8_t)(((a >> 24) & 0xff) - ((b >> 24) & 0xff)) << 24) |
           ((uint32_t)(uint8_t)(((a >> 16) & 0xff) - ((b >> 16) & 0xff)) << 16) |
           ((uint32_t)(uint8_t)(((a >> 8) & 0xff) - ((b >> 8) & 0xff)) << 8) |
           ((uint32_t)(uint8_t)((a & 0xff) - (b & 0xff)));
}

/// Per-byte add, with the same lane isolation as vsub4.
__device__ __forceinline__ uint32_t vadd4(uint32_t a, uint32_t b) {
    return ((uint32_t)(uint8_t)(((a >> 24) & 0xff) + ((b >> 24) & 0xff)) << 24) |
           ((uint32_t)(uint8_t)(((a >> 16) & 0xff) + ((b >> 16) & 0xff)) << 16) |
           ((uint32_t)(uint8_t)(((a >> 8) & 0xff) + ((b >> 8) & 0xff)) << 8) |
           ((uint32_t)(uint8_t)((a & 0xff) + (b & 0xff)));
}

/// Apply four sign bits to four packed bytes: negate byte i where bit i is set.
/// xor by the mask then subtract it is a conditional two's-complement negate,
/// so this is branch-free and does no per-element work.
__device__ __forceinline__ uint32_t apply_sign_bits4(uint32_t bytes,
                                                     uint32_t bits4) {
    const uint32_t m = vcmpeq4((bits4 * 0x01010101u) & 0x08040201u, 0x08040201u);
    return vsub4(bytes ^ m, m);
}

// ------------------------------------------------------------------ loading

/// Read one int from a byte stream with no alignment guarantee. GGUF block
/// layouts are the reason this exists: an iq2_xxs block is 66 bytes and an
/// mxfp4 block 17, so consecutive blocks land on odd addresses and the usual
/// aligned helpers are undefined behaviour on them.
__device__ __forceinline__ int load_int_unaligned(const void* p, int i32) {
    int v;
    __builtin_memcpy(&v, (const uint8_t*)p + sizeof(int) * i32, sizeof(int));
    return v;
}

/// Index a 16-entry signed-byte table with four packed nibbles, by byte
/// permute. Bit 3 of a nibble selects the upper half, so both halves are
/// permuted and blended on it rather than branching. `table` is 16 int8 as four
/// uint32. Returns the four low nibbles in `lo` and the four high in `hi`.
__device__ __forceinline__ void table_lookup_16(const uint32_t* table, int packed,
                                                int& lo, int& hi) {
    const uint32_t l = (uint32_t)packed;
    const uint32_t h = ((uint32_t)packed >> 4);
    const uint32_t l_lo = __builtin_amdgcn_perm(table[1], table[0], l & 0x07070707u);
    const uint32_t l_hi = __builtin_amdgcn_perm(table[3], table[2], l & 0x07070707u);
    const uint32_t h_lo = __builtin_amdgcn_perm(table[1], table[0], h & 0x07070707u);
    const uint32_t h_hi = __builtin_amdgcn_perm(table[3], table[2], h & 0x07070707u);
    const uint32_t l_sel = ((l >> 3) & 0x01010101u) * 0xFFu;
    const uint32_t h_sel = ((h >> 3) & 0x01010101u) * 0xFFu;
    lo = (int)((l_lo & ~l_sel) | (l_hi & l_sel));
    hi = (int)((h_lo & ~h_sel) | (h_hi & h_sel));
}

// -------------------------------------------------------- E8M0 scale decode
//
// Deliberately no bare `e8m0_decode`: see the file header. Pick the convention
// the format you are decoding actually specifies.

/// The byte IS the fp32 exponent field. Code 0 gives +0.0. Two ALU ops.
__device__ __forceinline__ float e8m0_decode_ieee(uint8_t e) {
    return __uint_as_float((uint32_t)e << 23);
}

/// value = 2^(code-127) throughout, so code 0 gives the subnormal 2^-127.
__device__ __forceinline__ float e8m0_decode_ldexp(uint8_t e) {
    return ldexpf(1.0f, (int)e - 127);
}

/// ggml/GGUF convention: code 0 is the smallest NORMAL, not zero and not a
/// subnormal. Used by the MXFP4 and imatrix GGUF paths, where treating it as
/// zero would silently drop a block.
__device__ __forceinline__ float e8m0_decode_ggml(uint8_t e) {
    const uint32_t bits = (e == 0) ? 0x00400000u : ((uint32_t)e << 23);
    float r;
    __builtin_memcpy(&r, &bits, 4);
    return r;
}

}  // namespace quixicore::quant
