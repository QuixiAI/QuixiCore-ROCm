#pragma once
/**
 * @file
 * @brief Shared MXFP8 codec primitives for the CDNA3 KV-cache kernels.
 *
 * Layout, quantization rule and the three easy-to-get-wrong details are
 * documented in kv_cache_mxfp8.cu. Both that codec and paged_attention_mxfp8.cu
 * include this so there is exactly one definition of the format.
 */
#include "hip/hip_runtime.h"
#include <cstdint>
#include <cstring>
#include <cmath>

namespace mxkv {

constexpr int kMxGroup = 32;
constexpr int kMxBlockBytes = 33;  // 1 scale byte + 32 code bytes
constexpr float kE4M3FnMax = 448.0f;

// ---- E8M0: value = 2^(code-127); 255 is NaN. Encode rounds the exponent UP. --
__host__ __device__ __forceinline__ float e8m0_decode_pow2(uint8_t code) {
    return ldexpf(1.0f, (int)code - 127);
}
__host__ __device__ __forceinline__ uint8_t e8m0_encode_up(float requested) {
    if (!(requested > 0.0f)) return 0;
    int exponent = (int)ceilf(log2f(requested));
    int biased = exponent + 127;
    if (biased < 0) biased = 0;
    if (biased > 254) biased = 254;
    return (uint8_t)biased;
}

// ---- E4M3FN, round-half-to-even. Mirrors the CPU float8_encode exactly. ----
__host__ __device__ __forceinline__ uint32_t round_right_even(uint32_t value, int shift) {
    if (shift <= 0) return value << (-shift);
    const uint32_t half = 1u << (shift - 1);
    const uint32_t truncated = value >> shift;
    const uint32_t remainder = value & ((1u << shift) - 1u);
    if (remainder > half) return truncated + 1u;
    if (remainder < half) return truncated;
    return truncated + (truncated & 1u);  // tie -> even
}

__host__ __device__ __forceinline__ uint8_t e4m3fn_encode(float value) {
    // Bit tests rather than isnan/signbit/isfinite: those are device-only in a
    // __host__ __device__ function under HIP, and this must be bit-identical on
    // both sides so the harness can check the kernel against a host replica.
    uint32_t vbits;
    memcpy(&vbits, &value, sizeof(vbits));
    const uint8_t sign = (uint8_t)((vbits >> 31) << 7);
    const uint32_t exponent_field = vbits & 0x7f800000u;
    const bool is_nan = (exponent_field == 0x7f800000u) && (vbits & 0x7fffffu);
    if (is_nan) return 0x7f;
    const float magnitude = fabsf(value);
    if (magnitude == 0.0f) return sign;
    const bool is_inf = (exponent_field == 0x7f800000u);
    if (is_inf || magnitude >= kE4M3FnMax) return 0x7e | sign;

    uint32_t bits;
    memcpy(&bits, &magnitude, sizeof(bits));
    const int source_exponent = (int)((bits >> 23) & 0xffu);
    if (source_exponent == 0) return sign;      // fp32 subnormal -> zero
    int unbiased = source_exponent - 127;
    const uint32_t significand = (1u << 23) | (bits & 0x7fffffu);
    constexpr int mantissa_bits = 3, bias = 7;

    uint32_t code;
    if (unbiased < 1 - bias) {                   // E4M3 subnormal
        const int shift = 24 - unbiased - bias - mantissa_bits;
        code = round_right_even(significand, shift);
    } else {
        uint32_t rounded = round_right_even(significand, 23 - mantissa_bits);
        if (rounded == (1u << (mantissa_bits + 1))) { rounded >>= 1; ++unbiased; }
        const int encoded_exponent = unbiased + bias;
        code = ((uint32_t)encoded_exponent << mantissa_bits) |
               (rounded - (1u << mantissa_bits));
    }
    return (uint8_t)(code | sign);
}

__host__ __device__ __forceinline__ float e4m3fn_decode(uint8_t code) {
    const uint32_t mantissa = code & 0x7u;
    const int exponent = (int)((code >> 3) & 0xfu);
    const float sign = (code & 0x80u) ? -1.0f : 1.0f;
    if (exponent == 0) return sign * ldexpf((float)mantissa, -9);   // subnormal
    if (exponent == 0xf && mantissa == 0x7) return sign * NAN;
    return sign * ldexpf(1.0f + (float)mantissa / 8.0f, exponent - 7);
}

__host__ __device__ __forceinline__ long long mx_group_base(
        long long slot, long long head, long long group,
        long long heads, long long groups) {
    return ((slot * heads + head) * groups + group) * kMxBlockBytes;
}

}  // namespace mxkv
