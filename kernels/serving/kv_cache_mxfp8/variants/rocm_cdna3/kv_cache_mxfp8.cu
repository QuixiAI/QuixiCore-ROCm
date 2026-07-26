#include "hip/hip_runtime.h"
/**
 * @file
 * @brief CDNA3 (gfx942) MXFP8 paged KV-cache codec: scatter and gather.
 *
 * Semantic source is ../QuixiCore-CPU/kernels/attention/attention_mxfp8.cpp
 * (kv_cache_scatter_mxfp8 :124, kv_cache_gather_mxfp8 :224).
 *
 * ## Canonical layout -- ONE interleaved plane, unlike q8_0
 *
 * The q8_0 codec in ../../kv_cache_q8_0 keeps codes and scales in two separate
 * arrays. MXFP8 does not: each 32-element MX group is a single 33-byte record,
 * scale byte first, and both key and value caches are one flat uint8 plane.
 *
 *   cache uint8 [max_slots, heads, head_dim/32, 33]
 *   group base  = ((slot * heads + head) * groups + group) * 33
 *   byte 0      = E8M0 scale code
 *   bytes 1..32 = E4M3FN codes
 *
 * Porting the q8_0 two-plane shape here produces plausible numbers and a cache
 * no other backend can read.
 *
 * ## Quantization rule (bit-exact, not approximate)
 *
 *   scale_code = e8m0_encode_up(amax(group) / 448.0f)
 *   inverse    = amax > 0 ? 1 / e8m0_decode(scale_code) : 0
 *   code       = e4m3fn_encode(v * inverse)
 *
 * Three details that a reasonable-looking implementation gets wrong:
 *
 *  - `e8m0_encode_up` rounds the exponent UP (ceil of log2), and clamps to
 *    [0,254]. It is NOT the raw exponent-field extraction used by the MX GEMM
 *    path in quant_formats.cuh's `e8m0_encode`. Using that one silently
 *    halves the scale on every non-power-of-two group.
 *  - 448 is the largest finite E4M3FN magnitude; the divide is what maps the
 *    group's amax onto the representable top of the format.
 *  - E4M3FN encoding is round-half-to-EVEN, with NaN -> 0x7f, +-0 -> 0x00/0x80,
 *    and overflow saturating to max-finite with the sign preserved. Q8_0 in the
 *    sibling codec rounds half AWAY from zero -- they are different rules and
 *    the caches are not interchangeable.
 *
 * ## Scatter is incremental, NOT a full rewrite
 *
 * The q8_0 reference zero-fills the entire cache before writing. The MXFP8
 * reference does not -- it writes only the requested slots and leaves the rest
 * untouched. Slots carrying a negative index are skipped. Do not add a
 * zero-fill kernel here "for symmetry"; it would diverge from the reference and
 * silently clear live pages.
 *
 * Non-finite input is rejected (sets *invalid), matching the reference's
 * early-out.
 *
 * Build: make      Test: make test  (self-checking, byte-exact vs host replica)
 */
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>

#include "mxfp8_common.cuh"

namespace mxkv {

// ===========================================================================
// scatter -- one warp-group per (token, head), lanes stride over MX groups
// ===========================================================================
template <int LANES>
__global__ void k_mx_scatter(const float *__restrict__ key,
                             const float *__restrict__ value,
                             const int *__restrict__ slots,
                             uint8_t *__restrict__ key_cache,
                             uint8_t *__restrict__ value_cache,
                             int count, int heads, int head_dim,
                             int *__restrict__ invalid) {
    const long long task = (long long)blockIdx.x * (blockDim.x / LANES) + threadIdx.x / LANES;
    if (task >= (long long)count * heads) return;
    const int lane = threadIdx.x % LANES;
    const int token = task / heads, head = task % heads;
    const int slot = slots[token];
    if (slot < 0) return;                       // skipped, left untouched

    const int groups = head_dim / kMxGroup;
    const long long source = ((long long)token * heads + head) * head_dim;

    for (int group = lane; group < groups; group += LANES) {
        const long long g0 = (long long)group * kMxGroup;
        const long long dst = mx_group_base(slot, head, group, heads, groups);

        float key_max = 0.0f, value_max = 0.0f;
        bool bad = false;
        for (int i = 0; i < kMxGroup; ++i) {
            const float kv = key[source + g0 + i];
            const float vv = value[source + g0 + i];
            if (!isfinite(kv) || !isfinite(vv)) bad = true;
            key_max = fmaxf(key_max, fabsf(kv));
            value_max = fmaxf(value_max, fabsf(vv));
        }
        if (bad) { atomicExch(invalid, 1); continue; }

        const uint8_t key_scale = e8m0_encode_up(key_max / kE4M3FnMax);
        const uint8_t value_scale = e8m0_encode_up(value_max / kE4M3FnMax);
        key_cache[dst] = key_scale;
        value_cache[dst] = value_scale;

        const float key_inv = key_max > 0.0f ? 1.0f / e8m0_decode(key_scale) : 0.0f;
        const float value_inv = value_max > 0.0f ? 1.0f / e8m0_decode(value_scale) : 0.0f;
        for (int i = 0; i < kMxGroup; ++i) {
            key_cache[dst + 1 + i] = e4m3fn_encode(key[source + g0 + i] * key_inv);
            value_cache[dst + 1 + i] = e4m3fn_encode(value[source + g0 + i] * value_inv);
        }
    }
}

// ===========================================================================
// gather -- indices must be in [0, max_slots); no negative-skip here
// ===========================================================================
template <int LANES>
__global__ void k_mx_gather(const uint8_t *__restrict__ key_cache,
                            const uint8_t *__restrict__ value_cache,
                            const int *__restrict__ indices,
                            float *__restrict__ key_out,
                            float *__restrict__ value_out,
                            int count, int heads, int head_dim) {
    const long long task = (long long)blockIdx.x * (blockDim.x / LANES) + threadIdx.x / LANES;
    if (task >= (long long)count * heads) return;
    const int lane = threadIdx.x % LANES;
    const int token = task / heads, head = task % heads;
    const int slot = indices[token];

    const int groups = head_dim / kMxGroup;
    const long long dest = ((long long)token * heads + head) * head_dim;

    for (int group = lane; group < groups; group += LANES) {
        const long long g0 = (long long)group * kMxGroup;
        const long long src = mx_group_base(slot, head, group, heads, groups);
        const float key_scale = e8m0_decode(key_cache[src]);
        const float value_scale = e8m0_decode(value_cache[src]);
        for (int i = 0; i < kMxGroup; ++i) {
            key_out[dest + g0 + i] = e4m3fn_decode(key_cache[src + 1 + i]) * key_scale;
            value_out[dest + g0 + i] = e4m3fn_decode(value_cache[src + 1 + i]) * value_scale;
        }
    }
}

}  // namespace mxkv

// ===========================================================================
// self-checking harness: byte-exact vs a host replica of the same rules
// ===========================================================================
using namespace mxkv;

#define CK(x) do { hipError_t e=(x); if(e){printf("HIP %s @%d\n",hipGetErrorString(e),__LINE__);exit(1);} } while(0)

static int g_fail = 0;
static void report(const char *name, bool ok, const char *detail = "") {
    printf("%-42s %s %s\n", name, ok ? "PASS" : "FAIL", detail);
    if (!ok) ++g_fail;
}

// Independent spot-check of the codec primitives against hand-computed values.
//
// This exists because the byte-exact scatter check CANNOT catch a wrong spec:
// the host replica calls the same __host__ __device__ helpers as the kernel, so
// mutating a shared helper moves both sides together and they still agree. That
// check proves host/device consistency only. These constants are derived from
// the format definitions by hand and are the actual guard on correctness.
static void check_primitives() {
    struct { float in; uint8_t want; } e8[] = {
        {0.0f, 0},          // non-positive -> 0
        {-1.0f, 0},
        {1.0f, 127},        // ceil(log2 1) = 0
        {0.75f, 127},       // ceil(log2 .75) = 0   <- rounds UP, not down
        {1.5f, 128},        // ceil(log2 1.5) = 1
        {2.0f, 128},        // ceil(log2 2) = 1
        {0.5f, 126},        // ceil(log2 .5) = -1
    };
    bool ok = true;
    for (auto &c : e8) if (e8m0_encode_up(c.in) != c.want) {
        printf("  e8m0_encode_up(%g) = %u, want %u\n", c.in,
               (unsigned)e8m0_encode_up(c.in), (unsigned)c.want);
        ok = false;
    }
    report("e8m0_encode_up vs hand-computed", ok);

    struct { float in; uint8_t want; } e4[] = {
        {0.0f, 0x00}, {-0.0f, 0x80},
        {1.0f, 0x38},        // exp 7 -> (7<<3)|0
        {2.0f, 0x40},
        {448.0f, 0x7e},      // at/above max finite -> saturate
        {1e30f, 0x7e}, {-1e30f, 0xfe},
        {1.25f, 0x3a},       // 1.25 = 1.010b -> mantissa 2
    };
    ok = true;
    for (auto &c : e4) if (e4m3fn_encode(c.in) != c.want) {
        printf("  e4m3fn_encode(%g) = 0x%02x, want 0x%02x\n", c.in,
               e4m3fn_encode(c.in), c.want);
        ok = false;
    }
    report("e4m3fn_encode vs hand-computed", ok);

    // decode(encode(x)) must be idempotent on exactly-representable values
    ok = true;
    for (float v : {1.0f, 2.0f, 1.25f, -3.5f, 0.5f})
        if (e4m3fn_decode(e4m3fn_encode(v)) != v) { ok = false; }
    report("e4m3fn round-trips exact values", ok);
}

int main() {
    check_primitives();
    const int count = 96, heads = 4, head_dim = 128, max_slots = 256;
    const int groups = head_dim / kMxGroup;
    std::mt19937 rng(20260726);
    std::uniform_real_distribution<float> uf(-6.0f, 6.0f);

    std::vector<float> key((size_t)count * heads * head_dim);
    std::vector<float> value(key.size());
    for (auto &v : key) v = uf(rng);
    for (auto &v : value) v = uf(rng);
    // exercise the zero-amax group path and the saturation path
    for (int i = 0; i < head_dim; ++i) key[i] = 0.0f;
    for (int i = 0; i < kMxGroup; ++i) value[head_dim + i] = 1e30f * ((i & 1) ? -1.f : 1.f);

    std::vector<int> slots(count);
    for (int t = 0; t < count; ++t) slots[t] = (t % 7 == 3) ? -1 : (t * 2) % max_slots;

    const size_t cache_bytes = (size_t)max_slots * heads * groups * kMxBlockBytes;
    std::vector<uint8_t> kc_ref(cache_bytes, 0xAB), vc_ref(cache_bytes, 0xAB);

    // ---- host replica of the scatter ----
    for (int t = 0; t < count; ++t) {
        if (slots[t] < 0) continue;
        for (int h = 0; h < heads; ++h)
            for (int g = 0; g < groups; ++g) {
                const size_t src = ((size_t)t * heads + h) * head_dim + (size_t)g * kMxGroup;
                const size_t dst = mx_group_base(slots[t], h, g, heads, groups);
                float km = 0.f, vm = 0.f;
                for (int i = 0; i < kMxGroup; ++i) {
                    km = std::max(km, std::fabs(key[src + i]));
                    vm = std::max(vm, std::fabs(value[src + i]));
                }
                const uint8_t ks = e8m0_encode_up(km / kE4M3FnMax);
                const uint8_t vs = e8m0_encode_up(vm / kE4M3FnMax);
                kc_ref[dst] = ks; vc_ref[dst] = vs;
                const float ki = km > 0.f ? 1.f / e8m0_decode(ks) : 0.f;
                const float vi = vm > 0.f ? 1.f / e8m0_decode(vs) : 0.f;
                for (int i = 0; i < kMxGroup; ++i) {
                    kc_ref[dst + 1 + i] = e4m3fn_encode(key[src + i] * ki);
                    vc_ref[dst + 1 + i] = e4m3fn_encode(value[src + i] * vi);
                }
            }
    }

    float *dk, *dv, *dko, *dvo;
    int *dslots, *dinvalid;
    uint8_t *dkc, *dvc;
    CK(hipMalloc(&dk, key.size() * 4));   CK(hipMemcpy(dk, key.data(), key.size() * 4, hipMemcpyHostToDevice));
    CK(hipMalloc(&dv, value.size() * 4)); CK(hipMemcpy(dv, value.data(), value.size() * 4, hipMemcpyHostToDevice));
    CK(hipMalloc(&dslots, count * 4));    CK(hipMemcpy(dslots, slots.data(), count * 4, hipMemcpyHostToDevice));
    CK(hipMalloc(&dkc, cache_bytes));     CK(hipMemset(dkc, 0xAB, cache_bytes));
    CK(hipMalloc(&dvc, cache_bytes));     CK(hipMemset(dvc, 0xAB, cache_bytes));
    CK(hipMalloc(&dinvalid, 4));          CK(hipMemset(dinvalid, 0, 4));
    CK(hipMalloc(&dko, key.size() * 4));  CK(hipMalloc(&dvo, value.size() * 4));

    constexpr int LANES = 8, BLOCK = 256;
    const int tasks = count * heads;
    const int blocks = (tasks + (BLOCK / LANES) - 1) / (BLOCK / LANES);
    k_mx_scatter<LANES><<<blocks, BLOCK>>>(dk, dv, dslots, dkc, dvc, count, heads, head_dim, dinvalid);
    CK(hipDeviceSynchronize());
    if (hipGetLastError() != hipSuccess) { printf("SCATTER KERNEL ERR\n"); return 1; }

    std::vector<uint8_t> kc(cache_bytes), vc(cache_bytes);
    CK(hipMemcpy(kc.data(), dkc, cache_bytes, hipMemcpyDeviceToHost));
    CK(hipMemcpy(vc.data(), dvc, cache_bytes, hipMemcpyDeviceToHost));
    size_t kdiff = 0, vdiff = 0;
    for (size_t i = 0; i < cache_bytes; ++i) {
        kdiff += (kc[i] != kc_ref[i]);
        vdiff += (vc[i] != vc_ref[i]);
    }
    char detail[128];
    snprintf(detail, sizeof detail, "(%zu key / %zu value bytes differ of %zu)", kdiff, vdiff, cache_bytes);
    report("kv_cache_scatter_mxfp8 (byte-exact)", kdiff == 0 && vdiff == 0, detail);

    int invalid = 1;
    CK(hipMemcpy(&invalid, dinvalid, 4, hipMemcpyDeviceToHost));
    report("  scatter rejects nothing on finite input", invalid == 0);

    // untouched slots must still hold the 0xAB fill (scatter is incremental)
    bool untouched_ok = true;
    for (int s = 1; s < max_slots && untouched_ok; s += 2) {   // odd slots never written
        const size_t base = mx_group_base(s, 0, 0, heads, groups);
        if (kc[base] != 0xAB) untouched_ok = false;
    }
    report("  scatter leaves unwritten slots untouched", untouched_ok);

    // ---- gather round-trip ----
    std::vector<int> indices(count);
    for (int t = 0; t < count; ++t) indices[t] = slots[t] >= 0 ? slots[t] : 0;
    int *dind; CK(hipMalloc(&dind, count * 4));
    CK(hipMemcpy(dind, indices.data(), count * 4, hipMemcpyHostToDevice));
    k_mx_gather<LANES><<<blocks, BLOCK>>>(dkc, dvc, dind, dko, dvo, count, heads, head_dim);
    CK(hipDeviceSynchronize());
    if (hipGetLastError() != hipSuccess) { printf("GATHER KERNEL ERR\n"); return 1; }

    std::vector<float> kout(key.size()), vout(value.size());
    CK(hipMemcpy(kout.data(), dko, key.size() * 4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(vout.data(), dvo, value.size() * 4, hipMemcpyDeviceToHost));

    size_t gmis = 0;
    for (int t = 0; t < count; ++t)
        for (int h = 0; h < heads; ++h)
            for (int g = 0; g < groups; ++g) {
                const size_t base = mx_group_base(indices[t], h, g, heads, groups);
                const float ks = e8m0_decode(kc_ref[base]);
                for (int i = 0; i < kMxGroup; ++i) {
                    const float want = e4m3fn_decode(kc_ref[base + 1 + i]) * ks;
                    const size_t o = ((size_t)t * heads + h) * head_dim + (size_t)g * kMxGroup + i;
                    if (kout[o] != want) ++gmis;
                }
            }
    snprintf(detail, sizeof detail, "(%zu of %zu values differ)", gmis, key.size());
    report("kv_cache_gather_mxfp8 (exact decode)", gmis == 0, detail);

    // round-trip error is bounded by the format, not by the kernel
    // Only tokens whose slot was actually written are comparable: scatter skips
    // slots[t] < 0, leaving those cache pages at their prior contents, so
    // gathering them back yields whatever was there (the 0xAB fill decodes to
    // ~2^44). Comparing them would measure the fill, not the codec.
    double worst = 0.0;
    for (int t = 0; t < count; ++t) {
        if (slots[t] < 0) continue;
        for (int h = 0; h < heads; ++h)
            for (int d = 0; d < head_dim; ++d) {
                const size_t i = ((size_t)t * heads + h) * head_dim + d;
                if (!std::isfinite(key[i]) || std::fabs(key[i]) > 1e20) continue;
                const double rel = std::fabs((double)kout[i] - key[i]) /
                                   std::max(1.0, std::fabs((double)key[i]));
                worst = std::max(worst, rel);
            }
    }
    snprintf(detail, sizeof detail, "(worst rel %.3e, e4m3 grid is ~6%%)", worst);
    report("  round-trip within E4M3FN resolution", worst < 0.08, detail);

    printf("\n%s (%d failures)\n", g_fail ? "FAILED" : "ALL PASS", g_fail);
    return g_fail ? 1 : 0;
}
