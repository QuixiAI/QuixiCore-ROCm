#include "hip/hip_runtime.h"
/**
 * @file
 * @brief Check this backend's quant decoders against the umbrella test vectors.
 *
 * The vectors in ../../../test-vectors/quant/ are derived from specs/formats/,
 * not from any implementation, so this answers "does ROCm agree with the spec"
 * rather than "does ROCm agree with itself". That distinction is the whole
 * point: the E8M0 spec has said code 0 decodes to 2^-127 for as long as it has
 * existed, and seven kernel families here decoded it to +0.0, because nothing
 * ever compared the two.
 *
 * Comparison is on IEEE-754 bits. 2^-127 is subnormal, so any tolerance-based
 * check would cheerfully accept +0.0 for it and report success.
 *
 * Build: make
 * Run:   HIP_VISIBLE_DEVICES=0 ./quant_conformance.out ../../../test-vectors/quant
 */
#include "quant_primitives.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace quixicore::quant;

static int g_fail = 0;
#define CK(x)                                                       \
    do {                                                            \
        hipError_t e = (x);                                         \
        if (e) {                                                    \
            printf("HIP %s @%d\n", hipGetErrorString(e), __LINE__); \
            exit(1);                                                \
        }                                                           \
    } while (0)

__global__ void run_e8m0(const uint8_t* codes, int n, float* exact,
                         float* fast, float* ggml) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    exact[i] = e8m0_decode_exact(codes[i]);
    fast[i] = e8m0_decode_fast(codes[i]);
    ggml[i] = e8m0_decode_ggml(codes[i]);
}

static uint32_t as_bits(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return u;
}

/// Minimal extraction of the fields this test needs; the vectors are flat
/// enough that a real JSON parser would be more dependency than value.
static std::string slurp(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        printf("cannot open %s\n", path.c_str());
        exit(1);
    }
    std::string s;
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) s.append(buf, n);
    fclose(f);
    return s;
}

/// Pull every `"key": <int>` and every `"bits": "0x..."` in document order.
static void scan(const std::string& j, const char* key,
                 std::vector<long>& out_num, std::vector<std::string>& out_str) {
    const std::string pat = std::string("\"") + key + "\":";
    size_t p = 0;
    while ((p = j.find(pat, p)) != std::string::npos) {
        size_t q = p + pat.size();
        while (q < j.size() && (j[q] == ' ' || j[q] == '\n')) ++q;
        if (j[q] == '"') {
            size_t e = j.find('"', q + 1);
            out_str.push_back(j.substr(q + 1, e - q - 1));
            out_num.push_back(-1);
        } else if (j.compare(q, 4, "null") == 0) {
            out_str.push_back("null");
            out_num.push_back(-1);
        } else {
            out_str.push_back("");
            out_num.push_back(strtol(j.c_str() + q, nullptr, 10));
        }
        p = q;
    }
}

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "../../../test-vectors/quant";

    // Two contracts, deliberately separate: MX for the spec decoder, ggml's
    // for the GGUF one. Checking both against MX would demand ggml return NaN
    // at 255, which would break reading the GGUF files this exists to read.
    const std::string j = slurp(dir + "/e8m0.json");
    const std::string jg = slurp(dir + "/e8m0_gguf.json");
    std::vector<long> gcodes_n, gdummy_n;
    std::vector<std::string> gcodes_s, gbits_s;
    scan(jg, "code", gcodes_n, gcodes_s);
    scan(jg, "bits", gdummy_n, gbits_s);
    std::vector<long> codes_n, dummy_n;
    std::vector<std::string> codes_s, bits_s;
    scan(j, "code", codes_n, codes_s);
    scan(j, "bits", dummy_n, bits_s);
    if (codes_n.size() != bits_s.size() || codes_n.empty()) {
        printf("vector parse failed (%zu codes, %zu bits)\n", codes_n.size(),
               bits_s.size());
        return 1;
    }

    const int n = (int)codes_n.size();
    std::vector<uint8_t> codes(n);
    for (int i = 0; i < n; ++i) codes[i] = (uint8_t)codes_n[i];

    uint8_t* d_codes;
    float *d_exact, *d_fast, *d_ggml;
    CK(hipMalloc(&d_codes, n));
    CK(hipMalloc(&d_exact, n * 4));
    CK(hipMalloc(&d_fast, n * 4));
    CK(hipMalloc(&d_ggml, n * 4));
    CK(hipMemcpy(d_codes, codes.data(), n, hipMemcpyHostToDevice));
    run_e8m0<<<(n + 63) / 64, 64>>>(d_codes, n, d_exact, d_fast, d_ggml);
    CK(hipDeviceSynchronize());
    std::vector<float> exact(n), fast(n), ggml(n);
    CK(hipMemcpy(exact.data(), d_exact, n * 4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(fast.data(), d_fast, n * 4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(ggml.data(), d_ggml, n * 4, hipMemcpyDeviceToHost));

    printf("E8M0 against %s/e8m0.json (%d codes)\n", dir.c_str(), n);
    printf("  %-6s %-12s %-12s %-12s %s\n", "code", "mx spec", "exact", "ggml",
           "fast");
    int exact_bad = 0, ggml_bad = 0, fast_div = 0;
    for (int i = 0; i < n; ++i) {
        const bool want_nan = bits_s[i] == "null";
        const uint32_t want = want_nan ? 0 : (uint32_t)strtoul(
                                             bits_s[i].c_str(), nullptr, 16);
        const bool e_ok = want_nan ? std::isnan(exact[i]) : as_bits(exact[i]) == want;
        // ggml is scored against e8m0_gguf.json, by code.
        bool g_ok = true;
        for (size_t k = 0; k < gcodes_n.size(); ++k) {
            if (gcodes_n[k] != codes_n[i]) continue;
            const uint32_t gw = (uint32_t)strtoul(gbits_s[k].c_str(), nullptr, 16);
            g_ok = as_bits(ggml[i]) == gw;
            break;
        }
        const bool f_ok = want_nan ? std::isnan(fast[i]) : as_bits(fast[i]) == want;
        if (!e_ok) ++exact_bad;
        if (!g_ok) ++ggml_bad;
        if (!f_ok) ++fast_div;
        if (!e_ok || !g_ok || !f_ok)
            printf("  %-6ld %-12s %-12s %-12s %s\n", codes_n[i],
                   want_nan ? "nan" : bits_s[i].c_str(),
                   e_ok ? "ok" : "MISMATCH", g_ok ? "ok" : "MISMATCH",
                   f_ok ? "ok" : "diverges");
    }

    printf("\n  e8m0_decode_exact : %s (%d/%d)\n",
           exact_bad ? "FAIL" : "conformant", n - exact_bad, n);
    printf("  e8m0_decode_ggml  : %s vs e8m0_gguf.json (%d/%d)\n",
           ggml_bad ? "FAIL" : "conformant", n - ggml_bad, n);
    printf("  e8m0_decode_fast  : diverges at %d/%d codes, as documented\n",
           fast_div, n);
    if (exact_bad || ggml_bad) g_fail = 1;

    // The fast form is only sound because of the producer contract. If it ever
    // stops diverging, or starts diverging somewhere new, the docs are stale.
    if (fast_div == 0) {
        printf("  note: fast no longer diverges -- update the header\n");
        g_fail = 1;
    }

    printf("\n%s\n", g_fail ? "FAILED" : "ALL PASS");
    return g_fail;
}
