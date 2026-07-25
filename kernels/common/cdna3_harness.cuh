/**
 * @file
 * @brief Shared CDNA3 (gfx942) kernel-harness header: fp64-oracle comparison,
 *        deterministic host RNG, wave64 reduction helpers, and a timing loop
 *        that reports median/min/max plus achieved GB/s and TFLOP/s.
 *
 * Every `kernels/<family>/<op>/variants/rocm_cdna3/<op>.cu` harness includes
 * this instead of re-implementing the pattern. The harnesses are deliberately
 * torch-free standalone binaries: kernel objects link system ROCm 7.2 and
 * conflict with the torch wheel's ROCm 6.4 HSA symbols when co-loaded.
 *
 * Tolerances mirror `../../registry/tolerances.yaml` so ROCm accepts exactly
 * what the Metal and CPU backends accept. Two independent verdicts are
 * available because they catch different failures:
 *
 *   - elementwise rtol/atol  -- catches a single bad lane or index;
 *   - aggregate rel-L1 + cosine -- catches systematic drift and layout errors
 *     that elementwise checks pass when the tolerance is loose (quantized).
 *
 * `Comparison::pass()` requires both. Note the CDNA3 test contract: bit-exact
 * kernel-vs-kernel self-checks can legitimately fail because cross-warp merge
 * order is not fixed on the 64-wide wavefront -- compare against the fp64
 * oracle with a real tolerance, or relax self-checks to ~1 fp16 ULP.
 *
 * Usage:
 *   #include "../../../../common/cdna3_harness.cuh"
 *   qc::Rng rng(1234);
 *   auto dx = qc::dnew(host_x);
 *   auto got = qc::d2h(dout, n);
 *   auto cmp = qc::compare(got, reference_fp64, qc::Tol::fp32());
 *   ok &= cmp.report("rms_norm D=4096");
 *   auto b = qc::bench([&]{ kernel<<<g,b>>>(...); });
 *   b.report_bandwidth("candidate", bytes_moved);
 */
#ifndef QUIXICORE_ROCM_CDNA3_HARNESS_CUH
#define QUIXICORE_ROCM_CDNA3_HARNESS_CUH

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#define QC_CHECK(x)                                                       \
    do {                                                                  \
        hipError_t _e = (x);                                              \
        if (_e != hipSuccess) {                                           \
            std::printf("HIP error: %s @ %s:%d\n", hipGetErrorString(_e), \
                        __FILE__, __LINE__);                              \
            std::exit(1);                                                 \
        }                                                                 \
    } while (0)

/** Synchronize and surface any async launch failure. Call after each kernel. */
#define QC_SYNC()                          \
    do {                                   \
        QC_CHECK(hipDeviceSynchronize());  \
        QC_CHECK(hipGetLastError());       \
    } while (0)

namespace qc {

// ---------------------------------------------------------------------------
// gfx942 geometry
// ---------------------------------------------------------------------------

/// CDNA3 wavefront width. Note this is 64, not NVIDIA's 32: `__shfl_*_sync`
/// would need a 64-bit lane mask here, so the mask-free `__shfl_*` is used.
constexpr int kWave = 64;
/// Default block size: 4 wavefronts.
constexpr int kThreads = 256;
constexpr int kWavesPerBlock = kThreads / kWave;
/// gfx942 LDS budget per workgroup.
constexpr int kLdsBytes = 64 * 1024;

/// Blocks needed to give each of `rows` rows its own wavefront.
__host__ __device__ __forceinline__ int wave_blocks(size_t rows,
                                                    int waves_per_block = kWavesPerBlock) {
    return static_cast<int>((rows + waves_per_block - 1) / waves_per_block);
}

/// Blocks needed to cover `n` elements at `per_block` elements per block.
__host__ __device__ __forceinline__ int grid_for(size_t n, int per_block) {
    return static_cast<int>((n + per_block - 1) / per_block);
}

// ---------------------------------------------------------------------------
// Wave64 reductions (every lane receives the full-wavefront result)
// ---------------------------------------------------------------------------

template <typename T>
__device__ __forceinline__ T wave_reduce_sum(T v) {
#pragma unroll
    for (int offset = kWave / 2; offset > 0; offset >>= 1)
        v += __shfl_xor(v, offset, kWave);
    return v;
}

template <typename T>
__device__ __forceinline__ T wave_reduce_max(T v) {
#pragma unroll
    for (int offset = kWave / 2; offset > 0; offset >>= 1)
        v = max(v, __shfl_xor(v, offset, kWave));
    return v;
}

template <typename T>
__device__ __forceinline__ T wave_reduce_min(T v) {
#pragma unroll
    for (int offset = kWave / 2; offset > 0; offset >>= 1)
        v = min(v, __shfl_xor(v, offset, kWave));
    return v;
}

/// Wavefront-wide inclusive prefix sum (Hillis-Steele over 64 lanes).
template <typename T>
__device__ __forceinline__ T wave_inclusive_scan(T v) {
    const int lane = threadIdx.x & (kWave - 1);
#pragma unroll
    for (int offset = 1; offset < kWave; offset <<= 1) {
        const T n = __shfl_up(v, offset, kWave);
        if (lane >= offset) v += n;
    }
    return v;
}

/// Argmax across a wavefront, tie-broken toward the LOWER index. The tie rule
/// is contractual for LM-head/argmax operations -- do not "optimize" it away.
template <typename T>
__device__ __forceinline__ void wave_reduce_argmax(T &value, int &index) {
#pragma unroll
    for (int offset = kWave / 2; offset > 0; offset >>= 1) {
        const T ov = __shfl_xor(value, offset, kWave);
        const int oi = __shfl_xor(index, offset, kWave);
        if (ov > value || (ov == value && oi < index)) {
            value = ov;
            index = oi;
        }
    }
}

/// Block-wide sum across all wavefronts in the block. `scratch` must hold at
/// least `blockDim.x / kWave` elements. Contains its own barriers.
template <typename T>
__device__ __forceinline__ T block_reduce_sum(T v, T *scratch) {
    const int lane = threadIdx.x & (kWave - 1);
    const int wave = threadIdx.x >> 6;
    const int waves = static_cast<int>(blockDim.x) >> 6;
    v = wave_reduce_sum(v);
    if (lane == 0) scratch[wave] = v;
    __syncthreads();
    T total = T(0);
    for (int w = 0; w < waves; ++w) total += scratch[w];
    return total;
}

// ---------------------------------------------------------------------------
// Host-side dtype conversion (everything compares in double)
// ---------------------------------------------------------------------------

__host__ inline double to_double(float v) { return static_cast<double>(v); }
__host__ inline double to_double(double v) { return v; }
__host__ inline double to_double(int v) { return static_cast<double>(v); }
__host__ inline double to_double(long v) { return static_cast<double>(v); }
__host__ inline double to_double(long long v) { return static_cast<double>(v); }
__host__ inline double to_double(unsigned v) { return static_cast<double>(v); }
__host__ inline double to_double(unsigned long v) { return static_cast<double>(v); }
__host__ inline double to_double(__half v) { return static_cast<double>(__half2float(v)); }
__host__ inline double to_double(__hip_bfloat16 v) {
    return static_cast<double>(__bfloat162float(v));
}

/// Round a double through fp16 storage. Use when the contract says the result
/// is rounded to output storage BEFORE a comparison or an argmax.
__host__ inline double round_fp16(double v) {
    return static_cast<double>(__half2float(__float2half(static_cast<float>(v))));
}
/// Round a double through bf16 storage (round-to-nearest-even).
__host__ inline double round_bf16(double v) {
    return static_cast<double>(__bfloat162float(__float2bfloat16(static_cast<float>(v))));
}

// ---------------------------------------------------------------------------
// Tolerances -- mirrors ../../registry/tolerances.yaml
// ---------------------------------------------------------------------------

struct Tol {
    double rtol;
    double atol;
    /// Aggregate relative-L1 ceiling.
    double rel;
    /// Aggregate cosine-similarity floor.
    double cosine;
    const char *name;

    static Tol fp32() { return {1e-5, 1e-6, 1e-5, 0.9999999, "fp32"}; }
    static Tol fp16() { return {1e-3, 1e-3, 2e-3, 0.999995, "fp16"}; }
    static Tol bf16() { return {2e-3, 2e-3, 4e-3, 0.99999, "bf16"}; }
    static Tol fp8() { return {2e-2, 2e-2, 3e-2, 0.9995, "fp8"}; }
    static Tol quantized() { return {3e-2, 3e-2, 4e-2, 0.999, "quantized"}; }
    /// Bit-exact: integer paths, packing/unpacking, index outputs.
    static Tol exact() { return {0.0, 0.0, 0.0, 1.0, "exact"}; }

    // -- Storage-ULP variants -------------------------------------------------
    // The registry tolerances describe how far a *computation* may drift. They
    // are not always achievable when the result is *stored* in a narrow float.
    //
    // bf16 keeps 8 mantissa bits, so one ULP is a relative 2^-8 = 3.9e-3. At
    // magnitude ~1.4 that is an absolute step of 7.8e-3 -- nearly 4x the
    // registry's bf16 atol of 2e-3. A kernel whose output lands one correctly-
    // rounded bf16 step from the oracle therefore fails an elementwise check it
    // could never have passed, and the check rewards luck rather than accuracy.
    //
    // These variants raise ONLY the elementwise relative bound to one storage
    // ULP. The aggregate rel-L1 and cosine bounds are untouched, and they remain
    // the real correctness signal: a genuinely wrong kernel moves those, while
    // one-ULP storage noise does not.
    //
    // Use these when the kernel's OUTPUT is bf16/fp16. Keep the plain fp32/fp16/
    // bf16 tolerances when the output is fp32 and the dtype only describes the
    // inputs.
    static Tol bf16_output() {
        Tol t = bf16();
        t.rtol = 3.90625e-3;  // 2^-8, one bf16 ULP
        t.name = "bf16-out";
        return t;
    }
    static Tol fp16_output() {
        Tol t = fp16();
        t.rtol = 9.765625e-4;  // 2^-10, one fp16 ULP
        t.name = "fp16-out";
        return t;
    }

    /// Relax only the elementwise bound, keeping the aggregate bounds. Use for
    /// kernels whose reduction order differs from the oracle's.
    Tol with_elementwise(double r, double a) const {
        Tol t = *this;
        t.rtol = r;
        t.atol = a;
        return t;
    }
};

// ---------------------------------------------------------------------------
// Comparison against an fp64 oracle
// ---------------------------------------------------------------------------

struct Comparison {
    size_t count = 0;
    size_t mismatches = 0;
    /// Aggregate sum|got-ref| / sum|ref|.
    double rel = 0.0;
    double max_abs = 0.0;
    double max_rel = 0.0;
    double cosine = 1.0;
    /// Flat index of the worst elementwise violation, or -1.
    long worst_index = -1;
    double worst_got = 0.0;
    double worst_ref = 0.0;
    Tol tol = Tol::fp32();

    bool pass() const {
        return mismatches == 0 && rel <= tol.rel && cosine >= tol.cosine &&
               std::isfinite(rel) && std::isfinite(cosine);
    }

    /// Print one result line. Returns pass() so callers can `ok &= report(...)`.
    bool report(const std::string &label) const {
        const bool ok = pass();
        std::printf("  %-52s [%s] n=%zu bad=%zu rel=%.3e max=%.3e cos=%.9f  %s\n",
                    label.c_str(), tol.name, count, mismatches, rel, max_abs,
                    cosine, ok ? "PASS" : "FAIL");
        if (!ok && worst_index >= 0)
            std::printf("      worst @%ld: got %.9g vs ref %.9g (abs %.3e)\n",
                        worst_index, worst_got, worst_ref,
                        std::fabs(worst_got - worst_ref));
        return ok;
    }
};

/// Compare a device-produced result against an fp64 host oracle.
template <typename G, typename R>
Comparison compare(const std::vector<G> &got, const std::vector<R> &ref, Tol tol) {
    Comparison c;
    c.tol = tol;
    c.count = std::min(got.size(), ref.size());
    if (got.size() != ref.size()) {
        std::printf("      size mismatch: got %zu vs ref %zu\n", got.size(), ref.size());
        c.mismatches = 1;
        return c;
    }
    double num = 0.0, den = 0.0, dot = 0.0, ng = 0.0, nr = 0.0;
    for (size_t i = 0; i < c.count; ++i) {
        const double g = to_double(got[i]);
        const double r = to_double(ref[i]);
        if (!std::isfinite(g) && std::isfinite(r)) {
            ++c.mismatches;
            if (c.worst_index < 0) { c.worst_index = (long)i; c.worst_got = g; c.worst_ref = r; }
            continue;
        }
        const double abs_err = std::fabs(g - r);
        const double rel_err = abs_err / std::max(std::fabs(r), 1e-300);
        if (abs_err > c.max_abs) {
            c.max_abs = abs_err;
            c.worst_index = (long)i;
            c.worst_got = g;
            c.worst_ref = r;
        }
        if (std::fabs(r) > 1e-12) c.max_rel = std::max(c.max_rel, rel_err);
        if (abs_err > tol.atol + tol.rtol * std::fabs(r)) ++c.mismatches;
        num += abs_err;
        den += std::fabs(r);
        dot += g * r;
        ng += g * g;
        nr += r * r;
    }
    c.rel = num / std::max(den, 1e-300);
    const double norm = std::sqrt(ng * nr);
    // Two all-zero vectors are identical, not undefined.
    c.cosine = norm > 1e-300 ? dot / norm : 1.0;
    return c;
}

// ---------------------------------------------------------------------------
// Deterministic host RNG -- same seed gives the same data every run, on any host
// ---------------------------------------------------------------------------

class Rng {
public:
    explicit Rng(uint64_t seed = 1234) : gen_(static_cast<uint32_t>(seed)) {}

    float normal(float mean = 0.f, float stddev = 1.f) {
        std::normal_distribution<float> d(mean, stddev);
        return d(gen_);
    }
    float uniform(float lo = -1.f, float hi = 1.f) {
        std::uniform_real_distribution<float> d(lo, hi);
        return d(gen_);
    }
    int integer(int lo, int hi) {  // inclusive
        std::uniform_int_distribution<int> d(lo, hi);
        return d(gen_);
    }

    std::vector<float> normals(size_t n, float stddev = 1.f) {
        std::vector<float> v(n);
        for (auto &x : v) x = normal(0.f, stddev);
        return v;
    }
    std::vector<float> uniforms(size_t n, float lo = -1.f, float hi = 1.f) {
        std::vector<float> v(n);
        for (auto &x : v) x = uniform(lo, hi);
        return v;
    }
    std::vector<int32_t> integers(size_t n, int lo, int hi) {
        std::vector<int32_t> v(n);
        for (auto &x : v) x = integer(lo, hi);
        return v;
    }

    /// Values spanning the awkward cases every kernel should survive: zero,
    /// denormal, tiny, huge, and both signs. Placed at fixed positions so a
    /// failure is reproducible.
    std::vector<float> adversarial(size_t n) {
        std::vector<float> v = normals(n);
        const float specials[] = {0.f,    -0.f,   1e-38f, -1e-38f, 1e30f,
                                  -1e30f, 1.f,    -1.f,   65504.f, -65504.f};
        const size_t k = sizeof(specials) / sizeof(float);
        for (size_t i = 0; i < k && i < n; ++i) v[i] = specials[i];
        return v;
    }

private:
    std::mt19937 gen_;
};

// ---------------------------------------------------------------------------
// Device buffer helpers
// ---------------------------------------------------------------------------

template <typename T>
T *dnew(const std::vector<T> &host) {
    T *d = nullptr;
    QC_CHECK(hipMalloc(&d, host.size() * sizeof(T)));
    QC_CHECK(hipMemcpy(d, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice));
    return d;
}

template <typename T>
T *dzero(size_t n) {
    T *d = nullptr;
    QC_CHECK(hipMalloc(&d, n * sizeof(T)));
    QC_CHECK(hipMemset(d, 0, n * sizeof(T)));
    return d;
}

template <typename T>
std::vector<T> d2h(const T *d, size_t n) {
    std::vector<T> h(n);
    QC_CHECK(hipMemcpy(h.data(), d, n * sizeof(T), hipMemcpyDeviceToHost));
    return h;
}

/// Frees every pointer handed to it; ignores nulls.
template <typename... Ts>
void dfree(Ts *...ptrs) {
    void *list[] = {static_cast<void *>(ptrs)...};
    for (void *p : list)
        if (p) QC_CHECK(hipFree(p));
}

/// Convert an f32 host vector to fp16/bf16 device storage.
template <typename T>
std::vector<T> to_storage(const std::vector<float> &src);

template <>
inline std::vector<__half> to_storage<__half>(const std::vector<float> &src) {
    std::vector<__half> out(src.size());
    for (size_t i = 0; i < src.size(); ++i) out[i] = __float2half(src[i]);
    return out;
}
template <>
inline std::vector<__hip_bfloat16> to_storage<__hip_bfloat16>(const std::vector<float> &src) {
    std::vector<__hip_bfloat16> out(src.size());
    for (size_t i = 0; i < src.size(); ++i) out[i] = __float2bfloat16(src[i]);
    return out;
}
template <>
inline std::vector<float> to_storage<float>(const std::vector<float> &src) {
    return src;
}

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

struct Bench {
    double median_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
    double mean_ms = 0.0;
    int warmups = 0;
    int iters = 0;

    double gbps(double bytes) const { return bytes / (median_ms * 1e-3) / 1e9; }
    double tflops(double flops) const { return flops / (median_ms * 1e-3) / 1e12; }
    /// max/min spread; the perf gate wants variance reported, not just a median.
    double spread() const { return min_ms > 0.0 ? max_ms / min_ms : 0.0; }

    void report(const std::string &label) const {
        std::printf("  %-40s %8.4f ms  (min %.4f max %.4f spread %.2fx, w%d/i%d)\n",
                    label.c_str(), median_ms, min_ms, max_ms, spread(), warmups, iters);
    }
    void report_bandwidth(const std::string &label, double bytes) const {
        std::printf("  %-40s %8.4f ms  %8.1f GB/s  (min %.4f max %.4f spread %.2fx, w%d/i%d)\n",
                    label.c_str(), median_ms, gbps(bytes), min_ms, max_ms, spread(),
                    warmups, iters);
    }
    void report_compute(const std::string &label, double flops) const {
        std::printf("  %-40s %8.4f ms  %8.1f TFLOP/s  (min %.4f max %.4f spread %.2fx, w%d/i%d)\n",
                    label.c_str(), median_ms, tflops(flops), min_ms, max_ms, spread(),
                    warmups, iters);
    }
};

/// Time `fn` with HIP events. Each iteration is timed individually so min/max
/// are real per-iteration extremes rather than an averaged window.
template <typename FN>
Bench bench(FN &&fn, int warmups = 10, int iters = 50) {
    for (int i = 0; i < warmups; ++i) fn();
    QC_SYNC();

    hipEvent_t start, stop;
    QC_CHECK(hipEventCreate(&start));
    QC_CHECK(hipEventCreate(&stop));
    std::vector<float> samples(iters);
    for (int i = 0; i < iters; ++i) {
        QC_CHECK(hipEventRecord(start));
        fn();
        QC_CHECK(hipEventRecord(stop));
        QC_CHECK(hipEventSynchronize(stop));
        QC_CHECK(hipEventElapsedTime(&samples[i], start, stop));
    }
    QC_CHECK(hipEventDestroy(start));
    QC_CHECK(hipEventDestroy(stop));

    std::sort(samples.begin(), samples.end());
    Bench b;
    b.warmups = warmups;
    b.iters = iters;
    b.median_ms = samples[iters / 2];
    b.min_ms = samples.front();
    b.max_ms = samples.back();
    b.mean_ms = std::accumulate(samples.begin(), samples.end(), 0.0) / iters;
    return b;
}

/// Speedup of `baseline` over `candidate`, printed the way the notebook wants.
inline void report_ab(const std::string &label, const Bench &baseline,
                      const Bench &candidate) {
    const double x = baseline.median_ms / candidate.median_ms;
    std::printf("  %-40s baseline %.4f ms -> candidate %.4f ms  = %.2fx (%+.1f%%)\n",
                label.c_str(), baseline.median_ms, candidate.median_ms, x,
                (x - 1.0) * 100.0);
}

// ---------------------------------------------------------------------------
// Run banner -- the perf gate requires GPU/runtime provenance in every result
// ---------------------------------------------------------------------------

inline void print_environment(const char *kernel_name) {
    hipDeviceProp_t prop{};
    int device = 0;
    QC_CHECK(hipGetDevice(&device));
    QC_CHECK(hipGetDeviceProperties(&prop, device));
    int runtime = 0, driver = 0;
    QC_CHECK(hipRuntimeGetVersion(&runtime));
    QC_CHECK(hipDriverGetVersion(&driver));
    std::printf("== %s ==\n", kernel_name);
    std::printf("   GPU: %s (%s)  CUs: %d  LDS/block: %zu KB\n", prop.name,
                prop.gcnArchName, prop.multiProcessorCount,
                prop.sharedMemPerBlock / 1024);
    std::printf("   HIP runtime: %d  driver: %d  device: %d of many\n", runtime, driver,
                device);
}

/// Standard exit: prints the verdict line the test scripts grep for.
inline int finish(bool ok) {
    std::printf("%s\n", ok ? "ALL PASS" : "FAIL");
    return ok ? 0 : 1;
}

/// True when the harness was invoked as `./<op>.out --bench`.
inline bool bench_requested(int argc, char **argv) {
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], "--bench") == 0) return true;
    return false;
}

}  // namespace qc

#endif  // QUIXICORE_ROCM_CDNA3_HARNESS_CUH
