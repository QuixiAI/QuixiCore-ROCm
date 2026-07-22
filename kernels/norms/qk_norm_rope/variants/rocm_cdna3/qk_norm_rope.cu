/**
 * @file
 * @brief CDNA3 port of the Metal qk_norm_rope contract.
 *
 * qkv is packed as (T, (Hq + Hk + Hv) * D), with each token/head row
 * contiguous. Q and K heads get RMSNorm over D, per-dim weights, and full RoPE.
 * V heads copy through. D is expected to be 64, 128, or 256.
 */
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#define CK(x) do { \
    hipError_t e = (x); \
    if (e != hipSuccess) { \
        printf("HIP %s @%d\n", hipGetErrorString(e), __LINE__); \
        exit(1); \
    } \
} while (0)

__device__ __forceinline__ float bf(const __hip_bfloat16 v) {
    return __bfloat162float(v);
}

__device__ __forceinline__ float block_sum(float v) {
    __shared__ float smem[256];
    const int tid = threadIdx.x;
    smem[tid] = v;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    return smem[0];
}

__global__ void qk_norm_rope_kernel(const __hip_bfloat16* __restrict__ qkv,
                                    const __hip_bfloat16* __restrict__ q_weight,
                                    const __hip_bfloat16* __restrict__ k_weight,
                                    const __hip_bfloat16* __restrict__ cosb,
                                    const __hip_bfloat16* __restrict__ sinb,
                                    const int* __restrict__ positions,
                                    __hip_bfloat16* __restrict__ out,
                                    int T,
                                    int hq,
                                    int hk,
                                    int hv,
                                    int D,
                                    float eps,
                                    int interleaved,
                                    int gemma) {
    const int head = blockIdx.x;
    const int token = blockIdx.y;
    const int HT = hq + hk + hv;
    if (token >= T || head >= HT) return;
    const long base = (long(token) * HT + head) * D;
    const int tid = threadIdx.x;

    if (head >= hq + hk) {
        for (int d = tid; d < D; d += blockDim.x) out[base + d] = qkv[base + d];
        return;
    }

    float ss = 0.0f;
    for (int d = tid; d < D; d += blockDim.x) {
        const float v = bf(qkv[base + d]);
        ss += v * v;
    }
    ss = block_sum(ss);
    const float inv_rms = rsqrtf(ss / float(D) + eps);
    const __hip_bfloat16* w = (head < hq) ? q_weight : k_weight;
    const int pos = positions[token];
    const long csbase = long(pos) * (D / 2);

    if (interleaved == 0) {
        for (int d = tid; d < D / 2; d += blockDim.x) {
            float w0 = bf(w[d]);
            float w1 = bf(w[d + D / 2]);
            if (gemma) {
                w0 += 1.0f;
                w1 += 1.0f;
            }
            const float x0 = bf(qkv[base + d]) * inv_rms * w0;
            const float x1 = bf(qkv[base + d + D / 2]) * inv_rms * w1;
            const float c = bf(cosb[csbase + d]);
            const float s = bf(sinb[csbase + d]);
            out[base + d] = __float2bfloat16(x0 * c - x1 * s);
            out[base + d + D / 2] = __float2bfloat16(x1 * c + x0 * s);
        }
    } else {
        for (int d = tid * 2; d < D; d += blockDim.x * 2) {
            float w0 = bf(w[d]);
            float w1 = bf(w[d + 1]);
            if (gemma) {
                w0 += 1.0f;
                w1 += 1.0f;
            }
            const float x0 = bf(qkv[base + d]) * inv_rms * w0;
            const float x1 = bf(qkv[base + d + 1]) * inv_rms * w1;
            const float c = bf(cosb[csbase + d / 2]);
            const float s = bf(sinb[csbase + d / 2]);
            out[base + d] = __float2bfloat16(x0 * c - x1 * s);
            out[base + d + 1] = __float2bfloat16(x0 * s + x1 * c);
        }
    }
}

template <typename T>
static T* dnew(const std::vector<T>& h) {
    T* d = nullptr;
    CK(hipMalloc(&d, h.size() * sizeof(T)));
    CK(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice));
    return d;
}

template <typename T>
static std::vector<T> d2h(const T* d, size_t n) {
    std::vector<T> h(n);
    CK(hipMemcpy(h.data(), d, n * sizeof(T), hipMemcpyDeviceToHost));
    return h;
}

template <typename FN>
static float bench_ms(FN&& fn, int warmup = 20, int iters = 100) {
    for (int i = 0; i < warmup; ++i) fn();
    CK(hipDeviceSynchronize());
    std::vector<float> t(iters);
    hipEvent_t a, b;
    CK(hipEventCreate(&a));
    CK(hipEventCreate(&b));
    for (int i = 0; i < iters; ++i) {
        CK(hipEventRecord(a));
        fn();
        CK(hipEventRecord(b));
        CK(hipEventSynchronize(b));
        CK(hipEventElapsedTime(&t[i], a, b));
    }
    CK(hipEventDestroy(a));
    CK(hipEventDestroy(b));
    std::sort(t.begin(), t.end());
    return t[iters / 2];
}

static std::vector<__hip_bfloat16> bf16v(const std::vector<float>& x) {
    std::vector<__hip_bfloat16> y(x.size());
    for (size_t i = 0; i < x.size(); ++i) y[i] = __float2bfloat16(x[i]);
    return y;
}

static float hbf(const __hip_bfloat16 x) {
    return __bfloat162float(x);
}

static std::vector<float> reference(const std::vector<float>& qkv,
                                    const std::vector<float>& qw,
                                    const std::vector<float>& kw,
                                    const std::vector<float>& ctab,
                                    const std::vector<float>& stab,
                                    const std::vector<int>& pos,
                                    int T,
                                    int hq,
                                    int hk,
                                    int hv,
                                    int D,
                                    float eps,
                                    bool interleaved,
                                    bool gemma) {
    const int HT = hq + hk + hv;
    std::vector<float> out(qkv.size());
    for (int t = 0; t < T; ++t) {
        for (int h = 0; h < HT; ++h) {
            const long base = (long(t) * HT + h) * D;
            if (h >= hq + hk) {
                for (int d = 0; d < D; ++d) out[base + d] = qkv[base + d];
                continue;
            }
            double ss = 0.0;
            for (int d = 0; d < D; ++d) ss += double(qkv[base + d]) * qkv[base + d];
            const float inv = float(1.0 / std::sqrt(ss / D + eps));
            const auto& w = h < hq ? qw : kw;
            const long cs = long(pos[t]) * (D / 2);
            if (!interleaved) {
                for (int d = 0; d < D / 2; ++d) {
                    float w0 = w[d];
                    float w1 = w[d + D / 2];
                    if (gemma) {
                        w0 += 1.0f;
                        w1 += 1.0f;
                    }
                    const float x0 = qkv[base + d] * inv * w0;
                    const float x1 = qkv[base + d + D / 2] * inv * w1;
                    const float c = ctab[cs + d];
                    const float s = stab[cs + d];
                    out[base + d] = x0 * c - x1 * s;
                    out[base + d + D / 2] = x1 * c + x0 * s;
                }
            } else {
                for (int d = 0; d < D; d += 2) {
                    float w0 = w[d];
                    float w1 = w[d + 1];
                    if (gemma) {
                        w0 += 1.0f;
                        w1 += 1.0f;
                    }
                    const float x0 = qkv[base + d] * inv * w0;
                    const float x1 = qkv[base + d + 1] * inv * w1;
                    const float c = ctab[cs + d / 2];
                    const float s = stab[cs + d / 2];
                    out[base + d] = x0 * c - x1 * s;
                    out[base + d + 1] = x0 * s + x1 * c;
                }
            }
        }
    }
    return out;
}

static int run_case(int D, bool interleaved, bool gemma, bool bench) {
    const int T = bench ? 4096 : 37;
    const int hq = 4, hk = 2, hv = 2;
    const int HT = hq + hk + hv;
    const int max_pos = 8192;
    const float eps = 1e-6f;
    std::mt19937 rng(17 + D + int(interleaved) * 11 + int(gemma) * 19);
    std::normal_distribution<float> nd(0.0f, 0.45f);
    std::uniform_int_distribution<int> pd(0, max_pos - 1);
    std::vector<float> qkv(size_t(T) * HT * D), qw(D), kw(D), ctab(size_t(max_pos) * D / 2),
        stab(size_t(max_pos) * D / 2);
    std::vector<int> pos(T);
    for (auto& x : qkv) x = nd(rng);
    for (int d = 0; d < D; ++d) {
        qw[d] = 0.75f + 0.02f * nd(rng);
        kw[d] = 0.80f + 0.02f * nd(rng);
    }
    for (int p = 0; p < max_pos; ++p) {
        for (int d = 0; d < D / 2; ++d) {
            const float a = 0.0007f * float(p) + 0.017f * float(d);
            ctab[size_t(p) * D / 2 + d] = std::cos(a);
            stab[size_t(p) * D / 2 + d] = std::sin(a);
        }
    }
    for (auto& x : pos) x = pd(rng);

    auto dqkv = dnew(bf16v(qkv));
    auto dqw = dnew(bf16v(qw));
    auto dkw = dnew(bf16v(kw));
    auto dc = dnew(bf16v(ctab));
    auto ds = dnew(bf16v(stab));
    auto dp = dnew(pos);
    __hip_bfloat16* dout = nullptr;
    CK(hipMalloc(&dout, qkv.size() * sizeof(__hip_bfloat16)));
    dim3 grid(HT, T);
    auto launch = [&] {
        qk_norm_rope_kernel<<<grid, 256>>>(dqkv, dqw, dkw, dc, ds, dp, dout, T, hq, hk, hv, D, eps,
                                           interleaved ? 1 : 0, gemma ? 1 : 0);
    };
    launch();
    CK(hipDeviceSynchronize());
    CK(hipGetLastError());
    auto got_bf = d2h(dout, qkv.size());
    auto ref = reference(qkv, qw, kw, ctab, stab, pos, T, hq, hk, hv, D, eps, interleaved, gemma);
    double sad = 0.0, sref = 0.0, maxe = 0.0, vcopy = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double g = hbf(got_bf[i]);
        const double d = std::abs(g - ref[i]);
        sad += d;
        sref += std::abs(ref[i]);
        maxe = std::max(maxe, d);
    }
    for (int t = 0; t < T; ++t) {
        for (int h = hq + hk; h < HT; ++h) {
            const long base = (long(t) * HT + h) * D;
            for (int d = 0; d < D; ++d)
                vcopy = std::max(vcopy, double(std::abs(hbf(got_bf[base + d]) - qkv[base + d])));
        }
    }
    const double rel = sad / std::max(sref, 1e-30);
    const bool ok = rel < 5e-3 && maxe < 7e-2 && vcopy < 8e-3;
    printf("qk_norm_rope D=%d interleaved=%d gemma=%d rel %.3e max %.3e vcopy %.3e %s\n",
           D, int(interleaved), int(gemma), rel, maxe, vcopy, ok ? "PASS" : "FAIL");
    if (bench && ok) {
        const float ms = bench_ms(launch);
        const double bytes = double(T) * HT * D * 2.0 * 2.0;
        printf("qk_norm_rope bench T=%d HT=%d D=%d: %.4f ms %.1f GB/s\n",
               T, HT, D, ms, bytes / (ms * 1e-3) / 1e9);
    }
    CK(hipFree(dqkv));
    CK(hipFree(dqw));
    CK(hipFree(dkw));
    CK(hipFree(dc));
    CK(hipFree(ds));
    CK(hipFree(dp));
    CK(hipFree(dout));
    return ok ? 0 : 1;
}

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    int rc = 0;
    rc |= run_case(64, false, false, bench);
    rc |= run_case(128, true, false, bench);
    rc |= run_case(256, false, true, bench);
    return rc;
}
