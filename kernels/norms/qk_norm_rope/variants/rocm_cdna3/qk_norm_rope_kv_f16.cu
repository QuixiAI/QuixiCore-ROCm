/**
 * @file
 * @brief CDNA3 port of Metal qk_norm_rope_kv_f16 split-store contract.
 */
#include "../../../../common/cdna3_harness.cuh"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

using bf16 = __hip_bfloat16;

__device__ __forceinline__ float bff(bf16 v) { return __bfloat162float(v); }

__device__ __forceinline__ float block_reduce_sum_256(float v) {
    __shared__ float scratch[256];
    const int tid = threadIdx.x;
    scratch[tid] = v;
    __syncthreads();
    for (int offset = 128; offset > 0; offset >>= 1) {
        if (tid < offset) scratch[tid] += scratch[tid + offset];
        __syncthreads();
    }
    return scratch[0];
}

template <int D>
__global__ void qk_norm_rope_kv_f16_wave_kernel(
    const bf16 *__restrict__ qkv, const bf16 *__restrict__ q_weight,
    const bf16 *__restrict__ k_weight, const bf16 *__restrict__ cosb,
    const bf16 *__restrict__ sinb, const int32_t *__restrict__ positions,
    bf16 *__restrict__ q_out, __half *__restrict__ k_out, __half *__restrict__ v_out,
    int T, int hq, int hk, int hv, float eps, int interleaved, int gemma) {
    const int head = blockIdx.x;
    const int token = blockIdx.y;
    const int tid = threadIdx.x;
    const int ht = hq + hk + hv;
    if (token >= T || head >= ht) return;

    const bool is_q = head < hq;
    const bool is_k = !is_q && head < hq + hk;
    const int kh = head - hq;
    const int vh = head - hq - hk;
    const size_t in_base = (static_cast<size_t>(token) * ht + head) * D;

    if (!is_q && !is_k) {
        const size_t out_base = (static_cast<size_t>(token) * hv + vh) * D;
        for (int d = tid; d < D; d += blockDim.x)
            v_out[out_base + d] = __float2half(bff(qkv[in_base + d]));
        return;
    }

    float ss = 0.0f;
    if (tid < D) {
        const float v = bff(qkv[in_base + tid]);
        ss = v * v;
    }
    ss = block_reduce_sum_256(ss);
    const float inv_rms = rsqrtf(ss / static_cast<float>(D) + eps);
    const bf16 *weight = is_q ? q_weight : k_weight;
    const int pos = positions[token];
    const size_t cs_base = static_cast<size_t>(pos) * (D / 2);
    const size_t out_base = is_q
        ? (static_cast<size_t>(token) * hq + head) * D
        : (static_cast<size_t>(token) * hk + kh) * D;

    if (interleaved == 0) {
        if (tid < D / 2) {
            float w0 = bff(weight[tid]);
            float w1 = bff(weight[tid + D / 2]);
            if (gemma != 0) {
                w0 += 1.0f;
                w1 += 1.0f;
            }
            const float x0 = bff(qkv[in_base + tid]) * inv_rms * w0;
            const float x1 = bff(qkv[in_base + tid + D / 2]) * inv_rms * w1;
            const float c = bff(cosb[cs_base + tid]);
            const float s = bff(sinb[cs_base + tid]);
            const float y0 = x0 * c - x1 * s;
            const float y1 = x1 * c + x0 * s;
            if (is_q) {
                q_out[out_base + tid] = __float2bfloat16(y0);
                q_out[out_base + tid + D / 2] = __float2bfloat16(y1);
            } else {
                k_out[out_base + tid] = __float2half(y0);
                k_out[out_base + tid + D / 2] = __float2half(y1);
            }
        }
    } else {
        for (int d = tid * 2; d < D; d += blockDim.x * 2) {
            float w0 = bff(weight[d]);
            float w1 = bff(weight[d + 1]);
            if (gemma != 0) {
                w0 += 1.0f;
                w1 += 1.0f;
            }
            const float x0 = bff(qkv[in_base + d]) * inv_rms * w0;
            const float x1 = bff(qkv[in_base + d + 1]) * inv_rms * w1;
            const float c = bff(cosb[cs_base + d / 2]);
            const float s = bff(sinb[cs_base + d / 2]);
            const float y0 = x0 * c - x1 * s;
            const float y1 = x0 * s + x1 * c;
            if (is_q) {
                q_out[out_base + d] = __float2bfloat16(y0);
                q_out[out_base + d + 1] = __float2bfloat16(y1);
            } else {
                k_out[out_base + d] = __float2half(y0);
                k_out[out_base + d + 1] = __float2half(y1);
            }
        }
    }
}

template <int D>
__global__ void qk_norm_rope_kv_f16_scalar_kernel(
    const bf16 *__restrict__ qkv, const bf16 *__restrict__ q_weight,
    const bf16 *__restrict__ k_weight, const bf16 *__restrict__ cosb,
    const bf16 *__restrict__ sinb, const int32_t *__restrict__ positions,
    bf16 *__restrict__ q_out, __half *__restrict__ k_out, __half *__restrict__ v_out,
    int T, int hq, int hk, int hv, float eps, int interleaved, int gemma) {
    const int head = blockIdx.x;
    const int token = blockIdx.y;
    if (threadIdx.x != 0) return;
    const int ht = hq + hk + hv;
    if (token >= T || head >= ht) return;

    const bool is_q = head < hq;
    const bool is_k = !is_q && head < hq + hk;
    const int kh = head - hq;
    const int vh = head - hq - hk;
    const size_t in_base = (static_cast<size_t>(token) * ht + head) * D;
    if (!is_q && !is_k) {
        const size_t out_base = (static_cast<size_t>(token) * hv + vh) * D;
        for (int d = 0; d < D; ++d) v_out[out_base + d] = __float2half(bff(qkv[in_base + d]));
        return;
    }

    double ss = 0.0;
    for (int d = 0; d < D; ++d) {
        const double v = bff(qkv[in_base + d]);
        ss += v * v;
    }
    const float inv_rms = static_cast<float>(
        1.0 / sqrt(ss / static_cast<double>(D) + static_cast<double>(eps)));
    const bf16 *weight = is_q ? q_weight : k_weight;
    const int pos = positions[token];
    const size_t cs_base = static_cast<size_t>(pos) * (D / 2);
    const size_t out_base = is_q
        ? (static_cast<size_t>(token) * hq + head) * D
        : (static_cast<size_t>(token) * hk + kh) * D;

    if (interleaved == 0) {
        for (int d = 0; d < D / 2; ++d) {
            float w0 = bff(weight[d]);
            float w1 = bff(weight[d + D / 2]);
            if (gemma != 0) {
                w0 += 1.0f;
                w1 += 1.0f;
            }
            const float x0 = bff(qkv[in_base + d]) * inv_rms * w0;
            const float x1 = bff(qkv[in_base + d + D / 2]) * inv_rms * w1;
            const float c = bff(cosb[cs_base + d]);
            const float s = bff(sinb[cs_base + d]);
            const float y0 = x0 * c - x1 * s;
            const float y1 = x1 * c + x0 * s;
            if (is_q) {
                q_out[out_base + d] = __float2bfloat16(y0);
                q_out[out_base + d + D / 2] = __float2bfloat16(y1);
            } else {
                k_out[out_base + d] = __float2half(y0);
                k_out[out_base + d + D / 2] = __float2half(y1);
            }
        }
    } else {
        for (int d = 0; d < D; d += 2) {
            float w0 = bff(weight[d]);
            float w1 = bff(weight[d + 1]);
            if (gemma != 0) {
                w0 += 1.0f;
                w1 += 1.0f;
            }
            const float x0 = bff(qkv[in_base + d]) * inv_rms * w0;
            const float x1 = bff(qkv[in_base + d + 1]) * inv_rms * w1;
            const float c = bff(cosb[cs_base + d / 2]);
            const float s = bff(sinb[cs_base + d / 2]);
            const float y0 = x0 * c - x1 * s;
            const float y1 = x0 * s + x1 * c;
            if (is_q) {
                q_out[out_base + d] = __float2bfloat16(y0);
                q_out[out_base + d + 1] = __float2bfloat16(y1);
            } else {
                k_out[out_base + d] = __float2half(y0);
                k_out[out_base + d + 1] = __float2half(y1);
            }
        }
    }
}

template <int D>
void reference(const std::vector<bf16> &qkv, const std::vector<bf16> &qw,
               const std::vector<bf16> &kw, const std::vector<bf16> &ctab,
               const std::vector<bf16> &stab, const std::vector<int32_t> &pos,
               int T, int hq, int hk, int hv, float eps, bool interleaved,
               bool gemma, std::vector<double> &q_ref, std::vector<double> &k_ref,
               std::vector<double> &v_ref) {
    const int ht = hq + hk + hv;
    q_ref.assign(static_cast<size_t>(T) * hq * D, 0.0);
    k_ref.assign(static_cast<size_t>(T) * hk * D, 0.0);
    v_ref.assign(static_cast<size_t>(T) * hv * D, 0.0);
    for (int t = 0; t < T; ++t) {
        for (int h = 0; h < ht; ++h) {
            const size_t in_base = (static_cast<size_t>(t) * ht + h) * D;
            if (h >= hq + hk) {
                const int vh = h - hq - hk;
                const size_t out_base = (static_cast<size_t>(t) * hv + vh) * D;
                for (int d = 0; d < D; ++d) v_ref[out_base + d] = qc::to_double(qkv[in_base + d]);
                continue;
            }
            double ss = 0.0;
            for (int d = 0; d < D; ++d) {
                const double v = qc::to_double(qkv[in_base + d]);
                ss += v * v;
            }
            const double inv = 1.0 / std::sqrt(ss / static_cast<double>(D) + eps);
            const bool is_q = h < hq;
            const auto &w = is_q ? qw : kw;
            const size_t out_base = is_q
                ? (static_cast<size_t>(t) * hq + h) * D
                : (static_cast<size_t>(t) * hk + (h - hq)) * D;
            auto &out = is_q ? q_ref : k_ref;
            const size_t cs_base = static_cast<size_t>(pos[t]) * (D / 2);
            if (!interleaved) {
                for (int d = 0; d < D / 2; ++d) {
                    double w0 = qc::to_double(w[d]);
                    double w1 = qc::to_double(w[d + D / 2]);
                    if (gemma) {
                        w0 += 1.0;
                        w1 += 1.0;
                    }
                    const double x0 = qc::to_double(qkv[in_base + d]) * inv * w0;
                    const double x1 = qc::to_double(qkv[in_base + d + D / 2]) * inv * w1;
                    const double c = qc::to_double(ctab[cs_base + d]);
                    const double s = qc::to_double(stab[cs_base + d]);
                    out[out_base + d] = x0 * c - x1 * s;
                    out[out_base + d + D / 2] = x1 * c + x0 * s;
                }
            } else {
                for (int d = 0; d < D; d += 2) {
                    double w0 = qc::to_double(w[d]);
                    double w1 = qc::to_double(w[d + 1]);
                    if (gemma) {
                        w0 += 1.0;
                        w1 += 1.0;
                    }
                    const double x0 = qc::to_double(qkv[in_base + d]) * inv * w0;
                    const double x1 = qc::to_double(qkv[in_base + d + 1]) * inv * w1;
                    const double c = qc::to_double(ctab[cs_base + d / 2]);
                    const double s = qc::to_double(stab[cs_base + d / 2]);
                    out[out_base + d] = x0 * c - x1 * s;
                    out[out_base + d + 1] = x0 * s + x1 * c;
                }
            }
        }
    }
}

template <int D>
bool run_case(int T, int hq, int hk, int hv, bool interleaved, bool gemma, bool bench) {
    const int ht = hq + hk + hv;
    const int max_pos = bench ? 8192 : 4096;
    qc::Rng rng(15000 + D + int(interleaved) * 31 + int(gemma) * 67 + T);
    std::vector<float> qkvf(static_cast<size_t>(T) * ht * D);
    std::vector<float> qwf(D), kwf(D);
    std::vector<float> cf(static_cast<size_t>(max_pos) * (D / 2));
    std::vector<float> sf(static_cast<size_t>(max_pos) * (D / 2));
    for (float &v : qkvf) v = rng.normal(0.0f, 0.45f);
    for (int d = 0; d < D; ++d) {
        qwf[d] = 1.0f + 0.05f * rng.normal();
        kwf[d] = 1.0f + 0.05f * rng.normal();
    }
    for (int p = 0; p < max_pos; ++p) {
        for (int d = 0; d < D / 2; ++d) {
            const float inv = 1.0f / std::pow(10000.0f, static_cast<float>(d) / (D / 2));
            const float a = static_cast<float>(p) * inv;
            cf[static_cast<size_t>(p) * (D / 2) + d] = std::cos(a);
            sf[static_cast<size_t>(p) * (D / 2) + d] = std::sin(a);
        }
    }
    std::vector<int32_t> pos(T);
    for (int &p : pos) p = rng.integer(0, max_pos - 1);

    auto qkv = qc::to_storage<bf16>(qkvf);
    auto qw = qc::to_storage<bf16>(qwf);
    auto kw = qc::to_storage<bf16>(kwf);
    auto ctab = qc::to_storage<bf16>(cf);
    auto stab = qc::to_storage<bf16>(sf);
    bf16 *dqkv = qc::dnew(qkv);
    bf16 *dqw = qc::dnew(qw);
    bf16 *dkw = qc::dnew(kw);
    bf16 *dc = qc::dnew(ctab);
    bf16 *ds = qc::dnew(stab);
    int32_t *dp = qc::dnew(pos);
    bf16 *dq = qc::dzero<bf16>(static_cast<size_t>(T) * hq * D);
    __half *dk = qc::dzero<__half>(static_cast<size_t>(T) * hk * D);
    __half *dv = qc::dzero<__half>(static_cast<size_t>(T) * hv * D);
    const float eps = 1e-6f;
    dim3 grid(ht, T);
    auto launch = [&] {
        qk_norm_rope_kv_f16_wave_kernel<D>
            <<<grid, 256>>>(dqkv, dqw, dkw, dc, ds, dp, dq, dk, dv, T, hq, hk, hv, eps,
                            interleaved ? 1 : 0, gemma ? 1 : 0);
    };
    launch();
    QC_SYNC();

    std::vector<double> qr, kr, vr;
    reference<D>(qkv, qw, kw, ctab, stab, pos, T, hq, hk, hv, eps, interleaved,
                 gemma, qr, kr, vr);
    char label[128];
    std::snprintf(label, sizeof(label), "qk_norm_rope_kv_f16 Q T=%d D=%d i=%d g=%d",
                  T, D, int(interleaved), int(gemma));
    bool ok = qc::compare(qc::d2h(dq, static_cast<size_t>(T) * hq * D), qr,
                          qc::Tol::bf16_output())
                  .report(label);
    std::snprintf(label, sizeof(label), "qk_norm_rope_kv_f16 K T=%d D=%d i=%d g=%d",
                  T, D, int(interleaved), int(gemma));
    ok &= qc::compare(qc::d2h(dk, static_cast<size_t>(T) * hk * D), kr,
                      qc::Tol::fp16_output())
              .report(label);
    std::snprintf(label, sizeof(label), "qk_norm_rope_kv_f16 V T=%d D=%d i=%d g=%d",
                  T, D, int(interleaved), int(gemma));
    ok &= qc::compare(qc::d2h(dv, static_cast<size_t>(T) * hv * D), vr,
                      qc::Tol::fp16_output())
              .report(label);

    if (bench && ok) {
        bf16 *dq_base = qc::dzero<bf16>(static_cast<size_t>(T) * hq * D);
        __half *dk_base = qc::dzero<__half>(static_cast<size_t>(T) * hk * D);
        __half *dv_base = qc::dzero<__half>(static_cast<size_t>(T) * hv * D);
        auto scalar = [&] {
            qk_norm_rope_kv_f16_scalar_kernel<D>
                <<<grid, 1>>>(dqkv, dqw, dkw, dc, ds, dp, dq_base, dk_base, dv_base,
                              T, hq, hk, hv, eps, interleaved ? 1 : 0, gemma ? 1 : 0);
        };
        const int scalar_repeat = 10;
        const int wave_repeat = 100;
        auto b0 = qc::bench([&] {
            for (int i = 0; i < scalar_repeat; ++i) scalar();
        }, 3, 10);
        auto b1 = qc::bench([&] {
            for (int i = 0; i < wave_repeat; ++i) launch();
        }, 10, 50);
        b0.median_ms /= scalar_repeat;
        b0.min_ms /= scalar_repeat;
        b0.max_ms /= scalar_repeat;
        b0.mean_ms /= scalar_repeat;
        b1.median_ms /= wave_repeat;
        b1.min_ms /= wave_repeat;
        b1.max_ms /= wave_repeat;
        b1.mean_ms /= wave_repeat;
        std::printf("  qk_norm_rope_kv_f16 timing batches: scalar %d launches/sample, wave-block %d launches/sample (per-launch shown)\n",
                    scalar_repeat, wave_repeat);
        const double bytes = static_cast<double>(T) * ht * D * sizeof(bf16) +
                             static_cast<double>(D * 2) * sizeof(bf16) +
                             static_cast<double>(T) * (hq * sizeof(bf16) +
                                                       (hk + hv) * sizeof(__half)) * D;
        b0.report_bandwidth("qk_norm_rope_kv_f16 scalar", bytes);
        b1.report_bandwidth("qk_norm_rope_kv_f16 wave-block", bytes);
        std::snprintf(label, sizeof(label), "qk_norm_rope_kv_f16 T=%d HT=%d D=%d",
                      T, ht, D);
        qc::report_ab(label, b0, b1);
        qc::dfree(dq_base, dk_base, dv_base);
    }

    qc::dfree(dqkv, dqw, dkw, dc, ds, dp, dq, dk, dv);
    return ok;
}

void run_benchmarks() {
    (void)run_case<128>(4096, 4, 2, 2, true, false, true);
    (void)run_case<256>(4096, 4, 2, 2, false, true, true);
}

}  // namespace

int main(int argc, char **argv) {
    qc::print_environment("qk_norm_rope_kv_f16");
    bool ok = true;
    for (bool interleaved : {false, true}) {
        for (bool gemma : {false, true}) {
            ok &= run_case<64>(33, 3, 1, 1, interleaved, gemma, false);
            ok &= run_case<128>(33, 3, 1, 1, interleaved, gemma, false);
            ok &= run_case<256>(33, 3, 1, 1, interleaved, gemma, false);
        }
    }
    if (qc::bench_requested(argc, argv)) run_benchmarks();
    return qc::finish(ok);
}
