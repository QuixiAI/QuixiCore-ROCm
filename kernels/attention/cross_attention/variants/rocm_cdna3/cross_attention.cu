/**
 * @file
 * @brief CDNA3 (gfx942) cross-attention: independent query/key lengths, a
 *        per-batch valid-key count, optional additive score bias, optional
 *        score softcap, and automatic-or-explicit scale.
 *
 * Semantic source is the CPU reference `quixicore_cpu::cross_attention`
 * (`../QuixiCore-CPU/kernels/attention/cross_attention_ref.cpp`). Contract,
 * matched exactly:
 *
 *   layout   Q [B, Hq, Lq, D]   K,V [B, Hkv, Lk, D]   O [B, Hq, Lq, D]
 *   GQA      kv_head = query_head / (Hq / Hkv),  Hq % Hkv == 0
 *   lengths  valid = clamp(key_lengths[b], 0, Lk)  -- per batch item, not per head
 *   scale    scale > 0 ? scale : 1/sqrt(D)
 *   bias     optional, indexed bias[((b*Hq + h)*Lq + i) * Lk + j]
 *   softcap  optional, score = softcap * tanh(score / softcap) when softcap > 0
 *   D        64, 128, or 256 only
 *   empty    valid == 0 emits an all-zero row (denominator stays 0; no NaN)
 *
 * Ordering matters and is contractual: bias is added to the SCALED score, and
 * the softcap is applied AFTER the bias. Reordering changes results whenever
 * both are present.
 *
 * This is cross-attention proper -- Lq and Lk are independent and neither is
 * required to be a multiple of the 16-wide MFMA tile, so partial query and key
 * tiles are masked rather than assumed away. That is the main structural
 * difference from the landed self-attention kernel
 * (`kernels/attention/gqa/variants/rocm_cdna3/attn_mfma.cuh`), which assumes
 * N % 16 == 0 and a single sequence length.
 *
 * Candidate: MFMA-tiled flash attention, BQ=BK=16, one 64-lane wavefront per
 * (query-block, head, batch), reusing each K/V tile across 16 queries and using
 * v_mfma_f32_16x16x16{bf16_1k,f16} for QK^T and P@V, with the softmax reducing
 * over an LDS transpose of S. Baseline for the A/B: one wavefront per query row
 * with the head dim split across lanes and a wavefront dot-product per key --
 * correct, but it re-reads K/V once per query instead of once per 16 queries.
 *
 *   hipcc -std=c++17 -O3 --offload-arch=gfx942 cross_attention.cu -o cross_attention.out
 */
#include "../../../../common/cdna3_harness.cuh"
#include "../../../../common/cdna3_mfma.cuh"

using qc::Bf16Traits;
using qc::float4_t;
using qc::Fp16Traits;

/// Scaled score -> bias -> softcap, in the reference's order.
__device__ __forceinline__ float apply_score_epilogue(float score, float bias_value,
                                                      bool has_bias, float softcap) {
    if (has_bias) score += bias_value;
    if (softcap > 0.0f) score = softcap * tanhf(score / softcap);
    return score;
}

// ---------------------------------------------------------------------------
// Candidate: MFMA-tiled, BQ=BK=16, one wavefront per (query block, head, batch)
// ---------------------------------------------------------------------------
template <typename T, int D>
__global__ __launch_bounds__(qc::kWave) void cross_attention_mfma(
    const typename T::storage *__restrict__ Q, const typename T::storage *__restrict__ K,
    const typename T::storage *__restrict__ V, const int *__restrict__ key_lengths,
    const float *__restrict__ bias, typename T::storage *__restrict__ O, int Hq,
    int Hkv, int Lq, int Lk, float scale, float softcap) {
    constexpr int BQ = 16, BK = 16, DT = D / 16;
    using storage = typename T::storage;
    using frag = typename T::frag;

    const int qb = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
    const int lane = threadIdx.x;
    const int q0 = qb * BQ;
    const int lo = lane >> 4, li = lane & 15;  // (lane/16, lane%16)
    const int kv_head = h / (Hq / Hkv);
    const int valid = min(max(key_lengths[b], 0), Lk);
    const bool has_bias = bias != nullptr;

    // Q [b, h, i, d] and K/V [b, kv_head, j, d]
    auto q_index = [&](int i, int d) {
        return (size_t)(((size_t)b * Hq + h) * Lq + (q0 + i)) * D + d;
    };
    auto kv_index = [&](int j, int d) {
        return (size_t)(((size_t)b * Hkv + kv_head) * Lk + j) * D + d;
    };
    auto bias_index = [&](int i, int j) {
        return (size_t)(((size_t)b * Hq + h) * Lq + (q0 + i)) * Lk + j;
    };

    // Preload Q fragments: a[v] = Q[m=li][k = s*16 + lo*4 + v].
    // Rows past Lq load zero so a partial query tile cannot read out of bounds.
    frag Qf[DT];
    const bool q_row_valid = (q0 + li) < Lq;
#pragma unroll
    for (int s = 0; s < DT; ++s) {
        const int d0 = s * 16 + lo * 4;
        frag f;
#pragma unroll
        for (int v = 0; v < 4; ++v)
            T::put(f, v, q_row_valid ? Q[q_index(li, d0 + v)] : T::from_float(0.0f));
        Qf[s] = f;
    }

    float4_t Oacc[DT];
#pragma unroll
    for (int s = 0; s < DT; ++s) Oacc[s] = float4_t{0.f, 0.f, 0.f, 0.f};

    __shared__ float sS[BQ][BK];
    __shared__ storage sP[BQ][BK];
    __shared__ float sm[BQ], sl[BQ], sc[BQ];
    if (lane < BQ) {
        sm[lane] = -INFINITY;
        sl[lane] = 0.0f;
    }
    __syncthreads();

    for (int k0 = 0; k0 < valid; k0 += BK) {
        // S[16x16] = Q @ K^T accumulated over DT head-dim steps.
        float4_t S = float4_t{0.f, 0.f, 0.f, 0.f};
        const bool k_row_valid = (k0 + li) < valid;
#pragma unroll
        for (int s = 0; s < DT; ++s) {
            const int d0 = s * 16 + lo * 4;
            frag Kf;
#pragma unroll
            for (int v = 0; v < 4; ++v)
                T::put(Kf, v,
                       k_row_valid ? K[kv_index(k0 + li, d0 + v)] : T::from_float(0.0f));
            S = T::mma(Qf[s], Kf, S);
        }
        // Lane holds S[qi = 4*lo + v][kj = li]; stage the scaled score in LDS so
        // the softmax can reduce along a row without a distributed-layout shuffle.
#pragma unroll
        for (int v = 0; v < 4; ++v) sS[4 * lo + v][li] = S[v] * scale;
        __syncthreads();

        // Softmax: lanes 0..15 each own one query row (online max / denominator).
        if (lane < BQ) {
            const int qi = lane;
            const float previous_max = sm[qi];
            float row_max = previous_max;
            for (int j = 0; j < BK; ++j) {
                const int kk = k0 + j;
                if (kk >= valid) continue;
                const float score = apply_score_epilogue(
                    sS[qi][j], has_bias ? bias[bias_index(qi, kk)] : 0.0f, has_bias,
                    softcap);
                sS[qi][j] = score;  // reuse the tile so the exp pass sees the same value
                row_max = fmaxf(row_max, score);
            }
            // First tile of an all-masked row leaves row_max at -inf; guard the
            // correction so exp(-inf - -inf) never produces a NaN.
            const float correction =
                (previous_max == -INFINITY) ? 0.0f : __expf(previous_max - row_max);
            float denominator = sl[qi] * correction;
            for (int j = 0; j < BK; ++j) {
                const int kk = k0 + j;
                const float p =
                    (kk < valid) ? __expf(sS[qi][j] - row_max) : 0.0f;
                sP[qi][j] = T::from_float(p);
                denominator += p;
            }
            sm[qi] = row_max;
            sl[qi] = denominator;
            sc[qi] = correction;
        }
        __syncthreads();

        // Rescale the output accumulator by each row's correction factor.
#pragma unroll
        for (int s = 0; s < DT; ++s) {
            float4_t o = Oacc[s];
#pragma unroll
            for (int v = 0; v < 4; ++v) o[v] *= sc[4 * lo + v];
            Oacc[s] = o;
        }

        // O += P @ V : a[v] = P[m=li][k = lo*4 + v], b[v] = V[k = lo*4 + v][n = s*16 + li]
        frag Pf;
#pragma unroll
        for (int v = 0; v < 4; ++v) T::put(Pf, v, sP[li][lo * 4 + v]);
#pragma unroll
        for (int s = 0; s < DT; ++s) {
            const int d = s * 16 + li;
            frag Vf;
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                const int kj = k0 + lo * 4 + v;
                T::put(Vf, v, kj < valid ? V[kv_index(kj, d)] : T::from_float(0.0f));
            }
            Oacc[s] = T::mma(Pf, Vf, Oacc[s]);
        }
        __syncthreads();
    }

    // Epilogue: O[qi][d] /= denominator. Oacc[s][v] holds O[qi = 4*lo + v][d = s*16 + li].
    // A row with no valid keys has denominator 0 and emits zeros, per the reference.
#pragma unroll
    for (int s = 0; s < DT; ++s) {
        const int d = s * 16 + li;
#pragma unroll
        for (int v = 0; v < 4; ++v) {
            const int qi = 4 * lo + v;
            if (q0 + qi >= Lq) continue;
            const float denominator = sl[qi];
            const float inverse = denominator > 0.0f ? 1.0f / denominator : 0.0f;
            O[q_index(qi, d)] = T::from_float(Oacc[s][v] * inverse);
        }
    }
}

// ---------------------------------------------------------------------------
// Baseline: one wavefront per query row, head dim split across lanes.
// Same math, no tiling -- K/V are re-read once per query instead of per 16.
// ---------------------------------------------------------------------------
template <typename T, int D>
__global__ __launch_bounds__(qc::kWave) void cross_attention_naive(
    const typename T::storage *__restrict__ Q, const typename T::storage *__restrict__ K,
    const typename T::storage *__restrict__ V, const int *__restrict__ key_lengths,
    const float *__restrict__ bias, typename T::storage *__restrict__ O, int Hq,
    int Hkv, int Lq, int Lk, float scale, float softcap) {
    constexpr int ITEMS = D / qc::kWave;
    const int i = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
    if (i >= Lq) return;
    const int lane = threadIdx.x;
    const int kv_head = h / (Hq / Hkv);
    const int valid = min(max(key_lengths[b], 0), Lk);
    const bool has_bias = bias != nullptr;

    const size_t q_base = (size_t)(((size_t)b * Hq + h) * Lq + i) * D;
    const size_t kv_base = (size_t)((size_t)b * Hkv + kv_head) * Lk * D;
    const size_t bias_base = (size_t)(((size_t)b * Hq + h) * Lq + i) * Lk;

    float q_val[ITEMS], acc[ITEMS];
#pragma unroll
    for (int t = 0; t < ITEMS; ++t) {
        q_val[t] = T::to_float(Q[q_base + lane + t * qc::kWave]);
        acc[t] = 0.0f;
    }

    float running_max = -INFINITY, denominator = 0.0f;
    for (int j = 0; j < valid; ++j) {
        const size_t k_row = kv_base + (size_t)j * D;
        float dot = 0.0f;
#pragma unroll
        for (int t = 0; t < ITEMS; ++t)
            dot = fmaf(q_val[t], T::to_float(K[k_row + lane + t * qc::kWave]), dot);
        dot = qc::wave_reduce_sum(dot) * scale;
        const float score = apply_score_epilogue(
            dot, has_bias ? bias[bias_base + j] : 0.0f, has_bias, softcap);

        const float next_max = fmaxf(running_max, score);
        const float old_weight =
            (running_max == -INFINITY) ? 0.0f : __expf(running_max - next_max);
        const float new_weight = __expf(score - next_max);
        denominator = denominator * old_weight + new_weight;
#pragma unroll
        for (int t = 0; t < ITEMS; ++t)
            acc[t] = acc[t] * old_weight +
                     T::to_float(V[k_row + lane + t * qc::kWave]) * new_weight;
        running_max = next_max;
    }

    const float inverse = denominator > 0.0f ? 1.0f / denominator : 0.0f;
#pragma unroll
    for (int t = 0; t < ITEMS; ++t)
        O[q_base + lane + t * qc::kWave] = T::from_float(acc[t] * inverse);
}

// ===========================================================================
// fp64 oracle -- mirrors cross_attention_ref.cpp
// ===========================================================================
// The CPU reference accumulates its output in float and rounds every step; this
// oracle accumulates in double. That is deliberate: the oracle should express
// the operation's mathematics, not the CPU backend's accumulator width. Kernel
// results are compared to it at the storage dtype's tolerance.
static std::vector<double> oracle(const std::vector<float> &Q, const std::vector<float> &K,
                                  const std::vector<float> &V,
                                  const std::vector<int32_t> &key_lengths,
                                  const std::vector<float> &bias, bool has_bias, int B,
                                  int Hq, int Hkv, int Lq, int Lk, int D, float scale,
                                  float softcap) {
    std::vector<double> out((size_t)B * Hq * Lq * D, 0.0);
    const double score_scale = scale > 0.0f ? (double)scale : 1.0 / std::sqrt((double)D);
    const int head_group = Hq / Hkv;

    for (int b = 0; b < B; ++b) {
        const int valid = std::min(std::max(key_lengths[b], 0), Lk);
        for (int h = 0; h < Hq; ++h) {
            const int kv_head = h / head_group;
            for (int i = 0; i < Lq; ++i) {
                const size_t q_base = (size_t)(((size_t)b * Hq + h) * Lq + i) * D;
                std::vector<double> acc(D, 0.0);
                double maximum = -std::numeric_limits<double>::infinity();
                double denominator = 0.0;
                for (int j = 0; j < valid; ++j) {
                    const size_t kv_base =
                        (size_t)(((size_t)b * Hkv + kv_head) * Lk + j) * D;
                    double dot = 0.0;
                    for (int d = 0; d < D; ++d)
                        dot += (double)Q[q_base + d] * (double)K[kv_base + d];
                    double score = dot * score_scale;
                    if (has_bias)
                        score += (double)bias[(size_t)(((size_t)b * Hq + h) * Lq + i) * Lk + j];
                    if (softcap > 0.0f)
                        score = (double)softcap * std::tanh(score / (double)softcap);

                    const double next_max = std::max(maximum, score);
                    const double old_weight =
                        std::isinf(maximum) ? 0.0 : std::exp(maximum - next_max);
                    const double new_weight = std::exp(score - next_max);
                    denominator = denominator * old_weight + new_weight;
                    for (int d = 0; d < D; ++d)
                        acc[d] = acc[d] * old_weight + (double)V[kv_base + d] * new_weight;
                    maximum = next_max;
                }
                const double inverse = denominator > 0.0 ? 1.0 / denominator : 0.0;
                for (int d = 0; d < D; ++d) out[q_base + d] = acc[d] * inverse;
            }
        }
    }
    return out;
}

// ===========================================================================
// Harness
// ===========================================================================

struct Case {
    const char *name;
    int B, Hq, Hkv, Lq, Lk, D;
    bool bias;
    float scale;    // 0 => automatic 1/sqrt(D)
    float softcap;  // 0 => disabled
    int length_mode; // 0 full, 1 ragged per batch, 2 first batch empty
};

template <typename T, int D>
static bool run_case(const Case &c, bool also_bench) {
    using storage = typename T::storage;
    qc::Rng rng(0x0C0FFEE + c.Lq * 131 + c.Lk * 17 + D);

    const size_t q_count = (size_t)c.B * c.Hq * c.Lq * D;
    const size_t kv_count = (size_t)c.B * c.Hkv * c.Lk * D;
    const size_t bias_count = (size_t)c.B * c.Hq * c.Lq * c.Lk;

    auto Qf = rng.normals(q_count, 0.6f);
    auto Kf = rng.normals(kv_count, 0.6f);
    auto Vf = rng.normals(kv_count, 0.6f);
    std::vector<float> Bf = c.bias ? rng.normals(bias_count, 0.3f) : std::vector<float>();

    std::vector<int32_t> lengths(c.B);
    for (int b = 0; b < c.B; ++b) {
        switch (c.length_mode) {
            case 1: lengths[b] = 1 + (b * 37 + 11) % c.Lk; break;
            case 2: lengths[b] = (b == 0) ? 0 : c.Lk; break;
            default: lengths[b] = c.Lk; break;
        }
    }
    // Also exercise the clamp: an over-long request must saturate, not read OOB.
    if (c.length_mode == 1 && c.B > 1) lengths[c.B - 1] = c.Lk + 5;

    // Round host inputs through storage first so the oracle sees exactly the
    // values the kernel sees; otherwise bf16 rounding shows up as "kernel error".
    auto Qs = qc::to_storage<storage>(Qf);
    auto Ks = qc::to_storage<storage>(Kf);
    auto Vs = qc::to_storage<storage>(Vf);
    std::vector<float> Qr(q_count), Kr(kv_count), Vr(kv_count);
    for (size_t i = 0; i < q_count; ++i) Qr[i] = (float)qc::to_double(Qs[i]);
    for (size_t i = 0; i < kv_count; ++i) {
        Kr[i] = (float)qc::to_double(Ks[i]);
        Vr[i] = (float)qc::to_double(Vs[i]);
    }

    const float scale = c.scale > 0.0f ? c.scale : 1.0f / std::sqrt((float)D);
    auto ref = oracle(Qr, Kr, Vr, lengths, Bf, c.bias, c.B, c.Hq, c.Hkv, c.Lq, c.Lk, D,
                      c.scale, c.softcap);

    auto dQ = qc::dnew(Qs);
    auto dK = qc::dnew(Ks);
    auto dV = qc::dnew(Vs);
    auto dLen = qc::dnew(lengths);
    float *dBias = c.bias ? qc::dnew(Bf) : nullptr;
    auto dO = qc::dzero<storage>(q_count);

    const dim3 grid_mfma(qc::grid_for(c.Lq, 16), c.Hq, c.B);
    const dim3 grid_naive(c.Lq, c.Hq, c.B);

    auto launch_mfma = [&] {
        cross_attention_mfma<T, D><<<grid_mfma, qc::kWave>>>(
            dQ, dK, dV, dLen, dBias, dO, c.Hq, c.Hkv, c.Lq, c.Lk, scale, c.softcap);
    };
    auto launch_naive = [&] {
        cross_attention_naive<T, D><<<grid_naive, qc::kWave>>>(
            dQ, dK, dV, dLen, dBias, dO, c.Hq, c.Hkv, c.Lq, c.Lk, scale, c.softcap);
    };

    char label[256];
    bool ok = true;

    launch_mfma();
    QC_SYNC();
    std::snprintf(label, sizeof(label), "%s [%s D=%d] mfma", c.name, T::name, D);
    ok &= qc::compare(qc::d2h(dO, q_count), ref,
                      std::is_same<T, Bf16Traits>::value ? qc::Tol::bf16_output() : qc::Tol::fp16_output())
              .report(label);

    QC_CHECK(hipMemset(dO, 0, q_count * sizeof(storage)));
    launch_naive();
    QC_SYNC();
    std::snprintf(label, sizeof(label), "%s [%s D=%d] naive", c.name, T::name, D);
    ok &= qc::compare(qc::d2h(dO, q_count), ref,
                      std::is_same<T, Bf16Traits>::value ? qc::Tol::bf16_output() : qc::Tol::fp16_output())
              .report(label);

    if (also_bench && ok) {
        const auto b_naive = qc::bench(launch_naive);
        const auto b_mfma = qc::bench(launch_mfma);
        // 2 GEMMs of Lq x Lk x D per (batch, head), 2 flops each.
        const double flops =
            4.0 * (double)c.B * c.Hq * c.Lq * c.Lk * D;
        std::snprintf(label, sizeof(label), "%s [%s D=%d] naive", c.name, T::name, D);
        b_naive.report_compute(label, flops);
        std::snprintf(label, sizeof(label), "%s [%s D=%d] mfma", c.name, T::name, D);
        b_mfma.report_compute(label, flops);
        std::snprintf(label, sizeof(label), "%s [%s D=%d]", c.name, T::name, D);
        qc::report_ab(label, b_naive, b_mfma);
    }

    qc::dfree(dQ, dK, dV, dO);
    qc::dfree(dLen);
    if (dBias) qc::dfree(dBias);
    return ok;
}

int main(int argc, char **argv) {
    const bool do_bench = qc::bench_requested(argc, argv);
    qc::print_environment("cross_attention (CDNA3 gfx942)");
    bool ok = true;

    // Correctness: contract surface first -- ragged lengths, empty rows, bias,
    // softcap, GQA ratios, and all three legal head dims.
    std::printf("correctness (fp64 oracle = quixicore_cpu::cross_attention):\n");
    const Case cases[] = {
        {"MHA full",          2, 8,  8, 64,  128, 0, false, 0.f,   0.f,  0},
        {"GQA 4:1 full",      2, 8,  2, 64,  128, 0, false, 0.f,   0.f,  0},
        {"GQA ragged len",    3, 8,  2, 37,  101, 0, false, 0.f,   0.f,  1},
        {"empty first batch", 2, 4,  1, 16,  64,  0, false, 0.f,   0.f,  2},
        {"bias",              2, 4,  2, 33,  67,  0, true,  0.f,   0.f,  1},
        {"softcap 30",        2, 4,  2, 33,  67,  0, false, 0.f,   30.f, 1},
        {"bias + softcap 20", 2, 4,  2, 33,  67,  0, true,  0.f,   20.f, 1},
        {"explicit scale",    1, 4,  4, 24,  48,  0, false, 0.125f, 0.f, 0},
        {"decode Lq=1",       4, 8,  2, 1,   512, 0, false, 0.f,   0.f,  1},
        {"long keys",         1, 4,  1, 8,   2048,0, false, 0.f,   0.f,  0},
    };
    for (const auto &c : cases) {
        ok &= run_case<Bf16Traits, 64>(c, false);
        ok &= run_case<Bf16Traits, 128>(c, false);
    }
    // D=256 and fp16 on a representative subset (register pressure / dtype path).
    ok &= run_case<Bf16Traits, 256>(cases[1], false);
    ok &= run_case<Bf16Traits, 256>(cases[2], false);
    ok &= run_case<Fp16Traits, 64>(cases[2], false);
    ok &= run_case<Fp16Traits, 128>(cases[4], false);
    ok &= run_case<Fp16Traits, 128>(cases[5], false);

    if (do_bench && ok) {
        std::printf("\nA/B naive wavefront-per-query vs MFMA BQ=16 tile "
                    "(HIP-event median, warmup 10 / iters 50):\n");
        const Case perf_cases[] = {
            {"prefill 512x512",  4, 32, 8, 512,  512,  0, false, 0.f, 0.f, 0},
            {"prefill 512x2048", 2, 32, 8, 512,  2048, 0, false, 0.f, 0.f, 0},
            {"encdec 128x1024",  8, 16, 4, 128,  1024, 0, false, 0.f, 0.f, 0},
            {"ragged 517x1031",  4, 16, 4, 517,  1031, 0, false, 0.f, 0.f, 1},
            {"bias 256x1024",    4, 16, 4, 256,  1024, 0, true,  0.f, 0.f, 0},
        };
        for (const auto &c : perf_cases) ok &= run_case<Bf16Traits, 128>(c, true);
        ok &= run_case<Bf16Traits, 64>(perf_cases[0], true);
    }

    return qc::finish(ok);
}
