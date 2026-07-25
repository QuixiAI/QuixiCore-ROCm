/**
 * @file
 * @brief CDNA3 (gfx942) biased attention: additive per-head score bias plus an
 *        optional boolean key mask shared across heads.
 *
 * Semantic source is the CPU reference `quixicore_cpu::biased_attention`
 * (`../QuixiCore-CPU/kernels/attention/attention_extended_ref.cpp`, ~L357).
 * Contract, matched exactly:
 *
 *   layout   Q [Hq, Lq, D]   K,V [Hkv, Lk, D]   O [Hq, Lq, D]   -- no batch dim
 *   GQA      kv_head = query_head / (Hq / Hkv),  requires Hq % Hkv == 0
 *   bias     optional, PER HEAD: bias[(h*Lq + i)*Lk + j]
 *   mask     optional, SHARED across heads: mask[i*Lk + j], 0 == skip the key
 *   scale    scale == 0 ? 1/sqrt(D) : scale
 *   masked   a fully masked query row emits zeros
 *
 * Two details separate this from `cross_attention`, and both are easy to get
 * wrong by analogy:
 *
 *   - The bias is indexed per head, but the mask is NOT. The mask is a single
 *     [Lq, Lk] plane reused by every head.
 *   - The automatic-scale trigger is `scale == 0`, not `scale > 0`. A negative
 *     explicit scale is legal here and must be honoured, not replaced by
 *     1/sqrt(D). (`cross_attention` really does use `scale > 0`; the two
 *     references differ, and this kernel follows its own.)
 *
 * A masked key is skipped entirely rather than driven to -inf, so it does not
 * participate in the running maximum. With a tiled softmax that distinction
 * matters: a tile in which every key is masked leaves the running max at -inf,
 * and `exp(-inf - -inf)` is NaN. The correction factor is special-cased there.
 *
 * Candidate: MFMA-tiled flash attention, BQ=BK=16, one 64-lane wavefront per
 * (query block, head), K/V tile reused across 16 queries, QK^T and P@V on
 * v_mfma_f32_16x16x16{bf16_1k,f16}. Baseline for the A/B: one wavefront per
 * query row with a wavefront dot-product per key.
 *
 *   hipcc -std=c++17 -O3 --offload-arch=gfx942 biased_attention.cu -o biased_attention.out
 */
#include "../../../../common/cdna3_harness.cuh"
#include "../../../../common/cdna3_mfma.cuh"

using qc::Bf16Traits;
using qc::float4_t;
using qc::Fp16Traits;

// ---------------------------------------------------------------------------
// Candidate: MFMA-tiled, BQ=BK=16, one wavefront per (query block, head)
// ---------------------------------------------------------------------------
template <typename T, int D>
__global__ __launch_bounds__(qc::kWave) void biased_attention_mfma(
    const typename T::storage *__restrict__ Q, const typename T::storage *__restrict__ K,
    const typename T::storage *__restrict__ V, const float *__restrict__ bias,
    const uint8_t *__restrict__ mask, typename T::storage *__restrict__ O, int Hq,
    int Hkv, int Lq, int Lk, float scale) {
    constexpr int BQ = 16, BK = 16, DT = D / 16;
    using storage = typename T::storage;
    using frag = typename T::frag;

    const int qb = blockIdx.x, h = blockIdx.y;
    const int lane = threadIdx.x;
    const int q0 = qb * BQ;
    const int lo = qc::mfma_lo(lane), li = qc::mfma_li(lane);
    const int kv_head = h / (Hq / Hkv);
    const bool has_bias = bias != nullptr;
    const bool has_mask = mask != nullptr;

    auto q_index = [&](int i, int d) { return (size_t)((size_t)h * Lq + (q0 + i)) * D + d; };
    auto kv_index = [&](int j, int d) { return (size_t)((size_t)kv_head * Lk + j) * D + d; };
    auto bias_index = [&](int i, int j) {
        return (size_t)((size_t)h * Lq + (q0 + i)) * Lk + j;  // per head
    };
    auto mask_index = [&](int i, int j) {
        return (size_t)(q0 + i) * Lk + j;  // shared across heads
    };

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
    for (int s = 0; s < DT; ++s) Oacc[s] = qc::mfma_zero();

    __shared__ float sS[BQ][BK];
    __shared__ storage sP[BQ][BK];
    __shared__ float sm[BQ], sl[BQ], sc[BQ];
    if (lane < BQ) {
        sm[lane] = -INFINITY;
        sl[lane] = 0.0f;
    }
    __syncthreads();

    for (int k0 = 0; k0 < Lk; k0 += BK) {
        // Skip a key tile in which no (query, key) pair is live. Without this
        // the tiled kernel does the full Lq x Lk work regardless of the mask and
        // loses badly to the baseline on sparse masks, which skip masked keys
        // outright -- measured as a 0.70x regression on a +-8 band before this
        // check existed. Reading 256 mask bytes is far cheaper than loading a
        // 16xD K/V tile and issuing DT MFMAs against it.
        //
        // The whole block is one wavefront, so the reduced predicate is uniform
        // and `continue` cannot desynchronize the __syncthreads() below.
        if (has_mask) {
            int live = 0;
            for (int e = lane; e < BQ * BK; e += qc::kWave) {
                const int qi = e / BK, j = e % BK;
                const int kk = k0 + j;
                if ((q0 + qi) < Lq && kk < Lk && mask[mask_index(qi, kk)] != 0) live = 1;
            }
            if (qc::wave_reduce_max(live) == 0) continue;
        }

        float4_t S = qc::mfma_zero();
        const bool k_row_valid = (k0 + li) < Lk;
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
#pragma unroll
        for (int v = 0; v < 4; ++v) sS[4 * lo + v][li] = S[v] * scale;
        __syncthreads();

        if (lane < BQ) {
            const int qi = lane;
            const bool row_in_range = (q0 + qi) < Lq;
            const float previous_max = sm[qi];
            float row_max = previous_max;
            bool any_live = false;
            for (int j = 0; j < BK; ++j) {
                const int kk = k0 + j;
                const bool live = row_in_range && kk < Lk &&
                                  (!has_mask || mask[mask_index(qi, kk)] != 0);
                if (!live) {
                    sS[qi][j] = -INFINITY;  // marks "skip" for the exp pass below
                    continue;
                }
                float score = sS[qi][j];
                if (has_bias) score += bias[bias_index(qi, kk)];
                sS[qi][j] = score;
                row_max = fmaxf(row_max, score);
                any_live = true;
            }
            // A tile with no live keys must not disturb the running state, and
            // exp(-inf - -inf) would be NaN.
            const float correction =
                (!any_live || previous_max == -INFINITY) ? (any_live ? 0.0f : 1.0f)
                                                         : __expf(previous_max - row_max);
            float denominator = sl[qi] * correction;
            for (int j = 0; j < BK; ++j) {
                const float score = sS[qi][j];
                const float p = (score == -INFINITY) ? 0.0f : __expf(score - row_max);
                sP[qi][j] = T::from_float(p);
                denominator += p;
            }
            sm[qi] = row_max;
            sl[qi] = denominator;
            sc[qi] = correction;
        }
        __syncthreads();

#pragma unroll
        for (int s = 0; s < DT; ++s) {
            float4_t o = Oacc[s];
#pragma unroll
            for (int v = 0; v < 4; ++v) o[v] *= sc[4 * lo + v];
            Oacc[s] = o;
        }

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
                T::put(Vf, v, kj < Lk ? V[kv_index(kj, d)] : T::from_float(0.0f));
            }
            Oacc[s] = T::mma(Pf, Vf, Oacc[s]);
        }
        __syncthreads();
    }

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
// ---------------------------------------------------------------------------
template <typename T, int D>
__global__ __launch_bounds__(qc::kWave) void biased_attention_naive(
    const typename T::storage *__restrict__ Q, const typename T::storage *__restrict__ K,
    const typename T::storage *__restrict__ V, const float *__restrict__ bias,
    const uint8_t *__restrict__ mask, typename T::storage *__restrict__ O, int Hq,
    int Hkv, int Lq, int Lk, float scale) {
    constexpr int ITEMS = D / qc::kWave > 0 ? D / qc::kWave : 1;
    const int i = blockIdx.x, h = blockIdx.y;
    if (i >= Lq) return;
    const int lane = threadIdx.x;
    const int kv_head = h / (Hq / Hkv);
    const bool has_bias = bias != nullptr;
    const bool has_mask = mask != nullptr;

    const size_t q_base = (size_t)((size_t)h * Lq + i) * D;
    const size_t kv_base = (size_t)kv_head * Lk * D;
    const size_t bias_base = (size_t)((size_t)h * Lq + i) * Lk;
    const size_t mask_base = (size_t)i * Lk;

    // D may be smaller than the 64-lane wavefront (swin uses D=32): lanes past
    // D contribute zero to the dot product and skip the accumulator write.
    float q_val[ITEMS], acc[ITEMS];
#pragma unroll
    for (int t = 0; t < ITEMS; ++t) {
        const int d = lane + t * qc::kWave;
        q_val[t] = d < D ? T::to_float(Q[q_base + d]) : 0.0f;
        acc[t] = 0.0f;
    }

    float running_max = -INFINITY, denominator = 0.0f;
    for (int j = 0; j < Lk; ++j) {
        if (has_mask && mask[mask_base + j] == 0) continue;
        const size_t k_row = kv_base + (size_t)j * D;
        float dot = 0.0f;
#pragma unroll
        for (int t = 0; t < ITEMS; ++t) {
            const int d = lane + t * qc::kWave;
            if (d < D) dot = fmaf(q_val[t], T::to_float(K[k_row + d]), dot);
        }
        dot = qc::wave_reduce_sum(dot) * scale;
        const float score = has_bias ? dot + bias[bias_base + j] : dot;

        const float next_max = fmaxf(running_max, score);
        const float old_weight =
            (running_max == -INFINITY) ? 0.0f : __expf(running_max - next_max);
        const float new_weight = __expf(score - next_max);
        denominator = denominator * old_weight + new_weight;
#pragma unroll
        for (int t = 0; t < ITEMS; ++t) {
            const int d = lane + t * qc::kWave;
            acc[t] = acc[t] * old_weight +
                     (d < D ? T::to_float(V[k_row + d]) : 0.0f) * new_weight;
        }
        running_max = next_max;
    }

    const float inverse = denominator > 0.0f ? 1.0f / denominator : 0.0f;
#pragma unroll
    for (int t = 0; t < ITEMS; ++t) {
        const int d = lane + t * qc::kWave;
        if (d < D) O[q_base + d] = T::from_float(acc[t] * inverse);
    }
}

// ===========================================================================
// fp64 oracle -- mirrors biased_attention (attention_extended_ref.cpp ~L357)
// ===========================================================================
static std::vector<double> oracle(const std::vector<float> &Q, const std::vector<float> &K,
                                  const std::vector<float> &V,
                                  const std::vector<float> &bias, bool has_bias,
                                  const std::vector<uint8_t> &mask, bool has_mask, int Hq,
                                  int Hkv, int Lq, int Lk, int D, float scale) {
    std::vector<double> out((size_t)Hq * Lq * D, 0.0);
    const double score_scale =
        scale == 0.0f ? 1.0 / std::sqrt((double)D) : (double)scale;
    const int group = Hq / Hkv;

    for (int h = 0; h < Hq; ++h) {
        const int kv_head = h / group;
        for (int i = 0; i < Lq; ++i) {
            const size_t q_base = (size_t)((size_t)h * Lq + i) * D;
            std::vector<double> acc(D, 0.0);
            double maximum = -std::numeric_limits<double>::infinity();
            double denominator = 0.0;
            for (int j = 0; j < Lk; ++j) {
                if (has_mask && mask[(size_t)i * Lk + j] == 0) continue;
                const size_t kv_base = (size_t)((size_t)kv_head * Lk + j) * D;
                double dot = 0.0;
                for (int d = 0; d < D; ++d)
                    dot += (double)Q[q_base + d] * (double)K[kv_base + d];
                double score = dot * score_scale;
                if (has_bias) score += (double)bias[(size_t)((size_t)h * Lq + i) * Lk + j];

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
    return out;
}

// ===========================================================================
// Harness
// ===========================================================================

struct Case {
    const char *name;
    int Hq, Hkv, Lq, Lk;
    bool bias;
    int mask_mode;  // 0 none, 1 banded, 2 causal, 3 one fully-masked row
    float scale;    // 0 => automatic 1/sqrt(D)
};

template <typename T, int D>
static bool run_case(const Case &c, bool also_bench) {
    using storage = typename T::storage;
    qc::Rng rng(0xB1A5ED + c.Lq * 131 + c.Lk * 17 + D);

    const size_t q_count = (size_t)c.Hq * c.Lq * D;
    const size_t kv_count = (size_t)c.Hkv * c.Lk * D;
    const size_t bias_count = (size_t)c.Hq * c.Lq * c.Lk;
    const size_t mask_count = (size_t)c.Lq * c.Lk;

    auto Qf = rng.normals(q_count, 0.6f);
    auto Kf = rng.normals(kv_count, 0.6f);
    auto Vf = rng.normals(kv_count, 0.6f);
    std::vector<float> Bf = c.bias ? rng.normals(bias_count, 0.3f) : std::vector<float>();

    std::vector<uint8_t> mask;
    if (c.mask_mode != 0) {
        mask.assign(mask_count, 1);
        for (int i = 0; i < c.Lq; ++i)
            for (int j = 0; j < c.Lk; ++j) {
                bool live = true;
                if (c.mask_mode == 1) live = std::abs(i - j) <= 8;         // band
                else if (c.mask_mode == 2) live = j <= i;                  // causal
                else if (c.mask_mode == 3) live = (i != c.Lq / 2);         // one dead row
                mask[(size_t)i * c.Lk + j] = live ? 1 : 0;
            }
    }

    auto Qs = qc::to_storage<storage>(Qf);
    auto Ks = qc::to_storage<storage>(Kf);
    auto Vs = qc::to_storage<storage>(Vf);
    std::vector<float> Qr(q_count), Kr(kv_count), Vr(kv_count);
    for (size_t i = 0; i < q_count; ++i) Qr[i] = (float)qc::to_double(Qs[i]);
    for (size_t i = 0; i < kv_count; ++i) {
        Kr[i] = (float)qc::to_double(Ks[i]);
        Vr[i] = (float)qc::to_double(Vs[i]);
    }

    const bool has_mask = c.mask_mode != 0;
    auto ref = oracle(Qr, Kr, Vr, Bf, c.bias, mask, has_mask, c.Hq, c.Hkv, c.Lq, c.Lk, D,
                      c.scale);
    const float scale = c.scale == 0.0f ? 1.0f / std::sqrt((float)D) : c.scale;

    auto dQ = qc::dnew(Qs);
    auto dK = qc::dnew(Ks);
    auto dV = qc::dnew(Vs);
    float *dBias = c.bias ? qc::dnew(Bf) : nullptr;
    uint8_t *dMask = has_mask ? qc::dnew(mask) : nullptr;
    auto dO = qc::dzero<storage>(q_count);

    const dim3 grid_mfma(qc::grid_for(c.Lq, 16), c.Hq);
    const dim3 grid_naive(c.Lq, c.Hq);
    auto launch_mfma = [&] {
        biased_attention_mfma<T, D><<<grid_mfma, qc::kWave>>>(
            dQ, dK, dV, dBias, dMask, dO, c.Hq, c.Hkv, c.Lq, c.Lk, scale);
    };
    auto launch_naive = [&] {
        biased_attention_naive<T, D><<<grid_naive, qc::kWave>>>(
            dQ, dK, dV, dBias, dMask, dO, c.Hq, c.Hkv, c.Lq, c.Lk, scale);
    };

    char label[256];
    bool ok = true;
    const qc::Tol tol =
        std::is_same<T, Bf16Traits>::value ? qc::Tol::bf16_output() : qc::Tol::fp16_output();

    launch_mfma();
    QC_SYNC();
    std::snprintf(label, sizeof(label), "%s [%s D=%d] mfma", c.name, T::name, D);
    ok &= qc::compare(qc::d2h(dO, q_count), ref, tol).report(label);

    QC_CHECK(hipMemset(dO, 0, q_count * sizeof(storage)));
    launch_naive();
    QC_SYNC();
    std::snprintf(label, sizeof(label), "%s [%s D=%d] naive", c.name, T::name, D);
    ok &= qc::compare(qc::d2h(dO, q_count), ref, tol).report(label);

    if (also_bench && ok) {
        const auto b_naive = qc::bench(launch_naive);
        const auto b_mfma = qc::bench(launch_mfma);
        const double flops = 4.0 * (double)c.Hq * c.Lq * c.Lk * D;
        std::snprintf(label, sizeof(label), "%s [%s D=%d] naive", c.name, T::name, D);
        b_naive.report_compute(label, flops);
        std::snprintf(label, sizeof(label), "%s [%s D=%d] mfma", c.name, T::name, D);
        b_mfma.report_compute(label, flops);
        std::snprintf(label, sizeof(label), "%s [%s D=%d]", c.name, T::name, D);
        qc::report_ab(label, b_naive, b_mfma);
    }

    qc::dfree(dQ, dK, dV, dO);
    if (dBias) qc::dfree(dBias);
    if (dMask) qc::dfree(dMask);
    return ok;
}

int main(int argc, char **argv) {
    const bool do_bench = qc::bench_requested(argc, argv);
    qc::print_environment("biased_attention (CDNA3 gfx942)");
    bool ok = true;

    std::printf("correctness (fp64 oracle = quixicore_cpu::biased_attention):\n");
    const Case cases[] = {
        {"MHA no bias/mask",   8, 8, 64, 128, false, 0, 0.f},
        {"GQA 4:1 bias",       8, 2, 64, 128, true,  0, 0.f},
        {"banded mask",        4, 2, 64, 128, false, 1, 0.f},
        {"causal mask",        4, 2, 64, 64,  false, 2, 0.f},
        {"bias + banded mask", 4, 2, 64, 128, true,  1, 0.f},
        {"fully masked row",   4, 1, 32, 64,  false, 3, 0.f},
        {"ragged 37x101",      4, 2, 37, 101, true,  1, 0.f},
        {"negative scale",     2, 1, 32, 64,  false, 0, -0.0625f},
        {"explicit scale",     2, 2, 32, 48,  true,  0, 0.125f},
        {"decode Lq=1",        8, 2, 1,  512, true,  0, 0.f},
    };
    for (const auto &c : cases) {
        ok &= run_case<Bf16Traits, 64>(c, false);
        ok &= run_case<Bf16Traits, 128>(c, false);
    }
    // D=32 is the Swin head dim; D=256 exercises register pressure.
    ok &= run_case<Bf16Traits, 32>(cases[2], false);
    ok &= run_case<Bf16Traits, 32>(cases[4], false);
    ok &= run_case<Bf16Traits, 256>(cases[1], false);
    ok &= run_case<Fp16Traits, 64>(cases[4], false);
    ok &= run_case<Fp16Traits, 128>(cases[6], false);

    if (do_bench && ok) {
        std::printf("\nA/B naive wavefront-per-query vs MFMA BQ=16 tile "
                    "(HIP-event median, warmup 10 / iters 50):\n");
        const Case perf_cases[] = {
            {"512x512 bias",      32, 8, 512, 512,  true,  0, 0.f},
            {"512x2048 bias",     32, 8, 512, 2048, true,  0, 0.f},
            {"1024x1024 banded",  16, 4, 1024, 1024, false, 1, 0.f},
            {"512x512 causal",    32, 8, 512, 512,  false, 2, 0.f},
        };
        for (const auto &c : perf_cases) ok &= run_case<Bf16Traits, 128>(c, true);
        ok &= run_case<Bf16Traits, 64>(perf_cases[0], true);
    }

    return qc::finish(ok);
}
