/**
 * @file
 * @brief CDNA3 port of Metal rms_norm_residual_next bf16 residual seam.
 */
#include "../../../../common/cdna3_harness.cuh"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

using bf16 = __hip_bfloat16;

__device__ __forceinline__ float bff(bf16 v) { return __bfloat162float(v); }

template <int D>
__global__ void rms_norm_residual_next_wave_kernel(
    const bf16 *__restrict__ x, const bf16 *__restrict__ post_weight,
    const bf16 *__restrict__ residual, const bf16 *__restrict__ next_weight,
    bf16 *__restrict__ res_out, bf16 *__restrict__ next_out, uint32_t M, float eps) {
    constexpr int ITEMS = D / qc::kWave;
    const int wave = threadIdx.x >> 6;
    const int lane = threadIdx.x & (qc::kWave - 1);
    const uint32_t row = blockIdx.x * qc::kWavesPerBlock + wave;
    if (row >= M) return;
    const size_t base = static_cast<size_t>(row) * D;

    float xs[ITEMS];
    float ss = 0.0f;
#pragma unroll
    for (int item = 0; item < ITEMS; ++item) {
        const int dim = lane + item * qc::kWave;
        const float v = bff(x[base + dim]);
        xs[item] = v;
        ss = fmaf(v, v, ss);
    }
    ss = qc::wave_reduce_sum(ss);
    const float pinv = rsqrtf(ss / static_cast<float>(D) + eps);

    float rv[ITEMS];
    float rss = 0.0f;
#pragma unroll
    for (int item = 0; item < ITEMS; ++item) {
        const int dim = lane + item * qc::kWave;
        const float r = bff(residual[base + dim]) +
                        xs[item] * pinv * bff(post_weight[dim]);
        rv[item] = r;
        res_out[base + dim] = __float2bfloat16(r);
        rss = fmaf(r, r, rss);
    }
    rss = qc::wave_reduce_sum(rss);
    const float rinv = rsqrtf(rss / static_cast<float>(D) + eps);
#pragma unroll
    for (int item = 0; item < ITEMS; ++item) {
        const int dim = lane + item * qc::kWave;
        next_out[base + dim] = __float2bfloat16(rv[item] * rinv * bff(next_weight[dim]));
    }
}

template <int D>
__global__ void rms_norm_residual_next_scalar_kernel(
    const bf16 *__restrict__ x, const bf16 *__restrict__ post_weight,
    const bf16 *__restrict__ residual, const bf16 *__restrict__ next_weight,
    bf16 *__restrict__ res_out, bf16 *__restrict__ next_out, uint32_t M, float eps) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;
    const size_t base = static_cast<size_t>(row) * D;
    float rv[D];
    double ss = 0.0;
    for (int d = 0; d < D; ++d) {
        const float v = bff(x[base + d]);
        ss += static_cast<double>(v) * v;
    }
    const float pinv = static_cast<float>(
        1.0 / sqrt(ss / static_cast<double>(D) + static_cast<double>(eps)));
    double rss = 0.0;
    for (int d = 0; d < D; ++d) {
        const float r = bff(residual[base + d]) + bff(x[base + d]) * pinv * bff(post_weight[d]);
        rv[d] = r;
        res_out[base + d] = __float2bfloat16(r);
        rss += static_cast<double>(r) * r;
    }
    const float rinv = static_cast<float>(
        1.0 / sqrt(rss / static_cast<double>(D) + static_cast<double>(eps)));
    for (int d = 0; d < D; ++d) {
        next_out[base + d] = __float2bfloat16(rv[d] * rinv * bff(next_weight[d]));
    }
}

template <int D>
void reference(const std::vector<bf16> &x, const std::vector<bf16> &post_weight,
               const std::vector<bf16> &residual,
               const std::vector<bf16> &next_weight, uint32_t M, float eps,
               std::vector<double> &res_ref, std::vector<double> &next_ref) {
    res_ref.assign(static_cast<size_t>(M) * D, 0.0);
    next_ref.assign(static_cast<size_t>(M) * D, 0.0);
    for (uint32_t row = 0; row < M; ++row) {
        const size_t base = static_cast<size_t>(row) * D;
        double ss = 0.0;
        for (int d = 0; d < D; ++d) {
            const double v = qc::to_double(x[base + d]);
            ss += v * v;
        }
        const double pinv = 1.0 / std::sqrt(ss / static_cast<double>(D) + eps);
        double rss = 0.0;
        for (int d = 0; d < D; ++d) {
            const double r = qc::to_double(residual[base + d]) +
                             qc::to_double(x[base + d]) * pinv *
                                 qc::to_double(post_weight[d]);
            res_ref[base + d] = r;
            rss += r * r;
        }
        const double rinv = 1.0 / std::sqrt(rss / static_cast<double>(D) + eps);
        for (int d = 0; d < D; ++d)
            next_ref[base + d] = res_ref[base + d] * rinv * qc::to_double(next_weight[d]);
    }
}

template <int D>
bool run_case(uint32_t M, bool bench) {
    qc::Rng rng(12000 + D + M);
    std::vector<float> xf(static_cast<size_t>(M) * D);
    std::vector<float> rf(static_cast<size_t>(M) * D);
    std::vector<float> pwf(D), nwf(D);
    for (float &v : xf) v = rng.normal(0.0f, 0.65f);
    for (float &v : rf) v = rng.normal(0.0f, 0.55f);
    for (float &v : pwf) v = rng.normal(0.0f, 0.4f);
    for (float &v : nwf) v = rng.normal(0.0f, 0.4f);
    auto x = qc::to_storage<bf16>(xf);
    auto r = qc::to_storage<bf16>(rf);
    auto pw = qc::to_storage<bf16>(pwf);
    auto nw = qc::to_storage<bf16>(nwf);
    bf16 *dx = qc::dnew(x);
    bf16 *dr = qc::dnew(r);
    bf16 *dpw = qc::dnew(pw);
    bf16 *dnw = qc::dnew(nw);
    bf16 *dres = qc::dzero<bf16>(static_cast<size_t>(M) * D);
    bf16 *dnext = qc::dzero<bf16>(static_cast<size_t>(M) * D);
    const float eps = 1e-6f;

    auto launch = [&] {
        rms_norm_residual_next_wave_kernel<D>
            <<<qc::wave_blocks(M), qc::kThreads>>>(dx, dpw, dr, dnw, dres, dnext, M, eps);
    };
    launch();
    QC_SYNC();

    std::vector<double> res_ref, next_ref;
    reference<D>(x, pw, r, nw, M, eps, res_ref, next_ref);
    char label[96];
    std::snprintf(label, sizeof(label), "rms_norm_residual_next res M=%u D=%d", M, D);
    bool ok = qc::compare(qc::d2h(dres, static_cast<size_t>(M) * D), res_ref,
                          qc::Tol::bf16_output())
                  .report(label);
    std::snprintf(label, sizeof(label), "rms_norm_residual_next next M=%u D=%d", M, D);
    ok &= qc::compare(qc::d2h(dnext, static_cast<size_t>(M) * D), next_ref,
                      qc::Tol::bf16_output())
              .report(label);

    if (bench && ok) {
        bf16 *dres_base = qc::dzero<bf16>(static_cast<size_t>(M) * D);
        bf16 *dnext_base = qc::dzero<bf16>(static_cast<size_t>(M) * D);
        auto scalar = [&] {
            rms_norm_residual_next_scalar_kernel<D>
                <<<qc::grid_for(M, 128), 128>>>(dx, dpw, dr, dnw, dres_base, dnext_base, M, eps);
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
        std::printf("  rms_norm_residual_next timing batches: scalar %d launches/sample, wave64 %d launches/sample (per-launch shown)\n",
                    scalar_repeat, wave_repeat);
        const double bytes = (4.0 * D + 2.0 * D * M) * sizeof(bf16) +
                             2.0 * D * M * sizeof(bf16);
        b0.report_bandwidth("rms_norm_residual_next scalar", bytes);
        b1.report_bandwidth("rms_norm_residual_next wave64", bytes);
        std::snprintf(label, sizeof(label), "rms_norm_residual_next M=%u D=%d", M, D);
        qc::report_ab(label, b0, b1);
        qc::dfree(dres_base, dnext_base);
    }

    qc::dfree(dx, dr, dpw, dnw, dres, dnext);
    return ok;
}

void run_benchmarks() {
    (void)run_case<768>(4096, true);
    (void)run_case<1024>(4096, true);
}

}  // namespace

int main(int argc, char **argv) {
    qc::print_environment("rms_norm_residual_next");
    bool ok = true;
    for (uint32_t M : {8u, 37u, 257u}) {
        ok &= run_case<256>(M, false);
        ok &= run_case<512>(M, false);
        ok &= run_case<768>(M, false);
        ok &= run_case<1024>(M, false);
    }
    if (qc::bench_requested(argc, argv)) run_benchmarks();
    return qc::finish(ok);
}
