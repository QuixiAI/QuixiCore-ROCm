/**
 * @file
 * @brief CDNA3 port of Metal mean_pool_rms_l2: bf16 mean pool, RMSNorm, L2.
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
__global__ void mean_pool_rms_l2_wave_kernel(const bf16 *__restrict__ x,
                                             const bf16 *__restrict__ weight,
                                             bf16 *__restrict__ out,
                                             uint32_t M, float eps) {
    constexpr int ITEMS = D / qc::kWave;
    const int lane = threadIdx.x & (qc::kWave - 1);
    float acc[ITEMS];
#pragma unroll
    for (int item = 0; item < ITEMS; ++item) acc[item] = 0.0f;

    for (uint32_t row = 0; row < M; ++row) {
        const size_t base = static_cast<size_t>(row) * D;
#pragma unroll
        for (int item = 0; item < ITEMS; ++item) {
            const int dim = lane + item * qc::kWave;
            acc[item] += bff(x[base + dim]);
        }
    }

    const float inv_m = 1.0f / static_cast<float>(M);
    float mean_sq = 0.0f;
#pragma unroll
    for (int item = 0; item < ITEMS; ++item) {
        acc[item] *= inv_m;
        mean_sq = fmaf(acc[item], acc[item], mean_sq);
    }
    mean_sq = qc::wave_reduce_sum(mean_sq);
    const float rms_inv = rsqrtf(mean_sq / static_cast<float>(D) + eps);

    float normed[ITEMS];
    float l2 = 0.0f;
#pragma unroll
    for (int item = 0; item < ITEMS; ++item) {
        const int dim = lane + item * qc::kWave;
        const float y = acc[item] * rms_inv * bff(weight[dim]);
        normed[item] = y;
        l2 = fmaf(y, y, l2);
    }
    l2 = qc::wave_reduce_sum(l2);
    const float l2_inv = rsqrtf(l2 + 1e-12f);
#pragma unroll
    for (int item = 0; item < ITEMS; ++item) {
        const int dim = lane + item * qc::kWave;
        out[dim] = __float2bfloat16(normed[item] * l2_inv);
    }
}

template <int D>
__global__ void mean_pool_rms_l2_scalar_kernel(const bf16 *__restrict__ x,
                                               const bf16 *__restrict__ weight,
                                               bf16 *__restrict__ out,
                                               uint32_t M, float eps) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float pooled[D];
    for (int d = 0; d < D; ++d) {
        double acc = 0.0;
        for (uint32_t row = 0; row < M; ++row)
            acc += static_cast<double>(bff(x[static_cast<size_t>(row) * D + d]));
        pooled[d] = static_cast<float>(acc / static_cast<double>(M));
    }
    double mean_sq = 0.0;
    for (int d = 0; d < D; ++d) mean_sq += static_cast<double>(pooled[d]) * pooled[d];
    const float rms_inv = static_cast<float>(
        1.0 / sqrt(mean_sq / static_cast<double>(D) + static_cast<double>(eps)));
    double l2 = 0.0;
    for (int d = 0; d < D; ++d) {
        pooled[d] = pooled[d] * rms_inv * bff(weight[d]);
        l2 += static_cast<double>(pooled[d]) * pooled[d];
    }
    const float l2_inv = static_cast<float>(1.0 / sqrt(l2 + 1e-12));
    for (int d = 0; d < D; ++d) out[d] = __float2bfloat16(pooled[d] * l2_inv);
}

template <int D>
std::vector<double> reference(const std::vector<bf16> &x,
                              const std::vector<bf16> &weight,
                              uint32_t M, float eps) {
    std::vector<double> pooled(D, 0.0);
    for (uint32_t row = 0; row < M; ++row) {
        const size_t base = static_cast<size_t>(row) * D;
        for (int d = 0; d < D; ++d) pooled[d] += qc::to_double(x[base + d]);
    }
    for (double &v : pooled) v /= static_cast<double>(M);
    double mean_sq = 0.0;
    for (double v : pooled) mean_sq += v * v;
    const double rms_inv = 1.0 / std::sqrt(mean_sq / static_cast<double>(D) + eps);
    double l2 = 0.0;
    for (int d = 0; d < D; ++d) {
        pooled[d] = pooled[d] * rms_inv * qc::to_double(weight[d]);
        l2 += pooled[d] * pooled[d];
    }
    const double l2_inv = 1.0 / std::sqrt(l2 + 1e-12);
    for (double &v : pooled) v *= l2_inv;
    return pooled;
}

template <int D>
bool run_case(uint32_t M, bool bench) {
    qc::Rng rng(9100 + D + M);
    std::vector<float> xf(static_cast<size_t>(M) * D);
    std::vector<float> wf(D);
    for (float &v : xf) v = rng.normal(0.0f, 0.7f);
    for (float &v : wf) v = rng.normal(0.0f, 0.5f);
    auto x = qc::to_storage<bf16>(xf);
    auto w = qc::to_storage<bf16>(wf);
    bf16 *dx = qc::dnew(x);
    bf16 *dw = qc::dnew(w);
    bf16 *dout = qc::dzero<bf16>(D);
    const float eps = 1e-6f;

    auto launch = [&] {
        mean_pool_rms_l2_wave_kernel<D><<<1, qc::kWave>>>(dx, dw, dout, M, eps);
    };
    launch();
    QC_SYNC();
    char label[96];
    std::snprintf(label, sizeof(label), "mean_pool_rms_l2 M=%u D=%d", M, D);
    const auto got = qc::d2h(dout, D);
    bool ok = qc::compare(got, reference<D>(x, w, M, eps), qc::Tol::bf16_output())
                  .report(label);

    if (bench && ok) {
        bf16 *dbase = qc::dzero<bf16>(D);
        auto scalar = [&] {
            mean_pool_rms_l2_scalar_kernel<D><<<1, 1>>>(dx, dw, dbase, M, eps);
        };
        const auto b0 = qc::bench(scalar, 3, 10);
        const auto b1 = qc::bench(launch, 10, 50);
        const double bytes = (static_cast<double>(M) * D + D + D) * sizeof(bf16);
        b0.report_bandwidth("mean_pool_rms_l2 scalar", bytes);
        b1.report_bandwidth("mean_pool_rms_l2 wave64", bytes);
        qc::report_ab(label, b0, b1);
        qc::dfree(dbase);
    }

    qc::dfree(dx, dw, dout);
    return ok;
}

void run_benchmarks() {
    (void)run_case<768>(1024, true);
    (void)run_case<1024>(1024, true);
}

}  // namespace

int main(int argc, char **argv) {
    qc::print_environment("mean_pool_rms_l2");
    bool ok = true;
    for (uint32_t M : {1u, 37u, 128u}) {
        ok &= run_case<256>(M, false);
        ok &= run_case<512>(M, false);
        ok &= run_case<768>(M, false);
        ok &= run_case<1024>(M, false);
    }
    if (qc::bench_requested(argc, argv)) run_benchmarks();
    return qc::finish(ok);
}
