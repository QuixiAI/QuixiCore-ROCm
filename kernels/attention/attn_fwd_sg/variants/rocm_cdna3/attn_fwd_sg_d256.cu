/**
 * @file
 * @brief CDNA3 port of Metal attn_fwd_sg_d256.
 */
#include "../../../../common/cdna3_harness.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <utility>
#include <vector>

namespace {

constexpr int kD = 256;

__device__ __forceinline__ float block_reduce_sum_256(float v) {
    __shared__ float scratch[kD];
    const int tid = threadIdx.x;
    scratch[tid] = v;
    __syncthreads();
    for (int offset = kD / 2; offset > 0; offset >>= 1) {
        if (tid < offset) scratch[tid] += scratch[tid + offset];
        __syncthreads();
    }
    return scratch[0];
}

__global__ void attn_fwd_sg_d256_block_kernel(
    const float *__restrict__ q, const __half *__restrict__ k, const __half *__restrict__ v,
    float *__restrict__ out, int T, int Hq, int Hkv, int window, float scale) {
    const int query = blockIdx.x;
    const int head = blockIdx.y;
    const int dim = threadIdx.x;
    if (query >= T || head >= Hq || dim >= kD) return;
    const int group_size = Hq / Hkv;
    const int kv_head = head / group_size;
    const size_t q_base = (static_cast<size_t>(query) * Hq + head) * kD;
    const int half_window = window / 2;
    const int first = window == 0 ? 0 : max(0, query - half_window);
    const int last = window == 0 ? T : min(T, query + half_window + 1);

    float row_max = -INFINITY;
    float row_sum = 0.0f;
    float acc = 0.0f;
    const float qd = q[q_base + dim] * scale;

    for (int key = first; key < last; ++key) {
        const size_t kv_base = (static_cast<size_t>(key) * Hkv + kv_head) * kD;
        const float partial = qd * __half2float(k[kv_base + dim]);
        const float score = block_reduce_sum_256(partial);
        const float next_max = fmaxf(row_max, score);
        const float alpha = __expf(row_max - next_max);
        const float prob = __expf(score - next_max);
        row_sum = row_sum * alpha + prob;
        acc = acc * alpha + prob * __half2float(v[kv_base + dim]);
        row_max = next_max;
    }
    out[q_base + dim] = row_sum == 0.0f ? 0.0f : acc / row_sum;
}

__global__ void attn_fwd_sg_d256_scalar_kernel(
    const float *__restrict__ q, const __half *__restrict__ k, const __half *__restrict__ v,
    float *__restrict__ out, int T, int Hq, int Hkv, int window, float scale) {
    const int query = blockIdx.x;
    const int head = blockIdx.y;
    if (threadIdx.x != 0 || query >= T || head >= Hq) return;
    const int group_size = Hq / Hkv;
    const int kv_head = head / group_size;
    const size_t q_base = (static_cast<size_t>(query) * Hq + head) * kD;
    const int half_window = window / 2;
    const int first = window == 0 ? 0 : max(0, query - half_window);
    const int last = window == 0 ? T : min(T, query + half_window + 1);
    float acc[kD];
    for (int d = 0; d < kD; ++d) acc[d] = 0.0f;
    float row_max = -INFINITY;
    float row_sum = 0.0f;
    for (int key = first; key < last; ++key) {
        const size_t kv_base = (static_cast<size_t>(key) * Hkv + kv_head) * kD;
        double score64 = 0.0;
        for (int d = 0; d < kD; ++d)
            score64 += static_cast<double>(q[q_base + d] * scale) *
                       static_cast<double>(__half2float(k[kv_base + d]));
        const float score = static_cast<float>(score64);
        const float next_max = fmaxf(row_max, score);
        const float alpha = __expf(row_max - next_max);
        const float prob = __expf(score - next_max);
        row_sum = row_sum * alpha + prob;
        for (int d = 0; d < kD; ++d)
            acc[d] = acc[d] * alpha + prob * __half2float(v[kv_base + d]);
        row_max = next_max;
    }
    const float inv = row_sum == 0.0f ? 0.0f : 1.0f / row_sum;
    for (int d = 0; d < kD; ++d) out[q_base + d] = acc[d] * inv;
}

std::vector<double> reference(const std::vector<float> &q, const std::vector<__half> &k,
                              const std::vector<__half> &v, int T, int Hq, int Hkv,
                              int window, float scale) {
    std::vector<double> out(static_cast<size_t>(T) * Hq * kD, 0.0);
    const int group_size = Hq / Hkv;
    for (int query = 0; query < T; ++query) {
        const int half_window = window / 2;
        const int first = window == 0 ? 0 : std::max(0, query - half_window);
        const int last = window == 0 ? T : std::min(T, query + half_window + 1);
        for (int head = 0; head < Hq; ++head) {
            const int kv_head = head / group_size;
            const size_t q_base = (static_cast<size_t>(query) * Hq + head) * kD;
            std::vector<double> scores(last - first);
            double max_score = -std::numeric_limits<double>::infinity();
            for (int key = first; key < last; ++key) {
                const size_t kv_base = (static_cast<size_t>(key) * Hkv + kv_head) * kD;
                double score = 0.0;
                for (int d = 0; d < kD; ++d)
                    score += static_cast<double>(q[q_base + d]) * scale *
                             static_cast<double>(__half2float(k[kv_base + d]));
                scores[key - first] = score;
                max_score = std::max(max_score, score);
            }
            double denom = 0.0;
            for (double &s : scores) {
                s = std::exp(s - max_score);
                denom += s;
            }
            for (int d = 0; d < kD; ++d) {
                double acc = 0.0;
                for (int key = first; key < last; ++key) {
                    const size_t kv_base = (static_cast<size_t>(key) * Hkv + kv_head) * kD;
                    acc += scores[key - first] * static_cast<double>(__half2float(v[kv_base + d]));
                }
                out[q_base + d] = acc / denom;
            }
        }
    }
    return out;
}

bool run_case(int T, int Hq, int Hkv, int window, bool bench) {
    qc::Rng rng(31000 + T + Hq * 13 + Hkv * 17 + window);
    std::vector<float> q(static_cast<size_t>(T) * Hq * kD);
    std::vector<float> kf(static_cast<size_t>(T) * Hkv * kD);
    std::vector<float> vf(static_cast<size_t>(T) * Hkv * kD);
    for (float &x : q) x = rng.normal(0.0f, 1.0f);
    for (float &x : kf) x = rng.normal(0.0f, 1.0f);
    for (float &x : vf) x = rng.normal(0.0f, 1.0f);
    auto k = qc::to_storage<__half>(kf);
    auto v = qc::to_storage<__half>(vf);
    float *dq = qc::dnew(q);
    __half *dk = qc::dnew(k);
    __half *dv = qc::dnew(v);
    float *dout = qc::dzero<float>(static_cast<size_t>(T) * Hq * kD);
    const float scale = 1.0f / std::sqrt(static_cast<float>(kD));
    dim3 grid(T, Hq);
    auto launch = [&] {
        attn_fwd_sg_d256_block_kernel<<<grid, kD>>>(dq, dk, dv, dout, T, Hq, Hkv, window, scale);
    };
    launch();
    QC_SYNC();
    char label[96];
    std::snprintf(label, sizeof(label), "attn_fwd_sg_d256 T=%d Hq=%d Hkv=%d w=%d",
                  T, Hq, Hkv, window);
    bool ok = qc::compare(qc::d2h(dout, static_cast<size_t>(T) * Hq * kD),
                          reference(q, k, v, T, Hq, Hkv, window, scale),
                          qc::Tol::fp32().with_elementwise(2e-2, 2e-2))
                  .report(label);

    if (bench && ok) {
        float *dbase = qc::dzero<float>(static_cast<size_t>(T) * Hq * kD);
        auto scalar = [&] {
            attn_fwd_sg_d256_scalar_kernel<<<grid, 1>>>(dq, dk, dv, dbase, T, Hq, Hkv, window, scale);
        };
        const int scalar_repeat = 3;
        const int block_repeat = 20;
        auto b0 = qc::bench([&] {
            for (int i = 0; i < scalar_repeat; ++i) scalar();
        }, 3, 12);
        auto b1 = qc::bench([&] {
            for (int i = 0; i < block_repeat; ++i) launch();
        }, 5, 20);
        b0.median_ms /= scalar_repeat;
        b0.min_ms /= scalar_repeat;
        b0.max_ms /= scalar_repeat;
        b0.mean_ms /= scalar_repeat;
        b1.median_ms /= block_repeat;
        b1.min_ms /= block_repeat;
        b1.max_ms /= block_repeat;
        b1.mean_ms /= block_repeat;
        std::printf("  attn_fwd_sg_d256 timing batches: scalar %d launches/sample, block %d launches/sample (per-launch shown)\n",
                    scalar_repeat, block_repeat);
        const double avg_keys = window == 0 ? static_cast<double>(T)
                                            : std::min<double>(T, window + 1.0);
        const double flops = 4.0 * T * Hq * avg_keys * kD;
        b0.report_compute("attn_fwd_sg_d256 scalar", flops);
        b1.report_compute("attn_fwd_sg_d256 block", flops);
        qc::report_ab(label, b0, b1);
        qc::dfree(dbase);
    }

    qc::dfree(dq, dk, dv, dout);
    return ok;
}

void run_benchmarks() {
    (void)run_case(256, 4, 2, 0, true);
    (void)run_case(256, 4, 2, 64, true);
}

}  // namespace

int main(int argc, char **argv) {
    qc::print_environment("attn_fwd_sg_d256");
    bool ok = true;
    for (int T : {8, 40, 65}) {
        for (auto heads : {std::pair<int, int>{3, 1}, std::pair<int, int>{4, 2},
                           std::pair<int, int>{2, 2}}) {
            ok &= run_case(T, heads.first, heads.second, 0, false);
            ok &= run_case(T, heads.first, heads.second, 16, false);
        }
    }
    if (qc::bench_requested(argc, argv)) run_benchmarks();
    return qc::finish(ok);
}
