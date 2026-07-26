/**
 * @file
 * @brief CDNA3 port of Metal fused packed-Q4_0 fp32 decode GEMV kernels.
 */
#include "../../../../common/cdna3_harness.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

constexpr int kQK = 32;

struct __align__(2) block_q4_0 {
    __half d;
    uint8_t qs[16];
};

static_assert(sizeof(block_q4_0) == 18, "Q4_0 block must be the Metal/GGUF 18-byte layout");

__device__ __host__ __forceinline__ float gelu_tanh(float x) {
    constexpr float c = 0.7978845608028654f;
    return 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x * x * x)));
}

__device__ __forceinline__ float q4_dot_wave(const block_q4_0 *__restrict__ w,
                                             const float *__restrict__ x,
                                             int row, int K) {
    const int lane = threadIdx.x & (qc::kWave - 1);
    const int bpr = K / kQK;
    float acc = 0.0f;
    for (int kb = lane; kb < bpr; kb += qc::kWave) {
        const block_q4_0 &blk = w[static_cast<size_t>(row) * bpr + kb];
        const float d = __half2float(blk.d);
        const int xbase = kb * kQK;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const uint8_t packed = blk.qs[i];
            acc += d * static_cast<float>((packed & 0x0f) - 8) * x[xbase + i];
            acc += d * static_cast<float>((packed >> 4) - 8) * x[xbase + i + 16];
        }
    }
    return qc::wave_reduce_sum(acc);
}

__device__ __forceinline__ float q4_dot_scalar(const block_q4_0 *__restrict__ w,
                                               const float *__restrict__ x,
                                               int row, int K) {
    const int bpr = K / kQK;
    float acc = 0.0f;
    for (int kb = 0; kb < bpr; ++kb) {
        const block_q4_0 &blk = w[static_cast<size_t>(row) * bpr + kb];
        const float d = __half2float(blk.d);
        const int xbase = kb * kQK;
        for (int i = 0; i < 16; ++i) {
            const uint8_t packed = blk.qs[i];
            acc += d * static_cast<float>((packed & 0x0f) - 8) * x[xbase + i];
            acc += d * static_cast<float>((packed >> 4) - 8) * x[xbase + i + 16];
        }
    }
    return acc;
}

__global__ void qgemv_up_gate_gelu_wave_kernel(
    float *__restrict__ out, const block_q4_0 *__restrict__ up,
    const block_q4_0 *__restrict__ gate, const float *__restrict__ x, int N, int K) {
    const int wave = threadIdx.x >> 6;
    const int lane = threadIdx.x & (qc::kWave - 1);
    const int row = blockIdx.x * qc::kWavesPerBlock + wave;
    if (row >= N) return;
    const float up_sum = q4_dot_wave(up, x, row, K);
    const float gate_sum = q4_dot_wave(gate, x, row, K);
    if (lane == 0) out[row] = gelu_tanh(gate_sum) * up_sum;
}

__global__ void qgemv_up_gate_wave_kernel(
    float *__restrict__ up_out, float *__restrict__ gate_out,
    const block_q4_0 *__restrict__ up, const block_q4_0 *__restrict__ gate,
    const float *__restrict__ x, int N, int K) {
    const int wave = threadIdx.x >> 6;
    const int lane = threadIdx.x & (qc::kWave - 1);
    const int combined = blockIdx.x * qc::kWavesPerBlock + wave;
    if (combined >= 2 * N) return;
    const bool is_gate = combined >= N;
    const int row = is_gate ? combined - N : combined;
    const float acc = q4_dot_wave(is_gate ? gate : up, x, row, K);
    if (lane == 0) (is_gate ? gate_out : up_out)[row] = acc;
}

__global__ void qgemv_qkv_wave_kernel(
    float *__restrict__ q_out, float *__restrict__ k_out, float *__restrict__ v_out,
    const block_q4_0 *__restrict__ qw, const block_q4_0 *__restrict__ kw,
    const block_q4_0 *__restrict__ vw, const float *__restrict__ x,
    int Nq, int Nkv, int K) {
    const int wave = threadIdx.x >> 6;
    const int lane = threadIdx.x & (qc::kWave - 1);
    const int combined = blockIdx.x * qc::kWavesPerBlock + wave;
    if (combined >= Nq + 2 * Nkv) return;
    const block_q4_0 *w = nullptr;
    float *out = nullptr;
    int row = 0;
    if (combined < Nq) {
        w = qw;
        out = q_out;
        row = combined;
    } else if (combined < Nq + Nkv) {
        w = kw;
        out = k_out;
        row = combined - Nq;
    } else {
        w = vw;
        out = v_out;
        row = combined - Nq - Nkv;
    }
    const float acc = q4_dot_wave(w, x, row, K);
    if (lane == 0) out[row] = acc;
}

__global__ void qgemv_up_gate_gelu_scalar_kernel(
    float *__restrict__ out, const block_q4_0 *__restrict__ up,
    const block_q4_0 *__restrict__ gate, const float *__restrict__ x, int N, int K) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;
    const float up_sum = q4_dot_scalar(up, x, row, K);
    const float gate_sum = q4_dot_scalar(gate, x, row, K);
    out[row] = gelu_tanh(gate_sum) * up_sum;
}

__global__ void qgemv_up_gate_scalar_kernel(
    float *__restrict__ up_out, float *__restrict__ gate_out,
    const block_q4_0 *__restrict__ up, const block_q4_0 *__restrict__ gate,
    const float *__restrict__ x, int N, int K) {
    const int combined = blockIdx.x * blockDim.x + threadIdx.x;
    if (combined >= 2 * N) return;
    const bool is_gate = combined >= N;
    const int row = is_gate ? combined - N : combined;
    (is_gate ? gate_out : up_out)[row] = q4_dot_scalar(is_gate ? gate : up, x, row, K);
}

__global__ void qgemv_qkv_scalar_kernel(
    float *__restrict__ q_out, float *__restrict__ k_out, float *__restrict__ v_out,
    const block_q4_0 *__restrict__ qw, const block_q4_0 *__restrict__ kw,
    const block_q4_0 *__restrict__ vw, const float *__restrict__ x,
    int Nq, int Nkv, int K) {
    const int combined = blockIdx.x * blockDim.x + threadIdx.x;
    if (combined >= Nq + 2 * Nkv) return;
    const block_q4_0 *w = nullptr;
    float *out = nullptr;
    int row = 0;
    if (combined < Nq) {
        w = qw;
        out = q_out;
        row = combined;
    } else if (combined < Nq + Nkv) {
        w = kw;
        out = k_out;
        row = combined - Nq;
    } else {
        w = vw;
        out = v_out;
        row = combined - Nq - Nkv;
    }
    out[row] = q4_dot_scalar(w, x, row, K);
}

std::vector<block_q4_0> quantize_q4_0(const std::vector<float> &src, int rows, int K) {
    const int bpr = K / kQK;
    std::vector<block_q4_0> out(static_cast<size_t>(rows) * bpr);
    for (int r = 0; r < rows; ++r) {
        for (int b = 0; b < bpr; ++b) {
            const float *base = src.data() + static_cast<size_t>(r) * K + b * kQK;
            float amax = 0.0f;
            float vmax = 0.0f;
            for (int i = 0; i < kQK; ++i) {
                if (std::fabs(base[i]) > amax) {
                    amax = std::fabs(base[i]);
                    vmax = base[i];
                }
            }
            const float d = vmax / -8.0f;
            const float id = d == 0.0f ? 0.0f : 1.0f / d;
            block_q4_0 &blk = out[static_cast<size_t>(r) * bpr + b];
            blk.d = __float2half(d);
            for (int i = 0; i < 16; ++i) {
                const int x0 = std::min(15, std::max(0, static_cast<int>(std::lround(base[i] * id + 8.0f))));
                const int x1 = std::min(15, std::max(0, static_cast<int>(std::lround(base[i + 16] * id + 8.0f))));
                blk.qs[i] = static_cast<uint8_t>(x0 | (x1 << 4));
            }
        }
    }
    return out;
}

std::vector<double> ref_dot(const std::vector<block_q4_0> &w,
                            const std::vector<float> &x, int rows, int K) {
    const int bpr = K / kQK;
    std::vector<double> out(rows, 0.0);
    for (int r = 0; r < rows; ++r) {
        double acc = 0.0;
        for (int kb = 0; kb < bpr; ++kb) {
            const block_q4_0 &blk = w[static_cast<size_t>(r) * bpr + kb];
            const double d = __half2float(blk.d);
            const int xbase = kb * kQK;
            for (int i = 0; i < 16; ++i) {
                const uint8_t packed = blk.qs[i];
                acc += d * static_cast<double>((packed & 0x0f) - 8) * x[xbase + i];
                acc += d * static_cast<double>((packed >> 4) - 8) * x[xbase + i + 16];
            }
        }
        out[r] = acc;
    }
    return out;
}

bool run_up_gate_gelu(int N, int K, bool bench) {
    qc::Rng rng(21000 + N + K);
    std::vector<float> upf(static_cast<size_t>(N) * K);
    std::vector<float> gatef(static_cast<size_t>(N) * K);
    std::vector<float> x(K);
    for (float &v : upf) v = rng.normal(0.0f, 0.3f);
    for (float &v : gatef) v = rng.normal(0.0f, 0.3f);
    for (float &v : x) v = rng.normal(0.0f, 0.8f);
    auto up = quantize_q4_0(upf, N, K);
    auto gate = quantize_q4_0(gatef, N, K);
    auto ref_up = ref_dot(up, x, N, K);
    auto ref_gate = ref_dot(gate, x, N, K);
    std::vector<double> ref(N);
    for (int i = 0; i < N; ++i) ref[i] = static_cast<double>(gelu_tanh(static_cast<float>(ref_gate[i]))) * ref_up[i];

    block_q4_0 *dup = qc::dnew(up);
    block_q4_0 *dgate = qc::dnew(gate);
    float *dx = qc::dnew(x);
    float *dout = qc::dzero<float>(N);
    auto launch = [&] {
        qgemv_up_gate_gelu_wave_kernel
            <<<qc::wave_blocks(N), qc::kThreads>>>(dout, dup, dgate, dx, N, K);
    };
    launch();
    QC_SYNC();
    char label[96];
    std::snprintf(label, sizeof(label), "qgemv_q4_0_f32_up_gate_gelu N=%d K=%d", N, K);
    bool ok = qc::compare(qc::d2h(dout, N), ref, qc::Tol::fp32().with_elementwise(2e-2, 2e-2))
                  .report(label);
    if (bench && ok) {
        float *dbase = qc::dzero<float>(N);
        auto scalar = [&] {
            qgemv_up_gate_gelu_scalar_kernel
                <<<qc::grid_for(N, 128), 128>>>(dbase, dup, dgate, dx, N, K);
        };
        const int scalar_repeat = 1000;
        const int wave_repeat = 2000;
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
        std::printf("  qgemv_up_gate_gelu timing batches: scalar %d launches/sample, wave64 %d launches/sample (per-launch shown)\n",
                    scalar_repeat, wave_repeat);
        const double ops = 4.0 * N * K;
        b0.report_compute("qgemv up_gate_gelu scalar", ops);
        b1.report_compute("qgemv up_gate_gelu wave64", ops);
        qc::report_ab(label, b0, b1);
        qc::dfree(dbase);
    }
    qc::dfree(dup, dgate, dx, dout);
    return ok;
}

bool run_up_gate(int N, int K, bool bench) {
    qc::Rng rng(22000 + N + K);
    std::vector<float> upf(static_cast<size_t>(N) * K);
    std::vector<float> gatef(static_cast<size_t>(N) * K);
    std::vector<float> x(K);
    for (float &v : upf) v = rng.normal(0.0f, 0.3f);
    for (float &v : gatef) v = rng.normal(0.0f, 0.3f);
    for (float &v : x) v = rng.normal(0.0f, 0.8f);
    auto up = quantize_q4_0(upf, N, K);
    auto gate = quantize_q4_0(gatef, N, K);
    auto ref_up = ref_dot(up, x, N, K);
    auto ref_gate = ref_dot(gate, x, N, K);
    block_q4_0 *dup = qc::dnew(up);
    block_q4_0 *dgate = qc::dnew(gate);
    float *dx = qc::dnew(x);
    float *du = qc::dzero<float>(N);
    float *dg = qc::dzero<float>(N);
    auto launch = [&] {
        qgemv_up_gate_wave_kernel
            <<<qc::wave_blocks(2 * static_cast<size_t>(N)), qc::kThreads>>>(du, dg, dup, dgate, dx, N, K);
    };
    launch();
    QC_SYNC();
    char label[96];
    std::snprintf(label, sizeof(label), "qgemv_q4_0_f32_up_gate up N=%d K=%d", N, K);
    bool ok = qc::compare(qc::d2h(du, N), ref_up, qc::Tol::fp32().with_elementwise(2e-2, 2e-2))
                  .report(label);
    std::snprintf(label, sizeof(label), "qgemv_q4_0_f32_up_gate gate N=%d K=%d", N, K);
    ok &= qc::compare(qc::d2h(dg, N), ref_gate, qc::Tol::fp32().with_elementwise(2e-2, 2e-2))
              .report(label);
    if (bench && ok) {
        float *du_base = qc::dzero<float>(N);
        float *dg_base = qc::dzero<float>(N);
        auto scalar = [&] {
            qgemv_up_gate_scalar_kernel
                <<<qc::grid_for(2 * static_cast<size_t>(N), 128), 128>>>(du_base, dg_base, dup, dgate, dx, N, K);
        };
        const int scalar_repeat = 1000;
        const int wave_repeat = 2000;
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
        std::printf("  qgemv_up_gate timing batches: scalar %d launches/sample, wave64 %d launches/sample (per-launch shown)\n",
                    scalar_repeat, wave_repeat);
        const double ops = 4.0 * N * K;
        b0.report_compute("qgemv up_gate scalar", ops);
        b1.report_compute("qgemv up_gate wave64", ops);
        std::snprintf(label, sizeof(label), "qgemv_q4_0_f32_up_gate N=%d K=%d", N, K);
        qc::report_ab(label, b0, b1);
        qc::dfree(du_base, dg_base);
    }
    qc::dfree(dup, dgate, dx, du, dg);
    return ok;
}

bool run_qkv(int Nq, int Nkv, int K, bool bench) {
    qc::Rng rng(23000 + Nq + Nkv + K);
    std::vector<float> qf(static_cast<size_t>(Nq) * K);
    std::vector<float> kf(static_cast<size_t>(Nkv) * K);
    std::vector<float> vf(static_cast<size_t>(Nkv) * K);
    std::vector<float> x(K);
    for (float &v : qf) v = rng.normal(0.0f, 0.3f);
    for (float &v : kf) v = rng.normal(0.0f, 0.3f);
    for (float &v : vf) v = rng.normal(0.0f, 0.3f);
    for (float &v : x) v = rng.normal(0.0f, 0.8f);
    auto qw = quantize_q4_0(qf, Nq, K);
    auto kw = quantize_q4_0(kf, Nkv, K);
    auto vw = quantize_q4_0(vf, Nkv, K);
    auto qref = ref_dot(qw, x, Nq, K);
    auto kref = ref_dot(kw, x, Nkv, K);
    auto vref = ref_dot(vw, x, Nkv, K);
    block_q4_0 *dqw = qc::dnew(qw);
    block_q4_0 *dkw = qc::dnew(kw);
    block_q4_0 *dvw = qc::dnew(vw);
    float *dx = qc::dnew(x);
    float *dq = qc::dzero<float>(Nq);
    float *dk = qc::dzero<float>(Nkv);
    float *dv = qc::dzero<float>(Nkv);
    auto launch = [&] {
        qgemv_qkv_wave_kernel
            <<<qc::wave_blocks(Nq + 2 * static_cast<size_t>(Nkv)), qc::kThreads>>>(
                dq, dk, dv, dqw, dkw, dvw, dx, Nq, Nkv, K);
    };
    launch();
    QC_SYNC();
    char label[96];
    std::snprintf(label, sizeof(label), "qgemv_q4_0_f32_qkv Q Nq=%d Nkv=%d K=%d", Nq, Nkv, K);
    bool ok = qc::compare(qc::d2h(dq, Nq), qref, qc::Tol::fp32().with_elementwise(2e-2, 2e-2))
                  .report(label);
    std::snprintf(label, sizeof(label), "qgemv_q4_0_f32_qkv K Nq=%d Nkv=%d K=%d", Nq, Nkv, K);
    ok &= qc::compare(qc::d2h(dk, Nkv), kref, qc::Tol::fp32().with_elementwise(2e-2, 2e-2))
              .report(label);
    std::snprintf(label, sizeof(label), "qgemv_q4_0_f32_qkv V Nq=%d Nkv=%d K=%d", Nq, Nkv, K);
    ok &= qc::compare(qc::d2h(dv, Nkv), vref, qc::Tol::fp32().with_elementwise(2e-2, 2e-2))
              .report(label);
    if (bench && ok) {
        float *dq_base = qc::dzero<float>(Nq);
        float *dk_base = qc::dzero<float>(Nkv);
        float *dv_base = qc::dzero<float>(Nkv);
        auto scalar = [&] {
            qgemv_qkv_scalar_kernel
                <<<qc::grid_for(Nq + 2 * static_cast<size_t>(Nkv), 128), 128>>>(
                    dq_base, dk_base, dv_base, dqw, dkw, dvw, dx, Nq, Nkv, K);
        };
        const int scalar_repeat = 1000;
        const int wave_repeat = 2000;
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
        std::printf("  qgemv_qkv timing batches: scalar %d launches/sample, wave64 %d launches/sample (per-launch shown)\n",
                    scalar_repeat, wave_repeat);
        const double ops = 2.0 * (Nq + 2.0 * Nkv) * K;
        b0.report_compute("qgemv qkv scalar", ops);
        b1.report_compute("qgemv qkv wave64", ops);
        std::snprintf(label, sizeof(label), "qgemv_q4_0_f32_qkv Nq=%d Nkv=%d K=%d", Nq, Nkv, K);
        qc::report_ab(label, b0, b1);
        qc::dfree(dq_base, dk_base, dv_base);
    }
    qc::dfree(dqw, dkw, dvw, dx, dq, dk, dv);
    return ok;
}

void run_benchmarks() {
    (void)run_up_gate_gelu(1152, 768, true);
    (void)run_up_gate(1152, 768, true);
    (void)run_qkv(768, 256, 768, true);
}

}  // namespace

int main(int argc, char **argv) {
    qc::print_environment("qgemv_fused");
    bool ok = true;
    ok &= run_up_gate_gelu(128, 256, false);
    ok &= run_up_gate_gelu(1152, 768, false);
    ok &= run_up_gate_gelu(256, 512, false);
    ok &= run_up_gate(128, 256, false);
    ok &= run_up_gate(1152, 768, false);
    ok &= run_qkv(768, 256, 768, false);
    ok &= run_qkv(512, 128, 256, false);
    if (qc::bench_requested(argc, argv)) run_benchmarks();
    return qc::finish(ok);
}
