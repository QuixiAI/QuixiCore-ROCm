/**
 * @file
 * @brief Phase 10 KD, optimizer, softmax-backward, and SiLU-backward ports.
 */
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include "../../../../common/cdna3_harness.cuh"

namespace {

constexpr float kTiny = 1e-30f;
constexpr float kNegInf = -std::numeric_limits<float>::infinity();

std::vector<double> to_ref(const std::vector<float> &values) {
    return std::vector<double>(values.begin(), values.end());
}

__device__ __forceinline__ float sigmoid_device(float value) {
    if (value >= 0.0f) return 1.0f / (1.0f + expf(-value));
    const float e = expf(value);
    return e / (1.0f + e);
}

float sigmoid_host(float value) {
    if (value >= 0.0f) return 1.0f / (1.0f + std::exp(-value));
    const float e = std::exp(value);
    return e / (1.0f + e);
}

__device__ __forceinline__ float block_reduce_sum_float(float value,
                                                        float *scratch) {
    const int lane = threadIdx.x & (qc::kWave - 1);
    const int wave = threadIdx.x >> 6;
    const int waves = blockDim.x >> 6;
    value = qc::wave_reduce_sum(value);
    if (lane == 0) scratch[wave] = value;
    __syncthreads();
    float total = 0.0f;
    for (int w = 0; w < waves; ++w) total += scratch[w];
    __syncthreads();
    return total;
}

__device__ __forceinline__ double block_reduce_sum_double(double value,
                                                          double *scratch) {
    const int lane = threadIdx.x & (qc::kWave - 1);
    const int wave = threadIdx.x >> 6;
    const int waves = blockDim.x >> 6;
    value = qc::wave_reduce_sum(value);
    if (lane == 0) scratch[wave] = value;
    __syncthreads();
    double total = 0.0;
    for (int w = 0; w < waves; ++w) total += scratch[w];
    __syncthreads();
    return total;
}

__device__ __forceinline__ float block_reduce_max_float(float value,
                                                        float *scratch) {
    const int lane = threadIdx.x & (qc::kWave - 1);
    const int wave = threadIdx.x >> 6;
    const int waves = blockDim.x >> 6;
    value = qc::wave_reduce_max(value);
    if (lane == 0) scratch[wave] = value;
    __syncthreads();
    float result = kNegInf;
    for (int w = 0; w < waves; ++w) result = fmaxf(result, scratch[w]);
    __syncthreads();
    return result;
}

// ---------------------------------------------------------------------------
// KD KL dense and CE-fused
// ---------------------------------------------------------------------------

__global__ void kd_kl_dense_fwd_scalar_kernel(
    const float *teacher, const float *student, float *loss,
    float *teacher_lse, float *student_lse, long long rows, long long vocab,
    float inverse_temperature) {
    const long long row = blockIdx.x;
    if (threadIdx.x != 0 || row >= rows) return;
    const float *tr = teacher + row * vocab;
    const float *sr = student + row * vocab;
    double tmax = -INFINITY;
    double smax = -INFINITY;
    for (long long token = 0; token < vocab; ++token) {
        tmax = fmax(tmax, static_cast<double>(tr[token]) * inverse_temperature);
        smax = fmax(smax, static_cast<double>(sr[token]) * inverse_temperature);
    }
    double tsum = 0.0;
    double ssum = 0.0;
    for (long long token = 0; token < vocab; ++token) {
        tsum += exp(static_cast<double>(tr[token]) * inverse_temperature - tmax);
        ssum += exp(static_cast<double>(sr[token]) * inverse_temperature - smax);
    }
    const double tlse = tmax + log(tsum);
    const double slse = smax + log(ssum);
    double value = 0.0;
    for (long long token = 0; token < vocab; ++token) {
        const double teacher_log = tr[token] * inverse_temperature - tlse;
        const double student_log = sr[token] * inverse_temperature - slse;
        value += exp(teacher_log) * (teacher_log - student_log);
    }
    loss[row] = static_cast<float>(value);
    teacher_lse[row] = static_cast<float>(tlse);
    student_lse[row] = static_cast<float>(slse);
}

__global__ void kd_kl_dense_fwd_kernel(
    const float *teacher, const float *student, float *loss,
    float *teacher_lse, float *student_lse, long long rows, long long vocab,
    float inverse_temperature) {
    const long long row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= rows) return;
    const long long base = row * vocab;
    __shared__ float scratch[4];
    float tmax_local = kNegInf;
    float smax_local = kNegInf;
    for (long long token = tid; token < vocab; token += blockDim.x) {
        tmax_local = fmaxf(tmax_local, teacher[base + token] * inverse_temperature);
        smax_local = fmaxf(smax_local, student[base + token] * inverse_temperature);
    }
    const float tmax = block_reduce_max_float(tmax_local, scratch);
    const float smax = block_reduce_max_float(smax_local, scratch);
    float tsum_local = 0.0f;
    float ssum_local = 0.0f;
    for (long long token = tid; token < vocab; token += blockDim.x) {
        tsum_local += expf(teacher[base + token] * inverse_temperature - tmax);
        ssum_local += expf(student[base + token] * inverse_temperature - smax);
    }
    const float tsum = block_reduce_sum_float(tsum_local, scratch);
    const float ssum = block_reduce_sum_float(ssum_local, scratch);
    const float tlse = tmax + logf(tsum);
    const float slse = smax + logf(ssum);
    float loss_local = 0.0f;
    for (long long token = tid; token < vocab; token += blockDim.x) {
        const float teacher_log = teacher[base + token] * inverse_temperature - tlse;
        const float student_log = student[base + token] * inverse_temperature - slse;
        loss_local += expf(teacher_log) * (teacher_log - student_log);
    }
    const float row_loss = block_reduce_sum_float(loss_local, scratch);
    if (tid == 0) {
        loss[row] = row_loss;
        teacher_lse[row] = tlse;
        student_lse[row] = slse;
    }
}

__global__ void kd_kl_dense_bwd_scalar_kernel(
    const float *teacher, const float *student, const float *teacher_lse,
    const float *student_lse, const float *grad_out, float *grad_student,
    long long rows, long long vocab, float inverse_temperature) {
    const long long row = blockIdx.x;
    if (threadIdx.x != 0 || row >= rows) return;
    for (long long token = 0; token < vocab; ++token) {
        const double q =
            exp(static_cast<double>(student[row * vocab + token]) *
                    inverse_temperature -
                student_lse[row]);
        const double p =
            exp(static_cast<double>(teacher[row * vocab + token]) *
                    inverse_temperature -
                teacher_lse[row]);
        grad_student[row * vocab + token] =
            static_cast<float>(grad_out[row] * inverse_temperature * (q - p));
    }
}

__global__ void kd_kl_dense_bwd_kernel(
    const float *teacher, const float *student, const float *teacher_lse,
    const float *student_lse, const float *grad_out, float *grad_student,
    long long count, long long vocab, float inverse_temperature) {
    const long long index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const long long row = index / vocab;
    const double q =
        exp(static_cast<double>(student[index]) * inverse_temperature -
            student_lse[row]);
    const double p =
        exp(static_cast<double>(teacher[index]) * inverse_temperature -
            teacher_lse[row]);
    grad_student[index] =
        static_cast<float>(grad_out[row] * inverse_temperature * (q - p));
}

__global__ void kd_ce_fused_fwd_scalar_kernel(
    const float *teacher, const float *student, const int *targets,
    float *ce_loss, float *kd_loss, float *raw_student_lse,
    float *tempered_student_lse, float *teacher_lse, long long rows,
    long long vocab, float inverse_temperature, int ignore_index) {
    const long long row = blockIdx.x;
    if (threadIdx.x != 0 || row >= rows) return;
    const long long base = row * vocab;
    double raw_max = -INFINITY;
    double st_max = -INFINITY;
    double t_max = -INFINITY;
    for (long long token = 0; token < vocab; ++token) {
        raw_max = fmax(raw_max, static_cast<double>(student[base + token]));
        st_max = fmax(st_max,
                      static_cast<double>(student[base + token]) *
                          inverse_temperature);
        t_max = fmax(t_max,
                     static_cast<double>(teacher[base + token]) *
                         inverse_temperature);
    }
    double raw_sum = 0.0;
    double st_sum = 0.0;
    double t_sum = 0.0;
    for (long long token = 0; token < vocab; ++token) {
        raw_sum += exp(static_cast<double>(student[base + token]) - raw_max);
        st_sum += exp(static_cast<double>(student[base + token]) *
                          inverse_temperature -
                      st_max);
        t_sum += exp(static_cast<double>(teacher[base + token]) *
                         inverse_temperature -
                     t_max);
    }
    const double raw_lse = raw_max + log(raw_sum);
    const double st_lse = st_max + log(st_sum);
    const double t_lse = t_max + log(t_sum);
    double kd = 0.0;
    for (long long token = 0; token < vocab; ++token) {
        const double teacher_log =
            teacher[base + token] * inverse_temperature - t_lse;
        const double student_log =
            student[base + token] * inverse_temperature - st_lse;
        kd += exp(teacher_log) * (teacher_log - student_log);
    }
    const int target = targets[row];
    raw_student_lse[row] = static_cast<float>(raw_lse);
    tempered_student_lse[row] = static_cast<float>(st_lse);
    teacher_lse[row] = static_cast<float>(t_lse);
    kd_loss[row] = static_cast<float>(kd);
    ce_loss[row] =
        target == ignore_index ? 0.0f
                               : static_cast<float>(raw_lse -
                                                    student[base + target]);
}

__global__ void kd_ce_fused_fwd_kernel(
    const float *teacher, const float *student, const int *targets,
    float *ce_loss, float *kd_loss, float *raw_student_lse,
    float *tempered_student_lse, float *teacher_lse, long long rows,
    long long vocab, float inverse_temperature, int ignore_index) {
    const long long row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= rows) return;
    const long long base = row * vocab;
    __shared__ float scratch[4];
    float raw_max_local = kNegInf;
    float st_max_local = kNegInf;
    float t_max_local = kNegInf;
    for (long long token = tid; token < vocab; token += blockDim.x) {
        raw_max_local = fmaxf(raw_max_local, student[base + token]);
        st_max_local = fmaxf(st_max_local,
                             student[base + token] * inverse_temperature);
        t_max_local = fmaxf(t_max_local,
                            teacher[base + token] * inverse_temperature);
    }
    const float raw_max = block_reduce_max_float(raw_max_local, scratch);
    const float st_max = block_reduce_max_float(st_max_local, scratch);
    const float t_max = block_reduce_max_float(t_max_local, scratch);
    float raw_sum_local = 0.0f;
    float st_sum_local = 0.0f;
    float t_sum_local = 0.0f;
    for (long long token = tid; token < vocab; token += blockDim.x) {
        raw_sum_local += expf(student[base + token] - raw_max);
        st_sum_local +=
            expf(student[base + token] * inverse_temperature - st_max);
        t_sum_local += expf(teacher[base + token] * inverse_temperature - t_max);
    }
    const float raw_lse = raw_max + logf(block_reduce_sum_float(raw_sum_local, scratch));
    const float st_lse = st_max + logf(block_reduce_sum_float(st_sum_local, scratch));
    const float t_lse = t_max + logf(block_reduce_sum_float(t_sum_local, scratch));
    float kd_local = 0.0f;
    for (long long token = tid; token < vocab; token += blockDim.x) {
        const float teacher_log = teacher[base + token] * inverse_temperature - t_lse;
        const float student_log = student[base + token] * inverse_temperature - st_lse;
        kd_local += expf(teacher_log) * (teacher_log - student_log);
    }
    const float kd = block_reduce_sum_float(kd_local, scratch);
    if (tid == 0) {
        const int target = targets[row];
        raw_student_lse[row] = raw_lse;
        tempered_student_lse[row] = st_lse;
        teacher_lse[row] = t_lse;
        kd_loss[row] = kd;
        ce_loss[row] =
            target == ignore_index ? 0.0f : raw_lse - student[base + target];
    }
}

__global__ void kd_ce_fused_bwd_kernel(
    const float *teacher, const float *student, const int *targets,
    const float *raw_student_lse, const float *tempered_student_lse,
    const float *teacher_lse, const float *grad_ce, const float *grad_kd,
    float *grad_student, long long count, long long vocab,
    float inverse_temperature, int ignore_index) {
    const long long index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const long long row = index / vocab;
    const long long token = index - row * vocab;
    const double student_probability =
        exp(static_cast<double>(student[index]) * inverse_temperature -
            tempered_student_lse[row]);
    const double teacher_probability =
        exp(static_cast<double>(teacher[index]) * inverse_temperature -
            teacher_lse[row]);
    double gradient =
        grad_kd[row] * inverse_temperature *
        (student_probability - teacher_probability);
    const int target = targets[row];
    if (target != ignore_index) {
        gradient += grad_ce[row] *
                    (exp(static_cast<double>(student[index]) -
                         raw_student_lse[row]) -
                     (token == target ? 1.0 : 0.0));
    }
    grad_student[index] = static_cast<float>(gradient);
}

__global__ void kd_ce_fused_bwd_scalar_kernel(
    const float *teacher, const float *student, const int *targets,
    const float *raw_student_lse, const float *tempered_student_lse,
    const float *teacher_lse, const float *grad_ce, const float *grad_kd,
    float *grad_student, long long rows, long long vocab,
    float inverse_temperature, int ignore_index) {
    const long long row = blockIdx.x;
    if (threadIdx.x != 0 || row >= rows) return;
    const int target = targets[row];
    for (long long token = 0; token < vocab; ++token) {
        const long long index = row * vocab + token;
        const double student_probability =
            exp(static_cast<double>(student[index]) * inverse_temperature -
                tempered_student_lse[row]);
        const double teacher_probability =
            exp(static_cast<double>(teacher[index]) * inverse_temperature -
                teacher_lse[row]);
        double gradient =
            grad_kd[row] * inverse_temperature *
            (student_probability - teacher_probability);
        if (target != ignore_index) {
            gradient += grad_ce[row] *
                        (exp(static_cast<double>(student[index]) -
                             raw_student_lse[row]) -
                         (token == target ? 1.0 : 0.0));
        }
        grad_student[index] = static_cast<float>(gradient);
    }
}

// ---------------------------------------------------------------------------
// KD KL top-k
// ---------------------------------------------------------------------------

__global__ void kd_kl_topk_fwd_scalar_kernel(
    const float *student, const int *teacher_indices,
    const float *teacher_probabilities, float *loss, float *student_lse,
    long long rows, long long vocab, long long top_k,
    float inverse_temperature, bool include_tail) {
    const long long row = blockIdx.x;
    if (threadIdx.x != 0 || row >= rows) return;
    const float *sr = student + row * vocab;
    double maximum = -INFINITY;
    for (long long token = 0; token < vocab; ++token) {
        maximum = fmax(maximum, static_cast<double>(sr[token]) * inverse_temperature);
    }
    double sum = 0.0;
    for (long long token = 0; token < vocab; ++token) {
        sum += exp(static_cast<double>(sr[token]) * inverse_temperature - maximum);
    }
    const double lse = maximum + log(sum);
    double probability_sum = 0.0;
    double student_selected = 0.0;
    for (long long item = 0; item < top_k; ++item) {
        const long long index = row * top_k + item;
        const int token = teacher_indices[index];
        if (token >= 0) {
            probability_sum += teacher_probabilities[index];
            student_selected += exp(sr[token] * inverse_temperature - lse);
        }
    }
    double value = 0.0;
    const double inverse_sum = 1.0 / fmax(probability_sum, static_cast<double>(kTiny));
    for (long long item = 0; item < top_k; ++item) {
        const long long index = row * top_k + item;
        const int token = teacher_indices[index];
        if (token < 0) continue;
        double p = teacher_probabilities[index];
        if (!include_tail) p *= inverse_sum;
        if (p > 0.0) {
            const double log_q = sr[token] * inverse_temperature - lse;
            value += p * (log(fmax(p, static_cast<double>(kTiny))) - log_q);
        }
    }
    if (include_tail) {
        const double tail = fmax(1.0 - probability_sum, 0.0);
        if (tail > 0.0) {
            value += tail * (log(fmax(tail, static_cast<double>(kTiny))) -
                             log(fmax(1.0 - student_selected,
                                      static_cast<double>(kTiny))));
        }
    }
    loss[row] = static_cast<float>(value);
    student_lse[row] = static_cast<float>(lse);
}

__global__ void kd_kl_topk_fwd_kernel(
    const float *student, const int *teacher_indices,
    const float *teacher_probabilities, float *loss, float *student_lse,
    long long rows, long long vocab, long long top_k,
    float inverse_temperature, bool include_tail) {
    const long long row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= rows) return;
    const long long base = row * vocab;
    const long long kbase = row * top_k;
    __shared__ float scratch[4];
    float max_local = kNegInf;
    for (long long token = tid; token < vocab; token += blockDim.x) {
        max_local = fmaxf(max_local, student[base + token] * inverse_temperature);
    }
    const float row_max = block_reduce_max_float(max_local, scratch);
    float sum_local = 0.0f;
    for (long long token = tid; token < vocab; token += blockDim.x) {
        sum_local += expf(student[base + token] * inverse_temperature - row_max);
    }
    const float lse = row_max + logf(block_reduce_sum_float(sum_local, scratch));
    float p_local = 0.0f;
    float selected_local = 0.0f;
    for (long long item = tid; item < top_k; item += blockDim.x) {
        const int token = teacher_indices[kbase + item];
        if (token >= 0) {
            const float p = teacher_probabilities[kbase + item];
            p_local += p;
            selected_local += expf(student[base + token] * inverse_temperature - lse);
        }
    }
    const float probability_sum = block_reduce_sum_float(p_local, scratch);
    const float student_selected = block_reduce_sum_float(selected_local, scratch);
    const float inverse_sum = 1.0f / fmaxf(probability_sum, kTiny);
    float loss_local = 0.0f;
    for (long long item = tid; item < top_k; item += blockDim.x) {
        const int token = teacher_indices[kbase + item];
        if (token < 0) continue;
        float p = teacher_probabilities[kbase + item];
        if (!include_tail) p *= inverse_sum;
        if (p > 0.0f) {
            const float log_q = student[base + token] * inverse_temperature - lse;
            loss_local += p * (logf(fmaxf(p, kTiny)) - log_q);
        }
    }
    float row_loss = block_reduce_sum_float(loss_local, scratch);
    if (include_tail) {
        const float tail = fmaxf(1.0f - probability_sum, 0.0f);
        if (tail > 0.0f && tid == 0) {
            row_loss += tail * (logf(fmaxf(tail, kTiny)) -
                                logf(fmaxf(1.0f - student_selected, kTiny)));
        }
    }
    if (tid == 0) {
        loss[row] = row_loss;
        student_lse[row] = lse;
    }
}

__global__ void kd_kl_topk_bwd_scalar_kernel(
    const float *student, const int *teacher_indices,
    const float *teacher_probabilities, const float *student_lse,
    const float *grad_out, float *grad_student, long long rows,
    long long vocab, long long top_k, float inverse_temperature,
    bool include_tail) {
    const long long row = blockIdx.x;
    if (threadIdx.x != 0 || row >= rows) return;
    const long long base = row * vocab;
    const long long kbase = row * top_k;
    double probability_sum = 0.0;
    double student_selected = 0.0;
    for (long long item = 0; item < top_k; ++item) {
        const int token = teacher_indices[kbase + item];
        if (token >= 0) {
            probability_sum += teacher_probabilities[kbase + item];
            student_selected +=
                exp(student[base + token] * inverse_temperature - student_lse[row]);
        }
    }
    const double tail = fmax(1.0 - probability_sum, 0.0);
    const double tail_c =
        include_tail && tail > 0.0
            ? tail / fmax(1.0 - student_selected, static_cast<double>(kTiny))
            : 0.0;
    const double q_coefficient =
        include_tail ? probability_sum - tail_c * student_selected : 1.0;
    const double inverse_sum = 1.0 / fmax(probability_sum, static_cast<double>(kTiny));
    const double go = grad_out[row] * inverse_temperature;
    for (long long token = 0; token < vocab; ++token) {
        const double q =
            exp(student[base + token] * inverse_temperature - student_lse[row]);
        double gradient = q_coefficient * q * go;
        for (long long item = 0; item < top_k; ++item) {
            if (teacher_indices[kbase + item] != token) continue;
            const double p = teacher_probabilities[kbase + item];
            const double correction =
                include_tail ? -p + tail_c * q : -p * inverse_sum;
            gradient += correction * go;
        }
        grad_student[base + token] = static_cast<float>(gradient);
    }
}

__global__ void kd_kl_topk_bwd_kernel(
    const float *student, const int *teacher_indices,
    const float *teacher_probabilities, const float *student_lse,
    const float *grad_out, float *grad_student, long long rows,
    long long vocab, long long top_k, float inverse_temperature,
    bool include_tail) {
    const long long row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= rows) return;
    const long long base = row * vocab;
    const long long kbase = row * top_k;
    __shared__ float scratch[4];
    float p_local = 0.0f;
    float selected_local = 0.0f;
    for (long long item = tid; item < top_k; item += blockDim.x) {
        const int token = teacher_indices[kbase + item];
        if (token >= 0) {
            const float p = teacher_probabilities[kbase + item];
            p_local += p;
            selected_local +=
                expf(student[base + token] * inverse_temperature - student_lse[row]);
        }
    }
    const float probability_sum = block_reduce_sum_float(p_local, scratch);
    const float student_selected = block_reduce_sum_float(selected_local, scratch);
    const float tail = fmaxf(1.0f - probability_sum, 0.0f);
    const float tail_c =
        include_tail && tail > 0.0f ? tail / fmaxf(1.0f - student_selected, kTiny)
                                    : 0.0f;
    const float q_coefficient =
        include_tail ? probability_sum - tail_c * student_selected : 1.0f;
    const float inverse_sum = 1.0f / fmaxf(probability_sum, kTiny);
    const float go = grad_out[row] * inverse_temperature;
    for (long long token = tid; token < vocab; token += blockDim.x) {
        const float q =
            expf(student[base + token] * inverse_temperature - student_lse[row]);
        float gradient = q_coefficient * q * go;
        for (long long item = 0; item < top_k; ++item) {
            if (teacher_indices[kbase + item] != token) continue;
            const float p = teacher_probabilities[kbase + item];
            const float correction = include_tail ? -p + tail_c * q : -p * inverse_sum;
            gradient += correction * go;
        }
        grad_student[base + token] = gradient;
    }
}

// ---------------------------------------------------------------------------
// Optimizer and elementwise backward kernels
// ---------------------------------------------------------------------------

__global__ void adamw_masked_segment_kernel(
    float *parameters, const float *gradients, float *first_moment,
    float *second_moment, const uint8_t *mask, long long count,
    long long segment_size, int mask_mode, float lr, float beta1, float beta2,
    float eps, float weight_decay, double first_correction,
    double second_correction) {
    const long long segment = blockIdx.x;
    if (threadIdx.x != 0) return;
    const bool active = mask[segment] != 0;
    const long long begin = segment * segment_size;
    const long long end = (begin + segment_size < count) ? begin + segment_size : count;
    for (long long index = begin; index < end; ++index) {
        if (!active && mask_mode == 0) continue;
        const float gradient = gradients[index];
        first_moment[index] = beta1 * first_moment[index] + (1.0f - beta1) * gradient;
        second_moment[index] =
            beta2 * second_moment[index] + (1.0f - beta2) * gradient * gradient;
        const double corrected_first = first_moment[index] / first_correction;
        const double corrected_second = second_moment[index] / second_correction;
        const double update =
            corrected_first / (sqrt(corrected_second) + eps) +
            (active ? weight_decay * parameters[index] : 0.0f);
        parameters[index] -= static_cast<float>(lr * update);
    }
}

__global__ void adamw_masked_kernel(
    float *parameters, const float *gradients, float *first_moment,
    float *second_moment, const uint8_t *mask, long long count,
    long long segment_size, int mask_mode, float lr, float beta1, float beta2,
    float eps, float weight_decay, double first_correction,
    double second_correction) {
    const long long index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const bool active = mask[index / segment_size] != 0;
    if (!active && mask_mode == 0) return;
    const float gradient = gradients[index];
    first_moment[index] = beta1 * first_moment[index] + (1.0f - beta1) * gradient;
    second_moment[index] =
        beta2 * second_moment[index] + (1.0f - beta2) * gradient * gradient;
    const double corrected_first = first_moment[index] / first_correction;
    const double corrected_second = second_moment[index] / second_correction;
    const double update =
        corrected_first / (sqrt(corrected_second) + eps) +
        (active ? weight_decay * parameters[index] : 0.0f);
    parameters[index] -= static_cast<float>(lr * update);
}

__global__ void sgd_chunk_scalar_kernel(float *parameters,
                                        const float *gradients,
                                        long long count, float lr,
                                        float weight_decay,
                                        long long chunk) {
    const long long block = blockIdx.x;
    if (threadIdx.x != 0) return;
    const long long begin = block * chunk;
    const long long end = (begin + chunk < count) ? begin + chunk : count;
    for (long long index = begin; index < end; ++index) {
        parameters[index] -= lr * (gradients[index] + weight_decay * parameters[index]);
    }
}

__global__ void sgd_kernel(float *parameters, const float *gradients,
                           long long count, float lr, float weight_decay) {
    const long long index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    parameters[index] -= lr * (gradients[index] + weight_decay * parameters[index]);
}

__global__ void softmax_backward_scalar_kernel(const float *grad_out,
                                               const float *softmax_output,
                                               float *grad_in,
                                               long long rows,
                                               long long dim) {
    const long long row = blockIdx.x;
    if (threadIdx.x != 0 || row >= rows) return;
    double dot = 0.0;
    for (long long i = 0; i < dim; ++i) {
        dot += static_cast<double>(grad_out[row * dim + i]) *
               softmax_output[row * dim + i];
    }
    for (long long i = 0; i < dim; ++i) {
        grad_in[row * dim + i] =
            softmax_output[row * dim + i] *
            (grad_out[row * dim + i] - static_cast<float>(dot));
    }
}

__global__ void softmax_backward_kernel(const float *grad_out,
                                        const float *softmax_output,
                                        float *grad_in, long long rows,
                                        long long dim) {
    const long long row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= rows) return;
    __shared__ double scratch[4];
    double local = 0.0;
    for (long long i = tid; i < dim; i += blockDim.x) {
        local += static_cast<double>(grad_out[row * dim + i]) *
                 softmax_output[row * dim + i];
    }
    const double dot = block_reduce_sum_double(local, scratch);
    for (long long i = tid; i < dim; i += blockDim.x) {
        grad_in[row * dim + i] =
            softmax_output[row * dim + i] *
            (grad_out[row * dim + i] - static_cast<float>(dot));
    }
}

__global__ void silu_backward_chunk_scalar_kernel(const float *grad_out,
                                                  const float *x,
                                                  float *grad_in,
                                                  long long count,
                                                  long long chunk) {
    const long long block = blockIdx.x;
    if (threadIdx.x != 0) return;
    const long long begin = block * chunk;
    const long long end = (begin + chunk < count) ? begin + chunk : count;
    for (long long index = begin; index < end; ++index) {
        const float probability = sigmoid_device(x[index]);
        grad_in[index] =
            grad_out[index] * probability * (1.0f + x[index] * (1.0f - probability));
    }
}

__global__ void silu_backward_kernel(const float *grad_out, const float *x,
                                     float *grad_in, long long count) {
    const long long index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float probability = sigmoid_device(x[index]);
    grad_in[index] =
        grad_out[index] * probability * (1.0f + x[index] * (1.0f - probability));
}

// ---------------------------------------------------------------------------
// Host references
// ---------------------------------------------------------------------------

double row_lse(const std::vector<float> &values, long long row, long long vocab,
               float multiplier) {
    const long long base = row * vocab;
    double maximum = -std::numeric_limits<double>::infinity();
    for (long long token = 0; token < vocab; ++token) {
        maximum = std::max(maximum,
                           static_cast<double>(values[base + token]) * multiplier);
    }
    double sum = 0.0;
    for (long long token = 0; token < vocab; ++token) {
        sum += std::exp(static_cast<double>(values[base + token]) * multiplier -
                        maximum);
    }
    return maximum + std::log(sum);
}

void kd_kl_dense_forward_ref(const std::vector<float> &teacher,
                             const std::vector<float> &student,
                             std::vector<float> &loss,
                             std::vector<float> &teacher_lse,
                             std::vector<float> &student_lse, long long rows,
                             long long vocab, float inverse_temperature) {
    loss.assign(static_cast<size_t>(rows), 0.0f);
    teacher_lse.assign(static_cast<size_t>(rows), 0.0f);
    student_lse.assign(static_cast<size_t>(rows), 0.0f);
    for (long long row = 0; row < rows; ++row) {
        const double tlse = row_lse(teacher, row, vocab, inverse_temperature);
        const double slse = row_lse(student, row, vocab, inverse_temperature);
        double value = 0.0;
        for (long long token = 0; token < vocab; ++token) {
            const long long index = row * vocab + token;
            const double teacher_log = teacher[index] * inverse_temperature - tlse;
            const double student_log = student[index] * inverse_temperature - slse;
            value += std::exp(teacher_log) * (teacher_log - student_log);
        }
        loss[row] = static_cast<float>(value);
        teacher_lse[row] = static_cast<float>(tlse);
        student_lse[row] = static_cast<float>(slse);
    }
}

std::vector<float> kd_kl_dense_backward_ref(
    const std::vector<float> &teacher, const std::vector<float> &student,
    const std::vector<float> &teacher_lse,
    const std::vector<float> &student_lse, const std::vector<float> &grad_out,
    long long rows, long long vocab, float inverse_temperature) {
    std::vector<float> grad(static_cast<size_t>(rows * vocab), 0.0f);
    for (long long row = 0; row < rows; ++row) {
        for (long long token = 0; token < vocab; ++token) {
            const long long index = row * vocab + token;
            const double q =
                std::exp(student[index] * inverse_temperature - student_lse[row]);
            const double p =
                std::exp(teacher[index] * inverse_temperature - teacher_lse[row]);
            grad[index] =
                static_cast<float>(grad_out[row] * inverse_temperature * (q - p));
        }
    }
    return grad;
}

void kd_kl_topk_forward_ref(const std::vector<float> &student,
                            const std::vector<int32_t> &teacher_indices,
                            const std::vector<float> &teacher_probabilities,
                            std::vector<float> &loss,
                            std::vector<float> &student_lse, long long rows,
                            long long vocab, long long top_k,
                            float inverse_temperature, bool include_tail) {
    loss.assign(static_cast<size_t>(rows), 0.0f);
    student_lse.assign(static_cast<size_t>(rows), 0.0f);
    for (long long row = 0; row < rows; ++row) {
        const double lse = row_lse(student, row, vocab, inverse_temperature);
        double probability_sum = 0.0;
        double student_selected = 0.0;
        for (long long item = 0; item < top_k; ++item) {
            const long long index = row * top_k + item;
            const int token = teacher_indices[index];
            if (token >= 0) {
                probability_sum += teacher_probabilities[index];
                student_selected += std::exp(student[row * vocab + token] *
                                                 inverse_temperature -
                                             lse);
            }
        }
        double value = 0.0;
        const double inverse_sum = 1.0 / std::max(probability_sum, double(kTiny));
        for (long long item = 0; item < top_k; ++item) {
            const long long index = row * top_k + item;
            const int token = teacher_indices[index];
            if (token < 0) continue;
            double p = teacher_probabilities[index];
            if (!include_tail) p *= inverse_sum;
            if (p > 0.0) {
                const double log_q = student[row * vocab + token] *
                                         inverse_temperature -
                                     lse;
                value += p * (std::log(std::max(p, double(kTiny))) - log_q);
            }
        }
        if (include_tail) {
            const double tail = std::max(1.0 - probability_sum, 0.0);
            if (tail > 0.0) {
                value += tail * (std::log(std::max(tail, double(kTiny))) -
                                 std::log(std::max(1.0 - student_selected,
                                                   double(kTiny))));
            }
        }
        loss[row] = static_cast<float>(value);
        student_lse[row] = static_cast<float>(lse);
    }
}

std::vector<float> kd_kl_topk_backward_ref(
    const std::vector<float> &student,
    const std::vector<int32_t> &teacher_indices,
    const std::vector<float> &teacher_probabilities,
    const std::vector<float> &student_lse, const std::vector<float> &grad_out,
    long long rows, long long vocab, long long top_k,
    float inverse_temperature, bool include_tail) {
    std::vector<float> grad(static_cast<size_t>(rows * vocab), 0.0f);
    for (long long row = 0; row < rows; ++row) {
        double probability_sum = 0.0;
        double student_selected = 0.0;
        for (long long item = 0; item < top_k; ++item) {
            const int token = teacher_indices[row * top_k + item];
            if (token >= 0) {
                probability_sum += teacher_probabilities[row * top_k + item];
                student_selected += std::exp(student[row * vocab + token] *
                                                 inverse_temperature -
                                             student_lse[row]);
            }
        }
        const double tail = std::max(1.0 - probability_sum, 0.0);
        const double tail_c = include_tail && tail > 0.0
                                  ? tail / std::max(1.0 - student_selected,
                                                    double(kTiny))
                                  : 0.0;
        const double q_coefficient =
            include_tail ? probability_sum - tail_c * student_selected : 1.0;
        const double inverse_sum = 1.0 / std::max(probability_sum, double(kTiny));
        const double go = grad_out[row] * inverse_temperature;
        for (long long token = 0; token < vocab; ++token) {
            const double q =
                std::exp(student[row * vocab + token] * inverse_temperature -
                         student_lse[row]);
            double gradient = q_coefficient * q * go;
            for (long long item = 0; item < top_k; ++item) {
                if (teacher_indices[row * top_k + item] != token) continue;
                const double p = teacher_probabilities[row * top_k + item];
                const double correction =
                    include_tail ? -p + tail_c * q : -p * inverse_sum;
                gradient += correction * go;
            }
            grad[row * vocab + token] = static_cast<float>(gradient);
        }
    }
    return grad;
}

void kd_ce_fused_forward_ref(
    const std::vector<float> &teacher, const std::vector<float> &student,
    const std::vector<int32_t> &targets, std::vector<float> &ce_loss,
    std::vector<float> &kd_loss, std::vector<float> &raw_student_lse,
    std::vector<float> &tempered_student_lse, std::vector<float> &teacher_lse,
    long long rows, long long vocab, float inverse_temperature,
    int ignore_index) {
    kd_kl_dense_forward_ref(teacher, student, kd_loss, teacher_lse,
                            tempered_student_lse, rows, vocab,
                            inverse_temperature);
    ce_loss.assign(static_cast<size_t>(rows), 0.0f);
    raw_student_lse.assign(static_cast<size_t>(rows), 0.0f);
    for (long long row = 0; row < rows; ++row) {
        const double lse = row_lse(student, row, vocab, 1.0f);
        raw_student_lse[row] = static_cast<float>(lse);
        const int target = targets[row];
        ce_loss[row] =
            target == ignore_index ? 0.0f
                                   : static_cast<float>(lse -
                                                        student[row * vocab + target]);
    }
}

std::vector<float> kd_ce_fused_backward_ref(
    const std::vector<float> &teacher, const std::vector<float> &student,
    const std::vector<int32_t> &targets,
    const std::vector<float> &raw_student_lse,
    const std::vector<float> &tempered_student_lse,
    const std::vector<float> &teacher_lse,
    const std::vector<float> &grad_ce, const std::vector<float> &grad_kd,
    long long rows, long long vocab, float inverse_temperature,
    int ignore_index) {
    std::vector<float> grad(static_cast<size_t>(rows * vocab), 0.0f);
    for (long long row = 0; row < rows; ++row) {
        const int target = targets[row];
        for (long long token = 0; token < vocab; ++token) {
            const long long index = row * vocab + token;
            const double student_probability =
                std::exp(student[index] * inverse_temperature -
                         tempered_student_lse[row]);
            const double teacher_probability =
                std::exp(teacher[index] * inverse_temperature - teacher_lse[row]);
            double gradient = grad_kd[row] * inverse_temperature *
                              (student_probability - teacher_probability);
            if (target != ignore_index) {
                gradient += grad_ce[row] *
                            (std::exp(student[index] - raw_student_lse[row]) -
                             (token == target ? 1.0 : 0.0));
            }
            grad[index] = static_cast<float>(gradient);
        }
    }
    return grad;
}

void adamw_masked_ref(std::vector<float> &parameters,
                      const std::vector<float> &gradients,
                      std::vector<float> &first_moment,
                      std::vector<float> &second_moment,
                      const std::vector<uint8_t> &mask, long long segment_size,
                      int mask_mode, float lr, float beta1, float beta2,
                      float eps, float weight_decay, long long step) {
    const double first_correction = 1.0 - std::pow(beta1, step);
    const double second_correction = 1.0 - std::pow(beta2, step);
    for (size_t index = 0; index < parameters.size(); ++index) {
        const bool active = mask[index / segment_size] != 0;
        if (!active && mask_mode == 0) continue;
        const float gradient = gradients[index];
        first_moment[index] = beta1 * first_moment[index] + (1.0f - beta1) * gradient;
        second_moment[index] =
            beta2 * second_moment[index] + (1.0f - beta2) * gradient * gradient;
        const double corrected_first = first_moment[index] / first_correction;
        const double corrected_second = second_moment[index] / second_correction;
        const double update =
            corrected_first / (std::sqrt(corrected_second) + eps) +
            (active ? weight_decay * parameters[index] : 0.0f);
        parameters[index] -= static_cast<float>(lr * update);
    }
}

std::vector<float> sgd_ref(std::vector<float> parameters,
                           const std::vector<float> &gradients, float lr,
                           float weight_decay) {
    for (size_t index = 0; index < parameters.size(); ++index) {
        parameters[index] -=
            lr * (gradients[index] + weight_decay * parameters[index]);
    }
    return parameters;
}

std::vector<float> softmax_rows(qc::Rng &rng, long long rows, long long dim) {
    std::vector<float> out(static_cast<size_t>(rows * dim), 0.0f);
    for (long long row = 0; row < rows; ++row) {
        std::vector<float> logits(static_cast<size_t>(dim));
        for (auto &x : logits) x = rng.uniform(-1.0f, 1.0f);
        const float max_value = *std::max_element(logits.begin(), logits.end());
        double sum = 0.0;
        for (float v : logits) sum += std::exp(v - max_value);
        for (long long i = 0; i < dim; ++i) {
            out[row * dim + i] = static_cast<float>(std::exp(logits[i] - max_value) / sum);
        }
    }
    return out;
}

std::vector<float> softmax_backward_ref(const std::vector<float> &grad_out,
                                        const std::vector<float> &softmax_output,
                                        long long rows, long long dim) {
    std::vector<float> grad(static_cast<size_t>(rows * dim), 0.0f);
    for (long long row = 0; row < rows; ++row) {
        double dot = 0.0;
        for (long long i = 0; i < dim; ++i) {
            dot += static_cast<double>(grad_out[row * dim + i]) *
                   softmax_output[row * dim + i];
        }
        for (long long i = 0; i < dim; ++i) {
            grad[row * dim + i] =
                softmax_output[row * dim + i] *
                (grad_out[row * dim + i] - static_cast<float>(dot));
        }
    }
    return grad;
}

std::vector<float> silu_backward_ref(const std::vector<float> &grad_out,
                                     const std::vector<float> &x) {
    std::vector<float> grad(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        const float probability = sigmoid_host(x[i]);
        grad[i] =
            grad_out[i] * probability * (1.0f + x[i] * (1.0f - probability));
    }
    return grad;
}

std::vector<int32_t> make_targets(long long rows, long long vocab,
                                  int ignore_index) {
    std::vector<int32_t> targets(static_cast<size_t>(rows));
    for (long long row = 0; row < rows; ++row) {
        targets[row] = row % 5 == 0 ? ignore_index : static_cast<int32_t>((row * 17) % vocab);
    }
    return targets;
}

void make_topk(long long rows, long long vocab, long long top_k,
               std::vector<int32_t> &indices, std::vector<float> &probabilities) {
    indices.resize(static_cast<size_t>(rows * top_k));
    probabilities.resize(static_cast<size_t>(rows * top_k));
    for (long long row = 0; row < rows; ++row) {
        double sum = 0.0;
        for (long long item = 0; item < top_k; ++item) {
            indices[row * top_k + item] =
                item == top_k - 1 && row % 4 == 0
                    ? -1
                    : static_cast<int32_t>((row * 131 + item * 17) % vocab);
            probabilities[row * top_k + item] = 0.01f + 0.003f * float((item % 7) + 1);
            if (indices[row * top_k + item] >= 0) {
                sum += probabilities[row * top_k + item];
            }
        }
        for (long long item = 0; item < top_k; ++item) {
            if (indices[row * top_k + item] >= 0) {
                probabilities[row * top_k + item] =
                    static_cast<float>(0.72 * probabilities[row * top_k + item] / sum);
            }
        }
    }
}

template <typename Fn>
qc::Bench bench_per_launch(Fn &&fn, int warmups = 10, int iters = 50,
                           int repeats = 1) {
    qc::Bench b = qc::bench([&] {
        for (int repeat = 0; repeat < repeats; ++repeat) fn();
        QC_CHECK(hipGetLastError());
    }, warmups, iters);
    if (repeats > 1) {
        b.median_ms /= static_cast<double>(repeats);
        b.min_ms /= static_cast<double>(repeats);
        b.max_ms /= static_cast<double>(repeats);
        b.mean_ms /= static_cast<double>(repeats);
    }
    return b;
}

void report_pair(const char *label, const qc::Bench &baseline,
                 const qc::Bench &candidate, double metric,
                 bool compute = false) {
    if (compute) {
        baseline.report_compute(std::string(label) + " scalar", metric);
        candidate.report_compute(std::string(label) + " candidate", metric);
    } else {
        baseline.report_bandwidth(std::string(label) + " scalar", metric);
        candidate.report_bandwidth(std::string(label) + " candidate", metric);
    }
    qc::report_ab(label, baseline, candidate);
}

bool run_correctness() {
    bool ok = true;
    int checks = 0;
    qc::Rng rng(0xa10);

    {
        constexpr long long rows = 7;
        constexpr long long vocab = 73;
        constexpr float invtemp = 0.7f;
        auto teacher = rng.uniforms(static_cast<size_t>(rows * vocab), -2.0f, 2.0f);
        auto student = rng.uniforms(static_cast<size_t>(rows * vocab), -2.0f, 2.0f);
        std::vector<float> ref_loss, ref_tlse, ref_slse;
        kd_kl_dense_forward_ref(teacher, student, ref_loss, ref_tlse, ref_slse,
                                rows, vocab, invtemp);
        float *dt = qc::dnew(teacher);
        float *ds = qc::dnew(student);
        float *dl = qc::dzero<float>(rows);
        float *dtl = qc::dzero<float>(rows);
        float *dsl = qc::dzero<float>(rows);
        kd_kl_dense_fwd_kernel<<<rows, 256>>>(dt, ds, dl, dtl, dsl, rows,
                                              vocab, invtemp);
        QC_SYNC();
        ok &= qc::compare(qc::d2h(dl, rows), to_ref(ref_loss),
                          qc::Tol::fp32().with_elementwise(2e-4, 2e-5))
                  .report("kd_kl_dense_fwd loss");
        ++checks;
        ok &= qc::compare(qc::d2h(dtl, rows), to_ref(ref_tlse),
                          qc::Tol::fp32().with_elementwise(2e-5, 2e-6))
                  .report("kd_kl_dense_fwd teacher_lse");
        ++checks;
        ok &= qc::compare(qc::d2h(dsl, rows), to_ref(ref_slse),
                          qc::Tol::fp32().with_elementwise(2e-5, 2e-6))
                  .report("kd_kl_dense_fwd student_lse");
        ++checks;
        auto grad_out = rng.uniforms(static_cast<size_t>(rows), -0.5f, 0.5f);
        const auto ref_grad = kd_kl_dense_backward_ref(teacher, student,
                                                       ref_tlse, ref_slse,
                                                       grad_out, rows, vocab,
                                                       invtemp);
        float *dgo = qc::dnew(grad_out);
        float *dgrad = qc::dzero<float>(ref_grad.size());
        kd_kl_dense_bwd_kernel<<<qc::grid_for(ref_grad.size(), 256), 256>>>(
            dt, ds, dtl, dsl, dgo, dgrad, ref_grad.size(), vocab, invtemp);
        QC_SYNC();
        ok &= qc::compare(qc::d2h(dgrad, ref_grad.size()), to_ref(ref_grad),
                          qc::Tol::fp32().with_elementwise(2e-4, 2e-5))
                  .report("kd_kl_dense_bwd grad_student");
        ++checks;
        qc::dfree(dt, ds, dl, dtl, dsl, dgo, dgrad);
    }

    {
        constexpr long long rows = 5;
        constexpr long long vocab = 67;
        constexpr long long top_k = 9;
        constexpr float invtemp = 0.8f;
        constexpr bool include_tail = true;
        auto student = rng.uniforms(static_cast<size_t>(rows * vocab), -2.0f, 2.0f);
        std::vector<int32_t> idx;
        std::vector<float> prob;
        make_topk(rows, vocab, top_k, idx, prob);
        std::vector<float> ref_loss, ref_lse;
        kd_kl_topk_forward_ref(student, idx, prob, ref_loss, ref_lse, rows,
                               vocab, top_k, invtemp, include_tail);
        float *ds = qc::dnew(student);
        int32_t *di = qc::dnew(idx);
        float *dp = qc::dnew(prob);
        float *dl = qc::dzero<float>(rows);
        float *dlse = qc::dzero<float>(rows);
        kd_kl_topk_fwd_kernel<<<rows, 256>>>(ds, di, dp, dl, dlse, rows, vocab,
                                             top_k, invtemp, include_tail);
        QC_SYNC();
        ok &= qc::compare(qc::d2h(dl, rows), to_ref(ref_loss),
                          qc::Tol::fp32().with_elementwise(3e-4, 3e-5))
                  .report("kd_kl_topk_fwd loss");
        ++checks;
        ok &= qc::compare(qc::d2h(dlse, rows), to_ref(ref_lse),
                          qc::Tol::fp32().with_elementwise(2e-5, 2e-6))
                  .report("kd_kl_topk_fwd student_lse");
        ++checks;
        auto grad_out = rng.uniforms(static_cast<size_t>(rows), -0.5f, 0.5f);
        const auto ref_grad = kd_kl_topk_backward_ref(student, idx, prob,
                                                      ref_lse, grad_out, rows,
                                                      vocab, top_k, invtemp,
                                                      include_tail);
        float *dgo = qc::dnew(grad_out);
        float *dg = qc::dzero<float>(ref_grad.size());
        kd_kl_topk_bwd_kernel<<<rows, 256>>>(ds, di, dp, dlse, dgo, dg, rows,
                                             vocab, top_k, invtemp,
                                             include_tail);
        QC_SYNC();
        ok &= qc::compare(qc::d2h(dg, ref_grad.size()), to_ref(ref_grad),
                          qc::Tol::fp32().with_elementwise(3e-4, 3e-5))
                  .report("kd_kl_topk_bwd grad_student");
        ++checks;
        qc::dfree(ds, di, dp, dl, dlse, dgo, dg);
    }

    {
        constexpr long long rows = 6;
        constexpr long long vocab = 71;
        constexpr float invtemp = 0.75f;
        constexpr int ignore_index = -100;
        auto teacher = rng.uniforms(static_cast<size_t>(rows * vocab), -2.0f, 2.0f);
        auto student = rng.uniforms(static_cast<size_t>(rows * vocab), -2.0f, 2.0f);
        auto targets = make_targets(rows, vocab, ignore_index);
        std::vector<float> ref_ce, ref_kd, ref_raw, ref_st, ref_t;
        kd_ce_fused_forward_ref(teacher, student, targets, ref_ce, ref_kd,
                                ref_raw, ref_st, ref_t, rows, vocab, invtemp,
                                ignore_index);
        float *dt = qc::dnew(teacher);
        float *ds = qc::dnew(student);
        int32_t *dta = qc::dnew(targets);
        float *dce = qc::dzero<float>(rows);
        float *dkd = qc::dzero<float>(rows);
        float *draw = qc::dzero<float>(rows);
        float *dst = qc::dzero<float>(rows);
        float *dtl = qc::dzero<float>(rows);
        kd_ce_fused_fwd_kernel<<<rows, 256>>>(dt, ds, dta, dce, dkd, draw, dst,
                                              dtl, rows, vocab, invtemp,
                                              ignore_index);
        QC_SYNC();
        ok &= qc::compare(qc::d2h(dce, rows), to_ref(ref_ce),
                          qc::Tol::fp32().with_elementwise(2e-4, 2e-5))
                  .report("kd_ce_fused_fwd ce");
        ++checks;
        ok &= qc::compare(qc::d2h(dkd, rows), to_ref(ref_kd),
                          qc::Tol::fp32().with_elementwise(2e-4, 2e-5))
                  .report("kd_ce_fused_fwd kd");
        ++checks;
        ok &= qc::compare(qc::d2h(draw, rows), to_ref(ref_raw),
                          qc::Tol::fp32().with_elementwise(2e-5, 2e-6))
                  .report("kd_ce_fused_fwd raw_lse");
        ++checks;
        ok &= qc::compare(qc::d2h(dst, rows), to_ref(ref_st),
                          qc::Tol::fp32().with_elementwise(2e-5, 2e-6))
                  .report("kd_ce_fused_fwd student_temp_lse");
        ++checks;
        ok &= qc::compare(qc::d2h(dtl, rows), to_ref(ref_t),
                          qc::Tol::fp32().with_elementwise(2e-5, 2e-6))
                  .report("kd_ce_fused_fwd teacher_lse");
        ++checks;
        auto grad_ce = rng.uniforms(static_cast<size_t>(rows), -0.5f, 0.5f);
        auto grad_kd = rng.uniforms(static_cast<size_t>(rows), -0.5f, 0.5f);
        const auto ref_grad = kd_ce_fused_backward_ref(
            teacher, student, targets, ref_raw, ref_st, ref_t, grad_ce, grad_kd,
            rows, vocab, invtemp, ignore_index);
        float *dgce = qc::dnew(grad_ce);
        float *dgkd = qc::dnew(grad_kd);
        float *dgrad = qc::dzero<float>(ref_grad.size());
        kd_ce_fused_bwd_kernel<<<qc::grid_for(ref_grad.size(), 256), 256>>>(
            dt, ds, dta, draw, dst, dtl, dgce, dgkd, dgrad, ref_grad.size(),
            vocab, invtemp, ignore_index);
        QC_SYNC();
        ok &= qc::compare(qc::d2h(dgrad, ref_grad.size()), to_ref(ref_grad),
                          qc::Tol::fp32().with_elementwise(2e-4, 2e-5))
                  .report("kd_ce_fused_bwd grad_student");
        ++checks;
        qc::dfree(dt, ds, dta, dce, dkd, draw, dst, dtl, dgce, dgkd, dgrad);
    }

    {
        constexpr long long count = 4099;
        constexpr long long segment = 17;
        auto params0 = rng.uniforms(static_cast<size_t>(count), -0.5f, 0.5f);
        auto params1 = params0;
        auto gradients = rng.uniforms(static_cast<size_t>(count), -0.2f, 0.2f);
        auto m0 = rng.uniforms(static_cast<size_t>(count), -0.05f, 0.05f);
        auto v0 = rng.uniforms(static_cast<size_t>(count), 0.0f, 0.05f);
        auto m1 = m0;
        auto v1 = v0;
        const long long segments = (count + segment - 1) / segment;
        std::vector<uint8_t> mask(static_cast<size_t>(segments), 0);
        for (long long i = 0; i < segments; ++i) mask[i] = (i % 3) != 0;
        adamw_masked_ref(params1, gradients, m1, v1, mask, segment, 1, 0.002f,
                         0.9f, 0.98f, 1e-6f, 0.01f, 7);
        float *dp = qc::dnew(params0);
        float *dg = qc::dnew(gradients);
        float *dm = qc::dnew(m0);
        float *dv = qc::dnew(v0);
        uint8_t *dmask = qc::dnew(mask);
        const double bc1 = 1.0 - std::pow(0.9, 7);
        const double bc2 = 1.0 - std::pow(0.98, 7);
        adamw_masked_kernel<<<qc::grid_for(static_cast<size_t>(count), 256), 256>>>(
            dp, dg, dm, dv, dmask, count, segment, 1, 0.002f, 0.9f, 0.98f,
            1e-6f, 0.01f, bc1, bc2);
        QC_SYNC();
        ok &= qc::compare(qc::d2h(dp, count), to_ref(params1),
                          qc::Tol::fp32().with_elementwise(2e-5, 2e-6))
                  .report("adamw_masked mode1 params");
        ++checks;
        ok &= qc::compare(qc::d2h(dm, count), to_ref(m1), qc::Tol::fp32())
                  .report("adamw_masked mode1 first_moment");
        ++checks;
        ok &= qc::compare(qc::d2h(dv, count), to_ref(v1), qc::Tol::fp32())
                  .report("adamw_masked mode1 second_moment");
        ++checks;
        qc::dfree(dp, dg, dm, dv, dmask);
    }

    {
        constexpr long long count = 4099;
        constexpr long long segment = 17;
        auto params0 = rng.uniforms(static_cast<size_t>(count), -0.5f, 0.5f);
        auto params1 = params0;
        auto gradients = rng.uniforms(static_cast<size_t>(count), -0.2f, 0.2f);
        auto m0 = rng.uniforms(static_cast<size_t>(count), -0.05f, 0.05f);
        auto v0 = rng.uniforms(static_cast<size_t>(count), 0.0f, 0.05f);
        auto m1 = m0;
        auto v1 = v0;
        const long long segments = (count + segment - 1) / segment;
        std::vector<uint8_t> mask(static_cast<size_t>(segments), 0);
        for (long long i = 0; i < segments; ++i) mask[i] = (i % 3) != 0;
        adamw_masked_ref(params1, gradients, m1, v1, mask, segment, 0, 0.002f,
                         0.9f, 0.98f, 1e-6f, 0.01f, 7);
        float *dp = qc::dnew(params0);
        float *dg = qc::dnew(gradients);
        float *dm = qc::dnew(m0);
        float *dv = qc::dnew(v0);
        uint8_t *dmask = qc::dnew(mask);
        const double bc1 = 1.0 - std::pow(0.9, 7);
        const double bc2 = 1.0 - std::pow(0.98, 7);
        adamw_masked_kernel<<<qc::grid_for(static_cast<size_t>(count), 256), 256>>>(
            dp, dg, dm, dv, dmask, count, segment, 0, 0.002f, 0.9f, 0.98f,
            1e-6f, 0.01f, bc1, bc2);
        QC_SYNC();
        ok &= qc::compare(qc::d2h(dp, count), to_ref(params1),
                          qc::Tol::fp32().with_elementwise(2e-5, 2e-6))
                  .report("adamw_masked mode0 params");
        ++checks;
        ok &= qc::compare(qc::d2h(dm, count), to_ref(m1), qc::Tol::fp32())
                  .report("adamw_masked mode0 first_moment");
        ++checks;
        ok &= qc::compare(qc::d2h(dv, count), to_ref(v1), qc::Tol::fp32())
                  .report("adamw_masked mode0 second_moment");
        ++checks;
        qc::dfree(dp, dg, dm, dv, dmask);
    }

    {
        constexpr long long count = 4099;
        auto params = rng.uniforms(static_cast<size_t>(count), -0.5f, 0.5f);
        auto gradients = rng.uniforms(static_cast<size_t>(count), -0.2f, 0.2f);
        const auto ref = sgd_ref(params, gradients, 0.01f, 0.1f);
        float *dp = qc::dnew(params);
        float *dg = qc::dnew(gradients);
        sgd_kernel<<<qc::grid_for(static_cast<size_t>(count), 256), 256>>>(
            dp, dg, count, 0.01f, 0.1f);
        QC_SYNC();
        ok &= qc::compare(qc::d2h(dp, count), to_ref(ref), qc::Tol::fp32())
                  .report("sgd params");
        ++checks;
        qc::dfree(dp, dg);
    }

    {
        constexpr long long rows = 11;
        constexpr long long dim = 97;
        auto softmax = softmax_rows(rng, rows, dim);
        auto grad_out = rng.uniforms(static_cast<size_t>(rows * dim), -0.5f, 0.5f);
        const auto ref = softmax_backward_ref(grad_out, softmax, rows, dim);
        float *dgo = qc::dnew(grad_out);
        float *dsm = qc::dnew(softmax);
        float *dgi = qc::dzero<float>(ref.size());
        softmax_backward_kernel<<<rows, 256>>>(dgo, dsm, dgi, rows, dim);
        QC_SYNC();
        ok &= qc::compare(qc::d2h(dgi, ref.size()), to_ref(ref),
                          qc::Tol::fp32().with_elementwise(2e-5, 2e-6))
                  .report("softmax_backward");
        ++checks;
        qc::dfree(dgo, dsm, dgi);
    }

    {
        constexpr long long count = 4099;
        auto x = rng.uniforms(static_cast<size_t>(count), -6.0f, 6.0f);
        auto grad_out = rng.uniforms(static_cast<size_t>(count), -0.5f, 0.5f);
        const auto ref = silu_backward_ref(grad_out, x);
        float *dx = qc::dnew(x);
        float *dgo = qc::dnew(grad_out);
        float *dgi = qc::dzero<float>(ref.size());
        silu_backward_kernel<<<qc::grid_for(static_cast<size_t>(count), 256), 256>>>(
            dgo, dx, dgi, count);
        QC_SYNC();
        ok &= qc::compare(qc::d2h(dgi, ref.size()), to_ref(ref),
                          qc::Tol::fp32().with_elementwise(2e-5, 2e-6))
                  .report("silu_backward");
        ++checks;
        qc::dfree(dx, dgo, dgi);
    }

    std::printf("Phase 10 correctness checks: %d\n", checks);
    return ok;
}

void run_benchmarks() {
    std::printf("\n== Phase 10 benchmarks ==\n");
    std::printf("   Timing note: medians are per launch; fast kernels use inner repeats.\n");
    qc::Rng rng(0xa1010);

    {
        constexpr long long rows = 1024;
        constexpr long long vocab = 2048;
        constexpr float invtemp = 0.7f;
        const size_t count = static_cast<size_t>(rows * vocab);
        auto teacher = rng.uniforms(count, -2.0f, 2.0f);
        auto student = rng.uniforms(count, -2.0f, 2.0f);
        float *dt = qc::dnew(teacher);
        float *ds = qc::dnew(student);
        float *dl = qc::dzero<float>(rows);
        float *dtl = qc::dzero<float>(rows);
        float *dsl = qc::dzero<float>(rows);
        const double bytes = double(count * 2 + rows * 3) * sizeof(float);
        const auto scalar = bench_per_launch([&] {
            kd_kl_dense_fwd_scalar_kernel<<<rows, 1>>>(dt, ds, dl, dtl, dsl,
                                                       rows, vocab, invtemp);
        }, 5, 20);
        const auto parallel = bench_per_launch([&] {
            kd_kl_dense_fwd_kernel<<<rows, 256>>>(dt, ds, dl, dtl, dsl, rows,
                                                  vocab, invtemp);
        }, 10, 40, 128);
        report_pair("kd_kl_dense_fwd", scalar, parallel, bytes);
        auto grad_out = rng.uniforms(static_cast<size_t>(rows), -0.5f, 0.5f);
        float *dgo = qc::dnew(grad_out);
        float *dg = qc::dzero<float>(count);
        const auto bwd_scalar = bench_per_launch([&] {
            kd_kl_dense_bwd_scalar_kernel<<<rows, 1>>>(dt, ds, dtl, dsl, dgo,
                                                       dg, rows, vocab,
                                                       invtemp);
        }, 5, 20);
        const auto bwd_parallel = bench_per_launch([&] {
            kd_kl_dense_bwd_kernel<<<qc::grid_for(count, 256), 256>>>(
                dt, ds, dtl, dsl, dgo, dg, count, vocab, invtemp);
        }, 10, 40, 128);
        report_pair("kd_kl_dense_bwd", bwd_scalar, bwd_parallel,
                    double(count * 3 + rows * 3) * sizeof(float));
        qc::dfree(dt, ds, dl, dtl, dsl, dgo, dg);
    }

    {
        constexpr long long rows = 2048;
        constexpr long long vocab = 4096;
        constexpr long long top_k = 16;
        constexpr float invtemp = 0.8f;
        constexpr bool include_tail = true;
        const size_t count = static_cast<size_t>(rows * vocab);
        auto student = rng.uniforms(count, -2.0f, 2.0f);
        std::vector<int32_t> idx;
        std::vector<float> prob;
        make_topk(rows, vocab, top_k, idx, prob);
        float *ds = qc::dnew(student);
        int32_t *di = qc::dnew(idx);
        float *dp = qc::dnew(prob);
        float *dl = qc::dzero<float>(rows);
        float *dlse = qc::dzero<float>(rows);
        const double bytes = double(count + idx.size() + prob.size() + rows * 2) * sizeof(float);
        const auto scalar = bench_per_launch([&] {
            kd_kl_topk_fwd_scalar_kernel<<<rows, 1>>>(ds, di, dp, dl, dlse,
                                                      rows, vocab, top_k,
                                                      invtemp, include_tail);
        }, 5, 20);
        const auto parallel = bench_per_launch([&] {
            kd_kl_topk_fwd_kernel<<<rows, 256>>>(ds, di, dp, dl, dlse, rows,
                                                 vocab, top_k, invtemp,
                                                 include_tail);
        }, 10, 40, 128);
        report_pair("kd_kl_topk_fwd", scalar, parallel, bytes);
        auto grad_out = rng.uniforms(static_cast<size_t>(rows), -0.5f, 0.5f);
        float *dgo = qc::dnew(grad_out);
        float *dg = qc::dzero<float>(count);
        const auto bwd_scalar = bench_per_launch([&] {
            kd_kl_topk_bwd_scalar_kernel<<<rows, 1>>>(
                ds, di, dp, dlse, dgo, dg, rows, vocab, top_k, invtemp,
                include_tail);
        }, 5, 20);
        const auto bwd_parallel = bench_per_launch([&] {
            kd_kl_topk_bwd_kernel<<<rows, 256>>>(ds, di, dp, dlse, dgo, dg,
                                                 rows, vocab, top_k, invtemp,
                                                 include_tail);
        }, 10, 40, 32);
        report_pair("kd_kl_topk_bwd", bwd_scalar, bwd_parallel,
                    double(count * 2 + idx.size() * 2 + prob.size() + rows * 2) *
                        sizeof(float));
        qc::dfree(ds, di, dp, dl, dlse, dgo, dg);
    }

    {
        constexpr long long rows = 1024;
        constexpr long long vocab = 2048;
        constexpr float invtemp = 0.75f;
        constexpr int ignore_index = -100;
        const size_t count = static_cast<size_t>(rows * vocab);
        auto teacher = rng.uniforms(count, -2.0f, 2.0f);
        auto student = rng.uniforms(count, -2.0f, 2.0f);
        auto targets = make_targets(rows, vocab, ignore_index);
        auto grad_ce = rng.uniforms(static_cast<size_t>(rows), -0.5f, 0.5f);
        auto grad_kd = rng.uniforms(static_cast<size_t>(rows), -0.5f, 0.5f);
        float *dt = qc::dnew(teacher);
        float *ds = qc::dnew(student);
        int32_t *dta = qc::dnew(targets);
        float *dce = qc::dzero<float>(rows);
        float *dkd = qc::dzero<float>(rows);
        float *draw = qc::dzero<float>(rows);
        float *dst = qc::dzero<float>(rows);
        float *dtl = qc::dzero<float>(rows);
        const double bytes = double(count * 2 + rows * 6) * sizeof(float);
        const auto fwd_scalar = bench_per_launch([&] {
            kd_ce_fused_fwd_scalar_kernel<<<rows, 1>>>(
                dt, ds, dta, dce, dkd, draw, dst, dtl, rows, vocab, invtemp,
                ignore_index);
        }, 3, 12);
        const auto fwd_parallel = bench_per_launch([&] {
            kd_ce_fused_fwd_kernel<<<rows, 256>>>(dt, ds, dta, dce, dkd, draw,
                                                  dst, dtl, rows, vocab,
                                                  invtemp, ignore_index);
        }, 10, 40, 128);
        report_pair("kd_ce_fused_fwd", fwd_scalar, fwd_parallel, bytes);
        float *dgce = qc::dnew(grad_ce);
        float *dgkd = qc::dnew(grad_kd);
        float *dg = qc::dzero<float>(count);
        const auto bwd_scalar = bench_per_launch([&] {
            kd_ce_fused_bwd_scalar_kernel<<<rows, 1>>>(
                dt, ds, dta, draw, dst, dtl, dgce, dgkd, dg, rows, vocab,
                invtemp, ignore_index);
        }, 10, 40);
        const auto bwd_parallel = bench_per_launch([&] {
            kd_ce_fused_bwd_kernel<<<qc::grid_for(count, 256), 256>>>(
                dt, ds, dta, draw, dst, dtl, dgce, dgkd, dg, count, vocab,
                invtemp, ignore_index);
        }, 10, 40, 128);
        report_pair("kd_ce_fused_bwd", bwd_scalar, bwd_parallel,
                    double(count * 3 + rows * 6) * sizeof(float));
        qc::dfree(dt, ds, dta, dce, dkd, draw, dst, dtl, dgce, dgkd, dg);
    }

    {
        constexpr long long count = 8 * 1024 * 1024;
        constexpr long long segment_size = 256;
        constexpr long long segments = (count + segment_size - 1) / segment_size;
        auto params = rng.uniforms(static_cast<size_t>(count), -0.5f, 0.5f);
        auto gradients = rng.uniforms(static_cast<size_t>(count), -0.2f, 0.2f);
        auto m = rng.uniforms(static_cast<size_t>(count), -0.05f, 0.05f);
        auto v = rng.uniforms(static_cast<size_t>(count), 0.0f, 0.05f);
        std::vector<uint8_t> mask(static_cast<size_t>(segments), 0);
        for (long long i = 0; i < segments; ++i) mask[i] = (i % 4) != 0;
        float *dp = qc::dnew(params);
        float *dg = qc::dnew(gradients);
        float *dm = qc::dnew(m);
        float *dv = qc::dnew(v);
        uint8_t *dmask = qc::dnew(mask);
        const double bc1 = 1.0 - std::pow(0.9, 11);
        const double bc2 = 1.0 - std::pow(0.98, 11);
        const double bytes = double(count * 4 + mask.size()) * sizeof(float);
        const auto scalar = bench_per_launch([&] {
            adamw_masked_segment_kernel<<<segments, 1>>>(
                dp, dg, dm, dv, dmask, count, segment_size, 1, 0.002f, 0.9f,
                0.98f, 1e-6f, 0.01f, bc1, bc2);
        }, 5, 20);
        const auto parallel = bench_per_launch([&] {
            adamw_masked_kernel<<<qc::grid_for(static_cast<size_t>(count), 256), 256>>>(
                dp, dg, dm, dv, dmask, count, segment_size, 1, 0.002f, 0.9f,
                0.98f, 1e-6f, 0.01f, bc1, bc2);
        }, 10, 40, 64);
        report_pair("adamw_masked", scalar, parallel, bytes);
        qc::dfree(dp, dg, dm, dv, dmask);
    }

    {
        constexpr long long count = 16 * 1024 * 1024;
        constexpr long long chunk = 512;
        auto params = rng.uniforms(static_cast<size_t>(count), -0.5f, 0.5f);
        auto gradients = rng.uniforms(static_cast<size_t>(count), -0.2f, 0.2f);
        float *dp = qc::dnew(params);
        float *dg = qc::dnew(gradients);
        const double bytes = double(count * 3) * sizeof(float);
        const auto scalar = bench_per_launch([&] {
            sgd_chunk_scalar_kernel<<<qc::grid_for(static_cast<size_t>(count), chunk), 1>>>(
                dp, dg, count, 0.01f, 0.1f, chunk);
        }, 5, 20);
        const auto parallel = bench_per_launch([&] {
            sgd_kernel<<<qc::grid_for(static_cast<size_t>(count), 256), 256>>>(
                dp, dg, count, 0.01f, 0.1f);
        }, 10, 40, 64);
        report_pair("sgd", scalar, parallel, bytes);
        qc::dfree(dp, dg);
    }

    {
        constexpr long long rows = 32768;
        constexpr long long dim = 256;
        auto softmax = softmax_rows(rng, rows, dim);
        auto grad_out = rng.uniforms(static_cast<size_t>(rows * dim), -0.5f, 0.5f);
        float *dgo = qc::dnew(grad_out);
        float *dsm = qc::dnew(softmax);
        float *dgi = qc::dzero<float>(softmax.size());
        const double bytes = double(softmax.size() * 3) * sizeof(float);
        const auto scalar = bench_per_launch([&] {
            softmax_backward_scalar_kernel<<<rows, 1>>>(dgo, dsm, dgi, rows,
                                                        dim);
        }, 5, 20);
        const auto parallel = bench_per_launch([&] {
            softmax_backward_kernel<<<rows, 256>>>(dgo, dsm, dgi, rows, dim);
        }, 10, 40, 64);
        report_pair("softmax_backward", scalar, parallel, bytes);
        qc::dfree(dgo, dsm, dgi);
    }

    {
        constexpr long long count = 16 * 1024 * 1024;
        constexpr long long chunk = 512;
        auto x = rng.uniforms(static_cast<size_t>(count), -6.0f, 6.0f);
        auto grad_out = rng.uniforms(static_cast<size_t>(count), -0.5f, 0.5f);
        float *dx = qc::dnew(x);
        float *dgo = qc::dnew(grad_out);
        float *dgi = qc::dzero<float>(count);
        const double bytes = double(count * 3) * sizeof(float);
        const auto scalar = bench_per_launch([&] {
            silu_backward_chunk_scalar_kernel<<<qc::grid_for(static_cast<size_t>(count), chunk), 1>>>(
                dgo, dx, dgi, count, chunk);
        }, 5, 20);
        const auto parallel = bench_per_launch([&] {
            silu_backward_kernel<<<qc::grid_for(static_cast<size_t>(count), 256), 256>>>(
                dgo, dx, dgi, count);
        }, 10, 40, 64);
        report_pair("silu_backward", scalar, parallel, bytes);
        qc::dfree(dx, dgo, dgi);
    }
}

}  // namespace

int main(int argc, char **argv) {
    qc::print_environment("phase10_training");
    const bool ok = run_correctness();
    if (qc::bench_requested(argc, argv)) run_benchmarks();
    return qc::finish(ok);
}
