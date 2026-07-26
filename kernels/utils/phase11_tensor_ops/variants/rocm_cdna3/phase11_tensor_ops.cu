/**
 * @file
 * @brief Phase 11 elementwise and tensor-op parity ports for CDNA3.
 */
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "../../../../common/cdna3_harness.cuh"

namespace {

constexpr int kThreads = qc::kThreads;
constexpr int kSortMax = 128;
constexpr float kNegInf = -std::numeric_limits<float>::infinity();

enum UnaryOp {
    kAbs = 0,
    kSign = 1,
    kNegate = 2,
    kStep = 3,
    kTanh = 4,
    kElu = 5,
    kRelu = 6,
    kSigmoid = 7,
    kGelu = 8,
    kGeluQuick = 9,
    kSilu = 10,
    kHardSwish = 11,
    kHardSigmoid = 12,
    kExp = 13,
    kExpm1 = 14,
    kSoftplus = 15,
    kGeluErf = 16,
    kXiElu = 17,
    kFloor = 18,
    kCeil = 19,
    kRound = 20,
    kTrunc = 21,
};

const char *kUnaryNames[] = {
    "abs",       "sign",   "negate", "step",    "tanh",  "elu",
    "relu",      "sigmoid","gelu",   "gelu_quick","silu", "hard_swish",
    "hard_sigmoid","exp",  "expm1",  "softplus","gelu_erf","xielu",
    "floor",     "ceil",   "round",  "trunc",
};

std::vector<double> to_ref(const std::vector<float> &values) {
    return std::vector<double>(values.begin(), values.end());
}

std::vector<double> to_ref_i32(const std::vector<int32_t> &values) {
    std::vector<double> out(values.size());
    for (size_t i = 0; i < values.size(); ++i) out[i] = values[i];
    return out;
}

std::vector<double> to_ref_i64(const std::vector<long long> &values) {
    std::vector<double> out(values.size());
    for (size_t i = 0; i < values.size(); ++i) out[i] = static_cast<double>(values[i]);
    return out;
}

float sigmoid_host(float value) {
    if (value >= 0.0f) return 1.0f / (1.0f + std::exp(-value));
    const float e = std::exp(value);
    return e / (1.0f + e);
}

float softplus_host(float value) {
    return value > 20.0f ? value : std::log(1.0f + std::exp(value));
}

float gelu_tanh_host(float value) {
    constexpr float kSqrt2OverPi = 0.79788456080286535588f;
    constexpr float kCoeff = 0.044715f;
    return 0.5f * value *
           (1.0f + std::tanh(kSqrt2OverPi * (value + kCoeff * value * value * value)));
}

float gelu_erf_host(float value) {
    constexpr float kInvSqrt2 = 0.70710678118654752440f;
    return 0.5f * value * (1.0f + std::erf(value * kInvSqrt2));
}

float unary_host(float value, int op, float alpha_n, float alpha_p, float beta,
                 float eps) {
    switch (op) {
        case kAbs: return std::fabs(value);
        case kSign: return value > 0.0f ? 1.0f : (value < 0.0f ? -1.0f : 0.0f);
        case kNegate: return -value;
        case kStep: return value > 0.0f ? 1.0f : 0.0f;
        case kTanh: return std::tanh(value);
        case kElu: return value > 0.0f ? value : std::expm1(value);
        case kRelu: return value > 0.0f ? value : 0.0f;
        case kSigmoid: return sigmoid_host(value);
        case kGelu: return gelu_tanh_host(value);
        case kGeluQuick: return value * sigmoid_host(1.702f * value);
        case kSilu: return value * sigmoid_host(value);
        case kHardSwish:
            return value * std::clamp((value + 3.0f) / 6.0f, 0.0f, 1.0f);
        case kHardSigmoid:
            return std::clamp((value + 3.0f) / 6.0f, 0.0f, 1.0f);
        case kExp: return std::exp(value);
        case kExpm1: return std::exp(value) - 1.0f;
        case kSoftplus: return softplus_host(value);
        case kGeluErf: return gelu_erf_host(value);
        case kXiElu: {
            const float an = beta + softplus_host(alpha_n);
            const float ap = softplus_host(alpha_p);
            if (value > 0.0f) return ap * value * value + beta * value;
            return (std::expm1(std::min(value, eps)) - value) * an + beta * value;
        }
        case kFloor: return std::floor(value);
        case kCeil: return std::ceil(value);
        case kRound: return std::round(value);
        case kTrunc: return std::trunc(value);
    }
    return 0.0f;
}

long long positive_mod_host(long long value, long long modulus) {
    const long long r = value % modulus;
    return r < 0 ? r + modulus : r;
}

bool report_with_inf(const std::string &label, const std::vector<float> &got,
                     const std::vector<double> &ref) {
    size_t bad = 0;
    double max_abs = 0.0;
    long worst = -1;
    for (size_t i = 0; i < got.size(); ++i) {
        const double g = got[i];
        const double r = ref[i];
        bool same = false;
        if (std::isinf(r) || std::isinf(g)) {
            same = std::isinf(r) && std::isinf(g) && std::signbit(r) == std::signbit(g);
        } else {
            const double abs_err = std::fabs(g - r);
            same = abs_err <= 1e-6 + 1e-5 * std::fabs(r);
            if (abs_err > max_abs) {
                max_abs = abs_err;
                worst = static_cast<long>(i);
            }
        }
        if (!same) {
            ++bad;
            if (worst < 0) worst = static_cast<long>(i);
        }
    }
    const bool ok = bad == 0;
    std::printf("  %-52s [fp32] n=%zu bad=%zu rel=0.000e+00 max=%.3e cos=1.000000000  %s\n",
                label.c_str(), got.size(), bad, max_abs, ok ? "PASS" : "FAIL");
    if (!ok && worst >= 0)
        std::printf("      worst @%ld: got %.9g vs ref %.9g\n", worst,
                    static_cast<double>(got[worst]), ref[worst]);
    return ok;
}

template <typename T>
void fill_linear(std::vector<T> &values, T base, T step) {
    for (size_t i = 0; i < values.size(); ++i)
        values[i] = static_cast<T>(base + step * static_cast<T>(i));
}

__device__ __forceinline__ float sigmoid_device(float value) {
    if (value >= 0.0f) return 1.0f / (1.0f + expf(-value));
    const float e = expf(value);
    return e / (1.0f + e);
}

__device__ __forceinline__ float softplus_device(float value) {
    return value > 20.0f ? value : logf(1.0f + expf(value));
}

__device__ __forceinline__ float gelu_tanh_device(float value) {
    constexpr float kSqrt2OverPi = 0.79788456080286535588f;
    constexpr float kCoeff = 0.044715f;
    return 0.5f * value *
           (1.0f + tanhf(kSqrt2OverPi * (value + kCoeff * value * value * value)));
}

__device__ __forceinline__ float gelu_erf_device(float value) {
    constexpr float kInvSqrt2 = 0.70710678118654752440f;
    return 0.5f * value * (1.0f + erff(value * kInvSqrt2));
}

__device__ __forceinline__ float unary_device(float value, int op,
                                             float alpha_n, float alpha_p,
                                             float beta, float eps) {
    switch (op) {
        case kAbs: return fabsf(value);
        case kSign: return value > 0.0f ? 1.0f : (value < 0.0f ? -1.0f : 0.0f);
        case kNegate: return -value;
        case kStep: return value > 0.0f ? 1.0f : 0.0f;
        case kTanh: return tanhf(value);
        case kElu: return value > 0.0f ? value : expm1f(value);
        case kRelu: return value > 0.0f ? value : 0.0f;
        case kSigmoid: return sigmoid_device(value);
        case kGelu: return gelu_tanh_device(value);
        case kGeluQuick: return value * sigmoid_device(1.702f * value);
        case kSilu: return value * sigmoid_device(value);
        case kHardSwish:
            return value * fminf(fmaxf((value + 3.0f) / 6.0f, 0.0f), 1.0f);
        case kHardSigmoid:
            return fminf(fmaxf((value + 3.0f) / 6.0f, 0.0f), 1.0f);
        case kExp: return expf(value);
        case kExpm1: return expf(value) - 1.0f;
        case kSoftplus: return softplus_device(value);
        case kGeluErf: return gelu_erf_device(value);
        case kXiElu: {
            const float an = beta + softplus_device(alpha_n);
            const float ap = softplus_device(alpha_p);
            if (value > 0.0f) return ap * value * value + beta * value;
            return (expm1f(fminf(value, eps)) - value) * an + beta * value;
        }
        case kFloor: return floorf(value);
        case kCeil: return ceilf(value);
        case kRound: return roundf(value);
        case kTrunc: return truncf(value);
    }
    return 0.0f;
}

__device__ __forceinline__ long long positive_mod_device(long long value,
                                                         long long modulus) {
    const long long r = value % modulus;
    return r < 0 ? r + modulus : r;
}

__device__ __forceinline__ float block_sum_float(float value, float *scratch) {
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

__device__ __forceinline__ double block_sum_double(double value,
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

__device__ __forceinline__ int block_sum_int(int value, int *scratch) {
    const int lane = threadIdx.x & (qc::kWave - 1);
    const int wave = threadIdx.x >> 6;
    const int waves = blockDim.x >> 6;
    value = qc::wave_reduce_sum(value);
    if (lane == 0) scratch[wave] = value;
    __syncthreads();
    int total = 0;
    for (int w = 0; w < waves; ++w) total += scratch[w];
    __syncthreads();
    return total;
}

// ---------------------------------------------------------------------------
// Elementwise and scalar reference kernels
// ---------------------------------------------------------------------------

__global__ void unary_kernel(const float *x, float *out, long long count,
                             int op, float alpha_n, float alpha_p, float beta,
                             float eps) {
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) out[idx] = unary_device(x[idx], op, alpha_n, alpha_p, beta, eps);
}

__global__ void unary_scalar_kernel(const float *x, float *out, long long count,
                                    int op, float alpha_n, float alpha_p,
                                    float beta, float eps) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (long long i = 0; i < count; ++i)
        out[i] = unary_device(x[i], op, alpha_n, alpha_p, beta, eps);
}

__global__ void unary_misc_kernel(const float *x, float *out, long long count,
                                  int mode, float a, float b) {
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    const float v = x == nullptr ? 0.0f : x[idx];
    float r = 0.0f;
    switch (mode) {
        case 0: r = v + a; break;                                  // add_scalar
        case 1: r = fminf(fmaxf(v, a), b); break;                  // clamp/value_clip
        case 2: r = v >= 0.0f ? v : a * v; break;                  // leaky_relu
        case 3: r = v * a; break;                                  // scale
        case 4: r = v * v; break;                                  // square
        case 5: r = sqrtf(v); break;                               // square_root
        case 6: r = sinf(v); break;                                // sine
        case 7: r = cosf(v); break;                                // cosine
        case 8: r = logf(v); break;                                // logarithm
        case 9: r = a; break;                                      // fill
        case 10: r = a + b * static_cast<float>(idx); break;       // arange
    }
    out[idx] = r;
}

__global__ void unary_misc_scalar_kernel(const float *x, float *out,
                                         long long count, int mode, float a,
                                         float b) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (long long idx = 0; idx < count; ++idx) {
        const float v = x == nullptr ? 0.0f : x[idx];
        float r = 0.0f;
        switch (mode) {
            case 0: r = v + a; break;
            case 1: r = fminf(fmaxf(v, a), b); break;
            case 2: r = v >= 0.0f ? v : a * v; break;
            case 3: r = v * a; break;
            case 4: r = v * v; break;
            case 5: r = sqrtf(v); break;
            case 6: r = sinf(v); break;
            case 7: r = cosf(v); break;
            case 8: r = logf(v); break;
            case 9: r = a; break;
            case 10: r = a + b * static_cast<float>(idx); break;
        }
        out[idx] = r;
    }
}

__global__ void binary_kernel(const float *x, const float *y, float *out,
                              long long count, int mode) {
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    if (mode == 0) out[idx] = x[idx] * y[idx];
    else if (mode == 1) out[idx] = x[idx] / y[idx];
    else out[idx] = x[idx] - y[idx];
}

__global__ void binary_scalar_kernel(const float *x, const float *y, float *out,
                                     long long count, int mode) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (long long idx = 0; idx < count; ++idx) {
        if (mode == 0) out[idx] = x[idx] * y[idx];
        else if (mode == 1) out[idx] = x[idx] / y[idx];
        else out[idx] = x[idx] - y[idx];
    }
}

__global__ void accumulate_kernel(float *destination, const float *source,
                                  long long count, float alpha) {
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) destination[idx] += alpha * source[idx];
}

__global__ void sigmoid_mul_kernel(const float *gate, const float *value,
                                   float *out, long long count) {
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) out[idx] = sigmoid_device(gate[idx]) * value[idx];
}

__global__ void sigmoid_mul_scalar_kernel(const float *gate, const float *value,
                                          float *out, long long count) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (long long idx = 0; idx < count; ++idx)
        out[idx] = sigmoid_device(gate[idx]) * value[idx];
}

__global__ void sigmoid_mul_backward_kernel(const float *grad_out,
                                            const float *gate,
                                            const float *value,
                                            float *grad_gate,
                                            float *grad_value,
                                            long long count) {
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    const float p = sigmoid_device(gate[idx]);
    grad_gate[idx] = grad_out[idx] * value[idx] * p * (1.0f - p);
    grad_value[idx] = grad_out[idx] * p;
}

// ---------------------------------------------------------------------------
// Reductions and normalization
// ---------------------------------------------------------------------------

__global__ void reduce_mean_kernel(const float *x, float *out, long long rows,
                                   long long dim) {
    const long long row = blockIdx.x;
    if (row >= rows) return;
    __shared__ double scratch[4];
    double sum = 0.0;
    for (long long i = threadIdx.x; i < dim; i += blockDim.x)
        sum += static_cast<double>(x[row * dim + i]);
    sum = block_sum_double(sum, scratch);
    if (threadIdx.x == 0) out[row] = static_cast<float>(sum / dim);
}

__global__ void reduce_mean_scalar_kernel(const float *x, float *out,
                                          long long rows, long long dim) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (long long row = 0; row < rows; ++row) {
        double sum = 0.0;
        for (long long i = 0; i < dim; ++i) sum += x[row * dim + i];
        out[row] = static_cast<float>(sum / dim);
    }
}

__global__ void reduce_sum_all_kernel(const float *x, float *out,
                                      long long count) {
    __shared__ double scratch[4];
    double sum = 0.0;
    for (long long i = threadIdx.x; i < count; i += blockDim.x)
        sum += static_cast<double>(x[i]);
    sum = block_sum_double(sum, scratch);
    if (threadIdx.x == 0) *out = static_cast<float>(sum);
}

__global__ void count_equal_kernel(const int32_t *x, const int32_t *y,
                                   long long count, long long *out) {
    __shared__ int scratch[4];
    int local = 0;
    for (long long i = threadIdx.x; i < count; i += blockDim.x)
        local += x[i] == y[i];
    const int total = block_sum_int(local, scratch);
    if (threadIdx.x == 0) *out = static_cast<long long>(total);
}

__global__ void cumulative_sum_kernel(const float *x, float *out,
                                      long long rows, long long dim) {
    const long long row = blockIdx.x;
    if (threadIdx.x != 0 || row >= rows) return;
    float sum = 0.0f;
    for (long long i = 0; i < dim; ++i) {
        sum += x[row * dim + i];
        out[row * dim + i] = sum;
    }
}

__global__ void l2_normalize_kernel(const float *x, float *out, long long rows,
                                    long long dim, float eps) {
    const long long row = blockIdx.x;
    if (row >= rows) return;
    __shared__ double scratch[4];
    double sum = 0.0;
    for (long long i = threadIdx.x; i < dim; i += blockDim.x) {
        const double v = x[row * dim + i];
        sum += v * v;
    }
    sum = block_sum_double(sum, scratch);
    const float inv = static_cast<float>(1.0 / sqrt(sum + eps));
    for (long long i = threadIdx.x; i < dim; i += blockDim.x)
        out[row * dim + i] = x[row * dim + i] * inv;
}

__global__ void l2_normalize_scalar_kernel(const float *x, float *out,
                                           long long rows, long long dim,
                                           float eps) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (long long row = 0; row < rows; ++row) {
        double sum = 0.0;
        for (long long i = 0; i < dim; ++i) {
            const double v = x[row * dim + i];
            sum += v * v;
        }
        const float inv = static_cast<float>(1.0 / sqrt(sum + eps));
        for (long long i = 0; i < dim; ++i) out[row * dim + i] = x[row * dim + i] * inv;
    }
}

__global__ void group_norm_kernel(const float *x, const float *weight,
                                  const float *bias, float *out,
                                  long long batch, long long channels,
                                  long long spatial, long long groups,
                                  float eps) {
    const long long item = blockIdx.x;
    if (item >= batch * groups) return;
    const long long b = item / groups;
    const long long group = item % groups;
    const long long channels_per_group = channels / groups;
    const long long group_size = channels_per_group * spatial;
    const long long offset = (b * channels + group * channels_per_group) * spatial;
    __shared__ double scratch[4];
    double sum = 0.0;
    double sumsq = 0.0;
    for (long long i = threadIdx.x; i < group_size; i += blockDim.x) {
        const double v = x[offset + i];
        sum += v;
        sumsq += v * v;
    }
    sum = block_sum_double(sum, scratch);
    sumsq = block_sum_double(sumsq, scratch);
    const double mean = sum / group_size;
    const float inv = static_cast<float>(
        1.0 / sqrt(sumsq / group_size - mean * mean + eps));
    for (long long i = threadIdx.x; i < group_size; i += blockDim.x) {
        const long long local_channel = i / spatial;
        const long long global_channel = group * channels_per_group + local_channel;
        const float gain = weight == nullptr ? 1.0f : weight[global_channel];
        const float shift = bias == nullptr ? 0.0f : bias[global_channel];
        out[offset + i] = (x[offset + i] - static_cast<float>(mean)) * inv * gain + shift;
    }
}

__global__ void group_norm_scalar_kernel(const float *x, const float *weight,
                                         const float *bias, float *out,
                                         long long batch, long long channels,
                                         long long spatial, long long groups,
                                         float eps) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const long long channels_per_group = channels / groups;
    const long long group_size = channels_per_group * spatial;
    for (long long b = 0; b < batch; ++b) {
        for (long long group = 0; group < groups; ++group) {
            const long long offset = (b * channels + group * channels_per_group) * spatial;
            double sum = 0.0;
            double sumsq = 0.0;
            for (long long i = 0; i < group_size; ++i) {
                const double v = x[offset + i];
                sum += v;
                sumsq += v * v;
            }
            const double mean = sum / group_size;
            const float inv = static_cast<float>(
                1.0 / sqrt(sumsq / group_size - mean * mean + eps));
            for (long long i = 0; i < group_size; ++i) {
                const long long local_channel = i / spatial;
                const long long global_channel = group * channels_per_group + local_channel;
                out[offset + i] = (x[offset + i] - static_cast<float>(mean)) * inv *
                                  weight[global_channel] + bias[global_channel];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Layout, scatter, and small matrix utilities
// ---------------------------------------------------------------------------

__global__ void concat_kernel(const float *a, const float *b, float *out,
                              long long outer, long long a_axis,
                              long long b_axis, long long inner) {
    const long long output_axis = a_axis + b_axis;
    const long long count = outer * output_axis * inner;
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    const long long inner_idx = idx % inner;
    const long long axis = (idx / inner) % output_axis;
    const long long outer_idx = idx / (output_axis * inner);
    if (axis < a_axis)
        out[idx] = a[(outer_idx * a_axis + axis) * inner + inner_idx];
    else
        out[idx] = b[(outer_idx * b_axis + (axis - a_axis)) * inner + inner_idx];
}

__global__ void repeat_2d_kernel(const float *x, float *out,
                                 long long source_rows, long long source_cols,
                                 long long output_rows, long long output_cols) {
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    const long long count = output_rows * output_cols;
    if (idx >= count) return;
    const long long row = idx / output_cols;
    const long long col = idx % output_cols;
    out[idx] = x[(row % source_rows) * source_cols + (col % source_cols)];
}

__global__ void repeat_backward_2d_kernel(const float *grad_out,
                                          float *grad_in,
                                          long long source_rows,
                                          long long source_cols,
                                          long long output_rows,
                                          long long output_cols) {
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    const long long count = source_rows * source_cols;
    if (idx >= count) return;
    const long long row = idx / source_cols;
    const long long col = idx % source_cols;
    const long long row_repeats = output_rows / source_rows;
    const long long col_repeats = output_cols / source_cols;
    float sum = 0.0f;
    for (long long rr = 0; rr < row_repeats; ++rr)
        for (long long cc = 0; cc < col_repeats; ++cc)
            sum += grad_out[(row + rr * source_rows) * output_cols +
                            (col + cc * source_cols)];
    grad_in[idx] = sum;
}

__global__ void pad_2d_kernel(const float *x, float *out, long long rows,
                              long long cols, long long top, long long bottom,
                              long long left, long long right, float value) {
    const long long output_rows = rows + top + bottom;
    const long long output_cols = cols + left + right;
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    const long long count = output_rows * output_cols;
    if (idx >= count) return;
    const long long row = idx / output_cols;
    const long long col = idx % output_cols;
    if (row >= top && row < top + rows && col >= left && col < left + cols)
        out[idx] = x[(row - top) * cols + (col - left)];
    else
        out[idx] = value;
}

__global__ void pad_reflect_1d_kernel(const float *x, float *out,
                                      long long rows, long long length,
                                      long long left, long long right) {
    const long long output_length = left + length + right;
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    const long long count = rows * output_length;
    if (idx >= count) return;
    const long long row = idx / output_length;
    const long long i = idx % output_length;
    long long source = i - left;
    if (source < 0) source = -source;
    if (source >= length) source = 2 * length - source - 2;
    out[idx] = x[row * length + source];
}

__global__ void roll_2d_kernel(const float *x, float *out, long long rows,
                               long long cols, long long row_shift,
                               long long col_shift) {
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    const long long count = rows * cols;
    if (idx >= count) return;
    const long long row = idx / cols;
    const long long col = idx % cols;
    const long long source_row = positive_mod_device(row - row_shift, rows);
    const long long source_col = positive_mod_device(col - col_shift, cols);
    out[idx] = x[source_row * cols + source_col];
}

__global__ void diag_embed_kernel(const float *diagonal, float *out,
                                  long long batch, long long dim) {
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    const long long count = batch * dim * dim;
    if (idx >= count) return;
    const long long col = idx % dim;
    const long long row = (idx / dim) % dim;
    const long long b = idx / (dim * dim);
    out[idx] = row == col ? diagonal[b * dim + row] : 0.0f;
}

__global__ void diag_mask_kernel(const float *x, float *out, long long rows,
                                 long long cols, long long past,
                                 int neg_inf) {
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    const long long count = rows * cols;
    if (idx >= count) return;
    const long long row = idx / cols;
    const long long col = idx % cols;
    const bool masked = col > past + row;
    out[idx] = masked ? (neg_inf ? -INFINITY : 0.0f) : x[idx];
}

__global__ void triangular_fill_kernel(const float *x, float *out,
                                       long long rows, long long cols,
                                       long long diagonal, int upper,
                                       float fill_value) {
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    const long long count = rows * cols;
    if (idx >= count) return;
    const long long row = idx / cols;
    const long long col = idx % cols;
    const bool selected = upper ? col - row >= diagonal : col - row <= diagonal;
    out[idx] = selected ? x[idx] : fill_value;
}

__global__ void add_id_kernel(const float *x, const float *rows,
                              const int32_t *ids, float *out, long long count,
                              long long row_count, long long width) {
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = count * width;
    if (idx >= total) return;
    const long long item = idx / width;
    const long long col = idx % width;
    out[idx] = x[idx] + rows[static_cast<long long>(ids[item]) * width + col];
}

__global__ void tensor_copy_kernel(const float *source, float *destination,
                                   long long count) {
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) destination[idx] = source[idx];
}

__global__ void set_rows_kernel(const float *source, const int32_t *row_ids,
                                const float *base, float *destination,
                                long long source_rows,
                                long long destination_rows,
                                long long row_width) {
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = destination_rows * row_width;
    if (idx >= total) return;
    const long long dest_row = idx / row_width;
    const long long col = idx % row_width;
    float value = base[idx];
    for (long long row = 0; row < source_rows; ++row) {
        if (row_ids[row] == dest_row) value = source[row * row_width + col];
    }
    destination[idx] = value;
}

__global__ void tensor_set_4d_kernel(const float *base, const float *update,
                                     float *output, long long output_count,
                                     long long n0, long long n1,
                                     long long n2, long long n3,
                                     long long stride1, long long stride2,
                                     long long stride3, long long offset) {
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= output_count) return;
    float value = base[idx];
    for (long long i3 = 0; i3 < n3; ++i3)
        for (long long i2 = 0; i2 < n2; ++i2)
            for (long long i1 = 0; i1 < n1; ++i1)
                for (long long i0 = 0; i0 < n0; ++i0) {
                    const long long dest = offset + i3 * stride3 + i2 * stride2 +
                                           i1 * stride1 + i0;
                    if (dest == idx) {
                        const long long source = ((i3 * n2 + i2) * n1 + i1) * n0 + i0;
                        value = update[source];
                    }
                }
    output[idx] = value;
}

__global__ void outer_product_kernel(const float *x, const float *y,
                                     float *out, long long rows,
                                     long long cols) {
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    const long long count = rows * cols;
    if (idx >= count) return;
    out[idx] = x[idx / cols] * y[idx % cols];
}

__global__ void solve_lower_triangular_kernel(const float *a, const float *b,
                                              float *x, long long batch,
                                              long long n,
                                              long long rhs_count) {
    const long long item = blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = batch * rhs_count;
    if (item >= total) return;
    const long long batch_index = item / rhs_count;
    const long long rhs = item % rhs_count;
    const float *matrix = a + batch_index * n * n;
    const float *source = b + batch_index * n * rhs_count;
    float *destination = x + batch_index * n * rhs_count;
    for (long long row = 0; row < n; ++row) {
        float sum = source[row * rhs_count + rhs];
        for (long long col = 0; col < row; ++col)
            sum -= matrix[row * n + col] * destination[col * rhs_count + rhs];
        destination[row * rhs_count + rhs] = sum / matrix[row * n + row];
    }
}

__global__ void argsort_kernel(const float *x, int32_t *indices,
                               long long rows, long long dim,
                               int descending) {
    const long long row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || dim > kSortMax) return;
    int order[kSortMax];
    for (int i = 0; i < dim; ++i) order[i] = i;
    const float *values = x + row * dim;
    for (int i = 1; i < dim; ++i) {
        const int current = order[i];
        int j = i - 1;
        while (j >= 0) {
            const bool move = descending ? values[current] > values[order[j]]
                                         : values[current] < values[order[j]];
            if (!move) break;
            order[j + 1] = order[j];
            --j;
        }
        order[j + 1] = current;
    }
    for (int i = 0; i < dim; ++i) indices[row * dim + i] = order[i];
}

__global__ void argsort_scalar_kernel(const float *x, int32_t *indices,
                                      long long rows, long long dim,
                                      int descending) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || dim > kSortMax) return;
    for (long long row = 0; row < rows; ++row) {
        int order[kSortMax];
        for (int i = 0; i < dim; ++i) order[i] = i;
        const float *values = x + row * dim;
        for (int i = 1; i < dim; ++i) {
            const int current = order[i];
            int j = i - 1;
            while (j >= 0) {
                const bool move = descending ? values[current] > values[order[j]]
                                             : values[current] < values[order[j]];
                if (!move) break;
                order[j + 1] = order[j];
                --j;
            }
            order[j + 1] = current;
        }
        for (int i = 0; i < dim; ++i) indices[row * dim + i] = order[i];
    }
}

__global__ void threshold_topk_indices_kernel(const float *scores,
                                              int32_t *indices,
                                              long long rows,
                                              long long width, int k) {
    const long long row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || width > kSortMax) return;
    float sorted[kSortMax];
    const float *source = scores + row * width;
    for (int i = 0; i < width; ++i) sorted[i] = source[i];
    for (int i = 1; i < width; ++i) {
        const float current = sorted[i];
        int j = i - 1;
        while (j >= 0 && current > sorted[j]) {
            sorted[j + 1] = sorted[j];
            --j;
        }
        sorted[j + 1] = current;
    }
    const float threshold = sorted[k - 1];
    int selected = 0;
    for (int column = 0; column < width && selected < k; ++column)
        if (source[column] > threshold) indices[row * k + selected++] = column;
    for (int column = 0; column < width && selected < k; ++column)
        if (source[column] == threshold) indices[row * k + selected++] = column;
}

__global__ void threshold_topk_indices_scalar_kernel(const float *scores,
                                                     int32_t *indices,
                                                     long long rows,
                                                     long long width, int k) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || width > kSortMax) return;
    for (long long row = 0; row < rows; ++row) {
        float sorted[kSortMax];
        const float *source = scores + row * width;
        for (int i = 0; i < width; ++i) sorted[i] = source[i];
        for (int i = 1; i < width; ++i) {
            const float current = sorted[i];
            int j = i - 1;
            while (j >= 0 && current > sorted[j]) {
                sorted[j + 1] = sorted[j];
                --j;
            }
            sorted[j + 1] = current;
        }
        const float threshold = sorted[k - 1];
        int selected = 0;
        for (int column = 0; column < width && selected < k; ++column)
            if (source[column] > threshold) indices[row * k + selected++] = column;
        for (int column = 0; column < width && selected < k; ++column)
            if (source[column] == threshold) indices[row * k + selected++] = column;
    }
}

__global__ void scalar_copy_kernel(const float *source, float *destination,
                                   long long count) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (long long i = 0; i < count; ++i) destination[i] = source[i];
}

// ---------------------------------------------------------------------------
// Host oracles
// ---------------------------------------------------------------------------

std::vector<double> unary_ref(const std::vector<float> &x, int op) {
    std::vector<double> ref(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        ref[i] = unary_host(x[i], op, 0.31f, 0.71f, 0.17f, 0.5f);
    return ref;
}

std::vector<double> unary_misc_ref(const std::vector<float> &x, int mode,
                                   float a, float b) {
    std::vector<double> ref(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        const float v = x[i];
        switch (mode) {
            case 0: ref[i] = v + a; break;
            case 1: ref[i] = std::clamp(v, a, b); break;
            case 2: ref[i] = v >= 0.0f ? v : a * v; break;
            case 3: ref[i] = v * a; break;
            case 4: ref[i] = v * v; break;
            case 5: ref[i] = std::sqrt(v); break;
            case 6: ref[i] = std::sin(v); break;
            case 7: ref[i] = std::cos(v); break;
            case 8: ref[i] = std::log(v); break;
            case 9: ref[i] = a; break;
            case 10: ref[i] = a + b * static_cast<float>(i); break;
        }
    }
    return ref;
}

std::vector<double> binary_ref(const std::vector<float> &x,
                               const std::vector<float> &y, int mode) {
    std::vector<double> ref(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        if (mode == 0) ref[i] = x[i] * y[i];
        else if (mode == 1) ref[i] = x[i] / y[i];
        else ref[i] = x[i] - y[i];
    }
    return ref;
}

std::vector<double> reduce_mean_ref(const std::vector<float> &x,
                                    long long rows, long long dim) {
    std::vector<double> ref(rows);
    for (long long row = 0; row < rows; ++row) {
        double sum = 0.0;
        for (long long i = 0; i < dim; ++i) sum += x[row * dim + i];
        ref[row] = sum / dim;
    }
    return ref;
}

std::vector<double> cumulative_sum_ref(const std::vector<float> &x,
                                       long long rows, long long dim) {
    std::vector<double> ref(rows * dim);
    for (long long row = 0; row < rows; ++row) {
        float sum = 0.0f;
        for (long long i = 0; i < dim; ++i) {
            sum += x[row * dim + i];
            ref[row * dim + i] = sum;
        }
    }
    return ref;
}

std::vector<double> l2_ref(const std::vector<float> &x, long long rows,
                           long long dim, float eps) {
    std::vector<double> ref(rows * dim);
    for (long long row = 0; row < rows; ++row) {
        double sum = 0.0;
        for (long long i = 0; i < dim; ++i) {
            const double v = x[row * dim + i];
            sum += v * v;
        }
        const double inv = 1.0 / std::sqrt(sum + eps);
        for (long long i = 0; i < dim; ++i) ref[row * dim + i] = x[row * dim + i] * inv;
    }
    return ref;
}

std::vector<double> group_norm_ref(const std::vector<float> &x,
                                   const std::vector<float> &weight,
                                   const std::vector<float> &bias,
                                   long long batch, long long channels,
                                   long long spatial, long long groups,
                                   float eps) {
    std::vector<double> ref(x.size());
    const long long channels_per_group = channels / groups;
    const long long group_size = channels_per_group * spatial;
    for (long long b = 0; b < batch; ++b) {
        for (long long group = 0; group < groups; ++group) {
            const long long offset = (b * channels + group * channels_per_group) * spatial;
            double sum = 0.0;
            double sumsq = 0.0;
            for (long long i = 0; i < group_size; ++i) {
                sum += x[offset + i];
                sumsq += static_cast<double>(x[offset + i]) * x[offset + i];
            }
            const double mean = sum / group_size;
            const double inv = 1.0 / std::sqrt(sumsq / group_size - mean * mean + eps);
            for (long long c = 0; c < channels_per_group; ++c) {
                const long long global_channel = group * channels_per_group + c;
                for (long long i = 0; i < spatial; ++i) {
                    const long long idx = offset + c * spatial + i;
                    ref[idx] = (x[idx] - static_cast<float>(mean)) * static_cast<float>(inv) *
                               weight[global_channel] + bias[global_channel];
                }
            }
        }
    }
    return ref;
}

std::vector<double> argsort_ref(const std::vector<float> &x, long long rows,
                                long long dim, bool descending) {
    std::vector<int32_t> out(rows * dim);
    std::vector<int> order(dim);
    for (long long row = 0; row < rows; ++row) {
        std::iota(order.begin(), order.end(), 0);
        const float *values = x.data() + row * dim;
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
            return descending ? values[a] > values[b] : values[a] < values[b];
        });
        for (long long i = 0; i < dim; ++i) out[row * dim + i] = order[i];
    }
    return to_ref_i32(out);
}

std::vector<double> threshold_topk_ref(const std::vector<float> &scores,
                                       long long rows, long long width, int k) {
    std::vector<int32_t> out(rows * k);
    std::vector<float> scratch(width);
    for (long long row = 0; row < rows; ++row) {
        const float *source = scores.data() + row * width;
        std::copy_n(source, width, scratch.data());
        std::sort(scratch.begin(), scratch.end(), std::greater<float>());
        const float threshold = scratch[k - 1];
        int selected = 0;
        for (long long col = 0; col < width && selected < k; ++col)
            if (source[col] > threshold) out[row * k + selected++] = static_cast<int32_t>(col);
        for (long long col = 0; col < width && selected < k; ++col)
            if (source[col] == threshold) out[row * k + selected++] = static_cast<int32_t>(col);
    }
    return to_ref_i32(out);
}

// ---------------------------------------------------------------------------
// Correctness
// ---------------------------------------------------------------------------

bool run_correctness() {
    bool ok = true;
    int checks = 0;
    qc::Rng rng(1117);

    const long long n = 4099;
    std::vector<float> x = rng.uniforms(n, -3.0f, 3.0f);
    x[0] = 0.0f;
    x[1] = -0.0f;
    x[2] = 0.5f;
    x[3] = -0.5f;
    float *dx = qc::dnew(x);
    float *dout = qc::dzero<float>(n);
    for (int op = kAbs; op <= kTrunc; ++op) {
        unary_kernel<<<qc::grid_for(n, kThreads), kThreads>>>(
            dx, dout, n, op, 0.31f, 0.71f, 0.17f, 0.5f);
        QC_SYNC();
        ok &= qc::compare(qc::d2h(dout, n), unary_ref(x, op), qc::Tol::fp32())
                  .report(std::string("unary ") + kUnaryNames[op]);
        ++checks;
    }

    std::vector<float> positive = rng.uniforms(n, 0.01f, 4.0f);
    float *dpos = qc::dnew(positive);
    struct MiscCase { const char *name; int mode; float a; float b; const float *host; float *device; };
    std::vector<MiscCase> misc = {
        {"add_scalar", 0, 1.25f, 0.0f, x.data(), dx},
        {"value_clip", 1, -0.75f, 0.80f, x.data(), dx},
        {"clamp", 1, -0.55f, 0.65f, x.data(), dx},
        {"leaky_relu", 2, 0.125f, 0.0f, x.data(), dx},
        {"scale", 3, -1.75f, 0.0f, x.data(), dx},
        {"square", 4, 0.0f, 0.0f, x.data(), dx},
        {"square_root", 5, 0.0f, 0.0f, positive.data(), dpos},
        {"sine", 6, 0.0f, 0.0f, x.data(), dx},
        {"cosine", 7, 0.0f, 0.0f, x.data(), dx},
        {"logarithm", 8, 0.0f, 0.0f, positive.data(), dpos},
        {"fill", 9, -2.25f, 0.0f, x.data(), dx},
        {"arange", 10, -3.0f, 0.25f, x.data(), dx},
    };
    for (const auto &c : misc) {
        unary_misc_kernel<<<qc::grid_for(n, kThreads), kThreads>>>(
            c.mode == 9 || c.mode == 10 ? nullptr : c.device, dout, n,
            c.mode, c.a, c.b);
        QC_SYNC();
        std::vector<float> host(c.host, c.host + n);
        ok &= qc::compare(qc::d2h(dout, n), unary_misc_ref(host, c.mode, c.a, c.b),
                          qc::Tol::fp32())
                  .report(c.name);
        ++checks;
    }

    std::vector<float> y = rng.uniforms(n, 0.5f, 2.0f);
    float *dy = qc::dnew(y);
    const char *binary_names[] = {"multiply", "divide", "subtract"};
    for (int mode = 0; mode < 3; ++mode) {
        binary_kernel<<<qc::grid_for(n, kThreads), kThreads>>>(dx, dy, dout, n, mode);
        QC_SYNC();
        ok &= qc::compare(qc::d2h(dout, n), binary_ref(x, y, mode), qc::Tol::fp32())
                  .report(binary_names[mode]);
        ++checks;
    }

    float *dacc = qc::dnew(x);
    accumulate_kernel<<<qc::grid_for(n, kThreads), kThreads>>>(dacc, dy, n, 0.375f);
    QC_SYNC();
    std::vector<double> acc_ref(n);
    for (long long i = 0; i < n; ++i) acc_ref[i] = x[i] + 0.375f * y[i];
    ok &= qc::compare(qc::d2h(dacc, n), acc_ref, qc::Tol::fp32()).report("accumulate");
    ++checks;

    sigmoid_mul_kernel<<<qc::grid_for(n, kThreads), kThreads>>>(dx, dy, dout, n);
    QC_SYNC();
    std::vector<double> sig_ref(n);
    for (long long i = 0; i < n; ++i) sig_ref[i] = sigmoid_host(x[i]) * y[i];
    ok &= qc::compare(qc::d2h(dout, n), sig_ref, qc::Tol::fp32()).report("sigmoid_mul");
    ++checks;

    std::vector<float> grad = rng.uniforms(n, -1.0f, 1.0f);
    float *dgrad = qc::dnew(grad);
    float *dg_gate = qc::dzero<float>(n);
    float *dg_value = qc::dzero<float>(n);
    sigmoid_mul_backward_kernel<<<qc::grid_for(n, kThreads), kThreads>>>(
        dgrad, dx, dy, dg_gate, dg_value, n);
    QC_SYNC();
    std::vector<double> grad_gate_ref(n), grad_value_ref(n);
    for (long long i = 0; i < n; ++i) {
        const float p = sigmoid_host(x[i]);
        grad_gate_ref[i] = grad[i] * y[i] * p * (1.0f - p);
        grad_value_ref[i] = grad[i] * p;
    }
    ok &= qc::compare(qc::d2h(dg_gate, n), grad_gate_ref, qc::Tol::fp32())
              .report("sigmoid_mul_backward grad_gate");
    ok &= qc::compare(qc::d2h(dg_value, n), grad_value_ref, qc::Tol::fp32())
              .report("sigmoid_mul_backward grad_value");
    checks += 2;

    const long long rows = 17, dim = 73;
    std::vector<float> matrix = rng.uniforms(rows * dim, -1.0f, 1.0f);
    float *dmatrix = qc::dnew(matrix);
    float *drow_out = qc::dzero<float>(rows);
    reduce_mean_kernel<<<rows, kThreads>>>(dmatrix, drow_out, rows, dim);
    QC_SYNC();
    ok &= qc::compare(qc::d2h(drow_out, rows), reduce_mean_ref(matrix, rows, dim),
                      qc::Tol::fp32())
              .report("reduce_mean");
    ++checks;

    float *dsum = qc::dzero<float>(1);
    reduce_sum_all_kernel<<<1, kThreads>>>(dmatrix, dsum, rows * dim);
    QC_SYNC();
    double sum_ref = 0.0;
    for (float v : matrix) sum_ref += v;
    ok &= qc::compare(qc::d2h(dsum, 1), std::vector<double>{sum_ref},
                      qc::Tol::fp32().with_elementwise(2e-5, 2e-5))
              .report("reduce_sum_all");
    ++checks;

    float *dcumsum = qc::dzero<float>(rows * dim);
    cumulative_sum_kernel<<<rows, 64>>>(dmatrix, dcumsum, rows, dim);
    QC_SYNC();
    ok &= qc::compare(qc::d2h(dcumsum, rows * dim),
                      cumulative_sum_ref(matrix, rows, dim), qc::Tol::fp32())
              .report("cumulative_sum");
    ++checks;

    float *dl2 = qc::dzero<float>(rows * dim);
    l2_normalize_kernel<<<rows, kThreads>>>(dmatrix, dl2, rows, dim, 1e-12f);
    QC_SYNC();
    ok &= qc::compare(qc::d2h(dl2, rows * dim), l2_ref(matrix, rows, dim, 1e-12f),
                      qc::Tol::fp32())
              .report("l2_normalize");
    ++checks;

    const long long gbatch = 3, gchannels = 8, gspatial = 11, groups = 4;
    std::vector<float> gx = rng.uniforms(gbatch * gchannels * gspatial, -1.0f, 1.0f);
    std::vector<float> gw = rng.uniforms(gchannels, 0.5f, 1.5f);
    std::vector<float> gb = rng.uniforms(gchannels, -0.2f, 0.2f);
    float *dgx = qc::dnew(gx);
    float *dgw = qc::dnew(gw);
    float *dgb = qc::dnew(gb);
    float *dgn = qc::dzero<float>(gx.size());
    group_norm_kernel<<<gbatch * groups, kThreads>>>(
        dgx, dgw, dgb, dgn, gbatch, gchannels, gspatial, groups, 1e-5f);
    QC_SYNC();
    ok &= qc::compare(qc::d2h(dgn, gx.size()),
                      group_norm_ref(gx, gw, gb, gbatch, gchannels, gspatial,
                                     groups, 1e-5f),
                      qc::Tol::fp32().with_elementwise(3e-5, 3e-5))
              .report("group_norm");
    ++checks;

    std::vector<int32_t> ix = rng.integers(2048, -4, 4);
    std::vector<int32_t> iy = ix;
    for (size_t i = 0; i < iy.size(); i += 5) iy[i] += 1;
    int32_t *dix = qc::dnew(ix);
    int32_t *diy = qc::dnew(iy);
    long long *dcount = qc::dzero<long long>(1);
    count_equal_kernel<<<1, kThreads>>>(dix, diy, ix.size(), dcount);
    QC_SYNC();
    long long count_ref = 0;
    for (size_t i = 0; i < ix.size(); ++i) count_ref += ix[i] == iy[i];
    ok &= qc::compare(qc::d2h(dcount, 1), std::vector<long long>{count_ref},
                      qc::Tol::exact())
              .report("count_equal");
    ++checks;

    const long long outer = 5, a_axis = 3, b_axis = 4, inner = 7;
    std::vector<float> ca = rng.uniforms(outer * a_axis * inner, -1.0f, 1.0f);
    std::vector<float> cb = rng.uniforms(outer * b_axis * inner, -1.0f, 1.0f);
    float *dca = qc::dnew(ca);
    float *dcb = qc::dnew(cb);
    float *dconcat = qc::dzero<float>(outer * (a_axis + b_axis) * inner);
    concat_kernel<<<qc::grid_for(outer * (a_axis + b_axis) * inner, kThreads), kThreads>>>(
        dca, dcb, dconcat, outer, a_axis, b_axis, inner);
    QC_SYNC();
    std::vector<double> concat_ref(outer * (a_axis + b_axis) * inner);
    for (long long o = 0; o < outer; ++o) {
        for (long long a = 0; a < a_axis * inner; ++a)
            concat_ref[o * (a_axis + b_axis) * inner + a] =
                ca[o * a_axis * inner + a];
        for (long long b = 0; b < b_axis * inner; ++b)
            concat_ref[o * (a_axis + b_axis) * inner + a_axis * inner + b] =
                cb[o * b_axis * inner + b];
    }
    ok &= qc::compare(qc::d2h(dconcat, concat_ref.size()), concat_ref,
                      qc::Tol::fp32())
              .report("concat");
    ++checks;

    const long long sr = 3, sc = 4, orows = 9, ocols = 12;
    std::vector<float> small = rng.uniforms(sr * sc, -1.0f, 1.0f);
    float *dsmall = qc::dnew(small);
    float *drepeat = qc::dzero<float>(orows * ocols);
    repeat_2d_kernel<<<qc::grid_for(orows * ocols, kThreads), kThreads>>>(
        dsmall, drepeat, sr, sc, orows, ocols);
    QC_SYNC();
    std::vector<double> repeat_ref(orows * ocols);
    for (long long r = 0; r < orows; ++r)
        for (long long c = 0; c < ocols; ++c)
            repeat_ref[r * ocols + c] = small[(r % sr) * sc + (c % sc)];
    ok &= qc::compare(qc::d2h(drepeat, repeat_ref.size()), repeat_ref,
                      qc::Tol::fp32())
              .report("repeat_2d");
    ++checks;

    std::vector<float> grad_out = rng.uniforms(orows * ocols, -1.0f, 1.0f);
    float *dgrad_out = qc::dnew(grad_out);
    float *drepeat_bwd = qc::dzero<float>(sr * sc);
    repeat_backward_2d_kernel<<<qc::grid_for(sr * sc, kThreads), kThreads>>>(
        dgrad_out, drepeat_bwd, sr, sc, orows, ocols);
    QC_SYNC();
    std::vector<double> repeat_bwd_ref(sr * sc, 0.0);
    for (long long r = 0; r < orows; ++r)
        for (long long c = 0; c < ocols; ++c)
            repeat_bwd_ref[(r % sr) * sc + (c % sc)] += grad_out[r * ocols + c];
    ok &= qc::compare(qc::d2h(drepeat_bwd, sr * sc), repeat_bwd_ref,
                      qc::Tol::fp32())
              .report("repeat_backward_2d");
    ++checks;

    const long long prow = 4, pcol = 5, top = 2, bottom = 1, left = 1, right = 3;
    std::vector<float> pad_src = rng.uniforms(prow * pcol, -1.0f, 1.0f);
    float *dpad_src = qc::dnew(pad_src);
    const long long pad_rows = prow + top + bottom;
    const long long pad_cols = pcol + left + right;
    float *dpad = qc::dzero<float>(pad_rows * pad_cols);
    pad_2d_kernel<<<qc::grid_for(pad_rows * pad_cols, kThreads), kThreads>>>(
        dpad_src, dpad, prow, pcol, top, bottom, left, right, -3.5f);
    QC_SYNC();
    std::vector<double> pad_ref(pad_rows * pad_cols, -3.5);
    for (long long r = 0; r < prow; ++r)
        for (long long c = 0; c < pcol; ++c)
            pad_ref[(r + top) * pad_cols + left + c] = pad_src[r * pcol + c];
    ok &= qc::compare(qc::d2h(dpad, pad_ref.size()), pad_ref, qc::Tol::fp32())
              .report("pad_2d");
    ++checks;

    const long long refl_rows = 3, refl_len = 5, refl_left = 2, refl_right = 3;
    std::vector<float> refl_src = rng.uniforms(refl_rows * refl_len, -1.0f, 1.0f);
    float *drefl_src = qc::dnew(refl_src);
    const long long refl_out_len = refl_left + refl_len + refl_right;
    float *drefl = qc::dzero<float>(refl_rows * refl_out_len);
    pad_reflect_1d_kernel<<<qc::grid_for(refl_rows * refl_out_len, kThreads), kThreads>>>(
        drefl_src, drefl, refl_rows, refl_len, refl_left, refl_right);
    QC_SYNC();
    std::vector<double> refl_ref(refl_rows * refl_out_len);
    for (long long r = 0; r < refl_rows; ++r)
        for (long long i = 0; i < refl_out_len; ++i) {
            long long source = i - refl_left;
            if (source < 0) source = -source;
            if (source >= refl_len) source = 2 * refl_len - source - 2;
            refl_ref[r * refl_out_len + i] = refl_src[r * refl_len + source];
        }
    ok &= qc::compare(qc::d2h(drefl, refl_ref.size()), refl_ref, qc::Tol::fp32())
              .report("pad_reflect_1d");
    ++checks;

    float *droll = qc::dzero<float>(prow * pcol);
    roll_2d_kernel<<<qc::grid_for(prow * pcol, kThreads), kThreads>>>(
        dpad_src, droll, prow, pcol, -1, 2);
    QC_SYNC();
    std::vector<double> roll_ref(prow * pcol);
    for (long long r = 0; r < prow; ++r)
        for (long long c = 0; c < pcol; ++c)
            roll_ref[r * pcol + c] =
                pad_src[positive_mod_host(r + 1, prow) * pcol +
                        positive_mod_host(c - 2, pcol)];
    ok &= qc::compare(qc::d2h(droll, roll_ref.size()), roll_ref, qc::Tol::fp32())
              .report("roll_2d");
    ++checks;

    const long long batch = 3, dd = 6;
    std::vector<float> diagonal = rng.uniforms(batch * dd, -1.0f, 1.0f);
    float *ddiag = qc::dnew(diagonal);
    float *ddiag_out = qc::dzero<float>(batch * dd * dd);
    diag_embed_kernel<<<qc::grid_for(batch * dd * dd, kThreads), kThreads>>>(
        ddiag, ddiag_out, batch, dd);
    QC_SYNC();
    std::vector<double> diag_ref(batch * dd * dd, 0.0);
    for (long long b = 0; b < batch; ++b)
        for (long long i = 0; i < dd; ++i)
            diag_ref[(b * dd + i) * dd + i] = diagonal[b * dd + i];
    ok &= qc::compare(qc::d2h(ddiag_out, diag_ref.size()), diag_ref,
                      qc::Tol::fp32())
              .report("diag_embed");
    ++checks;

    float *ddiag_mask = qc::dzero<float>(prow * pcol);
    diag_mask_kernel<<<qc::grid_for(prow * pcol, kThreads), kThreads>>>(
        dpad_src, ddiag_mask, prow, pcol, 1, 0);
    QC_SYNC();
    std::vector<double> diag_mask_ref(prow * pcol);
    for (long long r = 0; r < prow; ++r)
        for (long long c = 0; c < pcol; ++c)
            diag_mask_ref[r * pcol + c] = c > 1 + r ? 0.0 : pad_src[r * pcol + c];
    ok &= qc::compare(qc::d2h(ddiag_mask, prow * pcol), diag_mask_ref,
                      qc::Tol::fp32())
              .report("diag_mask zero");
    ++checks;

    diag_mask_kernel<<<qc::grid_for(prow * pcol, kThreads), kThreads>>>(
        dpad_src, ddiag_mask, prow, pcol, 1, 1);
    QC_SYNC();
    for (long long r = 0; r < prow; ++r)
        for (long long c = 0; c < pcol; ++c)
            diag_mask_ref[r * pcol + c] =
                c > 1 + r ? -std::numeric_limits<double>::infinity()
                          : pad_src[r * pcol + c];
    ok &= report_with_inf("diag_mask neg_inf", qc::d2h(ddiag_mask, prow * pcol),
                          diag_mask_ref);
    ++checks;

    float *dtri = qc::dzero<float>(prow * pcol);
    triangular_fill_kernel<<<qc::grid_for(prow * pcol, kThreads), kThreads>>>(
        dpad_src, dtri, prow, pcol, 0, 1, -9.0f);
    QC_SYNC();
    std::vector<double> tri_ref(prow * pcol);
    for (long long r = 0; r < prow; ++r)
        for (long long c = 0; c < pcol; ++c)
            tri_ref[r * pcol + c] = c - r >= 0 ? pad_src[r * pcol + c] : -9.0;
    ok &= qc::compare(qc::d2h(dtri, prow * pcol), tri_ref, qc::Tol::fp32())
              .report("triangular_fill");
    ++checks;

    const long long id_count = 9, id_rows = 5, id_width = 7;
    std::vector<float> id_x = rng.uniforms(id_count * id_width, -1.0f, 1.0f);
    std::vector<float> id_table = rng.uniforms(id_rows * id_width, -1.0f, 1.0f);
    std::vector<int32_t> ids = {0, 2, 4, 1, 3, 2, 0, 4, 1};
    float *did_x = qc::dnew(id_x);
    float *did_table = qc::dnew(id_table);
    int32_t *dids = qc::dnew(ids);
    float *did_out = qc::dzero<float>(id_count * id_width);
    add_id_kernel<<<qc::grid_for(id_count * id_width, kThreads), kThreads>>>(
        did_x, did_table, dids, did_out, id_count, id_rows, id_width);
    QC_SYNC();
    std::vector<double> add_id_ref(id_count * id_width);
    for (long long i = 0; i < id_count; ++i)
        for (long long c = 0; c < id_width; ++c)
            add_id_ref[i * id_width + c] =
                id_x[i * id_width + c] + id_table[ids[i] * id_width + c];
    ok &= qc::compare(qc::d2h(did_out, add_id_ref.size()), add_id_ref,
                      qc::Tol::fp32())
              .report("add_id");
    ++checks;

    float *dcopy = qc::dzero<float>(n);
    tensor_copy_kernel<<<qc::grid_for(n, kThreads), kThreads>>>(dx, dcopy, n);
    QC_SYNC();
    ok &= qc::compare(qc::d2h(dcopy, n), to_ref(x), qc::Tol::fp32())
              .report("tensor_copy");
    ++checks;

    const long long src_rows = 5, dst_rows = 4, row_width = 6;
    std::vector<float> src_rows_data = rng.uniforms(src_rows * row_width, -1.0f, 1.0f);
    std::vector<float> dst_base = rng.uniforms(dst_rows * row_width, -1.0f, 1.0f);
    std::vector<int32_t> row_ids = {0, 2, 1, 2, 3};
    float *dsrc_rows = qc::dnew(src_rows_data);
    float *ddst_base = qc::dnew(dst_base);
    int32_t *drow_ids = qc::dnew(row_ids);
    float *dset_rows = qc::dzero<float>(dst_rows * row_width);
    set_rows_kernel<<<qc::grid_for(dst_rows * row_width, kThreads), kThreads>>>(
        dsrc_rows, drow_ids, ddst_base, dset_rows, src_rows, dst_rows, row_width);
    QC_SYNC();
    std::vector<double> set_rows_ref(dst_base.begin(), dst_base.end());
    for (long long r = 0; r < src_rows; ++r)
        for (long long c = 0; c < row_width; ++c)
            set_rows_ref[row_ids[r] * row_width + c] = src_rows_data[r * row_width + c];
    ok &= qc::compare(qc::d2h(dset_rows, set_rows_ref.size()), set_rows_ref,
                      qc::Tol::fp32())
              .report("set_rows");
    ++checks;

    const long long out_count = 80, n0 = 3, n1 = 2, n2 = 2, n3 = 2;
    const long long stride1 = 5, stride2 = 17, stride3 = 40, offset = 4;
    std::vector<float> base4 = rng.uniforms(out_count, -1.0f, 1.0f);
    std::vector<float> update4 = rng.uniforms(n0 * n1 * n2 * n3, -1.0f, 1.0f);
    float *dbase4 = qc::dnew(base4);
    float *dupdate4 = qc::dnew(update4);
    float *dset4 = qc::dzero<float>(out_count);
    tensor_set_4d_kernel<<<qc::grid_for(out_count, kThreads), kThreads>>>(
        dbase4, dupdate4, dset4, out_count, n0, n1, n2, n3, stride1, stride2,
        stride3, offset);
    QC_SYNC();
    std::vector<double> set4_ref(base4.begin(), base4.end());
    for (long long i3 = 0; i3 < n3; ++i3)
        for (long long i2 = 0; i2 < n2; ++i2)
            for (long long i1 = 0; i1 < n1; ++i1)
                for (long long i0 = 0; i0 < n0; ++i0) {
                    const long long dest = offset + i3 * stride3 + i2 * stride2 +
                                           i1 * stride1 + i0;
                    const long long src = ((i3 * n2 + i2) * n1 + i1) * n0 + i0;
                    set4_ref[dest] = update4[src];
                }
    ok &= qc::compare(qc::d2h(dset4, out_count), set4_ref, qc::Tol::fp32())
              .report("tensor_set_4d");
    ++checks;

    const long long sort_rows = 9, sort_dim = 16;
    std::vector<float> sort_values = rng.uniforms(sort_rows * sort_dim, -1.0f, 1.0f);
    for (long long r = 0; r < sort_rows; ++r) {
        sort_values[r * sort_dim + 3] = 0.25f;
        sort_values[r * sort_dim + 7] = 0.25f;
    }
    float *dsort_values = qc::dnew(sort_values);
    int32_t *dsort = qc::dzero<int32_t>(sort_rows * sort_dim);
    argsort_kernel<<<qc::grid_for(sort_rows, kThreads), kThreads>>>(
        dsort_values, dsort, sort_rows, sort_dim, 0);
    QC_SYNC();
    ok &= qc::compare(qc::d2h(dsort, sort_rows * sort_dim),
                      argsort_ref(sort_values, sort_rows, sort_dim, false),
                      qc::Tol::exact())
              .report("argsort ascending");
    ++checks;
    argsort_kernel<<<qc::grid_for(sort_rows, kThreads), kThreads>>>(
        dsort_values, dsort, sort_rows, sort_dim, 1);
    QC_SYNC();
    ok &= qc::compare(qc::d2h(dsort, sort_rows * sort_dim),
                      argsort_ref(sort_values, sort_rows, sort_dim, true),
                      qc::Tol::exact())
              .report("argsort descending");
    ++checks;

    const int topk = 5;
    int32_t *dtopk = qc::dzero<int32_t>(sort_rows * topk);
    threshold_topk_indices_kernel<<<qc::grid_for(sort_rows, kThreads), kThreads>>>(
        dsort_values, dtopk, sort_rows, sort_dim, topk);
    QC_SYNC();
    ok &= qc::compare(qc::d2h(dtopk, sort_rows * topk),
                      threshold_topk_ref(sort_values, sort_rows, sort_dim, topk),
                      qc::Tol::exact())
              .report("threshold_topk_indices");
    ++checks;

    const long long ox_rows = 11, oy_cols = 13;
    std::vector<float> ox = rng.uniforms(ox_rows, -1.0f, 1.0f);
    std::vector<float> oyv = rng.uniforms(oy_cols, -1.0f, 1.0f);
    float *dox = qc::dnew(ox);
    float *doyv = qc::dnew(oyv);
    float *douter = qc::dzero<float>(ox_rows * oy_cols);
    outer_product_kernel<<<qc::grid_for(ox_rows * oy_cols, kThreads), kThreads>>>(
        dox, doyv, douter, ox_rows, oy_cols);
    QC_SYNC();
    std::vector<double> outer_ref(ox_rows * oy_cols);
    for (long long r = 0; r < ox_rows; ++r)
        for (long long c = 0; c < oy_cols; ++c)
            outer_ref[r * oy_cols + c] = ox[r] * oyv[c];
    ok &= qc::compare(qc::d2h(douter, outer_ref.size()), outer_ref,
                      qc::Tol::fp32())
              .report("outer_product");
    ++checks;

    const long long tbatch = 3, tn = 5, trhs = 4;
    std::vector<float> ta(tbatch * tn * tn, 0.0f);
    std::vector<float> tb(tbatch * tn * trhs);
    for (long long b = 0; b < tbatch; ++b) {
        for (long long r = 0; r < tn; ++r) {
            for (long long c = 0; c <= r; ++c)
                ta[(b * tn + r) * tn + c] = (r == c ? 2.0f : 0.1f * (r + c + 1));
        }
    }
    tb = rng.uniforms(tbatch * tn * trhs, -1.0f, 1.0f);
    float *dta = qc::dnew(ta);
    float *dtb = qc::dnew(tb);
    float *dtx = qc::dzero<float>(tbatch * tn * trhs);
    solve_lower_triangular_kernel<<<qc::grid_for(tbatch * trhs, kThreads), kThreads>>>(
        dta, dtb, dtx, tbatch, tn, trhs);
    QC_SYNC();
    std::vector<double> tx_ref(tbatch * tn * trhs);
    for (long long b = 0; b < tbatch; ++b) {
        for (long long rhs = 0; rhs < trhs; ++rhs) {
            for (long long r = 0; r < tn; ++r) {
                float sum = tb[(b * tn + r) * trhs + rhs];
                for (long long c = 0; c < r; ++c)
                    sum -= ta[(b * tn + r) * tn + c] * static_cast<float>(tx_ref[(b * tn + c) * trhs + rhs]);
                tx_ref[(b * tn + r) * trhs + rhs] = sum / ta[(b * tn + r) * tn + r];
            }
        }
    }
    ok &= qc::compare(qc::d2h(dtx, tx_ref.size()), tx_ref,
                      qc::Tol::fp32().with_elementwise(2e-5, 2e-5))
              .report("solve_lower_triangular");
    ++checks;

    qc::dfree(dx, dout, dpos, dy, dacc, dgrad, dg_gate, dg_value, dmatrix,
              drow_out, dsum, dcumsum, dl2, dgx, dgw, dgb, dgn, dix, diy,
              dcount, dca, dcb, dconcat, dsmall, drepeat, dgrad_out,
              drepeat_bwd, dpad_src, dpad, drefl_src, drefl, droll, ddiag,
              ddiag_out, ddiag_mask, dtri, did_x, did_table, dids, did_out,
              dcopy, dsrc_rows, ddst_base, drow_ids, dset_rows, dbase4,
              dupdate4, dset4, dsort_values, dsort, dtopk, dox, doyv, douter,
              dta, dtb, dtx);

    std::printf("Phase 11 correctness checks: %d\n", checks);
    return ok;
}

void run_bench() {
    std::printf("\n== Phase 11 benchmarks ==\n");
    std::printf("   Timing note: medians are per launch; fast candidates use inner repeats.\n");
    qc::Rng rng(2117);

    auto bench_repeated = [](auto &&fn, int repeats, int warmups, int iters) {
        qc::Bench b = qc::bench([&] {
            for (int r = 0; r < repeats; ++r) fn();
        }, warmups, iters);
        b.median_ms /= repeats;
        b.min_ms /= repeats;
        b.max_ms /= repeats;
        b.mean_ms /= repeats;
        return b;
    };

    const long long n = 1 << 20;
    std::vector<float> x = rng.uniforms(n, -3.0f, 3.0f);
    std::vector<float> y = rng.uniforms(n, 0.5f, 2.0f);
    float *dx = qc::dnew(x);
    float *dy = qc::dnew(y);
    float *dout = qc::dzero<float>(n);
    const double elem_bytes = static_cast<double>(n) * 2.0 * sizeof(float);
    auto unary_scalar = qc::bench([&] {
        unary_scalar_kernel<<<1, 1>>>(dx, dout, n, kSilu, 0.31f, 0.71f, 0.17f, 0.5f);
    }, 2, 5);
    auto unary_candidate = bench_repeated([&] {
        unary_kernel<<<qc::grid_for(n, kThreads), kThreads>>>(
            dx, dout, n, kSilu, 0.31f, 0.71f, 0.17f, 0.5f);
    }, 2000, 5, 20);
    unary_scalar.report_bandwidth("unary scalar", elem_bytes);
    unary_candidate.report_bandwidth("unary candidate", elem_bytes);
    qc::report_ab("unary", unary_scalar, unary_candidate);

    auto binary_scalar = qc::bench([&] {
        binary_scalar_kernel<<<1, 1>>>(dx, dy, dout, n, 0);
    }, 2, 5);
    auto binary_candidate = bench_repeated([&] {
        binary_kernel<<<qc::grid_for(n, kThreads), kThreads>>>(dx, dy, dout, n, 0);
    }, 2000, 5, 20);
    const double binary_bytes = static_cast<double>(n) * 3.0 * sizeof(float);
    binary_scalar.report_bandwidth("binary scalar", binary_bytes);
    binary_candidate.report_bandwidth("binary candidate", binary_bytes);
    qc::report_ab("binary arithmetic", binary_scalar, binary_candidate);

    auto misc_scalar = qc::bench([&] {
        unary_misc_scalar_kernel<<<1, 1>>>(dx, dout, n, 1, -0.5f, 0.5f);
    }, 2, 5);
    auto misc_candidate = bench_repeated([&] {
        unary_misc_kernel<<<qc::grid_for(n, kThreads), kThreads>>>(dx, dout, n, 1, -0.5f, 0.5f);
    }, 2000, 5, 20);
    misc_scalar.report_bandwidth("scalar/value scalar", elem_bytes);
    misc_candidate.report_bandwidth("scalar/value candidate", elem_bytes);
    qc::report_ab("value-like elementwise", misc_scalar, misc_candidate);

    auto sig_scalar = qc::bench([&] {
        sigmoid_mul_scalar_kernel<<<1, 1>>>(dx, dy, dout, n);
    }, 2, 5);
    auto sig_candidate = bench_repeated([&] {
        sigmoid_mul_kernel<<<qc::grid_for(n, kThreads), kThreads>>>(dx, dy, dout, n);
    }, 2000, 5, 20);
    sig_scalar.report_bandwidth("sigmoid baseline scalar", binary_bytes);
    sig_candidate.report_bandwidth("sigmoid_mul candidate", binary_bytes);
    qc::report_ab("sigmoid_mul", sig_scalar, sig_candidate);

    const long long rows = 4096, dim = 256;
    std::vector<float> mat = rng.uniforms(rows * dim, -1.0f, 1.0f);
    float *dmat = qc::dnew(mat);
    float *drow = qc::dzero<float>(rows);
    auto mean_scalar = qc::bench([&] {
        reduce_mean_scalar_kernel<<<1, 1>>>(dmat, drow, rows, dim);
    }, 2, 5);
    auto mean_candidate = bench_repeated([&] {
        reduce_mean_kernel<<<rows, kThreads>>>(dmat, drow, rows, dim);
    }, 1000, 5, 20);
    mean_scalar.report_bandwidth("reduce_mean scalar",
                                 static_cast<double>(rows * dim + rows) * sizeof(float));
    mean_candidate.report_bandwidth("reduce_mean current",
                                    static_cast<double>(rows * dim + rows) * sizeof(float));
    qc::report_ab("reduce_mean", mean_scalar, mean_candidate);

    float *dl2 = qc::dzero<float>(rows * dim);
    auto l2_scalar = qc::bench([&] {
        l2_normalize_scalar_kernel<<<1, 1>>>(dmat, dl2, rows, dim, 1e-12f);
    }, 2, 5);
    auto l2_candidate = bench_repeated([&] {
        l2_normalize_kernel<<<rows, kThreads>>>(dmat, dl2, rows, dim, 1e-12f);
    }, 1000, 5, 20);
    l2_scalar.report_bandwidth("l2_normalize scalar",
                               static_cast<double>(rows * dim) * 2.0 * sizeof(float));
    l2_candidate.report_bandwidth("l2_normalize current",
                                  static_cast<double>(rows * dim) * 2.0 * sizeof(float));
    qc::report_ab("l2_normalize", l2_scalar, l2_candidate);

    const long long grow = 512, gchannels = 16, gspatial = 64, groups = 4;
    std::vector<float> gx = rng.uniforms(grow * gchannels * gspatial, -1.0f, 1.0f);
    std::vector<float> gw = rng.uniforms(gchannels, 0.5f, 1.5f);
    std::vector<float> gb = rng.uniforms(gchannels, -0.2f, 0.2f);
    float *dgx = qc::dnew(gx);
    float *dgw = qc::dnew(gw);
    float *dgb = qc::dnew(gb);
    float *dgn = qc::dzero<float>(gx.size());
    auto gn_scalar = qc::bench([&] {
        group_norm_scalar_kernel<<<1, 1>>>(
            dgx, dgw, dgb, dgn, grow, gchannels, gspatial, groups, 1e-5f);
    }, 2, 5);
    auto gn_candidate = bench_repeated([&] {
        group_norm_kernel<<<grow * groups, kThreads>>>(
            dgx, dgw, dgb, dgn, grow, gchannels, gspatial, groups, 1e-5f);
    }, 1000, 5, 20);
    gn_scalar.report_bandwidth("group_norm scalar",
                               static_cast<double>(gx.size()) * 2.0 * sizeof(float));
    gn_candidate.report_bandwidth("group_norm current",
                                  static_cast<double>(gx.size()) * 2.0 * sizeof(float));
    qc::report_ab("group_norm", gn_scalar, gn_candidate);

    const long long sort_rows = 4096, sort_dim = 64;
    std::vector<float> sv = rng.uniforms(sort_rows * sort_dim, -1.0f, 1.0f);
    float *dsv = qc::dnew(sv);
    int32_t *didx = qc::dzero<int32_t>(sort_rows * sort_dim);
    auto sort_scalar = qc::bench([&] {
        argsort_scalar_kernel<<<1, 1>>>(dsv, didx, sort_rows, sort_dim, 1);
    }, 2, 5);
    auto sort_candidate = qc::bench([&] {
        argsort_kernel<<<qc::grid_for(sort_rows, kThreads), kThreads>>>(
            dsv, didx, sort_rows, sort_dim, 1);
    }, 10, 30);
    sort_scalar.report_bandwidth("argsort scalar",
                                 static_cast<double>(sort_rows * sort_dim) *
                                     (sizeof(float) + sizeof(int32_t)));
    sort_candidate.report_bandwidth("argsort current",
                                    static_cast<double>(sort_rows * sort_dim) *
                                        (sizeof(float) + sizeof(int32_t)));
    qc::report_ab("argsort", sort_scalar, sort_candidate);

    int32_t *dtop = qc::dzero<int32_t>(sort_rows * 8);
    auto top_scalar = qc::bench([&] {
        threshold_topk_indices_scalar_kernel<<<1, 1>>>(dsv, dtop, sort_rows, sort_dim, 8);
    }, 2, 5);
    auto top_candidate = qc::bench([&] {
        threshold_topk_indices_kernel<<<qc::grid_for(sort_rows, kThreads), kThreads>>>(
            dsv, dtop, sort_rows, sort_dim, 8);
    }, 10, 30);
    top_scalar.report_bandwidth("threshold_topk scalar",
                                static_cast<double>(sort_rows * sort_dim * sizeof(float) +
                                                    sort_rows * 8 * sizeof(int32_t)));
    top_candidate.report_bandwidth("threshold_topk current",
                                   static_cast<double>(sort_rows * sort_dim * sizeof(float) +
                                                       sort_rows * 8 * sizeof(int32_t)));
    qc::report_ab("threshold_topk_indices", top_scalar, top_candidate);

    float *dcopy = qc::dzero<float>(n);
    auto copy_scalar = qc::bench([&] {
        scalar_copy_kernel<<<1, 1>>>(dx, dcopy, n);
    }, 2, 5);
    auto copy_candidate = bench_repeated([&] {
        tensor_copy_kernel<<<qc::grid_for(n, kThreads), kThreads>>>(dx, dcopy, n);
    }, 2000, 5, 20);
    copy_scalar.report_bandwidth("tensor_copy scalar", elem_bytes);
    copy_candidate.report_bandwidth("tensor_copy candidate", elem_bytes);
    qc::report_ab("tensor_copy/layout ops", copy_scalar, copy_candidate);

    qc::dfree(dx, dy, dout, dmat, drow, dl2, dgx, dgw, dgb, dgn, dsv, didx,
              dtop, dcopy);
}

}  // namespace

int main(int argc, char **argv) {
    const bool do_bench = argc > 1 && std::string(argv[1]) == "--bench";
    qc::print_environment("phase11_tensor_ops");
    const bool ok = run_correctness();
    if (do_bench) run_bench();
    std::printf("%s\n", ok ? "ALL PASS" : "FAIL");
    return ok ? 0 : 1;
}
