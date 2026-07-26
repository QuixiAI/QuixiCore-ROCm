/**
 * @file
 * @brief Phase 3 CPU/Metal parity ports for decode matmul epilogues on CDNA3.
 *
 * Torch-free standalone harness. Correctness uses fp64 host oracles and
 * tolerances from kernels/common/cdna3_harness.cuh.
 */
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

#include "../../../../common/cdna3_harness.cuh"
#include "../../../../quantization/qgemv/variants/rocm_cdna3/quant_formats_tables.cuh"

namespace {

enum Activation : int {
    kNone = 0,
    kGeluErf = 1,
    kGeluTanh = 2,
    kSilu = 3,
    kRelu2 = 4,
};

template <typename S>
using storage_t = typename S::storage;

__device__ __host__ inline float gelu_erf_f(float x) {
#ifdef __HIP_DEVICE_COMPILE__
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865475f));
#else
    return 0.5f * x * (1.0f + std::erf(x * 0.7071067811865475f));
#endif
}

__device__ __host__ inline float gelu_tanh_f(float x) {
    constexpr float k = 0.7978845608028654f;
    constexpr float a = 0.044715f;
#ifdef __HIP_DEVICE_COMPILE__
    return 0.5f * x * (1.0f + tanhf(k * (x + a * x * x * x)));
#else
    return 0.5f * x * (1.0f + std::tanh(k * (x + a * x * x * x)));
#endif
}

__device__ __host__ inline float silu_f(float x) {
#ifdef __HIP_DEVICE_COMPILE__
    return x / (1.0f + expf(-x));
#else
    return x / (1.0f + std::exp(-x));
#endif
}

__device__ __host__ inline float activate_f(float x, int activation) {
    switch (activation) {
        case kGeluErf:
            return gelu_erf_f(x);
        case kGeluTanh:
            return gelu_tanh_f(x);
        case kSilu:
            return silu_f(x);
        case kRelu2:
            return x > 0.0f ? x * x : 0.0f;
        case kNone:
        default:
            return x;
    }
}

template <typename S>
__device__ __forceinline__ float load_s(const storage_t<S> *p, size_t i) {
    return S::to_float(p[i]);
}

template <typename S>
__global__ void linear_epilogue_wave64(
    const storage_t<S> *__restrict__ x,
    const storage_t<S> *__restrict__ weight,
    const storage_t<S> *__restrict__ bias,
    const storage_t<S> *__restrict__ residual,
    storage_t<S> *__restrict__ y, int rows, int input_dim, int output_dim,
    int activation, int use_bias, int use_residual,
    int residual_after_activation) {
    const int output = blockIdx.x;
    const int row = blockIdx.y;
    const int lane = threadIdx.x & 63;
    if (row >= rows || output >= output_dim) return;
    float acc = 0.0f;
    const size_t x_base = size_t(row) * input_dim;
    const size_t w_base = size_t(output) * input_dim;
    for (int input = lane; input < input_dim; input += 64) {
        acc += load_s<S>(x, x_base + input) * load_s<S>(weight, w_base + input);
    }
    acc = qc::wave_reduce_sum(acc);
    if (lane == 0) {
        const size_t out_idx = size_t(row) * output_dim + output;
        if (use_bias) acc += load_s<S>(bias, output);
        if (!residual_after_activation && use_residual) {
            acc += load_s<S>(residual, out_idx);
        }
        acc = activate_f(acc, activation);
        if (residual_after_activation && use_residual) {
            acc += load_s<S>(residual, out_idx);
        }
        y[out_idx] = S::from_float(acc);
    }
}

template <typename S>
__global__ void linear_epilogue_scalar(
    const storage_t<S> *__restrict__ x,
    const storage_t<S> *__restrict__ weight,
    const storage_t<S> *__restrict__ bias,
    const storage_t<S> *__restrict__ residual,
    storage_t<S> *__restrict__ y, int rows, int input_dim, int output_dim,
    int activation, int use_bias, int use_residual,
    int residual_after_activation) {
    const int output = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    if (row >= rows || output >= output_dim) return;
    float acc = 0.0f;
    const size_t x_base = size_t(row) * input_dim;
    const size_t w_base = size_t(output) * input_dim;
    for (int input = 0; input < input_dim; ++input) {
        acc += load_s<S>(x, x_base + input) * load_s<S>(weight, w_base + input);
    }
    const size_t out_idx = size_t(row) * output_dim + output;
    if (use_bias) acc += load_s<S>(bias, output);
    if (!residual_after_activation && use_residual) acc += load_s<S>(residual, out_idx);
    acc = activate_f(acc, activation);
    if (residual_after_activation && use_residual) acc += load_s<S>(residual, out_idx);
    y[out_idx] = S::from_float(acc);
}

__device__ __host__ inline float half_bits_to_float(const uint8_t *p) {
    uint16_t bits = uint16_t(p[0]) | (uint16_t(p[1]) << 8);
#ifdef __HIP_DEVICE_COMPILE__
    return __half2float(__ushort_as_half(bits));
#else
    __half h;
    std::memcpy(&h, &bits, sizeof(bits));
    return __half2float(h);
#endif
}

template <typename FMT>
__device__ __forceinline__ float packed_dequant(const uint8_t *packed,
                                                int output, int input,
                                                int input_dim) {
    const int blocks = input_dim / FMT::block_k;
    const int block = input / FMT::block_k;
    const int column = input - block * FMT::block_k;
    const uint8_t *base =
        packed + (size_t(output) * blocks + block) * size_t(FMT::block_bytes);
    return FMT::dequant(base, column);
}

__device__ __forceinline__ float q8_0_dequant(const uint8_t *packed, int output,
                                               int input, int input_dim) {
    return packed_dequant<tmq::q8_0>(packed, output, input, input_dim);
}

template <typename S>
__global__ void q8_epilogue_wave64(
    const storage_t<S> *__restrict__ x, const uint8_t *__restrict__ weight,
    const storage_t<S> *__restrict__ bias,
    const storage_t<S> *__restrict__ residual, storage_t<S> *__restrict__ y,
    int rows, int input_dim, int output_dim, int activation, int use_bias,
    int use_residual, int residual_after_activation) {
    const int output = blockIdx.x;
    const int row = blockIdx.y;
    const int lane = threadIdx.x & 63;
    if (row >= rows || output >= output_dim) return;
    float acc = 0.0f;
    const size_t x_base = size_t(row) * input_dim;
    for (int input = lane; input < input_dim; input += 64) {
        acc += load_s<S>(x, x_base + input) *
               q8_0_dequant(weight, output, input, input_dim);
    }
    acc = qc::wave_reduce_sum(acc);
    if (lane == 0) {
        const size_t out_idx = size_t(row) * output_dim + output;
        if (use_bias) acc += load_s<S>(bias, output);
        if (!residual_after_activation && use_residual) {
            acc += load_s<S>(residual, out_idx);
        }
        acc = activate_f(acc, activation);
        if (residual_after_activation && use_residual) {
            acc += load_s<S>(residual, out_idx);
        }
        y[out_idx] = S::from_float(acc);
    }
}

template <typename S, typename FMT>
__global__ void packed_epilogue_wave64(
    const storage_t<S> *__restrict__ x, const uint8_t *__restrict__ weight,
    const storage_t<S> *__restrict__ bias,
    const storage_t<S> *__restrict__ residual, storage_t<S> *__restrict__ y,
    int rows, int input_dim, int output_dim, int activation, int use_bias,
    int use_residual, int residual_after_activation) {
    const int output = blockIdx.x;
    const int row = blockIdx.y;
    const int lane = threadIdx.x & 63;
    if (row >= rows || output >= output_dim) return;
    float acc = 0.0f;
    const size_t x_base = size_t(row) * input_dim;
    for (int input = lane; input < input_dim; input += 64) {
        acc += load_s<S>(x, x_base + input) *
               packed_dequant<FMT>(weight, output, input, input_dim);
    }
    acc = qc::wave_reduce_sum(acc);
    if (lane == 0) {
        const size_t out_idx = size_t(row) * output_dim + output;
        if (use_bias) acc += load_s<S>(bias, output);
        if (!residual_after_activation && use_residual) {
            acc += load_s<S>(residual, out_idx);
        }
        acc = activate_f(acc, activation);
        if (residual_after_activation && use_residual) {
            acc += load_s<S>(residual, out_idx);
        }
        y[out_idx] = S::from_float(acc);
    }
}

template <typename S, typename FMT>
__global__ void packed_epilogue_scalar(
    const storage_t<S> *__restrict__ x, const uint8_t *__restrict__ weight,
    const storage_t<S> *__restrict__ bias,
    const storage_t<S> *__restrict__ residual, storage_t<S> *__restrict__ y,
    int rows, int input_dim, int output_dim, int activation, int use_bias,
    int use_residual, int residual_after_activation) {
    const int output = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    if (row >= rows || output >= output_dim) return;
    float acc = 0.0f;
    const size_t x_base = size_t(row) * input_dim;
    for (int input = 0; input < input_dim; ++input) {
        acc += load_s<S>(x, x_base + input) *
               packed_dequant<FMT>(weight, output, input, input_dim);
    }
    const size_t out_idx = size_t(row) * output_dim + output;
    if (use_bias) acc += load_s<S>(bias, output);
    if (!residual_after_activation && use_residual) acc += load_s<S>(residual, out_idx);
    acc = activate_f(acc, activation);
    if (residual_after_activation && use_residual) acc += load_s<S>(residual, out_idx);
    y[out_idx] = S::from_float(acc);
}

template <typename S>
__global__ void swiglu_dense_wave64(
    const storage_t<S> *__restrict__ x,
    const storage_t<S> *__restrict__ gate_weight,
    const storage_t<S> *__restrict__ up_weight,
    const storage_t<S> *__restrict__ gate_bias,
    const storage_t<S> *__restrict__ up_bias, storage_t<S> *__restrict__ y,
    int rows, int input_dim, int output_dim, int use_bias) {
    const int output = blockIdx.x;
    const int row = blockIdx.y;
    const int lane = threadIdx.x & 63;
    if (row >= rows || output >= output_dim) return;
    float gate = 0.0f;
    float up = 0.0f;
    const size_t x_base = size_t(row) * input_dim;
    const size_t w_base = size_t(output) * input_dim;
    for (int input = lane; input < input_dim; input += 64) {
        const float xv = load_s<S>(x, x_base + input);
        gate += xv * load_s<S>(gate_weight, w_base + input);
        up += xv * load_s<S>(up_weight, w_base + input);
    }
    gate = qc::wave_reduce_sum(gate);
    up = qc::wave_reduce_sum(up);
    if (lane == 0) {
        if (use_bias) {
            gate += load_s<S>(gate_bias, output);
            up += load_s<S>(up_bias, output);
        }
        y[size_t(row) * output_dim + output] = S::from_float(silu_f(gate) * up);
    }
}

template <typename S>
__global__ void swiglu_q8_wave64(
    const storage_t<S> *__restrict__ x, const uint8_t *__restrict__ gate_weight,
    const uint8_t *__restrict__ up_weight,
    const storage_t<S> *__restrict__ gate_bias,
    const storage_t<S> *__restrict__ up_bias, storage_t<S> *__restrict__ y,
    int rows, int input_dim, int output_dim, int use_bias) {
    const int output = blockIdx.x;
    const int row = blockIdx.y;
    const int lane = threadIdx.x & 63;
    if (row >= rows || output >= output_dim) return;
    float gate = 0.0f;
    float up = 0.0f;
    const size_t x_base = size_t(row) * input_dim;
    for (int input = lane; input < input_dim; input += 64) {
        const float xv = load_s<S>(x, x_base + input);
        gate += xv * q8_0_dequant(gate_weight, output, input, input_dim);
        up += xv * q8_0_dequant(up_weight, output, input, input_dim);
    }
    gate = qc::wave_reduce_sum(gate);
    up = qc::wave_reduce_sum(up);
    if (lane == 0) {
        if (use_bias) {
            gate += load_s<S>(gate_bias, output);
            up += load_s<S>(up_bias, output);
        }
        y[size_t(row) * output_dim + output] = S::from_float(silu_f(gate) * up);
    }
}

template <typename S, typename FMT>
__global__ void swiglu_packed_wave64(
    const storage_t<S> *__restrict__ x, const uint8_t *__restrict__ gate_weight,
    const uint8_t *__restrict__ up_weight,
    const storage_t<S> *__restrict__ gate_bias,
    const storage_t<S> *__restrict__ up_bias, storage_t<S> *__restrict__ y,
    int rows, int input_dim, int output_dim, int use_bias) {
    const int output = blockIdx.x;
    const int row = blockIdx.y;
    const int lane = threadIdx.x & 63;
    if (row >= rows || output >= output_dim) return;
    float gate = 0.0f;
    float up = 0.0f;
    const size_t x_base = size_t(row) * input_dim;
    for (int input = lane; input < input_dim; input += 64) {
        const float xv = load_s<S>(x, x_base + input);
        gate += xv * packed_dequant<FMT>(gate_weight, output, input, input_dim);
        up += xv * packed_dequant<FMT>(up_weight, output, input, input_dim);
    }
    gate = qc::wave_reduce_sum(gate);
    up = qc::wave_reduce_sum(up);
    if (lane == 0) {
        if (use_bias) {
            gate += load_s<S>(gate_bias, output);
            up += load_s<S>(up_bias, output);
        }
        y[size_t(row) * output_dim + output] = S::from_float(silu_f(gate) * up);
    }
}

template <typename S, typename FMT>
__global__ void swiglu_packed_scalar(
    const storage_t<S> *__restrict__ x, const uint8_t *__restrict__ gate_weight,
    const uint8_t *__restrict__ up_weight,
    const storage_t<S> *__restrict__ gate_bias,
    const storage_t<S> *__restrict__ up_bias, storage_t<S> *__restrict__ y,
    int rows, int input_dim, int output_dim, int use_bias) {
    const int output = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    if (row >= rows || output >= output_dim) return;
    float gate = 0.0f;
    float up = 0.0f;
    const size_t x_base = size_t(row) * input_dim;
    for (int input = 0; input < input_dim; ++input) {
        const float xv = load_s<S>(x, x_base + input);
        gate += xv * packed_dequant<FMT>(gate_weight, output, input, input_dim);
        up += xv * packed_dequant<FMT>(up_weight, output, input, input_dim);
    }
    if (use_bias) {
        gate += load_s<S>(gate_bias, output);
        up += load_s<S>(up_bias, output);
    }
    y[size_t(row) * output_dim + output] = S::from_float(silu_f(gate) * up);
}

template <typename S>
__global__ void gemm_gate_residual_wave64(
    const storage_t<S> *__restrict__ x,
    const storage_t<S> *__restrict__ weight,
    const storage_t<S> *__restrict__ bias, const storage_t<S> *__restrict__ gate,
    const storage_t<S> *__restrict__ residual, storage_t<S> *__restrict__ y,
    int rows, int output_dim, int input_dim, int use_bias, int use_gate,
    int use_residual) {
    const int output = blockIdx.x;
    const int row = blockIdx.y;
    const int lane = threadIdx.x & 63;
    if (row >= rows || output >= output_dim) return;
    float acc = 0.0f;
    for (int input = lane; input < input_dim; input += 64) {
        acc += load_s<S>(x, size_t(row) * input_dim + input) *
               load_s<S>(weight, size_t(input) * output_dim + output);
    }
    acc = qc::wave_reduce_sum(acc);
    if (lane == 0) {
        if (use_bias) acc += load_s<S>(bias, output);
        if (use_gate) acc *= load_s<S>(gate, output);
        const size_t out_idx = size_t(row) * output_dim + output;
        if (use_residual) acc += load_s<S>(residual, out_idx);
        y[out_idx] = S::from_float(acc);
    }
}

__global__ void lora_apply_direct_f16_wave64(
    const float *__restrict__ x, const __half *__restrict__ adapter_a,
    const __half *__restrict__ adapter_b, const float *__restrict__ base,
    float *__restrict__ out, int rows, int input_dim, int output_dim, int rank,
    float scale, int use_base) {
    const int output = blockIdx.x;
    const int row = blockIdx.y;
    const int lane = threadIdx.x & 63;
    if (row >= rows || output >= output_dim) return;
    double unavailable_on_device = 0.0;
    (void)unavailable_on_device;
    float sum = 0.0f;
    for (int r = 0; r < rank; ++r) {
        float low = 0.0f;
        for (int input = lane; input < input_dim; input += 64) {
            low += x[size_t(row) * input_dim + input] *
                   __half2float(adapter_a[size_t(r) * input_dim + input]);
        }
        low = qc::wave_reduce_sum(low);
        if (lane == 0) {
            const float rounded_low = __half2float(__float2half(low));
            sum += rounded_low *
                   __half2float(adapter_b[size_t(output) * rank + r]);
        }
    }
    if (lane == 0) {
        const float delta = __half2float(__float2half(sum));
        const size_t idx = size_t(row) * output_dim + output;
        out[idx] = (use_base ? base[idx] : 0.0f) + scale * delta;
    }
}

__global__ void complex_gemm_wave64(
    const float *__restrict__ a_real, const float *__restrict__ a_imag,
    const float *__restrict__ b_real, const float *__restrict__ b_imag,
    float *__restrict__ c_real, float *__restrict__ c_imag, int m, int n,
    int k) {
    const int col = blockIdx.x;
    const int row = blockIdx.y;
    const int lane = threadIdx.x & 63;
    if (row >= m || col >= n) return;
    float real = 0.0f;
    float imag = 0.0f;
    for (int inner = lane; inner < k; inner += 64) {
        const size_t ai = size_t(row) * k + inner;
        const size_t bi = size_t(inner) * n + col;
        const float ar = a_real[ai];
        const float ai_v = a_imag[ai];
        const float br = b_real[bi];
        const float bi_v = b_imag[bi];
        real += ar * br - ai_v * bi_v;
        imag += ar * bi_v + ai_v * br;
    }
    real = qc::wave_reduce_sum(real);
    imag = qc::wave_reduce_sum(imag);
    if (lane == 0) {
        c_real[size_t(row) * n + col] = real;
        c_imag[size_t(row) * n + col] = imag;
    }
}

__global__ void grouped_gemm_wave64(const float *__restrict__ a,
                                    const float *__restrict__ b,
                                    float *__restrict__ c, int groups, int m,
                                    int n, int k) {
    const int col = blockIdx.x;
    const int row = blockIdx.y;
    const int group = blockIdx.z;
    const int lane = threadIdx.x & 63;
    if (group >= groups || row >= m || col >= n) return;
    const float *a_row = a + (size_t(group) * m + row) * k;
    const float *b_group = b + size_t(group) * k * n;
    float acc = 0.0f;
    for (int inner = lane; inner < k; inner += 64) {
        acc += a_row[inner] * b_group[size_t(inner) * n + col];
    }
    acc = qc::wave_reduce_sum(acc);
    if (lane == 0) c[(size_t(group) * m + row) * n + col] = acc;
}

__global__ void grouped_gemm_scalar(const float *__restrict__ a,
                                    const float *__restrict__ b,
                                    float *__restrict__ c, int groups, int m,
                                    int n, int k) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    const int group = blockIdx.z;
    if (group >= groups || row >= m || col >= n) return;
    const float *a_row = a + (size_t(group) * m + row) * k;
    const float *b_group = b + size_t(group) * k * n;
    float acc = 0.0f;
    for (int inner = 0; inner < k; ++inner) {
        acc += a_row[inner] * b_group[size_t(inner) * n + col];
    }
    c[(size_t(group) * m + row) * n + col] = acc;
}

template <typename S>
std::vector<storage_t<S>> as_storage(const std::vector<float> &src) {
    return qc::to_storage<storage_t<S>>(src);
}

template <typename T>
double hval(const std::vector<T> &v, size_t idx) {
    return qc::to_double(v[idx]);
}

double host_activate(double x, int activation) {
    switch (activation) {
        case kGeluErf:
            return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0)));
        case kGeluTanh: {
            constexpr double k = 0.7978845608028654;
            constexpr double a = 0.044715;
            return 0.5 * x * (1.0 + std::tanh(k * (x + a * x * x * x)));
        }
        case kSilu:
            return x / (1.0 + std::exp(-x));
        case kRelu2:
            return x > 0.0 ? x * x : 0.0;
        case kNone:
        default:
            return x;
    }
}

template <typename S>
std::vector<double> ref_linear(
    const std::vector<storage_t<S>> &x, const std::vector<storage_t<S>> &w,
    const std::vector<storage_t<S>> &bias,
    const std::vector<storage_t<S>> &residual, int rows, int input_dim,
    int output_dim, int activation, bool use_bias, bool use_residual,
    bool residual_after_activation) {
    std::vector<double> ref(size_t(rows) * output_dim);
    for (int r = 0; r < rows; ++r) {
        for (int o = 0; o < output_dim; ++o) {
            double acc = 0.0;
            for (int i = 0; i < input_dim; ++i) {
                acc += hval(x, size_t(r) * input_dim + i) *
                       hval(w, size_t(o) * input_dim + i);
            }
            const size_t out_idx = size_t(r) * output_dim + o;
            if (use_bias) acc += hval(bias, o);
            if (!residual_after_activation && use_residual) acc += hval(residual, out_idx);
            acc = host_activate(acc, activation);
            if (residual_after_activation && use_residual) acc += hval(residual, out_idx);
            ref[out_idx] = acc;
        }
    }
    return ref;
}

void store_half_bytes(uint8_t *dst, float value) {
    __half h = __float2half(value);
    std::memcpy(dst, &h, sizeof(h));
}

float load_half_bytes_host(const uint8_t *src) {
    __half h;
    std::memcpy(&h, src, sizeof(h));
    return __half2float(h);
}

enum class PackedFormat {
    kQ4_0,
    kQ8_0,
    kQ6_K,
    kMxFp8,
    kNvFp4,
    kMxFp4,
};

const char *packed_format_name(PackedFormat fmt) {
    switch (fmt) {
        case PackedFormat::kQ4_0:
            return "q4_0";
        case PackedFormat::kQ8_0:
            return "q8_0";
        case PackedFormat::kQ6_K:
            return "q6_K";
        case PackedFormat::kMxFp8:
            return "mxfp8";
        case PackedFormat::kNvFp4:
            return "nvfp4";
        case PackedFormat::kMxFp4:
            return "mxfp4";
    }
    return "unknown";
}

int packed_block_k(PackedFormat fmt) {
    switch (fmt) {
        case PackedFormat::kQ4_0:
            return tmq::q4_0::block_k;
        case PackedFormat::kQ8_0:
            return tmq::q8_0::block_k;
        case PackedFormat::kQ6_K:
            return tmq::q6_K::block_k;
        case PackedFormat::kMxFp8:
            return tmq::mxfp8::block_k;
        case PackedFormat::kNvFp4:
            return tmq::nvfp4::block_k;
        case PackedFormat::kMxFp4:
            return tmq::mxfp4::block_k;
    }
    return 0;
}

int packed_block_bytes(PackedFormat fmt) {
    switch (fmt) {
        case PackedFormat::kQ4_0:
            return tmq::q4_0::block_bytes;
        case PackedFormat::kQ8_0:
            return tmq::q8_0::block_bytes;
        case PackedFormat::kQ6_K:
            return tmq::q6_K::block_bytes;
        case PackedFormat::kMxFp8:
            return tmq::mxfp8::block_bytes;
        case PackedFormat::kNvFp4:
            return tmq::nvfp4::block_bytes;
        case PackedFormat::kMxFp4:
            return tmq::mxfp4::block_bytes;
    }
    return 0;
}

void require_packed_shape(PackedFormat fmt, int input_dim) {
    const int block_k = packed_block_k(fmt);
    if (block_k <= 0 || input_dim % block_k != 0) {
        std::fprintf(stderr, "%s input_dim=%d is not a multiple of block_k=%d\n",
                     packed_format_name(fmt), input_dim, block_k);
        std::abort();
    }
}

double e2m1_decode_host(unsigned nib) {
    static constexpr double values[8] = {0.0, 0.5, 1.0, 1.5,
                                         2.0, 3.0, 4.0, 6.0};
    const double mag = values[nib & 7u];
    return (nib & 8u) ? -mag : mag;
}

double e8m0_decode_host(uint8_t e) {
    if (e == 0) return 0.0;
    return std::ldexp(1.0, int(e) - 127);
}

void fill_nibbles(uint8_t *qs, int bytes, qc::Rng &rng) {
    for (int i = 0; i < bytes; ++i) {
        const int lo = rng.integer(0, 15);
        const int hi = rng.integer(0, 15);
        qs[i] = uint8_t(lo | (hi << 4));
    }
}

std::vector<uint8_t> pack_q8_0(const std::vector<float> &weights, int rows,
                               int input_dim) {
    const int blocks = input_dim / 32;
    std::vector<uint8_t> packed(size_t(rows) * blocks * 34);
    for (int r = 0; r < rows; ++r) {
        for (int b = 0; b < blocks; ++b) {
            const float *src = weights.data() + size_t(r) * input_dim + b * 32;
            float amax = 0.0f;
            for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(src[j]));
            const float d = amax / 127.0f;
            const float inv = d != 0.0f ? 1.0f / d : 0.0f;
            uint8_t *dst = packed.data() + (size_t(r) * blocks + b) * 34;
            store_half_bytes(dst, d);
            int8_t *codes = reinterpret_cast<int8_t *>(dst + 2);
            for (int j = 0; j < 32; ++j) {
                float q = std::nearbyint(src[j] * inv);
                q = std::max(-127.0f, std::min(127.0f, q));
                codes[j] = static_cast<int8_t>(q);
            }
        }
    }
    return packed;
}

double q8_0_dequant_host(const std::vector<uint8_t> &packed, int output,
                         int input, int input_dim) {
    const int blocks = input_dim / 32;
    const int block = input >> 5;
    const int column = input & 31;
    const uint8_t *base =
        packed.data() + (size_t(output) * blocks + block) * 34;
    return double(load_half_bytes_host(base)) *
           double(reinterpret_cast<const int8_t *>(base + 2)[column]);
}

std::vector<uint8_t> pack_format_blocks(PackedFormat fmt, int rows,
                                        int input_dim, qc::Rng &rng) {
    require_packed_shape(fmt, input_dim);
    const int block_k = packed_block_k(fmt);
    const int block_bytes = packed_block_bytes(fmt);
    const int blocks = input_dim / block_k;
    std::vector<uint8_t> packed(size_t(rows) * blocks * block_bytes);
    static constexpr uint8_t kE4m3Codes[] = {
        0x00, 0x18, 0x20, 0x28, 0x30, 0x38, 0x40, 0x48, 0x50,
        0x80, 0x98, 0xa0, 0xa8, 0xb0, 0xb8, 0xc0, 0xc8, 0xd0,
    };
    static constexpr uint8_t kNvScales[] = {0x20, 0x28, 0x30, 0x38};
    for (int row = 0; row < rows; ++row) {
        for (int block = 0; block < blocks; ++block) {
            uint8_t *base =
                packed.data() + (size_t(row) * blocks + block) * block_bytes;
            switch (fmt) {
                case PackedFormat::kQ4_0:
                    store_half_bytes(base, rng.uniform(0.004f, 0.025f));
                    fill_nibbles(base + 2, 16, rng);
                    break;
                case PackedFormat::kQ8_0: {
                    store_half_bytes(base, rng.uniform(0.001f, 0.009f));
                    int8_t *codes = reinterpret_cast<int8_t *>(base + 2);
                    for (int i = 0; i < 32; ++i) {
                        codes[i] = static_cast<int8_t>(rng.integer(-80, 80));
                    }
                    break;
                }
                case PackedFormat::kQ6_K: {
                    std::fill(base, base + tmq::q6_K::block_bytes, uint8_t(0));
                    int8_t *scales = reinterpret_cast<int8_t *>(base + 192);
                    for (int i = 0; i < 16; ++i) {
                        int s = rng.integer(-6, 6);
                        if (s == 0) s = 1;
                        scales[i] = static_cast<int8_t>(s);
                    }
                    store_half_bytes(base + 208, rng.uniform(0.0008f, 0.0035f));
                    for (int col = 0; col < 256; ++col) {
                        const int code = rng.integer(0, 63);
                        const int chunk = col >> 7;
                        const int pos = col & 127;
                        const int group = pos >> 5;
                        const int lane = pos & 31;
                        const int ql_off = chunk * 64 + lane + 32 * (group & 1);
                        if (group & 2) {
                            base[ql_off] = uint8_t(base[ql_off] | ((code & 0xF) << 4));
                        } else {
                            base[ql_off] = uint8_t(base[ql_off] | (code & 0xF));
                        }
                        base[128 + chunk * 32 + lane] = uint8_t(
                            base[128 + chunk * 32 + lane] |
                            (((code >> 4) & 3) << (2 * group)));
                    }
                    break;
                }
                case PackedFormat::kMxFp8:
                    base[0] = static_cast<uint8_t>(rng.integer(120, 124));
                    for (int i = 0; i < 32; ++i) {
                        base[1 + i] =
                            kE4m3Codes[rng.integer(
                                0, int(sizeof(kE4m3Codes) / sizeof(kE4m3Codes[0])) - 1)];
                    }
                    break;
                case PackedFormat::kNvFp4:
                    base[0] =
                        kNvScales[rng.integer(
                            0, int(sizeof(kNvScales) / sizeof(kNvScales[0])) - 1)];
                    fill_nibbles(base + 1, 8, rng);
                    break;
                case PackedFormat::kMxFp4:
                    base[0] = static_cast<uint8_t>(rng.integer(120, 124));
                    fill_nibbles(base + 1, 16, rng);
                    break;
            }
        }
    }
    return packed;
}

double packed_dequant_host(PackedFormat fmt, const std::vector<uint8_t> &packed,
                           int output, int input, int input_dim) {
    require_packed_shape(fmt, input_dim);
    const int block_k = packed_block_k(fmt);
    const int block_bytes = packed_block_bytes(fmt);
    const int blocks = input_dim / block_k;
    const int block = input / block_k;
    const int col = input - block * block_k;
    const uint8_t *base =
        packed.data() + (size_t(output) * blocks + block) * block_bytes;
    switch (fmt) {
        case PackedFormat::kQ4_0: {
            const uint8_t *qs = base + 2;
            const int nib = (col < 16) ? (qs[col] & 0x0F) : (qs[col - 16] >> 4);
            return double(load_half_bytes_host(base)) * double(nib - 8);
        }
        case PackedFormat::kQ8_0:
            return double(load_half_bytes_host(base)) *
                   double(reinterpret_cast<const int8_t *>(base + 2)[col]);
        case PackedFormat::kQ6_K: {
            const uint8_t *ql = base;
            const uint8_t *qh = base + 128;
            const int8_t *sca = reinterpret_cast<const int8_t *>(base + 192);
            const double d = double(load_half_bytes_host(base + 208));
            const int chunk = col >> 7;
            const int pos = col & 127;
            const int group = pos >> 5;
            const int lane = pos & 31;
            const int ql_byte = ql[chunk * 64 + lane + 32 * (group & 1)];
            const int nib = (group & 2) ? (ql_byte >> 4) : (ql_byte & 0xF);
            const int hbits = (qh[chunk * 32 + lane] >> (2 * group)) & 3;
            const int q = (nib | (hbits << 4)) - 32;
            const int sc_idx = chunk * 8 + (lane >> 4) + group * 2;
            return d * double(int(sca[sc_idx])) * double(q);
        }
        case PackedFormat::kMxFp8:
            return e8m0_decode_host(base[0]) *
                   double(tmq::e4m3_decode_host(base[1 + col]));
        case PackedFormat::kNvFp4: {
            const uint8_t *qs = base + 1;
            const unsigned nib =
                (col < 8) ? (qs[col] & 0x0F) : (qs[col - 8] >> 4);
            return double(tmq::e4m3_decode_host(base[0])) * e2m1_decode_host(nib);
        }
        case PackedFormat::kMxFp4: {
            const uint8_t *qs = base + 1;
            const unsigned nib =
                (col < 16) ? (qs[col] & 0x0F) : (qs[col - 16] >> 4);
            return e8m0_decode_host(base[0]) * e2m1_decode_host(nib);
        }
    }
    return 0.0;
}

template <typename S>
std::vector<double> ref_q8_epilogue(
    const std::vector<storage_t<S>> &x, const std::vector<uint8_t> &packed,
    const std::vector<storage_t<S>> &bias,
    const std::vector<storage_t<S>> &residual, int rows, int input_dim,
    int output_dim, int activation, bool use_bias, bool use_residual,
    bool residual_after_activation) {
    std::vector<double> ref(size_t(rows) * output_dim);
    for (int r = 0; r < rows; ++r) {
        for (int o = 0; o < output_dim; ++o) {
            double acc = 0.0;
            for (int i = 0; i < input_dim; ++i) {
                acc += hval(x, size_t(r) * input_dim + i) *
                       q8_0_dequant_host(packed, o, i, input_dim);
            }
            const size_t out_idx = size_t(r) * output_dim + o;
            if (use_bias) acc += hval(bias, o);
            if (!residual_after_activation && use_residual) acc += hval(residual, out_idx);
            acc = host_activate(acc, activation);
            if (residual_after_activation && use_residual) acc += hval(residual, out_idx);
            ref[out_idx] = acc;
        }
    }
    return ref;
}

template <typename S>
std::vector<double> ref_packed_epilogue(
    PackedFormat fmt, const std::vector<storage_t<S>> &x,
    const std::vector<uint8_t> &packed, const std::vector<storage_t<S>> &bias,
    const std::vector<storage_t<S>> &residual, int rows, int input_dim,
    int output_dim, int activation, bool use_bias, bool use_residual,
    bool residual_after_activation) {
    std::vector<double> ref(size_t(rows) * output_dim);
    for (int r = 0; r < rows; ++r) {
        for (int o = 0; o < output_dim; ++o) {
            double acc = 0.0;
            for (int i = 0; i < input_dim; ++i) {
                acc += hval(x, size_t(r) * input_dim + i) *
                       packed_dequant_host(fmt, packed, o, i, input_dim);
            }
            const size_t out_idx = size_t(r) * output_dim + o;
            if (use_bias) acc += hval(bias, o);
            if (!residual_after_activation && use_residual) acc += hval(residual, out_idx);
            acc = host_activate(acc, activation);
            if (residual_after_activation && use_residual) acc += hval(residual, out_idx);
            ref[out_idx] = acc;
        }
    }
    return ref;
}

template <typename S>
bool run_linear_case(const char *label, int rows, int input_dim, int output_dim,
                     int activation, bool use_bias, bool use_residual,
                     bool residual_after_activation, qc::Rng &rng) {
    const auto x = as_storage<S>(rng.normals(size_t(rows) * input_dim, 0.08f));
    const auto w = as_storage<S>(rng.normals(size_t(output_dim) * input_dim, 0.08f));
    const auto bias = as_storage<S>(rng.normals(output_dim, 0.04f));
    const auto residual = as_storage<S>(rng.normals(size_t(rows) * output_dim, 0.03f));
    auto *dx = qc::dnew(x);
    auto *dw = qc::dnew(w);
    auto *db = qc::dnew(bias);
    auto *dr = qc::dnew(residual);
    auto *dy = qc::dzero<storage_t<S>>(size_t(rows) * output_dim);
    dim3 grid(output_dim, rows);
    linear_epilogue_wave64<S><<<grid, 64>>>(
        dx, dw, db, dr, dy, rows, input_dim, output_dim, activation, use_bias,
        use_residual, residual_after_activation);
    QC_SYNC();
    auto got = qc::d2h(dy, size_t(rows) * output_dim);
    auto ref = ref_linear<S>(x, w, bias, residual, rows, input_dim, output_dim,
                             activation, use_bias, use_residual,
                             residual_after_activation);
    const bool ok = qc::compare(got, ref, qc::tol_for<S>()).report(label);
    qc::dfree(dx, dw, db, dr, dy);
    return ok;
}

template <typename S>
bool run_q8_case(const char *label, int rows, int input_dim, int output_dim,
                 int activation, bool use_bias, bool use_residual,
                 bool residual_after_activation, qc::Rng &rng) {
    const auto x = as_storage<S>(rng.normals(size_t(rows) * input_dim, 0.08f));
    const auto wf = rng.normals(size_t(output_dim) * input_dim, 0.10f);
    const auto packed = pack_q8_0(wf, output_dim, input_dim);
    const auto bias = as_storage<S>(rng.normals(output_dim, 0.04f));
    const auto residual = as_storage<S>(rng.normals(size_t(rows) * output_dim, 0.03f));
    auto *dx = qc::dnew(x);
    auto *dw = qc::dnew(packed);
    auto *db = qc::dnew(bias);
    auto *dr = qc::dnew(residual);
    auto *dy = qc::dzero<storage_t<S>>(size_t(rows) * output_dim);
    dim3 grid(output_dim, rows);
    q8_epilogue_wave64<S><<<grid, 64>>>(
        dx, dw, db, dr, dy, rows, input_dim, output_dim, activation, use_bias,
        use_residual, residual_after_activation);
    QC_SYNC();
    auto got = qc::d2h(dy, size_t(rows) * output_dim);
    auto ref = ref_q8_epilogue<S>(x, packed, bias, residual, rows, input_dim,
                                  output_dim, activation, use_bias, use_residual,
                                  residual_after_activation);
    const bool ok = qc::compare(got, ref, qc::tol_for<S>()).report(label);
    qc::dfree(dx, dw, db, dr, dy);
    return ok;
}

template <typename S, typename FMT>
bool run_packed_epilogue_case(const char *label, PackedFormat fmt, int rows,
                              int input_dim, int output_dim, int activation,
                              bool use_bias, bool use_residual,
                              bool residual_after_activation, qc::Rng &rng) {
    const auto x = as_storage<S>(rng.normals(size_t(rows) * input_dim, 0.08f));
    const auto packed = pack_format_blocks(fmt, output_dim, input_dim, rng);
    const auto bias = as_storage<S>(rng.normals(output_dim, 0.04f));
    const auto residual = as_storage<S>(rng.normals(size_t(rows) * output_dim, 0.03f));
    auto *dx = qc::dnew(x);
    auto *dw = qc::dnew(packed);
    auto *db = qc::dnew(bias);
    auto *dr = qc::dnew(residual);
    auto *dy = qc::dzero<storage_t<S>>(size_t(rows) * output_dim);
    packed_epilogue_wave64<S, FMT><<<dim3(output_dim, rows), 64>>>(
        dx, dw, db, dr, dy, rows, input_dim, output_dim, activation, use_bias,
        use_residual, residual_after_activation);
    QC_SYNC();
    auto got = qc::d2h(dy, size_t(rows) * output_dim);
    auto ref = ref_packed_epilogue<S>(fmt, x, packed, bias, residual, rows,
                                      input_dim, output_dim, activation,
                                      use_bias, use_residual,
                                      residual_after_activation);
    const bool ok = qc::compare(got, ref, qc::tol_for<S>()).report(label);
    qc::dfree(dx, dw, db, dr, dy);
    return ok;
}

template <typename S>
std::vector<double> ref_swiglu_dense(
    const std::vector<storage_t<S>> &x, const std::vector<storage_t<S>> &gw,
    const std::vector<storage_t<S>> &uw,
    const std::vector<storage_t<S>> &gb,
    const std::vector<storage_t<S>> &ub, int rows, int input_dim,
    int output_dim, bool use_bias) {
    std::vector<double> ref(size_t(rows) * output_dim);
    for (int r = 0; r < rows; ++r) {
        for (int o = 0; o < output_dim; ++o) {
            double gate = 0.0;
            double up = 0.0;
            for (int i = 0; i < input_dim; ++i) {
                const double xv = hval(x, size_t(r) * input_dim + i);
                gate += xv * hval(gw, size_t(o) * input_dim + i);
                up += xv * hval(uw, size_t(o) * input_dim + i);
            }
            if (use_bias) {
                gate += hval(gb, o);
                up += hval(ub, o);
            }
            ref[size_t(r) * output_dim + o] =
                (gate / (1.0 + std::exp(-gate))) * up;
        }
    }
    return ref;
}

template <typename S>
std::vector<double> ref_swiglu_q8(
    const std::vector<storage_t<S>> &x, const std::vector<uint8_t> &gw,
    const std::vector<uint8_t> &uw, const std::vector<storage_t<S>> &gb,
    const std::vector<storage_t<S>> &ub, int rows, int input_dim,
    int output_dim, bool use_bias) {
    std::vector<double> ref(size_t(rows) * output_dim);
    for (int r = 0; r < rows; ++r) {
        for (int o = 0; o < output_dim; ++o) {
            double gate = 0.0;
            double up = 0.0;
            for (int i = 0; i < input_dim; ++i) {
                const double xv = hval(x, size_t(r) * input_dim + i);
                gate += xv * q8_0_dequant_host(gw, o, i, input_dim);
                up += xv * q8_0_dequant_host(uw, o, i, input_dim);
            }
            if (use_bias) {
                gate += hval(gb, o);
                up += hval(ub, o);
            }
            ref[size_t(r) * output_dim + o] =
                (gate / (1.0 + std::exp(-gate))) * up;
        }
    }
    return ref;
}

template <typename S>
std::vector<double> ref_swiglu_packed(
    PackedFormat fmt, const std::vector<storage_t<S>> &x,
    const std::vector<uint8_t> &gw, const std::vector<uint8_t> &uw,
    const std::vector<storage_t<S>> &gb, const std::vector<storage_t<S>> &ub,
    int rows, int input_dim, int output_dim, bool use_bias) {
    std::vector<double> ref(size_t(rows) * output_dim);
    for (int r = 0; r < rows; ++r) {
        for (int o = 0; o < output_dim; ++o) {
            double gate = 0.0;
            double up = 0.0;
            for (int i = 0; i < input_dim; ++i) {
                const double xv = hval(x, size_t(r) * input_dim + i);
                gate += xv * packed_dequant_host(fmt, gw, o, i, input_dim);
                up += xv * packed_dequant_host(fmt, uw, o, i, input_dim);
            }
            if (use_bias) {
                gate += hval(gb, o);
                up += hval(ub, o);
            }
            ref[size_t(r) * output_dim + o] =
                (gate / (1.0 + std::exp(-gate))) * up;
        }
    }
    return ref;
}

template <typename S>
bool run_swiglu_dense_case(const char *label, int rows, int input_dim,
                           int output_dim, bool use_bias, qc::Rng &rng) {
    const auto x = as_storage<S>(rng.normals(size_t(rows) * input_dim, 0.08f));
    const auto gw = as_storage<S>(rng.normals(size_t(output_dim) * input_dim, 0.08f));
    const auto uw = as_storage<S>(rng.normals(size_t(output_dim) * input_dim, 0.08f));
    const auto gb = as_storage<S>(rng.normals(output_dim, 0.04f));
    const auto ub = as_storage<S>(rng.normals(output_dim, 0.04f));
    auto *dx = qc::dnew(x);
    auto *dgw = qc::dnew(gw);
    auto *duw = qc::dnew(uw);
    auto *dgb = qc::dnew(gb);
    auto *dub = qc::dnew(ub);
    auto *dy = qc::dzero<storage_t<S>>(size_t(rows) * output_dim);
    swiglu_dense_wave64<S><<<dim3(output_dim, rows), 64>>>(
        dx, dgw, duw, dgb, dub, dy, rows, input_dim, output_dim, use_bias);
    QC_SYNC();
    auto got = qc::d2h(dy, size_t(rows) * output_dim);
    auto ref = ref_swiglu_dense<S>(x, gw, uw, gb, ub, rows, input_dim,
                                   output_dim, use_bias);
    const bool ok = qc::compare(got, ref, qc::tol_for<S>()).report(label);
    qc::dfree(dx, dgw, duw, dgb, dub, dy);
    return ok;
}

template <typename S>
bool run_swiglu_q8_case(const char *label, int rows, int input_dim,
                        int output_dim, bool use_bias, qc::Rng &rng) {
    const auto x = as_storage<S>(rng.normals(size_t(rows) * input_dim, 0.08f));
    const auto gwf = rng.normals(size_t(output_dim) * input_dim, 0.10f);
    const auto uwf = rng.normals(size_t(output_dim) * input_dim, 0.10f);
    const auto gw = pack_q8_0(gwf, output_dim, input_dim);
    const auto uw = pack_q8_0(uwf, output_dim, input_dim);
    const auto gb = as_storage<S>(rng.normals(output_dim, 0.04f));
    const auto ub = as_storage<S>(rng.normals(output_dim, 0.04f));
    auto *dx = qc::dnew(x);
    auto *dgw = qc::dnew(gw);
    auto *duw = qc::dnew(uw);
    auto *dgb = qc::dnew(gb);
    auto *dub = qc::dnew(ub);
    auto *dy = qc::dzero<storage_t<S>>(size_t(rows) * output_dim);
    swiglu_q8_wave64<S><<<dim3(output_dim, rows), 64>>>(
        dx, dgw, duw, dgb, dub, dy, rows, input_dim, output_dim, use_bias);
    QC_SYNC();
    auto got = qc::d2h(dy, size_t(rows) * output_dim);
    auto ref = ref_swiglu_q8<S>(x, gw, uw, gb, ub, rows, input_dim,
                                output_dim, use_bias);
    const bool ok = qc::compare(got, ref, qc::tol_for<S>()).report(label);
    qc::dfree(dx, dgw, duw, dgb, dub, dy);
    return ok;
}

template <typename S, typename FMT>
bool run_swiglu_packed_case(const char *label, PackedFormat fmt, int rows,
                            int input_dim, int output_dim, bool use_bias,
                            qc::Rng &rng) {
    const auto x = as_storage<S>(rng.normals(size_t(rows) * input_dim, 0.08f));
    const auto gw = pack_format_blocks(fmt, output_dim, input_dim, rng);
    const auto uw = pack_format_blocks(fmt, output_dim, input_dim, rng);
    const auto gb = as_storage<S>(rng.normals(output_dim, 0.04f));
    const auto ub = as_storage<S>(rng.normals(output_dim, 0.04f));
    auto *dx = qc::dnew(x);
    auto *dgw = qc::dnew(gw);
    auto *duw = qc::dnew(uw);
    auto *dgb = qc::dnew(gb);
    auto *dub = qc::dnew(ub);
    auto *dy = qc::dzero<storage_t<S>>(size_t(rows) * output_dim);
    swiglu_packed_wave64<S, FMT><<<dim3(output_dim, rows), 64>>>(
        dx, dgw, duw, dgb, dub, dy, rows, input_dim, output_dim, use_bias);
    QC_SYNC();
    auto got = qc::d2h(dy, size_t(rows) * output_dim);
    auto ref =
        ref_swiglu_packed<S>(fmt, x, gw, uw, gb, ub, rows, input_dim,
                             output_dim, use_bias);
    const bool ok = qc::compare(got, ref, qc::tol_for<S>()).report(label);
    qc::dfree(dx, dgw, duw, dgb, dub, dy);
    return ok;
}

template <typename S>
bool run_gemm_gate_case(const char *label, int rows, int input_dim,
                        int output_dim, qc::Rng &rng) {
    const auto x = as_storage<S>(rng.normals(size_t(rows) * input_dim, 0.08f));
    const auto w = as_storage<S>(rng.normals(size_t(input_dim) * output_dim, 0.08f));
    const auto bias = as_storage<S>(rng.normals(output_dim, 0.04f));
    const auto gate = as_storage<S>(rng.uniforms(output_dim, 0.5f, 1.5f));
    const auto residual = as_storage<S>(rng.normals(size_t(rows) * output_dim, 0.03f));
    auto *dx = qc::dnew(x);
    auto *dw = qc::dnew(w);
    auto *db = qc::dnew(bias);
    auto *dg = qc::dnew(gate);
    auto *dr = qc::dnew(residual);
    auto *dy = qc::dzero<storage_t<S>>(size_t(rows) * output_dim);
    gemm_gate_residual_wave64<S><<<dim3(output_dim, rows), 64>>>(
        dx, dw, db, dg, dr, dy, rows, output_dim, input_dim, 1, 1, 1);
    QC_SYNC();
    auto got = qc::d2h(dy, size_t(rows) * output_dim);
    std::vector<double> ref(size_t(rows) * output_dim);
    for (int r = 0; r < rows; ++r) {
        for (int o = 0; o < output_dim; ++o) {
            double acc = hval(bias, o);
            for (int i = 0; i < input_dim; ++i) {
                acc += hval(x, size_t(r) * input_dim + i) *
                       hval(w, size_t(i) * output_dim + o);
            }
            acc *= hval(gate, o);
            acc += hval(residual, size_t(r) * output_dim + o);
            ref[size_t(r) * output_dim + o] = acc;
        }
    }
    const bool ok = qc::compare(got, ref, qc::tol_for<S>()).report(label);
    qc::dfree(dx, dw, db, dg, dr, dy);
    return ok;
}

bool run_lora_case(qc::Rng &rng) {
    constexpr int rows = 3;
    constexpr int input_dim = 96;
    constexpr int output_dim = 80;
    constexpr int rank = 16;
    constexpr float scale = 0.35f;
    const auto x = rng.normals(size_t(rows) * input_dim, 0.08f);
    const auto af = rng.normals(size_t(rank) * input_dim, 0.08f);
    const auto bf = rng.normals(size_t(output_dim) * rank, 0.08f);
    const auto a = qc::to_storage<__half>(af);
    const auto b = qc::to_storage<__half>(bf);
    const auto base = rng.normals(size_t(rows) * output_dim, 0.03f);
    auto *dx = qc::dnew(x);
    auto *da = qc::dnew(a);
    auto *db = qc::dnew(b);
    auto *dbase = qc::dnew(base);
    auto *dy = qc::dzero<float>(size_t(rows) * output_dim);
    lora_apply_direct_f16_wave64<<<dim3(output_dim, rows), 64>>>(
        dx, da, db, dbase, dy, rows, input_dim, output_dim, rank, scale, 1);
    QC_SYNC();
    auto got = qc::d2h(dy, size_t(rows) * output_dim);
    std::vector<double> ref(size_t(rows) * output_dim);
    for (int r = 0; r < rows; ++r) {
        std::vector<double> low(rank);
        for (int ar = 0; ar < rank; ++ar) {
            double sum = 0.0;
            for (int i = 0; i < input_dim; ++i) {
                sum += double(x[size_t(r) * input_dim + i]) *
                       qc::to_double(a[size_t(ar) * input_dim + i]);
            }
            low[ar] = qc::round_fp16(sum);
        }
        for (int o = 0; o < output_dim; ++o) {
            double sum = 0.0;
            for (int ar = 0; ar < rank; ++ar) {
                sum += low[ar] * qc::to_double(b[size_t(o) * rank + ar]);
            }
            const double delta = qc::round_fp16(sum);
            ref[size_t(r) * output_dim + o] =
                double(base[size_t(r) * output_dim + o]) + double(scale) * delta;
        }
    }
    const bool ok = qc::compare(got, ref, qc::Tol::fp32()).report("lora_apply_direct_f16");
    qc::dfree(dx, da, db, dbase, dy);
    return ok;
}

bool run_complex_case(qc::Rng &rng) {
    constexpr int m = 13;
    constexpr int n = 17;
    constexpr int k = 33;
    const auto ar = rng.normals(size_t(m) * k, 0.08f);
    const auto ai = rng.normals(size_t(m) * k, 0.08f);
    const auto br = rng.normals(size_t(k) * n, 0.08f);
    const auto bi = rng.normals(size_t(k) * n, 0.08f);
    auto *dar = qc::dnew(ar);
    auto *dai = qc::dnew(ai);
    auto *dbr = qc::dnew(br);
    auto *dbi = qc::dnew(bi);
    auto *dcr = qc::dzero<float>(size_t(m) * n);
    auto *dci = qc::dzero<float>(size_t(m) * n);
    complex_gemm_wave64<<<dim3(n, m), 64>>>(dar, dai, dbr, dbi, dcr, dci, m, n, k);
    QC_SYNC();
    auto got_r = qc::d2h(dcr, size_t(m) * n);
    auto got_i = qc::d2h(dci, size_t(m) * n);
    std::vector<double> ref_r(size_t(m) * n), ref_i(size_t(m) * n);
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            double rr = 0.0;
            double ii = 0.0;
            for (int inner = 0; inner < k; ++inner) {
                const size_t aidx = size_t(row) * k + inner;
                const size_t bidx = size_t(inner) * n + col;
                rr += double(ar[aidx]) * br[bidx] - double(ai[aidx]) * bi[bidx];
                ii += double(ar[aidx]) * bi[bidx] + double(ai[aidx]) * br[bidx];
            }
            ref_r[size_t(row) * n + col] = rr;
            ref_i[size_t(row) * n + col] = ii;
        }
    }
    bool ok = true;
    ok &= qc::compare(got_r, ref_r, qc::Tol::fp32()).report("complex_gemm real");
    ok &= qc::compare(got_i, ref_i, qc::Tol::fp32()).report("complex_gemm imag");
    qc::dfree(dar, dai, dbr, dbi, dcr, dci);
    return ok;
}

bool run_grouped_gemm_case(qc::Rng &rng) {
    constexpr int groups = 5;
    constexpr int m = 7;
    constexpr int n = 29;
    constexpr int k = 65;
    const auto a = rng.normals(size_t(groups) * m * k, 0.08f);
    const auto b = rng.normals(size_t(groups) * k * n, 0.08f);
    auto *da = qc::dnew(a);
    auto *db = qc::dnew(b);
    auto *dc = qc::dzero<float>(size_t(groups) * m * n);
    grouped_gemm_scalar<<<dim3((n + 255) / 256, m, groups), 256>>>(
        da, db, dc, groups, m, n, k);
    QC_SYNC();
    auto got = qc::d2h(dc, size_t(groups) * m * n);
    std::vector<double> ref(size_t(groups) * m * n);
    for (int g = 0; g < groups; ++g) {
        for (int r = 0; r < m; ++r) {
            for (int col = 0; col < n; ++col) {
                double sum = 0.0;
                for (int inner = 0; inner < k; ++inner) {
                    sum += double(a[(size_t(g) * m + r) * k + inner]) *
                           b[(size_t(g) * k + inner) * n + col];
                }
                ref[(size_t(g) * m + r) * n + col] = sum;
            }
        }
    }
    const bool ok = qc::compare(got, ref, qc::Tol::fp32()).report("grouped_gemm fp32");
    qc::dfree(da, db, dc);
    return ok;
}

bool run_correctness() {
    qc::Rng rng(3003);
    bool ok = true;
    ok &= run_linear_case<qc::StorageF32>("decode_linear fp32",
                                          5, 97, 73, kGeluTanh, true, false,
                                          false, rng);
    ok &= run_linear_case<qc::StorageBf16>("decode_linear bf16",
                                           4, 96, 64, kNone, true, false,
                                           false, rng);
    ok &= run_linear_case<qc::StorageF32>("decode_linear_residual fp32",
                                          3, 113, 61, kNone, true, true,
                                          false, rng);
    ok &= run_q8_case<qc::StorageF32>("decode_linear_q8 q8_0 fp32",
                                      4, 128, 67, kGeluTanh, true, true,
                                      false, rng);
    ok &= run_linear_case<qc::StorageFp16>(
        "decode_linear_epilogue_dense fp16", 3, 128, 80, kSilu, true, true,
        true, rng);
    ok &= run_packed_epilogue_case<qc::StorageF32, tmq::q4_0>(
        "decode_linear_epilogue_packed q4_0 fp32", PackedFormat::kQ4_0,
        2, 512, 35, kGeluErf, true, true, true, rng);
    ok &= run_packed_epilogue_case<qc::StorageF32, tmq::q8_0>(
        "decode_linear_epilogue_packed q8_0 fp32", PackedFormat::kQ8_0,
        2, 512, 35, kGeluErf, true, true, true, rng);
    ok &= run_packed_epilogue_case<qc::StorageF32, tmq::q6_K>(
        "decode_linear_epilogue_packed q6_K fp32", PackedFormat::kQ6_K,
        2, 512, 35, kGeluErf, true, true, true, rng);
    ok &= run_packed_epilogue_case<qc::StorageF32, tmq::mxfp8>(
        "decode_linear_epilogue_packed mxfp8 fp32", PackedFormat::kMxFp8,
        2, 512, 35, kGeluErf, true, true, true, rng);
    ok &= run_packed_epilogue_case<qc::StorageF32, tmq::nvfp4>(
        "decode_linear_epilogue_packed nvfp4 fp32", PackedFormat::kNvFp4,
        2, 512, 35, kGeluErf, true, true, true, rng);
    ok &= run_packed_epilogue_case<qc::StorageF32, tmq::mxfp4>(
        "decode_linear_epilogue_packed mxfp4 fp32", PackedFormat::kMxFp4,
        2, 512, 35, kGeluErf, true, true, true, rng);
    ok &= run_swiglu_dense_case<qc::StorageBf16>(
        "decode_swiglu_dense bf16", 4, 96, 64, true, rng);
    ok &= run_swiglu_packed_case<qc::StorageF32, tmq::q4_0>(
        "decode_swiglu_packed q4_0 fp32", PackedFormat::kQ4_0,
        2, 512, 35, true, rng);
    ok &= run_swiglu_packed_case<qc::StorageF32, tmq::q8_0>(
        "decode_swiglu_packed q8_0 fp32", PackedFormat::kQ8_0,
        2, 512, 35, true, rng);
    ok &= run_swiglu_packed_case<qc::StorageF32, tmq::q6_K>(
        "decode_swiglu_packed q6_K fp32", PackedFormat::kQ6_K,
        2, 512, 35, true, rng);
    ok &= run_swiglu_packed_case<qc::StorageF32, tmq::mxfp8>(
        "decode_swiglu_packed mxfp8 fp32", PackedFormat::kMxFp8,
        2, 512, 35, true, rng);
    ok &= run_swiglu_packed_case<qc::StorageF32, tmq::nvfp4>(
        "decode_swiglu_packed nvfp4 fp32", PackedFormat::kNvFp4,
        2, 512, 35, true, rng);
    ok &= run_swiglu_packed_case<qc::StorageF32, tmq::mxfp4>(
        "decode_swiglu_packed mxfp4 fp32", PackedFormat::kMxFp4,
        2, 512, 35, true, rng);
    ok &= run_gemm_gate_case<qc::StorageF32>("gemm_gate_residual fp32",
                                             4, 91, 79, rng);
    ok &= run_grouped_gemm_case(rng);
    ok &= run_lora_case(rng);
    ok &= run_complex_case(rng);
    std::printf("%s\n", ok ? "ALL PASS" : "FAILED");
    return ok;
}

void run_bench() {
    qc::Rng rng(3004);
    constexpr int rows = 64;
    constexpr int input_dim = 4096;
    constexpr int output_dim = 2048;
    auto x = as_storage<qc::StorageF32>(rng.normals(size_t(rows) * input_dim, 0.08f));
    auto w =
        as_storage<qc::StorageF32>(rng.normals(size_t(output_dim) * input_dim, 0.08f));
    auto bias = as_storage<qc::StorageF32>(rng.normals(output_dim, 0.04f));
    auto residual =
        as_storage<qc::StorageF32>(rng.normals(size_t(rows) * output_dim, 0.03f));
    auto *dx = qc::dnew(x);
    auto *dw = qc::dnew(w);
    auto *db = qc::dnew(bias);
    auto *dr = qc::dnew(residual);
    auto *dy0 = qc::dzero<float>(size_t(rows) * output_dim);
    auto *dy1 = qc::dzero<float>(size_t(rows) * output_dim);
    auto scalar = [&] {
        linear_epilogue_scalar<qc::StorageF32>
            <<<dim3((output_dim + 255) / 256, rows), 256>>>(
                dx, dw, db, dr, dy0, rows, input_dim, output_dim, kNone, 1, 1, 0);
    };
    auto wave64 = [&] {
        linear_epilogue_wave64<qc::StorageF32><<<dim3(output_dim, rows), 64>>>(
            dx, dw, db, dr, dy1, rows, input_dim, output_dim, kNone, 1, 1, 0);
    };
    scalar();
    wave64();
    QC_SYNC();
    const auto base = qc::bench(scalar, 10, 50);
    const auto cand = qc::bench(wave64, 10, 50);
    const double flops = 2.0 * rows * input_dim * output_dim;
    std::printf("== decode_linear_residual fp32 A/B rows=%d in=%d out=%d\n",
                rows, input_dim, output_dim);
    base.report_compute("scalar one-thread/output", flops);
    cand.report_compute("wave64 one-wave/output", flops);
    std::printf("  speedup %.2fx  keep=%s\n", base.median_ms / cand.median_ms,
                cand.median_ms <= base.median_ms ? "wave64" : "scalar");

    constexpr int packed_rows = 64;
    constexpr int packed_input_dim = 4096;
    constexpr int packed_output_dim = 1024;
    auto px = as_storage<qc::StorageF32>(
        rng.normals(size_t(packed_rows) * packed_input_dim, 0.08f));
    auto pw = pack_format_blocks(PackedFormat::kMxFp4, packed_output_dim,
                                 packed_input_dim, rng);
    auto pbias = as_storage<qc::StorageF32>(rng.normals(packed_output_dim, 0.04f));
    auto packed_residual = as_storage<qc::StorageF32>(
        rng.normals(size_t(packed_rows) * packed_output_dim, 0.03f));
    auto *dpx = qc::dnew(px);
    auto *dpw = qc::dnew(pw);
    auto *dpbias = qc::dnew(pbias);
    auto *dpresidual = qc::dnew(packed_residual);
    auto *dpy0 = qc::dzero<float>(size_t(packed_rows) * packed_output_dim);
    auto *dpy1 = qc::dzero<float>(size_t(packed_rows) * packed_output_dim);
    auto packed_scalar = [&] {
        packed_epilogue_scalar<qc::StorageF32, tmq::mxfp4>
            <<<dim3((packed_output_dim + 255) / 256, packed_rows), 256>>>(
                dpx, dpw, dpbias, dpresidual, dpy0, packed_rows, packed_input_dim,
                packed_output_dim, kGeluErf, 1, 1, 1);
    };
    auto packed_wave64 = [&] {
        packed_epilogue_wave64<qc::StorageF32, tmq::mxfp4>
            <<<dim3(packed_output_dim, packed_rows), 64>>>(
                dpx, dpw, dpbias, dpresidual, dpy1, packed_rows, packed_input_dim,
                packed_output_dim, kGeluErf, 1, 1, 1);
    };
    packed_scalar();
    packed_wave64();
    QC_SYNC();
    const auto pbase = qc::bench(packed_scalar, 10, 50);
    const auto pcand = qc::bench(packed_wave64, 10, 50);
    const double pflops = 2.0 * packed_rows * packed_input_dim * packed_output_dim;
    std::printf("== decode_linear_epilogue_packed mxfp4 fp32 A/B rows=%d in=%d out=%d\n",
                packed_rows, packed_input_dim, packed_output_dim);
    pbase.report_compute("scalar one-thread/output", pflops);
    pcand.report_compute("wave64 one-wave/output", pflops);
    std::printf("  speedup %.2fx  keep=%s\n", pbase.median_ms / pcand.median_ms,
                pcand.median_ms <= pbase.median_ms ? "wave64" : "scalar");

    auto pgw = pack_format_blocks(PackedFormat::kMxFp4, packed_output_dim,
                                  packed_input_dim, rng);
    auto puw = pack_format_blocks(PackedFormat::kMxFp4, packed_output_dim,
                                  packed_input_dim, rng);
    auto pgbias = as_storage<qc::StorageF32>(rng.normals(packed_output_dim, 0.04f));
    auto pubias = as_storage<qc::StorageF32>(rng.normals(packed_output_dim, 0.04f));
    auto *dpgw = qc::dnew(pgw);
    auto *dpuw = qc::dnew(puw);
    auto *dpgbias = qc::dnew(pgbias);
    auto *dpubias = qc::dnew(pubias);
    auto *dpsy0 = qc::dzero<float>(size_t(packed_rows) * packed_output_dim);
    auto *dpsy1 = qc::dzero<float>(size_t(packed_rows) * packed_output_dim);
    auto packed_swiglu_scalar = [&] {
        swiglu_packed_scalar<qc::StorageF32, tmq::mxfp4>
            <<<dim3((packed_output_dim + 255) / 256, packed_rows), 256>>>(
                dpx, dpgw, dpuw, dpgbias, dpubias, dpsy0, packed_rows,
                packed_input_dim, packed_output_dim, 1);
    };
    auto packed_swiglu_wave64 = [&] {
        swiglu_packed_wave64<qc::StorageF32, tmq::mxfp4>
            <<<dim3(packed_output_dim, packed_rows), 64>>>(
                dpx, dpgw, dpuw, dpgbias, dpubias, dpsy1, packed_rows,
                packed_input_dim, packed_output_dim, 1);
    };
    packed_swiglu_scalar();
    packed_swiglu_wave64();
    QC_SYNC();
    const auto psbase = qc::bench(packed_swiglu_scalar, 10, 50);
    const auto pscand = qc::bench(packed_swiglu_wave64, 10, 50);
    const double psflops = 4.0 * packed_rows * packed_input_dim * packed_output_dim;
    std::printf("== decode_swiglu_packed mxfp4 fp32 A/B rows=%d in=%d out=%d\n",
                packed_rows, packed_input_dim, packed_output_dim);
    psbase.report_compute("scalar one-thread/output", psflops);
    pscand.report_compute("wave64 one-wave/output", psflops);
    std::printf("  speedup %.2fx  keep=%s\n", psbase.median_ms / pscand.median_ms,
                pscand.median_ms <= psbase.median_ms ? "wave64" : "scalar");

    constexpr int groups = 8;
    constexpr int gm = 64;
    constexpr int gn = 512;
    constexpr int gk = 2048;
    auto ga = rng.normals(size_t(groups) * gm * gk, 0.08f);
    auto gb = rng.normals(size_t(groups) * gk * gn, 0.08f);
    auto *dga = qc::dnew(ga);
    auto *dgb = qc::dnew(gb);
    auto *dgc0 = qc::dzero<float>(size_t(groups) * gm * gn);
    auto *dgc1 = qc::dzero<float>(size_t(groups) * gm * gn);
    auto grouped_scalar = [&] {
        grouped_gemm_scalar<<<dim3((gn + 255) / 256, gm, groups), 256>>>(
            dga, dgb, dgc0, groups, gm, gn, gk);
    };
    auto grouped_wave64 = [&] {
        grouped_gemm_wave64<<<dim3(gn, gm, groups), 64>>>(
            dga, dgb, dgc1, groups, gm, gn, gk);
    };
    grouped_scalar();
    grouped_wave64();
    QC_SYNC();
    const auto gbase = qc::bench(grouped_scalar, 10, 50);
    const auto gcand = qc::bench(grouped_wave64, 10, 50);
    const double gflops = 2.0 * groups * gm * gn * gk;
    std::printf("== grouped_gemm fp32 A/B groups=%d M=%d N=%d K=%d\n",
                groups, gm, gn, gk);
    gbase.report_compute("scalar one-thread/output", gflops);
    gcand.report_compute("wave64 one-wave/output", gflops);
    std::printf("  speedup %.2fx  keep=%s\n", gbase.median_ms / gcand.median_ms,
                gcand.median_ms <= gbase.median_ms ? "wave64" : "scalar");

    qc::dfree(dx, dw, db, dr, dy0, dy1, dpx, dpw, dpbias, dpresidual, dpy0,
              dpy1, dpgw, dpuw, dpgbias, dpubias, dpsy0, dpsy1, dga, dgb,
              dgc0, dgc1);
}

}  // namespace

int main(int argc, char **argv) {
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    const bool ok = run_correctness();
    if (!ok) return 1;
    if (bench) run_bench();
    return 0;
}
