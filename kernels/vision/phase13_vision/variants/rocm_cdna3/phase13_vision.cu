/**
 * @file
 * @brief Phase 13 vision parity ports for CDNA3.
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
constexpr long long kEdgeFeatures = 256;
constexpr long long kEdgeClasses = 7;

__host__ __device__ long long patch_out(long long input, long long kernel,
                                        long long stride, long long padding) {
    return (input + 2 * padding - kernel) / stride + 1;
}

__host__ __device__ long long pool_out(long long input, long long kernel,
                                       long long stride, bool ceil_mode) {
    return ceil_mode ? (input - kernel + stride - 1) / stride + 1
                     : (input - kernel) / stride + 1;
}

__host__ __device__ long long ceil_div(long long a, long long b) {
    return (a + b - 1) / b;
}

std::vector<double> to_ref(const std::vector<float> &values) {
    return std::vector<double>(values.begin(), values.end());
}

std::vector<double> to_ref_i32(const std::vector<int32_t> &values) {
    std::vector<double> out(values.size());
    for (size_t i = 0; i < values.size(); ++i)
        out[i] = static_cast<double>(values[i]);
    return out;
}

__host__ __device__ float gelu_erf_value(float x) {
#ifdef __HIP_DEVICE_COMPILE__
    return 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f));
#else
    return 0.5f * x *
           (1.0f + std::erf(x * 0.70710678118654752440f));
#endif
}

__host__ __device__ float patch_value_2d(const float *input, long long item,
                                         long long oy, long long ox,
                                         long long ky, long long kx,
                                         long long channel, long long height,
                                         long long width, long long channels,
                                         long long stride_h,
                                         long long stride_w, long long pad_h,
                                         long long pad_w) {
    const long long iy = oy * stride_h + ky - pad_h;
    const long long ix = ox * stride_w + kx - pad_w;
    if (iy < 0 || iy >= height || ix < 0 || ix >= width) return 0.0f;
    return input[((item * height + iy) * width + ix) * channels + channel];
}

__host__ __device__ float patch_value_3d(
    const float *input, long long item, long long ot, long long oy,
    long long ox, long long kt, long long ky, long long kx, long long channel,
    long long frames, long long height, long long width, long long channels,
    long long stride_t, long long stride_h, long long stride_w,
    long long pad_t, long long pad_h, long long pad_w) {
    const long long it = ot * stride_t + kt - pad_t;
    const long long iy = oy * stride_h + ky - pad_h;
    const long long ix = ox * stride_w + kx - pad_w;
    if (it < 0 || it >= frames || iy < 0 || iy >= height || ix < 0 ||
        ix >= width)
        return 0.0f;
    return input[(((item * frames + it) * height + iy) * width + ix) *
                     channels +
                 channel];
}

__host__ __device__ float space_to_depth_value(
    const float *input, long long item, long long oy, long long ox,
    long long feature, long long height, long long width, long long channels,
    long long block_size) {
    const long long patch_pixel = feature / channels;
    const long long channel = feature - patch_pixel * channels;
    const long long by = patch_pixel / block_size;
    const long long bx = patch_pixel - by * block_size;
    const long long iy = oy * block_size + by;
    const long long ix = ox * block_size + bx;
    if (iy >= height || ix >= width) return 0.0f;
    return input[((item * height + iy) * width + ix) * channels + channel];
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

__global__ void extract_patches_2d_kernel(
    const float *input, float *output, long long batch, long long height,
    long long width, long long channels, long long kernel_h, long long kernel_w,
    long long stride_h, long long stride_w, long long pad_h, long long pad_w) {
    const long long out_h = patch_out(height, kernel_h, stride_h, pad_h);
    const long long out_w = patch_out(width, kernel_w, stride_w, pad_w);
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    const long long total =
        batch * out_h * out_w * kernel_h * kernel_w * channels;
    if (idx >= total) return;
    long long t = idx;
    const long long channel = t % channels;
    t /= channels;
    const long long kx = t % kernel_w;
    t /= kernel_w;
    const long long ky = t % kernel_h;
    t /= kernel_h;
    const long long ox = t % out_w;
    t /= out_w;
    const long long oy = t % out_h;
    const long long item = t / out_h;
    output[idx] = patch_value_2d(input, item, oy, ox, ky, kx, channel, height,
                                 width, channels, stride_h, stride_w, pad_h,
                                 pad_w);
}

__global__ void extract_patches_2d_scalar_kernel(
    const float *input, float *output, long long batch, long long height,
    long long width, long long channels, long long kernel_h, long long kernel_w,
    long long stride_h, long long stride_w, long long pad_h, long long pad_w) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const long long out_h = patch_out(height, kernel_h, stride_h, pad_h);
    const long long out_w = patch_out(width, kernel_w, stride_w, pad_w);
    const long long total =
        batch * out_h * out_w * kernel_h * kernel_w * channels;
    for (long long idx = 0; idx < total; ++idx) {
        long long t = idx;
        const long long channel = t % channels;
        t /= channels;
        const long long kx = t % kernel_w;
        t /= kernel_w;
        const long long ky = t % kernel_h;
        t /= kernel_h;
        const long long ox = t % out_w;
        t /= out_w;
        const long long oy = t % out_h;
        const long long item = t / out_h;
        output[idx] = patch_value_2d(input, item, oy, ox, ky, kx, channel,
                                     height, width, channels, stride_h,
                                     stride_w, pad_h, pad_w);
    }
}

__global__ void extract_patches_3d_kernel(
    const float *input, float *output, long long batch, long long frames,
    long long height, long long width, long long channels, long long kernel_t,
    long long kernel_h, long long kernel_w, long long stride_t,
    long long stride_h, long long stride_w, long long pad_t, long long pad_h,
    long long pad_w) {
    const long long out_t = patch_out(frames, kernel_t, stride_t, pad_t);
    const long long out_h = patch_out(height, kernel_h, stride_h, pad_h);
    const long long out_w = patch_out(width, kernel_w, stride_w, pad_w);
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = batch * out_t * out_h * out_w * kernel_t *
                            kernel_h * kernel_w * channels;
    if (idx >= total) return;
    long long t = idx;
    const long long channel = t % channels;
    t /= channels;
    const long long kx = t % kernel_w;
    t /= kernel_w;
    const long long ky = t % kernel_h;
    t /= kernel_h;
    const long long kt = t % kernel_t;
    t /= kernel_t;
    const long long ox = t % out_w;
    t /= out_w;
    const long long oy = t % out_h;
    t /= out_h;
    const long long ot = t % out_t;
    const long long item = t / out_t;
    output[idx] = patch_value_3d(input, item, ot, oy, ox, kt, ky, kx, channel,
                                 frames, height, width, channels, stride_t,
                                 stride_h, stride_w, pad_t, pad_h, pad_w);
}

__global__ void vision_patch_projection_kernel(
    const float *input, const float *weights, const float *bias, float *output,
    long long batch, long long height, long long width, long long in_channels,
    long long out_channels, long long kernel_h, long long kernel_w,
    long long stride_h, long long stride_w, long long pad_h, long long pad_w) {
    const long long out_h = patch_out(height, kernel_h, stride_h, pad_h);
    const long long out_w = patch_out(width, kernel_w, stride_w, pad_w);
    const long long rows = batch * out_h * out_w;
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * out_channels) return;
    const long long oc = idx % out_channels;
    const long long row = idx / out_channels;
    const long long item = row / (out_h * out_w);
    const long long spatial = row - item * out_h * out_w;
    const long long oy = spatial / out_w;
    const long long ox = spatial - oy * out_w;
    const long long patch_dim = kernel_h * kernel_w * in_channels;
    const float *wptr = weights + oc * patch_dim;
    float sum = bias == nullptr ? 0.0f : bias[oc];
    long long feature = 0;
    for (long long ky = 0; ky < kernel_h; ++ky)
        for (long long kx = 0; kx < kernel_w; ++kx)
            for (long long ic = 0; ic < in_channels; ++ic, ++feature)
                sum += patch_value_2d(input, item, oy, ox, ky, kx, ic, height,
                                      width, in_channels, stride_h, stride_w,
                                      pad_h, pad_w) *
                       wptr[feature];
    output[idx] = sum;
}

__global__ void vision_patch_projection_scalar_kernel(
    const float *input, const float *weights, const float *bias, float *output,
    long long batch, long long height, long long width, long long in_channels,
    long long out_channels, long long kernel_h, long long kernel_w,
    long long stride_h, long long stride_w, long long pad_h, long long pad_w) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const long long out_h = patch_out(height, kernel_h, stride_h, pad_h);
    const long long out_w = patch_out(width, kernel_w, stride_w, pad_w);
    const long long rows = batch * out_h * out_w;
    for (long long idx = 0; idx < rows * out_channels; ++idx) {
        const long long oc = idx % out_channels;
        const long long row = idx / out_channels;
        const long long item = row / (out_h * out_w);
        const long long spatial = row - item * out_h * out_w;
        const long long oy = spatial / out_w;
        const long long ox = spatial - oy * out_w;
        const long long patch_dim = kernel_h * kernel_w * in_channels;
        const float *wptr = weights + oc * patch_dim;
        float sum = bias == nullptr ? 0.0f : bias[oc];
        long long feature = 0;
        for (long long ky = 0; ky < kernel_h; ++ky)
            for (long long kx = 0; kx < kernel_w; ++kx)
                for (long long ic = 0; ic < in_channels; ++ic, ++feature)
                    sum += patch_value_2d(input, item, oy, ox, ky, kx, ic,
                                          height, width, in_channels, stride_h,
                                          stride_w, pad_h, pad_w) *
                           wptr[feature];
        output[idx] = sum;
    }
}

__global__ void vision_patch_projection_3d_kernel(
    const float *input, const float *weights, const float *bias, float *output,
    long long batch, long long frames, long long height, long long width,
    long long in_channels, long long out_channels, long long kernel_t,
    long long kernel_h, long long kernel_w, long long stride_t,
    long long stride_h, long long stride_w, long long pad_t, long long pad_h,
    long long pad_w) {
    const long long out_t = patch_out(frames, kernel_t, stride_t, pad_t);
    const long long out_h = patch_out(height, kernel_h, stride_h, pad_h);
    const long long out_w = patch_out(width, kernel_w, stride_w, pad_w);
    const long long rows = batch * out_t * out_h * out_w;
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * out_channels) return;
    const long long oc = idx % out_channels;
    const long long row = idx / out_channels;
    const long long item = row / (out_t * out_h * out_w);
    long long spatial = row - item * out_t * out_h * out_w;
    const long long ot = spatial / (out_h * out_w);
    spatial -= ot * out_h * out_w;
    const long long oy = spatial / out_w;
    const long long ox = spatial - oy * out_w;
    const long long patch_dim = kernel_t * kernel_h * kernel_w * in_channels;
    const float *wptr = weights + oc * patch_dim;
    float sum = bias == nullptr ? 0.0f : bias[oc];
    long long feature = 0;
    for (long long kt = 0; kt < kernel_t; ++kt)
        for (long long ky = 0; ky < kernel_h; ++ky)
            for (long long kx = 0; kx < kernel_w; ++kx)
                for (long long ic = 0; ic < in_channels; ++ic, ++feature)
                    sum += patch_value_3d(input, item, ot, oy, ox, kt, ky, kx,
                                          ic, frames, height, width,
                                          in_channels, stride_t, stride_h,
                                          stride_w, pad_t, pad_h, pad_w) *
                           wptr[feature];
    output[idx] = sum;
}

__global__ void patch_merge_layer_norm_kernel(
    const float *input, const float *weight, const float *bias, float *output,
    long long batch, long long height, long long width, long long channels,
    float eps) {
    const long long out_h = (height + 1) / 2;
    const long long out_w = (width + 1) / 2;
    const long long features = 4 * channels;
    const long long row = blockIdx.x * blockDim.x + threadIdx.x;
    const long long rows = batch * out_h * out_w;
    if (row >= rows) return;
    const long long item = row / (out_h * out_w);
    const long long spatial = row - item * out_h * out_w;
    const long long oy = spatial / out_w;
    const long long ox = spatial - oy * out_w;
    double mean = 0.0;
    for (long long f = 0; f < features; ++f) {
        const long long pixel = f / channels;
        const long long channel = f - pixel * channels;
        const long long by = pixel / 2;
        const long long bx = pixel - by * 2;
        const long long iy = 2 * oy + by;
        const long long ix = 2 * ox + bx;
        const float v = (iy < height && ix < width)
                            ? input[((item * height + iy) * width + ix) *
                                        channels +
                                    channel]
                            : 0.0f;
        mean += static_cast<double>(v);
    }
    mean /= static_cast<double>(features);
    double var = 0.0;
    for (long long f = 0; f < features; ++f) {
        const long long pixel = f / channels;
        const long long channel = f - pixel * channels;
        const long long by = pixel / 2;
        const long long bx = pixel - by * 2;
        const long long iy = 2 * oy + by;
        const long long ix = 2 * ox + bx;
        const float v = (iy < height && ix < width)
                            ? input[((item * height + iy) * width + ix) *
                                        channels +
                                    channel]
                            : 0.0f;
        const double delta = static_cast<double>(v) - mean;
        var += delta * delta;
    }
    const double inv = 1.0 / sqrt(var / static_cast<double>(features) + eps);
    for (long long f = 0; f < features; ++f) {
        const long long pixel = f / channels;
        const long long channel = f - pixel * channels;
        const long long by = pixel / 2;
        const long long bx = pixel - by * 2;
        const long long iy = 2 * oy + by;
        const long long ix = 2 * ox + bx;
        const float v = (iy < height && ix < width)
                            ? input[((item * height + iy) * width + ix) *
                                        channels +
                                    channel]
                            : 0.0f;
        output[row * features + f] =
            static_cast<float>((static_cast<double>(v) - mean) * inv *
                                   weight[f] +
                               bias[f]);
    }
}

__global__ void patch_merge_layer_norm_scalar_kernel(
    const float *input, const float *weight, const float *bias, float *output,
    long long batch, long long height, long long width, long long channels,
    float eps) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const long long out_h = (height + 1) / 2;
    const long long out_w = (width + 1) / 2;
    const long long features = 4 * channels;
    for (long long row = 0; row < batch * out_h * out_w; ++row) {
        const long long item = row / (out_h * out_w);
        const long long spatial = row - item * out_h * out_w;
        const long long oy = spatial / out_w;
        const long long ox = spatial - oy * out_w;
        double mean = 0.0;
        for (long long f = 0; f < features; ++f) {
            const long long pixel = f / channels;
            const long long channel = f - pixel * channels;
            const long long by = pixel / 2;
            const long long bx = pixel - by * 2;
            const long long iy = 2 * oy + by;
            const long long ix = 2 * ox + bx;
            const float v = (iy < height && ix < width)
                                ? input[((item * height + iy) * width + ix) *
                                            channels +
                                        channel]
                                : 0.0f;
            mean += static_cast<double>(v);
        }
        mean /= static_cast<double>(features);
        double var = 0.0;
        for (long long f = 0; f < features; ++f) {
            const long long pixel = f / channels;
            const long long channel = f - pixel * channels;
            const long long by = pixel / 2;
            const long long bx = pixel - by * 2;
            const long long iy = 2 * oy + by;
            const long long ix = 2 * ox + bx;
            const float v = (iy < height && ix < width)
                                ? input[((item * height + iy) * width + ix) *
                                            channels +
                                        channel]
                                : 0.0f;
            const double delta = static_cast<double>(v) - mean;
            var += delta * delta;
        }
        const double inv =
            1.0 / sqrt(var / static_cast<double>(features) + eps);
        for (long long f = 0; f < features; ++f) {
            const long long pixel = f / channels;
            const long long channel = f - pixel * channels;
            const long long by = pixel / 2;
            const long long bx = pixel - by * 2;
            const long long iy = 2 * oy + by;
            const long long ix = 2 * ox + bx;
            const float v = (iy < height && ix < width)
                                ? input[((item * height + iy) * width + ix) *
                                            channels +
                                        channel]
                                : 0.0f;
            output[row * features + f] =
                static_cast<float>((static_cast<double>(v) - mean) * inv *
                                       weight[f] +
                                   bias[f]);
        }
    }
}

__global__ void space_to_depth_norm_linear_kernel(
    const float *input, const float *norm_weight, const float *norm_bias,
    const float *projection_weight, const float *projection_bias,
    float *output, long long batch, long long height, long long width,
    long long channels, long long out_channels, long long block_size,
    float eps) {
    const long long out_h = ceil_div(height, block_size);
    const long long out_w = ceil_div(width, block_size);
    const long long rows = batch * out_h * out_w;
    const long long features = block_size * block_size * channels;
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * out_channels) return;
    const long long oc = idx % out_channels;
    const long long row = idx / out_channels;
    const long long item = row / (out_h * out_w);
    const long long spatial = row - item * out_h * out_w;
    const long long oy = spatial / out_w;
    const long long ox = spatial - oy * out_w;
    double mean = 0.0;
    for (long long f = 0; f < features; ++f)
        mean += static_cast<double>(space_to_depth_value(
            input, item, oy, ox, f, height, width, channels, block_size));
    mean /= static_cast<double>(features);
    double var = 0.0;
    for (long long f = 0; f < features; ++f) {
        const double delta =
            static_cast<double>(space_to_depth_value(
                input, item, oy, ox, f, height, width, channels, block_size)) -
            mean;
        var += delta * delta;
    }
    const double inv = 1.0 / sqrt(var / static_cast<double>(features) + eps);
    double sum = projection_bias == nullptr ? 0.0 : projection_bias[oc];
    const float *proj = projection_weight + oc * features;
    for (long long f = 0; f < features; ++f) {
        const double normed =
            (static_cast<double>(space_to_depth_value(
                 input, item, oy, ox, f, height, width, channels,
                 block_size)) -
             mean) *
                inv * norm_weight[f] +
            (norm_bias == nullptr ? 0.0 : norm_bias[f]);
        sum += static_cast<double>(proj[f]) * normed;
    }
    output[idx] = static_cast<float>(sum);
}

__global__ void space_to_depth_norm_linear_scalar_kernel(
    const float *input, const float *norm_weight, const float *norm_bias,
    const float *projection_weight, const float *projection_bias,
    float *output, long long batch, long long height, long long width,
    long long channels, long long out_channels, long long block_size,
    float eps) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const long long out_h = ceil_div(height, block_size);
    const long long out_w = ceil_div(width, block_size);
    const long long rows = batch * out_h * out_w;
    const long long features = block_size * block_size * channels;
    for (long long row = 0; row < rows; ++row) {
        const long long item = row / (out_h * out_w);
        const long long spatial = row - item * out_h * out_w;
        const long long oy = spatial / out_w;
        const long long ox = spatial - oy * out_w;
        double mean = 0.0;
        for (long long f = 0; f < features; ++f)
            mean += static_cast<double>(space_to_depth_value(
                input, item, oy, ox, f, height, width, channels, block_size));
        mean /= static_cast<double>(features);
        double var = 0.0;
        for (long long f = 0; f < features; ++f) {
            const double delta =
                static_cast<double>(space_to_depth_value(
                    input, item, oy, ox, f, height, width, channels,
                    block_size)) -
                mean;
            var += delta * delta;
        }
        const double inv =
            1.0 / sqrt(var / static_cast<double>(features) + eps);
        for (long long oc = 0; oc < out_channels; ++oc) {
            double sum = projection_bias == nullptr ? 0.0 : projection_bias[oc];
            const float *proj = projection_weight + oc * features;
            for (long long f = 0; f < features; ++f) {
                const double normed =
                    (static_cast<double>(space_to_depth_value(
                         input, item, oy, ox, f, height, width, channels,
                         block_size)) -
                     mean) *
                        inv * norm_weight[f] +
                    (norm_bias == nullptr ? 0.0 : norm_bias[f]);
                sum += static_cast<double>(proj[f]) * normed;
            }
            output[row * out_channels + oc] = static_cast<float>(sum);
        }
    }
}

__global__ void edge_mlp_first_kernel(const float *hidden,
                                      const float *first_weight,
                                      const float *first_bias, float *left,
                                      float *right, long long batch,
                                      long long length) {
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    const long long rows = batch * length;
    if (idx >= rows * kEdgeFeatures) return;
    const long long feature = idx % kEdgeFeatures;
    const long long row = idx / kEdgeFeatures;
    const float *source = hidden + row * kEdgeFeatures;
    const float *weights = first_weight + feature * (2 * kEdgeFeatures);
    double lhs = 0.0;
    double rhs = first_bias[feature];
    for (long long i = 0; i < kEdgeFeatures; ++i) {
        lhs += static_cast<double>(source[i]) * weights[i];
        rhs += static_cast<double>(source[i]) * weights[kEdgeFeatures + i];
    }
    left[idx] = static_cast<float>(lhs);
    right[idx] = static_cast<float>(rhs);
}

__global__ void edge_mlp_pair_kernel(const float *left, const float *right,
                                     const float *second_weight,
                                     const float *second_bias, float *output,
                                     long long batch, long long length) {
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = batch * length * length * kEdgeClasses;
    if (idx >= total) return;
    const long long pair = idx % (length * length);
    const long long cls = (idx / (length * length)) % kEdgeClasses;
    const long long item = idx / (length * length * kEdgeClasses);
    const long long lhs_index = pair / length;
    const long long rhs_index = pair - lhs_index * length;
    const float *lhs = left + (item * length + lhs_index) * kEdgeFeatures;
    const float *rhs = right + (item * length + rhs_index) * kEdgeFeatures;
    const float *weights = second_weight + cls * kEdgeFeatures;
    double sum = second_bias[cls];
    for (long long f = 0; f < kEdgeFeatures; ++f)
        sum += static_cast<double>(weights[f]) *
               gelu_erf_value(lhs[f] + rhs[f]);
    output[idx] = static_cast<float>(sum);
}

__global__ void edge_mlp_scalar_kernel(
    const float *hidden, const float *first_weight, const float *first_bias,
    const float *second_weight, const float *second_bias, float *left,
    float *right, float *output, long long batch, long long length) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const long long rows = batch * length;
    for (long long row = 0; row < rows; ++row) {
        const float *source = hidden + row * kEdgeFeatures;
        for (long long feature = 0; feature < kEdgeFeatures; ++feature) {
            const float *weights = first_weight + feature * (2 * kEdgeFeatures);
            double lhs = 0.0;
            double rhs = first_bias[feature];
            for (long long i = 0; i < kEdgeFeatures; ++i) {
                lhs += static_cast<double>(source[i]) * weights[i];
                rhs += static_cast<double>(source[i]) *
                       weights[kEdgeFeatures + i];
            }
            left[row * kEdgeFeatures + feature] = static_cast<float>(lhs);
            right[row * kEdgeFeatures + feature] = static_cast<float>(rhs);
        }
    }
    for (long long item = 0; item < batch; ++item)
        for (long long lhs_index = 0; lhs_index < length; ++lhs_index)
            for (long long rhs_index = 0; rhs_index < length; ++rhs_index) {
                const long long pair = lhs_index * length + rhs_index;
                const float *lhs =
                    left + (item * length + lhs_index) * kEdgeFeatures;
                const float *rhs =
                    right + (item * length + rhs_index) * kEdgeFeatures;
                for (long long cls = 0; cls < kEdgeClasses; ++cls) {
                    const float *weights = second_weight + cls * kEdgeFeatures;
                    double sum = second_bias[cls];
                    for (long long f = 0; f < kEdgeFeatures; ++f)
                        sum += static_cast<double>(weights[f]) *
                               gelu_erf_value(lhs[f] + rhs[f]);
                    output[(item * kEdgeClasses + cls) * length * length +
                           pair] = static_cast<float>(sum);
                }
            }
}

__global__ void vision_rope_2d_kernel(
    const float *x, const float *cosine, const float *sine,
    const int32_t *positions, float *output, long long batch, long long heads,
    long long tokens, long long head_dim, long long max_position,
    bool global_split) {
    const long long pairs = head_dim / 4;
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    const long long rows = batch * heads * tokens;
    if (idx >= rows * pairs) return;
    const long long pair = idx % pairs;
    const long long row = idx / pairs;
    const long long token = row % tokens;
    const long long item = row / (heads * tokens);
    int px = positions[(item * tokens + token) * 2];
    int py = positions[(item * tokens + token) * 2 + 1];
    px = px < 0 ? 0 : (px >= max_position ? static_cast<int>(max_position - 1) : px);
    py = py < 0 ? 0 : (py >= max_position ? static_cast<int>(max_position - 1) : py);
    const float cx = cosine[px * pairs + pair];
    const float sx = sine[px * pairs + pair];
    const float cy = cosine[py * pairs + pair];
    const float sy = sine[py * pairs + pair];
    const float *src = x + row * head_dim;
    float *dst = output + row * head_dim;
    if (global_split) {
        const float x0 = src[pair];
        const float y0 = src[pairs + pair];
        const float x1 = src[2 * pairs + pair];
        const float y1 = src[3 * pairs + pair];
        dst[pair] = x0 * cx - x1 * sx;
        dst[pairs + pair] = y0 * cy - y1 * sy;
        dst[2 * pairs + pair] = x0 * sx + x1 * cx;
        dst[3 * pairs + pair] = y0 * sy + y1 * cy;
    } else {
        const float x0 = src[pair];
        const float x1 = src[pairs + pair];
        const float y0 = src[2 * pairs + pair];
        const float y1 = src[3 * pairs + pair];
        dst[pair] = x0 * cx - x1 * sx;
        dst[pairs + pair] = x0 * sx + x1 * cx;
        dst[2 * pairs + pair] = y0 * cy - y1 * sy;
        dst[3 * pairs + pair] = y0 * sy + y1 * cy;
    }
}

__global__ void vision_rope_2d_scalar_kernel(
    const float *x, const float *cosine, const float *sine,
    const int32_t *positions, float *output, long long batch, long long heads,
    long long tokens, long long head_dim, long long max_position,
    bool global_split) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const long long pairs = head_dim / 4;
    const long long rows = batch * heads * tokens;
    for (long long row = 0; row < rows; ++row) {
        const long long token = row % tokens;
        const long long item = row / (heads * tokens);
        int px = positions[(item * tokens + token) * 2];
        int py = positions[(item * tokens + token) * 2 + 1];
        px = px < 0 ? 0 : (px >= max_position ? static_cast<int>(max_position - 1) : px);
        py = py < 0 ? 0 : (py >= max_position ? static_cast<int>(max_position - 1) : py);
        for (long long pair = 0; pair < pairs; ++pair) {
            const float cx = cosine[px * pairs + pair];
            const float sx = sine[px * pairs + pair];
            const float cy = cosine[py * pairs + pair];
            const float sy = sine[py * pairs + pair];
            const float *src = x + row * head_dim;
            float *dst = output + row * head_dim;
            if (global_split) {
                const float x0 = src[pair];
                const float y0 = src[pairs + pair];
                const float x1 = src[2 * pairs + pair];
                const float y1 = src[3 * pairs + pair];
                dst[pair] = x0 * cx - x1 * sx;
                dst[pairs + pair] = y0 * cy - y1 * sy;
                dst[2 * pairs + pair] = x0 * sx + x1 * cx;
                dst[3 * pairs + pair] = y0 * sy + y1 * cy;
            } else {
                const float x0 = src[pair];
                const float x1 = src[pairs + pair];
                const float y0 = src[2 * pairs + pair];
                const float y1 = src[3 * pairs + pair];
                dst[pair] = x0 * cx - x1 * sx;
                dst[pairs + pair] = x0 * sx + x1 * cx;
                dst[2 * pairs + pair] = y0 * cy - y1 * sy;
                dst[3 * pairs + pair] = y0 * sy + y1 * cy;
            }
        }
    }
}

__global__ void interpolate_position_2d_kernel(
    const float *table, float *output, long long input_h, long long input_w,
    long long output_h, long long output_w, long long channels,
    bool align_corners) {
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = output_h * output_w * channels;
    if (idx >= total) return;
    const long long channel = idx % channels;
    const long long spatial = idx / channels;
    const long long oy = spatial / output_w;
    const long long ox = spatial - oy * output_w;
    const double sy = align_corners && output_h > 1
                          ? static_cast<double>(oy) * (input_h - 1) /
                                (output_h - 1)
                          : (static_cast<double>(oy) + 0.5) * input_h /
                                    output_h -
                                0.5;
    const double sx = align_corners && output_w > 1
                          ? static_cast<double>(ox) * (input_w - 1) /
                                (output_w - 1)
                          : (static_cast<double>(ox) + 0.5) * input_w /
                                    output_w -
                                0.5;
    const double cy = sy < 0.0 ? 0.0 : (sy > input_h - 1 ? input_h - 1 : sy);
    const double cx = sx < 0.0 ? 0.0 : (sx > input_w - 1 ? input_w - 1 : sx);
    const long long y0 = static_cast<long long>(floor(cy));
    const long long x0 = static_cast<long long>(floor(cx));
    const long long y1 = y0 + 1 < input_h ? y0 + 1 : input_h - 1;
    const long long x1 = x0 + 1 < input_w ? x0 + 1 : input_w - 1;
    const float wy = static_cast<float>(cy - y0);
    const float wx = static_cast<float>(cx - x0);
    const float a = table[(y0 * input_w + x0) * channels + channel];
    const float b = table[(y0 * input_w + x1) * channels + channel];
    const float c = table[(y1 * input_w + x0) * channels + channel];
    const float d = table[(y1 * input_w + x1) * channels + channel];
    const float top = a + wx * (b - a);
    const float bottom = c + wx * (d - c);
    output[idx] = top + wy * (bottom - top);
}

__global__ void factorized_position_2d_kernel(
    const int32_t *position_ids, const float *table, const int32_t *valid_mask,
    float *output, long long batch, long long tokens, long long max_position,
    long long channels) {
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = batch * tokens * channels;
    if (idx >= total) return;
    const long long channel = idx % channels;
    const long long token = idx / channels;
    const int px = position_ids[token * 2];
    const int py = position_ids[token * 2 + 1];
    if (valid_mask[token] == 0 || px < 0 || px >= max_position || py < 0 ||
        py >= max_position) {
        output[idx] = 0.0f;
        return;
    }
    output[idx] = table[px * channels + channel] +
                  table[(max_position + py) * channels + channel];
}

__global__ void get_relative_position_kernel(const float *table, float *output,
                                             long long width, long long dim) {
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = width * width * dim;
    if (idx >= total) return;
    const long long channel = idx % dim;
    const long long key = (idx / dim) % width;
    const long long query = idx / (dim * width);
    output[idx] = table[(width - query - 1 + key) * dim + channel];
}

__global__ void add_relative_position_2d_kernel(
    const float *attention, const float *relative_h, const float *relative_w,
    float *output, long long batches, long long query_h, long long query_w,
    long long key_h, long long key_w) {
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = batches * query_h * query_w * key_h * key_w;
    if (idx >= total) return;
    const long long kw = idx % key_w;
    const long long kh = (idx / key_w) % key_h;
    const long long qw = (idx / (key_w * key_h)) % query_w;
    const long long qh = (idx / (key_w * key_h * query_w)) % query_h;
    const long long b = idx / (key_w * key_h * query_w * query_h);
    output[idx] = attention[idx] +
                  relative_h[((b * query_h + qh) * query_w + qw) * key_h +
                             kh] +
                  relative_w[((b * query_h + qh) * query_w + qw) * key_w +
                             kw];
}

__global__ void window_partition_kernel(const float *image, float *windows,
                                        long long height, long long width,
                                        long long channels,
                                        long long window_size) {
    const long long windows_x = ceil_div(width, window_size);
    const long long windows_y = ceil_div(height, window_size);
    const long long total =
        windows_y * windows_x * window_size * window_size * channels;
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    long long t = idx;
    const long long c = t % channels;
    t /= channels;
    const long long x = t % window_size;
    t /= window_size;
    const long long y = t % window_size;
    t /= window_size;
    const long long wx = t % windows_x;
    const long long wy = t / windows_x;
    const long long iy = wy * window_size + y;
    const long long ix = wx * window_size + x;
    windows[idx] =
        (iy < height && ix < width) ? image[(iy * width + ix) * channels + c]
                                    : 0.0f;
}

__global__ void window_partition_scalar_kernel(
    const float *image, float *windows, long long height, long long width,
    long long channels, long long window_size) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const long long windows_x = ceil_div(width, window_size);
    const long long windows_y = ceil_div(height, window_size);
    const long long total =
        windows_y * windows_x * window_size * window_size * channels;
    for (long long idx = 0; idx < total; ++idx) {
        long long t = idx;
        const long long c = t % channels;
        t /= channels;
        const long long x = t % window_size;
        t /= window_size;
        const long long y = t % window_size;
        t /= window_size;
        const long long wx = t % windows_x;
        const long long wy = t / windows_x;
        const long long iy = wy * window_size + y;
        const long long ix = wx * window_size + x;
        windows[idx] = (iy < height && ix < width)
                           ? image[(iy * width + ix) * channels + c]
                           : 0.0f;
    }
}

__global__ void window_unpartition_kernel(const float *windows, float *image,
                                          long long height, long long width,
                                          long long channels,
                                          long long window_size) {
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = height * width * channels;
    if (idx >= total) return;
    const long long c = idx % channels;
    const long long x = (idx / channels) % width;
    const long long y = idx / (channels * width);
    const long long windows_x = ceil_div(width, window_size);
    const long long wy = y / window_size;
    const long long wx = x / window_size;
    const long long local_y = y - wy * window_size;
    const long long local_x = x - wx * window_size;
    image[idx] = windows[((((wy * windows_x + wx) * window_size + local_y) *
                               window_size +
                           local_x) *
                              channels +
                          c)];
}

__global__ void avg_pool2d_tokens_kernel(
    const float *input, float *output, long long batch, long long height,
    long long width, long long channels, long long kernel_h, long long kernel_w,
    long long stride_h, long long stride_w, bool ceil_mode) {
    const long long out_h = pool_out(height, kernel_h, stride_h, ceil_mode);
    const long long out_w = pool_out(width, kernel_w, stride_w, ceil_mode);
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = batch * out_h * out_w * channels;
    if (idx >= total) return;
    const long long channel = idx % channels;
    const long long spatial = (idx / channels) % (out_h * out_w);
    const long long item = idx / (channels * out_h * out_w);
    const long long oy = spatial / out_w;
    const long long ox = spatial - oy * out_w;
    const long long y0 = oy * stride_h;
    const long long x0 = ox * stride_w;
    const long long y1 = y0 + kernel_h < height ? y0 + kernel_h : height;
    const long long x1 = x0 + kernel_w < width ? x0 + kernel_w : width;
    float sum = 0.0f;
    for (long long y = y0; y < y1; ++y)
        for (long long x = x0; x < x1; ++x)
            sum += input[((item * height + y) * width + x) * channels +
                         channel];
    output[idx] =
        sum / static_cast<float>((y1 - y0) * (x1 - x0));
}

__global__ void pool_tokens_by_position_kernel(
    const float *input, const int32_t *position_ids, const int32_t *valid_mask,
    float *output, int32_t *output_mask, long long batch, long long tokens,
    long long channels, long long output_length, long long kernel_size,
    long long source_width) {
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = batch * output_length * channels;
    if (idx >= total) return;
    const long long channel = idx % channels;
    const long long bucket = (idx / channels) % output_length;
    const long long item = idx / (channels * output_length);
    const long long pooled_width = source_width / kernel_size;
    const float scale = sqrtf(static_cast<float>(channels)) /
                        static_cast<float>(kernel_size * kernel_size);
    float sum = 0.0f;
    int32_t mask = 0;
    for (long long token = 0; token < tokens; ++token) {
        const long long input_token = item * tokens + token;
        if (valid_mask[input_token] == 0) continue;
        const int px = position_ids[input_token * 2];
        const int py = position_ids[input_token * 2 + 1];
        if (px < 0 || py < 0) continue;
        const long long candidate =
            px / kernel_size + pooled_width * (py / kernel_size);
        if (candidate != bucket) continue;
        mask = 1;
        sum += input[input_token * channels + channel] * scale;
    }
    output[idx] = sum;
    if (channel == 0) output_mask[item * output_length + bucket] = mask;
}

__global__ void pool_tokens_by_position_scalar_kernel(
    const float *input, const int32_t *position_ids, const int32_t *valid_mask,
    float *output, int32_t *output_mask, long long batch, long long tokens,
    long long channels, long long output_length, long long kernel_size,
    long long source_width) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const long long pooled_width = source_width / kernel_size;
    const float scale = sqrtf(static_cast<float>(channels)) /
                        static_cast<float>(kernel_size * kernel_size);
    for (long long i = 0; i < batch * output_length * channels; ++i)
        output[i] = 0.0f;
    for (long long i = 0; i < batch * output_length; ++i) output_mask[i] = 0;
    for (long long item = 0; item < batch; ++item) {
        for (long long token = 0; token < tokens; ++token) {
            const long long input_token = item * tokens + token;
            if (valid_mask[input_token] == 0) continue;
            const int px = position_ids[input_token * 2];
            const int py = position_ids[input_token * 2 + 1];
            if (px < 0 || py < 0) continue;
            const long long bucket =
                px / kernel_size + pooled_width * (py / kernel_size);
            if (bucket < 0 || bucket >= output_length) continue;
            output_mask[item * output_length + bucket] = 1;
            for (long long channel = 0; channel < channels; ++channel)
                output[(item * output_length + bucket) * channels + channel] +=
                    input[input_token * channels + channel] * scale;
        }
    }
}

__global__ void timestep_embedding_kernel(const float *timesteps,
                                          float *output, long long count,
                                          long long dim, float max_period) {
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = count * dim;
    if (idx >= total) return;
    const long long col = idx % dim;
    const long long row = idx / dim;
    const long long half = dim / 2;
    if (col < half) {
        const float freq = expf(-logf(max_period) * col / half);
        output[idx] = cosf(timesteps[row] * freq);
    } else if (col < 2 * half) {
        const long long j = col - half;
        const float freq = expf(-logf(max_period) * j / half);
        output[idx] = sinf(timesteps[row] * freq);
    } else {
        output[idx] = 0.0f;
    }
}

__global__ void upscale_nearest_2d_kernel(const float *input, float *output,
                                          long long channels, long long height,
                                          long long width, long long scale_h,
                                          long long scale_w) {
    const long long out_h = height * scale_h;
    const long long out_w = width * scale_w;
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = channels * out_h * out_w;
    if (idx >= total) return;
    const long long ox = idx % out_w;
    const long long oy = (idx / out_w) % out_h;
    const long long c = idx / (out_w * out_h);
    output[idx] =
        input[(c * height + oy / scale_h) * width + ox / scale_w];
}

// ---------------------------------------------------------------------------
// Host references
// ---------------------------------------------------------------------------

std::vector<double> extract_patches_2d_ref(
    const std::vector<float> &input, long long batch, long long height,
    long long width, long long channels, long long kernel_h, long long kernel_w,
    long long stride_h, long long stride_w, long long pad_h, long long pad_w) {
    const long long out_h = patch_out(height, kernel_h, stride_h, pad_h);
    const long long out_w = patch_out(width, kernel_w, stride_w, pad_w);
    std::vector<double> out(batch * out_h * out_w * kernel_h * kernel_w *
                            channels);
    for (long long i = 0; i < static_cast<long long>(out.size()); ++i) {
        long long t = i;
        const long long c = t % channels;
        t /= channels;
        const long long kx = t % kernel_w;
        t /= kernel_w;
        const long long ky = t % kernel_h;
        t /= kernel_h;
        const long long ox = t % out_w;
        t /= out_w;
        const long long oy = t % out_h;
        const long long b = t / out_h;
        out[i] = patch_value_2d(input.data(), b, oy, ox, ky, kx, c, height,
                                width, channels, stride_h, stride_w, pad_h,
                                pad_w);
    }
    return out;
}

std::vector<double> extract_patches_3d_ref(
    const std::vector<float> &input, long long batch, long long frames,
    long long height, long long width, long long channels, long long kernel_t,
    long long kernel_h, long long kernel_w, long long stride_t,
    long long stride_h, long long stride_w, long long pad_t, long long pad_h,
    long long pad_w) {
    const long long out_t = patch_out(frames, kernel_t, stride_t, pad_t);
    const long long out_h = patch_out(height, kernel_h, stride_h, pad_h);
    const long long out_w = patch_out(width, kernel_w, stride_w, pad_w);
    std::vector<double> out(batch * out_t * out_h * out_w * kernel_t *
                            kernel_h * kernel_w * channels);
    for (long long i = 0; i < static_cast<long long>(out.size()); ++i) {
        long long t = i;
        const long long c = t % channels;
        t /= channels;
        const long long kx = t % kernel_w;
        t /= kernel_w;
        const long long ky = t % kernel_h;
        t /= kernel_h;
        const long long kt = t % kernel_t;
        t /= kernel_t;
        const long long ox = t % out_w;
        t /= out_w;
        const long long oy = t % out_h;
        t /= out_h;
        const long long ot = t % out_t;
        const long long b = t / out_t;
        out[i] = patch_value_3d(input.data(), b, ot, oy, ox, kt, ky, kx, c,
                                frames, height, width, channels, stride_t,
                                stride_h, stride_w, pad_t, pad_h, pad_w);
    }
    return out;
}

std::vector<double> vision_patch_projection_ref(
    const std::vector<float> &input, const std::vector<float> &weights,
    const std::vector<float> *bias, long long batch, long long height,
    long long width, long long in_channels, long long out_channels,
    long long kernel_h, long long kernel_w, long long stride_h,
    long long stride_w, long long pad_h, long long pad_w) {
    const long long out_h = patch_out(height, kernel_h, stride_h, pad_h);
    const long long out_w = patch_out(width, kernel_w, stride_w, pad_w);
    const long long rows = batch * out_h * out_w;
    const long long patch_dim = kernel_h * kernel_w * in_channels;
    std::vector<double> out(rows * out_channels);
    for (long long row = 0; row < rows; ++row) {
        const long long item = row / (out_h * out_w);
        const long long spatial = row - item * out_h * out_w;
        const long long oy = spatial / out_w;
        const long long ox = spatial - oy * out_w;
        for (long long oc = 0; oc < out_channels; ++oc) {
            double sum = bias == nullptr ? 0.0 : (*bias)[oc];
            const float *wptr = weights.data() + oc * patch_dim;
            long long feature = 0;
            for (long long ky = 0; ky < kernel_h; ++ky)
                for (long long kx = 0; kx < kernel_w; ++kx)
                    for (long long ic = 0; ic < in_channels; ++ic, ++feature)
                        sum += static_cast<double>(patch_value_2d(
                                   input.data(), item, oy, ox, ky, kx, ic,
                                   height, width, in_channels, stride_h,
                                   stride_w, pad_h, pad_w)) *
                               wptr[feature];
            out[row * out_channels + oc] = sum;
        }
    }
    return out;
}

std::vector<double> vision_patch_projection_3d_ref(
    const std::vector<float> &input, const std::vector<float> &weights,
    const std::vector<float> *bias, long long batch, long long frames,
    long long height, long long width, long long in_channels,
    long long out_channels, long long kernel_t, long long kernel_h,
    long long kernel_w, long long stride_t, long long stride_h,
    long long stride_w, long long pad_t, long long pad_h, long long pad_w) {
    const long long out_t = patch_out(frames, kernel_t, stride_t, pad_t);
    const long long out_h = patch_out(height, kernel_h, stride_h, pad_h);
    const long long out_w = patch_out(width, kernel_w, stride_w, pad_w);
    const long long rows = batch * out_t * out_h * out_w;
    const long long patch_dim = kernel_t * kernel_h * kernel_w * in_channels;
    std::vector<double> out(rows * out_channels);
    for (long long row = 0; row < rows; ++row) {
        const long long item = row / (out_t * out_h * out_w);
        long long spatial = row - item * out_t * out_h * out_w;
        const long long ot = spatial / (out_h * out_w);
        spatial -= ot * out_h * out_w;
        const long long oy = spatial / out_w;
        const long long ox = spatial - oy * out_w;
        for (long long oc = 0; oc < out_channels; ++oc) {
            double sum = bias == nullptr ? 0.0 : (*bias)[oc];
            const float *wptr = weights.data() + oc * patch_dim;
            long long feature = 0;
            for (long long kt = 0; kt < kernel_t; ++kt)
                for (long long ky = 0; ky < kernel_h; ++ky)
                    for (long long kx = 0; kx < kernel_w; ++kx)
                        for (long long ic = 0; ic < in_channels;
                             ++ic, ++feature)
                            sum += static_cast<double>(patch_value_3d(
                                       input.data(), item, ot, oy, ox, kt, ky,
                                       kx, ic, frames, height, width,
                                       in_channels, stride_t, stride_h,
                                       stride_w, pad_t, pad_h, pad_w)) *
                                   wptr[feature];
            out[row * out_channels + oc] = sum;
        }
    }
    return out;
}

std::vector<double> patch_merge_layer_norm_ref(
    const std::vector<float> &input, const std::vector<float> &weight,
    const std::vector<float> &bias, long long batch, long long height,
    long long width, long long channels, float eps) {
    const long long out_h = (height + 1) / 2;
    const long long out_w = (width + 1) / 2;
    const long long features = 4 * channels;
    std::vector<double> out(batch * out_h * out_w * features);
    std::vector<double> row(features);
    for (long long r = 0; r < batch * out_h * out_w; ++r) {
        const long long item = r / (out_h * out_w);
        const long long spatial = r - item * out_h * out_w;
        const long long oy = spatial / out_w;
        const long long ox = spatial - oy * out_w;
        for (long long f = 0; f < features; ++f) {
            const long long pixel = f / channels;
            const long long channel = f - pixel * channels;
            const long long by = pixel / 2;
            const long long bx = pixel - by * 2;
            const long long iy = 2 * oy + by;
            const long long ix = 2 * ox + bx;
            row[f] = (iy < height && ix < width)
                         ? input[((item * height + iy) * width + ix) *
                                     channels +
                                 channel]
                         : 0.0;
        }
        const double mean =
            std::accumulate(row.begin(), row.end(), 0.0) / features;
        double var = 0.0;
        for (double v : row) {
            const double d = v - mean;
            var += d * d;
        }
        const double inv = 1.0 / std::sqrt(var / features + eps);
        for (long long f = 0; f < features; ++f)
            out[r * features + f] =
                (row[f] - mean) * inv * weight[f] + bias[f];
    }
    return out;
}

std::vector<double> space_to_depth_norm_linear_ref(
    const std::vector<float> &input, const std::vector<float> &norm_weight,
    const std::vector<float> *norm_bias,
    const std::vector<float> &projection_weight,
    const std::vector<float> *projection_bias, long long batch,
    long long height, long long width, long long channels,
    long long out_channels, long long block_size, float eps) {
    const long long out_h = ceil_div(height, block_size);
    const long long out_w = ceil_div(width, block_size);
    const long long rows = batch * out_h * out_w;
    const long long features = block_size * block_size * channels;
    std::vector<double> out(rows * out_channels);
    std::vector<double> patch(features);
    for (long long row = 0; row < rows; ++row) {
        const long long item = row / (out_h * out_w);
        const long long spatial = row - item * out_h * out_w;
        const long long oy = spatial / out_w;
        const long long ox = spatial - oy * out_w;
        for (long long f = 0; f < features; ++f)
            patch[f] = space_to_depth_value(input.data(), item, oy, ox, f,
                                            height, width, channels,
                                            block_size);
        const double mean =
            std::accumulate(patch.begin(), patch.end(), 0.0) / features;
        double var = 0.0;
        for (double v : patch) {
            const double d = v - mean;
            var += d * d;
        }
        const double inv = 1.0 / std::sqrt(var / features + eps);
        for (long long f = 0; f < features; ++f)
            patch[f] = (patch[f] - mean) * inv * norm_weight[f] +
                       (norm_bias == nullptr ? 0.0 : (*norm_bias)[f]);
        for (long long oc = 0; oc < out_channels; ++oc) {
            double sum = projection_bias == nullptr ? 0.0 : (*projection_bias)[oc];
            const float *proj = projection_weight.data() + oc * features;
            for (long long f = 0; f < features; ++f)
                sum += static_cast<double>(proj[f]) * patch[f];
            out[row * out_channels + oc] = sum;
        }
    }
    return out;
}

std::vector<double> edge_mlp_ref(
    const std::vector<float> &hidden, const std::vector<float> &first_weight,
    const std::vector<float> &first_bias,
    const std::vector<float> &second_weight,
    const std::vector<float> &second_bias, long long batch,
    long long length) {
    const long long rows = batch * length;
    std::vector<float> left(rows * kEdgeFeatures);
    std::vector<float> right(rows * kEdgeFeatures);
    for (long long row = 0; row < rows; ++row) {
        const float *source = hidden.data() + row * kEdgeFeatures;
        for (long long feature = 0; feature < kEdgeFeatures; ++feature) {
            const float *weights =
                first_weight.data() + feature * (2 * kEdgeFeatures);
            double lhs = 0.0;
            double rhs = first_bias[feature];
            for (long long i = 0; i < kEdgeFeatures; ++i) {
                lhs += static_cast<double>(source[i]) * weights[i];
                rhs += static_cast<double>(source[i]) *
                       weights[kEdgeFeatures + i];
            }
            left[row * kEdgeFeatures + feature] = static_cast<float>(lhs);
            right[row * kEdgeFeatures + feature] = static_cast<float>(rhs);
        }
    }
    std::vector<double> out(batch * kEdgeClasses * length * length);
    for (long long item = 0; item < batch; ++item)
        for (long long lhs_index = 0; lhs_index < length; ++lhs_index)
            for (long long rhs_index = 0; rhs_index < length; ++rhs_index) {
                const long long pair = lhs_index * length + rhs_index;
                const float *lhs =
                    left.data() + (item * length + lhs_index) * kEdgeFeatures;
                const float *rhs =
                    right.data() + (item * length + rhs_index) * kEdgeFeatures;
                for (long long cls = 0; cls < kEdgeClasses; ++cls) {
                    const float *weights =
                        second_weight.data() + cls * kEdgeFeatures;
                    double sum = second_bias[cls];
                    for (long long f = 0; f < kEdgeFeatures; ++f)
                        sum += static_cast<double>(weights[f]) *
                               gelu_erf_value(lhs[f] + rhs[f]);
                    out[(item * kEdgeClasses + cls) * length * length + pair] =
                        sum;
                }
            }
    return out;
}

std::vector<double> vision_rope_ref(
    const std::vector<float> &x, const std::vector<float> &cosine,
    const std::vector<float> &sine, const std::vector<int32_t> &positions,
    long long batch, long long heads, long long tokens, long long head_dim,
    long long max_position, bool global_split) {
    const long long rows = batch * heads * tokens;
    const long long pairs = head_dim / 4;
    std::vector<double> out(x.size());
    for (long long row = 0; row < rows; ++row) {
        const long long token = row % tokens;
        const long long item = row / (heads * tokens);
        int px = positions[(item * tokens + token) * 2];
        int py = positions[(item * tokens + token) * 2 + 1];
        px = std::clamp(px, 0, static_cast<int>(max_position - 1));
        py = std::clamp(py, 0, static_cast<int>(max_position - 1));
        for (long long pair = 0; pair < pairs; ++pair) {
            const double cx = cosine[px * pairs + pair];
            const double sx = sine[px * pairs + pair];
            const double cy = cosine[py * pairs + pair];
            const double sy = sine[py * pairs + pair];
            const float *src = x.data() + row * head_dim;
            double *dst = out.data() + row * head_dim;
            if (global_split) {
                const double x0 = src[pair];
                const double y0 = src[pairs + pair];
                const double x1 = src[2 * pairs + pair];
                const double y1 = src[3 * pairs + pair];
                dst[pair] = x0 * cx - x1 * sx;
                dst[pairs + pair] = y0 * cy - y1 * sy;
                dst[2 * pairs + pair] = x0 * sx + x1 * cx;
                dst[3 * pairs + pair] = y0 * sy + y1 * cy;
            } else {
                const double x0 = src[pair];
                const double x1 = src[pairs + pair];
                const double y0 = src[2 * pairs + pair];
                const double y1 = src[3 * pairs + pair];
                dst[pair] = x0 * cx - x1 * sx;
                dst[pairs + pair] = x0 * sx + x1 * cx;
                dst[2 * pairs + pair] = y0 * cy - y1 * sy;
                dst[3 * pairs + pair] = y0 * sy + y1 * cy;
            }
        }
    }
    return out;
}

std::vector<double> interpolate_position_ref(
    const std::vector<float> &table, long long input_h, long long input_w,
    long long output_h, long long output_w, long long channels,
    bool align_corners) {
    std::vector<double> out(output_h * output_w * channels);
    for (long long spatial = 0; spatial < output_h * output_w; ++spatial) {
        const long long oy = spatial / output_w;
        const long long ox = spatial - oy * output_w;
        const double sy = align_corners && output_h > 1
                              ? static_cast<double>(oy) * (input_h - 1) /
                                    (output_h - 1)
                              : (static_cast<double>(oy) + 0.5) * input_h /
                                        output_h -
                                    0.5;
        const double sx = align_corners && output_w > 1
                              ? static_cast<double>(ox) * (input_w - 1) /
                                    (output_w - 1)
                              : (static_cast<double>(ox) + 0.5) * input_w /
                                        output_w -
                                    0.5;
        const double cy =
            std::clamp(sy, 0.0, static_cast<double>(input_h - 1));
        const double cx =
            std::clamp(sx, 0.0, static_cast<double>(input_w - 1));
        const long long y0 = static_cast<long long>(std::floor(cy));
        const long long x0 = static_cast<long long>(std::floor(cx));
        const long long y1 = std::min(y0 + 1, input_h - 1);
        const long long x1 = std::min(x0 + 1, input_w - 1);
        const float wy = static_cast<float>(cy - y0);
        const float wx = static_cast<float>(cx - x0);
        for (long long c = 0; c < channels; ++c) {
            const float a = table[(y0 * input_w + x0) * channels + c];
            const float b = table[(y0 * input_w + x1) * channels + c];
            const float v = table[(y1 * input_w + x0) * channels + c];
            const float d = table[(y1 * input_w + x1) * channels + c];
            const float top = a + wx * (b - a);
            const float bottom = v + wx * (d - v);
            out[spatial * channels + c] = top + wy * (bottom - top);
        }
    }
    return out;
}

std::vector<double> factorized_position_ref(
    const std::vector<int32_t> &positions, const std::vector<float> &table,
    const std::vector<int32_t> &mask, long long batch, long long tokens,
    long long max_position, long long channels) {
    std::vector<double> out(batch * tokens * channels, 0.0);
    for (long long token = 0; token < batch * tokens; ++token) {
        const int px = positions[token * 2];
        const int py = positions[token * 2 + 1];
        if (mask[token] == 0 || px < 0 || px >= max_position || py < 0 ||
            py >= max_position)
            continue;
        for (long long c = 0; c < channels; ++c)
            out[token * channels + c] =
                table[px * channels + c] +
                table[(max_position + py) * channels + c];
    }
    return out;
}

std::vector<double> get_relative_position_ref(
    const std::vector<float> &table, long long width, long long dim) {
    std::vector<double> out(width * width * dim);
    for (long long q = 0; q < width; ++q)
        for (long long k = 0; k < width; ++k)
            for (long long d = 0; d < dim; ++d)
                out[(q * width + k) * dim + d] =
                    table[(width - q - 1 + k) * dim + d];
    return out;
}

std::vector<double> add_relative_position_ref(
    const std::vector<float> &attention, const std::vector<float> &rel_h,
    const std::vector<float> &rel_w, long long batches, long long query_h,
    long long query_w, long long key_h, long long key_w) {
    std::vector<double> out(attention.size());
    for (long long b = 0; b < batches; ++b)
        for (long long qh = 0; qh < query_h; ++qh)
            for (long long qw = 0; qw < query_w; ++qw)
                for (long long kh = 0; kh < key_h; ++kh)
                    for (long long kw = 0; kw < key_w; ++kw) {
                        const long long idx =
                            ((((b * query_h + qh) * query_w + qw) * key_h +
                              kh) *
                                 key_w +
                             kw);
                        out[idx] =
                            attention[idx] +
                            rel_h[((b * query_h + qh) * query_w + qw) *
                                      key_h +
                                  kh] +
                            rel_w[((b * query_h + qh) * query_w + qw) *
                                      key_w +
                                  kw];
                    }
    return out;
}

std::vector<double> window_partition_ref(const std::vector<float> &image,
                                         long long height, long long width,
                                         long long channels,
                                         long long window_size) {
    const long long windows_x = ceil_div(width, window_size);
    const long long windows_y = ceil_div(height, window_size);
    std::vector<double> out(windows_y * windows_x * window_size * window_size *
                            channels);
    for (long long idx = 0; idx < static_cast<long long>(out.size()); ++idx) {
        long long t = idx;
        const long long c = t % channels;
        t /= channels;
        const long long x = t % window_size;
        t /= window_size;
        const long long y = t % window_size;
        t /= window_size;
        const long long wx = t % windows_x;
        const long long wy = t / windows_x;
        const long long iy = wy * window_size + y;
        const long long ix = wx * window_size + x;
        out[idx] = (iy < height && ix < width)
                       ? image[(iy * width + ix) * channels + c]
                       : 0.0;
    }
    return out;
}

std::vector<double> window_unpartition_ref(const std::vector<float> &windows,
                                           long long height, long long width,
                                           long long channels,
                                           long long window_size) {
    const long long windows_x = ceil_div(width, window_size);
    std::vector<double> out(height * width * channels);
    for (long long idx = 0; idx < static_cast<long long>(out.size()); ++idx) {
        const long long c = idx % channels;
        const long long x = (idx / channels) % width;
        const long long y = idx / (channels * width);
        const long long wy = y / window_size;
        const long long wx = x / window_size;
        const long long ly = y - wy * window_size;
        const long long lx = x - wx * window_size;
        out[idx] = windows[((((wy * windows_x + wx) * window_size + ly) *
                                 window_size +
                             lx) *
                                channels +
                            c)];
    }
    return out;
}

std::vector<double> avg_pool2d_tokens_ref(
    const std::vector<float> &input, long long batch, long long height,
    long long width, long long channels, long long kernel_h, long long kernel_w,
    long long stride_h, long long stride_w, bool ceil_mode) {
    const long long out_h = pool_out(height, kernel_h, stride_h, ceil_mode);
    const long long out_w = pool_out(width, kernel_w, stride_w, ceil_mode);
    std::vector<double> out(batch * out_h * out_w * channels);
    for (long long idx = 0; idx < static_cast<long long>(out.size()); ++idx) {
        const long long c = idx % channels;
        const long long spatial = (idx / channels) % (out_h * out_w);
        const long long b = idx / (channels * out_h * out_w);
        const long long oy = spatial / out_w;
        const long long ox = spatial - oy * out_w;
        const long long y0 = oy * stride_h;
        const long long x0 = ox * stride_w;
        const long long y1 = std::min(y0 + kernel_h, height);
        const long long x1 = std::min(x0 + kernel_w, width);
        double sum = 0.0;
        for (long long y = y0; y < y1; ++y)
            for (long long x = x0; x < x1; ++x)
                sum += input[((b * height + y) * width + x) * channels + c];
        out[idx] = sum / static_cast<double>((y1 - y0) * (x1 - x0));
    }
    return out;
}

void pool_tokens_by_position_ref(
    const std::vector<float> &input, const std::vector<int32_t> &positions,
    const std::vector<int32_t> &valid_mask, std::vector<double> &output,
    std::vector<int32_t> &output_mask, long long batch, long long tokens,
    long long channels, long long output_length, long long kernel_size,
    long long source_width) {
    const long long pooled_width = source_width / kernel_size;
    const float scale = std::sqrt(static_cast<float>(channels)) /
                        static_cast<float>(kernel_size * kernel_size);
    output.assign(batch * output_length * channels, 0.0);
    output_mask.assign(batch * output_length, 0);
    for (long long item = 0; item < batch; ++item) {
        for (long long token = 0; token < tokens; ++token) {
            const long long input_token = item * tokens + token;
            if (valid_mask[input_token] == 0) continue;
            const int px = positions[input_token * 2];
            const int py = positions[input_token * 2 + 1];
            if (px < 0 || py < 0) continue;
            const long long bucket =
                px / kernel_size + pooled_width * (py / kernel_size);
            if (bucket < 0 || bucket >= output_length) continue;
            output_mask[item * output_length + bucket] = 1;
            for (long long c = 0; c < channels; ++c)
                output[(item * output_length + bucket) * channels + c] +=
                    input[input_token * channels + c] * scale;
        }
    }
}

std::vector<double> timestep_embedding_ref(const std::vector<float> &timesteps,
                                           long long count, long long dim,
                                           float max_period) {
    std::vector<double> out(count * dim);
    const long long half = dim / 2;
    for (long long i = 0; i < count; ++i) {
        for (long long j = 0; j < half; ++j) {
            const double freq =
                std::exp(-std::log(max_period) * j / static_cast<double>(half));
            const double arg = timesteps[i] * freq;
            out[i * dim + j] = std::cos(arg);
            out[i * dim + half + j] = std::sin(arg);
        }
        if ((dim & 1) != 0) out[i * dim + dim - 1] = 0.0;
    }
    return out;
}

std::vector<double> upscale_nearest_2d_ref(const std::vector<float> &input,
                                           long long channels,
                                           long long height, long long width,
                                           long long scale_h,
                                           long long scale_w) {
    const long long out_h = height * scale_h;
    const long long out_w = width * scale_w;
    std::vector<double> out(channels * out_h * out_w);
    for (long long idx = 0; idx < static_cast<long long>(out.size()); ++idx) {
        const long long ox = idx % out_w;
        const long long oy = (idx / out_w) % out_h;
        const long long c = idx / (out_w * out_h);
        out[idx] = input[(c * height + oy / scale_h) * width + ox / scale_w];
    }
    return out;
}

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

bool run_correctness() {
    std::printf("\n== Phase 13 correctness ==\n");
    qc::Rng rng(2313);
    bool ok = true;
    int checks = 0;

    const long long b = 2, h = 5, w = 6, c = 3, kh = 3, kw = 2;
    const long long sh = 2, sw = 2, ph = 1, pw = 1;
    const long long oh = patch_out(h, kh, sh, ph);
    const long long ow = patch_out(w, kw, sw, pw);
    std::vector<float> image = rng.uniforms(b * h * w * c, -1.0f, 1.0f);
    float *dimage = qc::dnew(image);
    float *dpatches = qc::dzero<float>(b * oh * ow * kh * kw * c);
    extract_patches_2d_kernel<<<qc::grid_for(b * oh * ow * kh * kw * c, kThreads),
                                kThreads>>>(dimage, dpatches, b, h, w, c, kh,
                                             kw, sh, sw, ph, pw);
    QC_SYNC();
    ok &= qc::compare(qc::d2h(dpatches, b * oh * ow * kh * kw * c),
                      extract_patches_2d_ref(image, b, h, w, c, kh, kw, sh,
                                             sw, ph, pw),
                      qc::Tol::fp32())
              .report("extract_patches_2d");
    ++checks;

    const long long frames = 4, h3 = 4, w3 = 5, c3 = 2, kt = 2, kh3 = 3,
                    kw3 = 2, st = 2, sh3 = 1, sw3 = 2, pt = 1, ph3 = 1,
                    pw3 = 0;
    const long long ot = patch_out(frames, kt, st, pt);
    const long long oh3 = patch_out(h3, kh3, sh3, ph3);
    const long long ow3 = patch_out(w3, kw3, sw3, pw3);
    std::vector<float> volume =
        rng.uniforms(b * frames * h3 * w3 * c3, -1.0f, 1.0f);
    float *dvolume = qc::dnew(volume);
    float *dpatches3 =
        qc::dzero<float>(b * ot * oh3 * ow3 * kt * kh3 * kw3 * c3);
    extract_patches_3d_kernel<<<
        qc::grid_for(b * ot * oh3 * ow3 * kt * kh3 * kw3 * c3, kThreads),
        kThreads>>>(dvolume, dpatches3, b, frames, h3, w3, c3, kt, kh3, kw3,
                    st, sh3, sw3, pt, ph3, pw3);
    QC_SYNC();
    ok &= qc::compare(qc::d2h(dpatches3,
                              b * ot * oh3 * ow3 * kt * kh3 * kw3 * c3),
                      extract_patches_3d_ref(volume, b, frames, h3, w3, c3, kt,
                                             kh3, kw3, st, sh3, sw3, pt, ph3,
                                             pw3),
                      qc::Tol::fp32())
              .report("extract_patches_3d");
    ++checks;

    const long long outc = 5;
    std::vector<float> weights =
        rng.uniforms(outc * kh * kw * c, -0.25f, 0.25f);
    std::vector<float> bias = rng.uniforms(outc, -0.1f, 0.1f);
    float *dweights = qc::dnew(weights);
    float *dbias = qc::dnew(bias);
    float *dproj = qc::dzero<float>(b * oh * ow * outc);
    vision_patch_projection_kernel<<<qc::grid_for(b * oh * ow * outc, kThreads),
                                     kThreads>>>(dimage, dweights, dbias,
                                                  dproj, b, h, w, c, outc, kh,
                                                  kw, sh, sw, ph, pw);
    QC_SYNC();
    ok &= qc::compare(qc::d2h(dproj, b * oh * ow * outc),
                      vision_patch_projection_ref(image, weights, &bias, b, h,
                                                  w, c, outc, kh, kw, sh, sw,
                                                  ph, pw),
                      qc::Tol::fp32())
              .report("vision_patch_projection");
    ++checks;

    const long long outc3 = 4;
    std::vector<float> weights3 =
        rng.uniforms(outc3 * kt * kh3 * kw3 * c3, -0.25f, 0.25f);
    std::vector<float> bias3 = rng.uniforms(outc3, -0.1f, 0.1f);
    float *dweights3 = qc::dnew(weights3);
    float *dbias3 = qc::dnew(bias3);
    float *dproj3 = qc::dzero<float>(b * ot * oh3 * ow3 * outc3);
    vision_patch_projection_3d_kernel<<<
        qc::grid_for(b * ot * oh3 * ow3 * outc3, kThreads), kThreads>>>(
        dvolume, dweights3, dbias3, dproj3, b, frames, h3, w3, c3, outc3, kt,
        kh3, kw3, st, sh3, sw3, pt, ph3, pw3);
    QC_SYNC();
    ok &= qc::compare(qc::d2h(dproj3, b * ot * oh3 * ow3 * outc3),
                      vision_patch_projection_3d_ref(
                          volume, weights3, &bias3, b, frames, h3, w3, c3,
                          outc3, kt, kh3, kw3, st, sh3, sw3, pt, ph3, pw3),
                      qc::Tol::fp32())
              .report("vision_patch_projection_3d");
    ++checks;

    const long long pmc = 4;
    std::vector<float> merge_input =
        rng.uniforms(b * h * w * pmc, -1.0f, 1.0f);
    std::vector<float> merge_weight =
        rng.uniforms(4 * pmc, 0.7f, 1.2f);
    std::vector<float> merge_bias = rng.uniforms(4 * pmc, -0.1f, 0.1f);
    float *dmerge_input = qc::dnew(merge_input);
    float *dmerge_weight = qc::dnew(merge_weight);
    float *dmerge_bias = qc::dnew(merge_bias);
    const long long pmoh = (h + 1) / 2, pmow = (w + 1) / 2;
    float *dmerge = qc::dzero<float>(b * pmoh * pmow * 4 * pmc);
    patch_merge_layer_norm_kernel<<<qc::grid_for(b * pmoh * pmow, kThreads),
                                    kThreads>>>(
        dmerge_input, dmerge_weight, dmerge_bias, dmerge, b, h, w, pmc, 1e-5f);
    QC_SYNC();
    ok &= qc::compare(qc::d2h(dmerge, b * pmoh * pmow * 4 * pmc),
                      patch_merge_layer_norm_ref(merge_input, merge_weight,
                                                 merge_bias, b, h, w, pmc,
                                                 1e-5f),
                      qc::Tol::fp32())
              .report("patch_merge_layer_norm");
    ++checks;

    const long long block = 2, sdc = 3, sdout = 5;
    const long long sdoh = ceil_div(h, block), sdow = ceil_div(w, block);
    std::vector<float> sd_input = rng.uniforms(b * h * w * sdc, -1.0f, 1.0f);
    std::vector<float> norm_weight =
        rng.uniforms(block * block * sdc, 0.8f, 1.2f);
    std::vector<float> norm_bias =
        rng.uniforms(block * block * sdc, -0.05f, 0.05f);
    std::vector<float> projection_weight =
        rng.uniforms(sdout * block * block * sdc, -0.2f, 0.2f);
    std::vector<float> projection_bias = rng.uniforms(sdout, -0.05f, 0.05f);
    float *dsd_input = qc::dnew(sd_input);
    float *dnw = qc::dnew(norm_weight);
    float *dnb = qc::dnew(norm_bias);
    float *dpw = qc::dnew(projection_weight);
    float *dpb = qc::dnew(projection_bias);
    float *dsd_out = qc::dzero<float>(b * sdoh * sdow * sdout);
    space_to_depth_norm_linear_kernel<<<
        qc::grid_for(b * sdoh * sdow * sdout, kThreads), kThreads>>>(
        dsd_input, dnw, dnb, dpw, dpb, dsd_out, b, h, w, sdc, sdout, block,
        1e-5f);
    QC_SYNC();
    ok &= qc::compare(qc::d2h(dsd_out, b * sdoh * sdow * sdout),
                      space_to_depth_norm_linear_ref(
                          sd_input, norm_weight, &norm_bias, projection_weight,
                          &projection_bias, b, h, w, sdc, sdout, block, 1e-5f),
                      qc::Tol::fp32().with_elementwise(2e-5, 2e-5))
              .report("space_to_depth_norm_linear");
    ++checks;

    const long long eb = 1, el = 4;
    std::vector<float> hidden =
        rng.uniforms(eb * el * kEdgeFeatures, -0.1f, 0.1f);
    std::vector<float> first_weight =
        rng.uniforms(kEdgeFeatures * 2 * kEdgeFeatures, -0.04f, 0.04f);
    std::vector<float> first_bias =
        rng.uniforms(kEdgeFeatures, -0.02f, 0.02f);
    std::vector<float> second_weight =
        rng.uniforms(kEdgeClasses * kEdgeFeatures, -0.05f, 0.05f);
    std::vector<float> second_bias =
        rng.uniforms(kEdgeClasses, -0.02f, 0.02f);
    float *dhid = qc::dnew(hidden);
    float *dfw = qc::dnew(first_weight);
    float *dfb = qc::dnew(first_bias);
    float *dsw2 = qc::dnew(second_weight);
    float *dsb2 = qc::dnew(second_bias);
    float *dleft = qc::dzero<float>(eb * el * kEdgeFeatures);
    float *dright = qc::dzero<float>(eb * el * kEdgeFeatures);
    float *dedge = qc::dzero<float>(eb * kEdgeClasses * el * el);
    edge_mlp_first_kernel<<<qc::grid_for(eb * el * kEdgeFeatures, kThreads),
                            kThreads>>>(dhid, dfw, dfb, dleft, dright, eb, el);
    edge_mlp_pair_kernel<<<
        qc::grid_for(eb * kEdgeClasses * el * el, kThreads), kThreads>>>(
        dleft, dright, dsw2, dsb2, dedge, eb, el);
    QC_SYNC();
    ok &= qc::compare(qc::d2h(dedge, eb * kEdgeClasses * el * el),
                      edge_mlp_ref(hidden, first_weight, first_bias,
                                   second_weight, second_bias, eb, el),
                      qc::Tol::fp32().with_elementwise(4e-5, 4e-5))
              .report("edge_mlp_256x7");
    ++checks;

    const long long rb = 2, heads = 2, tokens = 5, dim = 64, max_pos = 8;
    std::vector<float> rope_x =
        rng.uniforms(rb * heads * tokens * dim, -0.5f, 0.5f);
    std::vector<float> cosine =
        rng.uniforms(max_pos * (dim / 4), 0.2f, 1.0f);
    std::vector<float> sine =
        rng.uniforms(max_pos * (dim / 4), -0.5f, 0.5f);
    std::vector<int32_t> positions = rng.integers(rb * tokens * 2, -2, 10);
    float *drope_x = qc::dnew(rope_x);
    float *dcos = qc::dnew(cosine);
    float *dsin = qc::dnew(sine);
    int32_t *dpos = qc::dnew(positions);
    float *drope = qc::dzero<float>(rope_x.size());
    vision_rope_2d_kernel<<<qc::grid_for(rb * heads * tokens * (dim / 4),
                                          kThreads),
                             kThreads>>>(drope_x, dcos, dsin, dpos, drope, rb,
                                         heads, tokens, dim, max_pos, false);
    QC_SYNC();
    ok &= qc::compare(qc::d2h(drope, rope_x.size()),
                      vision_rope_ref(rope_x, cosine, sine, positions, rb,
                                      heads, tokens, dim, max_pos, false),
                      qc::Tol::fp32())
              .report("vision_rope_2d");
    ++checks;
    vision_rope_2d_kernel<<<qc::grid_for(rb * heads * tokens * (dim / 4),
                                          kThreads),
                             kThreads>>>(drope_x, dcos, dsin, dpos, drope, rb,
                                         heads, tokens, dim, max_pos, true);
    QC_SYNC();
    ok &= qc::compare(qc::d2h(drope, rope_x.size()),
                      vision_rope_ref(rope_x, cosine, sine, positions, rb,
                                      heads, tokens, dim, max_pos, true),
                      qc::Tol::fp32())
              .report("qwen_vision_rope_2d");
    ++checks;

    const long long ihp = 3, iwp = 4, ohp = 5, owp = 6, pc = 3;
    std::vector<float> pos_table =
        rng.uniforms(ihp * iwp * pc, -0.5f, 0.5f);
    float *dpos_table = qc::dnew(pos_table);
    float *dinterp = qc::dzero<float>(ohp * owp * pc);
    interpolate_position_2d_kernel<<<qc::grid_for(ohp * owp * pc, kThreads),
                                     kThreads>>>(dpos_table, dinterp, ihp, iwp,
                                                  ohp, owp, pc, false);
    QC_SYNC();
    ok &= qc::compare(qc::d2h(dinterp, ohp * owp * pc),
                      interpolate_position_ref(pos_table, ihp, iwp, ohp, owp,
                                               pc, false),
                      qc::Tol::fp32())
              .report("interpolate_position_2d half-pixel");
    ++checks;
    interpolate_position_2d_kernel<<<qc::grid_for(ohp * owp * pc, kThreads),
                                     kThreads>>>(dpos_table, dinterp, ihp, iwp,
                                                  ohp, owp, pc, true);
    QC_SYNC();
    ok &= qc::compare(qc::d2h(dinterp, ohp * owp * pc),
                      interpolate_position_ref(pos_table, ihp, iwp, ohp, owp,
                                               pc, true),
                      qc::Tol::fp32())
              .report("interpolate_position_2d align-corners");
    ++checks;

    const long long fpb = 2, fpt = 5, fpm = 6, fpc = 4;
    std::vector<int32_t> fppos = rng.integers(fpb * fpt * 2, -1, 7);
    std::vector<int32_t> fpmask(fpb * fpt, 1);
    fpmask[1] = 0;
    fpmask[7] = 0;
    std::vector<float> fptable = rng.uniforms(2 * fpm * fpc, -0.5f, 0.5f);
    int32_t *dfppos = qc::dnew(fppos);
    int32_t *dfpmask = qc::dnew(fpmask);
    float *dfptable = qc::dnew(fptable);
    float *dfpout = qc::dzero<float>(fpb * fpt * fpc);
    factorized_position_2d_kernel<<<qc::grid_for(fpb * fpt * fpc, kThreads),
                                    kThreads>>>(dfppos, dfptable, dfpmask,
                                                 dfpout, fpb, fpt, fpm, fpc);
    QC_SYNC();
    ok &= qc::compare(qc::d2h(dfpout, fpb * fpt * fpc),
                      factorized_position_ref(fppos, fptable, fpmask, fpb, fpt,
                                              fpm, fpc),
                      qc::Tol::fp32())
              .report("factorized_position_2d");
    ++checks;

    const long long rw = 4, rdim = 3;
    std::vector<float> rel_table = rng.uniforms((2 * rw - 1) * rdim, -0.2f, 0.2f);
    float *drel_table = qc::dnew(rel_table);
    float *drel_pos = qc::dzero<float>(rw * rw * rdim);
    get_relative_position_kernel<<<qc::grid_for(rw * rw * rdim, kThreads),
                                   kThreads>>>(drel_table, drel_pos, rw, rdim);
    QC_SYNC();
    ok &= qc::compare(qc::d2h(drel_pos, rw * rw * rdim),
                      get_relative_position_ref(rel_table, rw, rdim),
                      qc::Tol::fp32())
              .report("get_relative_position");
    ++checks;

    const long long ab = 2, qh = 2, qw2 = 3, akh = 3, akw = 2;
    std::vector<float> attention =
        rng.uniforms(ab * qh * qw2 * akh * akw, -0.4f, 0.4f);
    std::vector<float> rel_h = rng.uniforms(ab * qh * qw2 * akh, -0.2f, 0.2f);
    std::vector<float> rel_w = rng.uniforms(ab * qh * qw2 * akw, -0.2f, 0.2f);
    float *dattn = qc::dnew(attention);
    float *drel_h = qc::dnew(rel_h);
    float *drel_w = qc::dnew(rel_w);
    float *drel_add = qc::dzero<float>(attention.size());
    add_relative_position_2d_kernel<<<qc::grid_for(attention.size(), kThreads),
                                      kThreads>>>(dattn, drel_h, drel_w,
                                                   drel_add, ab, qh, qw2, akh,
                                                   akw);
    QC_SYNC();
    ok &= qc::compare(qc::d2h(drel_add, attention.size()),
                      add_relative_position_ref(attention, rel_h, rel_w, ab,
                                                qh, qw2, akh, akw),
                      qc::Tol::fp32())
              .report("add_relative_position_2d");
    ++checks;

    const long long wh = 5, ww = 6, wc = 3, win = 4;
    std::vector<float> win_image = rng.uniforms(wh * ww * wc, -1.0f, 1.0f);
    const long long windows_count = ceil_div(wh, win) * ceil_div(ww, win) *
                                    win * win * wc;
    float *dwin_image = qc::dnew(win_image);
    float *dwindows = qc::dzero<float>(windows_count);
    window_partition_kernel<<<qc::grid_for(windows_count, kThreads),
                              kThreads>>>(dwin_image, dwindows, wh, ww, wc,
                                           win);
    QC_SYNC();
    ok &= qc::compare(qc::d2h(dwindows, windows_count),
                      window_partition_ref(win_image, wh, ww, wc, win),
                      qc::Tol::fp32())
              .report("window_partition");
    ++checks;
    float *dunwin = qc::dzero<float>(win_image.size());
    window_unpartition_kernel<<<qc::grid_for(win_image.size(), kThreads),
                                kThreads>>>(dwindows, dunwin, wh, ww, wc, win);
    QC_SYNC();
    ok &= qc::compare(qc::d2h(dunwin, win_image.size()), to_ref(win_image),
                      qc::Tol::fp32())
              .report("window_unpartition");
    ++checks;

    float *dpool = qc::dzero<float>(b * pool_out(h, kh, sh, false) *
                                    pool_out(w, kw, sw, false) * c);
    avg_pool2d_tokens_kernel<<<
        qc::grid_for(b * pool_out(h, kh, sh, false) * pool_out(w, kw, sw, false) *
                         c,
                     kThreads),
        kThreads>>>(dimage, dpool, b, h, w, c, kh, kw, sh, sw, false);
    QC_SYNC();
    ok &= qc::compare(qc::d2h(dpool, b * pool_out(h, kh, sh, false) *
                                         pool_out(w, kw, sw, false) * c),
                      avg_pool2d_tokens_ref(image, b, h, w, c, kh, kw, sh, sw,
                                            false),
                      qc::Tol::fp32())
              .report("avg_pool2d_tokens floor");
    ++checks;
    qc::dfree(dpool);
    dpool = qc::dzero<float>(b * pool_out(h, kh, sh, true) *
                             pool_out(w, kw, sw, true) * c);
    avg_pool2d_tokens_kernel<<<
        qc::grid_for(b * pool_out(h, kh, sh, true) * pool_out(w, kw, sw, true) *
                         c,
                     kThreads),
        kThreads>>>(dimage, dpool, b, h, w, c, kh, kw, sh, sw, true);
    QC_SYNC();
    ok &= qc::compare(qc::d2h(dpool, b * pool_out(h, kh, sh, true) *
                                         pool_out(w, kw, sw, true) * c),
                      avg_pool2d_tokens_ref(image, b, h, w, c, kh, kw, sh, sw,
                                            true),
                      qc::Tol::fp32())
              .report("avg_pool2d_tokens ceil");
    ++checks;

    const long long ptb = 2, ptt = 8, ptc = 4, output_len = 12,
                    kernel_size = 2, source_width = 8;
    std::vector<float> token_input = rng.uniforms(ptb * ptt * ptc, -0.6f, 0.6f);
    std::vector<int32_t> token_pos = {
        0, 0, 1, 1, 2, 1, 3, 1, 4, 2, 5, 2, 7, 5, -1, 3,
        0, 0, 2, 0, 2, 1, 7, 1, 8, 1, 4, 4, 1, 5, 3, -2};
    std::vector<int32_t> token_mask(ptb * ptt, 1);
    token_mask[3] = 0;
    token_mask[12] = 0;
    float *dtok = qc::dnew(token_input);
    int32_t *dtpos = qc::dnew(token_pos);
    int32_t *dtmask = qc::dnew(token_mask);
    float *dtout = qc::dzero<float>(ptb * output_len * ptc);
    int32_t *dtout_mask = qc::dzero<int32_t>(ptb * output_len);
    pool_tokens_by_position_kernel<<<
        qc::grid_for(ptb * output_len * ptc, kThreads), kThreads>>>(
        dtok, dtpos, dtmask, dtout, dtout_mask, ptb, ptt, ptc, output_len,
        kernel_size, source_width);
    QC_SYNC();
    std::vector<double> pool_ref;
    std::vector<int32_t> pool_mask_ref;
    pool_tokens_by_position_ref(token_input, token_pos, token_mask, pool_ref,
                                pool_mask_ref, ptb, ptt, ptc, output_len,
                                kernel_size, source_width);
    ok &= qc::compare(qc::d2h(dtout, ptb * output_len * ptc), pool_ref,
                      qc::Tol::fp32())
              .report("pool_tokens_by_position values");
    ++checks;
    ok &= qc::compare(qc::d2h(dtout_mask, ptb * output_len),
                      to_ref_i32(pool_mask_ref), qc::Tol::exact())
              .report("pool_tokens_by_position mask");
    ++checks;

    const long long tcount = 4, tdim = 7;
    std::vector<float> timesteps = {-1.0f, 0.0f, 1.25f, 32.0f};
    float *dtimestep = qc::dnew(timesteps);
    float *dtemb = qc::dzero<float>(tcount * tdim);
    timestep_embedding_kernel<<<qc::grid_for(tcount * tdim, kThreads),
                                kThreads>>>(dtimestep, dtemb, tcount, tdim,
                                             10000.0f);
    QC_SYNC();
    ok &= qc::compare(qc::d2h(dtemb, tcount * tdim),
                      timestep_embedding_ref(timesteps, tcount, tdim,
                                             10000.0f),
                      qc::Tol::fp32())
              .report("timestep_embedding");
    ++checks;

    const long long uc = 3, uh = 4, uw = 5, usy = 2, usx = 3;
    std::vector<float> up =
        rng.uniforms(uc * uh * uw, -1.0f, 1.0f);
    float *dup = qc::dnew(up);
    float *dupo = qc::dzero<float>(uc * uh * usy * uw * usx);
    upscale_nearest_2d_kernel<<<qc::grid_for(uc * uh * usy * uw * usx, kThreads),
                                kThreads>>>(dup, dupo, uc, uh, uw, usy, usx);
    QC_SYNC();
    ok &= qc::compare(qc::d2h(dupo, uc * uh * usy * uw * usx),
                      upscale_nearest_2d_ref(up, uc, uh, uw, usy, usx),
                      qc::Tol::fp32())
              .report("upscale_nearest_2d");
    ++checks;

    qc::dfree(dimage, dpatches, dvolume, dpatches3, dweights, dbias, dproj,
              dweights3, dbias3, dproj3, dmerge_input, dmerge_weight,
              dmerge_bias, dmerge, dsd_input, dnw, dnb, dpw, dpb, dsd_out,
              dhid, dfw, dfb, dsw2, dsb2, dleft, dright, dedge, drope_x, dcos,
              dsin, dpos, drope, dpos_table, dinterp, dfppos, dfpmask,
              dfptable, dfpout, drel_table, drel_pos, dattn, drel_h, drel_w,
              drel_add, dwin_image, dwindows, dunwin, dpool, dtok, dtpos,
              dtmask, dtout, dtout_mask, dtimestep, dtemb, dup, dupo);

    std::printf("Phase 13 correctness checks: %d\n", checks);
    return ok;
}

void run_bench() {
    std::printf("\n== Phase 13 benchmarks ==\n");
    std::printf("   Timing note: direct CDNA3 routes compared with scalar GPU baselines.\n");
    qc::Rng rng(2413);
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

    const long long b = 2, h = 32, w = 32, c = 8, kh = 3, kw = 3;
    const long long oh = patch_out(h, kh, 1, 1), ow = patch_out(w, kw, 1, 1);
    std::vector<float> image = rng.uniforms(b * h * w * c, -1.0f, 1.0f);
    float *dimage = qc::dnew(image);
    float *dpatches = qc::dzero<float>(b * oh * ow * kh * kw * c);
    auto patches_s = qc::bench([&] {
        extract_patches_2d_scalar_kernel<<<1, 1>>>(dimage, dpatches, b, h, w, c,
                                                   kh, kw, 1, 1, 1, 1);
    }, 2, 3);
    auto patches_c = bench_repeated([&] {
        extract_patches_2d_kernel<<<
            qc::grid_for(b * oh * ow * kh * kw * c, kThreads), kThreads>>>(
            dimage, dpatches, b, h, w, c, kh, kw, 1, 1, 1, 1);
    }, 5000, 5, 20);
    const double patch_bytes =
        static_cast<double>(b * oh * ow * kh * kw * c) * 2.0 * sizeof(float);
    patches_s.report_bandwidth("extract_patches_2d scalar", patch_bytes);
    patches_c.report_bandwidth("extract_patches_2d candidate", patch_bytes);
    qc::report_ab("extract_patches_2d", patches_s, patches_c);

    const long long outc = 16;
    std::vector<float> weights =
        rng.uniforms(outc * kh * kw * c, -0.1f, 0.1f);
    std::vector<float> bias = rng.uniforms(outc, -0.05f, 0.05f);
    float *dw = qc::dnew(weights);
    float *db = qc::dnew(bias);
    float *dproj = qc::dzero<float>(b * oh * ow * outc);
    auto proj_s = qc::bench([&] {
        vision_patch_projection_scalar_kernel<<<1, 1>>>(
            dimage, dw, db, dproj, b, h, w, c, outc, kh, kw, 1, 1, 1, 1);
    }, 2, 3);
    auto proj_c = bench_repeated([&] {
        vision_patch_projection_kernel<<<
            qc::grid_for(b * oh * ow * outc, kThreads), kThreads>>>(
            dimage, dw, db, dproj, b, h, w, c, outc, kh, kw, 1, 1, 1, 1);
    }, 20, 5, 20);
    const double proj_flops =
        static_cast<double>(b * oh * ow * outc * kh * kw * c) * 2.0;
    proj_s.report_compute("vision_patch_projection scalar", proj_flops);
    proj_c.report_compute("vision_patch_projection candidate", proj_flops);
    qc::report_ab("vision_patch_projection", proj_s, proj_c);

    const long long pmc = 16;
    std::vector<float> merge_input =
        rng.uniforms(b * h * w * pmc, -1.0f, 1.0f);
    std::vector<float> merge_w = rng.uniforms(4 * pmc, 0.8f, 1.2f);
    std::vector<float> merge_b = rng.uniforms(4 * pmc, -0.05f, 0.05f);
    float *dmi = qc::dnew(merge_input);
    float *dmw = qc::dnew(merge_w);
    float *dmb = qc::dnew(merge_b);
    float *dmo = qc::dzero<float>(b * ((h + 1) / 2) * ((w + 1) / 2) * 4 * pmc);
    auto merge_s = qc::bench([&] {
        patch_merge_layer_norm_scalar_kernel<<<1, 1>>>(dmi, dmw, dmb, dmo, b, h,
                                                       w, pmc, 1e-5f);
    }, 2, 3);
    auto merge_c = bench_repeated([&] {
        patch_merge_layer_norm_kernel<<<
            qc::grid_for(b * ((h + 1) / 2) * ((w + 1) / 2), kThreads),
            kThreads>>>(dmi, dmw, dmb, dmo, b, h, w, pmc, 1e-5f);
    }, 50, 5, 20);
    const double merge_bytes =
        static_cast<double>(b * ((h + 1) / 2) * ((w + 1) / 2) * 4 * pmc) *
        4.0 * sizeof(float);
    merge_s.report_bandwidth("patch_merge_layer_norm scalar", merge_bytes);
    merge_c.report_bandwidth("patch_merge_layer_norm candidate", merge_bytes);
    qc::report_ab("patch_merge_layer_norm", merge_s, merge_c);

    const long long sdout = 16, block = 2;
    std::vector<float> norm_w = rng.uniforms(block * block * c, 0.8f, 1.2f);
    std::vector<float> norm_b = rng.uniforms(block * block * c, -0.05f, 0.05f);
    std::vector<float> proj_w =
        rng.uniforms(sdout * block * block * c, -0.1f, 0.1f);
    std::vector<float> proj_b = rng.uniforms(sdout, -0.05f, 0.05f);
    float *dnw = qc::dnew(norm_w);
    float *dnb = qc::dnew(norm_b);
    float *dpw = qc::dnew(proj_w);
    float *dpb = qc::dnew(proj_b);
    float *dsdo = qc::dzero<float>(b * ceil_div(h, block) * ceil_div(w, block) * sdout);
    auto sd_s = qc::bench([&] {
        space_to_depth_norm_linear_scalar_kernel<<<1, 1>>>(
            dimage, dnw, dnb, dpw, dpb, dsdo, b, h, w, c, sdout, block, 1e-5f);
    }, 2, 3);
    auto sd_c = bench_repeated([&] {
        space_to_depth_norm_linear_kernel<<<
            qc::grid_for(b * ceil_div(h, block) * ceil_div(w, block) * sdout,
                         kThreads),
            kThreads>>>(dimage, dnw, dnb, dpw, dpb, dsdo, b, h, w, c, sdout,
                        block, 1e-5f);
    }, 20, 5, 20);
    const double sd_flops =
        static_cast<double>(b * ceil_div(h, block) * ceil_div(w, block) *
                            sdout * block * block * c) *
        2.0;
    sd_s.report_compute("space_to_depth_norm_linear scalar", sd_flops);
    sd_c.report_compute("space_to_depth_norm_linear candidate", sd_flops);
    qc::report_ab("space_to_depth_norm_linear", sd_s, sd_c);

    const long long rb = 2, heads = 4, tokens = 64, dim = 64, max_pos = 32;
    std::vector<float> rx = rng.uniforms(rb * heads * tokens * dim, -0.5f, 0.5f);
    std::vector<float> rcos = rng.uniforms(max_pos * (dim / 4), 0.2f, 1.0f);
    std::vector<float> rsin = rng.uniforms(max_pos * (dim / 4), -0.5f, 0.5f);
    std::vector<int32_t> rpos = rng.integers(rb * tokens * 2, 0, max_pos - 1);
    float *drx = qc::dnew(rx);
    float *drc = qc::dnew(rcos);
    float *drs = qc::dnew(rsin);
    int32_t *drp = qc::dnew(rpos);
    float *dro = qc::dzero<float>(rx.size());
    auto rope_s = qc::bench([&] {
        vision_rope_2d_scalar_kernel<<<1, 1>>>(drx, drc, drs, drp, dro, rb,
                                               heads, tokens, dim, max_pos,
                                               false);
    }, 2, 3);
    auto rope_c = bench_repeated([&] {
        vision_rope_2d_kernel<<<
            qc::grid_for(rb * heads * tokens * (dim / 4), kThreads), kThreads>>>(
            drx, drc, drs, drp, dro, rb, heads, tokens, dim, max_pos, false);
    }, 5000, 5, 20);
    const double rope_bytes = static_cast<double>(rx.size()) * 2.0 * sizeof(float);
    rope_s.report_bandwidth("vision_rope_2d scalar", rope_bytes);
    rope_c.report_bandwidth("vision_rope_2d candidate", rope_bytes);
    qc::report_ab("vision_rope_2d", rope_s, rope_c);

    const long long wh = 65, ww = 67, wc = 8, win = 8;
    std::vector<float> wimg = rng.uniforms(wh * ww * wc, -1.0f, 1.0f);
    const long long wcount =
        ceil_div(wh, win) * ceil_div(ww, win) * win * win * wc;
    float *dwimg = qc::dnew(wimg);
    float *dwin = qc::dzero<float>(wcount);
    auto win_s = qc::bench([&] {
        window_partition_scalar_kernel<<<1, 1>>>(dwimg, dwin, wh, ww, wc, win);
    }, 2, 3);
    auto win_c = bench_repeated([&] {
        window_partition_kernel<<<qc::grid_for(wcount, kThreads), kThreads>>>(
            dwimg, dwin, wh, ww, wc, win);
    }, 5000, 5, 20);
    const double win_bytes = static_cast<double>(wcount) * 2.0 * sizeof(float);
    win_s.report_bandwidth("window_partition scalar", win_bytes);
    win_c.report_bandwidth("window_partition candidate", win_bytes);
    qc::report_ab("window_partition", win_s, win_c);

    const long long ptb = 2, ptt = 256, ptc = 16, output_len = 128,
                    kernel_size = 2, source_width = 32;
    std::vector<float> toks = rng.uniforms(ptb * ptt * ptc, -0.5f, 0.5f);
    std::vector<int32_t> tpos(ptb * ptt * 2);
    std::vector<int32_t> tmask(ptb * ptt, 1);
    for (long long i = 0; i < ptb * ptt; ++i) {
        tpos[i * 2] = static_cast<int32_t>(i % source_width);
        tpos[i * 2 + 1] = static_cast<int32_t>((i / source_width) % 16);
        if ((i % 11) == 0) tmask[i] = 0;
    }
    float *dtoks = qc::dnew(toks);
    int32_t *dtpos = qc::dnew(tpos);
    int32_t *dtmask = qc::dnew(tmask);
    float *dtout = qc::dzero<float>(ptb * output_len * ptc);
    int32_t *dtout_mask = qc::dzero<int32_t>(ptb * output_len);
    auto pool_s = qc::bench([&] {
        pool_tokens_by_position_scalar_kernel<<<1, 1>>>(
            dtoks, dtpos, dtmask, dtout, dtout_mask, ptb, ptt, ptc,
            output_len, kernel_size, source_width);
    }, 2, 3);
    auto pool_c = qc::bench([&] {
        pool_tokens_by_position_kernel<<<
            qc::grid_for(ptb * output_len * ptc, kThreads), kThreads>>>(
            dtoks, dtpos, dtmask, dtout, dtout_mask, ptb, ptt, ptc,
            output_len, kernel_size, source_width);
    }, 5, 20);
    const double pool_bytes =
        static_cast<double>(ptb * output_len * ptc * ptt) * sizeof(float);
    pool_s.report_bandwidth("pool_tokens_by_position scalar", pool_bytes);
    pool_c.report_bandwidth("pool_tokens_by_position candidate", pool_bytes);
    qc::report_ab("pool_tokens_by_position", pool_s, pool_c);

    const long long eb = 1, el = 8;
    std::vector<float> hidden =
        rng.uniforms(eb * el * kEdgeFeatures, -0.1f, 0.1f);
    std::vector<float> first_w =
        rng.uniforms(kEdgeFeatures * 2 * kEdgeFeatures, -0.04f, 0.04f);
    std::vector<float> first_b = rng.uniforms(kEdgeFeatures, -0.02f, 0.02f);
    std::vector<float> second_w =
        rng.uniforms(kEdgeClasses * kEdgeFeatures, -0.05f, 0.05f);
    std::vector<float> second_b = rng.uniforms(kEdgeClasses, -0.02f, 0.02f);
    float *deh = qc::dnew(hidden);
    float *defw = qc::dnew(first_w);
    float *defb = qc::dnew(first_b);
    float *desw = qc::dnew(second_w);
    float *desb = qc::dnew(second_b);
    float *del = qc::dzero<float>(eb * el * kEdgeFeatures);
    float *der = qc::dzero<float>(eb * el * kEdgeFeatures);
    float *deo = qc::dzero<float>(eb * kEdgeClasses * el * el);
    auto edge_s = qc::bench([&] {
        edge_mlp_scalar_kernel<<<1, 1>>>(deh, defw, defb, desw, desb, del, der,
                                         deo, eb, el);
    }, 2, 3);
    auto edge_c = qc::bench([&] {
        edge_mlp_first_kernel<<<qc::grid_for(eb * el * kEdgeFeatures, kThreads),
                                kThreads>>>(deh, defw, defb, del, der, eb, el);
        edge_mlp_pair_kernel<<<
            qc::grid_for(eb * kEdgeClasses * el * el, kThreads), kThreads>>>(
            del, der, desw, desb, deo, eb, el);
    }, 5, 20);
    const double edge_flops =
        static_cast<double>(eb * el * kEdgeFeatures * kEdgeFeatures * 4 +
                            eb * el * el * kEdgeClasses * kEdgeFeatures * 2);
    edge_s.report_compute("edge_mlp_256x7 scalar", edge_flops);
    edge_c.report_compute("edge_mlp_256x7 candidate", edge_flops);
    qc::report_ab("edge_mlp_256x7", edge_s, edge_c);

    qc::dfree(dimage, dpatches, dw, db, dproj, dmi, dmw, dmb, dmo, dnw, dnb,
              dpw, dpb, dsdo, drx, drc, drs, drp, dro, dwimg, dwin, dtoks,
              dtpos, dtmask, dtout, dtout_mask, deh, defw, defb, desw, desb,
              del, der, deo);
}

}  // namespace

int main(int argc, char **argv) {
    const bool do_bench = argc > 1 && std::string(argv[1]) == "--bench";
    qc::print_environment("phase13_vision");
    const bool ok = run_correctness();
    if (do_bench) run_bench();
    std::printf("%s\n", ok ? "ALL PASS" : "FAIL");
    return ok ? 0 : 1;
}
