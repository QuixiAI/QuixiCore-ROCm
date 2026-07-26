/**
 * @file
 * @brief Phase 12 convolution, pooling, and audio parity ports for CDNA3.
 */
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include "../../../../common/cdna3_harness.cuh"

namespace {

constexpr int kThreads = qc::kThreads;

enum PoolMode { kAverage = 0, kMaximum = 1 };

__host__ __device__ long long conv_out(long long input, long long kernel,
                                       long long stride, long long padding,
                                       long long dilation) {
    return (input + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1;
}

__host__ __device__ long long audio_conv_out(long long length, long long kernel,
                                             long long stride,
                                             long long padding,
                                             long long dilation) {
    return (length + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1;
}

std::vector<double> to_ref(const std::vector<float> &x) {
    return std::vector<double>(x.begin(), x.end());
}

__host__ __device__ float silu_value(float x) {
#ifdef __HIP_DEVICE_COMPILE__
    return x / (1.0f + expf(-x));
#else
    return x / (1.0f + std::exp(-x));
#endif
}

__host__ __device__ float softplus_value(float x) {
#ifdef __HIP_DEVICE_COMPILE__
    return fmaxf(x, 0.0f) + log1pf(expf(-fabsf(x)));
#else
    return std::max(x, 0.0f) + std::log1p(std::exp(-std::fabs(x)));
#endif
}

// ---------------------------------------------------------------------------
// Vision conv/pool kernels
// ---------------------------------------------------------------------------

__global__ void im2col_2d_kernel(const float *image, float *columns,
                                 long long batch, long long channels,
                                 long long ih, long long iw, long long kh,
                                 long long kw, long long sh, long long sw,
                                 long long ph, long long pw, long long dh,
                                 long long dw) {
    const long long oh = conv_out(ih, kh, sh, ph, dh);
    const long long ow = conv_out(iw, kw, sw, pw, dw);
    const long long patch = channels * kh * kw;
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = batch * oh * ow * patch;
    if (idx >= total) return;
    const long long p = idx % patch;
    const long long ow_i = (idx / patch) % ow;
    const long long oh_i = (idx / (patch * ow)) % oh;
    const long long n = idx / (patch * ow * oh);
    const long long c = p / (kh * kw);
    const long long rem = p - c * kh * kw;
    const long long kh_i = rem / kw;
    const long long kw_i = rem % kw;
    const long long in_y = oh_i * sh + kh_i * dh - ph;
    const long long in_x = ow_i * sw + kw_i * dw - pw;
    columns[idx] = (in_y >= 0 && in_y < ih && in_x >= 0 && in_x < iw)
                       ? image[((n * channels + c) * ih + in_y) * iw + in_x]
                       : 0.0f;
}

__global__ void im2col_2d_scalar_kernel(const float *image, float *columns,
                                        long long batch, long long channels,
                                        long long ih, long long iw,
                                        long long kh, long long kw,
                                        long long sh, long long sw,
                                        long long ph, long long pw,
                                        long long dh, long long dw) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const long long oh = conv_out(ih, kh, sh, ph, dh);
    const long long ow = conv_out(iw, kw, sw, pw, dw);
    const long long patch = channels * kh * kw;
    for (long long idx = 0; idx < batch * oh * ow * patch; ++idx) {
        const long long p = idx % patch;
        const long long ow_i = (idx / patch) % ow;
        const long long oh_i = (idx / (patch * ow)) % oh;
        const long long n = idx / (patch * ow * oh);
        const long long c = p / (kh * kw);
        const long long rem = p - c * kh * kw;
        const long long kh_i = rem / kw;
        const long long kw_i = rem % kw;
        const long long in_y = oh_i * sh + kh_i * dh - ph;
        const long long in_x = ow_i * sw + kw_i * dw - pw;
        columns[idx] = (in_y >= 0 && in_y < ih && in_x >= 0 && in_x < iw)
                           ? image[((n * channels + c) * ih + in_y) * iw + in_x]
                           : 0.0f;
    }
}

__global__ void im2col_3d_kernel(const float *volume, float *columns,
                                 long long batch, long long channels,
                                 long long id, long long ih, long long iw,
                                 long long kd, long long kh, long long kw,
                                 long long sd, long long sh, long long sw,
                                 long long pd, long long ph, long long pw,
                                 long long dd, long long dh, long long dw) {
    const long long od = conv_out(id, kd, sd, pd, dd);
    const long long oh = conv_out(ih, kh, sh, ph, dh);
    const long long ow = conv_out(iw, kw, sw, pw, dw);
    const long long patch = channels * kd * kh * kw;
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = batch * od * oh * ow * patch;
    if (idx >= total) return;
    const long long p = idx % patch;
    const long long ox = (idx / patch) % ow;
    const long long oy = (idx / (patch * ow)) % oh;
    const long long oz = (idx / (patch * ow * oh)) % od;
    const long long n = idx / (patch * ow * oh * od);
    const long long c = p / (kd * kh * kw);
    const long long r0 = p - c * kd * kh * kw;
    const long long kz = r0 / (kh * kw);
    const long long r1 = r0 - kz * kh * kw;
    const long long ky = r1 / kw;
    const long long kx = r1 % kw;
    const long long iz = oz * sd + kz * dd - pd;
    const long long iy = oy * sh + ky * dh - ph;
    const long long ix = ox * sw + kx * dw - pw;
    columns[idx] = (iz >= 0 && iz < id && iy >= 0 && iy < ih && ix >= 0 && ix < iw)
                       ? volume[(((n * channels + c) * id + iz) * ih + iy) * iw + ix]
                       : 0.0f;
}

__global__ void col2im_1d_kernel(const float *columns, float *signal,
                                 long long time_in, long long channels,
                                 long long kernel, long long stride,
                                 long long padding) {
    const long long time_out = (time_in - 1) * stride + kernel - 2 * padding;
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = channels * time_out;
    if (idx >= total) return;
    const long long channel = idx / time_out;
    const long long output_index = idx % time_out;
    const long long absolute = output_index + padding;
    float sum = 0.0f;
    for (long long input_index = 0; input_index < time_in; ++input_index) {
        const long long tap = absolute - input_index * stride;
        if (tap >= 0 && tap < kernel)
            sum += columns[(input_index * channels + channel) * kernel + tap];
    }
    signal[idx] = sum;
}

__global__ void col2im_2d_kernel(const float *columns, float *image,
                                 long long batch, long long channels,
                                 long long ih, long long iw, long long kh,
                                 long long kw, long long sh, long long sw,
                                 long long ph, long long pw, long long dh,
                                 long long dw) {
    const long long oh = conv_out(ih, kh, sh, ph, dh);
    const long long ow = conv_out(iw, kw, sw, pw, dw);
    const long long patch = channels * kh * kw;
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = batch * channels * ih * iw;
    if (idx >= total) return;
    const long long ix = idx % iw;
    const long long iy = (idx / iw) % ih;
    const long long c = (idx / (iw * ih)) % channels;
    const long long n = idx / (iw * ih * channels);
    float sum = 0.0f;
    for (long long oy = 0; oy < oh; ++oy)
        for (long long ox = 0; ox < ow; ++ox)
            for (long long ky = 0; ky < kh; ++ky)
                for (long long kx = 0; kx < kw; ++kx) {
                    if (iy == oy * sh + ky * dh - ph &&
                        ix == ox * sw + kx * dw - pw) {
                        const long long source =
                            ((n * oh + oy) * ow + ox) * patch +
                            (c * kh + ky) * kw + kx;
                        sum += columns[source];
                    }
                }
    image[idx] = sum;
}

__global__ void conv2d_kernel(const float *input, const float *weights,
                              const float *bias, float *output,
                              long long batch, long long icount,
                              long long ocount, long long ih, long long iw,
                              long long kh, long long kw, long long sh,
                              long long sw, long long ph, long long pw,
                              long long dh, long long dw) {
    const long long oh = conv_out(ih, kh, sh, ph, dh);
    const long long ow = conv_out(iw, kw, sw, pw, dw);
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = batch * ocount * oh * ow;
    if (idx >= total) return;
    const long long ox = idx % ow;
    const long long oy = (idx / ow) % oh;
    const long long oc = (idx / (ow * oh)) % ocount;
    const long long n = idx / (ow * oh * ocount);
    float sum = bias == nullptr ? 0.0f : bias[oc];
    for (long long ic = 0; ic < icount; ++ic)
        for (long long ky = 0; ky < kh; ++ky) {
            const long long iy = oy * sh + ky * dh - ph;
            if (iy < 0 || iy >= ih) continue;
            for (long long kx = 0; kx < kw; ++kx) {
                const long long ix = ox * sw + kx * dw - pw;
                if (ix < 0 || ix >= iw) continue;
                sum += input[((n * icount + ic) * ih + iy) * iw + ix] *
                       weights[((oc * icount + ic) * kh + ky) * kw + kx];
            }
        }
    output[idx] = sum;
}

__global__ void conv2d_scalar_kernel(const float *input, const float *weights,
                                     const float *bias, float *output,
                                     long long batch, long long icount,
                                     long long ocount, long long ih,
                                     long long iw, long long kh, long long kw,
                                     long long sh, long long sw, long long ph,
                                     long long pw, long long dh, long long dw) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const long long oh = conv_out(ih, kh, sh, ph, dh);
    const long long ow = conv_out(iw, kw, sw, pw, dw);
    for (long long idx = 0; idx < batch * ocount * oh * ow; ++idx) {
        const long long ox = idx % ow, oy = (idx / ow) % oh;
        const long long oc = (idx / (ow * oh)) % ocount;
        const long long n = idx / (ow * oh * ocount);
        float sum = bias == nullptr ? 0.0f : bias[oc];
        for (long long ic = 0; ic < icount; ++ic)
            for (long long ky = 0; ky < kh; ++ky) {
                const long long iy = oy * sh + ky * dh - ph;
                if (iy < 0 || iy >= ih) continue;
                for (long long kx = 0; kx < kw; ++kx) {
                    const long long ix = ox * sw + kx * dw - pw;
                    if (ix < 0 || ix >= iw) continue;
                    sum += input[((n * icount + ic) * ih + iy) * iw + ix] *
                           weights[((oc * icount + ic) * kh + ky) * kw + kx];
                }
            }
        output[idx] = sum;
    }
}

__global__ void depthwise_conv2d_kernel(
    const float *input, const float *weights, const float *bias, float *output,
    long long batch, long long channels, long long multiplier, long long ih,
    long long iw, long long kh, long long kw, long long sh, long long sw,
    long long ph, long long pw, long long dh, long long dw) {
    const long long oh = conv_out(ih, kh, sh, ph, dh);
    const long long ow = conv_out(iw, kw, sw, pw, dw);
    const long long output_channels = channels * multiplier;
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = batch * output_channels * oh * ow;
    if (idx >= total) return;
    const long long ox = idx % ow;
    const long long oy = (idx / ow) % oh;
    const long long oc = (idx / (ow * oh)) % output_channels;
    const long long n = idx / (ow * oh * output_channels);
    const long long channel = oc / multiplier;
    const long long multiple = oc % multiplier;
    float sum = bias == nullptr ? 0.0f : bias[oc];
    for (long long ky = 0; ky < kh; ++ky) {
        const long long iy = oy * sh + ky * dh - ph;
        if (iy < 0 || iy >= ih) continue;
        for (long long kx = 0; kx < kw; ++kx) {
            const long long ix = ox * sw + kx * dw - pw;
            if (ix < 0 || ix >= iw) continue;
            sum += input[((n * channels + channel) * ih + iy) * iw + ix] *
                   weights[((channel * multiplier + multiple) * kh + ky) * kw + kx];
        }
    }
    output[idx] = sum;
}

__global__ void conv3d_kernel(const float *input, const float *weights,
                              const float *bias, float *output,
                              long long batch, long long icount,
                              long long ocount, long long id, long long ih,
                              long long iw, long long kd, long long kh,
                              long long kw, long long sd, long long sh,
                              long long sw, long long pd, long long ph,
                              long long pw, long long dd, long long dh,
                              long long dw) {
    const long long od = conv_out(id, kd, sd, pd, dd);
    const long long oh = conv_out(ih, kh, sh, ph, dh);
    const long long ow = conv_out(iw, kw, sw, pw, dw);
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = batch * ocount * od * oh * ow;
    if (idx >= total) return;
    const long long ox = idx % ow;
    const long long oy = (idx / ow) % oh;
    const long long oz = (idx / (ow * oh)) % od;
    const long long oc = (idx / (ow * oh * od)) % ocount;
    const long long n = idx / (ow * oh * od * ocount);
    float sum = bias == nullptr ? 0.0f : bias[oc];
    for (long long ic = 0; ic < icount; ++ic)
        for (long long kz = 0; kz < kd; ++kz) {
            const long long iz = oz * sd + kz * dd - pd;
            if (iz < 0 || iz >= id) continue;
            for (long long ky = 0; ky < kh; ++ky) {
                const long long iy = oy * sh + ky * dh - ph;
                if (iy < 0 || iy >= ih) continue;
                for (long long kx = 0; kx < kw; ++kx) {
                    const long long ix = ox * sw + kx * dw - pw;
                    if (ix < 0 || ix >= iw) continue;
                    sum += input[(((n * icount + ic) * id + iz) * ih + iy) * iw + ix] *
                           weights[(((oc * icount + ic) * kd + kz) * kh + ky) * kw + kx];
                }
            }
        }
    output[idx] = sum;
}

__global__ void conv_transpose_1d_kernel(
    const float *input, const float *weights, const float *bias, float *output,
    long long batch, long long icount, long long ocount, long long input_length,
    long long kernel, long long stride, long long padding) {
    const long long output_length = (input_length - 1) * stride - 2 * padding + kernel;
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = batch * ocount * output_length;
    if (idx >= total) return;
    const long long ox = idx % output_length;
    const long long oc = (idx / output_length) % ocount;
    const long long n = idx / (output_length * ocount);
    float sum = bias == nullptr ? 0.0f : bias[oc];
    for (long long ic = 0; ic < icount; ++ic)
        for (long long i = 0; i < input_length; ++i)
            for (long long kx = 0; kx < kernel; ++kx)
                if (ox == i * stride + kx - padding)
                    sum += input[(n * icount + ic) * input_length + i] *
                           weights[(ic * ocount + oc) * kernel + kx];
    output[idx] = sum;
}

__global__ void conv_transpose_2d_kernel(
    const float *input, const float *weights, const float *bias, float *output,
    long long batch, long long icount, long long ocount, long long ih,
    long long iw, long long kh, long long kw, long long sh, long long sw,
    long long ph, long long pw) {
    const long long oh = (ih - 1) * sh - 2 * ph + kh;
    const long long ow = (iw - 1) * sw - 2 * pw + kw;
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = batch * ocount * oh * ow;
    if (idx >= total) return;
    const long long ox = idx % ow;
    const long long oy = (idx / ow) % oh;
    const long long oc = (idx / (ow * oh)) % ocount;
    const long long n = idx / (ow * oh * ocount);
    float sum = bias == nullptr ? 0.0f : bias[oc];
    for (long long ic = 0; ic < icount; ++ic)
        for (long long iy = 0; iy < ih; ++iy)
            for (long long ix = 0; ix < iw; ++ix)
                for (long long ky = 0; ky < kh; ++ky)
                    for (long long kx = 0; kx < kw; ++kx)
                        if (oy == iy * sh + ky - ph && ox == ix * sw + kx - pw)
                            sum += input[((n * icount + ic) * ih + iy) * iw + ix] *
                                   weights[((ic * ocount + oc) * kh + ky) * kw + kx];
    output[idx] = sum;
}

__global__ void pool1d_kernel(const float *input, float *output,
                              long long batch, long long channels,
                              long long input_length, long long kernel,
                              long long stride, long long padding, int mode) {
    const long long length = conv_out(input_length, kernel, stride, padding, 1);
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = batch * channels * length;
    if (idx >= total) return;
    const long long o = idx % length;
    const long long c = (idx / length) % channels;
    const long long n = idx / (length * channels);
    float value = mode == kMaximum ? -INFINITY : 0.0f;
    long long count = 0;
    for (long long kx = 0; kx < kernel; ++kx) {
        const long long i = o * stride + kx - padding;
        if (i < 0 || i >= input_length) continue;
        const float sample = input[(n * channels + c) * input_length + i];
        value = mode == kMaximum ? fmaxf(value, sample) : value + sample;
        ++count;
    }
    output[idx] = mode == kAverage ? value / count : value;
}

__global__ void pool2d_kernel(const float *input, float *output,
                              long long batch, long long channels,
                              long long ih, long long iw, long long kh,
                              long long kw, long long sh, long long sw,
                              long long ph, long long pw, int mode) {
    const long long oh = conv_out(ih, kh, sh, ph, 1);
    const long long ow = conv_out(iw, kw, sw, pw, 1);
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = batch * channels * oh * ow;
    if (idx >= total) return;
    const long long ox = idx % ow;
    const long long oy = (idx / ow) % oh;
    const long long c = (idx / (ow * oh)) % channels;
    const long long n = idx / (ow * oh * channels);
    float value = mode == kMaximum ? -INFINITY : 0.0f;
    long long count = 0;
    for (long long ky = 0; ky < kh; ++ky)
        for (long long kx = 0; kx < kw; ++kx) {
            const long long iy = oy * sh + ky - ph;
            const long long ix = ox * sw + kx - pw;
            if (iy < 0 || iy >= ih || ix < 0 || ix >= iw) continue;
            const float sample = input[((n * channels + c) * ih + iy) * iw + ix];
            value = mode == kMaximum ? fmaxf(value, sample) : value + sample;
            ++count;
        }
    output[idx] = mode == kAverage ? value / count : value;
}

__global__ void pool2d_scalar_kernel(const float *input, float *output,
                                     long long batch, long long channels,
                                     long long ih, long long iw, long long kh,
                                     long long kw, long long sh, long long sw,
                                     long long ph, long long pw, int mode) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const long long oh = conv_out(ih, kh, sh, ph, 1);
    const long long ow = conv_out(iw, kw, sw, pw, 1);
    for (long long idx = 0; idx < batch * channels * oh * ow; ++idx) {
        const long long ox = idx % ow, oy = (idx / ow) % oh;
        const long long c = (idx / (ow * oh)) % channels;
        const long long n = idx / (ow * oh * channels);
        float value = mode == kMaximum ? -INFINITY : 0.0f;
        long long count = 0;
        for (long long ky = 0; ky < kh; ++ky)
            for (long long kx = 0; kx < kw; ++kx) {
                const long long iy = oy * sh + ky - ph;
                const long long ix = ox * sw + kx - pw;
                if (iy < 0 || iy >= ih || ix < 0 || ix >= iw) continue;
                const float sample = input[((n * channels + c) * ih + iy) * iw + ix];
                value = mode == kMaximum ? fmaxf(value, sample) : value + sample;
                ++count;
            }
        output[idx] = mode == kAverage ? value / count : value;
    }
}

__global__ void pool2d_backward_kernel(
    const float *input, const float *grad_out, float *grad_in, long long batch,
    long long channels, long long ih, long long iw, long long kh, long long kw,
    long long sh, long long sw, long long ph, long long pw, int mode) {
    const long long oh = conv_out(ih, kh, sh, ph, 1);
    const long long ow = conv_out(iw, kw, sw, pw, 1);
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = batch * channels * ih * iw;
    if (idx >= total) return;
    const long long ix = idx % iw;
    const long long iy = (idx / iw) % ih;
    const long long c = (idx / (iw * ih)) % channels;
    const long long n = idx / (iw * ih * channels);
    float sum = 0.0f;
    for (long long oy = 0; oy < oh; ++oy)
        for (long long ox = 0; ox < ow; ++ox) {
            long long count = 0, best_y = -1, best_x = -1;
            float best = -INFINITY;
            for (long long ky = 0; ky < kh; ++ky)
                for (long long kx = 0; kx < kw; ++kx) {
                    const long long py = oy * sh + ky - ph;
                    const long long px = ox * sw + kx - pw;
                    if (py < 0 || py >= ih || px < 0 || px >= iw) continue;
                    ++count;
                    const float sample = input[((n * channels + c) * ih + py) * iw + px];
                    if (sample > best) {
                        best = sample;
                        best_y = py;
                        best_x = px;
                    }
                }
            const float g = grad_out[((n * channels + c) * oh + oy) * ow + ox];
            if (mode == kMaximum) {
                if (iy == best_y && ix == best_x) sum += g;
            } else {
                bool inside = false;
                for (long long ky = 0; ky < kh; ++ky)
                    for (long long kx = 0; kx < kw; ++kx) {
                        const long long py = oy * sh + ky - ph;
                        const long long px = ox * sw + kx - pw;
                        inside |= (py == iy && px == ix);
                    }
                if (inside) sum += g / count;
            }
        }
    grad_in[idx] = sum;
}

// ---------------------------------------------------------------------------
// Audio kernels
// ---------------------------------------------------------------------------

__global__ void audio_conv1d_direct_kernel(
    const float *input, const float *weights, const float *bias, float *output,
    long long batch, long long input_length, long long input_channels,
    long long output_channels, long long kernel, long long stride,
    long long padding, long long dilation) {
    const long long output_length =
        audio_conv_out(input_length, kernel, stride, padding, dilation);
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = batch * output_length * output_channels;
    if (idx >= total) return;
    const long long oc = idx % output_channels;
    const long long t = (idx / output_channels) % output_length;
    const long long b = idx / (output_channels * output_length);
    float sum = bias == nullptr ? 0.0f : bias[oc];
    for (long long k = 0; k < kernel; ++k) {
        const long long it = t * stride + k * dilation - padding;
        if (it < 0 || it >= input_length) continue;
        for (long long c = 0; c < input_channels; ++c)
            sum += input[(b * input_length + it) * input_channels + c] *
                   weights[(oc * kernel + k) * input_channels + c];
    }
    output[idx] = sum;
}

__global__ void audio_conv1d_direct_scalar_kernel(
    const float *input, const float *weights, const float *bias, float *output,
    long long batch, long long input_length, long long input_channels,
    long long output_channels, long long kernel, long long stride,
    long long padding, long long dilation) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const long long output_length =
        audio_conv_out(input_length, kernel, stride, padding, dilation);
    for (long long idx = 0; idx < batch * output_length * output_channels; ++idx) {
        const long long oc = idx % output_channels;
        const long long t = (idx / output_channels) % output_length;
        const long long b = idx / (output_channels * output_length);
        float sum = bias == nullptr ? 0.0f : bias[oc];
        for (long long k = 0; k < kernel; ++k) {
            const long long it = t * stride + k * dilation - padding;
            if (it < 0 || it >= input_length) continue;
            for (long long c = 0; c < input_channels; ++c)
                sum += input[(b * input_length + it) * input_channels + c] *
                       weights[(oc * kernel + k) * input_channels + c];
        }
        output[idx] = sum;
    }
}

__global__ void audio_depthwise_conv1d_kernel(
    const float *input, const float *weights, const float *bias, float *output,
    long long batch, long long input_length, long long channels,
    long long kernel, long long stride, long long padding, long long dilation,
    int apply_silu) {
    const long long output_length =
        audio_conv_out(input_length, kernel, stride, padding, dilation);
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = batch * output_length * channels;
    if (idx >= total) return;
    const long long c = idx % channels;
    const long long t = (idx / channels) % output_length;
    const long long b = idx / (channels * output_length);
    float sum = bias == nullptr ? 0.0f : bias[c];
    for (long long k = 0; k < kernel; ++k) {
        const long long it = t * stride + k * dilation - padding;
        if (it >= 0 && it < input_length)
            sum += input[(b * input_length + it) * channels + c] *
                   weights[c * kernel + k];
    }
    output[idx] = apply_silu ? silu_value(sum) : sum;
}

__global__ void audio_causal_depthwise_conv1d_kernel(
    const float *input, const float *weights, const float *bias, float *output,
    long long batch, long long input_length, long long channels,
    long long kernel, long long dilation) {
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = batch * input_length * channels;
    if (idx >= total) return;
    const long long c = idx % channels;
    const long long t = (idx / channels) % input_length;
    const long long b = idx / (channels * input_length);
    const long long pad_left = dilation * (kernel - 1);
    float sum = bias == nullptr ? 0.0f : bias[c];
    for (long long k = 0; k < kernel; ++k) {
        const long long it = t + k * dilation - pad_left;
        if (it >= 0 && it < input_length)
            sum += input[(b * input_length + it) * channels + c] *
                   weights[c * kernel + k];
    }
    output[idx] = sum;
}

__device__ float relative_score_device(const float *scaled_query,
                                       const float *key,
                                       const float *relative,
                                       long long head_dim, float key_scale) {
    float sum = 0.0f;
    for (long long f = 0; f < head_dim; ++f) {
        const float r = relative == nullptr ? 0.0f : relative[f];
        sum += scaled_query[f] * (key[f] * key_scale + r);
    }
    return sum;
}

__global__ void audio_relative_attention_kernel(
    const float *q, const float *k, const float *v, const float *relative_k,
    const float *per_dim_scale, const int32_t *lengths, float *out,
    long long batch, long long length, long long heads, long long head_dim,
    long long relative_positions, long long chunk_size, long long left_context,
    long long right_context, float q_scale, float k_scale, float softcap) {
    const long long row = blockIdx.x * blockDim.x + threadIdx.x;
    const long long rows = batch * length * heads;
    if (row >= rows) return;
    const long long head = row % heads;
    const long long pos = (row / heads) % length;
    const long long b = row / (heads * length);
    const long long valid_length = max(0LL, min(length, static_cast<long long>(lengths[b])));
    float *dst = out + row * head_dim;
    for (long long f = 0; f < head_dim; ++f) dst[f] = 0.0f;
    if (pos >= valid_length) return;
    const float log_two = logf(2.0f);
    const float used_q_scale =
        q_scale > 0.0f ? q_scale : 1.0f / (sqrtf(static_cast<float>(head_dim)) * log_two);
    const float used_k_scale = k_scale > 0.0f ? k_scale : 1.0f / log_two;
    float scaled[256];
    for (long long f = 0; f < head_dim; ++f)
        scaled[f] = q[row * head_dim + f] * used_q_scale * softplus_value(per_dim_scale[f]);
    const long long query_in_chunk = pos % chunk_size;
    const long long block_start = (pos / chunk_size) * chunk_size;
    const long long context_start = block_start - (left_context - 1);
    const long long context_length = chunk_size + left_context - 1 + right_context;
    float maximum = -INFINITY;
    float denominator = 0.0f;
    for (long long context = 0; context < context_length; ++context) {
        const long long key_pos = context_start + context;
        if (key_pos < 0 || key_pos >= valid_length) continue;
        const long long kv_row = ((b * length + key_pos) * heads + head);
        const long long rel_idx = context - query_in_chunk;
        const float *relative =
            rel_idx >= 0 && rel_idx < relative_positions
                ? relative_k + (rel_idx * heads + head) * head_dim
                : nullptr;
        float score = relative_score_device(scaled, k + kv_row * head_dim, relative,
                                            head_dim, used_k_scale);
        if (softcap > 0.0f) score = softcap * tanhf(score / softcap);
        const float *value = v + kv_row * head_dim;
        if (score > maximum) {
            const float scale = isfinite(maximum) ? expf(maximum - score) : 0.0f;
            denominator = denominator * scale + 1.0f;
            for (long long f = 0; f < head_dim; ++f) dst[f] = dst[f] * scale + value[f];
            maximum = score;
        } else {
            const float p = expf(score - maximum);
            denominator += p;
            for (long long f = 0; f < head_dim; ++f) dst[f] += p * value[f];
        }
    }
    if (denominator > 0.0f) {
        const float inv = 1.0f / denominator;
        for (long long f = 0; f < head_dim; ++f) dst[f] *= inv;
    }
}

__global__ void audio_relative_attention_scalar_kernel(
    const float *q, const float *k, const float *v, const float *relative_k,
    const float *per_dim_scale, const int32_t *lengths, float *out,
    long long batch, long long length, long long heads, long long head_dim,
    long long relative_positions, long long chunk_size, long long left_context,
    long long right_context, float q_scale, float k_scale, float softcap) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const float log_two = logf(2.0f);
    const float used_q_scale =
        q_scale > 0.0f ? q_scale : 1.0f / (sqrtf(static_cast<float>(head_dim)) * log_two);
    const float used_k_scale = k_scale > 0.0f ? k_scale : 1.0f / log_two;
    for (long long row = 0; row < batch * length * heads; ++row) {
        const long long head = row % heads;
        const long long pos = (row / heads) % length;
        const long long b = row / (heads * length);
        const long long valid_length = max(0LL, min(length, static_cast<long long>(lengths[b])));
        float *dst = out + row * head_dim;
        for (long long f = 0; f < head_dim; ++f) dst[f] = 0.0f;
        if (pos >= valid_length) continue;
        float scaled[256];
        for (long long f = 0; f < head_dim; ++f)
            scaled[f] = q[row * head_dim + f] * used_q_scale *
                        softplus_value(per_dim_scale[f]);
        const long long query_in_chunk = pos % chunk_size;
        const long long block_start = (pos / chunk_size) * chunk_size;
        const long long context_start = block_start - (left_context - 1);
        const long long context_length = chunk_size + left_context - 1 + right_context;
        float maximum = -INFINITY;
        float denominator = 0.0f;
        for (long long context = 0; context < context_length; ++context) {
            const long long key_pos = context_start + context;
            if (key_pos < 0 || key_pos >= valid_length) continue;
            const long long kv_row = ((b * length + key_pos) * heads + head);
            const long long rel_idx = context - query_in_chunk;
            const float *relative =
                rel_idx >= 0 && rel_idx < relative_positions
                    ? relative_k + (rel_idx * heads + head) * head_dim
                    : nullptr;
            float score = relative_score_device(scaled, k + kv_row * head_dim,
                                                relative, head_dim,
                                                used_k_scale);
            if (softcap > 0.0f) score = softcap * tanhf(score / softcap);
            const float *value = v + kv_row * head_dim;
            if (score > maximum) {
                const float scale = isfinite(maximum) ? expf(maximum - score) : 0.0f;
                denominator = denominator * scale + 1.0f;
                for (long long f = 0; f < head_dim; ++f)
                    dst[f] = dst[f] * scale + value[f];
                maximum = score;
            } else {
                const float p = expf(score - maximum);
                denominator += p;
                for (long long f = 0; f < head_dim; ++f) dst[f] += p * value[f];
            }
        }
        if (denominator > 0.0f) {
            const float inv = 1.0f / denominator;
            for (long long f = 0; f < head_dim; ++f) dst[f] *= inv;
        }
    }
}

// ---------------------------------------------------------------------------
// Host references
// ---------------------------------------------------------------------------

std::vector<double> im2col_2d_ref(const std::vector<float> &image,
                                  long long batch, long long channels,
                                  long long ih, long long iw, long long kh,
                                  long long kw, long long sh, long long sw,
                                  long long ph, long long pw, long long dh,
                                  long long dw) {
    const long long oh = conv_out(ih, kh, sh, ph, dh);
    const long long ow = conv_out(iw, kw, sw, pw, dw);
    const long long patch = channels * kh * kw;
    std::vector<double> out(batch * oh * ow * patch);
    for (long long n = 0; n < batch; ++n)
        for (long long oy = 0; oy < oh; ++oy)
            for (long long ox = 0; ox < ow; ++ox)
                for (long long c = 0; c < channels; ++c)
                    for (long long ky = 0; ky < kh; ++ky)
                        for (long long kx = 0; kx < kw; ++kx) {
                            const long long iy = oy * sh + ky * dh - ph;
                            const long long ix = ox * sw + kx * dw - pw;
                            const long long p = (c * kh + ky) * kw + kx;
                            out[((n * oh + oy) * ow + ox) * patch + p] =
                                (iy >= 0 && iy < ih && ix >= 0 && ix < iw)
                                    ? image[((n * channels + c) * ih + iy) * iw + ix]
                                    : 0.0;
                        }
    return out;
}

std::vector<double> im2col_3d_ref(const std::vector<float> &volume,
                                  long long batch, long long channels,
                                  long long id, long long ih, long long iw,
                                  long long kd, long long kh, long long kw,
                                  long long sd, long long sh, long long sw,
                                  long long pd, long long ph, long long pw,
                                  long long dd, long long dh, long long dw) {
    const long long od = conv_out(id, kd, sd, pd, dd);
    const long long oh = conv_out(ih, kh, sh, ph, dh);
    const long long ow = conv_out(iw, kw, sw, pw, dw);
    const long long patch = channels * kd * kh * kw;
    std::vector<double> out(batch * od * oh * ow * patch);
    for (long long n = 0; n < batch; ++n)
        for (long long oz = 0; oz < od; ++oz)
            for (long long oy = 0; oy < oh; ++oy)
                for (long long ox = 0; ox < ow; ++ox) {
                    long long p = 0;
                    for (long long c = 0; c < channels; ++c)
                        for (long long kz = 0; kz < kd; ++kz)
                            for (long long ky = 0; ky < kh; ++ky)
                                for (long long kx = 0; kx < kw; ++kx, ++p) {
                                    const long long iz = oz * sd + kz * dd - pd;
                                    const long long iy = oy * sh + ky * dh - ph;
                                    const long long ix = ox * sw + kx * dw - pw;
                                    out[(((n * od + oz) * oh + oy) * ow + ox) * patch + p] =
                                        (iz >= 0 && iz < id && iy >= 0 && iy < ih && ix >= 0 && ix < iw)
                                            ? volume[(((n * channels + c) * id + iz) * ih + iy) * iw + ix]
                                            : 0.0;
                                }
                }
    return out;
}

std::vector<double> col2im_1d_ref(const std::vector<float> &columns,
                                  long long time_in, long long channels,
                                  long long kernel, long long stride,
                                  long long padding) {
    const long long time_out = (time_in - 1) * stride + kernel - 2 * padding;
    std::vector<double> out(channels * time_out);
    for (long long c = 0; c < channels; ++c)
        for (long long output_index = 0; output_index < time_out; ++output_index) {
            const long long absolute = output_index + padding;
            float sum = 0.0f;
            for (long long input_index = 0; input_index < time_in; ++input_index) {
                const long long tap = absolute - input_index * stride;
                if (tap >= 0 && tap < kernel)
                    sum += columns[(input_index * channels + c) * kernel + tap];
            }
            out[c * time_out + output_index] = sum;
        }
    return out;
}

std::vector<double> col2im_2d_ref(const std::vector<float> &columns,
                                  long long batch, long long channels,
                                  long long ih, long long iw, long long kh,
                                  long long kw, long long sh, long long sw,
                                  long long ph, long long pw, long long dh,
                                  long long dw) {
    const long long oh = conv_out(ih, kh, sh, ph, dh);
    const long long ow = conv_out(iw, kw, sw, pw, dw);
    const long long patch = channels * kh * kw;
    std::vector<double> out(batch * channels * ih * iw, 0.0);
    for (long long n = 0; n < batch; ++n)
        for (long long oy = 0; oy < oh; ++oy)
            for (long long ox = 0; ox < ow; ++ox)
                for (long long c = 0; c < channels; ++c)
                    for (long long ky = 0; ky < kh; ++ky) {
                        const long long iy = oy * sh + ky * dh - ph;
                        if (iy < 0 || iy >= ih) continue;
                        for (long long kx = 0; kx < kw; ++kx) {
                            const long long ix = ox * sw + kx * dw - pw;
                            if (ix < 0 || ix >= iw) continue;
                            out[((n * channels + c) * ih + iy) * iw + ix] +=
                                columns[((n * oh + oy) * ow + ox) * patch +
                                        (c * kh + ky) * kw + kx];
                        }
                    }
    return out;
}

template <typename Launch>
std::vector<float> run_float_kernel(size_t n, Launch launch) {
    float *out = qc::dzero<float>(n);
    launch(out);
    QC_SYNC();
    auto h = qc::d2h(out, n);
    qc::dfree(out);
    return h;
}

std::vector<double> conv2d_ref(const std::vector<float> &input,
                               const std::vector<float> &weights,
                               const std::vector<float> &bias, long long batch,
                               long long icount, long long ocount,
                               long long ih, long long iw, long long kh,
                               long long kw, long long sh, long long sw,
                               long long ph, long long pw, long long dh,
                               long long dw) {
    const long long oh = conv_out(ih, kh, sh, ph, dh);
    const long long ow = conv_out(iw, kw, sw, pw, dw);
    std::vector<double> out(batch * ocount * oh * ow);
    for (long long n = 0; n < batch; ++n)
        for (long long oc = 0; oc < ocount; ++oc)
            for (long long oy = 0; oy < oh; ++oy)
                for (long long ox = 0; ox < ow; ++ox) {
                    float sum = bias[oc];
                    for (long long ic = 0; ic < icount; ++ic)
                        for (long long ky = 0; ky < kh; ++ky) {
                            const long long iy = oy * sh + ky * dh - ph;
                            if (iy < 0 || iy >= ih) continue;
                            for (long long kx = 0; kx < kw; ++kx) {
                                const long long ix = ox * sw + kx * dw - pw;
                                if (ix < 0 || ix >= iw) continue;
                                sum += input[((n * icount + ic) * ih + iy) * iw + ix] *
                                       weights[((oc * icount + ic) * kh + ky) * kw + kx];
                            }
                        }
                    out[((n * ocount + oc) * oh + oy) * ow + ox] = sum;
                }
    return out;
}

std::vector<double> depthwise_ref(const std::vector<float> &input,
                                  const std::vector<float> &weights,
                                  const std::vector<float> &bias,
                                  long long batch, long long channels,
                                  long long multiplier, long long ih,
                                  long long iw, long long kh, long long kw,
                                  long long sh, long long sw, long long ph,
                                  long long pw, long long dh, long long dw) {
    const long long oh = conv_out(ih, kh, sh, ph, dh);
    const long long ow = conv_out(iw, kw, sw, pw, dw);
    const long long ocount = channels * multiplier;
    std::vector<double> out(batch * ocount * oh * ow);
    for (long long n = 0; n < batch; ++n)
        for (long long oc = 0; oc < ocount; ++oc)
            for (long long oy = 0; oy < oh; ++oy)
                for (long long ox = 0; ox < ow; ++ox) {
                    const long long c = oc / multiplier;
                    const long long m = oc % multiplier;
                    float sum = bias[oc];
                    for (long long ky = 0; ky < kh; ++ky) {
                        const long long iy = oy * sh + ky * dh - ph;
                        if (iy < 0 || iy >= ih) continue;
                        for (long long kx = 0; kx < kw; ++kx) {
                            const long long ix = ox * sw + kx * dw - pw;
                            if (ix < 0 || ix >= iw) continue;
                            sum += input[((n * channels + c) * ih + iy) * iw + ix] *
                                   weights[((c * multiplier + m) * kh + ky) * kw + kx];
                        }
                    }
                    out[((n * ocount + oc) * oh + oy) * ow + ox] = sum;
                }
    return out;
}

std::vector<double> conv3d_ref(const std::vector<float> &input,
                               const std::vector<float> &weights,
                               const std::vector<float> &bias, long long batch,
                               long long icount, long long ocount,
                               long long id, long long ih, long long iw,
                               long long kd, long long kh, long long kw,
                               long long sd, long long sh, long long sw,
                               long long pd, long long ph, long long pw,
                               long long dd, long long dh, long long dw) {
    const long long od = conv_out(id, kd, sd, pd, dd);
    const long long oh = conv_out(ih, kh, sh, ph, dh);
    const long long ow = conv_out(iw, kw, sw, pw, dw);
    std::vector<double> out(batch * ocount * od * oh * ow);
    for (long long n = 0; n < batch; ++n)
        for (long long oc = 0; oc < ocount; ++oc)
            for (long long oz = 0; oz < od; ++oz)
                for (long long oy = 0; oy < oh; ++oy)
                    for (long long ox = 0; ox < ow; ++ox) {
                        float sum = bias[oc];
                        for (long long ic = 0; ic < icount; ++ic)
                            for (long long kz = 0; kz < kd; ++kz) {
                                const long long iz = oz * sd + kz * dd - pd;
                                if (iz < 0 || iz >= id) continue;
                                for (long long ky = 0; ky < kh; ++ky) {
                                    const long long iy = oy * sh + ky * dh - ph;
                                    if (iy < 0 || iy >= ih) continue;
                                    for (long long kx = 0; kx < kw; ++kx) {
                                        const long long ix = ox * sw + kx * dw - pw;
                                        if (ix < 0 || ix >= iw) continue;
                                        sum += input[(((n * icount + ic) * id + iz) * ih + iy) * iw + ix] *
                                               weights[(((oc * icount + ic) * kd + kz) * kh + ky) * kw + kx];
                                    }
                                }
                            }
                        out[(((n * ocount + oc) * od + oz) * oh + oy) * ow + ox] = sum;
                    }
    return out;
}

std::vector<double> conv_transpose_1d_ref(
    const std::vector<float> &input, const std::vector<float> &weights,
    const std::vector<float> &bias, long long batch, long long icount,
    long long ocount, long long input_length, long long kernel,
    long long stride, long long padding) {
    const long long output_length = (input_length - 1) * stride - 2 * padding + kernel;
    std::vector<double> out(batch * ocount * output_length);
    for (long long n = 0; n < batch; ++n)
        for (long long oc = 0; oc < ocount; ++oc)
            for (long long x = 0; x < output_length; ++x)
                out[(n * ocount + oc) * output_length + x] = bias[oc];
    for (long long n = 0; n < batch; ++n)
        for (long long ic = 0; ic < icount; ++ic)
            for (long long i = 0; i < input_length; ++i)
                for (long long kx = 0; kx < kernel; ++kx) {
                    const long long ox = i * stride + kx - padding;
                    if (ox < 0 || ox >= output_length) continue;
                    for (long long oc = 0; oc < ocount; ++oc)
                        out[(n * ocount + oc) * output_length + ox] +=
                            input[(n * icount + ic) * input_length + i] *
                            weights[(ic * ocount + oc) * kernel + kx];
                }
    return out;
}

std::vector<double> conv_transpose_2d_ref(
    const std::vector<float> &input, const std::vector<float> &weights,
    const std::vector<float> &bias, long long batch, long long icount,
    long long ocount, long long ih, long long iw, long long kh, long long kw,
    long long sh, long long sw, long long ph, long long pw) {
    const long long oh = (ih - 1) * sh - 2 * ph + kh;
    const long long ow = (iw - 1) * sw - 2 * pw + kw;
    std::vector<double> out(batch * ocount * oh * ow);
    for (long long n = 0; n < batch; ++n)
        for (long long oc = 0; oc < ocount; ++oc)
            for (long long i = 0; i < oh * ow; ++i)
                out[(n * ocount + oc) * oh * ow + i] = bias[oc];
    for (long long n = 0; n < batch; ++n)
        for (long long ic = 0; ic < icount; ++ic)
            for (long long iy = 0; iy < ih; ++iy)
                for (long long ix = 0; ix < iw; ++ix)
                    for (long long ky = 0; ky < kh; ++ky)
                        for (long long kx = 0; kx < kw; ++kx) {
                            const long long oy = iy * sh + ky - ph;
                            const long long ox = ix * sw + kx - pw;
                            if (oy < 0 || oy >= oh || ox < 0 || ox >= ow) continue;
                            for (long long oc = 0; oc < ocount; ++oc)
                                out[((n * ocount + oc) * oh + oy) * ow + ox] +=
                                    input[((n * icount + ic) * ih + iy) * iw + ix] *
                                    weights[((ic * ocount + oc) * kh + ky) * kw + kx];
                        }
    return out;
}

std::vector<double> pool1d_ref(const std::vector<float> &input, long long batch,
                               long long channels, long long input_length,
                               long long kernel, long long stride,
                               long long padding, int mode) {
    const long long length = conv_out(input_length, kernel, stride, padding, 1);
    std::vector<double> out(batch * channels * length);
    for (long long n = 0; n < batch; ++n)
        for (long long c = 0; c < channels; ++c)
            for (long long o = 0; o < length; ++o) {
                float value = mode == kMaximum ? -std::numeric_limits<float>::infinity() : 0.0f;
                long long count = 0;
                for (long long kx = 0; kx < kernel; ++kx) {
                    const long long i = o * stride + kx - padding;
                    if (i < 0 || i >= input_length) continue;
                    const float sample = input[(n * channels + c) * input_length + i];
                    value = mode == kMaximum ? std::max(value, sample) : value + sample;
                    ++count;
                }
                out[(n * channels + c) * length + o] =
                    mode == kAverage ? value / count : value;
            }
    return out;
}

std::vector<double> pool2d_ref(const std::vector<float> &input, long long batch,
                               long long channels, long long ih, long long iw,
                               long long kh, long long kw, long long sh,
                               long long sw, long long ph, long long pw,
                               int mode) {
    const long long oh = conv_out(ih, kh, sh, ph, 1);
    const long long ow = conv_out(iw, kw, sw, pw, 1);
    std::vector<double> out(batch * channels * oh * ow);
    for (long long n = 0; n < batch; ++n)
        for (long long c = 0; c < channels; ++c)
            for (long long oy = 0; oy < oh; ++oy)
                for (long long ox = 0; ox < ow; ++ox) {
                    float value = mode == kMaximum ? -std::numeric_limits<float>::infinity() : 0.0f;
                    long long count = 0;
                    for (long long ky = 0; ky < kh; ++ky)
                        for (long long kx = 0; kx < kw; ++kx) {
                            const long long iy = oy * sh + ky - ph;
                            const long long ix = ox * sw + kx - pw;
                            if (iy < 0 || iy >= ih || ix < 0 || ix >= iw) continue;
                            const float sample = input[((n * channels + c) * ih + iy) * iw + ix];
                            value = mode == kMaximum ? std::max(value, sample) : value + sample;
                            ++count;
                        }
                    out[((n * channels + c) * oh + oy) * ow + ox] =
                        mode == kAverage ? value / count : value;
                }
    return out;
}

std::vector<double> pool2d_backward_ref(
    const std::vector<float> &input, const std::vector<float> &grad_out,
    long long batch, long long channels, long long ih, long long iw,
    long long kh, long long kw, long long sh, long long sw, long long ph,
    long long pw, int mode) {
    const long long oh = conv_out(ih, kh, sh, ph, 1);
    const long long ow = conv_out(iw, kw, sw, pw, 1);
    std::vector<double> out(batch * channels * ih * iw, 0.0);
    for (long long n = 0; n < batch; ++n)
        for (long long c = 0; c < channels; ++c)
            for (long long oy = 0; oy < oh; ++oy)
                for (long long ox = 0; ox < ow; ++ox) {
                    long long count = 0, best_y = -1, best_x = -1;
                    float best = -std::numeric_limits<float>::infinity();
                    for (long long ky = 0; ky < kh; ++ky)
                        for (long long kx = 0; kx < kw; ++kx) {
                            const long long iy = oy * sh + ky - ph;
                            const long long ix = ox * sw + kx - pw;
                            if (iy < 0 || iy >= ih || ix < 0 || ix >= iw) continue;
                            ++count;
                            const float sample = input[((n * channels + c) * ih + iy) * iw + ix];
                            if (sample > best) {
                                best = sample;
                                best_y = iy;
                                best_x = ix;
                            }
                        }
                    const float g = grad_out[((n * channels + c) * oh + oy) * ow + ox];
                    if (mode == kMaximum) {
                        out[((n * channels + c) * ih + best_y) * iw + best_x] += g;
                    } else {
                        for (long long ky = 0; ky < kh; ++ky)
                            for (long long kx = 0; kx < kw; ++kx) {
                                const long long iy = oy * sh + ky - ph;
                                const long long ix = ox * sw + kx - pw;
                                if (iy >= 0 && iy < ih && ix >= 0 && ix < iw)
                                    out[((n * channels + c) * ih + iy) * iw + ix] += g / count;
                            }
                    }
                }
    return out;
}

std::vector<double> audio_conv1d_ref(
    const std::vector<float> &input, const std::vector<float> &weights,
    const std::vector<float> &bias, long long batch, long long length,
    long long channels, long long out_channels, long long kernel,
    long long stride, long long padding, long long dilation) {
    const long long out_len = audio_conv_out(length, kernel, stride, padding, dilation);
    std::vector<double> out(batch * out_len * out_channels);
    for (long long b = 0; b < batch; ++b)
        for (long long t = 0; t < out_len; ++t)
            for (long long oc = 0; oc < out_channels; ++oc) {
                float sum = bias[oc];
                for (long long k = 0; k < kernel; ++k) {
                    const long long it = t * stride + k * dilation - padding;
                    if (it < 0 || it >= length) continue;
                    for (long long c = 0; c < channels; ++c)
                        sum += input[(b * length + it) * channels + c] *
                               weights[(oc * kernel + k) * channels + c];
                }
                out[(b * out_len + t) * out_channels + oc] = sum;
            }
    return out;
}

std::vector<double> audio_depthwise_ref(
    const std::vector<float> &input, const std::vector<float> &weights,
    const std::vector<float> &bias, long long batch, long long length,
    long long channels, long long kernel, long long stride, long long padding,
    long long dilation, bool apply_silu) {
    const long long out_len = audio_conv_out(length, kernel, stride, padding, dilation);
    std::vector<double> out(batch * out_len * channels);
    for (long long b = 0; b < batch; ++b)
        for (long long t = 0; t < out_len; ++t)
            for (long long c = 0; c < channels; ++c) {
                float sum = bias[c];
                for (long long k = 0; k < kernel; ++k) {
                    const long long it = t * stride + k * dilation - padding;
                    if (it >= 0 && it < length)
                        sum += input[(b * length + it) * channels + c] *
                               weights[c * kernel + k];
                }
                out[(b * out_len + t) * channels + c] = apply_silu ? silu_value(sum) : sum;
            }
    return out;
}

std::vector<double> audio_causal_depthwise_ref(
    const std::vector<float> &input, const std::vector<float> &weights,
    const std::vector<float> &bias, long long batch, long long length,
    long long channels, long long kernel, long long dilation) {
    std::vector<double> out(batch * length * channels);
    const long long pad_left = dilation * (kernel - 1);
    for (long long b = 0; b < batch; ++b)
        for (long long t = 0; t < length; ++t)
            for (long long c = 0; c < channels; ++c) {
                float sum = bias[c];
                for (long long k = 0; k < kernel; ++k) {
                    const long long it = t + k * dilation - pad_left;
                    if (it >= 0 && it < length)
                        sum += input[(b * length + it) * channels + c] *
                               weights[c * kernel + k];
                }
                out[(b * length + t) * channels + c] = sum;
            }
    return out;
}

std::vector<double> audio_relative_ref(
    const std::vector<float> &q, const std::vector<float> &k,
    const std::vector<float> &v, const std::vector<float> &relative_k,
    const std::vector<float> &per_dim_scale, const std::vector<int32_t> &lengths,
    long long batch, long long length, long long heads, long long head_dim,
    long long relative_positions, long long chunk_size, long long left_context,
    long long right_context, float q_scale, float k_scale, float softcap) {
    const float log_two = std::log(2.0f);
    const float used_q_scale =
        q_scale > 0.0f ? q_scale : 1.0f / (std::sqrt(static_cast<float>(head_dim)) * log_two);
    const float used_k_scale = k_scale > 0.0f ? k_scale : 1.0f / log_two;
    std::vector<float> qdim(head_dim);
    for (long long f = 0; f < head_dim; ++f)
        qdim[f] = used_q_scale * softplus_value(per_dim_scale[f]);
    std::vector<double> out(batch * length * heads * head_dim, 0.0);
    std::vector<float> scaled(head_dim);
    for (long long b = 0; b < batch; ++b) {
        const long long valid = std::clamp<long long>(lengths[b], 0, length);
        for (long long pos = 0; pos < length; ++pos) {
            for (long long head = 0; head < heads; ++head) {
                const long long row = (b * length + pos) * heads + head;
                if (pos >= valid) continue;
                for (long long f = 0; f < head_dim; ++f)
                    scaled[f] = q[row * head_dim + f] * qdim[f];
                const long long query_in_chunk = pos % chunk_size;
                const long long block_start = (pos / chunk_size) * chunk_size;
                const long long context_start = block_start - (left_context - 1);
                const long long context_length = chunk_size + left_context - 1 + right_context;
                float maximum = -std::numeric_limits<float>::infinity();
                float denominator = 0.0f;
                for (long long context = 0; context < context_length; ++context) {
                    const long long key_pos = context_start + context;
                    if (key_pos < 0 || key_pos >= valid) continue;
                    const long long kv_row = (b * length + key_pos) * heads + head;
                    const long long rel_idx = context - query_in_chunk;
                    float score = 0.0f;
                    for (long long f = 0; f < head_dim; ++f) {
                        const float rel =
                            rel_idx >= 0 && rel_idx < relative_positions
                                ? relative_k[(rel_idx * heads + head) * head_dim + f]
                                : 0.0f;
                        score += scaled[f] * (k[kv_row * head_dim + f] * used_k_scale + rel);
                    }
                    if (softcap > 0.0f) score = softcap * std::tanh(score / softcap);
                    if (score > maximum) {
                        const float scale = std::isfinite(maximum) ? std::exp(maximum - score) : 0.0f;
                        denominator = denominator * scale + 1.0f;
                        for (long long f = 0; f < head_dim; ++f)
                            out[row * head_dim + f] =
                                out[row * head_dim + f] * scale + v[kv_row * head_dim + f];
                        maximum = score;
                    } else {
                        const float p = std::exp(score - maximum);
                        denominator += p;
                        for (long long f = 0; f < head_dim; ++f)
                            out[row * head_dim + f] += p * v[kv_row * head_dim + f];
                    }
                }
                if (denominator > 0.0f) {
                    for (long long f = 0; f < head_dim; ++f)
                        out[row * head_dim + f] *= 1.0f / denominator;
                }
            }
        }
    }
    return out;
}

bool run_correctness() {
    bool ok = true;
    int checks = 0;
    qc::Rng rng(1212);

    const long long batch = 2, channels = 2, ih = 5, iw = 6;
    const long long kh = 3, kw = 2, sh = 2, sw = 1, ph = 1, pw = 0, dh = 1, dw = 2;
    const long long oh = conv_out(ih, kh, sh, ph, dh);
    const long long ow = conv_out(iw, kw, sw, pw, dw);
    const long long patch = channels * kh * kw;
    std::vector<float> image = rng.uniforms(batch * channels * ih * iw, -1.0f, 1.0f);
    float *dimage = qc::dnew(image);
    float *dcols = qc::dzero<float>(batch * oh * ow * patch);
    im2col_2d_kernel<<<qc::grid_for(batch * oh * ow * patch, kThreads), kThreads>>>(
        dimage, dcols, batch, channels, ih, iw, kh, kw, sh, sw, ph, pw, dh, dw);
    QC_SYNC();
    ok &= qc::compare(qc::d2h(dcols, batch * oh * ow * patch),
                      im2col_2d_ref(image, batch, channels, ih, iw, kh, kw, sh, sw, ph, pw, dh, dw),
                      qc::Tol::fp32()).report("im2col_2d");
    ++checks;

    float *drecon = qc::dzero<float>(image.size());
    col2im_2d_kernel<<<qc::grid_for(image.size(), kThreads), kThreads>>>(
        dcols, drecon, batch, channels, ih, iw, kh, kw, sh, sw, ph, pw, dh, dw);
    QC_SYNC();
    ok &= qc::compare(qc::d2h(drecon, image.size()),
                      col2im_2d_ref(qc::d2h(dcols, batch * oh * ow * patch),
                                    batch, channels, ih, iw, kh, kw, sh, sw, ph, pw, dh, dw),
                      qc::Tol::fp32()).report("col2im_2d");
    ++checks;

    const long long id = 4, kd = 2;
    const long long od = conv_out(id, kd, 1, 1, 1);
    std::vector<float> volume = rng.uniforms(batch * channels * id * ih * iw, -1.0f, 1.0f);
    float *dvolume = qc::dnew(volume);
    float *dcols3 = qc::dzero<float>(batch * od * oh * ow * channels * kd * kh * kw);
    im2col_3d_kernel<<<qc::grid_for(batch * od * oh * ow * channels * kd * kh * kw, kThreads), kThreads>>>(
        dvolume, dcols3, batch, channels, id, ih, iw, kd, kh, kw, 1, sh, sw, 1, ph, pw, 1, dh, dw);
    QC_SYNC();
    ok &= qc::compare(qc::d2h(dcols3, batch * od * oh * ow * channels * kd * kh * kw),
                      im2col_3d_ref(volume, batch, channels, id, ih, iw, kd, kh, kw,
                                    1, sh, sw, 1, ph, pw, 1, dh, dw),
                      qc::Tol::fp32()).report("im2col_3d");
    ++checks;

    const long long tin = 5, ach = 3, ak = 3, astride = 2, apad = 1;
    const long long tout = (tin - 1) * astride + ak - 2 * apad;
    std::vector<float> col1 = rng.uniforms(tin * ach * ak, -1.0f, 1.0f);
    float *dcol1 = qc::dnew(col1);
    float *dsig = qc::dzero<float>(ach * tout);
    col2im_1d_kernel<<<qc::grid_for(ach * tout, kThreads), kThreads>>>(
        dcol1, dsig, tin, ach, ak, astride, apad);
    QC_SYNC();
    ok &= qc::compare(qc::d2h(dsig, ach * tout),
                      col2im_1d_ref(col1, tin, ach, ak, astride, apad),
                      qc::Tol::fp32()).report("col2im_1d");
    ++checks;

    const long long icount = 3, ocount = 4;
    std::vector<float> conv_in = rng.uniforms(batch * icount * ih * iw, -1.0f, 1.0f);
    std::vector<float> conv_w = rng.uniforms(ocount * icount * kh * kw, -0.5f, 0.5f);
    std::vector<float> conv_b = rng.uniforms(ocount, -0.1f, 0.1f);
    float *dcin = qc::dnew(conv_in);
    float *dcw = qc::dnew(conv_w);
    float *dcb = qc::dnew(conv_b);
    float *dconv = qc::dzero<float>(batch * ocount * oh * ow);
    conv2d_kernel<<<qc::grid_for(batch * ocount * oh * ow, kThreads), kThreads>>>(
        dcin, dcw, dcb, dconv, batch, icount, ocount, ih, iw, kh, kw,
        sh, sw, ph, pw, dh, dw);
    QC_SYNC();
    ok &= qc::compare(qc::d2h(dconv, batch * ocount * oh * ow),
                      conv2d_ref(conv_in, conv_w, conv_b, batch, icount, ocount,
                                 ih, iw, kh, kw, sh, sw, ph, pw, dh, dw),
                      qc::Tol::fp32()).report("conv2d");
    ++checks;

    const long long mult = 2;
    std::vector<float> dw_w = rng.uniforms(channels * mult * kh * kw, -0.5f, 0.5f);
    std::vector<float> dw_b = rng.uniforms(channels * mult, -0.1f, 0.1f);
    float *ddww = qc::dnew(dw_w);
    float *ddwb = qc::dnew(dw_b);
    float *ddwo = qc::dzero<float>(batch * channels * mult * oh * ow);
    depthwise_conv2d_kernel<<<qc::grid_for(batch * channels * mult * oh * ow, kThreads), kThreads>>>(
        dimage, ddww, ddwb, ddwo, batch, channels, mult, ih, iw, kh, kw,
        sh, sw, ph, pw, dh, dw);
    QC_SYNC();
    ok &= qc::compare(qc::d2h(ddwo, batch * channels * mult * oh * ow),
                      depthwise_ref(image, dw_w, dw_b, batch, channels, mult,
                                    ih, iw, kh, kw, sh, sw, ph, pw, dh, dw),
                      qc::Tol::fp32()).report("depthwise_conv2d");
    ++checks;

    const long long cd = 4, cod = conv_out(cd, kd, 1, 1, 1);
    std::vector<float> c3_in = rng.uniforms(batch * icount * cd * ih * iw, -1.0f, 1.0f);
    std::vector<float> c3_w = rng.uniforms(ocount * icount * kd * kh * kw, -0.25f, 0.25f);
    float *dc3in = qc::dnew(c3_in);
    float *dc3w = qc::dnew(c3_w);
    float *dc3o = qc::dzero<float>(batch * ocount * cod * oh * ow);
    conv3d_kernel<<<qc::grid_for(batch * ocount * cod * oh * ow, kThreads), kThreads>>>(
        dc3in, dc3w, dcb, dc3o, batch, icount, ocount, cd, ih, iw, kd, kh, kw,
        1, sh, sw, 1, ph, pw, 1, dh, dw);
    QC_SYNC();
    auto c3_ref = conv3d_ref(c3_in, c3_w, conv_b, batch, icount, ocount, cd,
                             ih, iw, kd, kh, kw, 1, sh, sw, 1, ph, pw, 1,
                             dh, dw);
    ok &= qc::compare(qc::d2h(dc3o, c3_ref.size()), c3_ref,
                      qc::Tol::fp32()).report("conv3d");
    ++checks;

    const long long ti = 4, tk = 3, ts = 2, tp = 1;
    const long long to = (ti - 1) * ts - 2 * tp + tk;
    std::vector<float> tinv = rng.uniforms(batch * icount * ti, -1.0f, 1.0f);
    std::vector<float> tw1 = rng.uniforms(icount * ocount * tk, -0.5f, 0.5f);
    float *dtinv = qc::dnew(tinv);
    float *dtw1 = qc::dnew(tw1);
    float *dt1 = qc::dzero<float>(batch * ocount * to);
    conv_transpose_1d_kernel<<<qc::grid_for(batch * ocount * to, kThreads), kThreads>>>(
        dtinv, dtw1, dcb, dt1, batch, icount, ocount, ti, tk, ts, tp);
    QC_SYNC();
    auto t1_ref = conv_transpose_1d_ref(tinv, tw1, conv_b, batch, icount,
                                        ocount, ti, tk, ts, tp);
    ok &= qc::compare(qc::d2h(dt1, t1_ref.size()), t1_ref,
                      qc::Tol::fp32()).report("conv_transpose_1d");
    ++checks;

    const long long toh = (ih - 1) * 1 - 2 * 1 + kh;
    const long long tow = (iw - 1) * 1 - 2 * 0 + kw;
    std::vector<float> tw2 = rng.uniforms(icount * ocount * kh * kw, -0.5f, 0.5f);
    float *dtw2 = qc::dnew(tw2);
    float *dt2 = qc::dzero<float>(batch * ocount * toh * tow);
    conv_transpose_2d_kernel<<<qc::grid_for(batch * ocount * toh * tow, kThreads), kThreads>>>(
        dcin, dtw2, dcb, dt2, batch, icount, ocount, ih, iw, kh, kw, 1, 1, 1, 0);
    QC_SYNC();
    auto t2_ref = conv_transpose_2d_ref(conv_in, tw2, conv_b, batch, icount,
                                        ocount, ih, iw, kh, kw, 1, 1, 1, 0);
    ok &= qc::compare(qc::d2h(dt2, t2_ref.size()), t2_ref,
                      qc::Tol::fp32()).report("conv_transpose_2d");
    ++checks;

    const long long pk = 3, ps = 2, pp = 1;
    const long long pl = conv_out(tout, pk, ps, pp, 1);
    float *dpool1 = qc::dzero<float>(batch * ach * pl);
    std::vector<float> p1in = rng.uniforms(batch * ach * tout, -1.0f, 1.0f);
    float *dp1in = qc::dnew(p1in);
    pool1d_kernel<<<qc::grid_for(batch * ach * pl, kThreads), kThreads>>>(
        dp1in, dpool1, batch, ach, tout, pk, ps, pp, kAverage);
    QC_SYNC();
    auto p1avg = qc::d2h(dpool1, batch * ach * pl);
    pool1d_kernel<<<qc::grid_for(batch * ach * pl, kThreads), kThreads>>>(
        dp1in, dpool1, batch, ach, tout, pk, ps, pp, kMaximum);
    QC_SYNC();
    auto p1max = qc::d2h(dpool1, batch * ach * pl);
    ok &= qc::compare(p1avg, pool1d_ref(p1in, batch, ach, tout, pk, ps, pp, kAverage),
                      qc::Tol::fp32()).report("pool1d average");
    ok &= qc::compare(p1max, pool1d_ref(p1in, batch, ach, tout, pk, ps, pp, kMaximum),
                      qc::Tol::fp32()).report("pool1d maximum");
    checks += 2;

    const long long poh = conv_out(ih, kh, 2, 1, 1);
    const long long pow = conv_out(iw, kw, 1, 0, 1);
    float *dpool2 = qc::dzero<float>(batch * channels * poh * pow);
    pool2d_kernel<<<qc::grid_for(batch * channels * poh * pow, kThreads), kThreads>>>(
        dimage, dpool2, batch, channels, ih, iw, kh, kw, 2, 1, 1, 0, kAverage);
    QC_SYNC();
    auto p2avg = qc::d2h(dpool2, batch * channels * poh * pow);
    pool2d_kernel<<<qc::grid_for(batch * channels * poh * pow, kThreads), kThreads>>>(
        dimage, dpool2, batch, channels, ih, iw, kh, kw, 2, 1, 1, 0, kMaximum);
    QC_SYNC();
    auto p2max = qc::d2h(dpool2, batch * channels * poh * pow);
    ok &= qc::compare(p2avg, pool2d_ref(image, batch, channels, ih, iw, kh, kw,
                                        2, 1, 1, 0, kAverage),
                      qc::Tol::fp32()).report("pool2d average");
    ok &= qc::compare(p2max, pool2d_ref(image, batch, channels, ih, iw, kh, kw,
                                        2, 1, 1, 0, kMaximum),
                      qc::Tol::fp32()).report("pool2d maximum");
    checks += 2;

    std::vector<float> gout = rng.uniforms(batch * channels * poh * pow, -1.0f, 1.0f);
    float *dgout = qc::dnew(gout);
    float *dp2b = qc::dzero<float>(image.size());
    pool2d_backward_kernel<<<qc::grid_for(image.size(), kThreads), kThreads>>>(
        dimage, dgout, dp2b, batch, channels, ih, iw, kh, kw, 2, 1, 1, 0, kAverage);
    QC_SYNC();
    auto p2bavg = qc::d2h(dp2b, image.size());
    pool2d_backward_kernel<<<qc::grid_for(image.size(), kThreads), kThreads>>>(
        dimage, dgout, dp2b, batch, channels, ih, iw, kh, kw, 2, 1, 1, 0, kMaximum);
    QC_SYNC();
    auto p2bmax = qc::d2h(dp2b, image.size());
    ok &= qc::compare(p2bavg, pool2d_backward_ref(image, gout, batch, channels,
                                                  ih, iw, kh, kw, 2, 1, 1, 0,
                                                  kAverage),
                      qc::Tol::fp32()).report("pool2d_backward average");
    ok &= qc::compare(p2bmax, pool2d_backward_ref(image, gout, batch, channels,
                                                  ih, iw, kh, kw, 2, 1, 1, 0,
                                                  kMaximum),
                      qc::Tol::fp32()).report("pool2d_backward maximum");
    checks += 2;

    const long long ab = 2, alen = 9, ain = 3, aout = 4;
    const long long aol = audio_conv_out(alen, ak, 2, 1, 1);
    std::vector<float> ax = rng.uniforms(ab * alen * ain, -1.0f, 1.0f);
    std::vector<float> aw = rng.uniforms(aout * ak * ain, -0.5f, 0.5f);
    std::vector<float> abias = rng.uniforms(aout, -0.1f, 0.1f);
    float *dax = qc::dnew(ax);
    float *daw = qc::dnew(aw);
    float *dab = qc::dnew(abias);
    float *dao = qc::dzero<float>(ab * aol * aout);
    audio_conv1d_direct_kernel<<<qc::grid_for(ab * aol * aout, kThreads), kThreads>>>(
        dax, daw, dab, dao, ab, alen, ain, aout, ak, 2, 1, 1);
    QC_SYNC();
    auto aref = audio_conv1d_ref(ax, aw, abias, ab, alen, ain, aout, ak, 2, 1, 1);
    ok &= qc::compare(qc::d2h(dao, aref.size()), aref, qc::Tol::fp32())
              .report("audio_conv1d_direct");
    ++checks;

    std::vector<float> adw = rng.uniforms(ain * ak, -0.5f, 0.5f);
    std::vector<float> adb = rng.uniforms(ain, -0.1f, 0.1f);
    float *dadw = qc::dnew(adw);
    float *dadb = qc::dnew(adb);
    float *dado = qc::dzero<float>(ab * aol * ain);
    audio_depthwise_conv1d_kernel<<<qc::grid_for(ab * aol * ain, kThreads), kThreads>>>(
        dax, dadw, dadb, dado, ab, alen, ain, ak, 2, 1, 1, 1);
    QC_SYNC();
    auto adref = audio_depthwise_ref(ax, adw, adb, ab, alen, ain, ak, 2, 1, 1, true);
    ok &= qc::compare(qc::d2h(dado, adref.size()), adref, qc::Tol::fp32())
              .report("audio_depthwise_conv1d");
    ++checks;

    float *dcado = qc::dzero<float>(ab * alen * ain);
    audio_causal_depthwise_conv1d_kernel<<<qc::grid_for(ab * alen * ain, kThreads), kThreads>>>(
        dax, dadw, dadb, dcado, ab, alen, ain, 3, 1);
    QC_SYNC();
    auto cadref = audio_causal_depthwise_ref(ax, adw, adb, ab, alen, ain, 3, 1);
    ok &= qc::compare(qc::d2h(dcado, cadref.size()), cadref, qc::Tol::fp32())
              .report("audio_causal_depthwise_conv1d");
    ++checks;

    const long long rb = 2, rlen = 8, rheads = 2, rdim = 64, rpos = 8;
    std::vector<float> rq = rng.uniforms(rb * rlen * rheads * rdim, -0.2f, 0.2f);
    std::vector<float> rk = rng.uniforms(rq.size(), -0.2f, 0.2f);
    std::vector<float> rv = rng.uniforms(rq.size(), -1.0f, 1.0f);
    std::vector<float> rr = rng.uniforms(rpos * rheads * rdim, -0.1f, 0.1f);
    std::vector<float> rscale = rng.uniforms(rdim, -0.2f, 0.2f);
    std::vector<int32_t> rlengths = {8, 5};
    float *drq = qc::dnew(rq);
    float *drk = qc::dnew(rk);
    float *drv = qc::dnew(rv);
    float *drr = qc::dnew(rr);
    float *drs = qc::dnew(rscale);
    int32_t *drl = qc::dnew(rlengths);
    float *dro = qc::dzero<float>(rq.size());
    audio_relative_attention_kernel<<<qc::grid_for(rb * rlen * rheads, kThreads), kThreads>>>(
        drq, drk, drv, drr, drs, drl, dro, rb, rlen, rheads, rdim, rpos,
        4, 3, 2, -1.0f, -1.0f, 10.0f);
    QC_SYNC();
    auto rref = audio_relative_ref(rq, rk, rv, rr, rscale, rlengths, rb, rlen,
                                   rheads, rdim, rpos, 4, 3, 2, -1.0f,
                                   -1.0f, 10.0f);
    ok &= qc::compare(qc::d2h(dro, rq.size()), rref,
                      qc::Tol::fp32().with_elementwise(5e-5, 5e-5))
              .report("audio_relative_attention");
    ++checks;

    qc::dfree(dimage, dcols, drecon, dvolume, dcols3, dcol1, dsig, dcin, dcw,
              dcb, dconv, ddww, ddwb, ddwo, dc3in, dc3w, dc3o, dtinv, dtw1,
              dt1, dtw2, dt2, dpool1, dp1in, dpool2, dgout, dp2b, dax, daw,
              dab, dao, dadw, dadb, dado, dcado, drq, drk, drv, drr, drs, drl,
              dro);

    std::printf("Phase 12 correctness checks: %d\n", checks);
    return ok;
}

void run_bench() {
    std::printf("\n== Phase 12 benchmarks ==\n");
    std::printf("   Timing note: direct CDNA3 routes compared with scalar GPU baselines.\n");
    qc::Rng rng(2212);
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

    const long long b = 2, c = 8, h = 16, w = 16, kh = 3, kw = 3;
    const long long oh = conv_out(h, kh, 1, 1, 1), ow = conv_out(w, kw, 1, 1, 1);
    std::vector<float> img = rng.uniforms(b * c * h * w, -1.0f, 1.0f);
    float *dimg = qc::dnew(img);
    float *dcols = qc::dzero<float>(b * oh * ow * c * kh * kw);
    auto im2col_s = qc::bench([&] {
        im2col_2d_scalar_kernel<<<1, 1>>>(dimg, dcols, b, c, h, w, kh, kw, 1, 1, 1, 1, 1, 1);
    }, 2, 3);
    auto im2col_c = bench_repeated([&] {
        im2col_2d_kernel<<<qc::grid_for(b * oh * ow * c * kh * kw, kThreads), kThreads>>>(
            dimg, dcols, b, c, h, w, kh, kw, 1, 1, 1, 1, 1, 1);
    }, 1000, 5, 20);
    const double im2col_bytes = static_cast<double>(b * oh * ow * c * kh * kw) * 2.0 * sizeof(float);
    im2col_s.report_bandwidth("im2col_2d scalar", im2col_bytes);
    im2col_c.report_bandwidth("im2col_2d candidate", im2col_bytes);
    qc::report_ab("im2col_2d", im2col_s, im2col_c);

    const long long oc = 8;
    std::vector<float> weight = rng.uniforms(oc * c * kh * kw, -0.2f, 0.2f);
    std::vector<float> bias = rng.uniforms(oc, -0.1f, 0.1f);
    float *dw = qc::dnew(weight);
    float *dbias = qc::dnew(bias);
    float *dout = qc::dzero<float>(b * oc * oh * ow);
    auto conv_s = qc::bench([&] {
        conv2d_scalar_kernel<<<1, 1>>>(dimg, dw, dbias, dout, b, c, oc, h, w, kh, kw,
                                       1, 1, 1, 1, 1, 1);
    }, 2, 3);
    auto conv_c = bench_repeated([&] {
        conv2d_kernel<<<qc::grid_for(b * oc * oh * ow, kThreads), kThreads>>>(
            dimg, dw, dbias, dout, b, c, oc, h, w, kh, kw, 1, 1, 1, 1, 1, 1);
    }, 20, 5, 20);
    const double conv_flops = static_cast<double>(b) * oc * oh * ow * c * kh * kw * 2.0;
    conv_s.report_compute("conv2d scalar", conv_flops);
    conv_c.report_compute("conv2d candidate", conv_flops);
    qc::report_ab("conv2d", conv_s, conv_c);

    float *dpool = qc::dzero<float>(b * c * oh * ow);
    auto pool_s = qc::bench([&] {
        pool2d_scalar_kernel<<<1, 1>>>(dimg, dpool, b, c, h, w, kh, kw, 1, 1, 1, 1, kAverage);
    }, 2, 3);
    auto pool_c = bench_repeated([&] {
        pool2d_kernel<<<qc::grid_for(b * c * oh * ow, kThreads), kThreads>>>(
            dimg, dpool, b, c, h, w, kh, kw, 1, 1, 1, 1, kAverage);
    }, 1000, 5, 20);
    const double pool_bytes = static_cast<double>(b * c * oh * ow * kh * kw) * sizeof(float);
    pool_s.report_bandwidth("pool2d scalar", pool_bytes);
    pool_c.report_bandwidth("pool2d candidate", pool_bytes);
    qc::report_ab("pool2d", pool_s, pool_c);

    const long long ab = 2, len = 128, ain = 16, aout = 16, ak = 5;
    const long long aol = audio_conv_out(len, ak, 1, 2, 1);
    std::vector<float> ax = rng.uniforms(ab * len * ain, -1.0f, 1.0f);
    std::vector<float> aw = rng.uniforms(aout * ak * ain, -0.1f, 0.1f);
    std::vector<float> abias = rng.uniforms(aout, -0.1f, 0.1f);
    float *dax = qc::dnew(ax);
    float *daw = qc::dnew(aw);
    float *dab = qc::dnew(abias);
    float *dao = qc::dzero<float>(ab * aol * aout);
    auto aconv_s = qc::bench([&] {
        audio_conv1d_direct_scalar_kernel<<<1, 1>>>(dax, daw, dab, dao, ab, len, ain, aout, ak, 1, 2, 1);
    }, 2, 3);
    auto aconv_c = bench_repeated([&] {
        audio_conv1d_direct_kernel<<<qc::grid_for(ab * aol * aout, kThreads), kThreads>>>(
            dax, daw, dab, dao, ab, len, ain, aout, ak, 1, 2, 1);
    }, 20, 5, 20);
    const double aconv_flops = static_cast<double>(ab) * aol * aout * ak * ain * 2.0;
    aconv_s.report_compute("audio_conv1d scalar", aconv_flops);
    aconv_c.report_compute("audio_conv1d candidate", aconv_flops);
    qc::report_ab("audio_conv1d_direct", aconv_s, aconv_c);

    const long long rb = 2, rlen = 32, heads = 2, dim = 64, rpos = 8;
    std::vector<float> rq = rng.uniforms(rb * rlen * heads * dim, -0.2f, 0.2f);
    std::vector<float> rk = rng.uniforms(rq.size(), -0.2f, 0.2f);
    std::vector<float> rv = rng.uniforms(rq.size(), -1.0f, 1.0f);
    std::vector<float> rr = rng.uniforms(rpos * heads * dim, -0.1f, 0.1f);
    std::vector<float> rs = rng.uniforms(dim, -0.2f, 0.2f);
    std::vector<int32_t> rl(rb, static_cast<int32_t>(rlen));
    float *drq = qc::dnew(rq);
    float *drk = qc::dnew(rk);
    float *drv = qc::dnew(rv);
    float *drr = qc::dnew(rr);
    float *drs = qc::dnew(rs);
    int32_t *drl = qc::dnew(rl);
    float *dro = qc::dzero<float>(rq.size());
    auto rel_s = qc::bench([&] {
        audio_relative_attention_scalar_kernel<<<1, 1>>>(
            drq, drk, drv, drr, drs, drl, dro, rb, rlen, heads, dim, rpos, 16, 8, 4,
            -1.0f, -1.0f, 10.0f);
    }, 2, 3);
    auto rel_c = qc::bench([&] {
        audio_relative_attention_kernel<<<qc::grid_for(rb * rlen * heads, kThreads), kThreads>>>(
            drq, drk, drv, drr, drs, drl, dro, rb, rlen, heads, dim, rpos, 16, 8, 4,
            -1.0f, -1.0f, 10.0f);
    }, 10, 30);
    const double rel_flops = static_cast<double>(rb * rlen * heads * (16 + 8 + 4) * dim * 2);
    rel_s.report_compute("audio_relative_attention scalar", rel_flops);
    rel_c.report_compute("audio_relative_attention current",
                         rel_flops);
    qc::report_ab("audio_relative_attention", rel_s, rel_c);

    qc::dfree(dimg, dcols, dw, dbias, dout, dpool, dax, daw, dab, dao, drq,
              drk, drv, drr, drs, drl, dro);
}

}  // namespace

int main(int argc, char **argv) {
    const bool do_bench = argc > 1 && std::string(argv[1]) == "--bench";
    qc::print_environment("phase12_conv_audio");
    const bool ok = run_correctness();
    if (do_bench) run_bench();
    std::printf("%s\n", ok ? "ALL PASS" : "FAIL");
    return ok ? 0 : 1;
}
