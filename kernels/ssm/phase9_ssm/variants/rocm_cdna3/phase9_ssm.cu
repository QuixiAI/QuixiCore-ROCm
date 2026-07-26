/**
 * @file
 * @brief Phase 9 Mamba2/SSD backward, decode, and DSV4 hyper-connection ports.
 */
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "../../../../common/cdna3_harness.cuh"

namespace {

constexpr long long kConnections = 4;
constexpr long long kMixOffset = 8;
constexpr long long kMixSize = 24;

std::vector<double> to_ref(const std::vector<float> &values) {
    return std::vector<double>(values.begin(), values.end());
}

__device__ __forceinline__ double dot_device(const float *a, const float *b,
                                             long long dim) {
    double sum = 0.0;
    for (long long d = 0; d < dim; ++d) {
        sum += static_cast<double>(a[d]) * b[d];
    }
    return sum;
}

__device__ __forceinline__ double pair_similarity(const float *c, const float *b,
                                                  long long base,
                                                  long long target,
                                                  long long source,
                                                  long long dim) {
    return dot_device(c + base + target * dim, b + base + source * dim, dim);
}

__device__ __forceinline__ double pair_grad_dot_x(const float *grad_y,
                                                  const float *x,
                                                  long long base,
                                                  long long target,
                                                  long long source,
                                                  long long dim) {
    return dot_device(grad_y + base + target * dim, x + base + source * dim, dim);
}

__device__ __forceinline__ float hc_sigmoid_device(float value) {
    if (value >= 0.0f) return 1.0f / (1.0f + expf(-value));
    const float e = expf(value);
    return e / (1.0f + e);
}

// ---------------------------------------------------------------------------
// Mamba2 / SSD backward
// ---------------------------------------------------------------------------

__global__ void mamba2_backward_scalar_kernel(
    const float *c, const float *b, const float *x, const float *cumulative_log,
    const float *grad_y, float *grad_c, float *grad_b, float *grad_x,
    float *grad_cumulative_log, long long batch, long long heads,
    long long sequence, long long dim) {
    const long long bh = blockIdx.x;
    if (threadIdx.x != 0 || bh >= batch * heads) return;
    const long long base = bh * sequence * dim;
    const long long decay_base = bh * sequence;
    for (long long index = 0; index < sequence * dim; ++index) {
        grad_c[base + index] = 0.0f;
        grad_b[base + index] = 0.0f;
        grad_x[base + index] = 0.0f;
    }
    for (long long index = 0; index < sequence; ++index) {
        grad_cumulative_log[decay_base + index] = 0.0f;
    }
    for (long long target = 0; target < sequence; ++target) {
        for (long long source = 0; source <= target; ++source) {
            double similarity = 0.0;
            double grad_dot_x = 0.0;
            for (long long d = 0; d < dim; ++d) {
                similarity += static_cast<double>(c[base + target * dim + d]) *
                              b[base + source * dim + d];
                grad_dot_x += static_cast<double>(grad_y[base + target * dim + d]) *
                              x[base + source * dim + d];
            }
            const double decay =
                exp(static_cast<double>(cumulative_log[decay_base + target] -
                                        cumulative_log[decay_base + source]));
            const double score_grad = decay * grad_dot_x;
            for (long long d = 0; d < dim; ++d) {
                grad_c[base + target * dim + d] +=
                    static_cast<float>(score_grad * b[base + source * dim + d]);
                grad_b[base + source * dim + d] +=
                    static_cast<float>(score_grad * c[base + target * dim + d]);
                grad_x[base + source * dim + d] += static_cast<float>(
                    decay * similarity * grad_y[base + target * dim + d]);
            }
            const float decay_gradient =
                static_cast<float>(decay * similarity * grad_dot_x);
            grad_cumulative_log[decay_base + target] += decay_gradient;
            grad_cumulative_log[decay_base + source] -= decay_gradient;
        }
    }
}

__global__ void mamba2_grad_c_kernel(const float *c, const float *b,
                                     const float *x,
                                     const float *cumulative_log,
                                     const float *grad_y, float *grad_c,
                                     long long batch, long long heads,
                                     long long sequence, long long dim) {
    const long long target = blockIdx.x;
    const long long d = blockIdx.y;
    const long long bh = blockIdx.z;
    const int lane = threadIdx.x;
    if (bh >= batch * heads || target >= sequence || d >= dim) return;
    const long long base = bh * sequence * dim;
    const long long decay_base = bh * sequence;
    double partial = 0.0;
    for (long long source = lane; source <= target; source += qc::kWave) {
        const double grad_dot_x =
            pair_grad_dot_x(grad_y, x, base, target, source, dim);
        const double decay =
            exp(static_cast<double>(cumulative_log[decay_base + target] -
                                    cumulative_log[decay_base + source]));
        partial += decay * grad_dot_x * b[base + source * dim + d];
    }
    partial = qc::wave_reduce_sum(partial);
    if (lane == 0) grad_c[base + target * dim + d] = static_cast<float>(partial);
}

__global__ void mamba2_grad_bx_kernel(const float *c, const float *b,
                                      const float *x,
                                      const float *cumulative_log,
                                      const float *grad_y, float *grad_b,
                                      float *grad_x, long long batch,
                                      long long heads, long long sequence,
                                      long long dim) {
    const long long source = blockIdx.x;
    const long long d = blockIdx.y;
    const long long bh = blockIdx.z;
    const int lane = threadIdx.x;
    if (bh >= batch * heads || source >= sequence || d >= dim) return;
    const long long base = bh * sequence * dim;
    const long long decay_base = bh * sequence;
    double partial_b = 0.0;
    double partial_x = 0.0;
    for (long long target = source + lane; target < sequence; target += qc::kWave) {
        const double grad_dot_x =
            pair_grad_dot_x(grad_y, x, base, target, source, dim);
        const double similarity =
            pair_similarity(c, b, base, target, source, dim);
        const double decay =
            exp(static_cast<double>(cumulative_log[decay_base + target] -
                                    cumulative_log[decay_base + source]));
        partial_b += decay * grad_dot_x * c[base + target * dim + d];
        partial_x += decay * similarity * grad_y[base + target * dim + d];
    }
    partial_b = qc::wave_reduce_sum(partial_b);
    partial_x = qc::wave_reduce_sum(partial_x);
    if (lane == 0) {
        grad_b[base + source * dim + d] = static_cast<float>(partial_b);
        grad_x[base + source * dim + d] = static_cast<float>(partial_x);
    }
}

__global__ void mamba2_grad_cl_kernel(const float *c, const float *b,
                                      const float *x,
                                      const float *cumulative_log,
                                      const float *grad_y,
                                      float *grad_cumulative_log,
                                      long long batch, long long heads,
                                      long long sequence, long long dim) {
    const long long item = blockIdx.x;
    const long long bh = blockIdx.y;
    const int lane = threadIdx.x;
    if (bh >= batch * heads || item >= sequence) return;
    const long long base = bh * sequence * dim;
    const long long decay_base = bh * sequence;
    double plus = 0.0;
    double minus = 0.0;
    for (long long source = lane; source <= item; source += qc::kWave) {
        const double similarity = pair_similarity(c, b, base, item, source, dim);
        const double grad_dot_x = pair_grad_dot_x(grad_y, x, base, item, source, dim);
        const double decay =
            exp(static_cast<double>(cumulative_log[decay_base + item] -
                                    cumulative_log[decay_base + source]));
        plus += decay * similarity * grad_dot_x;
    }
    for (long long target = item + lane; target < sequence; target += qc::kWave) {
        const double similarity = pair_similarity(c, b, base, target, item, dim);
        const double grad_dot_x = pair_grad_dot_x(grad_y, x, base, target, item, dim);
        const double decay =
            exp(static_cast<double>(cumulative_log[decay_base + target] -
                                    cumulative_log[decay_base + item]));
        minus += decay * similarity * grad_dot_x;
    }
    plus = qc::wave_reduce_sum(plus);
    minus = qc::wave_reduce_sum(minus);
    if (lane == 0) grad_cumulative_log[decay_base + item] =
                       static_cast<float>(plus - minus);
}

// ---------------------------------------------------------------------------
// SSD decode
// ---------------------------------------------------------------------------

__global__ void ssd_decode_scalar_kernel(const float *state, const float *alpha,
                                         const float *x, const float *k,
                                         const float *q, float *y,
                                         float *next_state, long long batch,
                                         long long heads, long long dim) {
    const long long bh = blockIdx.x;
    if (threadIdx.x != 0 || bh >= batch * heads) return;
    const float *state_base = state + bh * dim * dim;
    float *next = next_state + bh * dim * dim;
    const float *xr = x + bh * dim;
    const float *kr = k + bh * dim;
    const float *qr = q + bh * dim;
    for (long long value = 0; value < dim; ++value) {
        double result = 0.0;
        for (long long key = 0; key < dim; ++key) {
            const long long index = value * dim + key;
            next[index] = alpha[bh] * state_base[index] + xr[value] * kr[key];
            result += static_cast<double>(next[index]) * qr[key];
        }
        y[bh * dim + value] = static_cast<float>(result);
    }
}

__global__ void ssd_decode_wave_kernel(const float *state, const float *alpha,
                                       const float *x, const float *k,
                                       const float *q, float *y,
                                       float *next_state, long long batch,
                                       long long heads, long long dim) {
    const long long value = blockIdx.x;
    const long long bh = blockIdx.y;
    const int lane = threadIdx.x;
    if (bh >= batch * heads || value >= dim || lane >= qc::kWave) return;
    const float *state_row = state + (bh * dim + value) * dim;
    float *next_row = next_state + (bh * dim + value) * dim;
    const float xv = x[bh * dim + value];
    const float a = alpha[bh];
    float partial = 0.0f;
    for (long long key = lane; key < dim; key += qc::kWave) {
        const float s = a * state_row[key] + xv * k[bh * dim + key];
        next_row[key] = s;
        partial += s * q[bh * dim + key];
    }
    const float result = qc::wave_reduce_sum(partial);
    if (lane == 0) y[bh * dim + value] = result;
}

// ---------------------------------------------------------------------------
// DSV4 hyper-connections
// ---------------------------------------------------------------------------

__global__ void dsv4_hc_comb_scalar_kernel(const float *mixes,
                                           const float *scale,
                                           const float *base, float *comb,
                                           long long tokens, float eps,
                                           int iterations) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (long long token = 0; token < tokens; ++token) {
        float matrix[16];
        for (long long source = 0; source < kConnections; ++source) {
            float maximum = -INFINITY;
            for (long long destination = 0; destination < kConnections; ++destination) {
                const long long index = destination + kConnections * source;
                matrix[index] = mixes[token * kMixSize + kMixOffset + index] *
                                    scale[2] +
                                base[kMixOffset + index];
                maximum = fmaxf(maximum, matrix[index]);
            }
            float sum = 0.0f;
            for (long long destination = 0; destination < kConnections; ++destination) {
                const long long index = destination + kConnections * source;
                matrix[index] = expf(matrix[index] - maximum);
                sum += matrix[index];
            }
            for (long long destination = 0; destination < kConnections; ++destination) {
                const long long index = destination + kConnections * source;
                matrix[index] = matrix[index] / sum + eps;
            }
        }
        for (long long destination = 0; destination < kConnections; ++destination) {
            float sum = eps;
            for (long long source = 0; source < kConnections; ++source) {
                sum += matrix[destination + kConnections * source];
            }
            for (long long source = 0; source < kConnections; ++source) {
                matrix[destination + kConnections * source] /= sum;
            }
        }
        for (int iteration = 1; iteration < iterations; ++iteration) {
            for (long long source = 0; source < kConnections; ++source) {
                float sum = eps;
                for (long long destination = 0; destination < kConnections; ++destination) {
                    sum += matrix[destination + kConnections * source];
                }
                for (long long destination = 0; destination < kConnections; ++destination) {
                    matrix[destination + kConnections * source] /= sum;
                }
            }
            for (long long destination = 0; destination < kConnections; ++destination) {
                float sum = eps;
                for (long long source = 0; source < kConnections; ++source) {
                    sum += matrix[destination + kConnections * source];
                }
                for (long long source = 0; source < kConnections; ++source) {
                    matrix[destination + kConnections * source] /= sum;
                }
            }
        }
        for (long long index = 0; index < 16; ++index) comb[token * 16 + index] = matrix[index];
    }
}

__global__ void dsv4_hc_comb_token_kernel(const float *mixes,
                                          const float *scale,
                                          const float *base, float *comb,
                                          long long tokens, float eps,
                                          int iterations) {
    const long long token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= tokens) return;
    float matrix[16];
    for (long long source = 0; source < kConnections; ++source) {
        float maximum = -INFINITY;
        for (long long destination = 0; destination < kConnections; ++destination) {
            const long long index = destination + kConnections * source;
            matrix[index] =
                mixes[token * kMixSize + kMixOffset + index] * scale[2] +
                base[kMixOffset + index];
            maximum = fmaxf(maximum, matrix[index]);
        }
        float sum = 0.0f;
        for (long long destination = 0; destination < kConnections; ++destination) {
            const long long index = destination + kConnections * source;
            matrix[index] = expf(matrix[index] - maximum);
            sum += matrix[index];
        }
        for (long long destination = 0; destination < kConnections; ++destination) {
            const long long index = destination + kConnections * source;
            matrix[index] = matrix[index] / sum + eps;
        }
    }
    for (long long destination = 0; destination < kConnections; ++destination) {
        float sum = eps;
        for (long long source = 0; source < kConnections; ++source) {
            sum += matrix[destination + kConnections * source];
        }
        for (long long source = 0; source < kConnections; ++source) {
            matrix[destination + kConnections * source] /= sum;
        }
    }
    for (int iteration = 1; iteration < iterations; ++iteration) {
        for (long long source = 0; source < kConnections; ++source) {
            float sum = eps;
            for (long long destination = 0; destination < kConnections; ++destination) {
                sum += matrix[destination + kConnections * source];
            }
            for (long long destination = 0; destination < kConnections; ++destination) {
                matrix[destination + kConnections * source] /= sum;
            }
        }
        for (long long destination = 0; destination < kConnections; ++destination) {
            float sum = eps;
            for (long long source = 0; source < kConnections; ++source) {
                sum += matrix[destination + kConnections * source];
            }
            for (long long source = 0; source < kConnections; ++source) {
                matrix[destination + kConnections * source] /= sum;
            }
        }
    }
    for (long long index = 0; index < 16; ++index) comb[token * 16 + index] = matrix[index];
}

__global__ void dsv4_hc_pre_scalar_kernel(const float *x, const float *weights,
                                          float *output, long long tokens,
                                          long long embedding) {
    const long long token = blockIdx.x;
    if (threadIdx.x != 0 || token >= tokens) return;
    for (long long dimension = 0; dimension < embedding; ++dimension) {
        float sum = 0.0f;
        for (long long connection = 0; connection < kConnections; ++connection) {
            sum += x[(token * kConnections + connection) * embedding + dimension] *
                   weights[token * kConnections + connection];
        }
        output[token * embedding + dimension] = sum;
    }
}

__global__ void dsv4_hc_pre_kernel(const float *x, const float *weights,
                                   float *output, long long count,
                                   long long embedding) {
    const long long item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= count) return;
    const long long token = item / embedding;
    const long long dimension = item - token * embedding;
    float sum = 0.0f;
    for (long long connection = 0; connection < kConnections; ++connection) {
        sum += x[(token * kConnections + connection) * embedding + dimension] *
               weights[token * kConnections + connection];
    }
    output[item] = sum;
}

__global__ void dsv4_hc_post_scalar_kernel(const float *x,
                                           const float *residual,
                                           const float *post,
                                           const float *comb, float *output,
                                           long long tokens,
                                           long long embedding) {
    const long long item = blockIdx.x;
    if (threadIdx.x != 0 || item >= tokens * kConnections) return;
    const long long token = item / kConnections;
    const long long destination = item % kConnections;
    for (long long dimension = 0; dimension < embedding; ++dimension) {
        float sum =
            x[token * embedding + dimension] * post[token * kConnections + destination];
        for (long long source = 0; source < kConnections; ++source) {
            sum += residual[(token * kConnections + source) * embedding + dimension] *
                   comb[token * 16 + source * kConnections + destination];
        }
        output[(token * kConnections + destination) * embedding + dimension] = sum;
    }
}

__global__ void dsv4_hc_post_kernel(const float *x, const float *residual,
                                    const float *post, const float *comb,
                                    float *output, long long count,
                                    long long embedding) {
    const long long item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= count) return;
    const long long token = item / (kConnections * embedding);
    const long long within = item - token * kConnections * embedding;
    const long long destination = within / embedding;
    const long long dimension = within - destination * embedding;
    float sum =
        x[token * embedding + dimension] * post[token * kConnections + destination];
    for (long long source = 0; source < kConnections; ++source) {
        sum += residual[(token * kConnections + source) * embedding + dimension] *
               comb[token * 16 + source * kConnections + destination];
    }
    output[item] = sum;
}

// ---------------------------------------------------------------------------
// Host references
// ---------------------------------------------------------------------------

void mamba2_backward_ref(const std::vector<float> &c,
                         const std::vector<float> &b,
                         const std::vector<float> &x,
                         const std::vector<float> &cumulative_log,
                         const std::vector<float> &grad_y,
                         std::vector<float> &grad_c,
                         std::vector<float> &grad_b,
                         std::vector<float> &grad_x,
                         std::vector<float> &grad_cumulative_log,
                         long long batch, long long heads, long long sequence,
                         long long dim) {
    const long long tensor_count = batch * heads * sequence * dim;
    const long long decay_count = batch * heads * sequence;
    grad_c.assign(static_cast<size_t>(tensor_count), 0.0f);
    grad_b.assign(static_cast<size_t>(tensor_count), 0.0f);
    grad_x.assign(static_cast<size_t>(tensor_count), 0.0f);
    grad_cumulative_log.assign(static_cast<size_t>(decay_count), 0.0f);
    for (long long bh = 0; bh < batch * heads; ++bh) {
        const long long base = bh * sequence * dim;
        const long long decay_base = bh * sequence;
        for (long long target = 0; target < sequence; ++target) {
            for (long long source = 0; source <= target; ++source) {
                double similarity = 0.0;
                double grad_dot_x = 0.0;
                for (long long d = 0; d < dim; ++d) {
                    similarity += static_cast<double>(c[base + target * dim + d]) *
                                  b[base + source * dim + d];
                    grad_dot_x += static_cast<double>(grad_y[base + target * dim + d]) *
                                  x[base + source * dim + d];
                }
                const double decay =
                    std::exp(static_cast<double>(cumulative_log[decay_base + target] -
                                                 cumulative_log[decay_base + source]));
                const double score_grad = decay * grad_dot_x;
                for (long long d = 0; d < dim; ++d) {
                    grad_c[base + target * dim + d] +=
                        static_cast<float>(score_grad * b[base + source * dim + d]);
                    grad_b[base + source * dim + d] +=
                        static_cast<float>(score_grad * c[base + target * dim + d]);
                    grad_x[base + source * dim + d] += static_cast<float>(
                        decay * similarity * grad_y[base + target * dim + d]);
                }
                const float decay_gradient =
                    static_cast<float>(decay * similarity * grad_dot_x);
                grad_cumulative_log[decay_base + target] += decay_gradient;
                grad_cumulative_log[decay_base + source] -= decay_gradient;
            }
        }
    }
}

void ssd_decode_ref(const std::vector<float> &state,
                    const std::vector<float> &alpha,
                    const std::vector<float> &x, const std::vector<float> &k,
                    const std::vector<float> &q, std::vector<float> &y,
                    std::vector<float> &next_state, long long batch,
                    long long heads, long long dim) {
    y.assign(static_cast<size_t>(batch * heads * dim), 0.0f);
    next_state.assign(static_cast<size_t>(batch * heads * dim * dim), 0.0f);
    for (long long bh = 0; bh < batch * heads; ++bh) {
        const float *state_base = state.data() + bh * dim * dim;
        float *next = next_state.data() + bh * dim * dim;
        const float *xr = x.data() + bh * dim;
        const float *kr = k.data() + bh * dim;
        const float *qr = q.data() + bh * dim;
        for (long long value = 0; value < dim; ++value) {
            double result = 0.0;
            for (long long key = 0; key < dim; ++key) {
                const long long index = value * dim + key;
                next[index] = alpha[bh] * state_base[index] + xr[value] * kr[key];
                result += static_cast<double>(next[index]) * qr[key];
            }
            y[bh * dim + value] = static_cast<float>(result);
        }
    }
}

std::vector<float> dsv4_hc_comb_ref(const std::vector<float> &mixes,
                                    const std::vector<float> &scale,
                                    const std::vector<float> &base,
                                    long long tokens, float eps,
                                    int iterations) {
    std::vector<float> comb(static_cast<size_t>(tokens * 16), 0.0f);
    for (long long token = 0; token < tokens; ++token) {
        float matrix[16];
        for (long long source = 0; source < kConnections; ++source) {
            float maximum = -INFINITY;
            for (long long destination = 0; destination < kConnections; ++destination) {
                const long long index = destination + kConnections * source;
                matrix[index] = mixes[token * kMixSize + kMixOffset + index] *
                                    scale[2] +
                                base[kMixOffset + index];
                maximum = std::max(maximum, matrix[index]);
            }
            float sum = 0.0f;
            for (long long destination = 0; destination < kConnections; ++destination) {
                const long long index = destination + kConnections * source;
                matrix[index] = std::exp(matrix[index] - maximum);
                sum += matrix[index];
            }
            for (long long destination = 0; destination < kConnections; ++destination) {
                const long long index = destination + kConnections * source;
                matrix[index] = matrix[index] / sum + eps;
            }
        }
        auto normalize_columns = [&] {
            for (long long destination = 0; destination < kConnections; ++destination) {
                float sum = eps;
                for (long long source = 0; source < kConnections; ++source) {
                    sum += matrix[destination + kConnections * source];
                }
                for (long long source = 0; source < kConnections; ++source) {
                    matrix[destination + kConnections * source] /= sum;
                }
            }
        };
        auto normalize_rows = [&] {
            for (long long source = 0; source < kConnections; ++source) {
                float sum = eps;
                for (long long destination = 0; destination < kConnections; ++destination) {
                    sum += matrix[destination + kConnections * source];
                }
                for (long long destination = 0; destination < kConnections; ++destination) {
                    matrix[destination + kConnections * source] /= sum;
                }
            }
        };
        normalize_columns();
        for (int iteration = 1; iteration < iterations; ++iteration) {
            normalize_rows();
            normalize_columns();
        }
        for (long long index = 0; index < 16; ++index) comb[token * 16 + index] = matrix[index];
    }
    return comb;
}

std::vector<float> dsv4_hc_pre_ref(const std::vector<float> &x,
                                   const std::vector<float> &weights,
                                   long long tokens, long long embedding) {
    std::vector<float> output(static_cast<size_t>(tokens * embedding), 0.0f);
    for (long long token = 0; token < tokens; ++token) {
        for (long long dimension = 0; dimension < embedding; ++dimension) {
            float sum = 0.0f;
            for (long long connection = 0; connection < kConnections; ++connection) {
                sum += x[(token * kConnections + connection) * embedding + dimension] *
                       weights[token * kConnections + connection];
            }
            output[token * embedding + dimension] = sum;
        }
    }
    return output;
}

std::vector<float> dsv4_hc_post_ref(const std::vector<float> &x,
                                    const std::vector<float> &residual,
                                    const std::vector<float> &post,
                                    const std::vector<float> &comb,
                                    long long tokens, long long embedding) {
    std::vector<float> output(static_cast<size_t>(tokens * kConnections * embedding),
                              0.0f);
    for (long long token = 0; token < tokens; ++token) {
        for (long long destination = 0; destination < kConnections; ++destination) {
            for (long long dimension = 0; dimension < embedding; ++dimension) {
                float sum =
                    x[token * embedding + dimension] *
                    post[token * kConnections + destination];
                for (long long source = 0; source < kConnections; ++source) {
                    sum += residual[(token * kConnections + source) * embedding +
                                    dimension] *
                           comb[token * 16 + source * kConnections + destination];
                }
                output[(token * kConnections + destination) * embedding + dimension] =
                    sum;
            }
        }
    }
    return output;
}

void make_cumulative_log(std::vector<float> &cumulative_log, qc::Rng &rng,
                         long long batch, long long heads, long long sequence) {
    for (long long bh = 0; bh < batch * heads; ++bh) {
        double sum = 0.0;
        for (long long t = 0; t < sequence; ++t) {
            sum += rng.uniform(-0.08f, 0.02f);
            cumulative_log[static_cast<size_t>(bh * sequence + t)] =
                static_cast<float>(sum);
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

void launch_mamba2_backward_candidate(float *dc, float *db, float *dx,
                                      float *dcl, const float *c,
                                      const float *b, const float *x,
                                      const float *cl, const float *dy,
                                      long long batch, long long heads,
                                      long long sequence, long long dim) {
    mamba2_grad_c_kernel<<<dim3(sequence, dim, batch * heads), qc::kWave>>>(
        c, b, x, cl, dy, dc, batch, heads, sequence, dim);
    mamba2_grad_bx_kernel<<<dim3(sequence, dim, batch * heads), qc::kWave>>>(
        c, b, x, cl, dy, db, dx, batch, heads, sequence, dim);
    mamba2_grad_cl_kernel<<<dim3(sequence, batch * heads), qc::kWave>>>(
        c, b, x, cl, dy, dcl, batch, heads, sequence, dim);
}

void report_pair(const char *label, const qc::Bench &baseline,
                 const qc::Bench &candidate, double metric,
                 bool compute = true) {
    if (compute) {
        baseline.report_compute(std::string(label) + " scalar", metric);
        candidate.report_compute(std::string(label) + " candidate", metric);
    } else {
        baseline.report_bandwidth(std::string(label) + " scalar", metric);
        candidate.report_bandwidth(std::string(label) + " candidate", metric);
    }
    qc::report_ab(label, baseline, candidate);
}

bool check_mamba2_backward(const char *label, long long batch, long long heads,
                           long long sequence, long long dim, qc::Rng &rng) {
    bool ok = true;
    const size_t tensor_count = static_cast<size_t>(batch * heads * sequence * dim);
    const size_t decay_count = static_cast<size_t>(batch * heads * sequence);
    auto c = rng.uniforms(tensor_count, -0.25f, 0.25f);
    auto b = rng.uniforms(tensor_count, -0.25f, 0.25f);
    auto x = rng.uniforms(tensor_count, -0.25f, 0.25f);
    auto grad_y = rng.uniforms(tensor_count, -0.25f, 0.25f);
    std::vector<float> cl(decay_count);
    make_cumulative_log(cl, rng, batch, heads, sequence);
    std::vector<float> rc, rb, rx, rcl;
    mamba2_backward_ref(c, b, x, cl, grad_y, rc, rb, rx, rcl, batch, heads,
                        sequence, dim);
    float *dc = qc::dzero<float>(tensor_count);
    float *db = qc::dzero<float>(tensor_count);
    float *dx = qc::dzero<float>(tensor_count);
    float *dcl = qc::dzero<float>(decay_count);
    float *dc_in = qc::dnew(c);
    float *db_in = qc::dnew(b);
    float *dx_in = qc::dnew(x);
    float *dcl_in = qc::dnew(cl);
    float *dy = qc::dnew(grad_y);
    launch_mamba2_backward_candidate(dc, db, dx, dcl, dc_in, db_in, dx_in,
                                     dcl_in, dy, batch, heads, sequence, dim);
    QC_SYNC();
    const auto tol = qc::Tol::fp32().with_elementwise(3e-4, 3e-5);
    ok &= qc::compare(qc::d2h(dc, tensor_count), to_ref(rc), tol)
              .report(std::string(label) + " grad_c");
    ok &= qc::compare(qc::d2h(db, tensor_count), to_ref(rb), tol)
              .report(std::string(label) + " grad_b");
    ok &= qc::compare(qc::d2h(dx, tensor_count), to_ref(rx), tol)
              .report(std::string(label) + " grad_x");
    ok &= qc::compare(qc::d2h(dcl, decay_count), to_ref(rcl), tol)
              .report(std::string(label) + " grad_cumulative_log");
    qc::dfree(dc, db, dx, dcl, dc_in, db_in, dx_in, dcl_in, dy);
    return ok;
}

bool run_correctness() {
    bool ok = true;
    int checks = 0;
    qc::Rng rng(0x9009);

    ok &= check_mamba2_backward("mamba2_backward", 2, 2, 5, 13, rng);
    checks += 4;
    ok &= check_mamba2_backward("ssd_chunked_backward", 2, 1, 6, 11, rng);
    checks += 4;

    {
        constexpr long long batch = 2;
        constexpr long long heads = 2;
        constexpr long long dim = 17;
        const size_t state_count = static_cast<size_t>(batch * heads * dim * dim);
        const size_t vector_count = static_cast<size_t>(batch * heads * dim);
        auto state = rng.uniforms(state_count, -0.20f, 0.20f);
        auto alpha = rng.uniforms(static_cast<size_t>(batch * heads), 0.70f, 0.98f);
        auto x = rng.uniforms(vector_count, -0.20f, 0.20f);
        auto k = rng.uniforms(vector_count, -0.20f, 0.20f);
        auto q = rng.uniforms(vector_count, -0.20f, 0.20f);
        std::vector<float> ref_y, ref_state;
        ssd_decode_ref(state, alpha, x, k, q, ref_y, ref_state, batch, heads,
                       dim);
        float *ds = qc::dnew(state);
        float *da = qc::dnew(alpha);
        float *dx = qc::dnew(x);
        float *dk = qc::dnew(k);
        float *dq = qc::dnew(q);
        float *dy = qc::dzero<float>(vector_count);
        float *dns = qc::dzero<float>(state_count);
        ssd_decode_wave_kernel<<<dim3(dim, batch * heads), qc::kWave>>>(
            ds, da, dx, dk, dq, dy, dns, batch, heads, dim);
        QC_SYNC();
        ok &= qc::compare(qc::d2h(dy, vector_count), to_ref(ref_y),
                          qc::Tol::fp32().with_elementwise(2e-5, 2e-6))
                  .report("ssd_decode y");
        ++checks;
        ok &= qc::compare(qc::d2h(dns, state_count), to_ref(ref_state),
                          qc::Tol::fp32().with_elementwise(2e-5, 2e-6))
                  .report("ssd_decode next_state");
        ++checks;
        qc::dfree(ds, da, dx, dk, dq, dy, dns);
    }

    {
        constexpr long long tokens = 31;
        constexpr float eps = 1e-4f;
        constexpr int iterations = 3;
        auto mixes = rng.uniforms(static_cast<size_t>(tokens * kMixSize), -0.8f, 0.8f);
        auto scale = rng.uniforms(4, 0.5f, 1.5f);
        auto base = rng.uniforms(kMixSize, -0.4f, 0.4f);
        const auto ref = dsv4_hc_comb_ref(mixes, scale, base, tokens, eps,
                                          iterations);
        float *dm = qc::dnew(mixes);
        float *ds = qc::dnew(scale);
        float *db = qc::dnew(base);
        float *dc = qc::dzero<float>(ref.size());
        dsv4_hc_comb_token_kernel<<<qc::grid_for(static_cast<size_t>(tokens), 128), 128>>>(
            dm, ds, db, dc, tokens, eps, iterations);
        QC_SYNC();
        ok &= qc::compare(qc::d2h(dc, ref.size()), to_ref(ref),
                          qc::Tol::fp32().with_elementwise(2e-5, 2e-6))
                  .report("dsv4_hc_comb");
        ++checks;
        qc::dfree(dm, ds, db, dc);
    }

    {
        constexpr long long tokens = 17;
        constexpr long long embedding = 37;
        auto x = rng.uniforms(static_cast<size_t>(tokens * kConnections * embedding), -0.5f, 0.5f);
        auto weights = rng.uniforms(static_cast<size_t>(tokens * kConnections), -0.5f, 0.5f);
        const auto ref = dsv4_hc_pre_ref(x, weights, tokens, embedding);
        float *dx = qc::dnew(x);
        float *dw = qc::dnew(weights);
        float *do_ = qc::dzero<float>(ref.size());
        dsv4_hc_pre_kernel<<<qc::grid_for(ref.size(), 256), 256>>>(
            dx, dw, do_, ref.size(), embedding);
        QC_SYNC();
        ok &= qc::compare(qc::d2h(do_, ref.size()), to_ref(ref), qc::Tol::fp32())
                  .report("dsv4_hc_pre");
        ++checks;
        qc::dfree(dx, dw, do_);
    }

    {
        constexpr long long tokens = 13;
        constexpr long long embedding = 29;
        auto x = rng.uniforms(static_cast<size_t>(tokens * embedding), -0.5f, 0.5f);
        auto residual = rng.uniforms(static_cast<size_t>(tokens * kConnections * embedding), -0.5f, 0.5f);
        auto post = rng.uniforms(static_cast<size_t>(tokens * kConnections), -0.5f, 0.5f);
        auto comb = rng.uniforms(static_cast<size_t>(tokens * 16), 0.0f, 1.0f);
        const auto ref = dsv4_hc_post_ref(x, residual, post, comb, tokens,
                                          embedding);
        float *dx = qc::dnew(x);
        float *dr = qc::dnew(residual);
        float *dp = qc::dnew(post);
        float *dc = qc::dnew(comb);
        float *do_ = qc::dzero<float>(ref.size());
        dsv4_hc_post_kernel<<<qc::grid_for(ref.size(), 256), 256>>>(
            dx, dr, dp, dc, do_, ref.size(), embedding);
        QC_SYNC();
        ok &= qc::compare(qc::d2h(do_, ref.size()), to_ref(ref), qc::Tol::fp32())
                  .report("dsv4_hc_post");
        ++checks;
        qc::dfree(dx, dr, dp, dc, do_);
    }

    std::printf("Phase 9 correctness checks: %d\n", checks);
    return ok;
}

void run_benchmarks() {
    std::printf("\n== Phase 9 benchmarks ==\n");
    std::printf("   Timing note: medians are per launch; fast kernels use inner repeats.\n");
    qc::Rng rng(0x9999);

    auto bench_backward = [&](const char *label, long long batch, long long heads,
                              long long sequence, long long dim) {
        const size_t tensor_count = static_cast<size_t>(batch * heads * sequence * dim);
        const size_t decay_count = static_cast<size_t>(batch * heads * sequence);
        auto c = rng.uniforms(tensor_count, -0.20f, 0.20f);
        auto b = rng.uniforms(tensor_count, -0.20f, 0.20f);
        auto x = rng.uniforms(tensor_count, -0.20f, 0.20f);
        auto gy = rng.uniforms(tensor_count, -0.20f, 0.20f);
        std::vector<float> cl(decay_count);
        make_cumulative_log(cl, rng, batch, heads, sequence);
        float *dc = qc::dzero<float>(tensor_count);
        float *db = qc::dzero<float>(tensor_count);
        float *dx = qc::dzero<float>(tensor_count);
        float *dcl = qc::dzero<float>(decay_count);
        float *dc_in = qc::dnew(c);
        float *db_in = qc::dnew(b);
        float *dx_in = qc::dnew(x);
        float *dcl_in = qc::dnew(cl);
        float *dy = qc::dnew(gy);
        const double pairs = double(batch * heads) * double(sequence) *
                             double(sequence + 1) * 0.5;
        const double flops = pairs * double(dim) * 8.0;
        const auto scalar = bench_per_launch([&] {
            mamba2_backward_scalar_kernel<<<batch * heads, 1>>>(
                dc_in, db_in, dx_in, dcl_in, dy, dc, db, dx, dcl, batch,
                heads, sequence, dim);
        }, 3, 12);
        const auto direct = bench_per_launch([&] {
            launch_mamba2_backward_candidate(dc, db, dx, dcl, dc_in, db_in,
                                             dx_in, dcl_in, dy, batch, heads,
                                             sequence, dim);
        }, 10, 30);
        report_pair(label, scalar, direct, flops);
        qc::dfree(dc, db, dx, dcl, dc_in, db_in, dx_in, dcl_in, dy);
    };

    bench_backward("mamba2_backward", 4, 4, 64, 32);
    bench_backward("ssd_chunked_backward", 2, 4, 48, 32);

    {
        constexpr long long batch = 512;
        constexpr long long heads = 8;
        constexpr long long dim = 64;
        const size_t state_count = static_cast<size_t>(batch * heads * dim * dim);
        const size_t vector_count = static_cast<size_t>(batch * heads * dim);
        auto state = rng.uniforms(state_count, -0.20f, 0.20f);
        auto alpha = rng.uniforms(static_cast<size_t>(batch * heads), 0.70f, 0.98f);
        auto x = rng.uniforms(vector_count, -0.20f, 0.20f);
        auto k = rng.uniforms(vector_count, -0.20f, 0.20f);
        auto q = rng.uniforms(vector_count, -0.20f, 0.20f);
        float *ds = qc::dnew(state);
        float *da = qc::dnew(alpha);
        float *dx = qc::dnew(x);
        float *dk = qc::dnew(k);
        float *dq = qc::dnew(q);
        float *dy = qc::dzero<float>(vector_count);
        float *dns = qc::dzero<float>(state_count);
        const double flops = double(batch * heads * dim * dim) * 4.0;
        const auto scalar = bench_per_launch([&] {
            ssd_decode_scalar_kernel<<<batch * heads, 1>>>(
                ds, da, dx, dk, dq, dy, dns, batch, heads, dim);
        }, 5, 20);
        const auto wave = bench_per_launch([&] {
            ssd_decode_wave_kernel<<<dim3(dim, batch * heads), qc::kWave>>>(
                ds, da, dx, dk, dq, dy, dns, batch, heads, dim);
        }, 10, 40, 32);
        report_pair("ssd_decode", scalar, wave, flops);
        qc::dfree(ds, da, dx, dk, dq, dy, dns);
    }

    {
        constexpr long long tokens = 65536;
        constexpr float eps = 1e-4f;
        constexpr int iterations = 3;
        auto mixes = rng.uniforms(static_cast<size_t>(tokens * kMixSize), -0.8f, 0.8f);
        auto scale = rng.uniforms(4, 0.5f, 1.5f);
        auto base = rng.uniforms(kMixSize, -0.4f, 0.4f);
        float *dm = qc::dnew(mixes);
        float *ds = qc::dnew(scale);
        float *db = qc::dnew(base);
        float *dc = qc::dzero<float>(static_cast<size_t>(tokens * 16));
        const double bytes = double(tokens * (kMixSize + 16) + kMixSize + 4) * sizeof(float);
        const auto scalar = bench_per_launch([&] {
            dsv4_hc_comb_scalar_kernel<<<1, 1>>>(dm, ds, db, dc, tokens, eps,
                                                 iterations);
        }, 2, 12);
        const auto token = bench_per_launch([&] {
            dsv4_hc_comb_token_kernel<<<qc::grid_for(static_cast<size_t>(tokens), 128), 128>>>(
                dm, ds, db, dc, tokens, eps, iterations);
        }, 10, 40, 16);
        report_pair("dsv4_hc_comb", scalar, token, bytes, false);
        qc::dfree(dm, ds, db, dc);
    }

    {
        constexpr long long tokens = 65536;
        constexpr long long embedding = 256;
        const size_t in_count = static_cast<size_t>(tokens * kConnections * embedding);
        const size_t out_count = static_cast<size_t>(tokens * embedding);
        auto x = rng.uniforms(in_count, -0.5f, 0.5f);
        auto weights = rng.uniforms(static_cast<size_t>(tokens * kConnections), -0.5f, 0.5f);
        float *dx = qc::dnew(x);
        float *dw = qc::dnew(weights);
        float *do_ = qc::dzero<float>(out_count);
        const double bytes = double(in_count + weights.size() + out_count) * sizeof(float);
        const auto scalar = bench_per_launch([&] {
            dsv4_hc_pre_scalar_kernel<<<tokens, 1>>>(dx, dw, do_, tokens,
                                                     embedding);
        }, 5, 20);
        const auto parallel = bench_per_launch([&] {
            dsv4_hc_pre_kernel<<<qc::grid_for(out_count, 256), 256>>>(
                dx, dw, do_, out_count, embedding);
        }, 10, 40, 128);
        report_pair("dsv4_hc_pre", scalar, parallel, bytes, false);
        qc::dfree(dx, dw, do_);
    }

    {
        constexpr long long tokens = 32768;
        constexpr long long embedding = 256;
        const size_t x_count = static_cast<size_t>(tokens * embedding);
        const size_t out_count = static_cast<size_t>(tokens * kConnections * embedding);
        auto x = rng.uniforms(x_count, -0.5f, 0.5f);
        auto residual = rng.uniforms(out_count, -0.5f, 0.5f);
        auto post = rng.uniforms(static_cast<size_t>(tokens * kConnections), -0.5f, 0.5f);
        auto comb = rng.uniforms(static_cast<size_t>(tokens * 16), 0.0f, 1.0f);
        float *dx = qc::dnew(x);
        float *dr = qc::dnew(residual);
        float *dp = qc::dnew(post);
        float *dc = qc::dnew(comb);
        float *do_ = qc::dzero<float>(out_count);
        const double bytes = double(x_count + residual.size() + post.size() +
                                    comb.size() + out_count) *
                             sizeof(float);
        const auto scalar = bench_per_launch([&] {
            dsv4_hc_post_scalar_kernel<<<tokens * kConnections, 1>>>(
                dx, dr, dp, dc, do_, tokens, embedding);
        }, 5, 20);
        const auto parallel = bench_per_launch([&] {
            dsv4_hc_post_kernel<<<qc::grid_for(out_count, 256), 256>>>(
                dx, dr, dp, dc, do_, out_count, embedding);
        }, 10, 40);
        report_pair("dsv4_hc_post", scalar, parallel, bytes, false);
        qc::dfree(dx, dr, dp, dc, do_);
    }
}

}  // namespace

int main(int argc, char **argv) {
    qc::print_environment("phase9_ssm");
    const bool ok = run_correctness();
    if (qc::bench_requested(argc, argv)) run_benchmarks();
    return qc::finish(ok);
}
