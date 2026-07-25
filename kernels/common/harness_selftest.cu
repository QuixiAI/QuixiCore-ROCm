/**
 * @file
 * @brief Self-test for `cdna3_harness.cuh`.
 *
 * The harness is shared by every CDNA3 kernel port, so a bug in its wave64
 * reductions or its oracle comparison would silently corrupt every downstream
 * correctness verdict. This checks the helpers themselves against hand-computed
 * answers: reductions vs. an fp64 host sum, the argmax tie rule, the scan, the
 * dtype round-trips, and -- importantly -- that `compare()` actually FAILS on
 * data it should reject.
 *
 * Build and run: `make -C kernels/common test`
 */
#include "cdna3_harness.cuh"

// --------------------------------------------------------------------------
// Device kernels exercising the reduction helpers, one wavefront per row.
// --------------------------------------------------------------------------

__global__ void k_reduce_sum(const float *in, float *out, int cols) {
    const int lane = threadIdx.x & (qc::kWave - 1);
    const int row = blockIdx.x;
    float acc = 0.f;
    for (int c = lane; c < cols; c += qc::kWave) acc += in[(size_t)row * cols + c];
    acc = qc::wave_reduce_sum(acc);
    if (lane == 0) out[row] = acc;
}

__global__ void k_reduce_max(const float *in, float *out, int cols) {
    const int lane = threadIdx.x & (qc::kWave - 1);
    const int row = blockIdx.x;
    float acc = -INFINITY;
    for (int c = lane; c < cols; c += qc::kWave)
        acc = fmaxf(acc, in[(size_t)row * cols + c]);
    acc = qc::wave_reduce_max(acc);
    if (lane == 0) out[row] = acc;
}

// Each lane owns one column; ties must resolve to the lowest column index.
__global__ void k_argmax(const float *in, int *out, int cols) {
    const int lane = threadIdx.x & (qc::kWave - 1);
    const int row = blockIdx.x;
    float best = -INFINITY;
    int best_i = 0;
    for (int c = lane; c < cols; c += qc::kWave) {
        const float v = in[(size_t)row * cols + c];
        if (v > best || (v == best && c < best_i)) { best = v; best_i = c; }
    }
    qc::wave_reduce_argmax(best, best_i);
    if (lane == 0) out[row] = best_i;
}

__global__ void k_scan(const float *in, float *out) {
    const int lane = threadIdx.x & (qc::kWave - 1);
    out[lane] = qc::wave_inclusive_scan(in[lane]);
}

// Uses all 4 wavefronts of a 256-thread block.
__global__ void k_block_sum(const float *in, float *out, int cols) {
    __shared__ float scratch[qc::kWavesPerBlock];
    const int row = blockIdx.x;
    float acc = 0.f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x)
        acc += in[(size_t)row * cols + c];
    const float total = qc::block_reduce_sum(acc, scratch);
    if (threadIdx.x == 0) out[row] = total;
}

// --------------------------------------------------------------------------

static bool test_reductions() {
    const int rows = 37, cols = 501;  // deliberately not multiples of 64
    qc::Rng rng(7);
    auto host = rng.normals((size_t)rows * cols);

    auto d_in = qc::dnew(host);
    auto d_sum = qc::dzero<float>(rows);
    auto d_max = qc::dzero<float>(rows);
    auto d_blk = qc::dzero<float>(rows);
    auto d_arg = qc::dzero<int>(rows);

    k_reduce_sum<<<rows, qc::kWave>>>(d_in, d_sum, cols);
    k_reduce_max<<<rows, qc::kWave>>>(d_in, d_max, cols);
    k_argmax<<<rows, qc::kWave>>>(d_in, d_arg, cols);
    k_block_sum<<<rows, qc::kThreads>>>(d_in, d_blk, cols);
    QC_SYNC();

    std::vector<double> ref_sum(rows), ref_max(rows);
    std::vector<int> ref_arg(rows);
    for (int r = 0; r < rows; ++r) {
        double s = 0.0, m = -INFINITY;
        int mi = 0;
        for (int c = 0; c < cols; ++c) {
            const double v = host[(size_t)r * cols + c];
            s += v;
            if (v > m) { m = v; mi = c; }
        }
        ref_sum[r] = s;
        ref_max[r] = m;
        ref_arg[r] = mi;
    }

    bool ok = true;
    // Float accumulation over 501 terms in a different order than the oracle:
    // relax the elementwise bound but hold the aggregate bounds tight.
    ok &= qc::compare(qc::d2h(d_sum, rows), ref_sum,
                      qc::Tol::fp32().with_elementwise(1e-5, 1e-4))
              .report("wave_reduce_sum (37x501)");
    ok &= qc::compare(qc::d2h(d_max, rows), ref_max, qc::Tol::exact())
              .report("wave_reduce_max (37x501)");
    ok &= qc::compare(qc::d2h(d_arg, rows), ref_arg, qc::Tol::exact())
              .report("wave_reduce_argmax (37x501)");
    ok &= qc::compare(qc::d2h(d_blk, rows), ref_sum,
                      qc::Tol::fp32().with_elementwise(1e-5, 1e-4))
              .report("block_reduce_sum (4 waves)");

    qc::dfree(d_in, d_sum, d_max, d_blk);
    qc::dfree(d_arg);
    return ok;
}

// The argmax tie rule is contractual (LM-head lower-token tie breaking), so it
// gets a dedicated all-equal input rather than relying on random data.
static bool test_argmax_ties() {
    const int cols = 64;
    std::vector<float> host(cols, 1.0f);  // every element ties
    auto d_in = qc::dnew(host);
    auto d_arg = qc::dzero<int>(1);
    k_argmax<<<1, qc::kWave>>>(d_in, d_arg, cols);
    QC_SYNC();
    const int got = qc::d2h(d_arg, 1)[0];
    const bool ok = got == 0;
    std::printf("  %-52s [exact] all-tie -> index %d (want 0)  %s\n",
                "wave_reduce_argmax lower-index tie rule", got, ok ? "PASS" : "FAIL");
    qc::dfree(d_in);
    qc::dfree(d_arg);
    return ok;
}

static bool test_scan() {
    qc::Rng rng(11);
    auto host = rng.uniforms(qc::kWave, 0.f, 1.f);
    auto d_in = qc::dnew(host);
    auto d_out = qc::dzero<float>(qc::kWave);
    k_scan<<<1, qc::kWave>>>(d_in, d_out);
    QC_SYNC();
    std::vector<double> ref(qc::kWave);
    double acc = 0.0;
    for (int i = 0; i < qc::kWave; ++i) { acc += host[i]; ref[i] = acc; }
    const bool ok = qc::compare(qc::d2h(d_out, qc::kWave), ref,
                                qc::Tol::fp32().with_elementwise(1e-5, 1e-5))
                        .report("wave_inclusive_scan (64 lanes)");
    qc::dfree(d_in, d_out);
    return ok;
}

static bool test_dtype_roundtrip() {
    const std::vector<float> src = {0.f, 1.f, -1.f, 0.5f, 3.14159f, 65504.f, -1e-4f};
    auto h16 = qc::to_storage<__half>(src);
    auto b16 = qc::to_storage<__hip_bfloat16>(src);
    std::vector<double> ref16(src.size()), refb16(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        ref16[i] = qc::round_fp16(src[i]);
        refb16[i] = qc::round_bf16(src[i]);
    }
    bool ok = true;
    // to_storage + to_double must round-trip exactly through the same rounding
    // that round_fp16/round_bf16 apply, or storage-rounding contracts break.
    ok &= qc::compare(h16, ref16, qc::Tol::exact()).report("fp16 storage round-trip");
    ok &= qc::compare(b16, refb16, qc::Tol::exact()).report("bf16 storage round-trip");
    return ok;
}

// compare() is only useful if it rejects bad data. These cases must FAIL.
static bool test_compare_rejects() {
    bool ok = true;
    const std::vector<float> good = {1.f, 2.f, 3.f, 4.f};

    const std::vector<double> one_bad = {1.0, 2.0, 3.5, 4.0};
    const bool r1 = qc::compare(good, one_bad, qc::Tol::fp32()).pass();

    std::vector<double> drifted = {1.0, 2.0, 3.0, 4.0};
    for (auto &v : drifted) v *= 1.02;  // 2% systematic scale error
    const bool r2 = qc::compare(good, drifted, qc::Tol::fp16()).pass();

    const std::vector<float> nan_out = {1.f, NAN, 3.f, 4.f};
    const std::vector<double> clean = {1.0, 2.0, 3.0, 4.0};
    const bool r3 = qc::compare(nan_out, clean, qc::Tol::quantized()).pass();

    const std::vector<double> wrong_size = {1.0, 2.0};
    const bool r4 = qc::compare(good, wrong_size, qc::Tol::fp32()).pass();

    ok &= !r1 && !r2 && !r3 && !r4;
    std::printf("  %-52s single-bad=%d drift=%d nan=%d size=%d  %s\n",
                "compare() rejects bad data", (int)r1, (int)r2, (int)r3, (int)r4,
                ok ? "PASS" : "FAIL");

    // ...and accepts data it should: identical vectors, and two all-zero
    // vectors (cosine is undefined there and must not be treated as failure).
    const std::vector<float> zeros4(4, 0.f);
    const std::vector<double> zeros4d(4, 0.0);
    const bool a1 = qc::compare(good, clean, qc::Tol::fp32()).pass();
    const bool a2 = qc::compare(zeros4, zeros4d, qc::Tol::fp32()).pass();
    const bool acc = a1 && a2;
    std::printf("  %-52s identical=%d all-zero=%d  %s\n",
                "compare() accepts good data", (int)a1, (int)a2, acc ? "PASS" : "FAIL");
    return ok && acc;
}

static bool test_bench() {
    const int n = 1 << 22;
    auto d_in = qc::dzero<float>(n);
    auto d_out = qc::dzero<float>(1024);
    auto b = qc::bench([&] { k_reduce_sum<<<1024, qc::kWave>>>(d_in, d_out, n / 1024); },
                       3, 11);
    const bool ok = b.median_ms > 0.0 && b.min_ms <= b.median_ms &&
                    b.median_ms <= b.max_ms && b.iters == 11;
    b.report_bandwidth("bench() sanity", (double)n * sizeof(float));
    std::printf("  %-52s ordering min<=med<=max  %s\n", "bench() statistics",
                ok ? "PASS" : "FAIL");
    qc::dfree(d_in, d_out);
    return ok;
}

int main(int argc, char **argv) {
    (void)argc;
    (void)argv;
    qc::print_environment("cdna3_harness.cuh self-test");
    bool ok = true;
    ok &= test_reductions();
    ok &= test_argmax_ties();
    ok &= test_scan();
    ok &= test_dtype_roundtrip();
    ok &= test_compare_rejects();
    ok &= test_bench();
    return qc::finish(ok);
}
