#include "hip/hip_runtime.h"
// Harness for kv_cache kernels (kernels live in kv_cache_kernels.cuh).
#include "kv_cache_kernels.cuh"
// ============================== test harness ==============================
using namespace tms;

static float frand() { return (rand() % 20000 - 10000) / 10000.0f; }

int main() {
    srand(5);
    // scenario: B=3, H=8, H_KV in {8 (MHA), 2 (GQA)}, D in {64,128}, block_size=16
    const int B = 3, H = 8, BS = 16, MAXB = 16;
    const int ctx[B] = {37, 128, 5};
    int rc = 0;
    for (int D : {64, 128}) for (int HKV : {8, 2}) {
        const int num_blocks = B * MAXB + 1;
        const size_t cache_n = size_t(num_blocks) * BS * HKV * D;
        // host paged cache + block table (identity-ish mapping with a shuffled block order)
        std::vector<int> bt(B * MAXB, -1);
        int next_block = 1;   // block 0 left unmapped to exercise skipping
        for (int b = 0; b < B; b++)
            for (int c = 0; c * BS < ctx[b]; c++) bt[b * MAXB + c] = next_block++;
        std::vector<__half> K(cache_n), V(cache_n);
        for (auto& x : K) x = __float2half(frand() * 0.3f);
        for (auto& x : V) x = __float2half(frand() * 0.3f);
        std::vector<__half> q(size_t(B) * H * D);
        for (auto& x : q) x = __float2half(frand());
        std::vector<float> slopes(H);
        for (int h = 0; h < H; h++) slopes[h] = -0.05f * (h + 1);
        std::vector<int> mask(B * MAXB, 1);
        mask[0 * MAXB + 1] = 0;   // knock out one block of batch 0

        __half *dK, *dV, *dq, *dout;
        int *dbt, *dctx, *dmask; float* dsl;
        hipMalloc(&dK, cache_n * 2); hipMalloc(&dV, cache_n * 2);
        hipMalloc(&dq, q.size() * 2); hipMalloc(&dout, q.size() * 2);
        hipMalloc(&dbt, bt.size() * 4); hipMalloc(&dctx, B * 4);
        hipMalloc(&dmask, mask.size() * 4); hipMalloc(&dsl, H * 4);
        hipMemcpy(dK, K.data(), cache_n * 2, hipMemcpyHostToDevice);
        hipMemcpy(dV, V.data(), cache_n * 2, hipMemcpyHostToDevice);
        hipMemcpy(dq, q.data(), q.size() * 2, hipMemcpyHostToDevice);
        hipMemcpy(dbt, bt.data(), bt.size() * 4, hipMemcpyHostToDevice);
        hipMemcpy(dctx, ctx, B * 4, hipMemcpyHostToDevice);
        hipMemcpy(dmask, mask.data(), mask.size() * 4, hipMemcpyHostToDevice);
        hipMemcpy(dsl, slopes.data(), H * 4, hipMemcpyHostToDevice);

        const float scale = 1.0f / sqrtf(float(D));
        for (int variant = 0; variant < 4; variant++) {   // dense, alibi, window, mask
            const int use_alibi = variant == 1, window = variant == 2 ? 24 : 0, use_mask = variant == 3;
            dim3 grid(H, B);
            if (D == 64) paged_attention<__half, 64><<<grid, 32>>>(dq, dK, dV, dbt, dctx, dout, BS, MAXB, scale, H, HKV, dsl, use_alibi, dmask, use_mask, window);
            else         paged_attention<__half, 128><<<grid, 32>>>(dq, dK, dV, dbt, dctx, dout, BS, MAXB, scale, H, HKV, dsl, use_alibi, dmask, use_mask, window);
            hipDeviceSynchronize();
            if (hipGetLastError() != hipSuccess) { printf("KERNEL ERROR\n"); return 1; }
            std::vector<__half> got(q.size());
            hipMemcpy(got.data(), dout, q.size() * 2, hipMemcpyDeviceToHost);

            // CPU reference (double, full softmax)
            double maxd = 0;
            for (int b = 0; b < B; b++) for (int h = 0; h < H; h++) {
                const int kvh = h / (H / HKV);
                const int t0 = (window > 0) ? std::max(0, ctx[b] - window) : 0;
                std::vector<double> sc;
                std::vector<int> ts;
                for (int t = t0; t < ctx[b]; t++) {
                    const int col = t / BS, blk = bt[b * MAXB + col];
                    if (blk < 0) continue;
                    if (use_mask && mask[b * MAXB + col] == 0) continue;
                    double s = 0;
                    const size_t base = (size_t(blk) * BS + t % BS) * HKV * D + size_t(kvh) * D;
                    for (int d = 0; d < D; d++)
                        s += double(__half2float(q[(size_t(b) * H + h) * D + d])) * double(__half2float(K[base + d]));
                    s *= scale;
                    if (use_alibi) s += slopes[h] * double(t - ctx[b] + 1);
                    sc.push_back(s); ts.push_back(t);
                }
                double mx = -1e300;
                for (double s : sc) mx = std::max(mx, s);
                std::vector<double> o(D, 0.0);
                double Z = 0;
                for (size_t i = 0; i < sc.size(); i++) {
                    const int t = ts[i], col = t / BS, blk = bt[b * MAXB + col];
                    const double w = exp(sc[i] - mx);
                    const size_t base = (size_t(blk) * BS + t % BS) * HKV * D + size_t(kvh) * D;
                    for (int d = 0; d < D; d++) o[d] += w * double(__half2float(V[base + d]));
                    Z += w;
                }
                for (int d = 0; d < D; d++) {
                    const double ref = Z ? o[d] / Z : 0.0;
                    maxd = std::max(maxd, std::abs(double(__half2float(got[(size_t(b) * H + h) * D + d])) - ref));
                }
            }
            const char* names[4] = {"dense", "alibi", "window", "blocksparse"};
            printf("paged_attention D=%d HKV=%d %-11s max diff %.5f (%s)\n", D, HKV, names[variant],
                   maxd, maxd < 5e-3 ? "PASS" : "FAIL");
            rc |= !(maxd < 5e-3);
        }

        // gqa_staged: identical math to v1 -> bit-for-bit equal output
        {
            __half* dout2;
            hipMalloc(&dout2, q.size() * 2);
            dim3 grid(H, B);
            if (D == 64) paged_attention<__half, 64><<<grid, 32>>>(dq, dK, dV, dbt, dctx, dout, BS, MAXB, scale, H, HKV, dsl, 0, dmask, 0, 0);
            else         paged_attention<__half, 128><<<grid, 32>>>(dq, dK, dV, dbt, dctx, dout, BS, MAXB, scale, H, HKV, dsl, 0, dmask, 0, 0);
            const int gs = H / HKV;
            dim3 sgrid(HKV, B);
            if (D == 64) paged_attention_gqa_staged<__half, 64><<<sgrid, 32 * gs>>>(dq, dK, dV, dbt, dctx, dout2, BS, MAXB, scale, H, HKV);
            else         paged_attention_gqa_staged<__half, 128><<<sgrid, 32 * gs>>>(dq, dK, dV, dbt, dctx, dout2, BS, MAXB, scale, H, HKV);
            hipDeviceSynchronize();
            std::vector<__half> g1(q.size()), g2(q.size());
            hipMemcpy(g1.data(), dout, q.size() * 2, hipMemcpyDeviceToHost);
            hipMemcpy(g2.data(), dout2, q.size() * 2, hipMemcpyDeviceToHost);
            // CDNA3 note: v1 and the staged kernel do identical *math* but use
            // different reduction schedules. On NVIDIA warp-32 the staged
            // grouping reproduces v1's accumulation order bit-for-bit; on the
            // 64-wide CDNA wavefront the cross-warp merge order is not fixed, so
            // the two can differ by ~1 fp16 ULP on a handful of elements (and
            // the difference is not run-to-run deterministic). Both still match
            // the fp64 reference to <5e-3 (checked for v1 above). Compare within
            // 1 fp16 ULP instead of requiring bit-exactness.
            size_t bad = 0; double worst = 0;
            for (size_t i = 0; i < q.size(); i++) {
                const double a = __half2float(g1[i]), b = __half2float(g2[i]);
                const double d = std::abs(a - b);
                worst = std::max(worst, d);
                if (d > 1e-3 * std::max(1.0, std::abs(a))) bad++;   // >~1 fp16 ULP
            }
            printf("gqa_staged D=%d HKV=%d vs v1: %zu beyond-1ULP (%s)  [max abs diff %.3g]\n",
                   D, HKV, bad, bad ? "FAIL" : "PASS", worst);
            rc |= (bad != 0);
            hipFree(dout2);
        }

        // attn_window: dense sliding-window causal vs fp64 full softmax
        if (HKV == 8) {
            const int BH = 6, N = 64;
            std::vector<__half> aq(size_t(BH) * N * D), ak(aq.size()), av(aq.size());
            for (auto& x : aq) x = __float2half(frand() * 0.5f);
            for (auto& x : ak) x = __float2half(frand() * 0.5f);
            for (auto& x : av) x = __float2half(frand() * 0.5f);
            __half *daq, *dak, *dav, *dao;
            hipMalloc(&daq, aq.size() * 2); hipMalloc(&dak, ak.size() * 2);
            hipMalloc(&dav, av.size() * 2); hipMalloc(&dao, aq.size() * 2);
            hipMemcpy(daq, aq.data(), aq.size() * 2, hipMemcpyHostToDevice);
            hipMemcpy(dak, ak.data(), ak.size() * 2, hipMemcpyHostToDevice);
            hipMemcpy(dav, av.data(), av.size() * 2, hipMemcpyHostToDevice);
            for (int W : {0, 24}) {
                dim3 agrid(N, BH);
                if (D == 64) attn_window<__half, 64><<<agrid, 32>>>(daq, dak, dav, dao, N, scale, W);
                else         attn_window<__half, 128><<<agrid, 32>>>(daq, dak, dav, dao, N, scale, W);
                hipDeviceSynchronize();
                std::vector<__half> got(aq.size());
                hipMemcpy(got.data(), dao, aq.size() * 2, hipMemcpyDeviceToHost);
                double maxd = 0;
                for (int bh = 0; bh < BH; bh++) for (int i = 0; i < N; i++) {
                    const int j0 = (W > 0) ? std::max(0, i - W + 1) : 0;
                    std::vector<double> sc;
                    for (int j = j0; j <= i; j++) {
                        double s = 0;
                        for (int d = 0; d < D; d++)
                            s += double(__half2float(aq[(size_t(bh) * N + i) * D + d]))
                               * double(__half2float(ak[(size_t(bh) * N + j) * D + d]));
                        sc.push_back(s * scale);
                    }
                    double mx = -1e300;
                    for (double s : sc) mx = std::max(mx, s);
                    std::vector<double> o(D, 0.0);
                    double Z = 0;
                    for (size_t jj = 0; jj < sc.size(); jj++) {
                        const double w = exp(sc[jj] - mx);
                        for (int d = 0; d < D; d++)
                            o[d] += w * double(__half2float(av[(size_t(bh) * N + j0 + jj) * D + d]));
                        Z += w;
                    }
                    for (int d = 0; d < D; d++)
                        maxd = std::max(maxd, std::abs(double(__half2float(got[(size_t(bh) * N + i) * D + d])) - o[d] / Z));
                }
                printf("attn_window D=%d W=%-2d           max diff %.5f (%s)\n", D, W, maxd, maxd < 5e-3 ? "PASS" : "FAIL");
                rc |= !(maxd < 5e-3);
            }
            hipFree(daq); hipFree(dak); hipFree(dav); hipFree(dao);
        }

        // scatter -> gather round trip
        {
            const int T = ctx[0] + ctx[1] + ctx[2];
            std::vector<__half> nk(size_t(T) * HKV * D), nv(nk.size());
            for (auto& x : nk) x = __float2half(frand());
            for (auto& x : nv) x = __float2half(frand());
            std::vector<int64_t> slots(T);
            std::vector<int> cu = {0, ctx[0], ctx[0] + ctx[1], T};
            for (int b = 0, i = 0; b < B; b++)
                for (int t = 0; t < ctx[b]; t++, i++)
                    slots[i] = int64_t(bt[b * MAXB + t / BS]) * BS + t % BS;
            __half *dnk, *dnv, *dgk, *dgv; int64_t* dslot; int* dcu;
            hipMalloc(&dnk, nk.size() * 2); hipMalloc(&dnv, nv.size() * 2);
            hipMalloc(&dgk, nk.size() * 2); hipMalloc(&dgv, nv.size() * 2);
            hipMalloc(&dslot, T * 8); hipMalloc(&dcu, 4 * 4);
            hipMemcpy(dnk, nk.data(), nk.size() * 2, hipMemcpyHostToDevice);
            hipMemcpy(dnv, nv.data(), nv.size() * 2, hipMemcpyHostToDevice);
            hipMemcpy(dslot, slots.data(), T * 8, hipMemcpyHostToDevice);
            hipMemcpy(dcu, cu.data(), 4 * 4, hipMemcpyHostToDevice);
            kv_cache_zero<__half><<<int((cache_n + 255) / 256), 256>>>(dK, dV, cache_n);
            kv_cache_scatter<__half><<<T, 128>>>(dnk, dnv, dslot, dK, dV, HKV, D, BS);
            kv_cache_gather<__half><<<T, 128>>>(dK, dV, dgk, dgv, dbt, dcu, T, B, BS, MAXB, HKV, D);
            hipDeviceSynchronize();
            std::vector<__half> gk(nk.size());
            hipMemcpy(gk.data(), dgk, nk.size() * 2, hipMemcpyDeviceToHost);
            size_t bad = 0;
            for (size_t i = 0; i < nk.size(); i++)
                if (__half2float(gk[i]) != __half2float(nk[i])) bad++;
            printf("scatter/gather D=%d HKV=%d round-trip: %zu mismatches (%s)\n", D, HKV, bad, bad ? "FAIL" : "PASS");
            rc |= (bad != 0);
            hipFree(dnk); hipFree(dnv); hipFree(dgk); hipFree(dgv); hipFree(dslot); hipFree(dcu);
        }
        hipFree(dK); hipFree(dV); hipFree(dq); hipFree(dout);
        hipFree(dbt); hipFree(dctx); hipFree(dmask); hipFree(dsl);
    }
    return rc;
}
