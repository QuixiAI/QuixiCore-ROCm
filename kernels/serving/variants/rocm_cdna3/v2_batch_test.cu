#include "hip/hip_runtime.h"
/**
 * @file
 * @brief Harness for the V2 model-runner batch-prep kernels
 * (v2_batch_kernels.cuh) and the slot-mapping / indexer-metadata pair
 * (slot_mapping_kernels.cuh).
 *
 * These are index arithmetic, not math: every output is an exact integer, so
 * the reference is a host replay and the bar is bitwise equality, not a
 * tolerance. Each case exercises the branchy paths the serving path actually
 * hits -- ragged query lengths, chunked prefill (num_computed > 0), negative
 * idx_mapping rows, zero-length rows, context-parallel interleave, and the
 * CUDA-graph padding tails that must stay fixed-size across replay.
 *
 * Build: make v2_batch_test.out
 * Run:   HIP_VISIBLE_DEVICES=0 ./v2_batch_test.out [--bench]
 */
#include "slot_mapping_kernels.cuh"
#include "v2_batch_kernels.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace tms;
static int g_fail = 0;
#define CK(x)                                                              \
    do {                                                                   \
        hipError_t e = (x);                                                \
        if (e) {                                                           \
            printf("HIP %s @%d\n", hipGetErrorString(e), __LINE__);        \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

template <typename T>
static T* dnew(const std::vector<T>& h) {
    T* d;
    CK(hipMalloc(&d, h.size() * sizeof(T)));
    CK(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice));
    return d;
}
template <typename T>
static std::vector<T> d2h(const T* d, size_t n) {
    std::vector<T> h(n);
    CK(hipMemcpy(h.data(), d, n * sizeof(T), hipMemcpyDeviceToHost));
    return h;
}
template <typename T>
static void rep(const char* nm, const std::vector<T>& got,
                const std::vector<T>& want) {
    long mm = 0;
    for (size_t i = 0; i < want.size(); ++i) mm += (got[i] != want[i]);
    printf("%-34s %s (%ld/%zu mismatch)\n", nm, mm ? "FAIL" : "PASS", mm,
           want.size());
    if (mm) ++g_fail;
}

static std::mt19937 rng(31);

// Ragged batch shared by the cases below: chunked prefill, pure decode, and a
// long prefill in one launch, which is what the serving path sees.
static const int NREQ = 6;
static const int MAXREQ = 16;
static const int MAXTOK = 512;
static const int QLEN[NREQ] = {7, 1, 1, 13, 1, 4};
static const int IDXMAP[NREQ] = {3, 0, 5, 1, 9, 2};

static std::vector<int> cu_of(const int* lens, int n) {
    std::vector<int> cu(n + 1, 0);
    for (int i = 0; i < n; ++i) cu[i + 1] = cu[i] + lens[i];
    return cu;
}

// ---------------------------------------------------------------- slot mapping
static void test_compute_slot_mapping(int cp_world, int cp_rank,
                                      int cp_interleave) {
    const int block_size = 16, bt_stride = 64;
    const long pad_id = -1;
    auto qsl = cu_of(QLEN, NREQ);
    const long num_tokens = qsl[NREQ];
    const long max_num_tokens = num_tokens + 11;  // graph padding tail

    std::vector<long> positions(num_tokens);
    std::uniform_int_distribution<long> pd(0, 4000);
    for (auto& p : positions) p = pd(rng);

    std::vector<int> bt((size_t)MAXREQ * bt_stride);
    std::uniform_int_distribution<int> bd(0, 900);
    for (auto& b : bt) b = bd(rng);

    std::vector<long> sm(max_num_tokens, 12345);
    auto *dq = dnew(qsl), *dbt = dnew(bt);
    auto* dp = dnew(positions);
    auto* dsm = dnew(sm);

    compute_slot_mapping<<<NREQ + 1, 256>>>(
        num_tokens, max_num_tokens, dq, dp, dbt, bt_stride, block_size, dsm,
        block_size, 1, cp_world, cp_rank, cp_interleave, pad_id);
    CK(hipDeviceSynchronize());

    std::vector<long> want(max_num_tokens, 12345);
    for (long i = num_tokens; i < max_num_tokens; ++i) want[i] = pad_id;
    const long vbs = (long)block_size * cp_world;
    for (int r = 0; r < NREQ; ++r) {
        for (long t = qsl[r]; t < qsl[r + 1]; ++t) {
            const long pos = positions[t];
            const long vbi = pos / vbs, vbo = pos - vbi * vbs;
            const bool local = ((vbo / cp_interleave) % cp_world) == cp_rank;
            const long lbo =
                (vbo / ((long)cp_world * cp_interleave)) * cp_interleave +
                (vbo % cp_interleave);
            long slot = pad_id;
            if (local) {
                const long bi = vbi * 1 + lbo / block_size;
                slot = (long)bt[(size_t)r * bt_stride + bi] * block_size +
                       (lbo % block_size);
            }
            want[t] = slot;
        }
    }
    char nm[64];
    snprintf(nm, sizeof(nm), "compute_slot_mapping cp%d", cp_world);
    rep(nm, d2h(dsm, max_num_tokens), want);
    CK(hipFree(dq));
    CK(hipFree(dbt));
    CK(hipFree(dp));
    CK(hipFree(dsm));
}

// ------------------------------------------------------------ indexer metadata
static void test_indexer_metadata(int dcp_world, int dcp_rank,
                                  int dcp_interleave) {
    const int ratio = 4;
    auto qsl = cu_of(QLEN, NREQ);
    std::vector<int> useq(NREQ), ccu(NREQ + 1, 0), rowcu(NREQ + 1, 0);
    for (int b = 0; b < NREQ; ++b) {
        useq[b] = QLEN[b] + 40 * (b + 1);
        ccu[b + 1] = ccu[b] + useq[b] / ratio;
        rowcu[b + 1] = rowcu[b] + useq[b] / ratio + 3;
    }
    const int ntok = qsl[NREQ];
    const int qs = 1, qe = ntok - 1;  // exercise a non-trivial query slice

    std::vector<int> t2s(ccu[NREQ], -7), ks(ntok, -7), ke(ntok, -7);
    auto *dq = dnew(qsl), *du = dnew(useq), *dc = dnew(ccu), *dr = dnew(rowcu);
    auto *dt = dnew(t2s), *dks = dnew(ks), *dke = dnew(ke);

    indexer_metadata<<<NREQ, 256>>>(dq, du, dc, dr, dt, dks, dke, qs, qe,
                                    dcp_rank, dcp_world, dcp_interleave, ratio);
    CK(hipDeviceSynchronize());

    std::vector<int> wt(ccu[NREQ], -7), wks(ntok, -7), wke(ntok, -7);
    for (int b = 0; b < NREQ; ++b) {
        const int qstart = qsl[b], qlen = qsl[b + 1] - qstart;
        const int start_pos = useq[b] - qlen;
        for (int off = 0; off < qlen; ++off) {
            const int abs_pos = qstart + off;
            if (abs_pos < qs || abs_pos >= qe) continue;
            const int out = abs_pos - qs;
            wks[out] = rowcu[b];
            int len = (start_pos + 1 + off) / ratio;
            if (dcp_world > 1) {
                const int base =
                    (len / dcp_interleave / dcp_world) * dcp_interleave;
                int rem = len - base * dcp_world;
                rem = std::min(std::max(rem - dcp_rank * dcp_interleave, 0),
                               dcp_interleave);
                len = base + rem;
            }
            wke[out] = rowcu[b] + len;
        }
        for (int off = 0; off < ccu[b + 1] - ccu[b]; ++off)
            wt[ccu[b] + off] = b;
    }
    char nm[64];
    snprintf(nm, sizeof(nm), "indexer_metadata dcp%d", dcp_world);
    rep(nm, d2h(dt, wt.size()), wt);
    snprintf(nm, sizeof(nm), "indexer_metadata cu_ks/ke dcp%d", dcp_world);
    std::vector<int> got = d2h(dks, ntok), got2 = d2h(dke, ntok);
    got.insert(got.end(), got2.begin(), got2.end());
    std::vector<int> wantc = wks;
    wantc.insert(wantc.end(), wke.begin(), wke.end());
    rep(nm, got, wantc);
    for (int* p : {dq, du, dc, dr, dt, dks, dke}) CK(hipFree(p));
}

// --------------------------------------------------------------- batch prep
static void test_pos_seq_lens() {
    auto qsl = cu_of(QLEN, NREQ);
    std::vector<int> nct(MAXREQ, 0);
    const int vals[NREQ] = {0, 5, 17, 2, 40, 100};
    for (int i = 0; i < NREQ; ++i) nct[IDXMAP[i]] = vals[i];
    std::vector<int> idx(IDXMAP, IDXMAP + NREQ);

    std::vector<long> pos(qsl[NREQ], -1);
    std::vector<int> sl(MAXREQ, -1);
    auto *dq = dnew(qsl), *di = dnew(idx), *dn = dnew(nct), *dsl = dnew(sl);
    auto* dp = dnew(pos);
    prepare_pos_seq_lens<<<NREQ + 1, 256>>>(dp, dsl, di, dq, dn, MAXREQ);
    CK(hipDeviceSynchronize());

    std::vector<long> wp(qsl[NREQ], -1);
    std::vector<int> ws(MAXREQ, -1);
    for (int r = 0; r < NREQ; ++r) {
        const int s = nct[IDXMAP[r]];
        for (int t = qsl[r]; t < qsl[r + 1]; ++t) wp[t] = s + (t - qsl[r]);
        ws[r] = s + QLEN[r];
    }
    for (int r = NREQ; r < MAXREQ; ++r) ws[r] = 0;  // padding rows
    rep("prepare_pos_seq_lens pos", d2h(dp, wp.size()), wp);
    rep("prepare_pos_seq_lens seq_lens", d2h(dsl, MAXREQ), ws);
    for (int* p : {dq, di, dn, dsl}) CK(hipFree(p));
    CK(hipFree(dp));
}

static void test_prefill_inputs() {
    auto qsl = cu_of(QLEN, NREQ);
    const int comp[NREQ] = {0, 5, 0, 2, 40, 0};
    const int plen[NREQ] = {64, 6, 1, 30, 40, 9};
    std::vector<int> nct(MAXREQ, 0), pl(MAXREQ, 0);
    for (int i = 0; i < NREQ; ++i) {
        nct[IDXMAP[i]] = comp[i];
        pl[IDXMAP[i]] = plen[i];
    }
    std::vector<int> all((size_t)MAXREQ * MAXTOK);
    std::uniform_int_distribution<int> td(0, 50000);
    for (auto& v : all) v = td(rng);
    std::vector<int> idx(IDXMAP, IDXMAP + NREQ), ids(qsl[NREQ], -1),
        nxt(MAXREQ, -1);

    auto *dq = dnew(qsl), *di = dnew(idx), *dn = dnew(nct), *dpl = dnew(pl),
         *da = dnew(all), *dids = dnew(ids), *dnx = dnew(nxt);
    prepare_prefill_inputs<<<NREQ, 256>>>(dids, dnx, di, dq, da, MAXTOK, dpl,
                                          dn);
    CK(hipDeviceSynchronize());

    std::vector<int> wi(qsl[NREQ], -1), wn(MAXREQ, -1);
    for (int r = 0; r < NREQ; ++r) {
        const int si = IDXMAP[r];
        if (nct[si] >= pl[si]) continue;  // decode row
        const int qlen = QLEN[r];
        for (int i = 0; i < qlen; ++i)
            wi[qsl[r] + i] = all[(size_t)si * MAXTOK + nct[si] + i];
        const int next = nct[si] + qlen;
        if (next < pl[si]) wn[si] = all[(size_t)si * MAXTOK + next];
    }
    rep("prepare_prefill_inputs ids", d2h(dids, wi.size()), wi);
    rep("prepare_prefill_inputs next", d2h(dnx, MAXREQ), wn);
    for (int* p : {dq, di, dn, dpl, da, dids, dnx}) CK(hipFree(p));
}

static void test_num_sampled_rejected() {
    const int nlog[NREQ] = {2, 1, 4, 1, 3, 1};
    auto cul = cu_of(nlog, NREQ);
    // seq_lens is indexed by batch position, prefill_len by request-state index.
    // Rows 2 and 4 are mid-chunked-prefill (seq_lens < prefill_len), which must
    // zero both counts; the rest keep the sampler's num_sampled input.
    const int sl[NREQ] = {70, 6, 40, 31, 20, 9};
    const int pl[NREQ] = {64, 6, 64, 30, 40, 9};
    std::vector<int> seq(sl, sl + NREQ), pre(MAXREQ, 0);
    for (int i = 0; i < NREQ; ++i) pre[IDXMAP[i]] = pl[i];
    std::vector<int> idx(IDXMAP, IDXMAP + NREQ);
    std::vector<int> ns_in{2, 1, 4, 0, 3, 1}, nr(NREQ, -1);
    auto *dc = dnew(cul), *ds = dnew(seq), *dp = dnew(pre), *di = dnew(idx),
         *dns = dnew(ns_in), *dnr = dnew(nr);
    get_num_sampled_and_rejected<<<NREQ, 256>>>(dns, dnr, ds, dc, di, dp, NREQ);
    CK(hipDeviceSynchronize());

    std::vector<int> wns(NREQ), wnr(NREQ);
    for (int r = 0; r < NREQ; ++r) {
        const bool chunked = seq[r] < pre[IDXMAP[r]];
        const int nl = cul[r + 1] - cul[r];
        wns[r] = chunked ? 0 : ns_in[r];
        wnr[r] = chunked ? 0 : nl - wns[r];
    }
    rep("get_num_sampled_and_rejected", d2h(dns, NREQ), wns);
    rep("  .. num_rejected", d2h(dnr, NREQ), wnr);
    for (int* p : {dc, ds, dp, di, dns, dnr}) CK(hipFree(p));
}

static void test_expand_idx_mapping() {
    const int nlog[NREQ] = {2, 1, 4, 1, 3, 1};
    auto cul = cu_of(nlog, NREQ);
    const int total = cul[NREQ];
    std::vector<int> idx(IDXMAP, IDXMAP + NREQ), em(total, -1), lp(total, -1);
    auto *dc = dnew(cul), *di = dnew(idx), *dem = dnew(em), *dlp = dnew(lp);
    expand_idx_mapping<<<NREQ, 256>>>(di, dem, dlp, dc);
    CK(hipDeviceSynchronize());

    std::vector<int> we(total), wl(total);
    for (int r = 0; r < NREQ; ++r)
        for (int i = cul[r]; i < cul[r + 1]; ++i) {
            we[i] = IDXMAP[r];
            wl[i] = i - cul[r];
        }
    rep("expand_idx_mapping", d2h(dem, total), we);
    rep("  .. local_pos", d2h(dlp, total), wl);
    for (int* p : {dc, di, dem, dlp}) CK(hipFree(p));
}

static void test_post_update_nct() {
    auto qsl = cu_of(QLEN, NREQ);
    std::vector<int> nct(MAXREQ);
    for (int i = 0; i < MAXREQ; ++i) nct[i] = i;
    std::vector<int> idx(IDXMAP, IDXMAP + NREQ);
    auto *dq = dnew(qsl), *di = dnew(idx), *dn = dnew(nct);
    post_update_num_computed_tokens<<<NREQ, 256>>>(di, dn, dq, NREQ);
    CK(hipDeviceSynchronize());

    std::vector<int> w = nct;
    for (int r = 0; r < NREQ; ++r) w[IDXMAP[r]] = nct[IDXMAP[r]] + QLEN[r];
    rep("post_update_num_computed_tokens", d2h(dn, MAXREQ), w);
    for (int* p : {dq, di, dn}) CK(hipFree(p));
}

static void test_uniform_decode() {
    const int ndec = 4, maxlen = 3, bt_stride = 48;
    const int sl[ndec] = {130, 7, 64, 999};
    std::vector<int> seq(sl, sl + ndec);
    std::vector<int> bt((size_t)ndec * bt_stride);
    std::uniform_int_distribution<int> bd(0, 900);
    for (auto& b : bt) b = bd(rng);
    const int ntok = ndec * maxlen;
    std::vector<int> dsl(ntok, -1), ebt((size_t)ntok * bt_stride, -1),
        dl(ntok, -1);
    auto *ds = dnew(seq), *dbt = dnew(bt), *ddsl = dnew(dsl), *debt = dnew(ebt),
         *ddl = dnew(dl);
    prepare_uniform_decode<<<ntok, 256>>>(ds, ddsl, dbt, bt_stride, debt,
                                          bt_stride, ddl, maxlen);
    CK(hipDeviceSynchronize());

    std::vector<int> wsl(ntok), wbt((size_t)ntok * bt_stride), wdl(ntok, 1);
    for (int t = 0; t < ntok; ++t) {
        const int req = t / maxlen, off = t % maxlen;
        wsl[t] = seq[req] - (maxlen - 1 - off);
        for (int b = 0; b < bt_stride; ++b)
            wbt[(size_t)t * bt_stride + b] = bt[(size_t)req * bt_stride + b];
    }
    rep("prepare_uniform_decode seq_lens", d2h(ddsl, ntok), wsl);
    rep("  .. expanded_block_table", d2h(debt, wbt.size()), wbt);
    rep("  .. decode_lens", d2h(ddl, ntok), wdl);
    for (int* p : {ds, dbt, ddsl, debt, ddl}) CK(hipFree(p));
}

// ---------------------------------------------------------------------- bench
__global__ void empty_kernel() {}

static void bench_floor() {
    // Establishes whether these kernels have any headroom at all: they are one
    // block per request over a handful of integers, so if they sit at the
    // empty-kernel latency there is nothing to optimize.
    const int reps = 2000;
    hipEvent_t a, b;
    CK(hipEventCreate(&a));
    CK(hipEventCreate(&b));
    for (int i = 0; i < 50; ++i) empty_kernel<<<64, 256>>>();
    CK(hipDeviceSynchronize());
    CK(hipEventRecord(a));
    for (int i = 0; i < reps; ++i) empty_kernel<<<64, 256>>>();
    CK(hipEventRecord(b));
    CK(hipEventSynchronize(b));
    float ms = 0;
    CK(hipEventElapsedTime(&ms, a, b));
    printf("empty_kernel launch floor        %.2f us/launch\n",
           ms * 1000.0f / reps);
    CK(hipEventDestroy(a));
    CK(hipEventDestroy(b));
}

static void bench() {
    // Decode-shaped: the batch-prep kernels run once per step, so per-step
    // latency at serving batch sizes is the only number that matters.
    const int reps = 2000;
    bench_floor();
    // (nreq, qlen): decode shapes first, then the chunked-prefill shapes the
    // same kernels serve -- a block-size choice tuned only on qlen=1 would be
    // tuned on the case where there is no work to parallelize.
    const int shapes[][2] = {{8, 1}, {32, 1}, {64, 1}, {256, 1},
                             {1, 4096}, {4, 4096}, {1, 16384}};
    for (auto& sh : shapes) {
        const int nreq = sh[0];
        std::vector<int> qlen(nreq, sh[1]);
        auto qsl = cu_of(qlen.data(), nreq);
        std::vector<int> idx(nreq), nct(nreq + 8, 3), sl(nreq + 8, 0);
        for (int i = 0; i < nreq; ++i) idx[i] = i;
        std::vector<long> pos(qsl[nreq], 0);
        auto *dq = dnew(qsl), *di = dnew(idx), *dn = dnew(nct), *ds = dnew(sl);
        auto* dp = dnew(pos);

        // One varied factor: threads per block. Each block handles a single
        // request's tokens, so 256 threads is mostly idle at decode shapes.
        for (int thr : {64, 128, 256}) {
            hipEvent_t a, b;
            CK(hipEventCreate(&a));
            CK(hipEventCreate(&b));
            for (int i = 0; i < 50; ++i)
                prepare_pos_seq_lens<<<nreq + 1, thr>>>(dp, ds, di, dq, dn,
                                                        nreq + 8);
            CK(hipDeviceSynchronize());
            CK(hipEventRecord(a));
            for (int i = 0; i < reps; ++i)
                prepare_pos_seq_lens<<<nreq + 1, thr>>>(dp, ds, di, dq, dn,
                                                        nreq + 8);
            CK(hipEventRecord(b));
            CK(hipEventSynchronize(b));
            float ms = 0;
            CK(hipEventElapsedTime(&ms, a, b));
            printf("prepare_pos_seq_lens  nreq=%-4d qlen=%-6d thr=%-4d "
                   "%.2f us/launch\n",
                   nreq, sh[1], thr, ms * 1000.0f / reps);
            CK(hipEventDestroy(a));
            CK(hipEventDestroy(b));
        }
        for (int* p : {dq, di, dn, ds}) CK(hipFree(p));
        CK(hipFree(dp));
    }
}

int main(int argc, char** argv) {
    if (argc > 1 && strcmp(argv[1], "--bench") == 0) {
        bench();
        return 0;
    }
    test_compute_slot_mapping(1, 0, 1);
    test_compute_slot_mapping(2, 1, 2);
    test_indexer_metadata(1, 0, 1);
    test_indexer_metadata(4, 2, 2);
    test_pos_seq_lens();
    test_prefill_inputs();
    test_num_sampled_rejected();
    test_expand_idx_mapping();
    test_post_update_nct();
    test_uniform_decode();
    printf("%s\n", g_fail ? "FAILED" : "ALL PASS");
    return g_fail ? 1 : 0;
}
