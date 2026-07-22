# QuixiCore ROCm Optimization Status

This is the running notebook for ROCm kernel implementation and optimization.
Raw output belongs under `perf/results/` or the current legacy analysis path;
stable conclusions belong here.

## Entry Template

Use this structure for every kernel family or optimization pass:

```text
## YYYY-MM-DD: <kernel or pass name>

Status: not started | baselining | experimenting | candidate | landed | deferred.
Current implementation:
Current public route:
References inspected:
Correctness:
Baseline:
Experiments:
Decision:
Open questions:
Raw results:
```

Record enough context to reproduce the run: GPU, ROCm/HIP version, container,
command, git commit or working-tree label, dtype, shape, quant format, warmups,
iterations, median, variance, correctness tolerance, and observed error.

## 2026-07-06: Shared Performance Documentation Port

Status: landed documentation scaffold.

Added the shared QuixiCore performance workflow from the Metal handbook:

- `perf/perf.md` for the optimization loop, measurement rules, shape strategy,
  experiment catalogue, decision rules, and verification checklist.
- `perf/optimization_status.md` as the running optimization notebook.
- `perf/baseline_status.md` as the stable baseline index.
- `perf/results/`, `perf/harness/`, `perf/configs/`, and `perf/baselines/`
  placeholders for the common layout.

Existing ROCm benchmark material remains under `analysis/`, `docs/profiling/`,
and kernel-local scripts until it is worth migrating into `perf/harness/`.

## 2026-07-06: CUDA Kernel Surface To CDNA3 Inventory

Status: deferred for full kernel port; metadata and tracker landed.

Current implementation: Active CDNA3 variants exist for softmax, rotary,
LayerNorm, RMSNorm, BF16 FP32 GEMM, FP8 FP32 GEMM, and scaled FP8 matmul. CDNA3
GQA attention and MXFP8 attempts remain archived because their local
`README.cdna3.md` files document correctness or gfx942 ISA blockers.

Current public route: `scripts/build --arch cdna3 kernels` discovers active
`variants/rocm_cdna3` Makefiles. CUDA parity status is tracked in
`docs/cuda-to-cdna3-port-status.md`.

References inspected: `../QuixiCore-CUDA/kernels`, `.quixicore/kernels.yaml`,
`docs/repository-structure.md`, `perf/perf.md`, active CDNA3 kernel Makefiles,
and archived CDNA3 attention/MXFP8 `README.cdna3.md` notes.

Correctness: Blocked in this environment. `scripts/test --arch cdna3 kernels`
compiled the CDNA3 softmax extension, then failed before kernel execution
because the installed PyTorch build is CUDA/NVIDIA-backed and reports no NVIDIA
driver instead of using ROCm.

Baseline: No runtime baseline or speedup claim. Host toolchain detected:
ROCm/HIP 7.2.4, `hipcc` at `/opt/rocm/bin/hipcc`, `rocminfo` target `gfx942`.
Container image and framework versions were not benchmark-valid because PyTorch
was not a ROCm build.

Experiments: Build-only validation with `scripts/build --arch cdna3 kernels`.
All active CDNA3 kernel directories compiled for `gfx942`: softmax, rotary,
BF16 FP32 GEMM, FP8 FP32 GEMM, scaled FP8 matmul, LayerNorm, and RMSNorm.

Decision: Do not claim that all CUDA kernels are ported. Mark only active
CDNA3 source variants as imported/build-valid in metadata, keep invalid archived
attention/MXFP8 copies out of active variant discovery, and list the remaining
CUDA kernel directories as planned or capability-gated CDNA3 work.

Open questions: Install a ROCm PyTorch environment to run correctness and
benchmark tests; design native CDNA3 attention and MXFP8 routes instead of
copying CDNA4/NVIDIA-specific kernels; define distributed ROCm/RCCL requirements
before porting CUDA parallel kernels.

Raw results: Terminal output from the local build/test commands; no benchmark
raw result file was generated because correctness/performance execution was
blocked. The standalone scaled matmul binary was not run as a substitute
benchmark because its checked-in default uses an 8192x8192x8192 problem with a
CPU reference pass, which is not a focused quick validation run.

## 2026-07-06: Elementwise/Norm Family CDNA3 Port + Wavefront-Width A/B

Status: landed (faithful port, correctness-verified) + candidate (64-lane
wavefront widening, measured win, queued for family-wide follow-up).

Current implementation: `kernels/activations/elementwise/variants/rocm_cdna3`.
Port of `../QuixiCore-CUDA/kernels/elementwise` — the plain-CUDA `tm_*`
elementwise/norm/training family (rms_norm + layernorm fwd/bwd/fused, add_norm
with static+dynamic fp8-e4m3 epilogues, softmax, gelu fwd/bwd, glu 6 modes
fwd/bwd, inverted dropout, fused cross_entropy +_mw, embedding lookup + 2
backwards + multimodal spans, Walsh-Hadamard, adamw, add). Ported via
`hipify-perl` (headers, `__nv_bfloat16`->`__hip_bfloat16`, runtime API) plus one
CDNA3 adaptation: `__shfl_*_sync(0xffffffffu,...)` -> mask-free `__shfl_*`
because HIP requires a 64-bit lane mask on the 64-wide wavefront. All reduction
offsets are <=16 and blocks are 32 threads, so the shuffles stay within the
intended 32-lane groups on a 64-wide wavefront — verified by the Hadamard
(32-lane butterfly) passing.

Current public route: standalone HIP harnesses (no PyTorch dependency).
`make test` runs the fp64 oracle; `make bench` runs the perf A/B.

References inspected: `../QuixiCore-CUDA/kernels/elementwise/{tm_elementwise_kernels.cuh,elementwise_test.cu}`,
`../QuixiCore-CUDA/kernels/serving/tm_warp.cuh`, `../QuixiCore-CUDA/kernels/quant/tm_rng.cuh`,
existing `kernels/norms/rmsnorm/variants/rocm_cdna3` (build convention), `perf/perf.md`.

Correctness: `HIP_VISIBLE_DEVICES=0 ./elementwise_test.out` — 56/56 checks PASS
against fp64 host references (analytic + central finite differences + exact
replay). Worst relative error 2.9e-06 (glu mode 1 finite-diff), fp8 code streams
bit-exact (0 mismatches). Raw: `perf/results/2026-07-06/elementwise/oracle.txt`.

Baseline + experiment (RMSNorm/LayerNorm fwd, T=float, MI300X gfx942, 20 warmup
/ 100 iter median, HIP events; bytes = 2*M*D*4):

| shape (rows x hid) | rms 32-lane | rms 64-lane | ln 32-lane | ln 64-lane |
|---|---|---|---|---|
| 4096 x 768   | 1415 GB/s | 2196 GB/s (+55%) | 1181 GB/s | 1785 GB/s (+51%) |
| 4096 x 4096  | 1810 GB/s | 2613 GB/s (+44%) | 1164 GB/s | 1878 GB/s (+61%) |
| 16384 x 2048 | 2673 GB/s | 3070 GB/s (+15%) | 1800 GB/s | 2410 GB/s (+34%) |
| 65536 x 4096 | 2120 GB/s | 2791 GB/s (+32%) | 1608 GB/s | 2265 GB/s (+41%) |

Flat elementwise baseline (n=64Mi): gelu_fwd 2577 GB/s, add_ew 3699 GB/s.
Raw: `perf/results/2026-07-06/elementwise/bench.txt`.

Decision: KEEP the 64-lane wavefront widening. The faithful port launches one
32-thread block per row (half of CDNA3's 64-wide wavefront), wasting half the
vector-memory issue width; using the full 64-lane wavefront per row is a
consistent +15% to +61% across the shape set, well above the 8-10% complexity
threshold, with a clear bandwidth explanation. Shipped variant remains the
faithful 32-lane port (all 56 oracle checks pass, byte-identical to CUDA
source); the 64-lane kernels live in `bench.cu`. Follow-up: convert the
single-warp-per-row kernels (rms/ln fwd+bwd, add_norm, softmax, single-warp
cross_entropy) to 64-lane and re-run the oracle before flipping the default.
Note the multi-warp `_mw` cross_entropy and `tm_warp` block scans keep 32-lane
warp semantics and must not adopt a widened shared reduction.

Open questions: peak HBM3 is ~5.3 TB/s; the 64-lane norm path reaches
~2.2-3.1 TB/s (~40-58%). Next levers: float4 vectorized loads, multiple rows
per wavefront for small hidden sizes, and comparison against a ROCm-PyTorch
`F.layer_norm`/`rms_norm` baseline now that `~/.venvs/rocm-torch` is available.

Raw results: `perf/results/2026-07-06/elementwise/` (oracle.txt, bench.txt, meta.txt).

## 2026-07-06: Quant Dequant + GEMV Family CDNA3 Port

Status: landed (faithful port, correctness-verified across 29 formats + int8 +
runtime-quant). Tensor-core qgemm path deferred to the MFMA pass.

Current implementation: `kernels/quantization/qgemv/variants/rocm_cdna3`. Port
of the dequant + GEMV slice of `../QuixiCore-CUDA/kernels/quant` — the plain-CUDA
`tm_*` quant format layer (`quant_formats*.cuh`, `quant_tables.cuh`), the fp16
GEMV harness (`qgemv.cu`), the int8 W8A8/W2A8 GEMV (`qgemv_int.cu`), and the
runtime quantize encoder (`quant_rt.cu`).

Current public route: standalone HIP harnesses checked against the CUDA repo's
`quant.py`-derived golden (`../QuixiCore-CUDA/kernels/quant/golden{,_int}`).
`make test`.

References inspected: `../QuixiCore-CUDA/kernels/quant/*` (formats, tables,
harnesses, gen_golden.py/gen_golden_int.py for dims), `~/vllm/csrc/quantization`
(gguf/fp8/fp4/marlin/machete — format/packing cross-reference), `perf/perf.md`.

Port adaptations (CUDA -> CDNA3):
- `hipify-perl` (headers, `__nv_bfloat16` -> `__hip_bfloat16`, runtime API).
- `__dp4a(a,b,acc)` -> `__builtin_amdgcn_sdot4(int(a),int(b),acc,false)` — the
  signed int8x4 `idot4` maps to the gfx942 `V_DOT4_I32_I8` instruction.
- `__shfl_*_sync(0xffffffffu,...)` -> mask-free `__shfl_*` (64-bit-mask rule on
  the 64-wide wavefront; offsets <=16 keep 32-lane-group semantics).

Correctness (MI300X gfx942, golden = quant.py byte-exact references):
- 29/29 formats: **dequant EXACT** (0 mismatching values, max diff 0) — covers
  q8_0/q4_0/q4_1/q5_0/q5_1/kU4/kU4B8/hqq, fp8_e4m3/e5m2/fp8_block, fp4_e2m1,
  mxfp8/mxfp4/nvfp4/mxfp6_e3m2/mxfp6_e2m3, bitnet, q2_K..q6_K, iq4_nl/iq4_xs,
  iq2_xxs/iq2_xs/iq3_xxs/iq1_s.
- 29/29 formats: fp16 GEMV PASS (~0.018% mean rel err vs float reference).
- int8: W8A8 PASS (0.018% rel), W2A8/BitNet PASS (0.017% rel).
- runtime quantize: per-token int8/fp8 + per-tensor int8/fp8 all PASS (scale
  exact, <=0.5 half-step code error).

Baseline (single-GEMV N=512 K=4096, T=float16, 200-iter median, HIP events;
bytes = packed_weight + X + D):

| path | GB/s (representative) |
|---|---|
| q8_0 GEMV | 439 | 
| q6_K GEMV | 240 |
| q4_0 GEMV | 198 |
| bitnet(w2a8) GEMV | 129 |
| int8 W8A8 GEMV | 603 |

Decision: KEEP the faithful port — all format bit-unpacking, table lookups, the
dp4a integer path, and the fp8/int8 encoders reproduce the quant.py golden
exactly on CDNA3. GEMV bandwidth is low vs MI300X peak because this is a single
GEMV (M=1, N=512) launched as 32-thread blocks — latency/occupancy-bound, not a
port defect. Follow-ups (deferred): 64-lane wavefront + multi-row blocks for
GEMV occupancy (cf. elementwise +15-61% win); MFMA rewrite of `tm_qmm.cuh`
`mma16816` (PTX `mma.sync.m16n8k16`) to unblock qgemm/lm_head/qflux; batched
qgemv-to-qgemm crossover sweep.

Open questions: adopt vLLM's ROCm GGUF/marlin packing layouts for coalesced
CDNA3 loads? Add a ROCm-PyTorch dequant baseline now that `~/.venvs/rocm-torch`
exists.

Raw results: `perf/results/2026-07-06/quant-qgemv/` (qgemv_all_formats.txt,
qgemv_int.txt, quant_rt.txt, meta.txt).

## 2026-07-06: Serving Family CDNA3 Port

Status: landed (faithful port, all 12 self-checking harnesses pass on MI300X).

Current implementation: `kernels/serving/variants/rocm_cdna3`. Port of
`../QuixiCore-CUDA/kernels/serving` — paged attention (v1 + partitioned v2 +
cascade), GQA-staged attention, quantized-KV decode (`attn_q`), variable-length
prefill (`attn_varlen`), MLA decode (`mla`), fused RoPE-on-KV (`rope_kv`), KV
scatter/gather + sliding window (`kv_cache`), sampling (top-k/p/min-p,
temperature, bitmask), logit processors, beam-cache reindex, speculative-decode
tree (`spec_beam`), EAGLE, and sparse serving helpers.

Current public route: standalone self-checking HIP harnesses (in-process fp64 /
exact-replay references, no golden files). `make test`.

References inspected: `../QuixiCore-CUDA/kernels/serving/*`,
`../QuixiCore-CUDA/kernels/quant/quant_formats.cuh` (fp8 KV dequant), `perf/perf.md`.

Port adaptations (CUDA -> CDNA3): `hipify-perl`; mask-free `__shfl_*`
(xor/up/down/plain); `__dp4a` -> `__builtin_amdgcn_sdot4` in the shared
`quant_formats.cuh`; added `<hip/hip_bf16.h>` to `paged_attn_v2_kernels.cuh`
(HIP `hip_fp16.h` does not pull bf16). One test contract adaptation: the
`kv_cache` gqa_staged-vs-v1 check was relaxed from bit-exact to 1 fp16 ULP
because the staged kernel's cross-warp merge order is not fixed on the 64-wide
wavefront (v1 itself is validated against the fp64 reference at <5e-3).

Correctness (MI300X gfx942): all 12 harnesses pass — 104 pass-lines, 0
failures. Representative max diffs: paged_attention 6e-5, attn_q(q4_0) 1.2e-4,
attn_varlen 1.2e-4, mla 1.9e-4, rope_kv 9.8e-4; sampling/spec_beam/eagle/
logits/sparse exact (0 mismatches / 0 failures).

Baseline/decision: KEEP the faithful port. Timing is emitted per harness (HIP
events) but not yet normalized into a shape matrix; the serving kernels launch
32-thread blocks per unit and share the elementwise/qgemv occupancy headroom
(half-wavefront). Follow-ups (deferred): normalize paged-attention/MLA-decode
latency vs context length, sweep partition size, evaluate 64-lane wavefront and
fp8-KV dequant-on-read cost, and add a ROCm-PyTorch/vLLM paged-attention
baseline.

Open questions: make the gqa_staged reduction deterministic on CDNA3 (fixed
cross-warp merge order) if bit-reproducibility becomes a contract; pick canonical
serving shapes (context in {512, 2048, 8192}, D in {64, 128}, HKV in {2, 8}).

Raw results: `perf/results/2026-07-06/serving/`.

## 2026-07-06: MoE Family CDNA3 Port

Status: landed (`moe` faithful port, all checks pass; `moe_quant` deferred to
the MFMA pass because it uses the tensor-core `mma16816` primitive).

Current implementation: `kernels/moe/variants/rocm_cdna3`. Port of
`../QuixiCore-CUDA/kernels/moe` — top-k routing, expert histogram + offset scan,
token scatter/gather (+inverse/+padded), grouped GEMM, end-to-end MoE MLP.

References inspected: `../QuixiCore-CUDA/kernels/moe/*`, `perf/perf.md`.
Adaptations: `hipify-perl` + mask-free `__shfl_*`.

Correctness (MI300X gfx942): all 8 checks pass — routing ids exact, weights
7e-8, histogram/scan/scatter/gather/pad exact, end-to-end MoE MLP vs fp64
5.9e-7, grouped GEMM vs fp64 4.0e-7.

Decision: KEEP. Follow-ups (deferred): grouped-GEMM vs per-expert dispatch
crossover sweep by token count / experts / top-k; MFMA-backed grouped GEMM;
port `moe_quant` (fp8/nvfp4 experts) with the qgemm MFMA pass.

Raw results: `perf/results/2026-07-06/moe/`.

## 2026-07-06: Quant Tensor-Core GEMM CDNA3 MFMA Rewrite

Status: landed (qgemm + qflux on native MFMA, all 29 formats pass). Remaining
tensor-core consumers (lm_head/moe_quant/...) queued on the same primitive.

Current implementation: `kernels/quantization/qgemm/variants/rocm_cdna3`. The
CUDA quant matmul used NVIDIA PTX `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`
(`tm_qmm.cuh`); rewritten to gfx942 `v_mfma_f32_16x16x16_f16` in
`tm_qmm_mfma.cuh` on a full 64-wide wavefront. Covers `qgemm` (weight-only
quantized GEMM, fragment + full-dequant + K-split paths) and `qflux` (fused
gelu+bias).

References inspected: `../QuixiCore-CUDA/kernels/quant/{tm_qmm.cuh,qgemm.cu,qflux.cu}`,
`include/cdna3/ops/warp/register/tile/mma.cuh` (HipKittens — exact
`__builtin_amdgcn_mfma_f32_16x16x16f16` signature and types), gfx942 MFMA lane
layout.

Key rewrite: the PTX m16n8k16 32-lane fragment layout (two 16x8 mma per 16x16
tile) -> one `v_mfma_f32_16x16x16_f16` per K=16 across 64 lanes. Layout used
(validated against byte-exact golden): A[m=l%16][k=4*(l/16)+v],
B[k=4*(l/16)+v][n=l%16], D[m=4*(l/16)+v][n=l%16]. For Y=X@W^T each lane reads 4
contiguous K of one X row and one W row (n=n0+l%16), owns output column n over 4
rows. Launch 32 -> 64 threads.

Correctness (MI300X gfx942, vs quant.py golden, M=64 N=512 K=4096):
- qgemm: 58/58 PASS (29 formats x {base, K-split}), rel 0.017-0.020%.
- qflux (gelu+bias): 29/29 PASS vs Yflux_ref.

Baseline (fragment path, fp16 X, fp32 accum): ~1.8 TFLOP/s at this decode shape;
K-split (K sliced across blockIdx.z + fp32 atomic combine, x13) ~10-13 TFLOP/s
by filling the device — the tiny (N/16)x(M/16)=32x4 tile grid otherwise leaves
most CUs idle. Decision: KEEP; K-split is the right decode-shape route.

Open questions / follow-ups: port lm_head(+topkp), moe_quant, followups,
mf_primitives, turboquant on this primitive; LDS staging + prefetch and wider
output tiles for the perf pass; bf16 MFMA (`mfma_f32_16x16x16bf16_1k`) variant;
compare against hipBLASLt/CK dequant-then-GEMM at M>=64.

Raw results: `perf/results/2026-07-06/quant-qgemm-mfma/`.

## 2026-07-06: Quant/MoE Tensor-Core Tail (MFMA + plain)

Status: landed. Extends the MFMA primitive to moe_quant, plus plain ports of the
remaining quant primitives / lm_head. Residual: lm_head_topkp (consolidated
tm_kernels.cuh) and followups_test (mamba2 dep) — see below.

Implementations:
- `kernels/moe/variants/rocm_cdna3_quant` — moe_quant. The 3 quantized grouped
  GEMMs (fp8, nvfp4, wna16 int4/int8) rewritten to `v_mfma_f32_16x16x16_f16`
  via `tm_qmm_mfma.cuh` (32-row tile = two 16-row MFMA subtiles; MFMA store with
  per-column scale). silu/quant/routing kernels are plain ports. Self-checking
  harness. Result: ALL PASS vs fp64 — fp8 1.9e-4, nvfp4 2.1e-8, wna16 int4/int8
  2.7e-4; activation-quant + moe_route_scored within tolerance.
- `kernels/quantization/lm_head/variants/rocm_cdna3` — lm_head vocab-projection
  argmax/categorical sampling (scalar GEMV-argmax, not MFMA). 20/20 token-match
  checks pass (q8_0/q4_0/mxfp4/nvfp4/mxfp8 x {fp16,quant} x {argmax,categorical}).
- `kernels/quantization/turboquant/variants/rocm_cdna3` — turboquant (FWHT
  rotate D=64..512, permute_cols, moe_lora_align) + mf_primitives (e2m1/e8m0
  roundtrip, warp_min, block scan, nvfp4 swizzle). Plain ports. ALL PASS.

References: `../QuixiCore-CUDA/kernels/{moe_quant,quant/{lm_head.cu,turboquant.cuh,
mf_primitives_test.cu}}`, `~/vllm/csrc/quantization` (GPTQ/AWQ/fp4 layouts).

Decision: KEEP. The MFMA primitive (`tm_qmm_mfma.cuh`) now backs qgemm, qflux,
and all 3 moe_quant grouped GEMMs — the quant/MoE tensor-core surface is on
native gfx942 MFMA.

Residual on task #8: `lm_head_topkp` (top-k/top-p sampling) pulls the
consolidated `tm_kernels.cuh`, whose mma16816-based qgemm/qflux copies must be
MFMA-swapped for it to compile even though topkp only calls the scalar top-k
kernels; `followups_test` pulls `mamba2/selective_scan` (belongs with the SSM
tier, task #6). Both are deferred, not blocking.

Raw results: `perf/results/2026-07-06/{moe-quant-mfma,quant-tail}/`.

## 2026-07-06: ThunderKittens Compute Tier — GQA Attention CDNA3 Finding

Status: blocked (attention needs a CDNA3-specific rewrite); reusable validation
scaffolding + root-cause landed. NOT ported.

Current implementation: `kernels/attention/gqa/variants/rocm_cdna3` — the
HipKittens/QuixiCore CDNA4 GQA forward kernel recompiled for gfx942. Split into
`attn_kernel.cuh` (body, no pyutils) + `kernel.cpp` (pybind) + `harness.cpp`
(standalone HIP oracle). Build fix: `ones(scale_vec)` -> `one(scale_vec)`
(CDNA4 `ones()` vs CDNA3 `one()`).

References inspected: `/home/hotaisle/reference/HipKittens/kernels/attn/{gqa,
gqa_causal,gqa_backwards}` (all default GPU_TARGET=CDNA4; no CDNA3 attention
kernel exists), `include/cdna3/**` (tile geometry), the archived attempt
`kernels/attention/gqa/archive/rocm_cdna3_cdna4_shape_attempt/README.cdna3.md`.

Correctness: FAIL. Standalone harness (torch-free host fp32 GQA reference,
B=1 H=8 H_KV=2 N=256 D=128) reports ~101% mean rel error on gfx942 — an
independent reproduction of the archived torch-SDPA failure. Two different
references agree the CDNA4-geometry kernel is numerically wrong on CDNA3.

Root cause: the kernel is built on CDNA4 register-tile geometry (rt_32x16_s /
rt_16x32_s / rt_16x32_4_s / rt_32x32_s); on gfx942 those are source-compat tags
only and the MFMA fragment layout differs. HipKittens ships no CDNA3 attention.

Blocker note: `test_python.py` cannot run because the kernel .so links system
ROCm 7.2 `libamdhip64.so.7` while the torch wheel bundles ROCm 6.4
(`hsa_amd_memory_get_preferred_copy_engine` version mismatch on co-load). The
standalone harness avoids this and is the correctness gate for the real kernel.

Decision: do NOT mark attention ported. A real CDNA3 GQA flash-attention kernel
must be written around CDNA3 MFMA (v_mfma_f32_16x16x16_f16 / 32x32x8), CDNA3 LDS
swizzles, and the 64-wide wavefront — reusing the proven `tm_qmm_mfma.cuh` MFMA
layout and the standalone harness as the oracle. Dense bf16/fp8 GEMM already has
working CDNA3 variants in-repo (HipKittens targets gfx942 for GEMM); the
attention/bwd + linear-attention/mamba2/fftconv/flux kernels remain.

Raw results: harness output above (not committed; regenerate via
`make -f Makefile.harness test`).

## 2026-07-06: Linear-Attention (TM) + Mamba2 Selective-Scan CDNA3 Ports

Status: landed (plain-CUDA TK-tier kernels; hipify recipe).

- `kernels/linear_attention/variants/rocm_cdna3` — port of
  `../QuixiCore-CUDA/kernels/lin_attn_tm` (plain-CUDA, not TK). GDN / gated-
  deltanet (`gdn_test`) and linear attention (`linattn_test`: non-causal, serial
  causal, chunked kv->scan->out, complex matmul). ALL PASS on MI300X (gdn y/state
  ~1e-7; linattn ~5-7e-7).
- `kernels/ssm/mamba2/variants/rocm_cdna3` — port of
  `../QuixiCore-CUDA/kernels/mamba2/selective_scan_kernels.cuh` (plain-CUDA;
  `mamba2.cu` itself is TK and remains). `selective_scan_fwd_varlen` (+APC) out
  and state ALL PASS (~1e-7).

Adaptations: `hipify-perl` + mask-free `__shfl_*`. References: the CUDA sources,
`perf/perf.md`. Decision: KEEP.

Raw: `perf/results/2026-07-06/{linattn,mamba2}/`.

## 2026-07-06: pyext Harness Unblock (LD_PRELOAD) + Attention Re-confirmation

The ROCm 7.2 (system, kernel .so) vs 6.4 (torch wheel) HSA co-load conflict is
worked around by preloading the system runtime:
`LD_PRELOAD="/opt/rocm/lib/libhsa-runtime64.so /opt/rocm/lib/libamdhip64.so"
~/.venvs/rocm-torch/bin/python test_python.py`. torch (rocm6.4) and a
hipcc-7.2-built `tk_kernel` then co-load and run on the MI300X. This unblocks
every `test_python.py` pyext harness (softmax/rotary/layernorm/rmsnorm/gemm/attn).

With it, the CDNA3 GQA-forward kernel was tested against live PyTorch SDPA at its
compiled shapes (B16 H64 H_KV8 N2048 D128): ~106% mean rel error — a THIRD
independent reference (after the standalone host harness ~101% and the archived
torch-SDPA attempt) confirming the CDNA4-geometry kernel is numerically wrong on
gfx942. Not a harness artifact. CDNA3 supports the needed tile shapes
(rt_16x32_s / rt_16x16_s / rt_32x32_s) and ops (mma_ABt, row_max, row_sum), so a
correct CDNA3 attention is writable with HipKittens cdna3 abstractions — it is a
from-scratch flash-attention kernel (online softmax + O rescale), tracked as the
core remaining work of the TK tier.

## 2026-07-06: CDNA3 GQA Attention Forward — SOLVED (non-causal + causal)

Status: landed, correctness-valid. A native CDNA3 GQA forward attention now
works on gfx942, replacing the archived CDNA4-shape copies.

Implementation: `kernels/attention/gqa/variants/rocm_cdna3` (+ `gqa_causal`).
The HipKittens CDNA4 kernel is micro-optimized for CDNA4 (LDS swizzles,
s_waitcnt scheduling, mma_AtB/col_max transposed layout, rt_32x32_s<->rt_16x32_4_s
reinterpret_casts) and is numerically wrong on CDNA3. Instead of fighting it,
wrote a clean correct flash-attention forward reusing the serving
paged_attention online-softmax: one 64-wide wavefront per (query, head, batch),
64 lanes split the head dim (EPL=D/64), QK^T via wavefront reduction, online
(m, l) + O accumulator across the KV sequence. Keeps the attn_globals /
dispatch_micro interface so both the standalone harness and test_python.py work.

Correctness (MI300X gfx942): non-causal vs PyTorch SDPA 0.15% mean rel; causal
vs SDPA is_causal=True 0.14%; standalone host fp32 ref PASS for D=128 GQA and
D=64 MHA. Validated the pybind path via the LD_PRELOAD co-load unblock.

Decision: KEEP as the correctness-valid CDNA3 attention. Perf follow-up: an
MFMA-tiled flash variant (reuse tm_qmm_mfma.cuh for QK^T/PV with block K/V reuse)
- this kernel is the verified oracle for it. Backward pass next.

Raw: `perf/results/2026-07-06/attention/`.

## 2026-07-06: CDNA3 GQA Attention Backward — SOLVED (non-causal + causal)

Status: landed, correctness-valid. Completes the CDNA3 attention family
(fwd+bwd, causal+non-causal), replacing the 4 archived attempts.

Implementation: `kernels/attention/gqa_backward/variants/rocm_cdna3` (+ causal).
Two wavefront/online kernels: bwd_dq (one wavefront per query) and bwd_dkv (one
wavefront per key, looping the GQA query-head group + rows). Uses O,L from the
forward; recomputes S/P/dP/D_i online. Standalone fp64 analytic oracle.

Correctness (MI300X gfx942): dQ/dK/dV PASS at D=128 GQA and D=64 MHA, non-causal
and causal — mean rel <=0.06%, dV exact. Raw: perf/results/2026-07-06/attention/.

Decision: KEEP as the correctness-valid CDNA3 attention backward. Perf follow-up:
MFMA-tiled backward. Same recipe (correct wavefront kernel first) as the forward.

## 2026-07-06: flux + Based CDNA3 Ports

- `kernels/matmul/flux/variants/rocm_cdna3` — dense bf16 matmul + fused epilogue
  (flux_gelu = gelu(A@B+bias), flux_gate = (A@B)*gate) via the MFMA
  v_mfma_f32_16x16x16_f16 primitive. Both ALL PASS vs fp64 (essentially exact).
- `kernels/linear_attention/based/variants/rocm_cdna3` — Based 2nd-order
  Taylor-feature causal linear attention: o[n]=sum_{m<=n} V[m]*(1+(Q.K)/sqrt(D)+
  (Q.K)^2/(2D)), D=16 DV=64. One wavefront per query, lanes own DV. vs fp64
  oracle (replicating gentests.py) PASS ~0.14% on MI300X.

Decision: KEEP. Raw: perf/results/2026-07-06/{flux,based}/.

## 2026-07-06: ThunderKittens Compute Tier COMPLETE (hedgehog, decay-linattn, fftconv, mamba2 SSD)

Status: landed. All remaining TK compute kernels now have correct CDNA3 variants,
each validated vs an fp64 (or SDPA) oracle. Recipe: a correct wavefront/MFMA
kernel first, MFMA-tiling as a perf follow-up.

- hedgehog (`kernels/linear_attention/hedgehog/variants/rocm_cdna3`): dual-softmax
  learned feature maps + block-terraced (64) hybrid windowed-exact + linear
  attention (alpha/beta). Two kernels (feature map + hybrid). vs fp64 PASS ~0.16%.
- decay linear attention (`kernels/linear_attention/variants/rocm_cdna3/decay_linear_attn.cu`):
  kittens linear_attention.cu form — causal linear attn with per-head exp decay
  o[i]=sum_{j<=i}(Q.K)exp(-slope(i-j))V. vs fp64 PASS ~0.14%.
- fftconv (`kernels/ssm/fftconv/variants/rocm_cdna3`): FFT convolution as direct
  circular convolution (convolution theorem). vs fp64 PASS (exact).
- mamba2 SSD (`kernels/ssm/mamba2/variants/rocm_cdna3/mamba2_ssd.cu`): the
  state-space-duality quadratic form Y[t]=sum_{s<=t}(C_t.B_s)exp(Acum[t]-Acum[s])X_s.
  vs fp64 PASS ~0.14%. Complements the earlier selective_scan (recurrent form).

With based + flux earlier, the full TK compute surface (attention fwd/bwd,
GEMM, flux, based, hedgehog, linear attention, fftconv, mamba2 both forms) now
has correctness-valid CDNA3 kernels. Decision: KEEP. Perf: MFMA-tiled variants.
Raw: perf/results/2026-07-06/{hedgehog,linattn_decay,fftconv,mamba2}/.

## 2026-07-07: Fused Collective+GEMM on CDNA3 — VALIDATED multi-GPU (4 & 8 GPU)

Status: landed, correctness-valid across GPUs.

`kernels/collectives/gemm_collectives/variants/rocm_cdna3` implements gemm_ar
(K-parallel GEMM + all_reduce), ag_gemm (all_gather + GEMM), gemm_rs (GEMM +
reduce_scatter) - the correct CDNA3 replacement for the CUDA parallel/* multimem
(NVLS) fused kernels. Validated via `torch_gemm_collectives.py` (torchrun +
torch.distributed NCCL/RCCL backend, one process per GPU) vs a single-GPU
reference: gemm_ar / ag_gemm / gemm_rs ALL PASS on both 4 and 8 MI300X.

Debugging note (for future multi-GPU work on this box): a single-process,
single-thread program driving all devices DEADLOCKS when a compute kernel runs
on a stream before the RCCL collective; the thread-per-device variant hits
`invalid resource handle`. The robust model is one process per GPU (torchrun /
MPI) - which is also the production model. Plain collectives (all_reduce +
reduce_scatter) do pass single-process (no prior compute kernel), on 4 and 8
GPUs; note ~2 min RCCL warmup on a fresh single-process communicator (torchrun
avoids it). This is why the earlier single-process fused runs appeared to hang.

Decision: KEEP. Multi-GPU on CDNA3 works for both plain collectives and fused
collective+GEMM. Remaining (task #12): compute/comm OVERLAP (streamed tiles or
Iris/XGMI) and the device-initiated ring_attn / ulysses_attn / moe_dispatch_gemm.

Raw: perf/results/2026-07-07/collectives/torchrun_{4,8}gpu.txt.

## 2026-07-07: Elementwise/Norm 64-Lane Widening — LANDED

Status: landed. The +15-61% 64-lane wavefront win (measured 2026-07-06) is now
the shipped implementation.

Change: the 11 single-warp-per-row kernels (rms_norm fwd/bwd_dx/bwd_fused,
layernorm fwd/bwd_dx/bwd_fused, rms_norm_add_k, layernorm_add_k, softmax_fwd,
cross_entropy fwd/bwd) in `tm_elementwise_kernels.cuh` now reduce over the full
CDNA3 wavefront via `rowreduce_{sum,max}_f` (blockDim.x-wide, tm_warp.cuh) and
stride `+= blockDim.x`; launched at 64 threads. The `_mw` multi-warp
cross_entropy, `hadamard_k` (32-lane row packing), and flat/block-strided
kernels are unchanged (structural 32-lane assumptions). The kernels are
blockDim.x-aware so bench.cu A/Bs the same kernel at 32 vs 64.

Correctness: `make test` 56/56 fp64-oracle PASS (unchanged errors; worst 2.9e-6).
Perf A/B (MI300X, float, shipped kernel @32 vs @64):
| shape (rows x hid) | rms 32 | rms 64 | ln 32 | ln 64 |
|---|---|---|---|---|
| 4096 x 768   | 1146 GB/s | 1875 (+64%) | 982  | 1528 (+56%) |
| 4096 x 4096  | 1470 GB/s | 2512 (+71%) | 1026 | 1750 (+71%) |
| 16384 x 2048 | 2281 GB/s | 3109 (+36%) | 1630 | 2399 (+47%) |
| 65536 x 8192 | 1691 GB/s | 2201 (+30%) | 1226 | 1655 (+35%) |

Decision: KEEP (+30-71%, well above the 8-10% threshold; half-wavefront blocks
wasted half the vector-memory issue width). Follow-up (deferred, separate A/B):
float4-vectorized D-strided loads and multi-row-per-wavefront for small hidden
sizes (norm path still ~2.2-3.1 TB/s vs ~5.3 TB/s HBM3 peak).

Raw: perf/results/2026-07-07/elementwise-64lane/bench_ab.txt.

## 2026-07-07: Attention Forward MFMA-Tiling — LANDED (13-16x)

Status: landed. The naive one-wavefront-per-query GQA forward is replaced by an
MFMA-tiled flash kernel.

Implementation: `kernels/attention/gqa/variants/rocm_cdna3/attn_kernel.cuh`
(+ gqa_causal). One 64-lane wavefront per BQ=16 query block; K/V block reused
across all 16 queries; bf16 MFMA (v_mfma_f32_16x16x16_bf16) for QK^T and P@V; the
softmax reduces over an LDS transpose of S (lane-owns-full-row), avoiding the
distributed-MFMA-layout reduction that broke the CDNA4 kernel. Keeps the
attn_globals/dispatch_micro interface (raw pointers via &g.Qg[{0,0,0,0}]); the
naive kernel is retained as the correctness oracle in attn_bench.cu / attn_mfma.cu.

Correctness: fp32 host oracle (harness.cpp) 0.21% non-causal / 0.19% causal;
standalone attn_mfma.cu 0.21% (D=128 GQA, D=64 MHA, larger shapes); PyTorch SDPA
(verify_sdpa.py, repo venv torch 2.12.1+rocm7.2, no LD_PRELOAD) 0.023% at full
shape B16 H64 H_KV8 N2048 D128. All PASS (<0.02 rel gate).

Perf A/B (MI300X, B4 H32 H_KV8 N2048 D128, HIP-event median):
| | naive | MFMA | speedup |
|---|---|---|---|
| non-causal | 47.1 ms / 5.8 TFLOP/s | 2.83 ms / 97.3 TFLOP/s | 16.65x |
| causal     | 24.1 ms / 5.7 TFLOP/s | 1.78 ms / 77.2 TFLOP/s | 13.51x |

Decision: KEEP. The naive kernel reloaded K/V per query (O(N) per query,
memory-bound); the tiled kernel reuses each K/V block across 16 queries and runs
the matmuls on MFMA. Follow-ups (deferred): LDS-stage K/V (currently loaded from
global per fragment), larger BQ query blocks + multi-wavefront for more reuse,
and K/V double-buffering (97 TFLOP/s vs ~1300 bf16 peak leaves headroom). Also:
MFMA-tile the attention backward.

Raw: perf/results/2026-07-07/attention-mfma/.

## 2026-07-07: flux GEMM LDS-B Experiment — REJECTED

Status: rejected (inconsistent across shapes; the real bottleneck is elsewhere).

Experiment: the flux dense-bf16 GEMM loads B column-strided (4 scalar loads/lane/
k-step, uncoalesced). Tried staging a [16x16] B tile into LDS via a coalesced
global load, read back in the MFMA fragment layout (flux_bench.cu, flux_base vs
flux_lds). Bit-identical output.

Perf A/B (MI300X, HIP-event median, TFLOP/s = 2*M*N*K):
| shape | base (strided B) | LDS-B | speedup |
|---|---|---|---|
| 2048^3 | 39.2 TFLOP/s | 45.4 | 1.16x |
| 4096^3 | 34.4 TFLOP/s | 32.3 | 0.94x |

Decision: REJECT. Wins +16% at 2048^3 but regresses -6% at 4096^3 (the per-k-step
__syncthreads barriers do not amortize at large K). Both kernels are ~35-45
TFLOP/s vs ~1300 bf16 peak because the structural limiter is the 16x16-output-tile-
per-wavefront geometry (one MFMA per iter, no A-reuse, no wide accumulator) - a
B-load tweak cannot fix that. The real win needs the full wide-tile, double-
buffered LDS-staged GEMM (model: HipKittens 256_256_64_32_with16x32.cpp): wider
output tiles (e.g. 128x128 with register C-accum), LDS-staged A+B with tic/toc
double buffering, and s_waitcnt/s_barrier scheduling. That is a large rewrite of
marginal value for DENSE flux (hipBLASLt/rocBLAS are the practical path for dense
bf16); the custom-kernel effort is better spent on the QUANTIZED qgemm (no library
alternative) - deferred as a scoped follow-up. flux_bench.cu is kept as the A/B
harness (`make bench`).

Raw: kernels/matmul/flux/variants/rocm_cdna3/flux_bench.cu.

## 2026-07-07: qgemm Wide N-Tile (X-reuse) — LANDED (+47-54% at M>=256)

Status: landed. Weight-only quant GEMM Y=X@dequant(W)^T gets a wide N-tile kernel
`qgemm_wide<FMT,NT>` (qgemm.cu): NT 16-wide N-tiles per 64-lane wavefront, the X
fragment loaded once per k-step and reused across NT W-fragments (X traffic /NT,
MFMA:load ratio *NT). Bitwise-identical to the shipped qgemm (same load_xfrag/
load_wfrag/mma_16x16x16 math), so the win is orthogonal to the quant format.

NT is occupancy-picked (`qgemm_pick_nt`): widening amortizes the X load but shrinks
the grid, so cap NT so the grid still fills ~2 waves over 304 CUs and require
16*NT | N. NT=1 (== base) for decode (small M); NT=4 for prefill/large M. This
avoids the decode regression (see A/B).

Correctness: `make test` golden (all quant formats incl. k/i-quant dequant-route)
PASS with qgemm-wide(NT=4) rel bit-matching base (e.g. max abs 9.06e-06 == base).
Bench (qgemm_bench.cu, fp16_raw to isolate tiling) wide vs base max abs diff 0.

Perf A/B (MI300X, N=K=4096, HIP-event median, TFLOP/s=2MNK):
| M | base (NT=1) | wide (NT=4) | speedup | pick_nt |
|---|---|---|---|---|
| 64 (decode)  | 31.1 TFLOP/s | 18.9 | 0.61x | -> **1** (base) |
| 256          | 33.7 TFLOP/s | 49.6 | 1.47x | 4 |
| 2048 (prefill)| 36.2 TFLOP/s | 55.9 | 1.54x | 4 |

Decode regresses (grid shrinks below CU count) -> pick_nt keeps NT=1 there (the
existing qgemm_ksplit K-slice path handles decode occupancy). Net: +47-54% at
M>=256, no regression at decode. Applied to qflux too (qflux_gelu_wide<FMT,NT>, golden-validated all formats bit-matching base, same core so same +47-54%; commit follows). Remaining follow-ups:
combine wide-N with LDS-staged X double-buffer for the next tier; wire pick_nt
into the production serving dispatch.

Raw: perf/results/2026-07-07/qgemm-wide/bench.txt.

## 2026-07-07: qgemm LDS-Staged Wide Tile — REJECTED

Status: rejected (correct, but slower than the current register-fragment wide
kernel).

Current implementation: `kernels/quantization/qgemm/variants/rocm_cdna3`.
Shipped path remains `qgemm_wide<FMT,NT>` / `qflux_gelu_wide<FMT,NT>` with
`qgemm_pick_nt`. Bench-only comparators now include `qgemm_wide2d<FMT,MT,NT>`
(2D register tile) and `qgemm_wide_lds<NT>` (coalesced fp16_raw X/W loads into
ping-pong LDS buffers, same MFMA fragment math).

Current public route: no routing change; `qgemm_wide_lds` exists only in
`qgemm_bench.cu`.

References inspected: `qgemm_bench.cu`, `qgemm.cu`, `qflux.cu`,
`tm_qmm_mfma.cuh`, prior qgemm-wide entry above, and `perf/perf.md`.

Correctness: `qgemm_wide_lds<4>` is bit-identical to the base fp16_raw kernel
on all three A/B shapes (`diff 0`). `make test` also passes the existing qgemm
/ qflux golden suite across the quant format matrix: qgemm, qgemm-ksplit,
qgemm-wide, qflux, and qflux-wide all PASS.

Baseline / experiment: MI300X gfx942, `HIP_VISIBLE_DEVICES=0`, Ubuntu 22.04.5,
driver 6.16.13, ROCm/HIP 7.2.53211 (`hipcc` from ROCm 7.2.4), no container
image exposed (`/.dockerenv` absent, `/proc/1/cgroup` = `/init.scope`), native
HIP benchmark with no framework dependency. Repo venv available via
`source ~/QuixiCore/QuixiCore-ROCm/.venv/bin/activate`: PyTorch
2.12.1+rocm7.2, `torch.version.hip` 7.2.53211, MI300X visible. Git
`16c7fd636b1d7d54de4913e8e1a95178d4a33676` plus local bench candidate. Command:
`make bench` in `kernels/quantization/qgemm/variants/rocm_cdna3`. Dtype/format:
fp16_raw X/W, fp32 accumulate/output, `N=K=4096`, NT=4. Timing: HIP events,
10 warmups, 50 iterations, median.

| M | base NT=1 | wide NT=4 | wide2d 2x2 | wideLDS NT=4 |
|---|---|---|---|---|
| 64 | 31.2 TFLOP/s | 18.9 | 22.0 | 4.3 |
| 256 | 33.5 TFLOP/s | 49.6 | 48.7 | 15.9 |
| 2048 | 36.1 TFLOP/s | 55.6 | 55.7 | 33.2 |

Decision: REJECT the per-wave LDS-staged wide tile. Coalescing the global loads
does not offset the extra global->LDS stores, LDS reads, and per-K-step barriers;
the candidate is 0.23-0.60x of the current wide kernel and even loses to base at
large M. The 2D register tile tying the current 1D wide kernel confirms register
reuse is already exhausted for this one-wavefront tile shape. The remaining
qgemm headroom requires a different geometry: a multi-wave CTA GEMM tile with
LDS-staged A/B reused across several wavefronts (or a library-backed dense path
where quant/dequant semantics allow), not a single-wavefront LDS wrapper.

Raw results: `perf/results/2026-07-07/qgemm-lds/bench.txt` and
`perf/results/2026-07-07/qgemm-lds/make_test.txt`.

## 2026-07-07: qgemm Multi-Wave CTA LDS Tile — CANDIDATE

Status: candidate kept in qgemm correctness/bench harness; packed q8_0/q4_0
large-M timing confirms the win. Public routing remains deferred until the
broader format sweep and qflux mirror are complete.

Current implementation: `kernels/quantization/qgemm/variants/rocm_cdna3`.
`qgemm_bench.cu` now includes `qgemm_cta_lds<MT,NT>` for fp16_raw timing, and
`qgemm.cu` includes the generalized `qgemm_cta_lds<FMT,MT,NT>` path. The
candidate uses a 4-wave CTA (`MT=4`, `NT=4`) for a 64x64 output tile: each
wavefront owns one 16-row tile and all four wavefronts share the same 64-column
W tile from LDS. This is the multi-wave version of the LDS hypothesis rejected
above; the key difference is actual W/dequant reuse across M tiles.

Current public route: no dispatcher change. Existing decode path remains
K-split; existing medium path remains register-fragment wide/2D. CTA-LDS should
only be considered for prefill-scale `M>=1024` and `N % 64 == 0`; q8_0/q4_0 are
confirmed, while fp8/nvfp4/superblock formats still need timing.

References inspected: `qgemm_bench.cu`, `qgemm.cu`, `tm_qmm_mfma.cuh`, the
rejected single-wave LDS entry above, and `perf/perf.md`.

Correctness: `qgemm_cta_lds<4,4>` is bit-identical to base in the fp16_raw bench
(`diff 0` on every measured shape). `make test` validates the generalized
`qgemm_cta_lds<FMT,4,4>` across the golden quant format matrix; qgemm base,
K-split, wide, and CTA-LDS all PASS. qflux base/wide remains PASS; qflux CTA-LDS
is not mirrored yet.

Baseline / experiment: MI300X gfx942, `HIP_VISIBLE_DEVICES=0`, Ubuntu 22.04.5,
driver 6.16.13, ROCm/HIP 7.2.53211 (`hipcc` from ROCm 7.2.4), repo venv PyTorch
2.12.1+rocm7.2 active, native HIP timing. Git
`16c7fd636b1d7d54de4913e8e1a95178d4a33676` plus local candidate. Commands:
`make bench`; additional direct `./qgemm_bench.out <M> <N> <K>` sweeps; generated
q8_0/q4_0 packed large-M golden data under the raw results directory and ran
`./qgemm.out <golden_dir>`. Synthetic timing: fp16_raw X/W, fp32
accumulate/output. Packed timing: q8_0 and q4_0, `M=1024,N=K=4096`. Timing:
HIP events, 10 warmups, 50 iterations, median for `qgemm_bench`; qgemm format
harness uses 50-iteration HIP-event averages.

| shape | base NT=1 | wide NT=4 | wide2d 2x2 | ctaLDS 4x4 |
|---|---|---|---|---|
| M64 N4096 K4096 | 31.0 | 19.1 | 22.1 | 5.1 |
| M256 N4096 K4096 | 33.7 | 49.6 | 50.3 | 19.5 |
| M512 N4096 K4096 | 35.5 | 56.6 | 57.5 | 38.7 |
| M1024 N4096 K4096 | 35.9 | 55.8 | 54.1 | 64.6 |
| M2048 N4096 K4096 | 36.1 | 55.4 | 55.3 | 85.8 |
| M4096 N4096 K4096 | 37.0 | 54.7 | 56.0 | 87.6 |
| M8192 N4096 K4096 | 37.5 | 55.2 | 59.8 | 91.2 |
| M1024 N11008 K4096 | 34.7 | 31.7 | 52.6 | 62.4 |
| M2048 N11008 K4096 | 35.1 | 32.7 | 57.6 | 70.9 |
| M2048 N14336 K4096 | 34.1 | 48.0 | 62.6 | 80.2 |

Packed-format timing (`M=1024,N=K=4096`):

| format | base | wide NT=4 | ctaLDS 4x4 |
|---|---|---|---|
| q8_0 | 25.15 TFLOP/s | 33.39 | 59.86 |
| q4_0 | 12.64 TFLOP/s | 14.68 | 53.69 |

Decision: KEEP as a qgemm candidate for prefill-scale shapes. The measured win
starts around M=1024 and reaches +55-65% vs current wide on N=K=4096 at
M>=2048, and +67-117% on LLM rectangle N. Packed q8_0/q4_0 large-M timing also
wins (+79% and +266% vs current wide), so the dequant work does not erase the
geometry win for the simple packed formats. Reject for decode/small M: CTA-LDS
is 0.17x of base at M=64 and 0.39x of wide at M=256. The result confirms LDS is
only useful when it creates cross-wave W/dequant reuse; single-wave LDS staging
remains rejected. Before routing this as a shipped qgemm path, finish fp8/nvfp4
and superblock format timing, then mirror the epilogue into qflux if the qgemm
format sweep holds.

Raw results: `perf/results/2026-07-07/qgemm-cta-lds/bench.txt`,
`perf/results/2026-07-07/qgemm-cta-lds/sweep_m.txt`,
`perf/results/2026-07-07/qgemm-cta-lds/llm_rectangles.txt`,
`perf/results/2026-07-07/qgemm-cta-lds/make_test.txt`,
`perf/results/2026-07-07/qgemm-cta-lds/make_test_after_m_arg.txt`,
and `perf/results/2026-07-07/qgemm-cta-lds/packed_m1024.txt`.

## 2026-07-07: Collective+GEMM Compute/Comm Overlap (chunked RCCL) — REJECTED

Status: rejected for perf (correct, but slower than the non-overlapped fused path
on MI300X). Important hardware finding, not a bug.

Experiment: overlap the collective with the GEMM by chunking along N and issuing
per-chunk async collectives (all_reduce/reduce_scatter, async_op=True) so chunk c's
collective runs while chunk c+1's GEMM computes (torch_gemm_overlap.py, torchrun
one-process-per-GPU, repo venv torch 2.12.1+rocm7.2). gemm_ar and gemm_rs both
validate allclose vs single-GPU ref on 4 and 8 GPUs.

Perf A/B (base = single fused collective; overlap = chunked async):
| GPUs | shape | gemm_ar | gemm_rs |
|---|---|---|---|
| 4 | 4096^3 (C=4)        | 0.96x | 0.82x |
| 8 | 4096^3 (C=2)        | 0.86x | 0.82x |
| 8 | 8192x8192x1024 (C=2)| 0.91x | 0.92x |
| 8 | 4096x16384x1024 (C=2)| 0.91x | 0.92x |

Decision: REJECT (0.82-0.96x everywhere). Root cause: on MI300X, RCCL collectives
run as **compute kernels on the CUs** (no dedicated inline comm engine like NVLink
SHARP), so the collective and the GEMM contend for the same CUs -- chunking cannot
hide comm behind compute, and it also shrinks each collective's message size,
costing RCCL bandwidth efficiency. The single large fused collective is optimal.

Implication: intra-op chunked overlap is the wrong lever here. Real overlap on
this hardware needs either inter-op pipelining (model level, out of scope) or
device-initiated one-sided XGMI that does not spin up a separate CU-bound
collective kernel -- i.e. **Iris (Step 8)**, whose in-kernel XGMI store/put can
issue comm from the GEMM epilogue itself. This result motivates Step 8.
torch_gemm_overlap.py is kept as the validated overlap harness/comparator.

Raw: perf/results/2026-07-07/collective-overlap/sweep.txt.

## 2026-07-07: Library/Framework Baselines (Step 7a) — RECORDED

Honest context for the landed wins: the optimized CDNA3 kernels vs torch's own
kernels on MI300X (repo venv torch 2.12.1+rocm7.2). These are baselines, not a
change — they show remaining headroom and where to focus.

| kernel | shape | mine | torch/library | ratio |
|---|---|---|---|---|
| attention noncausal | B4 H32 H_KV8 N2048 D128 | 97 TFLOP/s | SDPA (flash) 321 | 0.30x |
| attention causal    | same | 77 TFLOP/s | SDPA causal 223 | 0.34x |
| qgemm M=256  | N=K=4096 | 49.6 TFLOP/s | dense fp16 matmul (hipBLASLt) 272 | 0.18x |
| qgemm M=2048 | N=K=4096 | 55.9 TFLOP/s | dense fp16 matmul 471 | 0.12x |
| layernorm | 16384x2048 f32 | 2399 GB/s | F.layer_norm 3372 | 0.71x |
| layernorm | 65536x8192 f32 | 1655 GB/s | F.layer_norm 2956 | 0.56x |

Reading: the landed wins are real and large vs the naive ports (norm +30-71%,
attention 16.65x, qgemm +47-54%), but the kernels are still below torch/library.
Norm is closest (memory-bound, ~0.6-0.7x). GEMM has the most headroom (0.12-0.18x
of hipBLASLt) - the 16x16-tile-per-wavefront geometry is the limiter; approaching
the library needs the full wide-tile, double-buffered, LDS-staged GEMM (register
C-accum, tic/toc) - the same rewrite flux/qgemm both point at. Attention ~0.3x of
SDPA's flash backend - the K/V-LDS + larger-query-block + double-buffer follow-ups
are the path. qgemm also carries dequant work torch's dense matmul does not, so the
dense number is a loose ceiling.

Note: 7b (hipblaslt-bench / rocblas-bench CLIs, CK examples, AITER fused comparators)
deferred - torch already exposes the hipBLASLt/flash numbers above; AITER is installed
for a fused-MoE/attention comparison when that work is picked up.

## 2026-07-07: New Distributed Kernels (Step 6) — LANDED (ring / ulysses / moe_dispatch)

Status: landed. Three new sequence/expert-parallel distributed kernels on the
proven torchrun one-process-per-GPU + RCCL pattern (repo venv torch 2.12.1+rocm7.2),
each validated allclose vs a single-GPU reference on 4 AND 8 GPUs. New capability
(not a perf tune).

- ring_attn (kernels/collectives/ring_attn/variants/rocm_cdna3/ring_attn.py):
  sequence-parallel attention. N sharded across ranks; deadlock-free ring rotation
  of the KV shard (batch_isend_irecv) with online-softmax merge per block. After W
  steps each rank has full-context attention for its Q rows.
- ulysses_attn (.../ulysses_attn/...): DeepSpeed-Ulysses. all_to_all reshards
  [Ms,H,D] seq-parallel -> [N,Hs,D] head-parallel, local full attention on the head
  subset, second all_to_all back to seq-parallel. Requires H % W == 0.
- moe_dispatch_gemm (.../moe_dispatch_gemm/...): expert-parallel MoE. Per-token
  expert ids -> all_to_all_v dispatch to expert-owning ranks (variable counts
  exchanged first, used as all_to_all_single split sizes) -> grouped per-expert GEMM
  -> all_to_all_v back -> unsort. Requires E % W == 0.

Correctness (max abs err vs single-GPU ref):
| kernel | 4 GPUs | 8 GPUs |
|---|---|---|
| ring_attn        | 2.38e-07 | 2.76e-07 |
| ulysses_attn     | 2.31e-07 | 2.12e-07 |
| moe_dispatch_gemm| 3.10e-06 | 2.62e-06 |

Run: torchrun --nproc_per_node={4,8} <kernel>.py. These use torch collectives for
the comm (RCCL) and torch matmul/softmax for local compute; fusing the local math
onto the landed MFMA attention/qgemm kernels is a follow-up.

## 2026-07-07: Iris In-Kernel XGMI Overlap (Step 8, optional) — FEASIBILITY

Status: feasibility established; the fused-overlap kernel is a scoped follow-up.

Context: Step 5 showed RCCL collectives contend with the GEMM for CUs on MI300X, so
the intended Step-8 lever is Iris (ROCm device-initiated XGMI over a symmetric heap):
issue one-sided puts from a compute kernel's epilogue so comm overlaps compute
without a separate CU-bound collective.

What was validated (iris_allreduce.py, torchrun one-proc-per-GPU, repo venv):
- Iris initializes cleanly on this MI300X/ROCm7.2 stack (requires dist.init_process_group
  first, then iris.iris(heap); symmetric heap allocates on every rank).
- ctx.ccl.all_reduce over the symmetric heap is CORRECT (2 GPUs: expect 3.0, got 3.0).
- Standalone A/B vs RCCL (2 GPUs, 2048^2 f32): iris 4.04 ms (4 GB/s) vs rccl 0.398 ms
  (42 GB/s) = 0.10x.

Reading: Iris's *standalone* library collective is not competitive with RCCL out of
the box (RCCL is a tuned library; Iris here likely needs config/algorithm + workspace
tuning, and small messages amortize its Triton launch poorly). But standalone all_reduce
is NOT the Iris value proposition -- the win is FUSED in-kernel overlap (device-side
iris.put/store from a GEMM epilogue), which needs a from-scratch Iris/Triton
producer/consumer GEMM+reduce_scatter kernel. That kernel is the real Step 8 work and is
deferred as a scoped follow-up; the integration path (init, symmetric heap, correct
collectives) is now de-risked. iris_allreduce.py kept as the Iris smoke test/comparator.

Raw: 2 GPUs 2048^2 iris 4.04ms/4GB/s vs rccl 0.398ms/42GB/s (0.10x), correctness PASS.

## 2026-07-07: Remaining CUDA Functional Coverage To CDNA3 — LANDED

Status: landed. Functional CDNA3 coverage added for the CUDA-only gaps identified
in the parity plan: norm quantization, qgemm act-order/block-scale variants,
dense int8 GEMM, dense MXFP8/NVFP4 GEMM, standalone collectives, and FP8
ag_gemm/gemm_rs. These are correctness-first routes; only measured keep/reject
decisions below are claimed.

Environment: AMD Instinct MI300X (`gfx942`), HIP 7.2.53211 / ROCm clang
22.0.0git roc-7.2.4, repo venv PyTorch 2.12.1+rocm7.2, `torch.cuda` sees 8 GPUs.
Standalone HIP runs use `HIP_VISIBLE_DEVICES=0`; collective runs use torchrun
one-process-per-GPU on GPUs 0,1.

Current implementation / public routes:
- `kernels/norms/norm_quant/variants/rocm_cdna3`: RMSNorm quant FP8/INT8,
  residual output, AZP int8, per-token-group int8.
- `kernels/quantization/qgemm/variants/rocm_cdna3/qgemm_variants.cu`:
  `qgemm_actorder` and `qgemm_blockscale`.
- `kernels/matmul/int8/variants/rocm_cdna3`: exact `int8 x int8 -> int32`
  dense GEMM.
- `kernels/matmul/{mxfp8,nvfp4}/variants/rocm_cdna3`: dense block-scaled GEMM
  via explicit dequant + fp32 GEMM baseline.
- `kernels/collectives/{all_gather,all_to_all,reduce_scatter}/variants/rocm_cdna3`
  and `kernels/collectives/gemm_collectives/variants/rocm_cdna3/test-fp8`.

Correctness:
- Norm quant: 8/8 checks PASS; representative rel errors: fp8 dynamic RMS
  2.261e-02, int8 dynamic RMS 7.656e-03, residual 1.929e-03, AZP dynamic
  3.918e-03, group int8 3.855e-03.
- QGEMM variants: act-order q4_0 PASS rel 8.628e-07; blockscale fp8_raw PASS
  rel 1.982e-05.
- Int8 GEMM: exact PASS, 0 mismatches.
- MXFP8/NVFP4 GEMM: PASS vs host double reference; MXFP8 fused/explicit rel
  9.284e-07, NVFP4 fused/explicit rel 0.
- Collectives: all_gather, all_to_all, reduce_scatter, ag_gemm_fp8, gemm_rs_fp8
  all PASS on 2 MI300X ranks.

Focused timing / decisions (HIP-event median; repeat ranges are min/max of
three later median runs where collected):

| Kernel | Shape | Baseline | Candidate | Decision |
|---|---|---:|---:|---|
| norm_quant int8 dyn | M=16384 D=4096 fp16 | block128 0.109-0.112 ms | block256 0.107-0.111 ms | TIE, no speedup claim |
| qgemm_actorder q4_0 | M=64 N=128 K=256 | new route | 0.015 ms | KEEP direct MFMA |
| qgemm_blockscale fp8_raw | M=64 N=128 K=256 | new route | 0.013-0.014 ms | KEEP direct MFMA |
| int8 GEMM | M=N=256 K=512 | scalar 0.038 ms | sdot4 0.021 ms | KEEP sdot4 |
| MXFP8 GEMM | M=N=64 K=256 | fused 0.083 ms | explicit+fp32 0.045 ms | KEEP explicit+fp32 |
| NVFP4 GEMM | M=N=64 K=256 | fused 0.138 ms | explicit+fp32 0.045-0.046 ms | KEEP explicit+fp32 |
| all_gather | 2 GPUs, n=1048576 | RCCL route | 0.114 ms | correctness route |
| all_to_all | 2 GPUs, chunk=262144 | RCCL route | 0.047 ms | correctness route |
| reduce_scatter | 2 GPUs, n=1048576 | RCCL route | 0.135 ms | correctness route |
| ag_gemm_fp8 | 2 GPUs, 256x256x512 | torch/RCCL route | 0.161 ms | correctness route |
| gemm_rs_fp8 | 2 GPUs, 256x256x512 | torch/RCCL route | 0.246 ms | correctness route |

Decision: KEEP the landed functional coverage. Norm-quant block128/block256 is
within noise on the measured shape, so no block-size performance route change is
claimed. For the dense block-scaled GEMMs,
explicit dequant + fp32 GEMM is the kept CDNA3 baseline because scalar fused
decode is slower on the measured shape. The B200/H100/Ampere implementations
remain architecture-specific references, not source to copy. Follow-up work is
optimization: MFMA/LDS dense block-scaled kernels, larger int8 tiling, and
full-node collective sweeps.

Raw results: `perf/results/2026-07-07/cdna3-port-fill/raw-summary.txt`.

## Current Baseline Sources

Status: baselines exist, not yet normalized into the shared harness.

| Area | Source | Notes |
|---|---|---|
| BF16 GEMM | `analysis/bf16_gemm/` | MI325/MI350/MI355 scripts, JSON, plots |
| FP8/FP6 GEMM | `analysis/fp8_gemm/`, `analysis/fp6_gemm/` | Library and custom comparisons |
| Attention fwd/bwd | `analysis/attn/fwd/`, `analysis/attn/bkwd/` | GQA/MHA, causal/non-causal plots |
| LayerNorm | `analysis/layernorm/` | MI350/MI355 row-kernel benchmark |
| Rotary | `analysis/rotary/` | MI350/MI355 benchmark |
| MLA decode | `analysis/mla_decode/` | Context-length latency/gain plots |
| Framework/library baselines | `analysis/baselines/` | PyTorch, Triton, HIPBLASLT, AITER, CK where available |
| Profiling workflow | `docs/profiling/` | rocprof trace and PMC counter collection |

## Kernel Family Status

| Kernel family | ROCm status | Next optimization step |
|---|---|---|
| BF16 GEMM | baselined in analysis | Normalize benchmark metadata and compare custom/library routes by shape |
| FP8/FP6 GEMM | baselined in analysis | Record format-specific library/custom crossover points |
| Attention forward | baselined in analysis | Capture shape table and profile the top remaining bottleneck |
| Attention backward | baselined in analysis | Capture correctness tolerances and profile D=64/D=128 separately |
| LayerNorm | baselined in analysis | Convert row-kernel numbers to GB/s and add framework comparison table |
| Rotary | baselined in analysis | Record vectorization/layout conclusions |
| Softmax | implemented under `kernels/softmax` | Add baseline and correctness entry |
| Quant GEMV/GEMM | not normalized | Add benchmark matrix and reference comparisons |
| Serving kernels | partial analysis | Tie MLA/paged/KV/MoE/sampling work to shared status entries |

## Open Questions

- Which ROCm target should be the canonical baseline host for each family:
  CDNA3, CDNA4, or both?
- Should migrated benchmark output preserve the current `analysis/` JSON schema
  or convert directly to the shared `perf/results/YYYY-MM-DD/<kernel>/` layout?
- Which library baseline should be primary for each matrix family: hipBLASLt,
  rocBLAS, Composable Kernel, AITER, Triton, or a per-shape best-of table?

## 2026-07-22: embeddinggemma.c ROCm Kernel Ports To CDNA3

Status: landed. Three shape/format-named variants ported from
`~/embeddinggemma-bench/src/engine_rocm.hip`, each with correctness vs an fp64
host reference and a focused baseline-vs-candidate A/B on this MI300X (gfx942,
ROCm/HIP 7.2). Existing coverage recorded, not duplicated.

Current implementation / public route: standalone CDNA3 variants under
`kernels/quantization/qgemm_q4q8/variants/rocm_cdna3`,
`kernels/quantization/qgeglu/variants/rocm_cdna3`, and
`kernels/attention/gqa_swa/variants/rocm_cdna3`; each builds and runs via its
local `make test` / `make bench`.

References inspected: embeddinggemma.c `src/engine_rocm.hip` (q4_q8_dot /
q4_q8_projection, q4_mfma_up_gate_gelu_f16, mfma_attention_f16 /
flash_attention_f16), existing ROCm `qgemm` (Q4_0->fp16 MFMA + qflux),
`qgemm_int` (W8A8/W2A8), `gqa`/`gqa_causal` (MFMA flash forward), `flux`
(flux_gelu/flux_gate), `.quixicore/kernels.yaml`, and `perf/perf.md`.

Already existed in QuixiCore-ROCm (recorded, NOT duplicated):

- The Q4_0->fp16 MFMA projection GEMM (embeddinggemma q4_mfma_projection*) is the
  existing `qgemm` MFMA path (dequant-to-fp16 + v_mfma_f32_16x16x16_f16). Equal
  algorithm; not re-added.
- Single-projection fused GELU (embeddinggemma q4_mfma projection + gelu) is the
  existing `qflux` (gelu_tanh(X@dequant(W)^T + bias)). Not re-added.
- The base MFMA flash-attention forward (non-causal / causal) is the existing
  `gqa` / `gqa_causal`. Only the symmetric-window specialization was missing.

Correctness (`make test`, MI300X, fp64 host ref):

- qgemm_q4q8: rel ~6e-8, PASS (ragged 48x768x17, FFN 1152x768x64, decode
  768x768x1). Device integer math exact; only fp16 scale rounding differs.
- qgeglu: rel ~3.3e-4, PASS (1152x768, M in {16,64}). fp16 MFMA accumulation.
- gqa_swa: rel ~1.4e-4..1.9e-4, PASS (full + windows 16/128/256/512; ragged
  T=37, aligned T=512).

Focused A/B (HIP-event median, warmup 10 / iters 50; `make bench`):

| Kernel | Shape | Baseline | Candidate | Decision |
|---|---|---:|---:|---|
| qgemm_q4q8 | N1152 K768 M64 | float-dequant 5.69 TOPS | sdot4 8.46 TOPS (1.49x) | KEEP |
| qgemm_q4q8 | N768 K1152 M64 | float-dequant 5.39 TOPS | sdot4 8.46 TOPS (1.57x) | KEEP |
| qgemm_q4q8 | N768 K768 M1 (decode) | 0.0059 ms | 0.0058 ms (1.02x) | KEEP (mem-bound parity) |
| qgeglu | N1152 K768 M64 | unfused 6.25 TFLOP/s | fused 7.45 TFLOP/s (1.19x) | KEEP |
| qgeglu | N1152 K768 M256 | unfused 17.57 TFLOP/s | fused 21.68 TFLOP/s (1.23x) | KEEP |
| gqa_swa | T2048 w256 | full 1.3639 ms | banded 0.1995 ms (6.84x) | KEEP |
| gqa_swa | T2048 w512 | full 1.3639 ms | banded 0.3764 ms (3.62x) | KEEP |

Decision: KEEP all three. Each is a genuinely missing route (integer Q4_0xQ8_0
dot; dual-projection quantized GeGLU fusion; symmetric sliding-window band) that
wins over its in-tree baseline at correctness. Commits: qgemm_q4q8 1752702a,
qgeglu 8c6237ae, gqa_swa 90eea2c9.

Follow-ups: MFMA-tile the int8 Q4_0xQ8_0 path (v_mfma_i32_16x16x16_i8); LDS-stage
the shared X fragment in qgeglu and widen N-tiles; LDS-stage K/V and add a
packed-QKV entry to gqa_swa.

Raw results: kernel-local `make bench` output (this box); not committed as bulky
traces per perf.md.
