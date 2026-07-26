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

## 2026-07-07: Metal Gap Kernel Ports To CDNA3

Status: landed functional coverage; optimization deferred except for the
dense-GEMM direct-vs-LDS A/B.

Current implementation: CDNA3 HIP variants for the Metal-only contract surfaces:
`qk_norm_rope`, `qgemm_int`, `quant_rt` group variants, `matmul_custom`,
`gemm_staged`, and `utils/marginal`.

Current public route: standalone operation variants discovered by
`scripts/build|test|bench --arch cdna3` under:

- `kernels/norms/qk_norm_rope/variants/rocm_cdna3`
- `kernels/quantization/qgemm_int/variants/rocm_cdna3`
- `kernels/quantization/quant_rt/variants/rocm_cdna3`
- `kernels/matmul/{matmul_custom,gemm_staged}/variants/rocm_cdna3`
- `kernels/utils/marginal/variants/rocm_cdna3`

References inspected: `../QuixiCore-Metal/kernels/{norms/qk_norm_rope,
quantization/qgemm_int,quantization/quant_rt,matmul/matmul_custom,
matmul/gemm_staged,utils/marginal}`, existing ROCm `qgemv`, `qgemm`,
`norm_quant`, `int8`, and `flux` CDNA3 ports, `.quixicore/kernels.yaml`, and
`perf/perf.md`.

Correctness (MI300X/gfx942, ROCm/HIP 7.2.4, PyTorch 2.12.1+rocm7.2 in
`~/QuixiCore/QuixiCore-ROCm/.venv`, command `make -C <variant> test`):

- `qk_norm_rope`: D=64 split, D=128 interleaved, D=256 Gemma all PASS; rel
  2.29e-03 to 2.45e-03 vs host oracle; V copy max <=3.91e-03 on test shapes.
- `qgemm_int`: W8A8, W8A8-AZP, and W2A8 all PASS; rel <=2.67e-04.
- `quant_rt`: per-group FP8 UE8M0 exact encoded codes/scales, per-group int8
  <=0.5 code step, token int8 AZP rel 7.42e-03.
- `matmul_custom`: f32 rel 1.81e-07, bf16 rel 1.40e-03.
- `gemm_staged`: f32 rel 1.84e-07, bf16 rel 1.41e-03.
- `marginal`: tau_tail, packbits big/little, segment_packbits, permute_cols all PASS.

Focused timing (HIP-event median; warmups/iters are harness-local: qk and
quant/marginal 20/100, qgemm_int and GEMMs 10/50 or 5/30; command
`make -C <variant> bench`):

| Kernel | Shape / dtype-format | Baseline/current | Candidate timing | Decision |
|---|---|---:|---:|---|
| qk_norm_rope | T=4096 HT=8 D=64 bf16 split | no prior ROCm surface | 0.0315 ms, 266 GB/s | KEEP coverage |
| qk_norm_rope | T=4096 HT=8 D=128 bf16 interleaved | no prior ROCm surface | 0.0336 ms, 499 GB/s | KEEP coverage |
| qk_norm_rope | T=4096 HT=8 D=256 bf16 Gemma | no prior ROCm surface | 0.0380 ms, 883 GB/s | KEEP coverage |
| qgemm_w8a8 | N=512 M=64 K=1024 int8/half scales | no prior ROCm surface | 0.0208 ms, 3.23 TOPS | KEEP correctness route |
| qgemm_w8a8_azp | N=512 M=64 K=1024 int8/f32 act scale | no prior ROCm surface | 0.0211 ms, 3.18 TOPS | KEEP correctness route |
| qgemm_w2a8 | N=512 M=64 K=1024 BitNet W2A8 | no prior ROCm surface | 0.0216 ms, 3.10 TOPS-equivalent | KEEP correctness route |
| quant_rt group fp8 | rows=4096 D=512 G=128 ue8m0 | no prior group variant | 0.0155 ms, 678 GB/s | KEEP coverage |
| quant_rt group int8 | rows=4096 D=512 G=128 | norm_quant had related group int8 | 0.0139 ms, 754 GB/s | KEEP coverage |
| quant_rt token int8 AZP | rows=4096 D=512 | norm_quant had related AZP | 0.0115 ms, 909 GB/s | KEEP coverage |
| matmul_custom | 512^3 f32 | direct baseline | 0.0636 ms, 4.22 TFLOP/s | KEEP coverage |
| gemm_staged | 512^3 f32 | matmul_custom direct 0.0636 ms | 0.0286 ms, 9.39 TFLOP/s | KEEP LDS staging for this surface |
| matmul_custom | 512^3 bf16 | direct baseline | 0.0649 ms, 4.13 TFLOP/s | KEEP coverage |
| gemm_staged | 512^3 bf16 | matmul_custom direct 0.0649 ms | 0.0563 ms, 4.77 TFLOP/s | KEEP LDS staging for this surface |
| marginal tau_tail | T=4096 H=4 D=16 f32 | no prior ROCm surface | 0.0062 ms, 1013 GB/s | KEEP coverage |
| marginal packbits | n=4194307 uint8/bool | no prior ROCm surface | 0.0061 ms, 684 GB/s input | KEEP coverage |

Decision: KEEP all functional ports so Metal contract coverage exists in the
ROCm tree. For dense GEMM, the LDS-staged variant beats the direct baseline on
the measured 512^3 f32 shape and modestly improves bf16; this does not change
the existing tuned dense GEMM routes and is not a speedup claim over MFMA,
hipBLASLt, rocBLAS, or PyTorch. Follow-ups: replace the correctness-first dense
GEMM surfaces with MFMA-backed implementations if they become public routes,
tile `qgemm_int` beyond one CTA per output element, and add framework/library
baselines for each surface.

Raw results: `perf/results/2026-07-07/metal-gap-ports/`.

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

## 2026-07-25: Metal/CPU Parity Program — Phase 0 Harness Infrastructure

Status: landed scaffolding (no performance claim).

Diffed this backend against the Metal and CPU operation manifests, which publish
a much larger surface than QuixiCore-CUDA's family-level metadata. Result: 178
operations in 13 groups have no CDNA3 implementation, including two families
with no directory at all (vision, convolution/audio). Inventory and per-kernel
work list: `docs/metal-cpu-parity-gaps.md`.

The program commits to a focused perf run for **every** one of those 178
kernels, so Phase 0 builds the infrastructure that makes each run cheap rather
than bespoke:

- `kernels/common/cdna3_harness.cuh` — fp64-oracle comparison with tolerances
  taken from `../registry/tolerances.yaml`, deterministic host RNG (including an
  adversarial generator for zero/denormal/huge/signed cases), wave64 reductions
  (sum/max/min/inclusive-scan/argmax), and a timing loop reporting median, min,
  max, spread, GB/s, and TFLOP/s.
- `kernels/common/harness_selftest.cu` — validates the harness itself.
- `perf/harness/run_kernel_bench.sh` — runs a kernel harness, archives raw
  output under `perf/results/<date>/<label>/` with GPU/ROCm/HIP/commit/container
  provenance, and emits a pre-filled notebook entry.
- `perf/configs/shapes.yaml` — mirrors `../registry/benchmark-shapes.yaml` and
  adds per-family shape sets for the families being brought to parity.

Two decisions worth recording, because they affect every downstream verdict:

1. `Comparison::pass()` requires **both** an elementwise rtol/atol check and
   aggregate rel-L1 + cosine bounds. Elementwise alone misses systematic drift
   at loose quantized tolerances; aggregate alone misses a single bad lane. The
   self-test asserts that `compare()` rejects a single bad element, a 2%
   systematic scale error, a NaN, and a size mismatch — a comparison helper that
   cannot fail is worse than none.
2. `wave_reduce_argmax` breaks ties toward the **lower** index. This is
   contractual for LM-head/argmax operations (Metal and CPU both specify
   lower-token tie breaking) and is covered by a dedicated all-tie test rather
   than left to random data.

Correctness: `make -C kernels/common test` — ALL PASS, 12 checks on MI300X.
Reductions verified against an fp64 host oracle at a deliberately ragged 37x501
shape (neither dimension a multiple of the 64-wide wavefront).

Environment:
  GPU: AMD Instinct MI300X (gfx942:sramecc+:xnack-), 304 CUs, 64 KB LDS/block
  ROCm: 7.2.4   HIP: 7.2.53211-97f5574fe2
  Container: bare metal (no container)
  Command: HIP_VISIBLE_DEVICES=0 make -C kernels/common test

Also wired `scripts/test kernels` to run the harness self-test before the kernel
sweep. Kernel discovery only descends into `variants/rocm_<arch>/`, so
`kernels/common` was invisible to it; shared infrastructure that no sweep
exercises goes stale silently.

Baseline / Experiments: none — this entry is scaffolding and claims no speedup.
`AGENTS.md` permits docs/metadata commits to skip the kernel perf run provided
they make no performance claim. This one makes none.

Decision: keep. Phase 1 (attention and RoPE variants) is next; the first
measured kernel entries follow there.

Open questions: none blocking. The CDNA4 variant gap is out of scope and tracked
separately — MI300X is the only hardware here, and an unmeasured gfx950 claim
would violate the gate.

Raw results: perf/results/2026-07-25/common/

## 2026-07-25: cross_attention (Metal/CPU parity Phase 1)

Status: landed.

Current implementation: `kernels/attention/cross_attention/variants/rocm_cdna3`
Current public route: `.quixicore/kernels.yaml` operation `cross_attention`
(family `attention`).

References inspected: `../QuixiCore-CPU/kernels/attention/cross_attention_ref.cpp`
(semantic source), `kernels/attention/gqa/variants/rocm_cdna3/attn_mfma.cuh`
(validated MFMA fragment layout), `../QuixiCore-CPU/docs/sibling-port-matrix.md`
(published contract: per-batch key lengths, optional score bias, automatic or
explicit scale, score softcap, D64/D128/D256).

First kernel of the 178-operation Metal/CPU parity program
(`docs/metal-cpu-parity-gaps.md`). ROCm had no cross-attention: the landed
self-attention kernel assumes one sequence length and `N % 16 == 0`, neither of
which holds here.

Correctness: **50/50 PASS** on MI300X vs. an fp64 host oracle mirroring the CPU
reference. Both the MFMA candidate and the baseline are checked on every case.
  Tolerance: bf16 (rel 4e-3, cosine 0.99999) and fp16 (rel 2e-3, cosine 0.999995)
  from `../registry/tolerances.yaml`. Observed rel ~2.2e-3 / cosine ~0.999997
  (bf16) and rel ~2.6e-4 / cosine ~0.99999996 (fp16).
  Shapes: MHA and GQA 4:1; ragged per-batch key lengths; an over-long key length
  that must clamp; an empty batch item; bias; softcap 30; bias + softcap 20;
  explicit scale; Lq=1 decode; Lk=2048; D 64/128/256; bf16 and fp16.

Three contract details that a naive port gets wrong, all now covered by tests:

1. Bias is added to the **scaled** score and the softcap is applied **after** the
   bias. Any other order changes results when both are present.
2. A batch item with zero valid keys must emit an all-zero row, not divide by a
   zero denominator.
3. On the first key tile the running max is `-inf`, and `exp(-inf - -inf)` is
   `NaN`. The correction factor is special-cased to 0 there. Without this the
   kernel silently produces NaNs on exactly the ragged inputs cross-attention
   exists to serve.

Baseline: one wavefront per query row, head dim split across lanes, one
wavefront dot-product per key. Same math; re-reads K/V once per query.
Experiments: single factor changed — MFMA BQ=BK=16 tiling so each K/V tile is
read once per 16 queries, with QK^T and P@V on v_mfma_f32_16x16x16bf16_1k.

| Shape (B,Hq,Hkv,Lq,Lk,D) | Baseline | Candidate | Speedup |
|---|---:|---:|---:|
| prefill 4,32,8,512,512,64 | 2.5900 ms (3.3 TFLOP/s) | 0.2782 ms (30.9 TFLOP/s) | 9.31x |
| prefill 4,32,8,512,512,128 | 3.3491 ms (5.1 TFLOP/s) | 0.5347 ms (32.1 TFLOP/s) | 6.26x |
| prefill 2,32,8,512,2048,128 | 7.5734 ms (4.5 TFLOP/s) | 1.1357 ms (30.3 TFLOP/s) | 6.67x |
| enc-dec 8,16,4,128,1024,128 | 1.8896 ms (4.5 TFLOP/s) | 0.4714 ms (18.2 TFLOP/s) | 4.01x |
| ragged 4,16,4,517,1031,128 | 0.9731 ms (17.9 TFLOP/s) | 0.3998 ms (43.7 TFLOP/s) | 2.43x |
| bias 4,16,4,256,1024,128 | 1.8694 ms (4.6 TFLOP/s) | 0.5813 ms (14.8 TFLOP/s) | 3.22x |

Environment:
  GPU: AMD Instinct MI300X (gfx942), device index 1 (idle; the sweep held GPU 0)
  ROCm: 7.2.4   HIP: 7.2.53211-97f5574fe2
  Container: bare metal (no container)
  Commit: 7d969a1d (working tree dirty — kernel not yet committed at measure time)
  Command: HIP_VISIBLE_DEVICES=1 make -C kernels/attention/cross_attention/variants/rocm_cdna3 bench
  Warmups/iterations: w10/i50, HIP-event median; worst spread 1.07x

**Measurement hygiene — a result that inverted under contention.** The first A/B
was taken while an unrelated kernel sweep held GPU 0. It reported the ragged
shape as a **0.64x regression** with a 4.12x min/max spread, and two other shapes
showed 3.1x and 5.1x spread. Re-measured on an idle GPU the same ragged shape is
a **2.43x win** at 1.04x spread. The median was measuring contention, not the
kernel. `perf/harness/run_kernel_bench.sh` now warns when any reported spread
exceeds 1.2x, and the Makefile no longer hardcodes `HIP_VISIBLE_DEVICES=0` so a
run pinned to an idle device records that device index truthfully rather than
claiming a device it did not use.

Decision: **KEEP**. The win holds on every measured shape and is largest where
K/V reuse dominates. No shape regressed.

Open questions / next levers: LDS double-buffering of K/V tiles (the loop
currently loads the next tile after `__syncthreads()`, idling the MFMA units);
larger BQ (32/64 queries per wavefront) for arithmetic intensity — the same
lever already noted for the self-attention kernel at 0.30x of SDPA flash. fp32
storage and a packed-QKV entry point are deferred until a caller needs them.

Raw results: perf/results/2026-07-25/attention-cross_attention/

## 2026-07-25: biased_attention (Metal/CPU parity Phase 1)

Status: landed.

Current implementation: `kernels/attention/biased_attention/variants/rocm_cdna3`
Current public route: `.quixicore/kernels.yaml` operation `biased_attention`.

References inspected: `../QuixiCore-CPU/kernels/attention/attention_extended_ref.cpp`
(~L357, semantic source), `kernels/common/cdna3_mfma.cuh`, the `cross_attention`
entry above.

Correctness: **50/50 PASS** on MI300X vs. an fp64 host oracle. MHA and GQA 4:1;
per-head bias; banded, causal, and one-fully-masked-row masks; bias+mask; ragged
37x101; negative and positive explicit scale; Lq=1 decode; D 32/64/128/256
(32 is the Swin head dim); bf16 and fp16.

Two contract details differ from `cross_attention` and are pinned by tests:
the bias is indexed **per head** while the mask is a single `[Lq, Lk]` plane
**shared** across heads; and automatic scale triggers on `scale == 0`, not
`scale > 0`, so a negative explicit scale is legal and must be honoured.

Baseline: one wavefront per query row, wavefront dot-product per key — and it
**skips masked keys outright**.
Experiments: two, run in sequence.

**Experiment 1 — MFMA BQ=BK=16 tiling (as for `cross_attention`).** Won on dense
shapes but **lost on sparse masks**: a ±8 banded mask at Lq=Lk=1024 measured
**0.70x, a real regression**. Cause is structural, not incidental — the tiled
kernel walked all Lk tiles regardless of the mask, doing roughly 64x the
necessary work, while the baseline touches ~17 keys per query.

**Experiment 2 — masked-tile skipping.** The wavefront reads the 16x16 mask tile
(256 bytes), OR-reduces liveness, and skips the K/V load plus D/16 MFMAs when
nothing is live. The block is exactly one wavefront, so the reduced predicate is
uniform and `continue` cannot desynchronize the loop's `__syncthreads()`.

| Shape (Hq,Hkv,Lq,Lk,D) | Baseline | Candidate | Speedup |
|---|---:|---:|---:|
| 32,8,512,512,128 + bias | 1.3981 ms (3.1 TFLOP/s) | 0.2962 ms (14.5 TFLOP/s) | 4.72x |
| 32,8,512,2048,128 + bias | 5.8754 ms (2.9 TFLOP/s) | 1.1525 ms (14.9 TFLOP/s) | 5.10x |
| 16,4,1024,1024,128 banded | 0.3549 ms (24.2 TFLOP/s) | 0.0980 ms (87.7 TFLOP/s) | 3.62x |
| 32,8,512,512,128 causal | 0.8525 ms (5.0 TFLOP/s) | 0.2942 ms (14.6 TFLOP/s) | 2.90x |
| 32,8,512,512,64 + bias | 0.9856 ms (2.2 TFLOP/s) | 0.2025 ms (10.6 TFLOP/s) | 4.87x |

The banded case went **0.70x -> 3.62x** on experiment 2 alone, and is now the
fastest absolute result in the table (87.7 TFLOP/s) because the tile skip lets
the kernel do proportionally less work as the mask gets sparser.

Environment:
  GPU: AMD Instinct MI300X (gfx942), device index 1 (idle)
  ROCm: 7.2.4   HIP: 7.2.53211-97f5574fe2
  Container: bare metal (no container)
  Command: HIP_VISIBLE_DEVICES=1 make -C kernels/attention/biased_attention/variants/rocm_cdna3 bench
  Warmups/iterations: w10/i50, HIP-event median; worst spread 1.15x

### Harness change: a tolerance that could not be met

The causal 512x512 case initially failed with exactly one bad element out of
2,097,152: `got -1.3671875 vs ref -1.37182645`, absolute error 4.6e-3.

Not a kernel bug. bf16 keeps 8 mantissa bits, so near magnitude 1.4 representable
values are 7.8e-3 apart and the two candidates are 1.3671875 and 1.375 — the
kernel landed one ULP below correctly-rounded. The registry's bf16 atol of 2e-3
is **smaller than a single bf16 ULP** there, so for a bf16-*stored* output that
elementwise bound is unsatisfiable in the worst case and passing it is luck.

`kernels/common/cdna3_harness.cuh` now provides `qc::Tol::bf16_output()` and
`fp16_output()`, which raise **only** the elementwise relative bound to one
storage ULP (2^-8 / 2^-10) and leave the aggregate rel-L1 and cosine bounds at
the registry values. The aggregates remain the real correctness signal: a
genuinely wrong kernel moves them; one-ULP storage noise does not. Applies to any
kernel whose output is stored in a narrow float; kernels producing fp32 keep the
plain registry tolerances. `cross_attention` was switched to the same bound and
still passes (it also passed the stricter one).

Decision: **KEEP** both experiments. No shape regresses.

Open questions / next levers: derive a per-query-block key range for structured
(banded/causal) masks so the tile scan itself can be skipped; LDS
double-buffering of K/V; larger BQ.

Raw results: perf/results/2026-07-25/attention-biased_attention/

## 2026-07-25: rope_variants — 8 RoPE operations (Metal/CPU parity Phase 1)

Status: landed.

Current implementation: `kernels/attention/rope_variants/variants/rocm_cdna3`
Current public route: `.quixicore/kernels.yaml` operation `rope_variants`,
covering `rotary_positioned`, `mrope`, `rope_table`,
`rope_interleaved_to_split`, `rope_backward`, `qk_norm_rope_positioned`,
`qk_norm_rope_split`, `rope_q_norm`.

References inspected: `../QuixiCore-CPU/kernels/attention/{rotary_positioned_ref,
attention_extended_ref,attention_ref,attention_serving_ref}.cpp` and
`kernels/utils/tensor_ops_ref.cpp:515`.

Grouped into one operation directory as `kernels/serving` (12 kernels) and
`kernels/linear_attention` (3) already are: shared row geometry, shared lever.

Correctness: **78/78 PASS** on MI300X vs. fp64 oracles, every operation checked
at both lane widths. fp32 `rel ~2.5e-8 … 5.3e-8, cosine 1.000000000`; bf16 at the
one-storage-ULP bound. Coverage: split and interleaved layouts, partial rotary
(rd 64/96 against D=128), per-batch position tables, mrope sectioned and
section-interleaved, GQA splits, a no-V-head case, the Gemma weight offset,
qk_norm with mrope tables, split vs packed outputs, norm on/off, non-zero pos0.

### Experiment 1 — pow -> exp2 + sincos: KEEP

`rope_interleaved_to_split` and `rope_backward` derive angles from `base` and
`pos0` instead of a table and were calling `pow()` per element. They sat at
~1.2 TB/s while the table-driven variants reached ~4.0 TB/s, so the hypothesis
was transcendental-bound rather than memory-bound. `base^e == exp2(e*log2(base))`
with `log2(base)` loop-invariant turns a `pow` into one `exp2`; one `sincos`
replaces a separate `cos` and `sin`. Still evaluated in double — the frequency
spread across pairs is where float loses the low bits.

| Kernel | Before | After | Speedup |
|---|---:|---:|---:|
| rope_interleaved_to_split | 0.4466 ms (1202 GB/s) | 0.2656 ms (2021 GB/s) | 1.68x |
| rope_backward | 0.4416 ms (1216 GB/s) | 0.2809 ms (1912 GB/s) | 1.57x |

### Experiment 2 — 32 -> 64 lanes per row: REJECT

This repo's standard first lever for row kernels, worth +30-71% on the norm
kernels (commit `1b77c18b`). Here it **lost on all eight operations**, reproduced
across two independent runs:

| Operation (shape) | 32 lanes/row | 64 lanes/row | Ratio |
|---|---:|---:|---:|
| rotary_positioned (B8 H32 T2048 D128) | 0.1327 ms (4045 GB/s) | 0.1538 ms (3492 GB/s) | 0.86x |
| mrope (B8 H32 T2048 D128) | 0.1640 ms (3274 GB/s) | 0.1705 ms (3150 GB/s) | 0.96x |
| rope_table (T16384 H32 D128) | 0.1375 ms (3905 GB/s) | 0.1544 ms (3476 GB/s) | 0.89x |
| rope_interleaved_to_split (T16384 H32 D128) | 0.2656 ms (2021 GB/s) | 0.3194 ms (1681 GB/s) | 0.83x |
| rope_backward (T16384 H32 D128) | 0.2809 ms (1912 GB/s) | 0.3165 ms (1696 GB/s) | 0.89x |
| qk_norm_rope_positioned (T8192 Hq32 Hk8 Hv8 D128) | 0.1549 ms (2599 GB/s) | 0.1853 ms (2173 GB/s) | 0.84x |
| qk_norm_rope_split (T8192 Hq32 Hk8 Hv8 D128) | 0.1533 ms (2626 GB/s) | 0.1864 ms (2160 GB/s) | 0.82x |
| rope_q_norm (T16384 H32 D128) | 0.2493 ms (2154 GB/s) | 0.2758 ms (1947 GB/s) | 0.90x |

**Why the norm result did not transfer, and why that matters for the remaining
phases.** The norm win came from a shape where half the wavefront sat idle — 32
threads assigned to a row inside a 64-wide wavefront. These kernels pack
`kBlock / LANES` rows per block, so a 32-lane row already fills the wavefront
with two rows; there is no idle half to reclaim. Halving the lanes per row
instead doubles the pairs each lane owns, doubling the independent loads in
flight, and with no cross-lane reduction to amortize, that ILP beats the wider
row. Consistent with this, the three variants that *do* have a reduction
(`qk_norm_rope_positioned`, `qk_norm_rope_split`, `rope_q_norm`) lose by the most
(0.82-0.90x) — a 64-lane reduction costs an extra shuffle step.

The lever is "fill the wavefront", not "widen the row". Where rows are already
packed, widening is a regression. Selected configuration is `kRopeLanes = 32`,
recorded as a constant beside this reasoning in the source.

Environment:
  GPU: AMD Instinct MI300X (gfx942), device index 1 (idle)
  ROCm: 7.2.4   HIP: 7.2.53211-97f5574fe2
  Container: bare metal (no container)
  Command: HIP_VISIBLE_DEVICES=1 make -C kernels/attention/rope_variants/variants/rocm_cdna3 bench
  Warmups/iterations: w25/i50, HIP-event median

Measurement caveat, reported rather than suppressed: several timings show a
min/max spread above the 1.2x warning threshold (up to 2.1x). The samples show a
**single slow iteration** — min and median are tight (`min 0.1290 / median
0.1327 / max 0.2696`) and medians reproduce to within 1.7% across independent
runs. It reads as a clock or power transient, not contention; every conclusion
above holds on the minima as well as the medians.

Decision: **KEEP** experiment 1, **REJECT** experiment 2. Shipped bandwidth is
3.3-4.0 TB/s for the table-driven variants, roughly 75% of MI300X HBM3 peak.

Open questions: cache the cos/sin table row in LDS when many heads share a token
(at H=32 the same row is re-read 32 times); vectorize to float4 for the split
layout, where both halves of a pair are contiguous runs.

Raw results: perf/results/2026-07-25/attention-rope_variants/

## 2026-07-25: attn_composites — swin / decode-cache / cascade (Phase 1 complete)

Status: landed. Completes Metal/CPU parity Phase 1 (13/13 kernels).

Current implementation: `kernels/attention/attn_composites/variants/rocm_cdna3`
Current public route: `.quixicore/kernels.yaml` operation `attn_composites`,
covering `swin_attention_d32`, `decode_cache_attention`,
`cascade_attention_multi`.

References inspected: `../QuixiCore-CPU/kernels/attention/attention_extended_ref.cpp:602`,
`attention_composites_ref.cpp:332` and `:470`.

Correctness: **24/24 PASS** on MI300X vs. fp64 oracles, each operation checked in
both baseline and candidate form. `decode_cache_attention` also verifies the
mutated key cache, not just the attention output — the append stage is half the
operation and a port that gets the cache slot wrong still produces plausible
attention output. fp32 `rel ~1e-7 … 6e-7, cosine 1.000000000`.

Contract details pinned by tests: Swin's mask index is
`window % windows_per_image` (the shifted-window mask repeats per image);
decode's attention bound is `[0, context_lengths[item]]` **inclusive**, so the
token appended by stage 1 is visible to stage 2 in the same call; cascade runs
one online softmax across prefix levels *and then* the paged tail in contract
order, with `levels == 0` legal.

| Operation | Baseline | Candidate | Speedup |
|---|---:|---:|---:|
| swin_attention_d32 (W1024 T49 H8) | 1.1245 ms global K/V | 1.0836 ms LDS K/V | 1.04x |
| decode_cache_attention (B256 Hq32 Hkv8 D128 ctx4096) | 587.62 ms scalar | 13.16 ms wavefront | 44.64x |
| cascade_attention_multi (B128 Hq32 Hkv8 D128 levels=2) | 109.69 ms scalar | 3.43 ms wavefront | 32.02x |

Decision: **KEEP** all three. Two caveats recorded so the numbers are not
misread.

**The Swin LDS win is marginal — 3.8%, not the lever it looked like.** The
hypothesis was that staging a window's K/V in LDS would pay because the global
version re-reads them per query position. A Swin window's K and V are only
`2 x 49 x 32 x 4 B ~= 12 KB`, so they are already cache-resident across the query
positions of the same workgroup; LDS saves an L1/L2 round-trip, not a DRAM one.
Kept (consistent at 1.03x spread, reduces cache pressure at larger `tokens`) but
recorded as marginal rather than dressed up.

**The 44.64x decode figure conflates two effects.** Its baseline is a direct
scalar transliteration — one thread per output row — which *also* recomputes the
query rotation inside the per-key dot product instead of materializing it, as a
naive port would to avoid a scratch buffer. So the number mixes "scalar to
wavefront" with "stop recomputing the rotation". The candidate's real structural
advantages are lane-parallel dot products with coalesced cache reads plus holding
the rotated query in registers across the cache walk. The cascade comparison has
no such confound — both sides do identical arithmetic — and its **32.02x** is a
clean scalar-vs-wavefront result. Read that one as the honest measure of the
lever.

Environment:
  GPU: AMD Instinct MI300X (gfx942), device index 1 (idle)
  ROCm: 7.2.4   HIP: 7.2.53211-97f5574fe2
  Container: bare metal (no container)
  Command: HIP_VISIBLE_DEVICES=1 make -C kernels/attention/attn_composites/variants/rocm_cdna3 bench
  Warmups/iterations: w10-15/i30-50, HIP-event median; worst spread 1.10x

Open questions: MFMA the Swin path (D=32 x tokens=49 rounds to 4 MFMA tiles and
`biased_attention` already has the fragment layout); split-K over the context for
`decode_cache_attention` at long contexts, reusing `merge_attn_states`;
float4-vectorize the cache walk.

Raw results: perf/results/2026-07-25/attention-attn_composites/

## 2026-07-25: kv_cache_q8_0 — Q8_0 paged KV codec (Phase 2, 4/17)

Status: landed.

Current implementation: `kernels/serving/kv_cache_q8_0/variants/rocm_cdna3`
Current public route: `.quixicore/kernels.yaml` operation `kv_cache_q8_0`,
covering `kv_cache_scatter_q8_0`, `kv_cache_gather_q8_0`,
`kv_cache_copy_blocks_q8_0`, `paged_attention_q8_0`.

References inspected: `../QuixiCore-CPU/kernels/attention/attention_q8_kv.cpp`
(:102, :235, :351, :417).

Correctness: **28/28 PASS** on MI300X. Scatter code and scale planes and
`copy_blocks` compare **bit-exact**; gather is exact (a pure dequantize);
`paged_attention_q8_0` is `rel ~1.5e-7, cosine 1.000000000` against an fp64
oracle that mirrors the reference's 32-key tiling and its `double` denominator.
Coverage: head dims 64/128/256, page sizes 16/32, MQA and GQA, block-table holes,
skipped scatter slots, an all-zero quantization group, sliding window, explicit
score scale.

Four contract details pinned by tests, each of which a plausible port gets wrong:

1. The layout is **two separate planes** (int8 codes, raw fp16-bit scales), not
   the GGUF interleaved block. Packing a scale ahead of each 32-code run yields
   numbers that look right and a cache no sibling backend can read.
2. Rounding is `copysign(floor(|x| + 0.5), x)` — half **away from** zero, not
   `rintf`'s half-to-even. This is why the code planes can be asserted bit-exact
   instead of within a tolerance, which is a much stronger test.
3. Scatter **rewrites the whole cache**: zero-fill everything, then write the
   requested slots, leaving `-1` slots zeroed. The test pre-dirties the cache
   with `0x7f` first, so omitting the zero-fill fails rather than passing on the
   slots it happened to touch.
4. A negative block id is a **hole**: zeroed on gather, and skipped entirely in
   attention — which is not the same as contributing a zero score.

Baseline: direct scalar transliteration, one thread per output row.
Experiments: one factor — give each row a 32-lane group, hold the query in
registers, and compute each tile's score with a lane-parallel dot plus a
wavefront reduction. The 32-key tiling and its single rescale per tile are
preserved on both sides.

| Shape | Baseline | Candidate | Speedup |
|---|---:|---:|---:|
| B128 Hq32 Hkv8 D128 ctx512 | 7.2420 ms | 1.0480 ms | 6.91x |

Both sides do identical arithmetic, so unlike the `decode_cache_attention` A/B
recorded above this figure is not confounded by a second change.

Environment:
  GPU: AMD Instinct MI300X (gfx942), device index 1 (idle)
  ROCm: 7.2.4   HIP: 7.2.53211-97f5574fe2
  Container: bare metal (no container)
  Command: HIP_VISIBLE_DEVICES=1 make -C kernels/serving/kv_cache_q8_0/variants/rocm_cdna3 bench
  Warmups/iterations: w5-10/i20-30, HIP-event median; worst spread 1.06x

Decision: **KEEP**.

Open questions: `__builtin_amdgcn_sdot4` for the score dot product with a
per-tile quantized query — the repo's int8 GEMM already measured a win from
sdot4; split-K over the context with `merge_attn_states`; vectorized code loads.

Raw results: perf/results/2026-07-25/serving-kv_cache_q8_0/

## 2026-07-26: matmul-decode_epilogues Phase 3 CPU/Metal parity

Status: landed.

Current implementation: `kernels/matmul/decode_epilogues/variants/rocm_cdna3`.
Current public route: `.quixicore/kernels.yaml` operations `decode_linear`,
`decode_linear_residual`, `decode_linear_q8`,
`decode_linear_epilogue_dense`, `decode_swiglu_dense`,
`gemm_gate_residual`, `grouped_gemm`, `lora_apply_direct_f16`, and
`complex_gemm`.

References inspected: `../QuixiCore-CPU/kernels/matmul/matmul_extended_ref.cpp`,
`../QuixiCore-CPU/kernels/matmul/lora_ref.cpp`,
`../QuixiCore-CPU/kernels/matmul/dense_gemm_ref.cpp`,
`../QuixiCore-Metal/kernels/matmul/decode_linear/decode_linear.metal`,
`../QuixiCore-Metal/kernels/matmul/cmplx_matmul/cmplx_matmul.metal`,
`kernels/common/cdna3_harness.cuh`.

Correctness: **ALL PASS** on MI300X. The standalone harness checks 13 result
lines against fp64 or format-aware host oracles:
`decode_linear` fp32/bf16, `decode_linear_residual` fp32,
`decode_linear_q8` q8_0 fp32, `decode_linear_epilogue_dense` fp16,
`decode_linear_epilogue_packed` q8_0 fp32,
`decode_swiglu_dense` bf16, `decode_swiglu_packed` q8_0 fp32,
`gemm_gate_residual` fp32, `grouped_gemm` fp32,
`lora_apply_direct_f16` fp32, and complex GEMM real/imag fp32.
Tolerances: harness `Tol::fp32`, `bf16_output`, and
`fp16_output`; q8_0 reference dequantizes the packed bytes and compares the
actual quantized contract, not the pre-quantized float weights.

Baseline 1: scalar one-thread-per-output decode residual route,
rows=64, input_dim=4096, output_dim=2048, fp32, optional bias+residual enabled.
Median 2.7513 ms, 0.4 TFLOP/s, min 2.7294 ms, max 2.7700 ms, spread 1.01x.

Experiment 1: one factor changed: split each output dot product across one
64-lane CDNA3 wavefront and reduce with `qc::wave_reduce_sum`. Candidate median
0.2795 ms, 3.8 TFLOP/s, min 0.2749 ms, max 0.2952 ms, spread 1.07x,
**9.84x** faster than the scalar baseline. Keep wave64 for decode epilogues.

Baseline 2: scalar one-thread-per-output dense `grouped_gemm`, groups=8,
M=64, N=512, K=2048, fp32. Median 0.4731 ms, 2.3 TFLOP/s,
min 0.4710 ms, max 0.4881 ms, spread 1.04x.

Experiment 2: same wave64 split-dot lever for grouped GEMM. Candidate median
3.0423 ms, 0.4 TFLOP/s, min 2.9285 ms, max 3.1014 ms, spread 1.06x,
**0.16x** the scalar baseline. Reject wave64 for grouped GEMM and keep scalar.

Environment:
  GPU: AMD Instinct MI300X (gfx942), device index 0
  ROCm: 7.2.4   HIP: 7.2.53211-97f5574fe2
  Container: bare metal (no container)
  Commit: 0519f214 (working tree dirty)
  Command: `HIP_VISIBLE_DEVICES=0 make -C kernels/matmul/decode_epilogues/variants/rocm_cdna3 bench`
  Warmups/iterations: correctness once, performance w10/i50 with HIP-event median.

Decision: **KEEP** the wave64 decode implementation and **REJECT** the same
lever for grouped GEMM. Decode is latency-shaped and benefits from a wavefront
per output; grouped GEMM's tile shape is better served by the scalar
one-thread-per-output route until it is replaced by a real MFMA/LDS design.

Open questions: MFMA/LDS dense prefill route if these decode epilogues are ever
routed for large batches.

Raw results: `perf/results/2026-07-26/matmul-decode_epilogues-phase3-final/`.

## 2026-07-26: matmul-decode_epilogues packed Phase 3 completion

Status: landed.

Current implementation: `kernels/matmul/decode_epilogues/variants/rocm_cdna3`.
Current public route: `.quixicore/kernels.yaml` operations
`decode_linear_epilogue_packed` and `decode_swiglu_packed`.

References inspected: `../QuixiCore-CPU/kernels/matmul/matmul_extended_ref.cpp`,
`../QuixiCore-Metal/kernels/matmul/decode_linear/decode_linear.metal`,
`../QuixiCore-Metal/bindings/python/tk/quant.py`,
`kernels/quantization/qgemv/variants/rocm_cdna3/quant_formats.cuh`,
`kernels/quantization/qgemv/variants/rocm_cdna3/quant_formats_tables.cuh`,
and `kernels/common/cdna3_harness.cuh`.

Correctness: **ALL PASS** on MI300X. The standalone harness checks 24 result
lines against fp64 or byte-layout-aware packed host oracles. The new packed
coverage validates `decode_linear_epilogue_packed` and `decode_swiglu_packed`
for `q4_0`, `q8_0`, `q6_K`, `mxfp8`, `nvfp4`, and `mxfp4`, rows=2,
input_dim=512, output_dim=35, fp32 output, optional bias/residual/activation
where applicable. Tolerance: harness `Tol::fp32`.

Baseline 1: scalar one-thread-per-output `decode_linear_epilogue_packed`
`mxfp4`, rows=64, input_dim=4096, output_dim=1024, fp32, GELU+bias+residual.
Median 1.1557 ms, 0.5 TFLOP/s, min 1.1503 ms, max 1.1754 ms, spread 1.02x.

Experiment 1: one factor changed: split each packed dot product across one
64-lane CDNA3 wavefront and reduce with `qc::wave_reduce_sum`. Candidate median
0.2086 ms, 2.6 TFLOP/s, min 0.2047 ms, max 0.2180 ms, spread 1.07x,
**5.54x** faster than the scalar baseline. Keep wave64.

Baseline 2: scalar one-thread-per-output `decode_swiglu_packed` `mxfp4`,
rows=64, input_dim=4096, output_dim=1024, fp32, bias enabled. Median
4.1934 ms, 0.3 TFLOP/s, min 4.1354 ms, max 4.2707 ms, spread 1.03x.

Experiment 2: same wave64 split-dot lever for the two packed SwiGLU
projections. Candidate median 0.3587 ms, 3.0 TFLOP/s, min 0.3483 ms,
max 0.3994 ms, spread 1.15x, **11.69x** faster than the scalar baseline.
Keep wave64.

Environment:
  GPU: AMD Instinct MI300X (gfx942), device index 0
  ROCm: 7.2.4   HIP: 7.2.53211-97f5574fe2
  Container: bare metal (no container)
  Commit: 0335ef9d (working tree dirty)
  Command: `HIP_VISIBLE_DEVICES=0 make -C kernels/matmul/decode_epilogues/variants/rocm_cdna3 bench`
  Warmups/iterations: correctness once, performance w10/i50 with HIP-event median.

Decision: **KEEP** the wave64 packed epilogue implementation for both linear
and SwiGLU packed decode paths. The same split-dot factor is a clear win on the
representative `mxfp4` packed path and all six packed formats pass the
format-aware oracle.

Open questions: broader format-by-format packed performance sweeps and a real
MFMA/LDS dense prefill route for large-batch matmul.

Raw results:
`perf/results/2026-07-26/matmul-decode_epilogues-packed-phase3-rerun/`.

## 2026-07-26: MoE Phase 4 CPU/Metal parity completion

Status: landed.

Current implementation: `kernels/moe/variants/rocm_cdna3` and
`kernels/moe/variants/rocm_cdna3_quant`. Current public route:
`.quixicore/kernels.yaml` operations `moe_route_grouped`,
`moe_gather_backward`, `moe_finalize_backward`,
`moe_grouped_gemm_backward_input`, `moe_grouped_gemm_backward_weight`,
`moe_grouped_qgemm`, and `moe_grouped_qswiglu`.

References inspected: `../QuixiCore-CPU/kernels/moe/moe_ref.cpp`,
`../QuixiCore-CPU/kernels/moe/moe_extended_ref.cpp`,
`../QuixiCore-Metal/kernels/moe/moe/moe.metal`,
`kernels/moe/variants/rocm_cdna3/tm_moe_kernels.cuh`,
`kernels/moe/variants/rocm_cdna3_quant/tm_moe_quant_kernels.cuh`, and the
2026-07-06 MoE notebook entries above.

Correctness: **ALL PASS** on MI300X. Dense MoE reports 13/13 harness checks
passing, including grouped routing, gather backward, finalize backward, and
both grouped-GEMM backward paths. Quantized MoE reports 13/13 harness checks
passing, including `moe_grouped_qgemm q2_K` and `moe_grouped_qswiglu q2_K`.
Oracles are fp64 host replicas of the CPU/Metal contracts, with exact id checks
where applicable and fp32-dot tolerances for q2_K row-indexed paths.

Dense Phase 4 baseline/candidate: one factor changed from scalar row/token
baselines to the current row/warp-parallel kernels. MI300X, fp32, HIP-event
per-launch median with repeated launches inside each sample.

| Kernel | Shape | Baseline | Candidate | Speedup | Decision |
|---|---|---:|---:|---:|---|
| `moe_route_grouped` | tokens=16384 E=128 K=4 groups=16 top_groups=4 | 0.2804 ms | 0.1000 ms | 2.80x | keep warp grouped route |
| `moe_gather_backward` | gathered_rows=16384 tokens=4096 dim=1024 | 1.6436 ms | 0.0504 ms | 32.59x | keep row-parallel atomics |
| `moe_finalize_backward` | tokens=8192 K=4 dim=1024 | 10.7431 ms | 0.2292 ms | 46.86x | keep warp token backward |
| `moe_grouped_gemm_backward_input` | rows=2048 experts=16 K=256 N=256 | 13.4278 ms | 0.4934 ms | 27.21x | keep row-parallel input grad |
| `moe_grouped_gemm_backward_weight` | rows=2048 experts=16 K=128 N=128 | 12.2180 ms | 0.1281 ms | 95.40x | keep row-parallel weight grad |

Quantized Phase 4 baseline/candidate: one factor changed from scalar
one-thread-per-row q2_K dot products to the current row-parallel q2_K kernels.

| Kernel | Shape | Baseline | Candidate | Speedup | Decision |
|---|---|---:|---:|---:|---|
| `moe_grouped_qgemm` q2_K | rows=4096 experts=8 N=128 K=512 | 3.8500 ms | 0.0847 ms | 45.44x | keep row-parallel qgemm |
| `moe_grouped_qswiglu` q2_K | rows=4096 experts=8 inter=64 K=512 | 3.2027 ms | 0.0747 ms | 42.85x | keep row-parallel qswiglu |

Environment:
  GPU: AMD Instinct MI300X (gfx942), device index 0
  ROCm: 7.2.4   HIP: 7.2.53211-97f5574fe2
  Container: bare metal (no container)
  Commit: 8daae0c4 (working tree dirty)
  Commands:
    `HIP_VISIBLE_DEVICES=0 make -C kernels/moe/variants/rocm_cdna3 bench`
    `HIP_VISIBLE_DEVICES=0 make -C kernels/moe/variants/rocm_cdna3_quant bench`
  Warmups/iterations: correctness once; timing w5/i20 with per-sample launch
  repeats printed per row (`r10`..`r200`) and HIP-event median/min/max.

Decision: **KEEP** all current Phase 4 CDNA3 kernels. Every measured row beats
its scalar baseline by at least 2.80x, with the backward and q2_K row-parallel
paths showing the expected large wins from distributing row work across a block
or wavefront.

Open questions: sorted-by-expert q2_K rows should continue to prefer the
existing MFMA tile route (`moe_gemm_gguf`) when the schedule permits one expert
per 32-row tile; the row-indexed Phase 4 qgemm/qswiglu kernels are the generic
contract path.

Raw results:
`perf/results/2026-07-26/moe-phase4-dense-final2/` and
`perf/results/2026-07-26/moe-phase4-quant-final2/`.

## 2026-07-26: In-GEMM k-quant Decode (dequant4 MFMA fragment) — LANDED

Status: landed. The 256-superblock k-quants can now run the qgemm fragment path
directly from packed weights instead of the full-dequant route.

Current implementation: `kernels/quantization/qgemm/variants/rocm_cdna3`.
New `dequant4<FMT>` span helper (generic in `quant_formats.cuh`, specializations
for q2_K/q3_K/q4_K/q5_K/q6_K in `quant_formats_tables.cuh`) decodes the 4
contiguous K that one lane owns in a `v_mfma_f32_16x16x16_f16` B fragment
(k = k0 + 4*(lane/16) + {0..3}). Every span constant — sub-block index, packed
sub-scale, nibble shift, high-bit mask — depends only on bits >=4 of the
in-block column, and a 4-aligned 4-span never crosses a 16-wide sub-block, so
the bodies are the existing dequant8 bodies with a 4-iteration tail.
`load_wfrag<FMT>` (`tm_qmm_mfma.cuh`) now routes through `dequant4`; formats
without a specialization fall back to 4 plain `FMT::dequant` calls, unchanged.

Current public route: unchanged defaults. The harness now measures the in-GEMM
path alongside the dequant route for every superblock format so the two can be
compared per shape; production routing is a follow-up.

Motivation: the dequant route (`dequant_to_fp16` then `qgemm<fp16_raw>`)
materializes all of W as fp16. That is fine for a 512x4096 test tile and
unusable for serving — a 262 GB Q2_K checkpoint would need ~2 TB of fp16.

References inspected: `../QuixiCore-Metal` `dequant_into_shared` /
`dequant_into_register` and `qgemm_q2_K` / `qgemm_frag_q2_K` (the Metal backend
already ships both in-GEMM routes), `tm_qmm_mfma.cuh`, `qgemm.cu`, `perf/perf.md`.

Correctness: `make test` -> ALL PASS (0 failures), 203 PASS lines across the 29
golden formats (qgemm base / ksplit / wide / ctaLDS, qflux, qgemm_variants).
Every in-GEMM result is **bit-identical** to the dequant route (bitdiff
0/32768 on all 11 superblock formats), which is the expected outcome: same
decode math, same fp32 op order per element.

Baseline / experiment: MI300X gfx942, `HIP_VISIBLE_DEVICES=0`, ROCm/HIP
7.2.53211, hipcc from ROCm 7.2.4, AMD clang 22.0.0git, native HIP harness (no
framework). Git `0519f214d4aca37d94242c08d61f35d6d8653bde` plus working tree.
Command: `make test` / per-format `./qgemm.out <golden dir>`. Golden shape
N=512 K=4096 M=64, fp16 X, fp32 accumulate. HIP events, 50 iterations.

Decode-overhead = in-GEMM ms / `qgemm<fp16_raw>` ms on the same tile (W already
dequantized), i.e. the cost of decoding in the fragment:

| format | scalar dequant | dequant4 | in-gemm vs dequant-route | ksplit in-gemm |
|---|---|---|---|---|
| q2_K | 1.75x | **1.66x** | 0.95x | **13.76 TFLOP/s** (route: 13.20) |
| q4_K | 9.12x | **2.79x** | 0.57x | 9.57 TFLOP/s |
| q6_K | 1.96x | 2.05x | 0.78x | 11.45 TFLOP/s |
| q3_K | - | 2.98x | 0.54x | - |
| q5_K | - | - | 0.52x | - |

Decision: KEEP. Two separate findings.

1. `dequant4` is a large win exactly where the sub-scale unpack is branchy:
   q4_K goes 9.12x -> 2.79x (3.3x faster). Its 6-bit packed scale has an
   `if (sub < 4)` that blocks CSE across four inlined `FMT::dequant` calls.
   For the branch-free formats (q2_K, q6_K) the compiler was already CSE-ing
   the redundant scale decode, so the gain is ~5% or inside noise — q6_K
   measured 2% slower, which is noise at this shape and not a regression worth
   chasing.

2. The in-GEMM path is the one that matters for serving regardless of the
   single-tile ratio, because the dequant route's memory cost is prohibitive.
   At the decode-shape K-split geometry q2_K in-GEMM is **13.76 vs 13.20
   TFLOP/s** — it beats the dequant route outright while allocating nothing.

Open questions: the golden shape is occupancy-bound, not decode-bound (128
wavefronts on 304 CUs, ~43 GB/s effective of 5.3 TB/s peak), so the base-kernel
ratios above understate the in-GEMM path. Next levers, in order: (a) port
Metal's `dequant_into_shared` — a CTA cooperatively decodes a BN x BK tile into
LDS with coalesced reads, which fixes the scattered 84-byte-strided per-lane
loads that dominate q2_K; (b) instantiate the `moe_gemm_*` grouped GEMM for
q2_K; (c) sweep realistic serving shapes (N=6144/2048, K=6144/2048) rather than
the 512x4096 golden tile. The iq* codebook formats keep the generic 4-element
fallback and were not specialized.

Raw results: `perf/results/2026-07-26/qgemm-kquant-ingemm/make_test.txt`.

## 2026-07-26: LDS-Staged k-quant Tile Decode (span fill) — LANDED (1.4-2.0x on ctaLDS)

Status: landed. Port of QuixiCore-Metal's `dequant_into_shared` span fill into
the CDNA3 multi-wave CTA/LDS qgemm tile.

Current implementation: `stage_qgemm_cta_tile` in the new
`kernels/quantization/qgemm/variants/rocm_cdna3/qgemm_kernels.cuh` (kernels
split out of `qgemm.cu` so the golden harness and the shape bench share one
definition — the harness keeps its `// ---- harness ----` half).

The W fill was one scalar `FMT::dequant` per element: for a 16-wide K tile that
re-decodes the block/sub-block scale 16x per row segment and issues 16 scattered
byte reads. It now fills one 8-wide span per thread via `dequant8<FMT>` — one
scale unpack per span, and the 8 packed weights a span needs are contiguous in
the block, so the global read is a short burst. Bit-identical to the
per-element fill (same decode, same fp32 op order).

Current public route: unchanged; `qgemm_cta_lds` is still a bench/harness
candidate, not the default. Routing is the open follow-up (see below).

References inspected: `../QuixiCore-Metal/include/metal/ops/warp/register/tile/
dequant.metal` (`dequant_into_shared`, SPANS_PER_ROW = BK/8) and
`kernels/quantization/qgemm/qgemm.metal` (`qgemm_q2_K`), plus the prior
2026-07-07 qgemm LDS entries.

Correctness: `make test` -> ALL PASS (0 failures), 203 PASS lines, ctaLDS
included. Raw: `perf/results/2026-07-26/qgemm-kquant-ingemm/make_test_after_lds.txt`.

Baseline / experiment: MI300X gfx942, ROCm/HIP 7.2.53211, hipcc ROCm 7.2.4,
`HIP_VISIBLE_DEVICES=0`, native HIP bench (`qgemm_q2k_bench.cu`, new), q2_K
weights, fp16 X, fp32 accumulate, HIP events, 10 warmup / 50 iter median.
Shapes are the GLM-5.2 projections. ctaLDS MT=4 NT=4, 256 threads.

ctaLDS TFLOP/s, per-element fill vs 8-span fill:

| M | N x K | per-element | 8-span | gain |
|---|---|---|---|---|
| 1024 | 2048 x 6144  | 33.98 | 46.64  | 1.37x |
| 1024 | 6144 x 2048  | 44.75 | 83.56  | 1.87x |
| 1024 | 6144 x 16384 | 42.11 | 83.65  | 1.99x |
| 1024 | 16384 x 2048 | 62.37 | 99.39  | 1.59x |
| 4096 | 2048 x 6144  | 55.24 | 94.80  | 1.72x |
| 4096 | 6144 x 2048  | 58.69 | 101.83 | 1.74x |
| 4096 | 6144 x 16384 | 59.15 | 86.59  | 1.46x |
| 4096 | 16384 x 2048 | 66.81 | 105.86 | 1.58x |

Decision: KEEP, well above the 8-10% complexity threshold, with a clear
mechanism (scale decode 16x -> 2x per row segment; contiguous instead of
scattered weight reads) and bit-identical output.

Route shape (from the same bench, q2_K, x = time vs the fp16 upper bound on the
same tile — below 1.0 means the quantized kernel beats a pure fp16 GEMM because
it moves ~6x fewer weight bytes):

| M | fragment | ksplit | ctaLDS |
|---|---|---|---|
| 64   | 1.25-1.82x | **1.12-1.73x** | 1.09-5.69x |
| 256  | 1.72-1.82x | 1.60-1.83x | **0.42-2.29x** |
| 1024 | 1.71-1.82x | 1.73-1.83x | **0.34-0.74x** |
| 4096 | 1.66-1.76x | 1.67-1.77x | **0.31-0.39x** |

At prefill the ctaLDS path reaches 86-106 TFLOP/s, 5.4x the fragment path and
~3x faster than the fp16 GEMM on the same tile. At M=64 it is the wrong kernel
(the CTA grid stops filling the device) and ksplit wins.

Open questions / follow-ups: (a) wire the M threshold (~128) into a
`qgemm_pick_route` alongside `qgemm_pick_nt`/`qgemm_pick_kchunk` and make it the
public route; (b) apply the same span fill to `moe_gemm_gguf` (currently the
register-fragment geometry only, which by the table above is the prefill
loser); (c) MT/NT sweep — 4x4 was inherited, not tuned, and N=2048 K=6144 is
consistently the weakest shape.

Raw results: `perf/results/2026-07-26/qgemm-kquant-ingemm/q2k_shape_bench_span.txt`.

## 2026-07-26: Attention Weight-Format Downgrade (q8_0 -> q5_K) — REJECTED

Status: rejected. Hypothesis was wrong; recording it so it is not retried.

Hypothesis: for GLM-5.2 the attention/projection tensors are q8_0 and account
for ~56% of per-token decode weight traffic (14.4 of 25.6 GB, dominated by
o_proj at 6144x16384 and q_b at 16384x2048). Decode is weight-bandwidth bound
(arithmetic intensity ~6 FLOP/byte against an HBM roofline near 60), so
requantizing those tensors to q5_K should cut ~35% of their bytes and buy
roughly 20% end-to-end decode time.

Correctness: not applicable — timing-only A/B; no routing or kernel changed.

Baseline / experiment: MI300X gfx942, ROCm/HIP 7.2.53211, hipcc ROCm 7.2.4,
`HIP_VISIBLE_DEVICES=0`, `qgemm_q2k_bench.out --formats`, K-split fragment path
(the measured M<128 winner), fp16 X, fp32 accumulate, HIP events, 10 warmup /
50 iter median. Random weight bytes: no k-quant decoder branches on weight
values, only on column position, so this is timing-valid.

o_proj (N=6144, K=16384):

| format | bit/w | packed MB | M=16 ms | M=64 ms |
|---|---|---|---|---|
| q8_0 | 8.50 | 106.96 | **0.1433** | **0.5864** |
| q6_K | 6.56 |  82.58 | 0.2288 | 0.8952 |
| q5_K | 5.50 |  69.21 | 0.3020 | 1.1944 |
| q4_K | 4.50 |  56.62 | 0.2613 | 1.0126 |

Same ordering at q_b (16384x2048) and kv/gate (6144x2048): q5_K is 1.9-2.1x
SLOWER than q8_0 in wall clock at M=16 while moving 35% fewer bytes.

Decision: REJECT. Do not requantize attention below q8_0 for speed.

Why the hypothesis failed: the roofline argument is about aggregate model
traversal, but these kernels reach only 200-750 GB/s of the ~5.3 TB/s peak, so
they are nowhere near bandwidth-bound at the kernel level — they are
decode/latency bound. q8_0's decoder is a single scale multiply on an int8
(`s * float(qs[i])`); q5_K adds a packed 6-bit sub-scale extraction with a
branch plus a separate high-bit (qh) load per element. Fewer bytes at a much
costlier decode is a net loss until the kernels approach the bandwidth roofline.
Note q4_K beats q5_K (0.2613 vs 0.3020 ms) despite both having dequant4 — q5_K
pays the extra qh load that q4_K does not.

Corollary for the serving plan: kernel efficiency, not weight-byte count, is the
current decode bottleneck. Byte-reduction work should be revisited only after
the decode-path kernels get closer to the roofline.

Open questions: this measures the MFMA GEMM path at M=16/64. True batch-1
decode (M=1) runs the GEMV/MMVQ path in `kernels/quantization/qgemv`, whose
per-format cost ratios were not measured here and could differ — though the
mechanism (cheap q8_0 decode vs branchy k-quant decode) should hold or
strengthen, since there is less MFMA work to hide the decode behind.

Raw results: `perf/results/2026-07-26/qgemm-kquant-ingemm/format_ab_attention_shapes.txt`.

### Addendum (same day): 4-bit formats incl. nvfp4 — also rejected for decode

Extended the format A/B to q4_0, iq4_nl, mxfp4 and nvfp4. Every 4-bit format is
slower in wall clock than q8_0 at the attention shapes, and the ordering tracks
decoder complexity rather than byte count. o_proj (6144x16384), M=16:

| format | bit/w | packed MB | ms | GB/s |
|---|---|---|---|---|
| q8_0   | 8.50 | 106.96 | **0.1434** | **745.7** |
| q4_K   | 4.50 |  56.62 | 0.2589 | 218.7 |
| q4_0   | 4.50 |  56.62 | 0.2833 | 199.9 |
| mxfp4  | 4.25 |  53.48 | 0.2889 | 185.1 |
| iq4_nl | 4.50 |  56.62 | 0.3356 | 168.7 |
| nvfp4  | 4.50 |  56.62 | 0.3642 | 155.5 |

nvfp4 moves 47% fewer bytes and takes 2.5x longer — its per-element e4m3
block-scale decode is the costliest in the set.

Scope limit, stated explicitly: this runs every format through the fp16 K=16
MFMA path. It does NOT test nvfp4 decoded to fp8 through
`v_mfma_f32_16x16x32_fp8_fp8` (measured separately at 1.98x the fp16 MFMA
K-throughput, 2059 vs 1039 TFLOP/s). That 2x lands at prefill, where the kernel
is MFMA-bound; at M=16 q8_0 reaches only 745 GB/s of the ~5.3 TB/s peak (14%),
so nothing here is MFMA-bound and doubling MFMA issue would not move it.

The governing fact is that the decode-path kernel leaves ~7x on the table
against the bandwidth roofline, while any format swap is worth ~1.3x. Format
choice is currently dominated by a kernel inefficiency. Fix the small-M kernel
(span-staged staging at a geometry that still fills the device) before
re-evaluating any format question — the ordering in this table could flip.

Project decision (2026-07-26): stay on the antirez routed quant (q8_0
attention + q2_K routed experts) for GLM-5.2 serving. No requantization.

## 2026-07-26: BaseQ Phase 5 Canonical Family

Status: landed.

Current implementation: `kernels/quantization/base_q/variants/rocm_cdna3`.
Current public routes: `base_q_dequant`, `base_q_gemv`, `base_q_gemm`,
`base_q_embedding`, `base_q_gemv_qkv`, `base_q_gemv_swiglu`,
`base_q_lm_head_argmax`, `base_q_moe_gemm`, and `base_q_moe_swiglu`.

References inspected: `../QuixiCore-CPU/include/quixicore_cpu/base_q.h`,
`../QuixiCore-CPU/kernels/quantization/base_q_ref.cpp`, and
`../QuixiCore-CPU/tests/correctness/test_base_q.cpp`. No live BaseQ Metal
kernel exists; the CPU contract is canonical.

Correctness: `make -C kernels/quantization/base_q/variants/rocm_cdna3 test` ->
`ALL PASS` (246 checks). Oracle: fp64 host oracle with harness fp32/fp16/bf16
and exact integer tolerances. Shapes cover BaseQ bits 2/3/4/5/6/8, group sizes
32/64/128, BF16/F16/E8M0 scales, E4M3 for 8-bit weights, symmetric and affine
decode, FP32/FP16/BF16 output storage, Q/K/V row-count skew, LM-head lower-token
ties after storage rounding, and 32-row padded MoE expert schedules.

Hypothesis: BaseQ decode-dot consumers are latency/parallelism limited in a
scalar one-thread-per-output route. A CDNA3 wave64 split-dot should expose more
K-dimension parallelism while reusing the common packed-code and scale decoder.
For LM-head argmax, a single streaming kernel might avoid score writes, but may
underfill the GPU.

Baseline / experiment: MI300X gfx942, ROCm 7.2.4, HIP 7.2.53211,
`HIP_VISIBLE_DEVICES=0 make -C kernels/quantization/base_q/variants/rocm_cdna3 bench`.
BaseQ4 affine, BF16 scales/biases, group size 64, FP16 input unless noted. HIP
events, 5 warmups, 20 iterations, 100 repeated launches per sample. All spreads
<= 1.12x.

| operation | baseline/current | candidate/kept | decision |
|---|---:|---:|---|
| `base_q_dequant` | n/a | 0.0150 ms, 1273.1 GB/s | KEEP current |
| `base_q_gemv` | scalar 0.8204 ms | wave64 0.0119 ms, 68.99x | KEEP wave64 |
| `base_q_gemm` | scalar 0.8661 ms | wave64 0.1093 ms, 7.93x | KEEP wave64 |
| `base_q_embedding` | n/a | 0.0298 ms, 1408.8 GB/s | KEEP current |
| `base_q_gemv_qkv` | scalar 0.9173 ms | wave64 0.0126 ms, 72.85x | KEEP wave64 |
| `base_q_gemv_swiglu` | scalar 1.4720 ms | wave64 0.0187 ms, 78.81x | KEEP wave64 |
| `base_q_lm_head_argmax` | materialized 0.0389 ms | streaming 13.7230 ms | REJECT streaming; KEEP materialized |
| `base_q_moe_gemm` | scalar 0.8871 ms | wave64 0.1132 ms, 7.84x | KEEP wave64 |
| `base_q_moe_swiglu` | scalar 1.6136 ms | wave64 0.1672 ms, 9.65x | KEEP wave64 |

Decision: KEEP the shared decode core, wave64 split-dot projection routes,
current dequant/embedding routes, and materialized LM-head argmax. REJECT the
streaming LM-head candidate for the measured vocab shape; it saves score
storage but provides too little parallel work.

Open questions: specialize hot BaseQ4/6 prefill shapes with LDS-staged decode
or MFMA-friendly unpacking once a real model route exercises this format.

Raw results: `perf/results/2026-07-26/base-q-phase5-final3/`.

## 2026-07-26: MXFP8 paged KV-cache codec (Phase 2, 6/17) — LANDED

Status: landed (correctness-first port; no speedup claimed).

Current implementation: `kernels/serving/kv_cache_mxfp8/variants/rocm_cdna3`.
`kv_cache_scatter_mxfp8` and `kv_cache_gather_mxfp8`, ported from
`../QuixiCore-CPU/kernels/attention/attention_mxfp8.cpp` (:124, :224). Warp-group
per (token, head) with lanes striding MX groups — the same schedule as the
landed q8_0 codec.

Layout differs from q8_0 and the difference is the whole risk: q8_0 keeps codes
and scales in two separate planes, MXFP8 packs each 32-element group as one
33-byte record (E8M0 scale byte, then 32 E4M3FN codes) in a single uint8 plane.
Scatter is also incremental rather than a full rewrite — the q8_0 reference
zero-fills the cache first, MXFP8 does not, and adding a zero-fill "for
symmetry" would silently clear live pages.

Correctness: `make test` -> ALL PASS (0 failures).
  e8m0_encode_up vs hand-computed         PASS
  e4m3fn_encode vs hand-computed          PASS
  e4m3fn round-trips exact values         PASS
  scatter (byte-exact vs host replica)    0 of 135168 bytes differ
  scatter leaves unwritten slots untouched PASS
  gather (exact decode)                   0 of 49152 values differ
  round-trip within E4M3FN resolution     worst rel 5.882e-02

That last number is the format's own grid (~6%), not a fitted tolerance.

Harness note worth carrying to the remaining Phase 2 kernels: the byte-exact
scatter check CANNOT catch a wrong spec. The host replica calls the same
`__host__ __device__` helpers as the kernel, so mutating a shared helper moves
both sides together and they still agree. Verified by mutation — flipping
`e8m0_encode_up` from ceil to floor (the documented trap) left the byte-exact
check PASSING and was caught only by the round-trip check. Added independent
hand-computed primitive checks; the same mutation now fails immediately with
`e8m0_encode_up(0.75) = 126, want 127`.

Baseline: none. This is a functional port; no performance run and no speedup is
claimed. Perf work on the KV codecs should follow the paged-attention kernels
that consume them.

Decision: KEEP. Open: `paged_attention_mxfp8` completes the mxfp8 group.

Raw results: terminal output of `make test` in the variant directory.

## 2026-07-26: paged_attention_mxfp8 (Phase 2, 7/17) — LANDED

Status: landed (correctness-first port; no speedup claimed).

Decode attention straight against the MXFP8 paged cache. One 64-lane wavefront
per (request, query head); lanes stride the head dimension holding one double
partial per MX group, wave-reduce, then apply the group scales. Codec primitives
factored into `mxfp8_common.cuh` so the codec and this kernel share one
definition of the format.

Two contract details reproduced rather than "improved": the online softmax tiles
at **16** (the sibling q8_0 kernel tiles at 32, and tile size changes the
floating-point answer), and the reference's `old_weight` is 0 on the first tile
rather than 1.

Correctness: `make test` -> ALL PASS. vs host replica, worst relative error
2.499e-04 at window=0 and window=24, 0 non-finite. Not bit-exact by design: the
lane-strided reduction reorders the within-group summation relative to the CPU's
sequential loop.

### The harness was silently broken and mutation testing is what found it

First green run reported `worst rel 0.000e+00` on both windows. That is not a
plausible number for a reordered reduction, and it was wrong three ways:

1. The kernel was emitting **all NaN**.
2. `std::max(worst, nan)` returns `worst` — NaN comparisons are false — so every
   NaN left the running maximum at zero. **An entirely broken kernel PASSED.**
3. The NaN came from the test data: code bytes were filled with rng(0,254),
   which includes 0x7f, the E4M3FN NaN encoding. The encoder only ever emits
   that for NaN input, so such a cache is unreachable in practice.

Verified by mutation: perturbing the kernel's final write by 1% was NOT detected
before the fix and IS after (`worst rel 1.003e-02`, exactly the injection).
The harness now counts non-finite outputs explicitly and fails on any.

Known remaining limitation, same as the codec: mutating a constant the host
replica shares (e.g. kScoreTile) moves both sides together and cannot be caught
by a same-file replica. The guard against that is the hand-computed primitive
checks in the codec harness, not this comparison.

Baseline: none; functional port, no speedup claimed.

Raw results: terminal output of `make test` in the variant directory.

## 2026-07-26: quantized MLA absorbed decode, dense + sparse (Phase 2, 9/17) — LANDED

Status: landed (correctness-first port; no speedup claimed).

`kernels/serving/mla_absorbed/variants/rocm_cdna3`. This is the GLM-5.2 /
DeepSeek-V3.2 decode shape: the cache holds only the low-rank latent plus a RoPE
tail, kv_b is absorbed, and full K/V is never materialized. The sparse variant is
the DSA path — `token_indices [batch, max_topk]` resolved through the same page
table — which is exactly what GLM-5.2's indexer feeds.

One 64-lane wavefront per (request, head); `wave_sum`'s xor reduction leaves the
score in every lane so the online-softmax state stays in lockstep with no shared
staging. `query_latent` and `mixed_latent` live in dynamic shared memory
(2 * latent_dim floats = 4 KB at latent_dim 512).

Two contract details reproduced rather than tidied: the softmax is PER-KEY (the
sibling mxfp8 kernel tiles at 16), and its two branches are asymmetric — the
new-maximum branch adds the latent with an implicit weight of exactly 1 rather
than exp(0), and rescales in float while the denominator is double. Collapsing
them into one general update changes the result.

Correctness vs fp64 host replica, q8_0 packed kv_b, GLM-like geometry
(latent 512, rope 64, nope 64, value 64, page 16):
  quantized_mla_decode_absorbed          worst rel 1.470e-04, 0 non-finite
  quantized_mla_decode_absorbed_sparse   worst rel 3.557e-05, 0 non-finite

Not bit-exact by design: per-lane partials accumulate in double but the
cross-lane reduction is float, where the reference is double throughout.

Mutation-verified: collapsing the asymmetric branch (adding the latent with
weight exp(0)*1.0001 instead of 1) fails at 1.215e-02 / 3.408e-02. The harness
also counts non-finite outputs explicitly — `std::max(worst, nan)` returns
`worst`, so without that an all-NaN result reports zero error and PASSES. That
exact failure was hit for real on paged_attention_mxfp8 earlier today.

Baseline: none; functional port, no speedup claimed. Only q8_0 is instantiated;
the reference is generic over QuantFormat and the kernel is templated on FMT, so
other formats are a instantiation away.

Raw results: terminal output of `make test` in the variant directory.

## 2026-07-26: kv_cache_scale_update (Phase 2, 10/17) — LANDED

Status: landed (correctness-first port; no speedup claimed).

`kernels/serving/kv_cache_scale_update/variants/rocm_cdna3`. amax over key and
value, divided by 240, then a monotone max against the incoming scale so it
never shrinks. Non-finite input rejects the call.

The divisor is 240, NOT the 448 the MXFP8 codec next door uses (448 is the
largest finite E4M3FN magnitude). Copying the neighbouring constant yields
scales ~1.87x too small and a cache that quietly clips.

Reduction is block-local wave max then an atomicMax on the float bit pattern —
valid only because these are magnitudes, where IEEE-754 bit order matches
numeric order for non-negatives.

Correctness: `make test` -> ALL PASS, and **bit-exact** against the host, which a
max reduction should be (order-independent, unlike a sum): old=0, old-dominates,
single-element, and non-finite-rejected all exact.

Mutation-verified: swapping the divisor to 448 fails 2 of 4 cases. The
old-dominates case correctly still passes, since the incoming scale loses the
max regardless — a reminder that a test whose expected value is insensitive to
the mutation proves nothing about it.

Baseline: none; functional port, no speedup claimed.

## 2026-07-26: turboquant_query_transform (Phase 2, 11/17) — LANDED

Status: landed (correctness-first port; no speedup claimed).

`kernels/quantization/turboquant_query/variants/rocm_cdna3`. Sign flip against a
shared `signs` vector, unnormalized FWHT, then a single 1/sqrt(head_size).
head_size 64 / 128 / 256. One block per (row, head), blockDim == head_size,
transform in shared memory.

The FWHT is the plain a+b / a-b recurrence with NO per-stage 1/sqrt(2). Folding
the normalization into each stage is the usual orthonormal variant and gives a
different answer; the reference normalizes exactly once at the end.

Correctness: **bit-exact** at all three head sizes — 0 of 7104 / 10752 / 5632
elements differ, worst absolute difference 0. Exactness is the right bar here
and a tolerance would have hidden a real bug: every butterfly in a stage reads a
disjoint pair and writes those same two slots, so the answer cannot depend on
thread order, unlike the reduction-based kernels in this tree.

Mutation-verified: flipping one butterfly output (a-b -> b-a) fails with exactly
half the elements differing at every head size — the signature a sign error
should produce.

Baseline: none; functional port, no speedup claimed.

## 2026-07-26: Quant Authoring And Embedding Phase 6 — LANDED

Status: landed.

Current implementation: `kernels/quantization/quant_authoring/variants/rocm_cdna3`.
Current public route: Phase 6 operation entries in `.quixicore/kernels.yaml`:
`fake_quant_int8`, `fake_quant_float8`, `tq2_0_pack`, `tq2_0_unpack`,
`ternary_pack`, `ternary_unpack`, `ternary_stats`,
`ternary_code_flip_count`, `calibration_absmax`, `qgemm_backward_input`,
`dequant_gather`, `quantized_embedding`, `quantized_embedding_bag`, and
`mxfp4_gemv`.

References inspected: CPU `quantization/quantization_ref.cpp`,
`serving/basert_ref.cpp`, `quantization/qgemm_extended_ref.cpp`,
`serving/serving_quant_ref.cpp`, `quantization/microscale_ref.cpp`; Metal
`quantization/dequant_gather/dequant_gather.metal` and
`quantization/qgemm_bwd/qgemm_bwd.metal`; existing ROCm
`quantization/qgemv/variants/rocm_cdna3/quant_formats*.cuh`.

Correctness: `make -C kernels/quantization/quant_authoring/variants/rocm_cdna3 test`
and archived `bench` both report `ALL PASS`. The harness has 25 checks: INT8
fake-quant output/codes/scales; FP8 E4M3FN and E5M2 output/codes/scale;
byte-exact TQ2_0 pack plus unpack; byte-exact ternary pack plus unpack,
histogram, and flip count; NaN-preserving calibration absmax; q4_0/q8_0/q6_K
packed gather; q8_0 embedding with add; q6_K weighted mean embedding bag;
BitNet qgemm backward-input; and MXFP4 GEMV. Tolerances are exact for codes and
packing, fp32/fp8 for numeric outputs, with NaNs checked explicitly for
calibration.

Baseline and experiments on MI300X gfx942, ROCm 7.2.4, HIP 7.2.53211,
bare metal, command
`HIP_VISIBLE_DEVICES=0 make -C kernels/quantization/quant_authoring/variants/rocm_cdna3 bench`.
Benchmarks used printed warmups/iterations; fast kernels use repeated launches
per timing sample and report per-launch medians.

| route / shape | baseline | candidate/current | decision |
| --- | ---: | ---: | --- |
| `fake_quant_int8`, fp32 8192x1024 | scalar 0.6887 ms, 109.7 GB/s | wave64 0.0252 ms, 3001.1 GB/s, 27.36x | keep wave64 |
| `fake_quant_float8`, E4M3FN count 1,048,576 | scalar 550.1906 ms | wave64 10.8514 ms, 50.70x | keep wave64; encoder remains ALU-heavy |
| `tq2_0_pack`, rows 32768, K=512 | scalar 0.6982 ms, 198.4 GB/s | block-parallel 0.0786 ms, 1763.4 GB/s, 8.89x | keep block-parallel |
| `tq2_0_unpack`, rows 32768, K=512 | — | current 0.0404 ms, 3429.0 GB/s | landed measured route |
| `ternary_pack`, rows 65536, K=512, group=128 | scalar 1.7686 ms, 157.7 GB/s | block-parallel 0.3124 ms, 892.8 GB/s, 5.66x | keep block-parallel |
| `ternary_unpack`, rows 65536, K=512 | — | current 0.0828 ms, 3369.3 GB/s | landed measured route |
| `ternary_stats`, rows 65536, K=512 | — | current 0.0561 ms, 186.9 GB/s | landed measured route |
| `ternary_code_flip_count`, rows 65536, K=512 | — | current 0.0484 ms, 433.1 GB/s | landed measured route |
| `calibration_absmax`, fp32 4096x4096 | scalar 1.0833 ms, 62.0 GB/s | block-parallel 0.1876 ms, 358.0 GB/s, 5.78x | keep block-parallel |
| `dequant_gather`, q8_0 rows 65536, dim 256, tokens 65536 | — | current 0.0426 ms, 3565.3 GB/s | landed measured route |
| `quantized_embedding`, q8_0 add, same shape | — | current 0.0507 ms, 2998.8 GB/s | landed measured route |
| `quantized_embedding_bag`, q6_K rows 32768, dim 256, ids 65536, bags 8192 | — | current 0.0235 ms, 942.9 GB/s | landed measured route |
| `qgemm_backward_input`, BitNet M=64,N=1024,K=512 | scalar 0.1453 ms, 0.5 TFLOP/s | wave64 0.1340 ms, 0.5 TFLOP/s, 1.08x | keep wave64; simple split-dot win |
| `mxfp4_gemv`, N=65536,K=1024 | scalar 0.1354 ms, 1.0 TFLOP/s | wave64 0.0689 ms, 1.9 TFLOP/s, 1.97x | keep wave64 |

Decision: KEEP the Phase 6 ports. The canonical byte layouts are exact for the
authoring formats, packed-table consumers reuse the existing ROCm GGUF
dequantizers, and every measured A/B candidate is a win. Follow-up levers are a
multi-wave/block FP8 fake-quant reduction to replace the one-wave count path
and FP16/BF16 store variants for public embedding storage if ROCm grows a
shared storage dispatcher.

Raw results: `perf/results/2026-07-26/quant-authoring-phase6-final3/bench.txt`.

## 2026-07-26: Sampling And Embedding Stragglers Phase 7 - LANDED

Status: landed.

Current implementation: `kernels/sampling/phase7_stragglers/variants/rocm_cdna3`.
Current public route: Phase 7 operation entries in `.quixicore/kernels.yaml`:
`top_k_renorm`, `top_p_renorm`, `logits_softcap`,
`embedding_lookup_types`, `broadcast`, and `reduce_sum`.

References inspected: CPU `utils/utils_extended_ref.cpp`,
`serving/basert_ref.cpp`, and `collectives/collectives_ref.cpp`; Metal
`sampling/sampling/sampling_transforms.metal` and
`serving/embedding/embedding.metal`; existing ROCm RCCL collectives docs for
the distinction between single-device CPU tensor contracts and multi-GPU
transport.

Correctness: `make -C kernels/sampling/phase7_stragglers/variants/rocm_cdna3 test`
and archived `bench` both report `ALL PASS`. The harness has 8 checks: stable
top-k and top-p probability renormalization with lower-token tie breaks,
final-logit softcap, FP32/FP16/BF16 token+type embedding lookup with invalid ids
zeroed, root broadcast, and root reduce-sum. BF16 embedding expected values are
rounded through BF16 storage before accumulation and output comparison.

Baseline and experiments on MI300X gfx942, ROCm 7.2.4, HIP 7.2.53211,
bare metal, command
`HIP_VISIBLE_DEVICES=0 make -C kernels/sampling/phase7_stragglers/variants/rocm_cdna3 bench`.
Benchmarks used printed warmups/iterations; fast kernels use repeated launches
per timing sample and report per-launch medians.

| route / shape | baseline | candidate/current | decision |
| --- | ---: | ---: | --- |
| `top_k_renorm`, vocab 262144, k 64 | scalar 4.7255 ms, 0.2 GB/s | rank kernel 0.4260 ms, 2.5 GB/s, 11.09x | keep rank route |
| `top_p_renorm`, vocab 131072, p 0.91 | scalar 274.8679 ms | rank kernel 0.8937 ms, 0.6 GB/s, 307.58x | keep rank route |
| `logits_softcap`, 33554432 fp32 logits | scalar 3675.1438 ms | elementwise 0.0467 ms, 2874.7 GB/s | keep elementwise route |
| `embedding_lookup_types`, rows 32768, dim 1024, tokens 4096, fp32 | - | current 0.0505 ms, 2669.4 GB/s | landed measured route |
| `broadcast`, 4 roots x 9437184 fp32 values | scalar 1056.3555 ms | parallel copy 0.0192 ms, 1968.5 GB/s, 55085.76x | keep parallel route |
| `reduce_sum`, 4 inputs x 9437184 fp32 values | scalar 686.3865 ms | parallel sum 0.0090 ms, 4175.7 GB/s, 75926.73x | keep parallel route |

Decision: KEEP the Phase 7 ports. Sampling preserves the CPU/Metal stable
ordering contract while avoiding serial selection, softcap and the collective
tensor contracts are direct memory-bandwidth kernels, and typed embedding now
matches FP32/FP16/BF16 storage semantics. The collectives are deliberately not
RCCL replacements; they land the CPU root tensor operations on one device.

Raw results: `perf/results/2026-07-26/phase7-stragglers-final/bench.txt`.

## 2026-07-26: Linear Attention Phase 8 - LANDED

Status: landed.

Current implementation: `kernels/linear_attention/phase8_linear/variants/rocm_cdna3`.
Current public route: Phase 8 operation entries in `.quixicore/kernels.yaml`:
`gated_linear_attention`, `rwkv_wkv6`, `rwkv_wkv7`, `gdn_short_conv`,
`gdn_qkv_prepare`, `gdn_gate_beta`, `gdn_gated_rmsnorm`, `gdn_recurrence`, and
`linear_attention_unnormalized`.

References inspected: CPU `linear_attention/llama_recurrent_ref.cpp`,
`linear_attention/gdn_ref.cpp`, and
`linear_attention/linear_attention_extended_ref.cpp`; Metal
`linear_attention/gdn/gdn.metal` and
`linear_attention/linear_attn/linear_attn.metal`; existing ROCm
`kernels/linear_attention/variants/rocm_cdna3/gdn_kernels.cuh`.

Correctness: `make -C kernels/linear_attention/phase8_linear/variants/rocm_cdna3 test`
and archived `bench` both report `ALL PASS`. The harness has 17 checks covering
GLA output/final state, RWKV6 output/final state, RWKV7 output/final state,
unnormalized linear attention, GDN recurrence output/state, GDN short-conv
output/state including a negative slot, GDN Q/K/V preparation with default
NaN q_scale/k_scale handling, GDN gate/beta, and GDN gated RMSNorm.

Baseline and experiments on MI300X gfx942, ROCm 7.2.4, HIP 7.2.53211,
bare metal, command
`HIP_VISIBLE_DEVICES=0 make -C kernels/linear_attention/phase8_linear/variants/rocm_cdna3 bench`.
Benchmarks used printed warmups/iterations; fast kernels use repeated launches
per timing sample and report per-launch medians.

| route / shape | baseline | candidate/current | decision |
| --- | ---: | ---: | --- |
| `gated_linear_attention`, fp32 S=8,T=16,H=4,D=64 | task-serial 18.9735 ms | wave64 column route 0.0141 ms, 0.7 TFLOP/s, 1349.70x | keep wave64 |
| `rwkv_wkv6`, fp32 S=8,T=16,H=4,D=64 | task-serial 10.7257 ms | wave64 column route 0.0147 ms, 0.9 TFLOP/s, 729.48x | keep wave64 |
| `rwkv_wkv7`, fp32 S=8,T=16,H=4,D=64 | task-serial 22.6561 ms | wave64 row route 0.0178 ms, 0.9 TFLOP/s, 1270.82x | keep wave64 |
| `linear_attention_unnormalized`, fp32 B=4,H=4,N=32,D=32 | direct scalar 86.6738 ms | staged KV + Q@KV 0.0194 ms, 0.1 TFLOP/s, 4458.58x | keep staged route |
| `gdn_recurrence`, fp32 R=8,Hk=2,Hv=4,Dk=64,Dv=64 | request-serial 117.8900 ms | wave64 row route 0.0180 ms, 0.5 TFLOP/s, 6533.73x | keep wave64 |
| `gdn_short_conv`, fp32 R=64,C=512,K=4 | request-serial 1.7052 ms, 1.1 GB/s | channel-parallel 0.0083 ms, 221.2 GB/s, 204.67x | keep channel route |
| `gdn_qkv_prepare`, fp32 tokens=8192,Hk=4,Hv=8,Dk=64,Dv=64 | token-serial 0.2986 ms, 224.8 GB/s | block RMS + V copy 0.0494 ms, 1358.9 GB/s, 6.05x | keep parallel route |
| `gdn_gate_beta`, fp32 tokens=65536,Hv=8 | token-serial 0.1555 ms, 53.9 GB/s | elementwise 0.0057 ms, 1462.3 GB/s, 27.11x | keep elementwise route |
| `gdn_gated_rmsnorm`, fp32 rows=32768,Dv=128 | row-serial 0.3097 ms, 216.7 GB/s | block RMS 0.0259 ms, 2592.0 GB/s, 11.96x | keep block route |

Decision: KEEP the Phase 8 ports. The kernels preserve serial token order where
recurrence requires it, but split independent rows, columns, channels, or
normalization reductions across CDNA3 wave64/block work. The GDN ports are
fp32-only operation-level parity; FP16/BF16 storage wrappers are explicit
non-ports until a shared floating storage dispatcher exists in the ROCm backend.

Raw results: `perf/results/2026-07-26/phase8-linear-final2/bench.txt`.
