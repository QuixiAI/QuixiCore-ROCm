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
