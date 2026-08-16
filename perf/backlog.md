# ROCm Optimization Backlog

The beam: 3-5 active idea families, best first. Pick from the top. Update
after every concluded experiment. Kill criteria are binding — when one fires,
record the kill in `perf/findings.md` and remove the family.

## Beam

### 1. qgemm CTA-LDS route to production
- Parent result: `qgemm_cta_lds<4,4>` is +55–117% vs the wide kernel at
  M≥1024, and with the span fill reaches 86–106 TFLOP/s on q2_K GLM-5.2
  prefill shapes (2026-07-07 "qgemm Multi-Wave CTA LDS Tile — CANDIDATE";
  2026-07-26 "LDS-Staged k-quant Tile Decode (span fill) — LANDED").
- Hypothesis: with an M routing threshold (~128) and full format coverage this
  becomes the shipped prefill qgemm path with no decode regression.
- Evidence so far: bit-identical output on every measured shape; packed
  q8_0/q4_0 confirmed (+79% / +266% vs wide at M=1024); 0.17x of base at M=64
  so it must never route to decode; fp8/nvfp4/superblock formats untimed;
  MT/NT 4x4 was inherited, not tuned, and N=2048 K=6144 is consistently the
  weakest shape.
- Next action: run the fp8/nvfp4/superblock format sweep at M∈{256,1024,2048}
  on N=K=4096 plus the GLM rectangles (`make bench` /
  `./qgemm_bench.out <M> <N> <K>` in
  `kernels/quantization/qgemm/variants/rocm_cdna3`), then wire
  `qgemm_pick_route` beside `qgemm_pick_nt` and mirror the epilogue into
  qflux.
- Kill criteria: any covered format regresses >5% vs the current wide/ksplit
  winner at its routed shapes, or no MT/NT configuration removes the
  decode-shape regression once the route threshold is in place.

### 2. Small-M quantized decode efficiency
- Parent result: decode-shape k-quant kernels reach only 200–750 GB/s of the
  ~5.3 TB/s roofline (~4–14%), which is why every format downgrade lost
  (2026-07-26 "Attention Weight-Format Downgrade (q8_0 -> q5_K) — REJECTED"
  and its 4-bit addendum). Kernel efficiency, not weight-byte count, is the
  decode bottleneck.
- Hypothesis: CTA-cooperative staged decode at a geometry that still fills the
  device (K-split x span fill) multiplies small-M throughput; format questions
  reopen only after the kernel approaches the roofline.
- Evidence so far: ksplit in-GEMM q2_K already beats the dequant route (13.76
  vs 13.20 TFLOP/s); the golden shape is occupancy-bound (128 waves on 304
  CUs); scattered 84-byte-strided per-lane loads dominate q2_K (2026-07-26
  "In-GEMM k-quant Decode" open questions); `mxfp4_gguf` GEMV at 1840.7 GB/s
  (~1/3 peak) shows the same headroom at M=1 (2026-08-02).
- Next action: port Metal's `dequant_into_shared` (CTA cooperatively decodes a
  BN x BK tile into LDS with coalesced reads) into the K-split decode
  geometry and A/B at o_proj (6144x16384) and q_b (16384x2048), M∈{1,16,64}.
- Kill criteria: the staged small-M kernel is <10% faster than the current
  ksplit fragment path at M≤64 serving shapes, or achieved bandwidth stays
  <25% of roofline after the staging A/B — then record "decoder-ALU-bound at
  small M" in findings as the standing verdict and drop the family.

### 3. Attention MFMA next tier
- Parent result: the MFMA-tiled forward runs 97.3/77.2 TFLOP/s — 0.30–0.34x of
  SDPA flash (321/223) on the same shape (2026-07-07 "Attention Forward
  MFMA-Tiling — LANDED"; "Library/Framework Baselines — RECORDED").
- Hypothesis: LDS-staged K/V tiles, larger BQ (32/64 queries per block), and
  tic/toc double-buffering close a large part of the SDPA gap; the same levers
  apply to `cross_attention` and `biased_attention` (2026-07-25 entries, same
  open-questions list).
- Evidence so far: K/V is currently loaded from global per fragment; each K/V
  block is reused by only 16 queries; three independent entries name the same
  three levers as the next step.
- Next action: one factor first — LDS-stage K/V tiles in
  `kernels/attention/gqa/variants/rocm_cdna3/attn_kernel.cuh` and A/B at
  B4 H32 H_KV8 N2048 D128, causal and non-causal, against the SDPA baseline.
- Kill criteria: all three levers individually miss the ≥8% complexity bar or
  any regresses causal/GQA/D=64 shapes — then record that the custom tier
  stops here and library routing (SDPA/AITER) is the path for large shapes.

### 4. Launch-bound serving fusion + sampler geometry
- Parent result: the `v2_batch_prep` family is launch-bound — ~4.5 us kernels
  against a measured 1.56 us empty-launch floor — and block-size tuning was
  rejected there (2026-08-01 "serving v2_batch_prep — MIXED");
  `v2_temperature` thr=1024 was kept at 1.04–1.97x (2026-08-01 "serving
  v2_sample — KEPT").
- Hypothesis: fusing the per-step batch-prep kernels into one launch recovers
  most of N x (launch floor) per serving step; the remaining sampler kernels
  (`v2_min_p`, `v2_topk_log_softmax`) may take the thr sweep but interact with
  `s_part[32]` cross-warp merges, so each needs its own correctness re-check.
- Evidence so far: the whole batch-prep family sits within ~3 us of the floor;
  the thr sweep already inverted once between decode and prefill shapes.
- Next action: prototype one fused per-step batch-prep launch in
  `kernels/serving/variants/rocm_cdna3` and measure per-step microseconds at
  nreq∈{8,64,256}, qlen∈{1,16384} (the GLM-5.2 serving widths).
- Kill criteria: fusion saves <2 us per step, or the family never appears in
  an end-to-end serving profile — record the negative and drop.

### 5. Norm/elementwise bandwidth vs F.layer_norm
- Parent result: shipped 64-lane norms run 2.2–3.1 TB/s while `F.layer_norm`
  measured 3372 GB/s at 16384x2048 (ours 0.71x) and 2956 vs 1655 at
  65536x8192 (0.56x) (2026-07-07 "Library/Framework Baselines — RECORDED").
- Hypothesis: float4-vectorized D-strided loads plus multi-row-per-wavefront
  for small hidden sizes close the torch gap (the deferred follow-up named in
  both the 2026-07-06 and 2026-07-07 elementwise entries).
- Evidence so far: rope_variants shows packed rows + per-lane ILP reaching
  ~75% of peak with the same row geometry — the norm path is not at a
  hardware wall.
- Next action: float4-load A/B on rms/ln forward over the perf.md §8
  rows x hidden matrix (oracle re-run first; 64-lane reduction semantics
  unchanged), compared against `F.layer_norm` on the same shapes.
- Kill criteria: both levers <8% on every shape, or torch parity reached —
  either way record the outcome in findings and remove the family.

## Parked (not on the beam)

- Iris in-kernel XGMI fused GEMM+reduce_scatter — feasibility proven,
  standalone all_reduce 0.10x of RCCL; needs a from-scratch producer/consumer
  kernel (2026-07-07, Iris — FEASIBILITY).
- Dense flux wide-tile double-buffered rewrite — structurally capped by the
  16x16 tile; hipBLASLt is the practical dense path (2026-07-07, flux LDS-B —
  REJECTED).
- MFMA-tiled attention backward (2026-07-06 GQA backward follow-up).
- `moe_gemm_gguf` span fill + prefill-loser fix — the grouped GGUF GEMM still
  uses the register-fragment geometry (2026-07-26 span-fill open questions).
- sdot4 quantized-query score dot + split-K `merge_attn_states` for
  `kv_cache_q8_0` / `decode_cache_attention` long contexts (2026-07-25 open
  questions).
- RoPE cos/sin table row in LDS when many heads share a token; float4 loads
  for the split layout (2026-07-25, rope_variants open questions).
- Swin D=32 MFMA path via the biased_attention fragment layout (2026-07-25,
  attn_composites open questions).
- `qgemv_fused` Q4_0 block-load vectorization; multi-wave pooling reduction
  for `mean_pool_rms_l2` long sequences (2026-07-26, Recent Metal Operation
  Parity follow-ups).
- nvfp4-decoded-to-fp8 MFMA prefill route — fp8 MFMA measured 1.98x fp16
  K-throughput (2059 vs 1039 TFLOP/s); only pays where the kernel is
  MFMA-bound (2026-07-26 addendum).
- `v2_min_p` / `v2_topk_log_softmax` block-size sweeps with per-kernel
  correctness re-checks (2026-08-01, v2_sample follow-up).
- `mxfp4_gguf` wider per-wave tile / multi-row blocking once it is wired into
  a serving path and shapes are known (2026-08-02).
- MFMA/LDS grouped-GEMM design to replace the scalar `grouped_gemm` route
  (2026-07-26, matmul-decode_epilogues).
- Multi-wave FP8 fake-quant reduction; FP16/BF16 embedding store variants if a
  shared storage dispatcher lands (2026-07-26, Phase 6 follow-ups).
- im2col+MFMA or shared-tile convolution for production shapes (2026-07-26,
  Phase 12).
- hipBLASLt/CK dequant-then-GEMM comparison at M≥64; AITER fused-MoE /
  attention comparators (2026-07-06 qgemm MFMA and 2026-07-07 baselines
  notes).
- BaseQ4/6 prefill LDS/MFMA specialization once a real model route exercises
  the format (2026-07-26, BaseQ Phase 5).
- Canonical serving shape set + paged-attention/MLA latency-vs-context
  normalization and partition-size sweep (2026-07-06, Serving Family port).

## Migrated sources

- `perf/optimization_status.md` "Open questions:" lines — 2026-07-06
  (elementwise, qgemv, serving, moe, qgemm MFMA), 2026-07-25
  (cross_attention, biased_attention, rope_variants, attn_composites,
  kv_cache_q8_0), 2026-07-26 (in-GEMM k-quant, span fill, format A/B, Phases
  4–13), 2026-08-01/02 (v2_sample, mxfp4_gguf) → beam #2/#3/#4/#5 and Parked.
- CANDIDATE-verdict entry: 2026-07-07 "qgemm Multi-Wave CTA LDS Tile" → beam
  #1.
- Explicit next-step lists in the 07-26..08-02 entries — span-fill follow-ups
  (a)/(b)/(c) → beam #1, #2 and Parked (`moe_gemm_gguf`); v2_batch_prep
  fusion note → beam #4; v2_sample sweep note and mxfp4 "obvious next
  experiment" → Parked.
- `perf/findings.md` (formerly perf.md §7) — structural-limiter and LDS
  findings → beam #1 framing and the parked flux rewrite; format rule → beam
  #2 kill criteria.
- perf.md §11 kernel-specific hypotheses — attention levers and quant
  crossovers → beam #3 and #5.
- The superseded 2026-07-06 `perf/baseline_status.md` migration tasks →
  `perf/baseline_status.md` "Superseded (historical)", not the beam
  (infrastructure, largely delivered by the 2026-07-25 Phase 0 harness).
