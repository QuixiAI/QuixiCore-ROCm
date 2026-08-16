# ROCm Established Findings — Do Not Re-Derive

Distilled from `perf/optimization_status.md` through 2026-08-02. Treat as
current truth until re-measured; every entry names its date and notebook
entry so it can be challenged with new data.

## Environment anchor

Every finding below was measured on the same box (from the notebook's
provenance blocks, 2026-07-06 → 2026-08-02):

- GPU: AMD Instinct MI300X (`gfx942:sramecc+:xnack-`), 304 CUs, 64 KiB LDS per
  workgroup, 8x per node; runs pinned with `HIP_VISIBLE_DEVICES`.
- ROCm 7.2.4, HIP 7.2.53211-97f5574fe2 (`hipcc` from ROCm 7.2.4); bare metal
  (no container); Ubuntu 22.04.5, driver 6.16.13 (recorded 2026-07-07).
- Framework baseline: repo venv PyTorch 2.12.1+rocm7.2 (`.venv`). Note
  `perf/perf.md` §1's 2026-07-27 hardware check records a newer nightly
  (2.14.0.dev20260723+rocm7.2); cite whichever the entry you rely on used.
- Standard timing: HIP events, 10 warmup / 50 timed iterations, median;
  launch-bound kernels batch repeated launches per sample. Empty-kernel launch
  floor on this system: 1.56 us (2026-08-01, `v2_batch_prep`).

## Wins

| finding | effect | date | notebook entry |
|---|---|---|---|
| Row-kernel 64-lane widening | +15–61% GB/s (shipped A/B +30–71%; table in perf.md §3) | 2026-07-07 | Elementwise/Norm 64-Lane Widening — LANDED |
| Attention forward MFMA-tiling | **13–16x** | 2026-07-07 | Attention Forward MFMA-Tiling — LANDED |
| `qgemm` wide N-tile (X-reuse) | +47–54% at M≥256 | 2026-07-07 | qgemm Wide N-Tile (X-reuse) — LANDED |
| `qgemm` multi-wave CTA LDS tile | +55–117% vs wide at M≥1024 (candidate, not yet routed) | 2026-07-07 | qgemm Multi-Wave CTA LDS Tile — CANDIDATE |
| LDS-staged k-quant tile decode (span fill) | 1.4–2.0x on ctaLDS; 86–106 TFLOP/s at q2_K prefill | 2026-07-26 | LDS-Staged k-quant Tile Decode (span fill) — LANDED |
| In-GEMM k-quant decode (dequant4 MFMA fragment) | landed; q4_K unpack 3.3x faster; ksplit in-GEMM beats the dequant route at decode shapes | 2026-07-26 | In-GEMM k-quant Decode (dequant4 MFMA fragment) — LANDED |
| `v2_temperature` block size 1024 | 1.04–1.97x, largest at decode token counts | 2026-08-01 | serving `v2_sample` — KEPT |
| `indexer_k_quant` one wave per token (thr=64) | 1.08–1.42x at every token count | 2026-08-01 | serving `sparse_indexer` — KEPT |
| `fp8_mqa_logits` fp8-MFMA port | bitwise-equal to Triton `tl.dot`; 111.8 TFLOP/s at H32 M512 N2048 | 2026-08-01 | serving `fp8_mqa_logits` |
| `mxfp4_gguf` GEMV (perm-table + `v_dot4`) | 1840.7 GB/s at 4096x4096, ~1/3 of HBM peak; first implementation | 2026-08-02 | quantization `mxfp4_gguf` |

Flat elementwise reference points (n = 64 Mi): `gelu_fwd` 2577 GB/s,
`add_ew` 3699 GB/s. Row kernels peak ~3070 GB/s; table-driven RoPE reaches
3.3–4.0 TB/s, ~75% of peak (2026-07-25, rope_variants). Against a 5.3 TB/s
roofline these are 49–75% — reasonable for read+write streaming, and the bar a
new elementwise kernel must clear.

## Rejected — with the reason, so they are not retried

**`flux` GEMM LDS-B staging — REJECTED (2026-07-07).** Staging a 16×16 B tile
through LDS to fix uncoalesced column-strided loads gave +16% at 2048³ but
**−6% at 4096³** (per-k-step barriers stop amortizing at large K). The decisive
observation: *both* kernels sit at 35–45 TFLOP/s against ~1300 peak, because
the structural limiter is the 16×16 output tile, not the B load. **Lesson: find
the structural limiter before optimizing memory access.** A 3% load-path win is
noise when you are at 3% of peak.

**`qgemm` LDS-staged wide tile (single-wave) — REJECTED (2026-07-07).**
Bit-identical, but slower than the existing register-fragment wide kernel
(0.23–0.60x). Coalescing the global loads does not offset the extra
global→LDS stores, LDS reads, and per-K-step barriers when only one wavefront
consumes the staged tile. Kept in `qgemm_bench.cu` as a comparator only. The
multi-wave CTA version, which creates real cross-wave W/dequant reuse, is the
kept CANDIDATE.

**Chunked-RCCL compute/comm overlap — REJECTED (2026-07-07).** Chunking along N
with async per-chunk collectives measured **0.82–0.96x** vs a single fused
collective on 4 and 8 GPUs. Correct, just slower. A hardware finding, not a
bug: on MI300X, RCCL collectives run as compute kernels on the CUs, so the
collective and the GEMM contend for the same CUs, and chunking also shrinks
RCCL's message size. Real overlap needs device-initiated XGMI from the GEMM
epilogue (Iris), not re-tuned chunk counts.

**Attention weight downgrade q8_0 → q5_K — REJECTED (2026-07-26).** The
hypothesis was textbook-correct and still wrong, which is why it is worth
remembering. Decode is weight-bandwidth bound (AI ≈ 6 FLOP/byte vs a roofline
far above it), attention tensors are ~56% of per-token decode traffic, so ~35%
fewer bytes should have bought ~20% end-to-end. It did not: **the k-quant
decoder is ALU-bound, so narrower formats cost more unpack work than they save
in bytes.** Every denser format tested (q6_K, q5_K, q4_K, q4_0, iq4_nl, mxfp4,
nvfp4) was *slower in wall clock* than q8_0 at attention shapes despite moving
fewer bytes; nvfp4 was worst at ~2.5x slower while moving 47% fewer bytes.
The kernels reach only 200–750 GB/s of the ~5.3 TB/s peak — nowhere near
bandwidth-bound. Standing project decision: GLM-5.2 serving stays on q8_0
attention + q2_K routed experts; no requantization.

**Generalized rule:** on CDNA3, "fewer bytes" only wins if the kernel is
actually near the bandwidth roofline. Measure achieved fraction of roofline
*first*. At ~14% of roofline you are ALU-bound and format changes will
backfire.

**RoPE 64-lane row widening — REJECTED (2026-07-25).** The repo's standard
first lever (worth +30–71% on the norm kernels) **lost 0.82–0.96x on all eight
rope_variants operations**, reproduced across two runs. These kernels already
pack two 32-lane rows per wavefront, so there is no idle half to reclaim;
halving lanes per row instead doubles per-lane ILP, and the variants with a
reduction lose the most. `kRopeLanes = 32` is pinned in the source beside the
reasoning. Rule: the lever is "fill the wavefront", not "widen the row".

**Grouped-GEMM wave64 split-dot — REJECTED (2026-07-26, matmul-decode_epilogues
Phase 3).** The same wave-per-output lever that wins 5.5–11.7x on decode
epilogues measured **0.16x** on dense `grouped_gemm` (groups=8, M=64, N=512,
K=2048). Decode is latency-shaped and benefits from a wavefront per output;
GEMM tiles do not. Keep scalar until a real MFMA/LDS grouped-GEMM design
exists.

**Streaming LM-head argmax — REJECTED (2026-07-26, BaseQ Phase 5).** 13.72 ms
vs 0.0389 ms for the materialized-scores route: the streaming kernel saves
score storage but exposes too little parallel work at the measured vocab shape.

**Scalar fused decode for dense block-scaled GEMM — REJECTED (2026-07-07,
Remaining CUDA Functional Coverage).** For MXFP8/NVFP4 dense GEMM at the
measured shape, fused scalar dequant-in-GEMM was slower than explicit dequant +
fp32 GEMM (0.083 vs 0.045 ms; 0.138 vs 0.045 ms). Explicit+fp32 is the kept
CDNA3 baseline until an MFMA/LDS block-scaled kernel exists.

**`v2_batch_prep` decode-tuned block size — REJECTED (2026-08-01).** thr=64
wins ~15–18% at qlen=1 but is **0.42x at qlen=16384** — exactly the
chunked-prefill width the GLM-5.2 serving path runs. A block size tuned only on
decode shapes is tuned on the case with no work to parallelize. The family sits
within ~3 us of the 1.56 us launch floor: launch-bound, so fuse the per-step
kernels rather than tuning any of them.

## Patterns and generalized rules

**The LDS pattern (two-sided).** LDS staging **lost** for dense GEMM B-tiles
and for `qgemm` single-wave wide tiles, but **won 1.4–2.0x** for k-quant
span-fill decode and +55–117% for the multi-wave CTA tile (2026-07-07 and
2026-07-26 entries above). The distinction is reuse per barrier: staging pays
only when many lanes — and especially multiple wavefronts — read each staged
byte multiple times; cross-wave W/dequant reuse is what flipped the qgemm
result. Do not assume either direction — A/B it, and report which side of that
line the kernel is on.

- Find the structural limiter before optimizing memory access; at 3% of peak a
  load-path win is noise (2026-07-07, flux GEMM LDS-B Experiment).
- "Fewer bytes" wins only near the bandwidth roofline; measure achieved
  roofline fraction before any format change (2026-07-26, Attention
  Weight-Format Downgrade).
- Fill the wavefront, don't widen the row: where rows are already packed two
  per wave, 64-lane widening is a regression (2026-07-25, rope_variants).
- Wave-per-output split-dot wins for latency-shaped decode/GEMV work (5.5–79x
  across Phases 3–7) and loses for GEMM-shaped tiles (0.16x) (2026-07-26,
  matmul-decode_epilogues; BaseQ Phase 5).
- There is no default block size: the measured winners were 256
  (`v2_batch_prep`), 1024 (`v2_temperature`), and 64 (`indexer_k_quant`).
  Sweeps must span decode *and* prefill shapes, since the winner inverted
  between them once already (2026-08-01, serving entries).
- Kernels within a few us of the 1.56 us launch floor are launch-bound: fuse
  launches, don't tune geometry (2026-08-01, v2_batch_prep).
- Bit-exact Triton-parity contracts pin the lane width (32-lane bfly trees on
  the 64-wide wave), the `exp2` lowering, the IEEE divide, and the MFMA
  instruction + K order. They are contract terms, not tunables — the usual
  geometry sweep does not apply (2026-08-01, v2_sample and fp8_mqa_logits).
- To recover a reduction order, probe with `1 + 2^-24` sentinel sums instead of
  reading ISA; the probe found a bit-reversed head-to-lane mapping that static
  analysis had stalled on (2026-08-01, fp8_mqa_logits).
- Judge dot-product error as `|err| / sum|terms|`, not `|err| / |result|` — a
  cancelling denominator measures the data, not the kernel (2026-08-02,
  mxfp4_gguf).
- Re-run surprising results on an idle device: a median under GPU contention
  inverted a 2.43x win into a "0.64x regression"; the harness warns at >1.2x
  spread for this reason (2026-07-25, cross_attention).
- `std::max(worst, NaN)` keeps `worst`, so an all-NaN kernel can PASS a
  worst-rel check; count non-finite outputs explicitly (2026-07-26,
  paged_attention_mxfp8).
- A host replica that shares the kernel's helpers cannot catch a wrong spec —
  pin primitives with hand-computed values and verify harnesses by mutation
  (2026-07-26, MXFP8 paged KV-cache codec).

## Open contradictions

- **Dense-GEMM LDS staging.** `flux` LDS-B lost at 4096³ (2026-07-07, flux GEMM
  LDS-B — REJECTED) while the correctness-tier `gemm_staged` beat its direct
  twin 2.2x at 512³ f32 (2026-07-07, Metal Gap Kernel Ports — KEPT). Different
  geometries (MFMA vs scalar loop), but the pair leaves "when does LDS pay for
  dense B?" without a rule beyond reuse-per-barrier. Resolves with: one kernel
  A/B'd across 512³→4096³ with the reuse per staged byte counted per shape.
- **Norm kernels "near ceiling" vs torch.** The handbook treats 49–70% of
  roofline as the streaming bar, but `F.layer_norm` measured 3372 GB/s at
  16384x2048 vs our 2399 (0.71x) and 2956 vs 1655 at 65536x8192 (2026-07-07,
  Library/Framework Baselines — RECORDED) — the library exceeds the bar, so
  headroom exists. Resolves with: float4-vectorized loads + multi-row-per-wave
  A/B against `F.layer_norm` on the perf.md §8 shape matrix.
- **Quant format ordering is proven only for the M=16/64 MFMA path.** The M=1
  GEMV/MMVQ path was not measured, and the 2026-07-26 rejection itself warns
  the ordering "could flip" once the small-M kernel stops leaving ~7x against
  the roofline on the table. Resolves with: a format A/B on the `qgemv` path at
  M=1, and a re-run of the o_proj/q_b format table after the small-M staging
  fix.
