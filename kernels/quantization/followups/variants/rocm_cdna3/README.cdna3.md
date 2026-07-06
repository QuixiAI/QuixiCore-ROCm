# followups (cross-cutting integration) — CDNA3 (gfx942): correctness-valid

Port of QuixiCore-CUDA/kernels/quant/followups_test.cu, exercising turboquant
KV encode, mamba2 selective_scan (APC + chunk-state checkpoints), EAGLE
speculative helpers, sparse-serving indexer/merge_attn_states_fp8. Reuses the
already-ported serving / mamba2 / turboquant headers. ALL PASS on MI300X.
