# Based (Taylor linear attention) — CDNA3 (gfx942): correctness-valid

Port of QuixiCore-CUDA/kernels/based (ThunderKittens). Causal 2nd-order
Taylor-feature linear attention:
  o[n] = sum_{m<=n} V[m] * (1 + (Q_n.K_m)/sqrt(D) + (Q_n.K_m)^2/(2D))
(D=D_QK=16, DV=64). One 64-wide wavefront per (query,head,batch); lanes own the
DV value dims; the D=16 QK dot is per-lane. Validated vs an fp64 oracle
replicating gentests.py (`o vs fp64 ref` PASS ~0.14%). `make test`.
Perf follow-up: MFMA-tiled + explicit cumulative KV state output (kv_a1/kv_a2).
