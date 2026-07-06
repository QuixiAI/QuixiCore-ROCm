# Hedgehog (hybrid linear+exact attention) — CDNA3 (gfx942): correctness-valid

Port of QuixiCore-CUDA/kernels/hedgehog (ThunderKittens). Dual-softmax learned
feature maps φ(x,map)=concat(softmax(x@map),softmax(-x@map)).clamp(1e-6), then a
block-terraced (BLK=64) hybrid: recent 128-token window uses EXACT softmax
attention (beta-weighted), older tokens use LINEAR attention Qs·Ks (alpha-
weighted); combined and normalized. D_QK=D_VO=128, feat=128.

Two kernels: `feat_ker` (feature maps, 64-lane softmax) + `attn_ker` (one
wavefront per query, lanes own DVO, windowed-exact + linear + normalize).
Validated vs an fp64 oracle replicating gentests.py: PASS ~0.16% on MI300X.
`make test`. Perf follow-up: MFMA tiling + chunked recurrent KV state.
