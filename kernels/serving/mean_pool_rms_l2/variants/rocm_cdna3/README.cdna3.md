# mean_pool_rms_l2 - CDNA3

Port of the current QuixiCore-Metal `kernels/serving/mean_pool_rms_l2` contract.
This is not the older `pool_mean_rms_l2` embeddinggemma head: this Metal op
mean-pools first, applies RMSNorm with a bf16 weight, then L2-normalizes the
weighted vector.

Contract:

```text
x      : bf16 (M, D), D in {256,512,768,1024}
weight : bf16 (D)
out    : bf16 (D)

p = mean_rows(x)
n = p * rsqrt(mean(p^2) + eps) * weight
out = n * rsqrt(sum(n^2) + 1e-12)
```

The CDNA3 candidate uses one wave64 for the pooled vector. Each lane owns
`D/64` dimensions in registers, streams all `M` rows once, performs both
reductions with wave shuffles, and writes bf16 output. The harness also carries
a one-thread scalar baseline for the required focused A/B.

Run:

```bash
make test
make bench
```
