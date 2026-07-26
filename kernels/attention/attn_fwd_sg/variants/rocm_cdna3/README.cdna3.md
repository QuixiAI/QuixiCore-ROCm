# attn_fwd_sg_d256 - CDNA3

Port of the current QuixiCore-Metal `attn_fwd_sg_d256` contract.

Contract:

```text
q : fp32 (T, Hq, 256)
k : fp16 (T, Hkv, 256)
v : fp16 (T, Hkv, 256)
o : fp32 (T, Hq, 256)
Hq % Hkv == 0
window == 0 => full bidirectional attention
window > 0  => symmetric window with half-width window/2
score = (q * scale) @ k
```

This CDNA3 port is correctness-first and scratch-free. One block owns one
`(query, query-head)` row, the 256 threads map one lane to one head dimension,
and a block reduction computes each key score. The output accumulator is updated
with online softmax, so there is no score matrix allocation. The harness also
includes a scalar one-thread-per-query/head baseline for the required A/B.

The existing `gqa_swa` kernel remains as the embeddinggemma specialization
(`Hkv=1`, fp16 pre-scaled Q, `[T,H*D]` layout). This op is the exact Metal public
shape with fp32 Q/O, runtime GQA head counts, and explicit `scale`.
