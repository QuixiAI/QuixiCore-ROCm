# Phase 7 Stragglers CDNA3 Variant

This standalone HIP harness lands the Phase 7 parity rows:

- `top_k_renorm`
- `top_p_renorm`
- `logits_softcap`
- `embedding_lookup_types`
- `broadcast`
- `reduce_sum`

The sampling references are CPU `utils/utils_extended_ref.cpp` and Metal
`sampling/sampling/sampling_transforms.metal`. The embedding reference is CPU
`serving/basert_ref.cpp`, with the typed Metal embedding shape used as a layout
cross-check. `broadcast` and root `reduce_sum` mirror CPU
`collectives/collectives_ref.cpp` as single-device tensor kernels; multi-GPU
transport remains covered by the existing RCCL collectives tree.

The top-k/top-p renormalizers preserve CPU stable ordering: larger probability
wins, exact ties break toward the lower token id. The parallel rank kernels are
measured against scalar selection baselines; when the exact parallel rank route
is slower, the notebook records the rejection rather than claiming a win.
