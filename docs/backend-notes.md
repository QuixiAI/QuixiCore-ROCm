# Backend Notes

QuixiCore ROCm is the AMD implementation of the QuixiCore contract. ROCm source
may use HIP, CK, rocBLAS, rocWMMA, RCCL, and architecture-specific variants when
those choices remain behind the shared operation semantics.

Current migration concern: active kernels live in focused legacy directories
such as `attn`, `gemm`, `layernorm`, `rotary`, and `softmax`. New work should
use the semantic family layout documented in `docs/repository-structure.md`.
