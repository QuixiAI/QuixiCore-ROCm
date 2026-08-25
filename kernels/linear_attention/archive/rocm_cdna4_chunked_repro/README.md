# CDNA4 chunked linear-attention repro (upstream import)

Imported from upstream HipKittens `kernels/cdna4/attn/repro/` during the
2026-08 upstream sync. It is a compile-and-inspect scaffold, not a contract
kernel: the chunk loop loads Q/K/V into registers and stops at a `// TODO`, and
the Makefile builds it standalone with `-Rpass-analysis=kernel-resource-usage
--save-temps` so the gfx950 ISA and register/LDS usage can be read off the
build.

Kept under `archive/` because it has no oracle, no benchmark, and no measured
result on this backend. The generated `*-amd-amdhsa-gfx950.s` listing that
upstream committed alongside it is not tracked here — `make` regenerates it.

```bash
make            # builds ./attn_kernel and the .s listing
make clean
```
