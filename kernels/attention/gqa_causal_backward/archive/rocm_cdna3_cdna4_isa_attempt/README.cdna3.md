This directory is a preserved CDNA3 port attempt copied from the CDNA4 causal
GQA backward variant.

It is not an active `variants/rocm_cdna3` implementation because it shares the
same CDNA4 assembly-register dependency as non-causal GQA backward. The copied
path requires instructions that do not assemble for gfx942.

A real CDNA3 port needs a backward kernel written around the supported CDNA3
MFMA and LDS load primitives instead of the CDNA4 art assembly path.
