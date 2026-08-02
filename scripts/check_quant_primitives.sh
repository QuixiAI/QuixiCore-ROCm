#!/usr/bin/env bash
# Guard the quant primitives against re-privatisation.
#
# include/cdna3/common/quant_primitives.cuh is the single definition of the
# byte-wise integer SIMD helpers and the E8M0 scale decoders. Both had drifted
# into per-family copies before it existed, and the E8M0 case was not benign:
# three different behaviours at code 0 lived under two names, so a kernel that
# grabbed the nearest copy could silently decode a block to zero.
#
# This flags two things:
#   1. a kernel defining a primitive the common header already owns
#   2. a bare `e8m0_decode` -- the convention has to be named, because code 0
#      is +0.0, 2^-127 or 2^-126 depending on which format you are decoding
#
# Pre-existing offenders are listed in the allowlist below with the convention
# they actually implement. Removing an entry means migrating that family.
set -uo pipefail
cd "$(dirname "$0")/.."

fail=0

# Empty: every family now names the E8M0 convention it uses. Re-populating
# this is a decision, not a convenience -- an entry here means a file is
# allowed to hide which behaviour it implements.
ALLOW=''

allowed() { grep -qxF "$1" <<<"$ALLOW"; }

# 1. Primitives the common header owns, redefined locally.
while IFS=: read -r f _; do
  [ -z "$f" ] && continue
  allowed "$f" && continue
  echo "error: $f defines a primitive owned by include/cdna3/common/quant_primitives.cuh"
  fail=1
done < <(grep -rln --include=*.cu --include=*.cuh \
           -e '__forceinline__ uint32_t vcmpeq4' \
           -e '__forceinline__ uint32_t vsub4' \
           -e '__forceinline__ uint32_t vadd4' \
           -e '__forceinline__ int dp4a' \
           kernels/ 2>/dev/null | sed 's/$/:/')

# 2. Unqualified e8m0_decode: the convention must be in the name.
while IFS=: read -r f _; do
  [ -z "$f" ] && continue
  allowed "$f" && continue
  echo "error: $f uses a bare e8m0_decode; say _ieee, _ldexp or _ggml (code 0 differs)"
  fail=1
done < <(grep -rln --include=*.cu --include=*.cuh \
           -e 'e8m0_decode[^_a-zA-Z]' kernels/ 2>/dev/null | sed 's/$/:/')

if [ "$fail" = 0 ]; then
  echo "quant primitives: OK"
fi
exit "$fail"
