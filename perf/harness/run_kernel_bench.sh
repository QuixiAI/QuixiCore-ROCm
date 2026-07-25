#!/usr/bin/env bash
#
# Run one kernel's harness, archive the raw output with full provenance, and
# emit a pre-filled `perf/optimization_status.md` entry.
#
# AGENTS.md requires every kernel commit to carry a focused performance run
# recording the kernel, route, dtype/format, shape set, correctness, baseline
# and candidate timings, GPU/ROCm/HIP versions, command line, warmups,
# iterations, median and variance, and a keep/reject decision. This script
# captures everything that can be captured mechanically so the only thing left
# to write by hand is the hypothesis and the decision.
#
# Usage:
#   perf/harness/run_kernel_bench.sh kernels/norms/rmsnorm
#   perf/harness/run_kernel_bench.sh --label rmsnorm-64lane kernels/norms/rmsnorm
#   perf/harness/run_kernel_bench.sh --target test kernels/vision/edge_mlp
#
# The kernel directory may be given as `kernels/<family>/<op>` or as the
# variant directory itself. Raw output lands in
# `perf/results/<date>/<label>/` (gitignored) and the notebook entry goes to
# stdout for review before pasting.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARCH="${ARCH:-rocm_cdna3}"
TARGET="bench"
LABEL=""
GPU_INDEX="${HIP_VISIBLE_DEVICES:-0}"

usage() {
    sed -n '3,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        -h|--help) usage 0 ;;
        --label)   LABEL="${2:?--label needs a value}"; shift 2 ;;
        --target)  TARGET="${2:?--target needs a value}"; shift 2 ;;
        --arch)    ARCH="${2:?--arch needs a value}"; shift 2 ;;
        -*)        echo "unknown option: $1" >&2; usage 2 ;;
        *)         break ;;
    esac
done

[ "$#" -ge 1 ] || usage 2
KERNEL_ARG="$1"

# ---------------------------------------------------------------------------
# Resolve the variant directory
# ---------------------------------------------------------------------------
resolve_dir() {
    local arg="$1" candidate
    # Accept absolute or repo-relative paths.
    if [ -d "$arg" ]; then candidate="$arg"
    elif [ -d "$REPO_ROOT/$arg" ]; then candidate="$REPO_ROOT/$arg"
    else
        echo "no such directory: $arg" >&2
        exit 1
    fi
    candidate="$(cd "$candidate" && pwd)"
    # Already a variant dir with a Makefile? Use it.
    if [ -f "$candidate/Makefile" ] && [[ "$candidate" == */variants/* || "$candidate" == */common ]]; then
        echo "$candidate"; return
    fi
    # Otherwise descend into variants/<arch>.
    if [ -f "$candidate/variants/$ARCH/Makefile" ]; then
        echo "$candidate/variants/$ARCH"; return
    fi
    # Or find the single variant Makefile beneath it, ignoring archives.
    local found
    found="$(find "$candidate" -path '*/archive/*' -prune -o \
                 -path "*/variants/$ARCH/Makefile" -print 2>/dev/null | head -2)"
    if [ "$(printf '%s\n' "$found" | grep -c .)" -eq 1 ] && [ -n "$found" ]; then
        dirname "$found"; return
    fi
    if [ -f "$candidate/Makefile" ]; then echo "$candidate"; return; fi
    echo "no $ARCH variant Makefile under: $arg" >&2
    exit 1
}

VARIANT_DIR="$(resolve_dir "$KERNEL_ARG")"
REL_DIR="${VARIANT_DIR#"$REPO_ROOT"/}"

if [ -z "$LABEL" ]; then
    # kernels/<family>/<op>/variants/<arch> -> <family>-<op>
    LABEL="$(printf '%s' "$REL_DIR" \
        | sed -E 's#^kernels/##; s#/variants/.*$##; s#/#-#g')"
    [ -n "$LABEL" ] || LABEL="$(basename "$VARIANT_DIR")"
fi

# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
RUN_DATE="$(date +%F)"
RUN_STAMP="$(date -Is)"
OUT_DIR="$REPO_ROOT/perf/results/$RUN_DATE/$LABEL"
mkdir -p "$OUT_DIR"
RAW="$OUT_DIR/${TARGET}.txt"

GIT_COMMIT="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
GIT_DIRTY=""
# --porcelain also reports untracked files; a new-but-uncommitted kernel is
# exactly the case where provenance matters most, so `git diff` is not enough.
if [ -n "$(git -C "$REPO_ROOT" status --porcelain 2>/dev/null)" ]; then
    GIT_DIRTY=" (working tree dirty)"
fi

rocm_version() {
    if [ -r /opt/rocm/.info/version ]; then cat /opt/rocm/.info/version
    else echo unknown; fi
}
hip_version() {
    /opt/rocm/bin/hipcc --version 2>/dev/null | sed -n 's/^HIP version: //p' | head -1 \
        || echo unknown
}
gpu_name() {
    rocm-smi --showproductname 2>/dev/null \
        | sed -n "s/^GPU\[$GPU_INDEX\][[:space:]]*:[[:space:]]*Card Series:[[:space:]]*//p" \
        | head -1
}
gpu_arch() {
    rocm-smi --showproductname 2>/dev/null \
        | sed -n "s/^GPU\[$GPU_INDEX\][[:space:]]*:[[:space:]]*GFX Version:[[:space:]]*//p" \
        | head -1
}

ROCM_VERSION="$(rocm_version)"
HIP_VERSION="$(hip_version)"
GPU_NAME="$(gpu_name)"; GPU_NAME="${GPU_NAME:-unknown}"
GPU_ARCH="$(gpu_arch)"; GPU_ARCH="${GPU_ARCH:-unknown}"
CONTAINER="${QC_CONTAINER:-bare metal (no container)}"
CMD="make -C $REL_DIR $TARGET"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
{
    echo "# QuixiCore-ROCm kernel run"
    echo "kernel_dir:   $REL_DIR"
    echo "label:        $LABEL"
    echo "target:       $TARGET"
    echo "command:      HIP_VISIBLE_DEVICES=$GPU_INDEX $CMD"
    echo "timestamp:    $RUN_STAMP"
    echo "git_commit:   $GIT_COMMIT$GIT_DIRTY"
    echo "gpu:          $GPU_NAME ($GPU_ARCH), device index $GPU_INDEX"
    echo "rocm:         $ROCM_VERSION"
    echo "hip:          $HIP_VERSION"
    echo "container:    $CONTAINER"
    echo "host:         $(hostname)"
    echo "---"
} > "$RAW"

echo "Running: $CMD  (HIP_VISIBLE_DEVICES=$GPU_INDEX)" >&2
set +e
HIP_VISIBLE_DEVICES="$GPU_INDEX" make -C "$VARIANT_DIR" "$TARGET" 2>&1 | tee -a "$RAW"
STATUS="${PIPESTATUS[0]}"
set -e

# The harnesses print "ALL PASS" on success; treat a missing verdict as failure
# even when make exits 0, so a harness that silently skips its checks cannot
# masquerade as a passing run.
CORRECTNESS="UNKNOWN — no verdict line found"
if grep -q '^ALL PASS' "$RAW"; then
    CORRECTNESS="ALL PASS ($(grep -c 'PASS$' "$RAW") checks)"
elif grep -q '^FAIL' "$RAW"; then
    CORRECTNESS="FAIL — see raw output"
fi

# Guard against measuring GPU contention instead of the kernel. A timing whose
# max/min spread exceeds ~1.2x is not a usable median: another process was very
# likely sharing the device. This has already produced one inverted result (a
# 2.43x win that read as a 0.64x regression), so it is a hard warning, not a note.
SPREAD_LIMIT="${QC_SPREAD_LIMIT:-1.20}"
WORST_SPREAD="$(grep -oE 'spread [0-9]+\.[0-9]+x' "$RAW" \
    | grep -oE '[0-9]+\.[0-9]+' | sort -rn | head -1)"
CONTENDED=0
if [ -n "$WORST_SPREAD" ] && \
   awk "BEGIN{exit !($WORST_SPREAD > $SPREAD_LIMIT)}"; then
    CONTENDED=1
fi

echo >&2
if [ "$STATUS" -ne 0 ]; then
    echo "!! harness exited $STATUS — do NOT record this as a measured win." >&2
fi
if [ "$CONTENDED" -eq 1 ]; then
    echo "!! timing spread ${WORST_SPREAD}x exceeds ${SPREAD_LIMIT}x — the GPU was" >&2
    echo "   probably shared. Re-run on an idle device before recording a decision." >&2
    echo "   Check with: rocm-smi --showpids" >&2
fi
echo "Raw output: ${RAW#"$REPO_ROOT"/}" >&2
echo >&2

# ---------------------------------------------------------------------------
# Notebook entry skeleton
# ---------------------------------------------------------------------------
cat <<ENTRY
--- paste into perf/optimization_status.md, then fill the <> fields ---

## $RUN_DATE: $LABEL

Status: <not started | baselining | experimenting | candidate | landed | deferred>.

Current implementation: $REL_DIR
Current public route: <kernels.yaml operation name / dispatch entry point>

References inspected: <CPU reference file, Metal .metal source, related notebook entries>

Correctness: $CORRECTNESS
  Oracle: <fp64 host oracle / CPU reference>, tolerance <fp32|fp16|bf16|fp8|quantized>
  Shapes: <dtype/format and shape set covered>

Baseline: <baseline kernel or library, median ms, GB/s or TFLOP/s>
Experiments: <one factor changed; candidate median ms; delta vs baseline>

Environment:
  GPU: $GPU_NAME ($GPU_ARCH), device index $GPU_INDEX
  ROCm: $ROCM_VERSION   HIP: $HIP_VERSION
  Container: $CONTAINER
  Commit: $GIT_COMMIT$GIT_DIRTY
  Command: HIP_VISIBLE_DEVICES=$GPU_INDEX $CMD
  Warmups/iterations: <as printed by the harness, e.g. w10/i50>

Decision: <keep | reject> — <why>
Open questions: <next lever, or none>
Raw results: perf/results/$RUN_DATE/$LABEL/
ENTRY

exit "$STATUS"
