#!/usr/bin/env bash
#
# QuixiCore ROCm bench entrypoint. Thin wrapper around the shared core
# (run_bench_core.sh, synced from the umbrella); this file is hand-written
# and backend-owned.
#
#   perf/harness/run_bench.sh --label rmsnorm-ab -- kernels/norms/rmsnorm
#   perf/harness/run_bench.sh --dry-run -- kernels/norms/rmsnorm
#
# Wraps the existing per-kernel run_kernel_bench.sh (which owns provenance,
# the ALL PASS verdict guard, and the spread guard for the prose bench.txt),
# then synthesizes schema-1 run.json + results.jsonl from its output via
# bench_txt_to_jsonl.py so the run validates and diffs like every other
# backend. Kernel directories are passed after `--`.
# SCAFFOLDING NOTE: the wrapper itself has not run on an MI300X host; the
# bench.txt parser is verified against the committed perf/results fixtures.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
QC_BACKEND="rocm"

qc_bench_cmd() {
    if [ "${#QC_PASSTHROUGH[@]}" -eq 0 ]; then
        echo "usage: run_bench.sh [--label L] [--dry-run] -- <kernel-dir> [...]" >&2
        return 2
    fi
    local k
    for k in "${QC_PASSTHROUGH[@]}"; do
        qc_exec "$REPO_ROOT/perf/harness/run_kernel_bench.sh" \
            --label "$(basename "$OUT_DIR")-$(basename "$k")" "$k"
    done
    if [ "$QC_DRY_RUN" -eq 0 ]; then
        python3 "$REPO_ROOT/perf/harness/bench_txt_to_jsonl.py" "$OUT_DIR"
    fi
}

qc_device_info() {
    command -v rocm-smi >/dev/null 2>&1 && \
        echo "gpu=$(rocm-smi --showproductname 2>/dev/null | grep 'Card Series' | head -1)"
    [ -r /opt/rocm/.info/version ] && echo "rocm=$(cat /opt/rocm/.info/version)"
    echo "uname=$(uname -srm)"
}

source "$(dirname "${BASH_SOURCE[0]}")/run_bench_core.sh"
