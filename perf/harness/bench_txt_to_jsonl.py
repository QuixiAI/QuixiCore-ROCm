#!/usr/bin/env python3
"""Synthesize schema-1 run.json + results.jsonl from ROCm bench.txt output.

The per-kernel harnesses (kernels/common/cdna3_harness.cuh and older
standalone benches) print human-readable text; run_kernel_bench.sh archives
it with a provenance header. This adapter parses those files into the
umbrella reporting format (docs/benchmarking.md, schema 1) so ROCm runs
validate, diff, and gate like every other backend. It invents nothing: rows
come only from timing lines actually present, and fields the text does not
state are left absent.

Usage:
  bench_txt_to_jsonl.py <run-dir> [source ...]

With no explicit sources, *.txt inside <run-dir> plus sibling directories
named <run-dir-basename>-* (the layout run_bench.sh produces via
run_kernel_bench.sh) are parsed. run.json and results.jsonl are written into
<run-dir>; an existing results.jsonl is overwritten, an existing run.json is
kept (only missing keys are filled).

Recognized timing-line dialects:
  name dims: 0.076 ms, 112.3 TFLOP/s
  name dims   0.0126 ms  96.1 GB/s
  name        0.0123 ms  (min 0.0120 max 0.0130 spread 1.08x, w10/i50)
"""

import json
import re
import sys
from pathlib import Path

SCHEMA = 1

HEADER_RE = re.compile(r"^(kernel_dir|label|target|command|cmd|timestamp|git_commit|commit|gpu|rocm|hip|hipcc|container|host):\s+(.*)$")
INLINE_ENV_RE = re.compile(r"^GPU:\s*(.+?)\s{2,}ROCm\s+(\S+)\s+HIP\s+(\S+)")
TIMING_RE = re.compile(
    r"^(?P<label>\S.*?):?\s+(?P<ms>[0-9]+\.[0-9]+)\s*(?P<msunit>ms|us/launch)"
    r"(?:[,\s]+(?P<tp>[0-9]+\.[0-9]+)\s*(?P<unit>GB/s|TFLOP/s|TOP/s))?"
    r"(?:.*?\(min\s+(?P<min>[0-9.]+)\s+max\s+(?P<max>[0-9.]+)\s+spread\s+(?P<spread>[0-9.]+)x,\s*w(?P<w>\d+)/i(?P<i>\d+)\))?"
)
UNIT_FIELD = {"GB/s": "gbps", "TFLOP/s": "tflops", "TOP/s": "tops"}


def parse_file(path):
    """Return (rows, header_meta, verdict) for one bench text file."""
    header = {}
    rows = []
    all_pass = False
    any_fail = False
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.rstrip()
        m = HEADER_RE.match(line)
        if m:
            header[m.group(1)] = m.group(2).strip()
            continue
        m = INLINE_ENV_RE.match(line)
        if m:
            header.setdefault("gpu", m.group(1).strip())
            header.setdefault("rocm", m.group(2))
            header.setdefault("hip", m.group(3))
            continue
        if line.startswith("ALL PASS"):
            all_pass = True
            continue
        if re.match(r"^FAIL\b", line):
            any_fail = True
            continue
        if "PASS" in line:
            continue
        m = TIMING_RE.match(line.strip())
        if m and m.group("ms"):
            label = m.group("label").strip()
            kernel = label.split()[0]
            ms = float(m.group("ms"))
            if m.group("msunit") == "us/launch":
                ms = ms / 1000.0
            row = {
                "schema": SCHEMA,
                "kernel": kernel,
                "variant": label,
                "shape": {},
                "dtype": "none",
                "status": "ok",
                "target_ms": ms,
                "source": str(path.name),
            }
            if m.group("tp"):
                row[UNIT_FIELD[m.group("unit")]] = float(m.group("tp"))
            if m.group("min"):
                row["target_min_ms"] = float(m.group("min"))
                row["target_max_ms"] = float(m.group("max"))
                row["target_spread"] = float(m.group("spread"))
                row["warmup"] = int(m.group("w"))
                row["iters"] = int(m.group("i"))
            rows.append(row)
    verdict = "fail" if any_fail else ("pass" if all_pass else "unknown")
    return rows, header, verdict


def main():
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    run_dir = Path(sys.argv[1])
    run_dir.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) > 2:
        sources = [Path(s) for s in sys.argv[2:]]
    else:
        sources = [run_dir]
        base = run_dir.name
        sources += sorted(p for p in run_dir.parent.glob(f"{base}-*") if p.is_dir())
    txts = []
    for s in sources:
        txts += sorted(s.glob("*.txt")) + sorted(s.glob("*.log")) if s.is_dir() else [s]
    txts = [t for t in txts if t.suffix == ".txt" or t.name == "bench.log"]
    if not txts:
        print(f"bench_txt_to_jsonl: no bench text found under {run_dir}", file=sys.stderr)
        return 1

    all_rows, merged_header, verdicts = [], {}, []
    for t in txts:
        rows, header, verdict = parse_file(t)
        verdicts.append((t, verdict))
        for k, v in header.items():
            merged_header.setdefault(k, v)
        if verdict == "fail":
            all_rows.append({
                "schema": SCHEMA, "kernel": header.get("kernel_dir", t.stem),
                "variant": "correctness", "shape": {}, "dtype": "none",
                "status": "fail", "notes": f"FAIL verdict in {t.name}",
                "source": t.name,
            })
        elif verdict == "pass" and not rows:
            # correctness-only file: represent the ALL PASS evidence
            all_rows.append({
                "schema": SCHEMA, "kernel": header.get("kernel_dir", t.stem),
                "variant": "correctness", "shape": {}, "dtype": "none",
                "status": "ok", "check_passed": True,
                "notes": f"ALL PASS, no timing lines in {t.name}",
                "source": t.name,
            })
        all_rows.extend(rows)

    if not all_rows:
        print("bench_txt_to_jsonl: no timing or verdict lines parsed — nothing to record",
              file=sys.stderr)
        return 1

    with (run_dir / "results.jsonl").open("w") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")

    run_json = run_dir / "run.json"
    meta = json.loads(run_json.read_text()) if run_json.exists() else {}
    git = merged_header.get("git_commit", "unknown")
    if "dirty" in git:
        git = git.split()[0] + "-dirty"
    defaults = {
        "schema": SCHEMA,
        "backend": "rocm",
        "repo": "QuixiAI/QuixiCore-ROCm",
        "contract": "v0.1",
        "git": git if git != "unknown" else merged_header.get("commit", "unknown"),
        "timestamp": merged_header.get("timestamp", "unknown"),
        "os": "Linux",
        "arch": "gfx942" if "gfx942" in merged_header.get("gpu", "") else "unknown",
        "device": merged_header.get("gpu", "unknown"),
        "rocm": merged_header.get("rocm"),
        "hip": merged_header.get("hip"),
        "container": merged_header.get("container"),
        "hostname": merged_header.get("host"),
        "warmup": None,
        "iters": None,
        "sources": [t.name for t in txts],
    }
    for k, v in defaults.items():
        meta.setdefault(k, v)
    run_json.write_text(json.dumps(meta, indent=2) + "\n")

    ok = sum(1 for r in all_rows if r["status"] == "ok")
    fails = sum(1 for r in all_rows if r["status"] == "fail")
    unknowns = [t.name for t, v in verdicts if v == "unknown"]
    print(f"bench_txt_to_jsonl: {ok} timing row(s), {fails} fail row(s) -> {run_dir}/results.jsonl")
    if unknowns:
        print(f"bench_txt_to_jsonl: WARNING no ALL PASS/FAIL verdict in: {', '.join(unknowns)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
