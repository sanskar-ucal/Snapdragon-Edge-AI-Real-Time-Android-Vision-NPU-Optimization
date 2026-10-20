#!/usr/bin/env bash
# Builds every model variant, evaluates and benchmarks them on the host, and regenerates the report.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-$ROOT/ml/.venv/bin/python}"
cd "$ROOT/ml"

"$PY" export.py
"$PY" quantize.py
"$PY" evaluate.py
"$PY" bench_host.py
"$PY" aihub_profile.py
"$PY" report.py
