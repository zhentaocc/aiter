#!/bin/bash
# Run the CK FP8 block-scale batched GEMM benchmark inside the aiter container.
#
# Usage:
#   bash op_tests/op_benchmarks/ck/run_bench.sh [GPU_ID] [PRESET] [EXTRA_ARGS...]
#
# Examples:
#   bash op_tests/op_benchmarks/ck/run_bench.sh                 # GPU 0, dsv4 preset
#   bash op_tests/op_benchmarks/ck/run_bench.sh 3 smoke         # GPU 3, smoke preset
#   bash op_tests/op_benchmarks/ck/run_bench.sh 0 dsv4 --no-accuracy -o /results/r.csv
#
# Assumes a writable, self-consistent aiter checkout. By default it uses the
# directory this script lives in (resolved to the repo root). Override with
# AITER_ROOT=/path/to/aiter.
set -euo pipefail

GPU_ID="${1:-0}"
PRESET="${2:-dsv4}"
shift || true; shift || true   # drop the two positional args; rest are passthrough

# Resolve repo root (this script is at <root>/op_tests/op_benchmarks/ck/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AITER_ROOT="${AITER_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
BENCH="$AITER_ROOT/op_tests/op_benchmarks/ck/bench_batched_gemm_fp8_blockscale.py"

echo "[run_bench] AITER_ROOT=$AITER_ROOT"
echo "[run_bench] GPU=$GPU_ID  preset=$PRESET  extra='$*'"

cd "$AITER_ROOT"
HIP_VISIBLE_DEVICES="$GPU_ID" PYTHONPATH="$AITER_ROOT" \
    python3 "$BENCH" --preset "$PRESET" "$@"
