#!/bin/bash
# Parallel autotune across N GPUs. Splits the untuned CSV row-wise and runs
# one worker per pinned GPU; merges the per-worker tuned CSVs at the end.
#
# Usage (inside container):
#   bash run_tune_parallel.sh <untuned_csv> <out_tuned_csv> <gpus> <iters>
#
# Example:
#   bash op_tests/run_tune_parallel.sh \
#       aiter/configs/fp8_blockscale_untuned_batched_gemm.csv \
#       aiter/configs/fp8_blockscale_tuned_batched_gemm.csv \
#       "4,5,6,7" 20

set -eu

UNTUNED="${1:-aiter/configs/fp8_blockscale_untuned_batched_gemm.csv}"
OUT_TUNED="${2:-aiter/configs/fp8_blockscale_tuned_batched_gemm.csv}"
GPUS_CSV="${3:-4,5,6,7}"
ITERS="${4:-20}"

WORK_DIR="${WORK_DIR:-/tmp/tune_parallel}"
LOG_DIR="${LOG_DIR:-/results/tune_logs}"
TUNE_PY="${TUNE_PY:-csrc/ck_batched_gemm_fp8_blockscale/batched_gemm_fp8_blockscale_tune.py}"

mkdir -p "$WORK_DIR" "$LOG_DIR"
IFS=',' read -ra GPUS <<< "$GPUS_CSV"
N=${#GPUS[@]}

# ---- Split CSV row-wise (preserving header) into N chunks ----
header=$(head -1 "$UNTUNED")
tail -n +2 "$UNTUNED" > "$WORK_DIR/all_rows.csv"
total=$(wc -l < "$WORK_DIR/all_rows.csv")
per=$(( (total + N - 1) / N ))
split -d -l "$per" "$WORK_DIR/all_rows.csv" "$WORK_DIR/chunk_"

# ---- Launch workers ----
PIDS=()
for i in "${!GPUS[@]}"; do
  gpu="${GPUS[$i]}"
  chunk=$(printf "%s/chunk_%02d" "$WORK_DIR" "$i")
  if [ ! -f "$chunk" ]; then continue; fi

  in_csv="$WORK_DIR/in_gpu${gpu}.csv"
  out_csv="$WORK_DIR/out_gpu${gpu}.csv"
  log="$LOG_DIR/worker_gpu${gpu}.log"

  echo "$header" > "$in_csv"
  cat "$chunk" >> "$in_csv"

  echo "[launch] gpu=$gpu  shapes=$(wc -l < "$chunk")  log=$log"
  HIP_VISIBLE_DEVICES="$gpu" python3 "$TUNE_PY" \
    -i "$in_csv" -o "$out_csv" --iters "$ITERS" \
    > "$log" 2>&1 &
  PIDS+=($!)
done

# ---- Wait ----
echo "[parallel] waiting on ${#PIDS[@]} workers..."
fail=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then fail=$((fail+1)); fi
done
echo "[parallel] all workers exited (failed: $fail)"

# ---- Merge ----
mkdir -p "$(dirname "$OUT_TUNED")"
first=1
for gpu in "${GPUS[@]}"; do
  out_csv="$WORK_DIR/out_gpu${gpu}.csv"
  [ -f "$out_csv" ] || continue
  if [ "$first" -eq 1 ]; then
    cp "$out_csv" "$OUT_TUNED"; first=0
  else
    tail -n +2 "$out_csv" >> "$OUT_TUNED"
  fi
done

echo "[parallel] merged -> $OUT_TUNED"
echo "[parallel] summary:"
wc -l "$OUT_TUNED" "$WORK_DIR"/out_gpu*.csv 2>/dev/null || true
exit "$fail"
