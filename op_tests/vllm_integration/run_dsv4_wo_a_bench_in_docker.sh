#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Run vllm DeepSeek V4 inside the AMD-published docker image
# (rocm/atom-dev:vllm-latest), with aiter.batched_gemm_fp8_blockwise enabled
# for the wo_a projection, and benchmark vs the BF16 reference fallback.
#
# Configurations measured back-to-back:
#   A. baseline       -- VLLM_DSV4_WO_A_AITER unset; ROCm BF16 reference path
#   B. aiter_wo_a     -- VLLM_DSV4_WO_A_AITER=1; aiter.batched_gemm_fp8_blockwise
#
# Outputs end up in $OUT_DIR (default ./vllm_bench_out).
#
# Prerequisites:
#   * 4x MI355X visible at /dev/dri/renderD128..131 (default)
#   * docker installed
#   * Hugging Face token / weights for sgl-project/DeepSeek-V4-Flash-FP8
#     (set HF_HOME or mount $HOME/.cache/huggingface)
#   * Local checkouts of /home/zhenchen/projects/aiter and (optional)
#     /home/zhenchen/projects/vllm
#
# The bench uses vllm's bench_serving with random-prompt traffic.

set -euo pipefail

IMG="${IMG:-rocm/atom-dev:vllm-latest}"
OUT_DIR="${OUT_DIR:-./vllm_bench_out}"
mkdir -p "${OUT_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"

MODEL="${MODEL:-sgl-project/DeepSeek-V4-Flash-FP8}"
TP="${TP:-4}"
PORT="${PORT:-30010}"
NUM_PROMPTS="${NUM_PROMPTS:-200}"
RANDOM_INPUT_LEN="${RANDOM_INPUT_LEN:-1024}"
RANDOM_OUTPUT_LEN="${RANDOM_OUTPUT_LEN:-256}"
REQUEST_RATE="${REQUEST_RATE:-8}"

# Mount paths inside the container.
AITER_HOST="${AITER_HOST:-/home/zhenchen/projects/aiter}"
PATCH_DIR_HOST="${AITER_HOST}/op_tests/vllm_integration"
HF_CACHE_HOST="${HF_CACHE_HOST:-${HOME}/.cache/huggingface}"

# In the container.
AITER_CT=/workspace/aiter
PATCH_DIR_CT=${AITER_CT}/op_tests/vllm_integration
HF_CACHE_CT=/root/.cache/huggingface

run_one() {
    local label="$1"; shift
    local extra_env="$1"; shift

    echo "=========================================="
    echo "[bench] ${label}"
    echo "=========================================="

    # Boot vllm server in the docker container.
    docker run -d --rm \
        --name "vllm_dsv4_${label}_${TS}" \
        --device /dev/kfd --device /dev/dri \
        --network host \
        --ipc host --shm-size 32g \
        -v "${AITER_HOST}:${AITER_CT}" \
        -v "${HF_CACHE_HOST}:${HF_CACHE_CT}" \
        -e "PYTHONPATH=${AITER_CT}:${PATCH_DIR_CT}:\${PYTHONPATH:-}" \
        -e "VLLM_DSV4_WO_A_AITER=${extra_env}" \
        -e "VLLM_USE_TRITON_FLASH_ATTN=0" \
        "${IMG}" \
        bash -c "
            set -e
            # Reinstall aiter from the mounted source so we pick up our kernel.
            cd ${AITER_CT} && pip install -e . --no-deps --quiet
            # Auto-apply our wo_a patch by importing it.
            python -c 'import vllm_dsv4_wo_a_aiter_patch'
            # Launch.
            python -m vllm.entrypoints.openai.api_server \
                --model ${MODEL} \
                --tensor-parallel-size ${TP} \
                --port ${PORT} \
                --trust-remote-code \
                --enable-chunked-prefill \
                --max-model-len 8192 \
                2>&1 | tee /tmp/vllm_${label}.log
        "

    # Wait for /health.
    for i in $(seq 1 600); do
        if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
            echo "[bench] ${label} server up after ${i}s"
            break
        fi
        sleep 1
    done

    # Run bench (from host; it's just a HTTP client).
    python -m vllm.entrypoints.bench_serving \
        --backend openai \
        --base-url "http://127.0.0.1:${PORT}" \
        --model "${MODEL}" \
        --dataset-name random \
        --num-prompts "${NUM_PROMPTS}" \
        --random-input-len "${RANDOM_INPUT_LEN}" \
        --random-output-len "${RANDOM_OUTPUT_LEN}" \
        --request-rate "${REQUEST_RATE}" \
        --save-result \
        --result-dir "${OUT_DIR}" \
        --result-filename "${label}_${TS}.json"

    # Tear down.
    docker stop "vllm_dsv4_${label}_${TS}" >/dev/null 2>&1 || true
    docker logs "vllm_dsv4_${label}_${TS}" > "${OUT_DIR}/server_${label}_${TS}.log" 2>&1 || true
    sleep 5
}

run_one baseline   "0"
run_one aiter_wo_a "1"

echo
echo "Results in ${OUT_DIR}/{baseline,aiter_wo_a}_${TS}.json"
echo "Diff:"
python - <<EOF
import json, glob
for label in ("baseline", "aiter_wo_a"):
    f = sorted(glob.glob("${OUT_DIR}/" + label + "_${TS}.json"))[0]
    d = json.load(open(f))
    print(f"  {label:12s}  ttft={d.get('median_ttft_ms','?'):>6}ms  "
          f"tpot={d.get('median_tpot_ms','?'):>5}ms  "
          f"itl={d.get('median_itl_ms','?'):>5}ms  "
          f"output_throughput={d.get('output_throughput','?'):>7}tok/s")
EOF
