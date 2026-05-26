#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# End-to-end benchmark: PR #23608 (sglang + DeepSeek V4 on AMD) with the
# torch fallbacks vs PR #23608 with our two FlyDSL/Triton kernels enabled.
#
# This script does NOT spin up a docker container -- it assumes you are
# already inside an environment where:
#   * sglang is the PR #23608 branch (gh pr checkout 23608 in /home/zhenchen/projects/sglang)
#   * aiter is built and installed from /home/zhenchen/projects/aiter
#   * 4x MI355X are visible at CUDA indices 0..3
#   * DeepSeek-V4-Flash-FP8 weights are accessible (HF repo: sgl-project/DeepSeek-V4-Flash-FP8)
#
# It runs two configurations back to back and stores results in
# ./bench_out/{baseline,flydsl}_<timestamp>.json.

set -euo pipefail

OUT_DIR="${OUT_DIR:-./bench_out}"
mkdir -p "${OUT_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"

PORT="${PORT:-30010}"

# Common env block from PR #23608 launch command.
common_env() {
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    export SGLANG_OPT_USE_FUSED_COMPRESS=false
    export SGLANG_OPT_USE_OLD_COMPRESSOR=true
    export SGLANG_OPT_USE_TILELANG_SWA_PREPARE=false
    export SGLANG_OPT_USE_FUSED_HASH_TOPK=false
    export SGLANG_HACK_FLASHMLA_BACKEND=torch
    export SGLANG_OPT_DEEPGEMM_HC_PRENORM=false
    export SGLANG_OPT_USE_TILELANG_MHC_PRE=false
    export SGLANG_OPT_USE_TILELANG_MHC_POST=false
    export SGLANG_ENABLE_THINKING=1
    export SGLANG_USE_AITER=1
    export SGLANG_USE_ROCM700A=1
    export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1
    export SGLANG_DSV4_FP4_EXPERTS=false
    export SGLANG_OPT_DPSK_V4_RADIX=0
    export SGLANG_OPT_USE_OVERLAP_STORE_CACHE=false
    export SGLANG_OPT_USE_FUSED_STORE_CACHE=false
    export SGLANG_FORCE_TRITON_MOE_FP8=1
}

LAUNCH_ARGS=(
    --model-path sgl-project/DeepSeek-V4-Flash-FP8
    --trust-remote-code
    --tp 4
    --dp 4
    --enable-dp-attention
    --disable-radix-cache
    --attention-backend compressed
    --max-running-request 256
    --page-size 256
    --chunked-prefill-size 8192
    --port "${PORT}"
    --disable-shared-experts-fusion
    --disable-cuda-graph
    --tool-call-parser deepseekv4
    --reasoning-parser deepseek-v4
)

BENCH_ARGS=(
    --backend sglang
    --dataset-name random
    --num-prompts 200
    --random-input-len 1024
    --random-output-len 256
    --request-rate 8
    --port "${PORT}"
)

run_config() {
    local label="$1"; shift
    echo "=========================================="
    echo "[bench] starting ${label}"
    echo "=========================================="

    # Boot server in background.
    python3 -m sglang.launch_server "${LAUNCH_ARGS[@]}" \
        > "${OUT_DIR}/server_${label}_${TS}.log" 2>&1 &
    local SERVER_PID=$!
    trap "kill ${SERVER_PID} 2>/dev/null || true" EXIT

    # Wait for /health.
    for i in $(seq 1 600); do
        if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
            echo "[bench] ${label} server up after ${i}s"
            break
        fi
        sleep 1
    done

    # Run the bench.
    python3 -m sglang.bench_serving "${BENCH_ARGS[@]}" \
        --output-file "${OUT_DIR}/${label}_${TS}.json"

    # Tear down.
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
    trap - EXIT
    echo "[bench] ${label} done."
    sleep 5
}

# --- Configuration A: baseline (torch fallbacks as in PR #23608 launch cmd). ---
common_env
export SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=false
export SGLANG_TOPK_TRANSFORM_512_TORCH=1
export SGLANG_OPT_FLYDSL_FUSED_GATE=0
export SGLANG_OPT_FLYDSL_TOPK_TRANSFORM=0
run_config "baseline"

# --- Configuration B: enable our two FlyDSL/Triton kernels. ---
common_env
export SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=false   # leave the CUDA path off
export SGLANG_TOPK_TRANSFORM_512_TORCH=1            # leave the torch path as the
                                                    # "off" branch fallback;
                                                    # the FLYDSL flag takes
                                                    # precedence in indexer.py
export SGLANG_OPT_FLYDSL_FUSED_GATE=1
export SGLANG_OPT_FLYDSL_TOPK_TRANSFORM=1
run_config "flydsl"

echo
echo "Results written to ${OUT_DIR}/{baseline,flydsl}_${TS}.json"
echo "Diff with: python3 -c \"import json; b=json.load(open('${OUT_DIR}/baseline_${TS}.json')); f=json.load(open('${OUT_DIR}/flydsl_${TS}.json')); print({k: (b[k], f[k]) for k in ('median_ttft_ms','median_tpot_ms','median_itl_ms','output_throughput')})\""
