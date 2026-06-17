# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Tune driver for the CK FP8 block-wise *batched* GEMM.

Mirrors ``ck_batched_gemm_a8w8/batched_gemm_a8w8_tune.py`` -- iterates every
``(B, M, N, K)`` row in ``-i untuned.csv``, builds and times every
candidate kernel from ``batched_gemm_fp8_blockwise_instance.candidate_kernels_dict``,
verifies correctness against a torch-dequant + ``torch.bmm`` oracle, and
writes the best (kernelId, splitK, us) per shape to ``-o tuned.csv``.

Run inside the rocm/atom-dev:vllm-latest docker (or any image with hipcc):

    python3 csrc/ck_batched_gemm_fp8_blockwise/batched_gemm_fp8_blockwise_tune.py \\
        -i aiter/configs/fp8_blockwise_untuned_batched_gemm.csv \\
        -o aiter/configs/fp8_blockwise_tuned_batched_gemm.csv
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch

# Add the ck dir to sys.path so the codegen module is importable when this
# script is invoked from any CWD.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

from batched_gemm_fp8_blockwise_instance import candidate_kernels_dict


def _torch_oracle(XQ, WQ, x_scale, w_scale) -> torch.Tensor:
    """Torch dequant + bmm reference."""
    B, M, K = XQ.shape
    _, N, _ = WQ.shape
    Kg = K // 128
    Ng = N // 128
    a = XQ.to(torch.float32).view(B, M, Kg, 128) * x_scale.unsqueeze(-1)
    a = a.view(B, M, K).to(torch.bfloat16)
    w = WQ.to(torch.float32).view(B, Ng, 128, Kg, 128) * w_scale.view(B, Ng, 1, Kg, 1)
    w = w.view(B, N, K).to(torch.bfloat16)
    return torch.bmm(a, w.transpose(1, 2)).to(torch.bfloat16)


def _make(B, M, N, K, *, seed=0, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    A = (torch.randn(B, M, K, generator=g, device=device, dtype=torch.float32) * 0.5).clamp(-8, 8).to(torch.float8_e4m3fn)
    W = (torch.randn(B, N, K, generator=g, device=device, dtype=torch.float32) * 0.5).clamp(-8, 8).to(torch.float8_e4m3fn)
    A_s = torch.rand(B, M, K // 128, generator=g, device=device, dtype=torch.float32) * 0.1 + 0.01
    W_s = torch.rand(B, N // 128, K // 128, generator=g, device=device, dtype=torch.float32) * 0.1 + 0.01
    return A, W, A_s, W_s


def _bench_one(tune_fn, A, W, A_s, W_s, kid: int, *, iters=20, warmup=5) -> float:
    Y = torch.empty((A.size(0), A.size(1), W.size(1)), dtype=torch.bfloat16, device=A.device)
    # Correctness check first.
    ref = _torch_oracle(A, W, A_s, W_s)
    try:
        tune_fn(A, W, A_s, W_s, Y, kid, 0)
    except Exception as e:
        return float("inf")
    err = (Y.float() - ref.float()).abs().max().item() / max(ref.float().abs().max().item(), 1e-6)
    if err > 0.1:
        return float("inf")
    # Time.
    for _ in range(warmup):
        tune_fn(A, W, A_s, W_s, Y, kid, 0)
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        tune_fn(A, W, A_s, W_s, Y, kid, 0)
        torch.cuda.synchronize()
        samples.append((time.perf_counter_ns() - t0) / 1e3)
    samples.sort()
    return samples[len(samples) // 2]


def main() -> int:
    ap = argparse.ArgumentParser(description="Tune CK FP8 block-wise batched GEMM")
    ap.add_argument("-i", "--input_file", required=True, help="untuned shapes CSV (B,M,N,K columns)")
    ap.add_argument("-o", "--output_file", required=True, help="best kernel CSV out")
    ap.add_argument("--profile_file", default="", help="optional all-results CSV")
    ap.add_argument("--sort", default=True, type=lambda x: x.lower() != "false")
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("requires CUDA/HIP device for tuning")

    # Late import so the JIT build is triggered exactly once per (rebuild).
    import aiter  # noqa: F401  (ensures module_batched_gemm_fp8_blockwise_tune is JIT-built)
    from aiter.ops._ck_batched_gemm_fp8_blockwise_loader import batched_gemm_fp8_blockwise_tune as tune_fn

    cu_num = torch.cuda.get_device_properties(0).multi_processor_count

    df_in = pd.read_csv(args.input_file)
    rows = []
    profile_rows = []
    for _, row in df_in.iterrows():
        B, M, N, K = int(row["B"]), int(row["M"]), int(row["N"]), int(row["K"])
        A, W, A_s, W_s = _make(B, M, N, K, seed=B * M + N + K)
        best_kid, best_us = -1, float("inf")
        for kid in candidate_kernels_dict.keys():
            us = _bench_one(tune_fn, A, W, A_s, W_s, kid, iters=args.iters)
            profile_rows.append({
                "cu_num": cu_num, "libtype": "ck",
                "B": B, "M": M, "N": N, "K": K,
                "kernelId": kid, "splitK": 0, "us": us,
                "kernelName": candidate_kernels_dict[kid].name,
            })
            if us < best_us:
                best_us, best_kid = us, kid
            print(f"[tune] B={B} M={M} N={N} K={K} kid={kid:2d} us={us:.2f}")
        flops = 2 * B * M * N * K
        tflops = flops / (best_us * 1e-6) / 1e12 if best_us != float("inf") else 0.0
        rows.append({
            "cu_num": cu_num, "libtype": "ck",
            "B": B, "M": M, "N": N, "K": K,
            "kernelId": best_kid, "splitK": 0, "us": best_us,
            "kernelName": candidate_kernels_dict[best_kid].name if best_kid >= 0 else "",
            "tflops": tflops,
        })
        print(f"[best] B={B} M={M} N={N} K={K} -> kid={best_kid} us={best_us:.2f} ({tflops:.1f} TFLOPs)")

    out = pd.DataFrame(rows)
    if args.sort:
        out = out.sort_values(["cu_num", "B", "N", "M", "K"]).reset_index(drop=True)
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_file, index=False)
    print(f"[tune] wrote {args.output_file}")
    if args.profile_file:
        pd.DataFrame(profile_rows).to_csv(args.profile_file, index=False)
        print(f"[tune] wrote {args.profile_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
