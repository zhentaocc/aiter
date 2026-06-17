"""
Compare per-shape latency: untuned baseline (kid=0) vs tuned (per-shape best
from tuned.csv) vs torch dequant+bmm reference.

Reads the SAME 24 shapes that were tuned (from fp8_blockscale_untuned CSV).
"""

from __future__ import annotations

import argparse
import math
import statistics as stats
import time
from pathlib import Path

import pandas as pd
import torch

import aiter  # noqa: F401 -- triggers tune-module load
from aiter.ops._ck_batched_gemm_fp8_blockscale_loader import (
    batched_gemm_fp8_blockscale_tune as tune_fn,
)
from aiter.ops.batched_gemm_op_fp8_blockscale import (
    _torch_batched_gemm_fp8_blockscale,
    convert_scales_to_ue8m0,
)


def _make(B, M, N, K, *, seed=0, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    A = (torch.randn(B, M, K, generator=g, device=device, dtype=torch.float32) * 0.5).clamp(-8, 8).to(torch.float8_e4m3fn)
    W = (torch.randn(B, N, K, generator=g, device=device, dtype=torch.float32) * 0.5).clamp(-8, 8).to(torch.float8_e4m3fn)
    A_s = torch.rand(B, M, K // 128, generator=g, device=device, dtype=torch.float32) * 0.1 + 0.01
    W_s = torch.rand(B, N // 128, K // 128, generator=g, device=device, dtype=torch.float32) * 0.1 + 0.01
    return A, W, A_s, W_s


def _time_us(fn, *, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return (start.elapsed_time(end) / iters) * 1000.0  # us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuned_csv", default="aiter/configs/fp8_blockscale_tuned_batched_gemm.csv")
    ap.add_argument("--baseline_kid", type=int, default=0, help="kernel id for untuned baseline")
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    df = pd.read_csv(args.tuned_csv)
    print(f"# Device: {torch.cuda.get_device_name(0)}")
    print(f"# baseline_kid={args.baseline_kid}  iters={args.iters}")
    print()
    print(f"{'B':>3} {'M':>5} {'N':>5} {'K':>5} | {'baseline_us':>11} {'tuned_us':>9} {'tuned_kid':>9} {'torch_us':>9} | "
          f"{'tune_x':>7} {'vs_torch':>9}")
    print("-" * 110)

    rows = []
    for _, r in df.iterrows():
        B, M, N, K = int(r["B"]), int(r["M"]), int(r["N"]), int(r["K"])
        kid_tuned = int(r["kernelId"])
        A, W, A_s, W_s = _make(B, M, N, K, seed=B * 10007 + M * 101 + N + K)
        Y = torch.empty((B, M, N), dtype=torch.bfloat16, device="cuda")

        try:
            us_base = _time_us(lambda: tune_fn(A, W, A_s, W_s, Y, args.baseline_kid, 0), iters=args.iters)
        except Exception as e:
            us_base = float("nan")
        try:
            us_tuned = _time_us(lambda: tune_fn(A, W, A_s, W_s, Y, kid_tuned, 0), iters=args.iters)
        except Exception as e:
            us_tuned = float("nan")
        # Torch reference: dequant + bmm. Fewer iters since slow.
        torch_iters = max(3, args.iters // 4)
        us_torch = _time_us(lambda: _torch_batched_gemm_fp8_blockscale(A, W, A_s, W_s),
                            warmup=2, iters=torch_iters)

        tune_speedup = us_base / us_tuned if not math.isnan(us_tuned) and us_tuned > 0 else float("nan")
        vs_torch = us_torch / us_tuned if not math.isnan(us_tuned) and us_tuned > 0 else float("nan")
        rows.append((B, M, N, K, us_base, us_tuned, kid_tuned, us_torch, tune_speedup, vs_torch))
        print(f"{B:>3} {M:>5} {N:>5} {K:>5} | {us_base:>11.1f} {us_tuned:>9.1f} {kid_tuned:>9d} {us_torch:>9.1f} | "
              f"{tune_speedup:>6.2f}x {vs_torch:>8.2f}x")

    # Aggregate summary
    print()
    valid = [r for r in rows if not (math.isnan(r[8]) or math.isnan(r[9]))]
    if valid:
        tune_speedups = [r[8] for r in valid]
        torch_speedups = [r[9] for r in valid]
        print(f"# tune speedup (baseline kid={args.baseline_kid} -> tuned): "
              f"min={min(tune_speedups):.2f}x  median={stats.median(tune_speedups):.2f}x  max={max(tune_speedups):.2f}x")
        print(f"# tuned vs torch dequant+bmm:                                "
              f"min={min(torch_speedups):.2f}x  median={stats.median(torch_speedups):.2f}x  max={max(torch_speedups):.2f}x")


if __name__ == "__main__":
    main()
