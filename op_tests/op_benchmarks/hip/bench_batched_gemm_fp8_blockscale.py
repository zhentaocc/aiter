# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Microbenchmark + accuracy harness for ``aiter.batched_gemm_fp8_blockscale``
(CK FP8 block-wise batched GEMM).

Reports for every (B, M, N, K) shape:
  - CK kernel latency (us)
  - Achieved TFLOP/s, bandwidth (GB/s)
  - Torch dequant+bmm reference latency (us) and speedup
  - Max abs error vs torch oracle (correctness check)

Follows the layout of ``op_tests/op_benchmarks/triton/bench_batched_gemm_a8w8.py``
so output / CLI behave the same way for downstream consumers.

Usage:
    python op_tests/op_benchmarks/hip/bench_batched_gemm_fp8_blockscale.py
    python op_tests/op_benchmarks/hip/bench_batched_gemm_fp8_blockscale.py --preset dsv4
    python op_tests/op_benchmarks/hip/bench_batched_gemm_fp8_blockscale.py \
            -b 8 -m 4096 -n 1024 -k 4096 --backend ck --no-accuracy

Backends:
  - ``auto`` (default): dispatcher picks CK when M >= 128 && M %% 128 == 0
    && N %% 128 == 0, else torch fallback
  - ``ck``  : force CK kernel (raises if shape not supported)
  - ``torch``: torch reference oracle (sanity / accuracy baseline)
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from dataclasses import dataclass

import torch

import aiter
from aiter.ops.batched_gemm_op_fp8_blockscale import (
    _torch_batched_gemm_fp8_blockscale,
    convert_scales_to_ue8m0,
)


# ---------------------------------------------------------------------------
# Shape presets: same idea as triton/bench_batched_gemm_a8w8.py model_*
# presets, but for DSv4 wo_a per-rank shapes (the kernel's production caller).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Shape:
    B: int
    M: int
    N: int
    K: int
    tag: str = ""


# All N=1024 (o_lora_rank), K=4096 (heads_per_group*head_dim=8*512).
# B = G_per_rank derived from {Flash, Pro} x TP per the upstream DSv4 config.
DSV4_SHAPES = [
    # B=1: Flash TP=8
    *[Shape(1, M, 1024, 4096, f"flash_tp8_M{M}") for M in (128, 256, 512, 1024, 2048, 4096)],
    # B=2: Flash TP=4 / Pro TP=8 (identical per-rank shape)
    *[Shape(2, M, 1024, 4096, f"flash_tp4_pro_tp8_M{M}") for M in (128, 256, 512, 1024, 2048, 4096)],
    # B=8: Flash TP=1
    *[Shape(8, M, 1024, 4096, f"flash_tp1_M{M}") for M in (128, 256, 512, 1024, 2048, 4096)],
    # B=16: Pro TP=1
    *[Shape(16, M, 1024, 4096, f"pro_tp1_M{M}") for M in (128, 256, 512, 1024, 2048, 4096)],
]

# Smoke set for quick CI runs.
SMOKE_SHAPES = [
    Shape(1, 128, 1024, 4096, "smoke_b1"),
    Shape(2, 256, 1024, 4096, "smoke_b2"),
    Shape(8, 1024, 1024, 4096, "smoke_b8"),
]

PRESETS = {
    "dsv4":  DSV4_SHAPES,
    "smoke": SMOKE_SHAPES,
}


# ---------------------------------------------------------------------------
# Input generation (mirrors test_batched_gemm_fp8_blockscale.py).
# ---------------------------------------------------------------------------

def _make_inputs(B: int, M: int, N: int, K: int, *, seed: int = 0, device: str = "cuda",
                 scale_dtype: torch.dtype = torch.uint8):
    g = torch.Generator(device=device).manual_seed(seed)
    A = (torch.randn(B, M, K, generator=g, device=device, dtype=torch.float32) * 0.5
         ).clamp(-8.0, 8.0).to(torch.float8_e4m3fn)
    W = (torch.randn(B, N, K, generator=g, device=device, dtype=torch.float32) * 0.5
         ).clamp(-8.0, 8.0).to(torch.float8_e4m3fn)
    K_g, N_g = K // 128, N // 128
    A_scale = (torch.rand(B, M, K_g, generator=g, device=device, dtype=torch.float32) * 0.1 + 0.01)
    W_scale = (torch.rand(B, N_g, K_g, generator=g, device=device, dtype=torch.float32) * 0.1 + 0.01)
    if scale_dtype == torch.uint8:
        # CK accepts u8 + fp32; for production we recommend u8 (cached convert).
        A_scale = convert_scales_to_ue8m0(A_scale)
        W_scale = convert_scales_to_ue8m0(W_scale)
    return A, W, A_scale, W_scale


# ---------------------------------------------------------------------------
# Timing helpers.
# ---------------------------------------------------------------------------

def _time_us(fn, *, warmup: int = 10, iters: int = 50) -> float:
    """Median microseconds per call over ``iters`` after ``warmup``."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter_ns() - t0) / 1e3)
    return statistics.median(samples)


# ---------------------------------------------------------------------------
# Per-shape bench (one row of the report).
# ---------------------------------------------------------------------------

def bench_one(shape: Shape, *, backend: str, iters: int, accuracy: bool,
              scale_dtype: torch.dtype) -> dict:
    B, M, N, K = shape.B, shape.M, shape.N, shape.K
    out_dtype = torch.bfloat16

    A, W, A_scale, W_scale = _make_inputs(B, M, N, K, seed=B * 10007 + M * 101 + N + K,
                                          scale_dtype=scale_dtype)
    Y = torch.empty((B, M, N), dtype=out_dtype, device="cuda")

    # ---- accuracy ----
    max_err = float("nan")
    rel_err = float("nan")
    if accuracy:
        # Torch reference uses fp32 scales; convert back if u8.
        if A_scale.dtype == torch.uint8:
            A_scale_ref = torch.exp2((A_scale.float() - 127.0))
            W_scale_ref = torch.exp2((W_scale.float() - 127.0))
        else:
            A_scale_ref, W_scale_ref = A_scale, W_scale
        ref = _torch_batched_gemm_fp8_blockscale(A, W, A_scale_ref, W_scale_ref)
        out_check = aiter.batched_gemm_fp8_blockscale(A, W, A_scale, W_scale,
                                                     backend=backend)
        diff = (out_check.float() - ref.float()).abs()
        max_err = diff.max().item()
        denom = ref.float().abs().max().item()
        rel_err = max_err / max(denom, 1e-6)

    # ---- perf ----
    # CK call.
    call_ck = lambda: aiter.batched_gemm_fp8_blockscale(
        A, W, A_scale, W_scale, out=Y, backend=backend)
    ck_us = _time_us(call_ck, warmup=10, iters=iters)

    # Torch oracle (always at lower iters since it's slow).
    if A_scale.dtype == torch.uint8:
        A_s_ref = torch.exp2((A_scale.float() - 127.0))
        W_s_ref = torch.exp2((W_scale.float() - 127.0))
    else:
        A_s_ref, W_s_ref = A_scale, W_scale
    torch_us = _time_us(lambda: _torch_batched_gemm_fp8_blockscale(A, W, A_s_ref, W_s_ref),
                        warmup=2, iters=max(5, iters // 8))

    # Throughput / bandwidth derived metrics.
    flops = 2.0 * B * M * N * K
    tflops = flops / (ck_us * 1e-6) / 1e12
    bytes_ab = A.numel() * A.element_size() + W.numel() * W.element_size()
    bytes_scale = A_scale.numel() * A_scale.element_size() + W_scale.numel() * W_scale.element_size()
    bytes_out = Y.numel() * Y.element_size()
    gb_s = (bytes_ab + bytes_scale + bytes_out) / (ck_us * 1e-6) / 1e9

    return {
        "tag": shape.tag, "B": B, "M": M, "N": N, "K": K,
        "ck_us": ck_us, "tflops": tflops, "gb_s": gb_s,
        "torch_us": torch_us, "speedup": torch_us / ck_us,
        "max_err": max_err, "rel_err": rel_err,
    }


# ---------------------------------------------------------------------------
# CLI + main.
# ---------------------------------------------------------------------------

def _print_header():
    cols = ["shape", "B", "M", "N", "K", "ck_us", "TFLOPS", "GB/s",
            "torch_us", "vs_torch", "max_err", "rel_err"]
    widths = [22, 3, 5, 5, 5, 8, 7, 6, 9, 8, 8, 8]
    print(" ".join(c.rjust(w) for c, w in zip(cols, widths)))
    print("-" * (sum(widths) + len(widths) - 1))


def _print_row(r: dict):
    print(f"{r['tag']:>22} {r['B']:>3d} {r['M']:>5d} {r['N']:>5d} {r['K']:>5d} "
          f"{r['ck_us']:>8.1f} {r['tflops']:>7.1f} {r['gb_s']:>6.1f} "
          f"{r['torch_us']:>9.1f} {r['speedup']:>7.2f}x "
          f"{r['max_err']:>8.4f} {r['rel_err']:>7.4f}")


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                  description=__doc__)
    ap.add_argument("--preset", choices=list(PRESETS), default="dsv4",
                    help="Shape preset (default: dsv4)")
    ap.add_argument("-b", "--batch", type=int, help="Override B (single-shape mode)")
    ap.add_argument("-m", type=int, help="Override M")
    ap.add_argument("-n", type=int, help="Override N")
    ap.add_argument("-k", type=int, help="Override K")
    ap.add_argument("--backend", choices=["auto", "ck", "torch"], default="auto")
    ap.add_argument("--scale-dtype", choices=["u8", "fp32"], default="u8",
                    help="Scale dtype passed to the kernel (default: u8 ue8m0)")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--no-accuracy", action="store_true",
                    help="Skip torch-oracle accuracy check (faster on big M)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("requires CUDA/HIP device", file=sys.stderr)
        return 1

    if args.batch is not None:
        if not (args.m and args.n and args.k):
            print("Single-shape mode requires -b -m -n -k", file=sys.stderr)
            return 1
        shapes = [Shape(args.batch, args.m, args.n, args.k, "cli")]
    else:
        shapes = PRESETS[args.preset]

    scale_dtype = torch.uint8 if args.scale_dtype == "u8" else torch.float32

    print(f"# device: {torch.cuda.get_device_name(0)}")
    print(f"# backend={args.backend}  scale_dtype={args.scale_dtype}  iters={args.iters}  "
          f"accuracy={'OFF' if args.no_accuracy else 'ON'}")
    print()
    _print_header()

    rows = []
    fail = 0
    for s in shapes:
        try:
            r = bench_one(s, backend=args.backend, iters=args.iters,
                          accuracy=not args.no_accuracy,
                          scale_dtype=scale_dtype)
        except Exception as e:
            fail += 1
            print(f"{s.tag:>22} {s.B:>3d} {s.M:>5d} {s.N:>5d} {s.K:>5d}  FAIL: {e}")
            continue
        rows.append(r)
        _print_row(r)

    # Aggregate summary (matches the format in bench_batched_gemm_a8w8.py).
    if rows:
        print()
        ck_us = [r["ck_us"] for r in rows]
        speedup = [r["speedup"] for r in rows]
        tflops = [r["tflops"] for r in rows]
        max_err = [r["max_err"] for r in rows if not math.isnan(r["max_err"])]
        print(f"# aggregate over {len(rows)} shapes:")
        print(f"#   CK latency:  min={min(ck_us):.1f}us  median={statistics.median(ck_us):.1f}us  max={max(ck_us):.1f}us")
        print(f"#   TFLOP/s:     min={min(tflops):.1f}  median={statistics.median(tflops):.1f}  max={max(tflops):.1f}")
        print(f"#   vs torch:    min={min(speedup):.2f}x  median={statistics.median(speedup):.2f}x  max={max(speedup):.2f}x")
        if max_err:
            print(f"#   max abs err: max={max(max_err):.4f}  (bf16 quant noise OK ~ <0.5)")
        if fail:
            print(f"#   FAILED: {fail} / {len(shapes)}")

    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
