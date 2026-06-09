# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Microbench: ``aiter.batched_gemm_fp8_blockwise`` vs torch dequant + bf16 bmm
oracle, on DeepSeek V4 ``wo_a``-realistic shapes.

Shapes are derived from the V4-Flash / V4-Pro architecture comments in
sglang's ``deepseek_v4.py``:

  * V4-Flash decode: T=1 (or small batch), G=8, D=4096, R=8192   (per group, full)
  * V4-Pro decode:   T=1, G=16, D=4096, R=8192-16384

Tensor parallel TP=8 shards the ``wo_a`` output axis ``R`` per rank: ``N = R / TP``
(e.g. R=8192 → N=1024). Presets ``flash_decode`` / ``pro_decode`` / ``prefill`` use that.

For wo_a's einsum ``"bhr,hdr->bhd"``:
    B = G  (group count)
    M = T  (token count)
    K = D  (head_dim contracted)
    N = R  (output low-rank)

Run:
    python aiter/op_tests/bench_batched_gemm_fp8_blockwise.py
    AITER_BENCH_ITERS=100 python aiter/op_tests/bench_batched_gemm_fp8_blockwise.py
    python aiter/op_tests/bench_batched_gemm_fp8_blockwise.py --shapes flash_decode
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Callable, List

import torch

import aiter
from aiter.ops.batched_gemm_op_fp8_blockwise import _torch_batched_gemm_fp8_blockwise


def _bench(fn: Callable, *, iters: int, warmup: int = 10) -> float:
    """Return median microseconds per call."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter_ns() - t0) / 1e3)
    samples.sort()
    return samples[len(samples) // 2]


def _print_table(rows: List[List], headers: List[str]) -> None:
    widths = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]
    sep = "  ".join("-" * w for w in widths)
    print("  ".join(str(h).ljust(w) for h, w in zip(headers, widths)))
    print(sep)
    for r in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))


SHAPE_PRESETS = {
    # tiny: cheap CI sweep
    "tiny": [
        (4, 1, 256, 128), (4, 16, 256, 128), (4, 128, 256, 128),
        (8, 1, 512, 128), (8, 64, 512, 256), (16, 256, 512, 256),
    ],
    # flash_decode: V4-Flash per-group, TP=8 (N = 8192/8 = 1024 per rank)
    "flash_decode": [
        (8, 1, 1024, 4096),
        (8, 4, 1024, 4096),
        (8, 16, 1024, 4096),
        (8, 64, 1024, 4096),
    ],
    # pro_decode: V4-Pro per-group, TP=8
    "pro_decode": [
        (16, 1, 1024, 4096),
        (16, 4, 1024, 4096),
        (16, 16, 1024, 4096),
        (16, 64, 1024, 4096),
    ],
    # flash_prefill: V4-Flash prefill, TP=8
    "flash_prefill": [
        (8, 1024, 1024, 4096),
        (8, 4096, 1024, 4096),
        (8, 8192, 1024, 4096),
        (8, 16384, 1024, 4096),
    ],
    # pro_prefill: V4-Pro prefill, TP=8
    "pro_prefill": [
        (16, 1024, 1024, 4096),
        (16, 4096, 1024, 4096),
        (16, 8192, 1024, 4096),
        (16, 16384, 1024, 4096),
    ],
    # prefill: legacy alias = flash_prefill (Flash only)
    "prefill": [
        (8, 1024, 1024, 4096),
        (8, 4096, 1024, 4096),
        (8, 8192, 1024, 4096),
    ],
    # dsv4_all: comprehensive sweep across all DSv4 single-op shapes
    "dsv4_all": [
        # decode
        (8, 1, 1024, 4096),    (8, 16, 1024, 4096),   (8, 64, 1024, 4096),
        (16, 1, 1024, 4096),   (16, 16, 1024, 4096),  (16, 64, 1024, 4096),
        # prefill
        (8, 1024, 1024, 4096), (8, 4096, 1024, 4096), (8, 8192, 1024, 4096),
        (16, 1024, 1024, 4096),(16, 4096, 1024, 4096),(16, 8192, 1024, 4096),
    ],
}


def _make(B, M, N, K, *, device="cuda", seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    A = (torch.randn(B, M, K, generator=g, device=device, dtype=torch.float32) * 0.5).clamp(-8, 8).to(torch.float8_e4m3fn)
    W = (torch.randn(B, N, K, generator=g, device=device, dtype=torch.float32) * 0.5).clamp(-8, 8).to(torch.float8_e4m3fn)
    A_s = torch.rand(B, M, K // 128, generator=g, device=device, dtype=torch.float32) * 0.1 + 0.01
    W_s = torch.rand(B, N // 128, K // 128, generator=g, device=device, dtype=torch.float32) * 0.1 + 0.01
    return A, W, A_s, W_s


def _bench_shapes(name: str, shapes: List[tuple], iters: int) -> None:
    print(f"\n=== {name} (page-128 blockwise FP8 scales) ===")
    rows = []
    for B, M, N, K in shapes:
        A, W, A_s, W_s = _make(B, M, N, K, seed=B * M + N + K)
        # Pre-allocate output to avoid allocator noise.
        out = torch.empty((B, M, N), dtype=torch.bfloat16, device=A.device)
        torch_us = _bench(
            lambda: _torch_batched_gemm_fp8_blockwise(A, W, A_s, W_s),
            iters=iters,
        )
        auto_us = _bench(
            lambda: aiter.batched_gemm_fp8_blockwise(A, W, A_s, W_s, out=out, backend="auto"),
            iters=iters,
        )
        try:
            ck_us = _bench(
                lambda: aiter.batched_gemm_fp8_blockwise(A, W, A_s, W_s, out=out, backend="ck"),
                iters=iters,
            )
            ck_str = f"{ck_us:.1f}"
            ck_speedup = f"{torch_us / ck_us:.2f}x"
        except Exception:
            ck_str = "n/a"
            ck_speedup = "n/a"
        # FLOPs: 2 * B * M * N * K
        flops = 2 * B * M * N * K
        auto_tflops = flops / (auto_us * 1e-6) / 1e12
        rows.append([
            B, M, N, K,
            f"{torch_us:.1f}",
            f"{auto_us:.1f}",
            ck_str,
            f"{torch_us / auto_us:.2f}x",
            ck_speedup,
            f"{auto_tflops:.1f}",
        ])
    _print_table(
        rows,
        headers=["B", "M", "N", "K", "torch us", "auto us", "ck us", "auto/torch", "ck/torch", "auto TFLOPs"],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=int(os.environ.get("AITER_BENCH_ITERS", 30)))
    ap.add_argument("--shapes", nargs="+", default=["tiny"], choices=list(SHAPE_PRESETS.keys()))
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("requires CUDA/HIP device")
    torch.cuda.set_device(0)
    print(f"Device: {torch.cuda.get_device_name(0)}  |  iters={args.iters}")

    for name in args.shapes:
        _bench_shapes(name, SHAPE_PRESETS[name], args.iters)


if __name__ == "__main__":
    main()
