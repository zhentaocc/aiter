# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Microbenchmark: ``aiter.flydsl_moe_fused_gate_v4`` and
``aiter.flydsl_topk_transform_512`` vs sglang PR #23608's torch fallbacks.

Run:
    python aiter/op_tests/bench_flydsl_v4_routing.py
or:
    AITER_BENCH_ITERS=200 python aiter/op_tests/bench_flydsl_v4_routing.py

Two tables are printed: fused_gate sweep and topk_transform sweep.

This is the microbench complement to the e2e ``bench_serving`` runs
described in ``e2e_bench_flydsl_v4.sh`` -- the e2e captures fusion +
dispatch overhead, this captures per-kernel hot-path latency in
isolation.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Callable, List, Optional

import torch

import aiter
from aiter.ops.flydsl.v4_routing import (
    _torch_fused_gate_v4,
    _torch_topk_transform_512,
)


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


def bench_fused_gate(iters: int) -> None:
    print("\n=== Fused gate (V4 ungrouped, sigmoid, K=8, shared=1) ===")
    rows = []
    device = "cuda"
    torch.manual_seed(0)
    for N in [1, 8, 64, 256, 512, 2048, 8192]:
        for E in [257, 512]:
            x = torch.randn(N, E, device=device, dtype=torch.float32) * 0.5
            b = torch.randn(E, device=device, dtype=torch.float32) * 0.05
            kwargs = dict(
                topk=8,
                scoring_func="sigmoid",
                num_fused_shared_experts=1,
                renormalize=True,
                routed_scaling_factor=2.5,
            )
            torch_us = _bench(lambda: _torch_fused_gate_v4(
                x, b, **kwargs, apply_routed_scaling_factor_on_output=False
            ), iters=iters)
            triton_us = _bench(lambda: aiter.flydsl_moe_fused_gate_v4(
                x, b, backend="triton", **kwargs,
            ), iters=iters)
            try:
                flydsl_us = _bench(lambda: aiter.flydsl_moe_fused_gate_v4(
                    x, b, backend="flydsl", **kwargs,
                ), iters=iters)
                flydsl_str = f"{flydsl_us:.2f}"
                speedup_fly = f"{torch_us / flydsl_us:.2f}x"
            except Exception:  # NotImplementedError (FlyDSL still skeleton)
                flydsl_str = "n/a"
                speedup_fly = "n/a"
            rows.append([
                N, E,
                f"{torch_us:.2f}",
                f"{triton_us:.2f}",
                flydsl_str,
                f"{torch_us / triton_us:.2f}x",
                speedup_fly,
            ])
    _print_table(
        rows,
        headers=["N", "E", "torch us", "triton us", "flydsl us", "tri/torch", "fly/torch"],
    )


def bench_topk_transform(iters: int) -> None:
    print("\n=== topk_transform_512 (page=256) ===")
    rows = []
    device = "cuda"
    torch.manual_seed(1)
    page_size = 256
    for B in [1, 16, 64, 256]:
        for S in [1024, 4096, 16384]:
            scores = torch.randn(B, S, device=device, dtype=torch.float32)
            seq_lens = torch.full((B,), S, device=device, dtype=torch.int32)
            num_pages = (S + page_size - 1) // page_size
            page_tables = torch.randint(
                0, 1 << 20, (B, num_pages), device=device, dtype=torch.int32
            )
            out_pages = torch.empty((B, 512), dtype=torch.int32, device=device)
            torch_us = _bench(lambda: _torch_topk_transform_512(
                scores, seq_lens, page_tables, out_pages, page_size, None,
            ), iters=iters)
            triton_us = _bench(lambda: aiter.flydsl_topk_transform_512(
                scores, seq_lens, page_tables, out_pages, page_size, None, backend="triton",
            ), iters=iters)
            try:
                flydsl_us = _bench(lambda: aiter.flydsl_topk_transform_512(
                    scores, seq_lens, page_tables, out_pages, page_size, None, backend="flydsl",
                ), iters=iters)
                flydsl_str = f"{flydsl_us:.2f}"
                speedup_fly = f"{torch_us / flydsl_us:.2f}x"
            except Exception:
                flydsl_str = "n/a"
                speedup_fly = "n/a"
            rows.append([
                B, S,
                f"{torch_us:.2f}",
                f"{triton_us:.2f}",
                flydsl_str,
                f"{torch_us / triton_us:.2f}x",
                speedup_fly,
            ])
    _print_table(
        rows,
        headers=["B", "S", "torch us", "triton us", "flydsl us", "tri/torch", "fly/torch"],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=int(os.environ.get("AITER_BENCH_ITERS", 50)))
    ap.add_argument("--only", choices=["fused_gate", "topk_transform"], default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("requires CUDA/HIP device")

    torch.cuda.set_device(0)
    print(f"Device: {torch.cuda.get_device_name(0)}  |  iters={args.iters}")

    if args.only in (None, "fused_gate"):
        bench_fused_gate(args.iters)
    if args.only in (None, "topk_transform"):
        bench_topk_transform(args.iters)


if __name__ == "__main__":
    main()
