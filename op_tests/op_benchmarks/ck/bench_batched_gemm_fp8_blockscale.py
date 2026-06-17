#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Benchmark for ``aiter.batched_gemm_fp8_blockscale`` (CK FP8 block-scale
batched GEMM, DeepSeek V4 ``wo_a`` path).

Usage:
    # Single shape
    python bench_batched_gemm_fp8_blockscale.py --shape 2 128 1024 4096

    # DSv4 preset sweep (44 per-rank shapes from V4-Flash / V4-Pro)
    python bench_batched_gemm_fp8_blockscale.py --preset dsv4

    # Save results to CSV
    python bench_batched_gemm_fp8_blockscale.py --preset dsv4 -o results.csv

"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import torch

import aiter
# Torch oracle lives next to the correctness test (not in the production
# dispatcher module).
from op_tests.test_batched_gemm_fp8_blockscale import (
    _torch_batched_gemm_fp8_blockscale,
)

DEVICE = "cuda"

# -----------------------------------------------------------------------------
# Shape presets.
#
# DSv4 per-rank wo_a shapes (from upstream config.json):
#   * N = o_lora_rank = 1024  (constant across Flash/Pro/TP)
#   * K = heads_per_group * head_dim = 8 * 512 = 4096
#   * B = G_per_rank = o_groups / TP
#       - Flash o_groups=8  -> B in {1 (TP=8), 2 (TP=4), 8 (TP=1)}
#       - Pro   o_groups=16 -> B in {2 (TP=8), 16 (TP=1)}
# -----------------------------------------------------------------------------

DSV4_BATCHES = [1, 2, 8, 16]
DSV4_MS = [1, 4, 8, 16, 32, 48, 64, 96, 128, 256, 512, 1024, 2048, 4096, 8192]
DSV4_N = 1024
DSV4_K = 4096

SMOKE_BATCHES = [1, 2]
SMOKE_MS = [128, 1024]

PRESETS = {
    "dsv4":  [(b, m, DSV4_N, DSV4_K) for b in DSV4_BATCHES for m in DSV4_MS],
    "smoke": [(b, m, DSV4_N, DSV4_K) for b in SMOKE_BATCHES for m in SMOKE_MS],
}


# -----------------------------------------------------------------------------
# Input helpers.
# -----------------------------------------------------------------------------

def _make_inputs(B, M, N, K, *, seed=0):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    A = (torch.randn(B, M, K, generator=g, device=DEVICE, dtype=torch.float32) * 0.5
         ).clamp(-8.0, 8.0).to(torch.float8_e4m3fn)
    W = (torch.randn(B, N, K, generator=g, device=DEVICE, dtype=torch.float32) * 0.5
         ).clamp(-8.0, 8.0).to(torch.float8_e4m3fn)
    K_g, N_g = K // 128, N // 128
    A_scale = torch.rand(B, M, K_g, generator=g, device=DEVICE, dtype=torch.float32) * 0.1 + 0.01
    W_scale = torch.rand(B, N_g, K_g, generator=g, device=DEVICE, dtype=torch.float32) * 0.1 + 0.01
    return A, W, A_scale, W_scale


# -----------------------------------------------------------------------------
# Per-shape bench.
# -----------------------------------------------------------------------------

def _cuda_event_us(fn, *, warmup, iters):
    """Pure-GPU latency samples (us) via CUDA events.

    Each sample brackets a single kernel launch with start/end events on the
    stream, so only GPU execution time is measured -- CPU launch / Python
    dispatch overhead between iterations is excluded (events sit on the GPU
    timeline). Returns the sorted list of per-iteration microseconds.
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1e3)  # ms -> us
    samples.sort()
    return samples


def _pcts(samples):
    return (
        samples[len(samples) // 2],                              # p50
        samples[max(0, int(len(samples) * 0.20) - 1)],          # p20
        samples[min(len(samples) - 1, int(len(samples) * 0.80))],  # p80
    )


def bench_one(B, M, N, K, *, warmup, rep, accuracy):
    A, W, A_scale, W_scale = _make_inputs(B, M, N, K, seed=B * 10007 + M * 101 + N + K)
    Y = torch.empty((B, M, N), dtype=torch.bfloat16, device=DEVICE)

    # Accuracy vs torch oracle (same fp32 scales).
    max_err = float("nan")
    rel_err = float("nan")
    if accuracy:
        ref = _torch_batched_gemm_fp8_blockscale(A, W, A_scale, W_scale)
        out = aiter.batched_gemm_fp8_blockscale(A, W, A_scale, W_scale)
        diff = (out.float() - ref.float()).abs()
        max_err = diff.max().item()
        rel_err = max_err / max(ref.float().abs().max().item(), 1e-6)

    # CK kernel timing (pure GPU time via CUDA events).
    ck = _cuda_event_us(
        lambda: aiter.batched_gemm_fp8_blockscale(A, W, A_scale, W_scale, out=Y),
        warmup=warmup, iters=rep)
    us, p20, p80 = _pcts(ck)

    # Torch dequant + bf16 bmm reference, timed the same way (fewer iters --
    # it's much slower). Median only.
    torch_iters = max(3, rep // 4)
    torch_samples = _cuda_event_us(
        lambda: _torch_batched_gemm_fp8_blockscale(A, W, A_scale, W_scale),
        warmup=2, iters=torch_iters)
    torch_us = torch_samples[len(torch_samples) // 2]
    speedup = torch_us / us if us > 0 else float("nan")

    flops = 2.0 * B * M * N * K
    tflops = flops / (us * 1e-6) / 1e12
    bytes_io = (A.numel() * A.element_size() + W.numel() * W.element_size() +
                A_scale.numel() * A_scale.element_size() +
                W_scale.numel() * W_scale.element_size() +
                Y.numel() * Y.element_size())
    bw_gb_s = bytes_io / (us * 1e-6) / 1e9

    return {
        "B": B, "M": M, "N": N, "K": K,
        "median_us": us, "p20_us": p20, "p80_us": p80,
        "torch_us": torch_us, "speedup": speedup,
        "tflops": tflops, "bw_gb_s": bw_gb_s,
        "max_err": max_err, "rel_err": rel_err,
    }


# -----------------------------------------------------------------------------
# Runners (single shape + sweep, mirrors bench_topk_topp_sampling.py).
# -----------------------------------------------------------------------------

def run_single_benchmark(args):
    B, M, N, K = args.shape
    print(f"\nBenchmarking batched_gemm_fp8_blockscale: B={B} M={M} N={N} K={K}\n")
    r = bench_one(B, M, N, K, warmup=args.warmup, rep=args.rep,
                  accuracy=not args.no_accuracy)
    print("Results:")
    print(f"  CK median:      {r['median_us']:.2f} us  (p20={r['p20_us']:.2f}, p80={r['p80_us']:.2f})")
    print(f"  Torch ref:      {r['torch_us']:.2f} us")
    print(f"  Speedup:        {r['speedup']:.2f}x")
    print(f"  Throughput:     {r['tflops']:.2f} TFLOP/s")
    print(f"  Bandwidth:      {r['bw_gb_s']:.2f} GB/s")
    if not args.no_accuracy:
        print(f"  Max abs err:    {r['max_err']:.4f}  (rel={r['rel_err']:.4f})")

    if args.o:
        _save_results_csv(args.o, [r])


def run_sweep_benchmark(args):
    if args.preset is not None:
        shapes = PRESETS[args.preset]
    else:
        batches = args.batches or DSV4_BATCHES
        ms = args.ms or DSV4_MS
        ns = args.ns or [DSV4_N]
        ks = args.ks or [DSV4_K]
        shapes = list(itertools.product(batches, ms, ns, ks))

    print(f"\nRunning sweep across {len(shapes)} shapes "
          f"(preset={args.preset or 'custom'})\n")

    header = (f"{'B':>4} {'M':>5} {'N':>5} {'K':>5} "
              f"{'ck_us':>9} {'p20':>8} {'p80':>8} {'torch_us':>9} {'speedup':>8} "
              f"{'TFLOPS':>7} {'GB/s':>7} {'max_err':>9} {'rel_err':>9}")
    print(header)
    print("-" * len(header))

    results = []
    for B, M, N, K in shapes:
        try:
            r = bench_one(B, M, N, K, warmup=args.warmup, rep=args.rep,
                          accuracy=not args.no_accuracy)
        except Exception as e:
            print(f"{B:>4} {M:>5} {N:>5} {K:>5}  FAIL: {e}")
            continue
        results.append(r)
        print(f"{B:>4} {M:>5} {N:>5} {K:>5} "
              f"{r['median_us']:>9.2f} {r['p20_us']:>8.2f} {r['p80_us']:>8.2f} "
              f"{r['torch_us']:>9.2f} {r['speedup']:>7.2f}x "
              f"{r['tflops']:>7.1f} {r['bw_gb_s']:>7.1f} "
              f"{r['max_err']:>9.4f} {r['rel_err']:>9.4f}")

    print(f"\nCompleted {len(results)} / {len(shapes)} shapes.")
    if args.o:
        _save_results_csv(args.o, results)


def _save_results_csv(filepath, results):
    path = Path(filepath)
    with open(path, "w") as f:
        f.write("B,M,N,K,median_us,p20_us,p80_us,torch_us,speedup,tflops,bw_gb_s,max_err,rel_err\n")
        for r in results:
            f.write(f"{r['B']},{r['M']},{r['N']},{r['K']},"
                    f"{r['median_us']:.3f},{r['p20_us']:.3f},{r['p80_us']:.3f},"
                    f"{r['torch_us']:.3f},{r['speedup']:.4f},"
                    f"{r['tflops']:.4f},{r['bw_gb_s']:.4f},"
                    f"{r['max_err']:.6f},{r['rel_err']:.6f}\n")
    print(f"Results saved to {path.resolve()}")


# -----------------------------------------------------------------------------
# CLI.
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        prog="Benchmark batched_gemm_fp8_blockscale",
        description="Benchmark CK FP8 block-scale batched GEMM (DeepSeek V4 wo_a).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--shape", type=int, nargs=4, metavar=("B", "M", "N", "K"),
                   help="Single-shape mode: B M N K (skips sweep).")
    p.add_argument("--preset", choices=list(PRESETS),
                   help=f"Shape preset for sweep (one of {list(PRESETS)}).")
    p.add_argument("--batches", type=int, nargs="+",
                   help=f"Custom B sweep (default: {DSV4_BATCHES}).")
    p.add_argument("--ms", type=int, nargs="+",
                   help=f"Custom M sweep (default: {DSV4_MS}).")
    p.add_argument("--ns", type=int, nargs="+",
                   help=f"Custom N sweep (default: [{DSV4_N}]).")
    p.add_argument("--ks", type=int, nargs="+",
                   help=f"Custom K sweep (default: [{DSV4_K}]).")
    p.add_argument("--no-accuracy", action="store_true",
                   help="Skip torch-oracle accuracy check (faster on big M).")
    # Defaults match the tune driver (warmup=5, iters=20) so bench latencies
    # are directly comparable to the tuned-CSV ``us`` column.
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--rep", type=int, default=20, help="timed iterations (per-call sync)")
    p.add_argument("-o", type=str, metavar="FILE",
                   help="Output CSV file path for results.")
    return p.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("requires CUDA/HIP device")

    if args.shape:
        run_single_benchmark(args)
    else:
        if args.preset is None and not any([args.batches, args.ms, args.ns, args.ks]):
            args.preset = "dsv4"  # default sweep
        run_sweep_benchmark(args)


if __name__ == "__main__":
    main()
