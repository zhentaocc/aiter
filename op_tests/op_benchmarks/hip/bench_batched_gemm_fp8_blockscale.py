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

    # Force a backend / scale dtype
    python bench_batched_gemm_fp8_blockscale.py --preset smoke --backend ck --scale-dtype u8
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import torch
import triton

import aiter
from aiter.ops.batched_gemm_op_fp8_blockscale import convert_scales_to_ue8m0
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
DSV4_MS = [16, 32, 48, 64, 96, 128, 256, 512, 1024, 2048, 4096]
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

def _make_inputs(B, M, N, K, *, seed=0, scale_dtype=torch.uint8):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    A = (torch.randn(B, M, K, generator=g, device=DEVICE, dtype=torch.float32) * 0.5
         ).clamp(-8.0, 8.0).to(torch.float8_e4m3fn)
    W = (torch.randn(B, N, K, generator=g, device=DEVICE, dtype=torch.float32) * 0.5
         ).clamp(-8.0, 8.0).to(torch.float8_e4m3fn)
    K_g, N_g = K // 128, N // 128
    A_scale = torch.rand(B, M, K_g, generator=g, device=DEVICE, dtype=torch.float32) * 0.1 + 0.01
    W_scale = torch.rand(B, N_g, K_g, generator=g, device=DEVICE, dtype=torch.float32) * 0.1 + 0.01
    if scale_dtype == torch.uint8:
        A_scale = convert_scales_to_ue8m0(A_scale)
        W_scale = convert_scales_to_ue8m0(W_scale)
    return A, W, A_scale, W_scale


def _u8_to_fp32(s):
    return torch.exp2((s.float() - 127.0)) if s.dtype == torch.uint8 else s


# -----------------------------------------------------------------------------
# Per-shape bench.
# -----------------------------------------------------------------------------

def bench_one(B, M, N, K, *, backend, scale_dtype, warmup, rep, accuracy):
    A, W, A_scale, W_scale = _make_inputs(B, M, N, K, seed=B * 10007 + M * 101 + N + K,
                                          scale_dtype=scale_dtype)
    Y = torch.empty((B, M, N), dtype=torch.bfloat16, device=DEVICE)

    # Accuracy vs torch oracle (uses fp32 scales).
    max_err = float("nan")
    rel_err = float("nan")
    if accuracy:
        A_s_ref, W_s_ref = _u8_to_fp32(A_scale), _u8_to_fp32(W_scale)
        ref = _torch_batched_gemm_fp8_blockscale(A, W, A_s_ref, W_s_ref)
        out = aiter.batched_gemm_fp8_blockscale(A, W, A_scale, W_scale, backend=backend)
        diff = (out.float() - ref.float()).abs()
        max_err = diff.max().item()
        rel_err = max_err / max(ref.float().abs().max().item(), 1e-6)

    # CK kernel timing.
    fn = lambda: aiter.batched_gemm_fp8_blockscale(
        A, W, A_scale, W_scale, out=Y, backend=backend)
    ms, p20, p80 = triton.testing.do_bench(
        fn, warmup=warmup, rep=rep, quantiles=[0.5, 0.2, 0.8])

    flops = 2.0 * B * M * N * K
    tflops = flops / ms * 1e-9
    bytes_io = (A.numel() * A.element_size() + W.numel() * W.element_size() +
                A_scale.numel() * A_scale.element_size() +
                W_scale.numel() * W_scale.element_size() +
                Y.numel() * Y.element_size())
    bw_gb_s = bytes_io / (ms * 1e-3) * 1e-9

    return {
        "B": B, "M": M, "N": N, "K": K,
        "median_ms": ms, "p20_ms": p20, "p80_ms": p80,
        "tflops": tflops, "bw_gb_s": bw_gb_s,
        "max_err": max_err, "rel_err": rel_err,
    }


# -----------------------------------------------------------------------------
# Runners (single shape + sweep, mirrors bench_topk_topp_sampling.py).
# -----------------------------------------------------------------------------

def run_single_benchmark(args):
    B, M, N, K = args.shape
    print(f"\nBenchmarking batched_gemm_fp8_blockscale: B={B} M={M} N={N} K={K}\n")
    r = bench_one(B, M, N, K, backend=args.backend,
                  scale_dtype=torch.uint8 if args.scale_dtype == "u8" else torch.float32,
                  warmup=args.warmup, rep=args.rep,
                  accuracy=not args.no_accuracy)
    print("Results:")
    print(f"  Median latency: {r['median_ms']:.4f} ms")
    print(f"  P20 latency:    {r['p20_ms']:.4f} ms")
    print(f"  P80 latency:    {r['p80_ms']:.4f} ms")
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
          f"(preset={args.preset or 'custom'}, backend={args.backend}, "
          f"scale={args.scale_dtype})\n")

    header = (f"{'B':>4} {'M':>5} {'N':>5} {'K':>5} "
              f"{'median_ms':>10} {'p20_ms':>9} {'p80_ms':>9} "
              f"{'TFLOPS':>7} {'GB/s':>7} {'max_err':>9} {'rel_err':>9}")
    print(header)
    print("-" * len(header))

    results = []
    for B, M, N, K in shapes:
        try:
            r = bench_one(B, M, N, K, backend=args.backend,
                          scale_dtype=torch.uint8 if args.scale_dtype == "u8" else torch.float32,
                          warmup=args.warmup, rep=args.rep,
                          accuracy=not args.no_accuracy)
        except Exception as e:
            print(f"{B:>4} {M:>5} {N:>5} {K:>5}  FAIL: {e}")
            continue
        results.append(r)
        print(f"{B:>4} {M:>5} {N:>5} {K:>5} "
              f"{r['median_ms']:>10.4f} {r['p20_ms']:>9.4f} {r['p80_ms']:>9.4f} "
              f"{r['tflops']:>7.1f} {r['bw_gb_s']:>7.1f} "
              f"{r['max_err']:>9.4f} {r['rel_err']:>9.4f}")

    print(f"\nCompleted {len(results)} / {len(shapes)} shapes.")
    if args.o:
        _save_results_csv(args.o, results)


def _save_results_csv(filepath, results):
    path = Path(filepath)
    with open(path, "w") as f:
        f.write("B,M,N,K,median_ms,p20_ms,p80_ms,tflops,bw_gb_s,max_err,rel_err\n")
        for r in results:
            f.write(f"{r['B']},{r['M']},{r['N']},{r['K']},"
                    f"{r['median_ms']:.6f},{r['p20_ms']:.6f},{r['p80_ms']:.6f},"
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
    p.add_argument("--backend", choices=["auto", "ck", "torch"], default="auto")
    p.add_argument("--scale-dtype", choices=["u8", "fp32"], default="u8",
                   help="Scale dtype passed to the kernel.")
    p.add_argument("--no-accuracy", action="store_true",
                   help="Skip torch-oracle accuracy check (faster on big M).")
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--rep", type=int, default=100)
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
