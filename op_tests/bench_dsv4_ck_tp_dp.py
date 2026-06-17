"""
Bench CK FP8 block-wise batched GEMM for DeepSeek V4 wo_a across
TP/DP/concurrency matrix (per-rank shapes).

Model-side facts (user-provided):
  - G_total = 8 (V4-Flash) / 16 (V4-Pro)
  - G_per_rank = G_total / TP  (TP shards groups)
  - N = R = 1024 (per-rank, constant)
  - K = D = 4096 (per-rank, constant)
  - M = max(1, ceil(concurrency / DP))

Per user request:
  Flash: (TP=4, DP=1), (TP=8, DP=1), (TP=1, DP=4), (TP=1, DP=8)
  Pro:   (TP=8, DP=1),                              (TP=1, DP=8)
  Concurrency C in {1, 4, 8, 16, 32, 64, 128, 256}
"""

from __future__ import annotations

import argparse
import math
import time

import torch

import aiter
from aiter.ops.batched_gemm_op_fp8_blockwise import (
    _torch_batched_gemm_fp8_blockwise,
    convert_scales_to_ue8m0,
)


N_DIM = 1024
K_DIM = 4096

G_TOTAL = {"flash": 8, "pro": 16}

CONFIGS = {
    "flash": [(4, 1), (8, 1), (1, 4), (1, 8)],
    "pro":   [(8, 1), (1, 8)],
}

CONCURRENCIES = [1, 4, 8, 16, 32, 64, 128, 256]


def _make_inputs(B, M, N, K, *, seed=0, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    A_f = (torch.randn(B, M, K, generator=g, device=device, dtype=torch.float32) * 0.5).clamp_(-8.0, 8.0)
    W_f = (torch.randn(B, N, K, generator=g, device=device, dtype=torch.float32) * 0.5).clamp_(-8.0, 8.0)
    A = A_f.to(torch.float8_e4m3fn)
    W = W_f.to(torch.float8_e4m3fn)
    K_g, N_g = K // 128, N // 128
    A_scale = (torch.rand(B, M, K_g, generator=g, device=device, dtype=torch.float32) * 0.1) + 0.01
    W_scale = (torch.rand(B, N_g, K_g, generator=g, device=device, dtype=torch.float32) * 0.1) + 0.01
    return A, W, A_scale, W_scale


def _bench_one(B, M, N, K, *, warmup=5, iters=20, check=True):
    """Return (us_per_call, max_abs_err_or_nan, chosen_backend)."""
    A, W, A_scale, W_scale = _make_inputs(B, M, N, K, seed=B * 10007 + M * 101 + N + K)
    W_scale_u8 = convert_scales_to_ue8m0(W_scale)

    err = float("nan")
    if check and M <= 64:
        ref = _torch_batched_gemm_fp8_blockwise(A, W, A_scale, W_scale)
        out = aiter.batched_gemm_fp8_blockwise(A, W, A_scale, W_scale_u8, backend="auto")
        err = (out.float() - ref.float()).abs().max().item()

    for _ in range(warmup):
        aiter.batched_gemm_fp8_blockwise(A, W, A_scale, W_scale_u8, backend="auto")
    torch.cuda.synchronize()

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    for _ in range(iters):
        aiter.batched_gemm_fp8_blockwise(A, W, A_scale, W_scale_u8, backend="auto")
    end_evt.record()
    torch.cuda.synchronize()
    elapsed_ms = start_evt.elapsed_time(end_evt)
    us_per_call = (elapsed_ms / iters) * 1000.0

    from aiter.ops.batched_gemm_op_fp8_blockwise import _CK_BROKEN
    backend = "ck" if (M >= 128 and M % 128 == 0 and N % 128 == 0 and not _CK_BROKEN) else "torch"
    return us_per_call, err, backend


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["flash", "pro", "both"], default="both")
    p.add_argument("--iters", type=int, default=20)
    args = p.parse_args()

    models = ["flash", "pro"] if args.model == "both" else [args.model]

    print(f"# Device: {torch.cuda.get_device_name(0)}")
    print(f"# Per-rank shape: B=G_per_rank, M=ceil(C/DP), N={N_DIM}, K={K_DIM}")
    print(f"# Iters={args.iters} (after warmup=5)\n")

    rows = []  # (model, TP, DP, C, B, M, us, err, backend)
    for model_name in models:
        for TP, DP in CONFIGS[model_name]:
            G_per_rank = G_TOTAL[model_name] // TP
            for C in CONCURRENCIES:
                M = max(1, math.ceil(C / DP))
                try:
                    us, err, backend = _bench_one(G_per_rank, M, N_DIM, K_DIM, iters=args.iters)
                    rows.append((model_name, TP, DP, C, G_per_rank, M, us, err, backend))
                    err_s = f"err={err:.4f}" if not math.isnan(err) else "err=skip"
                    print(f"{model_name:<6} TP={TP} DP={DP} C={C:<4} B={G_per_rank} M={M:<4} {us:7.1f}us  {err_s}  [{backend}]")
                except Exception as e:
                    rows.append((model_name, TP, DP, C, G_per_rank, M, float("nan"), float("nan"), f"FAIL"))
                    print(f"{model_name:<6} TP={TP} DP={DP} C={C:<4} B={G_per_rank} M={M:<4} FAIL: {e}")

    # Per-model markdown tables
    print()
    for model_name in models:
        print(f"## {model_name.title()} (G_total={G_TOTAL[model_name]}, N={N_DIM}, K={K_DIM})")
        print()
        cfgs = CONFIGS[model_name]
        hdr = "| Concurrency |" + "".join(f" TP={TP}/DP={DP} |" for TP, DP in cfgs)
        sep = "|-------------|" + "".join("------------|" for _ in cfgs)
        print(hdr)
        print(sep)
        for C in CONCURRENCIES:
            cells = []
            for TP, DP in cfgs:
                hit = [r for r in rows if r[0] == model_name and r[1] == TP and r[2] == DP and r[3] == C]
                if hit:
                    _, _, _, _, B, M, us, _err, backend = hit[0]
                    us_s = f"{us:.1f}" if not math.isnan(us) else "FAIL"
                    cells.append(f" B={B} M={M} {us_s}us [{backend}] ")
                else:
                    cells.append(" - ")
            print(f"| C={C:<10} |" + "|".join(cells) + "|")
        print()


if __name__ == "__main__":
    main()
