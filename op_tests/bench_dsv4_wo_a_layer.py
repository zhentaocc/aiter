# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
End-to-end bench of the DeepSeek V4 wo_a projection layer.

Compares the BF16 reference path (what vllm/sglang currently use on ROCm
as a placeholder) vs the new ``aiter.batched_gemm_fp8_blockwise`` path
this PR adds. Both paths take the same hidden_states-style input
``[T, G, D] bf16`` and the same FP8-quantized weight ``[G, R, D] fp8 +
[G, R/128, D/128] fp32 scale``, producing ``[T, G, R] bf16``.

Three implementations:

  * ``bf16_ref``   -- the current ROCm fallback: dequant W to bf16, do
                      bf16 bmm.  This is what vllm uses today.
  * ``aiter_auto`` -- this PR's new path: per-row FP8 quant on A, then
                      ``batched_gemm_fp8_blockwise(backend="auto")``.
  * ``aiter_ck``   -- same but force ``backend="ck"`` (prefill regime).
  * ``aiter_fly``  -- same but force ``backend="flydsl"`` (decode regime).

Run:
  python op_tests/bench_dsv4_wo_a_layer.py --shapes flash_decode pro_decode
  python op_tests/bench_dsv4_wo_a_layer.py --shapes flash_prefill pro_prefill
  python op_tests/bench_dsv4_wo_a_layer.py --shapes dsv4_all

(CK requires libstdc++ >= 3.4.31; run inside a docker with newer libstdc++
if not available on bare-metal host.)
"""

from __future__ import annotations

import argparse
import gc
import os
import time
from typing import List, Tuple

import torch

import aiter


# ----------------------------------------------------------------------------
# DSv4 wo_a shape presets (TP=8 sharded; N = o_lora_rank = 1024 per rank,
# K = head_dim = 4096; G = n_local_groups).
# Numbers represent (G, T, R, D) i.e. (n_groups, tokens, o_lora_rank, head_dim).
# ----------------------------------------------------------------------------
SHAPE_PRESETS = {
    "flash_decode": [  # V4-Flash, G=8
        (8, 1, 1024, 4096),
        (8, 4, 1024, 4096),
        (8, 16, 1024, 4096),
        (8, 64, 1024, 4096),
    ],
    "pro_decode": [  # V4-Pro, G=16
        (16, 1, 1024, 4096),
        (16, 4, 1024, 4096),
        (16, 16, 1024, 4096),
        (16, 64, 1024, 4096),
    ],
    "flash_prefill": [
        (8, 1024, 1024, 4096),
        (8, 4096, 1024, 4096),
        (8, 8192, 1024, 4096),
    ],
    "pro_prefill": [
        (16, 1024, 1024, 4096),
        (16, 4096, 1024, 4096),
        (16, 8192, 1024, 4096),
    ],
    "dsv4_all": [
        (8, 1, 1024, 4096),    (8, 16, 1024, 4096),   (8, 64, 1024, 4096),
        (16, 1, 1024, 4096),   (16, 16, 1024, 4096),  (16, 64, 1024, 4096),
        (8, 1024, 1024, 4096), (8, 4096, 1024, 4096),
        (16, 1024, 1024, 4096),(16, 4096, 1024, 4096),
    ],
}


# ----------------------------------------------------------------------------
# Build a layer-realistic input + weight.
# ----------------------------------------------------------------------------


def _make_layer(G: int, T: int, R: int, D: int, *, device="cuda", seed=0):
    """Return (a_bf16 [T,G,D], w_fp8 [G,R,D], w_scale_inv [G,R/128,D/128]).

    Matches the layout vllm holds after _setup_fp8_wo_a_scales.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    a_bf16 = torch.randn(T, G, D, generator=g, device=device, dtype=torch.bfloat16) * 0.5
    # Weight: store fp8 with realistic per-block scales (so the bf16 ref path
    # actually has to dequant something meaningful).
    w_raw = torch.randn(G, R, D, generator=g, device=device, dtype=torch.float32) * 0.5
    w_raw = w_raw.clamp_(-8.0, 8.0)
    w_fp8 = w_raw.to(torch.float8_e4m3fn)
    # Per-(128 R, 128 D) block scale, computed as max-abs of the block.
    w_dq_blocks = w_raw.view(G, R // 128, 128, D // 128, 128).abs()
    w_scale_inv = (w_dq_blocks.amax(dim=(2, 4)) / 448.0).clamp_min(1e-6)  # [G, R/128, D/128]
    return a_bf16, w_fp8, w_scale_inv


# ----------------------------------------------------------------------------
# Per-row, per-128-block FP8 quant of A (the host-side step in vllm's
# fused_inv_rope_fp8_quant path; on ROCm we currently do it unfused).
# ----------------------------------------------------------------------------


def _quant_a_fp8(a_bf16: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """a_bf16 [T,G,D] -> a_fp8 [T,G,D], a_scale [T,G,D/128] (fp32)."""
    T, G, D = a_bf16.shape
    a_f32 = a_bf16.float().view(T, G, D // 128, 128)
    # max-abs per 128-block, divided by 448 (fp8 e4m3 max).
    a_scale = (a_f32.abs().amax(dim=-1) / 448.0).clamp_min(1e-6)  # [T,G,D/128]
    a_scaled = a_f32 / a_scale.unsqueeze(-1)
    a_fp8 = a_scaled.clamp_(-448.0, 448.0).view(T, G, D).to(torch.float8_e4m3fn)
    return a_fp8, a_scale


# ----------------------------------------------------------------------------
# Reference BF16 path (what vllm uses on ROCm today as a placeholder).
# ----------------------------------------------------------------------------


def wo_a_bf16_ref(a_bf16: torch.Tensor, w_fp8: torch.Tensor,
                   w_scale_inv: torch.Tensor) -> torch.Tensor:
    """[T,G,D] bf16  @  [G,R,D] fp8  ->  [T,G,R] bf16  (dequant W + bf16 bmm)."""
    G, R, D = w_fp8.shape
    T = a_bf16.shape[0]
    # Dequant W: [G, R, D] = ([G, R/128, 128, D/128, 128] * [G, R/128, 1, D/128, 1])
    w_dq = (w_fp8.float().view(G, R // 128, 128, D // 128, 128)
            * w_scale_inv.view(G, R // 128, 1, D // 128, 1)
            ).view(G, R, D).to(torch.bfloat16)
    # BMM: [G,T,D] @ [G,D,R] = [G,T,R] (then transpose to [T,G,R])
    a_gtd = a_bf16.transpose(0, 1).contiguous()        # [G,T,D]
    out_gtn = torch.bmm(a_gtd, w_dq.transpose(1, 2))    # [G,T,R]
    return out_gtn.transpose(0, 1).contiguous()         # [T,G,R]


# ----------------------------------------------------------------------------
# New aiter path: per-row FP8 quant on A, then blockwise FP8 batched GEMM.
# This is what vllm_dsv4_wo_a_aiter_patch.py installs as forward_o_proj.
# ----------------------------------------------------------------------------


def wo_a_aiter(a_bf16: torch.Tensor, w_fp8: torch.Tensor,
                w_scale_inv: torch.Tensor, *, backend: str = "auto") -> torch.Tensor:
    """Quant + aiter blockwise FP8 batched GEMM. Same I/O contract as bf16_ref."""
    G, R, D = w_fp8.shape
    T = a_bf16.shape[0]
    a_fp8, a_scale = _quant_a_fp8(a_bf16)
    # [T,G,D] -> [G,T,D] for kernel canonical layout
    a_gtd = a_fp8.transpose(0, 1).contiguous()             # [G,T,D]
    a_scale_gtd = a_scale.transpose(0, 1).contiguous()     # [G,T,D/128]
    out_gtn = aiter.batched_gemm_fp8_blockwise(
        a_gtd, w_fp8, a_scale_gtd, w_scale_inv, backend=backend,
    )                                                       # [G,T,R] bf16
    return out_gtn.transpose(0, 1).contiguous()             # [T,G,R]


# ----------------------------------------------------------------------------
# Timing harness.
# ----------------------------------------------------------------------------


def gpu_time_median(fn, *, iters: int = 30, warmup: int = 10) -> float:
    """Median GPU-event timing in microseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for s, e in zip(starts, ends):
        s.record(); fn(); e.record()
    torch.cuda.synchronize()
    samples = sorted([s.elapsed_time(e) * 1000 for s, e in zip(starts, ends)])
    return samples[iters // 2]


# ----------------------------------------------------------------------------
# Bench loop.
# ----------------------------------------------------------------------------


def _bench_one(G: int, T: int, R: int, D: int, iters: int, atol: float):
    a_bf16, w_fp8, w_scale_inv = _make_layer(G, T, R, D, seed=G * T + R + D)

    # Correctness check first (BF16 ref vs aiter auto; only on small shapes)
    err = None
    if T <= 64:
        ref = wo_a_bf16_ref(a_bf16, w_fp8, w_scale_inv)
        try:
            new = wo_a_aiter(a_bf16, w_fp8, w_scale_inv, backend="auto")
            err = (ref - new).abs().max().item()
        except Exception as e:
            err = float("nan")

    # Time each impl
    times = {}
    for label, fn in [
        ("bf16_ref",  lambda: wo_a_bf16_ref(a_bf16, w_fp8, w_scale_inv)),
        ("aiter_auto", lambda: wo_a_aiter(a_bf16, w_fp8, w_scale_inv, backend="auto")),
        ("aiter_fly", lambda: wo_a_aiter(a_bf16, w_fp8, w_scale_inv, backend="flydsl")),
        ("aiter_ck",  lambda: wo_a_aiter(a_bf16, w_fp8, w_scale_inv, backend="ck")),
    ]:
        try:
            times[label] = gpu_time_median(fn, iters=iters)
        except Exception as e:
            times[label] = float("inf")

    del a_bf16, w_fp8, w_scale_inv
    gc.collect(); torch.cuda.empty_cache()

    return times, err


def _fmt_us(us: float) -> str:
    return "    FAIL" if us == float("inf") else f"{us:>7.1f}u"


def _bench_preset(name: str, shapes: List[Tuple[int, int, int, int]], iters: int):
    print(f"\n=== {name}  (wo_a layer: [T,G,D] bf16 -> [T,G,R] bf16) ===")
    print(f"{'G':>3} {'T':>5} {'R':>4} {'K=D':>5}  "
          f"{'bf16_ref':>10} {'auto':>9} {'fly':>9} {'ck':>9}  "
          f"{'auto/bf16':>10}  {'max_err':>8}")
    print("-" * 95)
    for G, T, R, D in shapes:
        times, err = _bench_one(G, T, R, D, iters, atol=2e-2)
        bf16, auto = times["bf16_ref"], times["aiter_auto"]
        ratio_str = "    -"
        if bf16 not in (float("inf"), 0) and auto not in (float("inf"), 0):
            ratio = auto / bf16
            star = " ★" if ratio < 0.5 else ("★ " if ratio < 0.9 else "  ")
            ratio_str = f"{star}{ratio:>5.2f}x"
        err_str = "    -" if err is None else (f"{err:.4f}" if not (err != err) else "  fail")
        print(f"{G:>3} {T:>5} {R:>4} {D:>5}  "
              f"{_fmt_us(bf16)} {_fmt_us(auto)} {_fmt_us(times['aiter_fly'])} {_fmt_us(times['aiter_ck'])}  "
              f"{ratio_str:>10}  {err_str:>8}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=int(os.environ.get("AITER_BENCH_ITERS", 30)))
    ap.add_argument("--shapes", nargs="+", default=["flash_decode"],
                    choices=list(SHAPE_PRESETS.keys()))
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA/HIP device required.")
    for name in args.shapes:
        _bench_preset(name, SHAPE_PRESETS[name], args.iters)
    print("\nLegend:")
    print("  bf16_ref   = current ROCm path (dequant W to bf16 + bf16 bmm)")
    print("  aiter_auto = per-row FP8 quant A + batched_gemm_fp8_blockwise(backend='auto')")
    print("  aiter_fly  = same but forced backend='flydsl' (decode-optimised)")
    print("  aiter_ck   = same but forced backend='ck'    (prefill-optimised)")
    print("  auto/bf16  = aiter_auto / bf16_ref (<1.0 means aiter wins)")
    print("  max_err    = max abs diff between bf16_ref and aiter_auto outputs (skipped for T>64)")


if __name__ == "__main__":
    main()
