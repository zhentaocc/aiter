# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import aiter
from aiter import dtypes
from aiter.test_common import checkAllclose, perftest
import argparse

BLOCK = 128  # block-scale recipe (1, 1, 128): per-row A scale, per-128x128 W block


@perftest(num_iters=5)
def run_torch(x, weight, x_scale, w_scale, dtype=dtypes.bf16):
    # Dequant + bf16 bmm reference (verbatim DeepGEMM fp8_einsum recipe=(1,1,128)).
    B, M, K = x.shape
    _, N, _ = weight.shape
    Kg, Ng = K // BLOCK, N // BLOCK
    a = x.to(dtypes.fp32).view(B, M, Kg, BLOCK) * x_scale.unsqueeze(-1)
    a = a.view(B, M, K).to(dtypes.bf16)
    w = weight.to(dtypes.fp32).view(B, Ng, BLOCK, Kg, BLOCK) * w_scale.view(
        B, Ng, 1, Kg, 1
    )
    w = w.view(B, N, K).to(dtypes.bf16)
    return torch.bmm(a, w.transpose(1, 2)).to(dtype)


@perftest()
def run_gemm_ck(x, weight, x_scale, w_scale, dtype=dtypes.bf16):
    return aiter.batched_gemm_fp8_blockscale(x, weight, x_scale, w_scale)


def test_gemm(dtype, b, m, n, k):
    dim = (b, m, n, k)
    x = (
        (torch.randn(b, m, k, dtype=dtypes.fp32, device="cuda") * 0.5)
        .clamp(-8, 8)
        .to(dtypes.fp8)
    )
    weight = (
        (torch.randn(b, n, k, dtype=dtypes.fp32, device="cuda") * 0.5)
        .clamp(-8, 8)
        .to(dtypes.fp8)
    )
    x_scale = (
        torch.rand([b, m, k // BLOCK], dtype=dtypes.fp32, device="cuda") * 0.1 + 0.01
    )
    w_scale = (
        torch.rand([b, n // BLOCK, k // BLOCK], dtype=dtypes.fp32, device="cuda") * 0.1
        + 0.01
    )

    a, avg_a = run_torch(x, weight, x_scale, w_scale, dtype)
    c, avg_c = run_gemm_ck(x, weight, x_scale, w_scale, dtype)
    msg = f"[perf] dim: {str(dim):<24} dtype: {dtype}, torch avg: {avg_a:<8.2f} us, ck avg: {avg_c:<8.2f} us, uplift: {avg_a/avg_c-1:<5.1%}"
    checkAllclose(
        a, c, msg="a,c: " + msg, rtol=2e-2, atol=4e-2, catastrophic_check=True
    )


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=dtypes.str2Dtype,
    choices=[dtypes.d_dtypes["bf16"]],
    nargs="*",
    default="bf16,",
    metavar="{bf16}",
    help="""Data type.
    e.g.: -d bf16""",
)
parser.add_argument(
    "-b",
    "--batch",
    type=int,
    nargs="*",
    default=[1, 2, 8, 16],
    help="""Batch size (= DeepSeek V4 wo_a G_per_rank: Flash o_groups=8 / Pro=16, /TP).
    e.g.: -b 1 2 8 16""",
)
parser.add_argument(
    "-s",
    "--mnk",
    type=dtypes.str2tuple,
    nargs="*",
    default=[
        # DeepSeek V4 wo_a per-rank shapes: N = o_lora_rank = 1024, K = 8*512 = 4096.
        (1, 1024, 4096),
        (4, 1024, 4096),
        (8, 1024, 4096),
        (16, 1024, 4096),
        (32, 1024, 4096),
        (64, 1024, 4096),
        (128, 1024, 4096),
        (256, 1024, 4096),
        (512, 1024, 4096),
        (1024, 1024, 4096),
        (2048, 1024, 4096),
        (4096, 1024, 4096),
    ],
    help="""Shape of mnk.
    e.g.:   -s 256,1024,4096
            --mnk 256,1024,4096""",
)

args = parser.parse_args()


for dtype in args.dtype:
    for b in args.batch:
        for m, n, k in args.mnk:
            test_gemm(dtype, b, m, n, k)
