# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Lazy loader for the CK FP8 block-wise batched GEMM modules.

Two modules are exposed:

  * ``batched_gemm_fp8_blockwise(XQ, WQ, x_scale, w_scale, Out)`` -- the
    production dispatcher used by ``aiter.batched_gemm_fp8_blockwise``
    (backend="ck").  Picks the kernel from the lookup CSV / heuristic.

  * ``batched_gemm_fp8_blockwise_tune(XQ, WQ, x_scale, w_scale, Out,
    kernelId, splitK)`` -- direct kernel selection by integer id, for
    the tune driver in
    ``csrc/ck_batched_gemm_fp8_blockwise/batched_gemm_fp8_blockwise_tune.py``.
"""

from __future__ import annotations

from typing import Optional

import torch

from ..jit.core import compile_ops


def _gen_fake_out(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    Out: torch.Tensor,
) -> torch.Tensor:
    return Out


@compile_ops("module_batched_gemm_fp8_blockwise", fc_name="batched_gemm_fp8_blockwise",
             gen_fake=_gen_fake_out)
def _batched_gemm_fp8_blockwise(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    Out: torch.Tensor,
) -> torch.Tensor: ...


@compile_ops("module_batched_gemm_fp8_blockwise_tune", fc_name="batched_gemm_fp8_blockwise_tune",
             gen_fake=lambda XQ, WQ, x_scale, w_scale, Out, kernelId, splitK=0: Out)
def batched_gemm_fp8_blockwise_tune(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    Out: torch.Tensor,
    kernelId: int,
    splitK: int = 0,
) -> torch.Tensor: ...


def ck_batched_gemm_fp8_blockwise(
    A: torch.Tensor,
    W: torch.Tensor,
    A_scale: torch.Tensor,
    W_scale: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Public CK entry point. Same contract as ``aiter.batched_gemm_fp8_blockwise``."""
    if out is None:
        B, M, _ = A.shape
        N = W.shape[1]
        out = torch.empty((B, M, N), dtype=torch.bfloat16, device=A.device)
    _batched_gemm_fp8_blockwise(A, W, A_scale, W_scale, out)
    return out
