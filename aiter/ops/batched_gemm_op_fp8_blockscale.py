# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Public entry point for FP8 block-wise batched GEMM.

This is the kernel that DeepSeek V4's ``wo_a`` projection needs (and that
sglang PR #23608 currently sources from ``deep_gemm.fp8_einsum``).  Mirrors
the contract of:

    deep_gemm.fp8_einsum("bmk,bnk->bmn", (A, A_scale), (W, W_scale), out,
                         recipe=(1, 1, 128))

Backend selection (this branch ships the CK path only; the decode-optimised
flydsl kernel is on the ``wo_a_fp8_blockwise`` branch):

  * ``"auto"``   -> CK when M >= 128 && M % 128 == 0 && N % 128 == 0,
                    else torch reference. Empirically validated on DSv4
                    wo_a: CK wins from M=4096+ (1.4-1.7x over flydsl).
  * ``"ck"``     -> CK FP8 block-wise batched GEMM (prefill-optimized).
  * ``"torch"``  -> reference dequant + ``torch.bmm`` in bf16, the
                    correctness oracle.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch

from aiter import logger as _aiter_logger

logger = _aiter_logger
_log = logger.getChild("batched_gemm_fp8_blockscale")


Backend = Literal["auto", "ck", "torch"]

# Once the CK loader fails, remember it so we don't re-pay the import cost
# (and the warning spam) on every call.
_CK_BROKEN: bool = False

# CK shape constraints. Below this M (or when M % 128 != 0 / N % 128 != 0)
# we fall back to the torch reference. The flydsl decode-optimised path
# lives on a separate branch (wo_a_fp8_blockwise) and is not built here.
_AUTO_CK_THRESHOLD_M = 128


# ----------------------------------------------------------------------------
# Runtime fallback (dequant + bf16 bmm). Used when CK loading fails or when
# the user explicitly passes ``backend="torch"``. The test-only correctness
# oracle of the same semantics lives in ``op_tests/test_batched_gemm_fp8_blockscale.py``.
# ----------------------------------------------------------------------------


def _torch_fallback(
    A: torch.Tensor,
    W: torch.Tensor,
    A_scale: torch.Tensor,
    W_scale: torch.Tensor,
) -> torch.Tensor:
    B, M, K = A.shape
    Bw, N, Kw = W.shape
    assert B == Bw and K == Kw
    K_g = K // 128
    N_g = N // 128
    assert A_scale.shape == (B, M, K_g)
    assert W_scale.shape == (B, N_g, K_g)

    # Dequant A: per-row, per-128k-block.
    a_dq = A.to(torch.float32).view(B, M, K_g, 128) * A_scale.unsqueeze(-1)
    a_dq = a_dq.view(B, M, K).to(torch.bfloat16)

    # Dequant W: per (128n, 128k) block.
    w_dq = W.to(torch.float32).view(B, N_g, 128, K_g, 128) * W_scale.view(B, N_g, 1, K_g, 1)
    w_dq = w_dq.view(B, N, K).to(torch.bfloat16)

    return torch.bmm(a_dq, w_dq.transpose(1, 2)).to(torch.bfloat16)


# ----------------------------------------------------------------------------
# Public entry point.
# ----------------------------------------------------------------------------


def batched_gemm_fp8_blockscale(
    A: torch.Tensor,
    W: torch.Tensor,
    A_scale: torch.Tensor,
    W_scale: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    backend: Backend = "auto",
) -> torch.Tensor:
    """
    FP8 block-wise batched GEMM:  ``Out[b, m, n] = sum_k A[b, m, k] * W[b, n, k]``.

    Args:
        A:        ``[B, M, K]`` ``fp8_e4m3fn``.
        W:        ``[B, N, K]`` ``fp8_e4m3fn``.
        A_scale:  ``[B, M, K // 128]`` ``fp32`` or ``uint8`` (UE8M0) --
                  per-row, per-128k-block.
        W_scale:  ``[B, N // 128, K // 128]`` ``fp32`` or ``uint8`` (UE8M0) --
                  per 128x128 block. Must share dtype with ``A_scale``.
        out:      Optional pre-allocated ``[B, M, N]`` ``bfloat16`` output.
        backend:  ``"auto" | "ck" | "torch"``.  Default ``"auto"``.

    Returns:
        ``Out[B, M, N] bfloat16``.

    Notes:
        * ``K`` and ``N`` must be multiples of 128.
        * Mirrors ``deep_gemm.fp8_einsum("bmk,bnk->bmn", (A, A_scale),
          (W, W_scale), out, recipe=(1, 1, 128))`` exactly.
    """
    global _CK_BROKEN
    chosen = backend
    if chosen == "auto":
        # CK is the only optimized backend on this branch; fall to torch for
        # shapes CK cannot handle (M < 128 or N/M not multiple of 128).
        _, M, _ = A.shape
        N = W.shape[1]
        if (M >= _AUTO_CK_THRESHOLD_M and M % 128 == 0 and N % 128 == 0
                and not _CK_BROKEN):
            chosen = "ck"
        else:
            chosen = "torch"

    if chosen == "ck":
        if not _CK_BROKEN:
            try:
                from aiter.ops._ck_batched_gemm_fp8_blockscale_loader import ck_batched_gemm_fp8_blockscale
                return ck_batched_gemm_fp8_blockscale(A, W, A_scale, W_scale, out=out)
            except (ImportError, NotImplementedError, RuntimeError) as e:
                _log.warning(
                    "CK batched_gemm_fp8_blockscale unavailable (%s); "
                    "falling back to torch reference (silent for subsequent calls).", e,
                )
                _CK_BROKEN = True

    out_ref = _torch_fallback(A, W, A_scale, W_scale)
    if out is not None:
        out.copy_(out_ref)
        return out
    return out_ref
