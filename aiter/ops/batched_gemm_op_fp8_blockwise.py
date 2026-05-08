# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Public entry point for FP8 block-wise batched GEMM.

This is the kernel that DeepSeek V4's ``wo_a`` projection needs (and that
sglang PR #23608 currently sources from ``deep_gemm.fp8_einsum``).  Mirrors
the contract of:

    deep_gemm.fp8_einsum("bmk,bnk->bmn", (A, A_scale), (W, W_scale), out,
                         recipe=(1, 1, 128))

Backend selection follows the same pattern as ``aiter.flydsl_*`` ops:

  * ``"auto"``  -> CK if a tuned kernel for the (B,M,N,K) shape is available,
                   else Triton.  Today CK is a stub, so this resolves to Triton.
  * ``"ck"``    -> CK kernel (currently raises NotImplementedError; see
                   ``csrc/ck_batched_gemm_fp8_blockwise/README.md``).
  * ``"triton"``-> Triton kernel (always available on ROCm).
  * ``"torch"`` -> reference dequant + ``torch.bmm`` in bf16, the correctness
                   oracle.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch

from aiter import logger as _aiter_logger

logger = _aiter_logger
_log = logger.getChild("batched_gemm_fp8_blockwise")


Backend = Literal["auto", "ck", "triton", "torch"]

# Once the CK loader fails to import, remember it so we don't re-pay the import
# cost (and the warning spam) on every call.
_CK_BROKEN: bool = False


# ----------------------------------------------------------------------------
# Reference implementation (correctness oracle).  Verbatim semantics of
# DeepGEMM's fp8_einsum with recipe=(1, 1, 128).
# ----------------------------------------------------------------------------


def _torch_batched_gemm_fp8_blockwise(
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
    a_f32 = A.to(torch.float32)
    a_dq = a_f32.view(B, M, K_g, 128) * A_scale.unsqueeze(-1)  # broadcast scale across the 128-block
    a_dq = a_dq.view(B, M, K).to(torch.bfloat16)

    # Dequant W: per (128n, 128k) block.
    w_f32 = W.to(torch.float32).view(B, N_g, 128, K_g, 128)
    w_dq = w_f32 * W_scale.view(B, N_g, 1, K_g, 1)
    w_dq = w_dq.view(B, N, K).to(torch.bfloat16)

    return torch.bmm(a_dq, w_dq.transpose(1, 2)).to(torch.bfloat16)


# ----------------------------------------------------------------------------
# Public entry point.
# ----------------------------------------------------------------------------


def batched_gemm_fp8_blockwise(
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
        A_scale:  ``[B, M, K // 128]`` ``fp32``  -- per-row, per-128k-block.
        W_scale:  ``[B, N // 128, K // 128]`` ``fp32`` -- per 128x128 block.
        out:      Optional pre-allocated ``[B, M, N]`` ``bfloat16`` output.
        backend:  ``"auto" | "ck" | "triton" | "torch"``.  Default ``"auto"``.

    Returns:
        ``Out[B, M, N] bfloat16``.

    Notes:
        * ``K`` and ``N`` must be multiples of 128.
        * Mirrors ``deep_gemm.fp8_einsum("bmk,bnk->bmn", (A, A_scale),
          (W, W_scale), out, recipe=(1, 1, 128))`` exactly.
        * For the equivalent operation under DeepSeek V4 wo_a's
          ``"bhr,hdr->bhd"`` einsum, transpose the activation to
          ``[G, T, D]`` before calling and transpose the output back to
          ``[T, G, R]`` after.
    """
    chosen = backend
    if chosen == "auto":
        # CK kernel is a stub today (see csrc/ck_batched_gemm_fp8_blockwise/README.md).
        # Fall through to Triton, which is the production fast path.
        chosen = "triton"

    if chosen == "ck":
        global _CK_BROKEN
        if _CK_BROKEN:
            chosen = "triton"
        else:
            try:
                from aiter.ops._ck_batched_gemm_fp8_blockwise_loader import ck_batched_gemm_fp8_blockwise
                return ck_batched_gemm_fp8_blockwise(A, W, A_scale, W_scale, out=out)
            except (ImportError, NotImplementedError) as e:
                _log.warning(
                    "CK batched_gemm_fp8_blockwise unavailable (%s); "
                    "falling back to Triton (silent for subsequent calls).", e,
                )
                _CK_BROKEN = True
                chosen = "triton"

    if chosen == "triton":
        try:
            from aiter.ops.triton._triton_kernels.gemm.batched.batched_gemm_fp8_blockwise import (
                triton_batched_gemm_fp8_blockwise,
            )
            return triton_batched_gemm_fp8_blockwise(A, W, A_scale, W_scale, out=out)
        except (ImportError, RuntimeError) as e:
            _log.warning("Triton batched_gemm_fp8_blockwise failed (%s); falling back to torch.", e)
            chosen = "torch"

    out_ref = _torch_batched_gemm_fp8_blockwise(A, W, A_scale, W_scale)
    if out is not None:
        out.copy_(out_ref)
        return out
    return out_ref
