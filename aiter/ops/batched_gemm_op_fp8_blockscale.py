# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Public entry point for the CK FP8 block-wise batched GEMM kernel.

This is the kernel that DeepSeek V4's ``wo_a`` projection needs (and that
sglang PR #23608 currently sources from ``deep_gemm.fp8_einsum``). Mirrors
the contract of:

    deep_gemm.fp8_einsum("bmk,bnk->bmn", (A, A_scale), (W, W_scale), out,
                         recipe=(1, 1, 128))

Shape constraints: ``M >= 128 && M % 128 == 0 && N % 128 == 0 && K % 128 == 0``.
Shapes outside that envelope raise ``ValueError``; callers are responsible
for routing such shapes elsewhere (e.g. the decode-optimised flydsl kernel
on the ``wo_a_fp8_blockwise`` branch).

Scale dtype: ``A_scale`` and ``W_scale`` must share dtype, either ``fp32`` or
``uint8`` UE8M0. uint8 scales are converted to fp32 on the GPU before the
CK call (CK's template requires fp32); ``W_scale`` conversion is cached
via a weakref keyed on ``id(W_scale)`` so weights pay the ~10-50us cast
exactly once per tensor lifetime.
"""

from __future__ import annotations

import weakref as _weakref
from typing import Optional

import torch

from aiter import logger as _aiter_logger
from aiter.jit.core import compile_ops

logger = _aiter_logger
_log = logger.getChild("batched_gemm_fp8_blockscale")

__all__ = ["batched_gemm_fp8_blockscale", "batched_gemm_fp8_blockscale_tune"]


# ----------------------------------------------------------------------------
# JIT-compiled CK kernel bindings.
# ----------------------------------------------------------------------------


def _gen_fake_out(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    Out: torch.Tensor,
) -> torch.Tensor:
    return Out


@compile_ops(
    "module_batched_gemm_fp8_blockscale",
    fc_name="batched_gemm_fp8_blockscale",
    gen_fake=_gen_fake_out,
)
def _batched_gemm_fp8_blockscale(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    Out: torch.Tensor,
) -> torch.Tensor: ...


@compile_ops(
    "module_batched_gemm_fp8_blockscale_tune",
    fc_name="batched_gemm_fp8_blockscale_tune",
    gen_fake=lambda XQ, WQ, x_scale, w_scale, Out, kernelId, splitK=0: Out,
)
def batched_gemm_fp8_blockscale_tune(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    Out: torch.Tensor,
    kernelId: int,
    splitK: int = 0,
) -> torch.Tensor: ...


# ----------------------------------------------------------------------------
# uint8 UE8M0 -> fp32 scale conversion. CK's host check rejects non-fp32
# scales; for callers that hold u8 scales (e.g. weights pre-converted at
# model load time), do the one-way reverse conversion here. Result is
# CACHED so subsequent calls with the same u8 tensor reuse the converted
# fp32 view (zero overhead in the hot loop).
# ----------------------------------------------------------------------------

# Keyed by id(u8_tensor); values are weakrefs so converted scales auto-evict
# when the source weight scale tensor is freed.
_U8_TO_FP32_CACHE: "dict[int, _weakref.ReferenceType[torch.Tensor]]" = {}


def _ue8m0_to_fp32(scales_u8: torch.Tensor) -> torch.Tensor:
    """uint8 UE8M0 -> fp32 multiplicative scales (cached by tensor id)."""
    if scales_u8.dtype == torch.float32:
        return scales_u8
    key = id(scales_u8)
    cached_ref = _U8_TO_FP32_CACHE.get(key)
    cached = cached_ref() if cached_ref is not None else None
    if cached is not None and cached.device == scales_u8.device:
        return cached
    # scale[i] = 2 ** (u8[i] - 127); u8=0 -> 2^(-127) (effectively zero).
    fp32 = torch.exp2((scales_u8.to(torch.float32) - 127.0))
    fp32 = fp32.contiguous()
    _U8_TO_FP32_CACHE[key] = _weakref.ref(fp32)
    return fp32


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
) -> torch.Tensor:
    """
    FP8 block-wise batched GEMM:  ``Out[b, m, n] = sum_k A[b, m, k] * W[b, n, k]``.

    Args:
        A:        ``[B, M, K]`` ``fp8_e4m3fn``.
        W:        ``[B, N, K]`` ``fp8_e4m3fn``.
        A_scale:  ``[B, M, K // 128]`` ``fp32`` or ``uint8`` (UE8M0).
        W_scale:  ``[B, N // 128, K // 128]`` ``fp32`` or ``uint8`` (UE8M0).
                  Must share dtype with ``A_scale``.
        out:      Optional pre-allocated ``[B, M, N]`` ``bfloat16`` output.

    Returns:
        ``Out[B, M, N] bfloat16``.

    Raises:
        ValueError: M/N/K shape constraints not satisfied, or A_scale and
            W_scale dtypes don't match.
        TypeError:  scale dtype is not torch.float32 or torch.uint8.
        ImportError / RuntimeError: CK kernel cannot be loaded.
    """
    _, M, K = A.shape
    N = W.shape[1]
    if A_scale.dtype != W_scale.dtype:
        raise ValueError(
            f"batched_gemm_fp8_blockscale: A_scale.dtype ({A_scale.dtype}) "
            f"and W_scale.dtype ({W_scale.dtype}) must match -- pass both as "
            f"torch.float32 or both as torch.uint8 (UE8M0)."
        )
    if A_scale.dtype not in (torch.float32, torch.uint8):
        raise TypeError(
            f"batched_gemm_fp8_blockscale: scale dtype must be torch.float32 "
            f"or torch.uint8; got {A_scale.dtype}."
        )

    if out is None:
        out = torch.empty((A.shape[0], M, N), dtype=torch.bfloat16, device=A.device)

    # CK kernel requires fp32 scales. Convert if needed (W_scale cached).
    if A_scale.dtype == torch.uint8:
        A_scale = _ue8m0_to_fp32(A_scale)
        W_scale = _ue8m0_to_fp32(W_scale)

    _batched_gemm_fp8_blockscale(A, W, A_scale, W_scale, out)
    return out
