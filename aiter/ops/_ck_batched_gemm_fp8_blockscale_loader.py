# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Lazy loader for the CK FP8 block-wise batched GEMM modules.

Two modules are exposed:

  * ``batched_gemm_fp8_blockscale(XQ, WQ, x_scale, w_scale, Out)`` -- the
    production dispatcher used by ``aiter.batched_gemm_fp8_blockscale``
    (backend="ck").  Picks the kernel from the lookup CSV / heuristic.

  * ``batched_gemm_fp8_blockscale_tune(XQ, WQ, x_scale, w_scale, Out,
    kernelId, splitK)`` -- direct kernel selection by integer id, for
    the tune driver in
    ``csrc/ck_batched_gemm_fp8_blockscale/batched_gemm_fp8_blockscale_tune.py``.
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


@compile_ops("module_batched_gemm_fp8_blockscale", fc_name="batched_gemm_fp8_blockscale",
             gen_fake=_gen_fake_out)
def _batched_gemm_fp8_blockscale(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    Out: torch.Tensor,
) -> torch.Tensor: ...


@compile_ops("module_batched_gemm_fp8_blockscale_tune", fc_name="batched_gemm_fp8_blockscale_tune",
             gen_fake=lambda XQ, WQ, x_scale, w_scale, Out, kernelId, splitK=0: Out)
def batched_gemm_fp8_blockscale_tune(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    Out: torch.Tensor,
    kernelId: int,
    splitK: int = 0,
) -> torch.Tensor: ...


# Cache for u8 -> fp32 scale conversion. Keyed by id(u8_tensor).
# Holds weak references via a dict so converted scales auto-evict when the
# source weight scale tensor is freed.
import weakref as _weakref
_U8_TO_FP32_CACHE: "dict[int, _weakref.ReferenceType[torch.Tensor]]" = {}


def _ue8m0_to_fp32(scales_u8: torch.Tensor) -> torch.Tensor:
    """uint8 UE8M0 -> fp32 multiplicative scales.

    CK's host check rejects non-fp32 scales. For callers that hold u8
    scales (e.g. weights pre-converted at model load time), do the one-way
    reverse conversion here. Result is CACHED so subsequent calls with
    the same u8 tensor reuse the converted fp32 view (zero overhead in
    the hot loop).
    """
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


def ck_batched_gemm_fp8_blockscale(
    A: torch.Tensor,
    W: torch.Tensor,
    A_scale: torch.Tensor,
    W_scale: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Public CK entry point. Same contract as ``aiter.batched_gemm_fp8_blockscale``.

    Scale dtype:
      * Accepts either fp32 *or* uint8 (UE8M0) scales -- but ``A_scale`` and
        ``W_scale`` MUST share the same dtype. Mixing fp32 with u8 is an
        error (the underlying CK template treats both scale tensors via the
        same template parameter, and mixed-dtype is not legal at the model
        level either: production either pre-converts both at load time or
        keeps both in fp32).
      * uint8 scales are converted to fp32 on the GPU before the CK call
        (CK's template requires fp32). The W_scale conversion is cached
        via a weak-ref keyed on ``id(W_scale)`` so weights pay the
        ~10-50us cast exactly once per tensor lifetime.
    """
    if A_scale.dtype != W_scale.dtype:
        raise TypeError(
            f"ck_batched_gemm_fp8_blockscale: A_scale.dtype ({A_scale.dtype}) "
            f"and W_scale.dtype ({W_scale.dtype}) must match -- pass both as "
            f"torch.float32 or both as torch.uint8 (UE8M0)."
        )
    if A_scale.dtype not in (torch.float32, torch.uint8):
        raise TypeError(
            f"ck_batched_gemm_fp8_blockscale: scale dtype must be torch.float32 "
            f"or torch.uint8; got {A_scale.dtype}."
        )
    if out is None:
        B, M, _ = A.shape
        N = W.shape[1]
        out = torch.empty((B, M, N), dtype=torch.bfloat16, device=A.device)
    # CK kernel requires fp32 scales. Convert if needed (W_scale cached).
    if A_scale.dtype == torch.uint8:
        A_scale = _ue8m0_to_fp32(A_scale)
        W_scale = _ue8m0_to_fp32(W_scale)
    _batched_gemm_fp8_blockscale(A, W, A_scale, W_scale, out)
    return out
