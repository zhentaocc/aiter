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


# ----------------------------------------------------------------------------
# Einsum-form entry point.
#
# Avoids the host-side ``.transpose(...).contiguous()`` round-trips that
# callers otherwise need to map their natural layout (e.g. wo_a's
# "tgd,grd->tgr" or DeepGEMM's "bhr,hdr->bhd") into the kernel's canonical
# [B, M, K] / [B, N, K] / [B, M, N] form.  All ``permute``s here are
# stride-only views -- no data movement.
# ----------------------------------------------------------------------------


import functools


@functools.lru_cache(maxsize=64)
def _parse_bmnk(equation: str):
    """
    Parse a 3-tensor batched-GEMM einsum like ``"bhr,hdr->bhd"`` into the
    four role letters (B, M, K, N) plus per-tensor permutations to canonical
    BMK / BNK / BMN axis order.

    Constraints (no diagonals, no broadcasting, no reductions other than K):
      * each input has exactly 3 axes; output has exactly 3 axes
      * B = letter present in A, W, and Out
      * M = letter present in A and Out, absent from W
      * N = letter present in W and Out, absent from A
      * K = letter present in A and W, absent from Out (the contraction)
    """
    lhs, rhs = equation.replace(" ", "").split("->")
    a_eq, w_eq = lhs.split(",")
    o_eq = rhs
    if not (len(a_eq) == 3 and len(w_eq) == 3 and len(o_eq) == 3):
        raise ValueError(f"einsum {equation!r}: each tensor must have exactly 3 axes")

    sa, sw, so = set(a_eq), set(w_eq), set(o_eq)
    cand_B = sa & sw & so
    cand_M = (sa & so) - sw
    cand_N = (sw & so) - sa
    cand_K = (sa & sw) - so
    for label, s in (("B", cand_B), ("M", cand_M), ("N", cand_N), ("K", cand_K)):
        if len(s) != 1:
            raise ValueError(
                f"einsum {equation!r}: cannot uniquely identify {label} axis "
                f"(candidates: {s}); supported pattern is one B/M/N/K letter each"
            )
    B, M, N, K = cand_B.pop(), cand_M.pop(), cand_N.pop(), cand_K.pop()

    a_perm = (a_eq.index(B), a_eq.index(M), a_eq.index(K))      # → BMK
    w_perm = (w_eq.index(B), w_eq.index(N), w_eq.index(K))      # → BNK
    o_perm = (o_eq.index(B), o_eq.index(M), o_eq.index(N))      # → BMN

    return {
        "B": B, "M": M, "N": N, "K": K,
        "a_perm": a_perm, "w_perm": w_perm, "o_perm": o_perm,
        "a_eq": a_eq, "w_eq": w_eq, "o_eq": o_eq,
    }


def _ensure_k_innermost(t: torch.Tensor, k_axis: int, name: str) -> torch.Tensor:
    """
    The Triton kernel issues vector loads on the K axis (BLOCK_K=128 contiguous
    elements per iteration).  If after permute the K axis is not innermost
    with stride 1, we materialise a contiguous copy -- correctness over
    speed.  Most real callers (wo_a "tgd,grd->tgr" / "bhr,hdr->bhd") already
    have K innermost, so this is a no-op.
    """
    if t.stride(k_axis) == 1:
        return t
    _log.warning(
        "%s: K axis is not contiguous after einsum permute (stride=%d); "
        "materialising a contiguous copy. Consider passing the tensor with "
        "the contraction axis innermost to avoid this.",
        name, t.stride(k_axis),
    )
    return t.contiguous()


def batched_gemm_fp8_blockwise_einsum(
    equation: str,
    A: torch.Tensor,
    A_scale: torch.Tensor,
    W: torch.Tensor,
    W_scale: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    backend: Backend = "auto",
) -> torch.Tensor:
    """
    Einsum-form FP8 block-wise batched GEMM.  Equivalent to::

        deep_gemm.fp8_einsum(equation, (A, A_scale), (W, W_scale), out,
                             recipe=(1, 1, 128))

    but routes through aiter's kernel.

    Examples of supported equations:
      * ``"tgd,grd->tgr"``  -- sglang wo_a layout
      * ``"bhr,hdr->bhd"``  -- vllm/DeepGEMM wo_a layout
      * ``"bmk,bnk->bmn"``  -- canonical batched GEMM (no permute)

    Scale layout convention (matches DeepSeek V4 wo_a checkpoint):
      * ``A_scale`` shares A's einsum letters; the K axis size is ``K/128``.
      * ``W_scale`` shares W's einsum letters; both N and K sizes are ``/128``.

    No data is copied: ``permute`` is a stride-only view.  When
    ``out is None``, the output is allocated in the caller's einsum layout
    (so the caller never has to ``.contiguous()``).
    """
    info = _parse_bmnk(equation)
    a_eq, w_eq, o_eq = info["a_eq"], info["w_eq"], info["o_eq"]
    a_perm, w_perm, o_perm = info["a_perm"], info["w_perm"], info["o_perm"]

    # --- Permute A/W to BMK/BNK (stride-only) and check K is innermost.
    A_bmk = _ensure_k_innermost(A.permute(*a_perm), 2, "A")
    W_bnk = _ensure_k_innermost(W.permute(*w_perm), 2, "W")
    A_scale_bmk = A_scale.permute(*a_perm)
    W_scale_bnk = W_scale.permute(*w_perm)

    B = A_bmk.shape[0]
    M = A_bmk.shape[1]
    N = W_bnk.shape[1]

    # --- Allocate output in caller's einsum layout (or accept caller's `out`).
    dim_of = {info["B"]: B, info["M"]: M, info["N"]: N}
    out_shape = tuple(dim_of[c] for c in o_eq)
    if out is None:
        out = torch.empty(out_shape, dtype=torch.bfloat16, device=A.device)
    elif tuple(out.shape) != out_shape:
        raise ValueError(
            f"einsum {equation!r}: out shape {tuple(out.shape)} "
            f"does not match einsum-derived {out_shape}"
        )

    # --- Permute out (stride-only) to BMN view for the kernel.
    out_bmn = out.permute(*o_perm)

    # --- Dispatch to the existing BMK kernel.  Strides flow through.
    batched_gemm_fp8_blockwise(
        A_bmk, W_bnk, A_scale_bmk, W_scale_bnk,
        out=out_bmn, backend=backend,
    )

    return out


def _torch_batched_gemm_fp8_blockwise_einsum(
    equation: str,
    A: torch.Tensor,
    A_scale: torch.Tensor,
    W: torch.Tensor,
    W_scale: torch.Tensor,
) -> torch.Tensor:
    """Reference oracle for the einsum form (mirrors the wrapper above)."""
    info = _parse_bmnk(equation)
    A_bmk = A.permute(*info["a_perm"]).contiguous()
    W_bnk = W.permute(*info["w_perm"]).contiguous()
    A_scale_bmk = A_scale.permute(*info["a_perm"]).contiguous()
    W_scale_bnk = W_scale.permute(*info["w_perm"]).contiguous()
    out_bmn = _torch_batched_gemm_fp8_blockwise(A_bmk, W_bnk, A_scale_bmk, W_scale_bnk)
    inv_o = [info["o_perm"].index(i) for i in range(3)]
    return out_bmn.permute(*inv_o).contiguous()
