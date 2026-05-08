# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
DeepSeek V4 routing kernels for ROCm / MI355X.

Public entry points (used by sglang PR #23608 dispatcher patches under the
``SGLANG_OPT_FLYDSL_*`` env flags):

  * ``flydsl_moe_fused_gate_v4`` -- port of sglang's
    ``python/sglang/jit_kernel/csrc/moe/moe_fused_gate.cuh`` (V4 ungrouped
    fused gate: sigmoid|sqrt-softplus -> +bias -> iterative top-K -> renorm
    + optional shared-expert padding).
  * ``flydsl_topk_transform_512`` -- port of sglang's
    ``python/sglang/jit_kernel/csrc/deepseek_v4/topk.cuh`` (per-row radix
    top-512 + page-table transform for the CSA indexer).

Each entry point accepts a ``backend`` argument:

  * ``"flydsl"``  -- use the FlyDSL kernel (MLIR-builder pattern, requires
    a working FlyDSL toolchain; see ``kernels/*.py``).
  * ``"triton"``  -- use the Triton kernel (works on any ROCm + Triton
    install; the safe production fast-path while the FlyDSL kernels are
    still being tuned).
  * ``"torch"``   -- vendored copy of the sglang torch reference; primarily
    for correctness validation.
  * ``"auto"``    -- (default) tries FlyDSL, falls back to Triton, finally
    torch.  Failure to compile FlyDSL is logged once and remembered so we
    don't re-pay JIT cost.

The torch fallback in this module is *intentionally* a verbatim copy of the
sglang reference math so that it can be used as the bit-exact correctness
oracle in op_tests without taking an sglang dependency.
"""

from __future__ import annotations

import functools
import logging
import os
from typing import Literal, Optional, Tuple

import torch

from aiter import logger as _aiter_logger

logger = _aiter_logger
_log = logger.getChild("flydsl_v4")

# ----------------------------------------------------------------------------
# Backend selection.
# ----------------------------------------------------------------------------

Backend = Literal["auto", "flydsl", "triton", "torch"]

_FLYDSL_DISABLED = os.environ.get("AITER_DISABLE_FLYDSL_V4", "0") == "1"
_TRITON_DISABLED = os.environ.get("AITER_DISABLE_TRITON_V4", "0") == "1"

# Per-op flydsl-load failure cache.  Set on first ImportError or compile error.
_FLYDSL_BROKEN = {"fused_gate": False, "topk_transform": False}


def _resolve_backend(backend: Backend, op: str) -> str:
    """Resolve ``"auto"`` to a concrete backend; honour broken-cache entries."""
    if backend != "auto":
        return backend
    if not _FLYDSL_DISABLED and not _FLYDSL_BROKEN[op]:
        return "flydsl"
    if not _TRITON_DISABLED:
        return "triton"
    return "torch"


def _mark_flydsl_broken(op: str, exc: BaseException) -> None:
    if not _FLYDSL_BROKEN[op]:
        _log.warning(
            "FlyDSL backend for %s unavailable (%s: %s); falling back. "
            "Set AITER_DISABLE_FLYDSL_V4=1 to silence.",
            op,
            type(exc).__name__,
            exc,
        )
    _FLYDSL_BROKEN[op] = True


# ----------------------------------------------------------------------------
# Torch references (vendored from sglang PR #23608 for correctness oracle).
# ----------------------------------------------------------------------------


def _torch_fused_gate_v4(
    input: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    scoring_func: str,
    num_fused_shared_experts: int,
    renormalize: bool,
    routed_scaling_factor: float,
    apply_routed_scaling_factor_on_output: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Verbatim port of ``biased_topk_impl`` (deepseek_v4_topk.py:157) +
    ``MoEFusedGateKernel::run`` (moe_fused_gate.cuh:282) semantics.
    """
    assert input.dtype == torch.float32 and bias.dtype == torch.float32
    N, E = input.shape
    K_routed = topk - num_fused_shared_experts
    assert K_routed > 0

    if scoring_func == "sigmoid":
        scores = input.sigmoid()
    elif scoring_func == "sqrtsoftplus":
        scores = torch.nn.functional.softplus(input).sqrt()
    else:
        raise ValueError(f"unknown scoring_func: {scoring_func}")

    # CUDA kernel adds bias to scores for selection but uses *unbiased* score
    # for the output weight; tie-breaking favours the smaller expert id (see
    # moe_fused_gate.cuh:221).  torch.topk does not have that tie rule, so
    # we replicate it via a stable sort on (-biased, expert_id).
    biased = scores + bias.unsqueeze(0)  # [N, E]

    # Stable selection that exactly matches the CUDA tie rule.
    sort_keys = -biased  # largest biased -> smallest key
    sorted_keys, sorted_idx = torch.sort(sort_keys, dim=-1, stable=True)
    routed_idx = sorted_idx[:, :K_routed].to(torch.int32)
    routed_weights = scores.gather(1, routed_idx.long())

    # Pad shared-expert slots.
    out_w = torch.empty((N, topk), dtype=torch.float32, device=input.device)
    out_i = torch.empty((N, topk), dtype=torch.int32, device=input.device)
    out_w[:, :K_routed] = routed_weights
    out_i[:, :K_routed] = routed_idx

    if num_fused_shared_experts > 0:
        # CUDA assigns shared-expert ids by lane offset (see line 253).
        # Last `num_fused_shared_experts` slots get experts [E .. E+S-1].
        for s in range(num_fused_shared_experts):
            out_i[:, K_routed + s] = E + s
        # Shared weight = routed_sum / routed_scaling_factor (line 165).
        routed_sum = routed_weights.sum(dim=-1)
        for s in range(num_fused_shared_experts):
            out_w[:, K_routed + s] = routed_sum / routed_scaling_factor

    if renormalize:
        norm = routed_weights.sum(dim=-1, keepdim=True)
        norm = torch.where(norm > 0.0, norm, torch.ones_like(norm))
        out_w[:, :K_routed] = out_w[:, :K_routed] / norm
        if num_fused_shared_experts > 0:
            # CUDA divides shared by the same routed_sum norm too; see line 168
            # `const auto norm = renormalize && routed_sum > 0.0f ? routed_sum : 1.0f`.
            for s in range(num_fused_shared_experts):
                out_w[:, K_routed + s] = out_w[:, K_routed + s] / norm.squeeze(-1)

    if apply_routed_scaling_factor_on_output:
        out_w = out_w * routed_scaling_factor

    return out_w, out_i


def _torch_topk_transform_512(
    scores: torch.Tensor,           # [B, max_seq_len] fp32
    seq_lens: torch.Tensor,         # [B] i32
    page_tables: torch.Tensor,      # [B, num_pages] i32
    out_page_indices: torch.Tensor, # [B, 512] i32 -- written
    page_size: int,
    out_raw_indices: Optional[torch.Tensor] = None,
) -> None:
    """Vendored from sglang ``topk_transform_512_pytorch_vectorized`` (indexer.py:210)."""
    TOPK = 512
    B = scores.shape[0]
    max_seq_len = scores.shape[1]
    device = scores.device

    page_bits = (page_size - 1).bit_length() if page_size > 1 else 0
    page_mask = page_size - 1

    positions = torch.arange(max_seq_len, device=device).unsqueeze(0).expand(B, -1)
    valid_mask = positions < seq_lens.unsqueeze(1)

    masked = scores.clone()
    masked[~valid_mask] = float("-inf")

    actual_k = min(TOPK, max_seq_len)
    _, raw_idx = torch.topk(masked, k=actual_k, dim=1, largest=True, sorted=False)
    raw_idx = raw_idx.to(torch.int32)

    if actual_k < TOPK:
        pad = torch.zeros((B, TOPK - actual_k), dtype=torch.int32, device=device)
        raw_idx = torch.cat([raw_idx, pad], dim=1)

    # For short sequences (<= TOPK) just emit sequential indices padded with -1.
    needs_seq = seq_lens <= TOPK
    if needs_seq.any():
        seq_idx = torch.arange(TOPK, device=device, dtype=torch.int32).unsqueeze(0).expand(B, -1)
        seq_valid = seq_idx < seq_lens.unsqueeze(1)
        raw_idx = torch.where(
            needs_seq.unsqueeze(1).expand(-1, TOPK),
            torch.where(seq_valid, seq_idx, torch.tensor(-1, device=device, dtype=torch.int32)),
            raw_idx,
        )

    page_idx = raw_idx >> page_bits
    offset_in_page = raw_idx & page_mask
    page_idx_clamped = torch.clamp(page_idx, min=0)
    physical = torch.gather(page_tables, dim=1, index=page_idx_clamped.long())
    page_indices = ((physical << page_bits) | offset_in_page).to(torch.int32)

    # invalid raw_idx (-1) -> -1 page_idx
    invalid = raw_idx < 0
    page_indices = torch.where(invalid, torch.full_like(page_indices, -1), page_indices)

    out_page_indices.copy_(page_indices)
    if out_raw_indices is not None:
        out_raw_indices.copy_(raw_idx)


# ----------------------------------------------------------------------------
# Backend dispatchers (lazy imports keep aiter import-time light).
# ----------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _load_triton_fused_gate():
    from aiter.ops.flydsl.kernels.fused_gate_v4_triton import triton_moe_fused_gate_v4
    return triton_moe_fused_gate_v4


@functools.lru_cache(maxsize=1)
def _load_triton_topk_transform():
    from aiter.ops.flydsl.kernels.topk_transform_512_triton import triton_topk_transform_512
    return triton_topk_transform_512


@functools.lru_cache(maxsize=1)
def _load_flydsl_fused_gate():
    from aiter.ops.flydsl.kernels.fused_gate_v4 import flydsl_kernel_moe_fused_gate_v4
    return flydsl_kernel_moe_fused_gate_v4


@functools.lru_cache(maxsize=1)
def _load_flydsl_topk_transform():
    from aiter.ops.flydsl.kernels.topk_transform_512 import flydsl_kernel_topk_transform_512
    return flydsl_kernel_topk_transform_512


# ----------------------------------------------------------------------------
# Public API.
# ----------------------------------------------------------------------------


def flydsl_moe_fused_gate_v4(
    input: torch.Tensor,
    bias: torch.Tensor,
    *,
    topk: int,
    scoring_func: str = "sigmoid",
    num_fused_shared_experts: int = 0,
    renormalize: bool = True,
    routed_scaling_factor: float = 1.0,
    apply_routed_scaling_factor_on_output: bool = False,
    backend: Backend = "auto",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    DeepSeek V4 ungrouped fused MoE gate.

    Mirrors the contract of sglang's ``moe_fused_gate(input, bias, topk, ...)``
    in ``python/sglang/jit_kernel/moe_fused_gate.py``.

    Args:
        input: ``[N, E]`` fp32 router logits.
        bias:  ``[E]``    fp32 per-expert correction bias.
        topk:  total selected experts (including shared).  ``topk > num_fused_shared_experts``.
        scoring_func: ``"sigmoid"`` or ``"sqrtsoftplus"``.
        num_fused_shared_experts: shared experts appended to top-K (their ids
            are ``[E .. E+S-1]``).
        renormalize: divide routed weights by their sum.
        routed_scaling_factor: scaling for shared-expert weight (= ``routed_sum / rsf``)
            and (when ``apply_routed_scaling_factor_on_output``) for routed weights too.
        backend: see module docstring.

    Returns:
        ``(weights[N, topk] fp32, indices[N, topk] i32)``.
    """
    assert input.is_cuda and bias.is_cuda
    assert input.dtype == torch.float32 and bias.dtype == torch.float32
    assert input.dim() == 2 and bias.dim() == 1
    assert input.shape[1] == bias.shape[0]
    assert topk > num_fused_shared_experts

    chosen = _resolve_backend(backend, "fused_gate")

    if chosen == "flydsl":
        try:
            kernel = _load_flydsl_fused_gate()
            return kernel(
                input,
                bias,
                topk=topk,
                scoring_func=scoring_func,
                num_fused_shared_experts=num_fused_shared_experts,
                renormalize=renormalize,
                routed_scaling_factor=routed_scaling_factor,
                apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
            )
        except (ImportError, RuntimeError, NotImplementedError) as e:
            _mark_flydsl_broken("fused_gate", e)
            chosen = "triton"

    if chosen == "triton":
        try:
            kernel = _load_triton_fused_gate()
            return kernel(
                input,
                bias,
                topk=topk,
                scoring_func=scoring_func,
                num_fused_shared_experts=num_fused_shared_experts,
                renormalize=renormalize,
                routed_scaling_factor=routed_scaling_factor,
                apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
            )
        except (ImportError, RuntimeError) as e:
            _log.warning("Triton fused_gate failed (%s); falling back to torch.", e)
            chosen = "torch"

    return _torch_fused_gate_v4(
        input, bias, topk, scoring_func, num_fused_shared_experts,
        renormalize, routed_scaling_factor, apply_routed_scaling_factor_on_output,
    )


def flydsl_topk_transform_512(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
    out_raw_indices: Optional[torch.Tensor] = None,
    *,
    backend: Backend = "auto",
) -> None:
    """
    DeepSeek V4 CSA indexer: per-row radix top-512 over scores plus page-table
    transform to produce paged sparse indices.

    Mirrors the contract of sglang's ``topk_transform_512(scores, seq_lens,
    page_tables, out_page_indices, page_size, out_raw_indices=None)`` in
    ``python/sglang/jit_kernel/deepseek_v4.py:140``.

    Inputs:
        scores:      [B, max_seq_len] fp32  (may be strided)
        seq_lens:    [B] i32, contiguous
        page_tables: [B, num_pages] i32     (may be strided)
        out_page_indices: [B, 512] i32, contiguous -- written
        page_size:   power of 2
        out_raw_indices:  optional [B, 512] i32 -- raw absolute positions
            *before* page transform; written when provided.
    """
    assert scores.is_cuda and seq_lens.is_cuda and page_tables.is_cuda
    assert out_page_indices.is_cuda and out_page_indices.shape[1] == 512
    assert (page_size & (page_size - 1)) == 0, "page_size must be power of 2"

    chosen = _resolve_backend(backend, "topk_transform")

    if chosen == "flydsl":
        try:
            kernel = _load_flydsl_topk_transform()
            kernel(
                scores, seq_lens, page_tables,
                out_page_indices, page_size, out_raw_indices,
            )
            return
        except (ImportError, RuntimeError, NotImplementedError) as e:
            _mark_flydsl_broken("topk_transform", e)
            chosen = "triton"

    if chosen == "triton":
        try:
            kernel = _load_triton_topk_transform()
            kernel(
                scores, seq_lens, page_tables,
                out_page_indices, page_size, out_raw_indices,
            )
            return
        except (ImportError, RuntimeError) as e:
            _log.warning("Triton topk_transform_512 failed (%s); falling back to torch.", e)
            chosen = "torch"

    _torch_topk_transform_512(
        scores, seq_lens, page_tables,
        out_page_indices, page_size, out_raw_indices,
    )
