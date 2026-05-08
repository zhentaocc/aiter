# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
FlyDSL (MLIR-builder) port of DeepSeek V4 ungrouped fused MoE gate.

CUDA reference: ``python/sglang/jit_kernel/csrc/moe/moe_fused_gate.cuh``
(``moe_fused_gate_kernel`` and ``moe_fused_gate_kernel_small_token``).

Status: SKELETON.  The structural elements (workgroup layout, score
computation, iterative top-K loop body) are sketched in the FlyDSL /
flir MLIR-builder dialects following the patterns in
``aiter/aiter/ops/flydsl/kernels/reduce.py`` and
``aiter/aiter/ops/flydsl/kernels/moe_gemm_2stage.py``.  Inner numerical
correctness (in particular the warp-shuffle argmax with min-id tie-break,
the LDS scratch lifetime for the cross-wave reduction, and the
shared-expert append) requires iteration against the FlyDSL compiler;
see ``v4_routing.py`` which auto-falls-back to the Triton port until
this kernel is signed off.

When promoting this from skeleton to production, the implementation should
follow these design points (taken from the approved plan
``/home/zhenchen/.claude/plans/velvet-painting-brook.md``):

  * **Two variants** matching CUDA's ``kSmallTokenThreshold = 512``:
    - small-N (N <= 512): 1 token per workgroup, ``warps_per_token`` waves
      of 64 lanes covering up to E=512 experts, LDS holds ``2 * E * fp32``.
    - large-N (N > 512): 4 tokens per workgroup (CDNA-natural; CUDA uses 6),
      one wave-64 per token, LDS partitioned per warp.
  * **Iterative top-K**: ``for k in range(K_routed): warp_argmax via
    gpu.ShuffleOp xor mode width=64; lane0 writes selected[k] and overwrites
    biased[id] = -inf; barrier``.  ``K_routed <= 16`` so the tail fits in
    one wave.
  * **Score functions** branched as constexpr templates:
    sigmoid -> ``1 / (1 + exp(-x))``; sqrt-softplus -> ``sqrt(log1p(exp(x)))``.
  * **Renormalization** + shared-expert append in a final post-loop block,
    written by lanes ``[0, K_routed)`` and ``[K_routed, K)`` respectively.

Compile cache key: ``(N_bucket, E_rounded_up_to_64, topk, scoring_func,
num_fused_shared_experts)``.

This file intentionally does NOT raise on import; the wrapper imports it
lazily and catches NotImplementedError to fall back gracefully.
"""

from __future__ import annotations

import functools
from typing import Tuple

import torch

# FlyDSL toolchain imports.  Wrapped in try/except so import-time of aiter
# never fails when FlyDSL is missing/broken.
try:
    import flydsl.compiler as flyc  # noqa: F401
    from flydsl.dialects.ext.python_control_flow import lower_range_for_loops  # noqa: F401
    _FLYDSL_OK = True
except ImportError as _e:
    _FLYDSL_OK = False
    _FLYDSL_IMPORT_ERR = _e


# ---------------------------------------------------------------------------
# Cache keys.
# ---------------------------------------------------------------------------


def _bucket_n(N: int) -> int:
    for b in (1, 8, 32, 128, 512, 2048, 8192):
        if N <= b:
            return b
    return ((N + 8191) // 8192) * 8192


def _round_up_e(E: int) -> int:
    return ((E + 63) // 64) * 64


# ---------------------------------------------------------------------------
# Kernel module compile (cached).
#
# This is the placeholder where the actual MLIR-builder construction lives.
# We define the function, document the intended structure, and raise
# NotImplementedError so the wrapper's auto-mode degrades to Triton.
#
# A working implementation would:
#   1. Open a FlyDSL kernel context with grid=(num_blocks,) and
#      block=(WARP_SIZE * warps_per_block,).
#   2. Allocate LDS for ``2 * E_rounded * f32`` (biased + original scores).
#   3. Loop: cooperatively compute scores into LDS via ``flir.arith`` ops.
#   4. ``gpu.barrier()``.
#   5. For k in range(K_routed):
#        a. Per-thread local max over its strided expert range.
#        b. Warp reduce_max with tie-break via ``gpu.ShuffleOp`` (xor)
#           and ``flir.arith.MaximumFOp`` (the same pattern as
#           ``aiter/ops/flydsl/kernels/reduce.py``).
#        c. Cross-wave merge through LDS scratch (use the
#           ``make_block_reduce`` helper from reduce.py).
#        d. Lane 0 records selected[k] and overwrites LDS[selected]=-inf.
#        e. ``gpu.barrier()``.
#   6. Lanes [0, K_routed) load original_score[selected[lane]] and
#      participate in a warp reduce_sum to compute routed_sum.
#   7. Lanes [0, K) write final outputs with optional renorm + shared-expert
#      patch (id = E + lane - K_routed, weight = routed_sum / rsf).
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _build_module(
    N_bucket: int,
    E_rounded: int,
    topk: int,
    scoring_func: str,
    num_fused_shared_experts: int,
    renormalize: bool,
    apply_rsf_out: bool,
):
    if not _FLYDSL_OK:
        raise NotImplementedError(
            f"FlyDSL not importable: {_FLYDSL_IMPORT_ERR}"
        ) from _FLYDSL_IMPORT_ERR

    # The MLIR-builder construction goes here.  See the design comments
    # at the top of this file.  Until that is implemented, signal the
    # wrapper to use the Triton fallback.
    raise NotImplementedError(
        "FlyDSL fused_gate_v4 kernel module not yet implemented; "
        "the Triton port at fused_gate_v4_triton.py is the current fast path. "
        f"Requested config: N_bucket={N_bucket} E_rounded={E_rounded} "
        f"topk={topk} scoring={scoring_func} shared={num_fused_shared_experts} "
        f"renorm={renormalize} apply_rsf_out={apply_rsf_out}."
    )


def flydsl_kernel_moe_fused_gate_v4(
    input: torch.Tensor,
    bias: torch.Tensor,
    *,
    topk: int,
    scoring_func: str = "sigmoid",
    num_fused_shared_experts: int = 0,
    renormalize: bool = True,
    routed_scaling_factor: float = 1.0,
    apply_routed_scaling_factor_on_output: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Entry point.  See ``v4_routing.flydsl_moe_fused_gate_v4`` for the contract."""
    N, E = input.shape
    module = _build_module(
        N_bucket=_bucket_n(N),
        E_rounded=_round_up_e(E),
        topk=topk,
        scoring_func=scoring_func,
        num_fused_shared_experts=num_fused_shared_experts,
        renormalize=renormalize,
        apply_rsf_out=apply_routed_scaling_factor_on_output,
    )

    weights = torch.empty((N, topk), dtype=torch.float32, device=input.device)
    indices = torch.empty((N, topk), dtype=torch.int32, device=input.device)

    # When the kernel is implemented, the launch contract is:
    #   module(input, bias, weights, indices, N, E, routed_scaling_factor)
    # with grid X = ceil(N / tokens_per_block).
    module(
        input,
        bias,
        weights,
        indices,
        N,
        E,
        float(routed_scaling_factor),
    )
    return weights, indices
