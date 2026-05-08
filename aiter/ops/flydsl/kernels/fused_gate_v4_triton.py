# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Triton implementation of DeepSeek V4 ungrouped fused MoE gate.

This is the *production* fast path while ``fused_gate_v4.py`` (the FlyDSL
MLIR-builder kernel) is still being tuned.  Both expose the same call
signature; the wrapper in ``v4_routing.py`` selects between them.

Algorithm faithfully mirrors ``moe_fused_gate.cuh``:

  1. score = sigmoid(x) | sqrt(softplus(x))     (per-token, all experts)
  2. biased = score + bias                      (used for selection only)
  3. iterative top-K_routed:
        argmax of biased; on ties, choose smaller expert id
        record (id, score@id); set biased[id] = -inf; continue
  4. optional renormalize over routed weights
  5. append shared experts at the tail (id = E + s, weight = routed_sum / rsf)
  6. optional apply_routed_scaling_factor on output

Tile model:
  * one program (workgroup) per row.  BLOCK_E covers all experts (E <= 512
    in V4-Pro / V4-Flash).  topk_routed <= 16, so the iterative loop is a
    static range.

This intentionally does NOT use grouped-topk semantics -- the V3-style
grouped path is already covered by ``aiter.moe_fused_gate``.
"""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl


# Maximum experts supported.  V4-Flash uses E=257 (256 + 1 shared); V4-Pro
# uses E=512.  CUDA reference caps at 512.
_BLOCK_E_FOR_E = {
    128: 128,
    256: 256,
    257: 512,   # one routed bias slot beyond 256
    384: 512,
    512: 512,
}


def _block_e_for(E: int) -> int:
    if E in _BLOCK_E_FOR_E:
        return _BLOCK_E_FOR_E[E]
    # round up to next power of two >= 64
    bs = 64
    while bs < E:
        bs *= 2
    return bs


@triton.jit
def _fused_gate_v4_kernel(
    input_ptr,            # [N, E] fp32
    bias_ptr,             # [E] fp32
    weights_ptr,          # [N, K] fp32
    indices_ptr,          # [N, K] i32
    N,
    E,
    K,
    K_ROUTED: tl.constexpr,   # K - num_fused_shared_experts
    NUM_SHARED: tl.constexpr,
    BLOCK_E: tl.constexpr,
    SCORING_FUNC: tl.constexpr,   # 0=sigmoid, 1=sqrtsoftplus
    RENORMALIZE: tl.constexpr,
    APPLY_RSF_OUT: tl.constexpr,
    routed_scaling_factor,  # fp32 scalar
):
    row = tl.program_id(0)
    if row >= N:
        return

    offs_e = tl.arange(0, BLOCK_E)
    valid = offs_e < E

    x = tl.load(input_ptr + row * E + offs_e, mask=valid, other=0.0)
    b = tl.load(bias_ptr + offs_e, mask=valid, other=0.0)

    # Score in fp32.
    if SCORING_FUNC == 0:
        # sigmoid(x) = 1 / (1 + exp(-x))
        score = 1.0 / (1.0 + tl.exp(-x))
    else:
        # sqrt(log1p(exp(x)))
        # Numerically: softplus(x) = log1p(exp(x)) is ok for x in normal range.
        score = tl.sqrt(tl.log(1.0 + tl.exp(x)))

    # Mask-out invalid lanes so they never win the argmax.
    NEG_INF = float("-inf")
    biased = tl.where(valid, score + b, NEG_INF)
    score_v = tl.where(valid, score, 0.0)

    # Iterative top-K_routed.  Each iteration: argmax with tie-break = lowest id.
    # We accumulate routed_sum on the fly so we don't need a second pass.
    routed_sum = tl.zeros((), dtype=tl.float32)
    # Loop body uses two scalar stores per iteration.  K_ROUTED is constexpr (<=16).
    for k in tl.static_range(0, K_ROUTED):
        max_val = tl.max(biased, axis=0)
        # Tie-break: among lanes equal to max_val, take the smallest id.
        is_max = (biased == max_val) & valid
        id_or_big = tl.where(is_max, offs_e, E + 1)
        winner = tl.min(id_or_big, axis=0)
        # weight = score (unbiased) at winner
        winner_mask = (offs_e == winner) & valid
        w = tl.sum(tl.where(winner_mask, score_v, 0.0), axis=0)
        # store
        tl.store(indices_ptr + row * K + k, winner)
        tl.store(weights_ptr + row * K + k, w)
        routed_sum += w
        # mask out the winner so the next iteration picks the next-largest
        biased = tl.where(winner_mask, NEG_INF, biased)

    # Renormalization (apply in-place to the routed slots we just wrote).
    if RENORMALIZE:
        norm = tl.where(routed_sum > 0.0, routed_sum, 1.0)
        # We need to overwrite indices_ptr+row*K+[0:K_ROUTED] with normalized weights.
        for k in tl.static_range(0, K_ROUTED):
            w = tl.load(weights_ptr + row * K + k)
            scale = routed_scaling_factor if APPLY_RSF_OUT else 1.0
            tl.store(weights_ptr + row * K + k, (w / norm) * scale)
    elif APPLY_RSF_OUT:
        for k in tl.static_range(0, K_ROUTED):
            w = tl.load(weights_ptr + row * K + k)
            tl.store(weights_ptr + row * K + k, w * routed_scaling_factor)

    # Shared experts: id = E + s, weight = routed_sum / rsf (further /norm if renormalize).
    if NUM_SHARED > 0:
        denom = routed_scaling_factor
        if RENORMALIZE:
            norm_s = tl.where(routed_sum > 0.0, routed_sum, 1.0)
            shared_weight = (routed_sum / denom) / norm_s
        else:
            shared_weight = routed_sum / denom
        if APPLY_RSF_OUT:
            shared_weight = shared_weight * routed_scaling_factor
        for s in tl.static_range(0, NUM_SHARED):
            tl.store(indices_ptr + row * K + K_ROUTED + s, E + s)
            tl.store(weights_ptr + row * K + K_ROUTED + s, shared_weight)


_SCORING_MAP = {"sigmoid": 0, "sqrtsoftplus": 1}


def triton_moe_fused_gate_v4(
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
    """See ``flydsl_moe_fused_gate_v4`` for the contract."""
    N, E = input.shape
    K_routed = topk - num_fused_shared_experts
    assert K_routed > 0
    assert K_routed <= 16, f"V4 fused gate caps K_routed <= 16, got {K_routed}"
    assert E <= 512, f"V4 fused gate caps E <= 512, got {E}"
    assert scoring_func in _SCORING_MAP

    BLOCK_E = _block_e_for(E)

    weights = torch.empty((N, topk), dtype=torch.float32, device=input.device)
    indices = torch.empty((N, topk), dtype=torch.int32, device=input.device)

    grid = (N,)
    _fused_gate_v4_kernel[grid](
        input,
        bias,
        weights,
        indices,
        N,
        E,
        topk,
        K_ROUTED=K_routed,
        NUM_SHARED=num_fused_shared_experts,
        BLOCK_E=BLOCK_E,
        SCORING_FUNC=_SCORING_MAP[scoring_func],
        RENORMALIZE=bool(renormalize),
        APPLY_RSF_OUT=bool(apply_routed_scaling_factor_on_output),
        routed_scaling_factor=float(routed_scaling_factor),
        num_warps=4,
        num_stages=1,
    )
    return weights, indices
