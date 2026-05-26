# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Bit-exact correctness tests for ``aiter.flydsl_moe_fused_gate_v4``.

Reference is the vendored torch port of sglang PR #23608's
``biased_topk_impl`` (``deepseek_v4_topk.py:157``) +
``MoEFusedGateKernel::run`` (``moe_fused_gate.cuh:282``) semantics.  The
torch reference lives next to the kernel under
``aiter/ops/flydsl/v4_routing.py::_torch_fused_gate_v4`` so this test does
not need an sglang import.

The kernel uses an iterative argmax with a min-id tie-break (CUDA line
221).  ``torch.topk`` does NOT match that tie rule, so even the torch
reference uses a stable sort -- guaranteeing both the reference and the
kernel pick the same id when scores collide.

Every shape tuple is run for both ``backend="triton"`` (the immediate
fast path) and ``backend="auto"`` (which currently still resolves to
Triton until the FlyDSL kernel is signed off).
"""

from __future__ import annotations

import itertools

import pytest
import torch

import aiter
from aiter.ops.flydsl.v4_routing import _torch_fused_gate_v4


def _make_inputs(N, E, *, seed=0, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    # router logits in a realistic-ish range (post-Linear).
    x = torch.randn(N, E, generator=g, device=device, dtype=torch.float32) * 0.5
    # bias has small magnitude (V4 spec: per-expert correction bias).
    b = torch.randn(E, generator=g, device=device, dtype=torch.float32) * 0.05
    return x, b


# Shape grid kept reasonable for CI; full sweep is in bench_flydsl_v4_routing.py.
SHAPES = list(
    itertools.product(
        [1, 16, 256, 1024],         # N
        [128, 257, 512],            # E (V4-Flash uses 257 = 256 routed + 1 shared bias slot)
        [(8, 1), (8, 0), (6, 0)],   # (topk, num_fused_shared_experts)
        ["sigmoid", "sqrtsoftplus"],
    )
)
BACKENDS = ["triton"]  # add "flydsl" once the MLIR kernel lands and compiles


@pytest.mark.parametrize("N,E,tk_pair,scoring", SHAPES)
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("apply_rsf_out", [False, True])
@pytest.mark.parametrize("backend", BACKENDS)
def test_flydsl_moe_fused_gate_v4_correctness(
    N, E, tk_pair, scoring, renormalize, apply_rsf_out, backend
):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA/HIP device")

    topk, num_shared = tk_pair
    rsf = 2.5

    x, b = _make_inputs(N, E)

    ref_w, ref_i = _torch_fused_gate_v4(
        x, b,
        topk=topk,
        scoring_func=scoring,
        num_fused_shared_experts=num_shared,
        renormalize=renormalize,
        routed_scaling_factor=rsf,
        apply_routed_scaling_factor_on_output=apply_rsf_out,
    )

    out_w, out_i = aiter.flydsl_moe_fused_gate_v4(
        x, b,
        topk=topk,
        scoring_func=scoring,
        num_fused_shared_experts=num_shared,
        renormalize=renormalize,
        routed_scaling_factor=rsf,
        apply_routed_scaling_factor_on_output=apply_rsf_out,
        backend=backend,
    )

    # Indices: routed slots must match exactly (tie rule = min-id, both impls agree).
    K_routed = topk - num_shared
    assert torch.equal(out_i[:, :K_routed], ref_i[:, :K_routed]), (
        f"routed indices mismatch at N={N} E={E} topk={topk} shared={num_shared} "
        f"scoring={scoring}"
    )
    # Shared-expert slots: ids must be E..E+num_shared-1 in order.
    if num_shared > 0:
        expected_shared = torch.arange(
            E, E + num_shared, dtype=torch.int32, device=x.device
        ).unsqueeze(0).expand(N, -1)
        assert torch.equal(out_i[:, K_routed:], expected_shared)

    # Weights: fp32 -> fp32, expect tight tolerance (small drift from
    # accumulator order is possible; 1e-5 is comfortable for these magnitudes).
    torch.testing.assert_close(out_w, ref_w, atol=1e-5, rtol=1e-5)


def test_v4_flash_realistic_shape():
    """Smoke test on a representative V4-Flash decode shape."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA/HIP device")
    # V4-Flash: E=257 (256 routed + 1 fused shared), topk=8 (7 routed + 1 shared).
    N = 32
    E = 257
    x, b = _make_inputs(N, E, seed=7)

    out_w, out_i = aiter.flydsl_moe_fused_gate_v4(
        x, b,
        topk=8,
        scoring_func="sigmoid",
        num_fused_shared_experts=1,
        renormalize=True,
        routed_scaling_factor=2.5,
        backend="auto",
    )
    ref_w, ref_i = _torch_fused_gate_v4(
        x, b,
        topk=8,
        scoring_func="sigmoid",
        num_fused_shared_experts=1,
        renormalize=True,
        routed_scaling_factor=2.5,
        apply_routed_scaling_factor_on_output=False,
    )
    assert torch.equal(out_i[:, :7], ref_i[:, :7])
    torch.testing.assert_close(out_w, ref_w, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
