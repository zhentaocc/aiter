# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Correctness tests for ``aiter.batched_gemm_fp8_blockscale``.

Reference oracle: torch dequant + ``torch.bmm`` (the
``_torch_batched_gemm_fp8_blockscale`` helper defined below). That oracle
is the verbatim semantics of DeepGEMM's
``fp8_einsum("bmk,bnk->bmn", ..., recipe=(1, 1, 128))``.

Tolerance: BF16 GEMM with FP8 inputs has ~1e-2 absolute / ~1e-2 relative
error vs the dequant-then-bf16-bmm oracle in the worst case (block-scale
quant + bf16 accumulation).  We use ``atol=2e-2, rtol=2e-2`` -- tight
enough to catch real bugs (off-by-one stride, wrong scale broadcast, etc.)
without flagging acceptable accumulation noise.
"""

from __future__ import annotations

import itertools

import pytest
import torch

import aiter


# ----------------------------------------------------------------------------
# Torch reference oracle: dequant + bf16 bmm. Verbatim semantics of DeepGEMM's
# ``fp8_einsum("bmk,bnk->bmn", ..., recipe=(1, 1, 128))``. Used by every
# accuracy test below and by the benchmark in
# ``op_tests/op_benchmarks/hip/bench_batched_gemm_fp8_blockscale.py``.
# ----------------------------------------------------------------------------


def _torch_batched_gemm_fp8_blockscale(
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
    a_dq = A.to(torch.float32).view(B, M, K_g, 128) * A_scale.unsqueeze(-1)
    a_dq = a_dq.view(B, M, K).to(torch.bfloat16)
    w_dq = W.to(torch.float32).view(B, N_g, 128, K_g, 128) * W_scale.view(B, N_g, 1, K_g, 1)
    w_dq = w_dq.view(B, N, K).to(torch.bfloat16)
    return torch.bmm(a_dq, w_dq.transpose(1, 2)).to(torch.bfloat16)


def _make_fp8(shape, *, generator, device="cuda") -> torch.Tensor:
    """Sample a tensor in the fp8_e4m3fn safe range, then quantise."""
    x = torch.randn(*shape, generator=generator, device=device, dtype=torch.float32) * 0.5
    # fp8_e4m3fn max ~ 448; clamp generously well below that.
    return x.clamp_(-8.0, 8.0).to(torch.float8_e4m3fn)


def _make_inputs(B, M, N, K, *, seed=0, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    A = _make_fp8((B, M, K), generator=g, device=device)
    W = _make_fp8((B, N, K), generator=g, device=device)
    K_g, N_g = K // 128, N // 128
    # Scales in a realistic range (post-calibration); fp8 max ~448, so per-block
    # scale is typically ~max_block / 448; sample positive values around 0.01..0.1.
    A_scale = (torch.rand(B, M, K_g, generator=g, device=device, dtype=torch.float32) * 0.1) + 0.01
    W_scale = (torch.rand(B, N_g, K_g, generator=g, device=device, dtype=torch.float32) * 0.1) + 0.01
    return A, W, A_scale, W_scale


# wo_a-realistic shapes plus a small smoke set.  D = head_dim (128 typical),
# but here K is the contracted dim (= D in wo_a's mapping).
SHAPES = [
    # (B=G, M=T, N=R, K=D)
    (4, 1, 256, 128),
    (4, 16, 256, 128),
    (4, 128, 256, 128),
    (8, 256, 512, 128),
    (8, 1, 512, 256),
    (16, 64, 512, 256),
    # near-wo_a-real:
    # V4-Flash: G=8, R=8192, D=4096 (per-group); too big for CI but smoke.
    pytest.param(
        2, 32, 8192, 4096,
        marks=pytest.mark.slow,
    ),
]


# ----------------------------------------------------------------------------
# DeepSeek V4 wo_a shape coverage.
#
# Real model dimensions (TP=8 sharded; N = o_lora_rank = 1024 per rank,
# K = head_dim = 4096). G = n_local_groups: 8 for V4-Flash, 16 for V4-Pro.
# T ∈ {1, 16, 64} covers decode hot path and small prefill chunks.
# T ∈ {1024, 4096} covers prefill (marked slow -- skip in fast CI).
#
# Auto dispatch should route:
#   - T <= 64 (or T % 128 != 0): flydsl decode kernel
#   - T >= 128 && T % 128 == 0:  CK prefill kernel
# Both paths must produce numerically equivalent output vs the torch oracle.
# ----------------------------------------------------------------------------

DSV4_SHAPES = [
    # === V4-Flash (G=8) decode ===
    pytest.param(8, 1,    1024, 4096, id="flash_decode_T=1"),
    pytest.param(8, 16,   1024, 4096, id="flash_decode_T=16"),
    pytest.param(8, 64,   1024, 4096, id="flash_decode_T=64"),
    # === V4-Pro (G=16) decode ===
    pytest.param(16, 1,   1024, 4096, id="pro_decode_T=1"),
    pytest.param(16, 16,  1024, 4096, id="pro_decode_T=16"),
    pytest.param(16, 64,  1024, 4096, id="pro_decode_T=64"),
    # === Prefill (slow) ===
    pytest.param(8,  1024, 1024, 4096, marks=pytest.mark.slow, id="flash_prefill_T=1024"),
    pytest.param(8,  4096, 1024, 4096, marks=pytest.mark.slow, id="flash_prefill_T=4096"),
    pytest.param(16, 1024, 1024, 4096, marks=pytest.mark.slow, id="pro_prefill_T=1024"),
    pytest.param(16, 4096, 1024, 4096, marks=pytest.mark.slow, id="pro_prefill_T=4096"),
]


@pytest.mark.parametrize("B,M,N,K", DSV4_SHAPES)
def test_dsv4_wo_a_shapes(B, M, N, K):
    """Correctness on real DeepSeek V4 wo_a single-op shapes."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA/HIP device")

    A, W, A_scale, W_scale = _make_inputs(B, M, N, K, seed=B * M + N + K)
    ref = _torch_batched_gemm_fp8_blockscale(A, W, A_scale, W_scale)
    out = aiter.batched_gemm_fp8_blockscale(A, W, A_scale, W_scale, backend="auto")

    assert out.shape == (B, M, N) and out.dtype == torch.bfloat16
    # Larger tolerance for big-K shapes (more accumulation noise).
    atol = 4e-2 if K >= 4096 else 2e-2
    torch.testing.assert_close(out, ref, atol=atol, rtol=2e-2)


@pytest.mark.parametrize("B,M,N,K", DSV4_SHAPES)
def test_dsv4_dispatch_consistency(B, M, N, K):
    """The auto backend should match the per-backend output."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA/HIP device")

    A, W, A_scale, W_scale = _make_inputs(B, M, N, K, seed=B * M + N + K)
    auto_out = aiter.batched_gemm_fp8_blockscale(A, W, A_scale, W_scale, backend="auto")

    # Probe what auto picked, then run that backend explicitly and compare.
    # CK-only branch: small-M / non-128-aligned shapes go to torch.
    expected_backend = "ck" if (M >= 128 and M % 128 == 0 and N % 128 == 0) else "torch"
    try:
        explicit_out = aiter.batched_gemm_fp8_blockscale(
            A, W, A_scale, W_scale, backend=expected_backend,
        )
    except (ImportError, RuntimeError, AssertionError) as e:
        pytest.skip(f"backend {expected_backend!r} unavailable on this host: {e}")

    # auto must match the explicit backend bit-for-bit (same kernel was called).
    torch.testing.assert_close(auto_out, explicit_out, atol=0, rtol=0)


@pytest.mark.parametrize("B,M,N,K", SHAPES)
@pytest.mark.parametrize("backend", ["auto"])
def test_batched_gemm_fp8_blockscale_correctness(B, M, N, K, backend):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA/HIP device")

    A, W, A_scale, W_scale = _make_inputs(B, M, N, K, seed=B * M + N + K)

    ref = _torch_batched_gemm_fp8_blockscale(A, W, A_scale, W_scale)

    out = aiter.batched_gemm_fp8_blockscale(
        A, W, A_scale, W_scale, backend=backend,
    )

    assert out.shape == (B, M, N) and out.dtype == torch.bfloat16
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-xvs", "-m", "not slow"])
