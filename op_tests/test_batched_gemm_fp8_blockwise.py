# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Correctness tests for ``aiter.batched_gemm_fp8_blockwise``.

Reference oracle: torch dequant + ``torch.bmm`` (the
``_torch_batched_gemm_fp8_blockwise`` function in the wrapper module).
That oracle is itself the verbatim semantics of DeepGEMM's
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
from aiter.ops.batched_gemm_op_fp8_blockwise import _torch_batched_gemm_fp8_blockwise


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


@pytest.mark.parametrize("B,M,N,K", SHAPES)
@pytest.mark.parametrize("backend", ["triton"])
def test_batched_gemm_fp8_blockwise_correctness(B, M, N, K, backend):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA/HIP device")

    A, W, A_scale, W_scale = _make_inputs(B, M, N, K, seed=B * M + N + K)

    ref = _torch_batched_gemm_fp8_blockwise(A, W, A_scale, W_scale)

    out = aiter.batched_gemm_fp8_blockwise(
        A, W, A_scale, W_scale, backend=backend,
    )

    assert out.shape == (B, M, N) and out.dtype == torch.bfloat16
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


def test_wo_a_einsum_layout():
    """
    Smoke-test the wo_a "bhr,hdr->bhd" call shape, which is the actual
    DeepSeek V4 use case.  We transpose to standard BMK/BNK layout, run our
    kernel, then transpose back -- and verify the result matches the same
    transformation applied to the torch oracle.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA/HIP device")
    T, G, D = 8, 4, 256
    R = 512
    g = torch.Generator(device="cuda").manual_seed(42)
    o_fp8 = _make_fp8((T, G, D), generator=g)
    o_s = (torch.rand(T, G, D // 128, generator=g, device="cuda") * 0.05) + 0.005
    w_fp8 = _make_fp8((G, R, D), generator=g)
    w_s = (torch.rand(G, R // 128, D // 128, generator=g, device="cuda") * 0.05) + 0.005

    # As a caller would do for wo_a:
    A = o_fp8.transpose(0, 1).contiguous()    # [G, T, D]
    A_scale = o_s.transpose(0, 1).contiguous()  # [G, T, D//128]
    out_btn = aiter.batched_gemm_fp8_blockwise(A, w_fp8, A_scale, w_s, backend="triton")
    out = out_btn.transpose(0, 1).contiguous()  # [T, G, R]

    # Reference: same transform on the torch path.
    ref_btn = _torch_batched_gemm_fp8_blockwise(A, w_fp8, A_scale, w_s)
    ref = ref_btn.transpose(0, 1).contiguous()
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-xvs", "-m", "not slow"])
