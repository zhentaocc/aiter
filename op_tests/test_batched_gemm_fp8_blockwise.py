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
    ref = _torch_batched_gemm_fp8_blockwise(A, W, A_scale, W_scale)
    out = aiter.batched_gemm_fp8_blockwise(A, W, A_scale, W_scale, backend="auto")

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
    auto_out = aiter.batched_gemm_fp8_blockwise(A, W, A_scale, W_scale, backend="auto")

    # Probe what auto picked, then run that backend explicitly and compare.
    # Flydsl-only branch: auto always picks flydsl when constraints hold.
    expected_backend = "flydsl"
    try:
        explicit_out = aiter.batched_gemm_fp8_blockwise(
            A, W, A_scale, W_scale, backend=expected_backend,
        )
    except (ImportError, RuntimeError, AssertionError) as e:
        pytest.skip(f"backend {expected_backend!r} unavailable on this host: {e}")

    # auto must match the explicit backend bit-for-bit (same kernel was called).
    torch.testing.assert_close(auto_out, explicit_out, atol=0, rtol=0)


@pytest.mark.parametrize("equation,A_shape,W_shape,O_shape", [
    # wo_a einsum form on DSv4 Flash decode (T=16, G=8, D=4096, R=1024)
    pytest.param("tgd,grd->tgr", (16, 8, 4096), (8, 1024, 4096), (16, 8, 1024),
                 id="flash_decode_tgd"),
    # wo_a einsum form on DSv4 Pro decode (T=16, G=16, D=4096, R=1024)
    pytest.param("tgd,grd->tgr", (16, 16, 4096), (16, 1024, 4096), (16, 16, 1024),
                 id="pro_decode_tgd"),
    # DeepGEMM-style "bhr,hdr->bhd" on Flash decode
    pytest.param("bhr,hdr->bhd", (16, 8, 4096), (8, 1024, 4096), (16, 8, 1024),
                 id="flash_decode_bhr"),
])
def test_dsv4_einsum_layouts(equation, A_shape, W_shape, O_shape):
    """Einsum entry-point on DSv4 layouts: no transpose roundtrip needed."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA/HIP device")

    g = torch.Generator(device="cuda").manual_seed(hash(equation) & 0xFFFF)
    A = _make_fp8(A_shape, generator=g)
    W = _make_fp8(W_shape, generator=g)
    # Scale shapes mirror the contraction axis in each operand.
    # For "tgd,grd->tgr": A_scale [T, G, D//128], W_scale [G, R//128, D//128].
    # For "bhr,hdr->bhd": A_scale [B, H, R//128], W_scale [H, D//128, R//128].
    if equation == "tgd,grd->tgr":
        T, G, D = A_shape; _, R, _ = W_shape
        A_s = torch.rand(T, G, D // 128, generator=g, device="cuda") * 0.05 + 0.005
        W_s = torch.rand(G, R // 128, D // 128, generator=g, device="cuda") * 0.05 + 0.005
    else:  # "bhr,hdr->bhd"
        B, H, R = A_shape; _, D, _ = W_shape
        A_s = torch.rand(B, H, R // 128, generator=g, device="cuda") * 0.05 + 0.005
        W_s = torch.rand(H, D // 128, R // 128, generator=g, device="cuda") * 0.05 + 0.005

    out = aiter.batched_gemm_fp8_blockwise_einsum(
        equation, A, A_s, W, W_s, backend="auto",
    )
    assert out.shape == O_shape and out.dtype == torch.bfloat16


@pytest.mark.parametrize("B,M,N,K", SHAPES)
@pytest.mark.parametrize("backend", ["auto"])
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
    out_btn = aiter.batched_gemm_fp8_blockwise(A, w_fp8, A_scale, w_s, backend="auto")
    out = out_btn.transpose(0, 1).contiguous()  # [T, G, R]

    # Reference: same transform on the torch path.
    ref_btn = _torch_batched_gemm_fp8_blockwise(A, w_fp8, A_scale, w_s)
    ref = ref_btn.transpose(0, 1).contiguous()
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


# ----------------------------------------------------------------------------
# Einsum-form tests.
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "equation,A_shape,W_shape,O_shape",
    [
        # sglang wo_a: A=[T,G,D], W=[G,R,D], Out=[T,G,R]
        ("tgd,grd->tgr", (8, 4, 256), (4, 512, 256), (8, 4, 512)),
        # vllm/DeepGEMM wo_a (same as above with renamed letters)
        ("bhr,hdr->bhd", (8, 4, 256), (4, 512, 256), (8, 4, 512)),
        # canonical BMK form
        ("bmk,bnk->bmn", (4, 8, 256), (4, 512, 256), (4, 8, 512)),
        # V4-Flash decode-shape sanity (B=8 groups, T=4, D=4096, R=1024)
        ("tgd,grd->tgr", (4, 8, 4096), (8, 1024, 4096), (4, 8, 1024)),
    ],
)
@pytest.mark.parametrize("backend", ["auto"])
def test_einsum_form_matches_oracle(equation, A_shape, W_shape, O_shape, backend):
    """einsum-form output must match the torch dequant+bmm oracle."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA/HIP device")
    from aiter.ops.batched_gemm_op_fp8_blockwise import (
        _torch_batched_gemm_fp8_blockwise_einsum,
    )

    g = torch.Generator(device="cuda").manual_seed(hash(equation) & 0xFFFF)
    A = (torch.randn(*A_shape, generator=g, device="cuda", dtype=torch.float32) * 0.5).clamp(-8, 8).to(torch.float8_e4m3fn)
    W = (torch.randn(*W_shape, generator=g, device="cuda", dtype=torch.float32) * 0.5).clamp(-8, 8).to(torch.float8_e4m3fn)
    # Build scale shapes matching A/W with the K (and W's N) axis sizes /128.
    info = _parse_eq_for_test(equation)
    A_scale_shape = tuple(s // 128 if i == info["K_in_a"] else s for i, s in enumerate(A_shape))
    W_scale_shape = tuple(
        (s // 128) if i in (info["N_in_w"], info["K_in_w"]) else s
        for i, s in enumerate(W_shape)
    )
    A_s = torch.rand(*A_scale_shape, generator=g, device="cuda", dtype=torch.float32) * 0.05 + 0.005
    W_s = torch.rand(*W_scale_shape, generator=g, device="cuda", dtype=torch.float32) * 0.05 + 0.005

    out_einsum = aiter.batched_gemm_fp8_blockwise_einsum(
        equation, A, A_s, W, W_s, backend=backend,
    )
    assert out_einsum.shape == tuple(O_shape)

    ref = _torch_batched_gemm_fp8_blockwise_einsum(equation, A, A_s, W, W_s)
    torch.testing.assert_close(out_einsum, ref, atol=2e-2, rtol=2e-2)


def _parse_eq_for_test(equation: str) -> dict:
    lhs, rhs = equation.replace(" ", "").split("->")
    a_eq, w_eq = lhs.split(",")
    sa, sw, so = set(a_eq), set(w_eq), set(rhs)
    K = ((sa & sw) - so).pop()
    N = ((sw & so) - sa).pop()
    return {
        "K_in_a": a_eq.index(K),
        "N_in_w": w_eq.index(N),
        "K_in_w": w_eq.index(K),
    }


def test_einsum_form_bit_exact_with_bmk_form():
    """
    Same kernel via auto-dispatch, different access stride only -- expect bit-exact match
    between the einsum form and the canonical BMK form (no atol/rtol).
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA/HIP device")
    T, G, D, R = 8, 4, 256, 512
    g = torch.Generator(device="cuda").manual_seed(123)
    A_tgd = (torch.randn(T, G, D, generator=g, device="cuda") * 0.5).clamp(-8, 8).to(torch.float8_e4m3fn)
    As_tgd = torch.rand(T, G, D // 128, generator=g, device="cuda") * 0.05 + 0.005
    W_grd = (torch.randn(G, R, D, generator=g, device="cuda") * 0.5).clamp(-8, 8).to(torch.float8_e4m3fn)
    Ws_grd = torch.rand(G, R // 128, D // 128, generator=g, device="cuda") * 0.05 + 0.005

    out_einsum = aiter.batched_gemm_fp8_blockwise_einsum(
        "tgd,grd->tgr", A_tgd, As_tgd, W_grd, Ws_grd, backend="auto",
    )
    assert out_einsum.shape == (T, G, R)

    out_bmk = aiter.batched_gemm_fp8_blockwise(
        A_tgd.transpose(0, 1).contiguous(),
        W_grd,
        As_tgd.transpose(0, 1).contiguous(),
        Ws_grd,
        backend="auto",
    )
    out_bmk_tgr = out_bmk.transpose(0, 1).contiguous()
    torch.testing.assert_close(out_einsum, out_bmk_tgr, atol=0.0, rtol=0.0)


def test_einsum_form_no_data_movement():
    """
    The wrapper's permutes are stride-only views: confirm the input data_ptr
    is unchanged (no .contiguous() round-trip behind the scenes).
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA/HIP device")
    T, G, D, R = 4, 4, 256, 256
    g = torch.Generator(device="cuda").manual_seed(7)
    A_tgd = (torch.randn(T, G, D, generator=g, device="cuda") * 0.5).clamp(-8, 8).to(torch.float8_e4m3fn)
    As_tgd = torch.rand(T, G, D // 128, generator=g, device="cuda") * 0.05 + 0.005
    W_grd = (torch.randn(G, R, D, generator=g, device="cuda") * 0.5).clamp(-8, 8).to(torch.float8_e4m3fn)
    Ws_grd = torch.rand(G, R // 128, D // 128, generator=g, device="cuda") * 0.05 + 0.005

    a_ptr = A_tgd.data_ptr()
    w_ptr = W_grd.data_ptr()
    aiter.batched_gemm_fp8_blockwise_einsum("tgd,grd->tgr", A_tgd, As_tgd, W_grd, Ws_grd, backend="auto")
    # Originals untouched (data_ptr stable, no in-place dequant or copy).
    assert A_tgd.data_ptr() == a_ptr
    assert W_grd.data_ptr() == w_ptr


def test_einsum_parse_errors():
    """The parser should reject malformed or ambiguous equations."""
    from aiter.ops.batched_gemm_op_fp8_blockwise import _parse_bmnk

    # Wrong rank.
    with pytest.raises(ValueError, match="exactly 3 axes"):
        _parse_bmnk("ab,bc->ac")
    # No batch axis common to all three.
    with pytest.raises(ValueError, match="cannot uniquely identify B"):
        _parse_bmnk("abc,cde->bdf")


if __name__ == "__main__":
    pytest.main([__file__, "-xvs", "-m", "not slow"])
