# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Triton implementation of FP8 block-wise batched GEMM, matching the contract
of DeepGEMM's ``fp8_einsum("bmk,bnk->bmn", ..., recipe=(1, 1, 128))`` /
``fp8_einsum("bhr,hdr->bhd", ..., recipe=(1, 1, 128))`` (which is the same
operator, just renamed indices).

Quantization recipe (matches DeepSeek V4 ``wo_a`` checkpoint format):

  * **Activation A** ``[B, M, K]`` ``fp8_e4m3fn`` with scale
    ``A_scale[B, M, K // 128]`` ``fp32`` -- *per-row, per-128-column-block*
    (block_m=1, block_k=128).
  * **Weight W** ``[B, N, K]`` ``fp8_e4m3fn`` with scale
    ``W_scale[B, N // 128, K // 128]`` ``fp32`` -- *per 128x128 (N, K) block*
    (block_n=128, block_k=128).
  * **Output** ``[B, M, N]`` ``bfloat16``.

Constraints:

  * ``K`` and ``N`` must be multiples of 128.
  * ``B``, ``M``, ``N``, ``K`` are runtime values; ``BLOCK_M``, ``BLOCK_N``,
    ``BLOCK_K`` are compile-time constexprs.
  * ``BLOCK_K = 128`` is fixed (= scale block) so that scale loads happen
    once per K-tile.
  * ``BLOCK_N`` must divide 128 (or vice versa); we keep ``BLOCK_N = 128``
    so the W-scale lookup is a single load per K-tile.

This kernel intentionally does NOT support per-token-group activations on
the M-axis with block_m > 1; the wo_a recipe uses block_m=1.  If a future
caller needs other recipes, add a new compile-time switch.
"""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl


_BLOCK_K = 128  # equal to the scale-block size on K
_BLOCK_N = 128  # equal to the scale-block size on N (single W-scale load per K-tile)


@triton.jit
def _batched_gemm_fp8_blockwise_kernel(
    A_ptr,            # [B, M, K] fp8_e4m3fn
    W_ptr,            # [B, N, K] fp8_e4m3fn
    A_scale_ptr,      # [B, M, K_div_128] fp32
    W_scale_ptr,      # [B, N_div_128, K_div_128] fp32
    Out_ptr,          # [B, M, N] bf16
    M, N, K,
    stride_a_b, stride_a_m, stride_a_k,
    stride_w_b, stride_w_n, stride_w_k,
    stride_as_b, stride_as_m, stride_as_kg,
    stride_ws_b, stride_ws_ng, stride_ws_kg,
    stride_o_b, stride_o_m, stride_o_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    One workgroup computes a [BLOCK_M, BLOCK_N] tile of Out[b, :, :].

    Grid: (cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N), B).
    """
    pid_b = tl.program_id(1)
    pid = tl.program_id(0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # L2-friendly group-major scheduling (standard Triton pattern).
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m_actual = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m_actual)
    pid_n = (pid % num_pid_in_group) // group_size_m_actual

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Bounds masks for M (N is a multiple of BLOCK_N=128 by contract).
    m_mask = offs_m < M
    n_mask = offs_n < N

    # Per-row A-scale base: A_scale[pid_b, offs_m, k_block].
    # Per-block W-scale base: W_scale[pid_b, pid_n, k_block]   (since BLOCK_N == 128).
    a_scale_row_ptr = A_scale_ptr + pid_b * stride_as_b + offs_m * stride_as_m
    w_scale_row_ptr = W_scale_ptr + pid_b * stride_ws_b + pid_n * stride_ws_ng

    # Initialise the FP32 accumulator.  Use FMA in the dequant-then-multiply step.
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Tile bases.
    a_base = A_ptr + pid_b * stride_a_b + offs_m[:, None] * stride_a_m
    w_base = W_ptr + pid_b * stride_w_b + offs_n[:, None] * stride_w_n

    num_k_tiles = tl.cdiv(K, BLOCK_K)

    for k_tile in range(0, num_k_tiles):
        k_start = k_tile * BLOCK_K
        k_offs = k_start + offs_k
        k_mask = k_offs < K

        # Load A tile [BLOCK_M, BLOCK_K] fp8_e4m3fn -> fp32.
        a_ptrs = a_base + k_offs[None, :] * stride_a_k
        a_fp8 = tl.load(
            a_ptrs,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        # Load W tile [BLOCK_N, BLOCK_K] fp8_e4m3fn -> fp32.
        w_ptrs = w_base + k_offs[None, :] * stride_w_k
        w_fp8 = tl.load(
            w_ptrs,
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0,
        )

        # Dequantize using per-row A-scale and per-block W-scale.
        # A_scale[b, m, k_tile] is a scalar per row -> broadcast across K-tile.
        a_scale = tl.load(
            a_scale_row_ptr + k_tile * stride_as_kg,
            mask=m_mask,
            other=0.0,
        )  # [BLOCK_M]
        # W_scale[b, pid_n, k_tile] is a single scalar per (N-tile, K-tile).
        w_scale = tl.load(w_scale_row_ptr + k_tile * stride_ws_kg)  # scalar

        # Cast fp8 -> fp32 for the multiply-accumulate.  We do not use a
        # native fp8 dot here for two reasons: (a) the per-row activation
        # scale and per-block weight scale need to be applied *before* the
        # accumulation to stay numerically equivalent to DeepGEMM's
        # fp8_einsum, and (b) the gfx950 fp8 MFMA path applies a single
        # per-tile scale, which doesn't match recipe=(1, 1, 128).
        a_f32 = a_fp8.to(tl.float32) * a_scale[:, None]                  # [BM, BK]
        w_f32 = w_fp8.to(tl.float32) * w_scale                            # [BN, BK]

        # Cast back to bf16 for tl.dot to dispatch to MFMA bf16 instructions.
        a_bf16 = a_f32.to(tl.bfloat16)
        w_bf16 = w_f32.to(tl.bfloat16)

        # acc += A @ W.T  (W is [BN, BK]; contract on K).
        acc += tl.dot(a_bf16, tl.trans(w_bf16))

    # Store [BM, BN] in bf16.
    out_ptrs = (
        Out_ptr
        + pid_b * stride_o_b
        + offs_m[:, None] * stride_o_m
        + offs_n[None, :] * stride_o_n
    )
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=m_mask[:, None] & n_mask[None, :])


# ----------------------------------------------------------------------------
# Heuristic tile picker.  Hand-tuned starting point for MI355X (gfx950);
# replace with autotune CSV like ck_batched_gemm_a8w8 once the CK kernel lands.
# ----------------------------------------------------------------------------


def _pick_tile(M: int, N: int, K: int) -> Tuple[int, int, int, int]:
    """Returns (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M)."""
    if M <= 16:
        return (16, _BLOCK_N, _BLOCK_K, 4)   # decode: tall-skinny
    if M <= 64:
        return (32, _BLOCK_N, _BLOCK_K, 4)
    if M <= 256:
        return (64, _BLOCK_N, _BLOCK_K, 8)
    return (128, _BLOCK_N, _BLOCK_K, 8)


def triton_batched_gemm_fp8_blockwise(
    A: torch.Tensor,
    W: torch.Tensor,
    A_scale: torch.Tensor,
    W_scale: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute ``Out[b, m, n] = sum_k A[b, m, k] * W[b, n, k]`` with
    block-wise FP8 dequantisation matching DeepSeek V4's ``wo_a`` recipe.

    See the module docstring for the precise recipe contract.
    """
    assert A.dtype == torch.float8_e4m3fn, f"A dtype {A.dtype}, expected fp8_e4m3fn"
    assert W.dtype == torch.float8_e4m3fn, f"W dtype {W.dtype}, expected fp8_e4m3fn"
    assert A_scale.dtype == torch.float32, f"A_scale dtype {A_scale.dtype}"
    assert W_scale.dtype == torch.float32, f"W_scale dtype {W_scale.dtype}"
    assert A.dim() == 3 and W.dim() == 3
    B, M, K = A.shape
    Bw, N, Kw = W.shape
    assert B == Bw and K == Kw, f"A {A.shape} vs W {W.shape}"
    assert K % _BLOCK_K == 0, f"K={K} not multiple of {_BLOCK_K}"
    assert N % _BLOCK_N == 0, f"N={N} not multiple of {_BLOCK_N}"
    K_g = K // _BLOCK_K
    N_g = N // _BLOCK_N
    assert A_scale.shape == (B, M, K_g), f"A_scale {A_scale.shape} vs expected {(B, M, K_g)}"
    assert W_scale.shape == (B, N_g, K_g), f"W_scale {W_scale.shape} vs expected {(B, N_g, K_g)}"

    if out is None:
        out = torch.empty((B, M, N), dtype=torch.bfloat16, device=A.device)
    else:
        assert out.shape == (B, M, N) and out.dtype == torch.bfloat16

    BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = _pick_tile(M, N, K)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), B)

    _batched_gemm_fp8_blockwise_kernel[grid](
        A, W, A_scale, W_scale, out,
        M, N, K,
        A.stride(0), A.stride(1), A.stride(2),
        W.stride(0), W.stride(1), W.stride(2),
        A_scale.stride(0), A_scale.stride(1), A_scale.stride(2),
        W_scale.stride(0), W_scale.stride(1), W_scale.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_M,
        num_warps=4,
        num_stages=2,
    )
    return out
