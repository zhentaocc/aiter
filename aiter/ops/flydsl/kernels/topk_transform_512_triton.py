# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Triton implementation of DeepSeek V4 CSA top-512 + page-table transform.

Production fast-path for ``flydsl_topk_transform_512`` while the FlyDSL
MLIR-builder kernel in ``topk_transform_512.py`` is being tuned.

CUDA reference: ``python/sglang/jit_kernel/csrc/deepseek_v4/topk.cuh``
(``topk_512_transform`` -> ``radix_topk`` 4-stage 8-bit radix).

This Triton port collapses the 4-stage radix into a single per-row block
that:
  1. masks scores past ``seq_lens[b]`` to -inf;
  2. selects top-512 via tournament reductions in registers;
  3. transforms each raw position to a paged index via the per-row page table.

For rows where ``seq_len <= 512``, the kernel emits sequential indices padded
with -1, exactly matching the CUDA ``naive_transform`` branch (line 47).

Performance-wise, fusing mask + topk + page transform into one launch beats
the torch reference (which does ``masked.clone()`` + ``torch.topk`` +
``torch.gather`` as three separate kernels with full-tensor materialisation).
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


_TOPK = 512
# Triton requires globals accessed inside @jit'ed kernels to be wrapped:
_TOPK_CONST = tl.constexpr(_TOPK)


@triton.jit
def _topk_transform_512_kernel_radix(
    scores_ptr,           # [B, S] fp32 (strided)
    seq_lens_ptr,         # [B] i32
    page_table_ptr,       # [B, P] i32 (strided)
    out_page_idx_ptr,     # [B, 512] i32
    out_raw_idx_ptr,      # [B, 512] i32 OR a dummy ptr if not requested
    HAS_RAW: tl.constexpr,
    B,
    score_stride,
    page_table_stride,
    page_bits,            # int (host-resolved log2(page_size))
    BLOCK_S: tl.constexpr,    # power-of-two >= max_seq_len bucket
):
    """
    One program per batch row.  BLOCK_S covers the full row (V4 max seq_len
    per CSA chunk fits comfortably under the 64K/128K ranges we target).
    """
    b = tl.program_id(0)
    seq_len = tl.load(seq_lens_ptr + b).to(tl.int32)

    offs = tl.arange(0, BLOCK_S)
    valid = offs < seq_len

    # Short-sequence path: emit sequential 0..seq_len-1, pad with -1.
    short_path = seq_len <= _TOPK_CONST

    # Always materialise the masked scores (Triton's branchy behaviour
    # is rough; one extra load is cheaper than diverging programs).
    NEG_INF = float("-inf")
    s_row = tl.load(scores_ptr + b * score_stride + offs, mask=valid, other=NEG_INF)

    # ---------------------------------------------------------------
    # Top-512 selection.
    #
    # We use a register-resident tournament: in each of K=512 iterations,
    # find argmax over the row, store its raw index, then mask it to -inf.
    # That is O(BLOCK_S * K) work per row -- for BLOCK_S=4096 and K=512
    # this is 2M ops which is well under microsecond on MI355X.
    #
    # For larger BLOCK_S (>=16K) this loop dominates; the FlyDSL port
    # in topk_transform_512.py implements the proper 4-stage radix that
    # is O(BLOCK_S * 4 + topk) instead.
    # ---------------------------------------------------------------
    keys = s_row  # mutable copy
    # We'll store raw indices into LDS-like Triton-internal storage
    # and write them out in a final pass.  Simplest: store directly into
    # global out_raw_idx_ptr and (optionally) out_page_idx_ptr.

    if not short_path:
        for k in tl.static_range(0, 512):
            max_val = tl.max(keys, axis=0)
            is_max = (keys == max_val) & valid
            # Tie-break: smallest raw index wins (matches CUDA radix's
            # natural ordering after stable atomicAdd).
            id_or_big = tl.where(is_max, offs, BLOCK_S + 1)
            winner = tl.min(id_or_big, axis=0)
            # Compute paged index inline.
            page_idx = winner >> page_bits
            page_mask = (1 << page_bits) - 1
            offset_in_page = winner & page_mask
            # Clamp page_idx for the gather (we still write -1 for invalid winners).
            phys = tl.load(
                page_table_ptr + b * page_table_stride + page_idx,
                mask=(winner >= 0),
                other=0,
            )
            paged = (phys << page_bits) | offset_in_page
            paged = tl.where(winner >= 0, paged, -1)

            tl.store(out_page_idx_ptr + b * 512 + k, paged.to(tl.int32))
            if HAS_RAW:
                tl.store(out_raw_idx_ptr + b * 512 + k, winner.to(tl.int32))

            # Mask winner.
            keys = tl.where(offs == winner, NEG_INF, keys)
    else:
        # Short path: sequential indices, padded with -1.
        for k in tl.static_range(0, 512):
            in_range = k < seq_len
            raw = tl.where(in_range, k, -1)
            page_idx = raw >> page_bits
            page_mask = (1 << page_bits) - 1
            offset_in_page = raw & page_mask
            phys = tl.load(
                page_table_ptr + b * page_table_stride + page_idx,
                mask=in_range,
                other=0,
            )
            paged = (phys << page_bits) | offset_in_page
            paged = tl.where(in_range, paged, -1)
            tl.store(out_page_idx_ptr + b * 512 + k, paged.to(tl.int32))
            if HAS_RAW:
                tl.store(out_raw_idx_ptr + b * 512 + k, raw.to(tl.int32))


def _block_s_for(max_seq_len: int) -> int:
    """Round max_seq_len up to a Triton-friendly power of 2, capped at 65536."""
    bs = 512
    while bs < max_seq_len and bs < 65536:
        bs *= 2
    return min(bs, 65536)


def triton_topk_transform_512(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
    out_raw_indices: Optional[torch.Tensor] = None,
) -> None:
    """See ``flydsl_topk_transform_512`` for the contract."""
    assert out_page_indices.shape[1] == _TOPK
    B, max_seq_len = scores.shape
    assert seq_lens.shape == (B,)
    assert page_tables.shape[0] == B

    if max_seq_len > 65536:
        # Out-of-budget for the register-tournament approach; fall back to
        # the FlyDSL radix kernel or torch.  The wrapper's "auto" mode
        # handles this -- but here we just raise so the wrapper's
        # fallback kicks in.
        raise RuntimeError(
            f"triton_topk_transform_512: max_seq_len={max_seq_len} exceeds 65536; "
            "use backend='flydsl' (true radix) or backend='torch'."
        )

    page_bits = (page_size - 1).bit_length() if page_size > 1 else 0
    BLOCK_S = _block_s_for(max_seq_len)

    # Triton requires pointer args; supply a real dummy tensor for raw_indices
    # when the caller didn't ask for it.
    HAS_RAW = out_raw_indices is not None
    raw_arg = out_raw_indices if HAS_RAW else out_page_indices  # ptr unused under HAS_RAW=False

    grid = (B,)
    _topk_transform_512_kernel_radix[grid](
        scores,
        seq_lens,
        page_tables,
        out_page_indices,
        raw_arg,
        HAS_RAW=HAS_RAW,
        B=B,
        score_stride=scores.stride(0),
        page_table_stride=page_tables.stride(0),
        page_bits=page_bits,
        BLOCK_S=BLOCK_S,
        num_warps=8,
        num_stages=1,
    )
