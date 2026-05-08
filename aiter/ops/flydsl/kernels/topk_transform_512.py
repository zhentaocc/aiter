# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
FlyDSL (MLIR-builder) port of DeepSeek V4 CSA top-512 + page-table transform.

CUDA reference: ``python/sglang/jit_kernel/csrc/deepseek_v4/topk.cuh``
(``topk_512_transform`` -> ``radix_topk`` 4-stage 8-bit radix).

Status: SKELETON.  See file-level docstring of ``fused_gate_v4.py`` for the
explanation of why this is shipped as a skeleton + Triton fallback.

When promoted to production, the implementation must mirror the CUDA radix
faithfully because that algorithm is what makes the kernel sub-microsecond
on long sequences (the Triton register-tournament fallback is O(BLOCK_S * K)
and degrades linearly in BLOCK_S; the radix is O(BLOCK_S * 4 + topk)).

Design (from ``/home/zhenchen/.claude/plans/velvet-painting-brook.md``):

  * **Workgroup**: 1 batch row per WG, ``block_size=512`` threads = 8 waves
    of 64 lanes.  LDS budget 64 KB (the CUDA ``kSMEM`` constant) holds the
    two double-buffered ``int32`` overflow-index arrays plus the histogram
    (256 u32) and control vars; comfortably fits in MI355X's 160 KB / CU.

  * **Stage 1 (coarse 8-bit)**:
        bin = convert_to_uint8(score)   # monotonic key (line 29 of topk.cuh)
        atomicAdd(s_histogram[bin], 1)  # avoid LDS atomics by per-wave
                                        # private histograms then a wave-0
                                        # merge (see "Risks" in the plan)
        cumsum -> threshold_bin
        emit indices above threshold; stash threshold-bin indices into LDS.

  * **Stage 2 (4 refinement rounds, 8-bit each)**:
        repeat: histogram over byte (24 - round*8); narrow until 512 selected.

  * **Page transform** (final):
        page_indices[i] = (page_table[idx[i] >> page_bits] << page_bits)
                          | (idx[i] & ((1 << page_bits) - 1))

Compile cache key: ``(score_stride_bucket, page_table_stride_bucket,
page_bits, has_raw_indices)``.

Until the FlyDSL implementation is sign-off, this file raises
NotImplementedError so the wrapper falls back to the Triton port.
"""

from __future__ import annotations

import functools
from typing import Optional

import torch

try:
    import flydsl.compiler as flyc  # noqa: F401
    _FLYDSL_OK = True
except ImportError as _e:
    _FLYDSL_OK = False
    _FLYDSL_IMPORT_ERR = _e


def _bucket_stride(s: int) -> int:
    for b in (1024, 4096, 16384, 65536, 262144):
        if s <= b:
            return b
    return ((s + 65535) // 65536) * 65536


@functools.lru_cache(maxsize=None)
def _build_module(
    score_stride_bucket: int,
    page_table_stride_bucket: int,
    page_bits: int,
    has_raw: bool,
):
    if not _FLYDSL_OK:
        raise NotImplementedError(
            f"FlyDSL not importable: {_FLYDSL_IMPORT_ERR}"
        ) from _FLYDSL_IMPORT_ERR
    raise NotImplementedError(
        "FlyDSL topk_transform_512 kernel module not yet implemented; "
        "the Triton port at topk_transform_512_triton.py is the current fast path. "
        f"Requested config: score_stride_bucket={score_stride_bucket} "
        f"page_table_stride_bucket={page_table_stride_bucket} "
        f"page_bits={page_bits} has_raw={has_raw}."
    )


def flydsl_kernel_topk_transform_512(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
    out_raw_indices: Optional[torch.Tensor] = None,
) -> None:
    """Entry point.  See ``v4_routing.flydsl_topk_transform_512`` for the contract."""
    page_bits = (page_size - 1).bit_length() if page_size > 1 else 0
    module = _build_module(
        score_stride_bucket=_bucket_stride(scores.stride(0)),
        page_table_stride_bucket=_bucket_stride(page_tables.stride(0)),
        page_bits=page_bits,
        has_raw=(out_raw_indices is not None),
    )
    module(
        scores,
        seq_lens,
        page_tables,
        out_page_indices,
        out_raw_indices,
        page_bits,
    )
