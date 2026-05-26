# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Correctness tests for ``aiter.flydsl_topk_transform_512``.

Reference is the vendored ``topk_transform_512_pytorch_vectorized`` from
sglang PR #23608 (``indexer.py:210``); see
``aiter/ops/flydsl/v4_routing.py::_torch_topk_transform_512``.

We use *set-equality* on the raw indices because the radix kernel breaks
score ties differently from ``torch.topk`` (which is itself non-
deterministic on ties).  After confirming the index sets match, we check
that the *gathered scores* are identical -- which is the actual property
the downstream attention kernel cares about.

For short rows (``seq_len <= 512``) the contract is exact: sequential
indices padded with -1.
"""

from __future__ import annotations

import itertools

import pytest
import torch

import aiter
from aiter.ops.flydsl.v4_routing import _torch_topk_transform_512


_TOPK = 512


def _make_inputs(B, S, page_size, seed=0, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    scores = torch.randn(B, S, generator=g, device=device, dtype=torch.float32)
    # seq_lens: a mix of short, near-cap, and long rows.
    seq_lens = torch.randint(1, S + 1, (B,), generator=g, device=device, dtype=torch.int32)
    num_pages = (S + page_size - 1) // page_size
    page_tables = torch.randint(
        0, 1 << 20, (B, num_pages), generator=g, device=device, dtype=torch.int32
    )
    return scores, seq_lens, page_tables


SHAPES = list(
    itertools.product(
        [1, 16, 64],            # B
        [1024, 4096, 16384],    # max_seq_len
        [128, 256],             # page_size
    )
)
BACKENDS = ["triton"]  # add "flydsl" once the MLIR kernel lands


def _set_equal_per_row(a: torch.Tensor, b: torch.Tensor, valid_mask: torch.Tensor) -> bool:
    """Per-row set equality on the *valid* slots (raw_idx >= 0)."""
    assert a.shape == b.shape
    for i in range(a.shape[0]):
        va = a[i][valid_mask[i]].tolist()
        vb = b[i][valid_mask[i]].tolist()
        if set(va) != set(vb):
            return False
    return True


@pytest.mark.parametrize("B,S,page_size", SHAPES)
@pytest.mark.parametrize("with_raw", [True, False])
@pytest.mark.parametrize("backend", BACKENDS)
def test_topk_transform_512_correctness(B, S, page_size, with_raw, backend):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA/HIP device")

    scores, seq_lens, page_tables = _make_inputs(B, S, page_size, seed=B * S + page_size)

    # Reference.
    ref_pages = torch.empty((B, _TOPK), dtype=torch.int32, device=scores.device)
    ref_raw = torch.empty((B, _TOPK), dtype=torch.int32, device=scores.device) if with_raw else None
    _torch_topk_transform_512(
        scores, seq_lens, page_tables, ref_pages, page_size, ref_raw,
    )

    # Implementation under test.
    out_pages = torch.empty((B, _TOPK), dtype=torch.int32, device=scores.device)
    out_raw = torch.empty((B, _TOPK), dtype=torch.int32, device=scores.device) if with_raw else None
    aiter.flydsl_topk_transform_512(
        scores, seq_lens, page_tables, out_pages, page_size, out_raw, backend=backend,
    )

    # Short rows (seq_len <= TOPK) must match exactly per the CUDA naive_transform contract.
    short_rows = (seq_lens <= _TOPK).cpu().tolist()

    for i, is_short in enumerate(short_rows):
        if is_short:
            assert torch.equal(out_pages[i], ref_pages[i]), (
                f"short-row mismatch at B={B} S={S} ps={page_size} row={i}"
            )
            if with_raw:
                assert torch.equal(out_raw[i], ref_raw[i])

    # Long rows: index sets must be equal on the valid (>=0) slots, and
    # gathered scores must match (the property that matters downstream).
    if with_raw:
        ref_valid = ref_raw >= 0
        out_valid = out_raw >= 0
        # On long rows both should fill all 512.
        long_idx = [i for i, s in enumerate(short_rows) if not s]
        if long_idx:
            ref_long_raw = ref_raw[long_idx]
            out_long_raw = out_raw[long_idx]
            ref_long_valid = ref_valid[long_idx]
            out_long_valid = out_valid[long_idx]
            assert _set_equal_per_row(ref_long_raw, out_long_raw, ref_long_valid)
            assert ref_long_valid.equal(out_long_valid)

            # Gathered scores must match (set the invalid slots to a fixed value
            # so we don't compare junk).
            ref_gather = scores[long_idx].gather(
                1, ref_long_raw.clamp(min=0).long()
            )
            out_gather = scores[long_idx].gather(
                1, out_long_raw.clamp(min=0).long()
            )
            ref_sorted = ref_gather.sort(dim=-1).values
            out_sorted = out_gather.sort(dim=-1).values
            torch.testing.assert_close(out_sorted, ref_sorted, atol=0.0, rtol=0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
