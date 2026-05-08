# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ROCm/FlyDSL MegaMOE port contributors

"""Workspace byte layout for the FlyDSL MegaMOE port.

Direct Python port of ``deep_gemm/include/deep_gemm/layout/mega_moe.cuh``
from deepseek-ai/DeepGEMM (MIT). Field byte ordering, sizing math, and
accessor offsets remain bit-compatible with the C++ original so that
weights / SF tensors quantized by the upstream DeepGEMM Python utilities
can be consumed unmodified.

Field names match the C++ source verbatim — ``nvl_barrier_*`` is kept
even though on CDNA the link is xGMI, so a ``grep`` against the C++
source still hits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


# ── Constants from layout/mega_moe.cuh ────────────────────────────────────

#: Number of candidate per-expert M block sizes (must match C++).
NUM_CANDIDATE_BLOCK_MS: int = 7

#: Candidate BLOCK_M values picked at heuristic time.
CANDIDATE_BLOCK_M: Tuple[int, ...] = (8, 16, 32, 64, 96, 128, 192)

MAX_CANDIDATE_BLOCK_M: int = 192
MIN_CANDIDATE_BLOCK_M: int = 8

#: LCM of all candidate BLOCK_M values - per-rank token counts must
#: align to this so that any chosen BLOCK_M divides the worst case.
LCM_CANDIDATE_BLOCK_M: int = 384

#: 32-byte barrier signal block at the head of the workspace, holding:
#:   bytes [0..15] : 4 x uint32 grid-sync counters
#:   bytes [16..19]: uint32 NVLink/xGMI barrier counter
#:   bytes [20..27]: 2 x int32 NVLink/xGMI barrier signals (phase 0/1)
NUM_BARRIER_SIGNAL_BYTES: int = 32

#: Number of grid-sync counters in the barrier signal block.
NUM_MAX_GRID_SYNC_COUNTERS: int = 4

#: Size of a TokenSrcMetadata struct = (rank_idx, token_idx, topk_idx) x uint32.
TOKEN_SRC_METADATA_BYTES: int = 12


# ── Math helpers (constexpr-friendly: pure ints, no torch) ────────────────


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def align(x: int, a: int) -> int:
    """Round ``x`` up to the next multiple of ``a``."""
    return ((x + a - 1) // a) * a


def align_down(x: int, a: int) -> int:
    """Round ``x`` down to the previous multiple of ``a``.

    Mirrors the C++ ``align<uint32_t, /*kRoundUp=*/false>`` overload.
    """
    return (x // a) * a


def get_num_max_pool_tokens(
    num_ranks: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    num_experts_per_rank: int,
) -> int:
    """Mirror ``layout::get_num_max_pool_tokens``.

    Worst-case shared-expert token-pool capacity: every rank sends every
    token to every local expert it can route to, plus per-expert padding
    for the largest BLOCK_M. Result is aligned to the BLOCK_M LCM (384).
    """
    num_max_recv_tokens = num_ranks * num_max_tokens_per_rank
    num_max_experts_per_token = min(num_topk, num_experts_per_rank)
    raw = (
        num_max_recv_tokens * num_max_experts_per_token
        + num_experts_per_rank * (MAX_CANDIDATE_BLOCK_M - 1)
    )
    return align(raw, LCM_CANDIDATE_BLOCK_M)


def get_num_padded_sf_pool_tokens(num_max_pool_tokens: int, block_m: int) -> int:
    """Mirror ``layout::get_num_padded_sf_pool_tokens``.

    SFs share one contiguous region sized by ``num_pool_blocks * SF_BLOCK_M``
    where ``SF_BLOCK_M = align(BLOCK_M, 128)``.
    """
    return (num_max_pool_tokens // block_m) * align(block_m, 128)


# ── Workspace ──────────────────────────────────────────────────────────────


@dataclass
class Workspace:
    """Symmetric workspace shared across all ranks (xGMI peers).

    The byte layout matches ``deep_gemm::layout::Workspace`` exactly so
    weights produced by upstream tests can be loaded directly. See
    ``layout/mega_moe.cuh`` for the field-by-field source.
    """

    num_ranks: int
    num_experts: int
    num_max_tokens_per_rank: int
    num_topk: int

    # Derived (filled in __post_init__)
    num_experts_per_rank: int = field(init=False)
    num_max_recv_tokens_per_expert: int = field(init=False)
    num_max_pool_tokens: int = field(init=False)
    num_max_pool_blocks: int = field(init=False)

    def __post_init__(self) -> None:
        if self.num_experts % self.num_ranks != 0:
            raise ValueError(
                f"num_experts ({self.num_experts}) must be divisible by "
                f"num_ranks ({self.num_ranks})"
            )
        self.num_experts_per_rank = self.num_experts // self.num_ranks
        self.num_max_recv_tokens_per_expert = self.num_ranks * self.num_max_tokens_per_rank
        self.num_max_pool_tokens = get_num_max_pool_tokens(
            self.num_ranks,
            self.num_max_tokens_per_rank,
            self.num_topk,
            self.num_experts_per_rank,
        )
        # C++ uses kMinCandidateBlockM (=8) here, giving the worst-case
        # block count any chosen BLOCK_M can produce.
        self.num_max_pool_blocks = self.num_max_pool_tokens // MIN_CANDIDATE_BLOCK_M

    # ── Sizing ────────────────────────────────────────────────────────

    def get_num_bytes(self) -> int:
        """Total workspace size in bytes, 16-byte aligned (TMA parity)."""
        n = 0
        n += NUM_BARRIER_SIGNAL_BYTES
        n += self.num_experts * 8 * 2  # send + recv (uint64 each)
        n += self.num_experts_per_rank * 8  # recv_sum (uint64)
        n += align(self.num_max_pool_blocks, 2) * 4  # l1 arrival (uint32)
        n += self.num_max_pool_blocks * 8  # l2 arrival mask (uint64)
        n += (
            self.num_experts_per_rank
            * self.num_ranks
            * self.num_max_recv_tokens_per_expert
            * 4
        )  # src_token_topk_idx (int32)
        n += self.num_max_pool_tokens * TOKEN_SRC_METADATA_BYTES
        return align(n, 16)

    # ── Byte-offset accessors (mirror ``Workspace::get_*_ptr``) ────────

    @property
    def offset_grid_sync_counters(self) -> int:
        return 0

    @property
    def offset_nvl_barrier_counter(self) -> int:
        # On CDNA this is the xGMI barrier counter; name kept for C++ parity.
        return NUM_MAX_GRID_SYNC_COUNTERS * 4  # 16

    @property
    def offset_nvl_barrier_signals(self) -> int:
        # 2 x int32 phase signals.
        return self.offset_nvl_barrier_counter + 4  # 20

    @property
    def offset_expert_send_count(self) -> int:
        return NUM_BARRIER_SIGNAL_BYTES  # 32

    @property
    def offset_expert_recv_count(self) -> int:
        return self.offset_expert_send_count + self.num_experts * 8

    @property
    def offset_expert_recv_count_sum(self) -> int:
        return self.offset_expert_recv_count + self.num_ranks * self.num_experts_per_rank * 8

    @property
    def offset_l1_arrival_count(self) -> int:
        return self.offset_expert_recv_count_sum + self.num_experts_per_rank * 8

    @property
    def offset_l2_arrival_mask(self) -> int:
        # L1 padded to even-uint32 count so the L2 mask is 8-byte aligned.
        return self.offset_l1_arrival_count + align(self.num_max_pool_blocks, 2) * 4

    @property
    def offset_src_token_topk_idx(self) -> int:
        return self.offset_l2_arrival_mask + self.num_max_pool_blocks * 8

    @property
    def offset_token_src_metadata(self) -> int:
        return (
            self.offset_src_token_topk_idx
            + self.num_experts_per_rank
            * self.num_ranks
            * self.num_max_recv_tokens_per_expert
            * 4
        )

    # ── Per-element byte addressing helpers ───────────────────────────

    def src_token_topk_idx_byte_offset(
        self, expert_idx: int = 0, rank_idx: int = 0, token_idx: int = 0
    ) -> int:
        return (
            self.offset_src_token_topk_idx
            + (
                expert_idx * (self.num_ranks * self.num_max_recv_tokens_per_expert)
                + rank_idx * self.num_max_recv_tokens_per_expert
                + token_idx
            )
            * 4
        )

    def token_src_metadata_byte_offset(self, pool_token_idx: int = 0) -> int:
        return self.offset_token_src_metadata + pool_token_idx * TOKEN_SRC_METADATA_BYTES


__all__ = [
    "CANDIDATE_BLOCK_M",
    "MAX_CANDIDATE_BLOCK_M",
    "MIN_CANDIDATE_BLOCK_M",
    "LCM_CANDIDATE_BLOCK_M",
    "NUM_BARRIER_SIGNAL_BYTES",
    "NUM_MAX_GRID_SYNC_COUNTERS",
    "TOKEN_SRC_METADATA_BYTES",
    "Workspace",
    "ceil_div",
    "align",
    "align_down",
    "get_num_max_pool_tokens",
    "get_num_padded_sf_pool_tokens",
]
