# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ROCm/FlyDSL MegaMOE port contributors

"""MegaMoEScheduler — block-iteration state machine.

Direct port of ``deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh``
(deepseek-ai/DeepGEMM, MIT). The scheduler walks every per-expert
(L1, L2) block assigned to one CTA across an expert-wave swizzle.

Phase 0 ships only :class:`HostMegaMoEScheduler` — a pure-Python
reference used for host-side simulation, unit tests, and verifying the
schedule semantics without a GPU. The device-side emission of this
state machine inside a ``@flyc.kernel`` lands in Phase 5 alongside the
warp-specialised mega kernel; until then there is no device class to
import (avoids the temptation to wire to a stub).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Callable, Iterator, List, Optional

from .workspace import ceil_div, align, align_down


# ── Phase enum (mirror ``sched::BlockPhase``) ──────────────────────────────


class BlockPhase(enum.IntEnum):
    """Which of the two grouped GEMMs the block belongs to.

    Values match the C++ enum so any cross-language probe lines up.
    """

    NONE = 0
    LINEAR1 = 1  # gate+up grouped GEMM (output dim = 2 * intermediate_hidden)
    LINEAR2 = 2  # down grouped GEMM (output dim = hidden)


# ── Per-launch config ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class MegaMoESchedulerConfig:
    """Compile-time scheduler config; matches the C++ template params.

    Mirrors ``MegaMoEScheduler<...>`` template arguments. Every value
    is a constexpr-style int; pass through to ``@flyc.kernel`` as
    ``fx.Constexpr[int]``.
    """

    block_m: int
    block_n: int
    block_k: int
    l1_shape_n: int  # 2 * intermediate_hidden  (gate+up interleaved)
    l1_shape_k: int  # hidden
    l2_shape_n: int  # hidden
    l2_shape_k: int  # intermediate_hidden
    num_experts_per_rank: int
    num_experts_per_wave: int  # overlap granularity
    num_sms: int  # number of CTAs in the persistent grid
    num_ranks: int

    # Derived
    num_l1_block_ns: int = field(init=False)
    num_l2_block_ns: int = field(init=False)
    num_l1_block_ks: int = field(init=False)
    num_l2_block_ks: int = field(init=False)
    num_experts_per_lane: int = field(init=False)

    def __post_init__(self) -> None:
        # Hard validations identical to the C++ DG_STATIC_ASSERTs.
        if self.l1_shape_n % self.block_n != 0:
            raise ValueError("l1_shape_n must be divisible by block_n")
        if self.l2_shape_n % self.block_n != 0:
            raise ValueError("l2_shape_n must be divisible by block_n")
        if self.l1_shape_k % self.block_k != 0:
            raise ValueError("l1_shape_k must be divisible by block_k")
        if self.l2_shape_k % self.block_k != 0:
            raise ValueError("l2_shape_k must be divisible by block_k")
        if self.num_experts_per_rank % self.num_experts_per_wave != 0:
            raise ValueError("num_experts_per_rank must be divisible by num_experts_per_wave")
        # The C++ side requires the number of N blocks to be even because
        # NV uses a 2-CTA UMMA cluster sharing m_block_idx with adjacent
        # n_block_idx. CDNA has no 2-CTA cluster so we still require it
        # only for parity; relaxing it is safe on AMD but keeping the
        # constraint avoids accidental layout drift between platforms.
        l1_n = self.l1_shape_n // self.block_n
        l2_n = self.l2_shape_n // self.block_n
        if l1_n % 2 != 0:
            raise ValueError("L1 N block count must be even")
        if l2_n % 2 != 0:
            raise ValueError("L2 N block count must be even")
        if self.num_sms % 2 != 0:
            raise ValueError("num_sms must be even")

        object.__setattr__(self, "num_l1_block_ns", l1_n)
        object.__setattr__(self, "num_l2_block_ns", l2_n)
        object.__setattr__(self, "num_l1_block_ks", self.l1_shape_k // self.block_k)
        object.__setattr__(self, "num_l2_block_ks", self.l2_shape_k // self.block_k)
        object.__setattr__(self, "num_experts_per_lane", ceil_div(self.num_experts_per_rank, 32))


# ── Per-block descriptor returned by the iterator ─────────────────────────


@dataclass(frozen=True)
class BlockDescriptor:
    phase: BlockPhase
    expert_idx: int
    num_k_blocks: int
    m_block_idx: int
    n_block_idx: int


# ── Host-side reference simulator ─────────────────────────────────────────


class HostMegaMoEScheduler:
    """Pure-Python mirror of the device scheduler.

    Faithful to the C++ ``MegaMoEScheduler`` semantics: starts from
    expert 0, walks each expert's L1 blocks for the current wave, then
    its L2 blocks, then advances the wave. Used for unit tests, for
    sanity-checking the kernel output, and for visualizing the block
    schedule when tuning ``num_experts_per_wave``.

    Parameters
    ----------
    config:
        :class:`MegaMoESchedulerConfig` describing tile shapes.
    block_idx:
        The CTA's own block index in ``[0, num_sms)``. The sim returns
        only the blocks this CTA would visit.
    num_tokens_per_local_expert:
        List of length ``num_experts_per_rank`` giving the number of
        post-dispatch tokens routed to each local expert. In the real
        kernel these come from ``expert_recv_count_sum`` after dispatch.
    """

    def __init__(
        self,
        config: MegaMoESchedulerConfig,
        block_idx: int,
        num_tokens_per_local_expert: List[int],
    ) -> None:
        if len(num_tokens_per_local_expert) != config.num_experts_per_rank:
            raise ValueError(
                "num_tokens_per_local_expert length must equal num_experts_per_rank"
            )
        self.cfg = config
        self.block_idx0 = block_idx  # original CTA id (immutable)
        self.tokens = list(num_tokens_per_local_expert)

        # Mutable scheduler state (matches the C++ struct fields).
        self.next_phase = BlockPhase.LINEAR1
        self.current_local_expert_idx = 0
        self.current_num_tokens = 0
        self.current_pool_block_offset = 0
        self.block_idx = block_idx
        self.m_block_idx = 0
        self.n_block_idx = 0

        # Initialise (mirrors ``set_expert_idx(0)`` in for_each_block).
        self._set_expert_idx(0)

    # ── Helpers (pure Python; no warp-shuffle needed off-device) ────

    def _num_m_blocks(self, expert_idx: int) -> int:
        return ceil_div(self.tokens[expert_idx], self.cfg.block_m)

    def _pool_block_offset(self, expert_idx: int) -> int:
        return sum(self._num_m_blocks(i) for i in range(expert_idx))

    def _set_expert_idx(self, expert_idx: int) -> None:
        self.current_local_expert_idx = expert_idx
        self.current_num_tokens = (
            self.tokens[expert_idx] if expert_idx < self.cfg.num_experts_per_rank else 0
        )
        self.current_pool_block_offset = self._pool_block_offset(expert_idx)

    def _advance_expert_idx(self) -> None:
        self.current_pool_block_offset += ceil_div(self.current_num_tokens, self.cfg.block_m)
        self.current_local_expert_idx += 1
        if self.current_local_expert_idx < self.cfg.num_experts_per_rank:
            self.current_num_tokens = self.tokens[self.current_local_expert_idx]
        else:
            self.current_num_tokens = 0

    def _wave_end(self) -> int:
        return align(self.current_local_expert_idx + 1, self.cfg.num_experts_per_wave)

    # ── Per-phase block fetchers (mirror the inner whiles, but with a
    # single explicit return path so the FlyDSL device port can use
    # the same shape) ──────────────────────────────────────────────

    def _fetch_next_l1_block(self) -> bool:
        wave_end = self._wave_end()
        found = False
        while not found and self.current_local_expert_idx < wave_end:
            num_m = ceil_div(self.current_num_tokens, self.cfg.block_m)
            self.m_block_idx = self.block_idx // self.cfg.num_l1_block_ns
            if self.m_block_idx < num_m:
                found = True
            else:
                self.block_idx -= num_m * self.cfg.num_l1_block_ns
                self._advance_expert_idx()
        return found

    def _fetch_next_l2_block(self) -> bool:
        wave_end = self._wave_end()
        found = False
        while not found and self.current_local_expert_idx < wave_end:
            num_m = ceil_div(self.current_num_tokens, self.cfg.block_m)
            if self.block_idx < num_m * self.cfg.num_l2_block_ns:
                self.m_block_idx = self.block_idx // self.cfg.num_l2_block_ns
                found = True
            else:
                self.block_idx -= num_m * self.cfg.num_l2_block_ns
                self._advance_expert_idx()
        return found

    # ── Core state machine ─────────────────────────────────────────

    def _next_block(self) -> Optional[BlockDescriptor]:
        result: Optional[BlockDescriptor] = None
        while result is None and self.current_local_expert_idx < self.cfg.num_experts_per_rank:
            if self.next_phase == BlockPhase.LINEAR1:
                if self._fetch_next_l1_block():
                    self.n_block_idx = self.block_idx - self.m_block_idx * self.cfg.num_l1_block_ns
                    self.block_idx += self.cfg.num_sms
                    result = BlockDescriptor(
                        phase=BlockPhase.LINEAR1,
                        expert_idx=self.current_local_expert_idx,
                        num_k_blocks=self.cfg.num_l1_block_ks,
                        m_block_idx=self.m_block_idx,
                        n_block_idx=self.n_block_idx,
                    )
                else:
                    # L1 wave complete → switch to L2 starting at the
                    # wave's first expert.  C++ uses
                    # ``align<...,/*kRoundUp=*/false>(curr-1, wave)`` =
                    # floor-align, which is the wave-start expert.
                    self.next_phase = BlockPhase.LINEAR2
                    wave_start = align_down(
                        max(self.current_local_expert_idx - 1, 0),
                        self.cfg.num_experts_per_wave,
                    )
                    self._set_expert_idx(wave_start)
                    # Reset the CTA's block walker to start of the wave's
                    # L2 region. The original C++ doesn't explicitly
                    # reset block_idx here because the inner
                    # ``advance_expert_idx`` left it at zero w.r.t. the
                    # wave; we mirror by using the already-adjusted
                    # ``self.block_idx``.
            else:  # LINEAR2
                if self._fetch_next_l2_block():
                    self.n_block_idx = self.block_idx - self.m_block_idx * self.cfg.num_l2_block_ns
                    self.block_idx += self.cfg.num_sms
                    result = BlockDescriptor(
                        phase=BlockPhase.LINEAR2,
                        expert_idx=self.current_local_expert_idx,
                        num_k_blocks=self.cfg.num_l2_block_ks,
                        m_block_idx=self.m_block_idx,
                        n_block_idx=self.n_block_idx,
                    )
                else:
                    # Move to the next wave's L1.
                    self.next_phase = BlockPhase.LINEAR1
        return result

    # ── Public iteration API ──────────────────────────────────────

    def __iter__(self) -> Iterator[BlockDescriptor]:
        return self

    def __next__(self) -> BlockDescriptor:
        nb = self._next_block()
        if nb is None:
            raise StopIteration
        return nb

    def for_each_block(self, func: Callable[[BlockDescriptor], None]) -> None:
        """Mirror the device-side ``for_each_block(func)`` walk."""
        for blk in self:
            func(blk)


__all__ = [
    "BlockPhase",
    "MegaMoESchedulerConfig",
    "BlockDescriptor",
    "HostMegaMoEScheduler",
]
