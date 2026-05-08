# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ROCm/FlyDSL MegaMOE port contributors

"""CPU-only smoke tests for workspace + heuristics + scheduler.

Run as:  python -m unittest mega_moe.test_workspace_scheduler -v
or:      python -m pytest aiter/ops/flydsl/kernels/mega_moe/test_workspace_scheduler.py -v
"""

from __future__ import annotations

import unittest

from . import (
    BLOCK_K,
    BLOCK_N,
    BlockConfig,
    BlockPhase,
    HostMegaMoEScheduler,
    MegaMoESchedulerConfig,
    Workspace,
    align,
    ceil_div,
    expected_tokens_per_expert,
    get_num_max_pool_tokens,
    get_num_padded_sf_pool_tokens,
    pick_block_config,
    pick_num_experts_per_wave,
    CANDIDATE_BLOCK_M,
    LCM_CANDIDATE_BLOCK_M,
)


# ───────────────────────── Workspace math ─────────────────────────


class WorkspaceMathTests(unittest.TestCase):
    """Validate against constants taken straight from layout/mega_moe.cuh."""

    def test_lcm_candidate_block_m(self) -> None:
        from math import gcd
        from functools import reduce

        def lcm(a: int, b: int) -> int:
            return a * b // gcd(a, b)

        self.assertEqual(reduce(lcm, CANDIDATE_BLOCK_M), LCM_CANDIDATE_BLOCK_M)

    def test_pool_tokens_alignment(self) -> None:
        for n_ranks in (2, 4, 8, 16):
            for max_t in (32, 128, 512):
                for topk in (1, 4, 6, 8):
                    for n_eper in (4, 16, 32):
                        n = get_num_max_pool_tokens(n_ranks, max_t, topk, n_eper)
                        self.assertEqual(n % LCM_CANDIDATE_BLOCK_M, 0)
                        self.assertGreaterEqual(
                            n, n_ranks * max_t * min(topk, n_eper)
                        )

    def test_padded_sf_pool(self) -> None:
        n = get_num_max_pool_tokens(8, 256, 8, 32)
        for bm in CANDIDATE_BLOCK_M:
            self.assertEqual(
                get_num_padded_sf_pool_tokens(n, bm),
                (n // bm) * align(bm, 128),
            )


# ───────────────────────── Heuristics ─────────────────────────


class BlockConfigHeuristicTests(unittest.TestCase):
    """Validate ``pick_block_config`` against the C++ reference table.

    Each row: (E[tokens/expert], block_m, store_block_m, num_epilogue_warpgroups).
    Source: ``csrc/jit_kernels/heuristics/mega_moe.hpp``.
    """

    EXPECTED_TABLE = [
        # ≤ 8.5 → small / RL long-tail
        (1.0, 16, 8, 2),
        (8.5, 16, 8, 2),
        # ≤ 16.5 → small batch, small EP, decoding
        (8.6, 32, 16, 2),
        (16.5, 32, 16, 2),
        # ≤ 32.5 → medium batch
        (16.6, 64, 32, 1),
        (32.5, 64, 32, 1),
        # ≤ 64.5 → large batch
        (32.6, 96, 16, 2),
        (64.5, 96, 16, 2),
        # ≤ 96.5 → medium-EP decoding
        (64.6, 128, 32, 2),
        (96.5, 128, 32, 2),
        # > 96.5 → prefill / large EP
        (96.6, 192, 32, 2),
        (1024.0, 192, 32, 2),
    ]

    def test_block_config_table(self) -> None:
        # Drive E directly by choosing num_tokens such that
        # num_tokens * num_ranks * num_topk / num_experts == E.
        # Use num_ranks=1, num_topk=1, num_experts=1 → E == num_tokens.
        for e, bm, sbm, ewg in self.EXPECTED_TABLE:
            n_tokens = int(e) if e == int(e) else None
            # We can't pass a float num_tokens, so use a representative
            # int that produces the same E with chosen multipliers.
            # Easiest: scale to integers via ranks×topk.
            # Pick num_experts=10, num_ranks=1, num_topk=1, then
            # num_tokens = ceil(E*10) gives E roughly. Instead just call
            # the helper that takes the float E directly:
            cfg = pick_block_config(
                num_ranks=2,
                num_experts=20,
                num_topk=1,
                num_tokens=int(round(e * 10)),  # E = num_tokens*2*1/20 = num_tokens/10
            )
            self.assertEqual(cfg.block_m, bm, msg=f"E≈{e}")
            self.assertEqual(cfg.store_block_m, sbm, msg=f"E≈{e}")
            self.assertEqual(cfg.num_epilogue_warpgroups, ewg, msg=f"E≈{e}")

    def test_expected_tokens_formula(self) -> None:
        # Sanity check the formula: per-rank tokens × ranks × topk / experts.
        self.assertAlmostEqual(
            expected_tokens_per_expert(num_tokens=128, num_ranks=8, num_topk=8, num_experts=256),
            128 * 8 * 8 / 256,
        )

    def test_block_m_never_8(self) -> None:
        # The heuristic skips block_m=8 entirely (it's reserved for
        # num_max_pool_blocks worst-case sizing).
        for n_tokens in (1, 8, 64, 256, 1024, 8192):
            cfg = pick_block_config(num_ranks=8, num_experts=256, num_topk=8,
                                    num_tokens=n_tokens)
            self.assertNotEqual(cfg.block_m, 8)
            self.assertIn(cfg.block_m, CANDIDATE_BLOCK_M)


class ExpertsPerWaveHeuristicTests(unittest.TestCase):
    """Validate ``pick_num_experts_per_wave`` matches the C++ algorithm."""

    def test_under_one_token_per_expert_returns_all(self) -> None:
        # E[tokens/expert] < 1 ⇒ fuse all experts into one wave.
        n_per_wave = pick_num_experts_per_wave(
            num_experts_per_rank=32,
            num_tokens=1,
            num_topk=1,
            intermediate_hidden=2048,
            block_m=16,
            block_n=BLOCK_N,
            num_sms=128,
        )
        self.assertEqual(n_per_wave, 32)

    def test_returns_divisor_of_num_experts_per_rank(self) -> None:
        # The output must be a divisor so every wave is the same size.
        for n_tokens in (16, 64, 256, 1024):
            for ne in (8, 16, 32):
                bm = pick_block_config(8, ne * 8, 8, n_tokens).block_m
                w = pick_num_experts_per_wave(
                    num_experts_per_rank=ne,
                    num_tokens=n_tokens,
                    num_topk=8,
                    intermediate_hidden=2048,
                    block_m=bm,
                    block_n=BLOCK_N,
                    num_sms=128,
                )
                self.assertGreaterEqual(w, 1)
                self.assertLessEqual(w, ne)
                self.assertEqual(ne % w, 0, msg=f"n_tokens={n_tokens}, ne={ne}")

    def test_more_blocks_per_expert_means_smaller_wave(self) -> None:
        # When intermediate_hidden grows (more L1 N blocks per expert),
        # fewer experts are needed per wave to fill all SMs.
        small = pick_num_experts_per_wave(
            num_experts_per_rank=32, num_tokens=512, num_topk=8,
            intermediate_hidden=512, block_m=128, block_n=BLOCK_N, num_sms=128,
        )
        large = pick_num_experts_per_wave(
            num_experts_per_rank=32, num_tokens=512, num_topk=8,
            intermediate_hidden=8192, block_m=128, block_n=BLOCK_N, num_sms=128,
        )
        self.assertGreaterEqual(small, large)


# ───────────────────────── Workspace layout ─────────────────────────


class WorkspaceLayoutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ws = Workspace(
            num_ranks=8,
            num_experts=256,
            num_max_tokens_per_rank=128,
            num_topk=8,
        )

    def test_derived_dims(self) -> None:
        self.assertEqual(self.ws.num_experts_per_rank, 32)
        self.assertEqual(self.ws.num_max_recv_tokens_per_expert, 8 * 128)

    def test_layout_offsets_monotonic(self) -> None:
        offsets = [
            self.ws.offset_grid_sync_counters,
            self.ws.offset_nvl_barrier_counter,
            self.ws.offset_nvl_barrier_signals,
            self.ws.offset_expert_send_count,
            self.ws.offset_expert_recv_count,
            self.ws.offset_expert_recv_count_sum,
            self.ws.offset_l1_arrival_count,
            self.ws.offset_l2_arrival_mask,
            self.ws.offset_src_token_topk_idx,
            self.ws.offset_token_src_metadata,
        ]
        for prev, cur in zip(offsets, offsets[1:]):
            self.assertLess(prev, cur)

    def test_total_size_alignment(self) -> None:
        self.assertEqual(self.ws.get_num_bytes() % 16, 0)

    def test_known_byte_layout_prefix(self) -> None:
        self.assertEqual(self.ws.offset_grid_sync_counters, 0)
        self.assertEqual(self.ws.offset_nvl_barrier_counter, 16)
        self.assertEqual(self.ws.offset_nvl_barrier_signals, 20)
        self.assertEqual(self.ws.offset_expert_send_count, 32)

    def test_per_element_addressing(self) -> None:
        last = self.ws.src_token_topk_idx_byte_offset(
            expert_idx=self.ws.num_experts_per_rank - 1,
            rank_idx=self.ws.num_ranks - 1,
            token_idx=self.ws.num_max_recv_tokens_per_expert - 1,
        )
        self.assertGreaterEqual(last, self.ws.offset_src_token_topk_idx)
        self.assertLess(last, self.ws.offset_token_src_metadata)

        last_meta = self.ws.token_src_metadata_byte_offset(
            self.ws.num_max_pool_tokens - 1
        )
        self.assertGreaterEqual(last_meta, self.ws.offset_token_src_metadata)
        self.assertLessEqual(last_meta + 12, self.ws.get_num_bytes())


# ───────────────────────── Scheduler ─────────────────────────


def _make_cfg(num_experts_per_rank: int = 8,
              num_experts_per_wave: int = 4,
              block_m: int = 64,
              num_sms: int = 128) -> MegaMoESchedulerConfig:
    return MegaMoESchedulerConfig(
        block_m=block_m,
        block_n=BLOCK_N,
        block_k=BLOCK_K,
        l1_shape_n=2 * 1024,
        l1_shape_k=4096,
        l2_shape_n=4096,
        l2_shape_k=1024,
        num_experts_per_rank=num_experts_per_rank,
        num_experts_per_wave=num_experts_per_wave,
        num_sms=num_sms,
        num_ranks=8,
    )


class SchedulerStateMachineTests(unittest.TestCase):
    def test_config_validations(self) -> None:
        with self.assertRaises(ValueError):
            MegaMoESchedulerConfig(
                block_m=64, block_n=BLOCK_N, block_k=BLOCK_K,
                l1_shape_n=130, l1_shape_k=4096,  # not divisible
                l2_shape_n=4096, l2_shape_k=1024,
                num_experts_per_rank=8, num_experts_per_wave=4,
                num_sms=128, num_ranks=8,
            )

    def test_empty_schedule(self) -> None:
        cfg = _make_cfg()
        sched = HostMegaMoEScheduler(cfg, block_idx=0,
                                     num_tokens_per_local_expert=[0] * 8)
        self.assertEqual(list(sched), [])

    def test_uniform_routing_full_coverage(self) -> None:
        cfg = _make_cfg()
        tokens = [64] * cfg.num_experts_per_rank  # 1 m_block per expert
        seen_l1: set = set()
        seen_l2: set = set()
        for cta in range(cfg.num_sms):
            sched = HostMegaMoEScheduler(cfg, block_idx=cta,
                                         num_tokens_per_local_expert=tokens)
            for blk in sched:
                key = (blk.expert_idx, blk.m_block_idx, blk.n_block_idx)
                target = seen_l1 if blk.phase == BlockPhase.LINEAR1 else seen_l2
                self.assertNotIn(key, target,
                                 msg=f"{blk.phase.name} dup at CTA {cta}: {blk}")
                target.add(key)
        self.assertEqual(len(seen_l1),
                         cfg.num_experts_per_rank * cfg.num_l1_block_ns)
        self.assertEqual(len(seen_l2),
                         cfg.num_experts_per_rank * cfg.num_l2_block_ns)

    def test_phase_order_within_one_wave(self) -> None:
        cfg = _make_cfg()
        tokens = [64] * cfg.num_experts_per_rank
        sched = HostMegaMoEScheduler(cfg, block_idx=0,
                                     num_tokens_per_local_expert=tokens)
        first_wave_experts = set(range(cfg.num_experts_per_wave))
        l1_seen, l2_seen = False, False
        for blk in sched:
            if blk.expert_idx not in first_wave_experts:
                break
            if blk.phase == BlockPhase.LINEAR1:
                self.assertFalse(l2_seen)
                l1_seen = True
            else:
                l2_seen = True
        self.assertTrue(l1_seen or l2_seen)

    # ── Skewed routing tests ──────────────────────────────────────

    def test_skewed_routing_full_coverage(self) -> None:
        """One hot expert + many cold experts must still cover every block."""
        cfg = _make_cfg(num_experts_per_rank=8, num_experts_per_wave=4,
                        block_m=64)
        # Expert 3 hot (256 tokens = 4 m_blocks), expert 5 medium (32 tokens =
        # 1 m_block), rest cold (0 tokens).
        tokens = [0, 0, 0, 256, 0, 32, 0, 0]
        m_blocks_per_expert = [ceil_div(t, cfg.block_m) for t in tokens]

        seen_l1: set = set()
        seen_l2: set = set()
        for cta in range(cfg.num_sms):
            sched = HostMegaMoEScheduler(cfg, block_idx=cta,
                                         num_tokens_per_local_expert=tokens)
            for blk in sched:
                key = (blk.expert_idx, blk.m_block_idx, blk.n_block_idx)
                target = seen_l1 if blk.phase == BlockPhase.LINEAR1 else seen_l2
                self.assertNotIn(key, target)
                target.add(key)

        expected_l1 = sum(
            m * cfg.num_l1_block_ns for m in m_blocks_per_expert
        )
        expected_l2 = sum(
            m * cfg.num_l2_block_ns for m in m_blocks_per_expert
        )
        self.assertEqual(len(seen_l1), expected_l1)
        self.assertEqual(len(seen_l2), expected_l2)

        # Cold experts produce zero blocks.
        for blk in [b for cta in range(cfg.num_sms)
                    for b in HostMegaMoEScheduler(
                        cfg, cta, tokens)]:
            self.assertGreater(tokens[blk.expert_idx], 0,
                               msg=f"Cold expert produced a block: {blk}")

    def test_extreme_skew_single_hot_expert(self) -> None:
        """All tokens on one expert in one wave — the other waves must be no-ops."""
        cfg = _make_cfg(num_experts_per_rank=8, num_experts_per_wave=2,
                        block_m=64, num_sms=64)
        tokens = [0] * 8
        tokens[0] = 1024  # 16 m_blocks for expert 0
        m_blocks = ceil_div(1024, cfg.block_m)

        seen_l1: set = set()
        for cta in range(cfg.num_sms):
            sched = HostMegaMoEScheduler(cfg, block_idx=cta,
                                         num_tokens_per_local_expert=tokens)
            for blk in sched:
                if blk.phase == BlockPhase.LINEAR1:
                    self.assertEqual(blk.expert_idx, 0)
                    seen_l1.add((blk.m_block_idx, blk.n_block_idx))

        self.assertEqual(len(seen_l1), m_blocks * cfg.num_l1_block_ns)

    def test_multi_wave_continuation_single_cta(self) -> None:
        """A single CTA must traverse all waves it's responsible for."""
        # 16 experts, 4 waves, every expert has 1 m_block. With small CTA
        # count (8) and L1_N=16 blocks/expert, each wave has 4*16=64 L1
        # blocks ⇒ each CTA does 8 L1 + 8 L2 per wave × 4 waves.
        cfg = _make_cfg(num_experts_per_rank=16, num_experts_per_wave=4,
                        block_m=64, num_sms=8)
        tokens = [64] * 16

        # Track wave membership per emitted block.
        for cta in range(cfg.num_sms):
            sched = HostMegaMoEScheduler(cfg, block_idx=cta,
                                         num_tokens_per_local_expert=tokens)
            blocks = list(sched)
            # The CTA must emit at least one block from each of the 4 waves
            # in L1 *or* L2 (with 1 m_block per expert and 16 N blocks, every
            # wave produces 64 L1 blocks split across 8 CTAs ⇒ 8 each).
            wave_seen = {w: 0 for w in range(4)}
            for blk in blocks:
                wave_seen[blk.expert_idx // cfg.num_experts_per_wave] += 1
            for w, count in wave_seen.items():
                self.assertGreater(count, 0,
                                   msg=f"CTA {cta} skipped wave {w} entirely")

    def test_pool_block_offset_accumulation(self) -> None:
        """``current_pool_block_offset`` must equal sum of m_blocks of preceding experts."""
        cfg = _make_cfg(num_experts_per_rank=8, num_experts_per_wave=8,
                        block_m=64)
        tokens = [128, 64, 192, 0, 32, 256, 64, 128]
        sched = HostMegaMoEScheduler(cfg, block_idx=0,
                                     num_tokens_per_local_expert=tokens)
        # Walk to expert 3 (idx 3) and verify offset.
        expected = sum(ceil_div(tokens[i], cfg.block_m) for i in range(3))
        sched._set_expert_idx(3)
        self.assertEqual(sched.current_pool_block_offset, expected)

        # Walk via _advance_expert_idx from 0 to 4 and verify each step.
        sched._set_expert_idx(0)
        running = 0
        for e in range(8):
            self.assertEqual(sched.current_pool_block_offset, running)
            running += ceil_div(tokens[e], cfg.block_m)
            sched._advance_expert_idx()


def _run() -> None:  # pragma: no cover
    unittest.main(verbosity=2)


if __name__ == "__main__":  # pragma: no cover
    _run()
