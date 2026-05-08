# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ROCm/FlyDSL MegaMOE port contributors

"""Compile-time block / wave / pipeline heuristics.

Direct port of ``csrc/jit_kernels/heuristics/mega_moe.hpp`` from
deepseek-ai/DeepGEMM (MIT). The C++ heuristic returns three coupled
artifacts that we replicate one-for-one:

* :func:`pick_block_config` — a step function on E[tokens/expert] that
  picks ``block_m``, ``store_block_m``, and ``num_epilogue_warpgroups``.
  ``block_n`` and ``block_k`` are fixed at 128.

* :func:`pick_num_experts_per_wave` — chooses how many experts are
  fused per overlap wave so that L1 work fully fills all CTAs after a
  2× imbalance derate, then rounds up to a divisor of
  ``num_experts_per_rank`` so every wave is the same size.

The C++ also returns ``cluster_size`` (always 2 on Blackwell, the
2-CTA UMMA cluster). We omit it: CDNA has no equivalent and our port
always uses a 1-CTA mapping.

NV picks ``block_n=128``, ``block_k=128``. We keep the same constants
because the upstream weight tensors and SF format assume them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .workspace import CANDIDATE_BLOCK_M, ceil_div


#: NV uses the same N/K block sizes for every config; the SF format
#: depends on these so they are fixed by the input data, not free
#: parameters of our port.
BLOCK_N: int = 128
BLOCK_K: int = 128

#: Threads per warpgroup on NV (and per AMD wavefront-quad). The C++
#: side multiplies ``num_epilogue_warpgroups * 128`` to get the
#: epilogue thread count; we keep the same convention.
THREADS_PER_WARPGROUP: int = 128


@dataclass(frozen=True)
class BlockConfig:
    """Result of the BLOCK_M step-function heuristic.

    Mirrors the tuple returned by ``get_block_config_for_mega_moe`` in
    the C++ source, minus ``cluster_size`` (always 2 on Blackwell, N/A
    on CDNA).
    """

    block_m: int
    store_block_m: int
    num_epilogue_warpgroups: int

    @property
    def num_epilogue_threads(self) -> int:
        """Total epilogue thread count = warpgroups × 128."""
        return self.num_epilogue_warpgroups * THREADS_PER_WARPGROUP


def expected_tokens_per_expert(
    num_tokens: int,
    num_ranks: int,
    num_topk: int,
    num_experts: int,
) -> float:
    """E[tokens / expert] used by the BLOCK_M heuristic.

    The C++ helper computes ``num_tokens * num_ranks * num_topk /
    num_experts``. ``num_tokens`` here is the per-rank token count
    (not the global count); the rank multiplier produces the total
    routed-token volume per global expert.
    """
    return float(num_tokens) * num_ranks * num_topk / float(num_experts)


def pick_block_config(
    num_ranks: int,
    num_experts: int,
    num_topk: int,
    num_tokens: int,
) -> BlockConfig:
    """Step function on E[tokens/expert].

    Mirrors ``get_block_config_for_mega_moe`` exactly. Note that the
    heuristic never selects ``block_m=8`` even though it is in the
    candidate list — that value is reserved for ``num_max_pool_blocks``
    sizing only.
    """
    e = expected_tokens_per_expert(num_tokens, num_ranks, num_topk, num_experts)

    if e <= 8.5:
        # RL long-tail rollout etc.
        cfg = BlockConfig(block_m=16, store_block_m=8, num_epilogue_warpgroups=2)
    elif e <= 16.5:
        # Small batch, small EP, decoding (e.g. 6/384, EP8, bsz 128).
        cfg = BlockConfig(block_m=32, store_block_m=16, num_epilogue_warpgroups=2)
    elif e <= 32.5:
        # Medium batch, small EP, decoding (e.g. 6/384, EP8, bsz 256).
        cfg = BlockConfig(block_m=64, store_block_m=32, num_epilogue_warpgroups=1)
    elif e <= 64.5:
        # Large batch, small EP, decoding (e.g. 6/384, EP8, bsz 512).
        cfg = BlockConfig(block_m=96, store_block_m=16, num_epilogue_warpgroups=2)
    elif e <= 96.5:
        # Medium batch, medium EP, decoding (e.g. 6/384, EP16/bsz256
        # or EP32/bsz128).
        cfg = BlockConfig(block_m=128, store_block_m=32, num_epilogue_warpgroups=2)
    else:
        # Prefill, or large EP decoding.
        cfg = BlockConfig(block_m=192, store_block_m=32, num_epilogue_warpgroups=2)

    # Mirror the C++ host assert: block_m must be a candidate.
    assert cfg.block_m in CANDIDATE_BLOCK_M, (
        f"BlockConfig.block_m={cfg.block_m} not in CANDIDATE_BLOCK_M={CANDIDATE_BLOCK_M}"
    )
    return cfg


def pick_num_experts_per_wave(
    num_experts_per_rank: int,
    num_tokens: int,
    num_topk: int,
    intermediate_hidden: int,
    block_m: int,
    block_n: int,
    num_sms: int,
) -> int:
    """Mirror ``get_num_experts_per_wave_for_mega_moe``.

    Strategy:
    1. If the typical expert sees < 1 token, fuse all experts into one
       wave (L1 work is too small to need splitting).
    2. Otherwise, count L1 blocks/expert assuming even routing, derate
       by an imbalance factor of 2 (so a hot expert can soak up the
       slack), and pick the smallest wave size whose total L1 blocks
       cover all SMs.
    3. Round *up* to a divisor of ``num_experts_per_rank`` so every
       wave has identical work.
    """
    expected = float(num_tokens) * num_topk / float(num_experts_per_rank)
    if expected < 1.0:
        return num_experts_per_rank

    k_imbalance_factor = 2

    num_m_blocks = ceil_div(int(math.ceil(expected)), block_m)
    num_n_blocks = (2 * intermediate_hidden) // block_n
    num_l1_blocks_per_expert = num_m_blocks * num_n_blocks

    if num_l1_blocks_per_expert > 0:
        n_per_wave = ceil_div(k_imbalance_factor * num_sms, num_l1_blocks_per_expert)
    else:
        n_per_wave = 1

    n_per_wave = min(n_per_wave, num_experts_per_rank)

    # Round up to the nearest divisor of num_experts_per_rank.
    while (
        n_per_wave < num_experts_per_rank
        and num_experts_per_rank % n_per_wave != 0
    ):
        n_per_wave += 1

    return n_per_wave


__all__ = [
    "BLOCK_N",
    "BLOCK_K",
    "THREADS_PER_WARPGROUP",
    "BlockConfig",
    "expected_tokens_per_expert",
    "pick_block_config",
    "pick_num_experts_per_wave",
]
