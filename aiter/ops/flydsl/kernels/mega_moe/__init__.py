# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ROCm/FlyDSL MegaMOE port contributors

"""FlyDSL port of DeepSeek V4 MegaMOE for AMD MI355X (gfx950).

This package re-implements the layout, scheduler, and (later) GEMM /
dispatch / combine pieces of ``deep_gemm.fp8_fp4_mega_moe`` on top of
FlyDSL's CDNA4 backend.

Phase 0 (this drop): byte-layout, heuristics, and host-side scheduler
reference. CPU-only; no FlyDSL or GPU required to run the tests.

Phase 1 (next): FP8/FP4 grouped GEMM core based on
``sm100_fp8_fp4_gemm_1d1d.cuh``, mapped onto CDNA4 scaled MFMA.
"""

from .heuristics import (
    BLOCK_K,
    BLOCK_N,
    THREADS_PER_WARPGROUP,
    BlockConfig,
    expected_tokens_per_expert,
    pick_block_config,
    pick_num_experts_per_wave,
)
from .scheduler import (
    BlockDescriptor,
    BlockPhase,
    HostMegaMoEScheduler,
    MegaMoESchedulerConfig,
)
from .workspace import (
    CANDIDATE_BLOCK_M,
    LCM_CANDIDATE_BLOCK_M,
    MAX_CANDIDATE_BLOCK_M,
    MIN_CANDIDATE_BLOCK_M,
    NUM_BARRIER_SIGNAL_BYTES,
    NUM_MAX_GRID_SYNC_COUNTERS,
    TOKEN_SRC_METADATA_BYTES,
    Workspace,
    align,
    align_down,
    ceil_div,
    get_num_max_pool_tokens,
    get_num_padded_sf_pool_tokens,
)

__all__ = [
    # workspace
    "CANDIDATE_BLOCK_M",
    "LCM_CANDIDATE_BLOCK_M",
    "MAX_CANDIDATE_BLOCK_M",
    "MIN_CANDIDATE_BLOCK_M",
    "NUM_BARRIER_SIGNAL_BYTES",
    "NUM_MAX_GRID_SYNC_COUNTERS",
    "TOKEN_SRC_METADATA_BYTES",
    "Workspace",
    "align",
    "align_down",
    "ceil_div",
    "get_num_max_pool_tokens",
    "get_num_padded_sf_pool_tokens",
    # heuristics
    "BLOCK_K",
    "BLOCK_N",
    "THREADS_PER_WARPGROUP",
    "BlockConfig",
    "expected_tokens_per_expert",
    "pick_block_config",
    "pick_num_experts_per_wave",
    # scheduler
    "BlockDescriptor",
    "BlockPhase",
    "HostMegaMoEScheduler",
    "MegaMoESchedulerConfig",
]
