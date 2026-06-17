# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
KernelInstance dataclass + candidate / default kernel dicts for the FP8
block-wise *batched* GEMM (DeepSeek V4 wo_a path).

Mirrors ``ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_instance.py``; the
only differences are the kernel-name prefix (``a8w8_batched_blockscale``
instead of ``a8w8_blockscale``) and the lookup-key arity (B, M, N, K
instead of M, N, K).

All instances pin the recipe to ``(Scale_Block_M=1, Scale_Block_N=128,
Scale_Block_K=128)`` -- exactly the DeepSeek V4 ``wo_a`` recipe, equal
to DeepGEMM's ``recipe=(1, 1, 128)`` for the activation side and an
implied 128x128 weight-block grid.
"""

from dataclasses import dataclass


@dataclass
class KernelInstance:
    BLOCK_SIZE: int
    ScaleBlockM: int
    ScaleBlockN: int
    ScaleBlockK: int
    MPerBLOCK: int
    NPerBLOCK: int
    KPerBLOCK: int
    AK1: int
    BK1: int
    MPerXDL: int
    NPerXDL: int
    WAVE_MAP_M: int
    WAVE_MAP_N: int
    ABLOCK_TRANSFER: list[int]
    BBLOCK_TRANSFER: list[int]
    CSHUFFLE_MX_PER_WAVE_PERSHUFFLE: int
    CSHUFFLE_NX_PER_WAVE_PERSHUFFLE: int
    CBLOCK_TRANSFER: list[int]
    CBLOCK_SPV: list[int]
    PIPELINE_Sched: str
    PIPELINE_VERSION: int

    @property
    def name(self) -> str:
        return ("_").join(
            [
                "a8w8_batched_blockscale",
                ("x").join(
                    map(str, [self.ScaleBlockM, self.ScaleBlockN, self.ScaleBlockK])
                ),
                ("x").join(
                    map(
                        str,
                        [
                            self.BLOCK_SIZE,
                            self.MPerBLOCK,
                            self.NPerBLOCK,
                            self.KPerBLOCK,
                        ],
                    )
                ),
                ("x").join(map(str, [self.AK1, self.BK1])),
                ("x").join(map(str, [self.MPerXDL, self.NPerXDL])),
                ("x").join(map(str, [self.WAVE_MAP_M, self.WAVE_MAP_N])),
                ("x").join(map(str, self.ABLOCK_TRANSFER)),
                ("x").join(map(str, self.BBLOCK_TRANSFER)),
                ("x").join(map(str, self.CBLOCK_TRANSFER)),
                ("x").join(map(str, self.CBLOCK_SPV)),
                ("x").join(
                    map(
                        str,
                        [
                            self.CSHUFFLE_MX_PER_WAVE_PERSHUFFLE,
                            self.CSHUFFLE_NX_PER_WAVE_PERSHUFFLE,
                        ],
                    )
                ),
                self.PIPELINE_Sched.lower(),
                f"v{self.PIPELINE_VERSION}",
            ]
        )


# fmt: off
# Initial seed set: the same tile points that ``ck_gemm_a8w8_blockscale`` ships
# (its candidate_kernels_dict).  These are known-good gfx9-family templates;
# the autotune sweep can extend the dict per (B, M, N, K) shape.
candidate_kernels_dict = {
    ################| Block| Scale| Scale| Scale|  MPer|  NPer|  KPer| AK1| BK1|MPer| NPer| MXdl| NXdl|  ABlockTransfer|  BBlockTransfer|    CShuffle|    CShuffle|     CBlockTransferClusterLengths|  CBlockTransfer|  Block-wiseGemm|     Block-wiseGemm|
    # Compute-friendly (large MPerBlock, prefill regime)
    0:   KernelInstance(256,     1,   128,   128,   128,   128,   128,  16,  16,  32,   32,    2,    2,     [ 8, 32, 1],     [ 8, 32, 1],           1,           1,                   [1, 32, 1,  8],             [8],     "Intrawave",                  3,),
    1:   KernelInstance(256,     1,   128,   128,   128,    64,   128,  16,  16,  32,   32,    2,    1,     [ 8, 32, 1],     [ 8, 32, 1],           1,           1,                   [1, 32, 1,  8],             [8],     "Intrawave",                  3,),
    2:   KernelInstance(256,     1,   128,   128,    64,   128,   128,  16,  16,  32,   32,    1,    2,     [ 8, 32, 1],     [ 8, 32, 1],           1,           1,                   [1, 32, 1,  8],             [8],     "Intrawave",                  3,),
    3:   KernelInstance(256,     1,   128,   128,    64,    64,   128,  16,  16,  32,   32,    1,    1,     [ 8, 32, 1],     [ 8, 32, 1],           1,           1,                   [1, 32, 1,  8],             [8],     "Intrawave",                  3,),
    # Memory-friendly (small MPerBlock, decode regime -- the V4-Flash hot path).
    4:   KernelInstance(256,     1,   128,   128,    16,   256,   128,   8,  16,  16,   16,    1,    4,     [16, 16, 1],     [ 8, 32, 1],           1,           2,                   [1, 16, 1, 16],             [8],     "Intrawave",                  1,),
    5:   KernelInstance(256,     1,   128,   128,    16,   128,   128,   8,  16,  16,   16,    1,    2,     [16, 16, 1],     [ 8, 32, 1],           1,           2,                   [1, 16, 1, 16],             [8],     "Intrawave",                  1,),
    6:   KernelInstance(256,     1,   128,   128,    16,    64,   128,   8,  16,  16,   16,    1,    1,     [16, 16, 1],     [ 8, 32, 1],           1,           1,                   [1, 16, 1, 16],             [4],     "Intrawave",                  1,),
    7:   KernelInstance(256,     1,   128,   128,    16,   128,   256,  16,  16,  16,   16,    1,    2,     [16, 16, 1],     [16, 16, 1],           1,           2,                   [1, 16, 1, 16],             [8],     "Intrawave",                  1,),
    8:   KernelInstance(256,     1,   128,   128,    16,    64,   256,  16,  16,  16,   16,    1,    1,     [16, 16, 1],     [16, 16, 1],           1,           1,                   [1, 16, 1, 16],             [4],     "Intrawave",                  1,),
    9:   KernelInstance(256,     1,   128,   128,    32,   256,   128,  16,  16,  32,   32,    1,    2,     [ 8, 32, 1],     [ 8, 32, 1],           1,           1,                   [1, 32, 1,  8],             [8],     "Intrawave",                  1,),
    10:  KernelInstance(256,     1,   128,   128,    32,   128,   128,  16,  16,  32,   32,    1,    1,     [ 8, 32, 1],     [ 8, 32, 1],           1,           1,                   [1, 32, 1,  8],             [8],     "Intrawave",                  1,),
    11:  KernelInstance(256,     1,   128,   128,    32,    64,   128,  16,  16,  16,   16,    2,    1,     [ 8, 32, 1],     [ 8, 32, 1],           2,           1,                   [1, 32, 1,  8],             [8],     "Intrawave",                  1,),
    12:  KernelInstance(256,     1,   128,   128,    32,   128,   256,  16,  16,  32,   32,    1,    1,     [16, 16, 1],     [16, 16, 1],           1,           1,                   [1, 32, 1,  8],             [8],     "Intrawave",                  1,),
    13:  KernelInstance(256,     1,   128,   128,    32,    64,   256,  16,  16,  16,   16,    2,    1,     [16, 16, 1],     [16, 16, 1],           2,           1,                   [1, 32, 1,  8],             [8],     "Intrawave",                  1,),
    14:  KernelInstance(256,     1,   128,   128,    64,   256,   128,  16,  16,  32,   32,    2,    2,     [ 8, 32, 1],     [ 8, 32, 1],           1,           1,                   [1, 32, 1,  8],             [8],     "Intrawave",                  1,),
    15:  KernelInstance(256,     1,   128,   128,    64,   128,   128,  16,  16,  32,   32,    2,    1,     [ 8, 32, 1],     [ 8, 32, 1],           1,           1,                   [1, 32, 1,  8],             [8],     "Intrawave",                  1,),
    16:  KernelInstance(256,     1,   128,   128,    64,    64,   128,  16,  16,  32,   32,    1,    1,     [ 8, 32, 1],     [ 8, 32, 1],           1,           1,                   [1, 32, 1,  8],             [8],     "Intrawave",                  1,),
    17:  KernelInstance(256,     1,   128,   128,    64,   128,   256,  16,  16,  32,   32,    2,    1,     [16, 16, 1],     [16, 16, 1],           1,           1,                   [1, 32, 1,  8],             [8],     "Intrawave",                  1,),
    18:  KernelInstance(256,     1,   128,   128,    64,    64,   256,  16,  16,  32,   32,    1,    1,     [16, 16, 1],     [16, 16, 1],           1,           1,                   [1, 32, 1,  8],             [8],     "Intrawave",                  1,),

    # NOTE: CK's BlockwiseGemmXdlops_pipeline_v1_ab_scale<Interwave, ...> is
    # NOT specialised -- missing BlockHasHotloop, compile fails. So Interwave
    # is unavailable for the ABScale GEMM path. Don't add Interwave variants.

    # ----- MPerBlock=256 (big-M compute tile, BlockSize=256 -> 4 waves on M). -----
    19:  KernelInstance(256,     1,   128,   128,   256,   128,   128,  16,  16,  32,   32,    4,    2,     [ 8, 32, 1],     [ 8, 32, 1],           1,           2,                   [1, 32, 1,  8],             [8],     "Intrawave",                  3,),
    20:  KernelInstance(256,     1,   128,   128,   256,    64,   128,  16,  16,  32,   32,    4,    1,     [ 8, 32, 1],     [ 8, 32, 1],           1,           1,                   [1, 32, 1,  8],             [8],     "Intrawave",                  3,),
    # (dropped: MPerBlock=256 with v3 + KPerBlock=256 -- violates v3 scaleblocksliceK==1)
    # (dropped: MPerBlock=256, NPerBlock=256, v3 -- violates v3 scaleblocksliceN==1)

    # ----- BlockSize=128 (2 waves / WG -> lower LDS pressure, more WG/CU). -----
    21:  KernelInstance(128,     1,   128,   128,    64,   128,   128,  16,  16,  32,   32,    2,    2,     [ 8, 16, 1],     [ 8, 16, 1],           1,           1,                   [1, 16, 1,  8],             [8],     "Intrawave",                  3,),
    22:  KernelInstance(128,     1,   128,   128,    64,    64,   128,  16,  16,  32,   32,    2,    1,     [ 8, 16, 1],     [ 8, 16, 1],           1,           1,                   [1, 16, 1,  8],             [8],     "Intrawave",                  3,),
    23:  KernelInstance(128,     1,   128,   128,    32,   128,   128,  16,  16,  32,   32,    1,    2,     [ 8, 16, 1],     [ 8, 16, 1],           1,           1,                   [1, 16, 1,  8],             [8],     "Intrawave",                  1,),
    24:  KernelInstance(128,     1,   128,   128,    32,    64,   128,  16,  16,  32,   32,    1,    1,     [ 8, 16, 1],     [ 8, 16, 1],           1,           1,                   [1, 16, 1,  8],             [8],     "Intrawave",                  1,),
}

# ``batched_gemm_fp8_blockscale.cu`` ``batched_blockscale_heuristic_dispatch`` tiles (kernel id 7 is
# the CSV/default entry ``(-1)``). JIT codegen must emit these symbols into the manifest or the
# module fails to compile when the tune file lists only a single default kernel.
FP8_BLOCKSCALE_HEURISTIC_EXTRA_KERNEL_IDS = (10, 15, 0)


default_kernels_dict = {
    # Default fallback for every shape until the autotune CSV is populated.
    # Memory-friendly small-MPerBlock template -- right for the decode hot path
    # which is where the loop-over-B wrapper has the smallest per-batch overhead.
    (-1):KernelInstance(256,     1,   128,   128,    16,   128,   256,  16,  16,  16,   16,    1,    2,     [16, 16, 1],     [16, 16, 1],           1,           2,                   [1, 16, 1, 16],             [8],     "Intrawave",                  1,),
}
# fmt: on
