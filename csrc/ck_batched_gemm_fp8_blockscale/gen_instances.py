# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Codegen for CK FP8 block-wise *batched* GEMM instances.

Mirrors ``ck_gemm_a8w8_blockscale/gen_instances.py``.  Differences:

  * Lookup keys are 4-tuples ``(B, M, N, K)`` instead of 3-tuples ``(M, N, K)``.
  * Per-instance ``__forceinline__`` wrapper calls
    ``batched_gemm_fp8_blockscale_impl`` (host-side B-loop) instead of
    ``gemm_a8w8_blockscale_impl``.
  * Manifest signature is the batched signature from
    ``include/batched_gemm_fp8_blockscale.h``.
"""

import argparse
import os
import shutil
from pathlib import Path

import pandas as pd
import torch

from batched_gemm_fp8_blockscale_instance import (
    KernelInstance,
    FP8_BLOCKSCALE_HEURISTIC_EXTRA_KERNEL_IDS,
    candidate_kernels_dict,
    default_kernels_dict,
)


def _unique_instances_by_name(kernels_dict: dict) -> list:
    seen: set[str] = set()
    out = []
    for k in kernels_dict.values():
        if k.name not in seen:
            seen.add(k.name)
            out.append(k)
    return out


class batched_gemm_fp8_blockscale_codegen:
    def __init__(self, working_path: str, istune: bool = False, tune_file: str | None = None):
        self.working_path = working_path
        if not os.path.exists(working_path):
            os.makedirs(working_path)
        self.impl_path = os.path.join(working_path, "impl")
        self.instances_path = os.path.join(working_path, "instances")
        self.istune = istune
        self.tune_file = tune_file

    # ------------------------------------------------------------------
    # Tune CSV loader.  Mirrors gemm_a8w8_blockscale's get_tune_dict, but
    # the CSV key is (B, M, N, K) instead of (M, N, K).
    # ------------------------------------------------------------------
    def get_tune_dict(self, tune_dict_csv: str):
        tune_dict = dict(default_kernels_dict)
        if tune_dict_csv and os.path.exists(tune_dict_csv):
            tune_df = pd.read_csv(tune_dict_csv)
            if torch.cuda.is_available():
                gpu = torch.cuda.current_device()
                cu_num = torch.cuda.get_device_properties(gpu).multi_processor_count
                tune_df = tune_df[
                    (tune_df["cu_num"] == cu_num) & (tune_df["libtype"] == "ck")
                ].reset_index(drop=True)
            for i in range(len(tune_df)):
                B = int(tune_df.loc[i, "B"])
                M = int(tune_df.loc[i, "M"])
                N = int(tune_df.loc[i, "N"])
                K = int(tune_df.loc[i, "K"])
                kid = int(tune_df.loc[i, "kernelId"])
                if kid in candidate_kernels_dict:
                    tune_dict[(B, M, N, K)] = candidate_kernels_dict[kid]
                else:
                    print(f"[gen_instances] kernelId {kid} missing for ({B},{M},{N},{K})")
        return tune_dict

    # ------------------------------------------------------------------
    # Per-kernel codegen.
    # ------------------------------------------------------------------
    def gen_ck_instance(self, k: KernelInstance) -> None:
        INSTANCE_IMPL = f"""// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "batched_gemm_fp8_blockscale_common.cuh"

enum class GemmSpecialization {{
    Default    = 0,
    MPadding   = 1,
    NPadding   = 2,
    KPadding   = 3,
    MNPadding  = 4,
    MKPadding  = 5,
    NKPadding  = 6,
    MNKPadding = 7
}};

static const std::unordered_map<std::string, GemmSpecialization> g_gemm_spec_names{{
    {{"",    GemmSpecialization::Default}},
    {{"M",   GemmSpecialization::MPadding}},
    {{"N",   GemmSpecialization::NPadding}},
    {{"K",   GemmSpecialization::KPadding}},
    {{"MN",  GemmSpecialization::MNPadding}},
    {{"MK",  GemmSpecialization::MKPadding}},
    {{"NK",  GemmSpecialization::NKPadding}},
    {{"MNK", GemmSpecialization::MNKPadding}}
}};

static GemmSpecialization GetGemmSpec(int64_t m, int64_t n, int64_t k,
                                      int64_t mb, int64_t nb, int64_t kb) {{
    auto Ceil = [](int64_t x, int64_t y) {{ return (x + y - 1) / y; }};
    std::string spec;
    if (Ceil(m, mb) * mb - m != 0) spec += "M";
    if (Ceil(n, nb) * nb - n != 0) spec += "N";
    if (Ceil(k, kb) * kb - k != 0) spec += "K";
    return g_gemm_spec_names.at(spec);
}}

template <typename DDataType, typename EDataType>
torch::Tensor
{k.name}(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y)
{{
    const int M = XQ.size(1);
    const int N = WQ.size(1);
    const int K = XQ.size(2);

    auto gemm_spec = GetGemmSpec(M, N, K, {k.MPerBLOCK}, {k.NPerBLOCK}, {k.KPerBLOCK});

    if (gemm_spec == GemmSpecialization::Default)        {{ __INSTANCE_DEFAULT__ }}
    else if (gemm_spec == GemmSpecialization::MPadding)  {{ __INSTANCE_MPAD__ }}
    else if (gemm_spec == GemmSpecialization::NPadding)  {{ __INSTANCE_NPAD__ }}
    else if (gemm_spec == GemmSpecialization::KPadding)  {{ __INSTANCE_KPAD__ }}
    else if (gemm_spec == GemmSpecialization::MNPadding) {{ __INSTANCE_MNPAD__ }}
    else if (gemm_spec == GemmSpecialization::MKPadding) {{ __INSTANCE_MKPAD__ }}
    else if (gemm_spec == GemmSpecialization::NKPadding) {{ __INSTANCE_NKPAD__ }}
    else if (gemm_spec == GemmSpecialization::MNKPadding){{ __INSTANCE_MNKPAD__ }}
    else {{ throw std::runtime_error("Unsupported GemmSpecialization!"); }}
}}
"""

        INSTANCE_BODY = f"""using GemmInstance = DeviceGemmHelperF8BlockScalePerBatch<
            DDataType, EDataType,
            {k.BLOCK_SIZE},
            {k.ScaleBlockM}, {k.ScaleBlockN}, {k.ScaleBlockK},
            {k.MPerBLOCK}, {k.NPerBLOCK}, {k.KPerBLOCK},
            {k.AK1}, {k.BK1},
            {k.MPerXDL}, {k.NPerXDL},
            {k.WAVE_MAP_M}, {k.WAVE_MAP_N},
            S<{(", ").join(map(str, k.ABLOCK_TRANSFER))}>,
            S<{(", ").join(map(str, k.BBLOCK_TRANSFER))}>,
            {k.CSHUFFLE_MX_PER_WAVE_PERSHUFFLE},
            {k.CSHUFFLE_NX_PER_WAVE_PERSHUFFLE},
            S<{(", ").join(map(str, k.CBLOCK_TRANSFER))}>,
            S<{(", ").join(map(str, k.CBLOCK_SPV))}>,
            ck::BlockGemmPipelineScheduler::{k.PIPELINE_Sched},
            ck::BlockGemmPipelineVersion::v{k.PIPELINE_VERSION},
            ck::tensor_operation::device::GemmSpecialization::{{GemmSpec}}>;

        return batched_gemm_fp8_blockscale_impl<DDataType, EDataType, GemmInstance>(
            XQ, WQ, x_scale, w_scale, Y);
"""

        IMPL_str = (
            INSTANCE_IMPL
            .replace("__INSTANCE_DEFAULT__", INSTANCE_BODY.replace("{GemmSpec}", "Default"))
            .replace("__INSTANCE_MPAD__",    INSTANCE_BODY.replace("{GemmSpec}", "MPadding"))
            .replace("__INSTANCE_NPAD__",    INSTANCE_BODY.replace("{GemmSpec}", "NPadding"))
            .replace("__INSTANCE_KPAD__",    INSTANCE_BODY.replace("{GemmSpec}", "KPadding"))
            .replace("__INSTANCE_MNPAD__",   INSTANCE_BODY.replace("{GemmSpec}", "MNPadding"))
            .replace("__INSTANCE_MKPAD__",   INSTANCE_BODY.replace("{GemmSpec}", "MKPadding"))
            .replace("__INSTANCE_NKPAD__",   INSTANCE_BODY.replace("{GemmSpec}", "NKPadding"))
            .replace("__INSTANCE_MNKPAD__",  INSTANCE_BODY.replace("{GemmSpec}", "MNKPadding"))
        )

        Path(os.path.join(self.impl_path, f"{k.name}.cuh")).write_text(IMPL_str)

        INSTANCE_template = """// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "impl/{name}.cuh"

template torch::Tensor
{name}<{dtypes}>(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y);
"""
        Path(os.path.join(self.instances_path, f"{k.name}_dFP32_eBF16.cpp")).write_text(
            INSTANCE_template.format(name=k.name, dtypes="FP32, BF16")
        )
        Path(os.path.join(self.instances_path, f"{k.name}_dFP32_eFP16.cpp")).write_text(
            INSTANCE_template.format(name=k.name, dtypes="FP32, FP16")
        )

    # ------------------------------------------------------------------
    # Lookup table generator (4-tuple key).
    # ------------------------------------------------------------------
    def gen_lookup_dict(self, kernels_dict):
        LOOKUP_head = """#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#ifdef USE_ROCM

#define GENERATE_LOOKUP_TABLE(DTYPE, ETYPE)                                                   \\
   {                                                                                          \\"""
        LOOKUP_template = """
       {{{BMNK}, {kernel_name}<DTYPE, ETYPE>}},                                                \\"""
        LOOKUP_end = """
   }

#endif // USE_ROCM
"""
        with open(os.path.join(self.working_path, "batched_gemm_fp8_blockscale_lookup.h"), "w") as f:
            f.write(LOOKUP_head)
            for bmnk, k in kernels_dict.items():
                if not self.istune and isinstance(bmnk, tuple) and bmnk[0] > 0:
                    f.write(LOOKUP_template.format(
                        BMNK="{" + (", ").join(map(str, list(bmnk))) + "}",
                        kernel_name=k.name,
                    ))
                elif self.istune and isinstance(bmnk, int):
                    f.write(LOOKUP_template.format(BMNK=bmnk, kernel_name=k.name))
            f.write(LOOKUP_end)

    # ------------------------------------------------------------------
    # Manifest header: forward-declare every kernel symbol so the
    # dispatcher's lookup table compiles.
    # ------------------------------------------------------------------
    def gen_manifest_head(self, kernels_dict):
        MANIFEST_head = """#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#ifdef USE_ROCM

#include <cstdlib>
#include <torch/extension.h>
"""
        MANIFEST_template = """
template <typename DDataType, typename EDataType>
torch::Tensor
{kernel_name}(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y);
"""
        MANIFEST_end = """

#endif // USE_ROCM
"""
        with open(os.path.join(self.working_path, "batched_gemm_fp8_blockscale_manifest.h"), "w") as f:
            f.write(MANIFEST_head)
            for k in _unique_instances_by_name(kernels_dict):
                f.write(MANIFEST_template.format(kernel_name=k.name))
            f.write(MANIFEST_end)

    def gen_code(self, kernels_dict: dict) -> None:
        for k in _unique_instances_by_name(kernels_dict):
            self.gen_ck_instance(k)
        self.gen_lookup_dict(kernels_dict)
        self.gen_manifest_head(kernels_dict)

    def run(self) -> None:
        for path in (self.impl_path, self.instances_path):
            if os.path.exists(path):
                shutil.rmtree(path)
            os.mkdir(path)
        if self.istune:
            self.gen_code(candidate_kernels_dict)
        else:
            kernels_dict = self.get_tune_dict(self.tune_file)
            # Heuristic dispatcher references multiple CK tiles; merge them into the JIT blob when the
            # tune CSV does not already pull them in (see FP8_BLOCKSCALE_HEURISTIC_EXTRA_KERNEL_IDS).
            seen_names = {k.name for k in kernels_dict.values()}
            for idx, kid in enumerate(FP8_BLOCKSCALE_HEURISTIC_EXTRA_KERNEL_IDS):
                k = candidate_kernels_dict[kid]
                if k.name not in seen_names:
                    kernels_dict[(0, -(idx + 1), 0, 0)] = k
                    seen_names.add(k.name)
            self.gen_code(kernels_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate",
        description="Codegen for CK FP8 block-wise batched GEMM",
    )
    parser.add_argument("-w", "--working_path", default="./", required=False)
    parser.add_argument(
        "-f", "--tune_file",
        default="aiter/configs/fp8_blockscale_tuned_batched_gemm.csv",
        required=False,
    )
    parser.add_argument("--tune", action="store_true", required=False)
    args = parser.parse_args()
    batched_gemm_fp8_blockscale_codegen(args.working_path, args.tune, args.tune_file).run()
