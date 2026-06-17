// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Tune entry: instantiate every candidate kernel by integer ``kernelId`` so
// the Python tune driver can sweep them and pick the fastest per (B,M,N,K).
// Mirrors ``ck_batched_gemm_a8w8/batched_gemm_a8w8_tune.cu``.

#include "batched_gemm_fp8_blockscale_common.cuh"
#include "batched_gemm_fp8_blockscale_lookup.h"
#include "batched_gemm_fp8_blockscale_manifest.h"
#include <cmath>

using BatchedBlockscaleKernel = std::function<torch::Tensor(
    torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&, int)>;

using BatchedBlockscaleKernelMap = std::unordered_map<int, BatchedBlockscaleKernel>;

template <typename DDataType, typename EDataType>
torch::Tensor batched_gemm_fp8_blockscale_tune(torch::Tensor& XQ,
                                              torch::Tensor& WQ,
                                              torch::Tensor& x_scale,
                                              torch::Tensor& w_scale,
                                              torch::Tensor& Y,
                                              int kernelId,
                                              int splitK) {
    static const BatchedBlockscaleKernelMap lookup{GENERATE_LOOKUP_TABLE(DDataType, EDataType)};
    auto it = lookup.find(kernelId);
    TORCH_CHECK(it != lookup.end(),
                "FP8 block-wise batched GEMM tune: unknown kernelId ", kernelId);
    // splitK == 0 means "no split" (treated as KBatch=1); >= 1 sets that KBatch.
    const int kbatch = splitK <= 1 ? 1 : splitK;
    return it->second(XQ, WQ, x_scale, w_scale, Y, kbatch);
}

torch::Tensor batched_gemm_fp8_blockscale_tune(torch::Tensor& XQ,
                                              torch::Tensor& WQ,
                                              torch::Tensor& x_scale,
                                              torch::Tensor& w_scale,
                                              torch::Tensor& Y,
                                              int kernelId,
                                              int splitK) {
    if (Y.dtype() == at::ScalarType::Half) {
        return batched_gemm_fp8_blockscale_tune<FP32, FP16>(XQ, WQ, x_scale, w_scale, Y, kernelId, splitK);
    } else if (Y.dtype() == at::ScalarType::BFloat16) {
        return batched_gemm_fp8_blockscale_tune<FP32, BF16>(XQ, WQ, x_scale, w_scale, Y, kernelId, splitK);
    }
    TORCH_CHECK(false, "FP8 block-wise batched GEMM tune: unsupported output dtype");
}
