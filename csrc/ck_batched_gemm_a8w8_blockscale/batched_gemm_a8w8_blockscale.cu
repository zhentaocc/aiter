// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#include <cmath>
#include <climits>
#include <functional>
#include <unordered_map>

#include <torch/extension.h>

#include "batched_gemm_a8w8_blockscale_common.cuh"
#include "batched_gemm_a8w8_blockscale_lookup.h"
#include "batched_gemm_a8w8_blockscale_manifest.h"

using BatchedBlockscaleKernel = std::function<torch::Tensor(
    torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&)>;

// (B, M, N, K) -> kernel
struct IntTupleHash4 {
    size_t operator()(const std::tuple<int, int, int, int>& t) const {
        auto h1 = std::hash<int>{}(std::get<0>(t));
        auto h2 = std::hash<int>{}(std::get<1>(t));
        auto h3 = std::hash<int>{}(std::get<2>(t));
        auto h4 = std::hash<int>{}(std::get<3>(t));
        return ((h1 ^ (h2 << 1)) ^ (h3 << 2)) ^ (h4 << 3);
    }
};

using BatchedBlockscaleKernelMap =
    std::unordered_map<std::tuple<int, int, int, int>, BatchedBlockscaleKernel, IntTupleHash4>;

static constexpr int nextPow2(unsigned int num) {
    if (num <= 1) return 1;
    return 1 << (CHAR_BIT * sizeof(num) - __builtin_clz(num - 1));
}

template <typename DDataType, typename EDataType = DDataType>
static BatchedBlockscaleKernel batched_blockscale_heuristic_dispatch(int B, int M, int N, int K) {
    // Fallback for shapes absent from the tuned lookup table (the primary path).
    // The M bands below are taken from the autotune sweep on the DSv4 wo_a shapes
    // (aiter/configs/a8w8_blockscale_tuned_batched_gemm.csv; N=1024, K=4096,
    // B in {1,2,8,16}): MPerBlock grows with M as the regime shifts memory- to
    // compute-bound. Per-batch dispatch cost is ~constant in B, so the rule is
    // M-driven only. All tiles use KPerBlock=128 so the fallback stays valid for
    // any K % 128 == 0 (not just the tuned K=4096).
    if (M <= 256) {
        // Decode + small prefill: MPerBlock=16 was the measured winner for every
        // tuned M <= 256 (kernel id 6 here; id 8 is the same tile with KPerBlock
        // =256, used by tuning at K=4096 but not K-general).
        return a8w8_batched_blockscale_1x128x128_256x16x64x128_8x16_16x16_1x1_16x16x1_8x32x1_1x16x1x16_4_1x1_intrawave_v1<DDataType, EDataType>;
    }
    if (M <= 1024) {
        // Mid prefill: MPerBlock=64 / NPerBlock=64, intrawave v3 (tuned best at
        // M=512, 1024).  kernel id 3.
        return a8w8_batched_blockscale_1x128x128_256x64x64x128_16x16_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DDataType, EDataType>;
    }
    if (M <= 2048) {
        // MPerBlock=64 / NPerBlock=128, intrawave v3 (tuned best at M=2048).
        // kernel id 2.
        return a8w8_batched_blockscale_1x128x128_256x64x128x128_16x16_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DDataType, EDataType>;
    }
    // Large-M prefill: full 128x128 compute tile, intrawave v3 (tuned best at
    // M=4096).  kernel id 0.
    return a8w8_batched_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DDataType, EDataType>;
}

template <typename DDataType, typename EDataType = DDataType>
static BatchedBlockscaleKernel batched_blockscale_dispatch(int B, int M, int N, int K) {
    static const auto lookup = [] {
        if constexpr (std::is_same_v<EDataType, FP16>) {
            return BatchedBlockscaleKernelMap{GENERATE_LOOKUP_TABLE(DDataType, FP16)};
        } else if constexpr (std::is_same_v<EDataType, BF16>) {
            return BatchedBlockscaleKernelMap{GENERATE_LOOKUP_TABLE(DDataType, BF16)};
        } else {
            static_assert(false, "batched_blockscale_dispatch used with unsupported dtype!");
        }
    }();

    // Exact (B, M, N, K) lookup first.
    auto it = lookup.find({B, M, N, K});
    if (it != lookup.end()) return it->second;

    // Bucket M by next pow-of-2 (kernels are padded to MPerBlock anyway).
    int padded_m = M;
    if (M > 1 && M <= 16) padded_m = 16;
    else if (M <= 16384) padded_m = nextPow2(M);

    it = lookup.find({B, padded_m, N, K});
    if (it != lookup.end()) return it->second;

    return batched_blockscale_heuristic_dispatch<DDataType, EDataType>(B, M, N, K);
}

torch::Tensor batched_gemm_a8w8_blockscale(torch::Tensor& XQ,
                                         torch::Tensor& WQ,
                                         torch::Tensor& x_scale,
                                         torch::Tensor& w_scale,
                                         torch::Tensor& Y) {
    TORCH_CHECK(XQ.dtype() == WQ.dtype(),
                "FP8 block-wise: weight and activation dtypes must match");
    TORCH_CHECK(XQ.dtype() == at::ScalarType::Float8_e4m3fn ||
                XQ.dtype() == at::ScalarType::Float8_e4m3fnuz,
                "FP8 block-wise: input must be fp8_e4m3fn(uz)");
    TORCH_CHECK(x_scale.dtype() == w_scale.dtype(),
                "FP8 block-wise: scale dtypes must match");
    TORCH_CHECK(x_scale.dtype() == at::ScalarType::Float,
                "FP8 block-wise: scales must be fp32");

    const int B = XQ.size(0);
    const int M = XQ.size(1);
    const int N = WQ.size(1);
    const int K = XQ.size(2);

    if (Y.dtype() == at::ScalarType::Half) {
        batched_blockscale_dispatch<FP32, FP16>(B, M, N, K)(XQ, WQ, x_scale, w_scale, Y);
    } else if (Y.dtype() == at::ScalarType::BFloat16) {
        batched_blockscale_dispatch<FP32, BF16>(B, M, N, K)(XQ, WQ, x_scale, w_scale, Y);
    } else {
        TORCH_CHECK(false, "FP8 block-wise: unsupported output dtype (use fp16 or bf16)");
    }
    return Y;
}
