#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// CK template wrapper for FP8 block-wise *batched* GEMM.
//
// Composable Kernel ships ``DeviceGemmMultiD_ABScale_Xdl_CShuffle_V3`` (the
// non-batched FP8 AB-scale device op used by ``ck_gemm_a8w8_blockscale``)
// but does NOT ship a ``DeviceBatchedGemmMultiD_ABScale_Xdl_CShuffle_V3``
// equivalent.  We therefore implement the B dimension as an outer loop in
// the host wrapper -- one ``MakeArgument`` + ``invoker.Run`` per batch slice
// on the same hipStream.  For the wo_a use case (B = num_groups <= 16) this
// adds ~B * a-few-microseconds of dispatch overhead, which is small relative
// to the per-call HBM traffic for V4-Flash decode shapes (verified by Triton
// microbench: 80 us / call at B=8 M=1).
//
// When CK adds a true batched ABScale device op, swap ``GemmInstance`` with
// the batched variant and remove the loop -- nothing else needs to change.

#ifdef USE_ROCM

#undef __HIP_NO_HALF_OPERATORS__
#undef __HIP_NO_HALF_CONVERSIONS__

#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <numeric>

#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include <torch/extension.h>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3_ab_scale.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"

#include "ck/utility/blkgemmpipe_scheduler.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using BF16 = ck::bhalf_t;
using FP8  = ck::f8_t;
using FP32 = float;
using FP16 = ck::half_t;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using A0DataType       = FP8;
using A1DataType       = FP32;
using B0DataType       = FP8;
using B1DataType       = FP32;
using AccDataType      = FP32;
using CShuffleDataType = FP32;
using DsDataType       = ck::Tuple<>;

using A0Layout = Row;
using B0Layout = Col;
using DsLayout = ck::Tuple<>;
using ELayout  = Row;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using AElementOp   = PassThrough;
using BElementOp   = PassThrough;
using CDEElementOp = PassThrough;

// ---------------------------------------------------------------------------
// Device op alias.  Identical signature to ``DeviceLegacyGemmHelperF8BlockScale``
// in ``ck_gemm_a8w8_blockscale/include/gemm_a8w8_blockscale_common.cuh``;
// kept locally here so we don't take a transitive build dep on that module.
// ---------------------------------------------------------------------------

template <typename AB1DataType,
          typename EDataType,
          ck::index_t BlockSize,
          ck::index_t Scale_Block_M,
          ck::index_t Scale_Block_N,
          ck::index_t Scale_Block_K,
          ck::index_t MPerBlock,
          ck::index_t NPerBlock,
          ck::index_t KPerBlock,
          ck::index_t AK1,
          ck::index_t BK1,
          ck::index_t MPerXDL,
          ck::index_t NPerXDL,
          ck::index_t MXdlPerWave,
          ck::index_t NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          ck::index_t CSHUFFLE_MX_PER_WAVE_PERSHUFFLE,
          ck::index_t CSHUFFLE_NX_PER_WAVE_PERSHUFFLE,
          typename CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          typename CDEShuffleBlockTransferScalarPerVectors,
          ck::BlockGemmPipelineScheduler BlkGemmPipeSched =
              ck::BlockGemmPipelineScheduler::Intrawave,
          ck::BlockGemmPipelineVersion BlkGemmPipelineVer = ck::BlockGemmPipelineVersion::v1,
          auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::Default>
using DeviceGemmHelperF8BlockScalePerBatch =
    ck::tensor_operation::device::DeviceGemmMultiD_ABScale_Xdl_CShuffle_V3
    // clang-format off
         <A0Layout, B0Layout, DsLayout, ELayout,
          A0DataType, AB1DataType, B0DataType, AB1DataType, DsDataType, EDataType, AccDataType, CShuffleDataType,
          AElementOp,  BElementOp, CDEElementOp, GemmSpec,
          BlockSize, Scale_Block_M, Scale_Block_N, Scale_Block_K,
          MPerBlock, NPerBlock, KPerBlock,
          AK1, BK1,
          MPerXDL, NPerXDL,
          MXdlPerWave, NXdlPerWave,
          ABlockTransferThreadClusterLengths_AK0_M_AK1,
          S<1, 0, 2>, S<1, 0, 2>,
          2, AK1, AK1, 0,
          BBlockTransferThreadClusterLengths_BK0_N_BK1,
          S<1, 0, 2>, S<1, 0, 2>,
          2, BK1, BK1, 0,
          CSHUFFLE_MX_PER_WAVE_PERSHUFFLE,
          CSHUFFLE_NX_PER_WAVE_PERSHUFFLE,
          CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          CDEShuffleBlockTransferScalarPerVectors,
          BlkGemmPipeSched,
          BlkGemmPipelineVer, A0DataType>;
// clang-format on

// ---------------------------------------------------------------------------
// Host wrapper: loop over B and dispatch the per-batch invoker.
//
// Layout assumptions (matches the contract in batched_gemm_fp8_blockwise.h):
//   XQ        [B, M, K] fp8_e4m3fn   row-major,           strides {M*K, K, 1}
//   WQ        [B, N, K] fp8_e4m3fn   N-major-then-K,      strides {N*K, K, 1}
//                                    (CK B0Layout = Col means the [N, K] view
//                                    is treated as KxN col-major, which is
//                                    bit-identical to NxK row-major in memory)
//   x_scale   [B, M, K/128] fp32     row-major,           strides {M*Kg, Kg, 1}
//   w_scale   [B, N/128, K/128] fp32 row-major,           strides {Ng*Kg, Kg, 1}
//   Y         [B, M, N] bf16         row-major,           strides {M*N, N, 1}
// ---------------------------------------------------------------------------

template <typename DDataType, typename EDataType, typename GemmInstance>
__forceinline__ torch::Tensor batched_gemm_fp8_blockwise_impl(torch::Tensor& XQ,
                                                              torch::Tensor& WQ,
                                                              torch::Tensor& x_scale,
                                                              torch::Tensor& w_scale,
                                                              torch::Tensor& Y)
{
    const int B = XQ.size(0);
    const int M = XQ.size(1);
    const int K = XQ.size(2);
    const int N = WQ.size(1);

    TORCH_CHECK(WQ.size(0) == B && WQ.size(2) == K, "WQ shape mismatch");
    TORCH_CHECK(Y.size(0) == B && Y.size(1) == M && Y.size(2) == N, "Y shape mismatch");
    TORCH_CHECK(K % 128 == 0, "K must be a multiple of 128");
    TORCH_CHECK(N % 128 == 0, "N must be a multiple of 128");

    const int Kg = K / 128;
    const int Ng = N / 128;
    TORCH_CHECK(x_scale.size(0) == B && x_scale.size(1) == M && x_scale.size(2) == Kg,
                "x_scale shape mismatch");
    TORCH_CHECK(w_scale.size(0) == B && w_scale.size(1) == Ng && w_scale.size(2) == Kg,
                "w_scale shape mismatch");

    const int StrideA = K;  // per-batch row stride
    const int StrideB = K;
    const int StrideE = N;

    // Per-batch byte offsets for the loop.
    const std::size_t xq_b_stride       = static_cast<std::size_t>(M) * K;
    const std::size_t wq_b_stride       = static_cast<std::size_t>(N) * K;
    const std::size_t x_scale_b_stride  = static_cast<std::size_t>(M) * Kg;
    const std::size_t w_scale_b_stride  = static_cast<std::size_t>(Ng) * Kg;
    const std::size_t y_b_stride        = static_cast<std::size_t>(M) * N;

    auto* xq_ptr      = static_cast<A0DataType*>(XQ.data_ptr());
    auto* wq_ptr      = static_cast<B0DataType*>(WQ.data_ptr());
    auto* x_scale_ptr = static_cast<DDataType*>(x_scale.data_ptr());
    auto* w_scale_ptr = static_cast<DDataType*>(w_scale.data_ptr());
    auto* y_ptr       = reinterpret_cast<EDataType*>(Y.data_ptr());

    auto a_element_op   = AElementOp{};
    auto b_element_op   = BElementOp{};
    auto cde_element_op = CDEElementOp{};

    constexpr ck::index_t NumDTensor = DsDataType::Size();

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(XQ));
    auto device_gemm = GemmInstance{};
    auto invoker     = device_gemm.MakeInvoker();
    auto stream      = at::hip::getCurrentHIPStream();

    for (int b = 0; b < B; ++b) {
        auto argument = device_gemm.MakeArgument(
            xq_ptr + b * xq_b_stride,
            wq_ptr + b * wq_b_stride,
            std::array<const void*, NumDTensor>{},
            y_ptr + b * y_b_stride,
            M, N, K,
            StrideA, StrideB,
            std::array<ck::index_t, NumDTensor>{},
            StrideE,
            x_scale_ptr + b * x_scale_b_stride,
            w_scale_ptr + b * w_scale_b_stride,
            a_element_op, b_element_op, cde_element_op);

        TORCH_CHECK(device_gemm.IsSupportedArgument(argument),
                    "FP8 block-wise batched GEMM: unsupported argument for tile config");

        invoker.Run(argument, StreamConfig{stream});
    }

    return Y;
}

#endif // USE_ROCM
