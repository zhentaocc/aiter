#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#include <torch/all.h>
#include <torch/extension.h>

// Block-wise FP8 batched GEMM matching DeepGEMM's
//   fp8_einsum("bmk,bnk->bmn", (A, A_scale), (W, W_scale), out, recipe=(1, 1, 128))
//
// Inputs:
//   XQ      [B, M, K] fp8_e4m3fn
//   WQ      [B, N, K] fp8_e4m3fn
//   x_scale [B, M, K/128] fp32     -- per-row, per-128k-block
//   w_scale [B, N/128, K/128] fp32 -- per (128n, 128k) block
//   Y       [B, M, N] bf16 (or fp16)
//
// K and N must be multiples of 128.
torch::Tensor batched_gemm_a8w8_blockscale(torch::Tensor& XQ,
                                         torch::Tensor& WQ,
                                         torch::Tensor& x_scale,
                                         torch::Tensor& w_scale,
                                         torch::Tensor& Y);

torch::Tensor batched_gemm_a8w8_blockscale_tune(torch::Tensor& XQ,
                                              torch::Tensor& WQ,
                                              torch::Tensor& x_scale,
                                              torch::Tensor& w_scale,
                                              torch::Tensor& Y,
                                              int kernelId,
                                              int splitK);
