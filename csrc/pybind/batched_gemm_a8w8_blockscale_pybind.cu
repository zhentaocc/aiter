// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "batched_gemm_a8w8_blockscale.h"
#include "rocm_ops.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BATCHED_GEMM_A8W8_BLOCKSCALE_PYBIND;
}
