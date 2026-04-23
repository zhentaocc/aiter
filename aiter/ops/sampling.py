# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
from typing import Optional

from csrc.cpp_itfs.sampling.top_k_renorm_probs import (
    top_k_renorm_probs as top_k_renorm_probs_core,
)
from csrc.cpp_itfs.sampling.top_p_renorm_probs import (
    top_p_renorm_probs as top_p_renorm_probs_core,
)
from csrc.cpp_itfs.sampling.top_p_sampling_from_probs import (
    top_p_sampling_from_probs as top_p_sampling_from_probs_core,
)
from csrc.cpp_itfs.sampling.top_k_top_p_sampling_from_probs import (
    top_k_top_p_sampling_from_probs as top_k_top_p_sampling_from_probs_core,
)
from csrc.cpp_itfs.sampling.top_k_top_p_renorm_probs import (
    top_k_top_p_renorm_probs as top_k_top_p_renorm_probs_core,
)
from csrc.cpp_itfs.sampling.chain_speculative_sampling import (
    chain_speculative_sampling as chain_speculative_sampling_core,
)
from csrc.cpp_itfs.torch_utils import direct_register_custom_op


def top_k_renorm_probs(
    probs: torch.Tensor,
    maybe_top_k_arr: Optional[torch.Tensor],
    top_k_val: int,
) -> torch.Tensor:
    return top_k_renorm_probs_core(
        probs,
        maybe_top_k_arr,
        top_k_val,
    )


direct_register_custom_op(
    "top_k_renorm_probs",
    top_k_renorm_probs,
    [],
)


def top_p_renorm_probs(
    probs: torch.Tensor,
    maybe_top_p_arr: Optional[torch.Tensor],
    top_p_val: float,
) -> torch.Tensor:
    return top_p_renorm_probs_core(
        probs,
        maybe_top_p_arr,
        top_p_val,
    )


direct_register_custom_op(
    "top_p_renorm_probs",
    top_p_renorm_probs,
    [],
)


def top_p_sampling_from_probs(
    probs: torch.Tensor,
    indices: torch.Tensor,
    maybe_top_p_arr: Optional[torch.Tensor],
    top_p_val: float,
    deterministic: bool = False,
) -> torch.Tensor:
    return top_p_sampling_from_probs_core(
        probs,
        indices,
        maybe_top_p_arr,
        top_p_val,
        deterministic,
    )


direct_register_custom_op(
    "top_p_sampling_from_probs",
    top_p_sampling_from_probs,
    [],
)


def top_k_top_p_sampling_from_probs(
    probs: torch.Tensor,
    indices: torch.Tensor,
    maybe_top_k_arr: Optional[torch.Tensor],
    top_k_val: int,
    maybe_top_p_arr: Optional[torch.Tensor],
    top_p_val: float,
    deterministic: bool = False,
) -> torch.Tensor:
    return top_k_top_p_sampling_from_probs_core(
        probs,
        indices,
        maybe_top_k_arr,
        top_k_val,
        maybe_top_p_arr,
        top_p_val,
        deterministic,
    )


direct_register_custom_op(
    "top_k_top_p_sampling_from_probs",
    top_k_top_p_sampling_from_probs,
    [],
)


def top_k_top_p_renorm_probs(
    probs: torch.Tensor,
    maybe_top_k_arr: Optional[torch.Tensor],
    top_k_val: int,
    maybe_top_p_arr: Optional[torch.Tensor],
    top_p_val: float,
) -> torch.Tensor:
    return top_k_top_p_renorm_probs_core(
        probs,
        maybe_top_k_arr,
        top_k_val,
        maybe_top_p_arr,
        top_p_val,
    )


direct_register_custom_op(
    "top_k_top_p_renorm_probs",
    top_k_top_p_renorm_probs,
    [],
)


def chain_speculative_sampling(
    candidates: torch.Tensor,
    target_probs: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_for_final_sampling: torch.Tensor,
    threshold_single: float,
    threshold_acc: float,
    deterministic: bool = True,
) -> torch.Tensor:
    return chain_speculative_sampling_core(
        candidates,
        target_probs,
        uniform_samples,
        uniform_samples_for_final_sampling,
        threshold_single,
        threshold_acc,
        deterministic,
    )


direct_register_custom_op(
    "chain_speculative_sampling",
    chain_speculative_sampling,
    [],
)
