# SPDX-License-Identifier: MIT
# Copyright (C) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.


from jinja2 import Template
from csrc.cpp_itfs.utils import compile_template_op, AITER_CORE_DIR, str_to_bool
import math

MD_NAME = "chain_speculative_sampling"

with open(
    f"{AITER_CORE_DIR}/csrc/cpp_itfs/sampling/chain_speculative_sampling.cpp.jinja",
    "r",
) as f:
    src_template = Template(f.read())


def compile(
    vec_size: int,
    deterministic: bool,
    folder: str = None,
):
    return compile_template_op(
        src_template,
        MD_NAME,
        [
            f"{AITER_CORE_DIR}/csrc/cpp_itfs/utils.h",
            f"{AITER_CORE_DIR}/csrc/cpp_itfs/sampling/sampling.cuh",
            f"{AITER_CORE_DIR}/csrc/cpp_itfs/sampling/vec_dtypes.cuh",
        ],
        vec_size=vec_size,
        deterministic=deterministic,
        folder=folder,
    )


def chain_speculative_sampling(
    candidates,
    target_probs,
    uniform_samples,
    uniform_samples_for_final_sampling,
    threshold_single,
    threshold_acc,
    deterministic=True,
):
    import torch
    from csrc.cpp_itfs.torch_utils import torch_to_c_types

    target_probs = target_probs.float()
    uniform_samples = uniform_samples.float()
    uniform_samples_for_final_sampling = uniform_samples_for_final_sampling.float()
    threshold_single = float(threshold_single)
    threshold_acc = max(float(threshold_acc), 1e-9)

    batch_size = candidates.size(0)
    draft_len = candidates.size(1)
    vocab_size = target_probs.size(2)
    vec_size = math.gcd(16 // target_probs.element_size(), vocab_size)

    accept_length = torch.empty(batch_size, dtype=torch.int32, device=candidates.device)
    bonus_token_ids = torch.empty(batch_size, dtype=torch.int32, device=candidates.device)

    func = compile(vec_size, deterministic)
    (
        accept_length_ptr,
        bonus_token_ids_ptr,
        candidates_ptr,
        target_probs_ptr,
        uniform_samples_ptr,
        uniform_samples_final_ptr,
        batch_size,
        draft_len,
        vocab_size,
        threshold_single,
        threshold_acc,
        stream,
    ) = torch_to_c_types(
        accept_length,
        bonus_token_ids,
        candidates,
        target_probs,
        uniform_samples,
        uniform_samples_for_final_sampling,
        batch_size,
        draft_len,
        vocab_size,
        threshold_single,
        threshold_acc,
        torch.cuda.current_stream(),
    )
    func(
        accept_length_ptr,
        bonus_token_ids_ptr,
        candidates_ptr,
        target_probs_ptr,
        uniform_samples_ptr,
        uniform_samples_final_ptr,
        batch_size,
        draft_len,
        vocab_size,
        threshold_single,
        threshold_acc,
        stream,
    )
    return accept_length, bonus_token_ids


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, required=True)
    parser.add_argument("--deterministic", type=str_to_bool, required=True)
    parser.add_argument("--folder", type=str, default=None)
    args = parser.parse_args()
    compile(**vars(args))
