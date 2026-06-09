# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Monkey-patch vllm's DeepSeek V4 ``DeepseekV4MLAAttention.forward_o_proj`` so
that on ROCm it uses ``aiter.batched_gemm_fp8_blockwise`` instead of the
current BF16 reference fallback.

vllm main (post PR #40860) carries this comment in
``vllm/model_executor/layers/deepseek_v4_attention.py`` around line 311::

    # Keep ROCm on the BF16 reference wo_a path util kernel ready.
    if current_platform.is_rocm():
        z = rocm_inv_rope_einsum(...)
        return self.wo_b(z.flatten(1))

This module is the "kernel ready" delta.  When imported (e.g. via
``--load-format auto`` plus a ``VLLM_PLUGINS`` entry) it patches that
ROCm branch to do FP8 per-row activation quant + the AITER blockwise
batched GEMM.

Activation:
    VLLM_DSV4_WO_A_AITER=1   # any truthy value enables the patch

Designed to run inside the ``rocm/atom-dev:vllm-latest`` docker image
where vllm + aiter are already installed.  See
``run_dsv4_wo_a_bench_in_docker.sh`` for the wrapper.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch

logger = logging.getLogger("vllm_dsv4_wo_a_aiter_patch")


def _enabled() -> bool:
    return os.environ.get("VLLM_DSV4_WO_A_AITER", "0") not in ("0", "", "false", "False")


def _patched_rocm_wo_a(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """
    Drop-in replacement for the ROCm BF16 reference path.

    Steps:
      1. Inverse-RoPE on the rope head dim (reuse vllm's existing fused op).
      2. Per-row FP8 quant on the (T*G, D) flattened activation (group_size=128).
      3. ``aiter.batched_gemm_fp8_blockwise`` on (G, T, D) x (G, R, D) -> (G, T, R).
      4. Transpose back to (T, G, R) and feed to wo_b.
    """
    import aiter

    # vllm's wo_a stores its weight as fp8_e4m3fn with weight_scale_inv at
    # post_load_weights time (mirrors sglang's _setup_fp8_wo_a_scales).  We
    # re-shape into the (G, R, D) layout that batched_gemm_fp8_blockwise expects.
    G = self.n_local_groups
    R = self.o_lora_rank
    D = self.head_dim
    T = o.shape[0]

    # Step 1: inverse RoPE on the rope dim (in-place on the rope slice).
    # Reuse vllm's existing helper to stay numerically identical to the BF16 path.
    from vllm.model_executor.layers.deepseek_v4_attention import rocm_inv_rope_apply_only
    o = rocm_inv_rope_apply_only(
        self.rotary_emb,
        o,
        positions,
        self.rope_head_dim,
        self.n_local_groups,
    )

    # Step 2: per-row, per-128-block FP8 activation quant.  Reuse vllm's QuantFP8.
    # vllm's own NV path uses fused_inv_rope_fp8_quant which fuses 1+2; on ROCm
    # we keep them split for now (small overhead on small T).
    a_fp8, a_scale = self._wo_a_act_quant(
        o.reshape(T * G, D).contiguous()
    )
    a_fp8 = a_fp8.view(T, G, D).transpose(0, 1).contiguous()        # [G, T, D]
    a_scale = a_scale.view(T, G, D // 128).transpose(0, 1).contiguous()  # [G, T, D//128]

    # Step 3: aiter blockwise FP8 batched GEMM.
    # wo_a.weight is already (G, R, D) fp8_e4m3fn; wo_a.weight_scale_inv is
    # (G, R//128, D//128) fp32 (post _setup_fp8_wo_a_scales).
    w_fp8 = self.wo_a.weight.view(G, R, D)
    w_scale = self.wo_a.weight_scale_inv.view(G, R // 128, D // 128).to(torch.float32)

    z_btn = aiter.batched_gemm_fp8_blockwise(
        a_fp8, w_fp8, a_scale, w_scale, backend="auto",
    )                                                          # [G, T, R] bf16
    z = z_btn.transpose(0, 1).contiguous()                     # [T, G, R]

    # Step 4: wo_b projection.
    return self.wo_b(z.flatten(1))


def apply_patch() -> bool:
    """Returns True iff the patch was applied."""
    if not _enabled():
        logger.info("VLLM_DSV4_WO_A_AITER not set; leaving vllm DSv4 wo_a path untouched.")
        return False
    try:
        from vllm.model_executor.layers import deepseek_v4_attention as dv4_attn
        from vllm.platforms import current_platform
    except ImportError as e:
        logger.error("vllm DSv4 patch unavailable: %s", e)
        return False

    if not current_platform.is_rocm():
        logger.info("Not on ROCm; skipping aiter wo_a patch.")
        return False

    Attn = dv4_attn.DeepseekV4MLAAttention
    orig = Attn.forward_o_proj  # type: ignore[attr-defined]

    def forward_o_proj(self, hidden_states, positions, llama_4_scaling=None):
        # Reproduce the parent flow up to the ROCm branch, then call our path.
        num_tokens = hidden_states.shape[0]
        o_padded = torch.empty(
            (num_tokens, self.padded_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        torch.ops.vllm.deepseek_v4_attention(
            hidden_states, positions, o_padded, self.layer_name,
        )
        o = o_padded[:, : self.n_local_heads, :]
        return _patched_rocm_wo_a(self, o, positions)

    Attn.forward_o_proj = forward_o_proj  # type: ignore[assignment]
    logger.info("vllm DSv4 wo_a path patched to use aiter.batched_gemm_fp8_blockwise.")
    return True


# Auto-apply on import so a single VLLM_PLUGINS entry is enough to enable.
if __name__ != "__main__":
    apply_patch()
