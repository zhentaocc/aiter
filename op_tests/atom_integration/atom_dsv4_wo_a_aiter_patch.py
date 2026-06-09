# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Monkey-patch ATOM's DeepSeek V4 wo_a path to use
``aiter.batched_gemm_fp8_blockwise`` and pre-convert weight scales to U8
(UE8M0) at model-load time.

ATOM ships with a sglang-like DSv4 implementation. The wo_a output
projection is the same op as in vllm: a per-group batched FP8 block-wise
GEMM. ATOM's current ROCm fallback (like vllm) is a BF16 reference path.
This patch swaps it for the aiter kernel.

Activation:
    AITER_DSV4_WO_A=1                  # enables the wo_a kernel swap
    AITER_DSV4_WO_A_U8_SCALES=1        # additionally pre-converts
                                        # W_scale_inv to uint8 at load time
                                        # (eliminates the per-call fp32->u8
                                        # conversion -- the dominant overhead
                                        # if AITER_DSV4_WO_A_U8_SCALES=0)

Designed to run inside ``rocm/atom-dev:latest``. Importing this module
(via ATOM_PLUGINS=op_tests.atom_integration.atom_dsv4_wo_a_aiter_patch
or via direct ``import`` from a custom entrypoint) auto-applies the patch.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch

logger = logging.getLogger("atom_dsv4_wo_a_aiter_patch")


def _enabled() -> bool:
    return os.environ.get("AITER_DSV4_WO_A", "0") not in ("0", "", "false", "False")


def _u8_scales_enabled() -> bool:
    return os.environ.get("AITER_DSV4_WO_A_U8_SCALES", "1") not in ("0", "", "false", "False")


def _convert_w_scale_to_u8(self) -> None:
    """Replace self.wo_a.weight_scale_inv (fp32) with the uint8 UE8M0 form.

    Idempotent: if already uint8, no-op. Runs in model-load (CPU/GPU) thread
    so the cost is paid once per model load, not per inference step.
    """
    import aiter
    if not hasattr(self, "wo_a") or not hasattr(self.wo_a, "weight_scale_inv"):
        return
    cur = self.wo_a.weight_scale_inv
    if cur.dtype == torch.uint8:
        return  # already converted
    u8 = aiter.convert_scales_to_ue8m0(cur.detach())
    # Replace as Parameter so model state-dict ops work
    import torch.nn as nn
    self.wo_a.weight_scale_inv = nn.Parameter(u8, requires_grad=False)
    logger.info("DSv4 wo_a layer %s: weight_scale_inv converted fp32 -> uint8 (shape=%s)",
                getattr(self, "layer_name", "?"), tuple(u8.shape))


def _patched_rocm_wo_a(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Drop-in replacement for the ATOM ROCm BF16 reference wo_a path.

    Steps:
      1. Inverse-RoPE on the rope head dim (reuse ATOM's existing op).
      2. Per-row FP8 quant on the (T*G, D) flattened activation
         (group_size=128). Produces uint8 scales directly if ATOM's
         quant kernel supports it; else fp32 scales (wrapper handles
         the conversion silently).
      3. aiter.batched_gemm_fp8_blockwise on (G, T, D) x (G, R, D) -> (G, T, R).
         backend="auto" -> flydsl(decode) or CK(prefill).
      4. Transpose back to (T, G, R) and feed to wo_b.
    """
    import aiter

    G = self.n_local_groups
    R = self.o_lora_rank
    D = self.head_dim
    T = o.shape[0]

    # Step 1: inverse RoPE.
    # ATOM exposes the same helper under a slightly different module path
    # than vllm; try both.
    try:
        from atom.model_executor.layers.deepseek_v4_attention import rocm_inv_rope_apply_only
    except ImportError:
        from vllm.model_executor.layers.deepseek_v4_attention import rocm_inv_rope_apply_only
    o = rocm_inv_rope_apply_only(
        self.rotary_emb, o, positions, self.rope_head_dim, self.n_local_groups,
    )

    # Step 2: per-row, per-128-block FP8 activation quant.
    # ATOM's QuantFP8 (mirrors vllm's). Produces (a_fp8, a_scale_fp32).
    a_fp8, a_scale = self._wo_a_act_quant(
        o.reshape(T * G, D).contiguous(),
    )
    a_fp8 = a_fp8.view(T, G, D).transpose(0, 1).contiguous()          # [G, T, D]
    a_scale = a_scale.view(T, G, D // 128).transpose(0, 1).contiguous() # [G, T, D//128] fp32

    # Step 3: aiter blockwise FP8 batched GEMM.
    # wo_a.weight is fp8_e4m3fn [G, R, D].
    # wo_a.weight_scale_inv is either fp32 [G, R/128, D/128] (legacy) or
    # uint8 UE8M0 [G, R/128, D/128] (after _convert_w_scale_to_u8).
    w_fp8 = self.wo_a.weight.view(G, R, D)
    w_scale = self.wo_a.weight_scale_inv.view(G, R // 128, D // 128)
    # Wrapper accepts both dtypes; if w_scale is uint8 it skips the
    # per-call fp32->u8 conversion entirely.

    z_btn = aiter.batched_gemm_fp8_blockwise(
        a_fp8, w_fp8, a_scale, w_scale, backend="auto",
    )                                                              # [G, T, R] bf16
    z = z_btn.transpose(0, 1).contiguous()                         # [T, G, R]

    # Step 4: wo_b projection.
    return self.wo_b(z.flatten(1))


def _patch_attention_class(Attn) -> None:
    """Install our forward_o_proj on ATOM's DeepseekV4MLAAttention class."""
    orig_forward_o_proj = getattr(Attn, "forward_o_proj", None)
    orig_post_load = getattr(Attn, "post_load_weights", None)

    def forward_o_proj(self, hidden_states, positions, llama_4_scaling=None):
        # Reproduce parent flow up to the ROCm branch.
        num_tokens = hidden_states.shape[0]
        o_padded = torch.empty(
            (num_tokens, self.padded_heads, self.head_dim),
            dtype=hidden_states.dtype, device=hidden_states.device,
        )
        # ATOM exposes the per-layer attention op the same way as vllm.
        torch.ops.atom.deepseek_v4_attention(
            hidden_states, positions, o_padded, self.layer_name,
        ) if hasattr(torch.ops, "atom") else torch.ops.vllm.deepseek_v4_attention(
            hidden_states, positions, o_padded, self.layer_name,
        )
        o = o_padded[:, : self.n_local_heads, :]
        return _patched_rocm_wo_a(self, o, positions)

    def post_load_weights(self, *args, **kwargs):
        # Call ATOM's original load logic first (sets up weight_scale_inv).
        if orig_post_load is not None:
            orig_post_load(self, *args, **kwargs)
        # Then convert W_scale to uint8 once (if enabled).
        if _u8_scales_enabled():
            _convert_w_scale_to_u8(self)

    Attn.forward_o_proj = forward_o_proj  # type: ignore[assignment]
    if orig_post_load is not None:
        Attn.post_load_weights = post_load_weights  # type: ignore[assignment]


def apply_patch() -> bool:
    """Returns True iff the patch was applied."""
    if not _enabled():
        logger.info("AITER_DSV4_WO_A not set; leaving ATOM DSv4 wo_a path untouched.")
        return False

    # Try ATOM module path first; fall back to vllm if ATOM reuses vllm's class.
    Attn = None
    for mod_path in (
        "atom.model_executor.layers.deepseek_v4_attention",
        "vllm.model_executor.layers.deepseek_v4_attention",
    ):
        try:
            import importlib
            mod = importlib.import_module(mod_path)
            Attn = getattr(mod, "DeepseekV4MLAAttention", None)
            if Attn is not None:
                logger.info("Patching DeepseekV4MLAAttention from %s", mod_path)
                break
        except ImportError:
            continue

    if Attn is None:
        logger.error("DSv4 attention class not found in ATOM or vllm; patch skipped.")
        return False

    # Verify we're on ROCm. ATOM/vllm both expose current_platform.
    try:
        from vllm.platforms import current_platform
        if not current_platform.is_rocm():
            logger.info("Not on ROCm; skipping aiter wo_a patch.")
            return False
    except ImportError:
        logger.warning("Couldn't import vllm.platforms.current_platform; "
                       "applying patch unconditionally (caller's responsibility "
                       "to ensure ROCm).")

    _patch_attention_class(Attn)
    logger.info(
        "ATOM DSv4 wo_a path patched: aiter.batched_gemm_fp8_blockwise "
        "(U8 scales=%s).", _u8_scales_enabled(),
    )
    return True


# Auto-apply on import.
if __name__ != "__main__":
    apply_patch()
