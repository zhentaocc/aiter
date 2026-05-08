# SPDX-License-Identifier: Apache-2.0
"""Phase 1 step 2c — try shifting A/B byte offsets across multiple MFMA calls
to cover the full K=128.

Theory: a single FP8×FP4 mfma_scale call only consumes K=32 (per the per-K
active probe showing 32 active positions, not 128 or 64). So to cover K=128
we need 4 calls, each with a different byte-shifted view of A and B.

Per-K probe showed lane k_group=0 of A reads bytes 0..15 (K=0..15 fire)
and lane k_group=3 of A reads bytes 96..127 (K=112..127 fire). So a single
call sees only 2 lane groups (0 and 3), each providing 16 K elements.

If we shift the byte offsets so different (k_group=0, k_group=3) byte
regions are read each call, we should cover the full K.
"""

from __future__ import annotations

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import buffer_ops, rocdl, vector, range_constexpr
from flydsl.expr.typing import T
from flydsl._mlir import ir
from ._phase1_step2_fp4_b import _fp4_quant, _fp4_dequant, _FP4_TABLE


def make_kernel(num_calls: int):
    """Issue `num_calls` MFMA calls, each shifted in byte offset."""

    @flyc.kernel
    def k(a_ptr: fx.Tensor, b_ptr: fx.Tensor, c_ptr: fx.Tensor):
        tid = fx.thread_idx.x
        lane = tid
        row = lane % fx.Index(16)
        # Base offsets (lane k_group=0 byte region)
        a_dword_base = row * fx.Index(32) + (lane // fx.Index(16)) * fx.Index(8)
        b_dword_base = row * fx.Index(16) + (lane // fx.Index(16)) * fx.Index(4)

        a_rsrc = buffer_ops.create_buffer_resource(a_ptr, max_size=True)
        b_rsrc = buffer_ops.create_buffer_resource(b_ptr, max_size=True)
        c_rsrc = buffer_ops.create_buffer_resource(c_ptr, max_size=True)

        v8i32 = ir.VectorType.get([8], ir.IntegerType.get_signless(32))
        acc = fx.Vector.filled(4, 0.0, fx.Float32)

        # Each call shifts the byte offsets to cover a different K=32 chunk.
        # K=128, single call consumes K=32 effective → need 4 calls.
        # Shift A by 8 dwords (= 32 bytes = 32 fp8 elements) per call.
        # Shift B by 2 dwords (= 8 bytes = 16 fp4 elements) per call.
        # Wait — single call consumes K=32 total (lane 0 covers 16, lane 3 covers 16).
        # The active K positions form pairs: (lane 0 sees k=0..15, lane 3 sees k=112..127).
        # That's NOT contiguous, so straight byte shifts may not work.
        # Try: per-call shift by 8 dwords for A (32 bytes = 32 K of fp8) and 2 dwords for B.

        for call in range_constexpr(num_calls):
            shift_a = fx.Index(call * 2)  # 2 dwords = 8 bytes = 8 fp8 K-elements
            shift_b = fx.Index(call * 1)  # 1 dword = 4 bytes = 8 fp4 K-elements
            a_dword_off = a_dword_base + shift_a
            b_dword_off = b_dword_base + shift_b
            a_lo = buffer_ops.buffer_load(a_rsrc, a_dword_off, vec_width=4, dtype=T.i32)
            a_hi = buffer_ops.buffer_load(a_rsrc, a_dword_off + fx.Index(4),
                                           vec_width=4, dtype=T.i32)
            b128 = buffer_ops.buffer_load(b_rsrc, b_dword_off, vec_width=4, dtype=T.i32)
            a128 = vector.from_elements(
                v8i32, [a_lo[0], a_lo[1], a_lo[2], a_lo[3],
                        a_hi[0], a_hi[1], a_hi[2], a_hi[3]])
            acc = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                T.f32x4,
                [a128, b128, acc, 0, 4, 0, 0x7F7F7F7F, 0, 0x7F7F7F7F],
            )

        out_row_base = (lane // fx.Index(16)) * fx.Index(4)
        out_col = lane % fx.Index(16)
        for i in range_constexpr(4):
            c_off = (out_row_base + fx.Index(i)) * fx.Index(16) + out_col
            buffer_ops.buffer_store(acc[i], c_rsrc, c_off)

    @flyc.jit
    def launcher(a_ptr: fx.Tensor, b_ptr: fx.Tensor, c_ptr: fx.Tensor,
                 stream: fx.Stream = fx.Stream(None)):
        k(a_ptr, b_ptr, c_ptr).launch(grid=(1, 1, 1), block=(64, 1, 1),
                                       stream=stream)
    return launcher


def per_k_probe(launcher) -> list[int]:
    actives = []
    for k_pos in range(128):
        A_f32 = torch.zeros(16, 128, device='cuda'); A_f32[0, k_pos] = 1.0
        B_f32 = torch.zeros(16, 128, device='cuda'); B_f32[0, k_pos] = 1.0
        A = A_f32.to(torch.float8_e4m3fn).contiguous()
        B_packed = _fp4_quant(B_f32).contiguous()
        C = torch.zeros(16, 16, dtype=torch.float32, device='cuda')
        A_dl = flyc.from_dlpack(A).mark_layout_dynamic(leading_dim=1, divisibility=128)
        B_dl = flyc.from_dlpack(B_packed).mark_layout_dynamic(leading_dim=1, divisibility=64)
        C_dl = flyc.from_dlpack(C).mark_layout_dynamic(leading_dim=1, divisibility=16)
        launcher(A_dl, B_dl, C_dl, stream=torch.cuda.Stream())
        torch.cuda.synchronize()
        if abs(C[0, 0].item()) > 0.01:
            actives.append(k_pos)
    return actives


def main() -> None:
    for n in [1, 2, 4]:
        L = make_kernel(n)
        a = per_k_probe(L)
        print(f"num_calls={n}: {len(a)} active K positions: {a}")


if __name__ == "__main__":
    main()
