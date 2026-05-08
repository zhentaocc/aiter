# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ROCm/FlyDSL MegaMOE port contributors

"""Phase 1 step 2b — multi-MFMA accumulation for FP8×FP4.

Based on the discovery that a single mfma_scale_f32_16x16x128_f8f6f4
call in mixed FP8×FP4 mode only consumes ~32 of the 128 K-positions.
Per preshuffle_gemm.py:850-892, the FP4 path issues multiple MFMA calls
into one accumulator, varying opselA/opselB across the calls.

Hypothesis: each (opselA, opselB) ∈ [0,3]² selects a different K=32
sub-window for both A and B operands, and looping over all 16 (or some
subset) combinations covers the full K.

We test this by issuing all 4×4 = 16 calls accumulating into one acc.
"""

from __future__ import annotations

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import buffer_ops, rocdl, vector, range_constexpr
from flydsl.expr.typing import T
from flydsl._mlir import ir

from ._phase1_step2_fp4_b import _fp4_quant, _fp4_dequant, _FP4_TABLE


def make_kernel(opsel_pairs):
    """Build a kernel that issues one MFMA call per (opsel_a, opsel_b) pair."""

    @flyc.kernel
    def k(a_ptr: fx.Tensor, b_ptr: fx.Tensor, c_ptr: fx.Tensor):
        tid = fx.thread_idx.x
        lane = tid
        row = lane % fx.Index(16)
        a_dword_off = row * fx.Index(32) + (lane // fx.Index(16)) * fx.Index(8)
        b_dword_off = row * fx.Index(16) + (lane // fx.Index(16)) * fx.Index(4)

        a_rsrc = buffer_ops.create_buffer_resource(a_ptr, max_size=True)
        b_rsrc = buffer_ops.create_buffer_resource(b_ptr, max_size=True)
        c_rsrc = buffer_ops.create_buffer_resource(c_ptr, max_size=True)

        a_lo = buffer_ops.buffer_load(a_rsrc, a_dword_off, vec_width=4, dtype=T.i32)
        a_hi = buffer_ops.buffer_load(a_rsrc, a_dword_off + fx.Index(4),
                                       vec_width=4, dtype=T.i32)
        b128 = buffer_ops.buffer_load(b_rsrc, b_dword_off, vec_width=4, dtype=T.i32)

        v8i32 = ir.VectorType.get([8], ir.IntegerType.get_signless(32))
        a128 = vector.from_elements(
            v8i32, [a_lo[0], a_lo[1], a_lo[2], a_lo[3],
                    a_hi[0], a_hi[1], a_hi[2], a_hi[3]])

        acc = fx.Vector.filled(4, 0.0, fx.Float32)
        for op_a, op_b in opsel_pairs:
            acc = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                T.f32x4,
                [a128, b128, acc, 0, 4, op_a, 0x7F7F7F7F, op_b, 0x7F7F7F7F],
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


def run_test(opsel_pairs, label):
    torch.manual_seed(0)
    A_f32 = (torch.randn(16, 128, device="cuda") * 0.5).clamp(-2, 2)
    B_f32 = _FP4_TABLE.to("cuda")[torch.randint(0, 16, (16, 128), device="cuda")]
    A = A_f32.to(torch.float8_e4m3fn).contiguous()
    B_packed = _fp4_quant(B_f32).contiguous()
    C = torch.zeros(16, 16, dtype=torch.float32, device="cuda")
    A_dl = flyc.from_dlpack(A).mark_layout_dynamic(leading_dim=1, divisibility=128)
    B_dl = flyc.from_dlpack(B_packed).mark_layout_dynamic(leading_dim=1, divisibility=64)
    C_dl = flyc.from_dlpack(C).mark_layout_dynamic(leading_dim=1, divisibility=16)

    launcher = make_kernel(tuple(opsel_pairs))
    launcher(A_dl, B_dl, C_dl, stream=torch.cuda.Stream())
    torch.cuda.synchronize()

    A_ref = A.to(torch.float32)
    B_ref = _fp4_dequant(B_packed)
    expected = A_ref @ B_ref.T
    diff = (C - expected).abs()
    print(f"{label:40s}: max diff = {diff.max().item():.4e}  "
          f"mean diff = {diff.mean().item():.4e}  "
          f"close = {torch.allclose(C, expected, atol=5e-2, rtol=5e-2)}")


def main() -> None:
    print("Test 1: single (0,0) — baseline (known to under-count)")
    run_test([(0, 0)], "single (0,0)")

    print("\nTest 2: all 4 opsel-A diagonal (0,0)..(3,3)")
    run_test([(0, 0), (1, 1), (2, 2), (3, 3)], "diagonal opsels")

    print("\nTest 3: full 4x4 grid (16 calls)")
    pairs = [(a, b) for a in range(4) for b in range(4)]
    run_test(pairs, "full 4x4 opsel grid")

    print("\nTest 4: only (0,0) and (3,3)")
    run_test([(0, 0), (3, 3)], "only corners")

    print("\nTest 5: row 0 of opsels (0..3, 0)")
    run_test([(0, 0), (1, 0), (2, 0), (3, 0)], "(*, 0)")

    print("\nTest 6: col 0 of opsels (0, 0..3)")
    run_test([(0, 0), (0, 1), (0, 2), (0, 3)], "(0, *)")


if __name__ == "__main__":
    main()
