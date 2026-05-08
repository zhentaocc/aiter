# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ROCm/FlyDSL MegaMOE port contributors

"""Phase 1 step 1 — minimum-viable scaled MFMA tile.

Single workgroup (1 CTA, 64 lanes = 1 wave) computes one 16×16 FP32 tile
from a 16×128 FP8 A and a 16×128 FP8 B (B is K-major, shape [N=16, K=128]).
Scales = 1.0 (E8M0 byte 0x7F).

Lane mapping for CDNA 16x16x128 MFMA:
- A operand: lane ``l`` holds A[l%16, (l/16)*32 + 0..31]   (32 fp8 / lane)
- B operand: lane ``l`` holds B[l%16, (l/16)*32 + 0..31]
- C accumulator: lane ``l`` writes C[(l/16)*4 + 0..3, l%16]  (4 fp32 / lane)
"""

from __future__ import annotations

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import buffer_ops, rocdl, vector, range_constexpr, arith
from flydsl.expr.typing import T
from flydsl._mlir import ir


def _vec_ty(n: int, mlir_elem):
    return ir.VectorType.get([n], mlir_elem)


@flyc.kernel
def single_tile_mfma_kernel(
    a_ptr: fx.Tensor,
    b_ptr: fx.Tensor,
    c_ptr: fx.Tensor,
):
    tid = fx.thread_idx.x
    lane = tid

    # IMPORTANT: buffer_load/buffer_store offsets are in **element units**
    # (units of dtype), not bytes — the lowering multiplies by element
    # size automatically.  So for an i32 load on FP8 data we pass
    # `byte_offset / 4` (i.e. dword index).
    #
    # A and B are FP8 [16, 128], 1 byte/elem, 128 bytes/row = 32 dwords/row.
    # Per-lane byte tile offset: row*128 + k_off bytes.
    # In dword units: row*32 + k_off/4.
    row = lane % fx.Index(16)
    k_dword = (lane // fx.Index(16)) * fx.Index(8)   # k_off=32 bytes = 8 dwords
    a_dword_off = row * fx.Index(32) + k_dword
    b_dword_off = row * fx.Index(32) + k_dword

    a_rsrc = buffer_ops.create_buffer_resource(a_ptr, max_size=True)
    b_rsrc = buffer_ops.create_buffer_resource(b_ptr, max_size=True)
    c_rsrc = buffer_ops.create_buffer_resource(c_ptr, max_size=True)

    # Each load: 4 i32 = 16 bytes = 16 fp8 elements.
    # 2 loads/lane = 32 fp8 elements/lane.
    a_lo = buffer_ops.buffer_load(a_rsrc, a_dword_off, vec_width=4, dtype=T.i32)
    a_hi = buffer_ops.buffer_load(a_rsrc, a_dword_off + fx.Index(4),
                                   vec_width=4, dtype=T.i32)
    b_lo = buffer_ops.buffer_load(b_rsrc, b_dword_off, vec_width=4, dtype=T.i32)
    b_hi = buffer_ops.buffer_load(b_rsrc, b_dword_off + fx.Index(4),
                                   vec_width=4, dtype=T.i32)

    # Concatenate v4i32 + v4i32 → v8i32.  T.i32x8 doesn't exist in this
    # FlyDSL build, so construct the vector type via raw MLIR.
    v8i32 = _vec_ty(8, ir.IntegerType.get_signless(32))
    a128 = vector.from_elements(
        v8i32,
        [a_lo[0], a_lo[1], a_lo[2], a_lo[3], a_hi[0], a_hi[1], a_hi[2], a_hi[3]],
    )
    b128 = vector.from_elements(
        v8i32,
        [b_lo[0], b_lo[1], b_lo[2], b_lo[3], b_hi[0], b_hi[1], b_hi[2], b_hi[3]],
    )

    # Zero accumulator.  Vector.filled wants the Numeric *class*, not an MLIR type.
    acc0 = fx.Vector.filled(4, 0.0, fx.Float32)

    # cbsz=0 (FP8 e4m3fn), blgp=0 (FP8 e4m3fn),
    # opselA/opselB=0, scaleA/scaleB = 0x7F7F7F7F (all 1.0 in E8M0).
    # T.f32x4 is a property (not a method) in this build.
    acc = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
        T.f32x4,
        [a128, b128, acc0, 0, 0, 0, 0x7F7F7F7F, 0, 0x7F7F7F7F],
    )

    # Lane l writes 4 FP32 values to C[(l/16)*4 + i, l%16] for i in 0..3.
    # buffer_store offset is in element units when offset_is_bytes is False.
    out_row_base = (lane // fx.Index(16)) * fx.Index(4)
    out_col = lane % fx.Index(16)
    for i in range_constexpr(4):
        c_off = (out_row_base + fx.Index(i)) * fx.Index(16) + out_col
        buffer_ops.buffer_store(acc[i], c_rsrc, c_off)


@flyc.jit
def single_tile_launch(
    a_ptr: fx.Tensor, b_ptr: fx.Tensor, c_ptr: fx.Tensor,
    stream: fx.Stream = fx.Stream(None),
):
    single_tile_mfma_kernel(a_ptr, b_ptr, c_ptr).launch(
        grid=(1, 1, 1), block=(64, 1, 1), stream=stream
    )


def main() -> None:
    torch.manual_seed(0)
    A_f32 = (torch.randn(16, 128, device="cuda") * 0.1).clamp(-2, 2)
    B_f32 = (torch.randn(16, 128, device="cuda") * 0.1).clamp(-2, 2)
    A = A_f32.to(torch.float8_e4m3fn).contiguous()
    B = B_f32.to(torch.float8_e4m3fn).contiguous()
    C = torch.zeros(16, 16, dtype=torch.float32, device="cuda")

    A_dl = flyc.from_dlpack(A).mark_layout_dynamic(leading_dim=1, divisibility=128)
    B_dl = flyc.from_dlpack(B).mark_layout_dynamic(leading_dim=1, divisibility=128)
    C_dl = flyc.from_dlpack(C).mark_layout_dynamic(leading_dim=1, divisibility=16)

    single_tile_launch(A_dl, B_dl, C_dl, stream=torch.cuda.Stream())
    torch.cuda.synchronize()

    A_ref = A.to(torch.float32)
    B_ref = B.to(torch.float32)
    expected = A_ref @ B_ref.T
    diff = (C - expected).abs()
    print(f"max abs diff: {diff.max().item():.4e}")
    print(f"mean abs diff: {diff.mean().item():.4e}")
    print(f"expected[0,:4]: {expected[0,:4].tolist()}")
    print(f"got[0,:4]:      {C[0,:4].tolist()}")
    print(f"all close (atol=5e-2): {torch.allclose(C, expected, atol=5e-2, rtol=5e-2)}")


if __name__ == "__main__":
    main()
