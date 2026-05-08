# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ROCm/FlyDSL MegaMOE port contributors

"""Phase 1 step 3 — K-loop accumulation, FP4×FP4 single output tile.

Builds on _phase1_step2d_fp4fp4.py:
  - Same 16x16x128 MFMA tile, FP4×FP4
  - Single CTA, single output tile (16, 16) FP32
  - Now sweeps multiple K=128 chunks: total K = K_chunks * 128

Verifies that the accumulator is correctly threaded across MFMA calls.
This is the foundation for step 4 (grouping) and step 5 (persistent
scheduler integration).
"""

from __future__ import annotations

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import buffer_ops, rocdl, range_constexpr
from flydsl.expr.typing import T
from ._phase1_step2_fp4_b import _fp4_quant, _fp4_dequant, _FP4_TABLE


def make_kloop_kernel(k_chunks: int):
    """Make a kernel for total K = k_chunks * 128."""

    @flyc.kernel
    def kernel(a_ptr: fx.Tensor, b_ptr: fx.Tensor, c_ptr: fx.Tensor):
        tid = fx.thread_idx.x
        lane = tid
        row = lane % fx.Index(16)

        # Total bytes per row of A/B = k_chunks * 128 fp4 = k_chunks * 64 bytes
        # Per-chunk row stride = 64 bytes = 16 dwords
        # Per K-chunk advance = 16 dwords (per row).
        row_stride_dwords = 16 * k_chunks  # bytes_per_row / 4

        # Per-lane base offset (for chunk 0):
        #   row_offset = row * row_stride_dwords
        #   k_lane_offset = (lane // 16) * 4 dwords
        a_base = row * fx.Index(row_stride_dwords) + (lane // fx.Index(16)) * fx.Index(4)
        b_base = row * fx.Index(row_stride_dwords) + (lane // fx.Index(16)) * fx.Index(4)

        a_rsrc = buffer_ops.create_buffer_resource(a_ptr, max_size=True)
        b_rsrc = buffer_ops.create_buffer_resource(b_ptr, max_size=True)
        c_rsrc = buffer_ops.create_buffer_resource(c_ptr, max_size=True)

        acc = fx.Vector.filled(4, 0.0, fx.Float32)

        # K loop: each iteration covers K=128 fp4 elements per row = 64 bytes = 16 dwords.
        # Per-chunk A/B byte offset shift: 16 dwords (= 64 bytes) per row, BUT we want to
        # advance by ONE chunk's worth of K (not row).  The chunk stride within a row is
        # 16 dwords (= 128 fp4 along K).
        for chunk in range_constexpr(k_chunks):
            chunk_off = fx.Index(chunk * 16)  # advance 16 dwords per chunk
            a_dword_off = a_base + chunk_off
            b_dword_off = b_base + chunk_off
            a128 = buffer_ops.buffer_load(a_rsrc, a_dword_off, vec_width=4, dtype=T.i32)
            b128 = buffer_ops.buffer_load(b_rsrc, b_dword_off, vec_width=4, dtype=T.i32)
            acc = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                T.f32x4,
                [a128, b128, acc, 4, 4, 0, 0x7F7F7F7F, 0, 0x7F7F7F7F],
            )

        out_row_base = (lane // fx.Index(16)) * fx.Index(4)
        out_col = lane % fx.Index(16)
        for i in range_constexpr(4):
            c_off = (out_row_base + fx.Index(i)) * fx.Index(16) + out_col
            buffer_ops.buffer_store(acc[i], c_rsrc, c_off)

    @flyc.jit
    def launcher(a_ptr: fx.Tensor, b_ptr: fx.Tensor, c_ptr: fx.Tensor,
                 stream: fx.Stream = fx.Stream(None)):
        kernel(a_ptr, b_ptr, c_ptr).launch(grid=(1, 1, 1), block=(64, 1, 1),
                                            stream=stream)
    return launcher


def test_k(k_chunks: int) -> None:
    K = k_chunks * 128
    torch.manual_seed(42 + k_chunks)
    A_f32 = _FP4_TABLE.to("cuda")[torch.randint(0, 16, (16, K), device="cuda")]
    B_f32 = _FP4_TABLE.to("cuda")[torch.randint(0, 16, (16, K), device="cuda")]
    A_packed = _fp4_quant(A_f32).contiguous()  # [16, K/2] uint8
    B_packed = _fp4_quant(B_f32).contiguous()
    C = torch.zeros(16, 16, dtype=torch.float32, device="cuda")

    A_dl = flyc.from_dlpack(A_packed).mark_layout_dynamic(leading_dim=1, divisibility=K // 2)
    B_dl = flyc.from_dlpack(B_packed).mark_layout_dynamic(leading_dim=1, divisibility=K // 2)
    C_dl = flyc.from_dlpack(C).mark_layout_dynamic(leading_dim=1, divisibility=16)

    launcher = make_kloop_kernel(k_chunks)
    launcher(A_dl, B_dl, C_dl, stream=torch.cuda.Stream())
    torch.cuda.synchronize()

    A_ref = _fp4_dequant(A_packed)
    B_ref = _fp4_dequant(B_packed)
    expected = A_ref @ B_ref.T
    diff = (C - expected).abs()
    print(f"K={K} ({k_chunks} chunks): max abs diff = {diff.max().item():.4e}  "
          f"close = {torch.allclose(C, expected, atol=1e-3, rtol=1e-3)}")
    if diff.max().item() > 1e-2:
        print(f"  expected[0,:4]: {expected[0,:4].tolist()}")
        print(f"  got[0,:4]:      {C[0,:4].tolist()}")


def main() -> None:
    test_k(1)   # K = 128
    test_k(2)   # K = 256
    test_k(4)   # K = 512
    test_k(8)   # K = 1024


if __name__ == "__main__":
    main()
