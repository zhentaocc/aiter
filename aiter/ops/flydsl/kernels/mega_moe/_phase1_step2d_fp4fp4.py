# SPDX-License-Identifier: Apache-2.0
"""Phase 1 step 2d — FP4×FP4 single-tile (matches preshuffle_gemm idiom).

After discovering FP8×FP4 mixed mode only consumes K=32 of 128 in one
mfma_scale call, switch to **FP4×FP4** which is the verified mode used
by preshuffle_gemm.py:836 (`_fp4_cbsz=4, _fp4_blgp=4`).

For DeepSeek V4 the L1 weights are FP4. For activations, in MegaMOE the
dispatched activations might be FP8 in some recipes; here we test the
FP4×FP4 baseline first.

A: [16, 128] FP4 (= [16, 64] uint8 packed)
B: [16, 128] FP4 (= [16, 64] uint8 packed)
Both lane vectors: vector<4xi32> = 16 bytes = 32 fp4 per lane.
4 K-lanes × 32 = 128 K total ✓
"""

from __future__ import annotations

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import buffer_ops, rocdl, vector, range_constexpr
from flydsl.expr.typing import T
from flydsl._mlir import ir
from ._phase1_step2_fp4_b import _fp4_quant, _fp4_dequant, _FP4_TABLE


@flyc.kernel
def fp4_fp4_kernel(a_ptr: fx.Tensor, b_ptr: fx.Tensor, c_ptr: fx.Tensor):
    tid = fx.thread_idx.x
    lane = tid
    row = lane % fx.Index(16)

    # Both A and B are FP4 packed: row stride = 64 bytes = 16 dwords.
    # Per lane K-stride = 16 fp4 elements / 4 fp4 per dword = 4 dwords.
    # Wait, 1 dword = 4 bytes = 8 fp4. So 32 fp4 = 4 dwords. K-stride = 4 dwords.
    a_dword_off = row * fx.Index(16) + (lane // fx.Index(16)) * fx.Index(4)
    b_dword_off = row * fx.Index(16) + (lane // fx.Index(16)) * fx.Index(4)

    a_rsrc = buffer_ops.create_buffer_resource(a_ptr, max_size=True)
    b_rsrc = buffer_ops.create_buffer_resource(b_ptr, max_size=True)
    c_rsrc = buffer_ops.create_buffer_resource(c_ptr, max_size=True)

    # Each: 4 i32 = 16 bytes = 32 fp4 per lane.
    a128 = buffer_ops.buffer_load(a_rsrc, a_dword_off, vec_width=4, dtype=T.i32)
    b128 = buffer_ops.buffer_load(b_rsrc, b_dword_off, vec_width=4, dtype=T.i32)

    acc0 = fx.Vector.filled(4, 0.0, fx.Float32)
    # cbsz=4 (FP4 e2m1fn for A), blgp=4 (FP4 e2m1fn for B)
    acc = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
        T.f32x4,
        [a128, b128, acc0, 4, 4, 0, 0x7F7F7F7F, 0, 0x7F7F7F7F],
    )

    out_row_base = (lane // fx.Index(16)) * fx.Index(4)
    out_col = lane % fx.Index(16)
    for i in range_constexpr(4):
        c_off = (out_row_base + fx.Index(i)) * fx.Index(16) + out_col
        buffer_ops.buffer_store(acc[i], c_rsrc, c_off)


@flyc.jit
def fp4_fp4_launch(a_ptr: fx.Tensor, b_ptr: fx.Tensor, c_ptr: fx.Tensor,
                   stream: fx.Stream = fx.Stream(None)):
    fp4_fp4_kernel(a_ptr, b_ptr, c_ptr).launch(grid=(1, 1, 1), block=(64, 1, 1),
                                                stream=stream)


def per_k_probe() -> list[int]:
    actives = []
    for k_pos in range(128):
        A_f32 = torch.zeros(16, 128, device='cuda'); A_f32[0, k_pos] = 1.0
        B_f32 = torch.zeros(16, 128, device='cuda'); B_f32[0, k_pos] = 1.0
        A_packed = _fp4_quant(A_f32).contiguous()
        B_packed = _fp4_quant(B_f32).contiguous()
        C = torch.zeros(16, 16, dtype=torch.float32, device='cuda')
        A_dl = flyc.from_dlpack(A_packed).mark_layout_dynamic(leading_dim=1, divisibility=64)
        B_dl = flyc.from_dlpack(B_packed).mark_layout_dynamic(leading_dim=1, divisibility=64)
        C_dl = flyc.from_dlpack(C).mark_layout_dynamic(leading_dim=1, divisibility=16)
        fp4_fp4_launch(A_dl, B_dl, C_dl, stream=torch.cuda.Stream())
        torch.cuda.synchronize()
        if abs(C[0, 0].item()) > 0.01:
            actives.append(k_pos)
    return actives


def random_test() -> None:
    torch.manual_seed(0)
    A_f32 = _FP4_TABLE.to("cuda")[torch.randint(0, 16, (16, 128), device="cuda")]
    B_f32 = _FP4_TABLE.to("cuda")[torch.randint(0, 16, (16, 128), device="cuda")]
    A_packed = _fp4_quant(A_f32).contiguous()
    B_packed = _fp4_quant(B_f32).contiguous()
    C = torch.zeros(16, 16, dtype=torch.float32, device="cuda")
    A_dl = flyc.from_dlpack(A_packed).mark_layout_dynamic(leading_dim=1, divisibility=64)
    B_dl = flyc.from_dlpack(B_packed).mark_layout_dynamic(leading_dim=1, divisibility=64)
    C_dl = flyc.from_dlpack(C).mark_layout_dynamic(leading_dim=1, divisibility=16)
    fp4_fp4_launch(A_dl, B_dl, C_dl, stream=torch.cuda.Stream())
    torch.cuda.synchronize()

    A_ref = _fp4_dequant(A_packed)
    B_ref = _fp4_dequant(B_packed)
    expected = A_ref @ B_ref.T
    diff = (C - expected).abs()
    print(f"random FP4×FP4: max diff = {diff.max().item():.4e}  "
          f"mean diff = {diff.mean().item():.4e}  "
          f"close = {torch.allclose(C, expected, atol=5e-2, rtol=5e-2)}")
    print(f"  expected[0,:4]: {expected[0,:4].tolist()}")
    print(f"  got[0,:4]:      {C[0,:4].tolist()}")


def main() -> None:
    a = per_k_probe()
    print(f"per-K probe (FP4×FP4): {len(a)} active positions: {a if len(a) < 40 else f'{a[:8]}...{a[-8:]}'}")
    random_test()


if __name__ == "__main__":
    main()
