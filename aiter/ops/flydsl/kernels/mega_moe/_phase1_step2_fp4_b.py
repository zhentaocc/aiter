# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ROCm/FlyDSL MegaMOE port contributors

"""Phase 1 step 2 — single-tile MFMA with FP4 B operand.

Same as step 1 but B is now FP4 e2m1fn (cbsz=0, blgp=4).

Per the FlyROCDL CDNA4 atom (MmaAtom.cpp:180-189):
- A operand (FP8): vector<8xi32> per lane = 32 fp8 = 256 bits
- B operand (FP4): vector<4xi32> per lane = 32 fp4 = 128 bits
- K=128, M=N=16, single 16x16x128 MFMA tile

B tensor shape: [N=16, K=128] of FP4 = [16, 64] uint8 storage
                (2 fp4 elements packed per byte, low nibble first
                 per AMD MX/OCP convention).

Lane mapping (B):
- lane l ← B[l%16, (l/16)*32 .. (l/16)*32+31]   (32 fp4/lane = 16 bytes)
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
def single_tile_fp8_fp4_kernel(
    a_ptr: fx.Tensor,        # [16, 128] fp8 e4m3fn
    b_ptr: fx.Tensor,        # [16, 64] uint8 (= [16, 128] FP4 packed)
    c_ptr: fx.Tensor,        # [16, 16] fp32
):
    tid = fx.thread_idx.x
    lane = tid

    # A side (FP8, same as step 1):
    #   row = lane % 16, k_off = (lane // 16) * 32 bytes
    #   in dword units: row * 32 + (lane // 16) * 8
    row = lane % fx.Index(16)
    k_dword_a = (lane // fx.Index(16)) * fx.Index(8)
    a_dword_off = row * fx.Index(32) + k_dword_a

    # B side (FP4, packed): row stride is 64 bytes = 16 dwords;
    # k_off in *bytes* is (lane // 16) * 32 fp4-elements * 0.5 = (lane//16) * 16 bytes
    # in dword units: row * 16 + (lane // 16) * 4
    k_dword_b = (lane // fx.Index(16)) * fx.Index(4)
    b_dword_off = row * fx.Index(16) + k_dword_b

    a_rsrc = buffer_ops.create_buffer_resource(a_ptr, max_size=True)
    b_rsrc = buffer_ops.create_buffer_resource(b_ptr, max_size=True)
    c_rsrc = buffer_ops.create_buffer_resource(c_ptr, max_size=True)

    # A: 2 × dwordx4 = 8 i32 (= 32 fp8) per lane.
    a_lo = buffer_ops.buffer_load(a_rsrc, a_dword_off, vec_width=4, dtype=T.i32)
    a_hi = buffer_ops.buffer_load(a_rsrc, a_dword_off + fx.Index(4),
                                   vec_width=4, dtype=T.i32)

    # B: 1 × dwordx4 = 4 i32 (= 32 fp4) per lane.
    b128 = buffer_ops.buffer_load(b_rsrc, b_dword_off, vec_width=4, dtype=T.i32)

    v8i32 = _vec_ty(8, ir.IntegerType.get_signless(32))
    a128 = vector.from_elements(
        v8i32,
        [a_lo[0], a_lo[1], a_lo[2], a_lo[3], a_hi[0], a_hi[1], a_hi[2], a_hi[3]],
    )

    acc0 = fx.Vector.filled(4, 0.0, fx.Float32)

    # cbsz=0 (FP8 e4m3fn), blgp=4 (FP4 e2m1fn).
    # opselA/opselB=0, scaleA/scaleB = 0x7F7F7F7F (1.0).
    acc = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
        T.f32x4,
        [a128, b128, acc0, 0, 4, 0, 0x7F7F7F7F, 0, 0x7F7F7F7F],
    )

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
    single_tile_fp8_fp4_kernel(a_ptr, b_ptr, c_ptr).launch(
        grid=(1, 1, 1), block=(64, 1, 1), stream=stream
    )


# ── FP4 e2m1fn pack/unpack (OCP / AMD MX convention) ──────────────────
#
# E2M1FN values (4 bits): SE2M1, no NaN, no inf. Bias = 1.
# Code → value mapping:
#   0000 = +0.0    1000 = -0.0
#   0001 = +0.5    1001 = -0.5
#   0010 = +1.0    1010 = -1.0
#   0011 = +1.5    1011 = -1.5
#   0100 = +2.0    1100 = -2.0
#   0101 = +3.0    1101 = -3.0
#   0110 = +4.0    1110 = -4.0
#   0111 = +6.0    1111 = -6.0
#
# Two FP4 elements pack into one uint8: low nibble = even index, high
# nibble = odd index (i.e. byte b at position p contains elements 2p and
# 2p+1 with elem(2p) = b & 0xF, elem(2p+1) = (b >> 4) & 0xF).

_FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def _fp4_dequant(packed: torch.Tensor) -> torch.Tensor:
    """uint8 packed [..., K/2]  →  fp32 [..., K]."""
    table = _FP4_TABLE.to(packed.device)
    p = packed.to(torch.int32)
    lo = p & 0xF
    hi = (p >> 4) & 0xF
    out = torch.stack([table[lo], table[hi]], dim=-1)
    return out.flatten(-2)


def _fp4_quant(x: torch.Tensor) -> torch.Tensor:
    """fp32 [..., K]  →  uint8 packed [..., K/2]. Uses nearest match."""
    table = _FP4_TABLE.to(x.device)
    diffs = (x.unsqueeze(-1) - table).abs()  # [..., K, 16]
    codes = diffs.argmin(dim=-1).to(torch.int32)  # [..., K]
    even = codes[..., 0::2]
    odd = codes[..., 1::2]
    return ((odd << 4) | even).to(torch.uint8)


def main() -> None:
    torch.manual_seed(0)
    A_f32 = (torch.randn(16, 128, device="cuda") * 0.5).clamp(-2, 2)
    # FP4 has tight set; pick values from the FP4 grid for B to avoid
    # quantisation-induced noise in the reference.
    B_f32 = _FP4_TABLE.to("cuda")[
        torch.randint(0, 16, (16, 128), device="cuda")
    ]
    A = A_f32.to(torch.float8_e4m3fn).contiguous()
    B_packed = _fp4_quant(B_f32).contiguous()  # [16, 64] uint8
    C = torch.zeros(16, 16, dtype=torch.float32, device="cuda")

    A_dl = flyc.from_dlpack(A).mark_layout_dynamic(leading_dim=1, divisibility=128)
    B_dl = flyc.from_dlpack(B_packed).mark_layout_dynamic(leading_dim=1, divisibility=64)
    C_dl = flyc.from_dlpack(C).mark_layout_dynamic(leading_dim=1, divisibility=16)

    single_tile_launch(A_dl, B_dl, C_dl, stream=torch.cuda.Stream())
    torch.cuda.synchronize()

    A_ref = A.to(torch.float32)
    B_ref = _fp4_dequant(B_packed)
    expected = A_ref @ B_ref.T

    diff = (C - expected).abs()
    print(f"max abs diff: {diff.max().item():.4e}")
    print(f"mean abs diff: {diff.mean().item():.4e}")
    print(f"expected[0,:4]: {expected[0,:4].tolist()}")
    print(f"got[0,:4]:      {C[0,:4].tolist()}")
    print(f"all close (atol=5e-2): {torch.allclose(C, expected, atol=5e-2, rtol=5e-2)}")


if __name__ == "__main__":
    main()
