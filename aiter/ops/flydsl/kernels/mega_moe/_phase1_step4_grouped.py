# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ROCm/FlyDSL MegaMOE port contributors

"""Phase 1 step 4 — grouped GEMM, FP4×FP4, 2D launch grid.

For each expert e ∈ [0, E), compute C[e] = A[e] @ B[e].T, where:
  A: [E, M, K] FP4   (= [E, M, K/2] uint8 packed)
  B: [E, N, K] FP4   (= [E, N, K/2] uint8 packed)
  C: [E, M, N] FP32

Each CTA owns one (expert_idx, m_tile_idx, n_tile_idx) triple:
  by = expert_idx * (M/16) + m_tile_idx
  bx = n_tile_idx

Inside the CTA: same K-loop pattern as step 3, with byte offsets shifted
by the expert + tile origin.
"""

from __future__ import annotations

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import buffer_ops, rocdl, range_constexpr
from flydsl.expr.typing import T
from ._phase1_step2_fp4_b import _fp4_quant, _fp4_dequant, _FP4_TABLE


def make_grouped_kernel(M: int, N: int, K: int, E: int):
    """M, N, K, E are compile-time."""
    assert M % 16 == 0 and N % 16 == 0 and K % 128 == 0 and K % 8 == 0
    m_tiles = M // 16
    n_tiles = N // 16
    k_chunks = K // 128

    # Per-expert byte sizes (FP4 = 0.5 bytes).
    a_bytes_per_expert = M * K // 2
    b_bytes_per_expert = N * K // 2
    c_elems_per_expert = M * N
    a_dwords_per_expert = a_bytes_per_expert // 4
    b_dwords_per_expert = b_bytes_per_expert // 4
    a_dwords_per_row = K // 8         # K fp4 / 2 fp4-per-byte / 4 byte-per-dword = K/8
    b_dwords_per_row = K // 8

    @flyc.kernel
    def kernel(a_ptr: fx.Tensor, b_ptr: fx.Tensor, c_ptr: fx.Tensor):
        bx = fx.block_idx.x   # n_tile_idx ∈ [0, n_tiles)
        by = fx.block_idx.y   # expert_idx * m_tiles + m_tile_idx
        tid = fx.thread_idx.x
        lane = tid

        # Decode (expert_idx, m_tile_idx) from by.
        expert_idx = by // fx.Index(m_tiles)
        m_tile_idx = by % fx.Index(m_tiles)
        n_tile_idx = bx

        row = lane % fx.Index(16)
        k_lane = lane // fx.Index(16)  # 0..3, K-lane index within the wave

        # A row addressing:
        #   byte row offset = expert_idx * a_bytes_per_expert + (m_tile_idx*16 + row) * (K/2)
        # In dwords: expert_idx * a_dwords_per_expert + (m_tile_idx*16 + row) * a_dwords_per_row
        # Plus per-K-lane offset of 4 dwords.
        a_base = (expert_idx * fx.Index(a_dwords_per_expert)
                  + (m_tile_idx * fx.Index(16) + row) * fx.Index(a_dwords_per_row)
                  + k_lane * fx.Index(4))
        b_base = (expert_idx * fx.Index(b_dwords_per_expert)
                  + (n_tile_idx * fx.Index(16) + row) * fx.Index(b_dwords_per_row)
                  + k_lane * fx.Index(4))

        a_rsrc = buffer_ops.create_buffer_resource(a_ptr, max_size=True)
        b_rsrc = buffer_ops.create_buffer_resource(b_ptr, max_size=True)
        c_rsrc = buffer_ops.create_buffer_resource(c_ptr, max_size=True)

        acc = fx.Vector.filled(4, 0.0, fx.Float32)

        for chunk in range_constexpr(k_chunks):
            # Per K-chunk advance = 16 dwords (within a row).
            chunk_off = fx.Index(chunk * 16)
            a_dword_off = a_base + chunk_off
            b_dword_off = b_base + chunk_off
            a128 = buffer_ops.buffer_load(a_rsrc, a_dword_off, vec_width=4, dtype=T.i32)
            b128 = buffer_ops.buffer_load(b_rsrc, b_dword_off, vec_width=4, dtype=T.i32)
            acc = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                T.f32x4,
                [a128, b128, acc, 4, 4, 0, 0x7F7F7F7F, 0, 0x7F7F7F7F],
            )

        # C addressing:
        #   element offset (fp32) = expert_idx * c_elems_per_expert
        #                          + (m_tile_idx*16 + out_row) * N + n_tile_idx*16 + out_col
        out_row_base = k_lane * fx.Index(4)
        out_col = lane % fx.Index(16)
        c_expert_off = expert_idx * fx.Index(c_elems_per_expert)
        for i in range_constexpr(4):
            out_row = m_tile_idx * fx.Index(16) + out_row_base + fx.Index(i)
            out_col_global = n_tile_idx * fx.Index(16) + out_col
            c_off = c_expert_off + out_row * fx.Index(N) + out_col_global
            buffer_ops.buffer_store(acc[i], c_rsrc, c_off)

    @flyc.jit
    def launcher(a_ptr: fx.Tensor, b_ptr: fx.Tensor, c_ptr: fx.Tensor,
                 stream: fx.Stream = fx.Stream(None)):
        kernel(a_ptr, b_ptr, c_ptr).launch(
            grid=(n_tiles, m_tiles * E, 1), block=(64, 1, 1), stream=stream)
    return launcher


def test(M: int, N: int, K: int, E: int) -> None:
    torch.manual_seed(7)
    A_f32 = _FP4_TABLE.to("cuda")[torch.randint(0, 16, (E, M, K), device="cuda")]
    B_f32 = _FP4_TABLE.to("cuda")[torch.randint(0, 16, (E, N, K), device="cuda")]
    A_packed = _fp4_quant(A_f32.reshape(-1, K)).reshape(E, M, K // 2).contiguous()
    B_packed = _fp4_quant(B_f32.reshape(-1, K)).reshape(E, N, K // 2).contiguous()
    C = torch.zeros(E, M, N, dtype=torch.float32, device="cuda")

    A_dl = flyc.from_dlpack(A_packed).mark_layout_dynamic(leading_dim=2, divisibility=K // 2)
    B_dl = flyc.from_dlpack(B_packed).mark_layout_dynamic(leading_dim=2, divisibility=K // 2)
    C_dl = flyc.from_dlpack(C).mark_layout_dynamic(leading_dim=2, divisibility=N)

    launcher = make_grouped_kernel(M, N, K, E)
    launcher(A_dl, B_dl, C_dl, stream=torch.cuda.Stream())
    torch.cuda.synchronize()

    # Reference: per-expert FP4 matmul
    A_ref = _fp4_dequant(A_packed.reshape(-1, K // 2)).reshape(E, M, K)
    B_ref = _fp4_dequant(B_packed.reshape(-1, K // 2)).reshape(E, N, K)
    expected = torch.einsum("emk,enk->emn", A_ref, B_ref)
    diff = (C - expected).abs()
    print(f"M={M:3d} N={N:3d} K={K:4d} E={E}: "
          f"max diff = {diff.max().item():.4e}  "
          f"mean = {diff.mean().item():.4e}  "
          f"close = {torch.allclose(C, expected, atol=1e-3, rtol=1e-3)}")
    if diff.max().item() > 1e-2:
        # Find which expert's tile is wrong
        for e in range(E):
            d = (C[e] - expected[e]).abs()
            if d.max().item() > 1e-3:
                print(f"  expert {e}: max diff = {d.max().item():.4e}")
                idx = d.argmax().item()
                row, col = idx // N, idx % N
                print(f"    at C[{e}, {row}, {col}]: got={C[e, row, col].item()} "
                      f"expected={expected[e, row, col].item()}")
                break


def main() -> None:
    test(M=16, N=16, K=128, E=1)    # smallest case
    test(M=16, N=16, K=128, E=4)    # multi-expert
    test(M=32, N=32, K=128, E=2)    # multi-tile per expert
    test(M=32, N=32, K=256, E=2)    # multi-tile + multi-K-chunk
    test(M=64, N=64, K=512, E=4)    # bigger


if __name__ == "__main__":
    main()
