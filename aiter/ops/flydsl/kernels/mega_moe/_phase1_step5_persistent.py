# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ROCm/FlyDSL MegaMOE port contributors

"""Phase 1 step 5 — persistent CTA scheduling.

Replaces the (n_tiles, m_tiles*E) 2D grid of step 4 with a 1D grid of
``num_sms`` persistent CTAs. Each CTA loops over its assigned blocks:

  for iter ∈ [0, iters_per_cta):
      block_idx = bx + iter * num_sms
      decode (expert, m_tile, n_tile)
      compute the FP4×FP4 16×16 output tile for that triple

This is the simplest persistent pattern (one tile per iteration, no
phase ping-pong, no pre-fetched data carried across iterations).
The full ``HostMegaMoEScheduler`` L1↔L2 ping-pong + variable expert
sizes wait for Phase 5 mega-kernel integration.
"""

from __future__ import annotations

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import buffer_ops, rocdl, range_constexpr
from flydsl.expr.typing import T
from ._phase1_step2_fp4_b import _fp4_quant, _fp4_dequant, _FP4_TABLE


def make_persistent_kernel(M: int, N: int, K: int, E: int, num_sms: int):
    """All shape params are compile-time."""
    assert M % 16 == 0 and N % 16 == 0 and K % 128 == 0 and K % 8 == 0
    m_tiles = M // 16
    n_tiles = N // 16
    k_chunks = K // 128
    total_blocks = E * m_tiles * n_tiles
    blocks_per_expert = m_tiles * n_tiles

    # Each CTA does ceil(total_blocks / num_sms) iterations; bound by
    # block_idx < total_blocks check inside.
    iters_per_cta = (total_blocks + num_sms - 1) // num_sms

    a_dwords_per_expert = M * K // 8
    b_dwords_per_expert = N * K // 8
    c_elems_per_expert = M * N
    a_dwords_per_row = K // 8
    b_dwords_per_row = K // 8

    @flyc.kernel
    def kernel(a_ptr: fx.Tensor, b_ptr: fx.Tensor, c_ptr: fx.Tensor):
        bx = fx.block_idx.x
        tid = fx.thread_idx.x
        lane = tid

        row = lane % fx.Index(16)
        k_lane = lane // fx.Index(16)

        a_rsrc = buffer_ops.create_buffer_resource(a_ptr, max_size=True)
        b_rsrc = buffer_ops.create_buffer_resource(b_ptr, max_size=True)
        c_rsrc = buffer_ops.create_buffer_resource(c_ptr, max_size=True)

        # Persistent loop. Use range_constexpr because iters_per_cta is constexpr.
        # Inside, gate on block_idx < total_blocks (constexpr-foldable when
        # total_blocks % num_sms == 0).
        for it in range_constexpr(iters_per_cta):
            block_idx = bx + fx.Index(it * num_sms)

            # Decode (expert, m_tile, n_tile)
            expert_idx = block_idx // fx.Index(blocks_per_expert)
            within = block_idx % fx.Index(blocks_per_expert)
            m_tile_idx = within // fx.Index(n_tiles)
            n_tile_idx = within % fx.Index(n_tiles)

            # Per-block byte offsets (dword units)
            a_base = (expert_idx * fx.Index(a_dwords_per_expert)
                      + (m_tile_idx * fx.Index(16) + row) * fx.Index(a_dwords_per_row)
                      + k_lane * fx.Index(4))
            b_base = (expert_idx * fx.Index(b_dwords_per_expert)
                      + (n_tile_idx * fx.Index(16) + row) * fx.Index(b_dwords_per_row)
                      + k_lane * fx.Index(4))

            # Fresh accumulator for each block
            acc = fx.Vector.filled(4, 0.0, fx.Float32)

            # K-loop
            for chunk in range_constexpr(k_chunks):
                chunk_off = fx.Index(chunk * 16)
                a_dword_off = a_base + chunk_off
                b_dword_off = b_base + chunk_off
                a128 = buffer_ops.buffer_load(a_rsrc, a_dword_off, vec_width=4, dtype=T.i32)
                b128 = buffer_ops.buffer_load(b_rsrc, b_dword_off, vec_width=4, dtype=T.i32)
                acc = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                    T.f32x4,
                    [a128, b128, acc, 4, 4, 0, 0x7F7F7F7F, 0, 0x7F7F7F7F],
                )

            # Store: 4 fp32 per lane
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
        kernel(a_ptr, b_ptr, c_ptr).launch(grid=(num_sms, 1, 1),
                                            block=(64, 1, 1),
                                            stream=stream)
    return launcher


def test(M: int, N: int, K: int, E: int, num_sms: int) -> None:
    torch.manual_seed(11)
    A_f32 = _FP4_TABLE.to("cuda")[torch.randint(0, 16, (E, M, K), device="cuda")]
    B_f32 = _FP4_TABLE.to("cuda")[torch.randint(0, 16, (E, N, K), device="cuda")]
    A_packed = _fp4_quant(A_f32.reshape(-1, K)).reshape(E, M, K // 2).contiguous()
    B_packed = _fp4_quant(B_f32.reshape(-1, K)).reshape(E, N, K // 2).contiguous()
    C = torch.zeros(E, M, N, dtype=torch.float32, device="cuda")

    A_dl = flyc.from_dlpack(A_packed).mark_layout_dynamic(leading_dim=2, divisibility=K // 2)
    B_dl = flyc.from_dlpack(B_packed).mark_layout_dynamic(leading_dim=2, divisibility=K // 2)
    C_dl = flyc.from_dlpack(C).mark_layout_dynamic(leading_dim=2, divisibility=N)

    launcher = make_persistent_kernel(M, N, K, E, num_sms)
    launcher(A_dl, B_dl, C_dl, stream=torch.cuda.Stream())
    torch.cuda.synchronize()

    A_ref = _fp4_dequant(A_packed.reshape(-1, K // 2)).reshape(E, M, K)
    B_ref = _fp4_dequant(B_packed.reshape(-1, K // 2)).reshape(E, N, K)
    expected = torch.einsum("emk,enk->emn", A_ref, B_ref)
    diff = (C - expected).abs()

    total_blocks = E * (M // 16) * (N // 16)
    iters = (total_blocks + num_sms - 1) // num_sms
    print(f"M={M:3d} N={N:3d} K={K:4d} E={E:2d}  num_sms={num_sms:3d}  "
          f"total_blocks={total_blocks:4d}  iters/CTA={iters:2d}  "
          f"max_diff={diff.max().item():.4e}  "
          f"close={torch.allclose(C, expected, atol=1e-3, rtol=1e-3)}")
    if diff.max().item() > 1e-3:
        idx = diff.argmax().item()
        sN = N
        e = idx // (M * sN); rest = idx % (M * sN)
        m_ = rest // sN; n_ = rest % sN
        print(f"  worst: C[{e},{m_},{n_}] got={C[e,m_,n_].item()} expected={expected[e,m_,n_].item()}")


def main() -> None:
    # When total_blocks <= num_sms, each CTA does at most 1 block (or 0).
    test(M=16, N=16, K=128, E=1,  num_sms=4)    # total=1, 1 CTA active
    test(M=16, N=16, K=128, E=4,  num_sms=4)    # total=4, all CTAs 1 block
    test(M=16, N=16, K=128, E=16, num_sms=4)    # total=16, each CTA 4 blocks
    test(M=32, N=32, K=128, E=4,  num_sms=8)    # total=16, each CTA 2 blocks
    test(M=32, N=32, K=256, E=2,  num_sms=4)    # total=8, each CTA 2 blocks
    test(M=64, N=64, K=512, E=4,  num_sms=16)   # total=64, each CTA 4 blocks
    test(M=64, N=64, K=512, E=8,  num_sms=128)  # total=128, each CTA 1 block (matches gfx950)


if __name__ == "__main__":
    main()
