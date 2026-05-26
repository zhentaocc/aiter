# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Multi-wave streamlined splitk-style FlyDSL FP8 block-wise batched GEMM.

STATUS (2026-05-10): CORRECT + BENCHED. Not adopted.
  Bit-equal output to single-wave kernel (max bf16 err vs torch oracle = 0.5
  on the canonical (B=8, N=1024, K=4096) sweep, identical to sw).
  Perf vs single-wave kernel:
    M=16   decode      0.99x  (1% win)
    M=64   decode      0.97x  (3% win)
    M=1024 prefill     1.04x  (4% loss)
    M=4096 prefill     1.13x  (13% loss)
  Tiny win on decode (latency-bound, 128 WGs); steady loss on prefill
  (~32k WGs saturate the chip, so per-WG latency is invisible and per-WG
  throughput dominates -- and per-WG work is identical to sw).
  Estimated 152 TFLOPS at M=4096, ~2.5% of MI355X peak fp8 -- not
  memory-bound, just an under-pipelined inner loop. To actually win
  prefill we'd need 3-stage software pipelining + sched.barrier
  intrinsics + async global_load_lds, i.e. a port of splitk_hgemm.py
  proper, not this single-wave-extended skeleton. Deferred.

Architecture:
  * 4 waves per workgroup (256 threads), BLOCK_M_WARPS=1 × BLOCK_N_WARPS=4
  * BLOCK_M=16, BLOCK_N=64 output tile per WG (each wave writes 16×16)
  * LDS-A *double-buffer* (4 KB) -- shared across the 4 waves
  * Cooperative HBM->LDS load: all 256 lanes load 8 bytes each (= 2 KB)
  * Each wave loads its own 16×128-K W stripe from HBM
  * V4 (1, 1, 128) scales fed natively to MFMA scaleA / scaleB

FlyDSL footguns solved here (all real):
  * `_memref.store` raw value error -> use STensor.vec_store instead
  * `ir.BFloat16Type.get()` doesn't exist -> use `T.bf16` property
  * `scf.IfOp(hasElse=False)` -> `has_else=False`
  * `vector<NxFloat8E4M3FN>` LLVM lowering crash -> use vector<Nxi8>
    + bitcast at the MFMA boundary
  * GTensor i8 vec_load(.., 32) NaN -> use i32 buffer_load(vec_width=4)
    twice (= 8 dwords = 32 bytes) then bitcast v8i32 -> v32i8
  * wave-0-only LDS write left 3/4 waves idle -> cooperative load
  * single-buffer LDS forced barrier-per-iter -> double-buffer

Kept as reference for future kernel work; production path remains the
single-wave kernel in batched_gemm_fp8_blockwise_flydsl.py.
"""

from __future__ import annotations

import functools
from typing import Optional

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith as _arith, scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import buffer_ops, range_constexpr, rocdl, vector, gpu
from flydsl.expr.typing import T, Uint8, _to_raw
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from .tensor_shim import GTensor, STensor
from .batched_gemm_fp8_blockwise_flydsl import (
    _ue8m0_byte_pack4,
    _truncf_f32_to_bf16,
    _torch_scales_to_ue8m0,
)


def _vec_ty(n, mlir_elem):
    return ir.VectorType.get([n], mlir_elem)


_BLOCK_M = 16
_BLOCK_N = 64
_BLOCK_K = 128
_N_WAVES = 4
_WAVE_SIZE = 64
_BLOCK_THREADS = _N_WAVES * _WAVE_SIZE   # 256


@functools.lru_cache(maxsize=None)
def compile_bgfp8bw_mw_kernel(B: int, M: int, N: int, K: int):
    """Multi-wave kernel factory.  All shape values baked at compile time."""
    assert M % _BLOCK_M == 0
    assert N % _BLOCK_N == 0
    assert N % 128 == 0      # so 64-N tile fits inside one 128-N W_scale block
    assert K % _BLOCK_K == 0

    GRID_M = M // _BLOCK_M
    GRID_N = N // _BLOCK_N
    K_g = K // 128
    N_g = N // 128

    A_BATCH_STRIDE = M * K
    W_BATCH_STRIDE = N * K
    O_BATCH_STRIDE = M * N
    AS_BATCH_STRIDE = M * K_g
    WS_BATCH_STRIDE = N_g * K_g

    # LDS allocation (double-buffer A: 2 × 16 rows × 128 K bytes = 4 KB).
    # Allocator is set up at *factory* time but finalized inside the kernel
    # (mirroring splitk_hgemm pattern).
    allocator = SmemAllocator(None, arch="gfx950", global_sym_name="smem_mw")
    LDS_A_BYTES = _BLOCK_M * _BLOCK_K
    smem_a_offset_0 = allocator._align(allocator.ptr, 16)
    smem_a_offset_1 = smem_a_offset_0 + LDS_A_BYTES
    allocator.ptr = smem_a_offset_1 + LDS_A_BYTES

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def kernel(
        A: fx.Tensor,
        W: fx.Tensor,
        A_scale: fx.Tensor,
        W_scale: fx.Tensor,
        Out: fx.Tensor,
    ):
        # Finalize LDS allocation inside the gpu.module body
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        # Set up tensor wrappers.  Use FlyDSL's T.* properties (they create
        # MLIR types in the active context).
        fp8_t = T.f8
        bf16_t = T.bf16
        i8_t = T.i8

        # NOTE: LLVM's vector type lowering crashes on vector<NxFloat8E4M3FN>.
        # Workaround: treat fp8 storage as i8 throughout; bitcast to v8i32
        # at the MFMA boundary (which expects v8i32 packed input).
        A_ = GTensor(A, dtype=i8_t, shape=(B, M, K))
        W_ = GTensor(W, dtype=i8_t, shape=(B, N, K))
        Out_ = GTensor(Out, dtype=bf16_t, shape=(B, M, N))
        As_ = GTensor(A_scale, dtype=i8_t, shape=(B, M, K_g))
        Ws_ = GTensor(W_scale, dtype=i8_t, shape=(B, N_g, K_g))

        # LDS A views (double-buffer): two [BLOCK_M, BLOCK_K] i8 tiles.
        smem_a_ptr_0 = SmemPtr(allocator.get_base(), smem_a_offset_0, i8_t,
                                shape=(_BLOCK_M * _BLOCK_K,))
        smem_a_ptr_1 = SmemPtr(allocator.get_base(), smem_a_offset_1, i8_t,
                                shape=(_BLOCK_M * _BLOCK_K,))
        as_lds_bufs = [
            STensor(smem_a_ptr_0, dtype=i8_t, shape=(_BLOCK_M, _BLOCK_K)),
            STensor(smem_a_ptr_1, dtype=i8_t, shape=(_BLOCK_M, _BLOCK_K)),
        ]

        # ---- thread / wave / block coordinates ----
        tid = fx.thread_idx.x        # [0, 256)
        wave_id = tid // fx.Index(_WAVE_SIZE)   # [0, 4)
        lane = tid % fx.Index(_WAVE_SIZE)        # [0, 64)

        pid_m = fx.block_idx.x       # 16-tile in M
        pid_n = fx.block_idx.y       # 64-tile in N
        pid_b = fx.block_idx.z

        # Per-wave N base (each wave writes 16-N stripe).
        n_offset = pid_n * fx.Index(_BLOCK_N) + wave_id * fx.Index(16)

        # MFMA lane mapping (16x16x128 fp8):
        #   A operand: lane l holds A[l%16, (l/16)*32 + 0..31]
        #   B operand: lane l holds B[l%16, (l/16)*32 + 0..31]
        #   C accum:   lane l writes C[(l/16)*4 + 0..3, l%16]
        row = lane % fx.Index(16)
        k_subtile = lane // fx.Index(16)        # [0, 4)
        k_byte_in_lane = k_subtile * fx.Index(32)

        n_block_idx = (pid_n * fx.Index(_BLOCK_N)) // fx.Index(128)

        v4f32 = _vec_ty(4, ir.F32Type.get())
        v8i32 = _vec_ty(8, ir.IntegerType.get_signless(32))
        v32fp8 = _vec_ty(32, fp8_t)

        acc = fx.Vector.filled(4, 0.0, fx.Float32)

        # ====================================================================
        # K-LOOP with LDS-A sharing across 4 waves.
        # Each K-iter:
        #   1. Wave 0: load A tile from HBM via i32 buffer_load (proven path),
        #      bitcast to v32xi8, store to LDS via STensor.
        #   2. Barrier.
        #   3. All waves: load own A row from LDS (v32xi8), bitcast to v8i32
        #      for MFMA.  Each wave loads its own W stripe from HBM.
        #   4. MFMA + accumulate.
        # ====================================================================
        a_rsrc_raw = buffer_ops.create_buffer_resource(A, max_size=True)
        w_rsrc_raw = buffer_ops.create_buffer_resource(W, max_size=True)
        k_dword_lane = (lane // fx.Index(16)) * fx.Index(8)

        # Per-wave W base (each wave reads own N-stripe).
        w_row_dword_base = (
            (pid_b * fx.Index(W_BATCH_STRIDE)
             + (n_offset + row) * fx.Index(K))
            // fx.Index(4)
        )

        v32i8 = _vec_ty(32, T.i8)
        v8i8_t = _vec_ty(8, T.i8)

        # ---- Cooperative-load lane mapping (constant across K-iters) ----
        # 256 threads × 8 bytes/thread = 2048 bytes = exactly the LDS-A slot.
        ld_row = tid // fx.Index(16)
        ld_byte_off = (tid % fx.Index(16)) * fx.Index(8)

        def _hbm_a_dword_off_for(k_const):
            """k_const is a Python int (compile-time)."""
            return (
                (pid_b * fx.Index(A_BATCH_STRIDE)
                 + (pid_m * fx.Index(_BLOCK_M) + ld_row) * fx.Index(K)
                 + fx.Index(k_const * 128) + ld_byte_off)
                // fx.Index(4)
            )

        # ====================================================================
        # PROLOGUE: load tile 0 into buffer 0, barrier.
        # ====================================================================
        a_pref = buffer_ops.buffer_load(
            a_rsrc_raw, _hbm_a_dword_off_for(0), vec_width=2, dtype=T.i32)
        as_lds_bufs[0].vec_store(
            (ld_row, ld_byte_off), vector.bitcast(v8i8_t, a_pref), 8)
        gpu.barrier()

        # ====================================================================
        # K-LOOP, double-buffered.
        # Iter k:
        #   * If k+1 < K_g: issue HBM load of tile k+1 → register `a_next`
        #   * Read A from LDS buffer[k % 2]
        #   * Load W stripe (per-wave, per-lane)
        #   * MFMA + accumulate
        #   * If k+1 < K_g: store a_next → LDS buffer[(k+1) % 2], barrier
        # The HBM load of k+1 is *issued* before MFMA so the memory unit
        # overlaps with the matrix unit; the LDS write+barrier waits until
        # after compute, so wave-divergent VMEM latency is hidden.
        # ====================================================================
        for k_tile in range_constexpr(K_g):
            cur_buf = as_lds_bufs[k_tile % 2]
            nxt_buf_idx = (k_tile + 1) % 2

            # ---- Issue prefetch of next A tile (HBM → register) ----
            if k_tile + 1 < K_g:
                a_next = buffer_ops.buffer_load(
                    a_rsrc_raw, _hbm_a_dword_off_for(k_tile + 1),
                    vec_width=2, dtype=T.i32)

            # ---- Read current A from LDS into v8i32 register ----
            a_lds_vec = cur_buf.vec_load((row, k_dword_lane * fx.Index(4)), 32)
            a128 = vector.bitcast(v8i32, a_lds_vec)

            # ---- Each wave: load own W stripe from HBM (per-lane v8i32) ----
            k_dword_off = fx.Index(k_tile * 32)
            w_dword_off = w_row_dword_base + k_dword_lane + k_dword_off
            w_lo = buffer_ops.buffer_load(w_rsrc_raw, w_dword_off, vec_width=4, dtype=T.i32)
            w_hi = buffer_ops.buffer_load(w_rsrc_raw, w_dword_off + fx.Index(4),
                                          vec_width=4, dtype=T.i32)
            b128 = vector.from_elements(
                v8i32,
                [w_lo[0], w_lo[1], w_lo[2], w_lo[3],
                 w_hi[0], w_hi[1], w_hi[2], w_hi[3]],
            )

            # ---- Per-lane A_scale (lane → row pid_m*16 + lane%16) ----
            a_scale_byte = As_[
                pid_b,
                pid_m * fx.Index(_BLOCK_M) + row,
                fx.Index(k_tile),
            ]
            a_scale_packed = _ue8m0_byte_pack4(a_scale_byte)

            # ---- W_scale (uniform across the WG within one 128-N-block) ----
            w_scale_byte = Ws_[pid_b, n_block_idx, fx.Index(k_tile)]
            w_scale_packed = _ue8m0_byte_pack4(w_scale_byte)

            # ---- MFMA with both V4 scales fed natively ----
            tile_acc = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                T.f32x4,
                [a128, b128, fx.Vector.filled(4, 0.0, fx.Float32),
                 0, 0, 0, a_scale_packed, 0, w_scale_packed],
            )

            # ---- Accumulate ----
            new_vals = []
            for i in range_constexpr(4):
                new_vals.append(acc[i] + tile_acc[i])
            acc = vector.from_elements(v4f32, new_vals)

            # ---- Commit prefetched A_{k+1} to its LDS buffer + barrier ----
            if k_tile + 1 < K_g:
                as_lds_bufs[nxt_buf_idx].vec_store(
                    (ld_row, ld_byte_off), vector.bitcast(v8i8_t, a_next), 8)
                gpu.barrier()

        # ---- Store output (16x16 per wave, 16x64 per WG total) ----
        out_row_base = (lane // fx.Index(16)) * fx.Index(4)
        out_col = lane % fx.Index(16)
        for i in range_constexpr(4):
            m_idx = pid_m * fx.Index(_BLOCK_M) + out_row_base + fx.Index(i)
            n_idx = n_offset + out_col
            bf16_val = _truncf_f32_to_bf16(acc[i])
            Out_[pid_b, m_idx, n_idx] = bf16_val

    @flyc.jit
    def launcher(
        A: fx.Tensor, W: fx.Tensor,
        A_scale: fx.Tensor, W_scale: fx.Tensor,
        Out: fx.Tensor,
    ):
        kernel(A, W, A_scale, W_scale, Out).launch(
            grid=(GRID_M, GRID_N, B),
            block=(_BLOCK_THREADS, 1, 1),
        )

    return launcher


def flydsl_batched_gemm_fp8_blockwise_mw(
    A: torch.Tensor,
    W: torch.Tensor,
    A_scale: torch.Tensor,
    W_scale: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Multi-wave streamlined splitk-style FlyDSL kernel."""
    assert A.dtype == torch.float8_e4m3fn
    assert W.dtype == torch.float8_e4m3fn
    if A_scale.dtype not in (torch.float32, torch.uint8):
        raise TypeError(f"A_scale dtype: {A_scale.dtype}")
    if W_scale.dtype not in (torch.float32, torch.uint8):
        raise TypeError(f"W_scale dtype: {W_scale.dtype}")
    A_scale = _torch_scales_to_ue8m0(A_scale)
    W_scale = _torch_scales_to_ue8m0(W_scale)
    B, M, K = A.shape
    Bw, N, Kw = W.shape
    assert B == Bw and K == Kw

    if out is None:
        out = torch.empty((B, M, N), dtype=torch.bfloat16, device=A.device)

    launcher = compile_bgfp8bw_mw_kernel(B, M, N, K)
    cf = getattr(launcher, "_aiter_cf", None)
    if cf is None:
        cf = flyc.compile(launcher, A, W, A_scale, W_scale, out)
        launcher._aiter_cf = cf
    else:
        cf(A, W, A_scale, W_scale, out)
    return out
