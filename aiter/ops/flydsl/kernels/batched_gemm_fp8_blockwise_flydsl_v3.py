# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
v3: v2 geometry + CK-Intrawave-v3-style staged HBM->VGPR->LDS pipeline.

Same geometry as v2 (BLOCK 128x128x128, 4 waves, 16x16x128 fp8 MFMA, 16
MFMAs/wave/iter, single LDS A buffer). The change is the K-loop structure:

  v2: async DMA HBM->LDS (raw_ptr_buffer_load_lds), single PrefetchStages.
      HBM-load completion goes through lgkmcnt -- the SAME counter as
      ds_read -- so the compiler conservatively waits before any MFMA-input
      ds_read can issue.

  v3: staged HBM->VGPR->LDS (buffer_load + ds_write).
      HBM-load completion goes through vmcnt; ds_read uses lgkmcnt; they
      can complete independently. PrefetchStages = 1 in VGPR + Prefill = 1
      in LDS + LDS-prefetch = 1 in thread_buf:
        * one A-tile in flight HBM->VGPR (a_vgpr_next)
        * one A-tile in LDS being read into thread_buf
        * one A-tile already in thread_buf being MFMA-consumed
      Mirrors CK's blockwise_gemm_pipeline_xdlops_v3_ab_scale (with
      PrefetchStages reduced to 1 to keep VGPR pressure manageable
      without sched_group_barrier).

Cost: +16 VGPR/wave (the A staging slot). Expected occupancy 3 waves/SIMD.
Risk: FlyDSL has no sched_group_barrier primitive, so the LLVM scheduler
must figure out the interleaving on its own. If LLVM is not aggressive
enough, v3 may not beat v2 (Phase 5b found similar issues with manual
ping-pong). This file is the experiment to find out.

See OPTIMIZATION_JOURNEY.md Iter 8 deep-dive section for the full
analysis of why CK's v3 is structurally faster than v2.
"""

from __future__ import annotations

import functools
from typing import Optional

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith as _arith, scf
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import memref as _memref_dialect
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, range_constexpr, rocdl, vector, gpu
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


# ---- LDS XOR-swizzle (16-byte granularity) ----
# Eliminates the 16-way LDS bank conflict that arises when 16 lanes of a wave
# read different rows of the LDS A tile at the same K-byte offset.
# For row r and original byte-col c (16-byte aligned), the swizzled byte-col is
#   c XOR ((r & (k_blocks16 - 1)) * 16)
# where k_blocks16 = BLOCK_K / 16 = 128 / 16 = 8. Self-inverse (write-side and
# read-side use the same formula).
# Pattern borrowed from FlyDSL/kernels/mfma_preshuffle_pipeline.py:swizzle_xor16.
_K_BLOCKS16 = 8        # BLOCK_K // 16 (compile-time constant)
_K_BLOCKS16_MASK = 7   # _K_BLOCKS16 - 1


def _swizzle_xor16(row, col_bytes):
    """row: arith index (LDS row 0..127). col_bytes: arith index (16B-aligned)."""
    rem = arith.andi(row, arith.index(_K_BLOCKS16_MASK))
    return col_bytes ^ (rem * 16)


# ============================================================================
# Geometry constants — fixed for v2.
# ============================================================================
_BLOCK_M = 128                    # output rows per WG
_BLOCK_N = 128                    # output cols per WG (matches Triton + 128-N W_scale block)
_BLOCK_K = 128                    # K consumed per K-iter (== scale block size)
_N_WAVES = 4                      # waves per WG
_WAVE_SIZE = 64                   # gfx950 wave64
_BLOCK_THREADS = _N_WAVES * _WAVE_SIZE   # 256

# Per-wave compute: each wave covers 128 M x 32 N.
_N_PER_WAVE = _BLOCK_N // _N_WAVES        # 32
_M_SUB = _BLOCK_M // 16                   # 8 MFMA tiles along M per wave
_N_SUB = _N_PER_WAVE // 16                # 2 MFMA tiles along N per wave
# Per K-iter: M_SUB * N_SUB = 16 MFMAs per wave; 64 across all 4 waves.

# Cooperative HBM->LDS load decomposition:
# A tile = 128 rows x 128 K bytes = 16384 bytes = 16 KB.
# 256 threads x 64 bytes/thread = 16384 bytes (single round, 4 dwordx4 / thread).
# Layout: 2 threads per row, each owns 64 contiguous K bytes.
_LDS_A_BYTES = _BLOCK_M * _BLOCK_K        # 16384


@functools.lru_cache(maxsize=None)
def compile_bgfp8bw_v3_kernel(B: int, M: int, N: int, K: int):
    """v3 kernel factory. All shape values baked at compile time.

    Always uses 4-wave cooperative HBM->VGPR staging (no load_mode option
    -- v2's solo_unroll / solo_loop ablations don't apply here because the
    staging path is fundamentally different).
    """
    assert M % _BLOCK_M == 0, f"v3 needs M % {_BLOCK_M} == 0, got M={M}"
    assert N % _BLOCK_N == 0, f"v3 needs N % {_BLOCK_N} == 0, got N={N}"
    assert K % _BLOCK_K == 0, f"v3 needs K % {_BLOCK_K} == 0, got K={K}"
    assert K // _BLOCK_K >= 2, f"v3 needs at least 2 K-iters, got K_g={K // _BLOCK_K}"

    GRID_M = M // _BLOCK_M
    GRID_N = N // _BLOCK_N
    K_g = K // _BLOCK_K
    N_g = N // 128

    A_BATCH_STRIDE = M * K
    W_BATCH_STRIDE = N * K
    O_BATCH_STRIDE = M * N
    AS_BATCH_STRIDE = M * K_g
    WS_BATCH_STRIDE = N_g * K_g

    # ---- LDS allocation ----
    # Phase 1:    A tile single buffer 16 KB
    # Phase 4a:  + A_scale buffer (BLOCK_M × K_g bytes) + W_scale buffer
    # Phase 5b experiment: tried 32 KB double-buffer A but regressed 2%
    #   because per-CU LDS pushed WG occupancy from 7 → 4. Reverted.
    _LDS_A_BUF_BYTES = _LDS_A_BYTES           # 16 KB single buffer (Phase 5b reverted)
    _LDS_AS_BYTES = _BLOCK_M * K_g            # 128 × 32 = 4096
    _LDS_WS_BYTES = max(K_g, 16)              # K_g (=32) bytes; round up if smaller
    allocator = SmemAllocator(None, arch="gfx950", global_sym_name="smem_v2")
    smem_a_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = smem_a_offset + _LDS_A_BYTES         # 16 KB (Phase 5b reverted)
    smem_as_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = smem_as_offset + _LDS_AS_BYTES
    smem_ws_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = smem_ws_offset + _LDS_WS_BYTES

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def kernel(
        A: fx.Tensor,
        W: fx.Tensor,
        A_scale: fx.Tensor,
        W_scale: fx.Tensor,
        Out: fx.Tensor,
    ):
        # Finalize LDS allocation inside gpu.module body
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        i8_t = T.i8
        bf16_t = T.bf16

        # GTensor wrappers (i8 storage for fp8 — LLVM doesn't lower v8fp8)
        A_ = GTensor(A, dtype=i8_t, shape=(B, M, K))
        W_ = GTensor(W, dtype=i8_t, shape=(B, N, K))
        Out_ = GTensor(Out, dtype=bf16_t, shape=(B, M, N))
        As_ = GTensor(A_scale, dtype=i8_t, shape=(B, M, K_g))
        Ws_ = GTensor(W_scale, dtype=i8_t, shape=(B, N_g, K_g))

        # LDS A view: [BLOCK_M, BLOCK_K] = [128, 128] i8
        # (Phase 5b reverted to single-buffer; double-buffer dropped occupancy.)
        smem_a_ptr = SmemPtr(allocator.get_base(), smem_a_offset, i8_t,
                              shape=(_LDS_A_BYTES,))
        as_lds = STensor(smem_a_ptr, dtype=i8_t, shape=(_BLOCK_M, _BLOCK_K))
        # Raw LDS memref (for raw_ptr_buffer_load_lds, async DMA path)
        lds_a_memref = smem_a_ptr.get()

        # ---- Phase 4a: LDS for A_scale + W_scale ----
        # A_scale layout: [BLOCK_M, K_g] = [128, 32] i8 (one byte per row × K-iter)
        smem_as_ptr = SmemPtr(allocator.get_base(), smem_as_offset, i8_t,
                               shape=(_LDS_AS_BYTES,))
        as_scale_lds = STensor(smem_as_ptr, dtype=i8_t, shape=(_BLOCK_M, K_g))
        lds_as_memref = smem_as_ptr.get()

        # W_scale layout: [K_g] i8 (one byte per K-iter; same byte for all M)
        smem_ws_ptr = SmemPtr(allocator.get_base(), smem_ws_offset, i8_t,
                               shape=(_LDS_WS_BYTES,))
        ws_scale_lds = STensor(smem_ws_ptr, dtype=i8_t, shape=(_LDS_WS_BYTES,))
        lds_ws_memref = smem_ws_ptr.get()

        # ---- Thread / wave / block coordinates ----
        tid = fx.thread_idx.x                             # [0, 256)
        wave_id = tid // fx.Index(_WAVE_SIZE)             # [0, 4)
        lane = tid % fx.Index(_WAVE_SIZE)                 # [0, 64)

        pid_m = fx.block_idx.x       # 128-tile index in M
        pid_n = fx.block_idx.y       # 128-tile index in N
        pid_b = fx.block_idx.z

        # Per-wave N base (each wave handles 32 N starting here)
        wave_n_offset = wave_id * fx.Index(_N_PER_WAVE)    # 0, 32, 64, 96

        # ---- MFMA 16x16x128 lane mapping ----
        # A operand: lane l holds A[l%16, (l/16)*32 + 0..31]
        # B operand: lane l holds B[l%16, (l/16)*32 + 0..31]
        # C accum:   lane l writes C[(l/16)*4 + 0..3, l%16]
        row = lane % fx.Index(16)
        k_subtile = lane // fx.Index(16)                  # [0, 4)
        k_byte_in_lane = k_subtile * fx.Index(32)
        k_dword_lane = k_subtile * fx.Index(8)            # k_byte_in_lane / 4

        # n_block_idx for W_scale: which 128-N block is this WG in.
        # BLOCK_N = 128, so each WG is exactly one 128-N block.
        n_block_idx = pid_n

        v4f32 = _vec_ty(4, ir.F32Type.get())
        v8i32 = _vec_ty(8, ir.IntegerType.get_signless(32))
        v4i32_t = _vec_ty(4, ir.IntegerType.get_signless(32))
        v16i8_t = _vec_ty(16, T.i8)

        # ---- Per-wave W base (each wave reads its own 32-N stripe) ----
        # For lane l, the row of W to read is wave_n_offset + (n_sub*16) + (l%16).
        # Pre-compute per-(n_sub, lane) row dword bases.
        w_row_dword_bases = [
            (pid_b * fx.Index(W_BATCH_STRIDE)
             + (pid_n * fx.Index(_BLOCK_N) + wave_n_offset
                + fx.Index(n_sub * 16) + row) * fx.Index(K))
            // fx.Index(4)
            for n_sub in range_constexpr(_N_SUB)
        ]

        # ---- Per-(m_sub, lane) M-row index for output store + A_scale lookup ----
        # m_idx_per_sub_row[m_sub] = pid_m*128 + m_sub*16 + (lane%16)
        m_row_per_sub = [
            pid_m * fx.Index(_BLOCK_M) + fx.Index(m_sub * 16) + row
            for m_sub in range_constexpr(_M_SUB)
        ]
        # Output row mapping: lane l writes 4 rows starting at m_sub*16 + (l/16)*4
        out_row_base = (lane // fx.Index(16)) * fx.Index(4)

        # ---- Initialize 16 accumulators per lane ----
        accs = [
            [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range_constexpr(_N_SUB)]
            for _ in range_constexpr(_M_SUB)
        ]

        # ---- Phase 3b: async copy (HBM -> LDS, bypass VGPR) ----
        # raw_ptr_buffer_load_lds writes lane l's `dma_bytes` to LDS at
        # base + l*dma_bytes (hardware-fixed lane stride). The cooperative
        # mapping must therefore be:
        #   chunk c, wave w, lane l   ->   row = c*32 + w*8 + l//8
        #                                   col_byte = (l%8)*16
        # Each chunk handles 32 LDS rows; each wave inside a chunk handles 8 rows.
        # The XOR-swizzle (Phase 2) is applied to the HBM read column instead of
        # the LDS write column -- the LDS layout still ends up swizzled and the
        # read side keeps the same swizzled-read formula.
        _DMA_BYTES = 16
        row_in_wave = lane // fx.Index(8)              # 0..7
        col_byte_in_lane = (lane % fx.Index(8)) * fx.Index(16)   # 0,16,...,112

        # v3: A goes through staged HBM->VGPR->LDS, no raw_ptr_buffer_load_lds
        # for A. We still need _lds_ptr_type for scale prologue (which uses
        # raw_ptr_buffer_load_lds for A_scale + W_scale, unchanged from v2).
        _lds_ptr_type = ir.Type.parse("!llvm.ptr<3>")

        # Constant DMA-call args (reused across all chunks)
        _DMA_BYTES_T = arith.constant(_DMA_BYTES, type=T.i32)
        _SOFFSET_T = arith.constant(0, type=T.i32)
        _OFFSET_IMM_T = arith.constant(0, type=T.i32)
        _AUX_T = arith.constant(1, type=T.i32)

        a_rsrc = buffer_ops.create_buffer_resource(A, max_size=True)
        w_rsrc = buffer_ops.create_buffer_resource(W, max_size=True)
        as_rsrc = buffer_ops.create_buffer_resource(A_scale, max_size=True)
        ws_rsrc = buffer_ops.create_buffer_resource(W_scale, max_size=True)

        # ====================================================================
        # Phase 4a PROLOGUE: cooperative load A_scale + W_scale into LDS.
        # Done ONCE per WG, then reused 32 K-iters.
        #
        # A_scale per-WG: BLOCK_M(128) × K_g(32) bytes = 4 KB.
        #   256 threads × 16 B/thread = 4 KB (single DMA round, 1 dwordx4/thread).
        #   thread tid loads HBM bytes [tid*16, tid*16+16) of the 4 KB region.
        #   LDS layout naturally row-major: LDS[m_row*K_g + k_iter] = scale byte.
        #
        # W_scale per-WG: K_g(=32) bytes total. Wave 0 lane 0 loads bytes 0..15,
        #   wave 0 lane 1 loads bytes 16..31. Other lanes idle (scf.if).
        # ====================================================================
        _AS_DMA_BYTES = 16
        # A_scale: per-thread HBM byte offset
        _hbm_as_byte_off = (
            pid_b * fx.Index(AS_BATCH_STRIDE)
            + pid_m * fx.Index(_BLOCK_M * K_g)
            + tid * fx.Index(_AS_DMA_BYTES)
        )
        _hbm_as_off_i32 = arith.index_cast(T.i32, _hbm_as_byte_off)

        # A_scale LDS scalar base (per-wave shift like cooperative A load)
        _lds_as_wave_base_idx = (
            _memref_dialect.extract_aligned_pointer_as_index(lds_as_memref)
            + wave_id * fx.Index(_WAVE_SIZE * _AS_DMA_BYTES)
        )
        _lds_as_wave_base_i64 = rocdl.readfirstlane(
            T.i64, arith.index_cast(T.i64, _lds_as_wave_base_idx))
        _lds_as_ptr = _llvm.inttoptr(_lds_ptr_type, _lds_as_wave_base_i64)

        rocdl.raw_ptr_buffer_load_lds(
            as_rsrc, _lds_as_ptr,
            arith.constant(_AS_DMA_BYTES, type=T.i32),
            _hbm_as_off_i32,
            _SOFFSET_T, _OFFSET_IMM_T, _AUX_T,
        )

        # W_scale: K_g (=32) bytes total per WG. Have lane 0 of wave 0 load
        # all 32 bytes via 2 DMAs (16 B each), other lanes skip via scf.if.
        # n_block_idx = pid_n (since BLOCK_N=128 maps 1:1 to W_scale N blocks)
        _is_lane0_wave0 = arith.cmpi(_arith.CmpIPredicate.eq,
                                       tid, arith.constant(0, type=T.i32))
        _ws_if = scf.IfOp(_is_lane0_wave0, results_=[], has_else=False)
        with ir.InsertionPoint(_ws_if.then_block):
            _lds_ws_base_idx = _memref_dialect.extract_aligned_pointer_as_index(lds_ws_memref)
            _lds_ws_base_i64 = rocdl.readfirstlane(
                T.i64, arith.index_cast(T.i64, _lds_ws_base_idx))
            _hbm_ws_byte_off_base = (
                pid_b * fx.Index(WS_BATCH_STRIDE)
                + pid_n * fx.Index(K_g)   # n_block_idx == pid_n
            )
            for ws_chunk in range_constexpr((_LDS_WS_BYTES + 15) // 16):
                _lds_ws_ptr = _llvm.inttoptr(
                    _lds_ptr_type,
                    _lds_ws_base_i64 + arith.constant(ws_chunk * 16, type=T.i64))
                _hbm_ws_off_i32 = arith.index_cast(
                    T.i32, _hbm_ws_byte_off_base + fx.Index(ws_chunk * 16))
                rocdl.raw_ptr_buffer_load_lds(
                    ws_rsrc, _lds_ws_ptr,
                    arith.constant(16, type=T.i32),
                    _hbm_ws_off_i32,
                    _SOFFSET_T, _OFFSET_IMM_T, _AUX_T,
                )
            scf.YieldOp([])

        gpu.barrier()  # ensure scales visible to all waves before K-loop

        # ====================================================================
        # v3 K-LOOP — staged HBM->VGPR->LDS pipeline (CK Intrawave-v3 style)
        #
        # Per-iter data flow (matches CK Intrawave v3, modulo single-VGPR slot):
        #   PROLOGUE:
        #     a_stage = HBM_read[0]                   (4 dwordx4 per thread)
        #     ds_write(LDS_A, a_stage)                (LDS has iter 0)
        #     a_stage = HBM_read[1]                   (PrefetchStages=2: iter 1 in VGPR)
        #     barrier (LDS iter 0 visible)
        #
        #   MAIN LOOP iter k = 0 .. K_g - 2:
        #     W loads (direct HBM->VGPR, like v2)
        #     MFMA loop reading LDS A (iter k data, per-M_SUB-just-in-time)
        #     barrier #1 (MFMA reads done; LDS safe to overwrite)
        #     ds_write(LDS_A, a_stage)                (write iter k+1; LDS now iter k+1)
        #     a_stage = HBM_read[k+2]                 (issue iter k+2 prefetch — 300+ cyc lat
        #                                              hides into next iter's MFMA)
        #     barrier #2 (LDS iter k+1 visible to next iter's MFMA reads)
        #
        #   EPILOGUE iter K_g - 1:
        #     W loads + MFMA reading LDS (iter K_g-1 data, written in last
        #                                  main-loop iter's ds_write)
        #     (no further HBM reads or LDS writes)
        # ====================================================================

        # Helper: cooperative HBM -> VGPR A load. Returns list[4] of v4i32
        # (per thread). Per chunk c: this thread loads HBM row
        #   absolute_row = c*32 + wave*8 + lane//8
        # cols [(lane%8)*16 XOR swz, +16) — same swizzled column as v2.
        def _hbm_load_a_to_vgpr(k_byte_off_const):
            chunks_data = []
            for chunk in range_constexpr(4):
                absolute_row = (fx.Index(chunk * 32)
                                + wave_id * fx.Index(8)
                                + row_in_wave)
                swz_amount = arith.andi(
                    absolute_row, arith.index(_K_BLOCKS16_MASK)) * fx.Index(16)
                swz_col_byte = col_byte_in_lane ^ swz_amount
                hbm_byte_off = (
                    pid_b * fx.Index(A_BATCH_STRIDE)
                    + (pid_m * fx.Index(_BLOCK_M) + absolute_row) * fx.Index(K)
                    + k_byte_off_const + swz_col_byte
                )
                hbm_dword_off = hbm_byte_off // fx.Index(4)
                cd = buffer_ops.buffer_load(
                    a_rsrc, hbm_dword_off, vec_width=4, dtype=T.i32)
                chunks_data.append(cd)
            return chunks_data

        # Helper: cooperative VGPR -> LDS A write. Same per-thread layout.
        # LDS[absolute_row, col_byte_in_lane] = chunk c data. The XOR-swizzle
        # is baked into the data via the HBM read column (matches v2's effective
        # LDS layout, so the same _swizzle_xor16 read formula in MFMA loop works).
        def _vgpr_to_lds_a(chunks_data):
            for chunk in range_constexpr(4):
                absolute_row = (fx.Index(chunk * 32)
                                + wave_id * fx.Index(8)
                                + row_in_wave)
                v16i8_data = vector.bitcast(v16i8_t, chunks_data[chunk])
                as_lds.vec_store(
                    (absolute_row, col_byte_in_lane),
                    v16i8_data,
                    vec_size=16,
                )

        # Helper: per-iter W load + MFMA chain. Pulled out of the main loop
        # so the epilogue can reuse it without code duplication.
        def _w_load_and_mfma(k_tile_const):
            k_dword_off = fx.Index(k_tile_const * 32)

            b_tiles = []
            for n_sub in range_constexpr(_N_SUB):
                w_dword_off = w_row_dword_bases[n_sub] + k_dword_lane + k_dword_off
                w_lo = buffer_ops.buffer_load(
                    w_rsrc, w_dword_off, vec_width=4, dtype=T.i32)
                w_hi = buffer_ops.buffer_load(
                    w_rsrc, w_dword_off + fx.Index(4), vec_width=4, dtype=T.i32)
                b128 = vector.from_elements(
                    v8i32,
                    [w_lo[0], w_lo[1], w_lo[2], w_lo[3],
                     w_hi[0], w_hi[1], w_hi[2], w_hi[3]],
                )
                b_tiles.append(b128)

            w_scale_byte = ws_scale_lds[fx.Index(k_tile_const)]
            w_scale_packed = _ue8m0_byte_pack4(w_scale_byte)

            for m_sub in range_constexpr(_M_SUB):
                a_lds_row = fx.Index(m_sub * 16) + row
                swz_lo = _swizzle_xor16(a_lds_row, k_byte_in_lane)
                swz_hi = _swizzle_xor16(a_lds_row, k_byte_in_lane + fx.Index(16))
                half_lo_b = as_lds.vec_load((a_lds_row, swz_lo), 16)
                half_hi_b = as_lds.vec_load((a_lds_row, swz_hi), 16)
                half_lo = vector.bitcast(v4i32_t, half_lo_b)
                half_hi = vector.bitcast(v4i32_t, half_hi_b)
                a128 = vector.from_elements(
                    v8i32,
                    [half_lo[0], half_lo[1], half_lo[2], half_lo[3],
                     half_hi[0], half_hi[1], half_hi[2], half_hi[3]],
                )

                local_m_row = fx.Index(m_sub * 16) + row
                a_scale_byte = as_scale_lds[local_m_row, fx.Index(k_tile_const)]
                a_scale_packed = _ue8m0_byte_pack4(a_scale_byte)

                for n_sub in range_constexpr(_N_SUB):
                    tile_acc = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                        T.f32x4,
                        [a128, b_tiles[n_sub], fx.Vector.filled(4, 0.0, fx.Float32),
                         0, 0, 0, a_scale_packed, 0, w_scale_packed],
                    )
                    new_vals = []
                    for i in range_constexpr(4):
                        new_vals.append(accs[m_sub][n_sub][i] + tile_acc[i])
                    accs[m_sub][n_sub] = vector.from_elements(v4f32, new_vals)

        # ---- Prologue: 2-stage HBM prefetch + 1-stage LDS prefill ----
        # Stage 1: HBM_read[0] -> a_stage  (iter 0 in VGPR)
        a_stage = _hbm_load_a_to_vgpr(fx.Index(0))
        # Prefill 1: a_stage -> LDS  (iter 0 in LDS)
        _vgpr_to_lds_a(a_stage)
        # Stage 2: HBM_read[1] -> a_stage (iter 1 in VGPR; PrefetchStages=2)
        # NOTE: requires K_g >= 2 (asserted at compile time)
        a_stage = _hbm_load_a_to_vgpr(fx.Index(_BLOCK_K))
        # Wait LDS iter 0 visible to all waves
        gpu.barrier()

        # ---- Main loop iter k = 0 .. K_g - 2 ----
        # range_constexpr unrolls fully; flydsl uses Python int k_tile
        for k_tile in range_constexpr(K_g - 1):
            # Step 1: W load + MFMA on LDS (iter k data already there)
            _w_load_and_mfma(k_tile)

            # Step 2: barrier — wait MFMA's LDS reads done before overwriting
            gpu.barrier()

            # Step 3: ds_write iter k+1 to LDS (a_stage holds iter k+1 from
            # previous iter's prefetch / prologue)
            _vgpr_to_lds_a(a_stage)

            # Step 4: issue iter k+2 prefetch (latency hides into next iter's MFMA).
            # Skip on the last main-loop iter (k_tile = K_g - 2) since iter
            # K_g - 1 is the epilogue — no iter K_g to prefetch.
            if k_tile + 2 < K_g:
                a_stage = _hbm_load_a_to_vgpr(fx.Index((k_tile + 2) * _BLOCK_K))

            # Step 5: barrier — LDS iter k+1 visible to next iter's MFMA
            gpu.barrier()

        # ---- Epilogue: last iter k = K_g - 1 ----
        # LDS already has iter K_g-1 data (written in last main-loop iter's
        # Step 3). No further HBM reads or LDS writes needed.
        _w_load_and_mfma(K_g - 1)
        # ====================================================================
        # Output store: each wave writes its 128M x 32N sub-tile direct to HBM.
        # Per lane: M_SUB (8) x N_SUB (2) x 4 rows = 64 bf16 elements.
        # ====================================================================
        out_col = lane % fx.Index(16)
        for m_sub in range_constexpr(_M_SUB):
            for n_sub in range_constexpr(_N_SUB):
                for i in range_constexpr(4):
                    m_idx = (pid_m * fx.Index(_BLOCK_M)
                             + fx.Index(m_sub * 16) + out_row_base + fx.Index(i))
                    n_idx = (pid_n * fx.Index(_BLOCK_N) + wave_n_offset
                             + fx.Index(n_sub * 16) + out_col)
                    bf16_val = _truncf_f32_to_bf16(accs[m_sub][n_sub][i])
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


def flydsl_batched_gemm_fp8_blockwise_v3(
    A: torch.Tensor,
    W: torch.Tensor,
    A_scale: torch.Tensor,
    W_scale: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """v3 wrapper. Constraints: M % 128 == 0, N % 128 == 0, K % 128 == 0,
    K >= 2 * 128 (need at least 2 K-iters for the prefetch pipeline)."""
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

    launcher = compile_bgfp8bw_v3_kernel(B, M, N, K)
    cf = getattr(launcher, "_aiter_cf", None)
    if cf is None:
        cf = flyc.compile(launcher, A, W, A_scale, W_scale, out)
        launcher._aiter_cf = cf
    else:
        cf(A, W, A_scale, W_scale, out)
    return out
