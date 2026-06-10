# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
v2: 4 waves/WG + LDS-A (Phase 1: single buffer, sync HBM->LDS) + 128x128 tile.

Architecture:
  * BLOCK_M=128, BLOCK_N=128, BLOCK_K=128; one WG outputs a 128x128 tile.
  * 4 waves per WG (256 threads). Waves split BLOCK_N: each wave handles
    128 (M) x 32 (N), i.e. M_SUB=8, N_SUB=2 = 16 MFMAs per wave per K-iter.
  * LDS holds the entire A tile (128 rows x 128 K bytes = 16 KB), shared
    across all 4 waves. **Single buffer** in Phase 1 (will add ping/pong
    in Phase 3 if Phase 1 perf justifies it).
  * W loaded direct HBM -> VGPR per wave (each wave loads its own 32-N
    stripe). Matches the blockscale_preshuffle_gemm.py pattern.
  * Native fp8 MFMA `mfma_scale_f32_16x16x128_f8f6f4` with scaleA + scaleB
    fed natively (our advantage over Triton, which uses bf16 MFMA).

Goal: prefill <= 700 us at (B=8, M=4096, N=1024, K=4096) — currently
sw is 1072 us, Triton is 458 us. Closing the structural gap.
See OPTIMIZATION_JOURNEY.md Iter 8 for context and phase plan.
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


def _mfma_scale_32x32x64(result_type, a, b, c, scaleA, scaleB):
    """Functional wrapper for the 32x32x64 scaled fp8 MFMA.

    flydsl 0.1.3.1 only ships a functional wrapper for the 16x16x128 variant;
    for 32x32x64 we call the raw rocdl OpView and unwrap operands ourselves
    (mirrors the 16x16x128 wrapper's body). cbsz/blgp/opselA/opselB = 0 (fp8*fp8).
    Returns the result SSA value (a v16f32, indexable like the 16x16 result).
    """
    return rocdl.mfma_scale_f32_32x32x64_f8f6f4(
        result_type,
        rocdl._unwrap_mfma_operand(a),
        rocdl._unwrap_mfma_operand(b),
        rocdl._unwrap_mfma_operand(c),
        0, 0, 0,
        rocdl._unwrap_mfma_operand(scaleA),
        0,
        rocdl._unwrap_mfma_operand(scaleB),
    ).result


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
def compile_bgfp8bw_m32_kernel(B: int, M: int, N: int, K: int,
                                load_mode: str = "coop",
                                block_m: int = 128,
                                block_n: int = 128,
                                n_waves: int = 4,
                                sched_hint: int = 0):
    """v2 kernel factory. All shape + geometry values baked at compile time.

    Geometry (tunable for per-shape autotuning; defaults reproduce the
    original fixed 128x128 / 4-wave layout exactly):
      * block_m: output rows per WG (multiple of n_waves*8 and of 16).
      * block_n: output cols per WG (<= 128, divisible by n_waves, and
        block_n/n_waves a multiple of 16). <=128 keeps the "one W_scale
        byte covers the whole WG's N" assumption.
      * n_waves: waves per WG (thread count = n_waves*64).
      * BLOCK_K is fixed at 128 (== MFMA K dim and scale-block size).

    load_mode: one of:
      * "coop" (default): all waves cooperate on HBM->LDS A load,
        each wave issues block_m/(n_waves*8) DMA chunks/iter.
      * "solo_unroll" / "solo_loop": ablation-only single-wave variants;
        supported only at the default 128x128 / 4-wave geometry.
    Used for hardware behavior comparison experiments. Production = "coop".
    """
    assert load_mode in ("coop", "solo_unroll", "solo_loop"), \
        f"load_mode must be coop / solo_unroll / solo_loop, got {load_mode!r}"

    # ---- Geometry (local; shadows module-level defaults) ----
    _BLOCK_M = block_m
    _BLOCK_N = block_n
    _BLOCK_K = 128
    _N_WAVES = n_waves
    _WAVE_SIZE = 64
    _BLOCK_THREADS = _N_WAVES * _WAVE_SIZE
    _N_PER_WAVE = _BLOCK_N // _N_WAVES
    _M_SUB = _BLOCK_M // 16
    _N_SUB = _N_PER_WAVE // 16
    # ---- 32x32x64 MFMA grid (m32 variant) ----
    # Each MFMA produces 32M x 32N consuming 32M x 64K (A) and 64K x 32N (B).
    # BLOCK_K=128 -> KRepeat=2 (two 64-K MFMA steps per K-iter, same accum).
    _M_SUB32 = _BLOCK_M // 32              # 32-row M-blocks per wave
    _N_SUB32 = _N_PER_WAVE // 32           # 32-col N-blocks per wave
    _KREP = _BLOCK_K // 64                 # 2
    _LDS_A_BYTES = _BLOCK_M * _BLOCK_K
    # Cooperative HBM->LDS A load: tile = block_m rows x 128 K-bytes. Each lane
    # DMAs 16 bytes (one 16B col of one row); 8 lanes cover a row's 128 bytes,
    # so a wave's 64 lanes cover 8 rows; one chunk (all waves) covers
    # n_waves*8 rows. Number of chunks = block_m / (n_waves*8).
    _A_DMA_ROWS_PER_CHUNK = _N_WAVES * 8
    assert _BLOCK_M % _A_DMA_ROWS_PER_CHUNK == 0, \
        f"block_m={_BLOCK_M} must be divisible by n_waves*8={_A_DMA_ROWS_PER_CHUNK}"
    _A_DMA_CHUNKS = _BLOCK_M // _A_DMA_ROWS_PER_CHUNK

    if load_mode != "coop":
        assert (_BLOCK_M, _BLOCK_N, _N_WAVES) == (128, 128, 4), \
            "solo_* load modes are only supported at the default 128x128/4-wave geometry"

    assert _BLOCK_N <= 128, f"m32 requires block_n <= 128, got {_BLOCK_N}"
    assert _BLOCK_N % _N_WAVES == 0, f"block_n={_BLOCK_N} must divide by n_waves={_N_WAVES}"
    assert _N_PER_WAVE % 32 == 0 and _N_PER_WAVE >= 32, \
        f"block_n/n_waves={_N_PER_WAVE} must be a positive multiple of 32 (32x32 MFMA)"
    assert _BLOCK_M % 32 == 0, f"block_m={_BLOCK_M} must be a multiple of 32 (32x32 MFMA)"
    assert M % _BLOCK_M == 0, f"v2 needs M % {_BLOCK_M} == 0, got M={M}"
    assert N % _BLOCK_N == 0, f"v2 needs N % {_BLOCK_N} == 0, got N={N}"
    assert K % _BLOCK_K == 0, f"v2 needs K % {_BLOCK_K} == 0, got K={K}"

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

        # ---- MFMA 32x32x64 lane mapping (CDNA4) ----
        # A operand (32M x 64K, v8i32/lane): lane l holds
        #   A[l%32, (l//32)*32 + 0..31]    -> row32 = l%32, kgroup = l//32
        # B operand (64K x 32N, v8i32/lane): lane l holds
        #   B[l%32, (l//32)*32 + 0..31]
        # C accum (32M x 32N, v16f32/lane): col = l%32; reg i in 0..15 maps to
        #   row = (i//4)*8 + (l//32)*4 + (i%4)
        row32 = lane % fx.Index(32)
        kgroup = lane // fx.Index(32)                     # [0, 2)
        k_byte_in_lane = kgroup * fx.Index(32)            # 0 or 32 (within a 64-K MFMA)
        k_dword_lane = kgroup * fx.Index(8)               # /4

        # n_block_idx for W_scale: which 128-N block is this WG in.
        n_block_idx = (pid_n * fx.Index(_BLOCK_N)) // fx.Index(128)

        v16f32 = _vec_ty(16, ir.F32Type.get())
        v8i32 = _vec_ty(8, ir.IntegerType.get_signless(32))
        v4i32_t = _vec_ty(4, ir.IntegerType.get_signless(32))
        v16i8_t = _vec_ty(16, T.i8)

        # ---- Per-wave W base (each wave reads its own N stripe) ----
        # For lane l in N-block nb, the W row is wave_n_offset + nb*32 + (l%32).
        w_row_dword_bases = [
            (pid_b * fx.Index(W_BATCH_STRIDE)
             + (pid_n * fx.Index(_BLOCK_N) + wave_n_offset
                + fx.Index(nb * 32) + row32) * fx.Index(K))
            // fx.Index(4)
            for nb in range_constexpr(_N_SUB32)
        ]

        # ---- Per-(m_block, lane) M-row for A_scale lookup ----
        m_row_per_sub = [
            pid_m * fx.Index(_BLOCK_M) + fx.Index(mb * 32) + row32
            for mb in range_constexpr(_M_SUB32)
        ]

        # ---- Initialize M_SUB32 x N_SUB32 accumulators (v16f32) per lane ----
        accs = [
            [fx.Vector.filled(16, 0.0, fx.Float32) for _ in range_constexpr(_N_SUB32)]
            for _ in range_constexpr(_M_SUB32)
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

        # Scalar LDS base for THIS wave (broadcast to all 64 lanes via readfirstlane).
        # extract_aligned_pointer_as_index returns the byte address of the LDS
        # memref's start (= allocator.get_base() + smem_a_offset).
        # In "coop" mode: each wave shifts by wave_id*1024 (4-wave layout).
        # In "solo_*" modes: only wave 0 writes, no per-wave shift needed.
        # Scalar LDS base for THIS wave (broadcast to all 64 lanes via readfirstlane).
        _lds_base_raw_idx = _memref_dialect.extract_aligned_pointer_as_index(lds_a_memref)
        if load_mode == "coop":
            _lds_base_idx = (
                _lds_base_raw_idx
                + wave_id * fx.Index(_WAVE_SIZE * _DMA_BYTES)
            )
        else:
            _lds_base_idx = _lds_base_raw_idx
        _lds_base_i64 = rocdl.readfirstlane(
            T.i64, arith.index_cast(T.i64, _lds_base_idx))
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
        # Done ONCE per WG, then reused K_g iters.
        #
        # A_scale per-WG: block_m × K_g bytes (= 4 KB at the 128 default).
        #   block_threads threads × 16 B/thread = block_threads*16 B per round.
        #   Round r, thread tid loads flat bytes [(r*bt+tid)*16, +16) of the
        #   block_m*K_g region. LDS is row-major: LDS[m_row*K_g + k_iter].
        #   General geometry needs multiple rounds (+ a partial last round).
        #
        # W_scale per-WG: K_g(=32) bytes total. Wave 0 lane 0 loads bytes 0..15,
        #   wave 0 lane 1 loads bytes 16..31. Other lanes idle (scf.if).
        # ====================================================================
        _AS_DMA_BYTES = 16
        _AS_TILE_BYTES = _BLOCK_M * K_g
        _AS_BYTES_PER_ROUND = _BLOCK_THREADS * _AS_DMA_BYTES
        _AS_FULL_ROUNDS = _AS_TILE_BYTES // _AS_BYTES_PER_ROUND
        _AS_REM_BYTES = _AS_TILE_BYTES % _AS_BYTES_PER_ROUND

        # A_scale LDS scalar base for THIS wave (hardware adds lane*16 stride).
        _lds_as_wave_base_idx = (
            _memref_dialect.extract_aligned_pointer_as_index(lds_as_memref)
            + wave_id * fx.Index(_WAVE_SIZE * _AS_DMA_BYTES)
        )
        _lds_as_wave_base_i64 = rocdl.readfirstlane(
            T.i64, arith.index_cast(T.i64, _lds_as_wave_base_idx))
        _hbm_as_base = (
            pid_b * fx.Index(AS_BATCH_STRIDE)
            + pid_m * fx.Index(_AS_TILE_BYTES)
        )

        def _as_dma_round(r):
            _round_off = r * _AS_BYTES_PER_ROUND
            _lds_ptr = _llvm.inttoptr(
                _lds_ptr_type,
                _lds_as_wave_base_i64 + arith.constant(_round_off, type=T.i64))
            _hbm_off = arith.index_cast(
                T.i32,
                _hbm_as_base + tid * fx.Index(_AS_DMA_BYTES) + fx.Index(_round_off))
            rocdl.raw_ptr_buffer_load_lds(
                as_rsrc, _lds_ptr,
                arith.constant(_AS_DMA_BYTES, type=T.i32),
                _hbm_off, _SOFFSET_T, _OFFSET_IMM_T, _AUX_T,
            )

        for _r in range_constexpr(_AS_FULL_ROUNDS):
            _as_dma_round(_r)
        if _AS_REM_BYTES > 0:
            # Partial last round: only threads whose flat byte offset is in
            # range load (guard the LDS write so it doesn't overrun the region).
            _rem_threads = _AS_REM_BYTES // _AS_DMA_BYTES
            _as_in_range = arith.cmpi(
                _arith.CmpIPredicate.slt, tid,
                arith.constant(_rem_threads, type=T.i32))
            _as_if = scf.IfOp(_as_in_range, results_=[], has_else=False)
            with ir.InsertionPoint(_as_if.then_block):
                _as_dma_round(_AS_FULL_ROUNDS)
                scf.YieldOp([])

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
                + n_block_idx * fx.Index(K_g)
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
        # K-LOOP — Phase 1: simple unrolled, sync HBM->LDS, single LDS buffer.
        # Each iter:
        #   1. Cooperative load A[k] tile -> LDS (256 threads * 64 B = 16 KB)
        #   2. Barrier
        #   3. Each wave: load W[k] stripe (32 N x 128 K bytes) -> VGPR
        #   4. Each wave: load A[k] from LDS (M_SUB x 32 B per lane) -> VGPR
        #   5. Load A_scale and W_scale, pack
        #   6. Per wave: M_SUB * N_SUB = 16 MFMAs + accumulate
        #   7. Barrier (before next iter overwrites LDS-A)
        # ====================================================================
        for k_tile in range_constexpr(K_g):
            k_dword_off = fx.Index(k_tile * 32)             # k bytes / 4
            k_byte_off = fx.Index(k_tile * 128)

            # ---------- Phase 3b: async HBM->LDS A load (bypass VGPR) ----------
            if load_mode == "coop":
                # n_waves coop: each wave issues _A_DMA_CHUNKS DMAs per K-iter.
                # block_threads threads total. Per-thread layout per chunk:
                #   absolute_row = chunk*(n_waves*8) + wave_id*8 + lane//8
                #   col_byte = (lane%8)*16
                # _A_DMA_CHUNKS chunks * block_threads * 16 B = block_m*128 B tile.
                for chunk in range_constexpr(_A_DMA_CHUNKS):
                    # LDS ptr advances by block_threads*dma_bytes per chunk
                    _chunk_lds_addr = _lds_base_i64 + arith.constant(
                        chunk * _BLOCK_THREADS * _DMA_BYTES, type=T.i64)
                    _chunk_lds_ptr = _llvm.inttoptr(_lds_ptr_type, _chunk_lds_addr)

                    # Per-lane HBM byte offset (with XOR-swizzle on HBM read side).
                    absolute_row = (fx.Index(chunk * _A_DMA_ROWS_PER_CHUNK)
                                    + wave_id * fx.Index(8)
                                    + row_in_wave)
                    swz_amount = arith.andi(
                        absolute_row, arith.index(_K_BLOCKS16_MASK)) * fx.Index(16)
                    swz_col_byte = col_byte_in_lane ^ swz_amount

                    hbm_byte_off = (
                        pid_b * fx.Index(A_BATCH_STRIDE)
                        + (pid_m * fx.Index(_BLOCK_M) + absolute_row) * fx.Index(K)
                        + k_byte_off + swz_col_byte
                    )
                    _global_offset_i32 = arith.index_cast(T.i32, hbm_byte_off)

                    rocdl.raw_ptr_buffer_load_lds(
                        a_rsrc,
                        _chunk_lds_ptr,
                        _DMA_BYTES_T,
                        _global_offset_i32,
                        _SOFFSET_T,
                        _OFFSET_IMM_T,
                        _AUX_T,
                    )
            elif load_mode == "solo_unroll":
                # Experimental: only wave 0 issues 16 DMAs per K-iter,
                # FULLY UNROLLED via range_constexpr(16). Demonstrated in
                # Iter 8 Phase 3c to cause severe VGPR spill (138 -> 512 VGPR,
                # 6228 B scratch, 778 scratch ops, occupancy 1 wave/SIMD).
                # Result: 3x slower than coop. See OPTIMIZATION_JOURNEY.md.
                _is_wave_0 = arith.cmpi(_arith.CmpIPredicate.eq,
                                         wave_id, arith.index(0))
                _if_op = scf.IfOp(_is_wave_0, results_=[], has_else=False)
                with ir.InsertionPoint(_if_op.then_block):
                    for chunk in range_constexpr(16):
                        _chunk_lds_addr = _lds_base_i64 + arith.constant(
                            chunk * _WAVE_SIZE * _DMA_BYTES, type=T.i64)
                        _chunk_lds_ptr = _llvm.inttoptr(_lds_ptr_type, _chunk_lds_addr)

                        absolute_row = fx.Index(chunk * 8) + row_in_wave
                        swz_amount = arith.andi(
                            absolute_row, arith.index(_K_BLOCKS16_MASK)) * fx.Index(16)
                        swz_col_byte = col_byte_in_lane ^ swz_amount

                        hbm_byte_off = (
                            pid_b * fx.Index(A_BATCH_STRIDE)
                            + (pid_m * fx.Index(_BLOCK_M) + absolute_row) * fx.Index(K)
                            + k_byte_off + swz_col_byte
                        )
                        _global_offset_i32 = arith.index_cast(T.i32, hbm_byte_off)

                        rocdl.raw_ptr_buffer_load_lds(
                            a_rsrc,
                            _chunk_lds_ptr,
                            _DMA_BYTES_T,
                            _global_offset_i32,
                            _SOFFSET_T,
                            _OFFSET_IMM_T,
                            _AUX_T,
                        )
                    scf.YieldOp([])
            else:
                # solo_loop: tries to use scf.for to avoid the unroll-induced
                # spill from solo_unroll. Empirical result (Iter 8 Phase 3c):
                # **DOES NOT WORK as a real loop** — LLVM's loop unroller
                # aggressively unrolls the small constant-bound (16) loop
                # later in the pipeline, undoing the scf.for. Final ISA
                # shows 512 buffer_load_lds (fully unrolled) and the same
                # 512 VGPR / 6212 B scratch / 776 scratch ops as solo_unroll.
                # Wall time identical to solo_unroll (~3x slower than coop).
                # Definitively isolating the LSU/HBM cost would require
                # llvm.loop.unroll.disable metadata, which FlyDSL doesn't
                # currently expose. See OPTIMIZATION_JOURNEY.md Iter 8 Phase 3c.
                _is_wave_0 = arith.cmpi(_arith.CmpIPredicate.eq,
                                         wave_id, arith.index(0))
                _if_op = scf.IfOp(_is_wave_0, results_=[], has_else=False)
                with ir.InsertionPoint(_if_op.then_block):
                    # scf.for runtime loop. CRITICAL: bounds MUST be
                    # arith.index() values, not Python ints — FlyDSL's AST
                    # rewriter unrolls Python int range silently otherwise
                    # (see prefetch-data-load skill, pitfall #1).
                    for chunk_iv, _state in range(arith.index(0),
                                                   arith.index(16),
                                                   arith.index(1),
                                                   init=[]):
                        # All offsets that depend on chunk are computed at
                        # runtime from chunk_iv (an MLIR index value).
                        _chunk_lds_off_i64 = arith.index_cast(
                            T.i64, chunk_iv * fx.Index(_WAVE_SIZE * _DMA_BYTES))
                        _chunk_lds_addr = _lds_base_i64 + _chunk_lds_off_i64
                        _chunk_lds_ptr = _llvm.inttoptr(_lds_ptr_type, _chunk_lds_addr)

                        absolute_row = chunk_iv * fx.Index(8) + row_in_wave
                        swz_amount = arith.andi(
                            absolute_row, arith.index(_K_BLOCKS16_MASK)) * fx.Index(16)
                        swz_col_byte = col_byte_in_lane ^ swz_amount

                        hbm_byte_off = (
                            pid_b * fx.Index(A_BATCH_STRIDE)
                            + (pid_m * fx.Index(_BLOCK_M) + absolute_row) * fx.Index(K)
                            + k_byte_off + swz_col_byte
                        )
                        _global_offset_i32 = arith.index_cast(T.i32, hbm_byte_off)

                        rocdl.raw_ptr_buffer_load_lds(
                            a_rsrc,
                            _chunk_lds_ptr,
                            _DMA_BYTES_T,
                            _global_offset_i32,
                            _SOFFSET_T,
                            _OFFSET_IMM_T,
                            _AUX_T,
                        )
                        results = yield []
                    scf.YieldOp([])

            # ---------- W load co-issued with A DMA ----------
            # 32x32x64 MFMA: per N-block, per krep (2 K-halves of 64) the lane
            # loads its 32 B-bytes = B[w_row, krep*64 + (l//32)*32 + 0..31].
            # b_tiles[nb][krep] is a v8i32.
            b_tiles = [[None] * _KREP for _ in range_constexpr(_N_SUB32)]
            for nb in range_constexpr(_N_SUB32):
                for krep in range_constexpr(_KREP):
                    w_dword_off = (w_row_dword_bases[nb] + k_dword_off
                                   + fx.Index(krep * 16) + k_dword_lane)
                    w_lo = buffer_ops.buffer_load(
                        w_rsrc, w_dword_off, vec_width=4, dtype=T.i32)
                    w_hi = buffer_ops.buffer_load(
                        w_rsrc, w_dword_off + fx.Index(4), vec_width=4, dtype=T.i32)
                    b_tiles[nb][krep] = vector.from_elements(
                        v8i32,
                        [w_lo[0], w_lo[1], w_lo[2], w_lo[3],
                         w_hi[0], w_hi[1], w_hi[2], w_hi[3]],
                    )

            gpu.barrier()

            # ---------- W_scale (from LDS) ----------
            w_scale_byte = ws_scale_lds[fx.Index(k_tile)]
            w_scale_packed = _ue8m0_byte_pack4(w_scale_byte)

            # ---------- A from LDS + A_scale + MFMA (32x32x64, KRepeat=2) ----------
            for mb in range_constexpr(_M_SUB32):
                # A_scale: per M-block M-row = mb*32 + (l%32); one byte per K-iter
                # (same byte for both kreps within the 128-K block).
                local_m_row = fx.Index(mb * 32) + row32
                a_scale_byte = as_scale_lds[local_m_row, fx.Index(k_tile)]
                a_scale_packed = _ue8m0_byte_pack4(a_scale_byte)

                # Load A from LDS for both kreps: lane reads
                #   A[mb*32 + (l%32), krep*64 + (l//32)*32 + 0..31] = 32 bytes.
                a_tiles = [None] * _KREP
                for krep in range_constexpr(_KREP):
                    a_lds_row = fx.Index(mb * 32) + row32
                    col_byte = fx.Index(krep * 64) + k_byte_in_lane
                    swz_lo = _swizzle_xor16(a_lds_row, col_byte)
                    swz_hi = _swizzle_xor16(a_lds_row, col_byte + fx.Index(16))
                    half_lo_b = as_lds.vec_load((a_lds_row, swz_lo), 16)
                    half_hi_b = as_lds.vec_load((a_lds_row, swz_hi), 16)
                    half_lo = vector.bitcast(v4i32_t, half_lo_b)
                    half_hi = vector.bitcast(v4i32_t, half_hi_b)
                    a_tiles[krep] = vector.from_elements(
                        v8i32,
                        [half_lo[0], half_lo[1], half_lo[2], half_lo[3],
                         half_hi[0], half_hi[1], half_hi[2], half_hi[3]],
                    )

                for nb in range_constexpr(_N_SUB32):
                    acc = accs[mb][nb]
                    for krep in range_constexpr(_KREP):
                        acc = _mfma_scale_32x32x64(
                            v16f32,
                            a_tiles[krep], b_tiles[nb][krep], acc,
                            a_scale_packed, w_scale_packed,
                        )
                    accs[mb][nb] = acc

            # ---- Optional sched hints (empirical; bench decides) ----
            # _N_MFMA mfmas, _N_DSRD main A ds_reads, _N_VMEM global loads/iter.
            if sched_hint == 1:
                # Single fence: let LLVM re-pack the region between barriers.
                rocdl.sched_barrier(0)
            elif sched_hint == 2:
                # Interleave: hoist vmem, then pair ds_reads with mfmas.
                _NM = _M_SUB32 * _N_SUB32 * _KREP
                _ND = _M_SUB32 * _KREP * 2
                _NV = _A_DMA_CHUNKS + _N_SUB32 * _KREP * 2
                _d_per = max(1, _ND // _NM)
                rocdl.sched_barrier(0)
                rocdl.sched_vmem(_NV)
                for _j in range_constexpr(_NM):
                    rocdl.sched_dsrd(_d_per)
                    rocdl.sched_mfma(1)
                rocdl.sched_barrier(0)

            # Phase 5b: in coop mode the barrier waits for DMA[k+1] (issued
            # at iter start) to complete + wave sync. In solo modes (legacy)
            # it serves the original "before next iter overwrites LDS" purpose.
            gpu.barrier()

        # ====================================================================
        # Output store: 32x32x64 C layout. lane l writes 16 f32:
        #   col(n) = l % 32; reg i -> row(m) = (i//4)*8 + (l//32)*4 + (i%4).
        # ====================================================================
        out_col = lane % fx.Index(32)
        out_row_grp = (lane // fx.Index(32)) * fx.Index(4)   # 0 or 4
        for mb in range_constexpr(_M_SUB32):
            for nb in range_constexpr(_N_SUB32):
                n_idx = (pid_n * fx.Index(_BLOCK_N) + wave_n_offset
                         + fx.Index(nb * 32) + out_col)
                for i in range_constexpr(16):
                    m_idx = (pid_m * fx.Index(_BLOCK_M) + fx.Index(mb * 32)
                             + fx.Index((i // 4) * 8) + out_row_grp
                             + fx.Index(i % 4))
                    bf16_val = _truncf_f32_to_bf16(accs[mb][nb][i])
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


def flydsl_batched_gemm_fp8_blockwise_m32(
    A: torch.Tensor,
    W: torch.Tensor,
    A_scale: torch.Tensor,
    W_scale: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    load_mode: str = "coop",
    block_m: int = 128,
    block_n: int = 128,
    n_waves: int = 4,
    sched_hint: int = 0,
) -> torch.Tensor:
    """v2 wrapper. Constraints: M % block_m == 0, N % block_n == 0, K % 128 == 0.

    block_m / block_n / n_waves: per-WG tile geometry (see
    compile_bgfp8bw_m32_kernel). Defaults reproduce the original 128x128/4-wave
    layout; pass other values for per-shape tile autotuning.

    load_mode: HBM->LDS A load strategy. One of:
      * "coop" (default, production): n_waves-wave cooperative load.
      * "solo_unroll": single wave 0 with unrolled 16-chunk loop (causes
        VGPR spill — for ablation experiments only; default geometry only).
      * "solo_loop": single wave 0 with scf.for runtime loop (no unroll;
        isolates true LSU cost without spill confound; default geometry only).
    """
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

    launcher = compile_bgfp8bw_m32_kernel(
        B, M, N, K, load_mode=load_mode,
        block_m=block_m, block_n=block_n, n_waves=n_waves,
        sched_hint=sched_hint)
    cf = getattr(launcher, "_aiter_cf", None)
    if cf is None:
        cf = flyc.compile(launcher, A, W, A_scale, W_scale, out)
        launcher._aiter_cf = cf
    else:
        cf(A, W, A_scale, W_scale, out)
    return out
