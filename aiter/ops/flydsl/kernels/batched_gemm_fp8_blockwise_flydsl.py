# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL FP8 block-wise batched GEMM (DeepSeek V4 ``wo_a`` path).

File organisation mirrors ``small_m_hgemm.py`` so the two kernels can be read
with the same mental model:

  - top-level constants and tile options
  - small helper utilities (UE8M0 scale, bf16 truncation, vector type)
  - host-side scale conversion
  - kernel-name builder
  - ``compile_bgfp8bw_kernel``: geometry derivation → @flyc.kernel body →
    @flyc.jit launcher
  - shape-policy helpers (``_pick_block_m`` / ``_pick_block_n``)
  - public entry point ``flydsl_batched_gemm_fp8_blockwise``

The kernel body factors the per-iteration work into closures (``_load_*``,
``_emit_mma``) shared by both the unrolled K-loop (decode / small K) and the
``scf.for`` prefetch K-loop (prefill / multi-K). This kills the W-tile
assembly + scale-load + MFMA call triplication that the previous
mega-``if native_scale_mfma`` branch carried.

Two MMA modes survive, distinguished only inside ``_emit_mma``:

  * NATIVE_SCALE_MFMA=True  (BLOCK_M ≤ 32, decode hot path)
      per-row A_scale fed to MFMA ``scaleA`` slot; no post-MFMA fp32 mul
  * NATIVE_SCALE_MFMA=False (BLOCK_M == 64, prefill)
      MFMA scaleA pinned to 0x7F7F7F7F (=1.0); A_scale applied post-MFMA as
      per-row fp32 multiply on the accumulator (keeps the MFMA critical path
      free of per-lane byte loads)

W_scale is always shared across N_SUB inside a 128-N block (a single byte
covers all N sub-tiles in this WG; see notes below).

The HBM-direct load path is preserved end-to-end. An LDS + ``ASYNC_COPY``
double-buffered pipeline (mirroring small_m's ``ldg_sts_*_async`` /
``lds_matrix_*``) is the planned next step; the closures here are shaped so
that path can drop in by replacing ``_load_a_pair`` / ``_load_w_pair`` with
LDS-resident variants without touching ``_emit_mma`` or the loop drivers.
"""

from __future__ import annotations

import functools
from typing import Optional

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith as _arith
from flydsl._mlir.dialects import gpu as _gpu_dialect  # noqa: F401  (registered via import)
from flydsl._mlir.dialects import llvm, memref
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, gpu, range_constexpr, rocdl, vector
from flydsl.expr.typing import T, Uint8, _to_raw
from flydsl.expr.utils.arith import unwrap as _unwrap
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr


# ---- top-level constants (mirrors small_m_hgemm) ---------------------------
# FP8 block-wise BMM is single-wave per WG; no STAGES / WARP_N_STEPS knobs
# until the LDS pipeline lands.
WAVE_SIZE = 64
MFMA_K = 128                    # mfma_scale_f32_16x16x128_f8f6f4 K dim
MFMA_M = 16
MFMA_N = 16
DTYPE_BYTES = 1                 # fp8 element bytes
BLOCK_M_OPTIONS = (16, 32, 64)
BLOCK_N_OPTIONS = (16, 32)
NATIVE_SCALE_BLOCK_M_MAX = 32   # native-scaleA path enabled iff BLOCK_M ≤ this
# LDS + ASYNC_COPY pipeline (POST_SCALE / prefill path):
LDS_STAGES = 2
LDS_DMA_BYTES = 16              # raw_ptr_buffer_load_lds chunk = 16 bytes/lane
MAX_LDS_BYTES = 163840          # gfx950 LDS cap, used for asserts


# ---- small helpers ---------------------------------------------------------
def _vec_ty(n: int, mlir_elem):
    return ir.VectorType.get([n], mlir_elem)


def _truncf_f32_to_bf16(val_f32):
    """f32 scalar → bf16 scalar via raw MLIR arith.TruncFOp."""
    return _arith.TruncFOp(T.bf16, _to_raw(val_f32)).result


def _ue8m0_byte_to_f32(byte_val):
    """UE8M0 byte → fp32 via zero-cost bit ops.

    UE8M0 byte ``u`` is ``2^(u - 127)``. fp32 shares bias 127, so placing the
    byte into the fp32 exponent field (sign=0, mantissa=0) yields exactly
    ``2^(u - 127)``. Sequence: i8 → zext i32 → shl 23 → bitcast f32.
    """
    raw = _to_raw(byte_val)
    i32_val = _arith.ExtUIOp(T.i32, raw).result
    c23 = _arith.ConstantOp(T.i32, ir.IntegerAttr.get(T.i32, 23)).result
    shifted = _arith.ShLIOp(i32_val, c23).result
    return _arith.BitcastOp(T.f32, shifted).result


def _ue8m0_byte_pack4(byte_val):
    """Replicate one UE8M0 byte into an i32 (4× E8M0 bytes packed).

    The MFMA scaleA / scaleB inputs are i32 holding 4 packed E8M0 bytes
    (one per 32-K-element sub-tile). V4's per-128-K scale shares the same
    byte across all 4 sub-tiles, so broadcasting is correct. One mul, no
    shifts: ``u * 0x01010101``.
    """
    raw = _to_raw(byte_val)
    i32_val = _arith.ExtUIOp(T.i32, raw).result
    c_replicate = _arith.ConstantOp(
        T.i32, ir.IntegerAttr.get(T.i32, 0x01010101)
    ).result
    return _arith.MulIOp(i32_val, c_replicate).result


def _torch_scales_to_ue8m0(scales: torch.Tensor) -> torch.Tensor:
    """Host-side: fp32 multiplicative scales → uint8 UE8M0 bytes (pass-through
    if already uint8). ``u = round(log2(s)) + 127`` clamped to [0, 255];
    non-positive entries become 0.
    """
    if scales.dtype == torch.uint8:
        return scales.contiguous()
    if scales.dtype != torch.float32:
        raise TypeError(
            "FlyDSL fp8 blockwise expects A_scale/W_scale as torch.uint8 "
            f"(UE8M0) or torch.float32; got {scales.dtype}"
        )
    s = scales.float()
    pos = s > 0
    out = torch.zeros_like(s, dtype=torch.uint8)
    tiny = torch.tensor(
        torch.finfo(torch.float32).tiny, device=s.device, dtype=torch.float32
    )
    safe = torch.where(pos, s, tiny)
    biased = (torch.log2(safe) + 127.0).round().clamp(0, 255).to(torch.uint8)
    out[pos] = biased[pos]
    return out.contiguous()


# ---- kernel-name builder (mirrors small_m_kernel_name) ---------------------
def bmm_fp8_blockwise_kernel_name(
    B: int, M: int, N: int, K: int, *,
    block_m: int, block_n: int, native_scale: bool,
) -> str:
    name = f"bgfp8bw_{B}x{M}x{N}x{K}_BM{block_m}_BN{block_n}"
    name += "_NS" if native_scale else "_PS"   # NativeScale vs PostScale
    return name


# ---- main compile entry ----------------------------------------------------
@functools.lru_cache(maxsize=None)
def compile_bgfp8bw_kernel(
    B: int, M: int, N: int, K: int,
    BLOCK_M: int = 16, BLOCK_N: int = 16,
    *,
    scale_is_u8: bool = False,
    native_scale_mfma: bool = True,
    use_lds_pipeline: bool = False,
):
    """Build the kernel + launcher for fixed (B, M, N, K) and (BLOCK_M, BLOCK_N).

    BLOCK_M ∈ {16, 32, 64}: stacks M_SUB = BLOCK_M // 16 MFMAs in M per WG.
    BLOCK_N ∈ {16, 32}:     stacks N_SUB = BLOCK_N // 16 MFMAs in N per WG;
                            shares one W_scale byte across all N sub-tiles
                            (BLOCK_N ≤ 32 ≤ 128-N W_scale block).

    use_lds_pipeline: opt-in LDS + ``raw_ptr_buffer_load_lds`` double-buffered
        pipeline for the POST_SCALE prefill K-loop (multi-K only). Default is
        the HBM-direct loop-carried prefetch path (bit-exact with the original
        kernel). Only takes effect when NATIVE_SCALE_MFMA is False (i.e.
        BLOCK_M == 64) and K_g > 1. NATIVE_SCALE decode paths are unaffected
        — their unrolled HBM-direct form is already decode-optimal.
    """
    del scale_is_u8  # accepted for back-compat; scales are converted on host

    # ---- validate ----------------------------------------------------------
    assert K % MFMA_K == 0, f"K={K} must be a multiple of {MFMA_K}"
    assert M % BLOCK_M == 0, f"M={M} must be a multiple of BLOCK_M={BLOCK_M}"
    assert N % 128 == 0, f"N={N} must be a multiple of 128"
    assert N % BLOCK_N == 0, f"N={N} must be a multiple of BLOCK_N={BLOCK_N}"
    assert BLOCK_M in BLOCK_M_OPTIONS, f"BLOCK_M must be in {BLOCK_M_OPTIONS}, got {BLOCK_M}"
    assert BLOCK_N in BLOCK_N_OPTIONS, f"BLOCK_N must be in {BLOCK_N_OPTIONS}, got {BLOCK_N}"
    # Auto-disable native-scaleA on prefill widths where its per-lane byte
    # loads hurt more than they save (cf. _pick_block_m heuristic notes).
    NATIVE_SCALE_MFMA = bool(native_scale_mfma) and (BLOCK_M <= NATIVE_SCALE_BLOCK_M_MAX)

    # ---- geometry ----------------------------------------------------------
    M_SUB = BLOCK_M // MFMA_M
    N_SUB = BLOCK_N // MFMA_N
    GRID_M = M // BLOCK_M
    GRID_N = N // BLOCK_N
    K_g = K // MFMA_K                   # K-loop trip count
    N_g = N // 128                      # 128-N W_scale block count (per batch)
    # Element-unit strides (buffer_load addresses A/W in dwords, so the K-loop
    # converts row-offsets via // 4).
    A_BATCH_STRIDE = M * K
    W_BATCH_STRIDE = N * K
    O_BATCH_STRIDE = M * N
    AS_BATCH_STRIDE = M * K_g           # A_scale per (batch, m, k_tile)
    WS_BATCH_STRIDE = N_g * K_g         # W_scale per (batch, n_block, k_tile)

    # ---- LDS pipeline setup (POST_SCALE prefill only, opt-in) -------------
    USE_LDS_PIPELINE = bool(use_lds_pipeline) and (not NATIVE_SCALE_MFMA) and (K_g > 1)
    if USE_LDS_PIPELINE:
        GPU_ARCH = get_rocm_arch()
        # 16 byte (= LDS_DMA_BYTES) per-lane async chunks, single wave per WG.
        LDG_A_X_THREADS = MFMA_K // LDS_DMA_BYTES                   # slots per row
        LDG_A_TOTAL_VECS = (BLOCK_M * MFMA_K) // LDS_DMA_BYTES
        LDG_A_PER_LANE = (LDG_A_TOTAL_VECS + WAVE_SIZE - 1) // WAVE_SIZE
        LDG_W_X_THREADS = MFMA_K // LDS_DMA_BYTES
        LDG_W_TOTAL_VECS = (BLOCK_N * MFMA_K) // LDS_DMA_BYTES
        LDG_W_PER_LANE = (LDG_W_TOTAL_VECS + WAVE_SIZE - 1) // WAVE_SIZE
        assert LDG_A_TOTAL_VECS % WAVE_SIZE == 0, (
            f"LDS A tile not evenly divisible across the wave: "
            f"{LDG_A_TOTAL_VECS} not multiple of {WAVE_SIZE}"
        )
        assert LDG_W_TOTAL_VECS % WAVE_SIZE == 0, (
            f"LDS W tile not evenly divisible across the wave: "
            f"{LDG_W_TOTAL_VECS} not multiple of {WAVE_SIZE}"
        )
        # Row stride padded by 16 bytes to break the 32-bank wraparound
        # alias (row stride 144 ≠ multiple of 32 dword banks); shrinks
        # ds_read bank conflicts from ~16-way to ~1-way.
        LDS_ROW_PAD = 0
        LDS_ROW_STRIDE = MFMA_K + LDS_ROW_PAD       # = 144 bytes
        assert LDS_ROW_STRIDE % LDS_DMA_BYTES == 0   # vec store alignment   # vec store alignment
        assert LDS_ROW_STRIDE % 4 == 0               # dword view alignment
        A_LDS_BYTES = LDS_STAGES * BLOCK_M * LDS_ROW_STRIDE
        W_LDS_BYTES = LDS_STAGES * BLOCK_N * LDS_ROW_STRIDE
        TOTAL_LDS_BYTES = A_LDS_BYTES + W_LDS_BYTES
        assert TOTAL_LDS_BYTES <= MAX_LDS_BYTES, (
            f"LDS usage {TOTAL_LDS_BYTES} exceeds cap {MAX_LDS_BYTES}"
        )
        _allocator = SmemAllocator(None, arch=GPU_ARCH, global_sym_name="smem")
        _smem_a_off = _allocator._align(_allocator.ptr, 16)
        _allocator.ptr = _smem_a_off + A_LDS_BYTES
        _smem_w_off = _allocator._align(_allocator.ptr, 16)
        _allocator.ptr = _smem_w_off + W_LDS_BYTES
    else:
        _allocator = None
        _smem_a_off = _smem_w_off = 0
        LDG_A_X_THREADS = LDG_A_PER_LANE = 0
        LDG_W_X_THREADS = LDG_W_PER_LANE = 0
        A_LDS_BYTES = W_LDS_BYTES = 0
        LDS_ROW_STRIDE = MFMA_K

    KERNEL_NAME = bmm_fp8_blockwise_kernel_name(
        B, M, N, K,
        block_m=BLOCK_M, block_n=BLOCK_N, native_scale=NATIVE_SCALE_MFMA,
    )
    if USE_LDS_PIPELINE:
        KERNEL_NAME += "_LDS"

    @flyc.kernel
    def kernel(
        A_ptr: fx.Tensor,
        W_ptr: fx.Tensor,
        A_scale_ptr: fx.Tensor,
        W_scale_ptr: fx.Tensor,
        Out_ptr: fx.Tensor,
    ):
        # ---- lane / WG indexing -------------------------------------------
        lane = fx.thread_idx.x
        pid_m = fx.block_idx.x          # BLOCK_M tile index along M
        pid_n = fx.block_idx.y          # BLOCK_N tile index along N
        pid_b = fx.block_idx.z          # batch index

        # MFMA 16x16x128 lane → data convention:
        #   A operand: 32 bytes from row (l % 16), starting K-byte (l/16)*32
        #   B operand: 32 bytes from row (l % 16), starting K-byte (l/16)*32
        #   C accum  : 4 fp32 from rows (l/16)*4 + 0..3, col (l % 16)
        row = lane % fx.Index(16)
        k_dword_lane = (lane // fx.Index(16)) * fx.Index(8)  # 8 dwords = 32 bytes

        # Per-(m_sub, lane) A row-base in DWORDS (// 4 converts bytes → dwords).
        # m-row of this lane's A slice: pid_m*BLOCK_M + sub*16 + (lane % 16).
        a_row_dword_bases = [
            (pid_b * fx.Index(A_BATCH_STRIDE)
             + (pid_m * fx.Index(BLOCK_M) + fx.Index(sub * MFMA_M) + row)
               * fx.Index(K))
             // fx.Index(4)
            for sub in range_constexpr(M_SUB)
        ]
        # Per-(n_sub, lane) W row-base in DWORDS (W layout is (N, K)).
        w_row_dword_bases = [
            (pid_b * fx.Index(W_BATCH_STRIDE)
             + (pid_n * fx.Index(BLOCK_N) + fx.Index(n_sub * MFMA_N) + row)
               * fx.Index(K))
             // fx.Index(4)
            for n_sub in range_constexpr(N_SUB)
        ]

        a_rsrc = buffer_ops.create_buffer_resource(A_ptr, max_size=True)
        w_rsrc = buffer_ops.create_buffer_resource(W_ptr, max_size=True)
        as_rsrc = buffer_ops.create_buffer_resource(A_scale_ptr, max_size=True)
        ws_rsrc = buffer_ops.create_buffer_resource(W_scale_ptr, max_size=True)
        o_rsrc = buffer_ops.create_buffer_resource(Out_ptr, max_size=True)

        # Lane l writes 4 consecutive M rows starting at this base.
        out_row_base = (lane // fx.Index(16)) * fx.Index(4)
        # All N sub-tiles in this WG share one W_scale byte (BLOCK_N ≤ 32 < 128).
        n_block_idx = (pid_n * fx.Index(BLOCK_N)) // fx.Index(128)
        v4f32 = _vec_ty(4, ir.F32Type.get())
        v8i32 = _vec_ty(8, ir.IntegerType.get_signless(32))

        # Per-(m_sub, i ∈ 0..3) output row: pid_m*BLOCK_M + sub*16 + lane/16*4 + i
        m_idx_per_sub = [
            [pid_m * fx.Index(BLOCK_M) + fx.Index(sub * MFMA_M)
                + out_row_base + fx.Index(i)
             for i in range_constexpr(4)]
            for sub in range_constexpr(M_SUB)
        ]

        # Accumulators: M_SUB × N_SUB of vec<4 x f32>.
        accs = [
            [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range_constexpr(N_SUB)]
            for _ in range_constexpr(M_SUB)
        ]

        # LDS memref views (i32 element type so each linear index addresses a
        # dword; load via raw vector.load_op + manual linear offset compute).
        # The async store side uses raw byte addressing via
        # extract_aligned_pointer_as_index, so the view dtype is incidental
        # for stores.
        if USE_LDS_PIPELINE:
            _base_ptr = _allocator.get_base()
            _as_smem = SmemPtr(
                _base_ptr, _smem_a_off, T.i32,
                shape=(LDS_STAGES * BLOCK_M * LDS_ROW_STRIDE // 4,),
            )
            _ws_smem = SmemPtr(
                _base_ptr, _smem_w_off, T.i32,
                shape=(LDS_STAGES * BLOCK_N * LDS_ROW_STRIDE // 4,),
            )
            as_lds_memref = _as_smem.get()
            ws_lds_memref = _ws_smem.get()

        # =========================================================
        # Shared closures: load helpers + the unified _emit_mma
        # =========================================================

        def _load_a_pair(k_dword_off, sub):
            """HBM-load one (lo, hi) v4i32 pair for A[sub] at k_dword_off."""
            off = a_row_dword_bases[sub] + k_dword_lane + k_dword_off
            lo = buffer_ops.buffer_load(a_rsrc, off, vec_width=4, dtype=T.i32)
            hi = buffer_ops.buffer_load(a_rsrc, off + fx.Index(4),
                                        vec_width=4, dtype=T.i32)
            return lo, hi

        def _load_w_pair(k_dword_off, n_sub):
            """HBM-load one (lo, hi) v4i32 pair for W[n_sub] at k_dword_off."""
            off = w_row_dword_bases[n_sub] + k_dword_lane + k_dword_off
            lo = buffer_ops.buffer_load(w_rsrc, off, vec_width=4, dtype=T.i32)
            hi = buffer_ops.buffer_load(w_rsrc, off + fx.Index(4),
                                        vec_width=4, dtype=T.i32)
            return lo, hi

        def _pair_to_v8i32(pair):
            lo, hi = pair
            return vector.from_elements(
                v8i32,
                [lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]],
            )

        def _load_w_scale_packed(k_tile_idx_val):
            """Load the shared (across N_SUB) W_scale byte and pack 4× into i32."""
            off = (pid_b * fx.Index(WS_BATCH_STRIDE)
                   + n_block_idx * fx.Index(K_g)
                   + k_tile_idx_val)
            u8 = buffer_ops.buffer_load(ws_rsrc, off, vec_width=1, dtype=Uint8)
            return _ue8m0_byte_pack4(u8)

        def _load_a_scale_packed(sub, k_tile_idx_val):
            """Native path: per-lane A_scale byte packed 4× into i32 (one byte
            per (lane-row, k_tile))."""
            a_scale_row_idx = (pid_m * fx.Index(BLOCK_M)
                               + fx.Index(sub * MFMA_M) + row)
            off = (pid_b * fx.Index(AS_BATCH_STRIDE)
                   + a_scale_row_idx * fx.Index(K_g)
                   + k_tile_idx_val)
            u8 = buffer_ops.buffer_load(as_rsrc, off, vec_width=1, dtype=Uint8)
            return _ue8m0_byte_pack4(u8)

        def _load_a_scale_per_row_f32(sub, k_tile_idx_val):
            """Post-mul path: per-output-row A_scale as 4 fp32 (one per
            accumulator row)."""
            scales = []
            for i in range_constexpr(4):
                off = (pid_b * fx.Index(AS_BATCH_STRIDE)
                       + m_idx_per_sub[sub][i] * fx.Index(K_g)
                       + k_tile_idx_val)
                u8 = buffer_ops.buffer_load(as_rsrc, off,
                                            vec_width=1, dtype=Uint8)
                scales.append(_ue8m0_byte_to_f32(u8))
            return scales

        def _emit_mma(a_pairs, w_pairs, k_tile_idx_val, accs_in):
            """One K-iter: build operands, load scales, emit M_SUB×N_SUB MFMAs,
            accumulate. Branches on NATIVE_SCALE_MFMA only here.

            a_pairs: list[M_SUB] of (lo, hi) v4i32 pairs (A operand source)
            w_pairs: list[N_SUB] of (lo, hi) v4i32 pairs (W operand source)
            k_tile_idx_val: fx.Index value of the K-tile index (for scale offset)
            accs_in: 2D list [M_SUB][N_SUB] of vec<4xf32> accumulators
            Returns: 2D list of vec<4xf32> with the iter's contribution added.
            """
            w_scale_packed = _load_w_scale_packed(k_tile_idx_val)
            b_tiles = [_pair_to_v8i32(w_pairs[n_sub])
                       for n_sub in range_constexpr(N_SUB)]

            accs_out = [[None] * N_SUB for _ in range_constexpr(M_SUB)]
            for sub in range_constexpr(M_SUB):
                a128 = _pair_to_v8i32(a_pairs[sub])

                if NATIVE_SCALE_MFMA:
                    a_scale_packed = _load_a_scale_packed(sub, k_tile_idx_val)
                    a_scale_per_row = None
                else:
                    a_scale_packed = 0x7F7F7F7F   # 1.0 placeholder
                    a_scale_per_row = _load_a_scale_per_row_f32(
                        sub, k_tile_idx_val
                    )

                for n_sub in range_constexpr(N_SUB):
                    tile_acc = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                        T.f32x4,
                        [a128, b_tiles[n_sub],
                         fx.Vector.filled(4, 0.0, fx.Float32),
                         0, 0, 0,
                         a_scale_packed, 0, w_scale_packed],
                    )
                    new_vals = []
                    for i in range_constexpr(4):
                        contrib = tile_acc[i]
                        if not NATIVE_SCALE_MFMA:
                            contrib = contrib * a_scale_per_row[i]
                        new_vals.append(accs_in[sub][n_sub][i] + contrib)
                    accs_out[sub][n_sub] = vector.from_elements(v4f32, new_vals)
            return accs_out

        # =========================================================
        # LDS + async-copy helpers (POST_SCALE prefill only)
        # =========================================================
        if USE_LDS_PIPELINE:
            _lds_ptr_type = ir.Type.parse("!llvm.ptr<3>")

            _v4i32 = T.vec(4, T.i32)

            # Bank-conflict mitigation via row padding (no XOR swizzle).
            # Row stride = MFMA_K bytes = 128 bytes = 32 LDS dwords = exactly
            # one full bank wraparound, so an unpadded layout gives ~16-way
            # bank conflicts on every ds_read_b128. Add 16 bytes of pad per
            # row → row stride 144 bytes / 36 dwords (not a multiple of 32
            # banks), breaking the alias and dropping conflicts to ~1-way.
            # Padding is allocator-side only (no per-access address transform),
            # so it's bit-identical to the unpadded layout from the MFMA's
            # point of view.

            def _ldg_sts_async(rsrc, lds_memref, lds_byte_base,
                               row_block_base_elem, batch_stride,
                               x_threads, per_lane, k_byte_off):
                """Async-copy a tile from a global buffer resource into LDS.

                Each lane issues ``per_lane`` raw_ptr_buffer_load_lds calls,
                each moving LDS_DMA_BYTES (=16) bytes. Slot s ∈ [0, per_lane)
                maps to (row_local, k_local_bytes) via:
                    slot = WAVE_SIZE * s + lane
                    row_local = slot // x_threads
                    k_local_bytes = (slot % x_threads) * LDS_DMA_BYTES
                """
                for i in range_constexpr(per_lane):
                    slot = fx.Index(WAVE_SIZE * i) + lane
                    row_local = slot // fx.Index(x_threads)
                    k_local_bytes = (slot % fx.Index(x_threads)) * fx.Index(LDS_DMA_BYTES)
                    # Global byte offset (FP8: byte offset == element offset).
                    row_global = row_block_base_elem + row_local
                    global_byte = (pid_b * fx.Index(batch_stride)
                                   + row_global * fx.Index(K)
                                   + k_byte_off + k_local_bytes)
                    global_byte_i32 = arith.index_cast(T.i32, global_byte)
                    # LDS byte offset: padded row stride breaks bank-conflict
                    # alias for ds_read; no address transform needed.
                    lds_byte = (lds_byte_base
                                + row_local * fx.Index(LDS_ROW_STRIDE)
                                + k_local_bytes)
                    lds_addr = (memref.extract_aligned_pointer_as_index(lds_memref)
                                + lds_byte)
                    lds_addr_i64 = rocdl.readfirstlane(
                        T.i64, arith.index_cast(T.i64, lds_addr)
                    )
                    lds_ptr = llvm.inttoptr(_lds_ptr_type, lds_addr_i64)
                    rocdl.raw_ptr_buffer_load_lds(
                        rsrc, lds_ptr,
                        arith.constant(LDS_DMA_BYTES, type=T.i32),
                        global_byte_i32,
                        arith.constant(0, type=T.i32),
                        arith.constant(0, type=T.i32),
                        arith.constant(1, type=T.i32),
                    )

            def _ldg_sts_a_async(k_byte_off, lds_stage):
                lds_byte_base = fx.Index(lds_stage * BLOCK_M * LDS_ROW_STRIDE)
                row_base = pid_m * fx.Index(BLOCK_M)
                _ldg_sts_async(
                    a_rsrc, as_lds_memref, lds_byte_base,
                    row_base, A_BATCH_STRIDE,
                    LDG_A_X_THREADS, LDG_A_PER_LANE,
                    k_byte_off,
                )

            def _ldg_sts_w_async(k_byte_off, lds_stage):
                lds_byte_base = fx.Index(lds_stage * BLOCK_N * LDS_ROW_STRIDE)
                row_base = pid_n * fx.Index(BLOCK_N)
                _ldg_sts_async(
                    w_rsrc, ws_lds_memref, lds_byte_base,
                    row_base, W_BATCH_STRIDE,
                    LDG_W_X_THREADS, LDG_W_PER_LANE,
                    k_byte_off,
                )

            def _wait_vmem_lds():
                """Block until all in-flight raw_ptr_buffer_load_lds have
                committed to both VMEM (vmcnt) and LDS (lgkmcnt). gpu.barrier
                after this ensures every lane sees the loaded data."""
                llvm.InlineAsmOp(
                    None, [],
                    "s_waitcnt vmcnt(0) lgkmcnt(0)", "",
                    has_side_effects=True,
                )

            def _lds_load_pair(lds_memref, lds_stage, rows_per_stage, sub_idx,
                               atom_size):
                """Read this lane's 32-byte fragment from a (stage, row, k) LDS
                tile as two v4i32 halves. Uses the padded row stride.

                rows_per_stage: BLOCK_M (for A) or BLOCK_N (for W)
                atom_size:      MFMA_M (=16) for A; MFMA_N (=16) for W
                """
                row_in_tile = fx.Index(sub_idx * atom_size) + row  # row = lane % 16
                k_dword_start = (lane // fx.Index(16)) * fx.Index(8)
                stage_base = fx.Index(lds_stage * rows_per_stage * LDS_ROW_STRIDE // 4)
                row_dword_base = stage_base + row_in_tile * fx.Index(LDS_ROW_STRIDE // 4)
                idx_lo = row_dword_base + k_dword_start
                idx_hi = idx_lo + fx.Index(4)
                lo = vector.load_op(_v4i32, lds_memref, [idx_lo])
                hi = vector.load_op(_v4i32, lds_memref, [idx_hi])
                return lo, hi

            def _lds_load_a_pair(lds_stage, sub):
                return _lds_load_pair(as_lds_memref, lds_stage, BLOCK_M, sub,
                                      MFMA_M)

            def _lds_load_w_pair(lds_stage, n_sub):
                return _lds_load_pair(ws_lds_memref, lds_stage, BLOCK_N, n_sub,
                                      MFMA_N)

        # =========================================================
        # K-loop drivers — pick form per K_g and NATIVE_SCALE_MFMA
        # =========================================================
        if K_g == 1:
            # Trivial fast path: one MFMA per (m_sub, n_sub), no loop, no state.
            k_zero = fx.Index(0)
            a_pairs = [_load_a_pair(k_zero, s) for s in range_constexpr(M_SUB)]
            w_pairs = [_load_w_pair(k_zero, n) for n in range_constexpr(N_SUB)]
            accs_new = _emit_mma(a_pairs, w_pairs,
                                 arith.constant(0, index=True), accs)
            for sub in range_constexpr(M_SUB):
                for n_sub in range_constexpr(N_SUB):
                    accs[sub][n_sub] = accs_new[sub][n_sub]

        elif NATIVE_SCALE_MFMA:
            # Decode K-loop: fully unrolled. Per-K-iter overhead is small at
            # decode M sizes, the unroll exposes MFMA parallelism, and the
            # native-scaleA path keeps per-lane byte loads off the post-MFMA
            # critical path.
            for k_tile in range_constexpr(K_g):
                k_off = fx.Index(k_tile * 32)
                a_pairs = [_load_a_pair(k_off, s) for s in range_constexpr(M_SUB)]
                w_pairs = [_load_w_pair(k_off, n) for n in range_constexpr(N_SUB)]
                accs_new = _emit_mma(a_pairs, w_pairs,
                                     arith.constant(k_tile, index=True), accs)
                for sub in range_constexpr(M_SUB):
                    for n_sub in range_constexpr(N_SUB):
                        accs[sub][n_sub] = accs_new[sub][n_sub]

        elif USE_LDS_PIPELINE:
            # Prefill K-loop, LDS + ``raw_ptr_buffer_load_lds`` double-buffered
            # variant. Compared to the HBM-direct prefetch loop below, this
            # keeps the scf.for state tiny (k_tile_idx + stage + accs) by
            # parking prefetched tiles in LDS instead of carrying register
            # values through the loop-carried block arg list. The stage
            # pingpongs 0↔1 each iter.
            _ldg_sts_a_async(fx.Index(0), 0)
            _ldg_sts_w_async(fx.Index(0), 0)
            _wait_vmem_lds()
            gpu.barrier()

            def _pack_state_lds(k_idx, stage_idx, acc_grid):
                out = [k_idx, stage_idx]
                for s in range_constexpr(M_SUB):
                    for n in range_constexpr(N_SUB):
                        out.append(acc_grid[s][n])
                return [_unwrap(v) for v in out]

            def _unpack_state_lds(state):
                k_idx = state[0]
                stage_idx = state[1]
                idx = 2
                acc_grid = [[None] * N_SUB for _ in range_constexpr(M_SUB)]
                for s in range_constexpr(M_SUB):
                    for n in range_constexpr(N_SUB):
                        acc_grid[s][n] = state[idx]
                        idx += 1
                return k_idx, stage_idx, acc_grid

            init_state = _pack_state_lds(
                arith.constant(0, index=True),
                arith.constant(0, index=True),
                accs,
            )

            for _bki, state in range(0, K_g - 1, 1, init=init_state):
                k_tile_idx, cur_stage, accs_cur = _unpack_state_lds(state)

                # Issue prefetch for k+1 into the *other* stage.
                k_next = k_tile_idx + arith.constant(1, index=True)
                next_stage = arith.constant(1, index=True) - cur_stage
                k_byte_off_next = k_next * fx.Index(MFMA_K)
                # cur_stage / next_stage are runtime index values. We can't
                # range_constexpr-dispatch on them; emit both stage variants
                # under an scf.if so the LDS byte offset is correctly computed
                # for whichever stage holds the prefetch target.
                #
                # In practice we keep STAGES==2 so a single comparator suffices.
                is_next_stage_one = arith.cmpi(
                    arith.CmpIPredicate.eq, next_stage,
                    arith.constant(1, index=True),
                )
                from flydsl._mlir.dialects import scf as _scf
                ifop_a = _scf.IfOp(is_next_stage_one, results_=[], has_else=True)
                with ir.InsertionPoint(ifop_a.then_block):
                    _ldg_sts_a_async(k_byte_off_next, 1)
                    _ldg_sts_w_async(k_byte_off_next, 1)
                    _scf.YieldOp([])
                with ir.InsertionPoint(ifop_a.else_block):
                    _ldg_sts_a_async(k_byte_off_next, 0)
                    _ldg_sts_w_async(k_byte_off_next, 0)
                    _scf.YieldOp([])

                # Read cur_stage from LDS (also under scf.if for stage 0/1).
                ifop_b = _scf.IfOp(
                    arith.cmpi(arith.CmpIPredicate.eq, cur_stage,
                               arith.constant(1, index=True)),
                    results_=([T.vec(4, T.i32)] * (2 * (M_SUB + N_SUB))),
                    has_else=True,
                )
                def _read_stage(s):
                    out_vals = []
                    a_pairs = [_lds_load_a_pair(s, sub) for sub in range_constexpr(M_SUB)]
                    w_pairs = [_lds_load_w_pair(s, n) for n in range_constexpr(N_SUB)]
                    for lo, hi in a_pairs:
                        out_vals += [lo, hi]
                    for lo, hi in w_pairs:
                        out_vals += [lo, hi]
                    return out_vals
                with ir.InsertionPoint(ifop_b.then_block):
                    _scf.YieldOp(_read_stage(1))
                with ir.InsertionPoint(ifop_b.else_block):
                    _scf.YieldOp(_read_stage(0))

                # Reconstruct a_pairs / w_pairs from the scf.if results.
                rvals = list(ifop_b.results)
                a_cur = [(rvals[2 * s], rvals[2 * s + 1])
                         for s in range_constexpr(M_SUB)]
                base = 2 * M_SUB
                w_cur = [(rvals[base + 2 * n], rvals[base + 2 * n + 1])
                         for n in range_constexpr(N_SUB)]

                accs_new = _emit_mma(a_cur, w_cur, k_tile_idx, accs_cur)
                # LLVM's default scheduler already finds the right
                # MFMA/VMEM/DSRD interleaving for our wo_a per-iter MFMA
                # count (M_SUB*N_SUB ≤ 8). Adding rocdl.sched_* hints
                # — even a single sched_barrier(0) — was measured (MI355X)
                # to regress 1–5% by over-constraining the schedule.
                _wait_vmem_lds()
                gpu.barrier()
                results = yield _pack_state_lds(k_next, next_stage, accs_new)

            # Epilogue: read the final stage and emit the last MFMA.
            k_last, cur_stage_last, accs_last_in = _unpack_state_lds(results)
            from flydsl._mlir.dialects import scf as _scf
            ifop_c = _scf.IfOp(
                arith.cmpi(arith.CmpIPredicate.eq, cur_stage_last,
                           arith.constant(1, index=True)),
                results_=([T.vec(4, T.i32)] * (2 * (M_SUB + N_SUB))),
                has_else=True,
            )
            with ir.InsertionPoint(ifop_c.then_block):
                _scf.YieldOp(_read_stage(1))
            with ir.InsertionPoint(ifop_c.else_block):
                _scf.YieldOp(_read_stage(0))
            rvals = list(ifop_c.results)
            a_last = [(rvals[2 * s], rvals[2 * s + 1])
                      for s in range_constexpr(M_SUB)]
            base = 2 * M_SUB
            w_last = [(rvals[base + 2 * n], rvals[base + 2 * n + 1])
                      for n in range_constexpr(N_SUB)]
            accs_final = _emit_mma(a_last, w_last, k_last, accs_last_in)
            for sub in range_constexpr(M_SUB):
                for n_sub in range_constexpr(N_SUB):
                    accs[sub][n_sub] = accs_final[sub][n_sub]

        else:
            # Prefill K-loop: scf.for with loop-carried A+W prefetch. The
            # prologue pre-loads k=0 tiles; the body issues k+1's HBM loads
            # before computing k's MFMAs so load latency hides behind compute.
            # State layout (deterministic; packers/unpackers must match):
            #   [0]                          k_tile_idx (index)
            #   [1 .. 1+2*M_SUB)             M_SUB (a_lo, a_hi) pairs
            #   [1+2*M_SUB .. +2*N_SUB)      N_SUB (w_lo, w_hi) pairs
            #   [..end)                      M_SUB*N_SUB v4f32 accumulators
            k_zero_off = fx.Index(0)
            a_pref = [_load_a_pair(k_zero_off, s)
                      for s in range_constexpr(M_SUB)]
            w_pref = [_load_w_pair(k_zero_off, n)
                      for n in range_constexpr(N_SUB)]

            def _pack_state(k_idx, a_pairs, w_pairs, acc_grid):
                out = [k_idx]
                for lo, hi in a_pairs:
                    out += [lo, hi]
                for lo, hi in w_pairs:
                    out += [lo, hi]
                for s in range_constexpr(M_SUB):
                    for n in range_constexpr(N_SUB):
                        out.append(acc_grid[s][n])
                return [_unwrap(v) for v in out]

            def _unpack_state(state):
                k_idx = state[0]
                idx = 1
                a_pairs = [(state[idx + 2 * s], state[idx + 2 * s + 1])
                           for s in range_constexpr(M_SUB)]
                idx += 2 * M_SUB
                w_pairs = [(state[idx + 2 * n], state[idx + 2 * n + 1])
                           for n in range_constexpr(N_SUB)]
                idx += 2 * N_SUB
                acc_grid = [[None] * N_SUB for _ in range_constexpr(M_SUB)]
                for s in range_constexpr(M_SUB):
                    for n in range_constexpr(N_SUB):
                        acc_grid[s][n] = state[idx]
                        idx += 1
                return k_idx, a_pairs, w_pairs, acc_grid

            init_state = _pack_state(
                arith.constant(0, index=True), a_pref, w_pref, accs
            )

            # NB: bare ``range(...)`` with ``init=`` is intercepted by FlyDSL's
            # AST rewriter and lowered to scf.for (see prefetch-data-load skill).
            for _bki, state in range(0, K_g - 1, 1, init=init_state):
                k_tile_idx, a_cur, w_cur, accs_cur = _unpack_state(state)

                # Issue prefetches for k+1 (overlap with the MFMA below).
                k_next = k_tile_idx + arith.constant(1, index=True)
                k_dword_off_next = k_next * fx.Index(32)
                a_next = [_load_a_pair(k_dword_off_next, s)
                          for s in range_constexpr(M_SUB)]
                w_next = [_load_w_pair(k_dword_off_next, n)
                          for n in range_constexpr(N_SUB)]

                accs_new = _emit_mma(a_cur, w_cur, k_tile_idx, accs_cur)

                results = yield _pack_state(k_next, a_next, w_next, accs_new)

            # Epilogue: process the last K-iter using the final yielded state.
            k_last, a_last, w_last, accs_last_in = _unpack_state(results)
            accs_final = _emit_mma(a_last, w_last, k_last, accs_last_in)
            for sub in range_constexpr(M_SUB):
                for n_sub in range_constexpr(N_SUB):
                    accs[sub][n_sub] = accs_final[sub][n_sub]

        # ---- Write BLOCK_M × BLOCK_N bf16 output --------------------------
        out_col = lane % fx.Index(16)
        for sub in range_constexpr(M_SUB):
            for n_sub in range_constexpr(N_SUB):
                for i in range_constexpr(4):
                    m_idx = m_idx_per_sub[sub][i]
                    n_idx = (pid_n * fx.Index(BLOCK_N)
                             + fx.Index(n_sub * MFMA_N) + out_col)
                    elem_off = (pid_b * fx.Index(O_BATCH_STRIDE)
                                + m_idx * fx.Index(N) + n_idx)
                    bf16_val = _truncf_f32_to_bf16(accs[sub][n_sub][i])
                    buffer_ops.buffer_store(bf16_val, o_rsrc, elem_off)

    @flyc.jit
    def launcher(
        A: fx.Tensor, W: fx.Tensor,
        A_scale: fx.Tensor, W_scale: fx.Tensor,
        Out: fx.Tensor,
    ):
        # Default stream — omitted from both signature and .launch() to keep
        # the AOT CallState argument arity stable across replays.
        if USE_LDS_PIPELINE:
            _allocator.finalized = False
            ctx = CompilationContext.get_current()
            with ir.InsertionPoint(ctx.gpu_module_body):
                _allocator.finalize()
        kernel._func.__name__ = KERNEL_NAME
        kernel(A, W, A_scale, W_scale, Out).launch(
            grid=(GRID_M, GRID_N, B),
            block=(WAVE_SIZE, 1, 1),
        )

    return launcher


# ---- shape policy ----------------------------------------------------------
def _pick_block_m(M: int) -> int:
    """Heuristic for BLOCK_M (calibrated on MI355X):
      * BM=64 — clear win only at prefill (M ≥ 128). At T=64 the VGPR pressure
        from 4 accumulators / lane + tile temps regresses vs BM=32.
      * BM=32 — sweet spot at the T=64 boundary.
      * BM=16 — decode (T < 32).
    """
    if M >= 128 and M % 64 == 0:
        return 64
    if M >= 32 and M % 32 == 0:
        return 32
    return 16


def _pick_block_n(M: int, N: int) -> int:
    """Heuristic for BLOCK_N. Wider N (N_SUB=2) amortises per-K-iter overhead
    across 8 MFMAs (M_SUB=4 × N_SUB=2) vs 4 (M_SUB=4 × 1), helping prefill
    throughput. Decode (M < 128) keeps BLOCK_N=16 to maximise WG count for
    chip occupancy.
    """
    if M >= 128 and N % 32 == 0:
        return 32
    return 16


# ---- public entry ----------------------------------------------------------
def flydsl_batched_gemm_fp8_blockwise(
    A: torch.Tensor,
    W: torch.Tensor,
    A_scale: torch.Tensor,
    W_scale: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
) -> torch.Tensor:
    """Same contract as ``aiter.batched_gemm_fp8_blockwise``.

    Constraints:
      * K must be a multiple of 128.
      * M must be a multiple of 16 (and BLOCK_M, if specified). M < 16 is
        host-padded to 16 then sliced back on output.
      * N must be a multiple of 128.

    Prefill (M ≥ 128, M % 128 == 0) should route to the CK backend via
    ``aiter.batched_gemm_fp8_blockwise(backend="ck")``; this entry point is
    the decode-optimised single-wave kernel.
    """
    assert A.dtype == torch.float8_e4m3fn
    assert W.dtype == torch.float8_e4m3fn
    if A_scale.dtype not in (torch.float32, torch.uint8):
        raise TypeError(
            f"A_scale must be float32 or uint8 (UE8M0), got {A_scale.dtype}"
        )
    if W_scale.dtype not in (torch.float32, torch.uint8):
        raise TypeError(
            f"W_scale must be float32 or uint8 (UE8M0), got {W_scale.dtype}"
        )
    A_scale = _torch_scales_to_ue8m0(A_scale)
    W_scale = _torch_scales_to_ue8m0(W_scale)
    B, M_orig, K = A.shape
    Bw, N, Kw = W.shape
    assert B == Bw and K == Kw
    assert K % 128 == 0, f"K must be a multiple of 128, got K={K}"
    assert N % 128 == 0

    # M < 16 decode: pad on host to 16, compute, slice back.
    _pad_M = (M_orig % 16) != 0
    if _pad_M:
        M_padded = (M_orig + 15) // 16 * 16
        A_p = torch.zeros((B, M_padded, K), dtype=A.dtype, device=A.device)
        A_p[:, :M_orig, :] = A
        A = A_p
        K_g = K // 128
        A_s_p = torch.zeros(
            (B, M_padded, K_g), dtype=A_scale.dtype, device=A_scale.device,
        )
        A_s_p[:, :M_orig, :] = A_scale
        A_scale = A_s_p
        out_orig = out  # caller-provided out is for the unpadded shape
        out = None
        M = M_padded
    else:
        M = M_orig
        out_orig = None

    BLOCK_M = block_m if block_m is not None else _pick_block_m(M)
    BLOCK_N = block_n if block_n is not None else _pick_block_n(M, N)
    assert M % BLOCK_M == 0, f"M={M} must divide BLOCK_M={BLOCK_M}"
    assert N % BLOCK_N == 0, f"N={N} must divide BLOCK_N={BLOCK_N}"
    # Native-MFMA scaleA path is decode-friendly (BLOCK_M ≤ 32); at BLOCK_M=64
    # (prefill, M_SUB=4) the lost latency-hiding regresses, use the post-MFMA
    # fp32-mul path. The compile-time gate inside the kernel enforces the same.
    native_scale_mfma = (BLOCK_M <= NATIVE_SCALE_BLOCK_M_MAX)
    # LDS + async-copy pipeline auto-enable: POST_SCALE (BLOCK_M=64 prefill)
    # path with enough K-iters to amortise LDS staging + barriers. Measured
    # on MI355X: K_g ≥ 8 gives 1.2–1.5× speedup vs HBM-direct prefetch;
    # K_g ≤ 2 regresses ~2× because of the per-iter barrier cost.
    use_lds_pipeline = (not native_scale_mfma) and (K // 128 >= 8)

    if out is None:
        out = torch.empty((B, M, N), dtype=torch.bfloat16, device=A.device)
    else:
        assert out.shape == (B, M, N) and out.dtype == torch.bfloat16

    launcher = compile_bgfp8bw_kernel(
        B, M, N, K, BLOCK_M, BLOCK_N,
        native_scale_mfma=native_scale_mfma,
        use_lds_pipeline=use_lds_pipeline,
    )
    # AOT compile cache: first call compiles + executes, later calls reuse the
    # CompiledFunction with minimal dispatch overhead.
    cf = getattr(launcher, "_aiter_cf", None)
    if cf is None:
        cf = flyc.compile(launcher, A, W, A_scale, W_scale, out)
        launcher._aiter_cf = cf
    else:
        cf(A, W, A_scale, W_scale, out)

    if _pad_M:
        sliced = out[:, :M_orig, :].contiguous()
        if out_orig is not None:
            out_orig.copy_(sliced)
            return out_orig
        return sliced
    return out
