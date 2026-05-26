# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
FlyDSL implementation of FP8 block-wise batched GEMM (DeepSeek V4 wo_a path).

Status: MINIMUM-VIABLE.  Shape-baked kernel (B, M, N, K become compile-time
constants in the JIT).  Validated on the smallest tile shape:

    (B=1, M=16, N=128, K=128)   -- one MFMA tile per (M, N) workgroup.

Larger shapes work as long as M, N are multiples of 16 / 128 respectively
and K = 128 (single K-block).  Multi-K-block support requires an scf.ForOp
in the kernel body -- a documented next step.

Algorithm (single workgroup = 1 wave-64):
  1. Load a 16x128 fp8 A tile via 2x buffer_load(v4i32) per lane.
  2. Load a 16x128 fp8 W tile the same way (W is in [N, K] layout, so the
     "row" of the MFMA's B operand maps to the N coordinate).
  3. Run the gfx950 ``mfma_scale_f32_16x16x128_f8f6f4`` with MFMA scale
     slots fixed at E8M0 byte 0x7F (= 1.0).  Per-row / per-128×128 block
     scales from global memory are **UE8M0** (uint8 biased exponent,
     bias 127): ``scale_fp32 = exp2(float(u8) - 127)``, then applied to
     the MFMA fp32 accumulator (the V4 (1,1,128) recipe still does not map
     directly onto MFMA's per-tile scale slots).
  4. Lane ``l`` writes 4 fp32 -> bf16 to Out[b, m_off + (l/16)*4 + i,
     n_off + l%16].

Mirrors the structure of ``aiter/ops/flydsl/kernels/mega_moe/_phase1_step1_single_tile.py``
(which is the production-validated FP8 MFMA template) with three deltas:
  * batched: pid_b indexes the B dim
  * per-row A scale: load + multiply post-MFMA per row
  * per-block W scale: load + multiply post-MFMA across the tile
"""

from __future__ import annotations

import functools
from typing import Optional

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith as _arith
from flydsl._mlir.dialects import math as mlir_math
from flydsl.expr import arith, buffer_ops, range_constexpr, rocdl, vector
from flydsl.expr.typing import T, Uint8, _to_raw
from flydsl.expr.utils.arith import arith_const, unwrap as _unwrap

# Dispatch thresholds for the v2 (4-wave + LDS-A) prefill kernel.
# v2 needs M and N both multiples of 128. See OPTIMIZATION_JOURNEY.md Iter 8.
_BLOCK_M_V2 = 128
_BLOCK_N_V2 = 128


def _vec_ty(n: int, mlir_elem):
    return ir.VectorType.get([n], mlir_elem)


def _truncf_f32_to_bf16(val_f32):
    """f32 scalar -> bf16 scalar via raw MLIR arith.TruncFOp.  Unwrap the
    FlyDSL Numeric/ArithValue wrapper to a raw ir.Value first."""
    raw = _to_raw(val_f32)
    return _arith.TruncFOp(T.bf16, raw).result


def _ue8m0_byte_to_f32(byte_val):
    """Convert a ue8m0 byte (V4 scale format) to fp32 via zero-cost bit ops.

    ue8m0 byte is interpreted as ``2^(byte - 127)``.  fp32 has the same
    exponent bias 127, so placing the byte into fp32 exponent field
    (bits 23..30) with sign=0 mantissa=0 gives exactly 2^(byte-127).

    Sequence: i8 -> zero-extend i32 -> shift-left 23 -> bitcast to f32.
    All bit-level ops, no FP arithmetic.
    """
    raw = _to_raw(byte_val)
    i32_val = _arith.ExtUIOp(T.i32, raw).result
    c23 = _arith.ConstantOp(T.i32, ir.IntegerAttr.get(T.i32, 23)).result
    shifted = _arith.ShLIOp(i32_val, c23).result
    return _arith.BitcastOp(T.f32, shifted).result


def _ue8m0_byte_pack4(byte_val):
    """Pack one ue8m0 byte 4 times into an i32, suitable for MFMA scaleA/scaleB.

    The MFMA scale args are i32 holding 4 packed E8M0 bytes (one per
    32-K-element sub-tile).  V4's per-128-K scale means all 4 sub-tiles
    within a K=128 MFMA tile share the same byte -- so we broadcast.

    Cheapest path: ext the byte to i32, then ``byte * 0x01010101`` produces
    a 4-byte-replicated i32.  One mul, no shifts.
    """
    raw = _to_raw(byte_val)
    i32_val = _arith.ExtUIOp(T.i32, raw).result
    c_replicate = _arith.ConstantOp(T.i32, ir.IntegerAttr.get(T.i32, 0x01010101)).result
    return _arith.MulIOp(i32_val, c_replicate).result


def _torch_scales_to_ue8m0(scales: torch.Tensor) -> torch.Tensor:
    """Host-side: fp32 multiplicative scales → uint8 UE8M0 bytes, or pass uint8 through.

    UE8M0: ``multiplier ≈ 2 ** (int(u8) - 127)``.  Rounding matches common
    aiter tests: ``round(log2(s)) + 127`` clamped to [0, 255]; non-positive
    entries become 0 (decoded as minimum exponent on device).
    """
    if scales.dtype == torch.uint8:
        return scales.contiguous()
    if scales.dtype != torch.float32:
        raise TypeError(
            "FlyDSL fp8 blockwise expects A_scale/W_scale as torch.uint8 (UE8M0) "
            f"or torch.float32; got {scales.dtype}"
        )
    s = scales.float()
    pos = s > 0
    out = torch.zeros_like(s, dtype=torch.uint8)
    tiny = torch.tensor(torch.finfo(torch.float32).tiny, device=s.device, dtype=torch.float32)
    safe = torch.where(pos, s, tiny)
    biased = (torch.log2(safe) + 127.0).round().clamp(0, 255).to(torch.uint8)
    out[pos] = biased[pos]
    return out.contiguous()


@functools.lru_cache(maxsize=None)
def compile_bgfp8bw_kernel(B: int, M: int, N: int, K: int, BLOCK_M: int = 16,
                            BLOCK_N: int = 16,
                            scale_is_u8: bool = False,
                            native_scale_mfma: bool = True):
    """
    Build the kernel + launcher for fixed (B, M, N, K) and (BLOCK_M, BLOCK_N).

    BLOCK_M:
      * 16 -- one MFMA per K-iter per N-sub, single M tile per WG (original)
      * 32 -- two MFMAs per K-iter per N-sub, stacked in M; share W loads
      * 64 -- four MFMAs per K-iter per N-sub, stacked in M; share W loads

    BLOCK_N:
      * 16 -- one MFMA per K-iter per M-sub (original)
      * 32 -- two MFMAs per K-iter per M-sub, stacked in N; share A loads + A_scale
              + share the single W_scale byte (both 16-N sub-tiles fit inside
              the same 128-N W_scale block)

    Wider BLOCK_M × BLOCK_N:
      * shrinks the grid by BLOCK_M*BLOCK_N/256 in the M*N tile dim
      * per K-iter compute scales as M_SUB × N_SUB MFMAs sharing A (per m_sub)
        and W (per n_sub) loads
      * grows per-WG accumulator state by M_SUB*N_SUB*4 fp32 per lane
    """
    assert K % 128 == 0
    assert M % BLOCK_M == 0, f"M={M} must be a multiple of BLOCK_M={BLOCK_M}"
    assert N % 128 == 0
    assert BLOCK_M in (16, 32, 64), f"BLOCK_M must be 16/32/64, got {BLOCK_M}"
    assert BLOCK_N in (16, 32), f"BLOCK_N must be 16/32, got {BLOCK_N}"
    assert N % BLOCK_N == 0, f"N={N} must be a multiple of BLOCK_N={BLOCK_N}"

    # ----- tile-geometry constants (see OPTIMIZATION_JOURNEY.md §1) -----
    # The MFMA used here is mfma_*_16x16x128 -- ONE MFMA covers a 16-M x 16-N
    # output tile from a 16-M x 128-K A slice and 128-K x 16-N B slice.
    # M_SUB and N_SUB say how many of those 16x16 micro-tiles each WG stacks
    # to form its BLOCK_M x BLOCK_N output tile, executing all of them inside
    # ONE K-iter (so per-K-iter loop overhead is amortized across M_SUB*N_SUB
    # MFMAs, which is the main lever we have for prefill perf).
    M_SUB = BLOCK_M // 16          # 1, 2, or 4 (BLOCK_M = 16/32/64)
    N_SUB = BLOCK_N // 16          # 1 or 2     (BLOCK_N = 16/32)
    GRID_M = M // BLOCK_M          # how many WGs along M
    GRID_N = N // BLOCK_N          # how many WGs along N
    K_g = K // 128                 # K-loop trip count (1 K-iter consumes 128 K)
    N_g = N // 128                 # number of 128-N W_scale blocks (per batch)

    # Strides in the *element* dim (bytes for fp8/i8, but you'll see // 4 in
    # the actual buffer_load offsets because buffer_load takes dword offsets).
    A_BATCH_STRIDE = M * K         # one batch of A     = M*K bytes
    W_BATCH_STRIDE = N * K         # one batch of W     = N*K bytes
    O_BATCH_STRIDE = M * N         # one batch of out   = M*N bf16 elements
    AS_BATCH_STRIDE = M * K_g      # one batch of A_s   = M*K_g UE8M0 bytes
    WS_BATCH_STRIDE = N_g * K_g    # one batch of W_s   = N_g*K_g UE8M0 bytes

    @flyc.kernel
    def kernel(
        A_ptr: fx.Tensor,
        W_ptr: fx.Tensor,
        A_scale_ptr: fx.Tensor,
        W_scale_ptr: fx.Tensor,
        Out_ptr: fx.Tensor,
    ):
        # tid here = (0..63) because we launch 1 wave per WG (block=(64,1,1)).
        tid = fx.thread_idx.x
        lane = tid

        pid_m = fx.block_idx.x       # BLOCK_M-tile index in M (stride=BLOCK_M rows)
        pid_n = fx.block_idx.y       # BLOCK_N-tile index in N (stride=BLOCK_N cols)
        pid_b = fx.block_idx.z

        # ---- MFMA 16x16x128 lane->data convention (see JOURNEY.md §1) ----
        # For ONE 16x16x128 MFMA, lane l in [0, 64) holds:
        #   A operand: 32 bytes from row (l % 16), starting K-byte (l/16)*32
        #   B operand: 32 bytes from row (l % 16), starting K-byte (l/16)*32  (B = W^T conceptually)
        #   C accum  : 4 fp32 from rows (l/16)*4 + 0..3, col (l % 16)
        # `row` and `k_dword_lane` re-derive that mapping for our HBM addresses.
        row = lane % fx.Index(16)
        # Each lane needs 32 bytes of A/W along K = 8 dwords. Lanes 0..15 read
        # bytes 0..31, lanes 16..31 read 32..63, lanes 32..47 read 64..95, etc.
        # Express that *in dwords* (buffer_load takes dword offsets, not bytes).
        k_dword_lane = (lane // fx.Index(16)) * fx.Index(8)

        # Per-(m_sub, lane) row-base for A in DWORDS (// 4 converts bytes->dwords).
        # m_sub stacks 16 rows along M: m-row of this lane's A slice is
        #   pid_m * BLOCK_M + sub * 16 + (lane % 16)
        # so the byte offset of that row's start (within A_ptr) is:
        #   pid_b * (M*K)  +  m_row * K
        # which becomes dwords by // 4. Lane's own k offset (k_dword_lane) and
        # the K-iter offset (k_tile * 128 / 4 = k_tile * 32) are added inside
        # the K-loop.
        a_row_dword_bases = [
            (pid_b * fx.Index(A_BATCH_STRIDE)
             + (pid_m * fx.Index(BLOCK_M) + fx.Index(sub * 16) + row) * fx.Index(K))
             // fx.Index(4)
            for sub in range_constexpr(M_SUB)
        ]
        # Per-(n_sub, lane) row-base for W. Same shape as A but n_sub stacks
        # 16 cols along N (lane row maps to a W ROW because W is (N, K)).
        w_row_dword_bases = [
            (pid_b * fx.Index(W_BATCH_STRIDE)
             + (pid_n * fx.Index(BLOCK_N) + fx.Index(n_sub * 16) + row) * fx.Index(K))
             // fx.Index(4)
            for n_sub in range_constexpr(N_SUB)
        ]

        a_rsrc = buffer_ops.create_buffer_resource(A_ptr, max_size=True)
        w_rsrc = buffer_ops.create_buffer_resource(W_ptr, max_size=True)
        as_rsrc = buffer_ops.create_buffer_resource(A_scale_ptr, max_size=True)
        ws_rsrc = buffer_ops.create_buffer_resource(W_scale_ptr, max_size=True)
        o_rsrc = buffer_ops.create_buffer_resource(Out_ptr, max_size=True)

        # On output, lane l writes 4 consecutive M rows starting at this base
        # (matching the C-accumulator lane mapping: rows (l/16)*4 + 0..3, col l%16).
        out_row_base = (lane // fx.Index(16)) * fx.Index(4)
        # All N sub-tiles in this WG live in the same 128-N W_scale block
        # because BLOCK_N (<= 32) divides 128. So one w_scale byte covers all
        # N_SUB MFMAs in a K-iter -- saves N_SUB-1 byte loads per K-iter.
        n_block_idx = (pid_n * fx.Index(BLOCK_N)) // fx.Index(128)
        v4f32 = _vec_ty(4, ir.F32Type.get())
        v8i32 = _vec_ty(8, ir.IntegerType.get_signless(32))

        # m_idx for each sub, each i in 0..3 (lane writes 4 rows per sub-MFMA).
        # m_idx_per_sub[sub][i] = pid_m*BLOCK_M + sub*16 + (lane/16)*4 + i
        m_idx_per_sub = [
            [pid_m * fx.Index(BLOCK_M) + fx.Index(sub * 16) + out_row_base + fx.Index(i)
             for i in range_constexpr(4)]
            for sub in range_constexpr(M_SUB)
        ]

        # Per-(m_sub, n_sub) accumulator (4 fp32 per lane). Indexed accs[m_sub][n_sub].
        accs = [
            [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range_constexpr(N_SUB)]
            for _ in range_constexpr(M_SUB)
        ]

        # ====================================================================
        # K-LOOP — two structural variants depending on `native_scale_mfma`:
        #
        # PATH A  (native_scale_mfma == True, BLOCK_M <= 32, decode):
        #   range_constexpr fully-unrolled K-loop.  A_scale fed natively to
        #   MFMA scaleA per-lane (1 byte/(m_sub, k_tile)).  Decode is
        #   latency-bound; the unroll already amortizes overhead well.
        #
        # PATH B  (native_scale_mfma == False, BLOCK_M == 64, prefill):
        #   scf.for K-loop with **loop-carried prefetch of A and W tiles**.
        #   - prologue pre-loads A (M_SUB pairs) + W (N_SUB pairs) for k=0
        #   - body: issues HBM loads for k+1 BEFORE computing MFMA for k,
        #     so the load latency is hidden behind the next iter's MFMA
        #   - epilogue runs the final iter using the yielded results.
        #   Per-row a_scale + w_scale stay synchronous (small + cached).
        #   See OPTIMIZATION_JOURNEY.md Iter 7 + FlyDSL prefetch-data-load
        #   skill for the pattern + 3 critical pitfalls.
        # ====================================================================
        if native_scale_mfma:
            # ---- Path A: range_constexpr K-loop ----
            for k_tile in range_constexpr(K_g):
                k_dword_off = fx.Index(k_tile * 32)

                # Load N_SUB W tiles (shared across M-subs)
                b_tiles = []
                for n_sub in range_constexpr(N_SUB):
                    w_dword_off = w_row_dword_bases[n_sub] + k_dword_lane + k_dword_off
                    w_lo = buffer_ops.buffer_load(w_rsrc, w_dword_off, vec_width=4, dtype=T.i32)
                    w_hi = buffer_ops.buffer_load(w_rsrc, w_dword_off + fx.Index(4),
                                                  vec_width=4, dtype=T.i32)
                    b_tiles.append(vector.from_elements(
                        v8i32,
                        [w_lo[0], w_lo[1], w_lo[2], w_lo[3],
                         w_hi[0], w_hi[1], w_hi[2], w_hi[3]],
                    ))

                # W_scale (one byte, shared across N_SUB)
                w_scale_off = (pid_b * fx.Index(WS_BATCH_STRIDE)
                               + n_block_idx * fx.Index(K_g)
                               + fx.Index(k_tile))
                w_scale_u8 = buffer_ops.buffer_load(ws_rsrc, w_scale_off,
                                                    vec_width=1, dtype=Uint8)
                w_scale_packed = _ue8m0_byte_pack4(w_scale_u8)

                # Per-(m_sub): load A + A_scale; emit N_SUB MFMAs sharing both
                for sub in range_constexpr(M_SUB):
                    a_dword_off = a_row_dword_bases[sub] + k_dword_lane + k_dword_off
                    a_lo = buffer_ops.buffer_load(a_rsrc, a_dword_off, vec_width=4, dtype=T.i32)
                    a_hi = buffer_ops.buffer_load(a_rsrc, a_dword_off + fx.Index(4),
                                                  vec_width=4, dtype=T.i32)
                    a128 = vector.from_elements(
                        v8i32,
                        [a_lo[0], a_lo[1], a_lo[2], a_lo[3],
                         a_hi[0], a_hi[1], a_hi[2], a_hi[3]],
                    )

                    a_scale_row_idx = (pid_m * fx.Index(BLOCK_M)
                                       + fx.Index(sub * 16) + row)
                    a_scale_off = (pid_b * fx.Index(AS_BATCH_STRIDE)
                                   + a_scale_row_idx * fx.Index(K_g)
                                   + fx.Index(k_tile))
                    a_scale_u8 = buffer_ops.buffer_load(as_rsrc, a_scale_off,
                                                        vec_width=1, dtype=Uint8)
                    a_scale_packed = _ue8m0_byte_pack4(a_scale_u8)
                    for n_sub in range_constexpr(N_SUB):
                        tile_acc = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                            T.f32x4,
                            [a128, b_tiles[n_sub], fx.Vector.filled(4, 0.0, fx.Float32),
                             0, 0, 0, a_scale_packed, 0, w_scale_packed],
                        )
                        new_vals = []
                        for i in range_constexpr(4):
                            new_vals.append(accs[sub][n_sub][i] + tile_acc[i])
                        accs[sub][n_sub] = vector.from_elements(v4f32, new_vals)
        else:
            # ---- Path B: scf.for K-loop with A+W prefetch (prefill) ----
            #
            # State layout (deterministic order — packers must match unpackers):
            #   [0]                          k_tile_idx (index)
            #   [1 .. 1+2*M_SUB)             M_SUB (a_lo, a_hi) prefetched A pairs
            #   [1+2*M_SUB .. +2*N_SUB)      N_SUB (w_lo, w_hi) prefetched W pairs
            #   [..end)                      M_SUB*N_SUB v4f32 accumulators

            # ---------- Prologue: pre-load A + W tiles for k_tile=0 ----------
            k_zero_off = fx.Index(0)
            a_pref_pairs = []
            for sub in range_constexpr(M_SUB):
                off = a_row_dword_bases[sub] + k_dword_lane + k_zero_off
                a_lo = buffer_ops.buffer_load(a_rsrc, off, vec_width=4, dtype=T.i32)
                a_hi = buffer_ops.buffer_load(a_rsrc, off + fx.Index(4),
                                              vec_width=4, dtype=T.i32)
                a_pref_pairs.append((a_lo, a_hi))
            w_pref_pairs = []
            for n_sub in range_constexpr(N_SUB):
                off = w_row_dword_bases[n_sub] + k_dword_lane + k_zero_off
                w_lo = buffer_ops.buffer_load(w_rsrc, off, vec_width=4, dtype=T.i32)
                w_hi = buffer_ops.buffer_load(w_rsrc, off + fx.Index(4),
                                              vec_width=4, dtype=T.i32)
                w_pref_pairs.append((w_lo, w_hi))

            # ---------- Build init_state ----------
            init_state_raw = [arith.constant(0, index=True)]
            for lo, hi in a_pref_pairs:
                init_state_raw += [lo, hi]
            for lo, hi in w_pref_pairs:
                init_state_raw += [lo, hi]
            for sub in range_constexpr(M_SUB):
                for n_sub in range_constexpr(N_SUB):
                    init_state_raw.append(accs[sub][n_sub])
            init_state = [_unwrap(v) for v in init_state_raw]

            # ---------- Path B compute helper (used in body + epilogue) ----------
            def _do_compute_b(a_pairs, w_pairs, k_tile_idx_val, accs_in):
                """Emit MFMA + post-mul accumulate for one K-iter.

                a_pairs: list of (a_lo, a_hi) per m_sub
                w_pairs: list of (w_lo, w_hi) per n_sub
                k_tile_idx_val: arith index value
                accs_in: 2D list [M_SUB][N_SUB] of acc vectors
                Returns: 2D list of new acc vectors.
                """
                # W_scale (shared across N_SUB)
                w_scale_off_b = (pid_b * fx.Index(WS_BATCH_STRIDE)
                                 + n_block_idx * fx.Index(K_g)
                                 + k_tile_idx_val)
                w_scale_u8_b = buffer_ops.buffer_load(ws_rsrc, w_scale_off_b,
                                                      vec_width=1, dtype=Uint8)
                w_scale_packed_b = _ue8m0_byte_pack4(w_scale_u8_b)

                b_tiles_b = []
                for n_sub in range_constexpr(N_SUB):
                    w_lo_b, w_hi_b = w_pairs[n_sub]
                    b_tiles_b.append(vector.from_elements(
                        v8i32,
                        [w_lo_b[0], w_lo_b[1], w_lo_b[2], w_lo_b[3],
                         w_hi_b[0], w_hi_b[1], w_hi_b[2], w_hi_b[3]],
                    ))

                accs_out = [[None] * N_SUB for _ in range_constexpr(M_SUB)]
                for sub in range_constexpr(M_SUB):
                    a_lo_b, a_hi_b = a_pairs[sub]
                    a128_b = vector.from_elements(
                        v8i32,
                        [a_lo_b[0], a_lo_b[1], a_lo_b[2], a_lo_b[3],
                         a_hi_b[0], a_hi_b[1], a_hi_b[2], a_hi_b[3]],
                    )
                    a_scale_per_row_b = []
                    for i in range_constexpr(4):
                        a_scale_off_b = (pid_b * fx.Index(AS_BATCH_STRIDE)
                                         + m_idx_per_sub[sub][i] * fx.Index(K_g)
                                         + k_tile_idx_val)
                        a_scale_u8_b = buffer_ops.buffer_load(as_rsrc, a_scale_off_b,
                                                              vec_width=1, dtype=Uint8)
                        a_scale_per_row_b.append(_ue8m0_byte_to_f32(a_scale_u8_b))
                    for n_sub in range_constexpr(N_SUB):
                        tile_acc_b = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                            T.f32x4,
                            [a128_b, b_tiles_b[n_sub], fx.Vector.filled(4, 0.0, fx.Float32),
                             0, 0, 0, 0x7F7F7F7F, 0, w_scale_packed_b],
                        )
                        new_vals_b = []
                        for i in range_constexpr(4):
                            new_vals_b.append(accs_in[sub][n_sub][i]
                                              + tile_acc_b[i] * a_scale_per_row_b[i])
                        accs_out[sub][n_sub] = vector.from_elements(v4f32, new_vals_b)
                return accs_out

            # ---------- scf.for body: K_g - 1 iters ----------
            # NB: we use the bare `range(...)` builtin name; FlyDSL's AST
            # rewriter intercepts it when `init=` is present and lowers
            # to scf.for (see FlyDSL prefetch-data-load skill).
            for _bki, state in range(0, K_g - 1, 1, init=init_state):
                # Unpack state
                k_tile_idx = state[0]
                _idx = 1
                a_cur = [(state[_idx + 2 * s], state[_idx + 2 * s + 1])
                         for s in range_constexpr(M_SUB)]
                _idx += 2 * M_SUB
                w_cur = [(state[_idx + 2 * n], state[_idx + 2 * n + 1])
                         for n in range_constexpr(N_SUB)]
                _idx += 2 * N_SUB
                accs_cur = [[None] * N_SUB for _ in range_constexpr(M_SUB)]
                for sub in range_constexpr(M_SUB):
                    for n_sub in range_constexpr(N_SUB):
                        accs_cur[sub][n_sub] = state[_idx]
                        _idx += 1

                # Compute next-iter index + dword offset
                k_tile_idx_next = k_tile_idx + arith.constant(1, index=True)
                k_dword_off_next = k_tile_idx_next * fx.Index(32)

                # === Issue prefetches for k+1 (overlap with the MFMA below) ===
                a_next_pairs = []
                for sub in range_constexpr(M_SUB):
                    off = a_row_dword_bases[sub] + k_dword_lane + k_dword_off_next
                    a_lo = buffer_ops.buffer_load(a_rsrc, off, vec_width=4, dtype=T.i32)
                    a_hi = buffer_ops.buffer_load(a_rsrc, off + fx.Index(4),
                                                  vec_width=4, dtype=T.i32)
                    a_next_pairs.append((a_lo, a_hi))
                w_next_pairs = []
                for n_sub in range_constexpr(N_SUB):
                    off = w_row_dword_bases[n_sub] + k_dword_lane + k_dword_off_next
                    w_lo = buffer_ops.buffer_load(w_rsrc, off, vec_width=4, dtype=T.i32)
                    w_hi = buffer_ops.buffer_load(w_rsrc, off + fx.Index(4),
                                                  vec_width=4, dtype=T.i32)
                    w_next_pairs.append((w_lo, w_hi))

                # === Compute MFMA for current iter ===
                accs_new = _do_compute_b(a_cur, w_cur, k_tile_idx, accs_cur)

                # Yield next state
                next_state_raw = [k_tile_idx_next]
                for lo, hi in a_next_pairs:
                    next_state_raw += [lo, hi]
                for lo, hi in w_next_pairs:
                    next_state_raw += [lo, hi]
                for sub in range_constexpr(M_SUB):
                    for n_sub in range_constexpr(N_SUB):
                        next_state_raw.append(accs_new[sub][n_sub])
                results = yield [_unwrap(v) for v in next_state_raw]

            # ---------- Epilogue: process the last K-iter ----------
            k_tile_idx_last = results[0]
            _idx = 1
            a_last = [(results[_idx + 2 * s], results[_idx + 2 * s + 1])
                      for s in range_constexpr(M_SUB)]
            _idx += 2 * M_SUB
            w_last = [(results[_idx + 2 * n], results[_idx + 2 * n + 1])
                      for n in range_constexpr(N_SUB)]
            _idx += 2 * N_SUB
            accs_last_in = [[None] * N_SUB for _ in range_constexpr(M_SUB)]
            for sub in range_constexpr(M_SUB):
                for n_sub in range_constexpr(N_SUB):
                    accs_last_in[sub][n_sub] = results[_idx]
                    _idx += 1

            accs_final = _do_compute_b(a_last, w_last, k_tile_idx_last, accs_last_in)
            # Write back into `accs` so the output store loop below sees them.
            for sub in range_constexpr(M_SUB):
                for n_sub in range_constexpr(N_SUB):
                    accs[sub][n_sub] = accs_final[sub][n_sub]

        # ---- Write BLOCK_M x BLOCK_N bf16 output (M_SUB x N_SUB sub-tiles) ----
        out_col = lane % fx.Index(16)
        for sub in range_constexpr(M_SUB):
            for n_sub in range_constexpr(N_SUB):
                for i in range_constexpr(4):
                    m_idx = m_idx_per_sub[sub][i]
                    n_idx = pid_n * fx.Index(BLOCK_N) + fx.Index(n_sub * 16) + out_col
                    elem_off = (
                        pid_b * fx.Index(O_BATCH_STRIDE)
                        + m_idx * fx.Index(N) + n_idx
                    )
                    bf16_val = _truncf_f32_to_bf16(accs[sub][n_sub][i])
                    buffer_ops.buffer_store(bf16_val, o_rsrc, elem_off)

    @flyc.jit
    def launcher(
        A: fx.Tensor, W: fx.Tensor,
        A_scale: fx.Tensor, W_scale: fx.Tensor,
        Out: fx.Tensor,
    ):
        # Use the default (current) stream -- omit the stream= kwarg from
        # both the signature and .launch() to keep the AOT-compiled
        # CallState's argument arity stable across replays.
        kernel(A, W, A_scale, W_scale, Out).launch(
            grid=(GRID_M, GRID_N, B),
            block=(64, 1, 1),
        )

    return launcher


def _pick_block_m(M: int) -> int:
    """
    Heuristic for BLOCK_M, calibrated against MI355X measured perf:
      * BM=64 helps clearly only for prefill-sized M (>= 128).  At T=64 the
        VGPR pressure (4 accumulators / lane + tile temporaries) regresses
        vs BM=32.
      * BM=32 is the sweet spot for the boundary T=64 region.
      * BM=16 for tiny decode (T < 32).
    """
    if M >= 128 and M % 64 == 0:
        return 64
    if M >= 32 and M % 32 == 0:
        return 32
    return 16


def _pick_block_n(M: int, N: int) -> int:
    """
    Heuristic for BLOCK_N. Wider N tile (N_SUB=2) amortizes per-K-iter loop
    overhead across 8 MFMAs (M_SUB=4 * N_SUB=2) instead of 4 (M_SUB=4 * 1),
    helping prefill throughput. Decode shapes (M < 128) keep BLOCK_N=16
    because they want maximum WG count for chip occupancy.
    """
    if M >= 128 and N % 32 == 0:
        return 32
    return 16


def flydsl_batched_gemm_fp8_blockwise(
    A: torch.Tensor,
    W: torch.Tensor,
    A_scale: torch.Tensor,
    W_scale: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
) -> torch.Tensor:
    """
    Same contract as ``aiter.batched_gemm_fp8_blockwise``.

    Constraints:
      * K must be a multiple of 128 (single 128-block per K-iter; arbitrary K).
      * M must be a multiple of 16 (and BLOCK_M, if specified).
      * N must be a multiple of 128.

    block_m: optional override (16 / 32 / 64).  If None, picks based on M.
    """
    assert A.dtype == torch.float8_e4m3fn
    assert W.dtype == torch.float8_e4m3fn
    if A_scale.dtype not in (torch.float32, torch.uint8):
        raise TypeError(f"A_scale must be float32 or uint8 (UE8M0), got {A_scale.dtype}")
    if W_scale.dtype not in (torch.float32, torch.uint8):
        raise TypeError(f"W_scale must be float32 or uint8 (UE8M0), got {W_scale.dtype}")
    A_scale = _torch_scales_to_ue8m0(A_scale)
    W_scale = _torch_scales_to_ue8m0(W_scale)
    B, M, K = A.shape
    Bw, N, Kw = W.shape
    assert B == Bw and K == Kw
    assert K % 128 == 0, f"K must be a multiple of 128, got K={K}"
    assert M % 16 == 0
    assert N % 128 == 0

    # Dispatch to v2 (4 waves/WG + LDS-A + 128x128 tile) for prefill shapes.
    # v2 needs M % 128 and N % 128. Decode (M < 128 or M not div 128) stays
    # on the single-wave kernel below. See OPTIMIZATION_JOURNEY.md Iter 8.
    if (block_m is None and block_n is None
            and M >= 128 and M % _BLOCK_M_V2 == 0 and N % _BLOCK_N_V2 == 0):
        from .batched_gemm_fp8_blockwise_flydsl_v2 import (
            flydsl_batched_gemm_fp8_blockwise_v2 as _v2,
        )
        # v2's wrapper expects already-converted scales (uint8) -- but our
        # _torch_scales_to_ue8m0 above is idempotent if the scale is already
        # uint8, so pass through is safe.
        return _v2(A, W, A_scale, W_scale, out=out)
    BLOCK_M = block_m if block_m is not None else _pick_block_m(M)
    BLOCK_N = block_n if block_n is not None else _pick_block_n(M, N)
    assert M % BLOCK_M == 0, f"M={M} must divide BLOCK_M={BLOCK_M}"
    assert N % BLOCK_N == 0, f"N={N} must divide BLOCK_N={BLOCK_N}"
    # Native-MFMA scaleA path is decode-friendly (BLOCK_M <= 32) -- it removes
    # post-MFMA scale ops at the cost of putting per-lane byte loads on the
    # MFMA critical path.  At BLOCK_M=64 (prefill, compute-bound, M_SUB=4)
    # the lost latency-hiding regresses; use the post-MFMA fp32-mul path.
    native_scale_mfma = (BLOCK_M <= 32)

    if out is None:
        out = torch.empty((B, M, N), dtype=torch.bfloat16, device=A.device)
    else:
        assert out.shape == (B, M, N) and out.dtype == torch.bfloat16

    launcher = compile_bgfp8bw_kernel(B, M, N, K, BLOCK_M, BLOCK_N,
                                       native_scale_mfma=native_scale_mfma)
    # AOT-compile path: first call returns a CompiledFunction with minimal
    # dispatch overhead; subsequent calls reuse it.  Pass raw torch tensors
    # directly -- FlyDSL handles dlpack wrapping internally and the
    # CompiledFunction can be re-invoked with new tensors of the same shape.
    cf = getattr(launcher, "_aiter_cf", None)
    if cf is None:
        cf = flyc.compile(launcher, A, W, A_scale, W_scale, out)
        launcher._aiter_cf = cf
    else:
        cf(A, W, A_scale, W_scale, out)
    return out
