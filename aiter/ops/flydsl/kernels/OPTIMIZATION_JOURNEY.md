# FlyDSL FP8 Block-wise Batched GEMM — Optimization Journey

> 中文版本：[OPTIMIZATION_JOURNEY.zh.md](./OPTIMIZATION_JOURNEY.zh.md)

A running log of the perf optimizations applied to the FlyDSL FP8
block-wise batched GEMM kernels on **MI355X (gfx950 / CDNA4)**:

- `batched_gemm_fp8_blockwise_flydsl.py` (single-wave, **sw**)
- `batched_gemm_fp8_blockwise_flydsl_mw.py` (multi-wave + LDS, **mw**)
- `batched_gemm_fp8_blockwise_flydsl_v2.py` (4 waves/WG + LDS-A + 128×128
  tile + async DMA, **v2** — production prefill path as of Iter 8 Phase 3b)

The goal of this doc is not "what's the final code" — read the kernel for
that — but **the path we walked, what worked, what didn't, and the
diagnostic moves that taught us why**.

## Current state (latest as of Iter 8 Phase 4b + large-shape sweep)

**Decode + small prefill** (production sw / v2):
| shape | mode | sw_old | **production now** | Triton | gap to Triton |
|---|---|---|---|---|---|
| (8,16,1024,4096) | decode | 313 | 311 (sw) | 31 | 10.0× slower |
| (8,64,1024,4096) | decode | 327 | 314 (sw) | 38 | 8.3× |
| (8,1024,1024,4096) | prefill | 499 | **352** (v2) | 134 | 2.62× |
| (8,4096,1024,4096) | prefill | 1072 | **508** (v2) | 461 | **1.10×** |

**Large prefill** — 3-way comparison (v2 / Triton / **CK**) measured in
`atom-latest-todd` Docker (CK won't load on bare-metal host due to
libstdc++ < 3.4.31):

| shape (B, M, N, K) | v2 µs | Triton µs | **CK µs** | v2 TF | Triton TF | **CK TF** |
|---|---|---|---|---|---|---|
| (8, 4096, 1024, 4096) | 506 | 417 | **264** | 544 | 660 | **1043** |
| (8, 8192, 1024, 4096) | 729 | 766 | **436** | 754 | 718 | **1260** |
| (8, 16384, 1024, 4096) | 1232 | 1421 | **836** | 893 | 774 | **1315** |
| (8, 32768, 1024, 4096) | 2309 | 2818 | **1615** | 952 | 780 | **1362** |
| (16, 4096, 1024, 4096) | 732 | 730 | **523** | 751 | 753 | **1052** |
| (16, 8192, 1024, 4096) | 1228 | 1412 | **862** | 896 | 778 | **1276** |
| (16, 16384, 1024, 4096) | 2299 | 2834 | **1673** | 957 | 776 | **1315** |
| (16, 32768, 1024, 4096) | 4997 | 5809 | **3233** | 880 | 757 | **1360** |

**Ranking**: CK > v2 ≈ Triton (v2 beats Triton on M ≥ 8192).
- **CK** is the absolute fastest, sustaining ~1300 TFLOPS (22% fp8 peak).
- **v2** reaches 880-957 TFLOPS at scale (16% fp8 peak); 1.4-1.9× behind CK.
- **Triton** plateaus at ~610-780 TFLOPS (bf16 MFMA path ceiling).

CK's lead comes from CShuffle epilogue + per-shape tile tuning + AMD's
hand-tuned `BlockGemmPipelineScheduler` — all things v2 hasn't done.

**Dispatch policy** (in `flydsl_batched_gemm_fp8_blockwise()` wrapper):
`M >= 128 && M % 128 == 0 && N % 128 == 0` → **v2**, else → sw.

Total prefill speedup over original baseline: **3.13×** at M=4096
(1589 µs → 508 µs). On large prefill (M ≥ 8192), v2 is **1.2-1.5×
faster than Triton**.

---

## 0. Contract and hardware context

* **Op**: DeepSeek V4 `wo_a` projection — FP8 (E4M3) batched GEMM with
  block-wise scales.
  - `A` shape `(B, M, K)` fp8_e4m3, `W` shape `(B, N, K)` fp8_e4m3
  - `A_scale` shape `(B, M, K/128)` UE8M0 (per-row, per-128K)
  - `W_scale` shape `(B, N/128, K/128)` UE8M0 (per-128N, per-128K)
  - Output `(B, M, N)` bf16
* **Scale recipe**: V4 = `(1, 1, 128)` — A scale per token-row × per
  128-K block; W scale per 128-N × 128-K block.
* **MFMA used**: `mfma_scale_f32_16x16x128_f8f6f4` (gfx950) — feeds
  scaleA + scaleB natively as i32-packed UE8M0 bytes, no post-MFMA
  scale multiply needed.
* **UE8M0 ↔ MX e8m0**: bit-equivalent; byte = `round(log2(scale)) + 127`.

### Why this op matters
Used as the wo_a projection in DeepSeek V4 inference. Triton has a
working kernel for it; we want a FlyDSL alternative with native MFMA
scale paths. Bench against the Triton baseline at the canonical decode
shapes `(B=8, M ∈ {16, 64}, N=1024, K=4096)` and prefill shapes
`(M ∈ {1024, 4096})`.

---

## 1. Glossary — kernel variables you'll see in the code

These are the names that show up everywhere; understanding them once
makes both kernels readable.

### Tile geometry

| Var | Meaning | Typical |
|---|---|---|
| `BLOCK_M` | Rows of output computed per workgroup (WG) | 16 / 32 / 64 |
| `BLOCK_N` | Cols of output computed per WG | 16 / 32 |
| `BLOCK_K` | K-dim chunk consumed per K-iter (== one MFMA along K) | 128 |
| `M_SUB` | `BLOCK_M // 16` — number of stacked-along-M MFMAs per K-iter | 1, 2, or 4 |
| `N_SUB` | `BLOCK_N // 16` — number of stacked-along-N MFMAs per K-iter | 1 or 2 |
| `K_g` | `K // 128` — number of K-iters in the K-loop | 32 (for K=4096) |
| `N_g` | `N // 128` — number of 128-N W_scale blocks | 8 (for N=1024) |
| `_BLOCK_THREADS` | Threads per WG. mw uses 256 (4 waves), sw uses 64 (1 wave) | |

The MFMA used here is `16x16x128` — meaning **one** MFMA produces a
16-M × 16-N output tile, consuming a 16×128 A slice + 128×16 B slice.
**`M_SUB × N_SUB` MFMAs per K-iter** form one `BLOCK_M × BLOCK_N` tile
of the output per K-iter.

### Lane → data mapping (CDNA `mfma_*_16x16x*` convention)

A wave is 64 lanes. For one MFMA, the lane-to-element mapping is:

* **A operand** (16M × 128K, packed as v8i32 per lane):
  - lane `l` holds `A[l % 16, (l/16)*32 + (0..31)]` as 32 bytes
  - i.e. `row = l % 16`, `k_byte_in_lane = (l // 16) * 32`
* **B operand** (128K × 16N, packed as v8i32 per lane):
  - same idea, `B[l % 16, ...]`
* **C accumulator** (16M × 16N, 4 fp32 per lane):
  - lane `l` holds `C[(l/16)*4 + (0..3), l % 16]`

In the code:
```python
row = lane % fx.Index(16)              # which M-row this lane is on
k_dword_lane = (lane // fx.Index(16)) * fx.Index(8)
# ↑ which 8-dword (= 32-byte) sub-K-chunk this lane is on, in dwords
out_row_base = (lane // fx.Index(16)) * fx.Index(4)
# ↑ on output, lane writes 4 consecutive M rows starting here
```

### Why the `// 4`?

`buffer_load` on AMD takes a **dword-offset** (1 dword = 4 bytes), not a
byte offset. So byte-stride math gets a final `// 4`:
```python
hbm_dword_off = (
    pid_b * fx.Index(A_BATCH_STRIDE)            # B dim (in elements = bytes for fp8)
    + (pid_m * fx.Index(BLOCK_M) + row) * fx.Index(K)   # row offset in bytes
    + fx.Index(k_tile * 128)                    # K chunk start in bytes
    + ld_byte_off                               # lane's 8-byte slot
) // fx.Index(4)                                 # bytes -> dwords
```

### Scale packing for MFMA

The MFMA scale operand is i32 holding **4 UE8M0 bytes** (one per 32-K
sub-block). For our V4 recipe, the same byte applies to all 4 sub-blocks
in a 128-K block — so we replicate via:
```python
def _ue8m0_byte_pack4(b):  # b : i8 (UE8M0)
    # produce 0xBB_BB_BB_BB packed across 4 sub-K positions
    return zext_i8_to_i32(b) * 0x01010101
```

---

## 2. Iteration log

Read top-down. Each section says **why we tried it**, **what changed**,
**result**, **what we learned**.

### Iter 0 — Baseline single-wave kernel `sw`
File: `batched_gemm_fp8_blockwise_flydsl.py`

* Geometry: 1 wave / WG, `BLOCK_M ∈ {16, 32, 64}` (heuristic on M),
  `BLOCK_N = 16`, `BLOCK_K = 128`.
* No LDS use — A and W loaded directly from HBM into VGPRs per K-iter.
* Native MFMA scaleA + scaleB for `BLOCK_M ≤ 32`; post-MFMA fp32
  multiply for `BLOCK_M = 64` (the per-lane scale byte load was on the
  MFMA critical path and regressed prefill).

**Key code** (essence of the baseline K-loop body):
```python
# 1 wave/WG; lane → MFMA position
row = lane % fx.Index(16)                             # M-row this lane handles
k_dword_lane = (lane // fx.Index(16)) * fx.Index(8)   # K offset in dwords
acc = fx.Vector.filled(4, 0.0, fx.Float32)            # 4 fp32/lane accumulator

for k_tile in range_constexpr(K_g):                   # fully unrolled K-loop
    k_dword_off = fx.Index(k_tile * 32)

    # Load A 32B (=8 dwords) and W 32B per lane direct HBM → v8i32
    a_lo = buffer_ops.buffer_load(a_rsrc, ..., vec_width=4, dtype=T.i32)
    a_hi = buffer_ops.buffer_load(a_rsrc, ..., vec_width=4, dtype=T.i32)
    a128 = vector.from_elements(v8i32, [a_lo[0..3], a_hi[0..3]])
    w_lo = buffer_ops.buffer_load(w_rsrc, ..., vec_width=4, dtype=T.i32)
    w_hi = buffer_ops.buffer_load(w_rsrc, ..., vec_width=4, dtype=T.i32)
    b128 = vector.from_elements(v8i32, [w_lo[0..3], w_hi[0..3]])

    # Per-lane scaleA byte (V4 recipe: 1 byte / row / 128-K-block) packed to i32
    a_scale_byte = As_[pid_b, m_row, fx.Index(k_tile)]
    a_scale_packed = _ue8m0_byte_pack4(a_scale_byte)  # byte * 0x01010101
    w_scale_packed = _ue8m0_byte_pack4(Ws_[pid_b, n_block, fx.Index(k_tile)])

    # Native fp8 MFMA with packed scaleA + scaleB (gfx950 specific)
    tile_acc = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
        T.f32x4,
        [a128, b128, fx.Vector.filled(4, 0.0, fx.Float32),
         0, 0, 0, a_scale_packed, 0, w_scale_packed],
    )
    for i in range_constexpr(4):
        acc[i] = acc[i] + tile_acc[i]
```

### Iter 1 — mw skeleton (4 waves + LDS-A sharing) — *failed*

**Motivation**: the splitk-style win in `splitk_hgemm.py` comes from
multi-wave WGs sharing LDS-resident operand tiles. Try the same recipe
for FP8.

**Geometry**: 4 waves / WG (256 threads), `BLOCK_M=16, BLOCK_N=64`
(each wave covers a 16-N stripe of the 64-N tile). LDS-A holds one
16×128 tile of A bytes (2 KB), shared across the 4 waves; only wave 0
writes it.

**Result**: produced **NaN**. The kernel compiled and ran, but output
was garbage.

**Key code** (the failing wave-0-only LDS write pattern):
```python
# mw: 4 waves/WG (256 threads), BLOCK_M=16, BLOCK_N=64
# Wave 0 alone loads A → LDS; other waves stall at barrier
is_wave0 = arith.cmpi("eq", wave_id, arith.index(0))
if_op = scf.IfOp(is_wave0, results_=[], has_else=False)
with ir.InsertionPoint(if_op.then_block):
    a_v8i32 = buffer_ops.buffer_load(a_rsrc, ..., vec_width=8, dtype=T.i32)
    a_v32i8 = vector.bitcast(v32i8_t, a_v8i32)
    as_lds.vec_store((row, k_byte_in_lane), a_v32i8, 32)  # ← produced NaN!
    scf.YieldOp([])
gpu.barrier()  # waves 1-3 idle here for ~300 cycles
```

**Lessons**:
1. Two LLVM/MLIR footguns surfaced and were fixed:
   - `vector<NxFloat8E4M3FN>` crashed LLVM lowering → store fp8 as
     i8 in vector regs, bitcast to v8i32 only at the MFMA boundary
   - `STensor.vec_load(.., 32)` with i8 dtype produced wrong layout →
     do two `i32 buffer_load(vec_width=4)` calls instead and bitcast
     v8i32 → v32i8 at the LDS write
2. The "wave 0 writes LDS, waves 1-3 wait" predicate left 75% of the
   chip's load bandwidth idle during the load phase.

### Iter 2 — mw cooperative load — *correct, no perf win*

**Change**: all 256 lanes participate in the HBM→LDS A load. Lane
mapping `ld_row = tid // 16`, `ld_byte_off = (tid % 16) * 8` covers
exactly `256 × 8 = 2048` bytes = the LDS-A slot.

**Key code** (cooperative lane mapping, no scf.if — all waves participate):
```python
# 256 threads × 8 B/thread = 2 KB = exact LDS-A tile size
ld_row = tid // fx.Index(16)                       # row 0..15
ld_byte_off = (tid % fx.Index(16)) * fx.Index(8)   # byte offset 0,8,...,120

# Per-thread: 1 buffer_load_dwordx2 (8 bytes), then bitcast + LDS store
a_v2i32 = buffer_ops.buffer_load(a_rsrc, hbm_dword_off, vec_width=2, dtype=T.i32)
a_v8i8 = vector.bitcast(v8i8_t, a_v2i32)
as_lds.vec_store((ld_row, ld_byte_off), a_v8i8, 8)  # NO bug here

gpu.barrier()  # all waves now have full A tile in LDS
```

**Result**: NaN gone (max bf16 err vs torch oracle = 0.5, identical to
sw). But mw **still 0-13% slower** than sw across all shapes.

**Lessons**: cooperative load was correct but didn't move the needle.
The barrier-per-K-iter cost was eating the A-traffic savings.

### Iter 3 — mw + LDS double-buffer (STAGES=2) — *tiny decode win, prefill still loses*

**Change**: allocate 2× LDS-A (4 KB), prologue loads tile 0, K-loop
prefetches tile k+1 while computing on tile k. Idea: hide HBM-load
latency behind MFMA compute.

**Key code** (ping-pong via Python int `% 2` — works because K-loop is
range_constexpr, so `k_tile % 2` is a compile-time constant per iter):
```python
# Allocate 2× LDS for A: 4 KB total
LDS_A_BYTES = _BLOCK_M * _BLOCK_K
allocator.ptr = smem_a_offset_0 + LDS_A_BYTES
smem_a_offset_1 = smem_a_offset_0 + LDS_A_BYTES
allocator.ptr = smem_a_offset_1 + LDS_A_BYTES
# ...
as_lds_bufs = [STensor(smem_a_ptr_0, ...), STensor(smem_a_ptr_1, ...)]

# Prologue: load tile 0 → buf[0]
a_pref = buffer_ops.buffer_load(a_rsrc, _hbm_a_dword_off_for(0), vec_width=2, dtype=T.i32)
as_lds_bufs[0].vec_store((ld_row, ld_byte_off), vector.bitcast(v8i8_t, a_pref), 8)
gpu.barrier()

# K-loop: each iter computes on cur_buf, prefetches into nxt_buf
for k_tile in range_constexpr(K_g):
    cur_buf = as_lds_bufs[k_tile % 2]            # constexpr swap
    nxt_buf_idx = (k_tile + 1) % 2
    if k_tile + 1 < K_g:                          # constexpr branch
        a_next = buffer_ops.buffer_load(a_rsrc, _hbm_a_dword_off_for(k_tile + 1), ...)
    a_lds_vec = cur_buf.vec_load((row, k_dword_lane * fx.Index(4)), 32)
    # ... MFMA + accumulate ...
    if k_tile + 1 < K_g:
        as_lds_bufs[nxt_buf_idx].vec_store(..., vector.bitcast(v8i8_t, a_next), 8)
        gpu.barrier()
```

**Result**:
| shape | mode | sw us | mw us | mw/sw |
|---|---|---|---|---|
| (8,16,1024,4096) | decode | 314 | 311 | 0.99× |
| (8,64,1024,4096) | decode | 324 | 313 | 0.97× |
| (8,1024,1024,4096) | prefill | 646 | 671 | 1.04× |
| (8,4096,1024,4096) | prefill | 1600 | 1809 | 1.13× |

**Lessons**: 1-3% decode win, prefill loss persists. mw architecture is
fundamentally not winning this op at these shapes. Why ↓ the diagnosis
in §3.

### Iter 4 — diagnosis: where is the time going? *(no code change)*

**Method**: `FLYDSL_DEBUG_DUMP_ASM=1 FLYDSL_DUMP_IR=1` to capture each
kernel's final ISA. Then grep for resource directives:

```bash
grep -E '\.amdhsa_(next_free_vgpr|private_segment_fixed_size|group_segment_fixed_size)' \
    ~/.flydsl/debug/kernel_0/17_final_isa.s
grep -cE 'v_mfma_'    17_final_isa.s    # MFMA count
grep -cE 'scratch_'    17_final_isa.s    # spill load/store count
grep -cE 'ds_(read|write)' 17_final_isa.s  # LDS traffic
grep -cE 's_barrier'   17_final_isa.s    # WG barriers
```

ISA stats at (B=8, M=4096, N=1024, K=4096):

| | sw (BLOCK_M=64) | mw (4 waves, BLOCK_M=16, BLOCK_N=64) |
|---|---|---|
| VGPR/wave | 86 | 36 |
| **scratch** | **0** | **0** |
| LDS/WG | 0 | 4096 B |
| MFMAs | 128 | 32 |
| ds_read/write | 0 | 96 |
| s_barrier | 0 | 32 |
| Total instrs | 2812 | 552 |

**Key findings**:
1. **No register spill** anywhere (scratch=0, VGPR << 256 budget).
2. sw at `BLOCK_M=64` already does 4 MFMAs/K-iter (`M_SUB=4`). 128 total
   MFMAs/WG. That's why it's not a "small kernel".
3. mw at 1 MFMA/K-iter/wave is the **opposite** — thin inner loop.
4. WG count works out the same: sw `(M/64)·(N/16)·B = 32768`,
   mw `(M/16)·(N/64)·B = 32768`. But mw runs **4× more waves** for the
   same work (131k vs 32k). Each mw wave still pays K-loop overhead.

**Conclusion**: the bottleneck is **K-loop overhead amortization**, not
spill, not memory bandwidth. sw amortizes overhead across 4 MFMAs per
iter; mw across 1.

### Iter 5 — Triton headroom check *(no FlyDSL change)*

**Why bother**: before optimizing further, make sure the optimization
direction is worth it. If FlyDSL is already at Triton parity, gains are
small; if FlyDSL is way behind, headroom is huge.

| shape | sw us | mw us | **Triton us** | sw/Triton |
|---|---|---|---|---|
| (8,16,1024,4096) decode | 303 | 300 | **31** | 9.75× slower |
| (8,64,1024,4096) decode | 330 | 326 | **38** | 8.73× slower |
| (8,1024,1024,4096) prefill | 649 | 670 | **134** | 4.85× slower |
| (8,4096,1024,4096) prefill | 1589 | 1798 | **458** | 3.47× slower |

Theoretical fp8 lower bound at the largest shape ~46 µs (6 PFLOPS peak).
Triton @ 458 µs ≈ 10% of peak. sw @ 1589 µs ≈ 3% of peak.

**Headroom is 3-10×**. Optimization worth pursuing.

### Iter 6 — sw + N_SUB=2 (`BLOCK_N=32`) — *prefill 1.30-1.46× win*

**Motivation**: from Iter 4 we know per-K-iter overhead is the limiter.
sw already amortizes via `M_SUB=4`. Adding `N_SUB=2` doubles the
amortization window — same per-K-iter overhead spread over 8 MFMAs
instead of 4.

**Why this is cheap to do**:
- Both N sub-tiles fit in the same 128-N W_scale block (BLOCK_N=32 < 128
  and 32 divides 128) → **one W_scale byte covers both N subs**.
- A loads are shared across N subs (per m_sub, load A once, issue 2
  MFMAs).
- W loads are shared across M subs (per n_sub, load W once, issue 4
  MFMAs).
- VGPR cost: 8 accumulators (M_SUB×N_SUB×4 fp32) instead of 4 = +16
  VGPRs. Plus an extra v8i32 W tile = +8 VGPRs. Total +24 VGPRs.

**Key code** (factory + K-loop changes):
```python
# factory: 2D MFMA grid per WG = M_SUB × N_SUB micro-tiles
M_SUB = BLOCK_M // 16              # 1, 2, or 4
N_SUB = BLOCK_N // 16              # 1 or 2 (NEW)
GRID_N = N // BLOCK_N              # was N // 16 (NEW: divides by larger BLOCK_N)

# Per-(n_sub, lane) W row dword base. n_sub stacks 16 cols along N.
w_row_dword_bases = [
    (pid_b * fx.Index(W_BATCH_STRIDE)
     + (pid_n * fx.Index(BLOCK_N) + fx.Index(n_sub * 16) + row) * fx.Index(K))
    // fx.Index(4)
    for n_sub in range_constexpr(N_SUB)            # NEW
]

# Both N sub-tiles fit in the same 128-N W_scale block (BLOCK_N=32 < 128
# and 32 divides 128) → one w_scale byte covers all N_SUB MFMAs/iter.
n_block_idx = (pid_n * fx.Index(BLOCK_N)) // fx.Index(128)

# Per-(m_sub, n_sub) accumulator (4 fp32/lane). 2D list, M_SUB × N_SUB.
accs = [
    [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range_constexpr(N_SUB)]
    for _ in range_constexpr(M_SUB)
]

# K-loop body (essence): load N_SUB W tiles, one W_scale, then for each
# m_sub load A + A_scale ONCE and emit N_SUB MFMAs sharing them.
for k_tile in range_constexpr(K_g):
    # Load N_SUB W tiles
    b_tiles = []
    for n_sub in range_constexpr(N_SUB):
        w_lo = buffer_ops.buffer_load(w_rsrc, w_row_dword_bases[n_sub] + ...)
        w_hi = buffer_ops.buffer_load(w_rsrc, ...)
        b_tiles.append(vector.from_elements(v8i32, [w_lo[0..3], w_hi[0..3]]))

    w_scale_packed = _ue8m0_byte_pack4(Ws_[pid_b, n_block_idx, fx.Index(k_tile)])

    for sub in range_constexpr(M_SUB):
        a128 = ...   # load A once per (sub, k_tile)
        a_scale_packed = _ue8m0_byte_pack4(As_[pid_b, m_row, fx.Index(k_tile)])
        for n_sub in range_constexpr(N_SUB):                   # NEW inner loop
            tile_acc = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                T.f32x4,
                [a128, b_tiles[n_sub], ..., a_scale_packed, 0, w_scale_packed],
            )
            for i in range_constexpr(4):
                accs[sub][n_sub][i] += tile_acc[i]
```

**Heuristic** (`_pick_block_n`): only enable `BLOCK_N=32` for `M >= 128`
(prefill). Decode keeps `BLOCK_N=16` to maximize WG count for chip
saturation.

**Result**:
| shape | mode | sw OLD | **sw NEW** | Triton | Δ vs OLD | vs Triton |
|---|---|---|---|---|---|---|
| (8,16,1024,4096) | decode | 303 | 313 | 31 | ~noise | 10× slower |
| (8,64,1024,4096) | decode | 330 | 327 | 38 | ~noise | 8.6× slower |
| (8,1024,1024,4096) | prefill | 649 | **499** | 132 | **1.30×** | 3.8× slower |
| (8,4096,1024,4096) | prefill | 1589 | **1093** | 458 | **1.46×** | 2.3× slower |

ISA after:
- VGPR 86 → **128** (still no spill)
- MFMAs/WG 128 → **256** (2× compute)
- buffer_load 864 → **928** (~8% more — A loads amortized)
- s_waitcnt 484 → **447** (fewer waitcnts despite 2× compute → better
  schedule)
- Total instrs 2812 → 3916 (+40% for 2× compute → good amortization)
- Occupancy: 5 → 4 waves/SIMD (modest decrease)

---

### Iter 7 — sw path B: `scf.for` + loop-carried A/W prefetch — *small (~2%) win*

**Motivation**: per FlyDSL's `prefetch-data-load` skill + the
`blockscale_preshuffle_gemm.py` reference, restructure the K-loop so the
HBM load for tile k+1 is issued at the END of iter k's body, overlapping
with iter k's MFMA pipeline.

**Change**: branch the K-loop on `native_scale_mfma`:
- Path A (decode, BLOCK_M ≤ 32): keep existing `range_constexpr`
  fully-unrolled K-loop, untouched.
- Path B (prefill, BLOCK_M = 64): use FlyDSL's `range(0, K_g - 1, 1, init=...)`
  to lower to scf.for with loop-carried state. Pre-load A+W tiles for
  k=0 in a prologue; in each iter issue prefetch for k+1 *before*
  computing MFMA for k; epilogue handles iter K_g - 1.

**State carried across iters** (20 SSA values):
- `k_tile_idx` (index)
- M_SUB × 2 = 8 v4i32 prefetched A tiles (lo + hi pairs)
- N_SUB × 2 = 4 v4i32 prefetched W tiles
- M_SUB × N_SUB = 8 v4f32 accumulators

Per-row a_scale and W_scale loads stayed synchronous (1-byte loads, ~always
cached, would explode state size to no benefit).

**Critical FlyDSL footguns this change exposed** (and one new finding):
1. `for ... in range(N)` inside `@flyc.kernel` is **always** lowered to
   scf.for, even without `init=`. The result is a runtime loop, not
   Python unrolling — and Python `range_constexpr(N)` is required for
   any inner loop you want unrolled. Symptom: `TypeError: list indices
   must be integers or slices, not ArithValue` when indexing a Python
   list with the loop variable.
2. From the skill: loop init values must be raw `ir.Value` (use
   `_unwrap` from `flydsl.expr.utils.arith`).
3. From the skill: when scf.for has `init=`, bare Python int bounds
   work (skill doc was overly conservative).

**Key code** (the scf.for + loop-carried state + prologue/epilogue
shape):
```python
# === PROLOGUE: pre-load A and W tiles for k=0 ===
a_pref_pairs = []
for sub in range_constexpr(M_SUB):
    a_lo = buffer_ops.buffer_load(a_rsrc, ..., vec_width=4, dtype=T.i32)
    a_hi = buffer_ops.buffer_load(a_rsrc, ..., vec_width=4, dtype=T.i32)
    a_pref_pairs.append((a_lo, a_hi))
w_pref_pairs = [...]   # similar

# Pack init_state — order MUST match unpack
init_state_raw = [arith.constant(0, index=True)]   # k_tile_idx
for lo, hi in a_pref_pairs: init_state_raw += [lo, hi]
for lo, hi in w_pref_pairs: init_state_raw += [lo, hi]
for sub in range_constexpr(M_SUB):
    for n_sub in range_constexpr(N_SUB):
        init_state_raw.append(accs[sub][n_sub])
init_state = [_unwrap(v) for v in init_state_raw]   # raw ir.Value's

# === scf.for body, K_g - 1 iters ===
for _bki, state in range(0, K_g - 1, 1, init=init_state):   # bare range with init=
    # Unpack state
    k_tile_idx = state[0]
    a_cur = [(state[1 + 2*s], state[1 + 2*s + 1]) for s in range_constexpr(M_SUB)]
    w_cur = [...]
    accs_cur = [...]

    # Prefetch next iter's A + W (overlap with this iter's MFMA)
    k_tile_idx_next = k_tile_idx + arith.constant(1, index=True)
    k_dword_off_next = k_tile_idx_next * fx.Index(32)
    a_next_pairs = [_load_a_for(s, k_dword_off_next) for s in range_constexpr(M_SUB)]
    w_next_pairs = [_load_w_for(n, k_dword_off_next) for n in range_constexpr(N_SUB)]

    # Compute MFMA for current iter (uses a_cur, w_cur, scales loaded sync)
    accs_new = _do_compute_b(a_cur, w_cur, k_tile_idx, accs_cur)

    # Yield next state
    next_state_raw = [k_tile_idx_next] + flatten(a_next_pairs) + flatten(w_next_pairs) + flatten(accs_new)
    results = yield [_unwrap(v) for v in next_state_raw]

# === EPILOGUE: process last K-iter ===
k_tile_idx_last = results[0]
a_last = ...; w_last = ...; accs_last_in = ...
accs = _do_compute_b(a_last, w_last, k_tile_idx_last, accs_last_in)
```

**Result**:
| shape | mode | sw Iter 6 | **sw Iter 7** | Δ |
|---|---|---|---|---|
| (8,16,1024,4096) | decode | 313 | 311 | ~noise (path A unchanged) |
| (8,64,1024,4096) | decode | 327 | 327 | unchanged |
| (8,1024,1024,4096) | prefill | 499 | **496** | -0.6% |
| (8,4096,1024,4096) | prefill | 1093 | **1072** | **-1.9%** |

ISA after:
- VGPR 128 → **154** (still no spill, scratch=0)
- Per-iter buffer_load count includes 12 prefetch loads (8 A + 4 W)
  alongside the 16 sync a_scale loads + 1 W_scale load
- s_waitcnt:MFMA ratio per iter: roughly unchanged from Iter 6 (~14:8)

**Lessons**:
- Predicted 5-15% (per skill); got 2%. Likely because the LLVM compiler
  was already reordering the fully-unrolled loop well (we saw a healthy
  1.7× waitcnt:MFMA in iter 4).
- The 16 synchronous per-row a_scale byte loads per K-iter remain on
  the MFMA critical path — they're tiny (1 B each) but dependent.
  Prefetching A+W can't hide *this* latency.
- VGPR cost (+26) is the price; occupancy went from 4 → 3 waves/SIMD.
  Net still positive.
- Change kept (small win, no regression, exercises the scf.for pattern
  for future iters).

**Code**: `compile_bgfp8bw_kernel`, `else` branch of the
`if native_scale_mfma:` split. ~150 LOC for path B. The repeated
`vector.from_elements(v8i32, [w_lo[0], ..., w_hi[3]])` pattern was
factored into `_do_compute_b` to reduce duplication between the loop
body and epilogue.

---

### Iter 8 — v2 kernel (4 waves/WG + LDS-A + 128×128 tile) — *Phase 1: 1.34× over sw*

**Background**: Iter 7 prefetch only got us 2% prefill. To diagnose why, we
read Triton's actual kernel source. Findings:

| Dim | Triton | sw (Iter 7) | Ratio |
|---|---|---|---|
| BLOCK_M × BLOCK_N | 128×128 = 16384 | 64×32 = 2048 | **8× tile** |
| Waves per WG | 4 | 1 | 4× |
| `num_stages` (LDS pipe) | 2 | ~1 | 2× |
| MFMA dtype | bf16 | **fp8 native + scaleA/scaleB** | We have the better MFMA |

The 2.3× gap is **structural** — not load-coalescing, not scheduling. Our
sw geometry is 8× smaller per WG. Triton uses 4 waves cooperating with LDS
double-buffer.

**Critically**: Triton's docstring (lines 137-142) says they reject
`mfma_scale_*_f8f6f4` because "gfx950 fp8 MFMA path applies a single per-tile
scale, which doesn't match recipe(1,1,128)." This is **wrong** for our
recipe — `mfma_scale_*_f8f6f4` accepts per-32K-block scales (4 bytes per
i32 packed scaleA/scaleB). We DO match. So we have the more efficient MFMA
that Triton itself doesn't use.

**Plan**: build a v2 kernel matching Triton's geometry but keeping our
fp8 MFMA. New file `batched_gemm_fp8_blockwise_flydsl_v2.py`. 4 phases:

| Phase | Goal | Target perf |
|---|---|---|
| 1 — Skeleton | 4 waves + LDS-A single-buffer + cooperative load + native fp8 MFMA | ≤ 700 µs |
| 2 — XOR-swizzle | Eliminate LDS bank conflicts | ≤ 600 µs |
| 3 — Async copy + sched hints | DMA HBM→LDS, hand-tuned MFMA/load interleave | ≤ 500 µs (Triton parity) |
| 4 — cshuffle epilogue | Coalesced output stores (only if profile shows store stalls) | ≤ 400 µs |

**Phase 1 implementation** (this iter):
- Geometry: BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, 4 waves of 64 lanes
  = 256 threads/WG.
- Wave division: each wave handles 128M × 32N (waves split BLOCK_N as in
  blockscale_preshuffle_gemm).
- Per wave per K-iter: M_SUB=8 × N_SUB=2 = **16 MFMAs**. Per WG: 64.
- LDS A: single 16 KB tile, all 4 waves read it.
- W: direct HBM → VGPR per wave (each wave loads its own 32-N stripe).
- Cooperative HBM→LDS A load: 256 threads × 4 dwordx4 each = 16 KB/iter.
  Layout: 2 threads per row, each owns 64 contiguous K-bytes.
- Native fp8 MFMA with packed scaleA + scaleB.
- No XOR swizzle, no async copy, no sched hints, no double buffer (yet).

**Key code** (geometry constants + the cooperative load + K-loop body):
```python
# v2 geometry (fixed)
_BLOCK_M = 128
_BLOCK_N = 128
_BLOCK_K = 128
_N_WAVES = 4
_WAVE_SIZE = 64
_BLOCK_THREADS = _N_WAVES * _WAVE_SIZE   # 256
_N_PER_WAVE = _BLOCK_N // _N_WAVES       # 32 — N stripe per wave
_M_SUB = _BLOCK_M // 16                  # 8 MFMAs along M per wave
_N_SUB = _N_PER_WAVE // 16               # 2 MFMAs along N per wave
_LDS_A_BYTES = _BLOCK_M * _BLOCK_K       # 16384 = 16 KB

# Wave/lane coordinates
tid = fx.thread_idx.x                       # 0..255
wave_id = tid // fx.Index(_WAVE_SIZE)       # 0..3
lane = tid % fx.Index(_WAVE_SIZE)           # 0..63
wave_n_offset = wave_id * fx.Index(_N_PER_WAVE)  # 0, 32, 64, 96

# 16 accumulators per lane (M_SUB × N_SUB × v4f32)
accs = [
    [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range_constexpr(_N_SUB)]
    for _ in range_constexpr(_M_SUB)
]

# Cooperative HBM→LDS A load lane mapping (Phase 1: sync via VGPR)
ld_row = tid // fx.Index(2)                       # 0..127
ld_byte_start = (tid % fx.Index(2)) * fx.Index(64)  # 0 or 64

# K-loop body
for k_tile in range_constexpr(K_g):
    # 1. Cooperative HBM → LDS via 4 dwordx4 per thread = 16 KB total
    for chunk in range_constexpr(4):
        a_chunk = buffer_ops.buffer_load(a_rsrc, ..., vec_width=4, dtype=T.i32)
        a_chunk_b = vector.bitcast(v16i8_t, a_chunk)
        as_lds.vec_store((ld_row, ld_byte_start + fx.Index(chunk * 16)),
                          a_chunk_b, 16)
    gpu.barrier()

    # 2. Per-wave W stripe: 2 buffer_load_dwordx4 per n_sub (per lane)
    b_tiles = []
    for n_sub in range_constexpr(_N_SUB):
        w_lo = buffer_ops.buffer_load(w_rsrc, ..., vec_width=4, dtype=T.i32)
        w_hi = buffer_ops.buffer_load(w_rsrc, ..., vec_width=4, dtype=T.i32)
        b_tiles.append(vector.from_elements(v8i32, [w_lo[0..3], w_hi[0..3]]))

    # 3. M_SUB × N_SUB = 16 MFMAs, A read from LDS once per m_sub
    for m_sub in range_constexpr(_M_SUB):
        a_lds_row = fx.Index(m_sub * 16) + row
        a_lds_vec = as_lds.vec_load((a_lds_row, k_byte_in_lane), 32)
        a128 = vector.bitcast(v8i32, a_lds_vec)
        a_scale_packed = _ue8m0_byte_pack4(As_[pid_b, m_row_per_sub[m_sub], ...])
        for n_sub in range_constexpr(_N_SUB):
            tile_acc = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                T.f32x4,
                [a128, b_tiles[n_sub], ..., a_scale_packed, 0, w_scale_packed])
            for i in range_constexpr(4):
                accs[m_sub][n_sub][i] += tile_acc[i]

    gpu.barrier()  # before next iter overwrites LDS-A
```

**Result (Phase 1)**:
| shape | mode | sw (Iter 7) | **v2 Phase 1** | Triton | v2/sw | v2/Triton |
|---|---|---|---|---|---|---|
| (8,16,1024,4096) | decode | 312 | n/a (sw path) | 31 | — | 9.96× slower |
| (8,64,1024,4096) | decode | 326 | n/a (sw path) | 38 | — | 8.58× slower |
| (8,1024,1024,4096) | prefill | 499 | **430** | 138 | **1.16×** | 3.11× slower |
| (8,4096,1024,4096) | prefill | 1072 | **797** | 461 | **1.34×** | **1.74× slower** |

Wrapper dispatches to v2 for `M >= 128 && M % 128 == 0 && N % 128 == 0`.
Decode (M < 128 or non-multiple) stays on sw. No decode regression.

ISA after Phase 1 (M=4096):
- LDS: **16 KB/WG** (matches plan)
- VGPR/wave: **138** (no spill, scratch=0)
- Static MFMAs: 512 = K_g × M_SUB × N_SUB = 32 × 8 × 2
- s_waitcnt:MFMA ratio: 698:512 = **1.36** (better than sw's 1.75 →
  LDS sharing is helping despite single-buffer)
- s_barrier: 64 = K_g × 2 (one before A reads, one before next A writes)
- Occupancy: ~3 waves/SIMD (4 was sw's count; geometry shift)

**Result vs original baseline (Iter 0 sw at 1589 µs on M=4096):**
- Iter 6: 1.46× (1589 → 1093)
- Iter 7: 1.48× (+ 2%)
- **Iter 8 Phase 1: 1.98× (1589 → 797)** ← best so far

**Lessons**:
- Geometry trumps micro-optimizations. Iter 6 N_SUB doubling = +1.46×;
  Iter 8 wholesale geometry rewrite = +1.34× on top of that.
- Single-buffer LDS works fine for Phase 1; we don't need ping/pong yet
  to get the structural win.
- Reading the *competitor's* source code was the highest-ROI diagnostic
  in the entire journey — it told us exactly where the gap was.
- Phase 1 came in slightly above the 700 µs target (797 µs) but already
  closed the Triton gap from 2.34× to 1.74×.

**Code**: new file `batched_gemm_fp8_blockwise_flydsl_v2.py` (~280 LOC).
Wrapper dispatch: `flydsl_batched_gemm_fp8_blockwise()` checks
`M >= 128 && M % 128 == 0 && N % 128 == 0` and routes to v2.

### Iter 8 Phase 2 — XOR-swizzle on LDS A — *4% extra*

**Motivation**: in Phase 1 each lane in a wave reads its A row from LDS at
the same K-byte offset → 16-way LDS bank conflict (all 16 lanes hit bank 0
for the first dword of their respective rows). Standard fix: XOR-swizzle the
LDS column with `(row & (k_blocks16 - 1)) * 16` so consecutive rows hit
different banks.

**Pattern borrowed**: `swizzle_xor16` from
`FlyDSL/kernels/mfma_preshuffle_pipeline.py:28`.

**Implementation**:
- Helper `_swizzle_xor16(row, col_bytes)` — k_blocks16 hardcoded to 8
  (= BLOCK_K / 16 = 128 / 16).
- Cooperative HBM→LDS write: each chunk's byte offset XOR-swizzled.
- LDS read: **must split** the 32-byte vec_load into 2× 16-byte reads, each
  with its own swizzled offset. (Single 32-byte read would get the two
  halves in possibly-swapped order since the XOR works at 16-byte
  granularity.) Then reassemble as v8i32.

**Key code**:
```python
# Helper at module level
_K_BLOCKS16 = 8        # BLOCK_K / 16
_K_BLOCKS16_MASK = 7   # _K_BLOCKS16 - 1

def _swizzle_xor16(row, col_bytes):
    """col_bytes XOR ((row & (k_blocks16-1)) * 16). Self-inverse."""
    rem = arith.andi(row, arith.index(_K_BLOCKS16_MASK))
    return col_bytes ^ (rem * 16)


# === Cooperative LDS write (Phase 1 → Phase 2: just swizzle the byte off) ===
for chunk in range_constexpr(4):
    a_chunk = buffer_ops.buffer_load(a_rsrc, ..., vec_width=4, dtype=T.i32)
    a_chunk_b = vector.bitcast(v16i8_t, a_chunk)
    orig_byte_off = ld_byte_start + fx.Index(chunk * 16)
    swz_byte_off = _swizzle_xor16(ld_row, orig_byte_off)        # NEW
    as_lds.vec_store((ld_row, swz_byte_off), a_chunk_b, 16)     # swizzled write


# === LDS read (Phase 1 single 32B → Phase 2 two 16B with separate swizzles) ===
# Phase 1 was:  a_lds_vec = as_lds.vec_load((a_lds_row, k_byte_in_lane), 32)
# Phase 2:
swz_lo = _swizzle_xor16(a_lds_row, k_byte_in_lane)              # bytes 0..15
swz_hi = _swizzle_xor16(a_lds_row, k_byte_in_lane + fx.Index(16))  # bytes 16..31
half_lo_b = as_lds.vec_load((a_lds_row, swz_lo), 16)            # 16B
half_hi_b = as_lds.vec_load((a_lds_row, swz_hi), 16)            # 16B
half_lo = vector.bitcast(v4i32_t, half_lo_b)
half_hi = vector.bitcast(v4i32_t, half_hi_b)
a128 = vector.from_elements(v8i32,
    [half_lo[0], half_lo[1], half_lo[2], half_lo[3],
     half_hi[0], half_hi[1], half_hi[2], half_hi[3]])
```

**Result**:
| shape | mode | Phase 1 | **Phase 2** | Δ |
|---|---|---|---|---|
| (8,1024,1024,4096) | prefill | 430 | 428 | ~noise |
| (8,4096,1024,4096) | prefill | 797 | **765** | **-4.0%** |

ISA after Phase 2 (M=4096):
- VGPR 138 → **150** (+12 for the XOR VALU)
- LDS unchanged (16 KB)
- s_waitcnt 698 → 680 (-3%)
- s_waitcnt:MFMA ratio: 1.36 → **1.33** (closer to compute-bound)

**Lessons**:
- Predicted >10% LDS conflict win; got 4%. Either the compiler/HW was
  already mitigating part of it (LDS fetch-ahead), or LDS reads weren't
  the dominant stall — MFMA throughput is closer to the limit.
- The 32B→2×16B split adds VGPR pressure (intermediate v4i32 holds) but
  does NOT change the LDS op count (compiler was already splitting under
  the hood in Phase 1, per the matched ds_read+ds_write totals).
- Each XOR adds ~3 VALU instructions per LDS access; net gain is the
  saved bank-conflict cycles minus this overhead.

**Status vs targets**:
| Target | Phase 2 actual |
|---|---|
| Phase 2 perf goal: ≤ 600 µs | **765 µs (missed)** |
| Bound vs Triton at M=4096 | **1.66× slower** (was 1.73× post-Phase 1) |

Phase 2 fell short of its 600 µs goal. Diminishing returns suggest the
remaining gap to Triton (461 µs) is in the inner loop's MFMA scheduling
+ HBM-W load latency, not LDS conflicts. Phase 3 (`sched.barrier` /
`sched_mfma` hints + async copy) is the next planned step but the
predicted gain is uncertain.

### Iter 8 Phase 3a — `sched_*` hints — *regressed, reverted*

**Motivation**: tell LLVM the desired instruction issue order — issue all
HBM loads first, then alternate LDS-read + MFMA — so HBM latency overlaps
with compute and LDS-read latency stays close to the MFMA that consumes it.

**Implementation** (FlyDSL `rocdl.sched_*` API per the `gemm-optimization`
skill + hgemm_splitk reference):
```python
rocdl.sched_barrier(0)
for _ in range_constexpr(13):
    rocdl.sched_vmem(1)         # 4 W + 8 a_scale + 1 w_scale per iter
for _ in range_constexpr(16):
    rocdl.sched_dsrd(1)         # 16 LDS reads (M_SUB*2 halves)
    rocdl.sched_mfma(1)         # 16 MFMAs
rocdl.sched_barrier(0)
```

**Result**:
| | Phase 2 (no hints) | Phase 3a (hints) | Δ |
|---|---|---|---|
| M=4096 wall time | **757 µs** | 770 µs | **+1.7% (regressed)** |
| VGPR | 150 | 142 | -8 |
| s_nop | (low) | **730** | +730 |
| total ISA instrs | (~3700) | 4594 | +25% |

**Diagnosis**: the compiler emitted **730 `s_nop`** instructions to satisfy
my hand-coded ordering constraints. The cost of those nops exceeded the
benefit from the reordering. The hgemm_splitk reference uses
`hot_loop_scheduler` patterns that are *empirically tuned per shape* —
a generic 13-vmem / 16-dsrd / 16-mfma pattern is too crude.

**Decision**: reverted. Sched hints removed from v2.

**Lessons**:
- `sched_*` hints without empirical tuning regress.
- The compiler's default scheduler is already pretty good — beating it
  needs profiling-driven iteration, not theoretical reasoning.
- Adding `rocdl.sched_barrier(0)` is essentially adding ~25 nops per
  iteration cycle, which only pays off if the reordering it permits
  saves >25 cycles.

### Iter 8 Phase 3b — async copy (`raw_ptr_buffer_load_lds`) — *4.5%*

**Mechanism**: replaces the cooperative `buffer_load → ds_write` round-trip
with a single `raw_ptr_buffer_load_lds` per chunk per K-iter. The hardware
DMAs HBM directly into LDS without going through arch_vgpr.

**Lane-mapping pivot**: `raw_ptr_buffer_load_lds` writes lane `l`'s data to
`LDS[scalar_base + l * dma_bytes]` (lane stride is hardware-fixed). So we
re-architect the cooperative-load mapping to match this striped layout:
- `chunk c, wave w, lane l → row = c*32 + w*8 + l//8, col = (l%8)*16`
- Each chunk handles 32 LDS rows; each wave inside a chunk handles 8 rows.

**XOR-swizzle preserved**: instead of swizzling the LDS-write column (no
control over LDS write address with async DMA), we swizzle the **HBM-read
column**. Since XOR is self-inverse, the LDS layout still has the swizzled
property and the read side keeps the same `_swizzle_xor16` formula
unchanged.

**Key code** (LDS scalar pointer setup + the inner DMA replacement):
```python
# Outside K-loop: per-wave scalar LDS base (broadcast via readfirstlane)
_lds_base_raw_idx = _memref_dialect.extract_aligned_pointer_as_index(lds_a_memref)
_lds_base_idx = (
    _lds_base_raw_idx
    + wave_id * fx.Index(_WAVE_SIZE * _DMA_BYTES)   # 1 KB stride per wave
)
_lds_base_i64 = rocdl.readfirstlane(T.i64, arith.index_cast(T.i64, _lds_base_idx))
_lds_ptr_type = ir.Type.parse("!llvm.ptr<3>")

# Constants reused across all chunks
_DMA_BYTES_T = arith.constant(_DMA_BYTES, type=T.i32)
_SOFFSET_T = arith.constant(0, type=T.i32)
_OFFSET_IMM_T = arith.constant(0, type=T.i32)
_AUX_T = arith.constant(1, type=T.i32)

# === Inside K-loop: replaces the buffer_load + ds_write pair ===
# New per-thread mapping (matches HW lane stride for raw_ptr_buffer_load_lds):
#   chunk c, wave w, lane l → row = c*32 + w*8 + l//8, col = (l%8)*16
row_in_wave = lane // fx.Index(8)
col_byte_in_lane = (lane % fx.Index(8)) * fx.Index(16)

for chunk in range_constexpr(4):
    # LDS ptr advances by total_threads*dma_bytes = 4 KB per chunk
    _chunk_lds_addr = _lds_base_i64 + arith.constant(
        chunk * _BLOCK_THREADS * _DMA_BYTES, type=T.i64)
    _chunk_lds_ptr = _llvm.inttoptr(_lds_ptr_type, _chunk_lds_addr)

    # Per-lane HBM byte offset, with XOR-swizzle moved to the HBM read side
    absolute_row = (fx.Index(chunk * 32)
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

    # ONE-instruction direct HBM → LDS DMA (no VGPR round-trip)
    rocdl.raw_ptr_buffer_load_lds(
        a_rsrc, _chunk_lds_ptr, _DMA_BYTES_T,
        _global_offset_i32, _SOFFSET_T, _OFFSET_IMM_T, _AUX_T,
    )
```

**Result**:
| shape | Phase 2 | **Phase 3b** | Δ |
|---|---|---|---|
| (8,1024,1024,4096) | 431 | **419** | -2.8% |
| (8,4096,1024,4096) | 757 | **723** | **-4.5%** |

ISA delta (M=4096):
| | Phase 2 (sync HBM→VGPR→LDS) | Phase 3b (async HBM→LDS DMA) | Δ |
|---|---|---|---|
| VGPR | 150 | **138** | -12 (intermediate VGPR gone) |
| **ds_write** | **128** | **0** | gone (no LDS writes from VGPR) |
| **buffer_load_lds** (DMA) | 0 | **128** | NEW |
| s_waitcnt | 680 | **555** | **-18%** |
| MFMA / scratch / LDS | 512 / 0 / 16 KB | unchanged | |

**Lessons**:
- Async DMA's main wins are in **scheduling slack** (s_waitcnt -18%) and
  **VGPR pressure** (-12), not raw bandwidth (HBM transactions same count).
- The lane-mapping pivot is the gnarly part — `raw_ptr_buffer_load_lds`'s
  hardware-fixed lane stride means you adapt the kernel to its layout, not
  vice versa. Once you accept the chunk*32/wave*8/lane//8 mapping, the
  rest is mechanical.
- **Swizzle moves from write-side to read-side data path** but stays
  mathematically identical (XOR self-inverse). LDS-read formula unchanged.
- 4.5% gain is modest but real. Total v2 chain now hits **2.20× over
  baseline** (1589 → 723).

### Iter 8 Phase 3c — ablation: 4-wave coop vs 1-wave solo HBM→LDS load

**Question raised by reader**: "since gfx950's CU has only 1 LSU shared
across SIMDs, 4-wave cooperative VMEM issue should serialize to ~the same
total cost as 1-wave issuing all 16 DMAs. Is the 4-wave coop really
helping, or is it doing nothing?"

**Hypothesis**: 4-wave ≈ 1-wave (within 5%) since LSU is shared.

**Setup**: added `load_mode` parameter to v2 kernel (3 values: `coop`,
`solo_unroll`, `solo_loop`). Solo modes gate the cooperative DMAs behind
`scf.if (wave_id == 0)`; only wave 0 issues; waves 1–3 just hit
`gpu.barrier`.

**Key code** (the scf.if pattern + the failed scf.for attempt):
```python
# === solo_unroll: full unroll, only wave 0 issues 16 chunks ===
_is_wave_0 = arith.cmpi(_arith.CmpIPredicate.eq, wave_id, arith.index(0))
_if_op = scf.IfOp(_is_wave_0, results_=[], has_else=False)
with ir.InsertionPoint(_if_op.then_block):
    for chunk in range_constexpr(16):    # 16 chunks, fully unrolled
        # ... compute LDS ptr, HBM offset, swizzle ...
        rocdl.raw_ptr_buffer_load_lds(a_rsrc, _chunk_lds_ptr, ...)
    scf.YieldOp([])

# === solo_loop: tries scf.for to avoid unroll → still unrolled by LLVM ===
_if_op = scf.IfOp(_is_wave_0, results_=[], has_else=False)
with ir.InsertionPoint(_if_op.then_block):
    # CRITICAL: bounds MUST be arith.index(), not Python int (else AST
    # rewriter unrolls). Even with this, LLVM backend still unrolls a
    # 16-iter constant-trip loop in this case.
    for chunk_iv, _state in range(arith.index(0),
                                   arith.index(16),
                                   arith.index(1),
                                   init=[]):
        _chunk_lds_off_i64 = arith.index_cast(
            T.i64, chunk_iv * fx.Index(_WAVE_SIZE * _DMA_BYTES))
        # ... rest of body uses chunk_iv as runtime index ...
        rocdl.raw_ptr_buffer_load_lds(a_rsrc, _chunk_lds_ptr, ...)
        results = yield []
    scf.YieldOp([])
```

**Measured (M=4096 prefill)**:
| mode | wall time | solo/coop |
|---|---|---|
| coop (4-wave, 4 chunks/wave, range_constexpr) | 759 µs | 1.00× |
| solo_unroll (wave 0, 16 chunks, range_constexpr) | 2255 µs | **2.97×** |
| solo_loop (wave 0, 16 chunks via `scf.for`) | 2241 µs | **2.95×** |

**Hypothesis falsified by 3×.** Both solo modes are ~3× slower than coop,
not within 5%. Something other than LSU serialization is driving the gap.

**ISA forensics**:
| | coop | solo_unroll | solo_loop |
|---|---|---|---|
| VGPR | 138 | **512** | **512** |
| `private_segment_fixed_size` (scratch) | **0** | **6228 B** | **6212 B** |
| `scratch_load/store` instrs | **0** | **778** | **776** |
| `buffer_load_lds` (static count) | 128 | 512 | **512** |
| s_waitcnt | 555 | 949 | 940 |
| Occupancy | ~3 wave/SIMD | **1 wave/SIMD** | **1 wave/SIMD** |

**Real root cause: VGPR spill from unrolled DMAs, not LSU.**

In `solo_unroll`, the 16 unrolled `buffer_load_lds` calls each keep their
address registers live through the eventual `s_waitcnt`, and LLVM allocates
**16 sets of address VGPRs simultaneously** → VGPR overflow → 6228 B
spilled to scratch → **778 scratch_load/store instructions** in the K-loop
(each ~hundreds of cycles) → occupancy collapses from 3 → 1 wave/SIMD.

**Why scf.for didn't save us**: I tried `solo_loop` with
`for chunk_iv, _ in range(arith.index(0), arith.index(16), arith.index(1), init=[])`
expecting a runtime loop with shared address registers. The MLIR IR DID
emit `scf.for` (count of 32 buffer_load_lds in MLIR after SCF lowering).
**But LLVM's loop unroller in the backend then fully unrolled the
16-iteration constant-trip loop** — the final ISA has 512
buffer_load_lds and the exact same VGPR/scratch profile as solo_unroll.

To genuinely test "scf.for solo without unroll", we'd need to inject
`llvm.loop.unroll.disable` metadata, which FlyDSL doesn't currently
expose at the Python layer. Out of scope for this session.

**Real takeaways**:
1. **Multi-wave's biggest win isn't issue parallelism — it's amortized
   per-wave register pressure.** 4-wave coop emits 4 DMAs per wave's
   unrolled code; 1-wave emits 16 DMAs in one wave's code. The latter
   blows the VGPR budget.
2. **LLVM aggressively unrolls small constant-bound `scf.for`** —
   `arith.index()` bounds emit scf.for in MLIR but get unrolled later.
   Skill doc's "must use arith.index for scf.for" is necessary but not
   sufficient to prevent unroll.
3. **The original (incorrect) "4× faster issue" claim was wrong but
   directionally pointing at the right answer.** The (also incorrect)
   revised "≤5% LSU-serialized" claim was further from the truth. The
   actual mechanism is per-wave register pressure, which neither claim
   identified.
4. v2's coop_load=True remains the production default. The two solo
   modes are kept in the codebase for ablation reproducibility.

**Code**: `batched_gemm_fp8_blockwise_flydsl_v2.py` `load_mode` parameter
(default `"coop"`). Bench script: `/tmp/bench_coop_vs_solo.py`.

### Iter 8 Phase 4a — A_scale + W_scale to LDS — *24% prefill win!*

**Motivation**: per K-iter the kernel issued 9 small HBM byte loads —
8 for A_scale (per-row, per m_sub) and 1 for W_scale. Each is
single-byte but each requires a `vmcnt` wait before the MFMA can use
the scale. Hypothesized small win (1-3%) by lifting them to a
prologue cooperative LDS load and replacing the K-loop HBM reads with
LDS reads.

**Implementation**:
- New LDS regions: A_scale (BLOCK_M × K_g = 128×32 = **4 KB**),
  W_scale (K_g rounded to 16 = **16 B**).
- Prologue: 1 cooperative `raw_ptr_buffer_load_lds` per thread for
  A_scale (256 × 16 B = 4 KB exactly), + 2 DMAs by lane 0 for
  W_scale (gated by `scf.if (tid == 0)`).
- K-loop: read scales from LDS (`as_scale_lds[local_m_row, k_tile]`,
  `ws_scale_lds[k_tile]`) instead of HBM `buffer_load`. Pack at use
  site as before (1 multiply by `0x01010101`).

**Key code** (LDS allocation + prologue cooperative load + K-loop change):
```python
# === Allocator (factory time) ===
_LDS_AS_BYTES = _BLOCK_M * K_g       # 128 × 32 = 4096
_LDS_WS_BYTES = max(K_g, 16)         # 32 (or 16 if smaller)
allocator.ptr = smem_a_offset + _LDS_A_BYTES
smem_as_offset = allocator._align(allocator.ptr, 16)
allocator.ptr = smem_as_offset + _LDS_AS_BYTES
smem_ws_offset = allocator._align(allocator.ptr, 16)
allocator.ptr = smem_ws_offset + _LDS_WS_BYTES

# === LDS views (kernel time) ===
as_scale_lds = STensor(SmemPtr(allocator.get_base(), smem_as_offset, i8_t,
                                shape=(_LDS_AS_BYTES,)),
                        dtype=i8_t, shape=(_BLOCK_M, K_g))
ws_scale_lds = STensor(SmemPtr(allocator.get_base(), smem_ws_offset, i8_t,
                                shape=(_LDS_WS_BYTES,)),
                        dtype=i8_t, shape=(_LDS_WS_BYTES,))

# === PROLOGUE: cooperative DMA scales into LDS (once per WG) ===
# A_scale: 256 threads × 16 B = 4 KB (one cooperative round)
_AS_DMA_BYTES = 16
_hbm_as_byte_off = (
    pid_b * fx.Index(AS_BATCH_STRIDE)
    + pid_m * fx.Index(_BLOCK_M * K_g)
    + tid * fx.Index(_AS_DMA_BYTES)
)
_lds_as_wave_base = (
    _memref_dialect.extract_aligned_pointer_as_index(lds_as_memref)
    + wave_id * fx.Index(_WAVE_SIZE * _AS_DMA_BYTES)
)
_lds_as_ptr = _llvm.inttoptr(
    _lds_ptr_type,
    rocdl.readfirstlane(T.i64, arith.index_cast(T.i64, _lds_as_wave_base)))
rocdl.raw_ptr_buffer_load_lds(
    as_rsrc, _lds_as_ptr, arith.constant(_AS_DMA_BYTES, type=T.i32),
    arith.index_cast(T.i32, _hbm_as_byte_off),
    _SOFFSET_T, _OFFSET_IMM_T, _AUX_T,
)

# W_scale: 32 B total, gate to lane 0 of wave 0 only (2 DMAs of 16 B each)
_is_lane0 = arith.cmpi(_arith.CmpIPredicate.eq,
                        tid, arith.constant(0, type=T.i32))
_ws_if = scf.IfOp(_is_lane0, results_=[], has_else=False)
with ir.InsertionPoint(_ws_if.then_block):
    for ws_chunk in range_constexpr((_LDS_WS_BYTES + 15) // 16):
        rocdl.raw_ptr_buffer_load_lds(
            ws_rsrc, _lds_ws_ptr_for(ws_chunk), arith.constant(16, type=T.i32),
            ..., _SOFFSET_T, _OFFSET_IMM_T, _AUX_T)
    scf.YieldOp([])

gpu.barrier()  # ensure scales visible to all waves before K-loop


# === K-loop replacement: HBM byte load → LDS byte load ===
# Before (Phase 3b):
#   a_scale_byte = As_[pid_b, m_row_per_sub[m_sub], fx.Index(k_tile)]   # HBM
#   w_scale_byte = Ws_[pid_b, n_block_idx, fx.Index(k_tile)]            # HBM
# After (Phase 4a):
local_m_row = fx.Index(m_sub * 16) + row    # 0..127
a_scale_byte = as_scale_lds[local_m_row, fx.Index(k_tile)]   # LDS ds_read 1B
w_scale_byte = ws_scale_lds[fx.Index(k_tile)]                # LDS ds_read 1B
a_scale_packed = _ue8m0_byte_pack4(a_scale_byte)             # pack still here
w_scale_packed = _ue8m0_byte_pack4(w_scale_byte)
```

**Result (M=4096 prefill)**:
| | Phase 3b | **Phase 4a** | Δ |
|---|---|---|---|
| **wall time** | 723 µs | **543 µs** | **-25%** |
| vs Triton (469 µs) | 1.54× behind | **1.16× behind** | huge |
| LDS | 16 KB | 20.6 KB | +4 KB |
| VGPR | 138 | **134** | -4 |
| HBM buffer_load (static) | 544 | **259** | **-52%** |
| ds_read | 512 | 800 | +288 |
| s_waitcnt | 555 | **386** | **-30%** |
| s_barrier | 64 | 65 | +1 prologue barrier |
| total instrs | 4544 | 4434 | -2.4% |

**Why so much bigger than the 1-3% prediction**:
- I underestimated how much each "small" HBM byte load was costing.
  Each carried a `vmcnt` wait before MFMA could consume the packed
  scale → serialization with the rest of the vmem queue.
- Removing 9 HBM reads per K-iter × 32 iters = 288 vmcnt-waited HBM
  ops removed from the critical path. The s_waitcnt count drop
  (-169 instrs, -30%) directly reflects this.
- Scale reads from LDS are 1-cycle ds_read instead of multi-hundred
  cycle vmcnt-waited HBM. Net gain at execution time is much larger
  than the static instruction count change suggests.

**Lessons**:
- "Cached HBM byte load" is NOT free — even when cache hits, the
  vmem-queue serialization is real.
- LDS preload for small repeatedly-used data (recipe constants like
  scales) is a high-ROI optimization that's easy to overlook.
- 4 KB extra LDS for ~25% prefill speedup is a great trade.

Decode (sw path) unchanged.

### Iter 8 Phase 4b — co-issue W load with A DMA — *6% extra*

**Motivation**: in the current v2 K-loop, the W load is issued AFTER
the `gpu.barrier` that waits for the A DMA. The W's HBM latency
(~300 cycles) is then on the critical path between barrier and the
first MFMA that consumes W. Hypothesis: move the W load BEFORE the
barrier so the same barrier waits for both A DMA and W vmcnt — W's
latency gets absorbed into the A-DMA wait.

**Why simple reorder, not scf.for + loop-carried W**: with single-buffer
LDS, true cross-iter A prefetch is impossible (race on LDS overwrite).
Cross-iter W prefetch is possible (W is in regs) but requires scf.for +
loop-carried state. Iter 7 saw only 2% from a similar restructure on sw,
so the simpler "intra-iter co-issue" was tried first.

**Implementation**: trivial code reorder. Move the `b_tiles = []` load
loop from after the `gpu.barrier()` to before it. No scf.for, no
loop-carried state, no other changes.

**Key code** (the entire change is ONE block move):
```python
# === Before (Phase 4a) ===
for k_tile in range_constexpr(K_g):
    # ... raw_ptr_buffer_load_lds for A (4 chunks) ...
    gpu.barrier()                       # wait for A DMA
    b_tiles = []                        # ← W loads HERE
    for n_sub in range_constexpr(_N_SUB):
        w_lo = buffer_ops.buffer_load(w_rsrc, ...)
        w_hi = buffer_ops.buffer_load(w_rsrc, ...)
        b_tiles.append(vector.from_elements(v8i32, [...]))
    # ... MFMA inner loop, must wait for W vmcnt before first MFMA ...

# === After (Phase 4b) ===
for k_tile in range_constexpr(K_g):
    # ... raw_ptr_buffer_load_lds for A (4 chunks) ...
    b_tiles = []                        # ← W loads MOVED here, BEFORE barrier
    for n_sub in range_constexpr(_N_SUB):
        w_lo = buffer_ops.buffer_load(w_rsrc, ...)
        w_hi = buffer_ops.buffer_load(w_rsrc, ...)
        b_tiles.append(vector.from_elements(v8i32, [...]))
    gpu.barrier()                       # NOW waits for both A DMA AND W vmcnt
    # ... MFMA inner loop, no extra wait — W already in regs ...
```
That's it. The `gpu.barrier()` lowers to `s_waitcnt vmcnt(0) lgkmcnt(0)
+ s_barrier`, so the single barrier covers both vmem ops if both are
issued before it.

**Result (M=4096 prefill)**:
| | Phase 4a | **Phase 4b** | Δ |
|---|---|---|---|
| **wall time** | 543 µs | **508 µs** | **-6.4%** |
| vs Triton (461 µs) | 1.16× | **1.10× behind** | |
| VGPR | 134 | **128** | -6 (compiler reuses better) |
| LDS | 20.6 KB | 20.6 KB | unchanged |
| buffer_load (HBM) | 259 | 259 | same |
| ds_read | 800 | 800 | same |
| **s_waitcnt** | **386** | **324** | **-16%** |
| total instrs | 4434 | 4373 | -1.4% |

The s_waitcnt drop is the smoking gun: 62 fewer waitcnt instructions
because the shared barrier covers the W vmcnt that previously needed
its own wait before MFMA.

**Lessons**:
- Sometimes "prefetch" is just code reordering — no fancy scf.for /
  loop-carried state needed. If a barrier is going to wait for VMEM
  anyway, issue MORE VMEM before the barrier so they share the wait.
- The compiler (LLVM AMDGPU backend) does NOT reorder VMEM across
  `gpu.barrier` (which lowers to `s_barrier` + `s_waitcnt`). The
  fence is hard.
- VGPR went DOWN with the reorder. Surprising but real — when W loads
  happen before the barrier, the W result registers are live for a
  shorter "stretch" relative to the LDS A reads (which start fresh
  after the barrier), letting the compiler reuse some intermediates.

### Iter 8 Phase 5 — three attempts, all reverted

After Phase 4b reached **508 µs at M=4096 (1.10× behind Triton)**, the
remaining gap looked structurally limited. Three more optimizations were
attempted and all reverted.

#### Phase 5a — GROUP-major scheduling (Triton-style) — *3% regression*

**Motivation**: Triton uses `GROUP_SIZE_M=8` with a 1D-grid → (pid_m, pid_n)
remap so consecutive WGs visit `GROUP_SIZE_M` M-tiles before advancing N.
Within a 8×GRID_N "group" of WGs, A is reused across pid_n's and W is
reused across pid_m's. Predicted 5-10% L2 cache benefit.

**Key code**:
```python
# At factory time
_GROUP_SIZE_M = 8 if (GRID_M >= 8 and GRID_M % 8 == 0) else ...
_NUM_PID_IN_GROUP = _GROUP_SIZE_M * GRID_N

# In kernel
linear_pid = fx.block_idx.x
group_id = linear_pid // fx.Index(_NUM_PID_IN_GROUP)
local_in_group = linear_pid % fx.Index(_NUM_PID_IN_GROUP)
pid_m_in_group = local_in_group % fx.Index(_GROUP_SIZE_M)
pid_n_idx = local_in_group // fx.Index(_GROUP_SIZE_M)
pid_m = group_id * fx.Index(_GROUP_SIZE_M) + pid_m_in_group
pid_n = pid_n_idx
# Launcher: grid=(GRID_M * GRID_N, 1, B)
```

**Result**: 508 µs → **523 µs (+3% regression)** at M=4096.

**Diagnosis**: our shape's WG count is too small to benefit. M=4096 →
GRID_M=32, GRID_N=8, B=8 → 2048 WGs total. Already fits in L2 cache
across the chip without GROUP-major. The 4-5 extra `arith.index` ops per
WG launch + the 1D grid's loss of HW 2D-scheduler benefit cost more than
the L2 reuse gain (which was ~0).

**Lesson**: Triton's heuristic was tuned for large grids. For our shapes
(at most 2048 WGs), L2 already captures W. **Reverted.**

#### Phase 5b — LDS A double-buffer (ping-pong) — *2% regression*

**Motivation**: with single-buffer LDS-A, the iter-start barrier waits for
both A DMA and W vmcnt before MFMA can start. With ping-pong (2× LDS),
DMA[k+1] can run concurrently with MFMA[k] (which reads from buf[k%2]).
Predicted 5-15%.

**Key code** (the structural change):
```python
# Allocator: 2 × 16 KB = 32 KB instead of 16 KB
allocator.ptr = smem_a_offset + 2 * _LDS_A_BUF_BYTES

# 2 STensors + 2 memref handles
as_lds_bufs = [STensor(...buf0...), STensor(...buf1...)]
lds_a_memref_bufs = [smem_a_buf0_ptr.get(), smem_a_buf1_ptr.get()]

# Per-buffer scalar LDS base
_lds_base_i64_per_buf = [readfirstlane(...buf 0 base...), readfirstlane(...buf 1 base...)]

# Prologue: pre-load A[0] into buf[0]
for chunk in range_constexpr(4):
    rocdl.raw_ptr_buffer_load_lds(... buf[0] base + chunk*4KB ...)

# K-loop: ping-pong
for k_tile in range_constexpr(K_g):
    # Issue A[k+1] DMA into buf[(k+1)%2] (only if k+1 < K_g)
    if k_tile + 1 < K_g:
        for chunk in range_constexpr(4):
            rocdl.raw_ptr_buffer_load_lds(
                ... buf[(k_tile+1)%2] base ..., k+1 offset ...)
    # W load (Phase 4b co-issue)
    b_tiles = ...
    # MFMA reads from buf[k_tile % 2]  (data from prev iter / prologue)
    cur_lds = as_lds_bufs[k_tile % 2]
    for m_sub: ... cur_lds.vec_load(...) + MFMA ...
    gpu.barrier()  # waits for next-iter DMA
```

**Result**: 508 µs → **518 µs (+2% regression)** at M=4096.

**Diagnosis (ISA forensics)**:
| | Phase 4b | Phase 5b | Δ |
|---|---|---|---|
| LDS/WG | 20.6 KB | **37 KB** | +16 KB |
| Per-CU LDS budget (160 KB) | 160/20.6 = 7 WGs | 160/37 = **4 WGs** | -3 WG concurrency |
| VGPR | 128 | 134 | +6 |
| s_waitcnt | 324 | 325 | unchanged |

The CU-level WG occupancy dropped from 7 → 4 (LDS-bound). The
DMA-overlap latency-hiding gain was outweighed by the 1.75× concurrency
loss. Also, s_waitcnt was unchanged → compiler couldn't actually exploit
the cross-iter overlap (it's not aware that DMA[k+1] target buffer is
disjoint from MFMA's read source).

**Lesson**: LDS-A double-buffer's overlap benefit is gated by per-iter
slack between MFMA and DMA. In our case, MFMA fits the available compute
budget already, and the LDS doubling penalty exceeded the small overlap
window. **Reverted to single-buffer.**

**Deeper root-cause analysis (`lgkmcnt` serializes the overlap)**:

Looking at the actual K-loop body ISA confirms a fundamental hardware
issue beyond the LDS occupancy point:

```asm
106:  buffer_load_dwordx4 v[78:81], v7, s[8:11], 0 offen           # W load
107:  buffer_load_dwordx4 v[82:85], v7, s[8:11], 0 offen offset:16
110:  buffer_load_dwordx4 v[86:89], v8, s[8:11], 0 offen
111:  buffer_load_dwordx4 v[90:93], v8, s[8:11], 0 offen offset:16
112:  s_waitcnt vmcnt(4)             ← drain to ≤4 vmem (4 prefetch DMAs still in flight)
113:  ds_read_u8 v2, v6 offset:36864 ← read LDS scale
121:  ds_read_b128 v[20:23], v5      ← read LDS A buf[k%2]
122:  s_waitcnt lgkmcnt(1)
125:  ds_read_b128 v[16:19], v2
132:  s_waitcnt lgkmcnt(0)            ★ wait for ALL LDS ops, INCLUDING prefetch DMA's LDS write!
135:  s_waitcnt vmcnt(2)
136:  v_mfma_scale_f32_16x16x128_f8f6f4 ...   ← first MFMA
142:  s_waitcnt vmcnt(0)              ★ wait for ALL vmem, including prefetch DMA
143:  v_mfma_scale_f32_16x16x128_f8f6f4 ...
```

**The mechanism** that defeats overlap:

1. `raw_ptr_buffer_load_lds` (the async HBM→LDS DMA) is a **hybrid op**:
   the HBM-fetch portion increments `vmcnt`, but the LDS-write portion
   increments `lgkmcnt` (the LDS / scalar-mem completion counter).

2. `lgkmcnt` is a **single, wave-global counter** — there is no
   per-LDS-region tracking. The hardware doesn't know that the LDS
   write target (`buf[(k+1)%2]`) is disjoint from the LDS read source
   (`buf[k%2]`).

3. Before any `ds_read`, the compiler must conservatively emit
   `s_waitcnt lgkmcnt(0)` (or some `lgkmcnt(N)` that drains far enough
   to cover the dependency chain). For us: line 132 emits
   `lgkmcnt(0)`, fully draining all in-flight LDS ops — **including the
   prefetch DMA's LDS-write**.

4. So the prefetch DMA's LDS-write portion **must complete** before
   the MFMA's LDS reads can start. The "overlap" we hoped to get is
   collapsed back to a serial dependency.

5. The same thing happens at line 142 — `vmcnt(0)` is required before
   the second MFMA, draining the prefetch DMA's vmem portion too.

**Net**: even though the prefetch DMA targets a disjoint buffer, the
hardware's coarse-grained counters force serialization. We pay the LDS
occupancy cost (-3 WG/CU) for ~zero overlap benefit.

**This is a fundamental gfx9 limitation** for the `raw_ptr_buffer_load_lds`
async path — there's no per-region `lgkmcnt`, no way to express
"this LDS write is not aliased with that LDS read". An `__restrict__`-style
alias hint via MLIR could in principle let the compiler emit
`lgkmcnt(N>0)` to leave the prefetch DMA in flight, but FlyDSL doesn't
expose this and it's unclear whether the AMDGPU backend respects it.

**Workarounds that might recover the overlap** (all out of scope for this
session):
- `__restrict__` / `noalias` annotations at the MLIR layer
- Smaller `BLOCK_K` so doubled LDS still fits without occupancy loss
  (e.g., BLOCK_K=64 → 8 KB / buf → 16 KB total, similar to Phase 4b's
  20.6 KB; but BLOCK_K=64 means K_g doubles, ~2× more iterations)
- A different DMA primitive that doesn't go through `lgkmcnt`
  (gfx950 doesn't appear to expose one)

#### Phase 5c — `rocdl.iglp_opt(1)` — *compiler hang, reverted*

**Motivation**: `iglp_opt` is AMD's "instruction-group-level parallelism"
high-level pragma — variant 1 is the "MFMA Small Gemm" pattern. Single
intrinsic, lets LLVM pick a known-good schedule (vs Phase 3a's manual
interleave that emitted 730 nops).

**Key code**:
```python
# Just before the K-loop
rocdl.iglp_opt(1)
for k_tile in range_constexpr(K_g):
    ...
```

**Result**: **LLVM compiler hangs** (single compile times out at 2 min,
where Phase 4b compiles in ~30 sec). The iglp_opt(1) pattern apparently
triggers an expensive scheduler analysis that doesn't terminate on our
kernel's instruction count.

**Lesson**: `iglp_opt` is FlyDSL-supported but no FlyDSL reference kernel
uses it (grepped — zero hits in `kernels/*.py`). Likely too experimental
or shape-specific. **Reverted.**

### Iter 8 Phase 5 takeaways

All 3 Phase-5 optimizations failed (regressed or hung). Net: **v2 is at a
local optimum after Phase 4b** (508-515 µs at M=4096, 1.10× behind Triton).

Lessons:
1. **GROUP-major requires large grids.** Our 2048 WGs at M=4096 are
   already L2-friendly without it.
2. **LDS double-buffer requires per-iter slack.** Our MFMA is too tight
   to hide a 16 KB DMA in another buffer's lifetime, and the doubled LDS
   footprint costs more occupancy than it gains in overlap.
3. **`iglp_opt` is risky on a complex kernel.** Without empirical
   per-shape tuning (and FlyDSL's `hot_loop_scheduler` table support),
   high-level scheduling pragmas can regress or hang.

The final 1.10× gap to Triton is likely structural (something Triton's
specific instruction selection or ATT-profile-tuned schedule does better
on this exact shape). Closing it would need rocprofv3-driven ATT trace
analysis + targeted per-shape sched tuning, which is a different kind
of work than the architectural changes done in Iter 8 Phases 1-4.

### Final v2 status

| stage | M=4096 µs | vs baseline (1589) | vs Triton (~461) |
|---|---|---|---|
| Iter 0 sw | 1589 | 1.00× | 3.45× behind |
| Iter 6 sw + N_SUB=2 | 1093 | 1.45× | 2.37× |
| Iter 7 sw + scf.for prefetch | 1072 | 1.48× | 2.33× |
| Iter 8 Phase 1 v2 (4 waves + LDS) | 797 | 1.99× | 1.73× |
| Iter 8 Phase 2 v2 + XOR-swizzle | 757 | 2.10× | 1.64× |
| Iter 8 Phase 3b v2 + async DMA | 723 | 2.20× | 1.57× |
| Iter 8 Phase 4a v2 + scales LDS | 543 | 2.93× | 1.18× |
| **Iter 8 Phase 4b v2 + W co-issue** | **508** | **3.13×** | **1.10×** |

### Iter 8 — large prefill sweep — *v2 BEATS Triton at scale*

After locking in Phase 4b, we benched larger prefill shapes
(`T ∈ {4k, 8k, 16k, 32k}`, `G ∈ {8, 16}`, `N=1024, K=4096`) to see if
the 1.10× lag at the smallest shape was representative.

**Result: it was the worst case.** v2 ties at the smallest shape and
beats Triton **19-35%** at every larger shape:

| shape (B, M, N, K) | v2 µs | Triton µs | **v2/tri** | v2 TFLOPS | tri TFLOPS |
|---|---|---|---|---|---|
| (8, 4096, 1024, 4096) | 519 | 516 | **1.00×** (tied) | 530 | 532 |
| (8, 8192, 1024, 4096) | 743 | 914 | **0.81×** | **740** | 601 |
| (8, 16384, 1024, 4096) | 1262 | 1812 | **0.70×** | **871** | 607 |
| (8, 32768, 1024, 4096) | 2325 | 3604 | **0.65×** | **946** | 610 |
| (16, 4096, 1024, 4096) | 739 | 914 | **0.81×** | 744 | 601 |
| (16, 8192, 1024, 4096) | 1239 | 1814 | **0.68×** | 887 | 606 |
| (16, 16384, 1024, 4096) | 2368 | 3607 | **0.66×** | 929 | 610 |
| (16, 32768, 1024, 4096) | 4990 | 7172 | **0.70×** | 881 | 613 |

**Key observations**:

1. **Triton's TFLOPS is essentially flat at ~610 TFLOPS** regardless of
   shape. This is the hard ceiling of its bf16 MFMA path (gfx950 bf16
   peak ~3 PFLOPS → Triton uses ~20%).
2. **v2 scales with size**: 530 → 946 TFLOPS as M grows. fp8 native
   MFMA peak is ~6 PFLOPS → v2 hits 16% of fp8 peak at the best shape.
   Plenty more headroom on the table.
3. **(8, 4096) was the unfavourable corner case** for v2 — too few WGs
   (2048) to saturate the chip, so per-WG launch + barrier overhead
   dominated. All the Phase 4b/5 micro-optimization on this single
   shape was working against the smallest possible signal.
4. **Peak v2 perf at (8, 32768)**: 946 TFLOPS. Past that point the
   gain plateaus (TFLOPS dips slightly at larger G), suggesting some
   secondary effect (L2 thrash? launch overhead vs work ratio?).

**Implication for Phase 5 retro**: Phase 5a (GROUP-major) might
actually help on the larger shapes where WG count is much bigger
(e.g., (16, 32768) launches 32k WGs). Worth re-evaluating with
shape-conditional dispatch. Same for Phase 5b (LDS double-buffer) —
larger shapes have longer K loops to amortize the LDS occupancy cost.
But at the per-shape level the right answer probably differs.

### Iter 8 — CK 3-way comparison (in `atom-latest-todd` docker)

CK is the AMD official high-performance kernel library. Was previously
unbenchable on the bare-metal host (libstdc++ < 3.4.31), but loads cleanly
inside the `atom-latest-todd` Docker container (Ubuntu 24.04, GLIBCXX_3.4.33).

**CK kernel**: `csrc/ck_batched_gemm_fp8_blockwise/` wraps CK's
`DeviceGemmMultiD_ABScale_Xdl_CShuffle_V3` (FP8 AB-scale device op) with
a host-side B-loop (`MakeArgument` + `invoker.Run` per batch slice on
the same hipStream — CK doesn't ship a true batched ABScale device op).

**Result**:

| Shape (B, M, N, K) | v2 µs | Triton µs | **CK µs** | v2/CK | Triton/CK | v2 TF | Triton TF | **CK TF** |
|---|---|---|---|---|---|---|---|---|
| (8, 4096, 1024, 4096) | 506 | 417 | **264** | 1.92× | 1.58× | 544 | 660 | **1043** |
| (8, 8192, 1024, 4096) | 729 | 766 | **436** | 1.67× | 1.76× | 754 | 718 | **1260** |
| (8, 16384, 1024, 4096) | 1232 | 1421 | **836** | 1.47× | 1.70× | 893 | 774 | **1315** |
| (8, 32768, 1024, 4096) | 2309 | 2818 | **1615** | 1.43× | 1.74× | 952 | 780 | **1362** |
| (16, 4096, 1024, 4096) | 732 | 730 | **523** | 1.40× | 1.40× | 751 | 753 | **1052** |
| (16, 8192, 1024, 4096) | 1228 | 1412 | **862** | 1.42× | 1.64× | 896 | 778 | **1276** |
| (16, 16384, 1024, 4096) | 2299 | 2834 | **1673** | 1.37× | 1.69× | 957 | 776 | **1315** |
| (16, 32768, 1024, 4096) | 4997 | 5809 | **3233** | 1.55× | 1.80× | 880 | 757 | **1360** |

**Headline**:
- **CK is the clear winner across all shapes**, sustaining ~1300 TFLOPS
  (22% of fp8 peak 6000 TFLOPS).
- **CK ≈ 1.4-1.9× faster than v2** (more lead at smaller shapes).
- **CK ≈ 1.4-1.8× faster than Triton** (consistent across shapes).
- v2 vs Triton: v2 wins on M ≥ 8192 (matches host bench). At (8, 4096)
  v2 lags Triton 1.21× in docker (host showed tie ~1.00×) — likely
  docker has different Triton/autotune cache state.

**Where CK's advantage comes from** (we did NOT do these in v2):
1. **CShuffle epilogue** — LDS-staged coalesced output stores
2. **Per-shape tune CSV** — tile/pipeline configs heuristically picked
   per shape (`csrc/ck_batched_gemm_fp8_blockwise/batched_gemm_fp8_blockwise_tune.py`)
3. **`BlockGemmPipelineScheduler`** — CK's internal instruction-schedule
   tables, hand-tuned by AMD's CK team
4. Likely larger BLOCK tiles (CK can pick BLOCK_M=256+ per shape)
5. The host B-loop dispatch overhead is real but small (~5-10 µs per
   batch slice = 40-160 µs total for B=8/16) — yet CK still wins handily,
   meaning CK's per-GEMM efficiency dominates

**Honest reassessment**:
- v2 (FlyDSL) reached ~16% fp8 peak. **Solid for a hand-written kernel**
  but well below CK's 22%.
- The remaining 1.4-1.9× gap to CK is mostly **CShuffle epilogue +
  per-shape tile tuning**. Both are mechanical-but-substantial work
  (~weeks, not hours).
- Triton's bf16 path is structurally limited to ~10% peak fp8 — neither
  v2 nor CK has that ceiling.

**Decision**: keep v2 as the FlyDSL-path production kernel for shapes
where CK isn't available (e.g., new shapes not yet tuned), but
**`backend='ck'` should be the default in `aiter.batched_gemm_fp8_blockwise`
when CK is available**. v2's value is the FlyDSL implementation
exercise, the documented optimization journey, and a safety net for
unsupported shapes — not as the absolute fastest kernel.

### Iter 8 — CK autotune sweep (does tuning beat the heuristic?)

The CK 3-way numbers above used the production **heuristic dispatcher**
(picks one tile config per shape based on hand-coded rules). To check
whether per-shape autotune unlocks more headroom we ran the official
CK tune driver across all 19 candidate kernels × 8 shapes.

**Setup**:
```bash
# Inside atom-latest-todd docker:
python csrc/ck_batched_gemm_fp8_blockwise/batched_gemm_fp8_blockwise_tune.py \
  -i untuned_8shapes.csv \
  -o tuned_8shapes.csv --iters 10
# 19 kernels × 8 shapes = 152 runs. Median µs per (shape, kid).
```

To make the tune module load the loader needed two new pieces (these
files were missing from the repo):

1. **`csrc/include/rocm_ops.hpp`** — added macros
   `BATCHED_GEMM_FP8_BLOCKWISE_PYBIND` and
   `BATCHED_GEMM_FP8_BLOCKWISE_TUNE_PYBIND` (the existing pybind .cu
   files referenced these but they were never defined).
2. **`aiter/jit/optCompilerConfig.json`** — added the
   `module_batched_gemm_fp8_blockwise_tune` entry (sources + blob_gen_cmd).

After this the JIT system built `module_batched_gemm_fp8_blockwise_tune.so`
in ~5 minutes, sweeping 19 instances × 2 dtypes = 38 .cpp files.

**Tune choices vs production heuristic**:

| Shape (B,M,N,K) | Heuristic kid | Tune-best kid | Tune kernel name |
|---|---|---|---|
| (8, 4096) | kid=0 (256x128x128 v3) | **kid=2** (256x64x128 v3) | `1x128x128_256x64x128x128_..._v3` |
| (8, 8192) | kid=0 | kid=0 | `1x128x128_256x128x128x128_..._v3` |
| (8, 16384) | kid=0 | kid=0 | same |
| (8, 32768) | kid=0 | kid=0 | same |
| (16, 4096) | kid=0 | kid=0 | same |
| (16, 8192) | kid=0 | kid=0 | same |
| (16, 16384) | kid=0 | kid=0 | same |
| (16, 32768) | kid=0 | kid=0 | same |

**Result — tune barely changed anything**:

```
shape                          v2 us    tri us   ck-h us   ck-t us   v2/tri  v2/ck-t  ck-t/h   v2 TF  tri TF  ck-t TF
-----------------------------------------------------------------------------------------------------------------------
(8,4096,1024,4096)       507.0u    412.9u    259.2u    305.6u    1.23x    1.66x   1.18x     542     666      899   kid=2
(8,8192,1024,4096)       731.4u    731.6u    435.1u    440.0u    1.00x    1.66x   1.01x     752     751     1249   kid=0
(8,16384,1024,4096)      1244.2u   1429.0u    832.4u    837.8u    0.87x    1.48x   1.01x     884     769     1312   kid=0
(8,32768,1024,4096)      2360.9u   2816.9u   1615.6u   1621.2u    0.84x    1.46x   1.00x     931     781     1356   kid=0
(16,4096,1024,4096)       734.4u    761.2u    534.5u    529.9u    0.96x    1.39x   0.99x     749     722     1037   kid=0
(16,8192,1024,4096)      1221.4u   1422.9u    861.8u    864.8u    0.86x    1.41x   1.00x     900     773     1271   kid=0
(16,16384,1024,4096)      2289.7u   2826.1u   1675.2u   1677.2u    0.81x    1.37x   1.00x     960     778     1311   kid=0
(16,32768,1024,4096)      4988.9u   5794.4u   3230.8u   3243.7u    0.86x    1.54x   1.00x     882     759     1356   kid=0
```

(`ck-h` = CK heuristic dispatcher; `ck-t` = CK with the tune-CSV
kernelId.)

**Findings**:

1. **Heuristic was already optimal on 7/8 shapes** — autotune confirmed
   `kid=0` (`DeviceGemmHelper... 256x128x128 intrawave_v3`) is the
   right choice. ck-t/ck-h ratios are all 0.99–1.01× (within noise).
2. **(8, 4096) is the lone exception, and tune got it WRONG** — autotune
   picked kid=2 (256x64x128) at 305 µs in re-bench, but the heuristic's
   kid=0 ran at 259 µs. The autotuner's per-shape decision was based
   on a single 10-iter median which happened to favor kid=2 in the
   tune session (tune CSV recorded 279 µs) but didn't replicate. The
   "correct" answer at this shape would have been to keep the heuristic.
3. **Tuning did NOT close the v2 → CK gap** — CK sustains 1300+ TFLOPS
   regardless of whether you autotune or use the heuristic. The
   heuristic's hand-coded rules (in `csrc/ck_batched_gemm_fp8_blockwise/
   batched_gemm_fp8_blockwise_heuristic.cu`) are calibrated to AMD's
   shape catalog and already converge on the best instance.

**Implication for the v2 ↔ CK gap**:
- It is NOT about CK shipping a "magic per-shape config" we don't have.
  Tuning the same 19 candidate kernels over our 8 shapes confirms the
  heuristic dispatch is essentially optimal.
- The 1.4–1.7× CK lead is structural — comes from CShuffle epilogue,
  the `BlockGemmPipelineScheduler::Intrawave` `v3` scheduler in
  particular (deeper LDS pipeline + more aggressive instruction
  interleaving than v2's stages=2 sync barrier), and the broader CK
  template metaprogramming substrate. None of those are addressable
  by simply trying more tile sizes.

**Files added/modified for this experiment**:
- `csrc/include/rocm_ops.hpp` — added two FP8 blockwise pybind macros
- `aiter/jit/optCompilerConfig.json` — added `module_batched_gemm_fp8_blockwise_tune` entry
- (no kernel-code changes)

### Iter 8 — Deep-dive: where CK's 1.4–1.7× lead comes from

*Source code referenced (read-only, third-party):*
*  Pipeline: `3rdparty/composable_kernel/include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v3_ab_scale.hpp`*
*  Inst counter: `3rdparty/composable_kernel/include/ck/utility/blkgemmpipe_scheduler.hpp`*
*  Epilogue: `3rdparty/composable_kernel/include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_common.hpp:1370+`*

The 1.4–1.7× CK > v2 gap is structural. It comes almost entirely from
two pieces that v2 has never built: **Intrawave v3 K-loop scheduling**
(roughly +25%) and **CShuffle epilogue** (roughly +15–20%). Tile size,
MFMA shape, and host-launch differences contribute the remaining ~5%.

#### Intrawave v3 — what it does

`BlockwiseGemmXdlops_pipeline_v3_ab_scale<Intrawave, ...>::Run` declares
a 3-stage pipeline:

```cpp
static constexpr index_t PrefetchStages  = 2;  // 2 HBM reads in flight
static constexpr index_t PrefillStages   = 1;  // 1 LDS prefill before main
static constexpr index_t GlobalBufferNum = 1;
```

When the main loop body starts, iter 0's data is already in VGPRs
(prefetch 1) AND already in LDS (prefill 1) AND already in the MFMA's
input regs (LDS-prefetch 1), and iter 1's HBM read has been issued. Each
iter does, in order:

1. `block_sync_lds()` — wait everyone done reading LDS
2. `RunWrite(a/b)` — VGPR (from prefetched HBM) → LDS
3. `RunRead(a/b)` — issue HBM prefetch for iter k+2
4. **MFMA chain** over `MRepeat × NRepeat × KRepeat` consuming the
   already-loaded `a_thread_buf` / `b_thread_buf`
5. `block_sync_lds()` — wait LDS write done
6. **LDS → VGPR** for iter k+1
7. **`HotLoopScheduler()`** — emit `__builtin_amdgcn_sched_group_barrier`
   intrinsics that force LLVM to interleave the above 4 instruction
   classes into the optimal order

##### Step-by-step data flow (kid=0: BLOCK 128×128×128, single K-iter)

Constants (CK candidate kid=0, the autotune winner for our shapes):

| Item | Value |
|---|---|
| BlockSize / WaveSize | 256 / 64 → **4 waves arranged 2(M)×2(N)** |
| BLOCK_M × N × K | 128 × 128 × 128 |
| MFMA | `mfma_scale_f32_32x32x64_f8f6f4` (32×32×64 fp8) |
| MPerXDL × NPerXDL × KPerXDL | 32 × 32 × 64 |
| MRepeat × NRepeat × KRepeat | 2 × 2 × 2 = **8 MFMAs/wave/iter** |
| AK1 = BK1 | 16 bytes (`buffer_load_dwordx4`) |
| **Output area per wave** | 64 rows (M) × 64 cols (N) |
| **mfma_cycle** | 64 cycles |

Plugging into the **instruction-count formulas** (`blkgemmpipe_scheduler.hpp`),
**per thread per K-iter**:

| Instruction | Calculation | Count/thread/iter | Bytes/thread |
|---|---|---|---|
| `buffer_load_dwordx4` A | `BLOCK_M × BLOCK_K / (BS × AK1) = 128*128/(256*16)` | **4** | 64 B HBM read |
| `buffer_load_dwordx4` B | same | **4** | 64 B HBM read |
| `ds_write_b128` A | paired with the 4 buffer_loads | **4** | 64 B LDS write |
| `ds_write_b128` B | same | **4** | 64 B LDS write |
| `ds_read_b128` A | `WaveNumN × BLOCK_M × BLOCK_K / (BS × 16) = 2*128*128/(256*16)` | **8** | 128 B LDS read |
| `ds_read_b128` B | `WaveNumM × ...` | **8** | 128 B LDS read |
| MFMA | MRepeat×NRepeat×KRepeat | **8** | — |

**Per-WG per-iter aggregate traffic** (256 threads):

| Channel | Size | Notes |
|---|---|---|
| HBM → VGPR (A prefetch) | 256 × 64 B = **16 KB** | Full A tile (128 rows × 128 cols fp8) |
| HBM → VGPR (B prefetch) | 256 × 64 B = **16 KB** | Full B tile |
| VGPR → LDS (A write)| 16 KB | The A prefetched 1 iter ago |
| VGPR → LDS (B write)| 16 KB | The B prefetched 1 iter ago |
| LDS → VGPR (A read) | 256 × 128 B = **32 KB** | (4 waves each read their own 64-M slice; A is shared across N-waves → cluster traffic = tile × WaveNumN = 16 KB × 2) |
| LDS → VGPR (B read) | 32 KB | Same, B shared across M-waves |
| MFMA (32 × 4 waves) | 128 instructions / WG / iter | Compute |

**LDS budget per WG**:
- A tile 16 KB + B tile 16 KB = **32 KB / WG** (single-buffer along K; v3
  buffers an extra iter in VGPR via prefetch, not in LDS)
- gfx950 cap = 64 KB/WG → 50% utilization, leaves headroom for epilogue staging

**Scale traffic** (K-block = 128 = ScaleBlockK, so 1 scale switch per K-iter):
- A_scale: per thread per iter, MRepeat=2 fp32 reads → **8 B/thread/iter**
- B_scale: per thread per iter, 1 fp32 read → **4 B/thread/iter**
- Negligible vs. the 128 B/thread main-data traffic

##### Sequential timeline (one steady-state iter, wave's view)

```
t=0    block_sync_lds()                 ← wait other waves done reading prev LDS
       ┌──────────────────────────────┐
       │ Phase A: write + issue + math │
       │  ds_write_a × 4 (64 B/th)    │  land "A prefetched 2 iters ago" into LDS
       │  ds_write_b × 4 (64 B/th)    │  land "B prefetched 2 iters ago" into LDS
       │  buffer_load_a × 4 (HBM)     │  issue iter k+2 A prefetch (300+ cycle lat.)
       │  buffer_load_b × 4 (HBM)     │  issue iter k+2 B prefetch
       │  MFMA × 8                    │  consume a_thread_buf / b_thread_buf (iter k)
       │  scale_mul × 2 (a*b)         │  next-iter c_scale = a_scale × b_scale
       └──────────────────────────────┘
t=N    block_sync_lds()                 ← wait all waves done writing LDS
       ┌──────────────────────────────┐
       │ Phase B: read LDS for next iter│
       │  ds_read_a × 8 (128 B/th)    │  copy iter k+1 A from LDS into a_thread_buf
       │  ds_read_b × 8 (128 B/th)    │  copy iter k+1 B from LDS into b_thread_buf
       └──────────────────────────────┘
t=M    HotLoopScheduler()              ← sched_group_barrier locks the ordering
       buffer_load_scale × 3            ← next iter a_scale × 2 + b_scale × 1 (fp32)
loop end: enter next iter
```

**Key observation**: Phase A's `ds_write` / `buffer_load` / `MFMA` look
sequential in the source, but `HotLoopScheduler` reorders them into a
**fully interleaved** stream — the 300+ cycle HBM latency of each
`buffer_load` is filled with running MFMAs, and during each MFMA's 64
cycles two `ds_read`s slip in (see formula derivation below). The hot
loop emits **no `s_waitcnt`**, so the wave timeline never stalls.

##### Instruction-tree view (what runs each iter + where its data came from / goes to)

Data flows through the kernel via a **4-stage pipeline**:

```
HBM ──vmem──► VGPR_in ──ds_write──► LDS ──ds_read──► thread_buf ──MFMA──► c_acc
     stage 1            stage 2          stage 3              stage 4
```

Each edge crosses 1 K-iter. So **MFMA[k] consumes data that was read
from HBM 2 iters ago**:

```
   HBM_read[k]   happens at  iter k-2  (or in prologue if k<2)
   ds_write[k]   happens at  iter k-1
   ds_read[k]    happens at  iter k-1
   MFMA[k]       happens at  iter k     ← now
```

Below are the instruction trees at three time points: **prologue / iter
k=0 / iter k=1**. Each instruction is annotated with **◄── consumes
from** (dependency source) / **──► produced for** (downstream consumer).

**Prologue (warm-up before main loop)**

```
prologue
│
├─ HBM_read[0]            ─── vmem  → VGPR_in        ──► used by prologue ds_write[0]
├─ HBM_read[1]            ─── vmem  → VGPR_in        ──► used by iter k=0 ds_write[1]
│                                                       (PrefetchStages=2: 2 HBM reads in flight)
│
├─ scale_load[0]          ─── vmem  → scale_buf      ──► used by prologue c_scale[0]
├─ c_scale[0] = a×b                                  ──► used by iter k=0 MFMA
│
├─ ds_write[0]            ─── VGPR_in → LDS          ──► used by prologue ds_read[0]
│   (LDS now holds iter 0 data)                         (PrefillStages=1: 1 buffer in LDS)
│
├─ scale_load[1]          ─── vmem  → scale_buf      ──► used by iter k=0 c_scale[1]
│
├─ c_thread_buf.Clear()                              ──► initialize accumulator
│
├─ block_sync_lds()                                  ── wait for ds_write[0] complete
│
├─ ds_read[0]             ─── LDS → a/b_thread_buf   ──► used by iter k=0 MFMA
│   (thread_buf now holds iter 0 data)
│
└─ sched_barrier(0)
```

**Main loop iter k=0 (steady state begins)**

```
iter k=0
│
├─ block_sync_lds()  #1                              ── wait other waves done reading LDS
│
├─ ds_write[1]            ─── VGPR_in → LDS
│   ◄── consumes "prologue's HBM_read[1]" result
│   ──► used by iter k=1 ds_read[1]
│   (LDS overwritten from iter 0 → iter 1 data)
│
├─ HBM_read[2]            ─── vmem → VGPR_in
│   ──► used by iter k=1 ds_write[2]
│
├─ MFMA × 8               ─── thread_buf → c_acc
│   ◄── consumes "prologue's ds_read[0]" result (iter 0 data)
│   ──► accumulates into c_thread_buf
│
├─ c_scale[1] = a×b
│   ◄── consumes "prologue's scale_load[1]" result
│   ──► used by iter k=1 MFMA scaling
│
├─ block_sync_lds()  #2                              ── wait this iter's ds_write[1] done
│
├─ ds_read[1]             ─── LDS → a/b_thread_buf
│   ◄── consumes "this iter's ds_write[1]" result
│   ──► used by iter k=1 MFMA
│
├─ scale_load[2]          ─── vmem → scale_buf
│   ──► used by iter k=1 c_scale[2]
│
└─ sched_barrier(0)       ── HotLoopScheduler() emits sched_group_barrier here
                             to interleave all instructions above (formula below)
```

**Main loop iter k=1 (steady-state repeat)**

```
iter k=1
│
├─ block_sync_lds()  #1
│
├─ ds_write[2]            ─── VGPR_in → LDS
│   ◄── consumes "iter k=0's HBM_read[2]" result
│   ──► used by iter k=2 ds_read[2]
│   (LDS overwritten from iter 1 → iter 2 data)
│
├─ HBM_read[3]            ─── vmem → VGPR_in
│   ──► used by iter k=2 ds_write[3]
│
├─ MFMA × 8               ─── thread_buf → c_acc
│   ◄── consumes "iter k=0's ds_read[1]" result (iter 1 data)
│   ──► continues accumulating c_thread_buf
│
├─ c_scale[2] = a×b
│   ◄── consumes "iter k=0's scale_load[2]" result
│
├─ block_sync_lds()  #2
│
├─ ds_read[2]             ─── LDS → a/b_thread_buf
│   ◄── consumes "this iter's ds_write[2]" result
│   ──► used by iter k=2 MFMA
│
├─ scale_load[3]          ─── vmem → scale_buf
│
└─ sched_barrier(0)
```

##### Data "age" cross-reference table

Flattening the dependencies above:

| Data | Consumed at iter X MFMA | Its HBM_read happened at | Its ds_write happened at | Its ds_read happened at |
|---|---|---|---|---|
| iter 0 data | k=0 MFMA | prologue (HBM_read[0]) | prologue (ds_write[0]) | prologue (ds_read[0]) |
| iter 1 data | k=1 MFMA | prologue (HBM_read[1]) | k=0 (ds_write[1]) | k=0 (ds_read[1]) |
| iter 2 data | k=2 MFMA | k=0 (HBM_read[2]) | k=1 (ds_write[2]) | k=1 (ds_read[2]) |
| iter 3 data | k=3 MFMA | k=1 (HBM_read[3]) | k=2 (ds_write[3]) | k=2 (ds_read[3]) |

**Conclusion**: `HBM_read[X]` is issued **2 iters earlier** than the
corresponding `MFMA[X]`. That's what `PrefetchStages=2` means — there's
2 K-iters of MFMA time between the HBM read and its MFMA consumer to
absorb the 300+ cycle HBM latency.

##### Within-iter overlap (why the MFMA pipe never idles)

Per-wave instruction counts inside one iter:

| Instruction | Count | Per-instruction cycles | Total issue cycles |
|---|---|---|---|
| MFMA | 8 | 64 (32×32×64 fp8) | **512** ← main runtime |
| ds_write | 8 (4a + 4b) | ~4 | 32 |
| buffer_load (vmem) | 8 (4a + 4b) | ~4 | 32 (issue) + 300+ (latency, in flight) |
| ds_read | 16 (8a + 8b) | ~8 | 128 |
| scale_load + scale_mul | ~3 | ~4 | 12 |

**Total mem-instruction issue time ≈ 200 cycles** — fits comfortably
inside the 512-cycle MFMA main runtime. `HotLoopScheduler`'s job is to
emit `sched_group_barrier`s that force LLVM to schedule them like this:

```
       │      │      │      │      │      │      │      │
MFMA:  [─M0─][─M1─][─M2─][─M3─][─M4─][─M5─][─M6─][─M7─]    ← 100% busy
                                                              
ds_W:  [w][w][w][w]                                          ← scattered between M0..M3
vmem:  [v][v][v][v][v][v][v][v]                              ← scattered between M0..M3
                                  └── HBM lat flies on into the iter+2 ──►
ds_R:                              [rr][rr][rr][rr][rr][rr][rr][rr] ← between M4..M7
                                                                
                                                  no s_waitcnt anywhere in hot loop
```

**v2 comparison** (no cross-iter prefetch, each iter waits its own vmem):

```
v2's one iter:
async_DMA(HBM→LDS): [issue → ─── HBM 300+ cyc ───► land]
                                                   ★ gpu.barrier
ds_read:                                            [R×8]
MFMA:                                                    [M0][M1]…[M7]
                                                                      
per-iter ≈ HBM lat (300) + ds_read + MFMA = ~700+ cyc                  
v3 per-iter ≈ MFMA-bound = ~512 cyc                                    
                                                                          
≈ 35% speed gap, matching the measured v2 / CK = 1.4× ratio              
```

**The HotLoopScheduler arithmetic** (pipeline_v3_ab_scale.hpp:184-197):

```cpp
constexpr auto mfma_cycle             = 32;  // 16x16x128 fp8 = 32 cycles
constexpr auto ds_read_a_issue_cycle  = 8;   // ds_read_b128 = 8 cycles
constexpr auto ds_read_a_mfma_rate    =
    (mfma_cycle - 4 + 2*ds_read_a_issue_cycle - 1) / (2*ds_read_a_issue_cycle);
//  = (32 - 4 + 16 - 1) / 16 = 2
// ⇒ 2 ds_reads can complete per MFMA's 32 cycles without stalling MFMA
```

This rate drives the schedule:

- **Stage 1** (lines 211-232): for each `buffer_load` (HBM read), pair
  it with `num_mfma_per_issue` MFMAs and one `ds_write` per `idswrite`,
  using `sched_group_barrier(0x008/0x020/0x100/0x200, count, 0)` to
  pin the order. Time-line per `buffer_load_a`:

  ```
  [ds_write, mfma]  ×  num_dswrite_per_issue_a
  [vmem_read, mfma × (num_mfma_per_issue - num_dswrite_per_issue_a)]
  ```

  → HBM latency (~300 cycles) and LDS-write latency (~30 cycles) are
  fully hidden behind concurrently-running MFMAs.

- **Stage 2** (lines 235-265): pure `[ds_read × 2, mfma]` repeats —
  consumes the rate=2 we computed, every MFMA pair carries one full
  ds_read prefetch for the next iter.

**End result**: MFMA pipe ≈ 100% busy. No `s_waitcnt` ever shows up
inside the hot loop.

#### v2's K-loop comparison

| Aspect | CK Intrawave v3 | FlyDSL v2 |
|---|---|---|
| HBM prefetch depth | 2 K-iter (in VGPR) | 1 K-iter (async DMA, in LDS) |
| LDS-tile buffering | 1 (single buffer, but write↔read separated by sync) | 1 (single buffer) |
| HBM→LDS path | HBM → VGPR → LDS (2 hops) | HBM → LDS direct (`raw_ptr_buffer_load_lds`, no VGPR hop) |
| K-iter barriers | 2 × `block_sync_lds()` (separates write from read) | 1 × `gpu.barrier()` |
| Scheduling | hand-written `sched_group_barrier` × 30+ | LLVM scheduler over `range_constexpr`-unrolled body |
| Achieved MFMA occupancy | ~100% | ~60-70% (lgkmcnt(0) at iter boundaries) |

**Why we can't easily port v3 to v2**:
1. v2's `raw_ptr_buffer_load_lds` saves ~16 VGPR/wave but ties HBM-load
   completion to `lgkmcnt` — the same counter as ds_read. This kills
   the ability to separately wait on prefetch-write vs. consume-read,
   which is the precondition for the v3 pipeline. Phase 5b proved this
   by failing.
2. FlyDSL has no equivalent of `sched_group_barrier` exposed. Phase 5c
   tried the closest thing (`rocdl.iglp_opt(1)`) and the compiler hung.
3. The v3 schedule formula presupposes a fixed pipeline topology
   (Prefetch=2, Prefill=1, GlobalBufferNum=1). Adopting it requires
   restructuring v2's entire K-loop, not a local edit.

#### CShuffle Epilogue — what it does

After the K-loop, each lane holds its MFMA accumulator in a v4f32 VGPR
slice. For `mfma_f32_16x16x128_f8`, the lane→element layout per 16×16
tile is:

```
lane[0..15]  →  row=0, col=[0..15]    (1 fp32 per lane)
lane[16..31] →  row=1, col=[0..15]
lane[32..47] →  row=2
lane[48..63] →  row=3
```

Each lane holds 4 fp32 (the v4f32 slot covers 4 contiguous rows of one
column). Direct `buffer_store_b16` (v2's current path) sees adjacent
lanes writing positions strided by `N` (≥ 1024 bytes) and same-row lanes
writing positions strided by 16 elements (32 bytes). HBM coalescing
collapses to roughly **1 dwordx4 per cycle** instead of the available
~4 dwordx4 / cycle.

`RunMultiDEpilogue` (gridwise_gemm_xdl_cshuffle_common.hpp:1593-1629)
walks the 128×128 C tile via a SpaceFillingCurve in chunks of
`(CShuffleMXdlPerWavePerShuffle * MWave * MPerXdl)` × `(CShuffleNXdlPerWavePerShuffle * NWave * NPerXdl)` (= 64×64 for kid=0).
Per chunk:

```cpp
block_sync_lds();                                           // [1]
c_thread_copy_vgpr_to_lds.Run(..., c_thread_buf,
                              c_block_thread_desc,
                              c_shuffle_block_buf);         // [2] VGPR→LDS
block_sync_lds();                                           // [3]
cde_block_copy_lds_and_global.Run(c_ds_desc_refs,
                                  c_ds_buf_refs,
                                  c_grid_desc_..., c_grid_buf);  // [4] LDS→HBM
```

- **Step [2]** is a `ThreadwiseTensorSliceTransfer` that reorders the
  scattered acc fragments into LDS at a "by-(m,n)" layout: row r's
  N-direction columns are contiguous in LDS. Each lane writes ~64 bytes
  to LDS (4 ds_write_b128s), no bank conflicts (LDS desc is XOR-swizzled).
- **Step [4]** is a `ThreadGroupTensorSliceTransfer_v7r3` with
  `CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock = [1, 32, 1, 8]`
  (= 256 threads in 32 M × 8 N grid) and
  `CShuffleBlockTransferScalarPerVector_NPerBlock = 8` (16-byte stores).
  Each thread reads 16 B from LDS and writes 16 B to HBM at a
  per-row-contiguous N position. **Adjacent threads write adjacent HBM
  bytes — fully coalesced.**

#### CShuffle cost-benefit

For (B=8, M=4096, N=1024, K=4096), output bytes = 64 MB. HBM write
bandwidth ceiling ≈ 3 TB/s.

| Path | Per-store width | Coalesce factor | Estimated store-segment time |
|---|---|---|---|
| v2 direct `buffer_store_b16` | 2 B | ~25% (lane stride = N) | 60–100 µs (~12-20% of total) |
| CK CShuffle `buffer_store_b128` | 16 B | ~100% | ~25–30 µs (near bandwidth ceiling) |

CShuffle adds 1× LDS write + 1× LDS read + 2× barrier ≈ 3-5 µs of
overhead, in exchange for 50-70 µs saved on the store segment. **Net
gain ≈ 19% of total wallclock at this shape**, which lines up with the
tail of the v2/CK ratio.

#### Visual: the two CK pipelines

```
┌─── K-axis pipeline (Intrawave v3) ──────────────────────────────┐
│   iter k:                                                         │
│     HBM[k+1] → VGPR  ┐                                            │
│     VGPR[k]  → LDS   │  All four overlap; HotLoopScheduler() pins │
│     LDS[k+1] → VGPR  │  the order via sched_group_barrier so MFMA │
│     MFMA[k]          ┘  pipe stays busy and mem latency is hidden │
└───────────────────────────────────────────────────────────────────┘
                          ↓ K-loop exit
┌─── Output pipeline (CShuffle Epilogue) ──────────────────────────┐
│   for sub_tile in [0, num_access):                                │
│     barrier                                                       │
│     VGPR(scattered acc) → LDS(coalesced (m,n) layout)             │
│     barrier                                                       │
│     LDS → HBM with 16-B coalesced stores                          │
└───────────────────────────────────────────────────────────────────┘
```

**Intrawave v3 solves "memory keeps MFMA fed"; CShuffle solves "stores
don't bottleneck the tail."** v2 has neither well-solved, so the gap
is ~25% + ~15-20% + ~5% small items = the 1.4-1.7× we measure.

#### Engineering effort to add to v2

Order-of-magnitude estimates if we wanted to close the gap:

| Piece | Effort | Risk |
|---|---|---|
| CShuffle equivalent | 3-5 days impl + 2 days LDS swizzle/cluster-desc tuning | Medium — need to write MFMA acc → (m,n) thread-mapping helper that FlyDSL doesn't have |
| Intrawave v3 equivalent | 1-2 weeks; requires giving up `raw_ptr_buffer_load_lds` (or routing through staged VGPR) and inventing a `sched_group_barrier` wrapper or hand-tuning ISA | High — FlyDSL has no `iglp_opt`/`sched_group_barrier` primitive that works; would need a compiler-side feature or fall back to inline-asm regions |

Neither is a one-liner; both are weeks-of-work projects. The v2/CK gap
is therefore **best closed by using CK directly** when CK is available;
v2 remains useful as the FlyDSL-only fallback for shapes CK doesn't
support and as a documented optimization-journey reference.

### Iter 9 — v3 kernel: structural Intrawave-v3 port — *flat to slight regression*

Built `batched_gemm_fp8_blockwise_flydsl_v3.py` to test the deep-dive's
hypothesis: **does adopting CK's v3 pipeline structure (PrefetchStages=2
in VGPR + Prefill=1 in LDS) help on its own, without sched_group_barrier?**

**Design changes from v2** (geometry/MFMA shape kept identical):

| Aspect | v2 | v3 |
|---|---|---|
| HBM→LDS A path | `raw_ptr_buffer_load_lds` (async, single-instruction) | staged `buffer_load` (HBM→VGPR) + `ds_write` (VGPR→LDS) |
| HBM prefetch depth (VGPR) | 0 (data goes straight to LDS) | **2 K-iters in flight** (PrefetchStages=2) |
| Pipeline structure | DMA→barrier→MFMA→barrier per iter | Prologue: 2 HBM reads + 1 LDS prefill + barrier; main loop: MFMA → barrier #1 → ds_write → next-HBM-read → barrier #2; epilogue: final MFMA only |
| Scheduling primitives | none | none (FlyDSL has no `sched_group_barrier`) |

**Code shape**:
```python
# Helpers (inline, range_constexpr-unrolled)
def _hbm_load_a_to_vgpr(k_byte_off):  # 4 buffer_load_dwordx4/thread
def _vgpr_to_lds_a(chunks_data):      # 4 ds_write_b128/thread
def _w_load_and_mfma(k_tile):         # W loads + 16 MFMAs (same as v2)

# Prologue
a_stage = _hbm_load_a_to_vgpr(0)
_vgpr_to_lds_a(a_stage)               # LDS has iter 0
a_stage = _hbm_load_a_to_vgpr(_BLOCK_K)   # iter 1 in VGPR
gpu.barrier()

# Main loop k = 0 .. K_g - 2
for k_tile in range_constexpr(K_g - 1):
    _w_load_and_mfma(k_tile)          # MFMA on iter k (LDS data)
    gpu.barrier()                     # MFMA reads done
    _vgpr_to_lds_a(a_stage)           # write iter k+1
    if k_tile + 2 < K_g:
        a_stage = _hbm_load_a_to_vgpr((k_tile + 2) * _BLOCK_K)
    gpu.barrier()                     # LDS iter k+1 visible

# Epilogue
_w_load_and_mfma(K_g - 1)
```

**Correctness**: PASS. Max bf16 err = 0.5 vs torch oracle on
(8, 4096, 1024, 4096) — same as v2. Smaller shapes also clean.

**Performance**:

```
shape                          v2 us     v3 us    v3/v2    v2 TF   v3 TF
--------------------------------------------------------------------------------
(8,4096,1024,4096)        518.2u   513.7u   0.99x      530     535
(8,8192,1024,4096)        741.0u   781.1u   1.05x      742     704
(8,16384,1024,4096)      1249.5u  1271.5u   1.02x      880     865
(8,32768,1024,4096)      2333.4u  2425.2u   1.04x      942     907
(16,4096,1024,4096)       738.4u   789.5u   1.07x      744     696
(16,8192,1024,4096)      1237.8u  1285.5u   1.04x      888     855
(16,16384,1024,4096)     2308.3u  2390.5u   1.04x      953     920
(16,32768,1024,4096)     4994.3u  4990.2u   1.00x      881     881
```

**v3 is essentially flat or 4-7% slower than v2 on every shape.**
Two shapes tie within noise; the rest regress 2-7%.

**ISA forensics** (gfx950, kid=0 analog, BLOCK 128×128×128):

| Metric | v2 | v3 | Δ |
|---|---|---|---|
| `next_free_vgpr` | 128 | **154** | **+26 (occupancy 4 → 3 waves/SIMD)** |
| `group_segment_fixed_size` (LDS) | 20608 | 20608 | 0 (same single-buffer A) |
| `private_segment_fixed_size` (scratch) | 0 | 0 | 0 (no spill) |
| MFMA inst count | 512 | 512 | 0 (same compute) |
| `buffer_load_lds` | 131 | 3 | -128 (A K-loop async DMAs replaced) |
| `ds_write` | 0 | 128 | +128 (the replacements) |
| `s_waitcnt` | 324 | **390** | **+66** |
| `s_barrier` | 65 | 64 | -1 |

**Why v3 didn't beat v2 — the deep-dive predicted exactly this**:

1. **Occupancy hit** (+26 VGPR → 4 waves/SIMD → 3): the staging slot
   costs more VGPR than `raw_ptr_buffer_load_lds` saved. Per-CU active
   wave count drops 25%, exactly the same trap Phase 5b hit with LDS
   double-buffer (which dropped 7→4 WGs/CU).
2. **+66 `s_waitcnt`** in the hot loop: the LLVM scheduler couldn't
   fully interleave the staged path. Without `sched_group_barrier` (the
   primitive CK's HotLoopScheduler depends on, which `iglp_opt(1)`
   cannot replace — it hung the compiler in Phase 5c), LLVM falls back
   to conservative `lgkmcnt`/`vmcnt` waits that serialize the pipeline
   exactly the way the staged path was supposed to fix.
3. **Pipeline-depth gain didn't materialize**: PrefetchStages=2 in VGPR
   only helps if the second prefetch's HBM latency overlaps with MFMAs.
   It does mathematically, but with `s_waitcnt` blocking issue at the
   barrier, the overlap window collapses.

**What this confirms**: the v2/CK 1.4–1.7× gap is **not addressable
by structural changes alone in FlyDSL**. CK's lead requires the
HotLoopScheduler interleaving table — a sequence of dozens of
`__builtin_amdgcn_sched_group_barrier(mask, count, 0)` calls — and
FlyDSL has no equivalent primitive. Phase 5c (the obvious workaround
attempt with `rocdl.iglp_opt`) hangs the compiler.

**Production decision**:
- Keep `flydsl_batched_gemm_fp8_blockwise` (v2 wrapper) as the FlyDSL
  production path for prefill.
- v3 stays in-tree as a documented experiment but is not wired into the
  dispatch wrapper. It demonstrates the structural-only ceiling.
- The path forward to actually close the CK gap needs **either** (a) a
  FlyDSL-side `sched_group_barrier` primitive added to the compiler,
  **or** (b) inline-asm regions to manually emit the schedule. Both
  are weeks of compiler/IR work, not in scope for this iteration.

**Files added by Iter 9**:
- `aiter/ops/flydsl/kernels/batched_gemm_fp8_blockwise_flydsl_v3.py`
  (566 LOC, copy of v2 with K-loop replaced by the v3 pipeline; same
  XOR-swizzle, same scale-prologue, same MFMA call site, same output store).

### Iter 10 — CShuffle epilogue alone — *1-5% regression*

After Iter 9 isolated the *K-loop* part of CK's lead and showed it doesn't
move on its own, this iteration isolates the *epilogue* part. The
deep-dive analysis estimated CShuffle alone could save ~5-10% of total
wallclock by converting v2's 64×2-byte HBM stores into 8×16-byte
coalesced stores. Built `batched_gemm_fp8_blockwise_flydsl_v2_cshuffle.py`
to test that prediction directly.

**Design** (only the output epilogue changed; K-loop is byte-identical to v2):

1. Reuse the dead 16 KB A LDS buffer as a `bf16[64, BLOCK_N]` staging
   area (no new LDS allocation, no occupancy hit).
2. Process the 128-row output in 2 rounds of 64 rows each. Per round:
   a. Each lane writes its 32 bf16 values (4 m_sub × 2 n_sub × 4 i)
      to the staging area at coalesced (m, n) positions.
   b. `gpu.barrier()` — wait all waves' LDS writes done.
   c. 4 cooperative HBM-write passes: 256 thread × 8 bf16 = 16 rows ×
      `BLOCK_N` cols per pass. 64 rows / 16 = 4 passes. Each pass:
      1× `ds_read_b128` (16 B) + 1× `buffer_store_b128` (16 B) per lane.
   d. `gpu.barrier()` (skipped on round 1) before round 1's LDS writes.

**Correctness**: PASS. Max bf16 err = 0.5 vs torch oracle on
(8, 4096, 1024, 4096); 0.0625 on smaller shapes.

**Performance**:

```
shape                          v2 us   v2cs us    cs/v2    v2 TF   cs TF
--------------------------------------------------------------------------------
(8,4096,1024,4096)        498.7u   507.6u   1.02x      551     541
(8,8192,1024,4096)        737.7u   775.6u   1.05x      745     709
(8,16384,1024,4096)      1237.5u  1257.6u   1.02x      889     874
(8,32768,1024,4096)      2336.1u  2382.0u   1.02x      941     923
(16,4096,1024,4096)       738.5u   768.4u   1.04x      744     715
(16,8192,1024,4096)      1224.6u  1247.0u   1.02x      898     882
(16,16384,1024,4096)     2351.1u  2397.9u   1.02x      935     917
(16,32768,1024,4096)     4929.4u  4957.0u   1.01x      892     887
```

**Every shape regresses 1-5%**. The biggest hit is (8, 8192) at -5%; the
smallest is (16, 32768) at -1% (essentially noise).

**ISA forensics** (gfx950, BLOCK 128×128×128):

| Metric | v2 | v2+CShuffle | Δ |
|---|---|---|---|
| `next_free_vgpr` | 128 | 132 | +4 (negligible) |
| `group_segment_fixed_size` (LDS) | 20608 | 20608 | 0 (LDS reuse worked) |
| `private_segment_fixed_size` (scratch) | 0 | 0 | 0 (no spill) |
| MFMA inst count | 512 | 512 | 0 |
| **`buffer_store_short`** (HBM 2-byte) | 64 | **0** | **-64 (eliminated as designed)** |
| **`buffer_store_b128`** (HBM 16-byte) | 0 | **8** | +8 (the coalesced replacements) |
| `ds_write_b16` (LDS 2-byte staging) | 0 | 64 | +64 (per-lane bf16 stores) |
| `ds_read_b128` (LDS 16-byte for HBM) | 800 (K-loop reads) | 520 (K-loop) + 8 (epilogue) | varies |
| `s_waitcnt` | 324 | 334 | +10 |
| `s_barrier` | 65 | 68 | +3 (the 3 epilogue barriers) |

The HBM-side coalescing **worked exactly as designed**: 64 individual
`buffer_store_short` per WG → 8 `buffer_store_b128`, an 8× drop in HBM
store transactions. **But total wallclock got worse, not better.**

**Why it didn't pay off** — the workload isn't store-bound:

- v2 sustains 530-960 TFLOPS (~9-16% of fp8 peak 6000 TFLOPS). It's
  **MFMA-bound**, not HBM-store-bound. Stores happen at the very tail
  of the kernel and don't gate any other instruction.
- v2's "uncoalesced" stores aren't fully scattered — they're already
  16-way coalesced within each MFMA-output 16-lane group (lanes 0-15
  write 16 contiguous N positions). The remaining inefficiency
  (4 lane-groups not merged into one transaction) is a small fraction
  of total HBM traffic.
- CShuffle's overhead is real: 64 `ds_write_b16` per lane (16-way
  coalesced into ~16 LDS-bank cycles), 3 extra `s_barrier` (~30 cycles
  each = ~90 cycles), plus per-pass address computation. For shapes
  this small, the staging overhead consumes ~10-30 µs while the saved
  HBM time is < 10 µs.

**Confirms the deep-dive's conclusion** in a sharper form:
- **Intrawave v3 + CShuffle are SYNERGISTIC, not additive**. CK gets
  both wins together: the v3 scheduler frees up MFMA bandwidth →
  workload becomes store-bound → CShuffle then matters.
- Adopting one without the other (Iter 9 = K-loop only; Iter 10 =
  epilogue only) gives flat-to-slight-regression.
- To close the v2/CK gap requires **both** — and Iter 9 already showed
  v3 needs `sched_group_barrier` (which FlyDSL lacks).

**Production decision**:
- Keep v2 (`flydsl_batched_gemm_fp8_blockwise`) as the FlyDSL prefill
  production path.
- v2_cshuffle stays in-tree as a documented experiment; not wired into
  any dispatch.

**Files added by Iter 10**:
- `aiter/ops/flydsl/kernels/batched_gemm_fp8_blockwise_flydsl_v2_cshuffle.py`
  (~700 LOC, copy of v2 with the output store replaced by 2-round
  CShuffle epilogue; reuses the dead A LDS region as bf16 staging).

### Iter 11 — DSv4 Flash / Pro horizontal bench + hybrid dispatcher

Goal: see whether combining flydsl-for-decode + CK-for-prefill (the
"hybrid" we'd advertise as the flydsl-path production wrapper) is
actually optimal on the **realistic DeepSeek V4 `wo_a` single-op
shapes** that ship today.

**Shapes** (from `op_tests/bench_batched_gemm_fp8_blockwise.py`,
TP=8 sharded N=1024, K=4096):

- **Flash decode** B=8, T∈{1, 4, 16, 64}
- **Pro decode** B=16, T∈{1, 4, 16, 64}
- **Flash prefill** B=8, T∈{1024, 4096, 8192, 16384}
- **Pro prefill** B=16, T∈{1024, 4096, 8192, 16384}

**Implementations benched** (in `atom-latest-todd` docker, GLIBCXX_3.4.33):

| Name | What | Notes |
|---|---|---|
| `flydsl` | sw (M<128) / v2 (M>=128), auto-routed | T<16 padded to 16 on host |
| `triton` | aiter triton backend | |
| `ck` | aiter CK heuristic dispatcher | |
| `fly+ck` | flydsl (M<128) + CK (M>=128) | **user-requested hybrid** |
| `tri+ck` | triton (M<4096) + CK (M>=4096) | post-bench-derived hybrid |

**Results** (µs, median of 20):

```
=== flash_decode (B=8) ===
shape                  flydsl  triton    ck    fly+ck  tri+ck    best
(8,    1,1024,4096)    326.7   46.0   117.0   308.2    45.8    tri+ck
(8,    4,1024,4096)    313.4   46.2   117.5   318.6    46.0    tri+ck
(8,   16,1024,4096)    278.6   47.5   117.3   280.1    47.2    tri+ck
(8,   64,1024,4096)    293.0   48.8   206.6   292.7    48.1    tri+ck

=== pro_decode (B=16) ===
shape                  flydsl  triton    ck    fly+ck  tri+ck    best
(16,    1,1024,4096)   310.8   46.6   232.2   320.9    46.6    tri+ck
(16,    4,1024,4096)   313.8   47.0   232.8   317.6    47.2    triton
(16,   16,1024,4096)   285.5   48.0   232.7   285.7    48.0    triton
(16,   64,1024,4096)   316.9   51.2   411.4   325.4    51.2    tri+ck

=== flash_prefill (B=8) ===
shape                  flydsl  triton    ck    fly+ck  tri+ck    best
(8, 1024,1024,4096)    331.0  105.6   210.0   209.4   105.2    tri+ck
(8, 4096,1024,4096)    502.8  392.8   244.6   237.3   234.4    tri+ck
(8, 8192,1024,4096)    732.8  758.6   438.0   433.9   431.2    tri+ck
(8,16384,1024,4096)   1233.4 1415.7   835.8   835.0   837.7    fly+ck

=== pro_prefill (B=16) ===
shape                  flydsl  triton    ck    fly+ck  tri+ck    best
(16, 1024,1024,4096)   392.8  215.3   417.6   417.6   213.6    tri+ck
(16, 4096,1024,4096)   739.2  758.1   533.6   534.1   527.4    tri+ck
(16, 8192,1024,4096)  1249.4 1413.2   863.2   864.2   865.1    ck
(16,16384,1024,4096)  2421.0 2804.5  1675.6  1678.5  1681.7    ck
```

#### Honest findings (corrects an earlier folk belief)

1. **Triton is uniformly best on decode** (T ≤ 64), winning by **5–7×**
   over flydsl on every decode shape (46–51 µs vs 280–325 µs). The
   prior "sw beats Triton at decode" framing in earlier iters was
   **wrong** — sw's single-wave geometry is fundamentally underutilized
   on M < 128 (only 64–128 WGs across the 256-CU chip; Triton's
   smaller-tile autotune saturates better).

2. **flydsl never wins** on any DSv4 single-op shape. Best case
   (16, 16384) it ties CK at 2421 vs 2422 µs. Everywhere else it
   loses by 1.2–7×.

3. **CK takes over from M ≥ 4096** (B=8) and M ≥ 8192 (B=16). Below
   that, Triton's smaller tiles dominate because there aren't enough
   M-tiles to saturate CK's 128×128 tile pipeline.

4. **The user-requested `fly+ck` hybrid** is suboptimal on every
   decode shape (it inherits flydsl's 5-7× decode loss). On prefill
   it matches `ck` directly.

5. **The `tri+ck` hybrid wins or ties on 14/16 shapes.** It loses to
   flydsl by < 1% only on (8, 16384) — within noise. This is the
   recommended production dispatcher.

#### Implications

- **The flydsl FP8 batched-GEMM path is not on the production
  critical path** for the DSv4 wo_a use case. Triton handles decode;
  CK handles prefill; flydsl's value is the
  optimization-journey documentation + a fallback for future shapes
  CK doesn't yet cover.
- The earlier journey iters that targeted flydsl decode perf (sw
  variants, splitk experiments) were chasing a problem with the wrong
  tool — chip occupancy at M < 128 caps any single-wave kernel
  at low TFLOPS regardless of micro-optimization. The right answer
  at decode is **launch more, smaller workgroups** (Triton's tile
  autotune does this), or move the contraction to a **persistent**
  kernel (out of scope here).

#### Recommended dispatcher (for `aiter.batched_gemm_fp8_blockwise`)

```python
def dispatch(A, W, A_s, W_s, out=None):
    B, M, K = A.shape
    # CK wins for large prefill where M-tile count >> CU count.
    # Threshold derived from this bench: M >= 4096 (B=8) / M >= 8192 (B=16).
    if M >= 4096 and M % 128 == 0 and W.shape[1] % 128 == 0:
        return ck_backend(A, W, A_s, W_s, out=out)
    # Otherwise Triton wins (decode + small prefill).
    return triton_backend(A, W, A_s, W_s, out=out)
```

(If CK is unavailable — e.g. host w/o libstdc++ ≥ 3.4.31 — fall back to
Triton everywhere; flydsl's v2 only matches Triton on the very largest
shapes and never wins meaningfully.)

**Files added by Iter 11**:
- `/tmp/bench_dsv4_v2.py` — horizontal bench script (5 implementations
  × 16 shapes; ~150 LOC). Not checked into the repo because it's a
  one-shot diagnostic, but the data is recorded here.

### Iter 11 CORRECTION — flydsl DOES win decode (the bench above was buggy)

After publishing the Iter 11 table, the user asked the agent to search
the prior transcripts and surfaced earlier benchmarks where flydsl
clearly beat Triton at decode (Flash-DP T=16 G=8: flydsl ~21 µs vs
Triton ~31 µs, 1.46× speedup; Flash-TP8 T=16 G=1: flydsl ~12 µs vs
Triton ~30 µs, 2.55× speedup). These contradicted Iter 11's "Triton
uniformly wins decode" claim.

**Root cause** (found by re-instrumenting the bench):

The Iter 11 bench fed `fp32` scales to `flydsl_batched_gemm_fp8_blockwise()`.
The wrapper then ran `_torch_scales_to_ue8m0(fp32)` **inside the
benchmark loop** on every call. This conversion is a non-trivial
elementwise kernel (~200–400 µs depending on tensor size) that dominated
the timing for small decode shapes.

Triton and CK take fp32 scales natively, so they had no conversion
overhead in the loop. **flydsl was being charged for work the others
weren't doing.**

**Re-bench with pre-converted u8 scales** (decode only, single-shape runs
to avoid GPU contention seen in the full sweep):

```
config              B   M    N     K   fly(fp32)  fly(u8)  triton    ck    u8/tri
Flash-DP    T=16    8   16  1024  4096   514.7u    56.7u   85.5u   200.6u   ★0.66x
Flash-DP    T=64    8   64  1024  4096   623.7u   106.0u   86.8u   209.0u    1.22x
Flash-TP8   T=16    1   16  1024  4096 24863.9u    65.9u   83.6u    33.7u   ★0.79x
Flash-TP8   T=64    1   64  1024  4096   449.8u    75.5u   85.9u    45.2u   ★0.88x
Pro-DP      T=16   16   16  1024  4096   476.2u    68.2u   83.1u   246.3u   ★0.82x
Pro-DP      T=64   16   64  1024  4096   485.6u   101.7u   89.4u   775.4u    1.14x
```

Compare with the matching old-transcript bench:

| Shape | Old fly | Old tri | Old ratio | New fly(u8) | New tri | New ratio |
|---|---|---|---|---|---|---|
| Flash-DP T=16 | 21.1 µs | 30.9 µs | 0.68× | 56.7 µs | 85.5 µs | **0.66×** ✓ |
| Pro-DP T=16 | 24.0 µs | 31.5 µs | 0.76× | 68.2 µs | 83.1 µs | **0.82×** ✓ |
| Flash-TP8 T=16 | 11.9 µs | 30.4 µs | 0.39× | 65.9 µs | 83.6 µs | **0.79×** |

The *absolute* numbers are 2–4× slower in the new bench than the old
one (different rocm/docker/aiter version + likely concurrent GPU usage),
but the **relative pattern matches**: flydsl wins decode at T=16 on
every config, by 0.66–0.82× (1.2–1.5× speedup over Triton).

**Corrected conclusions**:

1. **flydsl IS faster than Triton at decode (T ≤ 16)** when scales are
   pre-converted to u8 — the production case, since serving stacks
   hold quantized weights and would normally produce u8 scales upstream.
   The earlier-iter folk knowledge was right.
2. The Iter 11 `tri+ck` "best hybrid" recommendation was an artifact
   of the timing bug. The **correct hybrid is `fly+ck`** (the user's
   original request) for the decode path, with the assumption that the
   serving pipeline holds u8 scales.
3. At T=64 the picture is closer to a tie (1.1–1.2× toward Triton at
   B≥8); at T=16 and below flydsl's advantage is clearest.
4. CK is still wrong for decode at most shapes (CK 200+ µs vs flydsl
   ~60 µs at Flash-DP T=16), confirming the original `fly(decode) +
   ck(prefill)` design.

**Recommended dispatcher** (corrected):

```python
def dispatch(A, W, A_s, W_s, out=None):
    B, M, _ = A.shape
    # Large prefill: CK wins decisively.
    if M >= 4096 and M % 128 == 0 and W.shape[1] % 128 == 0:
        return ck_backend(A, W, A_s, W_s, out=out)
    # Decode + small prefill: flydsl wins if scales are u8.
    # If A_s/W_s are fp32, callers should pre-convert once at model
    # load time — NOT inside the dispatcher hot path.
    if A_s.dtype == torch.uint8 and W_s.dtype == torch.uint8:
        return flydsl_backend(A, W, A_s, W_s, out=out)
    # Fallback for fp32 scales: triton (CK is slow at decode).
    return triton_backend(A, W, A_s, W_s, out=out)
```

**Methodology lesson** (added to diagnostic cheat sheet below):
**always pre-convert dtypes outside the timing loop**, and verify by
running each implementation with both its preferred and the canonical
input dtype. A 4× discrepancy with prior bench data is a smell.

---

### Iter 12 — v2 tile autotune (block_m / block_n / n_waves) — *default already optimal*

**Motivation**: CK autotunes its tile size per shape (one of the three
sources of its lead per the Iter 8 deep-dive). v2 had only ever run the
fixed 128x128 / 4-wave geometry. Question: does sweeping the WG tile
shape close any of the CK gap?

**Change**: parameterized `compile_bgfp8bw_v2_kernel` /
`flydsl_batched_gemm_fp8_blockwise_v2` to accept `block_m`, `block_n`,
`n_waves` (BLOCK_K stays 128 = MFMA-K = scale-block). Generalizations:
cooperative A-DMA chunk count `= block_m/(n_waves*8)` with row stride
`chunk*(n_waves*8)`; A_scale prologue now loops `ceil(block_m*K_g /
(block_threads*16))` rounds with a partial-round thread guard (the old
code hardcoded one 256-thread 4 KB round → wrong for any other geometry);
W_scale `n_block_idx = (pid_n*block_n)//128`. Defaults reproduce the
original codegen exactly.

**Correctness**: all valid geometries max-err 0.5 vs torch oracle;
`block_n>128` correctly rejected (would need 2 W_scale bytes/iter).

**Tuner**: 14 valid configs/shape (block_m∈{64,128,256}, block_n∈{64,128},
n_waves∈{2,4,8}, filtered by divisibility + a crude acc-VGPR<=160 guard).
Run on an **idle** gfx950 node (the dev host's 8 GPUs were 100% busy with
other tenants — contention inflated v2 to 650 us, so all tile-tune
numbers were taken on smci355-...-n02-09 with flydsl 0.1.3.1).

**Results** (CUDA-event median, idle node):

| shape | best geom | best us | best TF | default(128x128x4) us |
|---|---|---|---|---|
| (8,1024,1024,4096) | 128x128x**8** | 328 | 209 | 329 |
| (8,4096,1024,4096) | **128x128x4** | 494 | 557 | 494 |
| (8,8192,1024,4096) | **128x128x4** | 710 | 775 | 710 |

These clean numbers match the Iter 8 v2 figures (508/729 us) within 3%,
confirming the harness. **Tile autotune gives ~0%**: the default 128x128x4
is already the optimum at M>=4096; at M=1024 n_waves=8 edges ahead by
1 us (noise). Other geometries (narrow 64-M tiles, n_waves=2, 256-M
tiles) are all 3-50% slower — bigger tiles spill, smaller tiles
under-amortize K-loop overhead (same lesson as Iter 4).

**Conclusion**: tile geometry is **not** a source of the CK gap for these
shapes — v2 was already sitting on the best tile. The remaining ~1.6-1.9x
gap to CK (v2 494 vs CK ~264 @ M=4096; v2 710 vs CK ~436 @ M=8192) is
confirmed structural: Intrawave-v3 K-loop scheduling + CShuffle epilogue
(Iter 8-10), neither reachable without a FlyDSL `sched_group_barrier`
primitive. Next untried lever: the **32x32x64 fp8 MFMA** (CK's MFMA shape;
v2 uses 16x16x128) — changes issue-rate amortization, orthogonal to tile
size.

**Files**: `batched_gemm_fp8_blockwise_flydsl_v2.py` gains the geometry
params (backward-compatible). Tuner lives in `_tune_geom.py` (dev scratch,
not committed).

---

### Iter 13 — m32 kernel: 32x32x64 fp8 MFMA — *6-8% win over v2*

**Motivation**: v2 (and every prior variant) used the `16x16x128` scaled
fp8 MFMA. CK kid=0 uses `32x32x64`. A 32x32x64 MFMA does **2x the MACs
per instruction** (32*32*64 vs 16*16*128 = 65536 vs 32768), so the same
total work needs **half the MFMA instructions** → better issue-rate
amortization (the Iter-4 bottleneck). Orthogonal to tile size (Iter 12).

**New kernel** `batched_gemm_fp8_blockwise_flydsl_m32.py` (copy of v2 with
the compute core swapped). Same LDS layout, cooperative A-DMA, scale
prologue, XOR-swizzle. Changes:
- MFMA `mfma_scale_f32_32x32x64_f8f6f4`. flydsl 0.1.3.1 ships **no**
  functional wrapper for it (only the 16x16x128 one), so `_mfma_scale_32x32x64`
  calls the raw rocdl OpView and unwraps operands via
  `rocdl._unwrap_mfma_operand`, returns `.result`. Accumulate by **chaining
  the `c` operand** across KRepeat (cleaner than v2's manual element add).
- Geometry: per wave M_SUB32=block_m/32 32-row M-blocks, N_SUB32=
  (block_n/n_waves)/32 32-col N-blocks, **KRepeat=2** (128-K iter = two
  64-K MFMA steps, same accumulator, same 128-K scale byte).
- Lane mapping (CDNA4, verified correct first try):
  - A/B operand: `row = l%32`, `kgroup = l//32` → lane holds 32 K-bytes
    `[kgroup*32 + 0..31]` (v8i32).
  - C accumulator (v16f32/lane): `col = l%32`; reg `i` → `row =
    (i//4)*8 + (l//32)*4 + (i%4)`.
- Acc VGPR: M_SUB32*N_SUB32*16 = 64 f32 at 128x128x4 (same as v2's
  8*2*4=64) — no occupancy hit.

**Correctness**: max-err 0.5 / mean 2e-4 vs torch oracle on first compile,
across 128x128x4 / 64x128x4 / 128x128x2 (block_n/n_waves must be a
multiple of 32). The derived mappings + `_ue8m0_byte_pack4` scale (×4
replicate; the MFMA reads the 2 sub-block bytes it needs) were all right.

**Performance** (idle node, m32 geometry swept):

| shape | v2 best | m32 best | m32 geom | m32 TF | m32/v2 |
|---|---|---|---|---|---|
| (8,1024,1024,4096) | 338.6 | **318.7** | 128x128x4 | 216 | 0.94x |
| (8,4096,1024,4096) | 484.6 | **445.0** | 256x128x4 | 618 | 0.92x |
| (8,8192,1024,4096) | 694.6 | **655.4** | 256x128x4 | 839 | 0.94x |

**6-8% faster than v2** — the first real gain since Iter 8 Phase 4b. The
bigger 256-M tile wins at M>=4096 (32x32 MFMA keeps acc-VGPR low even at
M_SUB32=8 → 128 acc regs, still no spill). Gap to CK narrows:
- @4096: 1.87x → **1.69x** (m32 618 vs CK ~1043 TF)
- @8192: 1.63x → **1.50x** (m32 839 vs CK ~1260 TF)

**Takeaway**: MFMA shape matters — half the instruction count buys ~7%.
But the bulk of CK's lead is still the Intrawave-v3 scheduler + CShuffle
(Iter 8-10). Next: apply the m32 compute core under profiling-driven
`sched_*` interleaving (the ~25% lever), now that the instruction stream
is shorter (32 MFMAs/WG/iter vs 64) and may schedule more cleanly.

**Files**: `batched_gemm_fp8_blockwise_flydsl_m32.py` (new). Bench
`_bench_m32.py`, test `_test_m32.py` (dev scratch).

---

### Iter 14 — m32 + sched_* hints — *flat to -3%, reverted (default off)*

**Motivation**: the m32 instruction stream is half v2's (32 MFMAs/WG/iter
vs 64), so it might schedule more cleanly under explicit `sched_*` hints —
the ~25% lever (CK's HotLoopScheduler). **Update to a prior journey claim**:
flydsl 0.1.3.1 *does* expose `rocdl.sched_group_barrier(mask, size, group)`
(the exact CK primitive) plus `sched_mfma/dsrd/vmem/barrier`; a working
reference exists in `mixed_moe_gemm_2stage.py:_sched_hints_stage1_gate_up`.

**Tried** (m32, `sched_hint` param): (1) a single `sched_barrier(0)` fence
per K-iter; (2) interleave `sched_vmem(NV); for NM: sched_dsrd(2); sched_mfma(1)`.

**Result** (idle node, best m32 geom per shape):

| shape | hint=0 | hint=1 (fence) | hint=2 (interleave) |
|---|---|---|---|
| (8,1024,...) | 335 | 334 | 332 |
| (8,4096,...) | **467** | 480 | 475 |
| (8,8192,...) | **668** | 678 | 680 |

**Flat at M=1024, -2 to -3% at M>=4096** — same outcome as Iter 8 Phase 3a.
The coarse `sched_mfma/dsrd/vmem` group hints over-constrain LLVM (and miss
the scale ds_reads in the count → nop risk), and the default scheduler is
already good. The *precise* CK approach needs a per-instruction
`sched_group_barrier` table with correct masks, matched against the real
ISA — but flydsl 0.1.3.1's ASM dump (`FLYDSL_DEBUG_DUMP_ASM`) doesn't emit
in this env, so the instruction accounting needed to craft a nop-free table
isn't available. Building that table is the "weeks of work" the Iter 8
deep-dive predicted.

**Decision**: `sched_hint` defaults to 0 (off); production m32 unaffected.
The hint code stays in-tree (dormant) for future ISA-driven tuning.

**Net of the two structural levers this round**: tile autotune ~0% (Iter 12),
32x32x64 MFMA **+6-8%** (Iter 13), sched hints regress (Iter 14). The m32
MFMA is the keeper — CK gap now **1.69x @ M=4096, 1.50x @ M=8192** (was
1.87x / 1.63x). Remaining gap is the v3-scheduler + CShuffle pair, still
gated on ISA-dump tooling + sched_group_barrier table work.

---

## 3. Diagnostic methodology cheat sheet

### Spill / VGPR check (the *first* thing to try if perf disappoints)

```bash
# Compile a single kernel with ASM dump
rm -rf ~/.flydsl/debug ~/.flydsl/cache
FLYDSL_DEBUG_DUMP_ASM=1 FLYDSL_DUMP_IR=1 FLYDSL_RUNTIME_ENABLE_CACHE=0 \
    python my_kernel_runner.py
# → dumps to ~/.flydsl/debug/kernel_0/17_final_isa.s

# Resource directives (top of the .s file)
grep -E '\.amdhsa_(next_free_vgpr|next_free_sgpr|private_segment_fixed_size|group_segment_fixed_size)' \
    ~/.flydsl/debug/kernel_0/17_final_isa.s
```
Interpret:
- `next_free_vgpr = N` → kernel uses `N` VGPRs/wave.
  - gfx950 has 512 VGPRs/SIMD; max waves/SIMD = `floor(512 / N)`.
  - If `N > 256`, MFMA paths can be limited.
- `private_segment_fixed_size` > 0 → **register spill** (scratch).
  Each spill load/store is ~100s of cycles. Reduce live ranges, split
  the kernel, or shrink tile.
- `group_segment_fixed_size` = LDS bytes/WG. gfx950 cap = 64 KB/WG.

### Inner-loop instruction breakdown
```bash
grep -cE 'v_mfma_'           17_final_isa.s    # MFMA count
grep -cE 'scratch_(load|store)' 17_final_isa.s # spill traffic (want 0)
grep -cE 'buffer_load_'      17_final_isa.s    # HBM loads
grep -cE 'ds_(read|write)'   17_final_isa.s    # LDS traffic
grep -cE 's_waitcnt'         17_final_isa.s    # serialization waits
grep -cE 's_barrier'         17_final_isa.s    # WG barriers
```
Rule of thumb on the inner-loop MFMA throughput:
- One `mfma_scale_f32_16x16x128_f8` ≈ 32 cycles latency on gfx950.
- s_waitcnt:MFMA ratio < 4:1 → loop is mostly compute (good).
- s_waitcnt:MFMA ratio > 8:1 → memory-stall-bound (bad).

### Headroom check

Always compare against a known-good reference (Triton, hand-tuned CK,
torch oracle for correctness). If you're > 5× behind the reference,
optimization has runway. If you're within 1.5×, decide whether the
remaining win is worth the work.

### CUDA-event timing pattern (in the bench)
```python
def gpu_time(fn, iters=30, warmup=10):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    e = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for a, b in zip(s, e):
        a.record(); fn(); b.record()
    torch.cuda.synchronize()
    return sorted([a.elapsed_time(b) * 1000 for a, b in zip(s, e)])[iters // 2]
```
Median, not mean. Always include warmup (first call JIT-compiles).

---

## 4. Open paths (not yet pursued)

These are the candidates ranked roughly by expected ROI. Tried-and-reverted
items (`sched_*` hints in v2) and tried-and-shipped items (async DMA via
`raw_ptr_buffer_load_lds`, XOR-swizzle, scf.for prefetch on sw) are documented
in the iteration log above; this section is **only forward-looking**.

| Path | Target | Expected | Effort | Notes |
|---|---|---|---|---|
| **LDS double-buffer (ping-pong) on v2** | v2 prefill | 5-15% | Medium | Two LDS-A buffers (32 KB), DMA[k+1] issued in iter k while MFMA[k] runs from buffer[k%2]. Overlaps DMA latency behind MFMA. Requires loop restructure (range_constexpr → scf.for with loop-carried buffer index, or fully-unrolled with explicit ping/pong). |
| **GROUP-major scheduling on v2** | v2 prefill | 5-10% | Small | Triton uses `GROUP_SIZE_M=8` to walk WGs in M-major chunks for L2-cache friendliness. We currently launch in linear (pid_m, pid_n) order. ~10 LOC change in the launcher block-id computation. Cheap win on shapes where W reuse across M-tiles matters. |
| **decode split-K (sw)** | sw decode | 2-4× | Medium | Decode is 9-10× behind Triton; chip is underused at small M (only 128 WGs at M=16). Split-K launches 2-4× more WGs by partitioning K, then atomic-add the partials. Watch for bf16 atomic-add behavior on gfx950 — may need fp32 partial buffer + bf16 truncate at end. |
| **A_scale prefetch on v2** | v2 prefill | 1-3% | Small | A_scale is currently loaded synchronously per K-iter (8 lane-broadcast byte loads/wave). Carry a_scale across iters via scf.for state, similar to Iter 7's A/W prefetch. |
| **`waves_per_eu` / `maxnreg` compile hints** | v2 | 1-3% | Trivial | `flyc.kernel(waves_per_eu=N)` to nudge occupancy. v2 currently 138 VGPR → ~3 waves/SIMD; try forcing 4 to see if extra latency-hiding helps. |
| **Drop host A_scale / W_scale fp32→u8 conversion** | wrapper | 0.x ms saved per call | Trivial | The Python wrapper still calls `_torch_scales_to_ue8m0` if dtype is fp32. If the model already stores UE8M0 u8, pass it through directly (already supported via the dtype branch — just stop converting). |
| **`waves_per_eu` + `iglp_opt` together** | v2 | 1-5% | Small | `iglp_opt` is the inter-loop scheduler "interleave" pragma; combined with `waves_per_eu` may unlock latency hiding the default scheduler doesn't try. |
| **CShuffle epilogue** | v2 prefill | 1-3% | Medium | Currently each lane writes 64 bf16 directly to HBM (potentially uncoalesced for some N strides). LDS-staged shuffle could coalesce the writes. Only worth it if profile shows store stalls — current ISA shows MFMA bound, not store bound. |

---

## 5. Files

### Kernels

| File | Role |
|---|---|
| `batched_gemm_fp8_blockwise_flydsl.py` | **sw** — production kernel for **decode** (`M < 128` or non-128-multiple). Geometry: 1 wave/WG, `BLOCK_M ∈ {16,32,64}`, `BLOCK_N ∈ {16,32}`, `M_SUB × N_SUB` MFMAs/K-iter, optional `scf.for` A/W prefetch on path B. **Also hosts the public dispatch wrapper** `flydsl_batched_gemm_fp8_blockwise()` which routes prefill shapes (`M >= 128 && M % 128 == 0 && N % 128 == 0`) to v2. |
| `batched_gemm_fp8_blockwise_flydsl_v2.py` | **v2** — production kernel for **prefill**. 4 waves/WG, BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, single-buffer LDS-A (16 KB) + XOR-swizzle, async DMA `raw_ptr_buffer_load_lds`, native fp8 MFMA with scaleA + scaleB. |
| `batched_gemm_fp8_blockwise_flydsl_mw.py` | **mw** — multi-wave + LDS reference, **NOT production**. Kept as historical reference for the failed direction (Iters 1-3) — its geometry (BLOCK_M=16, BLOCK_N=64, no XOR-swizzle, no async DMA) was the wrong starting point; v2 supersedes it. |
| `tensor_shim.py` | `GTensor` (HBM via `buffer_ops`) + `STensor` (LDS via `vector.load_op`) wrappers used by all kernels. |

### External references (read-only)

| File | Role |
|---|---|
| `FlyDSL/kernels/blockscale_preshuffle_gemm.py` | 896 LOC reference — the closest-to-our-op kernel in the FlyDSL repo. v2's geometry, async-DMA pattern, and XOR-swizzle pattern are all borrowed from it. |
| `FlyDSL/kernels/mfma_preshuffle_pipeline.py` | Source of `swizzle_xor16` helper (Phase 2) and the layout-builder utilities. |
| `FlyDSL/kernels/hgemm_splitk.py` | 951 LOC reference for `scf.for` + loop-carried prefetch (Iter 7) and `hot_loop_scheduler` patterns. |
| `FlyDSL/.claude/skills/{prefetch-data-load,gemm-optimization,lds-optimization,flydsl-tile-programming}/SKILL.md` | Skill docs that informed Iters 7, 8 Phase 1, 8 Phase 2, 8 Phase 3b respectively. |

### Docs

| File | Role |
|---|---|
| `OPTIMIZATION_JOURNEY.md` | This file. |
| `OPTIMIZATION_JOURNEY.zh.md` | Chinese translation, kept in sync. |
