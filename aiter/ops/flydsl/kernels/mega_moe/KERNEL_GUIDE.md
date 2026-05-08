# FlyDSL MegaMOE Kernel Guide

A walkthrough of the Phase 0 + Phase 1 kernels in this directory, focused on
**every FlyDSL API used and why**, plus the reasoning behind each
hyperparameter choice. Read this if you are extending the kernels here or
writing your own FlyDSL code on AMD CDNA4 (gfx950 / MI355X).

## How to read this guide

Sections 1–9 cover the device-side APIs (what runs in `@flyc.kernel`).
Section 10 is the hyperparameter reference table.
Section 11 covers host-side patterns (Phase 0 byte layout / scheduler).
Sections 12–14 cover heuristics and the debugging probe idioms that caught
the real porting bugs.

The tutorial is built around the existing files; line numbers cite the
`_phase1_step*.py` series.

---

## 1. The two decorators — `@flyc.kernel` vs `@flyc.jit`

Every kernel in this directory has the same two-layer structure:

```python
@flyc.kernel
def my_kernel(a: fx.Tensor, b: fx.Tensor, c: fx.Tensor):
    # Device code — traced into MLIR, runs on the GPU.
    ...

@flyc.jit
def my_launcher(a: fx.Tensor, b: fx.Tensor, c: fx.Tensor,
                stream: fx.Stream = fx.Stream(None)):
    # Host code — sets the launch grid.
    my_kernel(a, b, c).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)
```

- `@flyc.kernel` marks the function whose body is *traced into MLIR*, not
  run as Python. Inside, every `+`, `*`, `//`, `%` on `fx.Index` values
  emits an `arith.addi` / `muli` / `divui` / `remui` op. The Python
  control-flow constructs (`for`, `if`) emit MLIR (`scf.for`, `scf.if`)
  rather than running natively.
- `@flyc.jit` marks the host launcher. It JIT-compiles `my_kernel` (cached
  by signature), then calls `.launch(grid, block, stream)`. The first call
  pays compile cost; subsequent calls reuse the cached binary.

**Why two layers**: separates "what to compile" from "how to launch". The
kernel is parameterized; the launcher fixes grid/block based on input
shapes. You can have multiple launchers wrapping the same kernel for
different launch shapes without recompiling.

---

## 2. Tensor lifecycle — `from_dlpack` + `mark_layout_dynamic`

From `_phase1_step1_single_tile.py:main()`:

```python
A = A_f32.to(torch.float8_e4m3fn).contiguous()             # torch fp8 tensor
A_dl = flyc.from_dlpack(A).mark_layout_dynamic(
    leading_dim=1, divisibility=128
)
```

- `flyc.from_dlpack(A)` wraps a torch tensor as an `fx.Tensor` via the
  DLPack protocol (zero-copy).
- `mark_layout_dynamic(leading_dim=1, divisibility=128)` tells the JIT:
  - **`leading_dim=1`** — the **stride-1** (innermost contiguous) dimension
    is dim 1. For a row-major `[16, 128]` tensor, dim 1 is contiguous
    (stride 1 byte), dim 0 has stride 128 bytes. Common bug: passing
    `leading_dim=0` raises `Leading dimension must have stride 1` because
    dim 0 has stride 128, not 1.
  - **`divisibility=128`** — the leading dim's *length* is divisible by
    128. This unlocks vectorized loads (dwordx4 = 16-byte loads) since the
    compiler knows row size is a multiple of 16 bytes.

**Why divisibility hint matters**: without it, the compiler generates
scalar fallback paths for partial-vector loads. With it, it emits clean
`buffer_load_dwordx4`.

---

## 3. Buffer descriptor — `create_buffer_resource`

```python
a_rsrc = buffer_ops.create_buffer_resource(a_ptr, max_size=True)
```

Produces a 128-bit AMD V# (vertex/buffer) descriptor — `<8>` ptr in the
IR. It packs:

- 64-bit base address
- 32-bit num_records (bounds-check limit, in element units)
- 32-bit stride / format / cache flags

`max_size=True` sets num_records = `0xFFFFFFFF` (4 GB), effectively
disabling bounds-check. `max_size=False` uses the tensor's actual element
count or a passed `num_records_bytes`.

The Phase 1 kernels use `max_size=True` for development. Bounds-checking
silently drops OOB stores — this hid the byte-vs-element offset bug for
several debug rounds (when 56 of 64 lanes' addresses went OOB, the
descriptor silently dropped them, leaving stale zeros that looked like a
lane-mapping bug).

---

## 4. The element-vs-byte offset trap

The single most important FlyDSL ABI gotcha:

```python
a_lo = buffer_ops.buffer_load(a_rsrc, a_dword_off, vec_width=4, dtype=T.i32)
```

The `offset` parameter is in **`dtype`-element units, not bytes**. The
lowering automatically multiplies by `sizeof(dtype)` (4 for i32, 1 for
i8, etc.).

So if you want to read FP8 byte 128 of A as i32:

| Wrong | Right |
|---|---|
| `offset = 128` | `offset = 128 / 4 = 32` |
| Hardware byte address `128 * 4 = 512` (OOB) | Hardware byte address `32 * 4 = 128` ✓ |

That is why all address math in the Phase 1 kernels divides by 4
explicitly:

```python
a_dwords_per_row = K // 8     # K fp4 / 2-fp4-per-byte / 4-byte-per-dword = K/8
```

(FP4 packs 2 elements/byte, so K fp4 elements = K/2 bytes = K/8 dwords.)

`buffer_store` does have an `offset_is_bytes=True` flag if you really want
raw bytes; `buffer_load` does not — always element units.

---

## 5. The MFMA call — the heart of everything

```python
acc = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
    T.f32x4,                                     # result type
    [a128, b128, acc_in,                          # operands
     0, 4,                                        # cbsz=FP8, blgp=FP4
     0, 0x7F7F7F7F,                               # opselA=0, scaleA=1.0
     0, 0x7F7F7F7F],                              # opselB=0, scaleB=1.0
)
```

### Result type

`T.f32x4` is a `@property`, **not** a `@method`. Calling `T.f32x4()`
attempts to call the resulting `VectorType` instance and fails with
`'VectorType' object is not callable`. The result type for a 16×16 MFMA
is always `vector<4xf32>` — each lane returns 4 fp32 accumulator values.

### Operand layout per lane

(from the CDNA4 atom definition `getThrValLayoutAB` in
`lib/Dialect/FlyROCDL/CDNA4/MmaAtom.cpp`)

64 lanes split as 16 (MN axis) × 4 (K axis): `lane = mn_idx + k_lane * 16`

| Operand | Vector | Per-lane content |
|---|---|---|
| A | `vector<8xi32>` | 32 fp8 elements (32 K positions) |
| B (FP4 mode) | `vector<4xi32>` | 32 fp4 elements (32 K positions) |
| B (FP8 mode) | `vector<8xi32>` | 32 fp8 elements (32 K positions) |
| C accumulator | `vector<4xf32>` | 4 outputs at C[(lane/16)*4 + 0..3, lane%16] |

### `cbsz` / `blgp` encoding

(from `MmaAtom.cpp:134-146`)

| Code | Type |
|---|---|
| 0 | FP8 e4m3fn |
| 1 | FP8 e5m2 |
| 2 | FP6 e2m3fn |
| 3 | FP6 e3m2fn |
| 4 | FP4 e2m1fn |

These are repurposed from the classic-MFMA broadcast-control fields to
dtype codes for the scaled variant.

### Scale operand format

`0x7F7F7F7F` = E8M0 ≈ 1.0 in all four sub-windows. E8M0 is an 8-bit
exponent-only float; bias is 127 (`0x7F`), so `0x7F` = 1.0. The i32 packs
4 such bytes, one per K=32 sub-window. `opsel ∈ [0, 3]` selects which
sub-window each scale byte applies to. `0x7F7F7F7F` = unscaled.

### Empirically measured behavior

| Mode | cbsz, blgp | Single-call K coverage | Status |
|---|---|---|---|
| FP8 × FP8 | 0, 0 | 128/128 ✅ | Verified |
| FP4 × FP4 | 4, 4 | 128/128 ✅ | Verified |
| FP8 × FP4 mixed | 0, 4 | **32/128** ❌ | Degenerate; needs different operand packing |

The mixed-mode finding is documented in
`daily-digest/kernels/megamoe.md` §12.

---

## 6. Per-lane addressing math

From `_phase1_step5_persistent.py`:

```python
row = lane % fx.Index(16)
k_lane = lane // fx.Index(16)
a_base = (expert_idx * fx.Index(a_dwords_per_expert)
          + (m_tile_idx * fx.Index(16) + row) * fx.Index(a_dwords_per_row)
          + k_lane * fx.Index(4))
```

Decoded:

- `row = lane % 16` — which of 16 MN rows this lane reads (matches the
  MFMA layout's `mn_idx`).
- `k_lane = lane // 16` — which of 4 K-groups (matches `k_lane_idx`).
- Three additive terms: `expert_offset + tile_row_offset + k_chunk_within_tile_offset`.
- `(m_tile_idx * 16 + row)` is the global row index within the expert.
- `* a_dwords_per_row` advances to that row's start.
- `+ k_lane * 4` is the per-K-lane offset (each lane reads 4 i32 = 16
  bytes = 32 fp4 covering 32 K positions).

**Why `fx.Index(16)` and not `16`**: arithmetic on `fx.Index` emits MLIR
ops; mixing Python ints would require explicit conversion. Wrapping ints
in `fx.Index(...)` tells the compiler "this is an index value,
type-check accordingly."

---

## 7. Vector packing — `ir.VectorType.get` + `vector.from_elements`

```python
from flydsl._mlir import ir
from flydsl.expr import vector

v8i32 = ir.VectorType.get([8], ir.IntegerType.get_signless(32))
a128 = vector.from_elements(
    v8i32,
    [a_lo[0], a_lo[1], a_lo[2], a_lo[3], a_hi[0], a_hi[1], a_hi[2], a_hi[3]],
)
```

**Why drop to raw MLIR**: the installed FlyDSL exposes `T.i32x4` but not
`T.i32x8`. To build a `vector<8xi32>` (the MFMA A-operand width), we
construct the type directly via MLIR's Python bindings.

`vector.from_elements(type, [scalar_values])` packs N scalars into a
vector. `a_lo[i]` indexes into a vector, returning the i-th element
scalar.

---

## 8. Loops — `range_constexpr` vs `fx.range`

```python
for chunk in range_constexpr(k_chunks):
    chunk_off = fx.Index(chunk * 16)
    ...
```

**`range_constexpr(N)`**: pure Python loop, **fully unrolled at trace
time**. `chunk` is a Python int, so `chunk * 16` is a Python int
operation (then wrapped in `fx.Index`). Each iteration emits a fresh copy
of the body in the MLIR. Use when N is a small compile-time constant
(≤ ~16) and unrolling is desired.

**`fx.range(start, stop, step, init=[carried_state])`**: emits an
`scf.for` op. Loop bounds can be runtime values; carried values are
explicitly threaded:

```python
for it, st in fx.range(0, loop_iters, 1, init=[acc, ptr]):
    new_acc = ...
    new_ptr = ...
    yield [new_acc, new_ptr]   # implicit via final assignment
```

I used `range_constexpr` everywhere in Phase 1 because tile counts are
compile-time. For real production kernels with runtime K, you would use
`fx.range`.

---

## 9. Initial values — `Vector.filled(N, val, NumericClass)`

```python
acc = fx.Vector.filled(4, 0.0, fx.Float32)
```

`Vector.filled` wants the **Numeric class** (`fx.Float32`, `fx.Int32`),
not the **MLIR type** (`T.f32`, `T.i32`). The Numeric class wraps both
the MLIR type and Python conversion logic (`dtype(0.0)` builds a typed
constant). Passing the bare MLIR type causes
`F32Type object is not callable` at trace time.

---

## 10. The hyperparameter reference table

| Hyperparameter | Value | Why |
|---|---|---|
| MFMA tile shape | **16×16×128** | Only two CDNA4 scaled-MFMA shapes exist: 16x16x128 and 32x32x64. 16×16×128 has finer M/N granularity → better fit for small-M MoE expert tiles. |
| `BLOCK_K` | **128** | Matches the MFMA K. DeepGEMM also fixes this. Smaller would need fragment-K accumulation. |
| `BLOCK_N` | **128** | DeepGEMM uses 128 always; SF format and weight preshuffle assume it. |
| `BLOCK_M` step function | 16/32/64/96/128/192 | Direct port of `csrc/jit_kernels/heuristics/mega_moe.hpp`. Step thresholds (E[tokens/expert] ≤ 8.5/16.5/32.5/64.5/96.5) cover routing-skew distribution from RL long-tail (E ≤ 8) up to prefill (E > 96). |
| `num_sms` | **128** for MI355X | gfx950 has 128 CUs; persistent kernel uses one CTA per CU. |
| `block=(64, 1, 1)` | 64 threads | One CDNA4 wavefront = 64 lanes. Single-wave kernel keeps things simple. |
| `vec_width=4` (dwordx4 loads) | 4 i32 = 16 bytes | AMD coalesced-load granule. Smaller loads (`dwordx2`, `dword`) waste bandwidth. |
| `0x7F7F7F7F` | E8M0 ≈ 1.0 ×4 | Default scale = identity; all 4 K=32 sub-windows get scale 1.0. |
| `cbsz=0, blgp=0` (FP8×FP8) | works at K=128 | Pure mode — full K coverage per call. |
| `cbsz=4, blgp=4` (FP4×FP4) | works at K=128 | Pure mode — full K coverage per call. |
| `cbsz=0, blgp=4` (FP8×FP4 mixed) | **degenerate** | Empirically only 32/128 K positions consumed; needs different operand packing. |
| `kNumStages` (pipeline depth) | not yet used | Phase 1 step 5 has no LDS double-buffer; step 3b would add a 2-stage pipeline. |

---

## 11. Output store pattern

```python
out_row_base = k_lane * fx.Index(4)
out_col = lane % fx.Index(16)
for i in range_constexpr(4):
    out_row = m_tile_idx * fx.Index(16) + out_row_base + fx.Index(i)
    c_off = c_expert_off + out_row * fx.Index(N) + (n_tile_idx * fx.Index(16) + out_col)
    buffer_ops.buffer_store(acc[i], c_rsrc, c_off)
```

The 4 fp32 outputs per lane go to **4 vertically-adjacent rows in the
same column**. The CDNA 16×16 MFMA layout is:

> Lane (k_lane = 0..3, mn_idx = 0..15) writes `C[k_lane * 4 + 0..3, mn_idx]`.

Per-row stride = N (number of fp32 columns); the offset-in-elements is
`row * N + col` — element units, not bytes (because store offset is also
element-units by default).

---

## 12. Phase 0 host-side patterns

`workspace.py` and `scheduler.py` are pure Python with type annotations.
No FlyDSL, no GPU. Two notable patterns worth borrowing.

### `@dataclass` + `field(init=False)` for derived attributes

```python
@dataclass
class Workspace:
    num_ranks: int
    num_experts: int
    ...
    num_experts_per_rank: int = field(init=False)

    def __post_init__(self) -> None:
        self.num_experts_per_rank = self.num_experts // self.num_ranks
```

`init=False` keeps the derived field out of `__init__`'s signature;
`__post_init__` computes it once after construction.

### Single-exit `_next_block` with sentinel

```python
def _next_block(self) -> Optional[BlockDescriptor]:
    result: Optional[BlockDescriptor] = None
    while result is None and self.current_local_expert_idx < self.cfg.num_experts_per_rank:
        if ...:
            result = BlockDescriptor(...)
        else:
            self.next_phase = ...
    return result
```

Built around FlyDSL's "single-exit control flow" rule — even though this
is Python, structuring it with a `result` sentinel and one `return` makes
the device port (`scf.while` with carried state) mechanical. If we had
used early `return`, the device port would need a major rewrite.

---

## 13. Heuristic table (`heuristics.py`)

```python
def pick_block_config(num_ranks, num_experts, num_topk, num_tokens) -> BlockConfig:
    e = expected_tokens_per_expert(num_tokens, num_ranks, num_topk, num_experts)
    if e <= 8.5:    cfg = BlockConfig(block_m=16,  store_block_m=8,  num_epilogue_warpgroups=2)
    elif e <= 16.5: cfg = BlockConfig(block_m=32,  store_block_m=16, num_epilogue_warpgroups=2)
    elif e <= 32.5: cfg = BlockConfig(block_m=64,  store_block_m=32, num_epilogue_warpgroups=1)
    elif e <= 64.5: cfg = BlockConfig(block_m=96,  store_block_m=16, num_epilogue_warpgroups=2)
    elif e <= 96.5: cfg = BlockConfig(block_m=128, store_block_m=32, num_epilogue_warpgroups=2)
    else:           cfg = BlockConfig(block_m=192, store_block_m=32, num_epilogue_warpgroups=2)
```

**Why this exact table**: it is a verbatim port of
`get_block_config_for_mega_moe` in DeepGEMM C++. The thresholds
(8.5 / 16.5 / 32.5 / 64.5 / 96.5) are NV-tuned but reflect expert
routing-skew patterns common across hardware:

- `E ≤ 8` → RL long-tail rollout (most experts cold)
- `E ≤ 16` → small-batch decoding with EP=8
- `E ≤ 32` → medium-batch decoding
- `E > 96` → prefill or large EP

The `num_epilogue_warpgroups=1` for the `E ∈ [16.5, 32.5]` case is a
quirk (others use 2); we keep it for parity. It could affect AMD-side
scheduling later but has not been measured.

---

## 14. Probe / debugging idioms

The most generally useful debugging patterns built during the FP4 work.

### Identity-matrix probe

```python
def probe_A_with_B_eye_partial():
    # B = block-identity along K<16: B[i, j]=1 if j==i else 0
    # Then C[m, n] = sum_k A[m,k] * B[n,k] = A[m, n]   for n<16
    # If C[:, :16] == A[:, :16], A's lane mapping is correct.
```

Reduces the validation question to a checkable identity: kernel output
should equal a slice of the input.

### Per-position probe

```python
def find_active_ks(launcher) -> list[int]:
    actives = []
    for k_pos in range(128):
        # set ONLY k_pos in both A and B; check if C[0,0] fires
        ...
```

Reverse-engineers the actual MFMA K-coverage. This is what caught the
FP8×FP4 mixed-mode degeneracy — a constant-fill probe sums over all K
and hides the issue, but a per-position probe reveals exactly which K
positions actually contribute.

### Tid-dump kernel

```python
@flyc.kernel
def tiddump(out):
    tid = fx.thread_idx.x
    div_i32 = arith.index_cast(T.i32, tid // fx.Index(16))
    mod_i32 = arith.index_cast(T.i32, tid % fx.Index(16))
    buffer_ops.buffer_store(div_i32, rsrc, tid * fx.Index(8))
    buffer_ops.buffer_store(mod_i32, rsrc, tid * fx.Index(8) + fx.Index(4))
```

Verifies lane indexing arithmetic by writing each lane's computed values
to a per-lane slot in global memory. Caught the "all 64 lanes are alive,
indexing math is fine" diagnosis that ruled out lane-decomposition bugs
during the byte-vs-element offset hunt.

### `FLYDSL_DUMP_IR=1`

Set the env var before running a kernel; FlyDSL dumps the MLIR after
each pipeline pass to `~/.flydsl/debug/<kernel_name>_<id>/`. Inspecting
`00_origin.mlir` is the fastest way to confirm an offset / type / bitcast
is what you intended. The byte-vs-element offset bug was diagnosed by
reading line 43-44 of `00_origin.mlir` and seeing
`arith.muli %16, %c4_i32` — exposing the auto-multiply that broke
addressing.

---

## 15. Summary mental model

When writing a FlyDSL kernel:

1. **Choose the tile shape** from the CDNA4 MFMA set (16×16×128 or
   32×32×64).
2. **Compute lane decomposition**: `lane = mn_idx + k_lane * 16` for the
   16×16×128 MFMA.
3. **For loads**: per-lane byte offset = (row stride × global row) +
   (k_lane × per-lane K-bytes); divide by `dtype` size for element units.
4. **For stores**: per-lane writes 4 fp32 → C[k_lane*4 + i, mn_idx] for
   i = 0..3; element units.
5. **Use `max_size=True`** during development to avoid silent OOB drops.
6. **Test at multiple scales** with identity probes before random data,
   to isolate "lane mapping wrong" from "operand packing wrong" from
   "MFMA semantics misunderstood".

---

## Running the tests

```bash
# Phase 0 (CPU only, no FlyDSL/GPU needed)
cd /home/zhenchen/projects/aiter/aiter/ops/flydsl/kernels
python -m unittest mega_moe.test_workspace_scheduler -v

# Phase 1 step 1 — single FP8×FP8 16×16×128 tile
HIP_VISIBLE_DEVICES=3 python -m mega_moe._phase1_step1_single_tile

# Phase 1 step 1 identity probes
HIP_VISIBLE_DEVICES=3 python -m mega_moe._phase1_step1_probe

# Phase 1 step 2d — single FP4×FP4 tile
HIP_VISIBLE_DEVICES=3 python -m mega_moe._phase1_step2d_fp4fp4

# Phase 1 step 3 — K-loop accumulation (FP4×FP4)
HIP_VISIBLE_DEVICES=3 python -m mega_moe._phase1_step3_kloop

# Phase 1 step 4 — 2D-grid grouped GEMM
HIP_VISIBLE_DEVICES=3 python -m mega_moe._phase1_step4_grouped

# Phase 1 step 5 — persistent CTA scheduling
HIP_VISIBLE_DEVICES=3 python -m mega_moe._phase1_step5_persistent
```

Set `FLYDSL_DUMP_IR=1` before any of the above to dump per-pass MLIR.
Add `FLYDSL_RUNTIME_ENABLE_CACHE=0` when iterating on kernel code to
disable the on-disk JIT cache (in-memory cache stays active).

---

## File index

| File | Status | What it teaches |
|---|---|---|
| `workspace.py` | Phase 0 ✅ | dataclass byte-layout port, derived fields |
| `heuristics.py` | Phase 0 ✅ | Step-function block_m heuristic |
| `scheduler.py` | Phase 0 ✅ | Single-exit state-machine for device port |
| `__init__.py` | Phase 0 ✅ | Public API surface |
| `test_workspace_scheduler.py` | Phase 0 ✅ | 22 CPU tests covering layout + scheduler |
| `_phase1_step1_single_tile.py` | ✅ | Smallest viable scaled-MFMA, FP8×FP8 |
| `_phase1_step1_probe.py` | ✅ | Identity-matrix probe pattern |
| `_phase1_step1_tiddump.py` | ✅ | Lane-activity diagnostic |
| `_phase1_step2_fp4_b.py` | Archived ❌ | FP8×FP4 mixed mode degenerate (kept as counter-example) |
| `_phase1_step2b_multimfma.py` | Archived | opsel sweep (proves opsel does not extend K coverage) |
| `_phase1_step2c_kshift.py` | Archived | byte-shift sweep (8 K added per call but lane 3 OOB) |
| `_phase1_step2d_fp4fp4.py` | ✅ | FP4×FP4 verified at K=128 |
| `_phase1_step3_kloop.py` | ✅ | K-loop accumulation across multiple K=128 chunks |
| `_phase1_step4_grouped.py` | ✅ | 2D-grid grouped GEMM, multi-expert |
| `_phase1_step5_persistent.py` | ✅ | Persistent CTA scheduling, simplest form |

For the broader migration plan and known issues, see
`/home/zhenchen/projects/daily-digest/kernels/megamoe.md`.
