# FlyDSL MegaMOE Kernel 教程

本目录下 Phase 0 + Phase 1 各 kernel 的逐步讲解，重点覆盖**用到的每一个
FlyDSL API 及其原因**，以及每个超参数的取值依据。如果你要扩展本目录的
kernel，或在 AMD CDNA4（gfx950 / MI355X）上写自己的 FlyDSL 代码，
请参考本文。

## 阅读指南

第 1–9 节讲设备端 API（在 `@flyc.kernel` 内运行的部分）。
第 10 节是超参数速查表。
第 11 节讲 host 端写法（Phase 0 字节布局 / 调度器）。
第 12–14 节讲启发式与捕获真实 porting bug 的调试探针套路。

教程围绕已有文件展开；行号引用 `_phase1_step*.py` 系列。

---

## 1. 两种装饰器 —— `@flyc.kernel` vs `@flyc.jit`

本目录每个 kernel 都是同一个两层结构：

```python
@flyc.kernel
def my_kernel(a: fx.Tensor, b: fx.Tensor, c: fx.Tensor):
    # 设备端代码 —— 被 trace 成 MLIR，运行在 GPU 上
    ...

@flyc.jit
def my_launcher(a: fx.Tensor, b: fx.Tensor, c: fx.Tensor,
                stream: fx.Stream = fx.Stream(None)):
    # Host 端代码 —— 设置 launch grid
    my_kernel(a, b, c).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)
```

- `@flyc.kernel` 标记的函数 body 会被 **trace 成 MLIR**，而不是当 Python
  跑。函数内每个 `+`、`*`、`//`、`%` 作用在 `fx.Index` 上时都会发出一个
  `arith.addi` / `muli` / `divui` / `remui` 操作。Python 控制流
  （`for`、`if`）也会发出 MLIR（`scf.for`、`scf.if`），而不是原生执行。
- `@flyc.jit` 标记 host 端的 launcher。它 JIT-编译 `my_kernel`（按签名
  缓存），然后调用 `.launch(grid, block, stream)`。第一次调用付编译开销，
  后续调用复用缓存的二进制。

**为什么分两层**：把 "编译什么" 和 "怎么 launch" 解耦。Kernel 是参数化的，
launcher 根据输入 shape 固定 grid/block。可以用多个 launcher 包同一个
kernel，对应不同 launch 形状，无需重新编译。

---

## 2. Tensor 生命周期 —— `from_dlpack` + `mark_layout_dynamic`

来自 `_phase1_step1_single_tile.py:main()`：

```python
A = A_f32.to(torch.float8_e4m3fn).contiguous()             # torch fp8 tensor
A_dl = flyc.from_dlpack(A).mark_layout_dynamic(
    leading_dim=1, divisibility=128
)
```

- `flyc.from_dlpack(A)` 通过 DLPack 协议（零拷贝）把 torch tensor 包成
  `fx.Tensor`。
- `mark_layout_dynamic(leading_dim=1, divisibility=128)` 告诉 JIT：
  - **`leading_dim=1`** —— **stride-1**（最内层连续）维度是 dim 1。
    对于行主序 `[16, 128]` tensor，dim 1 连续（stride 1 字节），dim 0
    stride 是 128 字节。常见 bug：传 `leading_dim=0` 会报
    `Leading dimension must have stride 1`，因为 dim 0 stride 是 128
    而不是 1。
  - **`divisibility=128`** —— leading dim 的 *长度* 能被 128 整除。这能
    解锁向量化 load（dwordx4 = 16 字节），因为编译器知道 row size 是 16
    字节的倍数。

**为什么 divisibility 提示重要**：没有它编译器就要为部分向量 load 生成标量
fallback 路径。有了它就能干净地发出 `buffer_load_dwordx4`。

---

## 3. Buffer 描述符 —— `create_buffer_resource`

```python
a_rsrc = buffer_ops.create_buffer_resource(a_ptr, max_size=True)
```

生成一个 128 位 AMD V# (vertex/buffer) 描述符 —— IR 中表现为 `<8>` ptr。
它打包了：

- 64 位基地址
- 32 位 num_records（边界检查上限，元素单位）
- 32 位 stride / format / cache flags

`max_size=True` 把 num_records 设为 `0xFFFFFFFF`（4 GB），等同关闭边界
检查。`max_size=False` 用 tensor 的实际元素数或传入的 `num_records_bytes`。

Phase 1 各 kernel 在开发期都用 `max_size=True`。边界检查会**静默丢弃**
OOB store —— 这把 byte-vs-element 偏移 bug 隐藏了好几轮调试（64 lane 中
56 个地址越界，描述符直接丢，留下原始零，看起来像 lane mapping bug）。

---

## 4. 元素 vs 字节偏移陷阱

这是 FlyDSL ABI 最重要的 gotcha：

```python
a_lo = buffer_ops.buffer_load(a_rsrc, a_dword_off, vec_width=4, dtype=T.i32)
```

`offset` 参数是 **`dtype` 元素单位，不是字节**。Lowering 自动乘
`sizeof(dtype)`（i32 是 4，i8 是 1，等等）。

所以想把 A 的字节 128 当 i32 读：

| 错误写法 | 正确写法 |
|---|---|
| `offset = 128` | `offset = 128 / 4 = 32` |
| 硬件字节地址 `128 * 4 = 512` (OOB) | 硬件字节地址 `32 * 4 = 128` ✓ |

这就是为什么 Phase 1 各 kernel 的地址数学都显式除以 4：

```python
a_dwords_per_row = K // 8     # K fp4 / 2-fp4-per-byte / 4-byte-per-dword = K/8
```

（FP4 每字节装 2 个元素，所以 K 个 fp4 = K/2 字节 = K/8 dword。）

`buffer_store` 有一个 `offset_is_bytes=True` 标志可以让你按原生字节传；
`buffer_load` 没有 —— 永远是元素单位。

---

## 5. MFMA 调用 —— 全部计算的核心

```python
acc = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
    T.f32x4,                                     # 返回类型
    [a128, b128, acc_in,                          # operand
     0, 4,                                        # cbsz=FP8, blgp=FP4
     0, 0x7F7F7F7F,                               # opselA=0, scaleA=1.0
     0, 0x7F7F7F7F],                              # opselB=0, scaleB=1.0
)
```

### 返回类型

`T.f32x4` 是 `@property`，**不是** `@method`。写 `T.f32x4()` 会试着调用
返回的 `VectorType` 实例并报 `'VectorType' object is not callable`。
16×16 MFMA 的返回类型永远是 `vector<4xf32>` —— 每个 lane 返回 4 个 fp32
累加器值。

### 单 lane operand 布局

（来自 `lib/Dialect/FlyROCDL/CDNA4/MmaAtom.cpp` 中的 CDNA4 atom 定义
`getThrValLayoutAB`）

64 lane 拆为 16 (MN 轴) × 4 (K 轴)：`lane = mn_idx + k_lane * 16`

| Operand | Vector | 单 lane 持有 |
|---|---|---|
| A | `vector<8xi32>` | 32 个 fp8 元素（32 K 位置） |
| B (FP4 模式) | `vector<4xi32>` | 32 个 fp4 元素（32 K 位置） |
| B (FP8 模式) | `vector<8xi32>` | 32 个 fp8 元素（32 K 位置） |
| C 累加器 | `vector<4xf32>` | 4 个输出位于 C[(lane/16)*4 + 0..3, lane%16] |

### `cbsz` / `blgp` 编码

（来自 `MmaAtom.cpp:134-146`）

| 编码 | 类型 |
|---|---|
| 0 | FP8 e4m3fn |
| 1 | FP8 e5m2 |
| 2 | FP6 e2m3fn |
| 3 | FP6 e3m2fn |
| 4 | FP4 e2m1fn |

这些字段在经典 MFMA 中是 broadcast 控制，scaled 变种把它们重新用作 dtype
编码。

### 比例因子格式

`0x7F7F7F7F` = E8M0ㅤ≈ 1.0，覆盖全部 4 个 sub-window。E8M0 是 8 位仅指数
浮点；偏移 127（`0x7F`），所以 `0x7F` = 1.0。i32 打包 4 个这样的字节，
每个对应一个 K=32 sub-window。`opsel ∈ [0, 3]` 选择每个 scale byte 应用
到哪个 sub-window。`0x7F7F7F7F` = 不缩放。

### 实测行为

| 模式 | cbsz, blgp | 单次调用 K 覆盖 | 状态 |
|---|---|---|---|
| FP8 × FP8 | 0, 0 | 128/128 ✅ | 已验证 |
| FP4 × FP4 | 4, 4 | 128/128 ✅ | 已验证 |
| FP8 × FP4 混合 | 0, 4 | **32/128** ❌ | Degenerate；需要不同的 operand 打包 |

混合模式的发现详见 `daily-digest/kernels/megamoe.md` §12。

---

## 6. 单 lane 寻址数学

来自 `_phase1_step5_persistent.py`：

```python
row = lane % fx.Index(16)
k_lane = lane // fx.Index(16)
a_base = (expert_idx * fx.Index(a_dwords_per_expert)
          + (m_tile_idx * fx.Index(16) + row) * fx.Index(a_dwords_per_row)
          + k_lane * fx.Index(4))
```

逐项解读：

- `row = lane % 16` —— 该 lane 读 16 个 MN 行中的哪一行（对应 MFMA layout
  的 `mn_idx`）。
- `k_lane = lane // 16` —— 4 个 K-group 中的哪一个（对应 `k_lane_idx`）。
- 三项相加：`expert 偏移 + tile 行偏移 + tile 内 K-chunk 偏移`。
- `(m_tile_idx * 16 + row)` 是 expert 内的全局行号。
- `* a_dwords_per_row` 推进到该行起始。
- `+ k_lane * 4` 是单 K-lane 偏移（每个 lane 读 4 个 i32 = 16 字节 = 32
  个 fp4，覆盖 32 个 K 位置）。

**为什么用 `fx.Index(16)` 而不是 `16`**：在 `fx.Index` 上做算术会发出
MLIR op；混用 Python int 需要显式转换。把 int 包成 `fx.Index(...)`
告诉编译器 "这是一个 index 值，请按 index 类型检查"。

---

## 7. 向量打包 —— `ir.VectorType.get` + `vector.from_elements`

```python
from flydsl._mlir import ir
from flydsl.expr import vector

v8i32 = ir.VectorType.get([8], ir.IntegerType.get_signless(32))
a128 = vector.from_elements(
    v8i32,
    [a_lo[0], a_lo[1], a_lo[2], a_lo[3], a_hi[0], a_hi[1], a_hi[2], a_hi[3]],
)
```

**为什么要降到原始 MLIR**：当前安装的 FlyDSL 暴露了 `T.i32x4` 但没有
`T.i32x8`。要构造 `vector<8xi32>`（MFMA A operand 宽度），就直接用 MLIR
Python binding 构造类型。

`vector.from_elements(type, [scalar_values])` 把 N 个标量打包成一个
向量。`a_lo[i]` 索引一个向量，返回第 i 个标量元素。

---

## 8. 循环 —— `range_constexpr` vs `fx.range`

```python
for chunk in range_constexpr(k_chunks):
    chunk_off = fx.Index(chunk * 16)
    ...
```

**`range_constexpr(N)`**：纯 Python 循环，**trace 时完全展开**。`chunk`
是 Python int，所以 `chunk * 16` 是 Python int 操作（再被 `fx.Index`
包起来）。每次迭代都在 MLIR 里发出一份 body 拷贝。当 N 是较小的编译期常量
（≤ ~16）且想展开时使用。

**`fx.range(start, stop, step, init=[carried_state])`**：发出 `scf.for`
op。循环边界可以是运行时值；carried 值显式串起来：

```python
for it, st in fx.range(0, loop_iters, 1, init=[acc, ptr]):
    new_acc = ...
    new_ptr = ...
    yield [new_acc, new_ptr]   # 通过最后赋值隐式 yield
```

Phase 1 各处用 `range_constexpr` 是因为 tile 数都是编译期已知。真实生产
kernel 处理运行时 K 时应该用 `fx.range`。

---

## 9. 初始值 —— `Vector.filled(N, val, NumericClass)`

```python
acc = fx.Vector.filled(4, 0.0, fx.Float32)
```

`Vector.filled` 要的是 **Numeric 类**（`fx.Float32`、`fx.Int32`），
**不是** **MLIR 类型**（`T.f32`、`T.i32`）。Numeric 类同时包含 MLIR
类型和 Python 转换逻辑（`dtype(0.0)` 构造一个带类型的常量）。传 MLIR
裸类型会在 trace 时报 `F32Type object is not callable`。

---

## 10. 超参数速查表

| 超参数 | 取值 | 原因 |
|---|---|---|
| MFMA tile 形状 | **16×16×128** | CDNA4 scaled MFMA 只有两种形状：16x16x128 和 32x32x64。16×16×128 的 M/N 粒度更细 → 更适合小 M 的 MoE expert tile。 |
| `BLOCK_K` | **128** | 与 MFMA K 对齐。DeepGEMM 也固定为 128。更小的话需要 fragment-K 累加。 |
| `BLOCK_N` | **128** | DeepGEMM 一律用 128；SF 格式和 weight preshuffle 都假设 128。 |
| `BLOCK_M` 步函数 | 16/32/64/96/128/192 | 直接移植 `csrc/jit_kernels/heuristics/mega_moe.hpp`。阈值（E[tokens/expert] ≤ 8.5/16.5/32.5/64.5/96.5）覆盖从 RL 长尾（E ≤ 8）到 prefill（E > 96）的路由分布。 |
| `num_sms` | MI355X 上 **128** | gfx950 有 128 个 CU；持久化 kernel 一个 CTA 对应一个 CU。 |
| `block=(64, 1, 1)` | 64 线程 | 一个 CDNA4 wavefront = 64 lane。单 wave kernel 简单清晰。 |
| `vec_width=4` (dwordx4 load) | 4 i32 = 16 字节 | AMD coalesced load 粒度。更小的 load (`dwordx2`、`dword`) 浪费带宽。 |
| `0x7F7F7F7F` | E8M0 ≈ 1.0 ×4 | 默认 scale = identity；4 个 K=32 sub-window 都 scale 1.0。 |
| `cbsz=0, blgp=0`（FP8×FP8） | K=128 全覆盖 | 纯模式 —— 单次调用覆盖完整 K。 |
| `cbsz=4, blgp=4`（FP4×FP4） | K=128 全覆盖 | 纯模式 —— 单次调用覆盖完整 K。 |
| `cbsz=0, blgp=4`（FP8×FP4 混合） | **degenerate** | 实测仅 32/128 K 位置被消费；需要不同的 operand 打包。 |
| `kNumStages`（流水深度） | 暂未使用 | Phase 1 step 5 还没有 LDS double-buffer；step 3b 会加 2-stage 流水。 |

---

## 11. 输出存储模式

```python
out_row_base = k_lane * fx.Index(4)
out_col = lane % fx.Index(16)
for i in range_constexpr(4):
    out_row = m_tile_idx * fx.Index(16) + out_row_base + fx.Index(i)
    c_off = c_expert_off + out_row * fx.Index(N) + (n_tile_idx * fx.Index(16) + out_col)
    buffer_ops.buffer_store(acc[i], c_rsrc, c_off)
```

每 lane 的 4 个 fp32 输出去到**同一列、纵向相邻的 4 行**。CDNA 16×16
MFMA 布局是：

> Lane (k_lane = 0..3, mn_idx = 0..15) 写入 `C[k_lane * 4 + 0..3, mn_idx]`。

每行 stride = N（fp32 列数）；元素单位偏移 = `row * N + col` —— 元素单位
而非字节单位（因为 store 偏移默认也是元素单位）。

---

## 12. Phase 0 host 端写法

`workspace.py` 和 `scheduler.py` 是带类型注解的纯 Python。无 FlyDSL，
无 GPU。两个值得借鉴的写法。

### `@dataclass` + `field(init=False)` 处理派生属性

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

`init=False` 把派生字段排除出 `__init__` 签名；`__post_init__` 在构造后
计算一次。

### 单出口 `_next_block` + 哨兵

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

围绕 FlyDSL 的"单出口控制流"规则设计 —— 即使这是 Python，用 `result`
哨兵 + 单 `return` 让设备端移植（`scf.while` + carried state）变成机械
工作。如果用早 `return`，设备端移植要大改。

---

## 13. 启发式表（`heuristics.py`）

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

**为什么是这张表**：DeepGEMM C++ `get_block_config_for_mega_moe` 的逐字
移植。阈值（8.5 / 16.5 / 32.5 / 64.5 / 96.5）是 NV 调过的，但反映了跨
硬件通用的 expert 路由分布：

- `E ≤ 8` → RL 长尾 rollout（大多数 expert 冷）
- `E ≤ 16` → EP=8 的小 batch 解码
- `E ≤ 32` → 中等 batch 解码
- `E > 96` → prefill 或大 EP

`E ∈ [16.5, 32.5]` 区间 `num_epilogue_warpgroups=1` 是一个特例（其他都是
2）；保留它是为了对照。这可能影响 AMD 端调度，但还没量化测过。

---

## 14. 探针 / 调试套路

FP4 工作中沉淀下来的最有用的几个调试模式。

### Identity-matrix 探针

```python
def probe_A_with_B_eye_partial():
    # B = K<16 上的 block 单位阵：B[i, j]=1 if j==i else 0
    # 那 C[m, n] = sum_k A[m,k] * B[n,k] = A[m, n]   for n<16
    # 如果 C[:, :16] == A[:, :16]，说明 A 的 lane mapping 正确。
```

把验证问题归约为可校验的等式：kernel 输出应等于输入的某个切片。

### 单位置探针

```python
def find_active_ks(launcher) -> list[int]:
    actives = []
    for k_pos in range(128):
        # A 和 B 中只设 k_pos 一个位置；看 C[0,0] 是否触发
        ...
```

反推 MFMA 实际的 K 覆盖。这就是发现 FP8×FP4 混合模式 degenerate 的关键
—— 常数填充探针把 K 全部加起来会掩盖问题，而单位置探针能精确显示哪些 K
位置真正贡献。

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

把每个 lane 算出的值写到 global memory 的对应 slot，验证 lane 索引
算术。在 byte-vs-element 偏移排查中确认了"64 lane 全部活动、索引数学
没问题"，排除了 lane 分解 bug。

### `FLYDSL_DUMP_IR=1`

跑 kernel 前设这个环境变量；FlyDSL 会在每个 pass 后把 MLIR dump 到
`~/.flydsl/debug/<kernel_name>_<id>/`。看 `00_origin.mlir` 是确认
"偏移 / 类型 / bitcast 是否符合预期" 最快的方式。byte-vs-element 偏移
bug 就是看 `00_origin.mlir` 第 43-44 行的 `arith.muli %16, %c4_i32`
诊断出来的 —— 直接看到 buffer_load 自动做了 ×4 偏移。

---

## 15. 心智模型小结

写 FlyDSL kernel 时：

1. **从 CDNA4 MFMA 集合里选 tile 形状**（16×16×128 或 32×32×64）。
2. **算 lane 分解**：16×16×128 MFMA 时 `lane = mn_idx + k_lane * 16`。
3. **load 时**：单 lane 字节偏移 = (行 stride × 全局行) + (k_lane ×
   单 lane K 字节)；除以 `dtype` 大小转成元素单位。
4. **store 时**：单 lane 写 4 fp32 → C[k_lane*4 + i, mn_idx]，
   i = 0..3；元素单位。
5. **开发期用 `max_size=True`**，避免 OOB store 静默丢失。
6. **多尺寸测**，先 identity 探针后随机数据，区分 "lane mapping 错"、
   "operand 打包错"、"MFMA 语义理解错"。

---

## 跑测试

```bash
# Phase 0（仅 CPU，不需 FlyDSL/GPU）
cd /home/zhenchen/projects/aiter/aiter/ops/flydsl/kernels
python -m unittest mega_moe.test_workspace_scheduler -v

# Phase 1 step 1 —— FP8×FP8 单 16×16×128 tile
HIP_VISIBLE_DEVICES=3 python -m mega_moe._phase1_step1_single_tile

# Phase 1 step 1 identity 探针
HIP_VISIBLE_DEVICES=3 python -m mega_moe._phase1_step1_probe

# Phase 1 step 2d —— FP4×FP4 单 tile
HIP_VISIBLE_DEVICES=3 python -m mega_moe._phase1_step2d_fp4fp4

# Phase 1 step 3 —— K-loop 累加（FP4×FP4）
HIP_VISIBLE_DEVICES=3 python -m mega_moe._phase1_step3_kloop

# Phase 1 step 4 —— 2D-grid grouped GEMM
HIP_VISIBLE_DEVICES=3 python -m mega_moe._phase1_step4_grouped

# Phase 1 step 5 —— 持久化 CTA 调度
HIP_VISIBLE_DEVICES=3 python -m mega_moe._phase1_step5_persistent
```

任意命令前加 `FLYDSL_DUMP_IR=1` 即可 dump 各 pass 的 MLIR。
迭代 kernel 代码时再加 `FLYDSL_RUNTIME_ENABLE_CACHE=0` 关闭磁盘 JIT
缓存（in-memory 缓存仍然激活）。

---

## 文件索引

| 文件 | 状态 | 教什么 |
|---|---|---|
| `workspace.py` | Phase 0 ✅ | dataclass 字节布局移植，派生字段 |
| `heuristics.py` | Phase 0 ✅ | block_m 步函数启发式 |
| `scheduler.py` | Phase 0 ✅ | 适配设备端移植的单出口状态机 |
| `__init__.py` | Phase 0 ✅ | 公共 API |
| `test_workspace_scheduler.py` | Phase 0 ✅ | 22 个 CPU 测试覆盖布局+调度器 |
| `_phase1_step1_single_tile.py` | ✅ | 最小可运行的 scaled MFMA，FP8×FP8 |
| `_phase1_step1_probe.py` | ✅ | Identity 探针套路 |
| `_phase1_step1_tiddump.py` | ✅ | Lane 活跃性诊断 |
| `_phase1_step2_fp4_b.py` | 归档 ❌ | FP8×FP4 混合 degenerate（保留作反例） |
| `_phase1_step2b_multimfma.py` | 归档 | opsel sweep（证明 opsel 不能扩展 K 覆盖） |
| `_phase1_step2c_kshift.py` | 归档 | byte-shift sweep（每次加 8 K，但 lane 3 OOB） |
| `_phase1_step2d_fp4fp4.py` | ✅ | FP4×FP4 K=128 验证 |
| `_phase1_step3_kloop.py` | ✅ | 跨多个 K=128 chunk 的 K-loop 累加 |
| `_phase1_step4_grouped.py` | ✅ | 2D-grid grouped GEMM，多 expert |
| `_phase1_step5_persistent.py` | ✅ | 最简形态的持久化 CTA 调度 |

更宏观的迁移规划与已知问题见
`/home/zhenchen/projects/daily-digest/kernels/megamoe.md`。
