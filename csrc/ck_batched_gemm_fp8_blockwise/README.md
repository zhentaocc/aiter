# CK Batched GEMM FP8 Block-wise (DeepSeek V4 `wo_a`)

CK port of `aiter.batched_gemm_fp8_blockwise`, mirroring the
`ck_batched_gemm_a8w8` directory structure and the
`ck_gemm_a8w8_blockscale` block-scale logic.

## Contract

```
A        [B, M, K] fp8_e4m3fn
W        [B, N, K] fp8_e4m3fn
A_scale  [B, M, K/128] fp32      -- per-row, per-128k-block
W_scale  [B, N/128, K/128] fp32  -- per (128n, 128k) block
Out      [B, M, N] bf16 (or fp16)
K, N must be multiples of 128.
```

Equivalent to:
```python
deep_gemm.fp8_einsum("bmk,bnk->bmn",
                     (A, A_scale), (W, W_scale), Out, recipe=(1, 1, 128))
```

## Layout

```
ck_batched_gemm_fp8_blockwise/
├── README.md                                      (this file)
├── batched_gemm_fp8_blockwise.cu                  -- production dispatcher (heuristic + lookup)
├── batched_gemm_fp8_blockwise_tune.cu             -- tune entry: kernel-by-id selector
├── batched_gemm_fp8_blockwise_tune.py             -- tune driver, sweeps candidate_kernels_dict
├── batched_gemm_fp8_blockwise_instance.py         -- KernelInstance dataclass + candidate kernels
├── gen_instances.py                               -- codegen: instance .cu, lookup.h, manifest.h
├── include/
│   ├── batched_gemm_fp8_blockwise.h               -- public header
│   └── batched_gemm_fp8_blockwise_common.cuh      -- CK template + host B-loop wrapper
└── instances/                                     -- (autogen) one .cpp per (tile, dtype)

# Generated next to gen_instances.py at JIT/build time:
#   batched_gemm_fp8_blockwise_lookup.h
#   batched_gemm_fp8_blockwise_manifest.h
#   impl/<kernel_name>.cuh
```

## Implementation note: B is a host-side loop

Composable Kernel ships `DeviceGemmMultiD_ABScale_Xdl_CShuffle_V3` (the
non-batched FP8 AB-scale device op used by `ck_gemm_a8w8_blockscale`)
but does **not** ship a `DeviceBatchedGemmMultiD_ABScale_Xdl_CShuffle_V3`
equivalent.  We implement the B dimension as an outer loop in the host
wrapper -- one `MakeArgument` + `invoker.Run` per batch slice on the
same hipStream (see `include/batched_gemm_fp8_blockwise_common.cuh:batched_gemm_fp8_blockwise_impl`).

For the wo_a use case (`B = num_groups <= 16`) this adds
`~B * ~few-microseconds` of dispatch overhead, which is small relative
to the per-call HBM traffic for V4-Flash decode shapes.  When CK adds
a true batched ABScale device op, swap the device-op alias in
`include/batched_gemm_fp8_blockwise_common.cuh` and remove the loop --
the rest of this directory does not change.

## Build

Wired into `aiter/jit/optCompilerConfig.json` as two modules:

  * `module_batched_gemm_fp8_blockwise`      (production dispatcher)
  * `module_batched_gemm_fp8_blockwise_tune` (tune harness)

Both are JIT-compiled on first use of `aiter.batched_gemm_fp8_blockwise`
or the loader entry points in
`aiter/aiter/ops/_ck_batched_gemm_fp8_blockwise_loader.py`.

To pre-build:
```bash
PREBUILD_KERNELS=1 python setup.py develop
```

## Tune

```bash
# 1) Add (B, M, N, K) shapes to the untuned CSV:
cat aiter/configs/fp8_blockwise_untuned_batched_gemm.csv

# 2) Run the sweep (will hipcc-compile every candidate per shape):
python3 csrc/ck_batched_gemm_fp8_blockwise/batched_gemm_fp8_blockwise_tune.py \
    -i aiter/configs/fp8_blockwise_untuned_batched_gemm.csv \
    -o aiter/configs/fp8_blockwise_tuned_batched_gemm.csv

# 3) Rebuild with the tuned table:
AITER_REBUILD=1 python op_tests/test_batched_gemm_fp8_blockwise.py
```

## Status

| Piece | State |
|---|---|
| Kernel template (CK ABScale + B-loop) | **Written** |
| Dispatcher (heuristic + lookup) | **Written** |
| Codegen (gen_instances.py mirrors blockscale) | **Written** |
| Tune harness (.cu + .py) | **Written** |
| 19 candidate tile instances | **Seeded** (same set as `ck_gemm_a8w8_blockscale`; gfx9-family validated) |
| pybind binding + JIT config | **Wired** |
| Python loader | **Written** (`aiter/ops/_ck_batched_gemm_fp8_blockwise_loader.py`) |
| Wrapper integration (`backend="ck"`) | **Wired** in `aiter/ops/batched_gemm_op_fp8_blockwise.py` |
| Compile validation (hipcc) | **Not run on this host** -- requires the `rocm/atom-dev:vllm-latest` image or equivalent |
| Tune sweep (hours of hipcc) | **Not run** -- caller's choice when to invest the time |

Compile validation should happen inside the docker image where hipcc and
the matching CK headers are guaranteed to be present:

```bash
docker run --rm --device /dev/kfd --device /dev/dri \
    -v $PWD:/workspace/aiter \
    rocm/atom-dev:vllm-latest \
    bash -c "cd /workspace/aiter && \
             AITER_REBUILD=1 python -c 'import aiter; \
             import torch; B,M,N,K=4,16,256,128; \
             A=torch.randn(B,M,K,device=\"cuda\").to(torch.float8_e4m3fn); \
             W=torch.randn(B,N,K,device=\"cuda\").to(torch.float8_e4m3fn); \
             As=torch.rand(B,M,K//128,device=\"cuda\"); \
             Ws=torch.rand(B,N//128,K//128,device=\"cuda\"); \
             out=aiter.batched_gemm_fp8_blockwise(A,W,As,Ws,backend=\"ck\"); \
             print(out.shape)'"
```
