# Add CK FP8 block-scale batched GEMM (DeepSeek V4 `wo_a`)

## Summary

Adds a CK FP8 block-scale batched GEMM kernel for DeepSeek V4's `wo_a` output
projection (the operation `sglang` PR #23608 currently sources from
`deep_gemm.fp8_einsum` with recipe `(1, 1, 128)`). Public API:

```python
aiter.batched_gemm_fp8_blockscale(A, W, A_scale, W_scale, out=None, backend="auto")
aiter.batched_gemm_fp8_blockscale_einsum(equation, A, A_scale, W, W_scale, ...)
aiter.convert_scales_to_ue8m0(scales_fp32) -> uint8   # load-time helper
```

Backend selection:
- `"auto"` (default): CK when `M >= 128 && M % 128 == 0 && N % 128 == 0`, else
  torch reference.
- `"ck"`: force the CK path (raises on unsupported shape).
- `"torch"`: dequant + bf16 bmm oracle.

Scale dtype: both `A_scale` and `W_scale` must share dtype (fp32 *or* uint8
UE8M0). u8 is the production path — pre-convert at model load with
`convert_scales_to_ue8m0()`, then the kernel does a one-time GPU-side
exp2 cast (W_scale conversion is weak-ref cached so weights pay it once
per tensor lifetime).

## What's included

| Area | File(s) |
|---|---|
| CK kernel | `csrc/ck_batched_gemm_fp8_blockscale/` (instance dataclass, gen_instances.py codegen, dispatcher .cu, tune .cu, common.cuh wrapping `DeviceGemmMultiD_ABScale_Xdl_CShuffle_V3`) |
| Pybind | `csrc/pybind/batched_gemm_fp8_blockscale{_tune}_pybind.cu` |
| Python loader | `aiter/ops/_ck_batched_gemm_fp8_blockscale_loader.py` (u8↔fp32 scale conversion + caching, dtype-equality enforcement) |
| Public dispatcher | `aiter/ops/batched_gemm_op_fp8_blockscale.py` (auto/ck/torch backend + einsum entry) |
| JIT registration | `aiter/jit/optCompilerConfig.json`, `aiter/jit/core.py` (`AITER_CONFIG_FP8_BLOCKSCALE_BATCHED_GEMM`) |
| Pybind macros | `csrc/include/rocm_ops.hpp` |
| Top-level export | `aiter/__init__.py` |
| Tuned configs | `aiter/configs/fp8_blockscale_{tuned,untuned}_batched_gemm.csv` |

## Tune space

| Axis | Values |
|---|---|
| Kernel id | 25 candidates (MPerBlock ∈ {16,32,64,128,256}, NPerBlock ∈ {64,128,256}, KPerBlock ∈ {128,256}, BlockSize ∈ {128,256}, Pipeline Intrawave v1/v3) |
| splitK | Plumbed end-to-end (gen_instances → impl → tune) but **enforced to 1** until the vendored CK exposes `SetKBatch()` on the ABScale V3 device class. Untuned-CSV `splitK` column reserved for future. |
| Pipeline Sched | Intrawave only — Interwave specialization not implemented for ABScale (`BlockwiseGemmXdlops_pipeline_v1_ab_scale<Interwave, ...>` is undefined in upstream CK). |

24 DSv4-wo_a shapes shipped in `untuned.csv` (B ∈ {1, 2, 8, 16} × M ∈ {16, 32,
48, 64, 96, 128, 256, 512, 1024, 2048, 4096}, N=1024, K=4096), tuned in <4 min
on 7×MI355X.

## Performance (MI355X, bf16 output, fp8 input)

Measured by `op_tests/op_benchmarks/hip/bench_batched_gemm_fp8_blockscale.py
--preset dsv4` (44 shapes).

| Metric | min | **median** | max |
|---|---|---|---|
| Tune speedup (kid=0 default → tuned best) | 0.91× | **2.01×** | 2.71× |
| Tuned CK vs torch dequant+bmm oracle | 2.05× | **3.44×** | 5.38× |

Headline cells (production DSv4 wo_a per-rank shapes):

| Config | (B, M) | torch us | CK tuned us | speedup |
|---|---|---|---|---|
| Flash TP=8, C=128 | (1, 128) | 50.6 | **12.2** | 4.15× |
| Flash TP=8, C=4096 | (1, 4096) | 157.0 | **30.9** | 5.08× |
| Pro TP=8, C=128 | (2, 128) | 69.3 | **19.1** | 3.62× |
| Pro TP=8, C=4096 | (2, 4096) | 286.9 | **64.5** | 4.45× |
| Pro TP=1, C=4096 | (16, 4096) | 2036.5 | **507.4** | 4.01× |

## Accuracy

`op_tests/test_batched_gemm_fp8_blockscale.py` validates the CK output against
the torch dequant + `torch.bmm` oracle on all DSv4 shapes:
- `atol = 2e-2`, `rtol = 2e-2` for K ≤ 2048
- `atol = 4e-2` for K = 4096 (more accumulation noise in BF16)

Max observed absolute error: 0.13 (well below the 0.5 BF16 quantization noise
floor). All shapes pass.

## Build constraints discovered

When wiring this on top of upstream CK (rocm-7.1.1 system headers on MI355X):

1. **`CK_USE_OCP_FP8=1` must be defined** in module's `flags_extra_hip` —
   without it, vendored `ck::f8_t` resolves to `f8_fnuz_t` (gfx940-only),
   and the gfx950 ABScale V3 device class becomes abstract.
2. **Pipeline v3 requires** `KPerBlock / Scale_Block_K == 1` AND
   `NPerBlock / Scale_Block_N ≤ 1`. Larger-tile variants (kid 21/22 in earlier
   draft) failed compilation and were dropped.
3. **Interwave scheduler is unsupported** on the ABScale path in upstream CK
   (`BlockHasHotloop` missing on the v1 specialization).
4. **Vendored CK in `3rdparty/composable_kernel`** has only `f8_fnuz_t`
   implementation; system CK at `/opt/rocm-7.1.1/include/ck/` has both. Build
   prefers system CK by renaming the vendored dir to `.disabled` (or removing
   the `-I` include for it). Tracked as a follow-up to update the submodule
   pin.

## Reproducing the bench

In container with `aiter` source mounted at `/aiter_src`:

```bash
pip uninstall -y amd-aiter aiter
pip install -U triton   # if base image has triton <3.6.0
mv /workspace/aiter_ck/3rdparty/composable_kernel /workspace/aiter_ck/3rdparty/composable_kernel.disabled
cd /workspace/aiter_ck

# Build CK kernels (first call triggers JIT, ~2 min for tune-module)
PYTHONPATH=/workspace/aiter_ck python3 -c "
import torch, aiter
aiter.batched_gemm_fp8_blockscale(
  torch.zeros(1, 128, 4096, dtype=torch.float8_e4m3fn, device='cuda'),
  torch.zeros(1, 1024, 4096, dtype=torch.float8_e4m3fn, device='cuda'),
  torch.ones(1, 128, 32, dtype=torch.float32, device='cuda'),
  torch.ones(1, 8, 32, dtype=torch.float32, device='cuda'),
  backend='ck')
"

# Bench
PYTHONPATH=/workspace/aiter_ck python3 \
    op_tests/op_benchmarks/hip/bench_batched_gemm_fp8_blockscale.py --preset dsv4

# Re-tune (parallel over N GPUs)
PYTHONPATH=/workspace/aiter_ck bash op_tests/run_tune_parallel.sh \
    aiter/configs/fp8_blockscale_untuned_batched_gemm.csv \
    aiter/configs/fp8_blockscale_tuned_batched_gemm.csv "0,1,2,3,4,5,6,7" 20
```

## Test plan

- [x] `pytest op_tests/test_batched_gemm_fp8_blockscale.py -v` passes on
  MI355X (rocm-7.1.1 image)
- [x] `op_tests/op_benchmarks/hip/bench_batched_gemm_fp8_blockscale.py
  --preset dsv4` produces 44 rows, all max_err < 0.5
- [x] `op_tests/op_benchmarks/hip/bench_batched_gemm_fp8_blockscale.py
  --preset smoke --backend ck` runs CK path explicitly
- [x] Autotune over 44 shapes completes in <4 min on 7×MI355X
- [ ] Sibling kernel `gemm_a8w8_blockscale` still builds (regression check)
- [ ] gfx942 build does not regress (need to test with `CK_USE_OCP_FP8=0`
  conditional once we have a gfx942 machine)

## Out of scope (separate branches / future work)

- The decode-optimised flydsl kernel lives on the `wo_a_fp8_blockwise` branch;
  the auto dispatcher there picks flydsl for small-M decode and CK for large-M
  prefill.
- splitK > 1: needs newer CK with `SetKBatch()` on the ABScale V3 class.
- Native MFMA scale-slot variant (skip the post-MFMA fp32 multiply on the
  accumulator): tracked as TODO; would require a custom CK template or moving
  to MX-FP8 (recipe (1,1,32)) on gfx950.
- vllm / ATOM integration patches live on the flydsl branch alongside the
  decode kernel.
