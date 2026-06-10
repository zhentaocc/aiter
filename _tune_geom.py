"""v2 tile autotuner: sweep (block_m, block_n, n_waves) per shape, correctness-gate,
bench with CUDA events, report the fastest geometry. Also benches Triton for context.

Run inside a flydsl-0.1.3.1 container with idle GPUs. Repo root via AITER_ROOT
env (default /aiter_src).
"""
import sys, os, types, importlib.util, itertools, gc
import torch

ROOT = os.environ.get("AITER_ROOT", "/aiter_src")
_pkg_root = f"{ROOT}/aiter/ops/flydsl/kernels"
_pkg = types.ModuleType("_kpkg"); _pkg.__path__ = [_pkg_root]; sys.modules["_kpkg"] = _pkg
def _load(m, fn):
    spec = importlib.util.spec_from_file_location(f"_kpkg.{m}", f"{_pkg_root}/{fn}")
    mod = importlib.util.module_from_spec(spec); sys.modules[f"_kpkg.{m}"] = mod
    spec.loader.exec_module(mod); return mod
_load("tensor_shim", "tensor_shim.py")
v2 = _load("batched_gemm_fp8_blockwise_flydsl_v2", "batched_gemm_fp8_blockwise_flydsl_v2.py")

trit = None
try:
    spec_t = importlib.util.spec_from_file_location("trit",
        f"{ROOT}/aiter/ops/triton/_triton_kernels/gemm/batched/batched_gemm_fp8_blockwise.py")
    trit = importlib.util.module_from_spec(spec_t); spec_t.loader.exec_module(trit)
except Exception as e:
    print("Triton load failed:", str(e)[:80])


def gpu_time(fn, iters=30, warmup=10):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    e = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for a, b in zip(s, e):
        a.record(); fn(); b.record()
    torch.cuda.synchronize()
    return sorted([a.elapsed_time(b)*1000 for a, b in zip(s, e)])[iters//2]


def oracle(A, W, A_s, W_s):
    B, M, K = A.shape; _, N, _ = W.shape; BL = 128
    a = (A.to(torch.float32).view(B, M, K//BL, BL) * A_s.unsqueeze(-1)).view(B, M, K)
    w = (W.to(torch.float32).view(B, N//BL, BL, K//BL, BL) *
         W_s.view(B, N//BL, 1, K//BL, 1)).view(B, N, K)
    return torch.bmm(a, w.transpose(1, 2)).to(torch.bfloat16)


def valid_configs(M, N):
    cfgs = []
    for bm, bn, nw in itertools.product([64, 128, 256], [64, 128], [2, 4, 8]):
        if M % bm or N % bn: continue
        if bn % nw: continue
        npw = bn // nw
        if npw < 16 or npw % 16: continue
        if bm % (nw * 8): continue
        if bm % 16: continue
        # crude VGPR guard: skip configs that obviously blow the 256 budget
        m_sub, n_sub = bm // 16, npw // 16
        acc_vgpr = m_sub * n_sub * 4
        if acc_vgpr > 160: continue
        cfgs.append((bm, bn, nw))
    return cfgs


SHAPES = [
    (8, 1024, 1024, 4096),
    (8, 4096, 1024, 4096),
    (8, 8192, 1024, 4096),
]
if len(sys.argv) > 1:
    # allow "B,M,N,K;B,M,N,K" override
    SHAPES = [tuple(int(x) for x in s.split(",")) for s in sys.argv[1].split(";")]

best_map = {}
for (B, M, N, K) in SHAPES:
    torch.manual_seed(B*M+N+K)
    A = (torch.randn(B, M, K, device='cuda')*0.5).clamp(-8,8).to(torch.float8_e4m3fn)
    W = (torch.randn(B, N, K, device='cuda')*0.5).clamp(-8,8).to(torch.float8_e4m3fn)
    A_s = (2.0**torch.randint(-2,2,(B,M,K//128),device='cuda')).float()
    W_s = (2.0**torch.randint(-2,2,(B,N//128,K//128),device='cuda')).float()
    flops = 2*B*M*N*K
    ref = oracle(A, W, A_s, W_s)
    print(f"\n=== shape ({B},{M},{N},{K}) ===")
    results = []
    for (bm, bn, nw) in valid_configs(M, N):
        try:
            out = v2.flydsl_batched_gemm_fp8_blockwise_v2(A, W, A_s, W_s,
                    block_m=bm, block_n=bn, n_waves=nw)
            err = (out.float()-ref.float()).abs().max().item()
            if err > 0.5:
                print(f"  bm={bm:3} bn={bn:3} nw={nw}  CORRECTNESS FAIL err={err:.3f}")
                continue
            us = gpu_time(lambda: v2.flydsl_batched_gemm_fp8_blockwise_v2(
                A, W, A_s, W_s, block_m=bm, block_n=bn, n_waves=nw))
            tf = flops/(us*1e-6)/1e12
            results.append((us, bm, bn, nw, tf))
            print(f"  bm={bm:3} bn={bn:3} nw={nw}  {us:8.1f} us  {tf:7.0f} TF")
        except Exception as e:
            print(f"  bm={bm:3} bn={bn:3} nw={nw}  EXC {type(e).__name__}: {str(e)[:70]}")
    tri_us = float('nan')
    if trit is not None:
        try: tri_us = gpu_time(lambda: trit.triton_batched_gemm_fp8_blockwise(A, W, A_s, W_s))
        except Exception as e: print("  triton fail", str(e)[:70])
    results.sort()
    if results:
        bus, bm, bn, nw, btf = results[0]
        best_map[(B,M,N,K)] = (bm, bn, nw)
        print(f"  BEST: bm={bm} bn={bn} nw={nw}  {bus:.1f}us {btf:.0f}TF   "
              f"triton={tri_us:.1f}us   v2best/tri={bus/tri_us:.2f}x")
    del A, W, A_s, W_s, ref; gc.collect(); torch.cuda.empty_cache()

print("\n=== BEST MAP ===")
for k, v in best_map.items():
    print(f"  {k}: block_m={v[0]} block_n={v[1]} n_waves={v[2]}")
