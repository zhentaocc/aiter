"""Remote CK vs v2(best-tuned) side-by-side on idle GPU. CK branch uses
aiter.batched_gemm_a8w8_blockscale. Run in zhenchen_aiter_ck container."""
import sys, os, types, importlib.util, gc
import torch

ROOT = "/aiter_src"
_pkg_root = f"{ROOT}/aiter/ops/flydsl/kernels"
_pkg = types.ModuleType("_kpkg"); _pkg.__path__ = [_pkg_root]; sys.modules["_kpkg"] = _pkg
def _load(m, fn):
    spec = importlib.util.spec_from_file_location(f"_kpkg.{m}", f"{_pkg_root}/{fn}")
    mod = importlib.util.module_from_spec(spec); sys.modules[f"_kpkg.{m}"] = mod
    spec.loader.exec_module(mod); return mod
_load("tensor_shim", "tensor_shim.py")
v2 = _load("batched_gemm_fp8_blockwise_flydsl_v2", "batched_gemm_fp8_blockwise_flydsl_v2.py")

sys.path.insert(0, ROOT)
ck = None
for name in ("batched_gemm_a8w8_blockscale", "batched_gemm_fp8_blockwise"):
    try:
        ck = getattr(__import__("aiter", fromlist=[name]), name)
        print("CK fn:", name); break
    except Exception as e:
        print(f"import {name} failed:", type(e).__name__, str(e)[:80])

def gpu_time(fn, iters=30, warmup=10):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s=[torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    e=[torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for a,b in zip(s,e): a.record(); fn(); b.record()
    torch.cuda.synchronize()
    return sorted([a.elapsed_time(b)*1000 for a,b in zip(s,e)])[iters//2]

# best geometry per shape from the tuner
BEST = {(8,1024,1024,4096):(128,128,8),(8,4096,1024,4096):(128,128,4),(8,8192,1024,4096):(128,128,4)}
SHAPES=[(8,1024,1024,4096),(8,4096,1024,4096),(8,8192,1024,4096)]

print(f"{'shape':<24}{'v2best us':>11}{'ck us':>10}{'v2 TF':>8}{'ck TF':>8}{'v2/ck':>8}  geom")
print("-"*80)
for (B,M,N,K) in SHAPES:
    torch.manual_seed(B*M+N+K)
    A=(torch.randn(B,M,K,device='cuda')*0.5).clamp(-8,8).to(torch.float8_e4m3fn)
    W=(torch.randn(B,N,K,device='cuda')*0.5).clamp(-8,8).to(torch.float8_e4m3fn)
    A_s=(2.0**torch.randint(-2,2,(B,M,K//128),device='cuda')).float()
    W_s=(2.0**torch.randint(-2,2,(B,N//128,K//128),device='cuda')).float()
    flops=2*B*M*N*K
    bm,bn,nw=BEST[(B,M,N,K)]
    v2us=gpu_time(lambda: v2.flydsl_batched_gemm_fp8_blockwise_v2(A,W,A_s,W_s,block_m=bm,block_n=bn,n_waves=nw))
    ckus=float('inf')
    if ck is not None:
        try: ckus=gpu_time(lambda: ck(A,W,A_s,W_s))
        except Exception as e: print("  ck fail",type(e).__name__,str(e)[:100])
    def tf(u): return flops/(u*1e-6)/1e12 if u not in (0,float('inf')) else 0
    print(f"({B},{M},{N},{K})  {v2us:>9.1f}u {ckus:>8.1f}u {tf(v2us):>7.0f} {tf(ckus):>7.0f} "
          f"{(v2us/ckus if ckus!=float('inf') else 0):>7.2f}x  {bm}x{bn}x{nw}")
    del A,W,A_s,W_s; gc.collect(); torch.cuda.empty_cache()
