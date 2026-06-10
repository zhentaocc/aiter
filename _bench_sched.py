import sys, os, types, importlib.util, gc
import torch
ROOT = os.environ.get("AITER_ROOT", "/aiter_src")
_pkg_root = f"{ROOT}/aiter/ops/flydsl/kernels"
_pkg = types.ModuleType("_kpkg"); _pkg.__path__ = [_pkg_root]; sys.modules["_kpkg"] = _pkg
def _load(m, fn):
    spec = importlib.util.spec_from_file_location(f"_kpkg.{m}", f"{_pkg_root}/{fn}")
    mod = importlib.util.module_from_spec(spec); sys.modules[f"_kpkg.{m}"] = mod
    spec.loader.exec_module(mod); return mod
_load("tensor_shim", "tensor_shim.py")
m32 = _load("batched_gemm_fp8_blockwise_flydsl_m32", "batched_gemm_fp8_blockwise_flydsl_m32.py")
def gt(fn, it=30, wu=10):
    for _ in range(wu): fn()
    torch.cuda.synchronize()
    s=[torch.cuda.Event(enable_timing=True) for _ in range(it)]; e=[torch.cuda.Event(enable_timing=True) for _ in range(it)]
    for a,b in zip(s,e): a.record(); fn(); b.record()
    torch.cuda.synchronize(); return sorted([a.elapsed_time(b)*1000 for a,b in zip(s,e)])[it//2]
def oracle(A,W,As,Ws):
    B,M,K=A.shape;_,N,_=W.shape;BL=128
    a=(A.to(torch.float32).view(B,M,K//BL,BL)*As.unsqueeze(-1)).view(B,M,K)
    w=(W.to(torch.float32).view(B,N//BL,BL,K//BL,BL)*Ws.view(B,N//BL,1,K//BL,1)).view(B,N,K)
    return torch.bmm(a,w.transpose(1,2)).to(torch.bfloat16)
# use the m32 best geoms from prior bench
GEOM={(8,1024,1024,4096):(128,128,4),(8,4096,1024,4096):(256,128,4),(8,8192,1024,4096):(256,128,4)}
for (B,M,N,K),(bm,bn,nw) in GEOM.items():
    torch.manual_seed(B*M+N+K)
    A=(torch.randn(B,M,K,device='cuda')*0.5).clamp(-8,8).to(torch.float8_e4m3fn)
    W=(torch.randn(B,N,K,device='cuda')*0.5).clamp(-8,8).to(torch.float8_e4m3fn)
    As=(2.0**torch.randint(-2,2,(B,M,K//128),device='cuda')).float()
    Ws=(2.0**torch.randint(-2,2,(B,N//128,K//128),device='cuda')).float()
    ref=oracle(A,W,As,Ws); flops=2*B*M*N*K
    print(f"({B},{M},{N},{K}) geom {bm}x{bn}x{nw}")
    for sh in [0,1,2]:
        try:
            o=m32.flydsl_batched_gemm_fp8_blockwise_m32(A,W,As,Ws,block_m=bm,block_n=bn,n_waves=nw,sched_hint=sh)
            err=(o.float()-ref.float()).abs().max().item()
            us=gt(lambda: m32.flydsl_batched_gemm_fp8_blockwise_m32(A,W,As,Ws,block_m=bm,block_n=bn,n_waves=nw,sched_hint=sh))
            print(f"   sched_hint={sh}: {us:8.1f}us {flops/(us*1e-6)/1e12:6.0f}TF  err={err}")
        except Exception as e:
            print(f"   sched_hint={sh}: EXC {type(e).__name__}: {str(e)[:80]}")
    del A,W,As,Ws,ref; gc.collect(); torch.cuda.empty_cache()
