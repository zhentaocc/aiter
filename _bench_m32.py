"""Bench v2 (16x16x128) vs m32 (32x32x64) at prefill shapes. Idle node."""
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
v2 = _load("batched_gemm_fp8_blockwise_flydsl_v2", "batched_gemm_fp8_blockwise_flydsl_v2.py")
m32 = _load("batched_gemm_fp8_blockwise_flydsl_m32", "batched_gemm_fp8_blockwise_flydsl_m32.py")

def gt(fn, it=30, wu=10):
    for _ in range(wu): fn()
    torch.cuda.synchronize()
    s=[torch.cuda.Event(enable_timing=True) for _ in range(it)]
    e=[torch.cuda.Event(enable_timing=True) for _ in range(it)]
    for a,b in zip(s,e): a.record(); fn(); b.record()
    torch.cuda.synchronize()
    return sorted([a.elapsed_time(b)*1000 for a,b in zip(s,e)])[it//2]

def oracle(A,W,As,Ws):
    B,M,K=A.shape;_,N,_=W.shape;BL=128
    a=(A.to(torch.float32).view(B,M,K//BL,BL)*As.unsqueeze(-1)).view(B,M,K)
    w=(W.to(torch.float32).view(B,N//BL,BL,K//BL,BL)*Ws.view(B,N//BL,1,K//BL,1)).view(B,N,K)
    return torch.bmm(a,w.transpose(1,2)).to(torch.bfloat16)

# m32 geometry sweep candidates (block_n/n_waves multiple of 32)
M32_CFGS=[(128,128,4),(128,128,2),(64,128,4),(256,128,4),(128,128,1)]
SHAPES=[(8,1024,1024,4096),(8,4096,1024,4096),(8,8192,1024,4096)]
print(f"{'shape':<22}{'v2 us':>9}{'m32best us':>12}{'m32 TF':>8}{'m32/v2':>8}  m32geom")
print("-"*70)
for (B,M,N,K) in SHAPES:
    torch.manual_seed(B*M+N+K)
    A=(torch.randn(B,M,K,device='cuda')*0.5).clamp(-8,8).to(torch.float8_e4m3fn)
    W=(torch.randn(B,N,K,device='cuda')*0.5).clamp(-8,8).to(torch.float8_e4m3fn)
    As=(2.0**torch.randint(-2,2,(B,M,K//128),device='cuda')).float()
    Ws=(2.0**torch.randint(-2,2,(B,N//128,K//128),device='cuda')).float()
    flops=2*B*M*N*K; ref=oracle(A,W,As,Ws)
    v2us=gt(lambda: v2.flydsl_batched_gemm_fp8_blockwise_v2(A,W,As,Ws))
    best=(float('inf'),None)
    for (bm,bn,nw) in M32_CFGS:
        if M%bm or N%bn or (bn//nw)%32 or bm%32 or bm%(nw*8): continue
        try:
            o=m32.flydsl_batched_gemm_fp8_blockwise_m32(A,W,As,Ws,block_m=bm,block_n=bn,n_waves=nw)
            if (o.float()-ref.float()).abs().max().item()>0.5:
                print(f"   m32 {bm}x{bn}x{nw} WRONG"); continue
            us=gt(lambda: m32.flydsl_batched_gemm_fp8_blockwise_m32(A,W,As,Ws,block_m=bm,block_n=bn,n_waves=nw))
            if us<best[0]: best=(us,(bm,bn,nw))
        except Exception as e:
            print(f"   m32 {bm}x{bn}x{nw} EXC {str(e)[:50]}")
    mus,mg=best
    tf=flops/(mus*1e-6)/1e12 if mus!=float('inf') else 0
    print(f"({B},{M},{N},{K}) {v2us:>8.1f}u {mus:>10.1f}u {tf:>7.0f} {(mus/v2us if mus!=float('inf') else 0):>6.2f}x  {mg}")
    del A,W,As,Ws,ref; gc.collect(); torch.cuda.empty_cache()
