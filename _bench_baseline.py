"""Baseline harness validation: v2 (FlyDSL) vs CK vs Triton at canonical prefill.
Run inside aiter_ck_bench docker: repo mounted at /aiter_src."""
import sys, types, importlib.util
import torch

ROOT = "/aiter_src"
_pkg_root = f"{ROOT}/aiter/ops/flydsl/kernels"
_pkg = types.ModuleType("_kpkg"); _pkg.__path__ = [_pkg_root]; sys.modules["_kpkg"] = _pkg
def _load(modname, fn):
    spec = importlib.util.spec_from_file_location(f"_kpkg.{modname}", f"{_pkg_root}/{fn}")
    m = importlib.util.module_from_spec(spec); sys.modules[f"_kpkg.{modname}"] = m
    spec.loader.exec_module(m); return m
_load("tensor_shim", "tensor_shim.py")
v2 = _load("batched_gemm_fp8_blockwise_flydsl_v2", "batched_gemm_fp8_blockwise_flydsl_v2.py")

spec_t = importlib.util.spec_from_file_location("trit",
    f"{ROOT}/aiter/ops/triton/_triton_kernels/gemm/batched/batched_gemm_fp8_blockwise.py")
trit = importlib.util.module_from_spec(spec_t); spec_t.loader.exec_module(trit)

sys.path.insert(0, ROOT)
ck = None
try:
    from aiter import batched_gemm_fp8_blockwise as ck
except Exception as e:
    print("CK import failed:", type(e).__name__, str(e)[:120])

def gpu_time(fn, iters=20, warmup=5):
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
    a = A.to(torch.float32).view(B, M, K//BL, BL) * A_s.unsqueeze(-1)
    a = a.view(B, M, K)
    w = W.to(torch.float32).view(B, N, K//BL, BL) * W_s.repeat_interleave(BL, dim=1).view(B, N, K//BL, BL).gather(1, torch.zeros(1)) if False else None
    # simpler: expand W_s (B, N/128, K/128) -> (B, N, K)
    w = (W.to(torch.float32).view(B, N//BL, BL, K//BL, BL) *
         W_s.view(B, N//BL, 1, K//BL, 1)).view(B, N, K)
    return torch.bmm(a, w.transpose(1, 2)).to(torch.bfloat16)

for (B, M, N, K) in [(8, 4096, 1024, 4096)]:
    torch.manual_seed(B*M+N+K)
    A = (torch.randn(B, M, K, device='cuda')*0.5).clamp(-8,8).to(torch.float8_e4m3fn)
    W = (torch.randn(B, N, K, device='cuda')*0.5).clamp(-8,8).to(torch.float8_e4m3fn)
    A_s = (2.0**torch.randint(-2,2,(B,M,K//128),device='cuda')).float()
    W_s = (2.0**torch.randint(-2,2,(B,N//128,K//128),device='cuda')).float()
    flops = 2*B*M*N*K
    ref = oracle(A, W, A_s, W_s)
    out = v2.flydsl_batched_gemm_fp8_blockwise_v2(A, W, A_s, W_s)
    err = (out.float()-ref.float()).abs().max().item()
    print(f"v2 correctness max abs err vs oracle: {err}")
    res = {}
    res['v2'] = gpu_time(lambda: v2.flydsl_batched_gemm_fp8_blockwise_v2(A, W, A_s, W_s))
    try: res['tri'] = gpu_time(lambda: trit.triton_batched_gemm_fp8_blockwise(A, W, A_s, W_s))
    except Exception as e: res['tri']=float('inf'); print("tri fail", str(e)[:80])
    if ck is not None:
        try:
            o_ck = ck(A, W, A_s, W_s, backend='ck')
            cke = (o_ck.float()-ref.float()).abs().max().item()
            print(f"ck correctness max abs err vs oracle: {cke}")
            res['ck'] = gpu_time(lambda: ck(A, W, A_s, W_s, backend='ck'))
        except Exception as e: res['ck']=float('inf'); print("ck fail", type(e).__name__, str(e)[:160])
    def tf(us): return flops/(us*1e-6)/1e12 if us not in (0,float('inf')) else 0
    print(f"shape ({B},{M},{N},{K}):")
    for k,v in res.items():
        print(f"  {k:5} {v:8.1f} us  {tf(v):7.0f} TF")
