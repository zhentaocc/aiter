import sys, os, types, importlib.util
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
B,M,N,K=8,1024,1024,4096
A=(torch.randn(B,M,K,device='cuda')*0.5).clamp(-8,8).to(torch.float8_e4m3fn)
W=(torch.randn(B,N,K,device='cuda')*0.5).clamp(-8,8).to(torch.float8_e4m3fn)
As=(2.0**torch.randint(-2,2,(B,M,K//128),device='cuda')).float()
Ws=(2.0**torch.randint(-2,2,(B,N//128,K//128),device='cuda')).float()
out=m32.flydsl_batched_gemm_fp8_blockwise_m32(A,W,As,Ws)
print("compiled m32, out", out.shape)
