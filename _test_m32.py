"""Correctness test for the 32x32x64 MFMA kernel (m32) vs torch oracle."""
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

def oracle(A, W, A_s, W_s):
    B, M, K = A.shape; _, N, _ = W.shape; BL = 128
    a = (A.to(torch.float32).view(B, M, K//BL, BL) * A_s.unsqueeze(-1)).view(B, M, K)
    w = (W.to(torch.float32).view(B, N//BL, BL, K//BL, BL) *
         W_s.view(B, N//BL, 1, K//BL, 1)).view(B, N, K)
    return torch.bmm(a, w.transpose(1, 2)).to(torch.bfloat16)

B, M, N, K = 8, 1024, 1024, 4096
torch.manual_seed(B*M+N+K)
A = (torch.randn(B, M, K, device='cuda')*0.5).clamp(-8,8).to(torch.float8_e4m3fn)
W = (torch.randn(B, N, K, device='cuda')*0.5).clamp(-8,8).to(torch.float8_e4m3fn)
A_s = (2.0**torch.randint(-2,2,(B,M,K//128),device='cuda')).float()
W_s = (2.0**torch.randint(-2,2,(B,N//128,K//128),device='cuda')).float()
ref = oracle(A, W, A_s, W_s)

CONFIGS = [(128,128,4), (64,128,4), (128,64,4), (128,128,2)]
for (bm, bn, nw) in CONFIGS:
    try:
        out = m32.flydsl_batched_gemm_fp8_blockwise_m32(A, W, A_s, W_s,
                block_m=bm, block_n=bn, n_waves=nw)
        err = (out.float()-ref.float()).abs().max().item()
        # also report mean err to distinguish "totally wrong" from "layout-ish"
        me = (out.float()-ref.float()).abs().mean().item()
        ok = "OK" if err <= 0.5 else "BAD"
        print(f"  bm={bm:3} bn={bn:3} nw={nw}  max_err={err:10.4f} mean={me:8.4f} {ok}")
    except Exception as e:
        import traceback
        print(f"  bm={bm:3} bn={bn:3} nw={nw}  EXC {type(e).__name__}: {str(e)[:120]}")
        if os.environ.get("TB"): traceback.print_exc()
