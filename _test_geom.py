"""Correctness sweep over v2 geometry configs vs torch oracle.
Run inside aiter_ck_bench: repo at /aiter_src, flydsl 0.1.3.1."""
import sys, types, importlib.util
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

CONFIGS = [
    (128,128,4),  # default
    (64,128,4),
    (256,128,4),
    (128,64,4),
    (128,128,2),
    (128,64,2),
    (64,64,4),
    (128,256,4),  # should be REJECTED (block_n>128)
]
print(f"shape ({B},{M},{N},{K}), oracle ready")
for (bm, bn, nw) in CONFIGS:
    try:
        out = v2.flydsl_batched_gemm_fp8_blockwise_v2(A, W, A_s, W_s,
                                                       block_m=bm, block_n=bn, n_waves=nw)
        err = (out.float()-ref.float()).abs().max().item()
        ok = "OK" if err <= 0.5 else "BAD"
        print(f"  bm={bm:3} bn={bn:3} nw={nw}  max_err={err:8.4f}  {ok}")
    except Exception as e:
        print(f"  bm={bm:3} bn={bn:3} nw={nw}  EXC {type(e).__name__}: {str(e)[:90]}")
