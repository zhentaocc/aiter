import torch, aiter
def gt(fn, it=30, wu=10):
    for _ in range(wu): fn()
    torch.cuda.synchronize()
    s=[torch.cuda.Event(enable_timing=True) for _ in range(it)]
    e=[torch.cuda.Event(enable_timing=True) for _ in range(it)]
    for a,b in zip(s,e): a.record(); fn(); b.record()
    torch.cuda.synchronize()
    return sorted([a.elapsed_time(b)*1000 for a,b in zip(s,e)])[it//2]
for B,M,N,K in [(8,1024,1024,4096),(8,4096,1024,4096),(8,8192,1024,4096)]:
    torch.manual_seed(B*M+N+K)
    A=(torch.randn(B,M,K,device="cuda")*0.5).clamp(-8,8).to(torch.float8_e4m3fn)
    W=(torch.randn(B,N,K,device="cuda")*0.5).clamp(-8,8).to(torch.float8_e4m3fn)
    As=(2.0**torch.randint(-2,2,(B,M,K//128),device="cuda")).float()
    Ws=(2.0**torch.randint(-2,2,(B,N//128,K//128),device="cuda")).float()
    us=gt(lambda: aiter.batched_gemm_fp8_blockscale(A,W,As,Ws))
    tf=2*B*M*N*K/(us*1e-6)/1e12
    print(f"CK ({B},{M},{N},{K}) {us:.1f}us {tf:.0f}TF")
