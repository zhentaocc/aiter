"""Benchmark new AITER sampling kernels vs PyTorch reference implementations."""
import time
import torch
import torch.nn.functional as F

from aiter.ops.sampling import (
    top_p_renorm_probs,
    chain_speculative_sampling,
    top_k_top_p_renorm_probs,
    top_k_renorm_probs,
)

torch.set_default_device("cuda")


def bench(name, fn, warmup=10, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / iters * 1e3
    print(f"  {name:45s}  {elapsed:8.3f} ms")
    return elapsed


def _top_p_renorm_torch(probs, top_p):
    top_p = top_p.to(dtype=probs.dtype, device=probs.device)
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    mask = cumsum - sorted_probs <= top_p.unsqueeze(1)
    sorted_probs = sorted_probs * mask
    probs_out = torch.zeros_like(probs)
    probs_out.scatter_(1, sorted_indices, sorted_probs)
    return probs_out / probs_out.sum(dim=-1, keepdim=True).clamp(min=1e-8)


def _chain_verify_torch(candidates, target_probs, uniform_samples, uniform_final, ts, ta):
    bs, draft_len = candidates.shape
    accept_len = torch.zeros(bs, dtype=torch.int32, device=candidates.device)
    bonus = torch.zeros(bs, dtype=torch.int64, device=candidates.device)
    ta_capped = max(ta, 1e-9)
    for i in range(bs):
        prob_acc = 0.0
        coin = float(uniform_samples[i, 0].item())
        n = 0
        cur = 0
        for j in range(1, draft_len):
            tid = int(candidates[i, j].item())
            tp = float(target_probs[i, cur, tid].item())
            prob_acc += tp
            if coin <= prob_acc / ta_capped or tp >= ts:
                n += 1; prob_acc = 0.0; cur = j
                coin = float(uniform_samples[i, j].item())
            else:
                break
        accept_len[i] = n
        row = target_probs[i, cur]
        u = float(uniform_final[i].item()) * float(row.sum().item())
        cdf = torch.cumsum(row, dim=0)
        nz = (cdf >= u).nonzero(as_tuple=True)[0]
        bonus[i] = int(nz[0].item()) if len(nz) > 0 else row.shape[0] - 1
    return accept_len, bonus


def bench_top_p_renorm(batch_size, vocab_size):
    print(f"\n=== top_p_renorm_probs  bs={batch_size}  V={vocab_size} ===")
    raw = torch.rand(batch_size, vocab_size)
    probs = raw / raw.sum(dim=-1, keepdim=True)
    top_p = torch.full((batch_size,), 0.9)

    t_pt = bench("PyTorch (sort+cumsum+scatter)", lambda: _top_p_renorm_torch(probs, top_p))
    t_k = bench("AITER kernel", lambda: top_p_renorm_probs(probs, top_p, 0.0))
    print(f"  Speedup: {t_pt / t_k:.2f}x")


def bench_chain_spec(batch_size, vocab_size, draft_len):
    print(f"\n=== chain_speculative_sampling  bs={batch_size}  V={vocab_size}  draft={draft_len} ===")
    raw = torch.rand(batch_size, draft_len, vocab_size)
    target_probs = raw / raw.sum(dim=-1, keepdim=True)
    candidates = torch.randint(0, vocab_size, (batch_size, draft_len))
    u = torch.rand(batch_size, draft_len)
    uf = torch.rand(batch_size)

    t_pt = bench("PyTorch (python loop)", lambda: _chain_verify_torch(candidates, target_probs, u, uf, 1.0, 1.0), warmup=2, iters=10)
    t_k = bench("AITER kernel", lambda: chain_speculative_sampling(candidates, target_probs, u, uf, 1.0, 1.0, True))
    print(f"  Speedup: {t_pt / t_k:.2f}x")


def bench_topk_topp_renorm(batch_size, vocab_size):
    print(f"\n=== top_k_top_p_renorm_probs (fused)  bs={batch_size}  V={vocab_size} ===")
    raw = torch.rand(batch_size, vocab_size)
    probs = raw / raw.sum(dim=-1, keepdim=True)
    top_k = torch.full((batch_size,), 50, dtype=torch.int32)
    top_p = torch.full((batch_size,), 0.9)

    def sequential_renorm():
        p = top_k_renorm_probs(probs, top_k, 0)
        return top_p_renorm_probs(p, top_p, 0.0)

    t_seq = bench("Sequential (top_k + top_p)", sequential_renorm)
    t_fused = bench("AITER fused kernel", lambda: top_k_top_p_renorm_probs(probs, top_k, 0, top_p, 0.0))
    print(f"  Speedup: {t_seq / t_fused:.2f}x")


if __name__ == "__main__":
    print("=" * 70)
    print("AITER Sampling Kernel Benchmarks")
    print("=" * 70)

    for bs in [1, 4, 16, 64]:
        for V in [32000, 128256]:
            bench_top_p_renorm(bs, V)

    for bs in [1, 4, 16]:
        for V in [32000, 128256]:
            for dl in [4, 8]:
                bench_chain_spec(bs, V, dl)

    for bs in [1, 4, 16, 64]:
        for V in [32000, 128256]:
            bench_topk_topp_renorm(bs, V)
