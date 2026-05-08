# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ROCm/FlyDSL MegaMOE port contributors

"""Identity-matrix probes for the single-tile scaled MFMA kernel.

Three probes diagnose A/B/C lane mapping independently:

* **probe_A**: B = identity-along-K (B[i, j] = 1 if j==i else 0, restricted
  to j<16), A = arbitrary. Output C[m, n] should equal A[m, n] (the n-th
  element of A's m-th row). Mismatches reveal which A bytes a lane
  actually loads.

* **probe_B**: A = identity-along-K, B = arbitrary. Output C[m, n] should
  equal B[n, m]. Mismatches reveal B's lane→byte map.

* **probe_C**: A = B = ones (everywhere). Expected C[m, n] = 128 (sum of
  K=128 ones). Any non-128 value reveals which output positions a lane
  actually writes.

Run all three to triangulate the bug.
"""

from __future__ import annotations

import torch

from ._phase1_step1_single_tile import single_tile_launch
import flydsl.compiler as flyc


def _to_fp8(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.float8_e4m3fn).contiguous()


def _wrap(t: torch.Tensor, leading: int = 1, divisibility: int = 16):
    return flyc.from_dlpack(t).mark_layout_dynamic(leading_dim=leading,
                                                   divisibility=divisibility)


def _run(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Run the kernel with given FP8 A, B; return FP32 C."""
    C = torch.zeros(16, 16, dtype=torch.float32, device="cuda")
    A_dl = _wrap(A, leading=1, divisibility=128)
    B_dl = _wrap(B, leading=1, divisibility=128)
    C_dl = _wrap(C, leading=1, divisibility=16)
    single_tile_launch(A_dl, B_dl, C_dl, stream=torch.cuda.Stream())
    torch.cuda.synchronize()
    return C


def probe_C_all_ones() -> None:
    """A=B=ones → expected C = 128 everywhere. Any deviation → broken
    lane-output mapping."""
    A = _to_fp8(torch.ones(16, 128, device="cuda"))
    B = _to_fp8(torch.ones(16, 128, device="cuda"))
    C = _run(A, B)
    print(f"\nprobe_C (A=B=ones, expect C[i,j]=128 everywhere):")
    print(f"  unique values: {torch.unique(C[~torch.isnan(C)]).tolist()}")
    print(f"  has NaN: {torch.isnan(C).any().item()} (count={int(torch.isnan(C).sum().item())})")
    print(f"  C[0,:8]: {C[0,:8].tolist()}")
    print(f"  C[:,0]:  {C[:,0].tolist()}")
    # Mark positions that hit 128 vs other values to expose any pattern.
    mask128 = (C == 128.0).cpu()
    print(f"  mask of (C == 128) per row (first 16):")
    for r in range(16):
        print(f"    row {r:2d}: {''.join('#' if mask128[r,c] else ('N' if torch.isnan(C[r,c]) else '.') for c in range(16))}")


def probe_A_with_B_eye_partial() -> None:
    """B = block-identity along the first 16 K columns (B[i, k] = 1 iff k==i,
    k<16; else 0).  Then C[m, n] = sum_k A[m,k] * B[n,k] = A[m, n] for n<16.

    If C[m, n] equals A[m, n] for the right (m, n), A's lane mapping is OK.
    """
    A_f32 = (torch.arange(16 * 128, dtype=torch.float32, device="cuda")
             .reshape(16, 128) * 0.01).contiguous()
    A = _to_fp8(A_f32)

    B_f32 = torch.zeros(16, 128, dtype=torch.float32, device="cuda")
    for n in range(16):
        B_f32[n, n] = 1.0
    B = _to_fp8(B_f32)

    C = _run(A, B)
    A_ref = A.to(torch.float32)
    expected = A_ref[:, :16]   # C[m, n] = A[m, n] for n in 0..15

    print(f"\nprobe_A (B=eye_k<16, expect C = A[:, :16]):")
    print(f"  expected[0,:8]: {expected[0,:8].tolist()}")
    print(f"  got[0,:8]:      {C[0,:8].tolist()}")
    print(f"  expected[1,:8]: {expected[1,:8].tolist()}")
    print(f"  got[1,:8]:      {C[1,:8].tolist()}")
    diff = (C - expected).abs()
    print(f"  max abs diff: {diff.max().item():.4e} (NaN={torch.isnan(C).any().item()})")
    print(f"  match per output position (#=match within 0.02, X=mismatch, N=nan):")
    for r in range(16):
        line = ""
        for c in range(16):
            v, e = C[r, c].item(), expected[r, c].item()
            if torch.isnan(C[r, c]):
                line += "N"
            elif abs(v - e) < 0.02:
                line += "#"
            else:
                line += "X"
        print(f"    row {r:2d}: {line}")


def probe_B_with_A_eye_partial() -> None:
    """A = block-identity (A[m, k] = 1 iff k==m, k<16; else 0).  Then
    C[m, n] = sum_k A[m,k] * B[n,k] = B[n, m] for m<16."""
    A_f32 = torch.zeros(16, 128, dtype=torch.float32, device="cuda")
    for m in range(16):
        A_f32[m, m] = 1.0
    A = _to_fp8(A_f32)

    B_f32 = (torch.arange(16 * 128, dtype=torch.float32, device="cuda")
             .reshape(16, 128) * 0.01).contiguous()
    B = _to_fp8(B_f32)

    C = _run(A, B)
    B_ref = B.to(torch.float32)
    # C[m, n] = B[n, m]; equivalently expected = B[:16, :16].T
    expected = B_ref[:, :16].T

    print(f"\nprobe_B (A=eye_k<16, expect C = B[:16, :16].T):")
    print(f"  expected[0,:8]: {expected[0,:8].tolist()}")
    print(f"  got[0,:8]:      {C[0,:8].tolist()}")
    diff = (C - expected).abs()
    print(f"  max abs diff: {diff.max().item():.4e} (NaN={torch.isnan(C).any().item()})")


if __name__ == "__main__":
    probe_C_all_ones()
    probe_A_with_B_eye_partial()
    probe_B_with_A_eye_partial()
