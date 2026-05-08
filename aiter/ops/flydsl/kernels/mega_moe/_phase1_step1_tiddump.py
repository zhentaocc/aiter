# SPDX-License-Identifier: Apache-2.0
"""Tid-dump probe to find which lanes actually issue stores."""

from __future__ import annotations
import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import buffer_ops, range_constexpr, arith
from flydsl.expr.typing import T


@flyc.kernel
def tiddump(out: fx.Tensor):
    tid = fx.thread_idx.x
    rsrc = buffer_ops.create_buffer_resource(out, max_size=True)
    # Each lane writes:
    #   out[tid*2]     = lane // 16  (expected 0,0,...,1,1,...,2,2,...,3,3,...)
    #   out[tid*2 + 1] = lane % 16   (expected 0..15 repeated 4 times)
    div = tid // fx.Index(16)
    mod = tid % fx.Index(16)
    div_i32 = arith.index_cast(T.i32, div)
    mod_i32 = arith.index_cast(T.i32, mod)
    buffer_ops.buffer_store(div_i32, rsrc, tid * fx.Index(8),
                            offset_is_bytes=True)
    buffer_ops.buffer_store(mod_i32, rsrc, tid * fx.Index(8) + fx.Index(4),
                            offset_is_bytes=True)


@flyc.jit
def tiddump_launch(out: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    tiddump(out).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)


def main() -> None:
    out = torch.full((256,), -1, dtype=torch.int32, device="cuda")
    out_dl = flyc.from_dlpack(out).mark_layout_dynamic(leading_dim=0,
                                                        divisibility=4)
    tiddump_launch(out_dl, stream=torch.cuda.Stream())
    torch.cuda.synchronize()
    arr = out.cpu().tolist()
    pairs = [(arr[2*i], arr[2*i+1]) for i in range(64)]
    print("(div, mod) per lane (lane → (lane//16, lane%16) expected):")
    for lane in range(64):
        exp = (lane // 16, lane % 16)
        got = pairs[lane]
        mark = " " if got == exp else " <-- MISMATCH"
        if mark or lane < 32 or lane > 60:
            print(f"  lane {lane:2d}: got={got}  exp={exp}{mark}")


if __name__ == "__main__":
    main()
