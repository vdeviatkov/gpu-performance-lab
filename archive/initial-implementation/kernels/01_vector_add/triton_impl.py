"""One contiguous 1024-element tile per program; tail loads/stores are masked."""

import torch
import triton
import triton.language as tl

from common.python.contracts import validate


@triton.jit
def _add(A, B, C, N, BLOCK: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(A + offsets, offsets < N, other=0).to(tl.float32)
    b = tl.load(B + offsets, offsets < N, other=0).to(tl.float32)
    tl.store(C + offsets, a + b, offsets < N)


def run(a, b, *, block_size=1024, num_warps=4):
    validate(a, ndim=1, cuda=True, other=b)
    if block_size not in (256, 512, 1024, 2048) or num_warps not in (4, 8):
        raise ValueError("Use block_size in {256,512,1024,2048} and num_warps in {4,8}")
    out = torch.empty_like(a)
    if a.numel():
        with torch.cuda.device(a.device):
            _add[(triton.cdiv(a.numel(), block_size),)](
                a, b, out, a.numel(), BLOCK=block_size, num_warps=num_warps
            )
    return out
