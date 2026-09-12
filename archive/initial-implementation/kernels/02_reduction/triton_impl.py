"""Hierarchical stages with FP32 partial sums; no global atomics."""

import torch
import triton
import triton.language as tl

from common.python.contracts import validate


@triton.jit
def _partial(X, P, N, BLOCK: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(X + offsets, offsets < N, other=0).to(tl.float32)
    tl.store(P + tl.program_id(0), tl.sum(values, axis=0))


def run(x, *, block_size=1024, num_warps=4):
    validate(x, ndim=1, cuda=True)
    if block_size not in (256, 512, 1024, 2048) or num_warps not in (4, 8):
        raise ValueError("Unsupported reduction configuration")
    with torch.cuda.device(x.device):
        if x.numel() == 0:
            return torch.zeros((), device=x.device, dtype=torch.float32)
        current = x
        # Hierarchical reduction keeps every program bounded, including very large N.
        while True:
            count = triton.cdiv(current.numel(), block_size)
            out = torch.empty((count,), device=x.device, dtype=torch.float32)
            _partial[(count,)](current, out, current.numel(), BLOCK=block_size, num_warps=num_warps)
            if count == 1:
                return out.view(())
            current = out
