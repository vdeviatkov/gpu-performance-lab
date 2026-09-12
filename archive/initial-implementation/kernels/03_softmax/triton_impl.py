"""One padded row per program, with FP32 max/sum and masked columns."""

import torch
import triton
import triton.language as tl

from common.python.contracts import validate
from common.python.registry import BackendUnavailable


@triton.jit
def _softmax(X, Y, COLS, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK)
    values = tl.load(X + row * COLS + col, col < COLS, other=-float("inf")).to(tl.float32)
    shifted = values - tl.max(values, axis=0)
    numerators = tl.exp(shifted)
    result = numerators / tl.sum(numerators, axis=0)
    tl.store(Y + row * COLS + col, result, col < COLS)


def run(x, *, num_warps=None):
    validate(x, ndim=2, cuda=True)
    rows, cols = x.shape
    if cols > 32768:
        raise BackendUnavailable("Triton row softmax supports at most 32768 columns")
    block = triton.next_power_of_2(cols)
    warps = num_warps if num_warps is not None else (4 if block <= 2048 else 8)
    if warps not in (4, 8, 16):
        raise ValueError("num_warps must be 4, 8, or 16")
    out = torch.empty_like(x)
    if rows:
        with torch.cuda.device(x.device):
            _softmax[(rows,)](x, out, cols, BLOCK=block, num_warps=warps)
    return out
