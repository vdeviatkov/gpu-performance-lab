"""Production eager PyTorch baseline, also used as the correctness reference."""

from common.python.contracts import validate


def run(a, b):
    validate(a, ndim=1, other=b)
    return a + b
