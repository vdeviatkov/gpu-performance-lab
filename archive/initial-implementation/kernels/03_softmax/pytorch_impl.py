"""Production eager PyTorch baseline, also used as the correctness reference."""

import torch

from common.python.contracts import validate


def run(x):
    validate(x, ndim=2)
    return torch.softmax(x, dim=-1)
