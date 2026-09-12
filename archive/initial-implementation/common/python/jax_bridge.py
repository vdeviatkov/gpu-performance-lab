"""DLPack conversion is preparation, never part of a JAX benchmark sample."""

import os

# Avoid competing with PyTorch for JAX's default preallocation on shared devices.
# A user's explicit setting takes precedence and is captured in run metadata.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def from_torch(x):
    import jax.dlpack

    return jax.dlpack.from_dlpack(x.detach())


def to_torch(x):
    import torch

    x.block_until_ready()
    return torch.utils.dlpack.from_dlpack(x)
