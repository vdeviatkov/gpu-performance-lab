"""Inputs are device-resident JAX arrays; conversion happens before timing."""

import jax

from common.python import jax_bridge as _bridge  # noqa: F401


@jax.jit
def run(a, b):
    return a + b
