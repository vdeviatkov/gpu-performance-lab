"""XLA baseline with FP32 accumulation and output."""

import jax
import jax.numpy as jnp

from common.python import jax_bridge as _bridge  # noqa: F401


@jax.jit
def run(x):
    return jnp.sum(x, dtype=jnp.float32)
