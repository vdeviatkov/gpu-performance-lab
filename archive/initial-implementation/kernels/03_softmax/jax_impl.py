"""Stable XLA softmax; explicit FP32 intermediate arithmetic."""

import jax
import jax.numpy as jnp

from common.python import jax_bridge as _bridge  # noqa: F401


@jax.jit
def run(x):
    return jax.nn.softmax(x.astype(jnp.float32), axis=-1).astype(x.dtype)
