"""JAX backend: `jax.random` keys; `scan`, `cond` and `switch` are the `lax` versions."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from ..ops import leaves


class JaxBackend:
    name = "jax"
    xp = jnp

    def key(self, seed):
        return jax.random.key(seed)

    def split(self, key, n):
        return jax.random.split(key, n)

    def normal(self, key, shape, dtype, device=None):         # device is ignored
        return jax.random.normal(key, tuple(shape), dtype)

    def uniform(self, key, shape, dtype, device=None):
        return jax.random.uniform(key, tuple(shape), dtype)

    def scan(self, f, carry, xs):
        # Under jax.disable_jit() (eager tests), lax.scan rejects length 0 because it cannot infer the
        # ys type without tracing f. The runner's S == 2 scan is empty and returns no ys, so pass through.
        if jax.config.jax_disable_jit and leaves(xs)[0].shape[0] == 0:
            return carry, None
        return lax.scan(f, carry, xs)

    def cond(self, pred, f_true, f_false, operand):
        return lax.cond(pred, f_true, f_false, operand)

    def switch(self, index, branches, operand):
        return lax.switch(index, branches, operand)

    def __str__(self):
        return self.name

    __repr__ = __str__
