"""JAX backend: `jax.random` keys; `scan`, `cond` and `switch` are the `lax` versions."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from ..ops import leaves


class JaxBackend:
    """JAX backend: `jax.random` keys and the `jax.lax` control flow.

    `scan`, `cond` and `switch` are the `jax.lax` versions, so everything can run under `jax.jit`.

    Attributes
    ----------
    name : str
        ``"jax"``.
    xp : module
        `jax.numpy`.

    Notes
    -----
    JAX places arrays itself: the `device` arguments of `normal` and `uniform` are ignored. Float64 needs
    ``jax.config.update("jax_enable_x64", True)``.
    """

    name = "jax"
    xp = jnp

    def key(self, seed):
        """Make a root key, ``jax.random.key(seed)``.

        Parameters
        ----------
        seed : int
            Seed.

        Returns
        -------
        key
            A typed JAX PRNG key.
        """
        return jax.random.key(seed)

    def split(self, key, n):
        """Split `key` into `n` keys, ``jax.random.split(key, n)``.

        Parameters
        ----------
        key : key
            A JAX PRNG key.
        n : int
            Number of keys.

        Returns
        -------
        array
            `n` keys stacked along the leading axis.
        """
        return jax.random.split(key, n)

    def normal(self, key, shape, dtype, device=None):         # device is ignored
        """Draw standard normal samples with ``jax.random.normal``.

        Parameters
        ----------
        key : key
            A JAX PRNG key.
        shape : tuple of int
            Output shape.
        dtype : dtype
            Floating dtype.
        device : optional
            Ignored.

        Returns
        -------
        array
            Samples of shape `shape`.
        """
        return jax.random.normal(key, tuple(shape), dtype)

    def uniform(self, key, shape, dtype, device=None):
        """Draw samples uniform on [0, 1) with ``jax.random.uniform``.

        Parameters
        ----------
        key : key
            A JAX PRNG key.
        shape : tuple of int
            Output shape.
        dtype : dtype
            Floating dtype.
        device : optional
            Ignored.

        Returns
        -------
        array
            Samples of shape `shape`.
        """
        return jax.random.uniform(key, tuple(shape), dtype)

    def scan(self, f, carry, xs):
        """``jax.lax.scan(f, carry, xs)``.

        Parameters
        ----------
        f : Callable[[carry, x], tuple[carry, y]]
            Step function, traced once.
        carry : pytree
            Initial carry. Its structure, shapes and dtypes must be the same after every step.
        xs : pytree of array
            Per-step inputs, all with the same leading length.

        Returns
        -------
        carry : pytree
            The carry after the last step.
        ys : pytree of array or None
            The steps' outputs stacked along a new leading axis; None if `f` returns None, or if `xs` is empty
            under ``jax.disable_jit()``.
        """
        # Under jax.disable_jit() (eager tests), lax.scan rejects length 0 because it cannot infer the
        # ys type without tracing f. The runner's S == 2 scan is empty and returns no ys, so pass through.
        if jax.config.jax_disable_jit and leaves(xs)[0].shape[0] == 0:
            return carry, None
        return lax.scan(f, carry, xs)

    def cond(self, pred, f_true, f_false, operand):
        """``jax.lax.cond(pred, f_true, f_false, operand)``.

        Parameters
        ----------
        pred : bool or array
            Scalar predicate; may be traced.
        f_true, f_false : Callable[[operand], result]
            Branches. Both must return results of the same structure, shapes and dtypes.
        operand : pytree
            Argument to the chosen branch.

        Returns
        -------
        result
            The chosen branch's output.
        """
        return lax.cond(pred, f_true, f_false, operand)

    def switch(self, index, branches, operand):
        """``jax.lax.switch(index, branches, operand)``; out-of-range indices are clamped.

        Parameters
        ----------
        index : int or array
            A Python int or a traced integer scalar.
        branches : sequence of Callable[[operand], result]
            Branches. All must return results of the same structure, shapes and dtypes.
        operand : pytree
            Argument to the chosen branch.

        Returns
        -------
        result
            The chosen branch's output.
        """
        return lax.switch(index, branches, operand)

    def __str__(self):
        return self.name

    __repr__ = __str__
