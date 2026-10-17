"""Backends (§4.2): randomness, scan, cond and switch, the operations the Array API doesn't cover."""

from __future__ import annotations

from types import ModuleType
from typing import Protocol


class Backend(Protocol):
    """The operations the Array API standard doesn't cover: randomness, `scan`, `cond` and `switch`.

    Library functions that need one of these take the backend as the keyword argument `bk`; nothing else does.
    Implementations: `JaxBackend`, `TorchBackend` and `StrictBackend` (tests only), all importable from
    `ttg.backend`. Each is imported lazily, so its array library is loaded only when asked for.

    Attributes
    ----------
    name : str
        ``"jax"``, ``"torch"`` or ``"strict"``.
    xp : module
        The backend's array namespace, for building inputs at the call site. Library code infers its namespace
        from the arrays it is given instead.
    """

    name: str                                    # "jax" | "torch" | "strict"
    xp: ModuleType                               # for constructors and tests only; library code infers xp

    def split(self, key, n: int):                # -> array of n keys (stackable, so scan can slice it)
        """Split a key into `n` independent keys.

        Parameters
        ----------
        key : key
            A key from `key` or a previous `split`.
        n : int
            Number of keys.

        Returns
        -------
        array
            `n` keys stacked along the leading axis, so `scan` can slice one per step.
        """
        ...

    def normal(self, key, shape, dtype, device=None):
        """Draw standard normal samples.

        Parameters
        ----------
        key : key
            Randomness for this draw. Each key should be used once.
        shape : tuple of int
            Output shape.
        dtype : dtype
            Floating dtype of the backend's namespace.
        device : device, optional
            Output device, where the backend supports one.

        Returns
        -------
        array
            Samples of shape `shape`.
        """
        ...

    def uniform(self, key, shape, dtype, device=None):
        """Draw samples uniform on [0, 1).

        Parameters
        ----------
        key : key
            Randomness for this draw. Each key should be used once.
        shape : tuple of int
            Output shape.
        dtype : dtype
            Floating dtype of the backend's namespace.
        device : device, optional
            Output device, where the backend supports one.

        Returns
        -------
        array
            Samples of shape `shape`.
        """
        ...

    def scan(self, f, carry, xs):                # -> (carry, stacked ys)
        """Loop `f` over the leading axis of `xs`, threading `carry`.

        Parameters
        ----------
        f : Callable[[carry, x], tuple[carry, y]]
            Step function. `x` is one leading-axis slice of every leaf of `xs`.
        carry : pytree
            Initial carry. Its structure, shapes and dtypes must be the same after every step.
        xs : pytree of array
            Per-step inputs, all with the same leading length.

        Returns
        -------
        carry : pytree
            The carry after the last step.
        ys : pytree of array or None
            The steps' `y` outputs stacked along a new leading axis, or None if `f` returns None.
        """
        ...

    def cond(self, pred, f_true, f_false, operand):
        """Return ``f_true(operand)`` if `pred` else ``f_false(operand)``.

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
        ...

    def switch(self, index, branches, operand):   # index: Python int (static) or traced int scalar
        """Return ``branches[index](operand)``.

        Parameters
        ----------
        index : int or array
            A Python int (static) or a traced integer scalar.
        branches : sequence of Callable[[operand], result]
            Branches. All must return results of the same structure, shapes and dtypes.
        operand : pytree
            Argument to the chosen branch.

        Returns
        -------
        result
            The chosen branch's output.
        """
        ...

    def key(self, seed: int):                    # make a root key
        """Make a root key.

        Parameters
        ----------
        seed : int
            Seed.

        Returns
        -------
        key
            A key for `split`, `normal` and `uniform`.
        """
        ...


def __getattr__(name):                           # import a backend's library only when asked for
    if name == "StrictBackend":
        from .strict import StrictBackend
        return StrictBackend
    if name == "JaxBackend":
        from .jax import JaxBackend
        return JaxBackend
    if name == "TorchBackend":
        from .torch import TorchBackend
        return TorchBackend
    raise AttributeError(name)


__all__ = ["Backend", "StrictBackend", "JaxBackend", "TorchBackend"]
