"""Eager control flow shared by the PyTorch and strict backends."""

from __future__ import annotations

from ..ops import leaves, namespace, tree_map


class EagerControl:
    """Eager `scan`, `cond` and `switch` as plain Python control flow.

    A mixin for `TorchBackend` and `StrictBackend`. Predicates and indices are converted to Python values, so
    they must be concrete.
    """

    def scan(self, f, carry, xs):
        """Python loop over the leading axis of `xs`; ys stacked, or None if every step returns None.

        Parameters
        ----------
        f : Callable[[carry, x], tuple[carry, y]]
            Step function. `x` is slice ``i`` of every leaf of `xs`.
        carry : pytree
            Initial carry.
        xs : pytree of array
            Per-step inputs, all with the same leading length.

        Returns
        -------
        carry : pytree
            The carry after the last step.
        ys : pytree of array or None
            The steps' outputs stacked along a new leading axis, or None if every step returned None.
        """
        n = leaves(xs)[0].shape[0]
        ys = []
        for i in range(n):
            carry, y = f(carry, tree_map(lambda a: a[i, ...], xs))
            ys.append(y)
        if all(y is None for y in ys):
            return carry, None
        return carry, tree_map(lambda *a: namespace(*a).stack(a), *ys)

    def cond(self, pred, f_true, f_false, operand):
        """Return ``f_true(operand)`` if ``bool(pred)`` else ``f_false(operand)``; only the chosen branch runs.

        Parameters
        ----------
        pred : bool or array
            Concrete scalar predicate.
        f_true, f_false : Callable[[operand], result]
            Branches.
        operand : pytree
            Argument to the chosen branch.

        Returns
        -------
        result
            The chosen branch's output.
        """
        return f_true(operand) if bool(pred) else f_false(operand)

    def switch(self, index, branches, operand):
        """Return ``branches[int(index)](operand)``; only the chosen branch runs.

        Parameters
        ----------
        index : int or array
            Concrete integer scalar.
        branches : sequence of Callable[[operand], result]
            Branches.
        operand : pytree
            Argument to the chosen branch.

        Returns
        -------
        result
            The chosen branch's output.
        """
        return branches[int(index)](operand)

    def __str__(self):
        return self.name

    __repr__ = __str__
