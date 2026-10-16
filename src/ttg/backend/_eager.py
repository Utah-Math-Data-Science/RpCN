"""Eager control flow shared by the PyTorch and strict backends."""

from __future__ import annotations

from ..ops import leaves, namespace, tree_map


class EagerControl:
    def scan(self, f, carry, xs):
        """Python loop over the leading axis of `xs`; ys stacked, or None if every step returns None."""
        n = leaves(xs)[0].shape[0]
        ys = []
        for i in range(n):
            carry, y = f(carry, tree_map(lambda a: a[i, ...], xs))
            ys.append(y)
        if all(y is None for y in ys):
            return carry, None
        return carry, tree_map(lambda *a: namespace(*a).stack(a), *ys)

    def cond(self, pred, f_true, f_false, operand):
        return f_true(operand) if bool(pred) else f_false(operand)

    def switch(self, index, branches, operand):
        return branches[int(index)](operand)

    def __str__(self):
        return self.name

    __repr__ = __str__
