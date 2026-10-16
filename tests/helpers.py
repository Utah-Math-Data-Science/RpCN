"""Shared builders for the runner tests: the Gaussian toy model of §18.2."""

from ttg import Feature
from ttg.testing import toy_evaluate

ALL = frozenset(Feature)


def toy(bk, sch, D=(2,), dtype=None):
    """-> (evaluate, mu, w) for data ~ N(mu, I), r(x0) = <w, x0>."""
    xp = bk.xp
    dtype = xp.float64 if dtype is None else dtype
    n = 1
    for d in D:
        n *= d
    base = xp.reshape(xp.arange(n, dtype=dtype), D)
    mu = 0.5 - 0.25 * base
    w = xp.cos(base)
    return toy_evaluate(mu, 1.0, w, sch), mu, w
