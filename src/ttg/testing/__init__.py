"""Test doubles (§18.2). Imported only by tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import array_api_compat
import numpy

from ..kernels import Gaussian
from ..ops import bcast, namespace, tree_map
from ..records import DuplicateEvaluation, Feature, Record

Array = Any

EPS32 = 2.0 ** -24                       # unit roundoff for float32
TINY32 = 2.0 ** -126                     # smallest normal float32: the underflow floor unit (§18.3)


class CountingNetwork:                    # contract C1
    def __init__(self, net):
        self.net, self.calls = net, 0

    def __call__(self, x, t):
        self.calls += 1
        return self.net(x, t)


def to_host(a):                           # eager tests only
    if array_api_compat.is_torch_array(a):
        a = a.detach().cpu()
    return numpy.from_dlpack(a)


class AuditingEvaluate:                   # invariant I1: counts rows, raises on a repeated (x, t)
    def __init__(self, evaluate):
        self.evaluate, self.rows, self.calls, self.seen = evaluate, 0, 0, set()

    def __call__(self, x, t, needs):
        for xi, ti in zip(to_host(x), to_host(t)):
            key = (xi.tobytes(), float(ti))
            if key in self.seen:
                raise DuplicateEvaluation(ti)
            self.seen.add(key)
        self.rows += t.shape[0]
        self.calls += 1
        return self.evaluate(x, t, needs)


def index_of(t_grid, t):                  # eager tests only: the k with t_grid[k, ...] == t (all rows agree)
    xp = namespace(t_grid, t)
    hits = [k for k in range(t_grid.shape[0]) if bool(xp.all(t_grid[k, ...] == t))]
    assert len(hits) == 1, hits
    return hits[0]


@dataclass(frozen=True)                   # deliberately NOT a pytree: lives only in eager tests
class ReplayGaussian(Gaussian):
    forced: Array = None

    def sample(self, key, bk):
        return self.forced


@dataclass(frozen=True)
class ReplayKernel:                       # proposal double: follows a given path, densities from `base`
    base: Any
    path: Array                           # [S, I, N, *D]
    t_grid: Array                         # [S, I, N]

    @property
    def requires(self):
        return self.base.requires

    def at(self, rec, t):
        d = self.base.at(rec, t)
        k = index_of(self.t_grid, t)
        return ReplayGaussian(d.mean, d.var, self.path[k, ...])           # sample() returns path[k]


def toy_evaluate(mu, sd, w, sch):
    """data ~ N(mu, sd^2 I); reward r(x0) = <w, x0>; all exact. mu, w: [*D] arrays; sd: float."""
    def evaluate(x, t, needs):
        xp = namespace(x, t)
        m, s = bcast(sch.mean(t), x), bcast(sch.std(t), x)
        v = m**2 * sd**2 + s**2                          # marginal variance of x_t
        c = m * sd**2 / v                                # posterior-mean gain
        x0 = mu + c * (x - m * mu)                       # E[x0 | x_t]
        eps = (x - m * x0) / s
        score = -(x - m * mu) / v                        # = -eps / s exactly (Tweedie)
        ev = tuple(range(1, x.ndim))
        reward = xp.sum(w * x0, axis=ev)
        grad = c * w                                     # d<w, x0>/dx, broadcast per row
        vals = {Feature.X0: x0, Feature.EPS: eps, Feature.SCORE: score,
                Feature.REWARD: reward, Feature.REWARD_GRAD: grad}
        return Record(x, t, **{f.value: (vals[f] if f in needs else None) for f in Feature})
    return evaluate


def toy_eps_network(mu, sd, sch):
    """The toy model's exact eps prediction, as a raw network(x, t) for the builders."""
    def net(x, t):
        m, s = bcast(sch.mean(t), x), bcast(sch.std(t), x)
        v = m**2 * sd**2 + s**2
        return s * (x - m * mu) / v
    return net


def oracle(f, *args):
    """Run f on float64 copies of float32 inputs."""
    def up(a):
        if hasattr(a, "dtype") and hasattr(a, "shape"):
            xp = namespace(a)
            if a.dtype == xp.float32:
                return xp.astype(a, xp.float64)
        return a
    return f(*tree_map(up, args))


__all__ = ["EPS32", "TINY32", "AuditingEvaluate", "CountingNetwork", "ReplayGaussian", "ReplayKernel",
           "index_of", "oracle", "to_host", "toy_evaluate", "toy_eps_network"]
