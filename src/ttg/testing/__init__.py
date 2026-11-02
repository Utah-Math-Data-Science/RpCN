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

UNIT_ROUNDOFF32 = 2.0 ** -24             # unit roundoff for float32
SMALLEST_NORMAL32 = 2.0 ** -126          # smallest normal float32: the underflow floor unit (§18.3)


class CountingNetwork:                    # contract C1
    """Wrap a raw network ``network(x, t)`` and count its calls.

    Parameters
    ----------
    network : Callable[[array, array], array]
        The network to wrap.

    Attributes
    ----------
    calls : int
        Number of calls so far.
    """

    def __init__(self, network):
        self.network, self.calls = network, 0

    def __call__(self, x, t):
        """Count the call and return ``network(x, t)``."""
        self.calls += 1
        return self.network(x, t)


def to_host(a):                           # eager tests only
    """Copy an array to a NumPy array on the host (torch tensors are detached and moved to CPU first).

    Parameters
    ----------
    a : array
        A concrete (not traced) array from any supported namespace.

    Returns
    -------
    numpy.ndarray
    """
    if array_api_compat.is_torch_array(a):
        a = a.detach().cpu()
    return numpy.from_dlpack(a)


class AuditingEvaluate:                   # invariant I1: counts rows, raises on a repeated (x, t)
    """Wrap an `evaluate` function, count its calls and rows, and raise if any row ``(x, t)`` is evaluated twice.

    Eager only: rows are copied to the host to be compared.

    Parameters
    ----------
    evaluate : Callable[[array, array, frozenset], Record]
        The evaluation function to wrap.

    Attributes
    ----------
    rows : int
        Total number of rows evaluated.
    calls : int
        Number of calls.
    seen : set
        The ``(x bytes, t)`` pairs evaluated so far.

    Raises
    ------
    DuplicateEvaluation
        On a call containing a row already seen.
    """

    def __init__(self, evaluate):
        self.evaluate, self.rows, self.calls, self.seen = evaluate, 0, 0, set()

    def __call__(self, x, t, needs):
        """Record the rows, raising `DuplicateEvaluation` on a repeat, and return ``evaluate(x, t, needs)``."""
        for x_row, t_row in zip(to_host(x), to_host(t)):
            fingerprint = (x_row.tobytes(), float(t_row))
            if fingerprint in self.seen:
                raise DuplicateEvaluation(t_row)
            self.seen.add(fingerprint)
        self.rows += t.shape[0]
        self.calls += 1
        return self.evaluate(x, t, needs)


def index_of(t_grid, t):                  # eager tests only: the k with t_grid[k, ...] == t (all rows agree)
    """Find the step index of a time on a grid.

    Parameters
    ----------
    t_grid : array
        Times per step, shape ``[S, ...]``.
    t : array
        Times, broadcastable against ``t_grid[k]``.

    Returns
    -------
    int
        The unique `k` with ``t_grid[k, ...] == t`` in every entry.

    Raises
    ------
    AssertionError
        If no step, or more than one, matches.
    """
    xp = namespace(t_grid, t)
    hits = [k for k in range(t_grid.shape[0]) if bool(xp.all(t_grid[k, ...] == t))]
    assert len(hits) == 1, hits
    return hits[0]


@dataclass(frozen=True)                   # deliberately NOT a pytree: lives only in eager tests
class ReplayGaussian(Gaussian):
    """A `Gaussian` whose `sample` returns a fixed array; densities are those of the Gaussian.

    Not a pytree, so for eager tests only.

    Parameters
    ----------
    mean : array
        Mean, shape ``[*lead, *D]``.
    var : array
        Per-row variance, shape ``[*lead]``.
    forced : array
        What `sample` returns, shape ``[*lead, *D]``.
    """

    forced: Array = None

    def sample(self, key, backend):
        """Return `forced`, ignoring `key` and `backend`."""
        return self.forced


@dataclass(frozen=True)
class ReplayKernel:                       # proposal double: follows a given path, densities from `base`
    """Proposal double that follows a given path, with the densities of `base`.

    At step `k` it samples ``path[k]``. Eager tests only.

    Parameters
    ----------
    base : Kernel
        Kernel providing the means, variances and `requires`.
    path : array
        Points to return, shape ``[S, I, N, *D]``.
    t_grid : array
        Times of the steps, shape ``[S, I, N]``; the target time `t_dst` of `at` selects the step.
    """

    base: Any
    path: Array                           # [S, I, N, *D]
    t_grid: Array                         # [S, I, N]

    @property
    def requires(self):
        """The features `base` requires."""
        return self.base.requires

    def at(self, record, t_dst):
        """Return `base`'s Gaussian from `record` to time `t_dst`, with `sample` fixed to the path point at `t_dst`.

        Parameters
        ----------
        record : Record
            Current point and its features, leading axes ``[I, N]``.
        t_dst : array
            Target times, shape ``[I, N]``; must equal one row of `t_grid`.

        Returns
        -------
        ReplayGaussian
        """
        base_dist = self.base.at(record, t_dst)
        grid_index = index_of(self.t_grid, t_dst)
        return ReplayGaussian(base_dist.mean, base_dist.var, self.path[grid_index, ...])   # sample() returns that point


def toy_evaluate(prior_mean, prior_std, reward_weights, schedule):
    """Exact `evaluate` for a Gaussian toy model: data ~ N(prior_mean, prior_std^2 I), r(x0) = <reward_weights, x0>.

    Every feature (x0, eps, score, reward, reward gradient) is computed in closed form from the schedule.

    Parameters
    ----------
    prior_mean : array
        Data mean, shape ``[*D]``.
    prior_std : float
        Data standard deviation.
    reward_weights : array
        Reward weights, shape ``[*D]``.
    schedule : Schedule
        Noise schedule.

    Returns
    -------
    Callable[[array, array, frozenset], Record]
        ``evaluate(x [B, *D], t [B], needs)``, returning exactly the requested features.
    """
    def evaluate(x, t, needs):
        xp = namespace(x, t)
        m, s = bcast(schedule.mean(t), x), bcast(schedule.std(t), x)
        marginal_var = m**2 * prior_std**2 + s**2        # marginal variance of x_t
        gain = m * prior_std**2 / marginal_var           # posterior-mean gain
        x0 = prior_mean + gain * (x - m * prior_mean)    # E[x0 | x_t]
        eps = (x - m * x0) / s
        score = -(x - m * prior_mean) / marginal_var     # = -eps / s exactly (Tweedie)
        event_axes = tuple(range(1, x.ndim))
        reward = xp.sum(reward_weights * x0, axis=event_axes)
        reward_grad = gain * reward_weights              # d<reward_weights, x0>/dx, broadcast per row
        features = {Feature.X0: x0, Feature.EPS: eps, Feature.SCORE: score,
                    Feature.REWARD: reward, Feature.REWARD_GRAD: reward_grad}
        return Record(x, t, **{f.value: (features[f] if f in needs else None) for f in Feature})
    return evaluate


def toy_eps_network(prior_mean, prior_std, schedule):
    """Return the toy model's exact eps prediction, as a raw ``network(x, t)`` for the evaluate builders.

    Parameters
    ----------
    prior_mean : array
        Data mean, shape ``[*D]``.
    prior_std : float
        Data standard deviation.
    schedule : Schedule
        Noise schedule.

    Returns
    -------
    Callable[[array, array], array]
        ``network(x [B, *D], t [B]) -> eps [B, *D]``.
    """
    def network(x, t):
        m, s = bcast(schedule.mean(t), x), bcast(schedule.std(t), x)
        marginal_var = m**2 * prior_std**2 + s**2
        return s * (x - m * prior_mean) / marginal_var
    return network


def oracle(f, *args):
    """Run `f` on float64 copies of float32 inputs: a higher-precision reference for precision tests.

    Parameters
    ----------
    f : Callable
        Function to call.
    *args : pytree
        Arguments; float32 array leaves are cast to float64, everything else is passed through.

    Returns
    -------
    Any
        ``f(*args)`` on the upcast arguments.
    """
    def to_float64(a):
        if hasattr(a, "dtype") and hasattr(a, "shape"):
            xp = namespace(a)
            if a.dtype == xp.float32:
                return xp.astype(a, xp.float64)
        return a
    return f(*tree_map(to_float64, args))


__all__ = ["UNIT_ROUNDOFF32", "SMALLEST_NORMAL32", "AuditingEvaluate", "CountingNetwork", "ReplayGaussian",
           "ReplayKernel", "index_of", "oracle", "to_host", "toy_evaluate", "toy_eps_network"]
