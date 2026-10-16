"""Potentials (§9): cumulative log-tilt C of a history, and per-step increments. State is lambda-free."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from .ops import device, namespace, pytree, static
from .records import Feature

Array = Any
State = Any


class Potential(Protocol):
    requires: frozenset[Feature]
    def init(self, rec) -> State: ...                          # state after the first point
    def update(self, state, rec, *, last: bool) -> State: ... # absorb the next point; `last` is static
    def log_tilt(self, state) -> Array: ...                    # C = log prod_{s >= t} G_s, shape [I, N]
    def increment(self, prev, state) -> Array: ...             # C(state) - C(prev), never formed as that difference
    # precondition: state = update(prev, rec, last=...) for some rec


class TiltDifference:
    """Mixin: the default increment, for potentials whose C stays O(one term)."""

    def increment(self, prev, st):
        return self.log_tilt(st) - self.log_tilt(prev)


@pytree
class TiltState:
    r: Array
    t: Array


@pytree
class PathState:
    run: Array
    acc: Array
    r: Array
    t: Array
    last: bool = static(False)


@pytree
class EndsState:
    r0: Array
    t0: Array
    r: Array
    t: Array
    last: bool = static(False)


@pytree
class RRState:
    sum: Array
    term: Array
    x: Array
    t: Array


def _zeros(t):
    return namespace(t).zeros(t.shape, dtype=t.dtype, device=device(t))


@dataclass(frozen=True)
class FKDiff(TiltDifference):
    lam: Callable
    requires = frozenset({Feature.REWARD})

    def init(self, rec):
        return TiltState(r=rec.need(Feature.REWARD), t=rec.t)

    def update(self, st, rec, *, last):
        return TiltState(r=rec.need(Feature.REWARD), t=rec.t)

    def log_tilt(self, st):
        return self.lam(st.t) * st.r


@dataclass(frozen=True)
class FKMax:
    """Path potential: intermediate tilt lambda * acc, acc a sum over steps of the running max of r_hat.

    With resampling it is reliable only when lambda * S * (the spread of r_hat) is small: the intermediate tilt is
    up to ~S times sharper than the final target, so resampled weights degenerate. Without resampling the
    telescoped weight is exp(lambda r_hat) at the end, i.e. plain importance sampling.
    """

    lam: Callable                     # constant in t (it may vary by ladder position)
    requires = frozenset({Feature.REWARD})

    def init(self, rec):
        r = rec.need(Feature.REWARD)
        return PathState(run=r, acc=r, r=r, t=rec.t, last=False)

    def _running(self, run, r):
        return namespace(run, r).maximum(run, r)

    def update(self, st, rec, *, last):
        r = rec.need(Feature.REWARD)
        run = self._running(st.run, r)
        return PathState(run=run, acc=st.acc + run, r=r, t=rec.t, last=last)

    def log_tilt(self, st):
        return self.lam(st.t) * (st.r if st.last else st.acc)

    def increment(self, prev, st):                                  # `last` is static
        return self.lam(st.t) * ((st.r - prev.acc) if st.last else st.run)


@dataclass(frozen=True)
class FKSum(FKMax):                   # identical except the running statistic
    """Path potential: as FKMax with the running sum of r_hat. With resampling it is reliable only when
    lambda * S**2 / 2 * (the spread of r_hat) is small; see FKMax."""

    def _running(self, run, r):
        return run + r


@dataclass(frozen=True)
class TiltEnds:                       # C = -l(start) + [last] l(end): CREPE's endpoint tilts
    lam: Callable
    requires = frozenset({Feature.REWARD})

    def init(self, rec):
        r = rec.need(Feature.REWARD)
        return EndsState(r0=r, t0=rec.t, r=r, t=rec.t, last=False)

    def update(self, st, rec, *, last):
        return EndsState(r0=st.r0, t0=st.t0, r=rec.need(Feature.REWARD), t=rec.t, last=last)

    def log_tilt(self, st):
        start = -self.lam(st.t0) * st.r0
        return start + self.lam(st.t) * st.r if st.last else start

    def increment(self, prev, st):
        return self.lam(st.t) * st.r if st.last else namespace(st.r).zeros_like(st.r)


@dataclass(frozen=True)
class Sum:
    parts: tuple

    def __init__(self, *parts):
        object.__setattr__(self, "parts", parts)

    @property
    def requires(self):
        return frozenset().union(*(p.requires for p in self.parts))

    def init(self, rec):
        return tuple(p.init(rec) for p in self.parts)

    def update(self, st, rec, *, last):
        return tuple(p.update(a, rec, last=last) for p, a in zip(self.parts, st))

    def log_tilt(self, st):
        return sum(p.log_tilt(a) for p, a in zip(self.parts, st))

    def increment(self, prev, st):
        return sum(p.increment(a, b) for p, a, b in zip(self.parts, prev, st))


@dataclass(frozen=True)
class Zero:
    requires = frozenset()

    def init(self, rec):
        return _zeros(rec.t)

    def update(self, st, rec, *, last):
        return st

    def log_tilt(self, st):
        return st

    def increment(self, prev, st):
        return st


@dataclass(frozen=True)
class ReverseRatio:                   # scores the reverse step with the kernels opposite to the run
    target_rev: Any
    proposal_rev: Any

    @property
    def trivial(self):                # static: term is identically 0
        return self.target_rev is self.proposal_rev

    @property
    def requires(self):
        return frozenset() if self.trivial else self.target_rev.requires | self.proposal_rev.requires

    def init(self, rec):
        z = _zeros(rec.t)
        return RRState(sum=z, term=z, x=rec.x, t=rec.t)

    def update(self, st, rec, *, last):
        if self.trivial:
            return st
        term = self.proposal_rev.at(rec, st.t).log_ratio(self.target_rev.at(rec, st.t), st.x)   # §7
        return RRState(sum=st.sum + term, term=term, x=rec.x, t=rec.t)

    def log_tilt(self, st):
        return st.sum

    def increment(self, prev, st):
        return st.term
