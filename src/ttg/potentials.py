"""Potentials (§9): cumulative log-tilt C of a history, and per-step increments. State is lambda-free."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from .ops import device, namespace, pytree, static
from .records import Feature

Array = Any
State = Any


class Potential(Protocol):
    """Protocol for a potential: the cumulative log-tilt of a particle's history, and its per-step increments.

    A potential's state stores raw statistics of the history and no lambda; lambda is applied in `log_tilt` and
    `increment` from each row's time and ladder position, so a state moved to another position (a swap) is read at
    that position's temperature.

    Attributes
    ----------
    requires : frozenset of Feature
        Features the potential reads from each record (e.g. ``{Feature.REWARD}``).
    """

    requires: frozenset[Feature]

    def init(self, rec) -> State:                              # state after the first point
        """Start a history at its first point.

        Parameters
        ----------
        rec : Record
            The first point, ``[I, N, ...]``, carrying `requires`.

        Returns
        -------
        State
            A pytree of arrays with leading axes ``[I, N]``.
        """
        ...

    def update(self, state, rec, *, last: bool) -> State:     # absorb the next point; `last` is static
        """Extend a history by one point.

        Parameters
        ----------
        state : State
            State of the history so far.
        rec : Record
            The next point, ``[I, N, ...]``, carrying `requires`.
        last : bool
            Static. True if `rec` is the final point of the run.

        Returns
        -------
        State
            State of the extended history.
        """
        ...

    def log_tilt(self, state) -> Array:                        # C = log prod_{s >= t} G_s, shape [I, N]
        """Cumulative log-tilt ``C = log prod_{s >= t} G_s`` of the history.

        Parameters
        ----------
        state : State
            State of the history.

        Returns
        -------
        array
            ``[I, N]``.

        Notes
        -----
        Used to compare whole histories (swaps between rows with different histories). Runners and moves use
        `increment` instead, since C can grow along the path and differencing it loses precision.
        """
        ...

    def increment(self, prev, state) -> Array:                 # C(state) - C(prev), never formed as that difference
        """One step's change in log-tilt, ``C(state) - C(prev)``, computed directly from the step's statistics.

        Parameters
        ----------
        prev : State
            State before the step.
        state : State
            ``update(prev, rec, last=...)`` for some `rec`.

        Returns
        -------
        array
            ``[I, N]``.
        """
        ...
    # precondition: state = update(prev, rec, last=...) for some rec


class TiltDifference:
    """Mixin: the default increment, for potentials whose C stays O(one term)."""

    def increment(self, prev, st):
        """``log_tilt(st) - log_tilt(prev)``; accurate only when C is a single term.

        Parameters
        ----------
        prev : State
            State before the step.
        st : State
            State after the step.

        Returns
        -------
        array
            ``[I, N]``.
        """
        return self.log_tilt(st) - self.log_tilt(prev)


@pytree
class TiltState:
    """State of `FKDiff`: the current ``r_hat`` and ``t``, ``[I, N]`` each."""

    r: Array
    t: Array


@pytree
class PathState:
    """State of `FKMax`/`FKSum`: running statistic, accumulator, current ``r_hat`` and ``t``; static `last`."""

    run: Array
    acc: Array
    r: Array
    t: Array
    last: bool = static(False)


@pytree
class EndsState:
    """State of `TiltEnds`: start and current ``r_hat`` and ``t``; static `last`."""

    r0: Array
    t0: Array
    r: Array
    t: Array
    last: bool = static(False)


@pytree
class RRState:
    """State of `ReverseRatio`: running sum, latest term, previous point ``x`` and ``t``."""

    sum: Array
    term: Array
    x: Array
    t: Array


def _zeros(t):
    return namespace(t).zeros(t.shape, dtype=t.dtype, device=device(t))


@dataclass(frozen=True)
class FKDiff(TiltDifference):
    """FK potential with the difference form: ``C = lambda(t) r_hat_t(x)`` at the current point.

    Each step's weight is ``exp(lambda(t') r_hat' - lambda(t) r_hat)``, telescoping to ``exp(lambda r_hat_0)`` at the
    end. With a tempered lambda (e.g. `das_lambda`) this is DAS's potential; with ``ladder(betas)`` it is RpCN's
    tempered reward twist.

    Parameters
    ----------
    lam : Callable[[array], array]
        Tilt strength ``lam(t [I, N]) -> [I, N]``, e.g. ``constant(10.0)``, ``ladder(betas)``, a `das_lambda` table.

    Attributes
    ----------
    requires : frozenset of Feature
        ``{Feature.REWARD}``.
    """

    lam: Callable
    requires = frozenset({Feature.REWARD})

    def init(self, rec):
        """State ``(r_hat, t)`` of the first point. See `Potential.init`."""
        return TiltState(r=rec.need(Feature.REWARD), t=rec.t)

    def update(self, st, rec, *, last):
        """State ``(r_hat, t)`` of the new point; the history is not needed. See `Potential.update`."""
        return TiltState(r=rec.need(Feature.REWARD), t=rec.t)

    def log_tilt(self, st):
        """``lambda(t) r_hat``, ``[I, N]``. See `Potential.log_tilt`."""
        return self.lam(st.t) * st.r


@dataclass(frozen=True)
class FKMax:
    """Path potential: intermediate tilt lambda * acc, acc a sum over steps of the running max of r_hat.

    At the last point C becomes ``lambda r_hat_0``, so the run targets ``exp(lambda r_hat_0)`` like every FK potential.

    Parameters
    ----------
    lam : Callable[[array], array]
        Tilt strength ``lam(t [I, N]) -> [I, N]``. Must be constant in t (it may vary by ladder position).

    Attributes
    ----------
    requires : frozenset of Feature
        ``{Feature.REWARD}``.

    Notes
    -----
    Without resampling (``Never()``) the telescoped weight is exp(lambda r_hat) at the end, i.e. plain importance
    sampling, and the target is exact. With resampling it is reliable only when lambda * S * sigma_r is small,
    where sigma_r is the spread of the final predicted reward across particles: the intermediate tilt is up to ~S
    times sharper than the final target, so resampled weights degenerate. The bound is conservative: the effective
    count is the number of steps on which r_hat already differs between particles.
    """

    lam: Callable                     # constant in t (it may vary by ladder position)
    requires = frozenset({Feature.REWARD})

    def init(self, rec):
        """State with running statistic, accumulator and ``r_hat`` all equal to the first ``r_hat``.

        See `Potential.init`.
        """
        r = rec.need(Feature.REWARD)
        return PathState(run=r, acc=r, r=r, t=rec.t, last=False)

    def _running(self, run, r):
        return namespace(run, r).maximum(run, r)

    def update(self, st, rec, *, last):
        """Update the running statistic with the new ``r_hat`` and add it to the accumulator.

        See `Potential.update`.
        """
        r = rec.need(Feature.REWARD)
        run = self._running(st.run, r)
        return PathState(run=run, acc=st.acc + run, r=r, t=rec.t, last=last)

    def log_tilt(self, st):
        """``lambda * acc``; at the last point, ``lambda * r_hat``. ``[I, N]``. See `Potential.log_tilt`."""
        return self.lam(st.t) * (st.r if st.last else st.acc)

    def increment(self, prev, st):                                  # `last` is static
        """``lambda * run`` (the new running statistic); at the last point, ``lambda (r_hat - acc_prev)``.

        See `Potential.increment`.
        """
        return self.lam(st.t) * ((st.r - prev.acc) if st.last else st.run)


@dataclass(frozen=True)
class FKSum(FKMax):                   # identical except the running statistic
    """Path potential: as FKMax with the running sum of r_hat.

    Parameters
    ----------
    lam : Callable[[array], array]
        Tilt strength ``lam(t [I, N]) -> [I, N]``. Must be constant in t (it may vary by ladder position).

    Notes
    -----
    Exact without resampling (``Never()``). With resampling it is reliable only when lambda * S**2 / 2 * sigma_r is
    small, sigma_r being the spread of the final predicted reward across particles; see FKMax.
    """

    def _running(self, run, r):
        return run + r


@dataclass(frozen=True)
class TiltEnds:                       # C = -l(start) + [last] l(end): CREPE's endpoint tilts
    """CREPE's endpoint tilts: ``C = -l(start) + [last] l(end)``, with ``l = lambda(t) r_hat``.

    The increment is exactly 0 at interior steps and exactly ``l(end)`` at the last.

    Parameters
    ----------
    lam : Callable[[array], array]
        Tilt strength ``lam(t) -> same shape``, e.g. `level_lambda`.

    Attributes
    ----------
    requires : frozenset of Feature
        ``{Feature.REWARD}``.
    """

    lam: Callable
    requires = frozenset({Feature.REWARD})

    def init(self, rec):
        """Record the start point's ``r_hat`` and ``t``. See `Potential.init`."""
        r = rec.need(Feature.REWARD)
        return EndsState(r0=r, t0=rec.t, r=r, t=rec.t, last=False)

    def update(self, st, rec, *, last):
        """Keep the start point; record the new point's ``r_hat`` and ``t``. See `Potential.update`."""
        return EndsState(r0=st.r0, t0=st.t0, r=rec.need(Feature.REWARD), t=rec.t, last=last)

    def log_tilt(self, st):
        """``-lambda(t0) r_hat_0``, plus ``lambda(t) r_hat`` at the last point. See `Potential.log_tilt`."""
        start = -self.lam(st.t0) * st.r0
        return start + self.lam(st.t) * st.r if st.last else start

    def increment(self, prev, st):
        """0 at interior points; ``lambda(t) r_hat`` at the last point. See `Potential.increment`."""
        return self.lam(st.t) * st.r if st.last else namespace(st.r).zeros_like(st.r)


@dataclass(frozen=True)
class Sum:
    """The sum of several potentials: log-tilts and increments add; the state is the tuple of the parts' states.

    Parameters
    ----------
    *parts : Potential
        The potentials to add.
    """

    parts: tuple

    def __init__(self, *parts):
        object.__setattr__(self, "parts", parts)

    @property
    def requires(self):
        """Union of the parts' `requires` (frozenset of Feature)."""
        return frozenset().union(*(p.requires for p in self.parts))

    def init(self, rec):
        """Tuple of the parts' initial states. See `Potential.init`."""
        return tuple(p.init(rec) for p in self.parts)

    def update(self, st, rec, *, last):
        """Tuple of the parts' updated states. See `Potential.update`."""
        return tuple(p.update(a, rec, last=last) for p, a in zip(self.parts, st))

    def log_tilt(self, st):
        """Sum of the parts' log-tilts. See `Potential.log_tilt`."""
        return sum(p.log_tilt(a) for p, a in zip(self.parts, st))

    def increment(self, prev, st):
        """Sum of the parts' increments. See `Potential.increment`."""
        return sum(p.increment(a, b) for p, a, b in zip(self.parts, prev, st))


@dataclass(frozen=True)
class Zero:
    """The trivial potential: C = 0. Its state is a zero array ``[I, N]`` in the run's dtype and device.

    Attributes
    ----------
    requires : frozenset of Feature
        Empty.
    """

    requires = frozenset()

    def init(self, rec):
        """Zeros shaped like ``rec.t`` (``[I, N]``). See `Potential.init`."""
        return _zeros(rec.t)

    def update(self, st, rec, *, last):
        """Return `st` unchanged. See `Potential.update`."""
        return st

    def log_tilt(self, st):
        """Return the zero state. See `Potential.log_tilt`."""
        return st

    def increment(self, prev, st):
        """Return the zero state. See `Potential.increment`."""
        return st


@dataclass(frozen=True)
class ReverseRatio:                   # scores the reverse step with the kernels opposite to the run
    """Scores the reverse of each step with the kernels opposite to the run's direction.

    Each step from ``x_prev`` (time ``t_prev``) to ``rec`` adds the term
    ``log proposal_rev(x_prev | rec) - log target_rev(x_prev | rec)``, computed with `Gaussian.log_ratio`.

    Parameters
    ----------
    target_rev : Kernel
        The target's reverse kernel.
    proposal_rev : Kernel
        The proposal's reverse kernel. If it is the same object as `target_rev` the term is identically 0 and the
        state is never updated.
    """

    target_rev: Any
    proposal_rev: Any

    @property
    def trivial(self):                # static: term is identically 0
        """bool: True (static) if `target_rev` is `proposal_rev`, so every term is identically 0."""
        return self.target_rev is self.proposal_rev

    @property
    def requires(self):
        """Union of the two kernels' `requires`, or empty if `trivial` (frozenset of Feature)."""
        return frozenset() if self.trivial else self.target_rev.requires | self.proposal_rev.requires

    def init(self, rec):
        """Zero sum and term; remember the start point. See `Potential.init`."""
        z = _zeros(rec.t)
        return RRState(sum=z, term=z, x=rec.x, t=rec.t)

    def update(self, st, rec, *, last):
        """Score the reverse step from `rec` back to the previous point. See `Potential.update`."""
        if self.trivial:
            return st
        term = self.proposal_rev.at(rec, st.t).log_ratio(self.target_rev.at(rec, st.t), st.x)   # §7
        return RRState(sum=st.sum + term, term=term, x=rec.x, t=rec.t)

    def log_tilt(self, st):
        """Return the running sum of terms. See `Potential.log_tilt`."""
        return st.sum

    def increment(self, prev, st):
        """Return the latest term. See `Potential.increment`."""
        return st.term
