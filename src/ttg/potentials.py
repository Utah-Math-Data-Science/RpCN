"""Potentials (§9): cumulative log-tilt C of a history, and per-step increments. State is lambda-free."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from .ops import device_of, namespace, pytree, static
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

    def init(self, record) -> State:                           # state after the first point
        """Start a history at its first point.

        Parameters
        ----------
        record : Record
            The first point, ``[I, N, ...]``, carrying `requires`.

        Returns
        -------
        State
            A pytree of arrays with leading axes ``[I, N]``.
        """
        ...

    def update(self, state, record, *, is_last: bool) -> State:     # absorb the next point; `is_last` is static
        """Extend a history by one point.

        Parameters
        ----------
        state : State
            State of the history so far.
        record : Record
            The next point, ``[I, N, ...]``, carrying `requires`.
        is_last : bool
            Static. True if `record` is the final point of the run.

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
            ``update(prev, record, is_last=...)`` for some `record`.

        Returns
        -------
        array
            ``[I, N]``.
        """
        ...
    # precondition: state = update(prev, record, is_last=...) for some record


class TiltDifference:
    """Mixin: the default increment, for potentials whose C stays O(one term)."""

    def increment(self, prev, state):
        """``log_tilt(state) - log_tilt(prev)``; accurate only when C is a single term.

        Parameters
        ----------
        prev : State
            State before the step.
        state : State
            State after the step.

        Returns
        -------
        array
            ``[I, N]``.
        """
        return self.log_tilt(state) - self.log_tilt(prev)


@pytree
class TiltState:
    """State of `FKDiff`: the current ``r_hat`` and ``t``, ``[I, N]`` each."""

    reward: Array
    t: Array


@pytree
class PathState:
    """State of `FKMax`/`FKSum`: running statistic, cumulative sum, current ``r_hat`` and ``t``; static `is_last`."""

    running: Array
    cumulative: Array
    reward: Array
    t: Array
    is_last: bool = static(False)


@pytree
class EndsState:
    """State of `TiltEnds`: start and current ``r_hat`` and ``t``; static `is_last`."""

    reward_start: Array
    t_start: Array
    reward: Array
    t: Array
    is_last: bool = static(False)


@pytree
class ReverseRatioState:
    """State of `ReverseRatio`: running total, latest term, previous point ``x`` and ``t``."""

    total: Array
    last_term: Array
    x: Array
    t: Array


def _zeros(t):
    return namespace(t).zeros(t.shape, dtype=t.dtype, device=device_of(t))


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

    def init(self, record):
        """State ``(r_hat, t)`` of the first point. See `Potential.init`."""
        return TiltState(reward=record.need(Feature.REWARD), t=record.t)

    def update(self, state, record, *, is_last):
        """State ``(r_hat, t)`` of the new point; the history is not needed. See `Potential.update`."""
        return TiltState(reward=record.need(Feature.REWARD), t=record.t)

    def log_tilt(self, state):
        """``lambda(t) r_hat``, ``[I, N]``. See `Potential.log_tilt`."""
        return self.lam(state.t) * state.reward


@dataclass(frozen=True)
class FKMax:
    """Path potential: intermediate tilt lambda * cumulative, the sum over steps of the running max of r_hat.

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

    def init(self, record):
        """State with running statistic, its cumulative sum and ``r_hat`` all equal to the first ``r_hat``.

        See `Potential.init`.
        """
        reward = record.need(Feature.REWARD)
        return PathState(running=reward, cumulative=reward, reward=reward, t=record.t, is_last=False)

    def _update_running(self, running, reward):
        return namespace(running, reward).maximum(running, reward)

    def update(self, state, record, *, is_last):
        """Update the running statistic with the new ``r_hat`` and add it to the cumulative sum.

        See `Potential.update`.
        """
        reward = record.need(Feature.REWARD)
        running = self._update_running(state.running, reward)
        return PathState(running=running, cumulative=state.cumulative + running, reward=reward, t=record.t,
                         is_last=is_last)

    def log_tilt(self, state):
        """``lambda * cumulative``; at the last point, ``lambda * r_hat``. ``[I, N]``. See `Potential.log_tilt`."""
        return self.lam(state.t) * (state.reward if state.is_last else state.cumulative)

    def increment(self, prev, state):                               # `is_last` is static
        """``lambda * running`` (the new running statistic); at the last point, ``lambda (r_hat - cumulative_prev)``.

        See `Potential.increment`.
        """
        return self.lam(state.t) * ((state.reward - prev.cumulative) if state.is_last else state.running)


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

    def _update_running(self, running, reward):
        return running + reward


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

    def init(self, record):
        """Record the start point's ``r_hat`` and ``t``. See `Potential.init`."""
        reward = record.need(Feature.REWARD)
        return EndsState(reward_start=reward, t_start=record.t, reward=reward, t=record.t, is_last=False)

    def update(self, state, record, *, is_last):
        """Keep the start point; record the new point's ``r_hat`` and ``t``. See `Potential.update`."""
        return EndsState(reward_start=state.reward_start, t_start=state.t_start, reward=record.need(Feature.REWARD),
                         t=record.t, is_last=is_last)

    def log_tilt(self, state):
        """``-lambda(t_start) r_hat_start``, plus ``lambda(t) r_hat`` at the last point. See `Potential.log_tilt`."""
        start_term = -self.lam(state.t_start) * state.reward_start
        return start_term + self.lam(state.t) * state.reward if state.is_last else start_term

    def increment(self, prev, state):
        """0 at interior points; ``lambda(t) r_hat`` at the last point. See `Potential.increment`."""
        return self.lam(state.t) * state.reward if state.is_last else namespace(state.reward).zeros_like(state.reward)


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
        return frozenset().union(*(part.requires for part in self.parts))

    def init(self, record):
        """Tuple of the parts' initial states. See `Potential.init`."""
        return tuple(part.init(record) for part in self.parts)

    def update(self, state, record, *, is_last):
        """Tuple of the parts' updated states. See `Potential.update`."""
        return tuple(part.update(part_state, record, is_last=is_last)
                     for part, part_state in zip(self.parts, state))

    def log_tilt(self, state):
        """Sum of the parts' log-tilts. See `Potential.log_tilt`."""
        return sum(part.log_tilt(part_state) for part, part_state in zip(self.parts, state))

    def increment(self, prev, state):
        """Sum of the parts' increments. See `Potential.increment`."""
        return sum(part.increment(part_prev, part_state)
                   for part, part_prev, part_state in zip(self.parts, prev, state))


@dataclass(frozen=True)
class Zero:
    """The trivial potential: C = 0. Its state is a zero array ``[I, N]`` in the run's dtype and device.

    Attributes
    ----------
    requires : frozenset of Feature
        Empty.
    """

    requires = frozenset()

    def init(self, record):
        """Zeros shaped like ``record.t`` (``[I, N]``). See `Potential.init`."""
        return _zeros(record.t)

    def update(self, state, record, *, is_last):
        """Return `state` unchanged. See `Potential.update`."""
        return state

    def log_tilt(self, state):
        """Return the zero state. See `Potential.log_tilt`."""
        return state

    def increment(self, prev, state):
        """Return the zero state. See `Potential.increment`."""
        return state


@pytree
class RelativeState:
    """State of `Relative`: the inner potential's state and ``C(start)``, ``[I, N]``."""

    inner: State
    start_tilt: Array


@dataclass(frozen=True)
class Relative:                       # C - C(start): for runs that start from points already tilted by C(start)
    """``C - C(start)``: the inner potential relative to its value at the start point.

    For runs whose start points are already distributed with the tilt ``C(start)`` (e.g. `crepe_finish` from
    tilted level-0 samples): the starting weight is exactly 0, and increments are the inner potential's.

    Parameters
    ----------
    inner : Potential
        The potential to offset.
    """

    inner: Any

    @property
    def requires(self):
        """The inner potential's `requires` (frozenset of Feature)."""
        return self.inner.requires

    def init(self, record):
        """The inner state and its log-tilt at the start. See `Potential.init`."""
        state = self.inner.init(record)
        return RelativeState(inner=state, start_tilt=self.inner.log_tilt(state))

    def update(self, state, record, *, is_last):
        """Update the inner state; keep ``C(start)``. See `Potential.update`."""
        return RelativeState(inner=self.inner.update(state.inner, record, is_last=is_last),
                             start_tilt=state.start_tilt)

    def log_tilt(self, state):
        """``inner.log_tilt - C(start)``; exactly 0 at the start point. See `Potential.log_tilt`."""
        return self.inner.log_tilt(state.inner) - state.start_tilt   # x - x == 0 exactly at the start

    def increment(self, prev, state):
        """The inner potential's increment. See `Potential.increment`."""
        return self.inner.increment(prev.inner, state.inner)


@dataclass(frozen=True)
class ReverseRatio:                   # scores the reverse step with the kernels opposite to the run
    """Scores the reverse of each step with the kernels opposite to the run's direction.

    Each step from ``x_prev`` (time ``t_prev``) to ``record`` adds the term
    ``log proposal_reverse(x_prev | record) - log target_reverse(x_prev | record)``, computed with `Gaussian.log_ratio`.

    Parameters
    ----------
    target_reverse : Kernel
        The target's reverse kernel.
    proposal_reverse : Kernel
        The proposal's reverse kernel. If it is the same object as `target_reverse` the term is identically 0 and the
        state is never updated.
    """

    target_reverse: Any
    proposal_reverse: Any

    @property
    def is_trivial(self):             # static: term is identically 0
        """bool: True (static) if `target_reverse` is `proposal_reverse`, so every term is identically 0."""
        return self.target_reverse is self.proposal_reverse

    @property
    def requires(self):
        """Union of the two kernels' `requires`, or empty if `is_trivial` (frozenset of Feature)."""
        return frozenset() if self.is_trivial else self.target_reverse.requires | self.proposal_reverse.requires

    def init(self, record):
        """Zero total and term; remember the start point. See `Potential.init`."""
        zeros = _zeros(record.t)
        return ReverseRatioState(total=zeros, last_term=zeros, x=record.x, t=record.t)

    def update(self, state, record, *, is_last):
        """Score the reverse step from `record` back to the previous point. See `Potential.update`."""
        if self.is_trivial:
            return state
        term = self.proposal_reverse.at(record, state.t).log_ratio(self.target_reverse.at(record, state.t),
                                                                   state.x)                     # §7
        return ReverseRatioState(total=state.total + term, last_term=term, x=record.x, t=record.t)

    def log_tilt(self, state):
        """Return the running total of terms. See `Potential.log_tilt`."""
        return state.total

    def increment(self, prev, state):
        """Return the latest term. See `Potential.increment`."""
        return state.last_term
