"""Moves (§11): MCMC kernels at a fixed time that leave the current target invariant (never touch log_weights)."""

from __future__ import annotations

from dataclasses import replace
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from .exchange import StateSwap, replica_exchange
from .kernels import Gaussian
from .ops import bcast, device_of, namespace, pytree, tree_where
from .records import Feature, Point, Record

State = Any


@pytree
class MoveContext:          # one pytree: swaps and accept/reject keep it consistent
    """Everything a move reads and updates, as one pytree so swaps and accept/reject keep it consistent.

    Parameters
    ----------
    record : Record
        Current point ``[I, N, ...]``, evaluated with this step's needs.
    potential_state : State
        Potential state including `record`.
    prev_potential_state : State or None
        Potential state before `record`; None at the start point.
    reference : Gaussian or None
        The target kernel's distribution that `record` was drawn from (the prior at the start point); the reference
        measure of pCN proposals.
    """

    record: Record          # current point (evaluated with this step's needs)
    potential_state: State                # potential state including record
    prev_potential_state: State | None    # potential state before record (None at the start point)
    reference: Gaussian | None            # the target kernel's distribution that record was drawn from


class Move(Protocol):
    """Protocol for a move: an MCMC kernel at a fixed time that leaves the current target invariant.

    A move never changes the log-weights (resample-move SMC).

    Attributes
    ----------
    requires : frozenset of Feature
        Features the move reads from ``ctx.record``.
    """

    requires: frozenset[Feature]          # features the move reads from ctx.record

    def __call__(self, ctx, *, evaluate_point, potential, is_last, move_count, key, backend) -> MoveContext:
        """Apply the move.

        Parameters
        ----------
        ctx : MoveContext
            The current point, its potential states and reference.
        evaluate_point : Callable[[Point], Record]
            Evaluates a candidate with this step's needs (supplied by the runner), so accepted candidates are
            complete records.
        potential : Potential
            The run's potential; acceptances compare its increments.
        is_last : bool
            Static. True at the final point of the run.
        move_count : int or array
            How many times this move has already fired in this run (traced inside scan). Used for counters such
            as the replica-exchange sweep.
        key : key
            Random key.
        backend : Backend
            Randomness (and control flow) for the move.

        Returns
        -------
        MoveContext
            The updated context.
        """
        ...
    # evaluate_point(point) -> Record, supplied by the runner with this step's needs (I4).
    # move_count: how many times this move has already fired in this run (traced inside scan).


def rescore(potential, prev_state, record, is_last):   # potential state if `record` replaced the current point
    """Return the potential state if `record` replaced the current point.

    Parameters
    ----------
    potential : Potential
        The run's potential.
    prev_state : State or None
        State before the current point; None at the start point.
    record : Record
        The replacement point.
    is_last : bool
        Static. True at the final point of the run.

    Returns
    -------
    State
        ``potential.init(record)`` if `prev_state` is None, else
        ``potential.update(prev_state, record, is_last=is_last)``.
    """
    if prev_state is None:
        return potential.init(record)
    return potential.update(prev_state, record, is_last=is_last)


def tilt_delta(potential, prev_state, state_a, state_b):   # C(state_a) - C(state_b), both extending one history
    """``C(state_a) - C(state_b)`` for two states that extend the same history `prev_state`.

    The shared prefix is never formed.

    Parameters
    ----------
    potential : Potential
        The run's potential.
    prev_state : State or None
        The shared history; None at the start point, where C is a single term and `log_tilt` is differenced.
    state_a, state_b : State
        States after two alternative points.

    Returns
    -------
    array
        ``[I, N]``; the log-acceptance of `state_a` over `state_b` under a reference-reversible proposal.
    """
    if prev_state is None:                       # static: at the start point, C is a single term
        return potential.log_tilt(state_a) - potential.log_tilt(state_b)
    return potential.increment(prev_state, state_a) - potential.increment(prev_state, state_b)   # no shared prefix


@dataclass(frozen=True)
class PCN:
    """`n_proposals` pCN proposals reversible w.r.t. ctx.reference, each optionally followed by an exchange sweep.

    With ``reference = N(mu, tau**2)``, the proposal is ``x' = mu + sqrt(1 - rho**2) (x - mu) + rho tau xi``, accepted
    with log-probability `tilt_delta` (for `FKDiff`, ``lambda [r_hat(x') - r_hat(x)]``). Each proposal costs one
    evaluation. RpCN's move.

    Parameters
    ----------
    n_proposals : int
        pCN proposals per move call.
    step_size : Callable[[array], array]
        ``rho`` in (0, 1) per row, ``step_size(t [I, N]) -> [I, N]``, e.g. ``constant(0.95)`` or ``ladder(step_sizes)``.
    pairs : PairSchedule or None
        If given, a replica-exchange sweep (`StateSwap`) along axis 1 after each pCN step.

    Attributes
    ----------
    requires : frozenset of Feature
        Empty: acceptance reads only the potential.
    """

    n_proposals: int                     # pCN proposals per move call (M)
    step_size: Callable                  # rho in (0, 1) per row, e.g. ladder(step_sizes)
    pairs: Any = None                    # PairSchedule: replica exchange along axis 1 after each pCN step
    requires = frozenset()               # acceptance reads only the potential

    def __call__(self, ctx, *, evaluate_point, potential, is_last, move_count, key, backend):
        """Apply `n_proposals` pCN steps (and exchange sweeps). See `Move.__call__`."""
        for substep in range(self.n_proposals):                          # static
            key_noise, key_accept, key_exchange, key = backend.split(key, 4)
            x, reference = ctx.record.x, ctx.reference
            xp = namespace(x)
            rho = bcast(self.step_size(ctx.record.t), x)
            noise = backend.normal(key_noise, x.shape, x.dtype, device=device_of(x))
            x_candidate = (reference.mean + xp.sqrt(1 - rho**2) * (x - reference.mean)
                           + rho * bcast(xp.sqrt(reference.var), x) * noise)
            candidate = evaluate_point(Point(x_candidate, ctx.record.t))   # 1 evaluation per proposal
            candidate_state = rescore(potential, ctx.prev_potential_state, candidate, is_last)
            log_accept_ratio = tilt_delta(potential, ctx.prev_potential_state, candidate_state, ctx.potential_state)
            uniform = backend.uniform(key_accept, log_accept_ratio.shape, log_accept_ratio.dtype,
                                      device=device_of(log_accept_ratio))
            accepted = xp.log(uniform) < log_accept_ratio
            ctx = replace(ctx, record=tree_where(accepted, candidate, ctx.record),
                          potential_state=tree_where(accepted, candidate_state, ctx.potential_state))
            if self.pairs is not None:                                   # static
                ctx = replica_exchange(ctx, self.pairs, StateSwap(potential),
                                       sweep=move_count * self.n_proposals + substep,   # pCN steps so far
                                       key=key_exchange, backend=backend)
        return ctx


@dataclass(frozen=True)
class LangevinMove:
    """CREPE's local exploration move; unadjusted (no Metropolis correction).

    Draws one sample from ``kernel.at(record, record.t)`` at the current time and always accepts it.

    Parameters
    ----------
    kernel : Kernel
        A same-time kernel, e.g. ``Guided(Langevin(h), tilt(lam, 0.5))``.
    """

    kernel: Any                          # e.g. Guided(Langevin(h), tilt(lam, 0.5))

    @property
    def requires(self):
        """Features the kernel requires (frozenset of Feature)."""
        return self.kernel.requires

    def __call__(self, ctx, *, evaluate_point, potential, is_last, move_count, key, backend):
        """Take one unadjusted step. See `Move.__call__`."""
        x = self.kernel.at(ctx.record, ctx.record.t).sample(key, backend)
        record = evaluate_point(Point(x, ctx.record.t))
        return replace(ctx, record=record,
                       potential_state=rescore(potential, ctx.prev_potential_state, record, is_last))
