"""Moves (§11): MCMC kernels at a fixed time that leave the current target invariant (never touch log_w)."""

from __future__ import annotations

from dataclasses import replace
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from .exchange import StateSwap, replica_exchange
from .kernels import Gaussian
from .ops import bcast, device, namespace, pytree, tree_where
from .records import Feature, Point, Record

State = Any


@pytree
class MoveContext:          # one pytree: swaps and accept/reject keep it consistent
    """Everything a move reads and updates, as one pytree so swaps and accept/reject keep it consistent.

    Parameters
    ----------
    rec : Record
        Current point ``[I, N, ...]``, evaluated with this step's needs.
    st : State
        Potential state including `rec`.
    st_prev : State or None
        Potential state before `rec`; None at the start point.
    ref : Gaussian or None
        The target kernel's distribution that `rec` was drawn from (the prior at the start point); the reference
        measure of pCN proposals.
    """

    rec: Record             # current point (evaluated with this step's needs)
    st: State               # potential state including rec
    st_prev: State | None   # potential state before rec (None at the start point)
    ref: Gaussian | None    # the target kernel's distribution that rec was drawn from


class Move(Protocol):
    """Protocol for a move: an MCMC kernel at a fixed time that leaves the current target invariant.

    A move never changes the log-weights (resample-move SMC).

    Attributes
    ----------
    requires : frozenset of Feature
        Features the move reads from ``ctx.rec``.
    """

    requires: frozenset[Feature]          # features the move reads from ctx.rec

    def __call__(self, ctx, *, ev, potential, last, count, key, bk) -> MoveContext:
        """Apply the move.

        Parameters
        ----------
        ctx : MoveContext
            The current point, its potential states and reference.
        ev : Callable[[Point], Record]
            Evaluates a candidate with this step's needs (supplied by the runner), so accepted candidates are
            complete records.
        potential : Potential
            The run's potential; acceptances compare its increments.
        last : bool
            Static. True at the final point of the run.
        count : int or array
            How many times this move has already fired in this run (traced inside scan). Used for counters such
            as the replica-exchange sweep.
        key : key
            Random key.
        bk : Backend
            Randomness (and control flow) for the move.

        Returns
        -------
        MoveContext
            The updated context.
        """
        ...
    # ev(point) -> Record, supplied by the runner with this step's needs (I4).
    # count: how many times this move has already fired in this run (traced inside scan).


def rescore(potential, st_prev, rec, last):      # potential state if `rec` replaced the current point
    """Return the potential state if `rec` replaced the current point.

    Parameters
    ----------
    potential : Potential
        The run's potential.
    st_prev : State or None
        State before the current point; None at the start point.
    rec : Record
        The replacement point.
    last : bool
        Static. True at the final point of the run.

    Returns
    -------
    State
        ``potential.init(rec)`` if `st_prev` is None, else ``potential.update(st_prev, rec, last=last)``.
    """
    return potential.init(rec) if st_prev is None else potential.update(st_prev, rec, last=last)


def tilt_delta(potential, st_prev, a, b):        # C(a) - C(b) for two states that extend the same history
    """``C(a) - C(b)`` for two states that extend the same history `st_prev`, without forming the shared prefix.

    Parameters
    ----------
    potential : Potential
        The run's potential.
    st_prev : State or None
        The shared history; None at the start point, where C is a single term and `log_tilt` is differenced.
    a, b : State
        States after two alternative points.

    Returns
    -------
    array
        ``[I, N]``; the log-acceptance of `a` over `b` under a ref-reversible proposal.
    """
    if st_prev is None:                          # static: at the start point, C is a single term
        return potential.log_tilt(a) - potential.log_tilt(b)
    return potential.increment(st_prev, a) - potential.increment(st_prev, b)   # shared prefix never formed


@dataclass(frozen=True)
class PCN:
    """`n_steps` pCN proposals reversible w.r.t. ctx.ref, each optionally followed by an exchange sweep.

    With ``ref = N(mu, tau**2)``, the proposal is ``x' = mu + sqrt(1 - rho**2) (x - mu) + rho tau xi``, accepted
    with log-probability `tilt_delta` (for `FKDiff`, ``lambda [r_hat(x') - r_hat(x)]``). Each proposal costs one
    evaluation. RpCN's move.

    Parameters
    ----------
    n_steps : int
        pCN proposals per move call.
    step : Callable[[array], array]
        ``rho`` in (0, 1) per row, ``step(t [I, N]) -> [I, N]``, e.g. ``constant(0.95)`` or ``ladder(step_sizes)``.
    pairs : PairSchedule or None
        If given, a replica-exchange sweep (`StateSwap`) along axis 1 after each pCN step.

    Attributes
    ----------
    requires : frozenset of Feature
        Empty: acceptance reads only the potential.
    """

    n_steps: int                         # pCN proposals per move call (M)
    step: Callable                       # rho in (0, 1) per row, e.g. ladder(step_sizes)
    pairs: Any = None                    # PairSchedule: replica exchange along axis 1 after each pCN step
    requires = frozenset()               # acceptance reads only the potential

    def __call__(self, ctx, *, ev, potential, last, count, key, bk):
        """Apply `n_steps` pCN steps (and exchange sweeps). See `Move.__call__`."""
        for m in range(self.n_steps):                                    # static
            k_p, k_a, k_x, key = bk.split(key, 4)
            x, ref = ctx.rec.x, ctx.ref
            xp = namespace(x)
            rho = bcast(self.step(ctx.rec.t), x)
            xi = bk.normal(k_p, x.shape, x.dtype, device=device(x))
            x_new = ref.mean + xp.sqrt(1 - rho**2) * (x - ref.mean) + rho * bcast(xp.sqrt(ref.var), x) * xi
            cand = ev(Point(x_new, ctx.rec.t))                           # 1 evaluation per proposal
            cand_st = rescore(potential, ctx.st_prev, cand, last)
            log_a = tilt_delta(potential, ctx.st_prev, cand_st, ctx.st)
            acc = xp.log(bk.uniform(k_a, log_a.shape, log_a.dtype, device=device(log_a))) < log_a
            ctx = replace(ctx, rec=tree_where(acc, cand, ctx.rec), st=tree_where(acc, cand_st, ctx.st))
            if self.pairs is not None:                                   # static
                ctx = replica_exchange(ctx, self.pairs, StateSwap(potential),
                                       sweep=count * self.n_steps + m, key=k_x, bk=bk)  # pCN steps so far
        return ctx


@dataclass(frozen=True)
class LangevinMove:
    """CREPE's local exploration move; unadjusted (no Metropolis correction).

    Draws one sample from ``kernel.at(rec, rec.t)`` at the current time and always accepts it.

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

    def __call__(self, ctx, *, ev, potential, last, count, key, bk):
        """Take one unadjusted step. See `Move.__call__`."""
        rec = ev(Point(self.kernel.at(ctx.rec, ctx.rec.t).sample(key, bk), ctx.rec.t))
        return replace(ctx, rec=rec, st=rescore(potential, ctx.st_prev, rec, last))
