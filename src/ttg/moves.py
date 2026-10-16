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
    rec: Record             # current point (evaluated with this step's needs)
    st: State               # potential state including rec
    st_prev: State | None   # potential state before rec (None at the start point)
    ref: Gaussian | None    # the target kernel's distribution that rec was drawn from


class Move(Protocol):
    requires: frozenset[Feature]          # features the move reads from ctx.rec
    def __call__(self, ctx, *, ev, potential, last, count, key, bk) -> MoveContext: ...
    # ev(point) -> Record, supplied by the runner with this step's needs (I4).
    # count: how many times this move has already fired in this run (traced inside scan).


def rescore(potential, st_prev, rec, last):      # potential state if `rec` replaced the current point
    return potential.init(rec) if st_prev is None else potential.update(st_prev, rec, last=last)


def tilt_delta(potential, st_prev, a, b):        # C(a) - C(b) for two states that extend the same history
    if st_prev is None:                          # static: at the start point, C is a single term
        return potential.log_tilt(a) - potential.log_tilt(b)
    return potential.increment(st_prev, a) - potential.increment(st_prev, b)   # shared prefix never formed


@dataclass(frozen=True)
class PCN:
    """`n_steps` pCN proposals reversible w.r.t. ctx.ref, each optionally followed by an exchange sweep."""

    n_steps: int                         # pCN proposals per move call (M)
    step: Callable                       # rho in (0, 1) per row, e.g. ladder(step_sizes)
    pairs: Any = None                    # PairSchedule: replica exchange along axis 1 after each pCN step
    requires = frozenset()               # acceptance reads only the potential

    def __call__(self, ctx, *, ev, potential, last, count, key, bk):
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
    """CREPE's local exploration move; unadjusted (no Metropolis correction)."""

    kernel: Any                          # e.g. Guided(Langevin(h), tilt(lam, 0.5))

    @property
    def requires(self):
        return self.kernel.requires

    def __call__(self, ctx, *, ev, potential, last, count, key, bk):
        rec = ev(Point(self.kernel.at(ctx.rec, ctx.rec.t).sample(key, bk), ctx.rec.t))
        return replace(ctx, rec=rec, st=rescore(potential, ctx.st_prev, rec, last))
