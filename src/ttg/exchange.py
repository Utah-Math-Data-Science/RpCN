"""Replica exchange (§12): which pairs (a PairSchedule) and how a pair proposes (an Exchange), one driver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .ops import axis1_size, device, namespace, put_levels, take_levels, tree_where

Tree = Any
Array = Any


class PairSchedule(Protocol):
    period: int                                              # rounds depend on `sweep` only via sweep % period
    def rounds(self, n: int, sweep: int) -> tuple[tuple[tuple[int, int], ...], ...]: ...
    # static; each round is a tuple of disjoint adjacent pairs (lo, lo + 1), proposed together


@dataclass(frozen=True)
class SequentialSweep:          # (0,1), then (1,2), ..., each its own round
    period = 1

    def rounds(self, n, sweep):
        return tuple(((k, k + 1),) for k in range(n - 1))


@dataclass(frozen=True)
class EvenOdd:                  # deterministic even-odd (DEO): all pairs of one parity at once, alternating
    period = 2

    def rounds(self, n, sweep):
        return (tuple((m - 1, m) for m in range(1, n) if m % 2 == sweep % 2),)


class Exchange(Protocol):
    def propose(self, ladder, lo, hi, key, bk) -> tuple[Tree, Tree, Array]: ...
    # candidates for positions lo and hi (pytrees [I, P, ...]) and log-acceptance [I, P]


def replica_exchange(ladder, pairs, exchange, *, sweep, key, bk):
    """sweep: Python int (static) or traced int scalar (e.g. from the runner's step index)."""
    n = axis1_size(ladder)

    def for_residue(res):                                                # res: static int in range(period)
        def run(ladder):
            rounds = [r for r in pairs.rounds(n, res) if r]
            for pr, kk in zip(rounds, bk.split(key, max(len(rounds), 1))):
                lo, hi = [a for a, _ in pr], [b for _, b in pr]
                k_p, k_a = bk.split(kk, 2)
                new_lo, new_hi, log_a = exchange.propose(ladder, lo, hi, k_p, bk)
                xp = namespace(log_a)
                u = bk.uniform(k_a, log_a.shape, log_a.dtype, device=device(log_a))
                acc = xp.log(u) < log_a                                  # [I, P]
                ladder = put_levels(ladder, lo, tree_where(acc, new_lo, take_levels(ladder, lo)))
                ladder = put_levels(ladder, hi, tree_where(acc, new_hi, take_levels(ladder, hi)))
            return ladder
        return run

    if isinstance(sweep, int):                                           # static: no dispatch
        return for_residue(sweep % pairs.period)(ladder)
    return bk.switch(sweep % pairs.period, [for_residue(r) for r in range(pairs.period)], ladder)


@dataclass(frozen=True)
class StateSwap:                # plain parallel tempering: exchange states; 0 evaluations
    potential: Any

    def propose(self, ctx, lo, hi, key, bk):
        perm = list(range(axis1_size(ctx)))
        for a, b in zip(lo, hi):
            perm[a], perm[b] = b, a
        swapped = take_levels(ctx, perm)                           # rec, st, st_prev, ref move together
        d = self.potential.log_tilt(swapped.st) - self.potential.log_tilt(ctx.st)   # [I, n]
        return take_levels(swapped, lo), take_levels(swapped, hi), take_levels(d, lo) + take_levels(d, hi)
