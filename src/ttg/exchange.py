"""Replica exchange (§12): which pairs (a PairSchedule) and how a pair proposes (an Exchange), one driver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .ops import axis1_size, bcast, device, leaves, namespace, put_levels, take_levels, tree_map, tree_where

Tree = Any
Array = Any


class PairSchedule(Protocol):
    """Protocol for which adjacent ladder pairs to propose swapping, in which rounds.

    Attributes
    ----------
    period : int
        Rounds depend on `sweep` only through ``sweep % period``.
    """

    period: int                                              # rounds depend on `sweep` only via sweep % period

    def rounds(self, n: int, sweep: int) -> tuple[tuple[tuple[int, int], ...], ...]:
        """Return the rounds of one sweep. Static.

        Parameters
        ----------
        n : int
            Ladder size (axis 1).
        sweep : int
            Sweep number, in ``range(period)`` when called by `replica_exchange`.

        Returns
        -------
        tuple
            Rounds in order; each round is a tuple of disjoint adjacent pairs ``(lo, lo + 1)``, proposed together.
        """
        ...
    # static; each round is a tuple of disjoint adjacent pairs (lo, lo + 1), proposed together


@dataclass(frozen=True)
class SequentialSweep:          # (0,1), then (1,2), ..., each its own round
    """Pair schedule: ``(0, 1)``, then ``(1, 2)``, ..., each its own round. Period 1."""

    period = 1

    def rounds(self, n, sweep):
        """``n - 1`` rounds of one pair each. See `PairSchedule.rounds`."""
        return tuple(((k, k + 1),) for k in range(n - 1))


@dataclass(frozen=True)
class EvenOdd:                  # deterministic even-odd (DEO): all pairs of one parity at once, alternating
    """Pair schedule, deterministic even-odd: all pairs of one parity at once, alternating parity. Period 2."""

    period = 2

    def rounds(self, n, sweep):
        """One round: the pairs ``(m - 1, m)`` with ``m % 2 == sweep % 2``. See `PairSchedule.rounds`."""
        return (tuple((m - 1, m) for m in range(1, n) if m % 2 == sweep % 2),)


class Exchange(Protocol):
    """Protocol for how a set of ladder pairs proposes a swap."""

    def propose(self, ladder, lo, hi, key, bk) -> tuple[Tree, Tree, Array]:
        """Propose new contents for positions `lo` and `hi`.

        Parameters
        ----------
        ladder : pytree
            Arrays whose axis 1 is the ladder.
        lo, hi : list of int
            P pair positions, ``hi[p] = lo[p] + 1``.
        key : key
            Random key.
        bk : Backend
            Randomness, if the proposal needs any.

        Returns
        -------
        new_lo, new_hi : pytree
            Candidates for positions `lo` and `hi`, ``[I, P, ...]``.
        log_a : array
            ``[I, P]`` log-acceptance of each pair's proposal.
        """
        ...
    # candidates for positions lo and hi (pytrees [I, P, ...]) and log-acceptance [I, P]


def replica_exchange(ladder, pairs, exchange, *, sweep, key, bk):
    """Run one replica-exchange sweep along axis 1 of `ladder`, accepting each pair independently.

    Parameters
    ----------
    ladder : pytree
        Arrays whose axis 1 is the ladder (e.g. a `MoveContext` or CREPE's replicas).
    pairs : PairSchedule
        Which pairs, in which rounds.
    exchange : Exchange
        How a pair proposes.
    sweep : int or array
        Python int (static) or traced int scalar (e.g. from the runner's step index); selects the rounds via
        ``sweep % pairs.period`` (with `bk.switch` if traced).
    key : key
        Random key.
    bk : Backend
        Uniform draws for acceptance; `switch` on a traced `sweep`.

    Returns
    -------
    pytree
        `ladder` after the sweep.
    """
    n = axis1_size(ladder)
    whole_states = hasattr(exchange, "log_accept")                      # static: e.g. StateSwap

    def for_residue(res):                                                # res: static int in range(period)
        def run(ladder):
            rounds = [r for r in pairs.rounds(n, res) if r]
            if whole_states:
                return _swap_states(ladder, rounds, exchange, key, bk)
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


def _permute(tree, perm):                        # perm [I, n]: position k takes the state at perm[i, k]
    def take(a):
        xp = namespace(a, perm)
        idx = xp.broadcast_to(bcast(perm, a), tuple(perm.shape) + tuple(a.shape[2:]))
        return xp.take_along_axis(a, idx, axis=1)

    return tree_map(take, tree)


def _swap_states(ladder, rounds, exchange, key, bk):
    """`replica_exchange`'s rounds for an exchange that swaps whole states (it has ``view`` and ``log_accept``).

    Each round decides acceptance from ``exchange.view(ladder)`` alone, through a per-row permutation of the
    ladder positions; the whole ladder is permuted once at the end. Same draws and result as the generic path.
    """
    view = exchange.view(ladder)
    lv = leaves(view)[0]
    xp = namespace(lv)
    n = lv.shape[1]
    perm = xp.broadcast_to(xp.arange(n, device=device(lv))[None, :], (lv.shape[0], n))
    for pr, kk in zip(rounds, bk.split(key, max(len(rounds), 1))):
        lo, hi = [a for a, _ in pr], [b for _, b in pr]
        _, k_a = bk.split(kk, 2)                                         # k_p unused, as in StateSwap.propose
        log_a = exchange.log_accept(_permute(view, perm), lo, hi)
        u = bk.uniform(k_a, log_a.shape, log_a.dtype, device=device(log_a))
        acc = xp.log(u) < log_a                                          # [I, P]
        p_lo, p_hi = take_levels(perm, lo), take_levels(perm, hi)
        perm = put_levels(put_levels(perm, lo, xp.where(acc, p_hi, p_lo)), hi, xp.where(acc, p_lo, p_hi))
    return _permute(ladder, perm) if rounds else ladder


@dataclass(frozen=True)
class StateSwap:                # plain parallel tempering: exchange states; 0 evaluations
    """Plain parallel tempering: exchange whole states between adjacent positions; no evaluations.

    The log-acceptance is the change in ``potential.log_tilt`` summed over both positions; for
    ``FKDiff(ladder(betas))`` that is ``(beta_{k+1} - beta_k) (r_hat(x_k) - r_hat(x_{k+1}))``.

    Parameters
    ----------
    potential : Potential
        Read at each row's ladder position, so its state must be lambda-free.
    """

    potential: Any

    def propose(self, ctx, lo, hi, key, bk):
        """Swap ``rec``, ``st``, ``st_prev`` and ``ref`` together. See `Exchange.propose`.

        Parameters
        ----------
        ctx : MoveContext
            ``[I, n, ...]``.
        lo, hi : list of int
            Pair positions.
        key : key
            Unused.
        bk : Backend
            Unused.
        """
        swapped = take_levels(ctx, _swap_perm(axis1_size(ctx), lo, hi))   # rec, st, st_prev, ref move together
        return take_levels(swapped, lo), take_levels(swapped, hi), self.log_accept(ctx.st, lo, hi)

    def view(self, ctx):
        """The part of the ladder `log_accept` reads: the potential state ``ctx.st``."""
        return ctx.st

    def log_accept(self, st, lo, hi):
        """``[I, P]`` log-acceptance of swapping positions ``lo[p]`` and ``hi[p]``, from the potential state ``st``.

        Lets `replica_exchange` decide swaps without moving the whole state each round.
        """
        swapped = take_levels(st, _swap_perm(axis1_size(st), lo, hi))
        d = self.potential.log_tilt(swapped) - self.potential.log_tilt(st)    # [I, n], lambda at each position
        return take_levels(d, lo) + take_levels(d, hi)


def _swap_perm(n, lo, hi):
    perm = list(range(n))
    for a, b in zip(lo, hi):
        perm[a], perm[b] = b, a
    return perm
