"""Replica exchange (§12): which pairs (a PairSchedule) and how a pair proposes (an Exchange), one driver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .ops import bcast, device_of, leaves, n_levels, namespace, put_levels, take_levels, tree_map, tree_where

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

    def rounds(self, n_levels: int, sweep: int) -> tuple[tuple[tuple[int, int], ...], ...]:
        """Return the rounds of one sweep. Static.

        Parameters
        ----------
        n_levels : int
            Ladder size (axis 1).
        sweep : int
            Sweep number, in ``range(period)`` when called by `replica_exchange`.

        Returns
        -------
        tuple
            Rounds in order; each round is a tuple of disjoint adjacent pairs ``(lower, lower + 1)``, proposed together.
        """
        ...
    # static; each round is a tuple of disjoint adjacent pairs (lower, lower + 1), proposed together


@dataclass(frozen=True)
class SequentialSweep:          # (0,1), then (1,2), ..., each its own round
    """Pair schedule: ``(0, 1)``, then ``(1, 2)``, ..., each its own round. Period 1."""

    period = 1

    def rounds(self, n_levels, sweep):
        """``n_levels - 1`` rounds of one pair each. See `PairSchedule.rounds`."""
        return tuple(((level, level + 1),) for level in range(n_levels - 1))


@dataclass(frozen=True)
class EvenOdd:                  # deterministic even-odd (DEO): all pairs of one parity at once, alternating
    """Pair schedule, deterministic even-odd: all pairs of one parity at once, alternating parity. Period 2."""

    period = 2

    def rounds(self, n_levels, sweep):
        """One round: the pairs ``(level - 1, level)`` with ``level % 2 == sweep % 2``. See `PairSchedule.rounds`."""
        return (tuple((level - 1, level) for level in range(1, n_levels) if level % 2 == sweep % 2),)


class Exchange(Protocol):
    """Protocol for how a set of ladder pairs proposes a swap."""

    def propose(self, ladder, lower, upper, key, backend) -> tuple[Tree, Tree, Array]:
        """Propose new contents for positions `lower` and `upper`.

        Parameters
        ----------
        ladder : pytree
            Arrays whose axis 1 is the ladder.
        lower, upper : list of int
            P pair positions, ``upper[p] = lower[p] + 1``.
        key : key
            Random key.
        backend : Backend
            Randomness, if the proposal needs any.

        Returns
        -------
        new_lower, new_upper : pytree
            Candidates for positions `lower` and `upper`, ``[I, P, ...]``.
        log_accept_ratio : array
            ``[I, P]`` log-acceptance of each pair's proposal.
        """
        ...
    # candidates for positions lower and upper (pytrees [I, P, ...]) and log-acceptance [I, P]


def replica_exchange(ladder, pairs, exchange, *, sweep, key, backend):
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
        ``sweep % pairs.period`` (with `backend.switch` if traced).
    key : key
        Random key.
    backend : Backend
        Uniform draws for acceptance; `switch` on a traced `sweep`.

    Returns
    -------
    pytree
        `ladder` after the sweep.
    """
    swaps_whole_states = hasattr(exchange, "log_accept")                # static: e.g. StateSwap

    def for_residue(residue):                                            # residue: static int in range(period)
        def run(ladder):
            rounds = [pairs_round for pairs_round in pairs.rounds(n_levels(ladder), residue) if pairs_round]
            if swaps_whole_states:
                return _swap_states(ladder, rounds, exchange, key, backend)
            for round_pairs, round_key in zip(rounds, backend.split(key, max(len(rounds), 1))):
                lower, upper = [a for a, _ in round_pairs], [b for _, b in round_pairs]
                key_propose, key_accept = backend.split(round_key, 2)
                new_lower, new_upper, log_accept_ratio = exchange.propose(ladder, lower, upper, key_propose, backend)
                xp = namespace(log_accept_ratio)
                uniform = backend.uniform(key_accept, log_accept_ratio.shape, log_accept_ratio.dtype,
                                          device=device_of(log_accept_ratio))
                accepted = xp.log(uniform) < log_accept_ratio            # [I, P]
                ladder = put_levels(ladder, lower, tree_where(accepted, new_lower, take_levels(ladder, lower)))
                ladder = put_levels(ladder, upper, tree_where(accepted, new_upper, take_levels(ladder, upper)))
            return ladder
        return run

    if isinstance(sweep, int):                                           # static: no dispatch
        return for_residue(sweep % pairs.period)(ladder)
    return backend.switch(sweep % pairs.period, [for_residue(r) for r in range(pairs.period)], ladder)


def _permute(tree, permutation):                 # permutation [I, n]: position k takes the state at permutation[i, k]
    def take(a):
        xp = namespace(a, permutation)
        idx = xp.broadcast_to(bcast(permutation, a), tuple(permutation.shape) + tuple(a.shape[2:]))
        return xp.take_along_axis(a, idx, axis=1)

    return tree_map(take, tree)


def _swap_states(ladder, rounds, exchange, key, backend):
    """`replica_exchange`'s rounds for an exchange that swaps whole states (it has ``view`` and ``log_accept``).

    Each round decides acceptance from ``exchange.view(ladder)`` alone, through a per-row permutation of the
    ladder positions; the whole ladder is permuted once at the end. Same draws and result as the generic path.
    """
    view = exchange.view(ladder)
    first_leaf = leaves(view)[0]
    xp = namespace(first_leaf)
    permutation = xp.broadcast_to(xp.arange(first_leaf.shape[1], device=device_of(first_leaf))[None, :],
                                  (first_leaf.shape[0], first_leaf.shape[1]))
    for round_pairs, round_key in zip(rounds, backend.split(key, max(len(rounds), 1))):
        lower, upper = [a for a, _ in round_pairs], [b for _, b in round_pairs]
        _, key_accept = backend.split(round_key, 2)                      # key_propose unused, as in StateSwap.propose
        log_accept_ratio = exchange.log_accept(_permute(view, permutation), lower, upper)
        uniform = backend.uniform(key_accept, log_accept_ratio.shape, log_accept_ratio.dtype,
                                  device=device_of(log_accept_ratio))
        accepted = xp.log(uniform) < log_accept_ratio                    # [I, P]
        perm_lower, perm_upper = take_levels(permutation, lower), take_levels(permutation, upper)
        permutation = put_levels(put_levels(permutation, lower, xp.where(accepted, perm_upper, perm_lower)),
                                 upper, xp.where(accepted, perm_lower, perm_upper))
    return _permute(ladder, permutation) if rounds else ladder


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

    def propose(self, ladder, lower, upper, key, backend):
        """Swap ``record``, ``potential_state``, ``prev_potential_state`` and ``reference`` together.

        See `Exchange.propose`.

        Parameters
        ----------
        ladder : MoveContext
            ``[I, n, ...]``.
        lower, upper : list of int
            Pair positions.
        key : key
            Unused.
        backend : Backend
            Unused.
        """
        swapped = take_levels(ladder, _swap_permutation(n_levels(ladder), lower, upper))   # all fields move together
        return (take_levels(swapped, lower), take_levels(swapped, upper),
                self.log_accept(ladder.potential_state, lower, upper))

    def view(self, ladder):
        """The part of the ladder `log_accept` reads: the potential state ``ladder.potential_state``."""
        return ladder.potential_state

    def log_accept(self, state, lower, upper):
        """``[I, P]`` log-acceptance of swapping positions ``lower[p]`` and ``upper[p]``, from the potential ``state``.

        Lets `replica_exchange` decide swaps without moving the whole state each round.
        """
        swapped = take_levels(state, _swap_permutation(n_levels(state), lower, upper))
        tilt_change = self.potential.log_tilt(swapped) - self.potential.log_tilt(state)   # [I, n], lambda per position
        return take_levels(tilt_change, lower) + take_levels(tilt_change, upper)


def _swap_permutation(n_levels, lower, upper):
    permutation = list(range(n_levels))
    for a, b in zip(lower, upper):
        permutation[a], permutation[b] = b, a
    return permutation
