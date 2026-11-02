"""The SMC runner (§13). Serves FK, DAS, RpCN, CREPE's paths and CREPE's initialisation (I8)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .evaluate import evaluate_rows, start_record
from .kernels import Gaussian
from .moves import MoveContext
from .ops import device_of, leading_shape_of, namespace, steps_first
from .records import Feature, Point, Record
from .resample import masked_resample, systematic

Array = Any


@dataclass(frozen=True)
class SMCResult:
    """Output of `smc`.

    Parameters
    ----------
    final : Record
        Particles at the last grid point, shape [I, N, ...]. Carries the features required by the potential and
        the move, plus `out_needs`.
    log_weights : array
        Unnormalised log-weights of the final particles, shape [I, N]. Normalise over axis 1 (per instance).
    """

    final: Record          # [I, N, ...], carries potential.requires | move.requires | out_needs
    log_weights: Array     # [I, N] unnormalised


def smc(start, u_grid, *, backend, evaluate, schedule, target, proposal, potential, policy, sampler=systematic,
        move=None, move_at=lambda step_index: True, out_needs=frozenset({Feature.X0}), key) -> SMCResult:
    """Run sequential Monte Carlo along a grid of times: FK steering, DAS, RpCN, and CREPE's paths.

    Starting from `start` at the first grid point, each step k = 1, ..., S-1 optionally resamples (per instance,
    as `policy` decides), proposes the next state from `proposal`, reweights by target/proposal when they differ,
    evaluates the new points once, adds the potential's increment to the log-weights, and optionally applies
    `move`. Each point is evaluated exactly once; the network sees batches [B] = [I * N].

    Parameters
    ----------
    start : Gaussian, Point or Record
        Initial particles with leading axes [I, N] (I independent instances, N particles each). A Gaussian
        (e.g. ``schedule.prior((I, N), D, ...)``) is sampled and also serves as the reference of a move at the
        start point. A Point is evaluated. A Record is used as is and must already carry every feature the run
        needs. Its dtype and device set the run's dtype and device.
    u_grid : array
        Schedule coordinates, shape [S] (shared by all rows) or [I, N, S] (per row), S >= 2. Mapped to times
        with ``schedule.time``. May be increasing or decreasing; for sampling, run from 1 (noise) to 0 (data).
    backend : Backend
        Randomness and control flow (scan, cond).
    evaluate : Callable[[array, array, frozenset], Record]
        The model, e.g. from `make_evaluate_torch` or `make_evaluate_jax`.
    schedule : Schedule
        Noise schedule; used only to map `u_grid` to times.
    target : Kernel
        The transition the weights correct towards (e.g. ``PosteriorStep(schedule)``). Its distribution at each
        step is also the reference of the move.
    proposal : Kernel
        The transition particles are drawn from. If it is `target` itself, no correction is added.
    potential : Potential
        The log-tilt whose increments are added to the log-weights (e.g. `FKDiff`; `Zero` for none).
    policy : Never, Every or ESSBelow
        When to resample. Called before each proposal with the step index k.
    sampler : Callable[[Backend, array, key], array], optional
        How to draw ancestors when resampling: `systematic` (default) or `ssp`. See `masked_resample`.
    move : Move, optional
        An MCMC move (e.g. `PCN`) applied after each step where `move_at` is true; it leaves the weights
        unchanged. None for no move.
    move_at : Callable[[int], bool], optional
        Given the step index k (possibly traced), whether to apply `move` after step k. k = 0 is the start point,
        where a move is applied only if `start` is a Gaussian (it needs a reference). Default: always.
    out_needs : iterable of Feature, optional
        Extra features to evaluate at the last grid point. Default ``{Feature.X0}``, the final denoised estimate,
        which costs nothing extra because that point is evaluated anyway.
    key : backend key
        Root random key, e.g. ``backend.key(0)``.

    Returns
    -------
    SMCResult
        Final particles and their log-weights.

    Raises
    ------
    ValueError
        If the grid has fewer than 2 points.
    PartialRecord
        If `start` is a Record lacking features the run needs.
    """
    start_x = start.mean if isinstance(start, Gaussian) else start.x
    xp = namespace(start_x)
    dtype, device = start_x.dtype, device_of(start_x)               # §4.1: run dtype and device
    t_grid = schedule.time(xp.asarray(u_grid, dtype=dtype, device=device))   # I5: the only schedule call
    t_by_step = steps_first(t_grid, leading_shape_of(start))         # -> [S, I, N]
    n_points = t_by_step.shape[0]
    if n_points < 2:
        raise ValueError("smc needs S >= 2 grid points")
    move_needs = move.requires if move is not None else frozenset()
    step_needs = target.requires | proposal.requires | potential.requires | move_needs   # I4
    last_step_needs = potential.requires | move_needs | frozenset(out_needs)
    evaluate_step = lambda point: evaluate_rows(evaluate, point, step_needs, backend)
    evaluate_last_step = lambda point: evaluate_rows(evaluate, point, last_step_needs, backend)
    key_start, key_first_move, key_steps = backend.split(key, 3)

    def apply_move(ctx, n_moves, step_index, evaluate_point, is_last, key):   # -> (ctx, n_moves); no-op without a move
        if move is None:
            return ctx, n_moves

        def run_move(operand):
            ctx, n_moves = operand
            return (move(ctx, evaluate_point=evaluate_point, potential=potential, is_last=is_last,
                         move_count=n_moves, key=key, backend=backend),
                    n_moves + 1)

        return backend.cond(move_at(step_index), run_move, lambda operand: operand, (ctx, n_moves))

    record, reference = start_record(start, t_by_step[0, ...], step_needs, evaluate, backend, key_start)
    state = potential.init(record)
    log_weights = potential.log_tilt(state)
    n_moves = xp.zeros((), dtype=xp.int32, device=device)            # moves fired so far in this run
    if reference is not None:                                        # static: a move at t_N needs a reference
        ctx, n_moves = apply_move(MoveContext(record, state, None, reference), n_moves, 0, evaluate_step, False,
                                  key_first_move)
        record, state = ctx.record, ctx.potential_state

    def advance(carry, t_next, step_index, key, evaluate_point, is_last):
        record, state, log_weights, n_moves = carry
        key_resample, key_propose, key_move = backend.split(key, 3)
        (record, state), log_weights = masked_resample(backend, (record, state), log_weights,
                                                       policy.should_resample(log_weights, step_index), key_resample,
                                                       sampler)
        target_dist = target.at(record, t_next)                       # also the pCN reference
        proposal_dist = target_dist if proposal is target else proposal.at(record, t_next)
        x = proposal_dist.sample(key_propose, backend)
        if proposal is not target:                                    # static
            log_weights = log_weights + target_dist.log_ratio(proposal_dist, x)   # I9: never two log_probs
        new_record = evaluate_point(Point(x, t_next))                 # I1
        new_state = potential.update(state, new_record, is_last=is_last)
        log_weights = log_weights + potential.increment(state, new_state)   # I9: never a C difference
        ctx, n_moves = apply_move(MoveContext(new_record, new_state, state, target_dist), n_moves, step_index,
                                  evaluate_point, is_last, key_move)  # log_weights unchanged
        return ctx.record, ctx.potential_state, log_weights, n_moves

    def scan_step(carry, step_inputs):
        t_next, step_index, step_key = step_inputs
        return advance(carry, t_next, step_index, step_key, evaluate_step, False), None

    step_keys = backend.split(key_steps, n_points - 1)
    step_indices = xp.arange(1, n_points - 1, dtype=xp.int32, device=device)
    carry, _ = backend.scan(scan_step, (record, state, log_weights, n_moves),
                            (t_by_step[1:-1, ...], step_indices, step_keys[:-1, ...]))               # S == 2: empty scan
    record, state, log_weights, _ = advance(carry, t_by_step[-1, ...], n_points - 1, step_keys[-1, ...],
                                            evaluate_last_step, True)
    return SMCResult(final=record, log_weights=log_weights)
