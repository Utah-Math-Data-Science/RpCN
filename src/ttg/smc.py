"""The SMC runner (§13). Serves FK, DAS, RpCN, CREPE's paths and CREPE's initialisation (I8)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .evaluate import evaluate_rows, start_record
from .kernels import Gaussian
from .moves import MoveContext
from .ops import device, lead_shape, namespace, steps_first
from .records import Point, Record
from .resample import masked_resample

Array = Any


@dataclass(frozen=True)
class SMCResult:
    final: Record          # [I, N, ...], carries potential.requires | move.requires | final_needs
    log_w: Array           # [I, N] unnormalised


def smc(start, u_grid, *, bk, evaluate, schedule, target, proposal, potential, policy,
        move=None, move_at=lambda k: True, final_needs=frozenset(), key) -> SMCResult:
    """start: Gaussian (sampled, and the first move's reference), Point, or Record; leading axes [I, N].
    u_grid: [S] (shared) or [I, N, S] (per row; increasing or decreasing -- the runner doesn't care)."""
    x_like = start.mean if isinstance(start, Gaussian) else start.x
    xp = namespace(x_like)
    dtype, dev = x_like.dtype, device(x_like)                       # §4.1: run dtype and device
    t_grid = schedule.time(xp.asarray(u_grid, dtype=dtype, device=dev))   # I5: the only schedule call
    t_steps = steps_first(t_grid, lead_shape(start))                 # -> [S, I, N]
    S = t_steps.shape[0]
    if S < 2:
        raise ValueError("smc needs S >= 2 grid points")
    mv = move.requires if move is not None else frozenset()
    need_mid = target.requires | proposal.requires | potential.requires | mv     # I4
    need_end = potential.requires | mv | frozenset(final_needs)
    ev_mid = lambda p: evaluate_rows(evaluate, p, need_mid, bk)
    ev_end = lambda p: evaluate_rows(evaluate, p, need_end, bk)
    k_start, k_mv0, k_steps = bk.split(key, 3)

    def apply_move(ctx, count, k, ev, last, key):                    # -> (ctx, count); no-op without a move
        if move is None:
            return ctx, count
        fire = lambda c: (move(c[0], ev=ev, potential=potential, last=last, count=c[1], key=key, bk=bk),
                          c[1] + 1)
        return bk.cond(move_at(k), fire, lambda c: c, (ctx, count))

    rec, ref = start_record(start, t_steps[0, ...], need_mid, evaluate, bk, k_start)
    st = potential.init(rec)
    log_w = potential.log_tilt(st)
    count = xp.zeros((), dtype=xp.int32, device=dev)                 # moves fired so far in this run
    if ref is not None:                                              # static: a move at t_N needs a reference
        ctx, count = apply_move(MoveContext(rec, st, None, ref), count, 0, ev_mid, False, k_mv0)
        rec, st = ctx.rec, ctx.st

    def advance(carry, t_next, k, key, ev, last):
        rec, st, log_w, count = carry
        k_res, k_prop, k_mv = bk.split(key, 3)
        (rec, st), log_w = masked_resample(bk, (rec, st), log_w, policy.should_resample(log_w, k), k_res)
        ref = target.at(rec, t_next)                                  # also the pCN reference
        q = ref if proposal is target else proposal.at(rec, t_next)
        x = q.sample(k_prop, bk)
        if proposal is not target:                                    # static
            log_w = log_w + ref.log_ratio(q, x)                       # I9: never two log_probs
        new = ev(Point(x, t_next))                                    # I1
        st_new = potential.update(st, new, last=last)
        log_w = log_w + potential.increment(st, st_new)               # I9: never a C difference
        ctx, count = apply_move(MoveContext(new, st_new, st, ref), count, k, ev, last, k_mv)   # log_w unchanged
        return ctx.rec, ctx.st, log_w, count

    keys = bk.split(k_steps, S - 1)
    body = lambda c, inp: (advance(c, inp[0], inp[1], inp[2], ev_mid, False), None)
    steps = xp.arange(1, S - 1, dtype=xp.int32, device=dev)
    carry, _ = bk.scan(body, (rec, st, log_w, count), (t_steps[1:-1, ...], steps, keys[:-1, ...]))   # S == 2: empty scan
    rec, st, log_w, _ = advance(carry, t_steps[-1, ...], S - 1, keys[-1, ...], ev_end, True)
    return SMCResult(final=rec, log_w=log_w)
