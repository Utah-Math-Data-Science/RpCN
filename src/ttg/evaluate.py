"""The evaluation contract, conversions, builders and runner-side helpers (§6)."""

from __future__ import annotations

from typing import Any, Callable

from .kernels import Gaussian
from .ops import bcast, flatten_leading, namespace, tree_map, unflatten_leading
from .records import Feature, PartialRecord, Point, Record

Array = Any

Evaluate = Callable[[Array, Array, frozenset], Record]
# evaluate(x [B, *D], t [B], needs) -> Record with leading axis [B]
#  (C1) calls the network AT MOST ONCE per call (CFG: one batched forward over [cond, uncond]);
#  (C2) returns EXACTLY the requested features (all others None);
#  (C3) rows are independent: row b's outputs depend only on (x[b], t[b]).
# `needs` is static (hashable) and may drive Python-level branching inside evaluate.


# ---- conversions -----------------------------------------------------------------------------

def x0_from_eps(x, t, eps, sch):
    return (x - bcast(sch.std(t), x) * eps) / bcast(sch.mean(t), x)


def eps_from_x0(x, t, x0, sch):
    return (x - bcast(sch.mean(t), x) * x0) / bcast(sch.std(t), x)


def score_from_eps(x, t, eps, sch):
    return -eps / bcast(sch.std(t), x)


def eps_from_score(x, t, sc, sch):
    return -bcast(sch.std(t), x) * sc


def x0_eps_from_velocity(x, t, v, sch):          # v ~ dm/dt x0 + ds/dt eps
    m, s, dm, ds = (bcast(f(t), x) for f in (sch.mean, sch.std, sch.dt_mean, sch.dt_std))
    den = m * ds - dm * s                        # = m^2 d(s/m)/dt != 0
    return (ds * x - s * v) / den, (m * v - dm * x) / den


# ---- builders --------------------------------------------------------------------------------

def _keep(needs, f, v):
    return v if f in needs else None


def make_evaluate_jax(network, to_x0_eps, sch, reward):
    """network(x, t) -> raw output; to_x0_eps(x, t, out, sch) -> (x0, eps); reward(x0) [B, *D] -> [B]."""
    import jax
    import jax.numpy as jnp

    def evaluate(x, t, needs):
        grad = Feature.REWARD_GRAD in needs
        want_r = grad or Feature.REWARD in needs

        def fwd(x):
            x0, eps = to_x0_eps(x, t, network(x, t), sch)       # the single forward pass
            r = reward(x0) if want_r else None
            return (jnp.sum(r) if grad else 0.0), (x0, eps, r)

        if grad:
            (_, (x0, eps, r)), g = jax.value_and_grad(fwd, has_aux=True)(x)
        else:
            _, (x0, eps, r) = fwd(x)
            g = None
        score = score_from_eps(x, t, eps, sch) if Feature.SCORE in needs else None
        return Record(x, t, x0=_keep(needs, Feature.X0, x0), eps=_keep(needs, Feature.EPS, eps),
                      score=score, reward=_keep(needs, Feature.REWARD, r),
                      reward_grad=_keep(needs, Feature.REWARD_GRAD, g))

    return evaluate


def make_evaluate_torch(network, to_x0_eps, sch, reward):
    """As make_evaluate_jax; everything returned is detached."""
    import torch

    def evaluate(x, t, needs):
        grad = Feature.REWARD_GRAD in needs
        want_r = grad or Feature.REWARD in needs
        with torch.enable_grad() if grad else torch.no_grad():
            xg = x.detach().requires_grad_(grad)
            x0, eps = to_x0_eps(xg, t, network(xg, t), sch)      # the single forward pass
            r = reward(x0) if want_r else None
            g = torch.autograd.grad(r.sum(), xg)[0] if grad else None

        def keep(f, v):
            return None if (f not in needs or v is None) else v.detach()

        score = score_from_eps(x, t, eps.detach(), sch) if Feature.SCORE in needs else None
        return Record(x, t, x0=keep(Feature.X0, x0), eps=keep(Feature.EPS, eps), score=score,
                      reward=keep(Feature.REWARD, r), reward_grad=keep(Feature.REWARD_GRAD, g))

    return evaluate


# ---- runner-side helpers ---------------------------------------------------------------------

def evaluate_rows(evaluate, p: Point, needs, bk):
    """[I, N] <-> [B]; the only call site of evaluate."""
    xp = namespace(p.x, p.t)
    lead = tuple(p.t.shape)
    rec = evaluate(flatten_leading(p.x, len(lead)), xp.reshape(p.t, (-1,)), frozenset(needs))
    return tree_map(lambda a: unflatten_leading(a, lead), rec)


def start_record(start, t0, needs, evaluate, bk, key):
    """-> (Record, reference Gaussian or None)."""
    if isinstance(start, Gaussian):                           # e.g. schedule.prior(...): sample it, keep it
        return evaluate_rows(evaluate, Point(start.sample(key, bk), t0), needs, bk), start
    if isinstance(start, Point):
        return evaluate_rows(evaluate, start, needs, bk), None
    if not start.provides(needs):
        raise PartialRecord(needs)                            # never top up (would re-evaluate)
    return start.project(needs), None                         # extras dropped: scan's carry must match C2
