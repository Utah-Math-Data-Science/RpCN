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
    """Denoised estimate from predicted noise: x0 = (x - std(t) eps) / mean(t).

    Parameters
    ----------
    x : array
        States, shape [B, *D].
    t : array
        Times, shape [B].
    eps : array
        Predicted noise, shape [B, *D].
    sch : Schedule
        The noise schedule, x = mean(t) x0 + std(t) eps.

    Returns
    -------
    array
        x0, shape [B, *D].
    """
    return (x - bcast(sch.std(t), x) * eps) / bcast(sch.mean(t), x)


def eps_from_x0(x, t, x0, sch):
    """Noise from a denoised estimate: eps = (x - mean(t) x0) / std(t).

    Parameters
    ----------
    x : array
        States, shape [B, *D].
    t : array
        Times, shape [B].
    x0 : array
        Denoised estimate, shape [B, *D].
    sch : Schedule
        The noise schedule, x = mean(t) x0 + std(t) eps.

    Returns
    -------
    array
        eps, shape [B, *D].
    """
    return (x - bcast(sch.mean(t), x) * x0) / bcast(sch.std(t), x)


def score_from_eps(x, t, eps, sch):
    """Score from predicted noise: score = -eps / std(t).

    Parameters
    ----------
    x : array
        States, shape [B, *D].
    t : array
        Times, shape [B].
    eps : array
        Predicted noise, shape [B, *D].
    sch : Schedule
        The noise schedule, x = mean(t) x0 + std(t) eps.

    Returns
    -------
    array
        The score grad_x log p_t(x), shape [B, *D].
    """
    return -eps / bcast(sch.std(t), x)


def eps_from_score(x, t, sc, sch):
    """Noise from a score: eps = -std(t) score.

    Parameters
    ----------
    x : array
        States, shape [B, *D].
    t : array
        Times, shape [B].
    sc : array
        Score grad_x log p_t(x), shape [B, *D].
    sch : Schedule
        The noise schedule, x = mean(t) x0 + std(t) eps.

    Returns
    -------
    array
        eps, shape [B, *D].
    """
    return -bcast(sch.std(t), x) * sc


def x0_eps_from_velocity(x, t, v, sch):          # v ~ dm/dt x0 + ds/dt eps
    """Denoised estimate and noise from a velocity prediction v = dmean/dt x0 + dstd/dt eps.

    Solves the 2x2 system formed by v and x = mean(t) x0 + std(t) eps (e.g. for flow-matching networks).

    Parameters
    ----------
    x : array
        States, shape [B, *D].
    t : array
        Times, shape [B].
    v : array
        Predicted velocity, shape [B, *D].
    sch : Schedule
        The noise schedule, x = mean(t) x0 + std(t) eps.

    Returns
    -------
    x0 : array
        Denoised estimate, shape [B, *D].
    eps : array
        Noise, shape [B, *D].
    """
    m, s, dm, ds = (bcast(f(t), x) for f in (sch.mean, sch.std, sch.dt_mean, sch.dt_std))
    den = m * ds - dm * s                        # = m^2 d(s/m)/dt != 0
    return (ds * x - s * v) / den, (m * v - dm * x) / den


# ---- builders --------------------------------------------------------------------------------

def _keep(needs, f, v):
    return v if f in needs else None


def make_evaluate_jax(network, to_x0_eps, sch, reward):
    """Build an `evaluate` function for a JAX network and reward.

    The returned ``evaluate(x [B, *D], t [B], needs) -> Record`` honours the evaluation contract: it calls the
    network at most once per call, returns exactly the features in `needs` (all others None), and treats rows
    independently. `needs` is a static frozenset of `Feature`. REWARD_GRAD is the gradient of the reward of
    x0_hat with respect to x, through the network (one ``jax.value_and_grad`` over the single forward pass).

    Parameters
    ----------
    network : Callable[[array, array], array]
        ``network(x [B, *D], t [B])`` -> raw output (eps, x0, velocity, ... as `to_x0_eps` expects).
    to_x0_eps : Callable[[array, array, array, Schedule], tuple[array, array]]
        ``to_x0_eps(x, t, out, sch)`` -> (x0, eps), e.g. ``lambda x, t, eps, s: (x0_from_eps(x, t, eps, s), eps)``.
    sch : Schedule
        The noise schedule.
    reward : Callable[[array], array]
        ``reward(x0 [B, *D])`` -> [B]; differentiable if REWARD_GRAD is needed.

    Returns
    -------
    Callable[[array, array, frozenset], Record]
        The evaluate function. It is traceable and may be used under ``jax.jit``.
    """
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
    """Build an `evaluate` function for a PyTorch network and reward.

    As `make_evaluate_jax`: the returned ``evaluate(x [B, *D], t [B], needs) -> Record`` calls the network at
    most once per call, returns exactly the features in `needs` (all others None), and treats rows
    independently. The forward pass runs under ``torch.no_grad()`` unless REWARD_GRAD is needed, in which case
    the reward of x0_hat is differentiated with respect to x through the network. Everything returned is
    detached.

    Parameters
    ----------
    network : Callable[[Tensor, Tensor], Tensor]
        ``network(x [B, *D], t [B])`` -> raw output (eps, x0, velocity, ... as `to_x0_eps` expects).
    to_x0_eps : Callable[[Tensor, Tensor, Tensor, Schedule], tuple[Tensor, Tensor]]
        ``to_x0_eps(x, t, out, sch)`` -> (x0, eps), e.g. ``lambda x, t, eps, s: (x0_from_eps(x, t, eps, s), eps)``.
    sch : Schedule
        The noise schedule.
    reward : Callable[[Tensor], Tensor]
        ``reward(x0 [B, *D])`` -> [B]; differentiable if REWARD_GRAD is needed.

    Returns
    -------
    Callable[[Tensor, Tensor, frozenset], Record]
        The evaluate function.
    """
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
    """Evaluate points with any leading axes by flattening them into one batch axis.

    The only call site of `evaluate` in the library.

    Parameters
    ----------
    evaluate : Callable[[array, array, frozenset], Record]
        The model.
    p : Point
        Points with leading axes L (e.g. [I, N]).
    needs : iterable of Feature
        Features to request.
    bk : Backend
        Unused; kept for a uniform signature.

    Returns
    -------
    Record
        The evaluated points, with leading axes L restored.
    """
    xp = namespace(p.x, p.t)
    lead = tuple(p.t.shape)
    rec = evaluate(flatten_leading(p.x, len(lead)), xp.reshape(p.t, (-1,)), frozenset(needs))
    return tree_map(lambda a: unflatten_leading(a, lead), rec)


def start_record(start, t0, needs, evaluate, bk, key):
    """Turn an `smc` start into an evaluated Record.

    Parameters
    ----------
    start : Gaussian, Point or Record
        A Gaussian is sampled and evaluated, a Point is evaluated, and a Record is projected onto `needs`.
    t0 : array
        Times for a sampled Gaussian start, shape [L...].
    needs : iterable of Feature
        Features the record must carry.
    evaluate : Callable[[array, array, frozenset], Record]
        The model.
    bk : Backend
        Randomness for sampling a Gaussian start.
    key : backend key
        Random key.

    Returns
    -------
    rec : Record
        The start, carrying exactly `needs`.
    ref : Gaussian or None
        The Gaussian start (the reference for a move at the start point), otherwise None.

    Raises
    ------
    PartialRecord
        If a Record start lacks features in `needs` (it is never re-evaluated to top them up).
    """
    if isinstance(start, Gaussian):                           # e.g. schedule.prior(...): sample it, keep it
        return evaluate_rows(evaluate, Point(start.sample(key, bk), t0), needs, bk), start
    if isinstance(start, Point):
        return evaluate_rows(evaluate, start, needs, bk), None
    if not start.provides(needs):
        raise PartialRecord(needs)                            # never top up (would re-evaluate)
    return start.project(needs), None                         # extras dropped: scan's carry must match C2
