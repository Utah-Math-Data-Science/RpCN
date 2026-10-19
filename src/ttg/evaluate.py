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

def x0_from_eps(x, t, eps, schedule):
    """Denoised estimate from predicted noise: x0 = (x - std(t) eps) / mean(t).

    Parameters
    ----------
    x : array
        States, shape [B, *D].
    t : array
        Times, shape [B].
    eps : array
        Predicted noise, shape [B, *D].
    schedule : Schedule
        The noise schedule, x = mean(t) x0 + std(t) eps.

    Returns
    -------
    array
        x0, shape [B, *D].
    """
    return (x - bcast(schedule.std(t), x) * eps) / bcast(schedule.mean(t), x)


def eps_from_x0(x, t, x0, schedule):
    """Noise from a denoised estimate: eps = (x - mean(t) x0) / std(t).

    Parameters
    ----------
    x : array
        States, shape [B, *D].
    t : array
        Times, shape [B].
    x0 : array
        Denoised estimate, shape [B, *D].
    schedule : Schedule
        The noise schedule, x = mean(t) x0 + std(t) eps.

    Returns
    -------
    array
        eps, shape [B, *D].
    """
    return (x - bcast(schedule.mean(t), x) * x0) / bcast(schedule.std(t), x)


def score_from_eps(x, t, eps, schedule):
    """Score from predicted noise: score = -eps / std(t).

    Parameters
    ----------
    x : array
        States, shape [B, *D].
    t : array
        Times, shape [B].
    eps : array
        Predicted noise, shape [B, *D].
    schedule : Schedule
        The noise schedule, x = mean(t) x0 + std(t) eps.

    Returns
    -------
    array
        The score grad_x log p_t(x), shape [B, *D].
    """
    return -eps / bcast(schedule.std(t), x)


def eps_from_score(x, t, score, schedule):
    """Noise from a score: eps = -std(t) score.

    Parameters
    ----------
    x : array
        States, shape [B, *D].
    t : array
        Times, shape [B].
    score : array
        Score grad_x log p_t(x), shape [B, *D].
    schedule : Schedule
        The noise schedule, x = mean(t) x0 + std(t) eps.

    Returns
    -------
    array
        eps, shape [B, *D].
    """
    return -bcast(schedule.std(t), x) * score


def x0_eps_from_velocity(x, t, velocity, schedule):   # velocity ~ dm/dt x0 + ds/dt eps
    """Denoised estimate and noise from a predicted velocity, velocity = dmean/dt x0 + dstd/dt eps.

    Solves the 2x2 system formed by velocity and x = mean(t) x0 + std(t) eps (e.g. for flow-matching networks).

    Parameters
    ----------
    x : array
        States, shape [B, *D].
    t : array
        Times, shape [B].
    velocity : array
        Predicted velocity, shape [B, *D].
    schedule : Schedule
        The noise schedule, x = mean(t) x0 + std(t) eps.

    Returns
    -------
    x0 : array
        Denoised estimate, shape [B, *D].
    eps : array
        Noise, shape [B, *D].
    """
    m, s, dm, ds = (bcast(f(t), x) for f in (schedule.mean, schedule.std, schedule.dt_mean, schedule.dt_std))
    denom = m * ds - dm * s                      # = m^2 d(s/m)/dt != 0
    return (ds * x - s * velocity) / denom, (m * velocity - dm * x) / denom


# ---- builders --------------------------------------------------------------------------------

def on_x0(reward_fn):            # reward of the denoiser's estimate only
    """Adapt a reward of the denoised estimate alone to the builders' ``reward(x0, x, t)`` signature.

    Parameters
    ----------
    reward_fn : Callable[[array], array]
        ``reward_fn(x0 [B, *D])`` -> [B].

    Returns
    -------
    Callable[[array, array, array], array]
        ``lambda x0, x, t: reward_fn(x0)``.
    """
    return lambda x0, x, t: reward_fn(x0)


def _keep(needs, feature, value):
    return value if feature in needs else None


def make_evaluate_jax(network, to_x0_eps, schedule, reward):
    """Build an `evaluate` function for a JAX network and reward.

    The returned ``evaluate(x [B, *D], t [B], needs) -> Record`` honours the evaluation contract: it calls the
    network at most once per call, returns exactly the features in `needs` (all others None), and treats rows
    independently. `needs` is a static frozenset of `Feature`. REWARD_GRAD is the gradient of the reward with
    respect to x, through both x0_hat (the network) and x (one ``jax.value_and_grad`` over the single forward pass).

    Parameters
    ----------
    network : Callable[[array, array], array]
        ``network(x [B, *D], t [B])`` -> raw output (eps, x0, velocity, ... as `to_x0_eps` expects).
    to_x0_eps : Callable[[array, array, array, Schedule], tuple[array, array]]
        ``to_x0_eps(x, t, out, schedule)`` -> (x0, eps), e.g. ``lambda x, t, eps, s: (x0_from_eps(x, t, eps, s), eps)``.
    schedule : Schedule
        The noise schedule.
    reward : Callable[[array, array, array], array]
        ``reward(x0 [B, *D], x [B, *D], t [B])`` -> [B]; differentiable if REWARD_GRAD is needed. Wrap a reward
        of the estimate alone with `on_x0`. It may call a separate reward model; C1 counts only `network`.

    Returns
    -------
    Callable[[array, array, frozenset], Record]
        The evaluate function. It is traceable and may be used under ``jax.jit``.
    """
    import jax
    import jax.numpy as jnp

    def evaluate(x, t, needs):
        want_grad = Feature.REWARD_GRAD in needs
        want_reward = want_grad or Feature.REWARD in needs

        def forward(x):
            x0, eps = to_x0_eps(x, t, network(x, t), schedule)  # the single forward pass
            reward_value = reward(x0, x, t) if want_reward else None
            return (jnp.sum(reward_value) if want_grad else 0.0), (x0, eps, reward_value)

        if want_grad:
            (_, (x0, eps, reward_value)), reward_grad = jax.value_and_grad(forward, has_aux=True)(x)
        else:
            _, (x0, eps, reward_value) = forward(x)
            reward_grad = None
        score = score_from_eps(x, t, eps, schedule) if Feature.SCORE in needs else None
        return Record(x, t, x0=_keep(needs, Feature.X0, x0), eps=_keep(needs, Feature.EPS, eps),
                      score=score, reward=_keep(needs, Feature.REWARD, reward_value),
                      reward_grad=_keep(needs, Feature.REWARD_GRAD, reward_grad))

    return evaluate


def make_evaluate_torch(network, to_x0_eps, schedule, reward):
    """Build an `evaluate` function for a PyTorch network and reward.

    As `make_evaluate_jax`: the returned ``evaluate(x [B, *D], t [B], needs) -> Record`` calls the network at
    most once per call, returns exactly the features in `needs` (all others None), and treats rows
    independently. The forward pass runs under ``torch.no_grad()`` unless REWARD_GRAD is needed, in which case
    the reward is differentiated with respect to x, through x0_hat (the network) and x itself. Everything returned is
    detached.

    Parameters
    ----------
    network : Callable[[Tensor, Tensor], Tensor]
        ``network(x [B, *D], t [B])`` -> raw output (eps, x0, velocity, ... as `to_x0_eps` expects).
    to_x0_eps : Callable[[Tensor, Tensor, Tensor, Schedule], tuple[Tensor, Tensor]]
        ``to_x0_eps(x, t, out, schedule)`` -> (x0, eps), e.g. ``lambda x, t, eps, s: (x0_from_eps(x, t, eps, s), eps)``.
    schedule : Schedule
        The noise schedule.
    reward : Callable[[Tensor, Tensor, Tensor], Tensor]
        ``reward(x0 [B, *D], x [B, *D], t [B])`` -> [B]; differentiable if REWARD_GRAD is needed. Wrap a reward
        of the estimate alone with `on_x0`. It may call a separate reward model; C1 counts only `network`.

    Returns
    -------
    Callable[[Tensor, Tensor, frozenset], Record]
        The evaluate function.
    """
    import torch

    def evaluate(x, t, needs):
        want_grad = Feature.REWARD_GRAD in needs
        want_reward = want_grad or Feature.REWARD in needs
        with torch.enable_grad() if want_grad else torch.no_grad():
            x_tracked = x.detach().requires_grad_(want_grad)
            x0, eps = to_x0_eps(x_tracked, t, network(x_tracked, t), schedule)   # the single forward pass
            reward_value = reward(x0, x_tracked, t) if want_reward else None
            reward_grad = torch.autograd.grad(reward_value.sum(), x_tracked)[0] if want_grad else None

        def keep(feature, value):
            return None if (feature not in needs or value is None) else value.detach()

        score = score_from_eps(x, t, eps.detach(), schedule) if Feature.SCORE in needs else None
        return Record(x, t, x0=keep(Feature.X0, x0), eps=keep(Feature.EPS, eps), score=score,
                      reward=keep(Feature.REWARD, reward_value), reward_grad=keep(Feature.REWARD_GRAD, reward_grad))

    return evaluate


# ---- runner-side helpers ---------------------------------------------------------------------

def evaluate_rows(evaluate, point: Point, needs, backend):
    """Evaluate points with any leading axes by flattening them into one batch axis.

    The only call site of `evaluate` in the library.

    Parameters
    ----------
    evaluate : Callable[[array, array, frozenset], Record]
        The model.
    point : Point
        Points with leading axes L (e.g. [I, N]).
    needs : iterable of Feature
        Features to request.
    backend : Backend
        Unused; kept for a uniform signature.

    Returns
    -------
    Record
        The evaluated points, with leading axes L restored.
    """
    xp = namespace(point.x, point.t)
    leading_shape = tuple(point.t.shape)
    record = evaluate(flatten_leading(point.x, len(leading_shape)), xp.reshape(point.t, (-1,)), frozenset(needs))
    return tree_map(lambda a: unflatten_leading(a, leading_shape), record)


def start_record(start, t_start, needs, evaluate, backend, key):
    """Turn an `smc` start into an evaluated Record.

    Parameters
    ----------
    start : Gaussian, Point or Record
        A Gaussian is sampled and evaluated, a Point is evaluated, and a Record is projected onto `needs`.
    t_start : array
        Times for a sampled Gaussian start, shape [L...].
    needs : iterable of Feature
        Features the record must carry.
    evaluate : Callable[[array, array, frozenset], Record]
        The model.
    backend : Backend
        Randomness for sampling a Gaussian start.
    key : backend key
        Random key.

    Returns
    -------
    record : Record
        The start, carrying exactly `needs`.
    reference : Gaussian or None
        The Gaussian start (the reference for a move at the start point), otherwise None.

    Raises
    ------
    PartialRecord
        If a Record start lacks features in `needs` (it is never re-evaluated to top them up).
    """
    if isinstance(start, Gaussian):                           # e.g. schedule.prior(...): sample it, keep it
        return evaluate_rows(evaluate, Point(start.sample(key, backend), t_start), needs, backend), start
    if isinstance(start, Point):
        return evaluate_rows(evaluate, start, needs, backend), None
    if not start.provides(needs):
        raise PartialRecord(needs)                            # never top up (would re-evaluate)
    return start.project(needs), None                         # extras dropped: scan's carry must match C2
