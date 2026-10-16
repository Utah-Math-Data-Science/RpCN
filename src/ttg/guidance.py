"""Guidance scales for `Guided` (§8a): kappa = scale(t, g, Var), one value per row."""

from __future__ import annotations

from .ops import namespace


def event_norm(g, lead_ndim):    # ||g|| per row, over the event dims
    xp = namespace(g)
    return xp.sqrt(xp.sum(g**2, axis=tuple(range(lead_ndim, g.ndim))))


def tilt(lam, factor=1.0):       # exact Gaussian tilt by exp(lam <g, x>): shift = factor * Var * lam(t) * g
    return lambda t, g, var: factor * var * lam(t)


def normalized(gamma, eps=1e-12):   # RpCN inverse problems: shift = gamma(t) * g / ||g||
    return lambda t, g, var: gamma(t) / (event_norm(g, var.ndim) + eps)
