"""Guidance scales for `Guided` (§8a): kappa = scale(t, g, Var), one value per row."""

from __future__ import annotations

from .ops import namespace


def event_norm(g, lead_ndim):    # ||g|| per row, over the event dims
    """Euclidean norm of each row over the event axes.

    Parameters
    ----------
    g : array
        ``[L..., *D]``.
    lead_ndim : int
        Number of leading (row) axes, ``len(L)``.

    Returns
    -------
    array
        ``[L...]``.
    """
    xp = namespace(g)
    return xp.sqrt(xp.sum(g**2, axis=tuple(range(lead_ndim, g.ndim))))


def tilt(lam, factor=1.0):       # exact Gaussian tilt by exp(lam <g, x>): shift = factor * Var * lam(t) * g
    """Guidance scale for the exact Gaussian tilt by ``exp(lam(t) <g, x>)``: ``kappa = factor * Var * lam(t)``.

    Parameters
    ----------
    lam : Callable[[array], array]
        Tilt strength as a function of ``t``, e.g. ``constant(2.0)`` or ``das_lambda(...)``.
    factor : float, default 1.0
        Extra multiplier. ``tilt(lam, 0.5)`` with ``Langevin`` (variance ``2 h``) gives the reward drift
        ``h lam g``.

    Returns
    -------
    Callable[[array, array, array], array]
        ``scale(t [L...], g [L..., *D], var [L...]) -> [L...]``, for ``Guided``.
    """
    return lambda t, g, var: factor * var * lam(t)


def normalized(gamma, eps=1e-12):   # RpCN inverse problems: shift = gamma(t) * g / ||g||
    """Normalized-gradient guidance scale ``kappa = gamma(t) / (||g|| + eps)``, so the shift has length ``gamma(t)``.

    Intended for inverse problems with RpCN; use the same guided kernel as both target and proposal.

    Parameters
    ----------
    gamma : Callable[[array], array]
        Step length as a function of ``t``.
    eps : float, default 1e-12
        Added to the norm to avoid division by zero.

    Returns
    -------
    Callable[[array, array, array], array]
        ``scale(t [L...], g [L..., *D], var [L...]) -> [L...]``, for ``Guided``.
    """
    return lambda t, g, var: gamma(t) / (event_norm(g, var.ndim) + eps)
