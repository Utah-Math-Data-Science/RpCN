"""Guidance scales for `Guided` (§8a): kappa = scale(t, grad, Var), one value per row."""

from __future__ import annotations

from .ops import namespace


def event_norm(grad, lead_ndim):    # ||grad|| per row, over the event dims
    """Euclidean norm of each row over the event axes.

    Parameters
    ----------
    grad : array
        ``[L..., *D]``.
    lead_ndim : int
        Number of leading (row) axes, ``len(L)``.

    Returns
    -------
    array
        ``[L...]``.
    """
    xp = namespace(grad)
    return xp.sqrt(xp.sum(grad**2, axis=tuple(range(lead_ndim, grad.ndim))))


def tilt(lam, factor=1.0):       # exact Gaussian tilt by exp(lam <grad, x>): shift = factor * Var * lam(t) * grad
    """Guidance scale for the exact Gaussian tilt by ``exp(lam(t) <grad, x>)``: ``kappa = factor * Var * lam(t)``.

    Parameters
    ----------
    lam : Callable[[array], array]
        Tilt strength as a function of ``t``, e.g. ``constant(2.0)`` or ``das_lambda(...)``.
    factor : float, default 1.0
        Extra multiplier. ``tilt(lam, 0.5)`` with ``Langevin`` (variance ``2 h``) gives the reward drift
        ``h lam grad``.

    Returns
    -------
    Callable[[array, array, array], array]
        ``scale(t [L...], grad [L..., *D], var [L...]) -> [L...]``, for ``Guided``.
    """
    return lambda t, grad, var: factor * var * lam(t)


def normalized(step_length, norm_floor=1e-12):   # RpCN inverse problems: shift = step_length(t) * grad / ||grad||
    """Normalized-gradient guidance scale ``kappa = step_length(t) / (||grad|| + norm_floor)``.

    The shift then has length ``step_length(t)``.

    Intended for inverse problems with RpCN; use the same guided kernel as both target and proposal.

    Parameters
    ----------
    step_length : Callable[[array], array]
        Step length as a function of ``t``.
    norm_floor : float, default 1e-12
        Added to the norm to avoid division by zero.

    Returns
    -------
    Callable[[array, array, array], array]
        ``scale(t [L...], grad [L..., *D], var [L...]) -> [L...]``, for ``Guided``.
    """
    return lambda t, grad, var: step_length(t) / (event_norm(grad, var.ndim) + norm_floor)
