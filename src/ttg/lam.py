"""Tilt strength lambda (§8): held by its users, evaluated per row from the t they are handed."""

from __future__ import annotations

from .ops import device_of, index_dtype, namespace


def _take_1d(values, indices):   # Array API `take` wants 1-D indices
    xp = namespace(values, indices)
    return xp.reshape(xp.take(values, xp.reshape(indices, (-1,))), indices.shape)


def constant(value):
    """Make a tilt strength (or step size) that is the same everywhere.

    Parameters
    ----------
    value : float
        The value.

    Returns
    -------
    Callable[[array], array]
        ``lam(t) -> value`` with ``t``'s shape and dtype.
    """
    return lambda t: namespace(t).full_like(t, value)


def ladder(values):
    """Make a value by position on axis 1 (RpCN's ``beta_k``, per-rung pCN step sizes), independent of ``t``.

    Parameters
    ----------
    values : array
        ``[N]``, one value per particle position.

    Returns
    -------
    Callable[[array], array]
        ``lam(t [I, N]) -> [I, N]``, ``values`` broadcast over instances and cast to ``t``'s dtype.
    """
    def lam(t):
        xp = namespace(t, values)
        return xp.broadcast_to(xp.astype(values, t.dtype)[None, :], t.shape)
    return lam


def tabulated(knots, values):
    """Make a table of values at given times, looked up by nearest knot.

    Exact at the knots; meant for times that are only ever knots (e.g. the run's own time grid).

    Parameters
    ----------
    knots : array
        ``[S]`` times.
    values : array
        ``[S]`` values at those times.

    Returns
    -------
    Callable[[array], array]
        ``lam(t) -> array`` with ``t``'s shape, in ``t``'s dtype.
    """
    def lam(t):
        xp = namespace(t, knots, values)
        distance = xp.abs(t[..., None] - xp.astype(knots, t.dtype))
        return _take_1d(xp.astype(values, t.dtype), xp.argmin(distance, axis=-1))
    return lam


def piecewise_linear(knots, values, hold=False):   # knots: 1-D increasing
    """Linear interpolation between knots; beyond the end knots, linear extrapolation or held end values.

    Parameters
    ----------
    knots : array
        ``[K]``, increasing, ``K >= 2``.
    values : array
        ``[K]`` values at the knots.
    hold : bool, optional
        Static. False (default) extrapolates the end segments linearly; True holds the end values constant
        beyond the end knots.

    Returns
    -------
    Callable[[array], array]
        ``lam(t) -> array`` with ``t``'s shape, in ``t``'s dtype.
    """
    def lam(t):
        xp = namespace(t, knots, values)
        knots_t, values_t = xp.astype(knots, t.dtype), xp.astype(values, t.dtype)
        if hold:                                     # static
            t = xp.clip(t, min=knots_t[0], max=knots_t[-1])
        # searchsorted(knots_t, t) (side="left") as a count of knots below t: no reshape of a possibly strided t
        upper_index = xp.sum(xp.astype(knots_t < t[..., None], index_dtype(xp, device_of(t))), axis=-1)
        upper_index = xp.clip(upper_index, 1, knots_t.shape[0] - 1)
        knot_lo, knot_hi = _take_1d(knots_t, upper_index - 1), _take_1d(knots_t, upper_index)
        value_lo, value_hi = _take_1d(values_t, upper_index - 1), _take_1d(values_t, upper_index)
        return value_lo + (t - knot_lo) / (knot_hi - knot_lo) * (value_hi - value_lo)
    return lam


def das_lambda(t_grid, growth, alpha_kl):
    """DAS tempering schedule on the run's time grid.

    Step ``k`` (0 = noisiest) gets ``min(1, (1 + growth)^k - 1) / alpha_kl``; the last step gets ``1 / alpha_kl``.

    Parameters
    ----------
    t_grid : array
        ``[S]`` times of the run, noisiest first (e.g. ``schedule.time(u)`` with ``u`` decreasing).
    growth : float
        Tempering growth rate, > 0.
    alpha_kl : float
        KL-regularization strength; the final tilt strength is ``1 / alpha_kl``.

    Returns
    -------
    Callable[[array], array]
        ``tabulated(t_grid, ...)``: exact on the grid's times.
    """
    xp = namespace(t_grid)
    step_index = xp.arange(t_grid.shape[0], dtype=t_grid.dtype, device=device_of(t_grid))   # float: no int/float mixing
    ramp = xp.where(step_index == t_grid.shape[0] - 1, 1.0, xp.clip((1 + growth) ** step_index - 1, max=1.0))
    return tabulated(t_grid, ramp / alpha_kl)


def level_lambda(t_levels, beta, rho):
    """CREPE's per-level tilt strength: ``beta`` at the data end, 0 at the noise end.

    Level ``m`` of ``M`` gets ``beta (1 - m / M)^rho``, interpolated linearly in ``t`` between levels. The end
    values are held: ``beta`` below level 0 (where `crepe_finish` continues denoising) and 0 above level M.

    Parameters
    ----------
    t_levels : array
        ``[M + 1]`` level times, increasing (data end first), e.g. ``schedule.time(u_levels)``.
    beta : float
        Tilt strength at level 0.
    rho : float
        Decay exponent.

    Returns
    -------
    Callable[[array], array]
        ``piecewise_linear(t_levels, ..., hold=True)``.
    """
    xp = namespace(t_levels)
    n_gaps = t_levels.shape[0] - 1
    level = xp.arange(n_gaps + 1, dtype=t_levels.dtype, device=device_of(t_levels))
    return piecewise_linear(t_levels, beta * (1 - level / n_gaps) ** rho, hold=True)
