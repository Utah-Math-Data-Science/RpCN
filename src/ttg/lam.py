"""Tilt strength lambda (§8): held by its users, evaluated per row from the t they are handed."""

from __future__ import annotations

from .ops import device, namespace


def take1d(v, idx):              # Array API `take` wants 1-D indices
    xp = namespace(v, idx)
    return xp.reshape(xp.take(v, xp.reshape(idx, (-1,))), idx.shape)


def constant(c):
    """Make a tilt strength (or step size) that is the same everywhere.

    Parameters
    ----------
    c : float
        The value.

    Returns
    -------
    Callable[[array], array]
        ``lam(t) -> c`` with ``t``'s shape and dtype.
    """
    return lambda t: namespace(t).full_like(t, c)


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
        dist = xp.abs(t[..., None] - xp.astype(knots, t.dtype))
        return take1d(xp.astype(values, t.dtype), xp.argmin(dist, axis=-1))
    return lam


def piecewise_linear(knots, values):     # knots: 1-D increasing
    """Linear interpolation between knots, extrapolated linearly beyond the end segments.

    Parameters
    ----------
    knots : array
        ``[K]``, increasing, ``K >= 2``.
    values : array
        ``[K]`` values at the knots.

    Returns
    -------
    Callable[[array], array]
        ``lam(t) -> array`` with ``t``'s shape, in ``t``'s dtype.
    """
    def lam(t):
        xp = namespace(t, knots, values)
        kn, va = xp.astype(knots, t.dtype), xp.astype(values, t.dtype)
        j = xp.clip(xp.searchsorted(kn, xp.reshape(t, (-1,))), 1, kn.shape[0] - 1)
        j = xp.reshape(j, t.shape)                   # searchsorted wants a 1-D sorted first argument
        k0, k1, v0, v1 = take1d(kn, j - 1), take1d(kn, j), take1d(va, j - 1), take1d(va, j)
        return v0 + (t - k0) / (k1 - k0) * (v1 - v0)
    return lam


def das_lambda(t_grid, gamma, alpha_kl):
    """DAS tempering schedule on the run's time grid.

    Step ``k`` (0 = noisiest) gets ``min(1, (1 + gamma)^k - 1) / alpha_kl``; the last step gets ``1 / alpha_kl``.

    Parameters
    ----------
    t_grid : array
        ``[S]`` times of the run, noisiest first (e.g. ``sch.time(u)`` with ``u`` decreasing).
    gamma : float
        Tempering growth rate, > 0.
    alpha_kl : float
        KL-regularization strength; the final tilt strength is ``1 / alpha_kl``.

    Returns
    -------
    Callable[[array], array]
        ``tabulated(t_grid, ...)``: exact on the grid's times.
    """
    xp = namespace(t_grid)
    k = xp.arange(t_grid.shape[0], dtype=t_grid.dtype, device=device(t_grid))   # float: no int/float mixing
    lt = xp.where(k == t_grid.shape[0] - 1, 1.0, xp.clip((1 + gamma) ** k - 1, max=1.0))
    return tabulated(t_grid, lt / alpha_kl)


def level_lambda(t_levels, beta, rho):
    """CREPE's per-level tilt strength: ``beta`` at the data end, 0 at the noise end.

    Level ``m`` of ``M`` gets ``beta (1 - m / M)^rho``, interpolated linearly in ``t`` between levels.

    Parameters
    ----------
    t_levels : array
        ``[M + 1]`` level times, increasing (data end first), e.g. ``sch.time(u_levels)``.
    beta : float
        Tilt strength at level 0.
    rho : float
        Decay exponent.

    Returns
    -------
    Callable[[array], array]
        ``piecewise_linear(t_levels, ...)``.
    """
    xp = namespace(t_levels)
    M = t_levels.shape[0] - 1
    k = xp.arange(M + 1, dtype=t_levels.dtype, device=device(t_levels))
    return piecewise_linear(t_levels, beta * (1 - k / M) ** rho)
