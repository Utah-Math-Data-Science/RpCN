"""Tilt strength lambda (§8): held by its users, evaluated per row from the t they are handed."""

from __future__ import annotations

from .ops import device, namespace


def take1d(v, idx):              # Array API `take` wants 1-D indices
    xp = namespace(v, idx)
    return xp.reshape(xp.take(v, xp.reshape(idx, (-1,))), idx.shape)


def constant(c):
    return lambda t: namespace(t).full_like(t, c)


def ladder(values):
    """Value by position on axis 1 (RpCN's beta_k, pCN step sizes); t: [I, N]."""
    def lam(t):
        xp = namespace(t, values)
        return xp.broadcast_to(xp.astype(values, t.dtype)[None, :], t.shape)
    return lam


def tabulated(knots, values):
    """Exact at the knots; nearest-knot lookup (only knots are ever visited)."""
    def lam(t):
        xp = namespace(t, knots, values)
        dist = xp.abs(t[..., None] - xp.astype(knots, t.dtype))
        return take1d(xp.astype(values, t.dtype), xp.argmin(dist, axis=-1))
    return lam


def piecewise_linear(knots, values):     # knots: 1-D increasing
    def lam(t):
        xp = namespace(t, knots, values)
        kn, va = xp.astype(knots, t.dtype), xp.astype(values, t.dtype)
        j = xp.clip(xp.searchsorted(kn, xp.reshape(t, (-1,))), 1, kn.shape[0] - 1)
        j = xp.reshape(j, t.shape)                   # searchsorted wants a 1-D sorted first argument
        k0, k1, v0, v1 = take1d(kn, j - 1), take1d(kn, j), take1d(va, j - 1), take1d(va, j)
        return v0 + (t - k0) / (k1 - k0) * (v1 - v0)
    return lam


def das_lambda(t_grid, gamma, alpha_kl):
    """t_grid[0] noisiest; lambda_tilde_k = min(1, (1+gamma)^k - 1); last = 1; divided by alpha_kl."""
    xp = namespace(t_grid)
    k = xp.arange(t_grid.shape[0], dtype=t_grid.dtype, device=device(t_grid))   # float: no int/float mixing
    lt = xp.where(k == t_grid.shape[0] - 1, 1.0, xp.clip((1 + gamma) ** k - 1, max=1.0))
    return tabulated(t_grid, lt / alpha_kl)


def level_lambda(t_levels, beta, rho):
    """CREPE: beta at the data end, 0 at the noise end."""
    xp = namespace(t_levels)
    M = t_levels.shape[0] - 1
    k = xp.arange(M + 1, dtype=t_levels.dtype, device=device(t_levels))
    return piecewise_linear(t_levels, beta * (1 - k / M) ** rho)
