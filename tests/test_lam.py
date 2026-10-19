"""L1: held end values of level_lambda; piecewise_linear's default linear extrapolation."""

import numpy as np
from conftest import backends, host
from hypothesis import given
from hypothesis import strategies as st

from ttg import level_lambda, piecewise_linear


@backends
@given(M=st.integers(1, 4), beta=st.floats(0.1, 10.0), rho=st.floats(0.5, 3.0),
       below=st.lists(st.floats(0.0, 1.0), min_size=1, max_size=5),
       above=st.lists(st.floats(0.0, 1.0), min_size=1, max_size=5))
def test_L1_level_lambda_holds_end_values(backend, M, beta, rho, below, above):
    xp = backend.xp
    t_levels = xp.asarray([0.2 + 0.6 * m / M for m in range(M + 1)], dtype=xp.float64)
    lam = level_lambda(t_levels, beta=beta, rho=rho)
    t_lo = xp.asarray([0.2 * b for b in below], dtype=xp.float64)          # t <= t_levels[0]
    t_hi = xp.asarray([0.8 + 0.2 * a for a in above], dtype=xp.float64)    # t >= t_levels[-1]
    assert np.all(host(lam(t_lo)) == beta)
    assert np.all(host(lam(t_hi)) == 0.0)


@backends
@given(t=st.lists(st.floats(-1.0, 2.0), min_size=1, max_size=5))
def test_L1_piecewise_linear_extrapolates_by_default(backend, t):
    xp = backend.xp
    knots = xp.asarray([0.0, 0.5, 1.0], dtype=xp.float64)
    values = xp.asarray([1.0, 2.0, 4.0], dtype=xp.float64)
    tt = np.asarray(t)
    want = np.where(tt <= 0.5, 1.0 + 2.0 * tt, 2.0 + 4.0 * (tt - 0.5))     # end segments continued linearly
    got = host(piecewise_linear(knots, values)(xp.asarray(t, dtype=xp.float64)))
    assert np.allclose(got, want, rtol=1e-12, atol=1e-12)
    held = host(piecewise_linear(knots, values, hold=True)(xp.asarray(t, dtype=xp.float64)))
    assert np.allclose(held, np.interp(tt, [0.0, 0.5, 1.0], [1.0, 2.0, 4.0]), rtol=1e-12, atol=1e-12)
