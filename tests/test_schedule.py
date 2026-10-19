"""S1-S5 (schedules) and V1-V2 (conversions)."""

import numpy as np
from conftest import backends, element_bounds, host, schedules, u_pair, xps_for
from hypothesis import given
from hypothesis import strategies as st

from ttg import eps_from_x0, x0_eps_from_velocity, x0_from_eps
from ttg.ops import bcast


def times(backend, sch, *us):
    xp = backend.xp
    return sch.time(xp.asarray(list(us), dtype=xp.float64))


@backends
@schedules
@given(data=st.data())
def test_S1_time_increasing_and_positive_coefficients(backend, make_schedule, data):
    sch = make_schedule()
    u_lo, u_hi = data.draw(u_pair())
    t = host(times(backend, sch, 0.0, u_lo, u_hi, 1.0))
    assert t[1] < t[2] and t[0] <= t[1] and t[2] <= t[3]
    tt = times(backend, sch, 0.0, u_lo, u_hi, 1.0)
    assert np.all(host(sch.mean(tt)) > 0) and np.all(host(sch.std(tt)) > 0)


@backends
@schedules
@given(data=st.data())
def test_S2_snr_strictly_decreasing(backend, make_schedule, data):
    sch = make_schedule()
    u_lo, u_hi = data.draw(u_pair())
    t = times(backend, sch, u_lo, u_hi)
    snr = host(sch.mean(t) ** 2 / sch.std(t) ** 2)
    assert snr[0] > snr[1]


@backends
@schedules
@given(u=st.floats(0.0, 1.0))
def test_S3_derivatives_match_central_differences(backend, make_schedule, u):
    sch = make_schedule()
    t = times(backend, sch, u)
    h = 1e-6 * t                                           # relative step, keeps t - h inside the range
    for f, df in ((sch.mean, sch.dt_mean), (sch.std, sch.dt_std)):
        fd = host((f(t + h) - f(t - h)) / (2 * h))
        d = host(df(t))
        scale = np.abs(host(f(t))) / host(h)               # roundoff of the difference quotient
        assert np.all(np.abs(fd - d) <= 1e-6 * (1 + np.abs(d)) + 1e-14 * scale)


@backends
@schedules
@given(data=st.data())
def test_S4_prior_shapes_and_variance(backend, make_schedule, data):
    xp, sch = backend.xp, make_schedule()
    lead = tuple(data.draw(st.lists(st.integers(1, 3), min_size=1, max_size=2)))
    event = tuple(data.draw(st.lists(st.integers(1, 3), min_size=1, max_size=3)))
    for dtype in (xp.float32, xp.float64):
        g = sch.prior(lead, event, xp=xp, dtype=dtype)
        assert tuple(g.mean.shape) == lead + event and tuple(g.var.shape) == lead
        assert g.mean.dtype == dtype and g.var.dtype == dtype
        assert np.all(host(g.var) > 0)


@backends
@schedules
@given(data=st.data())
def test_S5_log_snr_and_its_difference(backend, make_schedule, data):
    xp, sch = backend.xp, make_schedule()
    u_lo, u_hi = data.draw(u_pair())
    t = times(backend, sch, u_lo)
    s = times(backend, sch, u_hi)
    ref = 2 * (xp.log(sch.mean(t)) - xp.log(sch.std(t)))
    got = sch.log_snr(t)
    assert np.all(np.abs(host(got - ref)) <= 1e-12 * (1 + np.abs(host(ref))))
    diff = sch.log_snr_diff(s, t)
    ref_d = sch.log_snr(s) - sch.log_snr(t)
    scale = np.abs(host(sch.log_snr(s))) + np.abs(host(sch.log_snr(t)))
    assert np.all(np.abs(host(diff - ref_d)) <= 1e-12 * (1 + scale))


@backends
@schedules
@given(data=st.data())
def test_V1_eps_x0_round_trip(backend, make_schedule, data):
    xp, xps, sch = backend.xp, xps_for(backend), make_schedule()
    B = data.draw(st.integers(1, 4))
    D = data.draw(xps.array_shapes(min_dims=1, max_dims=2, max_side=3))
    x0 = data.draw(xps.arrays(xp.float64, (B, *D), elements=element_bounds()))
    eps = data.draw(xps.arrays(xp.float64, (B, *D), elements=element_bounds()))
    us = data.draw(st.lists(st.floats(0.0, 1.0), min_size=B, max_size=B))
    t = sch.time(xp.asarray(us, dtype=xp.float64))
    m, s = bcast(sch.mean(t), x0), bcast(sch.std(t), x0)
    x = m * x0 + s * eps
    x0_rec = x0_from_eps(x, t, eps, sch)
    eps_rec = eps_from_x0(x, t, x0_rec, sch)
    tol = 1e-12 * host((1 + xp.abs(x) + s * xp.abs(eps)) / m)
    assert np.all(np.abs(host(x0_rec - x0)) <= tol)
    assert np.all(np.abs(host(eps_rec - eps)) <= host(1e-12 * (1 + xp.abs(eps))) + host(m) * tol / host(s) * 4)
    assert np.all(np.abs(host(m * x0_rec + s * eps_rec - x)) <= 1e-12 * (1 + host(xp.abs(x))) * 8)


@backends
@schedules
@given(data=st.data())
def test_V2_velocity_round_trip(backend, make_schedule, data):
    xp, xps, sch = backend.xp, xps_for(backend), make_schedule()
    B = data.draw(st.integers(1, 4))
    D = data.draw(xps.array_shapes(min_dims=1, max_dims=2, max_side=3))
    x0 = data.draw(xps.arrays(xp.float64, (B, *D), elements=element_bounds()))
    eps = data.draw(xps.arrays(xp.float64, (B, *D), elements=element_bounds()))
    us = data.draw(st.lists(st.floats(0.0, 1.0), min_size=B, max_size=B))
    t = sch.time(xp.asarray(us, dtype=xp.float64))
    m, s, dm, ds = (bcast(f(t), x0) for f in (sch.mean, sch.std, sch.dt_mean, sch.dt_std))
    x = m * x0 + s * eps
    v = dm * x0 + ds * eps
    x0_rec, eps_rec = x0_eps_from_velocity(x, t, v, sch)
    den = xp.abs(m * ds - dm * s)
    mag = (xp.abs(ds) + xp.abs(dm)) * (xp.abs(x) + xp.abs(v)) * (1 + m + s) + 1
    tol = host(1e-12 * mag / den)
    assert np.all(np.abs(host(x0_rec - x0)) <= tol)
    assert np.all(np.abs(host(eps_rec - eps)) <= tol)
