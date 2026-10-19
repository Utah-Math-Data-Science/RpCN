"""K1-K13 (Gaussians and kernels), N2-N3 (precision of log_ratio and log_snr_diff)."""

import math

import numpy as np
import pytest
from conftest import backends, element_bounds, host, schedules, u_pair, xps_for
from hypothesis import assume, given
from hypothesis import strategies as st

from ttg import (EDM, DDIMStep, Direction, EulerMaruyama, ForwardSDE, Gaussian, Guided, Langevin, Noising, Point,
                 PosteriorStep, Record, ReverseSDE, VPLinear, constant, diffusion_sq_coeff, drift_coeff, normalized,
                 tilt, transition)
from ttg.ops import bcast
from ttg.testing import UNIT_ROUNDOFF32, SMALLEST_NORMAL32, oracle


def close(a, b, rtol, atol=0.0):
    a, b = host(a), host(b)
    return bool(np.all(np.abs(a - b) <= atol + rtol * np.abs(b)))


@st.composite
def problem(draw, backend, n_arrays=3):
    """(B, D, [arrays of shape (B, *D)]) in float64."""
    xp, xps = backend.xp, xps_for(backend)
    B = draw(st.integers(1, 4))
    D = draw(xps.array_shapes(min_dims=1, max_dims=2, max_side=3))
    arrs = [draw(xps.arrays(xp.float64, (B, *D), elements=element_bounds())) for _ in range(n_arrays)]
    return B, tuple(D), arrs


def full(backend, B, v):
    return backend.xp.full((B,), v, dtype=backend.xp.float64)


def positive(backend, B, data, lo=0.01, hi=4.0):
    return data.draw(xps_for(backend).arrays(backend.xp.float64, (B,), elements=element_bounds(lo, hi)))


@backends
@given(data=st.data())
def test_K1_log_prob_formula_and_mode(backend, data):
    xp = backend.xp
    B, D, (mu, x, _) = data.draw(problem(backend))
    var = positive(backend, B, data)
    g = Gaussian(mu, var)
    d = math.prod(D)
    ev = tuple(range(1, x.ndim))
    explicit = -xp.sum((x - mu) ** 2, axis=ev) / (2 * var) - d / 2 * xp.log(2 * math.pi * var)
    assert close(g.log_prob(x), explicit, 1e-12, 1e-12)
    assert np.all(host(g.log_prob(mu)) >= host(g.log_prob(x)))


@backends
@schedules
@given(data=st.data())
def test_K2_chapman_kolmogorov_for_noising(backend, make_schedule, data):
    xp, sch = backend.xp, make_schedule()
    B, D, (x,) = data.draw(problem(backend, 1))
    a = data.draw(st.floats(0.0, 0.98))
    b = data.draw(st.floats(a + 0.01, 0.99))
    c = data.draw(st.floats(b + 0.01, 1.0))
    tu, tt, ts = (full(backend, B, float(sch.time(xp.asarray(v, dtype=xp.float64)))) for v in (a, b, c))
    k = Noising(sch)
    one = k.at(Point(x, tu), tt)                       # u -> t
    m_ut, v_ut = transition(sch, tu, tt)
    m_ts, v_ts = transition(sch, tt, ts)
    m_us, v_us = transition(sch, tu, ts)
    assert close(m_ts * m_ut, m_us, 1e-12)
    assert close(m_ts**2 * v_ut + v_ts, v_us, 1e-10)
    two = k.at(Record(one.mean, tt), ts)               # composing the means
    assert close(two.mean, k.at(Point(x, tu), ts).mean, 1e-12, 1e-12)


@backends
@schedules
@given(data=st.data())
def test_K3_posterior_is_bayes_inverse_of_noising(backend, make_schedule, data):
    xp, sch = backend.xp, make_schedule()
    B, D, (x0, x_t, x_s) = data.draw(problem(backend))
    u_t, u_s = data.draw(u_pair())
    t = xp.full((B,), sch.time(u_t), dtype=xp.float64)
    s = xp.full((B,), sch.time(u_s), dtype=xp.float64)

    def marg(x, tt):
        return Gaussian(bcast(sch.mean(tt), x0) * x0, sch.std(tt) ** 2).log_prob(x)

    lhs = Noising(sch).at(Point(x_t, t), s).log_prob(x_s) + marg(x_t, t) - marg(x_s, s)
    rhs = PosteriorStep(sch).at(Record(x_s, s, x0=x0), t).log_prob(x_t)
    assert bool(xp.all(xp.abs(lhs - rhs) <= 1e-8 * (1 + xp.abs(rhs))))


@backends
@schedules
@given(data=st.data())
def test_K4_ddim_eta_one_is_posterior_step(backend, make_schedule, data):
    xp, sch = backend.xp, make_schedule()
    B, D, (x, x0, _) = data.draw(problem(backend))
    u_t, u_s = data.draw(u_pair())
    t, s = full(backend, B, sch.time(u_t)), full(backend, B, sch.time(u_s))
    eps = (x - bcast(sch.mean(s), x) * x0) / bcast(sch.std(s), x)
    rec = Record(x, s, x0=x0, eps=eps)
    p, d = PosteriorStep(sch).at(rec, t), DDIMStep(sch, 1.0).at(rec, t)
    scale = host(xp.abs(x) + xp.abs(x0) + 1)
    assert np.all(np.abs(host(p.mean - d.mean)) <= 1e-10 * scale)
    assert close(d.var, p.var, 1e-12)


@backends
@given(data=st.data())
def test_K5_guided_with_zero_scale_is_base(backend, data):
    xp = backend.xp
    sch = VPLinear()
    B, D, (x, x0, g) = data.draw(problem(backend))
    u_t, u_s = data.draw(u_pair())
    t, s = full(backend, B, sch.time(u_t)), full(backend, B, sch.time(u_s))
    rec = Record(x, s, x0=x0, reward_grad=g)
    base = PosteriorStep(sch)
    a, b = Guided(base, lambda t, g, var: xp.zeros_like(var)).at(rec, t), base.at(rec, t)
    assert bool(xp.all(a.mean == b.mean)) and bool(xp.all(a.var == b.var))


def proposal_kernels(sch):
    """(kernel, source u, destination u) builders: every kernel the library offers, used in its direction."""
    em_b = EulerMaruyama(ReverseSDE(sch), Direction.BACKWARD)
    em_f = EulerMaruyama(ForwardSDE(sch), Direction.FORWARD)
    lam = constant(1.5)
    return [("noising", Noising(sch), "up"), ("posterior", PosteriorStep(sch), "down"),
            ("ddim0.3", DDIMStep(sch, 0.3), "down"), ("ddim1", DDIMStep(sch, 1.0), "down"),
            ("em_back", em_b, "down"), ("em_fwd", em_f, "up"),
            ("em_back_eta0.5", EulerMaruyama(ReverseSDE(sch, 0.5), Direction.BACKWARD), "down"),
            ("langevin", Langevin(lambda t: 0.01 * sch.std(t) ** 2), "stay"),
            ("guided", Guided(em_b, tilt(lam)), "down"), ("guided_norm", Guided(PosteriorStep(sch), normalized(lam)), "down")]


def full_record(x, t, sch, g):
    m, s = bcast(sch.mean(t), x), bcast(sch.std(t), x)
    x0 = x / m
    eps = (x - m * x0) / s
    return Record(x, t, x0=x0, eps=eps, score=-eps / s, reward_grad=g)


@backends
@schedules
@given(data=st.data())
def test_K6_proposal_variances_positive(backend, make_schedule, data):
    sch = make_schedule()
    B, D, (x, g, _) = data.draw(problem(backend))
    u_lo, u_hi = data.draw(u_pair())
    lo, hi = full(backend, B, sch.time(u_lo)), full(backend, B, sch.time(u_hi))
    for name, k, way in proposal_kernels(sch):
        src, dst = {"up": (lo, hi), "down": (hi, lo), "stay": (lo, lo)}[way]
        var = k.at(full_record(x, src, sch, g), dst).var
        assert np.all(host(var) > 0), name


@backends
@given(data=st.data(), gamma=st.floats(0.1, 5.0))
def test_K7_normalized_shift_norm(backend, data, gamma):
    xp = backend.xp
    sch = VPLinear()
    B, D, (x, x0, g) = data.draw(problem(backend))
    gn = host(xp.sqrt(xp.sum(g**2, axis=tuple(range(1, g.ndim)))))
    u_t, u_s = data.draw(u_pair())
    t, s = full(backend, B, sch.time(u_t)), full(backend, B, sch.time(u_s))
    rec = Record(x, s, x0=x0, reward_grad=g)
    base = PosteriorStep(sch)
    shift = Guided(base, normalized(constant(gamma))).at(rec, t).mean - base.at(rec, t).mean
    norm = host(xp.sqrt(xp.sum(shift**2, axis=tuple(range(1, g.ndim)))))
    want = gamma * gn / (gn + 1e-12)
    ok = gn > 0
    assert np.all(np.abs(norm - want)[ok] <= 1e-9 * (1 + want[ok]))


@backends
@schedules
@given(data=st.data(), eta=st.floats(0.1, 1.0))
def test_K8_euler_maruyama_direction(backend, make_schedule, data, eta):
    sch = make_schedule()
    B, D, (x, g, _) = data.draw(problem(backend))
    u_lo, u_hi = data.draw(u_pair())
    lo, hi = full(backend, B, sch.time(u_lo)), full(backend, B, sch.time(u_hi))
    for sde, direction, (src, dst) in ((ReverseSDE(sch, eta), Direction.BACKWARD, (hi, lo)),
                                       (ForwardSDE(sch), Direction.FORWARD, (lo, hi))):
        k = EulerMaruyama(sde, direction)
        rec = full_record(x, src, sch, g)
        assert np.all(host(k.at(rec, dst).var) > 0)
        assert np.all(host(k.at(full_record(x, dst, sch, g), src).var) <= 0)   # misused: wrong way


@backends
@given(u=st.floats(0.0, 1.0))
def test_K9_diffusion_sq_coeff_closed_forms(backend, u):
    xp = backend.xp
    vp, edm = VPLinear(), EDM()
    t = vp.time(xp.asarray([u], dtype=xp.float64))
    assert close(diffusion_sq_coeff(vp, t), vp._beta(t), 1e-10)
    t = edm.time(xp.asarray([u], dtype=xp.float64))
    assert close(diffusion_sq_coeff(edm, t), 2 * t, 1e-12)


@backends
@schedules
@given(data=st.data(), lam=st.floats(-3.0, 3.0))
def test_K10_tilt_is_exact_gaussian_tilt(backend, make_schedule, data, lam):
    xp, sch = backend.xp, make_schedule()
    B, D, (x, g, xa) = data.draw(problem(backend))
    xb = data.draw(xps_for(backend).arrays(xp.float64, (B, *D), elements=element_bounds()))
    u_t, u_s = data.draw(u_pair())
    t, s = full(backend, B, sch.time(u_t)), full(backend, B, sch.time(u_s))
    rec = full_record(x, s, sch, g)
    base = EulerMaruyama(ReverseSDE(sch), Direction.BACKWARD)
    guided = Guided(base, tilt(constant(lam)))
    ev = tuple(range(1, x.ndim))

    def c(y):
        return guided.at(rec, t).log_prob(y) - base.at(rec, t).log_prob(y) - lam * xp.sum(g * y, axis=ev)

    ca, cb = c(xa), c(xb)
    mag = host(xp.abs(base.at(rec, t).log_prob(xa)) + xp.abs(guided.at(rec, t).log_prob(xb)) + 1)
    assert np.all(np.abs(host(ca - cb)) <= 1e-9 * mag)


@backends
@schedules
@given(data=st.data(), eta=st.floats(0.1, 2.0))
def test_K11_sdes(backend, make_schedule, data, eta):
    xp, sch = backend.xp, make_schedule()
    assert ForwardSDE(sch).requires == frozenset()
    B, D, (x, g, _) = data.draw(problem(backend))
    t = full(backend, B, sch.time(data.draw(st.floats(0.0, 1.0))))
    rec = full_record(x, t, sch, g)
    want = bcast(drift_coeff(sch, t), x) * x - bcast(diffusion_sq_coeff(sch, t), x) * rec.score
    scale = host(xp.abs(bcast(drift_coeff(sch, t), x) * x)
                 + xp.abs(bcast(diffusion_sq_coeff(sch, t), x) * rec.score) + 1)
    assert np.all(np.abs(host(ReverseSDE(sch, 1.0).drift(rec) - want)) <= 1e-12 * scale)
    for sde in (ForwardSDE(sch), ReverseSDE(sch, 1.0), ReverseSDE(sch, eta)):
        assert np.all(host(sde.diffusion_sq(t)) > 0)


@backends
@given(data=st.data())
def test_K12_log_ratio(backend, data):
    xp = backend.xp
    B, D, (ma, mb, x) = data.draw(problem(backend))
    va, vb = positive(backend, B, data), positive(backend, B, data)
    a, b = Gaussian(ma, va), Gaussian(mb, vb)
    ref = a.log_prob(x) - b.log_prob(x)
    mag = host(xp.abs(a.log_prob(x)) + xp.abs(b.log_prob(x)) + 1)
    assert np.all(np.abs(host(a.log_ratio(b, x) - ref)) <= 1e-11 * mag)
    a2 = Gaussian(ma, vb)                                  # equal variances: only the first term remains
    ev = tuple(range(1, x.ndim))
    delta, y = ma - mb, x - mb
    first = xp.sum(delta * (y - 0.5 * delta), axis=ev) / vb
    assert bool(xp.all(a2.log_ratio(b, x) == first))


@backends
@schedules
@given(data=st.data())
def test_K13_cancellation_free_forms_match_naive(backend, make_schedule, data):
    xp, sch = backend.xp, make_schedule()
    B, D, (x, x0, _) = data.draw(problem(backend))
    u_t, u_s = data.draw(u_pair())
    t, s = full(backend, B, sch.time(u_t)), full(backend, B, sch.time(u_s))
    m, var = transition(sch, t, s)
    naive = sch.std(s) ** 2 - (sch.mean(s) / sch.mean(t)) ** 2 * sch.std(t) ** 2
    assert close(m, sch.mean(s) / sch.mean(t), 1e-13)
    assert np.all(np.abs(host(var - naive)) <= 1e-12 * host(sch.std(s) ** 2))
    p = PosteriorStep(sch).at(Record(x, s, x0=x0), t)
    m_st = bcast(sch.mean(s) / sch.mean(t), x)
    ss2, st2 = bcast(sch.std(s) ** 2, x), bcast(sch.std(t) ** 2, x)
    mean = m_st * st2 / ss2 * x + bcast(sch.mean(t), x) * bcast(naive, x) / ss2 * x0
    assert np.all(np.abs(host(p.mean - mean)) <= 1e-10 * host(xp.abs(x) + xp.abs(x0) + 1))
    assert np.all(np.abs(host(p.var - naive * sch.std(t) ** 2 / sch.std(s) ** 2)) <= 1e-12 * host(sch.std(t) ** 2))


# ---- precision --------------------------------------------------------------------------------

@backends
@given(data=st.data())
def test_N2_log_ratio_shared_variance_precision(backend, data):
    xp, xps = backend.xp, xps_for(backend)
    B = data.draw(st.integers(1, 4))
    D = tuple(data.draw(xps.array_shapes(min_dims=1, max_dims=2, max_side=8)))
    arr = lambda lo=-3.0, hi=3.0: data.draw(xps.arrays(xp.float32, (B, *D), elements=element_bounds(lo, hi)))
    ma, mb, noise = arr(), arr(), arr()
    v = data.draw(xps.arrays(xp.float32, (B,), elements=element_bounds(2.0**-10, 2.0**10)))
    x = mb + bcast(xp.sqrt(v), noise) * noise                # x = mean of b plus noise (float32)
    for diff in (host(ma) - host(mb), host(x) - host(mb)):    # delta and y stay in the normal float32 range
        assume(np.all((diff == 0) | (np.abs(diff) >= SMALLEST_NORMAL32)))
    a, b = Gaussian(ma, v), Gaussian(mb, v)
    got = a.log_ratio(b, x)
    ref = oracle(lambda a, b, x: a.log_ratio(b, x), a, b, x)
    ev = tuple(range(1, x.ndim))
    a64, b64, x64 = (oracle(lambda z: z, z) for z in (a, b, x))
    delta, y = a64.mean - b64.mean, x64 - b64.mean
    # §18.3: 8 eps sum_i |delta_i| (|y_i| + |delta_i|) / v, plus the underflow floor: (number of terms) x the
    # smallest normal float32 on the D-term sum, which the result divides by v. No D log v term.
    bound = 8 * UNIT_ROUNDOFF32 * xp.sum(xp.abs(delta) * (xp.abs(y) + xp.abs(delta)), axis=ev) / b64.var
    bound = host(bound) + math.prod(D) * SMALLEST_NORMAL32 / host(b64.var)
    assert got.dtype == xp.float32
    assert np.all(np.abs(host(xp.astype(got, xp.float64) - ref)) <= bound)


@backends
@pytest.mark.parametrize("make_schedule", [VPLinear, EDM], ids=lambda c: c.__name__)
@given(u=st.floats(0.0, float(np.float32(0.999)), width=32, allow_subnormal=False),   # normal float32 scalars
       log_gap=st.floats(-6.0, -1.0, width=32))
def test_N3_log_snr_diff_overrides_relatively_accurate(backend, make_schedule, u, log_gap):
    xp, sch = backend.xp, make_schedule()
    t_lo, t_hi = (float(sch.time(xp.asarray(v, dtype=xp.float64))) for v in (0.0, 1.0))
    t64 = float(sch.time(xp.asarray(u, dtype=xp.float64)))
    s64 = min(t64 + 10.0**log_gap * (t_hi - t_lo), t_hi)
    t = xp.asarray([t64], dtype=xp.float32)
    s = xp.asarray([s64], dtype=xp.float32)
    if not bool(xp.all(s > t)):
        return                                                   # the gap vanished in float32
    got = sch.log_snr_diff(s, t)
    ref = oracle(sch.log_snr_diff, s, t)
    assert got.dtype == xp.float32
    bound = 16 * UNIT_ROUNDOFF32 * np.abs(host(ref)) + SMALLEST_NORMAL32
    assert np.all(np.abs(host(xp.astype(got, xp.float64) - ref)) <= bound)
    _, var = transition(sch, t, s)
    assert np.all(host(var) > 0)
