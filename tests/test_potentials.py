"""P1-P6 (potentials), N1 (increments don't see C), N4 (structural: who calls log_tilt)."""

import numpy as np
import pytest
from conftest import backends, eager, element_bounds, host, leading_shape, xps_for
from helpers import toy
from hypothesis import given
from hypothesis import strategies as st

from ttg import (PCN, Direction, EulerMaruyama, FKDiff, ForwardSDE, FKMax, FKSum, Gaussian, Never, Noising, Point, PosteriorStep, Record, Relative, ReverseRatio,
                 SequentialSweep, Sum, TiltEnds, VPLinear, Zero, constant, ladder, smc, tilt_delta)
from ttg.ops import take_levels
from ttg.potentials import PathState, TiltState
from ttg.testing import UNIT_ROUNDOFF32, SMALLEST_NORMAL32, oracle


def records(backend, data, I, N, S, dtype=None):
    """A reward sequence as records along a decreasing time grid."""
    xp, xps = backend.xp, xps_for(backend)
    dtype = xp.float64 if dtype is None else dtype
    r = data.draw(xps.arrays(dtype, (S, I, N), elements=element_bounds()))
    x = data.draw(xps.arrays(dtype, (S, I, N, 2), elements=element_bounds()))
    t = xp.linspace(1.0, 0.1, S, dtype=dtype)
    return [Record(x[k, ...], xp.broadcast_to(t[k], (I, N)), reward=r[k, ...]) for k in range(S)], r


def run(pot, recs):
    states = [pot.init(recs[0])]
    for k in range(1, len(recs)):
        states.append(pot.update(states[-1], recs[k], is_last=(k == len(recs) - 1)))
    return states


def close(a, b, rtol=1e-9):
    a, b = host(a), host(b)
    return bool(np.all(np.abs(a - b) <= rtol * (1 + np.abs(b))))


@backends
@pytest.mark.parametrize("Pot", [FKDiff, FKMax, FKSum])
@given(data=st.data(), c=st.floats(0.0, 5.0))
def test_P1_fk_potentials_telescope(backend, Pot, data, c):
    xp = backend.xp
    I, N = data.draw(leading_shape())
    S = data.draw(st.integers(2, 6))
    recs, r = records(backend, data, I, N, S)
    pot = Pot(constant(c))
    states = run(pot, recs)
    final = pot.log_tilt(states[-1])
    assert bool(xp.all(xp.abs(final - c * r[S - 1, ...]) <= 1e-9 * (1 + xp.abs(c * r[S - 1, ...]))))
    incs = sum(pot.increment(a, b) for a, b in zip(states[:-1], states[1:]))
    total = pot.log_tilt(states[-1]) - pot.log_tilt(states[0])
    mag = sum(xp.abs(pot.log_tilt(s_)) for s_ in states)
    assert np.all(np.abs(host(incs - total)) <= 1e-9 * (1 + host(mag)))


@backends
@given(data=st.data(), c=st.floats(-3.0, 3.0), d=st.floats(-3.0, 3.0))
def test_P2_tilt_ends_and_sum(backend, data, c, d):
    xp = backend.xp
    I, N = data.draw(leading_shape())
    S = data.draw(st.integers(2, 6))
    recs, r = records(backend, data, I, N, S)
    te = TiltEnds(constant(c))
    final = te.log_tilt(run(te, recs)[-1])
    assert close(final, c * r[S - 1, ...] - c * r[0, ...])
    parts = (TiltEnds(constant(c)), FKMax(constant(d)), FKDiff(constant(c + d)))
    total = Sum(*parts)
    assert total.requires == frozenset().union(*(p.requires for p in parts))
    s_sum = run(total, recs)
    s_parts = [run(p, recs) for p in parts]
    for k in range(S):
        assert close(total.log_tilt(s_sum[k]), sum(p.log_tilt(s[k]) for p, s in zip(parts, s_parts)))
        if k:
            want = sum(p.increment(s[k - 1], s[k]) for p, s in zip(parts, s_parts))
            assert close(total.increment(s_sum[k - 1], s_sum[k]), want)


@backends
@given(data=st.data())
def test_P3_reverse_ratio_same_kernel_is_zero(backend, data):
    xp = backend.xp
    I, N = data.draw(leading_shape())
    S = data.draw(st.integers(2, 5))
    recs, _ = records(backend, data, I, N, S)
    k = PosteriorStep(VPLinear())
    rr = ReverseRatio(k, k)
    assert rr.requires == frozenset()
    states = run(rr, recs)
    for a, b in zip(states[:-1], states[1:]):
        assert bool(xp.all(rr.increment(a, b) == 0))
    assert bool(xp.all(rr.log_tilt(states[-1]) == 0))


LADDER_POTENTIALS = [FKDiff, FKMax, FKSum, TiltEnds]


@backends
@pytest.mark.parametrize("Pot", LADDER_POTENTIALS)
@given(data=st.data())
def test_P4_states_are_lambda_free(backend, Pot, data):
    xp, xps = backend.xp, xps_for(backend)
    I = data.draw(st.integers(1, 3))
    N = data.draw(st.integers(2, 5))
    S = data.draw(st.integers(2, 4))
    recs, _ = records(backend, data, I, N, S)
    betas = data.draw(xps.arrays(xp.float64, (N,), elements=element_bounds(0.0, 3.0)))
    pot = Pot(ladder(betas))
    st_ = run(pot, recs)[data.draw(st.integers(0, S - 1))]
    perm = data.draw(st.permutations(range(N)))
    moved = pot.log_tilt(take_levels(st_, perm))
    for k, src in enumerate(perm):                      # state from position src, read at position k
        alone = Pot(constant(float(betas[k])))
        want = alone.log_tilt(take_levels(st_, [src]))[:, 0]
        assert close(moved[:, k], want)


ALL_POTENTIALS = {
    "zero": lambda: Zero(),
    "fkdiff": lambda: FKDiff(constant(0.8)),
    "fkmax": lambda: FKMax(constant(0.8)),
    "fksum": lambda: FKSum(constant(0.8)),
    "tiltends": lambda: TiltEnds(constant(0.8)),
    "relative": lambda: Relative(FKDiff(constant(0.8))),
    "relative_fkmax": lambda: Relative(FKMax(constant(0.8))),
    # the run below denoises, so its reverse kernels noise
    "reverse_ratio": lambda: ReverseRatio(Noising(VPLinear()), EulerMaruyama(ForwardSDE(VPLinear()), Direction.FORWARD)),
    "sum": lambda: Sum(TiltEnds(constant(0.8)), FKMax(constant(-0.3))),
}


@backends
@pytest.mark.parametrize("name", list(ALL_POTENTIALS))
@given(data=st.data())
def test_P5_increment_is_log_tilt_difference(backend, name, data):
    xp = backend.xp
    I, N = data.draw(leading_shape())
    S = data.draw(st.integers(2, 6))
    recs, _ = records(backend, data, I, N, S)
    pot = ALL_POTENTIALS[name]()
    states = run(pot, recs)
    for k in range(1, S):
        a, b = states[k - 1], states[k]
        mag = host(xp.abs(pot.log_tilt(a)) + xp.abs(pot.log_tilt(b)))
        assert np.all(np.abs(host(pot.increment(a, b) - (pot.log_tilt(b) - pot.log_tilt(a)))) <= 1e-9 * (1 + mag))
    k = data.draw(st.integers(1, S - 1))                # two candidates extending the same history
    prev, last = states[k - 1], k == S - 1
    alt = Record(recs[k].x + 0.5, recs[k].t, x0=None if recs[k].x0 is None else recs[k].x0 - 0.25,
                 reward=recs[k].reward * 0.5 - 1.0)
    a, b = pot.update(prev, recs[k], is_last=last), pot.update(prev, alt, is_last=last)
    mag = host(xp.abs(pot.log_tilt(a)) + xp.abs(pot.log_tilt(b)))
    assert np.all(np.abs(host(tilt_delta(pot, prev, a, b) - (pot.log_tilt(a) - pot.log_tilt(b)))) <= 1e-9 * (1 + mag))


@backends
@pytest.mark.parametrize("make", [lambda: FKDiff(constant(0.8)), lambda: FKMax(constant(-1.3)),
                                  lambda: Sum(TiltEnds(constant(0.5)), FKDiff(constant(2.0)))],
                         ids=["fkdiff", "fkmax", "sum"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@given(data=st.data())
def test_P6_relative_is_zero_at_start_and_keeps_increments(backend, make, dtype, data):
    xp = backend.xp
    I, N = data.draw(leading_shape())
    S = data.draw(st.integers(2, 5))
    recs, _ = records(backend, data, I, N, S, dtype=getattr(xp, dtype))
    inner = make()
    rel = Relative(inner)
    assert rel.requires == inner.requires
    s_rel, s_in = run(rel, recs), run(inner, recs)
    assert bool(xp.all(rel.log_tilt(s_rel[0]) == 0))                  # exactly, in any dtype
    for k in range(1, S):
        assert bool(xp.all(rel.increment(s_rel[k - 1], s_rel[k]) == inner.increment(s_in[k - 1], s_in[k])))
    if dtype == "float64":
        c0 = inner.log_tilt(inner.init(recs[0]))
        for k in range(S):
            assert close(rel.log_tilt(s_rel[k]), inner.log_tilt(s_in[k]) - c0)


# ---- N1 ------------------------------------------------------------------------------------------

@backends
@pytest.mark.parametrize("Pot", [FKDiff, FKMax, FKSum])
@given(data=st.data(), c=st.floats(-5.0, 5.0, width=32, allow_subnormal=False))   # normal float32 scalars
def test_N1_increments_do_not_see_the_accumulator(backend, Pot, data, c):
    xp, xps = backend.xp, xps_for(backend)
    I, N = data.draw(leading_shape())
    f32 = lambda lo, hi: data.draw(xps.arrays(xp.float32, (I, N), elements=element_bounds(lo, hi)))
    r_prev, r = f32(-3.0, 3.0), f32(-3.0, 3.0)
    run_, acc = f32(-1e6, 1e6), f32(-1e6, 1e6)
    t = xp.full((I, N), 0.5, dtype=xp.float32)
    pot = Pot(constant(c))
    if Pot is FKDiff:
        prev = TiltState(reward=r_prev, t=t)
    else:
        prev = PathState(running=run_, cumulative=acc, reward=r_prev, t=t, is_last=False)
    rec = Record(xp.zeros((I, N, 1), dtype=xp.float32), t, reward=r)
    inc = pot.increment(prev, pot.update(prev, rec, is_last=False))  # a non-final record
    ref = oracle(lambda p, rc: pot.increment(p, pot.update(p, rc, is_last=False)), prev, rec)
    assert inc.dtype == xp.float32
    bound = 4 * UNIT_ROUNDOFF32 * (np.abs(host(ref)) + abs(c) * (np.abs(host(r_prev)) + np.abs(host(r))))
    bound = bound + 2 * SMALLEST_NORMAL32                           # underflow floor: 2 terms (lambda r, lambda r_prev)
    assert np.all(np.abs(host(xp.astype(inc, xp.float64) - ref)) <= bound)


# ---- N4 ------------------------------------------------------------------------------------------

class Spy:
    """Delegates to `inner`, counting direct calls to log_tilt."""

    def __init__(self, inner):
        self.inner, self.log_tilt_calls, self.increment_calls = inner, 0, 0
        self.requires = inner.requires

    def init(self, rec):
        return self.inner.init(rec)

    def update(self, st, rec, *, is_last):
        return self.inner.update(st, rec, is_last=is_last)

    def log_tilt(self, st):
        self.log_tilt_calls += 1
        return self.inner.log_tilt(st)

    def increment(self, prev, st):
        self.increment_calls += 1
        return self.inner.increment(prev, st)


@backends
@pytest.mark.parametrize("Pot", [FKDiff, FKMax])
@given(lead=leading_shape(), S=st.integers(2, 5), n_steps=st.integers(1, 3), seed=st.integers(0, 2**20))
def test_N4_log_tilt_only_where_histories_differ(backend, Pot, lead, S, n_steps, seed):
    I, N = lead
    xp, sch = backend.xp, VPLinear()
    ev, _, _ = toy(backend, sch)
    P = PosteriorStep(sch)
    prior = sch.prior((I, N), (2,), xp=xp, dtype=xp.float64)
    point = Point(prior.sample(backend.key(seed + 1), backend), sch.time(xp.ones((I, N), dtype=xp.float64)))

    def go(start, move):
        spy = Spy(Pot(constant(1.0)))
        smc(start, xp.linspace(1.0, 0.0, S, dtype=xp.float64), backend=backend, evaluate=ev, schedule=sch, target=P,
            proposal=P, potential=spy, policy=Never(), move=move, key=backend.key(seed))
        return spy

    with eager(backend):
        assert go(point, None).log_tilt_calls == 1                       # the start point only
        assert go(point, PCN(n_steps, constant(0.4))).log_tilt_calls == 1   # st_prev is never None here
        assert go(prior, PCN(n_steps, constant(0.4))).log_tilt_calls == 1 + 2 * n_steps   # start move: st_prev None
        swaps = go(point, PCN(n_steps, constant(0.4), pairs=SequentialSweep()))
        rounds_per_sweep = max(N - 1, 0)                                  # StateSwap: 2 calls per round
        assert swaps.log_tilt_calls == 1 + 2 * rounds_per_sweep * n_steps * (S - 1)
