"""M1-M7 (moves and replica exchange) and the §17 RpCN count."""

from dataclasses import dataclass

import numpy as np
import pytest
from conftest import backends, eager, element_bounds, host, leading_shape, xps_for
from helpers import ALL, toy
from hypothesis import given
from hypothesis import strategies as st

from ttg import (PCN, EvenOdd, Feature, FKDiff, FKMax, Gaussian, Guided, MoveContext, Never, Point, PosteriorStep, Record,
                 SequentialSweep, StateSwap, VPLinear, constant, evaluate_rows, ladder, replica_exchange, smc, tilt)
from ttg.potentials import TiltState
from ttg.testing import AuditingEvaluate, ReplayKernel


@backends
@given(data=st.data(), rho=st.floats(0.01, 0.99))
def test_M1_pcn_is_reversible_wrt_reference(backend, data, rho):
    xp, xps = backend.xp, xps_for(backend)
    B = data.draw(st.integers(1, 4))
    D = data.draw(xps.array_shapes(min_dims=1, max_dims=2, max_side=3))
    arr = lambda: data.draw(xps.arrays(xp.float64, (B, *D), elements=element_bounds()))
    mu, x, x_hat = arr(), arr(), arr()
    tau2 = data.draw(xps.arrays(xp.float64, (B,), elements=element_bounds(0.01, 4.0)))
    ref = Gaussian(mu, tau2)
    q = lambda a: Gaussian(mu + (1 - rho**2) ** 0.5 * (a - mu), rho**2 * tau2)
    lhs = ref.log_prob(x) + q(x).log_prob(x_hat)
    rhs = ref.log_prob(x_hat) + q(x_hat).log_prob(x)
    assert bool(xp.all(xp.abs(lhs - rhs) <= 1e-8 * (1 + xp.abs(lhs))))


def context(backend, sch, ev, I, N, key, needs=ALL, with_prev=True):
    """A MoveContext at a mid-run point: rec drawn from a PosteriorStep reference."""
    xp = backend.xp
    k1, k2, k3 = backend.split(key, 3)
    t_prev = sch.time(xp.full((I, N), 0.6, dtype=xp.float64))
    t = sch.time(xp.full((I, N), 0.5, dtype=xp.float64))
    prev = evaluate_rows(ev, Point(backend.normal(k1, (I, N, 2), xp.float64), t_prev), needs, backend)
    ref = PosteriorStep(sch).at(prev, t)
    rec = evaluate_rows(ev, Point(ref.sample(k2, backend), t), needs, backend)
    return prev, rec, ref


@backends
@given(lead=leading_shape(), n_steps=st.integers(1, 4), seed=st.integers(0, 2**20), at_start=st.booleans())
def test_M2_zero_tilt_accepts_every_proposal(backend, lead, n_steps, seed, at_start):
    I, N = lead
    xp, sch = backend.xp, VPLinear()
    ev, _, _ = toy(backend, sch)
    pot = FKDiff(constant(0.0))
    prev, rec, ref = context(backend, sch, ev, I, N, backend.key(seed))
    st_prev = None if at_start else pot.init(prev)
    st_ = pot.init(rec) if at_start else pot.update(st_prev, rec, is_last=False)
    cands = []

    def spy(p):
        cands.append(p.x)
        return evaluate_rows(ev, p, ALL, backend)

    with eager(backend):
        out = PCN(n_steps, constant(0.5))(MoveContext(rec, st_, st_prev, ref), evaluate_point=spy, potential=pot,
                                          is_last=False, move_count=0, key=backend.key(seed + 1),
                                          backend=backend)
    assert len(cands) == n_steps
    assert bool(xp.all(out.record.x == cands[-1]))


@dataclass(frozen=True)
class Recording:
    """Wraps a move and records the contexts it receives and returns (eager tests only)."""
    inner: object
    log: list

    @property
    def requires(self):
        return self.inner.requires

    def __call__(self, ctx, **kw):
        out = self.inner(ctx, **kw)
        self.log.append(out)
        return out


@backends
@pytest.mark.parametrize("Pot", [FKDiff, FKMax])
@given(data=st.data(), lead=leading_shape(), S=st.integers(2, 5), seed=st.integers(0, 2**20), c=st.floats(0.0, 3.0))
def test_M3_moves_never_change_log_w(backend, Pot, data, lead, S, seed, c):
    I, N = lead
    xp, sch = backend.xp, VPLinear()
    ev, _, _ = toy(backend, sch)
    path = data.draw(xps_for(backend).arrays(xp.float64, (S, I, N, 2), elements=element_bounds()))
    u = xp.linspace(1.0, 0.0, S, dtype=xp.float64)
    t_grid = xp.broadcast_to(xp.reshape(sch.time(u), (S, 1, 1)), (S, I, N))
    P = PosteriorStep(sch)
    Qb = Guided(P, tilt(constant(0.5)))
    pot = Pot(constant(c))
    log = []
    move = Recording(PCN(2, constant(0.4)), log)
    with eager(backend):
        out = smc(Point(path[0, ...], t_grid[0, ...]), u, backend=backend, evaluate=ev, schedule=sch, target=P,
                  proposal=ReplayKernel(Qb, path, t_grid), potential=pot, policy=Never(), move=move,
                  key=backend.key(seed))
    assert len(log) == S - 1                                             # a Point start has no start move
    # Plain loop: weights from the moved point of each step, with no term for the move itself.
    rec = evaluate_rows(ev, Point(path[0, ...], t_grid[0, ...]), ALL, backend)
    st_ = pot.init(rec)
    log_w = pot.log_tilt(st_)
    for k in range(1, S):
        x = path[k, ...]
        log_w = log_w + P.at(rec, t_grid[k, ...]).log_prob(x) - Qb.at(rec, t_grid[k, ...]).log_prob(x)
        new = evaluate_rows(ev, Point(x, t_grid[k, ...]), ALL, backend)
        st_new = pot.update(st_, new, is_last=(k == S - 1))
        log_w = log_w + pot.increment(st_, st_new)
        rec, st_ = log[k - 1].record, log[k - 1].potential_state         # the move's output carries on
    assert np.all(np.abs(host(out.log_weights - log_w)) <= 1e-8 * (1 + np.abs(host(log_w))))


@backends
@given(data=st.data(), seed=st.integers(0, 2**20))
def test_M4_state_swap_acceptance(backend, data, seed):
    xp, xps = backend.xp, xps_for(backend)
    I = data.draw(st.integers(1, 3))
    n = data.draw(st.integers(2, 5))
    betas = data.draw(xps.arrays(xp.float64, (n,), elements=element_bounds(0.0, 3.0)))
    r = data.draw(xps.arrays(xp.float64, (I, n), elements=element_bounds()))
    t = xp.full((I, n), 0.3, dtype=xp.float64)
    pot = FKDiff(ladder(betas))
    rec = Record(xp.zeros((I, n, 1), dtype=xp.float64), t, reward=r)
    ctx = MoveContext(rec, pot.init(rec), None, None)
    for rnd in EvenOdd().rounds(n, data.draw(st.integers(0, 1))) + SequentialSweep().rounds(n, 0):
        if not rnd:
            continue
        lo, hi = [a for a, _ in rnd], [b for _, b in rnd]
        _, _, log_a = StateSwap(pot).propose(ctx, lo, hi, backend.key(seed), backend)
        for p, (a, b) in enumerate(rnd):
            want = (betas[b] - betas[a]) * (r[:, a] - r[:, b])
            assert np.all(np.abs(host(log_a[:, p] - want)) <= 1e-12 * (1 + np.abs(host(want))))


@given(n=st.integers(1, 8), sweep=st.integers(0, 10))
def test_M5_pair_schedules(n, sweep):
    for sched in (SequentialSweep(), EvenOdd()):
        seen = set()
        for s in range(sweep, sweep + sched.period):
            for rnd in sched.rounds(n, s):
                flat = [i for p in rnd for i in p]
                assert len(flat) == len(set(flat))                       # disjoint
                assert all(b == a + 1 and 0 <= a and b < n for a, b in rnd)   # adjacent, in range
                seen.update(rnd)
            assert sched.rounds(n, s) == sched.rounds(n, s + sched.period)
        assert seen == {(k, k + 1) for k in range(n - 1)}
    rounds = SequentialSweep().rounds(n, sweep)
    assert len(rounds) == max(n - 1, 0) and all(len(r) == 1 for r in rounds)


def labelled_context(backend, I, n, betas):
    """Every leaf of row j encodes j, so mixing rows is detectable."""
    xp = backend.xp
    lab = xp.broadcast_to(xp.astype(xp.arange(n), xp.float64)[None, :], (I, n))
    rec = Record(xp.broadcast_to(lab[..., None], (I, n, 2)) * 1.0, lab * 1.0 + 0.5, reward=lab * 0.0)
    pot = FKDiff(ladder(betas))
    st_prev = TiltState(reward=lab * 2.0, t=lab * 3.0)
    ref = Gaussian(xp.broadcast_to(lab[..., None], (I, n, 2)) * 4.0, lab * 5.0 + 1.0)
    return MoveContext(rec, TiltState(reward=lab * 0.0, t=lab * 1.0 + 0.5), st_prev, ref), pot


@backends
@given(I=st.integers(1, 3), n=st.integers(2, 6), seed=st.integers(0, 2**20))
def test_M6_equal_betas_sequential_sweep_carries_position_zero_to_the_end(backend, I, n, seed):
    xp = backend.xp
    ctx, pot = labelled_context(backend, I, n, xp.full((n,), 1.5, dtype=xp.float64))
    with eager(backend):
        out = replica_exchange(ctx, SequentialSweep(), StateSwap(pot), sweep=0, key=backend.key(seed), backend=backend)
    assert bool(xp.all(out.record.x[:, n - 1, ...] == 0))
    assert bool(xp.all(out.record.x[:, :n - 1, 0] == xp.astype(xp.arange(1, n), xp.float64)[None, :]))


@backends
@pytest.mark.parametrize("pairs", [SequentialSweep(), EvenOdd()], ids=lambda p: type(p).__name__)
@given(data=st.data(), I=st.integers(1, 3), n=st.integers(2, 6), seed=st.integers(0, 2**20),
       sweep=st.integers(0, 3))
def test_M7_swaps_keep_rows_consistent(backend, pairs, data, I, n, seed, sweep):
    xp = backend.xp
    betas = data.draw(xps_for(backend).arrays(xp.float64, (n,), elements=element_bounds(0.0, 3.0)))
    ctx, pot = labelled_context(backend, I, n, betas)
    with eager(backend):
        out = replica_exchange(ctx, pairs, StateSwap(pot), sweep=xp.asarray(sweep), key=backend.key(seed),
                               backend=backend)
    src = out.record.x[:, :, 0]
    for leaf, scale, shift in ((out.record.t, 1.0, 0.5), (out.potential_state.t, 1.0, 0.5),
                               (out.prev_potential_state.reward, 2.0, 0.0), (out.prev_potential_state.t, 3.0, 0.0),
                               (out.reference.mean[:, :, 1], 4.0, 0.0), (out.reference.var, 5.0, 1.0)):
        assert bool(xp.all(leaf == src * scale + shift))
    assert sorted(host(src[0, ...]).tolist()) == list(range(n))         # a permutation of the rows


@backends
@given(I=st.integers(1, 2), K=st.integers(1, 3), S=st.integers(2, 5), M=st.integers(1, 3),
       stride=st.integers(1, 3), seed=st.integers(0, 2**20))
def test_counts_rpcn(backend, I, K, S, M, stride, seed):
    xp, sch = backend.xp, VPLinear()
    ev, _, _ = toy(backend, sch)
    audit = AuditingEvaluate(ev)
    P = PosteriorStep(sch)
    betas = xp.linspace(0.0, 2.0, K + 1, dtype=xp.float64)
    steps = xp.linspace(0.5, 0.2, K + 1, dtype=xp.float64)
    with eager(backend):
        smc(sch.prior((I, K + 1), (2,), xp=xp, dtype=xp.float64), xp.linspace(1.0, 0.0, S, dtype=xp.float64),
            backend=backend, evaluate=audit, schedule=sch, target=P, proposal=P, potential=FKDiff(ladder(betas)),
            policy=Never(), move=PCN(n_proposals=M, step_size=ladder(steps), pairs=EvenOdd()),
            move_at=lambda k: k % stride == 0, out_needs={Feature.X0}, key=backend.key(seed))
    steered = sum(1 for k in range(S) if k % stride == 0)
    assert audit.rows == I * (K + 1) * (S + M * steered)
