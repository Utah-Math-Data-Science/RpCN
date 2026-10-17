"""C1-C2 (evaluation contract) and E1-E5 (the SMC runner), run eagerly."""

import numpy as np
import pytest
from conftest import BACKENDS, backends, eager, elems, host, leading, xps_for
from helpers import ALL, toy
from hypothesis import given
from hypothesis import strategies as st

from ttg import (PCN, ESSBelow, Every, Feature, FKDiff, FKMax, FKSum, Gaussian, Guided, Langevin, LangevinMove,
                 Never, PartialRecord, Point, PosteriorStep, Record, ReverseRatio, Sum, TiltEnds, VPLinear, Zero,
                 constant, evaluate_rows, make_evaluate_jax, make_evaluate_torch, on_x0, smc, start_record,
                 tilt, x0_from_eps)
from ttg.testing import AuditingEvaluate, CountingNetwork, ReplayKernel, toy_eps_network, toy_evaluate

AUTODIFF = [bk for bk in BACKENDS if bk.name in ("jax", "torch")]


def builder(bk, net, sch, w):
    xp = bk.xp
    to_x0_eps = lambda x, t, eps, sch: (x0_from_eps(x, t, eps, sch), eps)
    reward = on_x0(lambda x0: xp.sum(w * x0, axis=tuple(range(1, x0.ndim))))
    make = make_evaluate_jax if bk.name == "jax" else make_evaluate_torch
    return make(net, to_x0_eps, sch, reward)


@pytest.mark.parametrize("bk", AUTODIFF, ids=str)
@given(data=st.data(), needs=st.sets(st.sampled_from(Feature)))
def test_C1_builders_one_forward_exact_features(bk, data, needs):
    xp, sch = bk.xp, VPLinear()
    B = data.draw(st.integers(1, 4))
    x = data.draw(xps_for(bk).arrays(xp.float64, (B, 2), elements=elems()))
    t = sch.time(xp.asarray(data.draw(st.lists(st.floats(0.0, 1.0), min_size=B, max_size=B)), dtype=xp.float64))
    ref_eval, mu, w = toy(bk, sch)
    net = CountingNetwork(toy_eps_network(mu, 1.0, sch))
    rec = builder(bk, net, sch, w)(x, t, frozenset(needs))
    assert net.calls <= 1
    ref = ref_eval(x, t, ALL)
    for f in Feature:
        v = getattr(rec, f.value)
        assert (v is not None) == (f in needs)
        if v is not None:
            assert np.allclose(host(v), host(getattr(ref, f.value)), rtol=1e-9, atol=1e-9)


@backends
@given(data=st.data(), needs=st.sets(st.sampled_from(Feature)))
def test_C1_toy_evaluate_exact_features(bk, data, needs):
    xp, sch = bk.xp, VPLinear()
    ev, _, _ = toy(bk, sch)
    x = xp.zeros((3, 2), dtype=xp.float64)
    t = xp.full((3,), 0.5, dtype=xp.float64)
    rec = ev(x, t, frozenset(needs))
    assert all((getattr(rec, f.value) is not None) == (f in needs) for f in Feature)


@backends
def test_C2_start_record_rejects_partial_record(bk):
    xp, sch = bk.xp, VPLinear()
    ev, _, _ = toy(bk, sch)
    rec = Record(xp.zeros((1, 2, 2), dtype=xp.float64), xp.ones((1, 2), dtype=xp.float64),
                 reward=xp.zeros((1, 2), dtype=xp.float64))
    with pytest.raises(PartialRecord):
        start_record(rec, rec.t, {Feature.REWARD, Feature.X0}, ev, bk, bk.key(0))
    out, ref = start_record(rec, rec.t, {Feature.REWARD}, ev, bk, bk.key(0))
    assert ref is None and out.reward is rec.reward and out.x0 is None


# ---- E: the runner ------------------------------------------------------------------------------

def gaussian_start(bk, sch, I, N, key=123):
    xp = bk.xp
    prior = sch.prior((I, N), (2,), xp=xp, dtype=xp.float64)
    return prior


def u_grid(bk, S):
    return bk.xp.linspace(1.0, 0.0, S, dtype=bk.xp.float64)


def kinds_of_start(bk, sch, ev, I, N, S, needs):
    """Gaussian, Point and Record starts at u = 1."""
    xp = bk.xp
    g = gaussian_start(bk, sch, I, N)
    x = g.sample(bk.key(7), bk)
    t0 = sch.time(xp.ones((I, N), dtype=xp.float64))
    rec = evaluate_rows(ev, Point(x, t0), needs, bk)
    return {"gaussian": g, "point": Point(x + 1.0, t0), "record": rec}


MOVES = {
    "none": (None, 0),
    "pcn": (PCN(n_steps=2, step=constant(0.3)), 2),
    "langevin": (LangevinMove(Guided(Langevin(lambda t: 0.01 * VPLinear().std(t) ** 2), tilt(constant(1.0), 0.5))), 1),
}


@backends
@pytest.mark.parametrize("start_kind", ["gaussian", "point", "record"])
@pytest.mark.parametrize("move_kind", list(MOVES))
@given(lead=leading(), S=st.integers(2, 6), stride=st.integers(1, 3), seed=st.integers(0, 2**20))
def test_E1_row_counts_and_no_duplicates(bk, start_kind, move_kind, lead, S, stride, seed):
    I, N = lead
    sch = VPLinear()
    ev, _, _ = toy(bk, sch)
    move, per = MOVES[move_kind]
    P = PosteriorStep(sch)
    pot = FKDiff(constant(1.0))
    with eager(bk):
        start = kinds_of_start(bk, sch, ev, I, N, S, ALL)[start_kind]
        audit = AuditingEvaluate(ev)
        smc(start, u_grid(bk, S), bk=bk, evaluate=audit, schedule=sch, target=P, proposal=Guided(P, tilt(constant(1.0))),
            potential=pot, policy=Every(2), move=move, move_at=lambda k: k % stride == 0,
            final_needs={Feature.X0}, key=bk.key(seed))
    steps = range(0 if start_kind == "gaussian" else 1, S)
    moved = sum(1 for k in steps if k % stride == 0)
    rows = I * N * (S if start_kind != "record" else S - 1) + I * N * per * moved
    assert audit.rows == rows


@backends
@given(lead=leading(), S=st.integers(2, 6), seed=st.integers(0, 2**20),
       policy=st.sampled_from([Never(), Every(1), Every(2), ESSBelow(0.5)]))
def test_E2_proposal_is_target_and_zero_give_zero_weights(bk, lead, S, seed, policy):
    I, N = lead
    xp, sch = bk.xp, VPLinear()
    ev, _, _ = toy(bk, sch)
    P = PosteriorStep(sch)
    with eager(bk):
        out = smc(gaussian_start(bk, sch, I, N), u_grid(bk, S), bk=bk, evaluate=ev, schedule=sch, target=P,
                  proposal=P, potential=Zero(), policy=policy, key=bk.key(seed))
    assert bool(xp.all(out.log_w == 0))


POTENTIALS = {
    "fkdiff": lambda P, Q: FKDiff(constant(0.7)),
    "fkmax": lambda P, Q: FKMax(constant(0.7)),
    "fksum": lambda P, Q: FKSum(constant(0.7)),
    "tiltends": lambda P, Q: TiltEnds(constant(0.7)),
    "sum_rr": lambda P, Q: Sum(TiltEnds(constant(0.7)), ReverseRatio(Q, P)),
}


def replay_setup(bk, data, S, I, N):
    xp, sch = bk.xp, VPLinear()
    ev, _, _ = toy(bk, sch)
    path = data.draw(xps_for(bk).arrays(xp.float64, (S, I, N, 2), elements=elems()))
    u = u_grid(bk, S)
    t_grid = xp.broadcast_to(xp.reshape(sch.time(u), (S, 1, 1)), (S, I, N))
    return xp, sch, ev, path, u, t_grid


@backends
@pytest.mark.parametrize("pot_kind", list(POTENTIALS))
@given(data=st.data(), lead=leading(), S=st.integers(2, 5), c=st.floats(-2.0, 2.0))
def test_E3_replayed_weights_match_plain_loop(bk, pot_kind, data, lead, S, c):
    I, N = lead
    xp, sch, ev, path, u, t_grid = replay_setup(bk, data, S, I, N)
    P = PosteriorStep(sch)
    Qb = Guided(P, tilt(constant(c)))
    Q = ReplayKernel(Qb, path, t_grid)
    pot = POTENTIALS[pot_kind](P, Qb)
    with eager(bk):
        out = smc(Point(path[0, ...], t_grid[0, ...]), u, bk=bk, evaluate=ev, schedule=sch, target=P, proposal=Q,
                  potential=pot, policy=Never(), key=bk.key(0))
    recs = [evaluate_rows(ev, Point(path[k, ...], t_grid[k, ...]), ALL, bk) for k in range(S)]
    st_ = pot.init(recs[0])
    ratio = xp.zeros((I, N), dtype=xp.float64)
    for k in range(1, S):
        st_ = pot.update(st_, recs[k], last=(k == S - 1))
        x = path[k, ...]
        ratio = ratio + P.at(recs[k - 1], t_grid[k, ...]).log_prob(x) - Qb.at(recs[k - 1], t_grid[k, ...]).log_prob(x)
    ref = pot.log_tilt(st_) + ratio
    assert np.all(np.abs(host(out.log_w - ref)) <= 1e-8 * (1 + np.abs(host(ref))))
    assert bool(xp.all(out.final.x == path[S - 1, ...]))              # E4: Never() keeps row order


@backends
@given(data=st.data(), lead=leading(), S=st.integers(2, 5))
def test_E4_never_keeps_row_order(bk, data, lead, S):
    I, N = lead
    xp, sch, ev, path, u, t_grid = replay_setup(bk, data, S, I, N)
    P = PosteriorStep(sch)
    with eager(bk):
        out = smc(Point(path[0, ...], t_grid[0, ...]), u, bk=bk, evaluate=ev, schedule=sch, target=P,
                  proposal=ReplayKernel(P, path, t_grid), potential=FKDiff(constant(5.0)), policy=Never(),
                  key=bk.key(0))
    assert bool(xp.all(out.final.x == path[S - 1, ...]))


@backends
@pytest.mark.parametrize("move_kind", ["pcn", "langevin"])
@given(lead=leading(), S=st.integers(2, 6), seed=st.integers(0, 2**20))
def test_E5_move_never_fired_is_no_move(bk, move_kind, lead, S, seed):
    I, N = lead
    xp, sch = bk.xp, VPLinear()
    ev, _, _ = toy(bk, sch)
    P = PosteriorStep(sch)
    run = lambda **kw: smc(gaussian_start(bk, sch, I, N), u_grid(bk, S), bk=bk, evaluate=ev, schedule=sch,
                           target=P, proposal=Guided(P, tilt(constant(1.0))), potential=FKMax(constant(1.0)),
                           policy=Every(2), final_needs={Feature.X0}, key=bk.key(seed), **kw)
    with eager(bk):
        a = run(move=None)
        b = run(move=MOVES[move_kind][0], move_at=lambda k: False)
    assert bool(xp.all(a.log_w == b.log_w))
    for name in ("x", "t", "x0", "reward"):              # the move's own requires add fields to b only
        assert bool(xp.all(getattr(a.final, name) == getattr(b.final, name)))
