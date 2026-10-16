"""X1-X4: CREPE's path exchange, replica invariant and evaluation counts."""

import numpy as np
import pytest
from conftest import backends, eager, elems, host, xps_for
from helpers import ALL, toy
from hypothesis import given
from hypothesis import strategies as st

from ttg import (CREPEConfig, Direction, EulerMaruyama, EvenOdd, ForwardSDE, Guided, Langevin, LangevinMove,
                 Noising, PathExchange, Point, ReverseSDE, SequentialSweep, VPLinear, crepe, crepe_init,
                 evaluate_rows, level_lambda, per_row_grid, tilt)
from ttg.ops import steps_first, take_levels
from ttg.testing import AuditingEvaluate, ReplayKernel


@st.composite
def u_levels(draw, max_M=3, min_gap=0.05):
    M = draw(st.integers(1, max_M))
    inner = draw(st.lists(st.floats(0.0, 0.9), min_size=M, max_size=M, unique=True))
    pts = sorted(inner) + [1.0]
    if any(b - a < min_gap for a, b in zip(pts, pts[1:])):
        pts = [m / M for m in range(M + 1)]
    return tuple(pts)


def kernels(sch, lam):
    B_P = EulerMaruyama(ReverseSDE(sch), Direction.BACKWARD)
    return Noising(sch), B_P, Guided(B_P, tilt(lam))


def make_cfg(bk, sch, levels, K, pairs=EvenOdd(), local=True, beta=1.5, rho=1.0, **kinds):
    xp = bk.xp
    lam = level_lambda(sch.time(xp.asarray(levels, dtype=xp.float64)), beta=beta, rho=rho)
    F_P, B_P, B_Q = kernels(sch, lam)
    kw = dict(F_P=F_P, F_Q=F_P, B_P=B_P, B_Q=B_Q)
    kw.update(kinds)
    move = LangevinMove(Guided(Langevin(h=lambda t: 0.01 * sch.std(t) ** 2), tilt(lam, 0.5))) if local else None
    return CREPEConfig(u_levels=levels, K=K, lam=lam, pairs=pairs, local_move=move,
                       local_levels=tuple(range(len(levels) - 1)) if local else (), **kw)


def replicas(bk, sch, ev, cfg, I, seed, needs):
    """Replicas at exactly the level times, with exactly `needs`."""
    xp = bk.xp
    t = sch.time(xp.asarray(cfg.u_levels, dtype=xp.float64))
    x = bk.normal(bk.key(seed), (I, cfg.M + 1, 2), xp.float64)
    return evaluate_rows(ev, Point(x, xp.broadcast_to(t[None, :], (I, cfg.M + 1))), needs, bk)


def fine_times(bk, sch, cfg, lo, I, ascending):
    """[K+1, I, P] fine times, in run order, and the same as a list of [I, P] steps."""
    xp = bk.xp
    g = steps_first(sch.time(per_row_grid(cfg, lo, I, ascending, xp=xp, dtype=xp.float64)), (I, len(lo)))
    return g, [g[k, ...] for k in range(g.shape[0])]


def ell(cfg, rec):
    return cfg.lam(rec.t) * rec.reward


def chain(ev, xs, ts, bk):
    return [evaluate_rows(ev, Point(x, t), ALL, bk) for x, t in zip(xs, ts)]


def path_terms(recs, ts, P, Q):
    """sum_k log P(x_k | x_{k-1}) - log Q(x_k | x_{k-1}) along a chain in run order."""
    return sum(P.at(recs[k - 1], ts[k]).log_prob(recs[k].x) - Q.at(recs[k - 1], ts[k]).log_prob(recs[k].x)
               for k in range(1, len(recs)))


def draw_pair_round(data, cfg):
    sweep = data.draw(st.integers(0, 1))
    rnd = [r for r in EvenOdd().rounds(cfg.M + 1, sweep) if r]
    if not rnd:
        rnd = [((0, 1),)]
    lo, hi = [a for a, _ in rnd[0]], [b for _, b in rnd[0]]
    return lo, hi


@backends
@pytest.mark.parametrize("mode", ["four_kernel", "endpoints", "b_only"])
@given(data=st.data(), levels=u_levels(), K=st.integers(1, 3), I=st.integers(1, 2), seed=st.integers(0, 2**20))
def test_X1_X2_path_exchange_acceptance(bk, mode, data, levels, K, I, seed):
    xp, sch = bk.xp, VPLinear()
    ev, _, _ = toy(bk, sch)
    base = make_cfg(bk, sch, levels, K, local=False)
    lo, hi = draw_pair_round(data, base)
    P_ = len(lo)
    (t_up, ts_up), (t_dn, ts_dn) = fine_times(bk, sch, base, lo, I, True), fine_times(bk, sch, base, lo, I, False)
    paths = lambda: data.draw(xps_for(bk).arrays(xp.float64, (K + 1, I, P_, 2), elements=elems()))
    y_path, z_path = paths(), paths()
    F_P, B_P, B_Q = base.F_P, base.B_P, base.B_Q
    F_Qb = EulerMaruyama(ForwardSDE(sch), Direction.FORWARD)
    if mode == "four_kernel":                         # X1: all four kernels distinct, both paths replayed
        kinds = dict(F_P=F_P, F_Q=ReplayKernel(F_Qb, y_path, t_up), B_P=B_P, B_Q=ReplayKernel(B_Q, z_path, t_dn))
        dens = dict(FP=F_P, FQ=F_Qb, BP=B_P, BQ=B_Q)
    elif mode == "b_only":                            # X2: F^Q = F^P (one object) -> B-only formula
        F = ReplayKernel(F_P, y_path, t_up)
        kinds = dict(F_P=F, F_Q=F, B_P=B_P, B_Q=ReplayKernel(B_Q, z_path, t_dn))
        dens = dict(FP=F_P, FQ=F_P, BP=B_P, BQ=B_Q)
    else:                                             # X2: F^Q = F^P and B^Q = B^P -> endpoint tilts
        kinds = dict(F_P=ReplayKernel(F_P, y_path, t_up), B_P=ReplayKernel(B_P, z_path, t_dn))
        kinds.update(F_Q=kinds["F_P"], B_Q=kinds["B_P"])
        dens = None
    cfg = make_cfg(bk, sch, levels, K, local=False, **kinds)
    R = replicas(bk, sch, ev, cfg, I, seed, cfg.replica_needs)
    with eager(bk):
        new_lo, new_hi, log_a = PathExchange(cfg, ev, sch).propose(R, lo, hi, bk.key(seed + 1), bk)
    y = chain(ev, [take_levels(R, lo).x] + [y_path[k, ...] for k in range(1, K + 1)], ts_up, bk)
    z = chain(ev, [take_levels(R, hi).x] + [z_path[k, ...] for k in range(1, K + 1)], ts_dn, bk)  # run order
    assert bool(xp.all(new_hi.x == y[-1].x)) and bool(xp.all(new_lo.x == z[-1].x))
    tilts = ell(cfg, z[-1]) - ell(cfg, z[0]) + ell(cfg, y[-1]) - ell(cfg, y[0])
    want = tilts
    if dens is not None:
        rev = lambda c: c[::-1]                       # a chain read in the other direction
        # up: log R^Q(y) - log R^P(y); down: log R^P(z) - log R^Q(z); R^X = prod B^X / F^X
        want = (want + path_terms(rev(y), rev(ts_up), dens["BQ"], dens["BP"])
                + path_terms(y, ts_up, dens["FP"], dens["FQ"])
                + path_terms(z, ts_dn, dens["BP"], dens["BQ"])
                + path_terms(rev(z), rev(ts_dn), dens["FQ"], dens["FP"]))
    assert np.all(np.abs(host(log_a - want)) <= 1e-8 * (1 + np.abs(host(want))))


@backends
@pytest.mark.parametrize("pairs", [EvenOdd(), SequentialSweep()], ids=lambda p: type(p).__name__)
@given(levels=u_levels(), K=st.integers(1, 3), I=st.integers(1, 2), n_blocks=st.integers(1, 3),
       seed=st.integers(0, 2**20), local=st.booleans())
def test_X3_X4_replica_invariant_and_counts(bk, pairs, levels, K, I, n_blocks, seed, local):
    xp, sch = bk.xp, VPLinear()
    ev, _, _ = toy(bk, sch)
    cfg = make_cfg(bk, sch, levels, K, pairs=pairs, local=local)
    audit = AuditingEvaluate(ev)
    with eager(bk):
        R0 = crepe_init(cfg, I, (2,), bk=bk, evaluate=audit, schedule=sch, dtype=xp.float64, key=bk.key(seed))
        init_rows = audit.rows
        samples, R = crepe(cfg, I, (2,), n_blocks, lambda r: r.x, bk=bk, evaluate=audit, schedule=sch,
                           dtype=xp.float64, key=bk.key(seed))
    M = cfg.M
    assert init_rows == I * (1 + M * K)                                   # §17 crepe_init
    per_block = 0
    for n in range(cfg.pairs.period):
        P_n = sum(len(r) for r in cfg.pairs.rounds(M + 1, n))
        R_n = int(any(M in p for r in cfg.pairs.rounds(M + 1, n + 1) for p in r))
        per_block += I * (2 * K * P_n + len(cfg.local_levels) + R_n)
    assert audit.rows == 2 * init_rows + n_blocks * per_block              # X4 (crepe re-runs its own init)
    t_levels = sch.time(xp.asarray(levels, dtype=xp.float64))
    for rep in (R0, R):                                                    # X3: replica invariant
        assert bool(xp.all(rep.t == xp.broadcast_to(t_levels[None, :], (I, M + 1))))
        for f in ALL:
            assert (getattr(rep, f.value) is not None) == (f in cfg.replica_needs)
    assert tuple(samples.shape) == (n_blocks, cfg.pairs.period, I, 1, 2)
