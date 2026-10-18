"""X1-X8: CREPE's path exchange, replica invariant, evaluation counts, continuation, finish and crepe_sample."""

import math

import numpy as np
import pytest
from conftest import backends, eager, elems, host, xps_for
from helpers import ALL, toy
from hypothesis import given
from hypothesis import strategies as st

from ttg import (CREPEConfig, Direction, EulerMaruyama, EvenOdd, Feature, ForwardSDE, Guided, Langevin,
                 LangevinMove, Never, Noising, PartialRecord, PathExchange, Point, ReverseSDE, SequentialSweep,
                 VPLinear, Zero, crepe, crepe_finish, crepe_init, crepe_sample, evaluate_rows, finish_needs, level_lambda,
                 per_row_grid, pool_samples, smc, tail_grid, tilt)
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


def one_sweep_rows(cfg, I, n):
    """§17: evaluated rows of sweep n."""
    M = cfg.M
    P_n = sum(len(r) for r in cfg.pairs.rounds(M + 1, n))
    R_n = int(any(M in p for r in cfg.pairs.rounds(M + 1, n + 1) for p in r))
    return I * (2 * cfg.K * P_n + len(cfg.local_levels) + R_n)


def sweep_rows(cfg, I, L=None):
    """§17: evaluated rows per block of L sweeps (default: one period)."""
    return sum(one_sweep_rows(cfg, I, n) for n in range(cfg.pairs.period if L is None else L))


@backends
@pytest.mark.parametrize("pairs", [EvenOdd(), SequentialSweep()], ids=lambda p: type(p).__name__)
@given(levels=u_levels(max_M=2), I=st.integers(1, 2), n_blocks=st.integers(1, 3), thin=st.integers(1, 3),
       burn_in=st.integers(0, 2), seed=st.integers(0, 2**20))
def test_X5_pytree_emit_thin_burn_in_and_pool_samples(bk, pairs, levels, I, n_blocks, thin, burn_in, seed):
    xp, sch = bk.xp, VPLinear()
    ev, _, _ = toy(bk, sch)
    cfg = make_cfg(bk, sch, levels, 1, pairs=pairs, local=False)
    L = math.lcm(cfg.pairs.period, thin)
    E = L // thin
    audit = AuditingEvaluate(ev)
    seen = []                                                              # rows evaluated at each emit

    def emit(r):
        seen.append(audit.rows)
        return r.project(finish_needs(cfg)), r.x

    with eager(bk):
        samples, _ = crepe(cfg, I, (2,), n_blocks, emit, bk=bk, evaluate=audit, schedule=sch, dtype=xp.float64,
                           key=bk.key(seed), thin=thin, burn_in=burn_in)
    init = I * (1 + cfg.M * cfg.K)                                         # §17 crepe_init
    total = (burn_in + n_blocks) * L
    assert audit.rows == init + (burn_in + n_blocks) * sweep_rows(cfg, I, L)   # (burn_in + n_blocks) * L sweeps
    after = [init + sum(one_sweep_rows(cfg, I, n % L) for n in range(j)) for j in range(1, total + 1)]
    # after sweeps burn_in L + thin, burn_in L + 2 thin, ...
    assert seen == [after[j - 1] for j in range(burn_in * L + thin, total + 1, thin)]
    rec, x = samples
    for f in ALL:
        assert (getattr(rec, f.value) is not None) == (f in finish_needs(cfg))
    assert tuple(rec.t.shape) == (n_blocks, E, I, 1) and tuple(rec.x.shape) == (n_blocks, E, I, 1, 2)
    pooled = pool_samples(samples)
    assert tuple(pooled[0].t.shape) == (I, n_blocks * E)
    assert tuple(pooled[0].x.shape) == (I, n_blocks * E, 2)
    xs = host(x)
    flat = [xs[blk, e, :, 0] for blk in range(n_blocks) for e in range(E)]   # emit order
    want = np.stack(flat, axis=1)                                          # [I, n, 2]: row i is chain i only
    assert np.array_equal(host(pooled[1]), want) and np.array_equal(host(pooled[0].x), want)
    with pytest.raises(ValueError):
        crepe(cfg, I, (2,), 1, emit, bk=bk, evaluate=ev, schedule=sch, dtype=xp.float64, key=bk.key(seed), thin=0)
    with pytest.raises(ValueError):
        crepe(cfg, I, (2,), 1, emit, bk=bk, evaluate=ev, schedule=sch, dtype=xp.float64, key=bk.key(seed),
              burn_in=-1)


@backends
@pytest.mark.parametrize("pairs", [EvenOdd(), SequentialSweep()], ids=lambda p: type(p).__name__)
@given(levels=u_levels(max_M=2), K=st.integers(1, 2), I=st.integers(1, 2), n_blocks=st.integers(0, 2),
       seed=st.integers(0, 2**20), local=st.booleans())
def test_X6_continue_from_replicas(bk, pairs, levels, K, I, n_blocks, seed, local):
    xp, sch = bk.xp, VPLinear()
    ev, _, _ = toy(bk, sch)
    cfg = make_cfg(bk, sch, levels, K, pairs=pairs, local=local)
    R0 = replicas(bk, sch, ev, cfg, I, seed, ALL)                          # extras beyond replica_needs
    audit = AuditingEvaluate(ev)
    with eager(bk):
        _, R = crepe(cfg, I, (2,), n_blocks, lambda r: r.x, bk=bk, evaluate=audit, schedule=sch,
                     dtype=xp.float64, key=bk.key(seed + 1), R0=R0)
    assert audit.rows == n_blocks * sweep_rows(cfg, I)                     # no crepe_init rows
    t_levels = sch.time(xp.asarray(levels, dtype=xp.float64))
    assert bool(xp.all(R.t == xp.broadcast_to(t_levels[None, :], (I, cfg.M + 1))))
    for f in ALL:
        assert (getattr(R, f.value) is not None) == (f in cfg.replica_needs)
    if n_blocks == 0:
        want = R0.project(cfg.replica_needs)
        for f in ALL:
            v = getattr(want, f.value)
            assert v is None or bool(xp.all(getattr(R, f.value) == v))
        assert bool(xp.all(R.x == R0.x))
    partial = R0.project(cfg.replica_needs - {Feature.REWARD})
    with pytest.raises(PartialRecord):
        crepe(cfg, I, (2,), 1, lambda r: r.x, bk=bk, evaluate=ev, schedule=sch, dtype=xp.float64,
              key=bk.key(seed), R0=partial)


def finish_start(bk, sch, ev, cfg, I, N, seed):
    xp = bk.xp
    t0 = xp.broadcast_to(sch.time(xp.asarray(cfg.u_levels[0], dtype=xp.float64)), (I, N))
    x = bk.normal(bk.key(seed), (I, N, 2), xp.float64)
    return evaluate_rows(ev, Point(x, t0), finish_needs(cfg), bk)


@backends
@given(n_tail=st.integers(1, 4), I=st.integers(1, 2), N=st.integers(1, 3), seed=st.integers(0, 2**20),
       u0=st.floats(0.05, 0.6), data=st.data())
def test_X7_crepe_finish(bk, n_tail, I, N, seed, u0, data):
    xp, sch = bk.xp, VPLinear()
    ev, _, _ = toy(bk, sch)
    levels = (u0, (u0 + 1.0) / 2, 1.0)
    cfg = make_cfg(bk, sch, levels, 1, local=False)
    with pytest.raises(ValueError):
        crepe_finish(None, make_cfg(bk, sch, (0.0, 1.0), 1, local=False), n_tail, exact=True, bk=bk,
                     evaluate=ev, schedule=sch, key=bk.key(seed))
    grid = tail_grid(cfg, n_tail, xp=xp, dtype=xp.float64)
    assert float(grid[0]) == u0 and float(grid[-1]) == 0.0 and grid.shape[0] == n_tail + 1
    start = finish_start(bk, sch, ev, cfg, I, N, seed)
    with eager(bk):
        out = crepe_finish(start, cfg, n_tail, exact=False, bk=bk, evaluate=ev, schedule=sch, key=bk.key(seed + 1))
        ref = smc(start, grid, bk=bk, evaluate=ev, schedule=sch, target=cfg.B_Q, proposal=cfg.B_Q,
                  potential=Zero(), policy=Never(), final_needs=frozenset({Feature.X0}), key=bk.key(seed + 1))
    assert bool(xp.all(out.log_w == 0))
    assert bool(xp.all(out.final.x == ref.final.x)) and bool(xp.all(out.log_w == ref.log_w))

    # exact=True with Never(), along a replayed tail
    ts = [xp.broadcast_to(t_, (I, N)) for t_ in sch.time(grid)]
    t_steps = xp.stack(ts)
    path = data.draw(xps_for(bk).arrays(xp.float64, (n_tail + 1, I, N, 2), elements=elems()))
    B_Q = cfg.B_Q
    rcfg = make_cfg(bk, sch, levels, 1, local=False, B_Q=ReplayKernel(B_Q, path, t_steps))
    with eager(bk):
        out = crepe_finish(start, rcfg, n_tail, exact=True, policy=Never(), bk=bk, evaluate=ev, schedule=sch,
                           key=bk.key(seed + 2))
    recs = chain(ev, [start.x] + [path[k, ...] for k in range(1, n_tail + 1)], ts, bk)
    want = ell(cfg, recs[-1]) - ell(cfg, recs[0]) + path_terms(recs, ts, cfg.B_P, B_Q)
    assert bool(xp.all(out.final.x == path[-1, ...]))
    assert np.all(np.abs(host(out.log_w - want)) <= 1e-8 * (1 + np.abs(host(want))))


def same_tree(a, b):
    """Identical leaves (and identical None structure) in two Records."""
    for f in ("x", "t", *(f.value for f in ALL)):
        u, v = getattr(a, f), getattr(b, f)
        assert (u is None) == (v is None), f
        assert u is None or np.array_equal(host(u), host(v)), f


@backends
@pytest.mark.parametrize("exact", [True, False])
@given(I=st.integers(1, 2), n_blocks=st.integers(1, 2), thin=st.integers(1, 2), burn_in=st.integers(0, 1),
       n_tail=st.integers(1, 3), u0=st.floats(0.05, 0.5), seed=st.integers(0, 2**20))
def test_X8_crepe_sample_is_its_composition(bk, exact, I, n_blocks, thin, burn_in, n_tail, u0, seed):
    xp, sch = bk.xp, VPLinear()
    ev, _, _ = toy(bk, sch)
    cfg = make_cfg(bk, sch, (u0, (u0 + 1.0) / 2, 1.0), 1, local=False)
    audit = AuditingEvaluate(ev)
    key = bk.key(seed)
    with eager(bk):
        out, R = crepe_sample(cfg, I, (2,), n_blocks, n_tail=n_tail, exact=exact, bk=bk, evaluate=audit,
                              schedule=sch, dtype=xp.float64, key=key, thin=thin, burn_in=burn_in)
        rows = audit.rows
        k_chain, k_tail = bk.split(key, 2)
        samples, R_ref = crepe(cfg, I, (2,), n_blocks, lambda r: r.project(finish_needs(cfg)), bk=bk, evaluate=ev,
                               schedule=sch, dtype=xp.float64, key=k_chain, thin=thin, burn_in=burn_in)
        ref = crepe_finish(pool_samples(samples), cfg, n_tail, exact=exact, bk=bk, evaluate=ev, schedule=sch,
                           final_needs=cfg.out_needs, key=k_tail)
    same_tree(out.final, ref.final.project(cfg.out_needs))
    assert np.array_equal(host(out.log_w), host(ref.log_w))
    same_tree(R, R_ref)
    for f in ALL:
        assert (getattr(out.final, f.value) is not None) == (f in cfg.out_needs)
    L = math.lcm(cfg.pairs.period, thin)
    N = n_blocks * L // thin
    assert tuple(out.log_w.shape) == (I, N)
    init = I * (1 + cfg.M * cfg.K)                                         # §17
    assert rows == init + (burn_in + n_blocks) * sweep_rows(cfg, I, L) + I * N * n_tail


@backends
@given(I=st.integers(1, 2), n_blocks=st.integers(1, 2), thin=st.integers(1, 2), seed=st.integers(0, 2**20),
       exact=st.sampled_from([None, True, False]))
def test_X8_crepe_sample_without_finish(bk, I, n_blocks, thin, seed, exact):
    xp, sch = bk.xp, VPLinear()
    ev, _, _ = toy(bk, sch)
    cfg = make_cfg(bk, sch, (0.0, 0.5, 1.0), 1, local=False)
    audit = AuditingEvaluate(ev)
    key = bk.key(seed)
    with eager(bk):
        out, R = crepe_sample(cfg, I, (2,), n_blocks, exact=exact, bk=bk, evaluate=audit, schedule=sch,
                              dtype=xp.float64, key=key, thin=thin)        # n_tail and exact ignored at u_0 = 0
        rows = audit.rows
        samples, R_ref = crepe(cfg, I, (2,), n_blocks, lambda r: r.project(cfg.out_needs), bk=bk, evaluate=ev,
                               schedule=sch, dtype=xp.float64, key=bk.split(key, 2)[0], thin=thin)
    same_tree(out.final, pool_samples(samples).project(cfg.out_needs))
    assert bool(xp.all(out.log_w == 0)) and out.log_w.shape == out.final.t.shape
    same_tree(R, R_ref)
    for f in ALL:
        assert (getattr(out.final, f.value) is not None) == (f in cfg.out_needs)
    L = math.lcm(cfg.pairs.period, thin)
    assert rows == I * (1 + cfg.M * cfg.K) + n_blocks * sweep_rows(cfg, I, L)   # no finish rows
    kw = dict(bk=bk, evaluate=ev, schedule=sch, dtype=xp.float64, key=key)
    with pytest.raises(ValueError):
        crepe_sample(cfg, I, (2,), 0, **kw)
    fcfg = make_cfg(bk, sch, (0.2, 0.6, 1.0), 1, local=False)
    with pytest.raises(ValueError):
        crepe_sample(fcfg, I, (2,), 1, exact=True, **kw)                   # n_tail missing
    with pytest.raises(ValueError):
        crepe_sample(fcfg, I, (2,), 1, n_tail=2, **kw)                     # exact missing
