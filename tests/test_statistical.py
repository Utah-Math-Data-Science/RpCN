"""§18.4: seeded statistical tests on the Gaussian toy (slow; `uv run pytest -m slow`).

Every method's estimate of E[x] (the final state, not x0_hat) must match the plain sampler on the same grid,
reweighted by exp(lambda r_hat), within Monte Carlo error across 8 seeds. For CREPE the grid and kernel are its own.
"""

import math

import numpy as np
import pytest
from conftest import BACKENDS, host

from ttg import (PCN, CREPEConfig, Direction, ESSBelow, EulerMaruyama, EvenOdd, Every, Feature, FKDiff, FKMax,
                 FKSum, Guided, Langevin, LangevinMove, Never, Noising, PosteriorStep, ReverseSDE, VPLinear, Zero,
                 constant, crepe, crepe_finish, das_lambda, finish_needs, ladder, level_lambda, pool_samples, smc,
                 tilt)
from ttg.crepe import fine_grid, tail_grid
from ttg.testing import toy_evaluate

pytestmark = pytest.mark.slow

FAST = [bk for bk in BACKENDS if bk.name in ("jax", "torch")]
BK = next((bk for bk in FAST if bk.name == "jax"), FAST[0] if FAST else None)
SEEDS = 8
LAM = 2.0
S = 50


def compiled(f):
    """jit on JAX; plain eager call on PyTorch."""
    if BK.name == "jax":
        import jax

        return jax.jit(f)
    return f


def setup():
    xp, sch = BK.xp, VPLinear()
    mu = xp.asarray([0.5, -0.5], dtype=xp.float64)
    w = xp.asarray([1.0, 0.0], dtype=xp.float64)
    ev = toy_evaluate(mu, 1.0, w, sch)
    u = xp.linspace(1.0, 0.0, S, dtype=xp.float64)
    return xp, sch, ev, u, PosteriorStep(sch)


def softmax_mean(log_w, x):
    """Self-normalised mean of x [I, N, 2] per instance, averaged over instances. NumPy, float64."""
    lw, x = host(log_w), host(x)
    w = np.exp(lw - lw.max(axis=1, keepdims=True))
    w /= w.sum(axis=1, keepdims=True)
    return np.einsum("in,ind->id", w, x).mean(axis=0)


def over_seeds(estimate):
    """estimate(key) -> [2]; returns (mean, standard error) across the seeds."""
    vals = np.stack([np.asarray(estimate(BK.key(1000 + s))) for s in range(SEEDS)])
    return vals.mean(axis=0), vals.std(axis=0, ddof=1) / math.sqrt(SEEDS)


_REF = {}


def reference(lam=LAM):
    """8 seeds x 2^16 plain-sampler paths, self-normalised weights exp(lam * r_hat_{t_min}(x))."""
    if lam not in _REF:
        xp, sch, ev, u, P = setup()

        def run(key):
            out = smc(sch.prior((1, 2**16), (2,), xp=xp, dtype=xp.float64), u, bk=BK, evaluate=ev, schedule=sch,
                      target=P, proposal=P, potential=Zero(), policy=Never(), final_needs={Feature.REWARD}, key=key)
            return lam * out.final.reward, out.final.x

        run = compiled(run)
        _REF[lam] = over_seeds(lambda key: softmax_mean(*run(key)))
    return _REF[lam]


def fk_run(potential, proposal=None, policy=Every(5), move=None, move_at=lambda k: True, I=16, N=256):
    xp, sch, ev, u, P = setup()

    def run(key):
        out = smc(sch.prior((I, N), (2,), xp=xp, dtype=xp.float64), u, bk=BK, evaluate=ev, schedule=sch, target=P,
                  proposal=P if proposal is None else proposal, potential=potential, policy=policy, move=move,
                  move_at=move_at, key=key)
        return out.log_w, out.final.x

    run = compiled(run)
    return over_seeds(lambda key: softmax_mean(*run(key)))


def check(name, est, ref, slack=0.0):
    (m, se), (m_ref, se_ref) = est, ref
    tol = 4 * np.sqrt(se**2 + se_ref**2) + slack
    print(f"{name}: estimate {m}, reference {m_ref}, |diff| {np.abs(m - m_ref)}, tol {tol}")
    return bool(np.all(np.abs(m - m_ref) <= tol))


needs_fast = pytest.mark.skipif(BK is None, reason="§18.4 runs in float64 on JAX or PyTorch")


@needs_fast
@pytest.mark.parametrize("Pot, lam, policy, N", [
    (FKDiff, LAM, Every(5), 256),
    (FKMax, 0.2, Every(5), 256),     # resampled path potentials degenerate unless lambda S |spread r| is small
    (FKSum, LAM, Never(), 4096),     # FKSum only without resampling: plain IS (biased low at N = 256)
], ids=["FKDiff", "FKMax", "FKSum"])
def test_fk_steering(Pot, lam, policy, N):
    assert check(f"FK {Pot.__name__}", fk_run(Pot(constant(lam)), policy=policy, N=N), reference(lam))


@needs_fast
def test_das():
    xp, sch, ev, u, P = setup()
    lam = das_lambda(sch.time(u), gamma=0.008, alpha_kl=0.5)
    est = fk_run(FKDiff(lam), proposal=Guided(P, tilt(lam)), policy=ESSBelow(0.5))
    assert check("DAS", est, reference())


@needs_fast
def test_fk_with_pcn_moves():
    est = fk_run(FKDiff(constant(LAM)), move=PCN(n_steps=2, step=constant(0.3)), move_at=lambda k: k % 5 == 0)
    assert check("FK + pCN", est, reference())


@needs_fast
def test_rpcn():
    xp, sch, ev, u, P = setup()
    K = 3
    betas = xp.linspace(0.0, LAM, K + 1, dtype=xp.float64)
    steps = xp.full((K + 1,), 0.95, dtype=xp.float64)      # small steps accept ~100% and don't mix

    def run(key):
        out = smc(sch.prior((2**12, K + 1), (2,), xp=xp, dtype=xp.float64), u, bk=BK, evaluate=ev, schedule=sch,
                  target=P, proposal=P, potential=FKDiff(ladder(betas)), policy=Never(),
                  move=PCN(n_steps=20, step=ladder(steps), pairs=EvenOdd()), key=key)
        return out.final.x[:, K, ...]

    run = compiled(run)
    est = over_seeds(lambda key: host(run(key)).mean(axis=0))
    assert check("RpCN", est, reference(), slack=0.02)          # exact only as mixing becomes complete


def crepe_cfg(u_levels=(0.0, 0.25, 0.5, 0.75, 1.0)):
    xp, sch, ev, u, P = setup()
    lam = level_lambda(sch.time(xp.asarray(u_levels, dtype=xp.float64)), beta=LAM, rho=1.0)
    noise = Noising(sch)
    B_P = EulerMaruyama(ReverseSDE(sch), Direction.BACKWARD)
    cfg = CREPEConfig(u_levels=u_levels, K=10, F_P=noise, F_Q=noise, B_P=B_P, B_Q=Guided(B_P, tilt(lam)), lam=lam,
                      pairs=EvenOdd(),
                      local_move=LangevinMove(Guided(Langevin(h=lambda t: 0.01 * sch.std(t) ** 2), tilt(lam, 0.5))),
                      local_levels=tuple(range(len(u_levels) - 1)))
    return cfg


def crepe_estimate():
    xp, sch, ev, u, P = setup()
    cfg = crepe_cfg()

    def run(key):
        samples, _ = crepe(cfg, 2**10, (2,), 300, lambda r: r.x, bk=BK, evaluate=ev, schedule=sch, burn_in=100,
                           dtype=xp.float64, key=key)
        return samples

    run = compiled(run)
    return over_seeds(lambda key: host(run(key)).reshape(-1, 2).mean(axis=0))


def crepe_grid(cfg, xp):
    """CREPE's own grid: the fine grids of every gap concatenated, u_M down to u_0, [M K + 1]."""
    gaps = [xp.flip(fine_grid(cfg, m, xp=xp, dtype=xp.float64), axis=0) for m in reversed(range(cfg.M))]
    return xp.concat([gaps[0]] + [g[1:] for g in gaps[1:]])


@needs_fast
def test_crepe():
    """The reference is the plain sampler with cfg.B_P on CREPE's own grid (the fine grids, level M down to 0),
    reweighted by exp(lambda_0 r_hat) at level 0."""
    xp, sch, ev, u, P = setup()
    cfg = crepe_cfg()
    grid = crepe_grid(cfg, xp)
    lam0 = cfg.lam(sch.time(xp.asarray(cfg.u_levels[0], dtype=xp.float64)))

    def run(key):
        out = smc(sch.prior((1, 2**16), (2,), xp=xp, dtype=xp.float64), grid, bk=BK, evaluate=ev, schedule=sch,
                  target=cfg.B_P, proposal=cfg.B_P, potential=Zero(), policy=Never(),
                  final_needs={Feature.REWARD}, key=key)
        return lam0 * out.final.reward, out.final.x

    run = compiled(run)
    assert check("CREPE", crepe_estimate(), over_seeds(lambda key: softmax_mean(*run(key))))


@needs_fast
def test_crepe_finish():
    """u_0 = 0.25, thin=4, burn_in=100, then crepe_finish(exact=True) from the 300 emits per chain to u = 0.
    The reference is the plain sampler with cfg.B_P on CREPE's grid and then the tail grid, reweighted by
    exp(2 r_hat) at u = 0."""
    xp, sch, ev, u, P = setup()
    cfg = crepe_cfg((0.25, 0.5, 0.75, 1.0))

    def est(key):
        k_c, k_f = BK.split(key, 2)
        samples, _ = crepe(cfg, 2**10, (2,), 300, lambda r: r.project(finish_needs(cfg)), bk=BK, evaluate=ev,
                           schedule=sch, dtype=xp.float64, key=k_c, thin=4, burn_in=100)   # L = 4: 300 emits
        out = crepe_finish(pool_samples(samples), cfg, n_tail=10, exact=True, bk=BK, evaluate=ev,
                           schedule=sch, key=k_f)
        return out.log_w, out.final.x

    est = compiled(est)
    tail = tail_grid(cfg, 10, xp=xp, dtype=xp.float64)
    grid = xp.concat([crepe_grid(cfg, xp), tail[1:]])          # u_M down to u_0, then down to 0

    def ref(key):
        out = smc(sch.prior((1, 2**16), (2,), xp=xp, dtype=xp.float64), grid, bk=BK, evaluate=ev, schedule=sch,
                  target=cfg.B_P, proposal=cfg.B_P, potential=Zero(), policy=Never(),
                  final_needs={Feature.REWARD}, key=key)
        return LAM * out.final.reward, out.final.x

    ref = compiled(ref)
    assert check("CREPE + finish", over_seeds(lambda key: softmax_mean(*est(key))),
                 over_seeds(lambda key: softmax_mean(*ref(key))))


@needs_fast
def test_sanity_untilted_estimate_is_detected():
    """lambda = 0 against the lambda = 2 reference must FAIL the same tolerance: the test sees a missing tilt."""
    assert not check("FK lambda=0", fk_run(FKDiff(constant(0.0))), reference())
