"""I7: the same step functions run under JAX jit/scan and eagerly, with the same results."""

import numpy as np
import pytest
from conftest import BACKENDS, host
from helpers import toy

from ttg import (PCN, CREPEConfig, Direction, ESSBelow, EulerMaruyama, EvenOdd, Feature, FKMax, Guided, Langevin,
                 LangevinMove, Noising, PosteriorStep, ReverseSDE, VPLinear, constant, crepe, ladder, level_lambda,
                 smc, tilt)

JAX = [backend for backend in BACKENDS if backend.name == "jax"]
pytestmark = pytest.mark.skipif(not JAX, reason="needs JAX")


def both(f, key):
    import jax

    with jax.disable_jit():
        eager = f(key)
    return eager, jax.jit(f)(key)


def test_smc_jit_matches_eager():
    backend = JAX[0]
    xp, sch = backend.xp, VPLinear()
    ev, _, _ = toy(backend, sch)
    P = PosteriorStep(sch)
    betas = xp.linspace(0.0, 2.0, 4, dtype=xp.float64)
    steps = xp.asarray([0.5, 0.4, 0.3, 0.2], dtype=xp.float64)

    def f(key):
        out = smc(sch.prior((2, 4), (2,), xp=xp, dtype=xp.float64), xp.linspace(1.0, 0.0, 7, dtype=xp.float64),
                  backend=backend, evaluate=ev, schedule=sch, target=P, proposal=Guided(P, tilt(constant(0.5))),
                  potential=FKMax(ladder(betas)), policy=ESSBelow(0.5),
                  move=PCN(2, ladder(steps), pairs=EvenOdd()), move_at=lambda k: k % 2 == 0,
                  out_needs={Feature.X0}, key=key)
        return out.log_weights, out.final.x, out.final.x0

    for a, b in zip(*both(f, backend.key(3))):
        assert np.allclose(host(a), host(b), rtol=1e-10, atol=1e-12)


def test_crepe_jit_matches_eager():
    backend = JAX[0]
    xp, sch = backend.xp, VPLinear()
    ev, _, _ = toy(backend, sch)
    levels = (0.0, 0.5, 1.0)
    lam = level_lambda(sch.time(xp.asarray(levels, dtype=xp.float64)), beta=2.0, rho=1.0)
    B_P = EulerMaruyama(ReverseSDE(sch), Direction.BACKWARD)
    cfg = CREPEConfig(u_levels=levels, steps_per_gap=3, model_noising=Noising(sch), proposal_noising=Noising(sch),
                      model_denoising=B_P, proposal_denoising=Guided(B_P, tilt(lam)),
                      lam=lam, local_move=LangevinMove(Guided(Langevin(lambda t: 0.01 * sch.std(t) ** 2),
                                                              tilt(lam, 0.5))), local_levels=(0, 1))

    def f(key):
        samples, R = crepe(cfg, 2, (2,), 3, lambda r: r.x, backend=backend, evaluate=ev, schedule=sch, dtype=xp.float64,
                           key=key)
        return samples, R.x

    for a, b in zip(*both(f, backend.key(5))):
        assert np.allclose(host(a), host(b), rtol=1e-10, atol=1e-12)
