"""Gaussians and kernels (§7). Every kernel goes one way only; `at(rec, t_dst)` returns a per-row Gaussian."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from .ops import bcast, device, namespace, prod, pytree
from .records import Feature

Array = Any


@pytree
class Gaussian:                       # isotropic, per row; a pytree, so it can travel in MoveContext
    mean: Array                       # [L..., *D]
    var: Array                        # [L...], must be > 0 (I6)

    def sample(self, key, bk):
        xp = namespace(self.mean, self.var)
        noise = bk.normal(key, self.mean.shape, self.mean.dtype, device=device(self.mean))
        return self.mean + bcast(xp.sqrt(self.var), self.mean) * noise

    def log_prob(self, x):
        xp = namespace(x, self.mean, self.var)
        ev = tuple(range(self.var.ndim, x.ndim))
        d = prod(x.shape[self.var.ndim:])
        return -0.5 * xp.sum((x - self.mean) ** 2, axis=ev) / self.var - 0.5 * d * xp.log(2 * xp.pi * self.var)

    def log_ratio(self, other, x):    # log self(x) - log other(x), without forming either log-density
        xp = namespace(x, self.mean, other.mean)
        ev = tuple(range(self.var.ndim, x.ndim))
        d = prod(x.shape[self.var.ndim:])
        delta = self.mean - other.mean            # exact where the means are close (Sterbenz)
        y = x - other.mean                        # the sampled noise when x ~ other
        dv = self.var - other.var                 # exactly 0 when the variances are equal
        quad = xp.sum(delta * (y - 0.5 * delta), axis=ev) / self.var
        return (quad + 0.5 * xp.sum(y**2, axis=ev) * dv / (self.var * other.var)
                - 0.5 * d * xp.log1p(dv / other.var))


def transition(sch, t, s):
    """t < s: (m_{s|t}, sigma^2_{s|t}), cancellation-free (§2)."""
    xp = namespace(t, s)
    return sch.mean(s) / sch.mean(t), -sch.std(s) ** 2 * xp.expm1(sch.log_snr_diff(s, t))


class Kernel(Protocol):
    requires: frozenset[Feature]
    def at(self, rec, t_dst) -> Gaussian: ...    # t_dst has rec.t's shape [L...]


@dataclass(frozen=True)
class Noising:                    # exact forward kernel F^P
    sch: Any
    requires = frozenset()

    def at(self, rec, t):
        m, var = transition(self.sch, rec.t, t)                     # source rec.t < destination t
        return Gaussian(bcast(m, rec.x) * rec.x, var)


@dataclass(frozen=True)
class PosteriorStep:
    sch: Any
    requires = frozenset({Feature.X0})

    def at(self, rec, t):
        xp = namespace(rec.t, t)
        s = rec.t                                                   # per-row scalars, shape [L...]
        ss, mt, st = self.sch.std(s), self.sch.mean(t), self.sch.std(t)
        m_st = self.sch.mean(s) / mt
        e = -xp.expm1(self.sch.log_snr_diff(s, t))                  # = sigma^2_{s|t} / s_s^2, in (0, 1)
        a, b = m_st * st**2 / ss**2, mt * e
        mean = bcast(a, rec.x) * rec.x + bcast(b, rec.x) * rec.need(Feature.X0)
        return Gaussian(mean, st**2 * e)                            # tau^2 = sigma^2_{s|t} s_t^2 / s_s^2


@dataclass(frozen=True)
class DDIMStep:                   # eta in (0, 1]; eta = 1 equals PosteriorStep (test K4)
    sch: Any
    eta: float
    requires = frozenset({Feature.X0, Feature.EPS})

    def at(self, rec, t):
        xp = namespace(rec.t, t)
        s = rec.t
        e = -xp.expm1(self.sch.log_snr_diff(s, t))                  # sigma^2_{s|t} / s_s^2
        st = self.sch.std(t)
        var = self.eta**2 * st**2 * e                               # eta^2 tau^2
        c = st * xp.sqrt(1 - self.eta**2 * e)                       # sqrt(s_t^2 - eta^2 tau^2); > 0 for eta <= 1
        mean = (bcast(self.sch.mean(t), rec.x) * rec.need(Feature.X0)
                + bcast(c, rec.x) * rec.need(Feature.EPS))
        return Gaussian(mean, var)


@dataclass(frozen=True)
class Langevin:                   # one unadjusted Langevin step at fixed t
    h: Callable                   # step size as a function of t, e.g. lambda t: 0.01 * sch.std(t)**2
    requires = frozenset({Feature.SCORE})

    def at(self, rec, t):         # t == rec.t
        h = self.h(rec.t)
        return Gaussian(rec.x + bcast(h, rec.x) * rec.need(Feature.SCORE), 2 * h)


@dataclass(frozen=True)
class EulerMaruyama:              # one EM step of the injected SDE, integrated up or down
    sde: Any
    direction: Any                # sde.Direction

    @property
    def requires(self):
        return self.sde.requires

    def at(self, rec, t):
        delta = t - rec.t                                           # signed
        sign = float(self.direction.value)                          # static switch: +1 or -1
        mean = rec.x + bcast(delta, rec.x) * self.sde.drift(rec)    # drift at the source
        return Gaussian(mean, self.sde.diffusion2(rec.t) * sign * delta)   # var <= 0 if misused (K8)


@dataclass(frozen=True)
class Guided:                     # base mean + scale(t, g, var) * g, with g at the source
    base: Any
    scale: Callable

    @property
    def requires(self):
        return self.base.requires | {Feature.REWARD_GRAD}

    def at(self, rec, t):
        d = self.base.at(rec, t)
        g = rec.need(Feature.REWARD_GRAD)
        return Gaussian(d.mean + bcast(self.scale(t, g, d.var), g) * g, d.var)
