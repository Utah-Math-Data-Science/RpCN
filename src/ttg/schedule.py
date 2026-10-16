"""Schedules (§5). The schedule owns all edge cases: m_t > 0 and s_t > 0 on time([0, 1])."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .kernels import Gaussian
from .ops import namespace


class Schedule(Protocol):
    def time(self, u): ...           # u in [0, 1] -> native t; strictly increasing
    def mean(self, t): ...           # m_t > 0 on time([0, 1])
    def std(self, t): ...            # s_t > 0 on time([0, 1])   (so u = 0 is NOT s = 0)
    def dt_mean(self, t): ...        # d m_t / dt
    def dt_std(self, t): ...         # d s_t / dt
    def prior(self, lead, event, *, xp, dtype, device=None) -> Gaussian: ...
    def log_snr(self, t): ...        # lsnr_t = log(m_t^2 / s_t^2)
    def log_snr_diff(self, s, t): ...   # lsnr_s - lsnr_t


class ScheduleDefaults:
    """Mixin: the two optional methods."""

    def log_snr(self, t):
        xp = namespace(t)
        return 2 * (xp.log(self.mean(t)) - xp.log(self.std(t)))

    def log_snr_diff(self, s, t):    # accurate only to ~eps * |lsnr|; override for tiny gaps
        return self.log_snr(s) - self.log_snr(t)


@dataclass(frozen=True)
class VPLinear(ScheduleDefaults):    # beta(t) = b0 + t (b1 - b0),  alpha_bar = exp(-int beta)
    b0: float = 0.1
    b1: float = 20.0
    t_min: float = 1e-3
    t_max: float = 1.0

    def time(self, u):
        return self.t_min + u * (self.t_max - self.t_min)

    def _log_ab(self, t):
        return -(self.b0 * t + 0.5 * (self.b1 - self.b0) * t**2)

    def _beta(self, t):
        return self.b0 + t * (self.b1 - self.b0)

    def mean(self, t):
        return namespace(t).exp(0.5 * self._log_ab(t))

    def std(self, t):
        xp = namespace(t)
        return xp.sqrt(-xp.expm1(self._log_ab(t)))

    def dt_mean(self, t):
        return -0.5 * self._beta(t) * self.mean(t)

    def dt_std(self, t):
        return 0.5 * self._beta(t) * namespace(t).exp(self._log_ab(t)) / self.std(t)

    def prior(self, lead, event, *, xp, dtype, device=None):
        return Gaussian(xp.zeros((*lead, *event), dtype=dtype, device=device),
                        xp.ones(tuple(lead), dtype=dtype, device=device))

    def log_snr(self, t):
        xp = namespace(t)
        return self._log_ab(t) - xp.log(-xp.expm1(self._log_ab(t)))

    def log_snr_diff(self, s, t):
        xp = namespace(s, t)
        d = -(s - t) * (self.b0 + 0.5 * (self.b1 - self.b0) * (s + t))   # log ab_s - log ab_t, via s - t
        a_t = self._log_ab(t)
        return d - xp.log1p(xp.exp(a_t) * xp.expm1(d) / xp.expm1(a_t))  # minus log((1 - ab_s) / (1 - ab_t))


@dataclass(frozen=True)
class EDM(ScheduleDefaults):         # t = sigma; non-uniform spacing lives in time(u)
    s_min: float = 0.002
    s_max: float = 80.0
    rho: float = 7.0

    def time(self, u):
        a, b = self.s_min ** (1 / self.rho), self.s_max ** (1 / self.rho)
        return (a + u * (b - a)) ** self.rho

    def mean(self, t):
        return namespace(t).ones_like(t)

    def std(self, t):
        return t

    def dt_mean(self, t):
        return namespace(t).zeros_like(t)

    def dt_std(self, t):
        return namespace(t).ones_like(t)

    def prior(self, lead, event, *, xp, dtype, device=None):
        return Gaussian(xp.zeros((*lead, *event), dtype=dtype, device=device),
                        xp.full(tuple(lead), self.s_max**2, dtype=dtype, device=device))

    def log_snr(self, t):
        return -2 * namespace(t).log(t)

    def log_snr_diff(self, s, t):
        return -2 * namespace(s, t).log1p((s - t) / t)


@dataclass(frozen=True)
class RectifiedFlow(ScheduleDefaults):  # x_t = (1 - t) x0 + t eps, clipped to [t_min, t_max] inside (0, 1)
    t_min: float = 1e-3
    t_max: float = 0.999

    def time(self, u):
        return self.t_min + u * (self.t_max - self.t_min)

    def mean(self, t):
        return 1 - t

    def std(self, t):
        return t

    def dt_mean(self, t):
        return -namespace(t).ones_like(t)

    def dt_std(self, t):
        return namespace(t).ones_like(t)

    def prior(self, lead, event, *, xp, dtype, device=None):   # ignores the (1 - t_max) x0 part
        return Gaussian(xp.zeros((*lead, *event), dtype=dtype, device=device),
                        xp.full(tuple(lead), self.t_max**2, dtype=dtype, device=device))

    def log_snr(self, t):
        xp = namespace(t)
        return 2 * (xp.log1p(-t) - xp.log(t))

    def log_snr_diff(self, s, t):
        xp = namespace(s, t)
        return 2 * (xp.log1p(-(s - t) / (1 - t)) - xp.log1p((s - t) / t))
