"""Schedules (§5). The schedule owns all edge cases: m_t > 0 and s_t > 0 on time([0, 1])."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

from .kernels import Gaussian
from .ops import namespace


class Schedule(Protocol):
    """A noise schedule ``x_t = m_t x0 + s_t eps``, in its native time ``t``.

    The schedule coordinate ``u`` in ``[0, 1]`` (0 = data end, 1 = noise end) maps to ``t`` through ``time``.
    On ``time([0, 1])``, ``m_t > 0`` and ``s_t > 0``. All methods compute elementwise in the dtype of their input.
    """

    def time(self, u):               # u in [0, 1] -> native t; strictly increasing
        """Map the schedule coordinate to native time.

        Parameters
        ----------
        u : array
            Values in ``[0, 1]``, any shape.

        Returns
        -------
        array
            Native times ``t``, same shape; strictly increasing in ``u``.
        """
        ...

    def mean(self, t):               # m_t > 0 on time([0, 1])
        """Signal scale ``m_t`` (> 0)."""
        ...

    def std(self, t):                # s_t > 0 on time([0, 1])   (so u = 0 is NOT s = 0)
        """Noise scale ``s_t`` (> 0, also at ``u = 0``)."""
        ...

    def dt_mean(self, t):            # d m_t / dt
        """Time derivative ``d m_t / dt``."""
        ...

    def dt_std(self, t):             # d s_t / dt
        """Time derivative ``d s_t / dt``."""
        ...

    def prior(self, leading_shape, event_shape, *, xp, dtype, device=None) -> Gaussian:
        """Return the Gaussian at the noise end, ``u = 1``.

        Parameters
        ----------
        leading_shape : tuple of int
            Leading (row) shape, e.g. ``(I, N)``.
        event_shape : tuple of int
            Event shape ``D``.
        xp : namespace
            Array namespace to create the arrays with.
        dtype : dtype
            Floating dtype of the run.
        device : device, optional
            Device to create the arrays on.

        Returns
        -------
        Gaussian
            Mean ``[*leading_shape, *event_shape]`` (zeros), variance ``[*leading_shape]``.
        """
        ...

    def log_snr(self, t):            # lsnr_t = log(m_t^2 / s_t^2)
        """Log signal-to-noise ratio ``log(m_t^2 / s_t^2)``."""
        ...

    def log_snr_diff(self, t_a, t_b):    # lsnr(t_a) - lsnr(t_b)
        """``log_snr(t_a) - log_snr(t_b)``, accurate for small gaps ``t_a - t_b``.

        Parameters
        ----------
        t_a, t_b : array
            Same shape.

        Returns
        -------
        array
            Same shape.
        """
        ...


class ScheduleDefaults:
    """Mixin: default ``log_snr`` and ``log_snr_diff`` from ``mean`` and ``std``."""

    def log_snr(self, t):
        """``2 (log m_t - log s_t)``. See ``Schedule.log_snr``."""
        xp = namespace(t)
        return 2 * (xp.log(self.mean(t)) - xp.log(self.std(t)))

    def log_snr_diff(self, t_a, t_b):   # accurate only to ~eps * |lsnr|; override for tiny gaps
        """``log_snr(t_a) - log_snr(t_b)`` by direct subtraction.

        Notes
        -----
        Accurate only to about ``eps * |log_snr|``; schedules override it to stay relatively accurate for tiny
        gaps.
        """
        return self.log_snr(t_a) - self.log_snr(t_b)


class Beta(Protocol):
    """A noise rate ``beta(t) >= 0`` for a ``VP`` schedule, with ``alpha_bar_t = exp(-int_0^t beta)``.

    All methods compute elementwise in the dtype of their input.
    """

    def rate(self, t):               # beta(t)
        """Noise rate ``beta(t)``."""
        ...

    def integral(self, t):           # int_0^t beta = -log alpha_bar_t
        """Cumulative rate ``int_0^t beta`` (``= -log alpha_bar_t``)."""
        ...

    def integral_diff(self, t_a, t_b):   # integral(t_a) - integral(t_b)
        """``integral(t_a) - integral(t_b)``, accurate for small gaps ``t_a - t_b``.

        Parameters
        ----------
        t_a, t_b : array
            Same shape.

        Returns
        -------
        array
            Same shape.
        """
        ...


class BetaDefaults:
    """Mixin: default ``integral_diff`` by direct subtraction."""

    def integral_diff(self, t_a, t_b):   # accurate only to ~eps * |integral|; override for tiny gaps
        """``integral(t_a) - integral(t_b)`` by direct subtraction.

        Notes
        -----
        Accurate only to about ``eps * |integral|``; override it to stay relatively accurate for tiny gaps.
        """
        return self.integral(t_a) - self.integral(t_b)


@dataclass(frozen=True)
class LinearBeta(BetaDefaults):      # beta(t) = beta_0 + t (beta_1 - beta_0)
    """Linear noise rate ``beta(t) = beta_0 + t (beta_1 - beta_0)`` (Ho et al., 2020; Song et al., 2021).

    Parameters
    ----------
    beta_0 : float, default 0.1
        ``beta`` at ``t = 0``.
    beta_1 : float, default 20.0
        ``beta`` at ``t = 1``.
    """

    beta_0: float = 0.1
    beta_1: float = 20.0

    def rate(self, t):
        """``beta_0 + t (beta_1 - beta_0)``. See ``Beta.rate``."""
        return self.beta_0 + t * (self.beta_1 - self.beta_0)

    def integral(self, t):
        """``beta_0 t + (beta_1 - beta_0) t^2 / 2``. See ``Beta.integral``."""
        return self.beta_0 * t + 0.5 * (self.beta_1 - self.beta_0) * t**2

    def integral_diff(self, t_a, t_b):
        """``(t_a - t_b) (beta_0 + (beta_1 - beta_0) (t_a + t_b) / 2)``. See ``Beta.integral_diff``."""
        return (t_a - t_b) * (self.beta_0 + 0.5 * (self.beta_1 - self.beta_0) * (t_a + t_b))


@dataclass(frozen=True)
class CosineBeta(BetaDefaults):      # alpha_bar_t = cos^2(theta_t) / cos^2(theta_0), theta_t = pi/2 (t + o) / (1 + o)
    """Cosine schedule of Nichol and Dhariwal (2021): ``alpha_bar_t = cos^2(theta_t) / cos^2(theta_0)``.

    ``theta_t = (pi / 2) (t + offset) / (1 + offset)``. ``alpha_bar_1 = 0``, so use it with ``t_max < 1``
    (``VPCosine`` defaults to 0.999).

    Parameters
    ----------
    offset : float, default 0.008
        Keeps ``beta`` finite near ``t = 0``.
    """

    offset: float = 0.008

    def _tan_theta(self, t):         # sin(theta) / sin(pi/2 - theta): both sines are well conditioned on [0, 1)
        xp = namespace(t)
        theta = 0.5 * math.pi * (t + self.offset) / (1 + self.offset)
        phi = 0.5 * math.pi * (1 - t) / (1 + self.offset)            # pi/2 - theta, accurate near t = 1
        return xp.sin(theta) / xp.sin(phi)

    def rate(self, t):
        """``2 tan(theta_t) dtheta/dt``. See ``Beta.rate``."""
        return math.pi / (1 + self.offset) * self._tan_theta(t)

    def integral(self, t):
        """``-2 log(cos(theta_t) / cos(theta_0))``, as ``integral_diff(t, 0)``. See ``Beta.integral``."""
        return self.integral_diff(t, namespace(t).zeros_like(t))   # two log cos would cancel near t = 0

    def integral_diff(self, t_a, t_b):
        """``-2 log(cos(theta_a) / cos(theta_b))``, computed through ``t_a - t_b``. See ``Beta.integral_diff``."""
        xp = namespace(t_a, t_b)
        delta = 0.5 * math.pi * (t_a - t_b) / (1 + self.offset)   # theta_a - theta_b
        # cos(theta_b + delta) / cos(theta_b) = 1 - 2 sin^2(delta / 2) - tan(theta_b) sin(delta)
        return -2 * xp.log1p(-2 * xp.sin(0.5 * delta) ** 2 - self._tan_theta(t_b) * xp.sin(delta))


@dataclass(frozen=True)
class VP(ScheduleDefaults):          # alpha_bar = exp(-int_0^t beta), for any noise rate beta
    """Variance-preserving schedule ``m_t = sqrt(alpha_bar_t)``, ``s_t = sqrt(1 - alpha_bar_t)``.

    ``alpha_bar_t = exp(-int_0^t beta)`` for the noise rate ``beta``; ``time`` is linear from ``t_min`` to
    ``t_max``. Prior: ``N(0, 1)``. ``log_snr_diff`` is as accurate as ``beta.integral_diff``.

    Parameters
    ----------
    beta : Beta
        Noise rate, e.g. ``LinearBeta()`` or ``CosineBeta()``.
    t_min : float, default 1e-3
        ``time(0)``; keeps ``s_t > 0``.
    t_max : float, default 1.0
        ``time(1)``; must keep ``alpha_bar_t > 0``.
    """

    beta: Any
    t_min: float = 1e-3
    t_max: float = 1.0

    def time(self, u):
        """``t_min + u (t_max - t_min)``. See ``Schedule.time``."""
        return self.t_min + u * (self.t_max - self.t_min)

    def mean(self, t):
        """``sqrt(alpha_bar_t)``. See ``Schedule.mean``."""
        return namespace(t).exp(-0.5 * self.beta.integral(t))

    def std(self, t):
        """``sqrt(1 - alpha_bar_t)``. See ``Schedule.std``."""
        xp = namespace(t)
        return xp.sqrt(-xp.expm1(-self.beta.integral(t)))

    def dt_mean(self, t):
        """``-beta(t) m_t / 2``. See ``Schedule.dt_mean``."""
        return -0.5 * self.beta.rate(t) * self.mean(t)

    def dt_std(self, t):
        """``beta(t) alpha_bar_t / (2 s_t)``. See ``Schedule.dt_std``."""
        return 0.5 * self.beta.rate(t) * namespace(t).exp(-self.beta.integral(t)) / self.std(t)

    def prior(self, leading_shape, event_shape, *, xp, dtype, device=None):
        """Return the standard normal ``N(0, 1)``. See ``Schedule.prior``."""
        return Gaussian(xp.zeros((*leading_shape, *event_shape), dtype=dtype, device=device),
                        xp.ones(tuple(leading_shape), dtype=dtype, device=device))

    def log_snr(self, t):
        """``log(alpha_bar_t / (1 - alpha_bar_t))``. See ``Schedule.log_snr``."""
        xp = namespace(t)
        log_alpha_bar = -self.beta.integral(t)
        return log_alpha_bar - xp.log(-xp.expm1(log_alpha_bar))

    def log_snr_diff(self, t_a, t_b):
        """``log_snr(t_a) - log_snr(t_b)``, computed through ``beta.integral_diff``. See ``Schedule.log_snr_diff``."""
        xp = namespace(t_a, t_b)
        log_alpha_bar_diff = -self.beta.integral_diff(t_a, t_b)     # log ab_a - log ab_b
        log_alpha_bar_b = -self.beta.integral(t_b)
        return log_alpha_bar_diff - xp.log1p(xp.exp(log_alpha_bar_b) * xp.expm1(log_alpha_bar_diff)
                                             / xp.expm1(log_alpha_bar_b))   # minus log((1 - ab_a) / (1 - ab_b))


def VPLinear(beta_0=0.1, beta_1=20.0, t_min=1e-3, t_max=1.0):   # noqa: N802  (a schedule constructor)
    """``VP(LinearBeta(beta_0, beta_1), t_min, t_max)``: the linear-``beta`` VP schedule.

    Parameters
    ----------
    beta_0, beta_1 : float
        See ``LinearBeta``.
    t_min, t_max : float
        See ``VP``.

    Returns
    -------
    VP
    """
    return VP(LinearBeta(beta_0, beta_1), t_min, t_max)


def VPCosine(offset=0.008, t_min=1e-3, t_max=0.999):   # noqa: N802  (a schedule constructor)
    """``VP(CosineBeta(offset), t_min, t_max)``: the cosine VP schedule, stopped short of ``alpha_bar_1 = 0``.

    Parameters
    ----------
    offset : float
        See ``CosineBeta``.
    t_min, t_max : float
        See ``VP``.

    Returns
    -------
    VP
    """
    return VP(CosineBeta(offset), t_min, t_max)


@dataclass(frozen=True)
class EDM(ScheduleDefaults):         # t = sigma; non-uniform spacing lives in time(u)
    """Variance-exploding EDM schedule: ``t = sigma``, ``m_t = 1``, ``s_t = t``.

    ``time(u) = (s_min^(1/rho) + u (s_max^(1/rho) - s_min^(1/rho)))^rho``, so a uniform ``u`` grid gives the
    EDM step spacing. Prior: ``N(0, s_max^2)``.

    Parameters
    ----------
    s_min : float, default 0.002
        ``time(0)``.
    s_max : float, default 80.0
        ``time(1)``.
    rho : float, default 7.0
        Spacing exponent.
    """

    s_min: float = 0.002
    s_max: float = 80.0
    rho: float = 7.0

    def time(self, u):
        """EDM ``rho`` spacing from ``s_min`` to ``s_max``. See ``Schedule.time``."""
        root_min, root_max = self.s_min ** (1 / self.rho), self.s_max ** (1 / self.rho)
        return (root_min + u * (root_max - root_min)) ** self.rho

    def mean(self, t):
        """``1``. See ``Schedule.mean``."""
        return namespace(t).ones_like(t)

    def std(self, t):
        """``t``. See ``Schedule.std``."""
        return t

    def dt_mean(self, t):
        """``0``. See ``Schedule.dt_mean``."""
        return namespace(t).zeros_like(t)

    def dt_std(self, t):
        """``1``. See ``Schedule.dt_std``."""
        return namespace(t).ones_like(t)

    def prior(self, leading_shape, event_shape, *, xp, dtype, device=None):
        """``N(0, s_max^2)``. See ``Schedule.prior``."""
        return Gaussian(xp.zeros((*leading_shape, *event_shape), dtype=dtype, device=device),
                        xp.full(tuple(leading_shape), self.s_max**2, dtype=dtype, device=device))

    def log_snr(self, t):
        """``-2 log t``. See ``Schedule.log_snr``."""
        return -2 * namespace(t).log(t)

    def log_snr_diff(self, t_a, t_b):
        """``-2 log1p((t_a - t_b) / t_b)``. See ``Schedule.log_snr_diff``."""
        return -2 * namespace(t_a, t_b).log1p((t_a - t_b) / t_b)


@dataclass(frozen=True)
class RectifiedFlow(ScheduleDefaults):  # x_t = (1 - t) x0 + t eps, clipped to [t_min, t_max] inside (0, 1)
    """Rectified-flow schedule ``x_t = (1 - t) x0 + t eps``, with ``t`` in ``[t_min, t_max]`` inside ``(0, 1)``.

    Prior: ``N(0, t_max^2)``, which ignores the ``(1 - t_max) x0`` part of ``x_{t_max}``.

    Parameters
    ----------
    t_min : float, default 1e-3
        ``time(0)``.
    t_max : float, default 0.999
        ``time(1)``.
    """

    t_min: float = 1e-3
    t_max: float = 0.999

    def time(self, u):
        """``t_min + u (t_max - t_min)``. See ``Schedule.time``."""
        return self.t_min + u * (self.t_max - self.t_min)

    def mean(self, t):
        """``1 - t``. See ``Schedule.mean``."""
        return 1 - t

    def std(self, t):
        """``t``. See ``Schedule.std``."""
        return t

    def dt_mean(self, t):
        """``-1``. See ``Schedule.dt_mean``."""
        return -namespace(t).ones_like(t)

    def dt_std(self, t):
        """``1``. See ``Schedule.dt_std``."""
        return namespace(t).ones_like(t)

    def prior(self, leading_shape, event_shape, *, xp, dtype, device=None):   # ignores the (1 - t_max) x0 part
        """``N(0, t_max^2)``. See ``Schedule.prior``."""
        return Gaussian(xp.zeros((*leading_shape, *event_shape), dtype=dtype, device=device),
                        xp.full(tuple(leading_shape), self.t_max**2, dtype=dtype, device=device))

    def log_snr(self, t):
        """``2 (log(1 - t) - log t)``. See ``Schedule.log_snr``."""
        xp = namespace(t)
        return 2 * (xp.log1p(-t) - xp.log(t))

    def log_snr_diff(self, t_a, t_b):
        """``log_snr(t_a) - log_snr(t_b)``, computed through ``t_a - t_b``. See ``Schedule.log_snr_diff``."""
        xp = namespace(t_a, t_b)
        return 2 * (xp.log1p(-(t_a - t_b) / (1 - t_b)) - xp.log1p((t_a - t_b) / t_b))
