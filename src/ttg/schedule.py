"""Schedules (§5). The schedule owns all edge cases: m_t > 0 and s_t > 0 on time([0, 1])."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

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

    def prior(self, lead, event, *, xp, dtype, device=None) -> Gaussian:
        """Return the Gaussian at the noise end, ``u = 1``.

        Parameters
        ----------
        lead : tuple of int
            Leading (row) shape, e.g. ``(I, N)``.
        event : tuple of int
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
            Mean ``[*lead, *event]`` (zeros), variance ``[*lead]``.
        """
        ...

    def log_snr(self, t):            # lsnr_t = log(m_t^2 / s_t^2)
        """Log signal-to-noise ratio ``log(m_t^2 / s_t^2)``."""
        ...

    def log_snr_diff(self, s, t):    # lsnr_s - lsnr_t
        """``log_snr(s) - log_snr(t)``, accurate for small gaps ``s - t``.

        Parameters
        ----------
        s, t : array
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

    def log_snr_diff(self, s, t):    # accurate only to ~eps * |lsnr|; override for tiny gaps
        """``log_snr(s) - log_snr(t)`` by direct subtraction.

        Notes
        -----
        Accurate only to about ``eps * |log_snr|``; schedules override it to stay relatively accurate for tiny
        gaps.
        """
        return self.log_snr(s) - self.log_snr(t)


@dataclass(frozen=True)
class VPLinear(ScheduleDefaults):    # beta(t) = b0 + t (b1 - b0),  alpha_bar = exp(-int beta)
    """Variance-preserving schedule with linear ``beta(t) = b0 + t (b1 - b0)``.

    ``m_t = sqrt(alpha_bar_t)``, ``s_t = sqrt(1 - alpha_bar_t)`` with ``alpha_bar_t = exp(-int_0^t beta)``;
    ``time`` is linear from ``t_min`` to ``t_max``. Prior: ``N(0, 1)``.

    Parameters
    ----------
    b0 : float, default 0.1
        ``beta`` at ``t = 0``.
    b1 : float, default 20.0
        ``beta`` at ``t = 1``.
    t_min : float, default 1e-3
        ``time(0)``; keeps ``s_t > 0``.
    t_max : float, default 1.0
        ``time(1)``.
    """

    b0: float = 0.1
    b1: float = 20.0
    t_min: float = 1e-3
    t_max: float = 1.0

    def time(self, u):
        """``t_min + u (t_max - t_min)``. See ``Schedule.time``."""
        return self.t_min + u * (self.t_max - self.t_min)

    def _log_ab(self, t):
        return -(self.b0 * t + 0.5 * (self.b1 - self.b0) * t**2)

    def _beta(self, t):
        return self.b0 + t * (self.b1 - self.b0)

    def mean(self, t):
        """``sqrt(alpha_bar_t)``. See ``Schedule.mean``."""
        return namespace(t).exp(0.5 * self._log_ab(t))

    def std(self, t):
        """``sqrt(1 - alpha_bar_t)``. See ``Schedule.std``."""
        xp = namespace(t)
        return xp.sqrt(-xp.expm1(self._log_ab(t)))

    def dt_mean(self, t):
        """``-beta(t) m_t / 2``. See ``Schedule.dt_mean``."""
        return -0.5 * self._beta(t) * self.mean(t)

    def dt_std(self, t):
        """``beta(t) alpha_bar_t / (2 s_t)``. See ``Schedule.dt_std``."""
        return 0.5 * self._beta(t) * namespace(t).exp(self._log_ab(t)) / self.std(t)

    def prior(self, lead, event, *, xp, dtype, device=None):
        """Return the standard normal ``N(0, 1)``. See ``Schedule.prior``."""
        return Gaussian(xp.zeros((*lead, *event), dtype=dtype, device=device),
                        xp.ones(tuple(lead), dtype=dtype, device=device))

    def log_snr(self, t):
        """``log(alpha_bar_t / (1 - alpha_bar_t))``. See ``Schedule.log_snr``."""
        xp = namespace(t)
        return self._log_ab(t) - xp.log(-xp.expm1(self._log_ab(t)))

    def log_snr_diff(self, s, t):
        """``log_snr(s) - log_snr(t)``, computed through ``s - t``. See ``Schedule.log_snr_diff``."""
        xp = namespace(s, t)
        d = -(s - t) * (self.b0 + 0.5 * (self.b1 - self.b0) * (s + t))   # log ab_s - log ab_t, via s - t
        a_t = self._log_ab(t)
        return d - xp.log1p(xp.exp(a_t) * xp.expm1(d) / xp.expm1(a_t))  # minus log((1 - ab_s) / (1 - ab_t))


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
        a, b = self.s_min ** (1 / self.rho), self.s_max ** (1 / self.rho)
        return (a + u * (b - a)) ** self.rho

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

    def prior(self, lead, event, *, xp, dtype, device=None):
        """``N(0, s_max^2)``. See ``Schedule.prior``."""
        return Gaussian(xp.zeros((*lead, *event), dtype=dtype, device=device),
                        xp.full(tuple(lead), self.s_max**2, dtype=dtype, device=device))

    def log_snr(self, t):
        """``-2 log t``. See ``Schedule.log_snr``."""
        return -2 * namespace(t).log(t)

    def log_snr_diff(self, s, t):
        """``-2 log1p((s - t) / t)``. See ``Schedule.log_snr_diff``."""
        return -2 * namespace(s, t).log1p((s - t) / t)


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

    def prior(self, lead, event, *, xp, dtype, device=None):   # ignores the (1 - t_max) x0 part
        """``N(0, t_max^2)``. See ``Schedule.prior``."""
        return Gaussian(xp.zeros((*lead, *event), dtype=dtype, device=device),
                        xp.full(tuple(lead), self.t_max**2, dtype=dtype, device=device))

    def log_snr(self, t):
        """``2 (log(1 - t) - log t)``. See ``Schedule.log_snr``."""
        xp = namespace(t)
        return 2 * (xp.log1p(-t) - xp.log(t))

    def log_snr_diff(self, s, t):
        """``log_snr(s) - log_snr(t)``, computed through ``s - t``. See ``Schedule.log_snr_diff``."""
        xp = namespace(s, t)
        return 2 * (xp.log1p(-(s - t) / (1 - t)) - xp.log1p((s - t) / t))
