"""Gaussians and kernels (§7). Every kernel goes one way only; `at(record, t_dst)` returns a per-row Gaussian."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from .ops import bcast, device_of, namespace, prod, pytree
from .records import Feature

Array = Any


@pytree
class Gaussian:                       # isotropic, per row; a pytree, so it can travel in MoveContext
    """Isotropic Gaussian with one mean and one variance per row.

    The leading axes ``[L...]`` index rows (e.g. ``[I, N]``); the trailing axes are the event shape ``D``.
    A pytree, so it can be carried through ``scan`` and ``jit``.

    Parameters
    ----------
    mean : array
        ``[L..., *D]``. Per-row mean.
    var : array
        ``[L...]``. Per-row variance, shared by all event entries; must be > 0.
    """

    mean: Array                       # [L..., *D]
    var: Array                        # [L...], must be > 0 (I6)

    def sample(self, key, backend):
        """Draw one sample per row.

        Parameters
        ----------
        key : backend key
            Random key.
        backend : Backend
            Supplies the normal draws.

        Returns
        -------
        array
            ``[L..., *D]``, in the dtype and on the device of ``mean``.
        """
        xp = namespace(self.mean, self.var)
        noise = backend.normal(key, self.mean.shape, self.mean.dtype, device=device_of(self.mean))
        return self.mean + bcast(xp.sqrt(self.var), self.mean) * noise

    def log_prob(self, x):
        """Log-density of ``x`` under each row's Gaussian.

        Parameters
        ----------
        x : array
            ``[L..., *D]``.

        Returns
        -------
        array
            ``[L...]``, summed over the event axes.

        Notes
        -----
        For a ratio of two densities use ``log_ratio``, which avoids the cancellation of two large log-densities.
        """
        xp = namespace(x, self.mean, self.var)
        event_axes = tuple(range(self.var.ndim, x.ndim))
        event_size = prod(x.shape[self.var.ndim:])
        return (-0.5 * xp.sum((x - self.mean) ** 2, axis=event_axes) / self.var
                - 0.5 * event_size * xp.log(2 * xp.pi * self.var))

    def log_ratio(self, other, x):    # log self(x) - log other(x), without forming either log-density
        """``log self(x) - log other(x)``, computed without forming either log-density.

        Parameters
        ----------
        other : Gaussian
            Same leading and event shapes as ``self``.
        x : array
            ``[L..., *D]``, typically a sample from ``other``.

        Returns
        -------
        array
            ``[L...]``.

        Notes
        -----
        Written in terms of the mean difference and ``x - other.mean``, so it stays accurate in float32 when the
        two means are close and the variances are equal (the variance terms then vanish exactly).
        """
        xp = namespace(x, self.mean, other.mean)
        event_axes = tuple(range(self.var.ndim, x.ndim))
        event_size = prod(x.shape[self.var.ndim:])
        mean_diff = self.mean - other.mean        # exact where the means are close (Sterbenz)
        centered = x - other.mean                 # the sampled noise when x ~ other
        var_diff = self.var - other.var           # exactly 0 when the variances are equal
        quadratic = xp.sum(mean_diff * (centered - 0.5 * mean_diff), axis=event_axes) / self.var
        return (quadratic + 0.5 * xp.sum(centered**2, axis=event_axes) * var_diff / (self.var * other.var)
                - 0.5 * event_size * xp.log1p(var_diff / other.var))


def transition(schedule, t_src, t_dst):
    """Forward transition coefficients from time ``t_src`` to a noisier time ``t_dst``.

    For ``t_src < t_dst``, writing ``t = t_src`` and ``s = t_dst``: ``x_s | x_t ~ N(m_{s|t} x_t, sigma^2_{s|t})``.
    The variance is computed from ``log_snr_diff``, so it is free of cancellation for small gaps.

    Parameters
    ----------
    schedule : Schedule
        The noise schedule.
    t_src : array
        ``[L...]``. Source (less noisy) times.
    t_dst : array
        ``[L...]``. Destination times, ``t_dst > t_src``.

    Returns
    -------
    mean_ratio : array
        ``[L...]``. ``m_{s|t} = m_s / m_t``.
    var : array
        ``[L...]``. ``sigma^2_{s|t} = s_s^2 - m_{s|t}^2 s_t^2``.
    """
    xp = namespace(t_src, t_dst)
    return (schedule.mean(t_dst) / schedule.mean(t_src),
            -schedule.std(t_dst) ** 2 * xp.expm1(schedule.log_snr_diff(t_dst, t_src)))


class Kernel(Protocol):
    """A one-way Markov kernel: given a source record and a destination time, a per-row Gaussian.

    Attributes
    ----------
    requires : frozenset of Feature
        The features ``at`` reads from the source record.
    """

    requires: frozenset[Feature]

    def at(self, record, t_dst) -> Gaussian:    # t_dst has record.t's shape [L...]
        """Return the kernel's distribution for moving from ``record`` to time ``t_dst``.

        Parameters
        ----------
        record : Record
            Source point, ``x`` of shape ``[L..., *D]`` and ``t`` of shape ``[L...]``, carrying ``requires``.
        t_dst : array
            ``[L...]``. Destination times.

        Returns
        -------
        Gaussian
            Mean ``[L..., *D]``, variance ``[L...]``.
        """
        ...


@dataclass(frozen=True)
class Noising:                    # exact forward kernel F^P
    """The exact forward (noising) kernel of the schedule.

    ``at(record, t_dst)`` requires ``record.t < t_dst`` and returns ``N(m_{t|s} x, sigma^2_{t|s})``. Needs no network.

    Parameters
    ----------
    schedule : Schedule
        The noise schedule.
    """

    schedule: Any
    requires = frozenset()

    def at(self, record, t_dst):
        """Noise ``record`` forward to the later time ``t_dst``. See ``Kernel.at``."""
        mean_ratio, var = transition(self.schedule, record.t, t_dst)        # source record.t < destination t_dst
        return Gaussian(bcast(mean_ratio, record.x) * record.x, var)


@dataclass(frozen=True)
class PosteriorStep:
    """Denoising step: the forward process's posterior ``q(x_t | x_s, x0 = x0_hat)``, for ``t < s``.

    Reads ``X0`` (the network's prediction of the clean sample) from the source record.

    Parameters
    ----------
    schedule : Schedule
        The noise schedule.
    """

    schedule: Any
    requires = frozenset({Feature.X0})

    def at(self, record, t_dst):
        """Denoise ``record`` (at time ``t_src = record.t``) to the earlier time ``t_dst``. See ``Kernel.at``."""
        xp = namespace(record.t, t_dst)
        t_src = record.t                                            # per-row scalars, shape [L...]
        std_src, mean_dst, std_dst = self.schedule.std(t_src), self.schedule.mean(t_dst), self.schedule.std(t_dst)
        mean_ratio = self.schedule.mean(t_src) / mean_dst
        var_fraction = -xp.expm1(self.schedule.log_snr_diff(t_src, t_dst))   # = sigma^2_{s|t} / s_s^2, in (0, 1)
        coeff_x, coeff_x0 = mean_ratio * std_dst**2 / std_src**2, mean_dst * var_fraction
        mean = bcast(coeff_x, record.x) * record.x + bcast(coeff_x0, record.x) * record.need(Feature.X0)
        return Gaussian(mean, std_dst**2 * var_fraction)            # tau^2 = sigma^2_{s|t} s_t^2 / s_s^2


@dataclass(frozen=True)
class DDIMStep:                   # eta in (0, 1]; eta = 1 equals PosteriorStep (test K4)
    """DDIM denoising step with stochasticity ``eta``.

    Mean ``m_t x0_hat + sqrt(s_t^2 - eta^2 tau^2) eps_hat`` and variance ``eta^2 tau^2``, where ``tau^2`` is
    the ``PosteriorStep`` variance. Reads ``X0`` and ``EPS`` from the source record.

    Parameters
    ----------
    schedule : Schedule
        The noise schedule.
    eta : float
        In ``(0, 1]``. ``eta = 1`` gives the same distribution as ``PosteriorStep``; ``eta = 0`` (deterministic
        DDIM) has no density and is not supported.
    """

    schedule: Any
    eta: float
    requires = frozenset({Feature.X0, Feature.EPS})

    def at(self, record, t_dst):
        """Denoise ``record`` to the earlier time ``t_dst``. See ``Kernel.at``."""
        xp = namespace(record.t, t_dst)
        t_src = record.t
        var_fraction = -xp.expm1(self.schedule.log_snr_diff(t_src, t_dst))   # sigma^2_{s|t} / s_s^2
        std_dst = self.schedule.std(t_dst)
        var = self.eta**2 * std_dst**2 * var_fraction               # eta^2 tau^2
        eps_coeff = std_dst * xp.sqrt(1 - self.eta**2 * var_fraction)   # sqrt(s_t^2 - eta^2 tau^2); > 0 for eta <= 1
        mean = (bcast(self.schedule.mean(t_dst), record.x) * record.need(Feature.X0)
                + bcast(eps_coeff, record.x) * record.need(Feature.EPS))
        return Gaussian(mean, var)


@dataclass(frozen=True)
class Langevin:                   # one unadjusted Langevin step at fixed t
    """One unadjusted Langevin step at fixed time: ``N(x + h score, 2 h)`` with ``h = step_size(t)``.

    Reads ``SCORE`` from the source record. Add a reward drift with ``Guided(Langevin(step_size), tilt(lam, 0.5))``.

    Parameters
    ----------
    step_size : Callable[[array], array]
        Step size ``h`` as a function of ``t`` (``[L...] -> [L...]``), e.g. ``lambda t: 0.01 * schedule.std(t)**2``.
    """

    step_size: Callable           # h as a function of t, e.g. lambda t: 0.01 * schedule.std(t)**2
    requires = frozenset({Feature.SCORE})

    def at(self, record, t_dst):     # t_dst == record.t
        """Langevin step from ``record``; ``t_dst`` must equal ``record.t``. See ``Kernel.at``."""
        h = self.step_size(record.t)
        return Gaussian(record.x + bcast(h, record.x) * record.need(Feature.SCORE), 2 * h)


@dataclass(frozen=True)
class EulerMaruyama:              # one EM step of the injected SDE, integrated up or down
    """One Euler-Maruyama step of an SDE, integrated in the given direction.

    Mean ``x + (t - s) b(x, s)`` and variance ``sigma_sde(s)^2 |t - s|``, with the drift and diffusion
    evaluated at the source. ``requires`` is the SDE's.

    Parameters
    ----------
    sde : SDE
        E.g. ``ForwardSDE(schedule)`` or ``ReverseSDE(schedule)``.
    direction : Direction
        ``Direction.FORWARD`` (``t`` increases) or ``Direction.BACKWARD`` (``t`` decreases). Stepping against
        the direction gives a non-positive variance.
    """

    sde: Any
    direction: Any                # sde.Direction

    @property
    def requires(self):
        """Return the SDE's required features (frozenset of Feature)."""
        return self.sde.requires

    def at(self, record, t_dst):
        """Integrate from ``record`` to time ``t_dst``. See ``Kernel.at``."""
        dt = t_dst - record.t                                       # signed
        sign = float(self.direction.value)                          # static switch: +1 or -1
        mean = record.x + bcast(dt, record.x) * self.sde.drift(record)   # drift at the source
        return Gaussian(mean, self.sde.diffusion_sq(record.t) * sign * dt)   # var <= 0 if misused (K8)


@dataclass(frozen=True)
class Guided:                     # base mean + scale(t, grad, var) * grad, with grad at the source
    """Shift a base kernel's mean along the reward gradient: ``mean + scale(t, grad, var) * grad``.

    ``grad`` is ``REWARD_GRAD`` at the source record; the variance is the base kernel's.

    Parameters
    ----------
    base : Kernel
        The kernel to guide.
    scale : Callable[[array, array, array], array]
        ``scale(t [L...], grad [L..., *D], var [L...]) -> [L...]``, e.g. ``tilt(lam)`` or ``normalized(step_length)``.
    """

    base: Any
    scale: Callable

    @property
    def requires(self):
        """Return the base kernel's required features plus ``REWARD_GRAD`` (frozenset of Feature)."""
        return self.base.requires | {Feature.REWARD_GRAD}

    def at(self, record, t_dst):
        """Return the guided distribution for moving from ``record`` to time ``t_dst``. See ``Kernel.at``."""
        base_dist = self.base.at(record, t_dst)
        grad = record.need(Feature.REWARD_GRAD)
        return Gaussian(base_dist.mean + bcast(self.scale(t_dst, grad, base_dist.var), grad) * grad, base_dist.var)
