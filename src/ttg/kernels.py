"""Gaussians and kernels (§7). Every kernel goes one way only; `at(rec, t_dst)` returns a per-row Gaussian."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from .ops import bcast, device, namespace, prod, pytree
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

    def sample(self, key, bk):
        """Draw one sample per row.

        Parameters
        ----------
        key : backend key
            Random key.
        bk : Backend
            Supplies the normal draws.

        Returns
        -------
        array
            ``[L..., *D]``, in the dtype and on the device of ``mean``.
        """
        xp = namespace(self.mean, self.var)
        noise = bk.normal(key, self.mean.shape, self.mean.dtype, device=device(self.mean))
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
        ev = tuple(range(self.var.ndim, x.ndim))
        d = prod(x.shape[self.var.ndim:])
        return -0.5 * xp.sum((x - self.mean) ** 2, axis=ev) / self.var - 0.5 * d * xp.log(2 * xp.pi * self.var)

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
        ev = tuple(range(self.var.ndim, x.ndim))
        d = prod(x.shape[self.var.ndim:])
        delta = self.mean - other.mean            # exact where the means are close (Sterbenz)
        y = x - other.mean                        # the sampled noise when x ~ other
        dv = self.var - other.var                 # exactly 0 when the variances are equal
        quad = xp.sum(delta * (y - 0.5 * delta), axis=ev) / self.var
        return (quad + 0.5 * xp.sum(y**2, axis=ev) * dv / (self.var * other.var)
                - 0.5 * d * xp.log1p(dv / other.var))


def transition(sch, t, s):
    """Forward transition coefficients from time ``t`` to a noisier time ``s``.

    For ``t < s``, ``x_s | x_t ~ N(m_{s|t} x_t, sigma^2_{s|t})``. The variance is computed from
    ``log_snr_diff``, so it is free of cancellation for small gaps.

    Parameters
    ----------
    sch : Schedule
        The noise schedule.
    t : array
        ``[L...]``. Source (less noisy) times.
    s : array
        ``[L...]``. Destination times, ``s > t``.

    Returns
    -------
    m : array
        ``[L...]``. ``m_{s|t} = m_s / m_t``.
    var : array
        ``[L...]``. ``sigma^2_{s|t} = s_s^2 - m_{s|t}^2 s_t^2``.
    """
    xp = namespace(t, s)
    return sch.mean(s) / sch.mean(t), -sch.std(s) ** 2 * xp.expm1(sch.log_snr_diff(s, t))


class Kernel(Protocol):
    """A one-way Markov kernel: given a source record and a destination time, a per-row Gaussian.

    Attributes
    ----------
    requires : frozenset of Feature
        The features ``at`` reads from the source record.
    """

    requires: frozenset[Feature]

    def at(self, rec, t_dst) -> Gaussian:    # t_dst has rec.t's shape [L...]
        """Return the kernel's distribution for moving from ``rec`` to time ``t_dst``.

        Parameters
        ----------
        rec : Record
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

    ``at(rec, t)`` requires ``rec.t < t`` and returns ``N(m_{t|s} x, sigma^2_{t|s})``. Needs no network.

    Parameters
    ----------
    sch : Schedule
        The noise schedule.
    """

    sch: Any
    requires = frozenset()

    def at(self, rec, t):
        """Noise ``rec`` forward to the later time ``t``. See ``Kernel.at``."""
        m, var = transition(self.sch, rec.t, t)                     # source rec.t < destination t
        return Gaussian(bcast(m, rec.x) * rec.x, var)


@dataclass(frozen=True)
class PosteriorStep:
    """Denoising step: the forward process's posterior ``q(x_t | x_s, x0 = x0_hat)``, for ``t < s``.

    Reads ``X0`` (the network's prediction of the clean sample) from the source record.

    Parameters
    ----------
    sch : Schedule
        The noise schedule.
    """

    sch: Any
    requires = frozenset({Feature.X0})

    def at(self, rec, t):
        """Denoise ``rec`` (at time ``s = rec.t``) to the earlier time ``t``. See ``Kernel.at``."""
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
    """DDIM denoising step with stochasticity ``eta``.

    Mean ``m_t x0_hat + sqrt(s_t^2 - eta^2 tau^2) eps_hat`` and variance ``eta^2 tau^2``, where ``tau^2`` is
    the ``PosteriorStep`` variance. Reads ``X0`` and ``EPS`` from the source record.

    Parameters
    ----------
    sch : Schedule
        The noise schedule.
    eta : float
        In ``(0, 1]``. ``eta = 1`` gives the same distribution as ``PosteriorStep``; ``eta = 0`` (deterministic
        DDIM) has no density and is not supported.
    """

    sch: Any
    eta: float
    requires = frozenset({Feature.X0, Feature.EPS})

    def at(self, rec, t):
        """Denoise ``rec`` to the earlier time ``t``. See ``Kernel.at``."""
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
    """One unadjusted Langevin step at fixed time: ``N(x + h score, 2 h)``.

    Reads ``SCORE`` from the source record. Add a reward drift with ``Guided(Langevin(h), tilt(lam, 0.5))``.

    Parameters
    ----------
    h : Callable[[array], array]
        Step size as a function of ``t`` (``[L...] -> [L...]``), e.g. ``lambda t: 0.01 * sch.std(t)**2``.
    """

    h: Callable                   # step size as a function of t, e.g. lambda t: 0.01 * sch.std(t)**2
    requires = frozenset({Feature.SCORE})

    def at(self, rec, t):         # t == rec.t
        """Langevin step from ``rec``; ``t`` must equal ``rec.t``. See ``Kernel.at``."""
        h = self.h(rec.t)
        return Gaussian(rec.x + bcast(h, rec.x) * rec.need(Feature.SCORE), 2 * h)


@dataclass(frozen=True)
class EulerMaruyama:              # one EM step of the injected SDE, integrated up or down
    """One Euler-Maruyama step of an SDE, integrated in the given direction.

    Mean ``x + (t - s) b(x, s)`` and variance ``sigma_sde(s)^2 |t - s|``, with the drift and diffusion
    evaluated at the source. ``requires`` is the SDE's.

    Parameters
    ----------
    sde : SDE
        E.g. ``ForwardSDE(sch)`` or ``ReverseSDE(sch)``.
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

    def at(self, rec, t):
        """Integrate from ``rec`` to time ``t``. See ``Kernel.at``."""
        delta = t - rec.t                                           # signed
        sign = float(self.direction.value)                          # static switch: +1 or -1
        mean = rec.x + bcast(delta, rec.x) * self.sde.drift(rec)    # drift at the source
        return Gaussian(mean, self.sde.diffusion2(rec.t) * sign * delta)   # var <= 0 if misused (K8)


@dataclass(frozen=True)
class Guided:                     # base mean + scale(t, g, var) * g, with g at the source
    """Shift a base kernel's mean along the reward gradient: ``mean + scale(t, g, var) * g``.

    ``g`` is ``REWARD_GRAD`` at the source record; the variance is the base kernel's.

    Parameters
    ----------
    base : Kernel
        The kernel to guide.
    scale : Callable[[array, array, array], array]
        ``scale(t [L...], g [L..., *D], var [L...]) -> [L...]``, e.g. ``tilt(lam)`` or ``normalized(gamma)``.
    """

    base: Any
    scale: Callable

    @property
    def requires(self):
        """Return the base kernel's required features plus ``REWARD_GRAD`` (frozenset of Feature)."""
        return self.base.requires | {Feature.REWARD_GRAD}

    def at(self, rec, t):
        """Return the guided distribution for moving from ``rec`` to time ``t``. See ``Kernel.at``."""
        d = self.base.at(rec, t)
        g = rec.need(Feature.REWARD_GRAD)
        return Gaussian(d.mean + bcast(self.scale(t, g, d.var), g) * g, d.var)
