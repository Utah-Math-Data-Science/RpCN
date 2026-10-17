"""SDEs in forward-time convention (§7a): dx = b(x, t) dt + sigma_sde(t) dW."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from .ops import bcast
from .records import Feature

Array = Any


class Direction(Enum):                    # integration direction; the value is the sign of Delta = t - s
    """Integration direction of an ``EulerMaruyama`` step; the value is the sign of ``t_dst - t_src``."""

    FORWARD = 1                           # noising: t increases
    BACKWARD = -1                         # denoising: t decreases


class SDE(Protocol):
    """An SDE ``dx = b(x, t) dt + sigma_sde(t) dW`` in forward-time convention.

    Attributes
    ----------
    requires : frozenset of Feature
        What the drift reads from the source record.
    """

    requires: frozenset[Feature]          # what the drift reads from the source record

    def drift(self, rec) -> Array:        # b(x, t) at the source, [L..., *D]
        """Return the drift ``b(x, t)`` at the record.

        Parameters
        ----------
        rec : Record
            ``x`` of shape ``[L..., *D]``, ``t`` of shape ``[L...]``, carrying ``requires``.

        Returns
        -------
        array
            ``[L..., *D]``.
        """
        ...

    def diffusion2(self, t) -> Array:     # sigma_sde(t)^2, [L...]
        """Return the squared diffusion coefficient ``sigma_sde(t)^2``.

        Parameters
        ----------
        t : array
            ``[L...]``.

        Returns
        -------
        array
            ``[L...]``.
        """
        ...


def f_coeff(sch, t):                      # f = m'/m
    """Linear drift coefficient of the schedule's noising SDE, ``f(t) = m'_t / m_t``.

    Parameters
    ----------
    sch : Schedule
        The noise schedule.
    t : array
        ``[L...]``.

    Returns
    -------
    array
        ``[L...]``.
    """
    return sch.dt_mean(t) / sch.mean(t)


def g2_coeff(sch, t):                     # gamma^2 = 2 s s' - 2 f s^2
    """Squared diffusion of the schedule's noising SDE, ``gamma(t)^2 = 2 s_t s'_t - 2 f(t) s_t^2``.

    Parameters
    ----------
    sch : Schedule
        The noise schedule.
    t : array
        ``[L...]``.

    Returns
    -------
    array
        ``[L...]``.
    """
    return 2 * sch.std(t) * sch.dt_std(t) - 2 * f_coeff(sch, t) * sch.std(t) ** 2


@dataclass(frozen=True)
class ForwardSDE:                         # the noising SDE whose marginals the schedule defines; no network
    """The noising SDE ``dx = f(t) x dt + gamma(t) dW`` whose marginals the schedule defines. Needs no network.

    Parameters
    ----------
    sch : Schedule
        The noise schedule.
    """

    sch: Any
    requires = frozenset()

    def drift(self, rec):
        """``f(t) x``. See ``SDE.drift``."""
        return bcast(f_coeff(self.sch, rec.t), rec.x) * rec.x

    def diffusion2(self, t):
        """``gamma(t)^2``. See ``SDE.diffusion2``."""
        return g2_coeff(self.sch, t)


@dataclass(frozen=True)
class ReverseSDE:                         # same marginals, run backwards; eta = 1 is the standard reverse-time SDE
    """Reverse-time SDE with the schedule's marginals, for integrating with ``Direction.BACKWARD``.

    Drift ``f x - (1 + eta^2) / 2 * gamma^2 score``, diffusion ``eta^2 gamma^2``. Reads ``SCORE``.

    Parameters
    ----------
    sch : Schedule
        The noise schedule.
    eta : float, default 1.0
        Noise level, > 0. ``eta = 1`` is the standard reverse-time SDE; ``eta = 0`` (the probability-flow ODE)
        has no density and is not supported.
    """

    sch: Any
    eta: float = 1.0                      # eta > 0 required (eta = 0 is the probability-flow ODE: no density, I6)
    requires = frozenset({Feature.SCORE})

    def drift(self, rec):
        """``f x - (1 + eta^2) / 2 * gamma^2 score``. See ``SDE.drift``."""
        c = -0.5 * (1 + self.eta**2) * g2_coeff(self.sch, rec.t)
        return bcast(f_coeff(self.sch, rec.t), rec.x) * rec.x + bcast(c, rec.x) * rec.need(Feature.SCORE)

    def diffusion2(self, t):
        """``eta^2 gamma(t)^2``. See ``SDE.diffusion2``."""
        return self.eta**2 * g2_coeff(self.sch, t)
