"""SDEs in forward-time convention (§7a): dx = b(x, t) dt + sigma_sde(t) dW."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from .ops import bcast
from .records import Feature

Array = Any


class Direction(Enum):                    # integration direction; the value is the sign of Delta = t - s
    FORWARD = 1                           # noising: t increases
    BACKWARD = -1                         # denoising: t decreases


class SDE(Protocol):
    requires: frozenset[Feature]          # what the drift reads from the source record
    def drift(self, rec) -> Array: ...    # b(x, t) at the source, [L..., *D]
    def diffusion2(self, t) -> Array: ... # sigma_sde(t)^2, [L...]


def f_coeff(sch, t):                      # f = m'/m
    return sch.dt_mean(t) / sch.mean(t)


def g2_coeff(sch, t):                     # gamma^2 = 2 s s' - 2 f s^2
    return 2 * sch.std(t) * sch.dt_std(t) - 2 * f_coeff(sch, t) * sch.std(t) ** 2


@dataclass(frozen=True)
class ForwardSDE:                         # the noising SDE whose marginals the schedule defines; no network
    sch: Any
    requires = frozenset()

    def drift(self, rec):
        return bcast(f_coeff(self.sch, rec.t), rec.x) * rec.x

    def diffusion2(self, t):
        return g2_coeff(self.sch, t)


@dataclass(frozen=True)
class ReverseSDE:                         # same marginals, run backwards; eta = 1 is the standard reverse-time SDE
    sch: Any
    eta: float = 1.0                      # eta > 0 required (eta = 0 is the probability-flow ODE: no density, I6)
    requires = frozenset({Feature.SCORE})

    def drift(self, rec):
        c = -0.5 * (1 + self.eta**2) * g2_coeff(self.sch, rec.t)
        return bcast(f_coeff(self.sch, rec.t), rec.x) * rec.x + bcast(c, rec.x) * rec.need(Feature.SCORE)

    def diffusion2(self, t):
        return self.eta**2 * g2_coeff(self.sch, t)
