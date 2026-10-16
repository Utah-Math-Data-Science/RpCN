"""Features, points and evaluation records (§6)."""

from __future__ import annotations

from dataclasses import replace
from enum import Enum
from typing import Any

from .ops import pytree

Array = Any


class Feature(Enum):
    X0 = "x0"
    EPS = "eps"
    SCORE = "score"
    REWARD = "reward"
    REWARD_GRAD = "reward_grad"


class MissingFeature(KeyError):
    """A consumer read a feature nobody requested: a `requires` bug."""


class PartialRecord(ValueError):
    """A Record start lacks features the run needs."""


class DuplicateEvaluation(AssertionError):
    """Testing only: invariant I1 violated."""


@pytree
class Point:                         # unevaluated
    x: Array                         # [L..., *D]
    t: Array                         # [L...]


@pytree
class Record:                        # evaluated; absent features are None
    x: Array
    t: Array
    x0: Array | None = None
    eps: Array | None = None
    score: Array | None = None
    reward: Array | None = None
    reward_grad: Array | None = None

    def project(self, needs) -> "Record":   # keep exactly `needs` (C2's structure); static
        return replace(self, **{f.value: None for f in Feature if f not in needs})

    def need(self, f: Feature) -> Array:
        v = getattr(self, f.value)
        if v is None:
            raise MissingFeature(f)  # never silently re-evaluate
        return v

    def provides(self, needs) -> bool:   # static: depends only on pytree structure
        return all(getattr(self, f.value) is not None for f in needs)
