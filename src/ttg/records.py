"""Features, points and evaluation records (§6)."""

from __future__ import annotations

from dataclasses import replace
from enum import Enum
from typing import Any

from .ops import pytree

Array = Any


class Feature(Enum):
    """A model output that `evaluate` can produce for a point (x, t).

    Members are X0 (the denoised estimate x0_hat), EPS (the predicted noise), SCORE (grad_x log p_t(x)),
    REWARD (r_hat = reward(x0_hat), shape [B]) and REWARD_GRAD (grad_x of REWARD). Each member's value is the
    name of the matching `Record` field.
    """

    X0 = "x0"
    EPS = "eps"
    SCORE = "score"
    REWARD = "reward"
    REWARD_GRAD = "reward_grad"


class MissingFeature(KeyError):
    """Raised by `Record.need` when a consumer reads a feature nobody requested: a `requires` bug."""


class PartialRecord(ValueError):
    """Raised by `smc` when a `Record` start lacks features the run needs (they are never topped up)."""


class DuplicateEvaluation(AssertionError):
    """Testing only: raised by auditing test doubles when a point is evaluated more than once."""


@pytree
class Point:                         # unevaluated
    """An unevaluated point: a state and its time.

    Parameters
    ----------
    x : array
        States, shape [L..., *D] (leading axes L, e.g. [I, N], then the event shape D).
    t : array
        Times, shape [L...].
    """

    x: Array                         # [L..., *D]
    t: Array                         # [L...]


@pytree
class Record:                        # evaluated; absent features are None
    """An evaluated point: the state, its time and the features requested from `evaluate`.

    Features that were not requested are None. Every array shares the leading axes of `t`.

    Parameters
    ----------
    x : array
        States, shape [L..., *D].
    t : array
        Times, shape [L...].
    x0 : array or None, optional
        Denoised estimate x0_hat, shape [L..., *D].
    eps : array or None, optional
        Predicted noise, shape [L..., *D].
    score : array or None, optional
        Score grad_x log p_t(x), shape [L..., *D].
    reward : array or None, optional
        Reward r_hat of x0_hat, shape [L...].
    reward_grad : array or None, optional
        Gradient of `reward` with respect to `x`, shape [L..., *D].
    """

    x: Array
    t: Array
    x0: Array | None = None
    eps: Array | None = None
    score: Array | None = None
    reward: Array | None = None
    reward_grad: Array | None = None

    def project(self, needs) -> "Record":   # keep exactly `needs` (C2's structure); static
        """Drop every feature not in `needs`.

        Parameters
        ----------
        needs : iterable of Feature
            Features to keep.

        Returns
        -------
        Record
            A copy carrying exactly the features of `needs` that are present here.
        """
        return replace(self, **{f.value: None for f in Feature if f not in needs})

    def need(self, f: Feature) -> Array:
        """Return a feature, failing loudly if it was not evaluated.

        Parameters
        ----------
        f : Feature
            The feature to read.

        Returns
        -------
        array
            The feature's array.

        Raises
        ------
        MissingFeature
            If the feature is None, i.e. no component requested it. It is never re-evaluated.
        """
        v = getattr(self, f.value)
        if v is None:
            raise MissingFeature(f)  # never silently re-evaluate
        return v

    def provides(self, needs) -> bool:   # static: depends only on pytree structure
        """Whether every feature in `needs` is present.

        Parameters
        ----------
        needs : iterable of Feature
            Features to check.

        Returns
        -------
        bool
            True if none of them is None. Depends only on the pytree structure, so it is static under jit.
        """
        return all(getattr(self, f.value) is not None for f in needs)
