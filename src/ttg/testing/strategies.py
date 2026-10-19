"""Hypothesis strategies (§18.1): element_bounds, u_pair, leading_shape, xps_for."""

from __future__ import annotations

import hypothesis.extra.array_api
from hypothesis import strategies as st
from hypothesis.extra.array_api import make_strategies_namespace


def xps_for(backend):
    """Hypothesis array strategies for a backend's namespace.

    Hypothesis may not know the newest standard revision yet; array generation uses nothing newer
    than 2023.12, so declaring an older version is safe.

    Parameters
    ----------
    backend : Backend
        Backend whose `xp` the strategies generate arrays for.

    Returns
    -------
    module-like
        The `hypothesis.extra.array_api` strategies namespace, at the newest revision Hypothesis has released.
    """
    return make_strategies_namespace(backend.xp, api_version=max(hypothesis.extra.array_api.RELEASED_VERSIONS))


def element_bounds(lo=-3.0, hi=3.0):
    """Keyword arguments for finite, normal-range float elements in ``[lo, hi]``.

    Parameters
    ----------
    lo, hi : float
        Bounds.

    Returns
    -------
    dict
        For the `elements` argument of ``xps.arrays``: no NaN, infinity or subnormals.
    """
    return {"min_value": lo, "max_value": hi, "allow_nan": False,
            "allow_infinity": False, "allow_subnormal": False}


@st.composite
def u_pair(draw, gap=1e-2):           # u_lo < u_hi, both in [0, 1]
    """Strategy for a pair ``(u_lo, u_hi)`` in [0, 1] with ``u_hi >= u_lo + gap``.

    Parameters
    ----------
    gap : float, default 1e-2
        Minimum separation.

    Returns
    -------
    SearchStrategy[tuple[float, float]]
    """
    u_lo = draw(st.floats(0.0, 1.0 - gap))
    return u_lo, draw(st.floats(u_lo + gap, 1.0))


@st.composite
def leading_shape(draw, max_instances=3, max_particles=4):
    """Strategy for leading axes ``(I, N)``: instances and particles, each at least 1.

    Parameters
    ----------
    max_instances, max_particles : int
        Upper bounds.

    Returns
    -------
    SearchStrategy[tuple[int, int]]
    """
    return draw(st.integers(1, max_instances)), draw(st.integers(1, max_particles))
