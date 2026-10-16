"""Hypothesis strategies (§18.1): elems, u_pair, leading, xps_for."""

from __future__ import annotations

import hypothesis.extra.array_api
from hypothesis import strategies as st
from hypothesis.extra.array_api import make_strategies_namespace


def xps_for(bk):
    """Hypothesis may not know the newest standard revision yet; array generation uses nothing newer
    than 2023.12, so declaring an older version is safe."""
    return make_strategies_namespace(bk.xp, api_version=max(hypothesis.extra.array_api.RELEASED_VERSIONS))


def elems(lo=-3.0, hi=3.0):
    return {"min_value": lo, "max_value": hi, "allow_nan": False,
            "allow_infinity": False, "allow_subnormal": False}


@st.composite
def u_pair(draw, gap=1e-2):           # u_lo < u_hi, both in [0, 1]
    a = draw(st.floats(0.0, 1.0 - gap))
    return a, draw(st.floats(a + gap, 1.0))


@st.composite
def leading(draw, max_I=3, max_N=4):
    return draw(st.integers(1, max_I)), draw(st.integers(1, max_N))
