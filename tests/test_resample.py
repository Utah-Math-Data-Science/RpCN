"""R1-R7: systematic resampling, masking, alignment, ESS, categorical selection, SSP resampling."""

import math

import numpy as np
import pytest
from conftest import backends, element_bounds, host, leading_shape, xps_for
from hypothesis import given, settings
from hypothesis import strategies as st

from ttg import Record, categorical, ess, masked_resample, ssp, systematic
from ttg.ops import device_of, index_dtype, logsumexp, namespace


def draw_log_w(backend, data, I, N):
    return data.draw(xps_for(backend).arrays(backend.xp.float64, (I, N), elements=element_bounds(-5.0, 5.0)))


@backends
@given(data=st.data(), seed=st.integers(0, 2**20))
def test_R1_systematic_counts(backend, data, seed):
    I, N = data.draw(leading_shape(max_particles=8))
    log_w = draw_log_w(backend, data, I, N)
    idx = host(systematic(backend, log_w, backend.key(seed)))
    lw = host(log_w)
    w = np.exp(lw - lw.max(axis=1, keepdims=True))
    w /= w.sum(axis=1, keepdims=True)
    for i in range(I):
        counts = np.bincount(idx[i], minlength=N)
        assert np.all(np.abs(counts - N * w[i]) <= 1 + 1e-9)


@backends
@given(data=st.data(), seed=st.integers(0, 2**20))
def test_R2_no_resampling_is_identity(backend, data, seed):
    xp = backend.xp
    I, N = data.draw(leading_shape())
    log_w = draw_log_w(backend, data, I, N)
    tree = (data.draw(xps_for(backend).arrays(xp.float64, (I, N, 3), elements=element_bounds())), log_w * 2)
    out, lw = masked_resample(backend, tree, log_w, xp.zeros((I,), dtype=xp.bool), backend.key(seed))
    assert bool(xp.all(lw == log_w))
    assert all(bool(xp.all(a == b)) for a, b in zip(out, tree))


@backends
@pytest.mark.parametrize("sampler", [systematic, ssp], ids=lambda sampler: sampler.__name__)
@given(data=st.data(), seed=st.integers(0, 2**20))
def test_R3_resampling_keeps_fields_aligned(backend, sampler, data, seed):
    xp = backend.xp
    I, N = data.draw(leading_shape())
    log_w = draw_log_w(backend, data, I, N)
    a = data.draw(xps_for(backend).arrays(xp.float64, (I, N, 2), elements=element_bounds()))
    rec = Record(a, xp.sum(a, axis=2), reward=xp.sum(a**2, axis=2) + 1.0)    # t = f(x), reward = g(x)
    do = data.draw(xps_for(backend).arrays(xp.bool, (I,)))
    out, lw = masked_resample(backend, rec, log_w, do, backend.key(seed), sampler)
    assert out.x0 is None and out.eps is None and out.score is None and out.reward_grad is None
    assert bool(xp.all(out.t == xp.sum(out.x, axis=2)))
    assert bool(xp.all(out.reward == xp.sum(out.x**2, axis=2) + 1.0))
    assert bool(xp.all(xp.where(do[:, None], lw == 0, lw == log_w)))


@backends
@given(data=st.data())
def test_R4_ess_range_and_extremes(backend, data):
    xp = backend.xp
    I, N = data.draw(leading_shape(max_particles=8))
    e = host(ess(draw_log_w(backend, data, I, N)))
    assert np.all(e >= 1 - 1e-12) and np.all(e <= N * (1 + 1e-12))
    assert np.allclose(host(ess(xp.zeros((I, N), dtype=xp.float64))), N, rtol=1e-12)
    j = data.draw(st.integers(0, N - 1))
    one_hot = xp.where(xp.arange(N)[None, :] == j, 0.0, -math.inf * xp.ones((I, N), dtype=xp.float64))
    assert np.all(host(ess(one_hot)) == 1)


@backends
@given(data=st.data(), seed=st.integers(0, 2**20))
def test_R5_categorical_range_and_one_hot(backend, data, seed):
    xp = backend.xp
    I, N = data.draw(leading_shape(max_particles=8))
    idx = host(categorical(backend, draw_log_w(backend, data, I, N), backend.key(seed)))
    assert idx.shape == (I,) and np.all((idx >= 0) & (idx < N))
    j = data.draw(st.integers(0, N - 1))
    one_hot = xp.where(xp.arange(N)[None, :] == j, 0.0, -math.inf * xp.ones((I, N), dtype=xp.float64))
    assert np.all(host(categorical(backend, one_hot, backend.key(seed))) == j)


@backends
def test_R5_categorical_frequencies(backend):
    xp = backend.xp
    I = 20_000
    w = np.array([0.1, 0.2, 0.3, 0.4])
    log_w = xp.asarray(np.tile(np.log(w) + 7.0, (I, 1)))              # unnormalised on purpose
    freq = np.bincount(host(categorical(backend, log_w, backend.key(0))), minlength=w.size) / I
    assert np.all(np.abs(freq - w) < 5 * np.sqrt(w * (1 - w) / I))


@backends
@given(data=st.data(), seed=st.integers(0, 2**20))
def test_R6_ssp_counts_are_floor_or_ceil(backend, data, seed):
    I, N = data.draw(leading_shape(max_particles=8))
    log_w = draw_log_w(backend, data, I, N)
    idx = host(ssp(backend, log_w, backend.key(seed)))
    lw = host(log_w)
    w = np.exp(lw - lw.max(axis=1, keepdims=True))
    w /= w.sum(axis=1, keepdims=True)
    assert idx.shape == (I, N) and np.all((idx >= 0) & (idx < N)) and np.all(np.diff(idx, axis=1) >= 0)
    for i in range(I):
        counts = np.bincount(idx[i], minlength=N)
        assert np.all(counts >= np.floor(N * w[i] - 1e-9)) and np.all(counts <= np.ceil(N * w[i] + 1e-9))


@backends
def test_R6_ssp_counts_are_unbiased(backend):
    xp = backend.xp
    I = 20_000
    w = np.array([0.05, 0.3, 0.15, 0.25, 0.1, 0.15])
    N = w.size
    log_w = xp.asarray(np.tile(np.log(w) + 7.0, (I, 1)))              # unnormalised on purpose
    idx = host(ssp(backend, log_w, backend.key(0)))
    counts = np.stack([np.bincount(row, minlength=N) for row in idx])
    assert np.all(counts.sum(axis=1) == N)
    frac = N * w - np.floor(N * w)                                     # count - floor is Bernoulli(frac)
    assert np.all(np.abs(counts.mean(axis=0) - N * w) <= 5 * np.sqrt(frac * (1 - frac) / I) + 1e-12)


def ssp_reference(backend, log_weights, key):                # [I, N] -> ancestors [I, N]
    """Independent SSP implementation (pairwise pivotal updates over an explicit left/right pair), for R7.

    It draws its uniforms in a different layout and orientation from `ssp`, so the two agree in distribution only,
    not draw for draw.
    """
    xp = namespace(log_weights)
    I, N = log_weights.shape
    device = device_of(log_weights)
    int_dtype = index_dtype(xp, device)
    particle = xp.arange(N, dtype=int_dtype, device=device)[None, :]
    if N == 1:
        return xp.zeros((I, 1), dtype=int_dtype, device=device)

    weights = xp.exp(log_weights - logsumexp(log_weights, axis=1, keepdims=True))
    expected = N * weights
    children = xp.astype(xp.floor(expected), int_dtype)
    residual = expected - xp.astype(children, expected.dtype)
    uniforms = backend.uniform(key, (I, N - 1), expected.dtype, device=device)
    left = xp.zeros((I,), dtype=int_dtype, device=device)
    right = xp.ones((I,), dtype=int_dtype, device=device)

    def at(values, indices):
        return xp.sum(xp.where(particle == indices[:, None], values, xp.zeros_like(values)), axis=1)

    for step in range(N - 1):
        residual_left = at(residual, left)
        residual_right = at(residual, right)
        delta_left = xp.minimum(residual_right, 1.0 - residual_left)
        delta_right = xp.minimum(residual_left, 1.0 - residual_right)
        delta_sum = delta_left + delta_right
        probability = xp.where(delta_sum > 0, delta_left / xp.where(delta_sum > 0, delta_sum, 1.0), 0.0)
        swap = uniforms[:, step] < probability
        next_left = xp.where(swap, right, left)
        next_right = xp.where(swap, left, right)
        delta = xp.where(swap, delta_right, delta_left)

        residual_left = at(residual, next_left)
        residual_right = at(residual, next_right)
        increase_left = residual_right < 1.0 - residual_left
        left_mask = particle == next_left[:, None]
        right_mask = particle == next_right[:, None]
        residual = residual + xp.where(
            increase_left[:, None],
            xp.where(left_mask, delta[:, None], 0.0),
            xp.where(right_mask, -delta[:, None], 0.0),
        )
        children = children + xp.astype((~increase_left)[:, None] & left_mask, int_dtype)

        next_index = xp.full((I,), step + 2, dtype=int_dtype, device=device)
        left = xp.where(increase_left, next_left, next_index)
        right = xp.where(increase_left, next_index, next_right)

    # Match the reference implementation's correction for accumulated floating-point roundoff.
    last = xp.where(right == N, left, right)
    residual_last = at(residual, last)
    missing_one = (xp.sum(children, axis=1) == N - 1) & (residual_last > 0.99)
    children = children + xp.astype(missing_one[:, None] & (particle == last[:, None]), int_dtype)

    cumulative = xp.cumulative_sum(children, axis=1)
    slots = xp.arange(N, dtype=int_dtype, device=device)[None, :, None]
    ancestors = xp.sum(xp.astype(cumulative[:, None, :] <= slots, int_dtype), axis=2)
    return xp.clip(ancestors, max=N - 1)


def count_pmf(ancestors, N):
    """{count vector: frequency} over the rows of [I, N] ancestors."""
    counts = np.stack([np.bincount(row, minlength=N) for row in ancestors])
    patterns, hits = np.unique(counts, axis=0, return_counts=True)
    return {tuple(pattern): hit / counts.shape[0] for pattern, hit in zip(patterns, hits)}


@backends
@settings(max_examples=20)
@given(data=st.data(), N=st.integers(2, 6), seed=st.integers(0, 2**20))
def test_R7_ssp_matches_reference_in_distribution(backend, data, N, seed):
    xp = backend.xp
    I = 20_000
    log_w = xp.broadcast_to(draw_log_w(backend, data, 1, N), (I, N))   # I independent draws for one weight row
    keys = backend.split(backend.key(seed), 2)
    pmf = count_pmf(host(ssp(backend, log_w, keys[0])), N)
    pmf_reference = count_pmf(host(ssp_reference(backend, log_w, keys[1])), N)
    for pattern in pmf.keys() | pmf_reference.keys():
        p, q = pmf.get(pattern, 0.0), pmf_reference.get(pattern, 0.0)
        pooled = (p + q) / 2
        assert abs(p - q) <= 6 * math.sqrt(pooled * (1 - pooled) * 2 / I) + 1e-12, (pattern, p, q)
