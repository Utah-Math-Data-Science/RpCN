"""R1-R5: systematic resampling, masking, alignment, ESS, categorical selection."""

import math

import numpy as np
from conftest import backends, element_bounds, host, leading_shape, xps_for
from hypothesis import given
from hypothesis import strategies as st

from ttg import Record, categorical, ess, masked_resample, systematic


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
@given(data=st.data(), seed=st.integers(0, 2**20))
def test_R3_resampling_keeps_fields_aligned(backend, data, seed):
    xp = backend.xp
    I, N = data.draw(leading_shape())
    log_w = draw_log_w(backend, data, I, N)
    a = data.draw(xps_for(backend).arrays(xp.float64, (I, N, 2), elements=element_bounds()))
    rec = Record(a, xp.sum(a, axis=2), reward=xp.sum(a**2, axis=2) + 1.0)    # t = f(x), reward = g(x)
    do = data.draw(xps_for(backend).arrays(xp.bool, (I,)))
    out, lw = masked_resample(backend, rec, log_w, do, backend.key(seed))
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
