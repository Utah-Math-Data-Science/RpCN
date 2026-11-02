"""Resampling and policies (§10). Branch-free and per instance."""

from __future__ import annotations

from dataclasses import dataclass

from .ops import device_of, index_dtype, logsumexp, namespace, take_rows, tree_map


def ess(log_weights):                             # [I, N] -> [I]
    """Effective sample size of each instance's weights, ``1 / sum_n w_n**2`` with normalised ``w``.

    Parameters
    ----------
    log_weights : array
        ``[I, N]`` unnormalised log-weights.

    Returns
    -------
    array
        ``[I]``, in ``[1, N]``, in the dtype of `log_weights`.
    """
    xp = namespace(log_weights)
    log_weights_normalized = log_weights - logsumexp(log_weights, axis=1, keepdims=True)
    return xp.exp(-logsumexp(2 * log_weights_normalized, axis=1))


def _cdf(log_weights):                            # [I, N] -> [I, N]
    xp = namespace(log_weights)
    return xp.cumulative_sum(xp.exp(log_weights - logsumexp(log_weights, axis=1, keepdims=True)), axis=1)


def systematic(backend, log_weights, key):                   # [I, N] -> ancestors [I, N]
    """Systematic resampling indices, independently per instance.

    Parameters
    ----------
    backend : Backend
        Supplies the one uniform draw per instance.
    log_weights : array
        ``[I, N]`` unnormalised log-weights.
    key : key
        Random key.

    Returns
    -------
    array
        ``[I, N]`` integer (`index_dtype`) ancestor indices in ``[0, N)``, non-decreasing along axis 1.

    Notes
    -----
    Costs O(N**2) memory and time per instance (a comparison against the whole CDF), which avoids a 2-D
    searchsorted.
    """
    xp = namespace(log_weights)
    I, N = log_weights.shape
    device = device_of(log_weights)
    cdf = _cdf(log_weights)
    positions = (backend.uniform(key, (I, 1), cdf.dtype, device=device)
                 + xp.arange(N, dtype=cdf.dtype, device=device)[None, :]) / N
    # O(N^2); avoids 2-D searchsorted
    ancestors = xp.sum(xp.astype(cdf[:, None, :] < positions[:, :, None], index_dtype(xp, device)), axis=2)
    return xp.clip(ancestors, max=N - 1)


def ssp(backend, log_weights, key):                          # [I, N] -> ancestors [I, N]
    """Srinivasan Sampling Process (SSP) resampling indices, independently per instance.

    Each index n is kept ``floor(N w_n)`` times plus 0 or 1 more, with ``E[count_n] = N w_n``. The fractional parts
    are made integral two at a time (Srinivasan, 2001; Gerber, Chopin and Whiteley, 2019): the open fraction ``a``
    meets the next fraction ``b``, and one of them becomes 0 or 1 while the other carries ``a + b`` (or
    ``a + b - 1``) forward, with probabilities that keep each one's mean.

    Parameters
    ----------
    backend : Backend
        Supplies one uniform draw per instance and pair, and the `scan` over the N - 1 pairings.
    log_weights : array
        ``[I, N]`` unnormalised log-weights.
    key : key
        Random key.

    Returns
    -------
    array
        ``[I, N]`` integer (`index_dtype`) ancestor indices in ``[0, N)``, non-decreasing along axis 1.

    Notes
    -----
    Sequential in N (a `scan` of N - 1 steps, vectorised over instances); the counts-to-indices step costs O(N**2)
    memory and time per instance, like `systematic`.
    """
    xp = namespace(log_weights)
    I, N = log_weights.shape
    device = device_of(log_weights)
    idx_dtype = index_dtype(xp, device)
    if N == 1:                                                   # static: nothing to pair
        return xp.zeros((I, 1), dtype=idx_dtype, device=device)
    expected = N * xp.exp(log_weights - logsumexp(log_weights, axis=1, keepdims=True))
    whole = xp.floor(expected)
    fraction = expected - whole                                  # [I, N] in [0, 1)
    uniforms = backend.uniform(key, (N - 1, I), fraction.dtype, device=device)
    columns = xp.arange(1, N, dtype=idx_dtype, device=device)

    def pair(carry, step):                                       # carry: open index and fraction, [I] each
        open_index, a = carry
        column, b, uniform = step
        total = a + b
        below_one = total < 1
        # P(open keeps the mass): a / (a + b) if a + b < 1, else P(open does not become 1) = (1 - a) / (2 - a - b)
        keep_prob = xp.where(below_one, a / xp.where(total > 0, total, 1.0), (1 - a) / (2 - total))
        keep_open = uniform < keep_prob
        resolved_index = xp.where(keep_open, column, open_index)
        resolved_value = xp.astype(~below_one, total.dtype)
        open_index = xp.where(keep_open, open_index, column)
        return (open_index, xp.where(below_one, total, total - 1)), (resolved_index, resolved_value)

    start = (xp.zeros((I,), dtype=idx_dtype, device=device), fraction[:, 0])
    steps = (xp.broadcast_to(columns[:, None], (N - 1, I)), xp.permute_dims(fraction[:, 1:], (1, 0)), uniforms)
    (open_index, leftover), (resolved_index, resolved_value) = backend.scan(pair, start, steps)
    # the fractions sum to an integer, so the last open one is 0 or 1 up to rounding
    resolved_index = xp.concat([resolved_index, open_index[None, :]], axis=0)                       # [N, I]
    resolved_value = xp.concat([resolved_value, xp.round(leftover)[None, :]], axis=0)
    all_index = xp.arange(N, dtype=idx_dtype, device=device)
    extra = xp.sum(xp.where(resolved_index[:, :, None] == all_index[None, None, :], resolved_value[:, :, None],
                            xp.zeros_like(resolved_value[:, :, None])), axis=0)                    # [I, N]
    counts = xp.astype(xp.round(whole + extra), idx_dtype)
    ends = xp.cumulative_sum(counts, axis=1)                     # copies of index n fill [ends[n-1], ends[n])
    ancestors = xp.sum(xp.astype(ends[:, None, :] <= all_index[None, :, None], idx_dtype), axis=2)
    return xp.clip(ancestors, max=N - 1)


def categorical(backend, log_weights, key):                  # [I, N] -> ancestors [I]
    """Draw one index per instance from its normalised weights, by inverting the CDF.

    Use it to pick a single particle per instance, e.g. from an `smc` run's final ``log_weights``. (A fixed column
    of `systematic` is not such a draw: column j always falls in the j-th 1/N stratum of the CDF.)

    Parameters
    ----------
    backend : Backend
        Supplies the one uniform draw per instance.
    log_weights : array
        ``[I, N]`` unnormalised log-weights.
    key : key
        Random key.

    Returns
    -------
    array
        ``[I]`` integer (`index_dtype`) indices in ``[0, N)``.
    """
    xp = namespace(log_weights)
    I, N = log_weights.shape
    device = device_of(log_weights)
    cdf = _cdf(log_weights)
    uniform = backend.uniform(key, (I, 1), cdf.dtype, device=device)
    ancestors = xp.sum(xp.astype(cdf < uniform, index_dtype(xp, device)), axis=1)
    return xp.clip(ancestors, max=N - 1)


def masked_resample(backend, tree, log_weights, resample_mask, key, sampler=systematic):    # resample_mask: [I] bool
    """Resample the instances selected by `resample_mask`; leave the others unchanged. Branch-free.

    Parameters
    ----------
    backend : Backend
        Randomness for `sampler`.
    tree : pytree
        Arrays with leading axes ``[I, N]`` (e.g. a ``(Record, State)`` pair), gathered along axis 1.
    log_weights : array
        ``[I, N]`` unnormalised log-weights.
    resample_mask : array
        ``[I]`` bool: which instances to resample.
    key : key
        Random key.
    sampler : Callable[[Backend, array, key], array], optional
        Draws ``[I, N]`` ancestor indices from ``(backend, log_weights, key)``, e.g. `systematic` (default) or
        `ssp`. It runs for every instance; `resample_mask` picks which results are used.

    Returns
    -------
    tree : pytree
        Resampled where `resample_mask`, unchanged elsewhere.
    log_weights : array
        ``[I, N]``: zeros where `resample_mask`, `log_weights` elsewhere.
    """
    xp = namespace(log_weights, resample_mask)
    I, N = log_weights.shape
    device = device_of(log_weights)
    own_index = xp.arange(N, dtype=index_dtype(xp, device), device=device)[None, :]
    ancestors = xp.where(resample_mask[:, None], sampler(backend, log_weights, key), own_index)
    tree = tree_map(lambda a: take_rows(a, ancestors), tree)
    return tree, xp.where(resample_mask[:, None], xp.zeros_like(log_weights), log_weights)


@dataclass(frozen=True)
class Never:                                      # step_index: possibly traced
    """Resampling policy: never resample. Required when axis 1 is a temperature ladder (RpCN)."""

    def should_resample(self, log_weights, step_index):
        """All False.

        Parameters
        ----------
        log_weights : array
            ``[I, N]`` log-weights.
        step_index : int or array
            Step index, possibly traced.

        Returns
        -------
        array
            ``[I]`` bool.
        """
        xp = namespace(log_weights)
        return xp.zeros(log_weights.shape[0], dtype=xp.bool, device=device_of(log_weights))


@dataclass(frozen=True)
class Every:
    """Resampling policy: resample every instance on steps whose index is a multiple of `interval`.

    Parameters
    ----------
    interval : int
        Stride in steps.
    """

    interval: int

    def should_resample(self, log_weights, step_index):
        """``step_index % interval == 0``, broadcast to every instance.

        Parameters
        ----------
        log_weights : array
            ``[I, N]`` log-weights.
        step_index : int or array
            Step index, possibly traced. `smc` resamples before advancing to grid point `step_index`,
            ``step_index = 1 .. S-1``.

        Returns
        -------
        array
            ``[I]`` bool.
        """
        xp = namespace(log_weights)
        return xp.broadcast_to(xp.asarray(step_index % self.interval == 0, device=device_of(log_weights)),
                               (log_weights.shape[0],))


@dataclass(frozen=True)
class ESSBelow:
    """Resampling policy: resample the instances whose effective sample size is below ``fraction * N``.

    Parameters
    ----------
    fraction : float
        Threshold as a fraction of the particle count N, e.g. 0.5.
    """

    fraction: float

    def should_resample(self, log_weights, step_index):
        """``ess(log_weights) < fraction * N`` per instance.

        Parameters
        ----------
        log_weights : array
            ``[I, N]`` log-weights.
        step_index : int or array
            Step index (unused).

        Returns
        -------
        array
            ``[I]`` bool.
        """
        return ess(log_weights) < self.fraction * log_weights.shape[1]
