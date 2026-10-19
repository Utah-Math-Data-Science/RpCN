"""Resampling and policies (§10). Branch-free and per instance."""

from __future__ import annotations

from dataclasses import dataclass

from .ops import device_of, logsumexp, namespace, take_rows, tree_map


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
        ``[I, N]`` int64 ancestor indices in ``[0, N)``, non-decreasing along axis 1.

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
    ancestors = xp.sum(xp.astype(cdf[:, None, :] < positions[:, :, None], xp.int64), axis=2)
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
        ``[I]`` int64 indices in ``[0, N)``.
    """
    xp = namespace(log_weights)
    I, N = log_weights.shape
    cdf = _cdf(log_weights)
    uniform = backend.uniform(key, (I, 1), cdf.dtype, device=device_of(log_weights))
    ancestors = xp.sum(xp.astype(cdf < uniform, xp.int64), axis=1)
    return xp.clip(ancestors, max=N - 1)


def masked_resample(backend, tree, log_weights, resample_mask, key):    # resample_mask: [I] bool
    """Resample the instances selected by `resample_mask`; leave the others unchanged. Branch-free.

    Parameters
    ----------
    backend : Backend
        Randomness for `systematic`.
    tree : pytree
        Arrays with leading axes ``[I, N]`` (e.g. a ``(Record, State)`` pair), gathered along axis 1.
    log_weights : array
        ``[I, N]`` unnormalised log-weights.
    resample_mask : array
        ``[I]`` bool: which instances to resample.
    key : key
        Random key.

    Returns
    -------
    tree : pytree
        Resampled where `resample_mask`, unchanged elsewhere.
    log_weights : array
        ``[I, N]``: zeros where `resample_mask`, `log_weights` elsewhere.
    """
    xp = namespace(log_weights, resample_mask)
    I, N = log_weights.shape
    own_index = xp.arange(N, dtype=xp.int64, device=device_of(log_weights))[None, :]
    ancestors = xp.where(resample_mask[:, None], systematic(backend, log_weights, key), own_index)
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
