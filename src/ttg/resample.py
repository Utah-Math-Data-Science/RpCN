"""Resampling and policies (§10). Branch-free and per instance."""

from __future__ import annotations

from dataclasses import dataclass

from .ops import device, logsumexp, namespace, take_rows, tree_map


def ess(log_w):                                   # [I, N] -> [I]
    """Effective sample size of each instance's weights, ``1 / sum_n w_n**2`` with normalised ``w``.

    Parameters
    ----------
    log_w : array
        ``[I, N]`` unnormalised log-weights.

    Returns
    -------
    array
        ``[I]``, in ``[1, N]``, in the dtype of `log_w`.
    """
    xp = namespace(log_w)
    lw = log_w - logsumexp(log_w, axis=1, keepdims=True)
    return xp.exp(-logsumexp(2 * lw, axis=1))


def systematic(bk, log_w, key):                   # [I, N] -> idx [I, N]
    """Systematic resampling indices, independently per instance.

    Parameters
    ----------
    bk : Backend
        Supplies the one uniform draw per instance.
    log_w : array
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
    xp = namespace(log_w)
    I, N = log_w.shape
    dev = device(log_w)
    w = xp.exp(log_w - logsumexp(log_w, axis=1, keepdims=True))
    cdf = xp.cumulative_sum(w, axis=1)
    u = (bk.uniform(key, (I, 1), w.dtype, device=dev) + xp.arange(N, dtype=w.dtype, device=dev)[None, :]) / N
    idx = xp.sum(xp.astype(cdf[:, None, :] < u[:, :, None], xp.int64), axis=2)   # O(N^2); avoids 2-D searchsorted
    return xp.clip(idx, max=N - 1)


def masked_resample(bk, tree, log_w, do, key):    # do: [I] bool
    """Resample the instances selected by `do`; leave the others unchanged. Branch-free.

    Parameters
    ----------
    bk : Backend
        Randomness for `systematic`.
    tree : pytree
        Arrays with leading axes ``[I, N]`` (e.g. a ``(Record, State)`` pair), gathered along axis 1.
    log_w : array
        ``[I, N]`` unnormalised log-weights.
    do : array
        ``[I]`` bool: which instances to resample.
    key : key
        Random key.

    Returns
    -------
    tree : pytree
        Resampled where `do`, unchanged elsewhere.
    log_w : array
        ``[I, N]``: zeros where `do`, `log_w` elsewhere.
    """
    xp = namespace(log_w, do)
    I, N = log_w.shape
    keep = xp.arange(N, dtype=xp.int64, device=device(log_w))[None, :]
    idx = xp.where(do[:, None], systematic(bk, log_w, key), keep)
    tree = tree_map(lambda a: take_rows(a, idx), tree)
    return tree, xp.where(do[:, None], xp.zeros_like(log_w), log_w)


@dataclass(frozen=True)
class Never:                                      # k: step index, possibly traced
    """Resampling policy: never resample. Required when axis 1 is a temperature ladder (RpCN)."""

    def should_resample(self, log_w, k):
        """All False.

        Parameters
        ----------
        log_w : array
            ``[I, N]`` log-weights.
        k : int or array
            Step index, possibly traced.

        Returns
        -------
        array
            ``[I]`` bool.
        """
        xp = namespace(log_w)
        return xp.zeros(log_w.shape[0], dtype=xp.bool, device=device(log_w))


@dataclass(frozen=True)
class Every:
    """Resampling policy: resample every instance on steps whose index is a multiple of `n`.

    Parameters
    ----------
    n : int
        Stride in steps.
    """

    n: int

    def should_resample(self, log_w, k):
        """``k % n == 0``, broadcast to every instance.

        Parameters
        ----------
        log_w : array
            ``[I, N]`` log-weights.
        k : int or array
            Step index, possibly traced. `smc` resamples before advancing to grid point `k`, ``k = 1 .. S-1``.

        Returns
        -------
        array
            ``[I]`` bool.
        """
        xp = namespace(log_w)
        return xp.broadcast_to(xp.asarray(k % self.n == 0, device=device(log_w)), (log_w.shape[0],))


@dataclass(frozen=True)
class ESSBelow:
    """Resampling policy: resample the instances whose effective sample size is below ``frac * N``.

    Parameters
    ----------
    frac : float
        Threshold as a fraction of the particle count N, e.g. 0.5.
    """

    frac: float

    def should_resample(self, log_w, k):
        """``ess(log_w) < frac * N`` per instance.

        Parameters
        ----------
        log_w : array
            ``[I, N]`` log-weights.
        k : int or array
            Step index (unused).

        Returns
        -------
        array
            ``[I]`` bool.
        """
        return ess(log_w) < self.frac * log_w.shape[1]
