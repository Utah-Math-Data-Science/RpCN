"""Resampling and policies (§10). Branch-free and per instance."""

from __future__ import annotations

from dataclasses import dataclass

from .ops import device, logsumexp, namespace, take_rows, tree_map


def ess(log_w):                                   # [I, N] -> [I]
    xp = namespace(log_w)
    lw = log_w - logsumexp(log_w, axis=1, keepdims=True)
    return xp.exp(-logsumexp(2 * lw, axis=1))


def systematic(bk, log_w, key):                   # [I, N] -> idx [I, N]
    xp = namespace(log_w)
    I, N = log_w.shape
    dev = device(log_w)
    w = xp.exp(log_w - logsumexp(log_w, axis=1, keepdims=True))
    cdf = xp.cumulative_sum(w, axis=1)
    u = (bk.uniform(key, (I, 1), w.dtype, device=dev) + xp.arange(N, dtype=w.dtype, device=dev)[None, :]) / N
    idx = xp.sum(xp.astype(cdf[:, None, :] < u[:, :, None], xp.int64), axis=2)   # O(N^2); avoids 2-D searchsorted
    return xp.clip(idx, max=N - 1)


def masked_resample(bk, tree, log_w, do, key):    # do: [I] bool
    xp = namespace(log_w, do)
    I, N = log_w.shape
    keep = xp.arange(N, dtype=xp.int64, device=device(log_w))[None, :]
    idx = xp.where(do[:, None], systematic(bk, log_w, key), keep)
    tree = tree_map(lambda a: take_rows(a, idx), tree)
    return tree, xp.where(do[:, None], xp.zeros_like(log_w), log_w)


@dataclass(frozen=True)
class Never:                                      # k: step index, possibly traced
    def should_resample(self, log_w, k):
        xp = namespace(log_w)
        return xp.zeros(log_w.shape[0], dtype=xp.bool, device=device(log_w))


@dataclass(frozen=True)
class Every:
    n: int

    def should_resample(self, log_w, k):
        xp = namespace(log_w)
        return xp.broadcast_to(xp.asarray(k % self.n == 0, device=device(log_w)), (log_w.shape[0],))


@dataclass(frozen=True)
class ESSBelow:
    frac: float

    def should_resample(self, log_w, k):
        return ess(log_w) < self.frac * log_w.shape[1]
