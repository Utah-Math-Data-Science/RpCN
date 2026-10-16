"""`array_api_strict` backend, for tests only. Keys are NumPy int64 seeds."""

from __future__ import annotations

import numpy as np

from ._eager import EagerControl

_SEED_HIGH = 2**62


class StrictBackend(EagerControl):
    name = "strict"

    def __init__(self):
        import array_api_strict

        self.xp = array_api_strict

    def key(self, seed):
        return np.asarray(seed, dtype=np.int64)

    def split(self, key, n):
        return np.random.default_rng(int(key)).integers(0, _SEED_HIGH, size=n, dtype=np.int64)

    def _from_numpy(self, a, dtype, device):
        np_dtype = np.float32 if dtype == self.xp.float32 else np.float64
        kw = {} if device is None else {"device": device}
        return self.xp.asarray(np.asarray(a, dtype=np_dtype), dtype=dtype, **kw)

    def normal(self, key, shape, dtype, device=None):
        return self._from_numpy(np.random.default_rng(int(key)).standard_normal(tuple(shape)), dtype, device)

    def uniform(self, key, shape, dtype, device=None):
        return self._from_numpy(np.random.default_rng(int(key)).random(tuple(shape)), dtype, device)
