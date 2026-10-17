"""`array_api_strict` backend, for tests only. Keys are NumPy int64 seeds."""

from __future__ import annotations

import numpy as np

from ._eager import EagerControl

_SEED_HIGH = 2**62


class StrictBackend(EagerControl):
    """`array_api_strict` backend, for tests only: checks that library code uses nothing beyond the standard.

    Randomness comes from NumPy: a key is a NumPy int64 seed for ``numpy.random.default_rng``. `scan`, `cond`
    and `switch` are eager (see `EagerControl`).

    Attributes
    ----------
    name : str
        ``"strict"``.
    xp : module
        `array_api_strict`, imported on construction.
    """

    name = "strict"

    def __init__(self):
        import array_api_strict

        self.xp = array_api_strict

    def key(self, seed):
        """Make a root key.

        Parameters
        ----------
        seed : int
            Seed.

        Returns
        -------
        numpy.ndarray
            The seed as a NumPy int64 scalar.
        """
        return np.asarray(seed, dtype=np.int64)

    def split(self, key, n):
        """Split `key` into `n` keys: `n` int64 seeds drawn from a generator seeded with `key`.

        Parameters
        ----------
        key : numpy.ndarray
            An int64 seed.
        n : int
            Number of keys.

        Returns
        -------
        numpy.ndarray
            `n` int64 seeds, shape ``[n]``.
        """
        return np.random.default_rng(int(key)).integers(0, _SEED_HIGH, size=n, dtype=np.int64)

    def _from_numpy(self, a, dtype, device):
        np_dtype = np.float32 if dtype == self.xp.float32 else np.float64
        kw = {} if device is None else {"device": device}
        return self.xp.asarray(np.asarray(a, dtype=np_dtype), dtype=dtype, **kw)

    def normal(self, key, shape, dtype, device=None):
        """Draw standard normal samples from a NumPy generator seeded with `key`.

        Parameters
        ----------
        key : numpy.ndarray
            An int64 seed.
        shape : tuple of int
            Output shape.
        dtype : dtype
            `array_api_strict` float32 or float64.
        device : device, optional
            `array_api_strict` device.

        Returns
        -------
        array
            Samples of shape `shape`.
        """
        return self._from_numpy(np.random.default_rng(int(key)).standard_normal(tuple(shape)), dtype, device)

    def uniform(self, key, shape, dtype, device=None):
        """Draw samples uniform on [0, 1) from a NumPy generator seeded with `key`.

        Parameters
        ----------
        key : numpy.ndarray
            An int64 seed.
        shape : tuple of int
            Output shape.
        dtype : dtype
            `array_api_strict` float32 or float64.
        device : device, optional
            `array_api_strict` device.

        Returns
        -------
        array
            Samples of shape `shape`.
        """
        return self._from_numpy(np.random.default_rng(int(key)).random(tuple(shape)), dtype, device)
