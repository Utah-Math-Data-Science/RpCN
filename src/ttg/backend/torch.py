"""PyTorch backend, through `array_api_compat.torch`. A key is an int64 tensor seed."""

from __future__ import annotations

import array_api_compat.torch as xp_torch
import torch

from ._eager import EagerControl

_SEED_HIGH = 2**62


def _generator(key, device):
    g = torch.Generator(device=device if device is not None else "cpu")
    g.manual_seed(int(key))
    return g


class TorchBackend(EagerControl):
    """PyTorch backend, through `array_api_compat.torch`.

    A key is an int64 tensor seed; each draw seeds a fresh `torch.Generator` from it. PyTorch has no tracing
    control flow here, so `scan`, `cond` and `switch` are eager Python (see `EagerControl`).

    Parameters
    ----------
    device : str or torch.device, default "cpu"
        Stored as `self.device`.

    Attributes
    ----------
    name : str
        ``"torch"``.
    xp : module
        `array_api_compat.torch`.
    device : torch.device
        The device given at construction.

    Notes
    -----
    The run's device is the device of its start arrays, and `normal`/`uniform` place samples on the `device`
    they are passed (CPU if None), not on `self.device`.
    """

    name = "torch"
    xp = xp_torch

    def __init__(self, device="cpu"):
        self.device = torch.device(device)

    def key(self, seed):
        """Make a root key.

        Parameters
        ----------
        seed : int
            Seed.

        Returns
        -------
        torch.Tensor
            The seed as an int64 scalar tensor (on CPU).
        """
        return torch.tensor(seed, dtype=torch.int64)

    def split(self, key, n):
        """Split `key` into `n` keys: `n` int64 seeds drawn from a CPU generator seeded with `key`.

        Parameters
        ----------
        key : torch.Tensor
            An int64 seed.
        n : int
            Number of keys.

        Returns
        -------
        torch.Tensor
            `n` int64 seeds, shape ``[n]``, on CPU.
        """
        return torch.randint(0, _SEED_HIGH, (n,), generator=_generator(key, "cpu"), dtype=torch.int64)

    def normal(self, key, shape, dtype, device=None):
        """Draw standard normal samples with ``torch.randn``, from a generator on `device` seeded with `key`.

        Parameters
        ----------
        key : torch.Tensor
            An int64 seed.
        shape : tuple of int
            Output shape.
        dtype : torch.dtype
            Floating dtype.
        device : torch.device or str, optional
            Output device; CPU if None.

        Returns
        -------
        torch.Tensor
            Samples of shape `shape`.
        """
        return torch.randn(tuple(shape), generator=_generator(key, device), dtype=dtype, device=device)

    def uniform(self, key, shape, dtype, device=None):
        """Draw samples uniform on [0, 1) with ``torch.rand``, from a generator on `device` seeded with `key`.

        Parameters
        ----------
        key : torch.Tensor
            An int64 seed.
        shape : tuple of int
            Output shape.
        dtype : torch.dtype
            Floating dtype.
        device : torch.device or str, optional
            Output device; CPU if None.

        Returns
        -------
        torch.Tensor
            Samples of shape `shape`.
        """
        return torch.rand(tuple(shape), generator=_generator(key, device), dtype=dtype, device=device)
