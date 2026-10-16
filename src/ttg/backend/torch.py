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
    name = "torch"
    xp = xp_torch

    def __init__(self, device="cpu"):
        self.device = torch.device(device)

    def key(self, seed):
        return torch.tensor(seed, dtype=torch.int64)

    def split(self, key, n):
        return torch.randint(0, _SEED_HIGH, (n,), generator=_generator(key, "cpu"), dtype=torch.int64)

    def normal(self, key, shape, dtype, device=None):
        return torch.randn(tuple(shape), generator=_generator(key, device), dtype=dtype, device=device)

    def uniform(self, key, shape, dtype, device=None):
        return torch.rand(tuple(shape), generator=_generator(key, device), dtype=dtype, device=device)
