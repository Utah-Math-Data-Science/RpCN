"""Backends (§4.2): randomness, scan, cond and switch, the operations the Array API doesn't cover."""

from __future__ import annotations

from types import ModuleType
from typing import Protocol


class Backend(Protocol):
    name: str                                    # "jax" | "torch" | "strict"
    xp: ModuleType                               # for constructors and tests only; library code infers xp
    def split(self, key, n: int): ...            # -> array of n keys (stackable, so scan can slice it)
    def normal(self, key, shape, dtype, device=None): ...
    def uniform(self, key, shape, dtype, device=None): ...
    def scan(self, f, carry, xs): ...            # -> (carry, stacked ys)
    def cond(self, pred, f_true, f_false, operand): ...
    def switch(self, index, branches, operand): ...   # index: Python int (static) or traced int scalar
    def key(self, seed: int): ...                # make a root key


def __getattr__(name):                           # import a backend's library only when asked for
    if name == "StrictBackend":
        from .strict import StrictBackend
        return StrictBackend
    if name == "JaxBackend":
        from .jax import JaxBackend
        return JaxBackend
    if name == "TorchBackend":
        from .torch import TorchBackend
        return TorchBackend
    raise AttributeError(name)


__all__ = ["Backend", "StrictBackend", "JaxBackend", "TorchBackend"]
