"""Array-API-only helpers (§4.3). No module-level namespace: `xp` is inferred from the arguments."""

from __future__ import annotations

import importlib.util
import math
from dataclasses import MISSING, dataclass, field, fields, is_dataclass, replace

import array_api_compat


def namespace(*arrays):
    """The array namespace of the given arrays (§4.1)."""
    return array_api_compat.array_namespace(*arrays)


def importable(name):
    return importlib.util.find_spec(name) is not None


def jax_available():
    return importable("jax")


def pytree(cls):
    """Frozen dataclass, registered with JAX if JAX is importable. PyTorch needs no registration."""
    cls = dataclass(frozen=True)(cls)
    if jax_available():
        import jax

        data = [f.name for f in fields(cls) if not f.metadata.get("static")]
        meta = [f.name for f in fields(cls) if f.metadata.get("static")]
        jax.tree_util.register_dataclass(cls, data_fields=data, meta_fields=meta)
    return cls


def static(default=MISSING):
    return field(default=default, metadata={"static": True})


def tree_map(f, *trees):
    """Pure Python map over dataclasses, tuples, lists and dicts. None leaves stay None; static fields are kept."""
    t = trees[0]
    if t is None:
        return None
    if is_dataclass(t):
        return replace(t, **{fl.name: tree_map(f, *(getattr(x, fl.name) for x in trees))
                             for fl in fields(t) if not fl.metadata.get("static")})
    if isinstance(t, (tuple, list)):
        return type(t)(tree_map(f, *xs) for xs in zip(*trees))
    if isinstance(t, dict):
        return {k: tree_map(f, *(x[k] for x in trees)) for k in t}
    return f(*trees)


def leaves(tree):
    """The non-None array leaves, in tree_map order."""
    out = []
    tree_map(lambda a: out.append(a) or a, tree)
    return out


prod = math.prod


def axis1_size(tree):
    return leaves(tree)[0].shape[1]


def device(a):
    return array_api_compat.device(a)


def bcast(v, like):
    """[L...] -> broadcastable against [L..., *D]."""
    xp = namespace(v)
    return xp.reshape(v, tuple(v.shape) + (1,) * (like.ndim - v.ndim))


def logsumexp(a, axis, keepdims=False):
    xp = namespace(a)
    m = xp.max(a, axis=axis, keepdims=True)
    m = xp.where(xp.isfinite(m), m, 0.0)
    out = m + xp.log(xp.sum(xp.exp(a - m), axis=axis, keepdims=True))
    return out if keepdims else xp.squeeze(out, axis=axis)


def flatten_leading(a, k):
    xp = namespace(a)
    return xp.reshape(a, (-1,) + tuple(a.shape[k:]))


def unflatten_leading(a, lead):
    xp = namespace(a)
    return xp.reshape(a, tuple(lead) + tuple(a.shape[1:]))


def take_rows(a, idx):
    """a [I, N, ...], idx [I, N] -> a[i, idx[i, j]]."""
    xp = namespace(a, idx)
    I, N = idx.shape
    flat = flatten_leading(a, 2)
    flat_idx = xp.reshape(xp.arange(I, dtype=idx.dtype, device=device(idx))[:, None] * N + idx, (-1,))
    return unflatten_leading(xp.take(flat, flat_idx, axis=0), (I, N))


def take_levels(tree, levels):
    """Select a static list of positions on axis 1."""
    levels = list(levels)

    def take(a):
        xp = namespace(a)
        return xp.take(a, xp.asarray(levels, device=device(a)), axis=1)

    return tree_map(take, tree)


def put_levels(tree, levels, sub):
    """Write `sub` (axis 1 = len(levels)) into positions `levels` of `tree`; rebuilds axis 1 statically."""
    levels = list(levels)

    def put(a, b):
        xp = namespace(a, b)
        cols = [b[:, levels.index(m), ...] if m in levels else a[:, m, ...] for m in range(a.shape[1])]
        return xp.stack(cols, axis=1)

    return tree_map(put, tree, sub)


def tree_where(mask, a, b):
    """mask [I] or [I, P]; a and b must have identical structure."""
    def where(u, v):
        xp = namespace(mask, u, v)
        return xp.where(bcast(mask, u), u, v)

    return tree_map(where, a, b)


def stack_levels(levels):
    """List of trees with axis 1 of size 1 -> one tree."""
    def cat(*a):
        return namespace(*a).concat(a, axis=1)

    return tree_map(cat, *levels)


def steps_first(t_grid, lead):
    """[S] or [*lead, S] -> [S, *lead]."""
    xp = namespace(t_grid)
    lead = tuple(lead)
    if t_grid.ndim == 1:
        return xp.broadcast_to(xp.reshape(t_grid, (-1,) + (1,) * len(lead)), (t_grid.shape[0], *lead))
    return xp.moveaxis(t_grid, -1, 0)


def lead_shape(start):
    """Gaussian -> var.shape; Point / Record -> t.shape."""
    from .kernels import Gaussian

    return tuple(start.var.shape) if isinstance(start, Gaussian) else tuple(start.t.shape)
