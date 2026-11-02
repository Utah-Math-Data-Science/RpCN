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

        data = [dataclass_field.name for dataclass_field in fields(cls) if not dataclass_field.metadata.get("static")]
        meta = [dataclass_field.name for dataclass_field in fields(cls) if dataclass_field.metadata.get("static")]
        jax.tree_util.register_dataclass(cls, data_fields=data, meta_fields=meta)
    return cls


def static(default=MISSING):
    return field(default=default, metadata={"static": True})


def tree_map(fn, *trees):
    """Pure Python map over dataclasses, tuples, lists and dicts. None leaves stay None; static fields are kept."""
    first = trees[0]
    if first is None:
        return None
    if is_dataclass(first):
        return replace(first, **{dataclass_field.name: tree_map(fn, *(getattr(tree, dataclass_field.name)
                                                                      for tree in trees))
                                 for dataclass_field in fields(first) if not dataclass_field.metadata.get("static")})
    if isinstance(first, (tuple, list)):
        return type(first)(tree_map(fn, *children) for children in zip(*trees))
    if isinstance(first, dict):
        return {k: tree_map(fn, *(tree[k] for tree in trees)) for k in first}
    return fn(*trees)


def leaves(tree):
    """The non-None array leaves, in tree_map order."""
    out = []
    tree_map(lambda a: out.append(a) or a, tree)
    return out


prod = math.prod


def n_levels(tree):
    return leaves(tree)[0].shape[1]


def device_of(a):
    return array_api_compat.device(a)


def index_dtype(xp, device=None):
    """The namespace's default indexing dtype (int32 on JAX without x64, int64 on NumPy and PyTorch)."""
    return xp.__array_namespace_info__().default_dtypes(device=device)["indexing"]


def bcast(v, like):
    """[L...] -> broadcastable against [L..., *D]."""
    xp = namespace(v)
    return xp.reshape(v, tuple(v.shape) + (1,) * (like.ndim - v.ndim))


def logsumexp(a, axis, keepdims=False):
    xp = namespace(a)
    shift = xp.max(a, axis=axis, keepdims=True)
    shift = xp.where(xp.isfinite(shift), shift, 0.0)
    out = shift + xp.log(xp.sum(xp.exp(a - shift), axis=axis, keepdims=True))
    return out if keepdims else xp.squeeze(out, axis=axis)


def flatten_leading(a, n_leading):
    xp = namespace(a)
    return xp.reshape(a, (-1,) + tuple(a.shape[n_leading:]))


def unflatten_leading(a, leading_shape):
    xp = namespace(a)
    return xp.reshape(a, tuple(leading_shape) + tuple(a.shape[1:]))


def take_rows(a, idx):
    """a [I, N, ...], idx [I, N] -> a[i, idx[i, j]]."""
    xp = namespace(a, idx)
    I, N = idx.shape
    flat = flatten_leading(a, 2)
    flat_idx = xp.reshape(xp.arange(I, dtype=idx.dtype, device=device_of(idx))[:, None] * N + idx, (-1,))
    return unflatten_leading(xp.take(flat, flat_idx, axis=0), (I, N))


def take_levels(tree, levels):
    """Select a static list of positions on axis 1."""
    runs = []                                    # maximal runs of consecutive levels, as (start, stop)
    for level in levels:
        if runs and runs[-1][1] == level:
            runs[-1] = (runs[-1][0], level + 1)
        else:
            runs.append((level, level + 1))

    def take(a):                                 # slices, not an index array: no host-to-device copy (a GPU sync)
        parts = [a[:, start:stop, ...] for start, stop in runs]
        return parts[0] if len(parts) == 1 else namespace(a).concat(parts, axis=1)

    return tree_map(take, tree)


def put_levels(tree, levels, values):
    """Write `values` (axis 1 = len(levels)) into positions `levels` of `tree`; rebuilds axis 1 statically."""
    levels = list(levels)

    def put(a, b):
        xp = namespace(a, b)
        columns = [b[:, levels.index(level), ...] if level in levels else a[:, level, ...]
                   for level in range(a.shape[1])]
        return xp.stack(columns, axis=1)

    return tree_map(put, tree, values)


def tree_where(mask, a, b):
    """mask [I] or [I, P]; a and b must have identical structure."""
    def where(on_true, on_false):
        xp = namespace(mask, on_true, on_false)
        return xp.where(bcast(mask, on_true), on_true, on_false)

    return tree_map(where, a, b)


def stack_levels(levels):
    """List of trees with axis 1 of size 1 -> one tree."""
    def concat_levels(*a):
        return namespace(*a).concat(a, axis=1)

    return tree_map(concat_levels, *levels)


def steps_first(t_grid, leading_shape):
    """[S] or [*leading_shape, S] -> [S, *leading_shape]."""
    xp = namespace(t_grid)
    leading_shape = tuple(leading_shape)
    if t_grid.ndim == 1:
        return xp.broadcast_to(xp.reshape(t_grid, (-1,) + (1,) * len(leading_shape)),
                               (t_grid.shape[0], *leading_shape))
    return xp.moveaxis(t_grid, -1, 0)


def leading_shape_of(start):
    """Gaussian -> var.shape; Point / Record -> t.shape."""
    from .kernels import Gaussian

    return tuple(start.var.shape) if isinstance(start, Gaussian) else tuple(start.t.shape)
