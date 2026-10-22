import contextlib
import warnings

import array_api_strict
import pytest
from hypothesis import settings

from ttg import EDM, RectifiedFlow, VPCosine, VPLinear
from ttg.backend import StrictBackend
from ttg.ops import importable
from ttg.testing.strategies import element_bounds, leading_shape, u_pair, xps_for  # noqa: F401  (re-exported for tests)

settings.register_profile("default", deadline=None, max_examples=50)   # JAX compile time breaks deadlines
settings.register_profile("ci", deadline=None, max_examples=200, derandomize=True)
settings.load_profile("default")

array_api_strict.set_array_api_strict_flags(api_version="2025.12")
warnings.filterwarnings("ignore", message=".*CUDA.*")
warnings.filterwarnings("ignore", message=".*Could not determine whether module.*")   # Hypothesis on torch


def available_backends():              # strict always; jax (with jax_enable_x64) and torch if importable
    out = [StrictBackend()]
    if importable("jax"):
        import jax

        from ttg.backend import JaxBackend

        jax.config.update("jax_enable_x64", True)
        out.append(JaxBackend())
    if importable("torch"):
        from ttg.backend import TorchBackend

        out.append(TorchBackend(device="cpu"))
    return out


BACKENDS = available_backends()
SCHEDULES = [VPLinear, VPCosine, EDM, RectifiedFlow]   # constructors take no arguments; defaults as in §5

backends = pytest.mark.parametrize("backend", BACKENDS, ids=str)
schedules = pytest.mark.parametrize("make_schedule", SCHEDULES, ids=lambda c: c.__name__)


def eager(backend):
    """Counting and duplicate tests run eagerly: under jit, Python side effects run once per trace."""
    if backend.name == "jax":
        import jax

        return jax.disable_jit()
    return contextlib.nullcontext()


def host(a):
    """A NumPy copy, for assertions that are easier in NumPy."""
    from ttg.testing import to_host

    return to_host(a)
