Test-Time Sequential Steering of Diffusion Models via Preconditioned Crank-Nicolson
===================================================================================

Code repository for `Test-Time Sequential Steering of Diffusion Models via Preconditioned Crank-Nicolson <https://openreview.net/forum?id=N7zXHlKXnl>`_.
Implementations for Residual preconditioned Crank-Nicolson (RpCN) and other test-time guidance methods are provided as an installable Python package (see `Usage`_).

The code for replicating the paper's experiments are on different branches of this Git repository:

   * `toy <https://github.com/Utah-Math-Data-Science/RpCN/tree/toy>`_: Code for **Section 5.1: Synthetic Expermiment**.
   * `image_gen <https://github.com/Utah-Math-Data-Science/RpCN/tree/image_gen>`_: Code for **Section 5.2: Conditional Image Generation**.
   * `dynamical-systems <https://github.com/Utah-Math-Data-Science/RpCN/tree/dynamical-systems>`_: Code for **Section 5.3: Dynamical Systems**.
   * `inverse_problems <https://github.com/Utah-Math-Data-Science/RpCN/tree/inverse_problems>`_: Code for **Section 5.4: Inverse Problems**.

Library Installation
====================

Clone the repository and install using ``pip`` or ``uv``:

.. code:: bash

   pip install .

.. warning::

   The dependency versions for PyTorch and Jax have not been specified to maximize compatibility potential without verification of compatibility.
   We hope it is compatible with the versions of PyTorch or Jax that your project is using.

Usage
====

Test-time guidance for diffusion models: FK steering, DAS, CREPE and RpCN in one codebase, written against
the Array API standard (2025.12). It runs on JAX, PyTorch (through `array-api-compat`) and `array_api_strict`.
The specification is `../impl.md`.

## Commands

```sh
uv sync --all-extras                       # set up (JAX and PyTorch extras)
uv run pytest                              # fast suite (Hypothesis, `default` profile)
uv run pytest --hypothesis-profile=ci      # fast suite, `ci` profile (200 examples, derandomized)
uv run pytest -m slow                      # statistical tests (§18.4); float64 on JAX or PyTorch
```

`uv sync` without extras, followed by `uv run pytest`, runs the suite on `array_api_strict` only, with the
JAX and PyTorch tests skipped.

## Layout

```
src/ttg/
  backend/      Backend protocol; jax.py, torch.py, strict.py (randomness, scan, cond, switch)
  ops.py        Array-API helpers and the pure-Python pytree map
  schedule.py   VPLinear, EDM, RectifiedFlow
  records.py    Feature, Point, Record
  evaluate.py   evaluation contract, conversions, JAX/PyTorch builders, evaluate_rows, start_record
  kernels.py    Gaussian, transition, Noising, PosteriorStep, DDIMStep, EulerMaruyama, Langevin, Guided
  sde.py        Direction, ForwardSDE, ReverseSDE
  guidance.py   tilt, normalized
  lam.py        constant, ladder, tabulated, piecewise_linear, das_lambda, level_lambda
  potentials.py FKDiff, FKMax, FKSum, TiltEnds, ReverseRatio, Sum, Zero
  resample.py   ess, systematic, masked_resample; Never, Every, ESSBelow
  moves.py      MoveContext, PCN, LangevinMove
  exchange.py   SequentialSweep, EvenOdd, StateSwap, replica_exchange
  smc.py        smc()
  crepe.py      CREPEConfig, PathExchange, crepe_init, crepe()
  testing/      test doubles, the Gaussian toy model, Hypothesis strategies
tests/          one file per property group of §18.3, plus test_statistical.py (§18.4)
```
