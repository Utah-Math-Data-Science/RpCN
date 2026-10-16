# ttg

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
