# Test-time guidance for diffusion models: implementation specification

FK steering, DAS, CREPE and RpCN, one codebase across JAX and PyTorch (and `array_api_strict` for tests).

All code below is **pseudocode in Python syntax**. It is meant to pin down interfaces and dataflow, not to be pasted in as-is. Where it is concrete (formulas, signatures, the wiring rules in §4.1), follow it exactly.

---

## 0. Scope and build plan

**This spec is the complete source of truth.** Implement exactly what it says. Don't consult papers or outside knowledge of these methods. If something is ambiguous or inconsistent, ask.

**Requirements** (declare these as lower bounds in `pyproject.toml`; `uv.lock` pins exact versions):

| package | minimum | why |
|---|---|---|
| Python | 3.10 | PEP 604 union annotations |
| `array-api-compat` | 1.14 | targets Array API 2025.12 |
| `array-api-strict` | 2.5 | 2025.12 is its default |
| `hypothesis` | a release with `hypothesis.extra.array_api` | §18 |
| `jax` (optional extra) | a release with `jax.tree_util.register_dataclass(cls, data_fields, meta_fields)` | §4.3 |
| `torch` (optional extra) | 2.x | via `array_api_compat.torch` |

**Packaging: a `uv` library package.**

- **Create it** with `uv init --lib ttg`. That gives a `src/ttg/` layout (organised as in §3) and a `pyproject.toml` with uv's build backend. Put `tests/` at the project root.
- **Dependencies:**
  - **Runtime:** `uv add array-api-compat`.
  - **Optional extras:** `uv add --optional jax jax` and `uv add --optional torch torch`. These become `[project.optional-dependencies]`.
  - **Development:** `uv add --dev array-api-strict hypothesis pytest`. This is the `dev` dependency group, and `ttg.testing` may import from it, since only tests import `ttg.testing`.
- **`requires-python = ">=3.10"`.** Commit `uv.lock`.
- **Commands:**
  - `uv sync --all-extras` to set up;
  - `uv run pytest` for the fast suite;
  - `uv run pytest -m slow` for §18.4;
  - `uv run pytest --hypothesis-profile=ci` for the `ci` profile.
- **pytest configuration** in `pyproject.toml`, under `[tool.pytest.ini_options]`:
  - register the `slow` marker;
  - set `addopts = "-m 'not slow'"`, so the plain command skips §18.4. A later `-m slow` on the command line overrides it.
- **Strict-only check:** `uv sync` without extras followed by `uv run pytest` must also pass, with JAX and PyTorch tests skipped.
- **Don't configure a torch index** (CPU or CUDA wheels) in `pyproject.toml`. That choice belongs to the machine.

**Build order.** Each stage is done when its tests pass on every installed backend (`strict` always; `jax` and `torch` when present), under the `ci` Hypothesis profile.

| stage | modules | done when these pass |
|---|---|---|
| 1 | `backend/`, `ops.py`, `schedule.py`, `records.py`, `evaluate.py`, `kernels.py`, `sde.py`, `guidance.py`, `lam.py`, `testing/` (toy model, doubles) | S1–S5, V1–V2, K1–K17, L1, C1–C2, N2–N3 |
| 2 | `potentials.py`, `resample.py`, `smc.py` (FK, DAS) | P1–P6, R1–R5, E1–E4, N1, N4 |
| 3 | `moves.py`, `exchange.py` (FK + pCN, RpCN) | M1–M7, E5, §17 counts with moves |
| 4 | `crepe.py` | X1–X8 |
| 5 | statistical tests (§18.4) | every row passes; the $\lambda = 0$ sanity check fails as intended |

**Where to start reading:** §1 (invariants), §4.1 (wiring rules), then each module's section in build order.

## 1. Invariants

Every design choice below exists to keep these true.

| # | invariant |
|---|---|
| I1 | The network is never evaluated twice on the same $(x, t)$. Every evaluation goes through `evaluate_rows`, with the features chosen by the runner. Moves evaluate proposals, which are new points. |
| I2 | No memoization. Evaluation records are passed by hand. |
| I3 | All per-particle (per-replica) state is one pytree. Resampling, swapping and accept/reject act on every leaf at once. |
| I4 | A runner knows **who** consumes each point, never **what** they consume. Consumers (kernels, potentials, moves) declare `requires`, and the runner requests the union. |
| I5 | The runner's only use of the schedule is $u \mapsto t$. It passes $t$ around and does nothing else with it. The schedule owns all edge cases: $m_t > 0$ and $s_t > 0$ on its whole range. |
| I6 | Every proposal kernel has strictly positive variance. |
| I7 | Control flow never branches on array *values*. Step-index-dependent branches use `backend.cond` or `backend.switch`. Step functions are pure and take explicit PRNG keys, so the same code runs under JAX `jit`/`scan` and in an eager PyTorch loop. |
| I8 | One implementation wherever one API covers another. The SMC runner serves FK, DAS, RpCN, CREPE's paths and CREPE's initialisation. One replica-exchange driver serves RpCN and CREPE. |
| I9 | No result is formed by subtracting two large, nearly equal numbers that the library could have avoided forming. Weights add per-step `increment`s, not differences of cumulative tilts. Density ratios use `Gaussian.log_ratio`, not two `log_prob`s. Transition variances use `log_snr_diff`. The library runs in the model's dtype; float64 appears only as a test oracle (§18). |

## 2. Notation

| symbol | meaning | code |
|---|---|---|
| $u \in [0, 1]$ | runner-facing index; $u = 0$ is the schedule's last (cleanest) level, $u = 1$ the reference end | `u` |
| $t = \tau(u)$ | the model's native time (e.g. $t = \sigma$ for EDM) | `t = schedule.time(u)` |
| $m_t,\ s_t$ | $x_t = m_t x_0 + s_t \varepsilon$ | `mean(t)`, `std(t)` |
| $\dot m_t,\ \dot s_t$ | derivatives with respect to $t$ | `dt_mean(t)`, `dt_std(t)` |
| $\mathrm{lsnr}_t = \log(m_t^2 / s_t^2)$ | log signal-to-noise ratio, strictly decreasing in $t$ | `log_snr(t)`; differences `log_snr_diff(t_a, t_b)` |
| $\hat x_0,\ \hat\varepsilon,\ \nabla \log p_t$ | denoiser outputs, related by $x = m_t \hat x_0 + s_t \hat\varepsilon$ and $\nabla \log p_t \approx -\hat\varepsilon / s_t$ | `X0`, `EPS`, `SCORE` |
| $\hat r_t(x) = r(\hat x_0(x)),\ g_t(x) = \nabla_x \hat r_t(x)$ | reward and its gradient through the network | `REWARD`, `REWARD_GRAD` |
| $F^P,\ B^P$ | model kernels: exact noising, and the model's denoising step | `cfg.model_noising` = `Noising(schedule)`, `cfg.model_denoising` = `PosteriorStep` / `DDIMStep` / `ReverseDiffusionStep` / `EulerMaruyama(ReverseSDE(schedule), Direction.BACKWARD)` |
| $F^Q,\ B^Q$ | proposal kernels: $F^Q$ in the **noising** direction, $B^Q$ in the **denoising** direction | `cfg.proposal_noising`, `cfg.proposal_denoising` (any `Kernel`) |
| $\lambda(t),\ \ell_t = \lambda(t)\,\hat r_t$ | tilt strength and log-tilt; may also depend on ladder position | `lam` |
| $C_t$ | cumulative log-tilt $\log \prod_{s \ge t} G_s$ along a particle's history | `potential.log_tilt(state)` |
| $I,\ N$ | independent instances; particles (or ladder positions) per instance | leading axes `[I, N, ...]`; CREPE's `n_instances` |
| $S$ | grid points in an SMC run | |
| $M,\ K$ | CREPE (§16): level gaps, fine steps per gap. RpCN (§15, §17): pCN steps per steered step, top ladder index ($N = K + 1$) | CREPE: `cfg.n_gaps`, `cfg.steps_per_gap`; RpCN: `PCN(n_proposals=M)` |
| $\beta_k,\ \rho_k$ | RpCN inverse temperatures, pCN step sizes | `ladder(betas)`, `PCN(step_size=...)` |

For two times $t < s$:

$$
m_{s|t} = \frac{m_s}{m_t}, \qquad \sigma^2_{s|t} = s_s^2 - m_{s|t}^2 s_t^2 = -\,s_s^2\,\operatorname{expm1}\!\big(\mathrm{lsnr}_s - \mathrm{lsnr}_t\big) \;(>0 \text{ because the SNR decreases}).
$$

Always use the right-hand ($\operatorname{expm1}$) form. The left-hand form cancels catastrophically when $s \approx t$.

## 3. Module map

```
ttg/
  backend/       Backend protocol; jax.py, torch.py (via array-api-compat), strict.py (array_api_strict)
  ops.py         xp-only helpers: bcast, logsumexp, take_rows, flatten/unflatten, take/put_levels, tree_where
  schedule.py    Schedule protocol; ScheduleDefaults (log_snr, log_snr_diff); Beta, BetaDefaults, LinearBeta, CosineBeta; VP (VPLinear, VPCosine), EDM, RectifiedFlow
  records.py     Feature, Point, Record
  evaluate.py    Evaluate contract, conversions, make_evaluate builders (one per backend), on_x0
  kernels.py     Gaussian (log_prob, log_ratio); transition; Noising, PosteriorStep, DDIMStep, ReverseDiffusionStep, EulerMaruyama, Langevin, Guided
  sde.py         Direction (FORWARD, BACKWARD); SDE protocol; ForwardSDE, ReverseSDE; drift_coeff, diffusion_sq_coeff
  guidance.py    guidance scales: tilt, normalized
  lam.py         constant, ladder, tabulated, piecewise_linear, das_lambda, level_lambda
  potentials.py  TiltDifference (default increment); Zero, FKDiff, FKMax, FKSum, TiltEnds, ReverseRatio, Sum, Relative
  resample.py    ess, systematic, categorical, masked_resample; policies Never, Every, ESSBelow
  moves.py       MoveContext, rescore, tilt_delta, PCN, LangevinMove
  exchange.py    PairSchedule (SequentialSweep, EvenOdd), Exchange (StateSwap), replica_exchange
  smc.py         smc()
  crepe.py       CREPEConfig, PathExchange, crepe_init, crepe(), pool_samples, finish_needs, crepe_finish, crepe_sample
  testing/       CountingNetwork, AuditingEvaluate, ReplayKernel, toy Gaussian model, strategies
```

## 4. Backend

### 4.1 Wiring rules

These rules decide how every module gets its array namespace, dtype, device and pytree behaviour. Every code block in this spec follows them.

| question | rule |
|---|---|
| where does `xp` come from? | **Inferred from the arguments:** `xp = array_api_compat.array_namespace(*arrays)` at the top of any function that receives arrays. In the pseudocode, a bare `xp` means exactly this. No module holds a global namespace. |
| what if a function receives no arrays? | **Constructors take `xp`, `dtype` and `device` explicitly:** `Schedule.prior(leading_shape, event_shape, *, xp, dtype, device=None)`, CREPE's `top_replica`/`crepe_init`/`crepe` (which take `dtype` and `device` and use `backend.xp`), and test builders. Nothing else creates arrays from nothing. |
| what needs `backend`? | **Only randomness, `scan`, `cond` and `switch`.** These are the operations the Array API standard doesn't cover. `backend` is passed explicitly (keyword `backend=`) to every function that needs one of them, and to nothing else. |
| pytrees | **`ops.tree_map` is pure Python** over dataclasses, tuples, lists and dicts. It skips `None` leaves and static fields, and is the same for every backend. Library dataclasses use the `@pytree` decorator (`ops.py`). It marks fields declared `field(metadata={"static": True})` as static, and registers the class with `jax.tree_util.register_dataclass(cls, data_fields, meta_fields)` if JAX is importable. PyTorch needs no registration. |
| dtype | **The run's dtype is the start's dtype** (`start.mean`, `start.x`). `smc` casts `u_grid` to it once, so `t` has it. Schedules, $\lambda$ helpers, kernels and potentials compute in the dtype of the `t`, `x` or `r` they receive. They never create float64 implicitly: `xp.zeros(shape, dtype=t.dtype, device=device_of(t))`, `xp.arange(n, dtype=t.dtype)`. Python scalars are fine in arithmetic: they adopt the array's dtype. Evaluating the network in another dtype is the user's `evaluate`'s business, but the `Record` it returns must use the run dtype. |
| device | The run's device is the start's device. `backend.normal`/`backend.uniform` take `device=`; the torch backend uses it, JAX ignores it. |
| integer and float arithmetic | **Never mix int and float arrays** (`array_api_strict` raises). Cast with `xp.astype` or create with the float `dtype`. |
| static vs traced | Python `int`/`bool` arguments named `is_last`, `static`, or documented as static drive Python control flow. The step index inside `scan` is **traced**. Anything that chooses structure from it goes through `backend.cond` or `backend.switch`. |
| scalars | Python scalars appear only in arithmetic operators and in `where`/`clip`; `jax.numpy` and `array_api_compat.torch` both accept those. Don't pass scalars to other functions (e.g. write `xp.clip(a, max=1.0)`, not `xp.minimum(1.0, a)`). |
| Array API version | **2025.12**, the current revision (Python scalars in `where`, `clip`, etc. are allowed). Requires `array-api-compat >= 1.14` and `array-api-strict >= 2.5`, where it is the default. `conftest.py` sets `array_api_strict.set_array_api_strict_flags(api_version="2025.12")` explicitly. |

### 4.2 Backend

```python
class Backend(Protocol):
    name: str                                    # "jax" | "torch" | "strict"
    xp: ModuleType                               # for constructors and tests only; library code infers xp
    def split(self, key, n_keys: int): ...       # -> array of n_keys keys (stackable, so scan can slice it)
    def normal(self, key, shape, dtype, device=None): ...
    def uniform(self, key, shape, dtype, device=None): ...
    def scan(self, f, carry, xs): ...            # -> (carry, stacked ys)
    def cond(self, pred, f_true, f_false, operand): ...
    def switch(self, index, branches, operand): ...   # index: Python int (static) or traced int scalar
    def key(self, seed: int): ...                # make a root key
```

- **JAX:**
  - keys are `jax.random` keys;
  - `scan`, `cond` and `switch` are the `lax` versions.
- **PyTorch:**
  - **Keys:** a key is an `int64` tensor seed. `split(key, n_keys)` seeds a CPU `torch.Generator` with it and draws `n_keys` new `int64` seeds. `normal` and `uniform` seed a generator on `device` with the key.
  - **`scan`** is a Python loop over the leading axis of `xs` (`tree_map`-sliced). If every step returns `ys = None`, it returns `None`; otherwise it stacks the `ys` with `tree_map(xp.stack)`.
  - **`cond` and `switch`** evaluate `pred`/`index` with `bool(...)`/`int(...)` and call one branch.
- **strict:** the same as PyTorch, using NumPy for seeds. It exists only for tests.
- **`cond` and `switch` are used only for step-index predicates** (e.g. "steer at this step", "even or odd sweep"). They evaluate only the branch taken, so moves at unsteered steps cost nothing. Masking would evaluate proposals and then throw them away.

### 4.3 Helpers (`ops.py`)

Written against the Array API standard:

```python
def pytree(cls):                                 # frozen dataclass + JAX registration if available
    cls = dataclass(frozen=True)(cls)
    if jax_available():
        data = [dataclass_field.name for dataclass_field in fields(cls) if not dataclass_field.metadata.get("static")]
        meta = [dataclass_field.name for dataclass_field in fields(cls) if dataclass_field.metadata.get("static")]
        jax.tree_util.register_dataclass(cls, data_fields=data, meta_fields=meta)
    return cls

def static(default=MISSING): return field(default=default, metadata={"static": True})

def tree_map(fn, *trees):                        # pure Python; None leaves are skipped (stay None)
    first = trees[0]
    if first is None: return None
    if is_dataclass(first):
        return replace(first, **{dataclass_field.name: tree_map(fn, *(getattr(tree, dataclass_field.name) for tree in trees))
                                 for dataclass_field in fields(first) if not dataclass_field.metadata.get("static")})
    if isinstance(first, (tuple, list)): return type(first)(tree_map(fn, *children) for children in zip(*trees))
    if isinstance(first, dict): return {k: tree_map(fn, *(tree[k] for tree in trees)) for k in first}
    return fn(*trees)                            # an array leaf

def leaves(tree):                                # the non-None array leaves, in tree_map order
    out = []; tree_map(lambda a: out.append(a) or a, tree); return out
def importable(name): return importlib.util.find_spec(name) is not None
def jax_available(): return importable("jax")
prod = math.prod
def n_levels(tree): return leaves(tree)[0].shape[1]
def device_of(a): return array_api_compat.device(a)

def bcast(v, like):                              # [L...] -> broadcastable against [L..., *D]
    return xp.reshape(v, v.shape + (1,) * (like.ndim - v.ndim))

def logsumexp(a, axis, keepdims=False):          # not in the Array API standard
    shift = xp.max(a, axis=axis, keepdims=True)
    shift = xp.where(xp.isfinite(shift), shift, 0.0)
    out = shift + xp.log(xp.sum(xp.exp(a - shift), axis=axis, keepdims=True))
    return out if keepdims else xp.squeeze(out, axis=axis)

def flatten_leading(a, n_leading):       return xp.reshape(a, (-1,) + a.shape[n_leading:])
def unflatten_leading(a, leading_shape): return xp.reshape(a, tuple(leading_shape) + a.shape[1:])

def take_rows(a, idx):                           # a [I, N, ...], idx [I, N] -> a[i, idx[i, j]]
    I, N = idx.shape
    flat = flatten_leading(a, 2)
    flat_idx = xp.reshape(xp.arange(I, dtype=idx.dtype, device=device_of(idx))[:, None] * N + idx, (-1,))
    return unflatten_leading(xp.take(flat, flat_idx, axis=0), (I, N))

def take_levels(tree, levels):                   # static list of positions on axis 1
    runs = []                                    # maximal runs of consecutive levels, as (start, stop)
    for level in levels:
        if runs and runs[-1][1] == level: runs[-1] = (runs[-1][0], level + 1)
        else: runs.append((level, level + 1))
    def take(a):                                 # slices, not an index array: no host-to-device copy (a GPU sync)
        parts = [a[:, start:stop, ...] for start, stop in runs]
        return parts[0] if len(parts) == 1 else xp.concat(parts, axis=1)
    return tree_map(take, tree)

def put_levels(tree, levels, values):            # no functional index-set in the standard:
    def put(a, b):                               # rebuild axis 1 statically
        columns = [b[:, levels.index(level)] if level in levels else a[:, level] for level in range(a.shape[1])]
        return xp.stack(columns, axis=1)
    return tree_map(put, tree, values)

def tree_where(mask, a, b):                      # mask [I] or [I, P]; a, b must have identical structure
    return tree_map(lambda on_true, on_false: xp.where(bcast(mask, on_true), on_true, on_false), a, b)

def stack_levels(levels):                        # list of trees with axis 1 of size 1 -> one tree
    return tree_map(lambda *a: xp.concat(a, axis=1), *levels)

def steps_first(t_grid, leading_shape):          # [S] or [*leading_shape, S] -> [S, *leading_shape]
    if t_grid.ndim == 1:
        return xp.broadcast_to(xp.reshape(t_grid, (-1,) + (1,) * len(leading_shape)),
                               (t_grid.shape[0], *leading_shape))
    return xp.moveaxis(t_grid, -1, 0)

def leading_shape_of(start):                     # Gaussian -> var.shape; Point / Record -> t.shape
    return start.var.shape if isinstance(start, Gaussian) else start.t.shape
```

**Array-API gaps to check per backend:**
- `cumulative_sum`;
- `take` with `axis`.

Nothing uses `searchsorted` (some backends accept only a 1-D sorted first argument and contiguous needles). Sorted lookups count instead: the index of `v` in sorted `a` (side `"left"`) is `sum(a < v)` over `a`'s axis. Systematic and categorical resampling (§10) and `piecewise_linear` (§8) all do this.

## 5. Schedule

```python
class Schedule(Protocol):
    def time(self, u): ...           # u in [0, 1] -> native t; strictly increasing
    def mean(self, t): ...           # m_t > 0 on time([0, 1])
    def std(self, t): ...            # s_t > 0 on time([0, 1])   (so u = 0 is NOT s = 0)
    def dt_mean(self, t): ...        # d m_t / dt
    def dt_std(self, t): ...         # d s_t / dt
    def prior(self, leading_shape, event_shape, *, xp, dtype, device=None) -> Gaussian: ...
    # approx. p at time(1); pCN needs mean and variance. event_shape: tuple, e.g. (C, H, W)
    # all methods except prior take arrays and compute in their dtype (§4.1)
    def log_snr(self, t): ...        # lsnr_t = log(m_t^2 / s_t^2); default from mean/std
    def log_snr_diff(self, t_a, t_b): ...   # lsnr(t_a) - lsnr(t_b); default log_snr(t_a) - log_snr(t_b)

class ScheduleDefaults:              # mixin: the two optional methods
    def log_snr(self, t):
        return 2 * (xp.log(self.mean(t)) - xp.log(self.std(t)))
    def log_snr_diff(self, t_a, t_b):    # accurate only to ~eps * |lsnr|; override for tiny gaps
        return self.log_snr(t_a) - self.log_snr(t_b)
```

**`log_snr_diff`.** The defaults are correct for any schedule. Where a schedule can write the difference as a closed form in $s - t$, override it (as below): the default loses accuracy on very fine grids.

The schedule owns its edge cases by choosing $\tau$'s range. Implement these three families, `VP` with any noise rate `Beta`, `EDM` and `RectifiedFlow`. `VP`'s `log_snr_diff` is as accurate as `beta.integral_diff`, so a user-written `Beta` should override it where it has a closed form in $t_a - t_b$:

```python
class Beta(Protocol):                # noise rate of a VP schedule: alpha_bar_t = exp(-int_0^t beta)
    def rate(self, t): ...           # beta(t)
    def integral(self, t): ...       # int_0^t beta = -log alpha_bar_t
    def integral_diff(self, t_a, t_b): ...   # integral(t_a) - integral(t_b); default by subtraction

class BetaDefaults:                  # mixin
    def integral_diff(self, t_a, t_b):   # accurate only to ~eps * |integral|; override for tiny gaps
        return self.integral(t_a) - self.integral(t_b)

@dataclass(frozen=True)
class LinearBeta(BetaDefaults):      # beta(t) = beta_0 + t (beta_1 - beta_0)
    beta_0: float = 0.1; beta_1: float = 20.0
    def rate(self, t):     return self.beta_0 + t * (self.beta_1 - self.beta_0)
    def integral(self, t): return self.beta_0 * t + 0.5 * (self.beta_1 - self.beta_0) * t**2
    def integral_diff(self, t_a, t_b):
        return (t_a - t_b) * (self.beta_0 + 0.5 * (self.beta_1 - self.beta_0) * (t_a + t_b))

@dataclass(frozen=True)
class CosineBeta(BetaDefaults):      # alpha_bar_t = cos^2(theta_t) / cos^2(theta_0), theta_t = pi/2 (t + o) / (1 + o)
    offset: float = 0.008            # alpha_bar_1 = 0: use with t_max < 1
    def _tan_theta(self, t):         # sin(theta) / sin(pi/2 - theta): both sines are well conditioned on [0, 1)
        theta = 0.5 * math.pi * (t + self.offset) / (1 + self.offset)
        phi = 0.5 * math.pi * (1 - t) / (1 + self.offset)            # pi/2 - theta, accurate near t = 1
        return xp.sin(theta) / xp.sin(phi)
    def rate(self, t):     return math.pi / (1 + self.offset) * self._tan_theta(t)
    def integral(self, t): return self.integral_diff(t, xp.zeros_like(t))   # two log cos would cancel near t = 0
    def integral_diff(self, t_a, t_b):
        delta = 0.5 * math.pi * (t_a - t_b) / (1 + self.offset)   # theta_a - theta_b
        # cos(theta_b + delta) / cos(theta_b) = 1 - 2 sin^2(delta / 2) - tan(theta_b) sin(delta)
        return -2 * xp.log1p(-2 * xp.sin(0.5 * delta) ** 2 - self._tan_theta(t_b) * xp.sin(delta))

@dataclass(frozen=True)
class VP(ScheduleDefaults):          # alpha_bar = exp(-int_0^t beta), for any noise rate beta
    beta: Beta; t_min: float = 1e-3; t_max: float = 1.0   # t_max must keep alpha_bar > 0
    def time(self, u):    return self.t_min + u * (self.t_max - self.t_min)
    def mean(self, t):    return xp.exp(-0.5 * self.beta.integral(t))
    def std(self, t):     return xp.sqrt(-xp.expm1(-self.beta.integral(t)))
    def dt_mean(self, t): return -0.5 * self.beta.rate(t) * self.mean(t)
    def dt_std(self, t):  return 0.5 * self.beta.rate(t) * xp.exp(-self.beta.integral(t)) / self.std(t)
    def prior(self, leading_shape, event_shape, *, xp, dtype, device=None):
        return Gaussian(xp.zeros((*leading_shape, *event_shape), dtype=dtype, device=device),
                        xp.ones(leading_shape, dtype=dtype, device=device))
    def log_snr(self, t):
        log_alpha_bar = -self.beta.integral(t)
        return log_alpha_bar - xp.log(-xp.expm1(log_alpha_bar))
    def log_snr_diff(self, t_a, t_b):
        log_alpha_bar_diff = -self.beta.integral_diff(t_a, t_b)     # log ab_a - log ab_b
        log_alpha_bar_b = -self.beta.integral(t_b)
        return log_alpha_bar_diff - xp.log1p(xp.exp(log_alpha_bar_b) * xp.expm1(log_alpha_bar_diff)
                                             / xp.expm1(log_alpha_bar_b))   # minus log((1 - ab_a) / (1 - ab_b))

def VPLinear(beta_0=0.1, beta_1=20.0, t_min=1e-3, t_max=1.0): return VP(LinearBeta(beta_0, beta_1), t_min, t_max)
def VPCosine(offset=0.008, t_min=1e-3, t_max=0.999):         return VP(CosineBeta(offset), t_min, t_max)

@dataclass(frozen=True)
class EDM(ScheduleDefaults):         # t = sigma; non-uniform spacing lives in time(u)
    s_min: float = 0.002; s_max: float = 80.0; rho: float = 7.0
    def time(self, u):
        root_min, root_max = self.s_min**(1/self.rho), self.s_max**(1/self.rho)
        return (root_min + u*(root_max - root_min))**self.rho
    def mean(self, t):    return xp.ones_like(t)
    def std(self, t):     return t
    def dt_mean(self, t): return xp.zeros_like(t)
    def dt_std(self, t):  return xp.ones_like(t)
    def prior(self, leading_shape, event_shape, *, xp, dtype, device=None):
        return Gaussian(xp.zeros((*leading_shape, *event_shape), dtype=dtype, device=device),
                        xp.full(leading_shape, self.s_max**2, dtype=dtype, device=device))
    def log_snr(self, t): return -2 * xp.log(t)
    def log_snr_diff(self, t_a, t_b): return -2 * xp.log1p((t_a - t_b) / t_b)

@dataclass(frozen=True)
class RectifiedFlow(ScheduleDefaults):  # x_t = (1 - t) x0 + t eps, clipped to [t_min, t_max] inside (0, 1)
    t_min: float = 1e-3; t_max: float = 0.999
    def time(self, u):    return self.t_min + u * (self.t_max - self.t_min)
    def mean(self, t):    return 1 - t
    def std(self, t):     return t
    def dt_mean(self, t): return -xp.ones_like(t)
    def dt_std(self, t):  return xp.ones_like(t)
    def prior(self, leading_shape, event_shape, *, xp, dtype, device=None):   # ignores the (1 - t_max) x0 part
        return Gaussian(xp.zeros((*leading_shape, *event_shape), dtype=dtype, device=device),
                        xp.full(leading_shape, self.t_max**2, dtype=dtype, device=device))
    def log_snr(self, t): return 2 * (xp.log1p(-t) - xp.log(t))
    def log_snr_diff(self, t_a, t_b): return 2 * (xp.log1p(-(t_a - t_b) / (1 - t_b)) - xp.log1p((t_a - t_b) / t_b))
```

`VPLinear`, `VPCosine`, `EDM` and `RectifiedFlow` are used by the tests (`SCHEDULES`, §18).

**Final denoising** is the default. `u = 0` maps to `t_min`, not to clean data, so `smc`, `crepe_finish` and `CREPEConfig` all default `out_needs` to `{X0}`, and `final.x0` gives $\hat x_0$ at the last point. This costs nothing extra, because that point is evaluated anyway. Pass `out_needs=frozenset()` to keep only `x` and the features the run itself needs.

## 6. Records and the evaluation contract

```python
class Feature(Enum):
    X0 = "x0"; EPS = "eps"; SCORE = "score"; REWARD = "reward"; REWARD_GRAD = "reward_grad"

@pytree
class Point:                         # unevaluated
    x: Array                         # [L..., *D]
    t: Array                         # [L...]

@pytree
class Record:                        # evaluated; absent features are None
    x: Array; t: Array
    x0: Array | None = None; eps: Array | None = None; score: Array | None = None
    reward: Array | None = None; reward_grad: Array | None = None
    def project(self, needs) -> "Record":   # keep exactly `needs` (C2's structure); static
        return replace(self, **{f.value: None for f in Feature if f not in needs})
    def need(self, feature: Feature) -> Array:
        value = getattr(self, feature.value)
        if value is None:
            raise MissingFeature(feature)  # never silently re-evaluate
        return value
    def provides(self, needs) -> bool:   # static: depends only on pytree structure
        return all(getattr(self, f.value) is not None for f in needs)

class MissingFeature(KeyError): ...      # a consumer read a feature nobody requested: a requires bug
class PartialRecord(ValueError): ...     # a Record start lacks features the run needs
class DuplicateEvaluation(AssertionError): ...   # testing only: I1 violated
```

**The contract.** User-supplied, or built by a helper:

```python
Evaluate = Callable[[Array, Array, frozenset[Feature]], Record]
# evaluate(x [B, *D], t [B], needs) -> Record with leading axis [B]
#  (C1) calls the network AT MOST ONCE per call (CFG: one batched forward over [cond, uncond]);
#  (C2) returns EXACTLY the requested features (all others None). This keeps pytree
#       structure predictable, which tree_where and stacking require;
#  (C3) rows are independent: row b's outputs depend only on (x[b], t[b]).
#       This is what makes per-row gradients via grad(sum(r)) correct. Beware BatchNorm
#       in training mode, or anything else that mixes rows.
# `needs` is static (hashable) and may drive Python-level branching inside evaluate.
```

**Conversion helpers.** Users may also do these conversions themselves.

```python
def x0_from_eps(x, t, eps, schedule):     return (x - bcast(schedule.std(t), x) * eps) / bcast(schedule.mean(t), x)
def eps_from_x0(x, t, x0, schedule):      return (x - bcast(schedule.mean(t), x) * x0) / bcast(schedule.std(t), x)
def score_from_eps(x, t, eps, schedule):  return -eps / bcast(schedule.std(t), x)
def eps_from_score(x, t, score, schedule): return -bcast(schedule.std(t), x) * score
def x0_eps_from_velocity(x, t, velocity, schedule):   # velocity ~ dm/dt x0 + ds/dt eps
    m, s, dm, ds = (bcast(f(t), x) for f in (schedule.mean, schedule.std, schedule.dt_mean, schedule.dt_std))
    denom = m * ds - dm * s                      # = m^2 d(s/m)/dt != 0
    return (ds * x - s * velocity) / denom, (m * velocity - dm * x) / denom
```

**Builders.** These are optional; users may write `evaluate` directly. There is one builder per backend that has autodiff. `array_api_strict` has no autodiff, so on it `REWARD_GRAD` comes only from the analytic toy model (§18.2). Builder for JAX:

```python
def make_evaluate_jax(network, to_x0_eps, schedule, reward):
    def evaluate(x, t, needs):
        want_grad = Feature.REWARD_GRAD in needs
        want_reward = want_grad or Feature.REWARD in needs
        def forward(x):
            x0, eps = to_x0_eps(x, t, network(x, t), schedule)  # the single forward pass
            reward_value = reward(x0, x, t) if want_reward else None
            return (xp.sum(reward_value) if want_grad else 0.0), (x0, eps, reward_value)
        if want_grad:
            (_, (x0, eps, reward_value)), reward_grad = jax.value_and_grad(forward, has_aux=True)(x)
        else:
            _, (x0, eps, reward_value) = forward(x); reward_grad = None
        keep = lambda feature, value: value if feature in needs else None
        return Record(x, t, x0=keep(X0, x0), eps=keep(EPS, eps),
                      score=keep(SCORE, score_from_eps(x, t, eps, schedule)),
                      reward=keep(REWARD, reward_value), reward_grad=keep(REWARD_GRAD, reward_grad))
    return evaluate

def make_evaluate_torch(network, to_x0_eps, schedule, reward):
    def evaluate(x, t, needs):
        want_grad = Feature.REWARD_GRAD in needs
        want_reward = want_grad or Feature.REWARD in needs
        with torch.enable_grad() if want_grad else torch.no_grad():
            x_tracked = x.detach().requires_grad_(want_grad)
            x0, eps = to_x0_eps(x_tracked, t, network(x_tracked, t), schedule)   # the single forward pass
            reward_value = reward(x0, x_tracked, t) if want_reward else None
            reward_grad = torch.autograd.grad(reward_value.sum(), x_tracked)[0] if want_grad else None
        det = lambda value: None if value is None else value.detach()
        keep = lambda feature, value: det(value) if feature in needs else None
        return Record(x, t, x0=keep(X0, x0), eps=keep(EPS, eps),
                      score=keep(SCORE, score_from_eps(x, t, eps, schedule)),
                      reward=keep(REWARD, reward_value), reward_grad=keep(REWARD_GRAD, reward_grad))
    return evaluate
```

`reward(x0, x, t)` maps `x0 [B, *D]`, `x [B, *D]`, `t [B]` to `[B]`. Gradients flow through both `x0` (that is, through the network) and `x`, so a reward may use the denoiser's estimate, the noisy state, the time, or any mix. The common case is a reward of the estimate alone; wrap it with

```python
def on_x0(reward_fn):             # reward of the denoiser's estimate only
    return lambda x0, x, t: reward_fn(x0)
```

A separate reward model (a classifier, an aesthetic scorer, ...) may be called inside `reward`; contract C1 counts only calls to the diffusion network. The PyTorch builder detaches everything it returns; JAX values carry no graph. The library never differentiates through a `Record`.

**Runner-side helpers:**

```python
def evaluate_rows(evaluate, point: Point, needs, backend):   # [I, N] <-> [B]; the only call site of evaluate
    leading_shape = point.t.shape
    record = evaluate(flatten_leading(point.x, len(leading_shape)), xp.reshape(point.t, (-1,)), frozenset(needs))
    return tree_map(lambda a: unflatten_leading(a, leading_shape), record)

def start_record(start, t_start, needs, evaluate, backend, key):   # -> (Record, reference Gaussian or None)
    if isinstance(start, Gaussian):                           # e.g. schedule.prior(...): sample it, keep it
        return evaluate_rows(evaluate, Point(start.sample(key, backend), t_start), needs, backend), start
    if isinstance(start, Point):
        return evaluate_rows(evaluate, start, needs, backend), None
    if not start.provides(needs):
        raise PartialRecord(needs)                            # never top up (would re-evaluate)
    return start.project(needs), None                         # extras dropped: scan's carry must match C2
```

- **Start times aren't checked** (that would be a value comparison). A `Point` or `Record` start must already be at $t = \tau(u_0)$. A `Gaussian` start is meant for $u_0 = 1$.
- **`smc` needs $S \ge 2$ grid points.**

## 7. Kernels

```python
@pytree
class Gaussian:                       # isotropic, per row; a pytree, so it can travel in MoveContext
    mean: Array                       # [L..., *D]
    var: Array                        # [L...], must be > 0 (I6)
    def sample(self, key, backend):
        noise = backend.normal(key, self.mean.shape, self.mean.dtype, device=device_of(self.mean))
        return self.mean + bcast(xp.sqrt(self.var), self.mean) * noise
    def log_prob(self, x):
        event_axes = tuple(range(self.var.ndim, x.ndim)); event_size = prod(x.shape[self.var.ndim:])
        return (-0.5 * xp.sum((x - self.mean) ** 2, axis=event_axes) / self.var
                - 0.5 * event_size * xp.log(2 * xp.pi * self.var))
    def log_ratio(self, other, x):    # log self(x) - log other(x), without forming either log-density
        event_axes = tuple(range(self.var.ndim, x.ndim)); event_size = prod(x.shape[self.var.ndim:])
        mean_diff = self.mean - other.mean        # exact where the means are close (Sterbenz)
        centered = x - other.mean                 # the sampled noise when x ~ other
        var_diff = self.var - other.var           # exactly 0 when the variances are equal
        quadratic = xp.sum(mean_diff * (centered - 0.5 * mean_diff), axis=event_axes) / self.var
        return (quadratic + 0.5 * xp.sum(centered**2, axis=event_axes) * var_diff / (self.var * other.var)
                          - 0.5 * event_size * xp.log1p(var_diff / other.var))

def transition(schedule, t_src, t_dst):   # t_src < t_dst: (m_{s|t}, sigma^2_{s|t}), cancellation-free (§2)
    return (schedule.mean(t_dst) / schedule.mean(t_src),
            -schedule.std(t_dst) ** 2 * xp.expm1(schedule.log_snr_diff(t_dst, t_src)))

class Kernel(Protocol):
    requires: frozenset[Feature]
    def at(self, record, t_dst) -> Gaussian: ...   # every kernel goes one way only
    # t_dst has record.t's shape [L...] (one destination time per row); the result is per row
```

| kernel | direction | distribution, from source $x$ at $s$ to $t$ | `requires` |
|---|---|---|---|
| `Noising(schedule)` = $F^P$ | forward ($t > s$) | $\mathcal N\!\big(m_{t\mid s} x,\ \sigma^2_{t\mid s}\big)$ | none |
| `PosteriorStep(schedule)` | backward | $\mathcal N\!\big(\tfrac{m_{s\mid t} s_t^2}{s_s^2} x + \tfrac{m_t \sigma^2_{s\mid t}}{s_s^2}\hat x_0,\ \tau^2\big)$, $\tau^2 = \tfrac{\sigma^2_{s\mid t} s_t^2}{s_s^2}$ | `X0` |
| `DDIMStep(schedule, eta)`, $0 < \eta \le 1$ | backward | $\mathcal N\!\big(m_t \hat x_0 + \sqrt{s_t^2 - \eta^2 \tau^2}\,\hat\varepsilon,\ \eta^2 \tau^2\big)$ | `X0`, `EPS` |
| `ReverseDiffusionStep(schedule)` | backward | $\mathcal N\!\big((2 - m_{s\mid t})\,x + \sigma^2_{s\mid t} \nabla \log p_s,\ \sigma^2_{s\mid t}\big)$ with $\nabla \log p_s = (m_s \hat x_0 - x)/s_s^2$, i.e. mean $(2 - m_{s\mid t} - v)\,x + m_s v\,\hat x_0$, $v = \sigma^2_{s\mid t}/s_s^2$ | `X0` |
| `EulerMaruyama(sde, direction)` | `Direction.FORWARD` or `Direction.BACKWARD` | $\mathcal N\!\big(x + \Delta\, b(x, s),\ \sigma^2_{\mathrm{sde}}(s)\,\lvert\Delta\rvert\big)$, signed $\Delta = t - s$; drift $b$ and diffusion $\sigma^2_{\mathrm{sde}}$ from the injected `SDE` | `sde.requires` |
| `Langevin(step_size)`, $h$ = `step_size`, a callable of $t$ | stays at $s$ | $\mathcal N\!\big(x + h(s) \nabla \log p,\ 2h(s)\big)$ | `SCORE` |
| `Guided(base, scale)` | as base | base mean $+\ \kappa\, g$ with $\kappa = \texttt{scale}(t, g, \mathrm{Var}_{\text{base}})$, $g$ **at the source**; same variance | base + `REWARD_GRAD` |

- **Coefficients:** the drift and diffusion come from the schedule, $f = \dot m / m$ and $\gamma^2 = 2 s \dot s - 2 f s^2$. For VP this gives $\gamma^2 = \beta$.
- **`EulerMaruyama` is one class for both directions, and the drift is injected.** It takes one Euler–Maruyama step of an `SDE`, $\mathrm dx = b(x, t)\,\mathrm dt + \sigma_{\mathrm{sde}}(t)\,\mathrm dW$, written in forward-time convention, from $s$ to $t$ with signed $\Delta = t - s$.
  - **The SDE supplies both drift and diffusion.** The drift is evaluated at the source and declares its own `requires`.
  - **The integration direction is the only other choice.** `Direction.BACKWARD` integrates with $\Delta < 0$, as for a reverse-time SDE. `Direction.FORWARD` integrates with $\Delta > 0$.
  - **The switch fixes the sign used in the variance**, $\sigma^2_{\mathrm{sde}} \cdot \mathrm{sign} \cdot \Delta$. A kernel used in the wrong direction therefore gets a non-positive variance and fails test K8, rather than silently running backwards.
- **Provided SDEs** (§7a): `ForwardSDE` (the noising SDE, $b = f x$, no network) and `ReverseSDE(eta)`. Anything else (for example a tempering drift) is a user-written `SDE`. `EulerMaruyama(ForwardSDE(schedule), Direction.FORWARD)` is the Euler–Maruyama approximation of noising, which differs from the exact `Noising` kernel.
- **`Guided` is one class for every gradient guidance.** The per-row scale $\kappa = \texttt{scale}(t, g, \mathrm{Var})$ sees the gradient and the base kernel's variance (§8a). $t$ is the **destination** time, so $\lambda$ is the one of the target the step proposes for, while $g$ is taken at the source:
  - `tilt(lam)` gives $\kappa = \mathrm{Var} \cdot \lambda(t)$: the exact Gaussian tilt by $e^{\lambda \langle g, x \rangle}$. Used by DAS and CREPE's $B^Q$.
  - `tilt(lam, 0.5)` gives Langevin's reward drift $h\lambda g$, since $\mathrm{Var} = 2h$.
  - `normalized(step_length)` gives $\kappa = \gamma(t)/\lVert g \rVert$, with $\gamma$ = `step_length`: for inverse problems, with the gradient taken at the source point. Use it as **both** target and proposal: there is then no weight correction, and pCN preserves it.

**CREPE's default proposals:**

| kernel | drift | implementation | `requires` |
|---|---|---|---|
| $F^Q$ (noising) | plain noising drift $f x$ | `Noising(schedule)`, i.e. $F^Q = F^P$ | none |
| $B^Q$ (denoising) | reward-tilted reverse drift $f x - \gamma^2(\nabla \log p + \lambda g)$ | `Guided(EulerMaruyama(ReverseSDE(schedule), Direction.BACKWARD), tilt(lam))` | `SCORE`, `REWARD_GRAD` |

Use the same base discretisation for $B^P$ and $B^Q$, so that $\log B^P - \log B^Q$ reflects only the guidance shift.

Implementations:

```python
@dataclass(frozen=True)
class Noising:                    # exact forward kernel F^P
    schedule: Schedule
    requires = frozenset()
    def at(self, record, t_dst):
        mean_ratio, var = transition(self.schedule, record.t, t_dst)   # source record.t < destination t_dst
        return Gaussian(bcast(mean_ratio, record.x) * record.x, var)

@dataclass(frozen=True)
class PosteriorStep:
    schedule: Schedule
    requires = frozenset({Feature.X0})
    def at(self, record, t_dst):
        t_src = record.t                                            # per-row scalars, shape [L...]
        std_src, mean_dst, std_dst = self.schedule.std(t_src), self.schedule.mean(t_dst), self.schedule.std(t_dst)
        mean_ratio = self.schedule.mean(t_src) / mean_dst
        var_fraction = -xp.expm1(self.schedule.log_snr_diff(t_src, t_dst))   # = sigma^2_{s|t} / s_s^2, in (0, 1)
        coeff_x, coeff_x0 = mean_ratio * std_dst**2 / std_src**2, mean_dst * var_fraction
        mean = bcast(coeff_x, record.x) * record.x + bcast(coeff_x0, record.x) * record.need(Feature.X0)
        return Gaussian(mean, std_dst**2 * var_fraction)            # tau^2 = sigma^2_{s|t} s_t^2 / s_s^2

@dataclass(frozen=True)
class DDIMStep:                   # eta in (0, 1]; eta = 1 equals PosteriorStep (test K4)
    schedule: Schedule; eta: float
    requires = frozenset({Feature.X0, Feature.EPS})
    def at(self, record, t_dst):
        t_src = record.t
        var_fraction = -xp.expm1(self.schedule.log_snr_diff(t_src, t_dst))   # sigma^2_{s|t} / s_s^2
        std_dst = self.schedule.std(t_dst)
        var = self.eta**2 * std_dst**2 * var_fraction               # eta^2 tau^2
        eps_coeff = std_dst * xp.sqrt(1 - self.eta**2 * var_fraction)   # sqrt(s_t^2 - eta^2 tau^2); > 0 for eta <= 1
        mean = (bcast(self.schedule.mean(t_dst), record.x) * record.need(Feature.X0)
                + bcast(eps_coeff, record.x) * record.need(Feature.EPS))
        return Gaussian(mean, var)

@dataclass(frozen=True)
class ReverseDiffusionStep:       # Song et al. (2021) reverse diffusion sampler, with the exact forward transition
    schedule: Schedule
    requires = frozenset({Feature.X0})
    def at(self, record, t_dst):
        t_src = record.t
        mean_src, std_src = self.schedule.mean(t_src), self.schedule.std(t_src)
        mean_ratio = mean_src / self.schedule.mean(t_dst)                     # m_{s|t}
        log_snr_diff = self.schedule.log_snr_diff(t_src, t_dst)               # lsnr_s - lsnr_t < 0
        var_fraction = -xp.expm1(log_snr_diff)                                # v = sigma^2_{s|t} / s_s^2, in (0, 1)
        # 2 - m_{s|t} - v, as (1 - m_{s|t}) + (1 - v): no cancellation as v -> 1 when m_{s|t} <= 1
        coeff_x, coeff_x0 = (1 - mean_ratio) + xp.exp(log_snr_diff), mean_src * var_fraction
        mean = bcast(coeff_x, record.x) * record.x + bcast(coeff_x0, record.x) * record.need(Feature.X0)
        return Gaussian(mean, std_src**2 * var_fraction)                      # sigma^2_{s|t}

@dataclass(frozen=True)
class Langevin:                   # one unadjusted Langevin step at fixed t
    step_size: Callable           # h as a function of t, e.g. lambda t: 0.01 * schedule.std(t)**2
    requires = frozenset({Feature.SCORE})
    def at(self, record, t_dst):  # t_dst == record.t
        h = self.step_size(record.t)
        return Gaussian(record.x + bcast(h, record.x) * record.need(Feature.SCORE), 2 * h)

@dataclass(frozen=True)
class EulerMaruyama:              # one EM step of the injected SDE, integrated up or down
    sde: SDE
    direction: Direction
    @property
    def requires(self): return self.sde.requires
    def at(self, record, t_dst):
        dt = t_dst - record.t                                       # signed
        sign = float(self.direction.value)                          # static switch: +1 or -1
        mean = record.x + bcast(dt, record.x) * self.sde.drift(record)   # drift at the source
        return Gaussian(mean, self.sde.diffusion_sq(record.t) * sign * dt)   # var <= 0 if misused (K8)

@dataclass(frozen=True)
class Guided:                     # base mean + scale(t, grad, var) * grad, with grad at the source
    base: Kernel; scale: Callable
    @property
    def requires(self): return self.base.requires | {Feature.REWARD_GRAD}
    def at(self, record, t_dst):
        base_dist = self.base.at(record, t_dst); grad = record.need(Feature.REWARD_GRAD)
        return Gaussian(base_dist.mean + bcast(self.scale(t_dst, grad, base_dist.var), grad) * grad, base_dist.var)
```

`PosteriorStep` and `DDIMStep` both write $\sigma^2_{s|t}/s_s^2 = -\operatorname{expm1}(\mathrm{lsnr}_s - \mathrm{lsnr}_t)$. In `DDIMStep`, $s_t^2 - \eta^2\tau^2 = s_t^2\,\big(1 - \eta^2 \sigma^2_{s|t}/s_s^2\big)$, which is positive because $\sigma^2_{s|t} < s_s^2$ and $\eta \le 1$.

**`ReverseDiffusionStep`** is the reverse diffusion sampler of Song et al. (2021, Appendix E): the reverse SDE discretised in the same form as the forward step, $x_s = m_{s|t} x_t + \sigma_{s|t} z$, so with drift $(m_{s|t} - 1)\,x$ and noise $\sigma^2_{s|t}$. For VP ($\beta = 1 - m_{s|t}^2$) it is their Algorithm 3 predictor; for VE ($m \equiv 1$) it is Algorithm 2's, and its mean equals `PosteriorStep`'s.
- **Variance:** $s_s^2 v = \sigma^2_{s|t}$, the same expression as `transition(schedule, t, s)`. It is `PosteriorStep`'s $\tau^2$ times $s_s^2 / s_t^2$, so it lies above every `DDIMStep` variance.
- **Mean:** agrees with `PosteriorStep`'s and with `EulerMaruyama(ReverseSDE(schedule), Direction.BACKWARD)`'s to first order in the step, while the step is small next to $s_t$ and keeps $m_{s|t}$ near 1.
- **Numerics:** $2 - m_{s|t} - v$ is computed as $(1 - m_{s|t}) + e^{\mathrm{lsnr}_s - \mathrm{lsnr}_t}$, since $1 - v = e^{\mathrm{lsnr}_s - \mathrm{lsnr}_t}$. With $m_{s|t} \le 1$ (all provided schedules) both terms are non-negative, so nothing cancels when $v \to 1$ (a VE step from $s_s \gg s_t$).

Kernels and SDEs are plain frozen dataclasses (configuration, not pytrees). They never travel through `scan` or `tree_where`; the `Gaussian`s they return do.

### Density ratios: `log_ratio`, never two `log_prob`s

Every density **ratio** in the library goes through `log_ratio`: proposal corrections (§13), `ReverseRatio` terms (§9), and so CREPE's path weights (§16). With $\Delta = \mu_P - \mu_Q$ and $y = x - \mu_Q$:

$$
\log \mathcal N(x; \mu_P, v_P) - \log \mathcal N(x; \mu_Q, v_Q)
= \frac{\langle \Delta,\ y - \tfrac12 \Delta \rangle}{v_P}
+ \frac{\lVert y \rVert^2\,(v_P - v_Q)}{2\, v_P v_Q}
- \frac{D}{2}\,\operatorname{log1p}\!\Big(\frac{v_P - v_Q}{v_Q}\Big).
$$

- **Never subtract two `log_prob`s.** In high dimension each is $O(D)$, so in float32 the difference loses the $O(1)$ ratio.
- **Equal variances** (`Guided` against its base, and the same-base $B^P$/$B^Q$ pairing of §7): `var_diff` is exactly 0, so only the first term remains.
- **Use the float `mean` arrays as they are.** The density must be that of the Gaussian actually sampled.
- `log_prob` stays for single densities and tests.

### 7a. SDEs

An `SDE` is written in forward-time convention: $\mathrm dx = b(x, t)\,\mathrm dt + \sigma_{\mathrm{sde}}(t)\,\mathrm dW$. A reverse-time SDE is the same object, integrated with $\Delta < 0$.

```python
class Direction(Enum):                    # integration direction; the value is the sign of Delta = t - s
    FORWARD = 1                           # noising: t increases
    BACKWARD = -1                         # denoising: t decreases

class SDE(Protocol):
    requires: frozenset[Feature]          # what the drift reads from the source record
    def drift(self, record) -> Array: ... # b(x, t) at the source, [L..., *D]
    def diffusion_sq(self, t) -> Array: ...   # sigma_sde(t)^2, [L...]

def drift_coeff(schedule, t): return schedule.dt_mean(t) / schedule.mean(t)                  # f = m'/m
def diffusion_sq_coeff(schedule, t):                                                        # gamma^2
    return 2 * schedule.std(t) * schedule.dt_std(t) - 2 * drift_coeff(schedule, t) * schedule.std(t) ** 2

@dataclass(frozen=True)
class ForwardSDE:                         # the noising SDE whose marginals the schedule defines; no network
    schedule: Schedule
    requires = frozenset()
    def drift(self, record): return bcast(drift_coeff(self.schedule, record.t), record.x) * record.x
    def diffusion_sq(self, t): return diffusion_sq_coeff(self.schedule, t)

@dataclass(frozen=True)
class ReverseSDE:                         # same marginals, run backwards; eta = 1 is the standard reverse-time SDE
    schedule: Schedule; eta: float = 1.0  # eta > 0 required (eta = 0 is the probability-flow ODE: no density, I6)
    requires = frozenset({Feature.SCORE})
    def drift(self, record):
        score_coeff = -0.5 * (1 + self.eta**2) * diffusion_sq_coeff(self.schedule, record.t)
        return (bcast(drift_coeff(self.schedule, record.t), record.x) * record.x
                + bcast(score_coeff, record.x) * record.need(Feature.SCORE))
    def diffusion_sq(self, t): return self.eta**2 * diffusion_sq_coeff(self.schedule, t)
```

- **`ReverseSDE(eta)`** is the standard family that shares the forward marginals: drift $f x - \tfrac{1+\eta^2}{2}\gamma^2 \nabla \log p$, diffusion $\eta^2 \gamma^2$.
- **Reward guidance goes through `Guided`, not into the drift.** That keeps one path for guidance, and for an Euler–Maruyama step the two are identical: a drift term $-\gamma^2 \lambda g$ integrated downward shifts the mean by exactly $\gamma^2 \lvert\Delta\rvert \lambda g = \mathrm{Var} \cdot \lambda \cdot g$, i.e. `tilt(lam)`. A custom `SDE` may still put anything in its drift.

## 8. Tilt strength $\lambda$

The runner never sees $\lambda$. It is held by the objects that use it, which evaluate it per row from the $t$ they are handed. It may also depend on the row's **position** on axis 1 (a ladder).

```python
def _take_1d(values, indices):   # Array API `take` wants 1-D indices
    return xp.reshape(xp.take(values, xp.reshape(indices, (-1,))), indices.shape)

def constant(value):            return lambda t: xp.full_like(t, value)
def ladder(values):             # value by position on axis 1 (RpCN's beta_k, pCN step sizes); t: [I, N]
    return lambda t: xp.broadcast_to(xp.astype(values, t.dtype)[None, :], t.shape)
def tabulated(knots, values):   # exact at the knots; nearest-knot lookup (only knots are ever visited)
    return lambda t: _take_1d(xp.astype(values, t.dtype), xp.argmin(xp.abs(t[..., None] - xp.astype(knots, t.dtype)), axis=-1))
def piecewise_linear(knots, values, hold=False):   # knots: 1-D increasing
    # hold=False: extrapolate the end segments linearly; hold=True: constant beyond the end knots
    def lam(t):
        knots_t, values_t = xp.astype(knots, t.dtype), xp.astype(values, t.dtype)
        if hold:                                     # static
            t = xp.clip(t, min=knots_t[0], max=knots_t[-1])
        # searchsorted(knots_t, t) (side="left") as a count of knots below t: no reshape of a possibly strided t
        upper_index = xp.sum(xp.astype(knots_t < t[..., None], xp.int64), axis=-1)
        upper_index = xp.clip(upper_index, 1, knots_t.shape[0] - 1)
        knot_lo, knot_hi = _take_1d(knots_t, upper_index - 1), _take_1d(knots_t, upper_index)
        value_lo, value_hi = _take_1d(values_t, upper_index - 1), _take_1d(values_t, upper_index)
        return value_lo + (t - knot_lo) / (knot_hi - knot_lo) * (value_hi - value_lo)
    return lam

def das_lambda(t_grid, growth, alpha_kl):  # t_grid[0] noisiest; lambda_tilde_k = min(1, (1+growth)^k - 1); last = 1
    step_index = xp.arange(t_grid.shape[0], dtype=t_grid.dtype)   # float: no int/float mixing (§4.1)
    ramp = xp.where(step_index == t_grid.shape[0] - 1, 1.0, xp.clip((1 + growth) ** step_index - 1, max=1.0))
    return tabulated(t_grid, ramp / alpha_kl)

def level_lambda(t_levels, beta, rho):     # CREPE: beta at the data end, 0 at the noise end
    n_gaps = t_levels.shape[0] - 1
    level = xp.arange(n_gaps + 1, dtype=t_levels.dtype)
    return piecewise_linear(t_levels, beta * (1 - level / n_gaps) ** rho, hold=True)
```

`level_lambda` **holds** its end values: $\beta$ for every $t$ below level 0 and $0$ above level $M$. Below level 0 is where `crepe_finish` (§16) continues denoising, with the same `cfg.lam` and the same guided kernel `cfg.proposal_denoising`.

### 8a. Guidance scales

`Guided` takes a guidance scale $\kappa = \texttt{scale}(t, g, \mathrm{Var})$, one value per row. It extends the tilt strength $\lambda(t)$ by also seeing the gradient $g$ and the base kernel's variance. Potentials keep plain $\lambda(t)$, so DAS can pass the same `lam` to both.

```python
def event_norm(grad, lead_ndim):    # ||grad|| per row, over the event dims
    return xp.sqrt(xp.sum(grad**2, axis=tuple(range(lead_ndim, grad.ndim))))

def tilt(lam, factor=1.0):       # exact Gaussian tilt by exp(lam <grad, x>): shift = factor * Var * lam(t) * grad
    return lambda t, grad, var: factor * var * lam(t)

def normalized(step_length, norm_floor=1e-12):   # RpCN inverse problems: shift = step_length(t) * grad / ||grad||
    return lambda t, grad, var: step_length(t) / (event_norm(grad, var.ndim) + norm_floor)
```

## 9. Potentials

A potential exposes the **cumulative** log-tilt $C$ of a particle's history, for comparing whole histories (swaps). It also exposes the **increment** $C(\text{new}) - C(\text{prev})$ of one step, computed directly from the step's statistics. The runner accumulates increments, and moves compare increments over a shared history.

```python
class Potential(Protocol):
    requires: frozenset[Feature]
    def init(self, record) -> State: ...                       # state after the first point
    def update(self, state, record, *, is_last: bool) -> State: ...   # absorb the next point; `is_last` is static
    def log_tilt(self, state) -> Array: ...                    # C = log prod_{s >= t} G_s, shape [I, N]
    def increment(self, prev, state) -> Array: ...             # C(state) - C(prev), never formed as that difference
    # precondition: state = update(prev, record, is_last=...) for some record
# runner: log_weights += increment(state, update(state, new_record))

class TiltDifference:                 # mixin: the default increment, for potentials whose C stays O(one term)
    def increment(self, prev, state): return self.log_tilt(state) - self.log_tilt(prev)
```

**The runner and moves use `increment`, never `log_tilt` differences.** $C$ grows along the path for `FKMax`, `FKSum` and `ReverseRatio`, so differencing it loses precision. `FKDiff` uses the default (its $C$ is a single term). `TiltEnds` returns exactly $0$ at interior steps and exactly $\ell$ at the last.

**Rule: state is $\lambda$-free.** It stores raw statistics, and $\lambda$ is applied in `log_tilt` and `increment` from the row's time and position. That is what makes swaps correct: a state moved to another ladder position is automatically read at that position's temperature.

| potential | state | `log_tilt` | `increment(prev, state)` |
|---|---|---|---|
| `FKDiff(lam)` (also DAS, with tempered $\lambda$) | $(\hat r, t)$ | $\lambda(t)\,\hat r$ | default: $\lambda(t')\,\hat r' - \lambda(t)\,\hat r$ |
| `FKMax(lam)` | running max, $\sum$ of running maxes (`cumulative`), $\hat r$, $t$, `is_last` | $\lambda \cdot$ `cumulative`; at the last point, $\lambda \hat r$ | $\lambda \cdot$ new running max; at the last point, $\lambda(\hat r - \text{cumulative}_{\text{prev}})$ |
| `FKSum(lam)` | running sum, $\sum$ of running sums, $\hat r$, $t$, `is_last` | same pattern | same pattern, with the running sum |
| `TiltEnds(lam)` | $(\hat r_{\mathrm{start}}, t_{\mathrm{start}}, \hat r, t)$, `is_last` | $-\ell_{\mathrm{start}} + [\text{last}]\,\ell$ | $0$; at the last point, $\ell$ |
| `ReverseRatio(target_reverse, proposal_reverse)` | $(\text{total}, \text{last term}, x_{\mathrm{prev}}, t_{\mathrm{prev}})$; left unchanged by `update` when `is_trivial` | total | the latest term |
| `Sum(*ps)` | tuple of the parts' states | sum | sum of the parts' increments |
| `Zero()` | zeros of shape `[I, N]` (the shape is what `smc` needs for `log_weights`) | that zero array | that zero array |
| `Relative(p)` | `p`'s state and $C_{\text{start}}$ | `p.log_tilt` minus $C_{\text{start}}$; exactly $0$ at the start point | `p.increment` |

- **`FKMax` and `FKSum`** require $\lambda$ constant in $t$ (it may vary by ladder position), as in FK steering.
- **Usage limit of `FKMax` and `FKSum`** (put it in both docstrings): they target $e^{\lambda \hat r_0}$ exactly without resampling (`Never()`). With resampling, they are reliable only when $\lambda S \sigma_r$ (max) or $\lambda \tfrac{S^2}{2} \sigma_r$ (sum) is small, where $\sigma_r$ is the spread of the final predicted reward across particles. These bounds are conservative: the effective counts are the steps on which $\hat r$ already differs between particles.
- **Telescoping.** All FK potentials end with $C = \lambda \hat r_0$, which is property P1 (§18).

```python
@pytree
class TiltState:   reward: Array; t: Array
@pytree
class PathState:   running: Array; cumulative: Array; reward: Array; t: Array; is_last: bool = static(False)
@pytree
class EndsState:   reward_start: Array; t_start: Array; reward: Array; t: Array; is_last: bool = static(False)
@pytree
class ReverseRatioState:   total: Array; last_term: Array; x: Array; t: Array

@dataclass(frozen=True)
class FKDiff(TiltDifference):
    lam: Callable
    requires = frozenset({Feature.REWARD})
    def init(self, record): return TiltState(reward=record.need(Feature.REWARD), t=record.t)
    def update(self, state, record, *, is_last): return TiltState(reward=record.need(Feature.REWARD), t=record.t)
    def log_tilt(self, state): return self.lam(state.t) * state.reward

@dataclass(frozen=True)
class FKMax:
    lam: Callable
    requires = frozenset({Feature.REWARD})
    def init(self, record):
        reward = record.need(Feature.REWARD)
        return PathState(running=reward, cumulative=reward, reward=reward, t=record.t, is_last=False)
    def update(self, state, record, *, is_last):
        reward = record.need(Feature.REWARD); running = xp.maximum(state.running, reward)
        return PathState(running=running, cumulative=state.cumulative + running, reward=reward, t=record.t,
                         is_last=is_last)                        # FKSum: running = state.running + reward
    def log_tilt(self, state): return self.lam(state.t) * (state.reward if state.is_last else state.cumulative)
    def increment(self, prev, state):                               # `is_last` is static
        return self.lam(state.t) * ((state.reward - prev.cumulative) if state.is_last else state.running)

@dataclass(frozen=True)
class FKSum(FKMax):                   # identical except the running statistic
    def update(self, state, record, *, is_last):
        reward = record.need(Feature.REWARD); running = state.running + reward
        return PathState(running=running, cumulative=state.cumulative + running, reward=reward, t=record.t,
                         is_last=is_last)

@dataclass(frozen=True)
class TiltEnds:                       # C = -l(start) + [last] l(end): CREPE's endpoint tilts
    lam: Callable
    requires = frozenset({Feature.REWARD})
    def init(self, record):
        reward = record.need(Feature.REWARD)
        return EndsState(reward_start=reward, t_start=record.t, reward=reward, t=record.t, is_last=False)
    def update(self, state, record, *, is_last):
        return EndsState(reward_start=state.reward_start, t_start=state.t_start, reward=record.need(Feature.REWARD),
                         t=record.t, is_last=is_last)
    def log_tilt(self, state):
        start_term = -self.lam(state.t_start) * state.reward_start
        return start_term + self.lam(state.t) * state.reward if state.is_last else start_term
    def increment(self, prev, state):
        return self.lam(state.t) * state.reward if state.is_last else xp.zeros_like(state.reward)

@dataclass(frozen=True)
class Sum:
    parts: tuple[Potential, ...]
    def __init__(self, *parts): object.__setattr__(self, "parts", parts)
    @property
    def requires(self): return frozenset().union(*(part.requires for part in self.parts))
    def init(self, record): return tuple(part.init(record) for part in self.parts)
    def update(self, state, record, *, is_last):
        return tuple(part.update(part_state, record, is_last=is_last) for part, part_state in zip(self.parts, state))
    def log_tilt(self, state): return sum(part.log_tilt(part_state) for part, part_state in zip(self.parts, state))
    def increment(self, prev, state):
        return sum(part.increment(part_prev, part_state) for part, part_prev, part_state in zip(self.parts, prev, state))

@dataclass(frozen=True)
class Zero:
    requires = frozenset()
    def init(self, record): return xp.zeros(record.t.shape, dtype=record.t.dtype, device=device_of(record.t))
    def update(self, state, record, *, is_last): return state
    def log_tilt(self, state): return state
    def increment(self, prev, state): return state

@pytree
class RelativeState:   inner: State; start_tilt: Array

@dataclass(frozen=True)
class Relative:                       # C - C(start): for runs that start from points already tilted by C(start)
    inner: Potential
    @property
    def requires(self): return self.inner.requires
    def init(self, record):
        state = self.inner.init(record); return RelativeState(inner=state, start_tilt=self.inner.log_tilt(state))
    def update(self, state, record, *, is_last):
        return RelativeState(inner=self.inner.update(state.inner, record, is_last=is_last), start_tilt=state.start_tilt)
    def log_tilt(self, state): return self.inner.log_tilt(state.inner) - state.start_tilt   # x - x == 0 exactly at the start
    def increment(self, prev, state): return self.inner.increment(prev.inner, state.inner)

@dataclass(frozen=True)
class ReverseRatio:                   # scores the reverse step with the kernels opposite to the run
    target_reverse: Kernel; proposal_reverse: Kernel
    @property
    def is_trivial(self): return self.target_reverse is self.proposal_reverse   # static: term is identically 0
    @property
    def requires(self):
        return frozenset() if self.is_trivial else self.target_reverse.requires | self.proposal_reverse.requires
    def init(self, record):
        zeros = xp.zeros(record.t.shape, dtype=record.t.dtype, device=device_of(record.t))
        return ReverseRatioState(total=zeros, last_term=zeros, x=record.x, t=record.t)
    def update(self, state, record, *, is_last):
        if self.is_trivial:
            return state
        term = self.proposal_reverse.at(record, state.t).log_ratio(self.target_reverse.at(record, state.t), state.x)   # §7
        return ReverseRatioState(total=state.total + term, last_term=term, x=record.x, t=record.t)
    def log_tilt(self, state): return state.total
    def increment(self, prev, state): return state.last_term
```

## 10. Resampling and policies

Everything here is branch-free and per instance.

```python
def ess(log_weights):                             # [I, N] -> [I]
    log_weights_normalized = log_weights - logsumexp(log_weights, axis=1, keepdims=True)
    return xp.exp(-logsumexp(2 * log_weights_normalized, axis=1))

def _cdf(log_weights):                            # [I, N] -> [I, N]
    return xp.cumulative_sum(xp.exp(log_weights - logsumexp(log_weights, axis=1, keepdims=True)), axis=1)

def systematic(backend, log_weights, key):        # [I, N] -> ancestors [I, N]
    I, N = log_weights.shape
    device = device_of(log_weights)
    cdf = _cdf(log_weights)
    positions = (backend.uniform(key, (I, 1), cdf.dtype, device=device) + xp.arange(N, dtype=cdf.dtype, device=device)[None, :]) / N
    ancestors = xp.sum(xp.astype(cdf[:, None, :] < positions[:, :, None], xp.int64), axis=2)   # O(N^2); avoids 2-D searchsorted
    return xp.clip(ancestors, max=N - 1)

def categorical(backend, log_weights, key):       # [I, N] -> ancestors [I]: one draw per instance, by CDF inversion
    I, N = log_weights.shape
    cdf = _cdf(log_weights)
    uniform = backend.uniform(key, (I, 1), cdf.dtype, device=device_of(log_weights))
    ancestors = xp.sum(xp.astype(cdf < uniform, xp.int64), axis=1)
    return xp.clip(ancestors, max=N - 1)

def masked_resample(backend, tree, log_weights, resample_mask, key):    # resample_mask: [I] bool
    I, N = log_weights.shape
    own_index = xp.arange(N, dtype=xp.int64, device=device_of(log_weights))[None, :]
    ancestors = xp.where(resample_mask[:, None], systematic(backend, log_weights, key), own_index)
    tree = tree_map(lambda a: take_rows(a, ancestors), tree)
    return tree, xp.where(resample_mask[:, None], xp.zeros_like(log_weights), log_weights)

@dataclass(frozen=True)
class Never:                                      # step_index: possibly traced
    def should_resample(self, log_weights, step_index):
        return xp.zeros(log_weights.shape[0], dtype=xp.bool, device=device_of(log_weights))

@dataclass(frozen=True)
class Every:
    interval: int
    def should_resample(self, log_weights, step_index):
        return xp.broadcast_to(xp.asarray(step_index % self.interval == 0, device=device_of(log_weights)),
                               (log_weights.shape[0],))

@dataclass(frozen=True)
class ESSBelow:
    fraction: float
    def should_resample(self, log_weights, step_index): return ess(log_weights) < self.fraction * log_weights.shape[1]
```

**Picking one particle.** To select a single particle per instance (e.g. from an `smc` run's final `log_weights`), use `categorical`. A fixed column of `systematic` is not such a draw: column $j$ always falls in the $j$-th $1/N$ stratum of the CDF.

**Ladders and resampling.** When axis 1 is a temperature ladder (RpCN), positions have different targets, so resampling along it is invalid. Use `Never()`.

## 11. Moves

A move is an MCMC kernel applied at a fixed time that leaves the current target invariant. It therefore never changes `log_weights` (resample-move SMC). Moves are used by `smc` (RpCN's pCN; optional for FK) and by CREPE (Langevin local exploration).

```python
@pytree
class MoveContext:                        # one pytree: swaps and accept/reject keep it consistent
    record: Record                        # current point (evaluated with this step's needs)
    potential_state: State                # potential state including record
    prev_potential_state: State | None    # potential state before record (None at the start point)
    reference: Gaussian | None            # the target kernel's distribution that record was drawn from

class Move(Protocol):
    requires: frozenset[Feature]          # features the move reads from ctx.record
    def __call__(self, ctx, *, evaluate_point, potential, is_last, move_count, key, backend) -> MoveContext: ...
    # evaluate_point(point) -> Record: supplied by the runner; it requests this step's needs, so accepted
    # candidates are complete records (I4). The move never knows what those needs are.
    # move_count: how many times this move has already fired in this run (an int scalar, traced
    # inside scan; the runner carries it). Moves use it for counters such as the replica-
    # exchange sweep, so EvenOdd parity alternates across calls instead of restarting, and
    # it does so even when move_at fires at a fixed stride (e.g. every 20th step).

def rescore(potential, prev_state, record, is_last):   # potential state if `record` replaced the current point
    return potential.init(record) if prev_state is None else potential.update(prev_state, record, is_last=is_last)

def tilt_delta(potential, prev_state, state_a, state_b):   # C(state_a) - C(state_b), both extending one history
    if prev_state is None:                       # static: at the start point, C is a single term
        return potential.log_tilt(state_a) - potential.log_tilt(state_b)
    return potential.increment(prev_state, state_a) - potential.increment(prev_state, state_b)   # no shared prefix
```

**MH acceptances use `tilt_delta`.** A candidate and the current point share `prev_potential_state`, so only their increments are compared. For `FKDiff` this is $\lambda[\hat r(\text{cand}) - \hat r(\text{cur})]$.

**pCN, with optional replica exchange.** RpCN's move: `n_proposals` pCN proposals, each followed by an exchange sweep.

```python
@dataclass(frozen=True)
class PCN:
    n_proposals: int                     # pCN proposals per move call (M)
    step_size: Callable                  # rho in (0, 1) per row, e.g. ladder(step_sizes)
    pairs: PairSchedule | None = None    # replica exchange along axis 1 after each pCN step (§12)
    requires = frozenset()               # acceptance reads only the potential
    def __call__(self, ctx, *, evaluate_point, potential, is_last, move_count, key, backend):
        for substep in range(self.n_proposals):                          # static
            key_noise, key_accept, key_exchange, key = backend.split(key, 4)
            x, reference = ctx.record.x, ctx.reference
            rho = bcast(self.step_size(ctx.record.t), x)
            noise = backend.normal(key_noise, x.shape, x.dtype, device=device_of(x))
            x_candidate = reference.mean + xp.sqrt(1 - rho**2) * (x - reference.mean) + rho * bcast(xp.sqrt(reference.var), x) * noise
            candidate = evaluate_point(Point(x_candidate, ctx.record.t))     # 1 evaluation per proposal
            candidate_state = rescore(potential, ctx.prev_potential_state, candidate, is_last)
            log_accept_ratio = tilt_delta(potential, ctx.prev_potential_state, candidate_state, ctx.potential_state)
            uniform = backend.uniform(key_accept, log_accept_ratio.shape, log_accept_ratio.dtype, device=device_of(log_accept_ratio))
            accepted = xp.log(uniform) < log_accept_ratio
            ctx = replace(ctx, record=tree_where(accepted, candidate, ctx.record),
                          potential_state=tree_where(accepted, candidate_state, ctx.potential_state))
            if self.pairs is not None:                                   # static
                ctx = replica_exchange(ctx, self.pairs, StateSwap(potential),
                                       sweep=move_count * self.n_proposals + substep,   # pCN steps so far
                                       key=key_exchange, backend=backend)
        return ctx
```

The proposal $\hat x = \mu + \sqrt{1-\rho^2}\,(x - \mu) + \rho\,\tau\,\xi$ is the pCN proposal. It is reversible with respect to `reference` $= \mathcal N(\mu, \tau^2)$, so the acceptance needs only the cumulative tilt. For `FKDiff` that is $\beta_k[\hat r(\hat x) - \hat r(x)]$.

**Langevin.** CREPE's local exploration move. It is unadjusted (no Metropolis correction).

```python
@dataclass(frozen=True)
class LangevinMove:
    kernel: Kernel                       # e.g. Guided(Langevin(h), tilt(lam, 0.5))
    @property
    def requires(self): return self.kernel.requires
    def __call__(self, ctx, *, evaluate_point, potential, is_last, move_count, key, backend):
        x = self.kernel.at(ctx.record, ctx.record.t).sample(key, backend)
        record = evaluate_point(Point(x, ctx.record.t))
        return replace(ctx, record=record, potential_state=rescore(potential, ctx.prev_potential_state, record, is_last))
```

## 12. Replica exchange

Replica exchange is split into **which pairs** (a schedule) and **how a pair proposes** (an exchange), with one driver. RpCN and CREPE both use it. The ladder is axis 1 of any pytree.

```python
class PairSchedule(Protocol):
    period: int                                              # rounds depend on `sweep` only via sweep % period
    def rounds(self, n_levels: int, sweep: int) -> tuple[tuple[tuple[int, int], ...], ...]: ...
    # static; each round is a tuple of disjoint adjacent pairs (lower, lower + 1), proposed together

@dataclass(frozen=True)
class SequentialSweep:          # (0,1), then (1,2), ..., each its own round
    period = 1
    def rounds(self, n_levels, sweep): return tuple(((level, level + 1),) for level in range(n_levels - 1))

@dataclass(frozen=True)
class EvenOdd:                  # deterministic even-odd (DEO): all pairs of one parity at once, alternating
    period = 2
    def rounds(self, n_levels, sweep):
        return (tuple((level - 1, level) for level in range(1, n_levels) if level % 2 == sweep % 2),)

class Exchange(Protocol):
    def propose(self, ladder, lower, upper, key, backend) -> tuple[Tree, Tree, Array]: ...
    # candidates for positions lower and upper (pytrees [I, P, ...]) and log-acceptance [I, P]
    # Optional, for exchanges that swap whole states (StateSwap): view(ladder) -> Tree, the part of the
    # ladder the acceptance reads, and log_accept(view, lower, upper) -> [I, P]. Their presence selects the fast path.

def replica_exchange(ladder, pairs: PairSchedule, exchange: Exchange, *, sweep, key, backend):
    # sweep: Python int (static) or traced int scalar (e.g. from the runner's step index)
    swaps_whole_states = hasattr(exchange, "log_accept")                # static: e.g. StateSwap
    def for_residue(residue):                                            # residue: static int in range(period)
        def run(ladder):
            rounds = [pairs_round for pairs_round in pairs.rounds(n_levels(ladder), residue) if pairs_round]
            if swaps_whole_states:
                return _swap_states(ladder, rounds, exchange, key, backend)
            for round_pairs, round_key in zip(rounds, backend.split(key, max(len(rounds), 1))):
                lower, upper = [a for a, _ in round_pairs], [b for _, b in round_pairs]
                key_propose, key_accept = backend.split(round_key, 2)
                new_lower, new_upper, log_accept_ratio = exchange.propose(ladder, lower, upper, key_propose, backend)
                uniform = backend.uniform(key_accept, log_accept_ratio.shape, log_accept_ratio.dtype,
                                          device=device_of(log_accept_ratio))
                accepted = xp.log(uniform) < log_accept_ratio            # [I, P]
                ladder = put_levels(ladder, lower, tree_where(accepted, new_lower, take_levels(ladder, lower)))
                ladder = put_levels(ladder, upper, tree_where(accepted, new_upper, take_levels(ladder, upper)))
            return ladder
        return run
    if isinstance(sweep, int):                                           # static: no dispatch
        return for_residue(sweep % pairs.period)(ladder)
    return backend.switch(sweep % pairs.period, [for_residue(r) for r in range(pairs.period)], ladder)

def _permute(tree, permutation):                 # permutation [I, n]: position k takes the state at permutation[i, k]
    def take(a):
        idx = xp.broadcast_to(bcast(permutation, a), tuple(permutation.shape) + tuple(a.shape[2:]))
        return xp.take_along_axis(a, idx, axis=1)
    return tree_map(take, tree)

def _swap_states(ladder, rounds, exchange, key, backend):
    # Rounds decide acceptance from exchange.view(ladder) alone, tracking a per-row permutation of
    # ladder positions; the whole ladder is permuted once at the end. Same draws and result as the generic path.
    view = exchange.view(ladder)
    first_leaf = leaves(view)[0]
    permutation = xp.broadcast_to(xp.arange(first_leaf.shape[1], device=device_of(first_leaf))[None, :],
                                  (first_leaf.shape[0], first_leaf.shape[1]))
    for round_pairs, round_key in zip(rounds, backend.split(key, max(len(rounds), 1))):
        lower, upper = [a for a, _ in round_pairs], [b for _, b in round_pairs]
        _, key_accept = backend.split(round_key, 2)                      # key_propose unused, as in StateSwap.propose
        log_accept_ratio = exchange.log_accept(_permute(view, permutation), lower, upper)
        uniform = backend.uniform(key_accept, log_accept_ratio.shape, log_accept_ratio.dtype,
                                  device=device_of(log_accept_ratio))
        accepted = xp.log(uniform) < log_accept_ratio                    # [I, P]
        perm_lower, perm_upper = take_levels(permutation, lower), take_levels(permutation, upper)
        permutation = put_levels(put_levels(permutation, lower, xp.where(accepted, perm_upper, perm_lower)),
                                 upper, xp.where(accepted, perm_lower, perm_upper))
    return _permute(ladder, permutation) if rounds else ladder

def _swap_permutation(n_levels, lower, upper):
    permutation = list(range(n_levels))
    for a, b in zip(lower, upper): permutation[a], permutation[b] = b, a
    return permutation

@dataclass(frozen=True)
class StateSwap:                # plain parallel tempering: exchange states; 0 evaluations
    potential: Potential
    def propose(self, ladder, lower, upper, key, backend):
        swapped = take_levels(ladder, _swap_permutation(n_levels(ladder), lower, upper))   # all fields move together
        return (take_levels(swapped, lower), take_levels(swapped, upper),
                self.log_accept(ladder.potential_state, lower, upper))
    def view(self, ladder): return ladder.potential_state                 # all that log_accept reads
    def log_accept(self, state, lower, upper):                            # [I, P]
        swapped = take_levels(state, _swap_permutation(n_levels(state), lower, upper))
        tilt_change = self.potential.log_tilt(swapped) - self.potential.log_tilt(state)   # [I, n], lambda per position
        return take_levels(tilt_change, lower) + take_levels(tilt_change, upper)
```

**Whole-state fast path.** For an exchange with `view` and `log_accept`, `replica_exchange` never moves the full ladder (records, `prev_potential_state`, `reference`, ...) inside a round. It permutes only the small `view` to score each round, and applies the accumulated permutation to the ladder once. It must consume the same keys as the generic path (`propose`'s `key_propose` is split off and ignored), so both paths give identical results. `propose` stays the reference definition.

For `FKDiff(ladder(betas))`, `StateSwap`'s log-acceptance is

$$
(\beta_{k+1} - \beta_k)\big(\hat r(x^{(k)}) - \hat r(x^{(k+1)})\big).
$$

**`StateSwap` is the one place `log_tilt` is differenced.** This is intended: the two rows have different histories.

CREPE's `PathExchange` is in §16.

## 13. The SMC runner

```python
@dataclass(frozen=True)
class SMCResult:
    final: Record          # [I, N, ...], carries potential.requires | move.requires | out_needs
    log_weights: Array     # [I, N] unnormalised

def smc(start, u_grid, *, backend, evaluate, schedule, target, proposal, potential, policy,
        move=None, move_at=lambda step_index: True, out_needs=frozenset({Feature.X0}), key) -> SMCResult:
    """start: Gaussian (sampled, and the first move's reference), Point, or Record; leading axes [I, N].
    u_grid: [S] (shared) or [I, N, S] (per row; increasing or decreasing -- the runner doesn't care)."""
    start_x = start.mean if isinstance(start, Gaussian) else start.x
    dtype, device = start_x.dtype, device_of(start_x)               # §4.1: run dtype and device
    t_grid = schedule.time(xp.asarray(u_grid, dtype=dtype, device=device))   # I5: the only schedule call
    t_by_step = steps_first(t_grid, leading_shape_of(start))         # -> [S, I, N]
    n_points = t_by_step.shape[0]
    move_needs = move.requires if move is not None else frozenset()
    step_needs = target.requires | proposal.requires | potential.requires | move_needs     # I4
    last_step_needs = potential.requires | move_needs | frozenset(out_needs)
    evaluate_step = lambda point: evaluate_rows(evaluate, point, step_needs, backend)
    evaluate_last_step = lambda point: evaluate_rows(evaluate, point, last_step_needs, backend)
    key_start, key_first_move, key_steps = backend.split(key, 3)

    def apply_move(ctx, n_moves, step_index, evaluate_point, is_last, key):   # -> (ctx, n_moves); no-op without a move
        if move is None:
            return ctx, n_moves
        def run_move(operand):
            ctx, n_moves = operand
            return (move(ctx, evaluate_point=evaluate_point, potential=potential, is_last=is_last,
                         move_count=n_moves, key=key, backend=backend),
                    n_moves + 1)
        return backend.cond(move_at(step_index), run_move, lambda operand: operand, (ctx, n_moves))

    record, reference = start_record(start, t_by_step[0], step_needs, evaluate, backend, key_start)
    state = potential.init(record)
    log_weights = potential.log_tilt(state)
    n_moves = xp.zeros((), dtype=xp.int32, device=device)           # moves fired so far in this run
    if reference is not None:                                        # static: a move at t_N needs a reference
        ctx, n_moves = apply_move(MoveContext(record, state, None, reference), n_moves, 0, evaluate_step, False,
                                  key_first_move)
        record, state = ctx.record, ctx.potential_state

    def advance(carry, t_next, step_index, key, evaluate_point, is_last):
        record, state, log_weights, n_moves = carry
        key_resample, key_propose, key_move = backend.split(key, 3)
        (record, state), log_weights = masked_resample(backend, (record, state), log_weights,
                                                       policy.should_resample(log_weights, step_index), key_resample)
        target_dist = target.at(record, t_next)                       # also the pCN reference
        proposal_dist = target_dist if proposal is target else proposal.at(record, t_next)
        x = proposal_dist.sample(key_propose, backend)
        if proposal is not target:                                    # static
            log_weights = log_weights + target_dist.log_ratio(proposal_dist, x)   # I9: never two log_probs
        new_record = evaluate_point(Point(x, t_next))                 # I1
        new_state = potential.update(state, new_record, is_last=is_last)
        log_weights = log_weights + potential.increment(state, new_state)   # I9: never a C difference
        ctx, n_moves = apply_move(MoveContext(new_record, new_state, state, target_dist), n_moves, step_index,
                                  evaluate_point, is_last, key_move)  # log_weights unchanged
        return ctx.record, ctx.potential_state, log_weights, n_moves

    def scan_step(carry, step_inputs):
        t_next, step_index, step_key = step_inputs
        return advance(carry, t_next, step_index, step_key, evaluate_step, False), None

    step_keys = backend.split(key_steps, n_points - 1)
    step_indices = xp.arange(1, n_points - 1, dtype=xp.int32, device=device)
    carry, _ = backend.scan(scan_step, (record, state, log_weights, n_moves),
                            (t_by_step[1:-1], step_indices, step_keys[:-1]))   # S == 2: empty scan
    record, state, log_weights, _ = advance(carry, t_by_step[-1], n_points - 1, step_keys[-1],
                                            evaluate_last_step, True)
    return SMCResult(final=record, log_weights=log_weights)
```

## 14. FK steering, DAS, and FK with pCN moves

```python
xp = backend.xp                                    # call-site code may use backend.xp to build inputs
schedule = EDM()                                   # injected
u = xp.linspace(1.0, 0.0, 100, dtype=xp.float32)
P = PosteriorStep(schedule)
start = schedule.prior((I, N), D, xp=xp, dtype=xp.float32)   # D: event shape tuple, e.g. (4, 64, 64)

# FK steering: FKDiff; optionally Guided(P, tilt(lam)) as proposal.
# FKMax / FKSum plug in the same way, within their usage limit (§9).
fk = smc(start, u, backend=backend, evaluate=evaluate, schedule=schedule, target=P, proposal=P,
         potential=FKDiff(constant(10.0)), policy=Every(5), key=key)

# DAS
lam = das_lambda(schedule.time(u), growth=0.008, alpha_kl=0.01)
das = smc(start, u, backend=backend, evaluate=evaluate, schedule=schedule, target=P, proposal=Guided(P, tilt(lam)),
          potential=FKDiff(lam), policy=ESSBelow(0.5), key=key)

# FK + pCN moves (resample-move): valid for any FK potential, because the move targets
# the conditional of the FK target given the rest of the path, via the cumulative log_tilt
fk_pcn = smc(start, u, backend=backend, evaluate=evaluate, schedule=schedule, target=P, proposal=P,
             potential=FKDiff(constant(10.0)), policy=Every(5),
             move=PCN(n_proposals=2, step_size=constant(0.3)), move_at=lambda step_index: step_index % 5 == 0,
             key=key)
```

## 15. RpCN

The particle axis is the temperature ladder ($N = K + 1$). The potential is `FKDiff` with $\beta$ by position, so `log_tilt` $= \beta_k\,\hat r_t(x)$, the tempered reward twist of rung $k$. There is no resampling.

```python
betas = xp.linspace(0.0, beta, K + 1, dtype=xp.float32)  # 0 = beta_0 < ... < beta_K = beta
P = DDIMStep(schedule, eta=1.0)                           # = PosteriorStep (test K4); inverse problems: Guided(P, normalized(step_length))
out = smc(schedule.prior((I, K + 1), D, xp=xp, dtype=xp.float32), u, backend=backend, evaluate=evaluate,
          schedule=schedule, target=P, proposal=P, potential=FKDiff(ladder(betas)), policy=Never(),
          move=PCN(n_proposals=M, step_size=ladder(step_sizes), pairs=SequentialSweep()),   # or EvenOdd()
          move_at=lambda step_index: step_index % 20 == 0, key=key)
coldest = out.final.x0[:, K]                              # the beta_K = beta replica, denoised
```

- At the start point, `reference` is the prior; afterwards it is the target kernel's distribution for the current step.
- **Swap order:** `SequentialSweep()` or `EvenOdd()`. Both are valid.

## 16. CREPE

**Layout.**
- **Replicas** are one `Record` with axes `[I, M+1, ...]`: $I$ chains, levels $m = 0..M$ at $u_0 < \dots < u_M = 1$.
- **The lowest level may sit above the data end** ($u_0 > 0$). Level 0 then samples $\propto p_{u_0}\, e^{\lambda(\tau(u_0))\, \hat r}$, and `crepe_finish` (below) carries its samples the rest of the way to $u = 0$. `crepe_sample` (end of this section) runs the chains and the finish in one call.
- **Replica invariant:** every replica always carries exactly `replica_needs`, and the replica at level $m$ always has $t = \tau(u_m)$.

```python
@dataclass(frozen=True)
class CREPEConfig:
    u_levels: tuple[float, ...]          # increasing; u_levels[-1] = 1
    steps_per_gap: int                   # fine steps per gap (K)
    model_noising: Kernel; model_denoising: Kernel; proposal_noising: Kernel; proposal_denoising: Kernel
    lam: Callable                        # e.g. level_lambda(time(u_levels), beta, rho)
    pairs: PairSchedule = EvenOdd()
    local_move: Move | None = None       # e.g. LangevinMove(Guided(Langevin(h), tilt(lam, 0.5)))
    local_levels: tuple[int, ...] = ()   # static subset of 0..M-1
    out_needs: frozenset = frozenset({Feature.X0})
    @property
    def n_gaps(self): return len(self.u_levels) - 1   # M
    @property
    def replica_needs(self):
        kernels = (self.model_noising, self.model_denoising, self.proposal_noising, self.proposal_denoising)
        requirements = [kernel.requires for kernel in kernels]
        requirements += [self.local_move.requires] if self.local_move else []
        return frozenset().union(*requirements) | {Feature.REWARD} | self.out_needs
```

Default configuration for reward tilting:

```python
noise = Noising(schedule)
lam = level_lambda(schedule.time(xp.asarray(u_levels)), beta=10.0, rho=5.0)
cfg = CREPEConfig(u_levels=u_levels, steps_per_gap=5,
                  model_noising=noise, proposal_noising=noise,             # same object -> F terms cancel statically
                  model_denoising=EulerMaruyama(ReverseSDE(schedule), Direction.BACKWARD),
                  proposal_denoising=Guided(EulerMaruyama(ReverseSDE(schedule), Direction.BACKWARD), tilt(lam)),
                  lam=lam, local_move=LangevinMove(Guided(Langevin(step_size=lambda t: 0.01 * schedule.std(t) ** 2),
                                                          tilt(lam, 0.5))),
                  local_levels=tuple(range(len(u_levels) - 1)))
```

**The path exchange.** Two one-directional `smc` calls per round.

```python
@dataclass(frozen=True)
class PathExchange:
    cfg: CREPEConfig; evaluate: Evaluate; schedule: Schedule
    def propose(self, ladder, lower, upper, key, backend):   # pair (lower[p], upper[p] = lower[p] + 1) is gap lower[p]
        cfg = self.cfg; key_up, key_down = backend.split(key, 2)
        n_instances, dtype, device = ladder.t.shape[0], ladder.t.dtype, device_of(ladder.t)
        path_potential = lambda target_reverse, proposal_reverse: Sum(TiltEnds(cfg.lam),
                                                                      ReverseRatio(target_reverse, proposal_reverse))
        run_path = lambda start, grid, target, proposal, potential, key: smc(start, grid, backend=backend,
                  evaluate=self.evaluate, schedule=self.schedule, target=target, proposal=proposal, potential=potential,
                  policy=Never(), out_needs=cfg.replica_needs, key=key)
        gap_grids = lambda ascending: per_row_grid(cfg, lower, n_instances, ascending, xp=xp, dtype=dtype, device=device)
        upward = run_path(take_levels(ladder, lower), gap_grids(True), cfg.model_noising, cfg.proposal_noising,
                          path_potential(cfg.model_denoising, cfg.proposal_denoising), key_up)
        downward = run_path(take_levels(ladder, upper), gap_grids(False), cfg.model_denoising, cfg.proposal_denoising,
                            path_potential(cfg.model_noising, cfg.proposal_noising), key_down)
        return downward.final, upward.final, upward.log_weights + downward.log_weights   # lower <- z_0, upper <- y_K

def fine_grid(cfg, gap, *, xp, dtype, device=None):   # gap: [K+1] ascending; endpoints exactly u_gap, u_{gap+1}
    u_lo, u_hi = cfg.u_levels[gap], cfg.u_levels[gap + 1]
    points = [u_lo, *(u_lo + (u_hi - u_lo) * j / cfg.steps_per_gap for j in range(1, cfg.steps_per_gap)), u_hi]
    return xp.asarray(points, dtype=dtype, device=device)

def per_row_grid(cfg, gaps, n_instances, ascending, *, xp, dtype, device=None):   # -> [I, P, K+1]
    grids = xp.stack([fine_grid(cfg, gap, xp=xp, dtype=dtype, device=device) for gap in gaps])
    grids = grids if ascending else xp.flip(grids, axis=-1)
    return xp.broadcast_to(grids[None], (n_instances, *grids.shape))
```

**Build every CREPE grid with `fine_grid`, never `xp.linspace`.** This keeps every run starting and ending at exactly the level times (the replica invariant).

With $\log R^X(x_{0:K}) = \sum_k [\log B^X(x_{k-1} \mid x_k) - \log F^X(x_k \mid x_{k-1})]$, this is CREPE's swap acceptance:

$$
\log w_{\mathrm{up}} = \ell(y_K) - \ell(y_0) + \log R^Q(y) - \log R^P(y), \qquad
\log w_{\mathrm{dn}} = \ell(z_0) - \ell(z_K) + \log R^P(z) - \log R^Q(z).
$$

The driver applies $\min(0, \cdot)$ implicitly: it accepts iff $\log U < \log w_{\mathrm{up}} + \log w_{\mathrm{dn}}$.

**With the default proposals** ($F^Q = F^P$):

- **The $F$ terms cancel statically.** `ReverseRatio(cfg.model_noising, cfg.proposal_noising)` is trivial, and
$$
\log A = \min\Big(0,\ \ell(z_0) - \ell(z_K) - \ell(y_0) + \ell(y_K) + \sum_{k=1}^{K} \big[\log B^P - \log B^Q\big](z_{k-1} \mid z_k) - \sum_{k=1}^{K} \big[\log B^P - \log B^Q\big](y_{k-1} \mid y_k)\Big).
$$
- **Sampling the up path needs no network.** Its $K$ evaluations score $y$ under the denoising kernels.
- **The cost is $2K$ evaluations per pair.**

**Initialisation:** $M$ chained `smc` calls with the model's own kernel.

```python
def top_replica(cfg, n_instances, event_shape, *, backend, evaluate, schedule, dtype, device, key):   # fresh level-M replica
    xp = backend.xp
    prior = schedule.prior((n_instances, 1), event_shape, xp=xp, dtype=dtype, device=device)
    t_top = schedule.time(xp.ones((n_instances, 1), dtype=dtype, device=device))
    return start_record(prior, t_top, cfg.replica_needs, evaluate, backend, key)[0]

def crepe_init(cfg, n_instances, event_shape, *, backend, evaluate, schedule, dtype, device=None, key):
    xp = backend.xp
    key_top, _unused_key, *gap_keys = backend.split(key, cfg.n_gaps + 2)   # the second key is never used
    top = top_replica(cfg, n_instances, event_shape, backend=backend, evaluate=evaluate, schedule=schedule,
                      dtype=dtype, device=device, key=key_top)
    levels, current = [None] * cfg.n_gaps + [top], top
    for level in range(cfg.n_gaps, 0, -1):                           # static loop
        result = smc(current, xp.flip(fine_grid(cfg, level - 1, xp=xp, dtype=dtype, device=device), axis=0),
                     backend=backend, evaluate=evaluate, schedule=schedule, target=cfg.model_denoising,
                     proposal=cfg.model_denoising, potential=Zero(), policy=Never(), out_needs=cfg.replica_needs,
                     key=gap_keys[level - 1])
        levels[level - 1] = current = result.final
    return stack_levels(levels)                                      # [I, M+1, ...]
```

**The sweep loop.** A block is $L = \operatorname{lcm}(\texttt{cfg.pairs.period}, \texttt{thin})$ sweeps, so the scan body's static structure is fixed and the pair-schedule parity lines up across blocks. `crepe` first runs `burn_in` blocks that emit nothing, then `n_blocks` blocks that emit after every `thin`-th sweep: two scans, because whether a block emits must be static.

```python
def crepe(cfg, n_instances, event_shape, n_blocks, emit, *, backend, evaluate, schedule, dtype, device=None, key,
          initial_ladder=None, thin=1, burn_in=0):
    # emit: Record [I, 1, ...] -> any pytree of arrays (e.g. r.x0, (r.x0, r.reward), or a projected Record)
    # initial_ladder: replicas from an earlier call (its returned ladder) to continue from; None starts with crepe_init
    # thin: emit after every thin-th sweep only (static int >= 1); the sweeps in between still run
    # burn_in: blocks (L sweeps each) run first with nothing emitted or stored (static int >= 0)
    xp = backend.xp
    key_init, key_run = backend.split(key, 2)                        # same split whether or not initial_ladder is given
    if initial_ladder is None:                                       # static
        ladder = crepe_init(cfg, n_instances, event_shape, backend=backend, evaluate=evaluate, schedule=schedule,
                            dtype=dtype, device=device, key=key_init)
    else:
        if not initial_ladder.provides(cfg.replica_needs):
            raise PartialRecord(cfg.replica_needs)
        ladder = initial_ladder.project(cfg.replica_needs)           # level times are the caller's responsibility
    evaluate_point = lambda point: evaluate_rows(evaluate, point, cfg.replica_needs, backend)
    path_exchange = PathExchange(cfg, evaluate, schedule)
    top_used_in_sweep = lambda sweep: any(cfg.n_gaps in pair for pairs_round in cfg.pairs.rounds(cfg.n_gaps + 1, sweep)
                                          for pair in pairs_round)   # static
    sweeps_per_block = math.lcm(cfg.pairs.period, thin)              # static: sweeps per scan body (L)
    def block(ladder, key, emitting):                                # emitting: static
        emitted = []
        for sweep, sweep_key in enumerate(backend.split(key, sweeps_per_block)):
            key_exchange, key_local, key_refresh = backend.split(sweep_key, 3)
            ladder = replica_exchange(ladder, cfg.pairs, path_exchange, sweep=sweep, key=key_exchange, backend=backend)
            if cfg.local_move is not None:                           # static
                local_ladder = take_levels(ladder, cfg.local_levels)
                ctx = cfg.local_move(MoveContext(local_ladder, Zero().init(local_ladder), None, None),
                                     evaluate_point=evaluate_point, potential=Zero(), is_last=False, move_count=sweep,
                                     key=key_local, backend=backend)
                ladder = put_levels(ladder, cfg.local_levels, ctx.record)
            if top_used_in_sweep(sweep + 1):                         # refresh only if used next sweep
                top = top_replica(cfg, n_instances, event_shape, backend=backend, evaluate=evaluate,
                                  schedule=schedule, dtype=dtype, device=device, key=key_refresh)
                ladder = put_levels(ladder, [cfg.n_gaps], top)
            if emitting and (sweep + 1) % thin == 0:                 # static
                emitted.append(emit(take_levels(ladder, [0])))       # e.g. lambda r: r.x0
        return ladder, (tree_map(lambda *a: xp.stack(a), *emitted) if emitting else None)   # any pytree
    key_burn_in, key_sample = backend.split(key_run, 2)
    if burn_in > 0:                                                  # static
        ladder, _ = backend.scan(lambda ladder, key: block(ladder, key, False), ladder,
                                 backend.split(key_burn_in, burn_in))
    ladder, samples = backend.scan(lambda ladder, key: block(ladder, key, True), ladder,
                                   backend.split(key_sample, n_blocks))
    return samples, ladder   # samples: every leaf [n_blocks, L // thin, I, 1, ...]; ladder: [I, M+1, ...]
```

- **Sweeps run:** `(burn_in + n_blocks) * L`, of which `n_blocks * L // thin` are emitted. The burn-in sweeps cost the same evaluations as any other sweep but store nothing. With the defaults (`thin=1`, `burn_in=0`), $L$ is the pair schedule's period and nothing changes.
- **Burn-in with continuation:** a run continued from `initial_ladder` usually needs `burn_in=0`, because its replicas are already past burn-in.
- **Choosing `thin`:** emit a cheap scalar first (e.g. `lambda r: r.reward`), estimate its integrated autocorrelation time $\tau_{\text{int}}$ in sweeps, and set `thin` to about $2\tau_{\text{int}}$. Kept samples are then close to independent. That matters for `crepe_finish`: every kept sample costs `n_tail` evaluations there, and correlated particles make its effective sample size, and therefore its resampling decisions, overstate the information in the ensemble.
- **When not to thin:** for plain averages over CREPE's own samples, thinning only discards information. Use it to cut memory or downstream cost.

**Continuing a run.** Pass the returned `ladder` back as `initial_ladder`, with a fresh `key`. The continuation skips `crepe_init` (no initialisation evaluations) and keeps the replica invariant.

**Refreshing the top replica** with a fresh prior draw is exact only if two things hold:
- `prior` is the exact marginal at $\tau(1)$;
- $\lambda(\tau(1)) = 0$. `level_lambda` guarantees this.

Otherwise the top level is approximately, not exactly, at its target. The approximation is the same as the one in the prior itself.

**Finishing from $u_0 > 0$.** `crepe_finish` runs one `smc` from the level-0 samples to $u = 0$, over `n_tail` steps. Most callers use `crepe_sample` (below), which composes `crepe`, `pool_samples` and `crepe_finish`.

```python
def finish_needs(cfg):          # what emit must keep for crepe_finish (emit = lambda r: r.project(finish_needs(cfg)))
    return cfg.model_denoising.requires | cfg.proposal_denoising.requires | {Feature.REWARD}

def pool_samples(samples):
    # leaves [n_blocks, E, I, 1, ...] (E = emits per block) -> [I, n_blocks * E, ...]:
    # one instance per chain; its particles are that chain's emitted samples in order (burn-in is crepe's job)
    def pool(a):
        n_blocks, n_emits, n_instances = a.shape[0], a.shape[1], a.shape[2]
        flat = xp.reshape(a, (n_blocks * n_emits, n_instances) + tuple(a.shape[4:]))   # level axis (size 1) dropped
        return xp.moveaxis(flat, 0, 1)
    return tree_map(pool, samples)

def tail_grid(cfg, n_tail, *, xp, dtype, device=None):   # [n_tail + 1], exactly u_levels[0] down to exactly 0
    u0 = cfg.u_levels[0]
    return xp.asarray([u0 * (1 - j / n_tail) for j in range(n_tail)] + [0.0], dtype=dtype, device=device)

def crepe_finish(start, cfg, n_tail, *, exact, backend, evaluate, schedule, policy=None,
                 out_needs=frozenset({Feature.X0}), key) -> SMCResult:
    """start: level-0 Records with leading axes [I, N] (e.g. pool_samples(samples)), at t = tau(u_levels[0])."""
    if cfg.u_levels[0] <= 0:
        raise ValueError("crepe_finish needs u_levels[0] > 0")
    xp = namespace(start.t)
    grid = tail_grid(cfg, n_tail, xp=xp, dtype=start.t.dtype, device=device_of(start.t))
    if exact:                                                        # static: weighted, exact target
        target, proposal = cfg.model_denoising, cfg.proposal_denoising
        potential = Relative(FKDiff(cfg.lam))
        policy = ESSBelow(0.5) if policy is None else policy
    else:                                                            # unweighted guided denoising (CREPE paper's finish)
        target = proposal = cfg.proposal_denoising
        potential, policy = Zero(), Never()
    return smc(start, grid, backend=backend, evaluate=evaluate, schedule=schedule, target=target, proposal=proposal,
               potential=potential, policy=policy, out_needs=out_needs, key=key)
```

- **`exact=True`** targets $p_0\, e^{\lambda(\tau(0))\, \hat r}$, the tilted target at $u = 0$ on the path law of `cfg.model_denoising`. The start samples already carry $e^{\lambda(\tau(u_0))\, \hat r}$, so `Relative` makes the starting weight exactly 0 and the final weight $\lambda(\tau(0))\,\hat r_{\text{end}} - \lambda(\tau(u_0))\,\hat r_{\text{start}}$ plus the proposal correction $\log B^P - \log B^Q$ along the tail. The result is weighted (`log_weights`); with `level_lambda`, $\lambda$ is $\beta$ along the whole tail.
- **`exact=False`** denoises with the guided kernel `cfg.proposal_denoising` and no weights (`log_weights` is identically 0). It is cheaper and approximate: its output is not exactly the tilted target.
- **Grouping:** `pool_samples` makes each chain one instance and its post-burn-in samples that instance's particles, so `exact=True` resamples only within a chain. Instances therefore stay as independent as the chains themselves (the API's `[I, ...]` contract). The particles within an instance are correlated unless `crepe` ran with enough `thin`.
- **Memory:** `emit = lambda r: r.project(finish_needs(cfg))` keeps only what the finish reads. `start_record` then projects to the finish's needs (§6), and nothing is re-evaluated (I1).

**One call.** `crepe_sample` runs CREPE end to end: the chains, the pooling, and, when $u_0 > 0$, the finish to $u = 0$.

```python
def crepe_sample(cfg, n_instances, event_shape, n_blocks, *, n_tail=None, exact=None, backend, evaluate, schedule,
                 dtype, device=None, key, initial_ladder=None, thin=1, burn_in=0,
                 policy=None) -> tuple[SMCResult, Record]:
    """-> (result, ladder). result.final: [I, N, ...] at u = 0, carrying exactly cfg.out_needs, with
    N = n_blocks * L // thin; result.log_weights: [I, N]. ladder: crepe's replicas (pass back as initial_ladder)."""
    xp = backend.xp
    if n_blocks < 1:
        raise ValueError("crepe_sample needs n_blocks >= 1")
    needs_finish = cfg.u_levels[0] > 0                               # static
    if needs_finish and (n_tail is None or exact is None):
        raise ValueError("u_levels[0] > 0: crepe_sample needs n_tail and exact")
    emit_needs = finish_needs(cfg) if needs_finish else cfg.out_needs   # both subsets of cfg.replica_needs
    key_chain, key_tail = backend.split(key, 2)
    samples, ladder = crepe(cfg, n_instances, event_shape, n_blocks, lambda r: r.project(emit_needs), backend=backend,
                            evaluate=evaluate, schedule=schedule, dtype=dtype, device=device, key=key_chain,
                            initial_ladder=initial_ladder, thin=thin, burn_in=burn_in)
    start = pool_samples(samples)                                    # [I, N, ...], one instance per chain
    if not needs_finish:                                             # static: level 0 is already u = 0
        return SMCResult(final=start, log_weights=xp.zeros_like(start.t)), ladder
    finished = crepe_finish(start, cfg, n_tail, exact=exact, backend=backend, evaluate=evaluate, schedule=schedule,
                            policy=policy, out_needs=cfg.out_needs, key=key_tail)
    return SMCResult(final=finished.final.project(cfg.out_needs), log_weights=finished.log_weights), ladder
```

- **Why a separate function:** `crepe` stays the bare chain sampler, with any `emit` (diagnostics, autocorrelation estimates for `thin`, the level-0 samples themselves). `crepe_sample` chooses `emit` itself, so the finish never re-evaluates (I1) and the caller can't get `emit` wrong. `crepe`, `pool_samples` and `crepe_finish` stay public.
- **One output type:** the result is an `SMCResult` whether or not $u_0 > 0$. With $u_0 = 0$ there is no finish: `final` is the pooled level-0 samples and `log_weights` is exactly 0. `final` always carries exactly `cfg.out_needs`, so its structure doesn't depend on $u_0$ or `exact`, and the self-normalised `log_weights` average is the estimator in every case.
- **`n_tail` and `exact`** are required when $u_0 > 0$ and ignored when $u_0 = 0$. `policy` is passed to `crepe_finish` (default `ESSBelow(0.5)` for `exact=True`).
- **Keys:** `backend.split(key, 2)`; the first goes to `crepe`, the second to `crepe_finish`.
- **Continuing and memory:** pass `ladder` back as `initial_ladder` with a fresh `key` (and usually `burn_in=0`). Each call finishes only its own samples, so with `exact=True` resampling never mixes samples from different calls. The finish batch is $I N$ rows; to bound it, draw the same total number of samples over several continued calls.


## 17. Exact evaluation counts

These counts are asserted by the tests.

| run | rows evaluated |
|---|---|
| `smc`, `Gaussian` or `Point` start, no move | $I N S$ |
| `smc`, `Record` start, no move | $I N (S - 1)$ |
| plus `PCN` at a moved step | $+\, I N \cdot$ `n_proposals` |
| plus `LangevinMove` at a moved step | $+\, I N$ |
| RpCN | $I (K+1)\,\big(S + M \cdot \#\{\text{steered steps}\}\big)$ |
| `StateSwap` round | $0$ |
| `PathExchange` round | $2K$ per pair, times $I$ |
| `crepe_init` | $I(1 + M K)$ |
| CREPE sweep $n$ | $I\,(2K P_n + \lvert\text{local levels}\rvert + R_n)$ |
| `crepe` with `initial_ladder` given | no `crepe_init` rows; sweeps as above |
| `crepe` with `burn_in` $= b$ | $b L$ sweeps before sampling, each counted as above |
| `crepe_finish`, start `[I, N]` | $I N \cdot$ `n_tail` |
| `crepe_sample` | the `crepe` rows, plus, if $u_0 > 0$, the `crepe_finish` row with $N$ = `n_blocks * L // thin` |

For a CREPE sweep:
- $P_n$ is the number of pairs in `cfg.pairs.rounds(M+1, n)`;
- $R_n = 1$ iff level $M$ appears in sweep $n+1$'s rounds.

## 18. Testing with Hypothesis

### 18.1 Setup

```python
# tests/conftest.py
import pytest
from hypothesis import settings, strategies as st
from hypothesis.extra.array_api import make_strategies_namespace

settings.register_profile("default", deadline=None, max_examples=50)   # JAX compile time breaks deadlines
settings.register_profile("ci", deadline=None, max_examples=200, derandomize=True)
settings.load_profile("default")

array_api_strict.set_array_api_strict_flags(api_version="2025.12")

def available_backends():              # strict always; jax (with jax_enable_x64) and torch if importable
    out = [StrictBackend()]
    if importable("jax"):   jax.config.update("jax_enable_x64", True); out.append(JaxBackend())
    if importable("torch"): out.append(TorchBackend(device="cpu"))
    return out

BACKENDS = available_backends()
SCHEDULES = [VPLinear, VPCosine, EDM, RectifiedFlow]   # constructors take no arguments; defaults as in §5

def xps_for(backend):                  # Hypothesis may not know the newest standard revision yet:
    return make_strategies_namespace(backend.xp, api_version=max(hypothesis.extra.array_api.RELEASED_VERSIONS))
    # array generation uses nothing newer than 2023.12, so declaring an older version is safe
    # (testing/strategies = element_bounds, u_pair, leading_shape and xps_for, as defined in this file)

def element_bounds(lo=-3.0, hi=3.0):
    return {"min_value": lo, "max_value": hi, "allow_nan": False,
            "allow_infinity": False, "allow_subnormal": False}

@st.composite
def u_pair(draw, gap=1e-2):           # u_lo < u_hi, both in [0, 1]
    u_lo = draw(st.floats(0.0, 1.0 - gap)); return u_lo, draw(st.floats(u_lo + gap, 1.0))

@st.composite
def leading_shape(draw, max_instances=3, max_particles=4):
    return draw(st.integers(1, max_instances)), draw(st.integers(1, max_particles))
```

**Pattern.** Parametrize over `backend` and `make_schedule` with `pytest.mark.parametrize`. Inside the test, build `xps = xps_for(backend)` and draw with `st.data()`. This avoids function-scoped fixtures, which trigger a Hypothesis health check.

**Backend notes:**
- **JAX:** `jax_enable_x64` is on in tests only, for the float64 oracle. With it on, JAX's defaults become float64, so every array a test builds names its dtype (§4.1 makes the library itself dtype-explicit).
- **PyTorch:** goes through `array_api_compat.torch`. Hypothesis may warn that it can't verify compliance; that is expected.
- **Fallback:** if array generation misbehaves on a backend, draw with `make_strategies_namespace(array_api_strict)` and convert with `backend.xp.from_dlpack`.

**Execution mode:** counting and duplicate tests must run eagerly (`jax.disable_jit()`). Under `jit`, Python side effects run once per trace.

**Float64 is an oracle, not a library mode.** Identity tests (K, P, M, X) run in float64, so their tolerances test the algebra. Precision tests (N) draw float32 inputs, run the library in float32, and compare against the same computation on the inputs **upcast** to float64. The comparison isolates the library's rounding from the rounding of the inputs. On JAX with `jax_enable_x64`, every float32 array in an N test is created with an explicit dtype.

```python
UNIT_ROUNDOFF32 = 2.0 ** -24             # unit roundoff for float32

def oracle(f, *args):                    # run f on float64 copies of float32 inputs
    to_float64 = lambda a: xp.astype(a, xp.float64) if hasattr(a, "dtype") and a.dtype == xp.float32 else a
    return f(*tree_map(to_float64, args))
```

### 18.2 Test doubles (`ttg/testing`)

```python
class CountingNetwork:                    # contract C1
    def __init__(self, network): self.network, self.calls = network, 0
    def __call__(self, x, t): self.calls += 1; return self.network(x, t)

class AuditingEvaluate:                   # invariant I1: counts rows, raises on a repeated (x, t)
    def __init__(self, evaluate): self.evaluate, self.rows, self.seen = evaluate, 0, set()
    def __call__(self, x, t, needs):
        for x_row, t_row in zip(to_host(x), to_host(t)):
            fingerprint = (x_row.tobytes(), float(t_row))
            if fingerprint in self.seen: raise DuplicateEvaluation(t_row)
            self.seen.add(fingerprint)
        self.rows += t.shape[0]
        return self.evaluate(x, t, needs)

def to_host(a): return numpy.from_dlpack(a)    # eager tests only (torch: .cpu() first)

def index_of(t_grid, t):                  # eager tests only: the k with t_grid[k] == t (all rows agree)
    hits = [k for k in range(t_grid.shape[0]) if bool(xp.all(t_grid[k] == t))]
    assert len(hits) == 1; return hits[0]

@dataclass(frozen=True)                   # deliberately NOT a pytree: lives only in eager tests
class ReplayGaussian(Gaussian):
    forced: Array = None
    def sample(self, key, backend): return self.forced

@dataclass(frozen=True)
class ReplayKernel:                       # proposal double: follows a given path, densities from `base`
    base: Kernel; path: Array; t_grid: Array        # path [S, I, N, *D]; t_grid [S, I, N]
    @property
    def requires(self): return self.base.requires
    def at(self, record, t_dst):
        base_dist = self.base.at(record, t_dst); grid_index = index_of(self.t_grid, t_dst)
        return ReplayGaussian(base_dist.mean, base_dist.var, self.path[grid_index])   # sample() returns that point

def toy_evaluate(prior_mean, prior_std, reward_weights, schedule):
    # data ~ N(prior_mean, prior_std^2 I); reward r(x0) = <reward_weights, x0>; all exact
    # prior_mean, reward_weights: [*D] arrays; prior_std: float. Supports every Feature on every backend (strict included).
    def evaluate(x, t, needs):
        m, s = bcast(schedule.mean(t), x), bcast(schedule.std(t), x)
        marginal_var = m**2 * prior_std**2 + s**2        # marginal variance of x_t
        gain = m * prior_std**2 / marginal_var           # posterior-mean gain
        x0 = prior_mean + gain * (x - m * prior_mean)    # E[x0 | x_t]
        eps = (x - m * x0) / s
        score = -(x - m * prior_mean) / marginal_var     # = -eps / s exactly (Tweedie)
        event_axes = tuple(range(1, x.ndim))
        reward = xp.sum(reward_weights * x0, axis=event_axes)
        reward_grad = gain * reward_weights              # d<reward_weights, x0>/dx, broadcast per row
        features = {X0: x0, EPS: eps, SCORE: score, REWARD: reward, REWARD_GRAD: reward_grad}
        return Record(x, t, **{f.value: (features[f] if f in needs else None) for f in Feature})
    return evaluate
```

The toy target is Gaussian, so tilting by $e^{\lambda \langle w, x_0 \rangle}$ has a closed form. But §18.4 compares methods on the **same grid** rather than against that closed form (see there).

### 18.3 Properties

**Schedules** (all of `SCHEDULES`):

| # | property |
|---|---|
| S1 | `time` is strictly increasing on $[0, 1]$; $m_t > 0$ and $s_t > 0$ there |
| S2 | the SNR $m_t^2 / s_t^2$ is strictly decreasing |
| S3 | `dt_mean`, `dt_std` match central finite differences (tolerance scaled to the step) |
| S4 | `prior` has positive variance and the requested shapes |
| S5 | `log_snr(t)` equals $2(\log m_t - \log s_t)$, and `log_snr_diff(t_a, t_b)` equals `log_snr(t_a) - log_snr(t_b)` (float64; covers the overrides) |

**Conversions:**

| # | property |
|---|---|
| V1 | `eps_from_x0(x0_from_eps(eps)) == eps`; `x == m x0 + s eps` |
| V2 | velocity round trip: build $v = \dot m x_0 + \dot s \varepsilon$ from drawn $x_0, \varepsilon$; recover both |

**Gaussians and kernels:**

| # | property |
|---|---|
| K1 | `log_prob` equals the explicit formula, and is maximised at the mean |
| K2 | **Chapman–Kolmogorov for `Noising`:** composing $u \to t \to s$ gives the $u \to s$ coefficients |
| K3 | **Bayes identity:** with $\hat x_0 = x_0$, `PosteriorStep` density $= F(x_s \mid x_t)\, q(x_t \mid x_0) / q(x_s \mid x_0)$ |
| K4 | `DDIMStep(eta=1)` equals `PosteriorStep` (mean and variance), for any schedule |
| K5 | `Guided(base, scale)` with `scale` $\equiv 0$ equals `base` |
| K6 | every proposal kernel has `var > 0` on `time([0, 1])` (I6) |
| K7 | `Guided(base, normalized(step_length))`'s mean shift has norm $\gamma(t)\,\lVert g \rVert / (\lVert g \rVert + 10^{-12})$ (so $\gamma(t)$ up to the guard), for $g \ne 0$ |
| K8 | `EulerMaruyama` has `var > 0` when used in its configured direction, and `var <= 0` when misused |
| K9 | `diffusion_sq_coeff` equals `beta.rate(t)` for `VPLinear` and `VPCosine`, and $2t$ for `EDM` |
| K11 | `ForwardSDE` requires nothing; `ReverseSDE(eta=1)` drift equals $f x - \gamma^2 \nabla \log p$; `diffusion_sq > 0` for every provided SDE with $\eta > 0$ |
| K10 | **`tilt` is the exact Gaussian tilt:** $\log$ `Guided(base, tilt(lam))` $(x) - \log$ `base` $(x) - \lambda \langle g, x \rangle$ is the same for any two $x$ |
| K12 | `a.log_ratio(b, x)` equals `a.log_prob(x) - b.log_prob(x)` for any means and positive variances; with `a.var` and `b.var` equal, it equals the first term alone, exactly |
| K13 | `transition` and `PosteriorStep` match the $s_s^2 - m_{s\mid t}^2 s_t^2$ forms of §2 and §7 |
| K14 | `ReverseDiffusionStep` for `EDM` ($m \equiv 1$): mean $= x + (s_s^2 - s_t^2)(\hat x_0 - x)/s_s^2$, variance $= s_s^2 - s_t^2$, and the mean equals `PosteriorStep`'s |
| K15 | `ReverseDiffusionStep` for `VPLinear` equals Song et al.'s Algorithm 3 predictor, $\mathcal N\big((2 - \sqrt{1-\beta})\,x + \beta \nabla\log p_s,\ \beta\big)$ with $\beta = 1 - m_{s\mid t}^2$ |
| K16 | `ReverseDiffusionStep`'s variance equals `transition(schedule, t, s)[1]`, and `PosteriorStep`'s times $s_s^2 / s_t^2$, for any schedule |
| K17 | `ReverseDiffusionStep`'s coefficients of $x$ and $\hat x_0$ and its variance agree with `EulerMaruyama(ReverseSDE(schedule), Direction.BACKWARD)`'s and `PosteriorStep`'s to first order: halving the gap ($0.01 \to 0.005$ in $u$, source $u \in [0.1, 0.9]$) shrinks every nonzero difference more than 3x, for any schedule |

**Potentials** (pure, no model):

| # | property |
|---|---|
| P1 | telescoping: for any reward sequence, the final `log_tilt` equals $\lambda\,\hat r_{\text{last}}$ for `FKDiff`, `FKMax`, `FKSum`, and the summed increments equal the final `log_tilt` minus the initial one |
| P2 | `TiltEnds` gives $\ell(\text{end}) - \ell(\text{start})$; `Sum` is additive |
| P3 | `ReverseRatio(K, K)` contributes exactly 0 and requires nothing |
| P4 | $\lambda$-freedom: permuting states along a `ladder` axis and then reading `log_tilt` equals reading each moved state at its new position's $\lambda$ |
| P5 | `increment(prev, update(prev, record, is_last))` equals `log_tilt(new) - log_tilt(prev)` for every potential, at every step including the last; `tilt_delta(p, prev, state_a, state_b)` equals `log_tilt(state_a) - log_tilt(state_b)` for two updates of the same `prev` |
| P6 | `Relative(p)`: `log_tilt(init(record))` is exactly 0 (any backend, any dtype); its increments equal `p`'s; `log_tilt` equals `p.log_tilt(state) - p.log_tilt(p.init(record_0))` (float64) |
| L1 | `level_lambda` is exactly $\beta$ for every $t \le t_{\text{levels}}[0]$ and exactly $0$ for every $t \ge t_{\text{levels}}[-1]$; `piecewise_linear(..., hold=False)` still extrapolates its end segments linearly |

**Precision** (float32 against the upcast float64 oracle). In every N test, draw scalars (λ included) from the **normal** float32 range: some backends, JAX on CPU among them, flush subnormal numbers to zero. Every bound also adds an **underflow floor**: (number of summed terms) × the smallest normal float32, scaled the same way as the bound (divided by $v$ in N2).

| # | property |
|---|---|
| N1 | **increments don't see $C$:** for drawn float32 states with accumulators up to $10^6$ and a non-final record, every FK potential's `increment` is within $4\epsilon\,\big(\lvert \text{inc} \rvert + \lvert\lambda\rvert(\lvert \hat r_{\text{prev}} \rvert + \lvert \hat r \rvert)\big)$ of the oracle. The bound contains no accumulator, so it fails if a $C$ difference ever creeps back in |
| N2 | **`log_ratio` with shared variance:** for $x$ = mean of `b` plus noise, `a.log_ratio(b, x)` is within $8\epsilon \sum_i \lvert \Delta_i \rvert \big(\lvert y_i \rvert + \lvert \Delta_i \rvert\big) / v$, plus the underflow floor, of the oracle. The bound is independent of $D \log v$ |
| N3 | **`log_snr_diff` overrides are relatively accurate:** for `VPLinear`, `VPCosine` and `EDM`, with gaps $s - t$ down to $10^{-6}$ of the range, the float32 result is within $16\epsilon\,\lvert \text{oracle} \rvert$; consequently, `transition` variances are $> 0$ on every such pair (I6) |
| N4 | **structural:** with a spy potential, `smc` calls `log_tilt` only at the start point. `PCN` calls it only when `prev_potential_state is None`, and `StateSwap` (different histories, §12) always does. Everything else goes through `increment`. |

**Resampling:**

| # | property |
|---|---|
| R1 | systematic counts: each index count is within 1 of $N w_i$, per instance |
| R2 | `resample_mask = False` gives an identical tree and identical `log_weights` |
| R3 | alignment: a tree with fields $a$ and $b = f(a)$ still satisfies $b = f(a)$ after resampling; `None` leaves stay `None` |
| R4 | `ess` lies in $[1, N]$; it equals $N$ for uniform weights and $1$ for a one-hot |
| R5 | `categorical` returns `[I]` indices in $[0, N)$, always $j$ for a one-hot at $j$; with $I = 20000$ rows of unnormalised weights $\propto (0.1, 0.2, 0.3, 0.4)$, each frequency is within $5$ standard errors of its weight |

**Moves and replica exchange:**

| # | property |
|---|---|
| M1 | **pCN reversibility:** $\mathcal N(x; \mu, \tau^2)\, q(x \to \hat x) = \mathcal N(\hat x; \mu, \tau^2)\, q(\hat x \to x)$, with $q = \mathcal N(\mu + \sqrt{1-\rho^2}(x - \mu), \rho^2\tau^2)$, exactly (log space) |
| M2 | with $\lambda \equiv 0$, every pCN proposal is accepted |
| M3 | a move never changes `log_weights` (runner, replayed path) |
| M4 | `StateSwap` log-acceptance equals $(\beta_{k+1} - \beta_k)(\hat r(x^{(k)}) - \hat r(x^{(k+1)}))$ (§12) for `FKDiff(ladder(betas))` |
| M5 | schedules: pairs within a round are disjoint and adjacent; over one `period`, every adjacent pair appears; `SequentialSweep` has $n-1$ single-pair rounds |
| M6 | with all $\beta_k$ equal, `SequentialSweep` swaps are always accepted, so the state at position 0 ends at position $n-1$ |
| M7 | a swap keeps each `MoveContext` row internally consistent (`record`, `potential_state`, `prev_potential_state`, `reference` from the same source row) |

**Contract and runner** (eager):

| # | property |
|---|---|
| C1 | for every subset of features (`st.sets(st.sampled_from(Feature))`), one `evaluate` call costs $\le 1$ network call, and returns exactly the requested features |
| C2 | `start_record` raises on a partial `Record` |
| E1 | `AuditingEvaluate` row totals match §17 for drawn $(I, N, S)$, with and without moves; no duplicate raises |
| E2 | `proposal is target` and `Zero()` give `log_weights == 0` exactly |
| E3 | with `ReplayKernel` and `Never()`, the runner's `log_weights` equals a plain-loop reference along the replayed path: $C_{\text{final}} + \sum_k [\log P - \log Q](x_k \mid x_{k-1})$. The runner starts from `log_weights` $= C_{\text{start}}$, the start point's own tilt (FK's $G_T$; `TiltEnds`'s $-\ell_{\text{start}}$), so $C_{\text{start}}$ is **not** subtracted |
| E4 | `Never()` keeps row order (final $x$ = replayed path end, per row) |
| E5 | `move_at(k) == False` for all $k$ gives results identical to `move=None` |

**CREPE:**

| # | property |
|---|---|
| X1 | with replayed $y$, $z$ paths: `PathExchange`'s log-acceptance equals the four-kernel formula, computed directly |
| X2 | with $F^Q = F^P$ and $B^Q = B^P$: it reduces to endpoint tilts; with $F^Q = F^P$ only: to the $B$-only formula of §16 |
| X3 | the replica invariant (fixed level times, exactly `replica_needs`) holds after any number of blocks |
| X4 | evaluation counts per sweep match §17 for both `EvenOdd` and `SequentialSweep` |
| X5 | `emit` may return any pytree: with `emit = lambda r: r.project(finish_needs(cfg))`, `samples` is a `Record` whose leaves have leading axes `[n_blocks, L // thin, I, 1]` with $L = \operatorname{lcm}(\text{period}, \text{thin})$. With a counting `emit`, the run makes `n_blocks * L` sweeps (counts per §17) and emits `n_blocks * L // thin` times, after sweeps `thin, 2 thin, ...`. With `burn_in=b`, the run makes `(b + n_blocks) * L` sweeps, `emit` is first called after sweep `b L + thin`, and `samples` keeps the same shape. `pool_samples(samples)` has leading axes `[I, n_blocks * L // thin]`, keeps every chain's samples in emit order, and never mixes chains within an instance |
| X6 | `crepe(..., initial_ladder=ladder)` makes no `crepe_init` evaluations (counts per §17), keeps the replica invariant, raises `PartialRecord` if `initial_ladder` lacks `replica_needs`, and with `n_blocks = 0` returns `initial_ladder.project(cfg.replica_needs)` unchanged |
| X7 | `crepe_finish` raises `ValueError` when `u_levels[0] == 0`. With `exact=False`, its `log_weights` is identically 0, and it equals `smc(start, tail_grid(...), target=proposal=cfg.proposal_denoising, potential=Zero(), policy=Never())` for the same key. With `exact=True` and `Never()`, its `log_weights` equals, along a replayed tail, $\lambda(\tau(0))\,\hat r_{\text{end}} - \lambda(\tau(u_0))\,\hat r_{\text{start}} + \sum [\log B^P - \log B^Q]$; `tail_grid` starts at exactly `u_levels[0]` and ends at exactly 0 |
| X8 | `crepe_sample` equals, for the same key, the composition it is defined by: `crepe` with `emit = lambda r: r.project(finish_needs(cfg))` and the first key of `backend.split(key, 2)`, then `crepe_finish(pool_samples(samples), ..., out_needs=cfg.out_needs)` with the second, its `final` projected to `cfg.out_needs`; it returns the same `ladder`. Its evaluation counts match §17 (the finish adds nothing beyond its `n_tail` steps). With `u_levels[0] == 0` it runs no finish, `log_weights` is identically 0, and `final` equals `pool_samples` of the level-0 samples projected to `cfg.out_needs`. `final` carries exactly `cfg.out_needs` for both values of `exact` and both cases of `u_levels[0]`. It raises `ValueError` if `n_blocks < 1`, or if `u_levels[0] > 0` and `n_tail` or `exact` is missing |

Three worked examples:

```python
@pytest.mark.parametrize("backend", BACKENDS, ids=str)
@pytest.mark.parametrize("make_schedule", SCHEDULES)
@given(data=st.data())
def test_K3_posterior_is_bayes_inverse_of_noising(backend, make_schedule, data):
    xp, xps, sch = backend.xp, xps_for(backend), make_schedule()
    B = data.draw(st.integers(1, 4))
    D = data.draw(xps.array_shapes(min_dims=1, max_dims=2, max_side=3))
    arr = lambda: data.draw(xps.arrays(xp.float64, (B, *D), elements=element_bounds()))
    x0, x_t, x_s = arr(), arr(), arr()
    u_t, u_s = data.draw(u_pair())
    t, s = xp.full((B,), sch.time(u_t), dtype=xp.float64), xp.full((B,), sch.time(u_s), dtype=xp.float64)
    marg = lambda x, tt: Gaussian(bcast(sch.mean(tt), x0) * x0, sch.std(tt) ** 2).log_prob(x)
    lhs = Noising(sch).at(Point(x_t, t), s).log_prob(x_s) + marg(x_t, t) - marg(x_s, s)
    rhs = PosteriorStep(sch).at(Record(x_s, s, x0=x0), t).log_prob(x_t)
    assert xp.all(xp.abs(lhs - rhs) <= 1e-8 * (1 + xp.abs(rhs)))

@pytest.mark.parametrize("backend", BACKENDS, ids=str)
@pytest.mark.parametrize("Pot", [FKDiff, FKMax, FKSum])
@given(data=st.data(), c=st.floats(0.0, 5.0))
def test_P1_fk_potentials_telescope(backend, Pot, data, c):
    xp, xps = backend.xp, xps_for(backend)
    I, N = data.draw(leading_shape()); S = data.draw(st.integers(2, 6))
    r = data.draw(xps.arrays(xp.float64, (S, I, N), elements=element_bounds()))
    t = xp.linspace(1.0, 0.0, S, dtype=xp.float64)
    rec = lambda k: Record(xp.zeros((I, N, 1), dtype=xp.float64), xp.broadcast_to(t[k], (I, N)), reward=r[k])
    pot = Pot(constant(c))
    st_ = pot.init(rec(0))
    for k in range(1, S):
        st_ = pot.update(st_, rec(k), is_last=(k == S - 1))
    final = pot.log_tilt(st_)
    assert xp.all(xp.abs(final - c * r[S - 1]) <= 1e-9 * (1 + xp.abs(c * r[S - 1])))

@pytest.mark.parametrize("backend", BACKENDS, ids=str)
@given(data=st.data(), rho=st.floats(0.01, 0.99))
def test_M1_pcn_is_reversible_wrt_reference(backend, data, rho):
    xp, xps = backend.xp, xps_for(backend)
    B = data.draw(st.integers(1, 4)); D = data.draw(xps.array_shapes(min_dims=1, max_dims=2, max_side=3))
    arr = lambda: data.draw(xps.arrays(xp.float64, (B, *D), elements=element_bounds()))
    mu, x, x_hat = arr(), arr(), arr()
    tau2 = data.draw(xps.arrays(xp.float64, (B,), elements=element_bounds(0.01, 4.0)))
    ref = Gaussian(mu, tau2)
    q = lambda a: Gaussian(mu + (1 - rho**2) ** 0.5 * (a - mu), rho**2 * tau2)
    lhs = ref.log_prob(x) + q(x).log_prob(x_hat)
    rhs = ref.log_prob(x_hat) + q(x_hat).log_prob(x)
    assert xp.all(xp.abs(lhs - rhs) <= 1e-8 * (1 + xp.abs(lhs)))
```

### 18.4 Statistical tests (not Hypothesis)

These are seeded and marked `slow`. On the Gaussian toy, FK, DAS, FK + pCN, RpCN and CREPE estimates of $\mathbb E[x_0]$ must match the plain sampler on the **same grid with the same model kernel**, reweighted by $e^{\lambda r}$, within Monte Carlo error. Don't compare against the analytic tilted target.

**Concrete setup** (float64, on JAX or PyTorch):

| setting | value |
|---|---|
| toy model | $D = (2,)$, $\mu = (0.5, -0.5)$, $\mathrm{sd} = 1$, $w = (1, 0)$, so $r(x_0) = x_{0,1}$ |
| schedule and kernel | `VPLinear()`, $S = 50$ points uniform in $u$, `PosteriorStep` as $B^P$ |
| tilt | $\lambda = 2$ (constant) unless a row says otherwise; DAS uses `das_lambda(..., alpha_kl=0.5)`, whose final value is $1/\alpha_{\mathrm{KL}} = 2$ |
| estimand | $\mathbb E[x]$ of the **final state** $x$ at $u = 0$ (not $\hat x_0$) |
| reference | 8 seeds × $2^{16}$ independent plain-sampler paths (`smc` with `proposal=target`, `Zero()`, `Never()`) on the **same grid and kernel as the method** (`PosteriorStep` on the 50-point grid, except CREPE), self-normalised weights $e^{\lambda \hat r_{t_{\min}}(x)}$ at the method's $\lambda$. These use the same reward-through-$\hat x_0$ the potentials use, so the reference and the methods target the same distribution |
| FK | $N = 256$, $I = 16$ for each of: `FKDiff(constant(2.0))` with `Every(5)`; `FKMax(constant(0.2))` with `Every(5)` (against the $\lambda = 0.2$ reference); `FKSum(constant(2.0))` with `Never()` and $N = 4096$. `FKMax`/`FKSum` aren't tested with resampling at $\lambda = 2$: that is outside their usage limit (§9) |
| DAS | `Guided(P, tilt(lam))` proposal, `FKDiff(lam)`, `ESSBelow(0.5)`, `lam = das_lambda(t_grid, growth=0.008, alpha_kl=0.5)`; $N = 256$, $I = 16$ |
| FK + pCN | as FK with `FKDiff`, plus `PCN(n_proposals=2, step_size=constant(0.3))` and `move_at=lambda k: k % 5 == 0` |
| RpCN | $K = 3$, `betas = linspace(0, 2, 4)`, `step_sizes = (0.95, 0.95, 0.95, 0.95)`, `PCN(n_proposals=20, step_size=ladder(step_sizes), pairs=EvenOdd())` at **every** step, `Never()`; $I = 2^{12}$; the estimate uses rung $K$ only. RpCN is exact only as mixing becomes complete, so its bound adds $0.02$ for that bias: $4\sqrt{\mathrm{SE}^2 + \mathrm{SE}_{\text{ref}}^2} + 0.02$ |
| CREPE | `u_levels = (0.0, 0.25, 0.5, 0.75, 1.0)`, $K = 10$, `level_lambda(..., beta=2.0, rho=1.0)` (so $\lambda = 2$ at level 0 and $0$ at the top), the default kernels of §16, `Langevin(step_size=lambda t: 0.01 * schedule.std(t)**2)` on levels 0–3, `EvenOdd()`; $I = 2^{10}$ chains, run through `crepe_sample` with `burn_in=100`, `n_blocks=300` (no finish, since $u_0 = 0$); the estimate is the average of `final.x`. **Reference:** the plain sampler using `cfg.model_denoising` on CREPE's own grid (the `fine_grid`s of every gap concatenated, from level $M$ down to level 0), reweighted by $e^{\lambda_0 \hat r}$ at level 0, where $\lambda_0 = 2$ |
| CREPE + finish | as the CREPE row, but `u_levels = (0.25, 0.5, 0.75, 1.0)` (so $u_0 > 0$), run through `crepe_sample` with `n_tail=10`, `exact=True`, `thin=4`, `burn_in=100`, `n_blocks=300` (so $L = 4$ and 300 emits per chain) and the finish's default `ESSBelow(0.5)`; the estimate is the self-normalised `log_weights` average of the final `x`. **Reference:** the plain sampler using `cfg.model_denoising` on CREPE's grid from level $M$ down to level 0, then on `tail_grid(cfg, 10)` down to $u = 0$, reweighted by $e^{2 \hat r}$ at the end. `exact=False` has no statistical test: it is approximate by design |
| pass condition | $\lvert \bar m - \bar m_{\text{ref}} \rvert \le 4\sqrt{\mathrm{SE}^2 + \mathrm{SE}_{\text{ref}}^2}$ for both coordinates, with each standard error taken from the spread across the 8 seeds |
| sanity check | for each tested $\lambda$, the same tolerance must **fail** for $\lambda = 0$ against that $\lambda$'s reference, so the test can detect a missing tilt |
