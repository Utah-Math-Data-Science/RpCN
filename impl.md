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
| 1 | `backend/`, `ops.py`, `schedule.py`, `records.py`, `evaluate.py`, `kernels.py`, `sde.py`, `guidance.py`, `lam.py`, `testing/` (toy model, doubles) | S1–S5, V1–V2, K1–K13, C1–C2, N2–N3 |
| 2 | `potentials.py`, `resample.py`, `smc.py` (FK, DAS) | P1–P5, R1–R4, E1–E4, N1, N4 |
| 3 | `moves.py`, `exchange.py` (FK + pCN, RpCN) | M1–M7, E5, §17 counts with moves |
| 4 | `crepe.py` | X1–X4 |
| 5 | statistical tests (§18.4) | all five methods pass; the $\lambda = 0$ sanity check fails as intended |

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
| I7 | Control flow never branches on array *values*. Step-index-dependent branches use `bk.cond` or `bk.switch`. Step functions are pure and take explicit PRNG keys, so the same code runs under JAX `jit`/`scan` and in an eager PyTorch loop. |
| I8 | One implementation wherever one API covers another. The SMC runner serves FK, DAS, RpCN, CREPE's paths and CREPE's initialisation. One replica-exchange driver serves RpCN and CREPE. |
| I9 | No result is formed by subtracting two large, nearly equal numbers that the library could have avoided forming. Weights add per-step `increment`s, not differences of cumulative tilts. Density ratios use `Gaussian.log_ratio`, not two `log_prob`s. Transition variances use `log_snr_diff`. The library runs in the model's dtype; float64 appears only as a test oracle (§18). |

## 2. Notation

| symbol | meaning | code |
|---|---|---|
| $u \in [0, 1]$ | runner-facing index; $u = 0$ is the schedule's last (cleanest) level, $u = 1$ the reference end | `u` |
| $t = \tau(u)$ | the model's native time (e.g. $t = \sigma$ for EDM) | `t = schedule.time(u)` |
| $m_t,\ s_t$ | $x_t = m_t x_0 + s_t \varepsilon$ | `mean(t)`, `std(t)` |
| $\dot m_t,\ \dot s_t$ | derivatives with respect to $t$ | `dt_mean(t)`, `dt_std(t)` |
| $\mathrm{lsnr}_t = \log(m_t^2 / s_t^2)$ | log signal-to-noise ratio, strictly decreasing in $t$ | `log_snr(t)`; differences `log_snr_diff(s, t)` |
| $\hat x_0,\ \hat\varepsilon,\ \nabla \log p_t$ | denoiser outputs, related by $x = m_t \hat x_0 + s_t \hat\varepsilon$ and $\nabla \log p_t \approx -\hat\varepsilon / s_t$ | `X0`, `EPS`, `SCORE` |
| $\hat r_t(x) = r(\hat x_0(x)),\ g_t(x) = \nabla_x \hat r_t(x)$ | reward and its gradient through the network | `REWARD`, `REWARD_GRAD` |
| $F^P,\ B^P$ | model kernels: exact noising, and the model's denoising step | `Noising`, `PosteriorStep` / `DDIMStep` / `EulerMaruyama(ReverseSDE(sch), Direction.BACKWARD)` |
| $F^Q,\ B^Q$ | proposal kernels: $F^Q$ in the **noising** direction, $B^Q$ in the **denoising** direction | any `Kernel` |
| $\lambda(t),\ \ell_t = \lambda(t)\,\hat r_t$ | tilt strength and log-tilt; may also depend on ladder position | `lam` |
| $C_t$ | cumulative log-tilt $\log \prod_{s \ge t} G_s$ along a particle's history | `potential.log_tilt(state)` |
| $I,\ N$ | independent instances; particles (or ladder positions) per instance | leading axes `[I, N, ...]` |
| $S$ | grid points in an SMC run | |
| $M,\ K$ | CREPE (§16): level gaps, fine steps per gap. RpCN (§15, §17): pCN steps per steered step, top ladder index ($N = K + 1$) | |
| $\beta_k,\ \rho_k$ | RpCN inverse temperatures, pCN step sizes | `ladder(betas)`, `PCN(step=...)` |

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
  schedule.py    Schedule protocol; ScheduleDefaults (log_snr, log_snr_diff); VPLinear, EDM, ...
  records.py     Feature, Point, Record
  evaluate.py    Evaluate contract, conversions, make_evaluate builders (one per backend)
  kernels.py     Gaussian (log_prob, log_ratio); transition; Noising, PosteriorStep, DDIMStep, EulerMaruyama, Langevin, Guided
  sde.py         Direction (FORWARD, BACKWARD); SDE protocol; ForwardSDE, ReverseSDE; f_coeff, g2_coeff
  guidance.py    guidance scales: tilt, normalized
  lam.py         constant, ladder, tabulated, piecewise_linear, das_lambda, level_lambda
  potentials.py  TiltDifference (default increment); Zero, FKDiff, FKMax, FKSum, TiltEnds, ReverseRatio, Sum
  resample.py    ess, systematic, masked_resample; policies Never, Every, ESSBelow
  moves.py       MoveContext, rescore, tilt_delta, PCN, LangevinMove
  exchange.py    PairSchedule (SequentialSweep, EvenOdd), Exchange (StateSwap), replica_exchange
  smc.py         smc()
  crepe.py       CREPEConfig, PathExchange, crepe_init, crepe()
  testing/       CountingNetwork, AuditingEvaluate, ReplayKernel, toy Gaussian model, strategies
```

## 4. Backend

### 4.1 Wiring rules

These rules decide how every module gets its array namespace, dtype, device and pytree behaviour. Every code block in this spec follows them.

| question | rule |
|---|---|
| where does `xp` come from? | **Inferred from the arguments:** `xp = array_api_compat.array_namespace(*arrays)` at the top of any function that receives arrays. In the pseudocode, a bare `xp` means exactly this. No module holds a global namespace. |
| what if a function receives no arrays? | **Constructors take `xp`, `dtype` and `device` explicitly:** `Schedule.prior(lead, event, *, xp, dtype, device=None)`, CREPE's `top_replica`/`crepe_init`/`crepe` (which take `dtype` and `device` and use `bk.xp`), and test builders. Nothing else creates arrays from nothing. |
| what needs `bk`? | **Only randomness, `scan`, `cond` and `switch`.** These are the operations the Array API standard doesn't cover. `bk` is passed explicitly (keyword `bk=`) to every function that needs one of them, and to nothing else. |
| pytrees | **`ops.tree_map` is pure Python** over dataclasses, tuples, lists and dicts. It skips `None` leaves and static fields, and is the same for every backend. Library dataclasses use the `@pytree` decorator (`ops.py`). It marks fields declared `field(metadata={"static": True})` as static, and registers the class with `jax.tree_util.register_dataclass(cls, data_fields, meta_fields)` if JAX is importable. PyTorch needs no registration. |
| dtype | **The run's dtype is the start's dtype** (`start.mean`, `start.x`). `smc` casts `u_grid` to it once, so `t` has it. Schedules, $\lambda$ helpers, kernels and potentials compute in the dtype of the `t`, `x` or `r` they receive. They never create float64 implicitly: `xp.zeros(shape, dtype=t.dtype, device=device(t))`, `xp.arange(n, dtype=t.dtype)`. Python scalars are fine in arithmetic: they adopt the array's dtype. Evaluating the network in another dtype is the user's `evaluate`'s business, but the `Record` it returns must use the run dtype. |
| device | The run's device is the start's device. `bk.normal`/`bk.uniform` take `device=`; the torch backend uses it, JAX ignores it. |
| integer and float arithmetic | **Never mix int and float arrays** (`array_api_strict` raises). Cast with `xp.astype` or create with the float `dtype`. |
| static vs traced | Python `int`/`bool` arguments named `last`, `static`, or documented as static drive Python control flow. The step index inside `scan` is **traced**. Anything that chooses structure from it goes through `bk.cond` or `bk.switch`. |
| scalars | Python scalars appear only in arithmetic operators and in `where`/`clip`; `jax.numpy` and `array_api_compat.torch` both accept those. Don't pass scalars to other functions (e.g. write `xp.clip(a, max=1.0)`, not `xp.minimum(1.0, a)`). |
| Array API version | **2025.12**, the current revision (Python scalars in `where`, `clip`, etc. are allowed). Requires `array-api-compat >= 1.14` and `array-api-strict >= 2.5`, where it is the default. `conftest.py` sets `array_api_strict.set_array_api_strict_flags(api_version="2025.12")` explicitly. |

### 4.2 Backend

```python
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
```

- **JAX:**
  - keys are `jax.random` keys;
  - `scan`, `cond` and `switch` are the `lax` versions.
- **PyTorch:**
  - **Keys:** a key is an `int64` tensor seed. `split(key, n)` seeds a CPU `torch.Generator` with it and draws `n` new `int64` seeds. `normal` and `uniform` seed a generator on `device` with the key.
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
        data = [f.name for f in fields(cls) if not f.metadata.get("static")]
        meta = [f.name for f in fields(cls) if f.metadata.get("static")]
        jax.tree_util.register_dataclass(cls, data_fields=data, meta_fields=meta)
    return cls

def static(default=MISSING): return field(default=default, metadata={"static": True})

def tree_map(f, *trees):                         # pure Python; None leaves are skipped (stay None)
    t = trees[0]
    if t is None: return None
    if is_dataclass(t):
        return replace(t, **{fl.name: tree_map(f, *(getattr(x, fl.name) for x in trees))
                             for fl in fields(t) if not fl.metadata.get("static")})
    if isinstance(t, (tuple, list)): return type(t)(tree_map(f, *xs) for xs in zip(*trees))
    if isinstance(t, dict): return {k: tree_map(f, *(x[k] for x in trees)) for k in t}
    return f(*trees)                             # an array leaf

def leaves(tree):                                # the non-None array leaves, in tree_map order
    out = []; tree_map(lambda a: out.append(a) or a, tree); return out
def importable(name): return importlib.util.find_spec(name) is not None
def jax_available(): return importable("jax")
prod = math.prod
def axis1_size(tree): return leaves(tree)[0].shape[1]
def device(a): return array_api_compat.device(a)

def bcast(v, like):                              # [L...] -> broadcastable against [L..., *D]
    return xp.reshape(v, v.shape + (1,) * (like.ndim - v.ndim))

def logsumexp(a, axis, keepdims=False):          # not in the Array API standard
    m = xp.max(a, axis=axis, keepdims=True)
    m = xp.where(xp.isfinite(m), m, 0.0)
    out = m + xp.log(xp.sum(xp.exp(a - m), axis=axis, keepdims=True))
    return out if keepdims else xp.squeeze(out, axis=axis)

def flatten_leading(a, k):   return xp.reshape(a, (-1,) + a.shape[k:])
def unflatten_leading(a, lead): return xp.reshape(a, tuple(lead) + a.shape[1:])

def take_rows(a, idx):                           # a [I, N, ...], idx [I, N] -> a[i, idx[i, j]]
    I, N = idx.shape
    flat = flatten_leading(a, 2)
    flat_idx = xp.reshape(xp.arange(I, dtype=idx.dtype, device=device(idx))[:, None] * N + idx, (-1,))
    return unflatten_leading(xp.take(flat, flat_idx, axis=0), (I, N))

def take_levels(tree, levels):                   # static list of positions on axis 1
    return tree_map(lambda a: xp.take(a, xp.asarray(levels, device=device(a)), axis=1), tree)

def put_levels(tree, levels, sub):               # no functional index-set in the standard:
    def put(a, b):                               # rebuild axis 1 statically
        cols = [b[:, levels.index(m)] if m in levels else a[:, m] for m in range(a.shape[1])]
        return xp.stack(cols, axis=1)
    return tree_map(put, tree, sub)

def tree_where(mask, a, b):                      # mask [I] or [I, P]; a, b must have identical structure
    return tree_map(lambda u, v: xp.where(bcast(mask, u), u, v), a, b)

def stack_levels(levels):                        # list of trees with axis 1 of size 1 -> one tree
    return tree_map(lambda *a: xp.concat(a, axis=1), *levels)

def steps_first(t_grid, lead):                   # [S] or [*lead, S] -> [S, *lead]
    if t_grid.ndim == 1:
        return xp.broadcast_to(xp.reshape(t_grid, (-1,) + (1,) * len(lead)), (t_grid.shape[0], *lead))
    return xp.moveaxis(t_grid, -1, 0)

def lead_shape(start):                           # Gaussian -> var.shape; Point / Record -> t.shape
    return start.var.shape if isinstance(start, Gaussian) else start.t.shape
```

**Array-API gaps to check per backend:**
- `cumulative_sum`;
- `searchsorted` (1-D sorted first argument only);
- `take` with `axis`.

Systematic resampling avoids `searchsorted` on 2-D arrays (§10).

## 5. Schedule

```python
class Schedule(Protocol):
    def time(self, u): ...           # u in [0, 1] -> native t; strictly increasing
    def mean(self, t): ...           # m_t > 0 on time([0, 1])
    def std(self, t): ...            # s_t > 0 on time([0, 1])   (so u = 0 is NOT s = 0)
    def dt_mean(self, t): ...        # d m_t / dt
    def dt_std(self, t): ...         # d s_t / dt
    def prior(self, lead, event, *, xp, dtype, device=None) -> Gaussian: ...
    # approx. p at time(1); pCN needs mean and variance. event: tuple, e.g. (C, H, W)
    # all methods except prior take arrays and compute in their dtype (§4.1)
    def log_snr(self, t): ...        # lsnr_t = log(m_t^2 / s_t^2); default from mean/std
    def log_snr_diff(self, s, t): ...   # lsnr_s - lsnr_t; default log_snr(s) - log_snr(t)

class ScheduleDefaults:              # mixin: the two optional methods
    def log_snr(self, t):
        return 2 * (xp.log(self.mean(t)) - xp.log(self.std(t)))
    def log_snr_diff(self, s, t):    # accurate only to ~eps * |lsnr|; override for tiny gaps
        return self.log_snr(s) - self.log_snr(t)
```

**`log_snr_diff`.** The defaults are correct for any schedule. Where a schedule can write the difference as a closed form in $s - t$, override it (as below): the default loses accuracy on very fine grids.

The schedule owns its edge cases by choosing $\tau$'s range. Implement these three:

```python
@dataclass(frozen=True)
class VPLinear(ScheduleDefaults):    # beta(t) = b0 + t (b1 - b0),  alpha_bar = exp(-int beta)
    b0: float = 0.1; b1: float = 20.0; t_min: float = 1e-3; t_max: float = 1.0
    def time(self, u):    return self.t_min + u * (self.t_max - self.t_min)
    def _log_ab(self, t): return -(self.b0 * t + 0.5 * (self.b1 - self.b0) * t**2)
    def _beta(self, t):   return self.b0 + t * (self.b1 - self.b0)
    def mean(self, t):    return xp.exp(0.5 * self._log_ab(t))
    def std(self, t):     return xp.sqrt(-xp.expm1(self._log_ab(t)))
    def dt_mean(self, t): return -0.5 * self._beta(t) * self.mean(t)
    def dt_std(self, t):  return 0.5 * self._beta(t) * xp.exp(self._log_ab(t)) / self.std(t)
    def prior(self, lead, event, *, xp, dtype, device=None):
        return Gaussian(xp.zeros((*lead, *event), dtype=dtype, device=device), xp.ones(lead, dtype=dtype, device=device))
    def log_snr(self, t): return self._log_ab(t) - xp.log(-xp.expm1(self._log_ab(t)))
    def log_snr_diff(self, s, t):
        d = -(s - t) * (self.b0 + 0.5 * (self.b1 - self.b0) * (s + t))   # log ab_s - log ab_t, via s - t
        a_t = self._log_ab(t)
        return d - xp.log1p(xp.exp(a_t) * xp.expm1(d) / xp.expm1(a_t))  # minus log((1 - ab_s) / (1 - ab_t))

@dataclass(frozen=True)
class EDM(ScheduleDefaults):         # t = sigma; non-uniform spacing lives in time(u)
    s_min: float = 0.002; s_max: float = 80.0; rho: float = 7.0
    def time(self, u):    a, b = self.s_min**(1/self.rho), self.s_max**(1/self.rho); return (a + u*(b - a))**self.rho
    def mean(self, t):    return xp.ones_like(t)
    def std(self, t):     return t
    def dt_mean(self, t): return xp.zeros_like(t)
    def dt_std(self, t):  return xp.ones_like(t)
    def prior(self, lead, event, *, xp, dtype, device=None):
        return Gaussian(xp.zeros((*lead, *event), dtype=dtype, device=device),
                        xp.full(lead, self.s_max**2, dtype=dtype, device=device))
    def log_snr(self, t): return -2 * xp.log(t)
    def log_snr_diff(self, s, t): return -2 * xp.log1p((s - t) / t)

@dataclass(frozen=True)
class RectifiedFlow(ScheduleDefaults):  # x_t = (1 - t) x0 + t eps, clipped to [t_min, t_max] inside (0, 1)
    t_min: float = 1e-3; t_max: float = 0.999
    def time(self, u):    return self.t_min + u * (self.t_max - self.t_min)
    def mean(self, t):    return 1 - t
    def std(self, t):     return t
    def dt_mean(self, t): return -xp.ones_like(t)
    def dt_std(self, t):  return xp.ones_like(t)
    def prior(self, lead, event, *, xp, dtype, device=None):   # ignores the (1 - t_max) x0 part
        return Gaussian(xp.zeros((*lead, *event), dtype=dtype, device=device),
                        xp.full(lead, self.t_max**2, dtype=dtype, device=device))
    def log_snr(self, t): return 2 * (xp.log1p(-t) - xp.log(t))
    def log_snr_diff(self, s, t): return 2 * (xp.log1p(-(s - t) / (1 - t)) - xp.log1p((s - t) / t))
```

All three are used by the tests (`SCHEDULES`, §18).

**Final denoising** is the user's choice. Requesting `X0` in `final_needs` returns $\hat x_0$ at the last point at no extra cost, because that point is evaluated anyway.

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
    def need(self, f: Feature) -> Array:
        v = getattr(self, f.value)
        if v is None:
            raise MissingFeature(f)  # never silently re-evaluate
        return v
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
def x0_from_eps(x, t, eps, sch):   return (x - bcast(sch.std(t), x) * eps) / bcast(sch.mean(t), x)
def eps_from_x0(x, t, x0, sch):    return (x - bcast(sch.mean(t), x) * x0) / bcast(sch.std(t), x)
def score_from_eps(x, t, eps, sch): return -eps / bcast(sch.std(t), x)
def eps_from_score(x, t, sc, sch):  return -bcast(sch.std(t), x) * sc
def x0_eps_from_velocity(x, t, v, sch):          # v ~ dm/dt x0 + ds/dt eps
    m, s, dm, ds = (bcast(f(t), x) for f in (sch.mean, sch.std, sch.dt_mean, sch.dt_std))
    den = m * ds - dm * s                        # = m^2 d(s/m)/dt != 0
    return (ds * x - s * v) / den, (m * v - dm * x) / den
```

**Builders.** These are optional; users may write `evaluate` directly. There is one builder per backend that has autodiff. `array_api_strict` has no autodiff, so on it `REWARD_GRAD` comes only from the analytic toy model (§18.2). Builder for JAX:

```python
def make_evaluate_jax(network, to_x0_eps, sch, reward):
    def evaluate(x, t, needs):
        grad = Feature.REWARD_GRAD in needs
        want_r = grad or Feature.REWARD in needs
        def fwd(x):
            x0, eps = to_x0_eps(x, t, network(x, t), sch)       # the single forward pass
            r = reward(x0) if want_r else None
            return (xp.sum(r) if grad else 0.0), (x0, eps, r)
        if grad:
            (_, (x0, eps, r)), g = jax.value_and_grad(fwd, has_aux=True)(x)
        else:
            _, (x0, eps, r) = fwd(x); g = None
        keep = lambda f, v: v if f in needs else None
        return Record(x, t, x0=keep(X0, x0), eps=keep(EPS, eps),
                      score=keep(SCORE, score_from_eps(x, t, eps, sch)),
                      reward=keep(REWARD, r), reward_grad=keep(REWARD_GRAD, g))
    return evaluate

def make_evaluate_torch(network, to_x0_eps, sch, reward):
    def evaluate(x, t, needs):
        grad = Feature.REWARD_GRAD in needs
        want_r = grad or Feature.REWARD in needs
        with torch.enable_grad() if grad else torch.no_grad():
            xg = x.detach().requires_grad_(grad)
            x0, eps = to_x0_eps(xg, t, network(xg, t), sch)      # the single forward pass
            r = reward(x0) if want_r else None
            g = torch.autograd.grad(r.sum(), xg)[0] if grad else None
        det = lambda v: None if v is None else v.detach()
        keep = lambda f, v: det(v) if f in needs else None
        return Record(x, t, x0=keep(X0, x0), eps=keep(EPS, eps),
                      score=keep(SCORE, score_from_eps(x, t, eps, sch)),
                      reward=keep(REWARD, r), reward_grad=keep(REWARD_GRAD, g))
    return evaluate
```

`reward(x0)` maps `[B, *D] -> [B]`. The PyTorch builder detaches everything it returns; JAX values carry no graph. The library never differentiates through a `Record`.

**Runner-side helpers:**

```python
def evaluate_rows(evaluate, p: Point, needs, bk):            # [I, N] <-> [B]; the only call site of evaluate
    lead = p.t.shape
    rec = evaluate(flatten_leading(p.x, len(lead)), xp.reshape(p.t, (-1,)), frozenset(needs))
    return tree_map(lambda a: unflatten_leading(a, lead), rec)

def start_record(start, t0, needs, evaluate, bk, key):       # -> (Record, reference Gaussian or None)
    if isinstance(start, Gaussian):                           # e.g. schedule.prior(...): sample it, keep it
        return evaluate_rows(evaluate, Point(start.sample(key, bk), t0), needs, bk), start
    if isinstance(start, Point):
        return evaluate_rows(evaluate, start, needs, bk), None
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
    def sample(self, key, bk):
        noise = bk.normal(key, self.mean.shape, self.mean.dtype, device=device(self.mean))
        return self.mean + bcast(xp.sqrt(self.var), self.mean) * noise
    def log_prob(self, x):
        ev = tuple(range(self.var.ndim, x.ndim)); d = prod(x.shape[self.var.ndim:])
        return -0.5 * xp.sum((x - self.mean) ** 2, axis=ev) / self.var - 0.5 * d * xp.log(2 * xp.pi * self.var)
    def log_ratio(self, other, x):    # log self(x) - log other(x), without forming either log-density
        ev = tuple(range(self.var.ndim, x.ndim)); d = prod(x.shape[self.var.ndim:])
        delta = self.mean - other.mean            # exact where the means are close (Sterbenz)
        y = x - other.mean                        # the sampled noise when x ~ other
        dv = self.var - other.var                 # exactly 0 when the variances are equal
        quad = xp.sum(delta * (y - 0.5 * delta), axis=ev) / self.var
        return (quad + 0.5 * xp.sum(y**2, axis=ev) * dv / (self.var * other.var)
                     - 0.5 * d * xp.log1p(dv / other.var))

def transition(sch, t, s):            # t < s: (m_{s|t}, sigma^2_{s|t}), cancellation-free (§2)
    return sch.mean(s) / sch.mean(t), -sch.std(s) ** 2 * xp.expm1(sch.log_snr_diff(s, t))

class Kernel(Protocol):
    requires: frozenset[Feature]
    def at(self, rec, t_dst) -> Gaussian: ...    # every kernel goes one way only
    # t_dst has rec.t's shape [L...] (one destination time per row); the result is per row
```

| kernel | direction | distribution, from source $x$ at $s$ to $t$ | `requires` |
|---|---|---|---|
| `Noising(sch)` = $F^P$ | forward ($t > s$) | $\mathcal N\!\big(m_{t\mid s} x,\ \sigma^2_{t\mid s}\big)$ | none |
| `PosteriorStep(sch)` | backward | $\mathcal N\!\big(\tfrac{m_{s\mid t} s_t^2}{s_s^2} x + \tfrac{m_t \sigma^2_{s\mid t}}{s_s^2}\hat x_0,\ \tau^2\big)$, $\tau^2 = \tfrac{\sigma^2_{s\mid t} s_t^2}{s_s^2}$ | `X0` |
| `DDIMStep(sch, eta)`, $0 < \eta \le 1$ | backward | $\mathcal N\!\big(m_t \hat x_0 + \sqrt{s_t^2 - \eta^2 \tau^2}\,\hat\varepsilon,\ \eta^2 \tau^2\big)$ | `X0`, `EPS` |
| `EulerMaruyama(sde, direction)` | `Direction.FORWARD` or `Direction.BACKWARD` | $\mathcal N\!\big(x + \Delta\, b(x, s),\ \sigma^2_{\mathrm{sde}}(s)\,\lvert\Delta\rvert\big)$, signed $\Delta = t - s$; drift $b$ and diffusion $\sigma^2_{\mathrm{sde}}$ from the injected `SDE` | `sde.requires` |
| `Langevin(h)`, $h$ a callable of $t$ | stays at $s$ | $\mathcal N\!\big(x + h(s) \nabla \log p,\ 2h(s)\big)$ | `SCORE` |
| `Guided(base, scale)` | as base | base mean $+\ \kappa\, g$ with $\kappa = \texttt{scale}(t, g, \mathrm{Var}_{\text{base}})$, $g$ **at the source**; same variance | base + `REWARD_GRAD` |

- **Coefficients:** the drift and diffusion come from the schedule, $f = \dot m / m$ and $\gamma^2 = 2 s \dot s - 2 f s^2$. For VP this gives $\gamma^2 = \beta$.
- **`EulerMaruyama` is one class for both directions, and the drift is injected.** It takes one Euler–Maruyama step of an `SDE`, $\mathrm dx = b(x, t)\,\mathrm dt + \sigma_{\mathrm{sde}}(t)\,\mathrm dW$, written in forward-time convention, from $s$ to $t$ with signed $\Delta = t - s$.
  - **The SDE supplies both drift and diffusion.** The drift is evaluated at the source and declares its own `requires`.
  - **The integration direction is the only other choice.** `Direction.BACKWARD` integrates with $\Delta < 0$, as for a reverse-time SDE. `Direction.FORWARD` integrates with $\Delta > 0$.
  - **The switch fixes the sign used in the variance**, $\sigma^2_{\mathrm{sde}} \cdot \mathrm{sign} \cdot \Delta$. A kernel used in the wrong direction therefore gets a non-positive variance and fails test K8, rather than silently running backwards.
- **Provided SDEs** (§7a): `ForwardSDE` (the noising SDE, $b = f x$, no network) and `ReverseSDE(eta)`. Anything else (for example a tempering drift) is a user-written `SDE`. `EulerMaruyama(ForwardSDE(sch), Direction.FORWARD)` is the Euler–Maruyama approximation of noising, which differs from the exact `Noising` kernel.
- **`Guided` is one class for every gradient guidance.** The per-row scale $\kappa = \texttt{scale}(t, g, \mathrm{Var})$ sees the gradient and the base kernel's variance (§8a). $t$ is the **destination** time, so $\lambda$ is the one of the target the step proposes for, while $g$ is taken at the source:
  - `tilt(lam)` gives $\kappa = \mathrm{Var} \cdot \lambda(t)$: the exact Gaussian tilt by $e^{\lambda \langle g, x \rangle}$. Used by DAS and CREPE's $B^Q$.
  - `tilt(lam, 0.5)` gives Langevin's reward drift $h\lambda g$, since $\mathrm{Var} = 2h$.
  - `normalized(gamma)` gives $\kappa = \gamma(t)/\lVert g \rVert$: for inverse problems, with the gradient taken at the source point. Use it as **both** target and proposal: there is then no weight correction, and pCN preserves it.

**CREPE's default proposals:**

| kernel | drift | implementation | `requires` |
|---|---|---|---|
| $F^Q$ (noising) | plain noising drift $f x$ | `Noising(sch)`, i.e. $F^Q = F^P$ | none |
| $B^Q$ (denoising) | reward-tilted reverse drift $f x - \gamma^2(\nabla \log p + \lambda g)$ | `Guided(EulerMaruyama(ReverseSDE(sch), Direction.BACKWARD), tilt(lam))` | `SCORE`, `REWARD_GRAD` |

Use the same base discretisation for $B^P$ and $B^Q$, so that $\log B^P - \log B^Q$ reflects only the guidance shift.

Implementations:

```python
@dataclass(frozen=True)
class Noising:                    # exact forward kernel F^P
    sch: Schedule
    requires = frozenset()
    def at(self, rec, t):
        m, var = transition(self.sch, rec.t, t)                     # source rec.t < destination t
        return Gaussian(bcast(m, rec.x) * rec.x, var)

@dataclass(frozen=True)
class PosteriorStep:
    sch: Schedule
    requires = frozenset({Feature.X0})
    def at(self, rec, t):
        s = rec.t                                                   # per-row scalars, shape [L...]
        ss, mt, st = self.sch.std(s), self.sch.mean(t), self.sch.std(t)
        m_st = self.sch.mean(s) / mt
        e = -xp.expm1(self.sch.log_snr_diff(s, t))                  # = sigma^2_{s|t} / s_s^2, in (0, 1)
        a, b = m_st * st**2 / ss**2, mt * e
        mean = bcast(a, rec.x) * rec.x + bcast(b, rec.x) * rec.need(Feature.X0)
        return Gaussian(mean, st**2 * e)                            # tau^2 = sigma^2_{s|t} s_t^2 / s_s^2

@dataclass(frozen=True)
class DDIMStep:                   # eta in (0, 1]; eta = 1 equals PosteriorStep (test K4)
    sch: Schedule; eta: float
    requires = frozenset({Feature.X0, Feature.EPS})
    def at(self, rec, t):
        s = rec.t
        e = -xp.expm1(self.sch.log_snr_diff(s, t))                  # sigma^2_{s|t} / s_s^2
        st = self.sch.std(t)
        var = self.eta**2 * st**2 * e                               # eta^2 tau^2
        c = st * xp.sqrt(1 - self.eta**2 * e)                       # sqrt(s_t^2 - eta^2 tau^2); > 0 for eta <= 1
        mean = bcast(self.sch.mean(t), rec.x) * rec.need(Feature.X0) + bcast(c, rec.x) * rec.need(Feature.EPS)
        return Gaussian(mean, var)

@dataclass(frozen=True)
class Langevin:                   # one unadjusted Langevin step at fixed t
    h: Callable                   # step size as a function of t, e.g. lambda t: 0.01 * sch.std(t)**2
    requires = frozenset({Feature.SCORE})
    def at(self, rec, t):         # t == rec.t
        h = self.h(rec.t)
        return Gaussian(rec.x + bcast(h, rec.x) * rec.need(Feature.SCORE), 2 * h)

@dataclass(frozen=True)
class EulerMaruyama:              # one EM step of the injected SDE, integrated up or down
    sde: SDE
    direction: Direction
    @property
    def requires(self): return self.sde.requires
    def at(self, rec, t):
        delta = t - rec.t                                           # signed
        sign = float(self.direction.value)                          # static switch: +1 or -1
        mean = rec.x + bcast(delta, rec.x) * self.sde.drift(rec)    # drift at the source
        return Gaussian(mean, self.sde.diffusion2(rec.t) * sign * delta)   # var <= 0 if misused (K8)

@dataclass(frozen=True)
class Guided:                     # base mean + scale(t, g, var) * g, with g at the source
    base: Kernel; scale: Callable
    @property
    def requires(self): return self.base.requires | {Feature.REWARD_GRAD}
    def at(self, rec, t):
        d = self.base.at(rec, t); g = rec.need(Feature.REWARD_GRAD)
        return Gaussian(d.mean + bcast(self.scale(t, g, d.var), g) * g, d.var)
```

`PosteriorStep` and `DDIMStep` both write $\sigma^2_{s|t}/s_s^2 = -\operatorname{expm1}(\mathrm{lsnr}_s - \mathrm{lsnr}_t)$. In `DDIMStep`, $s_t^2 - \eta^2\tau^2 = s_t^2\,\big(1 - \eta^2 \sigma^2_{s|t}/s_s^2\big)$, which is positive because $\sigma^2_{s|t} < s_s^2$ and $\eta \le 1$.

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
- **Equal variances** (`Guided` against its base, and the same-base $B^P$/$B^Q$ pairing of §7): `dv` is exactly 0, so only the first term remains.
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
    def drift(self, rec) -> Array: ...    # b(x, t) at the source, [L..., *D]
    def diffusion2(self, t) -> Array: ... # sigma_sde(t)^2, [L...]

def f_coeff(sch, t):  return sch.dt_mean(t) / sch.mean(t)                                   # f = m'/m
def g2_coeff(sch, t): return 2 * sch.std(t) * sch.dt_std(t) - 2 * f_coeff(sch, t) * sch.std(t) ** 2   # gamma^2

@dataclass(frozen=True)
class ForwardSDE:                         # the noising SDE whose marginals the schedule defines; no network
    sch: Schedule
    requires = frozenset()
    def drift(self, rec): return bcast(f_coeff(self.sch, rec.t), rec.x) * rec.x
    def diffusion2(self, t): return g2_coeff(self.sch, t)

@dataclass(frozen=True)
class ReverseSDE:                         # same marginals, run backwards; eta = 1 is the standard reverse-time SDE
    sch: Schedule; eta: float = 1.0       # eta > 0 required (eta = 0 is the probability-flow ODE: no density, I6)
    requires = frozenset({Feature.SCORE})
    def drift(self, rec):
        c = -0.5 * (1 + self.eta**2) * g2_coeff(self.sch, rec.t)
        return bcast(f_coeff(self.sch, rec.t), rec.x) * rec.x + bcast(c, rec.x) * rec.need(Feature.SCORE)
    def diffusion2(self, t): return self.eta**2 * g2_coeff(self.sch, t)
```

- **`ReverseSDE(eta)`** is the standard family that shares the forward marginals: drift $f x - \tfrac{1+\eta^2}{2}\gamma^2 \nabla \log p$, diffusion $\eta^2 \gamma^2$.
- **Reward guidance goes through `Guided`, not into the drift.** That keeps one path for guidance, and for an Euler–Maruyama step the two are identical: a drift term $-\gamma^2 \lambda g$ integrated downward shifts the mean by exactly $\gamma^2 \lvert\Delta\rvert \lambda g = \mathrm{Var} \cdot \lambda \cdot g$, i.e. `tilt(lam)`. A custom `SDE` may still put anything in its drift.

## 8. Tilt strength $\lambda$

The runner never sees $\lambda$. It is held by the objects that use it, which evaluate it per row from the $t$ they are handed. It may also depend on the row's **position** on axis 1 (a ladder).

```python
def take1d(v, idx):              # Array API `take` wants 1-D indices
    return xp.reshape(xp.take(v, xp.reshape(idx, (-1,))), idx.shape)

def constant(c):                return lambda t: xp.full_like(t, c)
def ladder(values):             # value by position on axis 1 (RpCN's beta_k, pCN step sizes); t: [I, N]
    return lambda t: xp.broadcast_to(xp.astype(values, t.dtype)[None, :], t.shape)
def tabulated(knots, values):   # exact at the knots; nearest-knot lookup (only knots are ever visited)
    return lambda t: take1d(xp.astype(values, t.dtype), xp.argmin(xp.abs(t[..., None] - xp.astype(knots, t.dtype)), axis=-1))
def piecewise_linear(knots, values):     # knots: 1-D increasing
    def lam(t):
        kn, va = xp.astype(knots, t.dtype), xp.astype(values, t.dtype)
        j = xp.clip(xp.searchsorted(kn, xp.reshape(t, (-1,))), 1, kn.shape[0] - 1)
        j = xp.reshape(j, t.shape)                   # searchsorted wants a 1-D needle-free API; flatten t
        k0, k1, v0, v1 = take1d(kn, j - 1), take1d(kn, j), take1d(va, j - 1), take1d(va, j)
        return v0 + (t - k0) / (k1 - k0) * (v1 - v0)
    return lam

def das_lambda(t_grid, gamma, alpha_kl):   # t_grid[0] noisiest; lambda_tilde_k = min(1, (1+gamma)^k - 1); last = 1
    k = xp.arange(t_grid.shape[0], dtype=t_grid.dtype)           # float: no int/float mixing (§4.1)
    lt = xp.where(k == t_grid.shape[0] - 1, 1.0, xp.clip((1 + gamma) ** k - 1, max=1.0))
    return tabulated(t_grid, lt / alpha_kl)

def level_lambda(t_levels, beta, rho):     # CREPE: beta at the data end, 0 at the noise end
    M = t_levels.shape[0] - 1
    return piecewise_linear(t_levels, beta * (1 - xp.arange(M + 1, dtype=t_levels.dtype) / M) ** rho)
```

### 8a. Guidance scales

`Guided` takes a guidance scale $\kappa = \texttt{scale}(t, g, \mathrm{Var})$, one value per row. It extends the tilt strength $\lambda(t)$ by also seeing the gradient $g$ and the base kernel's variance. Potentials keep plain $\lambda(t)$, so DAS can pass the same `lam` to both.

```python
def event_norm(g, lead_ndim):    # ||g|| per row, over the event dims
    return xp.sqrt(xp.sum(g**2, axis=tuple(range(lead_ndim, g.ndim))))

def tilt(lam, factor=1.0):       # exact Gaussian tilt by exp(lam <g, x>): shift = factor * Var * lam(t) * g
    return lambda t, g, var: factor * var * lam(t)

def normalized(gamma, eps=1e-12):   # RpCN inverse problems: shift = gamma(t) * g / ||g||
    return lambda t, g, var: gamma(t) / (event_norm(g, var.ndim) + eps)
```

## 9. Potentials

A potential exposes the **cumulative** log-tilt $C$ of a particle's history, for comparing whole histories (swaps). It also exposes the **increment** $C(\text{new}) - C(\text{prev})$ of one step, computed directly from the step's statistics. The runner accumulates increments, and moves compare increments over a shared history.

```python
class Potential(Protocol):
    requires: frozenset[Feature]
    def init(self, rec) -> State: ...                          # state after the first point
    def update(self, state, rec, *, last: bool) -> State: ... # absorb the next point; `last` is static
    def log_tilt(self, state) -> Array: ...                    # C = log prod_{s >= t} G_s, shape [I, N]
    def increment(self, prev, state) -> Array: ...             # C(state) - C(prev), never formed as that difference
    # precondition: state = update(prev, rec, last=...) for some rec
# runner: log_w += increment(st, update(st, new))

class TiltDifference:                 # mixin: the default increment, for potentials whose C stays O(one term)
    def increment(self, prev, st): return self.log_tilt(st) - self.log_tilt(prev)
```

**The runner and moves use `increment`, never `log_tilt` differences.** $C$ grows along the path for `FKMax`, `FKSum` and `ReverseRatio`, so differencing it loses precision. `FKDiff` uses the default (its $C$ is a single term). `TiltEnds` returns exactly $0$ at interior steps and exactly $\ell$ at the last.

**Rule: state is $\lambda$-free.** It stores raw statistics, and $\lambda$ is applied in `log_tilt` and `increment` from the row's time and position. That is what makes swaps correct: a state moved to another ladder position is automatically read at that position's temperature.

| potential | state | `log_tilt` | `increment(prev, st)` |
|---|---|---|---|
| `FKDiff(lam)` (also DAS, with tempered $\lambda$) | $(\hat r, t)$ | $\lambda(t)\,\hat r$ | default: $\lambda(t')\,\hat r' - \lambda(t)\,\hat r$ |
| `FKMax(lam)` | running max, $\sum$ of running maxes, $\hat r$, $t$, `last` | $\lambda \cdot$ accumulator; at the last point, $\lambda \hat r$ | $\lambda \cdot$ new running max; at the last point, $\lambda(\hat r - \text{acc}_{\text{prev}})$ |
| `FKSum(lam)` | running sum, $\sum$ of running sums, $\hat r$, $t$, `last` | same pattern | same pattern, with the running sum |
| `TiltEnds(lam)` | $(\hat r_{\mathrm{start}}, t_{\mathrm{start}}, \hat r, t, \text{last})$ | $-\ell_{\mathrm{start}} + [\text{last}]\,\ell$ | $0$; at the last point, $\ell$ |
| `ReverseRatio(P_rev, Q_rev)` | $(\text{sum}, \text{term}, x_{\mathrm{prev}}, t_{\mathrm{prev}})$; left unchanged by `update` when trivial | sum | the latest term |
| `Sum(*ps)` | tuple of the parts' states | sum | sum of the parts' increments |
| `Zero()` | zeros of shape `[I, N]` (the shape is what `smc` needs for `log_w`) | that zero array | that zero array |

- **`FKMax` and `FKSum`** require $\lambda$ constant in $t$ (it may vary by ladder position), as in FK steering.
- **Usage limit of `FKMax` and `FKSum`** (put it in both docstrings): they target $e^{\lambda \hat r_0}$ exactly without resampling (`Never()`). With resampling, they are reliable only when $\lambda S \sigma_r$ (max) or $\lambda \tfrac{S^2}{2} \sigma_r$ (sum) is small, where $\sigma_r$ is the spread of the final predicted reward across particles. These bounds are conservative: the effective counts are the steps on which $\hat r$ already differs between particles.
- **Telescoping.** All FK potentials end with $C = \lambda \hat r_0$, which is property P1 (§18).

```python
@pytree
class TiltState:  r: Array; t: Array
@pytree
class PathState:  run: Array; acc: Array; r: Array; t: Array; last: bool = static(False)
@pytree
class EndsState:  r0: Array; t0: Array; r: Array; t: Array; last: bool = static(False)
@pytree
class RRState:    sum: Array; term: Array; x: Array; t: Array

@dataclass(frozen=True)
class FKDiff(TiltDifference):
    lam: Callable
    requires = frozenset({Feature.REWARD})
    def init(self, rec): return TiltState(r=rec.need(Feature.REWARD), t=rec.t)
    def update(self, st, rec, *, last): return TiltState(r=rec.need(Feature.REWARD), t=rec.t)
    def log_tilt(self, st): return self.lam(st.t) * st.r

@dataclass(frozen=True)
class FKMax:
    lam: Callable
    requires = frozenset({Feature.REWARD})
    def init(self, rec):
        r = rec.need(Feature.REWARD); return PathState(run=r, acc=r, r=r, t=rec.t, last=False)
    def update(self, st, rec, *, last):
        r = rec.need(Feature.REWARD); run = xp.maximum(st.run, r)
        return PathState(run=run, acc=st.acc + run, r=r, t=rec.t, last=last)     # FKSum: run = st.run + r
    def log_tilt(self, st): return self.lam(st.t) * (st.r if st.last else st.acc)
    def increment(self, prev, st):                                  # `last` is static
        return self.lam(st.t) * ((st.r - prev.acc) if st.last else st.run)

@dataclass(frozen=True)
class FKSum(FKMax):                   # identical except the running statistic
    def update(self, st, rec, *, last):
        r = rec.need(Feature.REWARD); run = st.run + r
        return PathState(run=run, acc=st.acc + run, r=r, t=rec.t, last=last)

@dataclass(frozen=True)
class TiltEnds:                       # C = -l(start) + [last] l(end): CREPE's endpoint tilts
    lam: Callable
    requires = frozenset({Feature.REWARD})
    def init(self, rec):
        r = rec.need(Feature.REWARD); return EndsState(r0=r, t0=rec.t, r=r, t=rec.t, last=False)
    def update(self, st, rec, *, last):
        return EndsState(r0=st.r0, t0=st.t0, r=rec.need(Feature.REWARD), t=rec.t, last=last)
    def log_tilt(self, st):
        start = -self.lam(st.t0) * st.r0
        return start + self.lam(st.t) * st.r if st.last else start
    def increment(self, prev, st):
        return self.lam(st.t) * st.r if st.last else xp.zeros_like(st.r)

@dataclass(frozen=True)
class Sum:
    parts: tuple[Potential, ...]
    def __init__(self, *parts): object.__setattr__(self, "parts", parts)
    @property
    def requires(self): return frozenset().union(*(p.requires for p in self.parts))
    def init(self, rec): return tuple(p.init(rec) for p in self.parts)
    def update(self, st, rec, *, last): return tuple(p.update(a, rec, last=last) for p, a in zip(self.parts, st))
    def log_tilt(self, st): return sum(p.log_tilt(a) for p, a in zip(self.parts, st))
    def increment(self, prev, st):
        return sum(p.increment(a, b) for p, a, b in zip(self.parts, prev, st))

@dataclass(frozen=True)
class Zero:
    requires = frozenset()
    def init(self, rec): return xp.zeros(rec.t.shape, dtype=rec.t.dtype, device=device(rec.t))
    def update(self, st, rec, *, last): return st
    def log_tilt(self, st): return st
    def increment(self, prev, st): return st

@dataclass(frozen=True)
class ReverseRatio:                   # scores the reverse step with the kernels opposite to the run
    target_rev: Kernel; proposal_rev: Kernel
    @property
    def trivial(self): return self.target_rev is self.proposal_rev     # static: term is identically 0
    @property
    def requires(self):
        return frozenset() if self.trivial else self.target_rev.requires | self.proposal_rev.requires
    def init(self, rec):
        z = xp.zeros(rec.t.shape, dtype=rec.t.dtype, device=device(rec.t))
        return RRState(sum=z, term=z, x=rec.x, t=rec.t)
    def update(self, st, rec, *, last):
        if self.trivial:
            return st
        term = self.proposal_rev.at(rec, st.t).log_ratio(self.target_rev.at(rec, st.t), st.x)   # §7
        return RRState(sum=st.sum + term, term=term, x=rec.x, t=rec.t)
    def log_tilt(self, st): return st.sum
    def increment(self, prev, st): return st.term
```

## 10. Resampling and policies

Everything here is branch-free and per instance.

```python
def ess(log_w):                                   # [I, N] -> [I]
    lw = log_w - logsumexp(log_w, axis=1, keepdims=True)
    return xp.exp(-logsumexp(2 * lw, axis=1))

def systematic(bk, log_w, key):                   # [I, N] -> idx [I, N]
    I, N = log_w.shape
    w = xp.exp(log_w - logsumexp(log_w, axis=1, keepdims=True))
    cdf = xp.cumulative_sum(w, axis=1)
    u = (bk.uniform(key, (I, 1), w.dtype, device=device(w)) + xp.arange(N, dtype=w.dtype, device=device(w))[None, :]) / N
    idx = xp.sum(xp.astype(cdf[:, None, :] < u[:, :, None], xp.int64), axis=2)   # O(N^2); avoids 2-D searchsorted
    return xp.minimum(idx, N - 1)

def masked_resample(bk, tree, log_w, do, key):    # do: [I] bool
    I, N = log_w.shape
    idx = xp.where(do[:, None], systematic(bk, log_w, key), xp.arange(N, dtype=xp.int64, device=device(log_w))[None, :])
    tree = tree_map(lambda a: take_rows(a, idx), tree)
    return tree, xp.where(do[:, None], xp.zeros_like(log_w), log_w)

@dataclass(frozen=True)
class Never:                                      # k: step index, possibly traced
    def should_resample(self, log_w, k): return xp.zeros(log_w.shape[0], dtype=xp.bool, device=device(log_w))

@dataclass(frozen=True)
class Every:
    n: int
    def should_resample(self, log_w, k):
        return xp.broadcast_to(xp.asarray(k % self.n == 0, device=device(log_w)), (log_w.shape[0],))

@dataclass(frozen=True)
class ESSBelow:
    frac: float
    def should_resample(self, log_w, k): return ess(log_w) < self.frac * log_w.shape[1]
```

**Ladders and resampling.** When axis 1 is a temperature ladder (RpCN), positions have different targets, so resampling along it is invalid. Use `Never()`.

## 11. Moves

A move is an MCMC kernel applied at a fixed time that leaves the current target invariant. It therefore never changes `log_w` (resample-move SMC). Moves are used by `smc` (RpCN's pCN; optional for FK) and by CREPE (Langevin local exploration).

```python
@pytree
class MoveContext:          # one pytree: swaps and accept/reject keep it consistent
    rec: Record             # current point (evaluated with this step's needs)
    st: State               # potential state including rec
    st_prev: State | None   # potential state before rec (None at the start point)
    ref: Gaussian | None    # the target kernel's distribution that rec was drawn from

class Move(Protocol):
    requires: frozenset[Feature]          # features the move reads from ctx.rec
    def __call__(self, ctx, *, ev, potential, last, count, key, bk) -> MoveContext: ...
    # ev(point) -> Record: supplied by the runner; it requests this step's needs, so accepted
    # candidates are complete records (I4). The move never knows what those needs are.
    # count: how many times this move has already fired in this run (an int scalar, traced
    # inside scan; the runner carries it). Moves use it for counters such as the replica-
    # exchange sweep, so EvenOdd parity alternates across calls instead of restarting, and
    # it does so even when move_at fires at a fixed stride (e.g. every 20th step).

def rescore(potential, st_prev, rec, last):      # potential state if `rec` replaced the current point
    return potential.init(rec) if st_prev is None else potential.update(st_prev, rec, last=last)

def tilt_delta(potential, st_prev, a, b):        # C(a) - C(b) for two states that extend the same history
    if st_prev is None:                          # static: at the start point, C is a single term
        return potential.log_tilt(a) - potential.log_tilt(b)
    return potential.increment(st_prev, a) - potential.increment(st_prev, b)   # shared prefix never formed
```

**MH acceptances use `tilt_delta`.** A candidate and the current point share `st_prev`, so only their increments are compared. For `FKDiff` this is $\lambda[\hat r(\text{cand}) - \hat r(\text{cur})]$.

**pCN, with optional replica exchange.** RpCN's move: `n_steps` pCN proposals, each followed by an exchange sweep.

```python
@dataclass(frozen=True)
class PCN:
    n_steps: int                         # pCN proposals per move call (M)
    step: Callable                       # rho in (0, 1) per row, e.g. ladder(step_sizes)
    pairs: PairSchedule | None = None    # replica exchange along axis 1 after each pCN step (§12)
    requires = frozenset()               # acceptance reads only the potential
    def __call__(self, ctx, *, ev, potential, last, count, key, bk):
        for m in range(self.n_steps):                                    # static
            k_p, k_a, k_x, key = bk.split(key, 4)
            x, ref = ctx.rec.x, ctx.ref
            rho = bcast(self.step(ctx.rec.t), x)
            xi = bk.normal(k_p, x.shape, x.dtype, device=device(x))
            x_new = ref.mean + xp.sqrt(1 - rho**2) * (x - ref.mean) + rho * bcast(xp.sqrt(ref.var), x) * xi
            cand = ev(Point(x_new, ctx.rec.t))                           # 1 evaluation per proposal
            cand_st = rescore(potential, ctx.st_prev, cand, last)
            log_a = tilt_delta(potential, ctx.st_prev, cand_st, ctx.st)
            acc = xp.log(bk.uniform(k_a, log_a.shape, log_a.dtype, device=device(log_a))) < log_a
            ctx = replace(ctx, rec=tree_where(acc, cand, ctx.rec), st=tree_where(acc, cand_st, ctx.st))
            if self.pairs is not None:                                   # static
                ctx = replica_exchange(ctx, self.pairs, StateSwap(potential),
                                       sweep=count * self.n_steps + m, key=k_x, bk=bk)  # pCN steps so far
        return ctx
```

The proposal $\hat x = \mu + \sqrt{1-\rho^2}\,(x - \mu) + \rho\,\tau\,\xi$ is the pCN proposal. It is reversible with respect to `ref` $= \mathcal N(\mu, \tau^2)$, so the acceptance needs only the cumulative tilt. For `FKDiff` that is $\beta_k[\hat r(\hat x) - \hat r(x)]$.

**Langevin.** CREPE's local exploration move. It is unadjusted (no Metropolis correction).

```python
@dataclass(frozen=True)
class LangevinMove:
    kernel: Kernel                       # e.g. Guided(Langevin(h), tilt(lam, 0.5))
    @property
    def requires(self): return self.kernel.requires
    def __call__(self, ctx, *, ev, potential, last, count, key, bk):
        rec = ev(Point(self.kernel.at(ctx.rec, ctx.rec.t).sample(key, bk), ctx.rec.t))
        return replace(ctx, rec=rec, st=rescore(potential, ctx.st_prev, rec, last))
```

## 12. Replica exchange

Replica exchange is split into **which pairs** (a schedule) and **how a pair proposes** (an exchange), with one driver. RpCN and CREPE both use it. The ladder is axis 1 of any pytree.

```python
class PairSchedule(Protocol):
    period: int                                              # rounds depend on `sweep` only via sweep % period
    def rounds(self, n: int, sweep: int) -> tuple[tuple[tuple[int, int], ...], ...]: ...
    # static; each round is a tuple of disjoint adjacent pairs (lo, lo + 1), proposed together

@dataclass(frozen=True)
class SequentialSweep:          # (0,1), then (1,2), ..., each its own round
    period = 1
    def rounds(self, n, sweep): return tuple(((k, k + 1),) for k in range(n - 1))

@dataclass(frozen=True)
class EvenOdd:                  # deterministic even-odd (DEO): all pairs of one parity at once, alternating
    period = 2
    def rounds(self, n, sweep): return (tuple((m - 1, m) for m in range(1, n) if m % 2 == sweep % 2),)

class Exchange(Protocol):
    def propose(self, ladder, lo, hi, key, bk) -> tuple[Tree, Tree, Array]: ...
    # candidates for positions lo and hi (pytrees [I, P, ...]) and log-acceptance [I, P]

def replica_exchange(ladder, pairs: PairSchedule, exchange: Exchange, *, sweep, key, bk):
    # sweep: Python int (static) or traced int scalar (e.g. from the runner's step index)
    n = axis1_size(ladder)
    def for_residue(res):                                                # res: static int in range(period)
        def run(ladder):
            rounds = [r for r in pairs.rounds(n, res) if r]
            for pr, kk in zip(rounds, bk.split(key, max(len(rounds), 1))):
                lo, hi = [a for a, _ in pr], [b for _, b in pr]
                k_p, k_a = bk.split(kk, 2)
                new_lo, new_hi, log_a = exchange.propose(ladder, lo, hi, k_p, bk)
                u = bk.uniform(k_a, log_a.shape, log_a.dtype, device=device(log_a))
                acc = xp.log(u) < log_a                                  # [I, P]
                ladder = put_levels(ladder, lo, tree_where(acc, new_lo, take_levels(ladder, lo)))
                ladder = put_levels(ladder, hi, tree_where(acc, new_hi, take_levels(ladder, hi)))
            return ladder
        return run
    if isinstance(sweep, int):                                           # static: no dispatch
        return for_residue(sweep % pairs.period)(ladder)
    return bk.switch(sweep % pairs.period, [for_residue(r) for r in range(pairs.period)], ladder)

@dataclass(frozen=True)
class StateSwap:                # plain parallel tempering: exchange states; 0 evaluations
    potential: Potential
    def propose(self, ctx, lo, hi, key, bk):
        perm = list(range(axis1_size(ctx)))
        for a, b in zip(lo, hi): perm[a], perm[b] = b, a
        swapped = take_levels(ctx, perm)                           # rec, st, st_prev, ref move together
        d = self.potential.log_tilt(swapped.st) - self.potential.log_tilt(ctx.st)   # [I, n]
        return take_levels(swapped, lo), take_levels(swapped, hi), take_levels(d, lo) + take_levels(d, hi)
```

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
    final: Record          # [I, N, ...], carries potential.requires | move.requires | final_needs
    log_w: Array           # [I, N] unnormalised

def smc(start, u_grid, *, bk, evaluate, schedule, target, proposal, potential, policy,
        move=None, move_at=lambda k: True, final_needs=frozenset(), key) -> SMCResult:
    """start: Gaussian (sampled, and the first move's reference), Point, or Record; leading axes [I, N].
    u_grid: [S] (shared) or [I, N, S] (per row; increasing or decreasing -- the runner doesn't care)."""
    x_like = start.mean if isinstance(start, Gaussian) else start.x
    dtype, dev = x_like.dtype, device(x_like)                       # §4.1: run dtype and device
    t_grid = schedule.time(xp.asarray(u_grid, dtype=dtype, device=dev))   # I5: the only schedule call
    t_steps = steps_first(t_grid, lead_shape(start))                 # -> [S, I, N]
    S = t_steps.shape[0]
    mv = move.requires if move is not None else frozenset()
    need_mid = target.requires | proposal.requires | potential.requires | mv     # I4
    need_end = potential.requires | mv | frozenset(final_needs)
    ev_mid = lambda p: evaluate_rows(evaluate, p, need_mid, bk)
    ev_end = lambda p: evaluate_rows(evaluate, p, need_end, bk)
    k_start, k_mv0, k_steps = bk.split(key, 3)

    def apply_move(ctx, count, k, ev, last, key):                    # -> (ctx, count); no-op without a move
        if move is None:
            return ctx, count
        fire = lambda c: (move(c[0], ev=ev, potential=potential, last=last, count=c[1], key=key, bk=bk),
                          c[1] + 1)
        return bk.cond(move_at(k), fire, lambda c: c, (ctx, count))

    rec, ref = start_record(start, t_steps[0], need_mid, evaluate, bk, k_start)
    st = potential.init(rec)
    log_w = potential.log_tilt(st)
    count = xp.zeros((), dtype=xp.int32, device=dev)                 # moves fired so far in this run
    if ref is not None:                                              # static: a move at t_N needs a reference
        ctx, count = apply_move(MoveContext(rec, st, None, ref), count, 0, ev_mid, False, k_mv0)
        rec, st = ctx.rec, ctx.st

    def advance(carry, t_next, k, key, ev, last):
        rec, st, log_w, count = carry
        k_res, k_prop, k_mv = bk.split(key, 3)
        (rec, st), log_w = masked_resample(bk, (rec, st), log_w, policy.should_resample(log_w, k), k_res)
        ref = target.at(rec, t_next)                                  # also the pCN reference
        q = ref if proposal is target else proposal.at(rec, t_next)
        x = q.sample(k_prop, bk)
        if proposal is not target:                                    # static
            log_w = log_w + ref.log_ratio(q, x)                       # I9: never two log_probs
        new = ev(Point(x, t_next))                                    # I1
        st_new = potential.update(st, new, last=last)
        log_w = log_w + potential.increment(st, st_new)               # I9: never a C difference
        ctx, count = apply_move(MoveContext(new, st_new, st, ref), count, k, ev, last, k_mv)   # log_w unchanged
        return ctx.rec, ctx.st, log_w, count

    keys = bk.split(k_steps, S - 1)
    body = lambda c, inp: (advance(c, inp[0], inp[1], inp[2], ev_mid, False), None)
    steps = xp.arange(1, S - 1, dtype=xp.int32, device=dev)
    carry, _ = bk.scan(body, (rec, st, log_w, count), (t_steps[1:-1], steps, keys[:-1]))   # S == 2: empty scan
    rec, st, log_w, _ = advance(carry, t_steps[-1], S - 1, keys[-1], ev_end, True)
    return SMCResult(final=rec, log_w=log_w)
```

## 14. FK steering, DAS, and FK with pCN moves

```python
xp = bk.xp                                         # call-site code may use bk.xp to build inputs
sch = EDM()                                        # injected
u = xp.linspace(1.0, 0.0, 100, dtype=xp.float32)
P = PosteriorStep(sch)
start = sch.prior((I, N), D, xp=xp, dtype=xp.float32)   # D: event shape tuple, e.g. (4, 64, 64)

# FK steering: FKDiff; optionally Guided(P, tilt(lam)) as proposal.
# FKMax / FKSum plug in the same way, within their usage limit (§9).
fk = smc(start, u, bk=bk, evaluate=evaluate, schedule=sch, target=P, proposal=P,
         potential=FKDiff(constant(10.0)), policy=Every(5), final_needs={Feature.X0}, key=key)

# DAS
lam = das_lambda(sch.time(u), gamma=0.008, alpha_kl=0.01)
das = smc(start, u, bk=bk, evaluate=evaluate, schedule=sch, target=P, proposal=Guided(P, tilt(lam)),
          potential=FKDiff(lam), policy=ESSBelow(0.5), final_needs={Feature.X0}, key=key)

# FK + pCN moves (resample-move): valid for any FK potential, because the move targets
# the conditional of the FK target given the rest of the path, via the cumulative log_tilt
fk_pcn = smc(start, u, bk=bk, evaluate=evaluate, schedule=sch, target=P, proposal=P,
             potential=FKDiff(constant(10.0)), policy=Every(5),
             move=PCN(n_steps=2, step=constant(0.3)), move_at=lambda k: k % 5 == 0,
             final_needs={Feature.X0}, key=key)
```

## 15. RpCN

The particle axis is the temperature ladder ($N = K + 1$). The potential is `FKDiff` with $\beta$ by position, so `log_tilt` $= \beta_k\,\hat r_t(x)$, the tempered reward twist of rung $k$. There is no resampling.

```python
betas = xp.linspace(0.0, beta, K + 1, dtype=xp.float32)  # 0 = beta_0 < ... < beta_K = beta
P = DDIMStep(sch, eta=1.0)                                # = PosteriorStep (test K4); inverse problems: Guided(P, normalized(gamma))
out = smc(sch.prior((I, K + 1), D, xp=xp, dtype=xp.float32), u, bk=bk, evaluate=evaluate, schedule=sch,
          target=P, proposal=P, potential=FKDiff(ladder(betas)), policy=Never(),
          move=PCN(n_steps=M, step=ladder(step_sizes), pairs=SequentialSweep()),   # or EvenOdd()
          move_at=lambda k: k % 20 == 0, final_needs={Feature.X0}, key=key)
coldest = out.final.x[:, K]                               # the beta_K = beta replica at the last level; or .x0
```

- At the start point, `ref` is the prior; afterwards it is the target kernel's distribution for the current step.
- **Swap order:** `SequentialSweep()` or `EvenOdd()`. Both are valid.

## 16. CREPE

**Layout.**
- **Replicas** are one `Record` with axes `[I, M+1, ...]`: $I$ chains, levels $m = 0..M$ at $u_0 < \dots < u_M = 1$.
- **Replica invariant:** every replica always carries exactly `replica_needs`, and the replica at level $m$ always has $t = \tau(u_m)$.

```python
@dataclass(frozen=True)
class CREPEConfig:
    u_levels: tuple[float, ...]          # increasing; u_levels[-1] = 1
    K: int                               # fine steps per gap
    F_P: Kernel; B_P: Kernel; F_Q: Kernel; B_Q: Kernel
    lam: Callable                        # e.g. level_lambda(time(u_levels), beta, rho)
    pairs: PairSchedule = EvenOdd()
    local_move: Move | None = None       # e.g. LangevinMove(Guided(Langevin(h), tilt(lam, 0.5)))
    local_levels: tuple[int, ...] = ()   # static subset of 0..M-1
    out_needs: frozenset = frozenset({Feature.X0})
    @property
    def M(self): return len(self.u_levels) - 1
    @property
    def replica_needs(self):
        reqs = [k.requires for k in (self.F_P, self.B_P, self.F_Q, self.B_Q)]
        reqs += [self.local_move.requires] if self.local_move else []
        return frozenset().union(*reqs) | {Feature.REWARD} | self.out_needs
```

Default configuration for reward tilting:

```python
noise = Noising(sch)
lam = level_lambda(sch.time(xp.asarray(u_levels)), beta=10.0, rho=5.0)
cfg = CREPEConfig(u_levels=u_levels, K=5,
                  F_P=noise, F_Q=noise,                                    # same object -> F terms cancel statically
                  B_P=EulerMaruyama(ReverseSDE(sch), Direction.BACKWARD), B_Q=Guided(EulerMaruyama(ReverseSDE(sch), Direction.BACKWARD), tilt(lam)),
                  lam=lam, local_move=LangevinMove(Guided(Langevin(h=lambda t: 0.01 * sch.std(t) ** 2), tilt(lam, 0.5))),
                  local_levels=tuple(range(len(u_levels) - 1)))
```

**The path exchange.** Two one-directional `smc` calls per round.

```python
@dataclass(frozen=True)
class PathExchange:
    cfg: CREPEConfig; evaluate: Evaluate; schedule: Schedule
    def propose(self, R, lo, hi, key, bk):           # pair (lo[p], hi[p] = lo[p] + 1) is gap lo[p]
        c = self.cfg; k_up, k_dn = bk.split(key, 2)
        I, dtype, dev = R.t.shape[0], R.t.dtype, device(R.t)
        pot = lambda P_rev, Q_rev: Sum(TiltEnds(c.lam), ReverseRatio(P_rev, Q_rev))
        run = lambda start, grid, tgt, prop, p, k: smc(start, grid, bk=bk, evaluate=self.evaluate,
                  schedule=self.schedule, target=tgt, proposal=prop, potential=p, policy=Never(),
                  final_needs=c.replica_needs, key=k)
        grid = lambda asc: per_row_grid(c, lo, I, asc, xp=xp, dtype=dtype, device=dev)
        up = run(take_levels(R, lo), grid(True), c.F_P, c.F_Q, pot(c.B_P, c.B_Q), k_up)
        dn = run(take_levels(R, hi), grid(False), c.B_P, c.B_Q, pot(c.F_P, c.F_Q), k_dn)
        return dn.final, up.final, up.log_w + dn.log_w           # lower <- z_0, upper <- y_K

def fine_grid(cfg, m, *, xp, dtype, device=None):   # gap m: [K+1] ascending; endpoints exactly u_m, u_{m+1}
    u0, u1 = cfg.u_levels[m], cfg.u_levels[m + 1]
    pts = [u0, *(u0 + (u1 - u0) * j / cfg.K for j in range(1, cfg.K)), u1]
    return xp.asarray(pts, dtype=dtype, device=device)

def per_row_grid(cfg, gaps, I, ascending, *, xp, dtype, device=None):   # -> [I, P, K+1]
    g = xp.stack([fine_grid(cfg, m, xp=xp, dtype=dtype, device=device) for m in gaps])
    g = g if ascending else xp.flip(g, axis=-1)
    return xp.broadcast_to(g[None], (I, *g.shape))
```

**Build every CREPE grid with `fine_grid`, never `xp.linspace`.** This keeps every run starting and ending at exactly the level times (the replica invariant).

With $\log R^X(x_{0:K}) = \sum_k [\log B^X(x_{k-1} \mid x_k) - \log F^X(x_k \mid x_{k-1})]$, this is CREPE's swap acceptance:

$$
\log w_{\mathrm{up}} = \ell(y_K) - \ell(y_0) + \log R^Q(y) - \log R^P(y), \qquad
\log w_{\mathrm{dn}} = \ell(z_0) - \ell(z_K) + \log R^P(z) - \log R^Q(z).
$$

The driver applies $\min(0, \cdot)$ implicitly: it accepts iff $\log U < \log w_{\mathrm{up}} + \log w_{\mathrm{dn}}$.

**With the default proposals** ($F^Q = F^P$):

- **The $F$ terms cancel statically.** `ReverseRatio(F_P, F_Q)` is trivial, and
$$
\log A = \min\Big(0,\ \ell(z_0) - \ell(z_K) - \ell(y_0) + \ell(y_K) + \sum_{k=1}^{K} \big[\log B^P - \log B^Q\big](z_{k-1} \mid z_k) - \sum_{k=1}^{K} \big[\log B^P - \log B^Q\big](y_{k-1} \mid y_k)\Big).
$$
- **Sampling the up path needs no network.** Its $K$ evaluations score $y$ under the denoising kernels.
- **The cost is $2K$ evaluations per pair.**

**Initialisation:** $M$ chained `smc` calls with the model's own kernel.

```python
def top_replica(cfg, I, D, *, bk, evaluate, schedule, dtype, device, key):   # fresh level-M replica
    xp = bk.xp
    prior = schedule.prior((I, 1), D, xp=xp, dtype=dtype, device=device)
    t1 = schedule.time(xp.ones((I, 1), dtype=dtype, device=device))
    return start_record(prior, t1, cfg.replica_needs, evaluate, bk, key)[0]

def crepe_init(cfg, I, D, *, bk, evaluate, schedule, dtype, device=None, key):
    xp = bk.xp
    k0, k1, *ks = bk.split(key, cfg.M + 2)
    top = top_replica(cfg, I, D, bk=bk, evaluate=evaluate, schedule=schedule, dtype=dtype, device=device, key=k0)
    levels, cur = [None] * cfg.M + [top], top
    for m in range(cfg.M, 0, -1):                                    # static loop
        res = smc(cur, xp.flip(fine_grid(cfg, m - 1, xp=xp, dtype=dtype, device=device), axis=0),
                  bk=bk, evaluate=evaluate, schedule=schedule, target=cfg.B_P, proposal=cfg.B_P,
                  potential=Zero(), policy=Never(), final_needs=cfg.replica_needs, key=ks[m - 1])
        levels[m - 1] = cur = res.final
    return stack_levels(levels)                                      # [I, M+1, ...]
```

**The sweep loop.** The scan body covers `cfg.pairs.period` sweeps, so its static structure is fixed.

```python
def crepe(cfg, I, D, n_blocks, emit, *, bk, evaluate, schedule, dtype, device=None, key):
    xp = bk.xp
    k_init, k_run = bk.split(key, 2)
    R = crepe_init(cfg, I, D, bk=bk, evaluate=evaluate, schedule=schedule, dtype=dtype, device=device, key=k_init)
    ev = lambda p: evaluate_rows(evaluate, p, cfg.replica_needs, bk)
    xchg = PathExchange(cfg, evaluate, schedule)
    top_used = lambda n: any(cfg.M in p for rnd in cfg.pairs.rounds(cfg.M + 1, n) for p in rnd)   # static
    def body(R, key):
        outs = []
        for n, kk in zip(range(cfg.pairs.period), bk.split(key, cfg.pairs.period)):
            k_x, k_l, k_r = bk.split(kk, 3)
            R = replica_exchange(R, cfg.pairs, xchg, sweep=n, key=k_x, bk=bk)
            if cfg.local_move is not None:                           # static
                sub = take_levels(R, cfg.local_levels)
                ctx = cfg.local_move(MoveContext(sub, Zero().init(sub), None, None), ev=ev, potential=Zero(),
                                     last=False, count=n, key=k_l, bk=bk)
                R = put_levels(R, cfg.local_levels, ctx.rec)
            if top_used(n + 1):                                      # refresh only if used next sweep
                top = top_replica(cfg, I, D, bk=bk, evaluate=evaluate, schedule=schedule,
                                  dtype=dtype, device=device, key=k_r)
                R = put_levels(R, [cfg.M], top)
            outs.append(emit(take_levels(R, [0])))                   # e.g. lambda r: r.x0
        return R, xp.stack(outs)
    R, samples = bk.scan(body, R, bk.split(k_run, n_blocks))
    return samples, R                    # samples: [n_blocks, period, I, 1, ...]
```

**Refreshing the top replica** with a fresh prior draw is exact only if two things hold:
- `prior` is the exact marginal at $\tau(1)$;
- $\lambda(\tau(1)) = 0$. `level_lambda` guarantees this.

Otherwise the top level is approximately, not exactly, at its target. The approximation is the same as the one in the prior itself.

## 17. Exact evaluation counts

These counts are asserted by the tests.

| run | rows evaluated |
|---|---|
| `smc`, `Gaussian` or `Point` start, no move | $I N S$ |
| `smc`, `Record` start, no move | $I N (S - 1)$ |
| plus `PCN` at a moved step | $+\, I N \cdot$ `n_steps` |
| plus `LangevinMove` at a moved step | $+\, I N$ |
| RpCN | $I (K+1)\,\big(S + M \cdot \#\{\text{steered steps}\}\big)$ |
| `StateSwap` round | $0$ |
| `PathExchange` round | $2K$ per pair, times $I$ |
| `crepe_init` | $I(1 + M K)$ |
| CREPE sweep $n$ | $I\,(2K P_n + \lvert\text{local levels}\rvert + R_n)$ |

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
SCHEDULES = [VPLinear, EDM, RectifiedFlow]   # constructors take no arguments; defaults as in §5

def xps_for(bk):                       # Hypothesis may not know the newest standard revision yet:
    return make_strategies_namespace(bk.xp, api_version=max(hypothesis.extra.array_api.RELEASED_VERSIONS))
    # array generation uses nothing newer than 2023.12, so declaring an older version is safe
    # (testing/strategies = elems, u_pair, leading and xps_for, as defined in this file)

def elems(lo=-3.0, hi=3.0):
    return {"min_value": lo, "max_value": hi, "allow_nan": False,
            "allow_infinity": False, "allow_subnormal": False}

@st.composite
def u_pair(draw, gap=1e-2):           # u_lo < u_hi, both in [0, 1]
    a = draw(st.floats(0.0, 1.0 - gap)); return a, draw(st.floats(a + gap, 1.0))

@st.composite
def leading(draw, max_I=3, max_N=4):
    return draw(st.integers(1, max_I)), draw(st.integers(1, max_N))
```

**Pattern.** Parametrize over `bk` and `make_schedule` with `pytest.mark.parametrize`. Inside the test, build `xps = xps_for(bk)` and draw with `st.data()`. This avoids function-scoped fixtures, which trigger a Hypothesis health check.

**Backend notes:**
- **JAX:** `jax_enable_x64` is on in tests only, for the float64 oracle. With it on, JAX's defaults become float64, so every array a test builds names its dtype (§4.1 makes the library itself dtype-explicit).
- **PyTorch:** goes through `array_api_compat.torch`. Hypothesis may warn that it can't verify compliance; that is expected.
- **Fallback:** if array generation misbehaves on a backend, draw with `make_strategies_namespace(array_api_strict)` and convert with `bk.xp.from_dlpack`.

**Execution mode:** counting and duplicate tests must run eagerly (`jax.disable_jit()`). Under `jit`, Python side effects run once per trace.

**Float64 is an oracle, not a library mode.** Identity tests (K, P, M, X) run in float64, so their tolerances test the algebra. Precision tests (N) draw float32 inputs, run the library in float32, and compare against the same computation on the inputs **upcast** to float64. The comparison isolates the library's rounding from the rounding of the inputs. On JAX with `jax_enable_x64`, every float32 array in an N test is created with an explicit dtype.

```python
EPS32 = 2.0 ** -24                       # unit roundoff for float32

def oracle(f, *args):                    # run f on float64 copies of float32 inputs
    up = lambda a: xp.astype(a, xp.float64) if hasattr(a, "dtype") and a.dtype == xp.float32 else a
    return f(*tree_map(up, args))
```

### 18.2 Test doubles (`ttg/testing`)

```python
class CountingNetwork:                    # contract C1
    def __init__(self, net): self.net, self.calls = net, 0
    def __call__(self, x, t): self.calls += 1; return self.net(x, t)

class AuditingEvaluate:                   # invariant I1: counts rows, raises on a repeated (x, t)
    def __init__(self, evaluate): self.evaluate, self.rows, self.seen = evaluate, 0, set()
    def __call__(self, x, t, needs):
        for xi, ti in zip(to_host(x), to_host(t)):
            key = (xi.tobytes(), float(ti))
            if key in self.seen: raise DuplicateEvaluation(ti)
            self.seen.add(key)
        self.rows += t.shape[0]
        return self.evaluate(x, t, needs)

def to_host(a): return numpy.from_dlpack(a)    # eager tests only (torch: .cpu() first)

def index_of(t_grid, t):                  # eager tests only: the k with t_grid[k] == t (all rows agree)
    hits = [k for k in range(t_grid.shape[0]) if bool(xp.all(t_grid[k] == t))]
    assert len(hits) == 1; return hits[0]

@dataclass(frozen=True)                   # deliberately NOT a pytree: lives only in eager tests
class ReplayGaussian(Gaussian):
    forced: Array = None
    def sample(self, key, bk): return self.forced

@dataclass(frozen=True)
class ReplayKernel:                       # proposal double: follows a given path, densities from `base`
    base: Kernel; path: Array; t_grid: Array        # path [S, I, N, *D]; t_grid [S, I, N]
    @property
    def requires(self): return self.base.requires
    def at(self, rec, t):
        d = self.base.at(rec, t); k = index_of(self.t_grid, t)
        return ReplayGaussian(d.mean, d.var, self.path[k])           # sample() returns path[k]

def toy_evaluate(mu, sd, w, sch):         # data ~ N(mu, sd^2 I); reward r(x0) = <w, x0>; all exact
    # mu, w: [*D] arrays; sd: float. Supports every Feature on every backend (strict included).
    def evaluate(x, t, needs):
        m, s = bcast(sch.mean(t), x), bcast(sch.std(t), x)
        v = m**2 * sd**2 + s**2                          # marginal variance of x_t
        c = m * sd**2 / v                                # posterior-mean gain
        x0 = mu + c * (x - m * mu)                       # E[x0 | x_t]
        eps = (x - m * x0) / s
        score = -(x - m * mu) / v                        # = -eps / s exactly (Tweedie)
        ev = tuple(range(1, x.ndim))
        reward = xp.sum(w * x0, axis=ev)
        grad = c * w                                     # d<w, x0>/dx, broadcast per row
        vals = {X0: x0, EPS: eps, SCORE: score, REWARD: reward, REWARD_GRAD: grad}
        return Record(x, t, **{f.value: (vals[f] if f in needs else None) for f in Feature})
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
| S5 | `log_snr(t)` equals $2(\log m_t - \log s_t)$, and `log_snr_diff(s, t)` equals `log_snr(s) - log_snr(t)` (float64; covers the overrides) |

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
| K7 | `Guided(base, normalized(gamma))`'s mean shift has norm $\gamma(t)\,\lVert g \rVert / (\lVert g \rVert + 10^{-12})$ (so $\gamma(t)$ up to the guard), for $g \ne 0$ |
| K8 | `EulerMaruyama` has `var > 0` when used in its configured direction, and `var <= 0` when misused |
| K9 | `g2_coeff` equals $\beta(t)$ for `VPLinear`, and $2t$ for `EDM` |
| K11 | `ForwardSDE` requires nothing; `ReverseSDE(eta=1)` drift equals $f x - \gamma^2 \nabla \log p$; `diffusion2 > 0` for every provided SDE with $\eta > 0$ |
| K10 | **`tilt` is the exact Gaussian tilt:** $\log$ `Guided(base, tilt(lam))` $(x) - \log$ `base` $(x) - \lambda \langle g, x \rangle$ is the same for any two $x$ |
| K12 | `a.log_ratio(b, x)` equals `a.log_prob(x) - b.log_prob(x)` for any means and positive variances; with `a.var` and `b.var` equal, it equals the first term alone, exactly |
| K13 | `transition` and `PosteriorStep` match the $s_s^2 - m_{s\mid t}^2 s_t^2$ forms of §2 and §7 |

**Potentials** (pure, no model):

| # | property |
|---|---|
| P1 | telescoping: for any reward sequence, the final `log_tilt` equals $\lambda\,\hat r_{\text{last}}$ for `FKDiff`, `FKMax`, `FKSum`, and the summed increments equal the final `log_tilt` minus the initial one |
| P2 | `TiltEnds` gives $\ell(\text{end}) - \ell(\text{start})$; `Sum` is additive |
| P3 | `ReverseRatio(K, K)` contributes exactly 0 and requires nothing |
| P4 | $\lambda$-freedom: permuting states along a `ladder` axis and then reading `log_tilt` equals reading each moved state at its new position's $\lambda$ |
| P5 | `increment(prev, update(prev, rec, last))` equals `log_tilt(new) - log_tilt(prev)` for every potential, at every step including the last; `tilt_delta(p, prev, a, b)` equals `log_tilt(a) - log_tilt(b)` for two updates of the same `prev` |

**Precision** (float32 against the upcast float64 oracle). In every N test, draw scalars (λ included) from the **normal** float32 range: some backends, JAX on CPU among them, flush subnormal numbers to zero. Every bound also adds an **underflow floor**: (number of summed terms) × the smallest normal float32, scaled the same way as the bound (divided by $v$ in N2).

| # | property |
|---|---|
| N1 | **increments don't see $C$:** for drawn float32 states with accumulators up to $10^6$ and a non-final record, every FK potential's `increment` is within $4\epsilon\,\big(\lvert \text{inc} \rvert + \lvert\lambda\rvert(\lvert \hat r_{\text{prev}} \rvert + \lvert \hat r \rvert)\big)$ of the oracle. The bound contains no accumulator, so it fails if a $C$ difference ever creeps back in |
| N2 | **`log_ratio` with shared variance:** for $x$ = mean of `b` plus noise, `a.log_ratio(b, x)` is within $8\epsilon \sum_i \lvert \Delta_i \rvert \big(\lvert y_i \rvert + \lvert \Delta_i \rvert\big) / v$, plus the underflow floor, of the oracle. The bound is independent of $D \log v$ |
| N3 | **`log_snr_diff` overrides are relatively accurate:** for `VPLinear` and `EDM`, with gaps $s - t$ down to $10^{-6}$ of the range, the float32 result is within $16\epsilon\,\lvert \text{oracle} \rvert$; consequently, `transition` variances are $> 0$ on every such pair (I6) |
| N4 | **structural:** with a spy potential, `smc` calls `log_tilt` only at the start point. `PCN` calls it only when `st_prev is None`, and `StateSwap` (different histories, §12) always does. Everything else goes through `increment`. |

**Resampling:**

| # | property |
|---|---|
| R1 | systematic counts: each index count is within 1 of $N w_i$, per instance |
| R2 | `do = False` gives an identical tree and identical `log_w` |
| R3 | alignment: a tree with fields $a$ and $b = f(a)$ still satisfies $b = f(a)$ after resampling; `None` leaves stay `None` |
| R4 | `ess` lies in $[1, N]$; it equals $N$ for uniform weights and $1$ for a one-hot |

**Moves and replica exchange:**

| # | property |
|---|---|
| M1 | **pCN reversibility:** $\mathcal N(x; \mu, \tau^2)\, q(x \to \hat x) = \mathcal N(\hat x; \mu, \tau^2)\, q(\hat x \to x)$, with $q = \mathcal N(\mu + \sqrt{1-\rho^2}(x - \mu), \rho^2\tau^2)$, exactly (log space) |
| M2 | with $\lambda \equiv 0$, every pCN proposal is accepted |
| M3 | a move never changes `log_w` (runner, replayed path) |
| M4 | `StateSwap` log-acceptance equals $(\beta_{k+1} - \beta_k)(\hat r(x^{(k)}) - \hat r(x^{(k+1)}))$ (§12) for `FKDiff(ladder(betas))` |
| M5 | schedules: pairs within a round are disjoint and adjacent; over one `period`, every adjacent pair appears; `SequentialSweep` has $n-1$ single-pair rounds |
| M6 | with all $\beta_k$ equal, `SequentialSweep` swaps are always accepted, so the state at position 0 ends at position $n-1$ |
| M7 | a swap keeps each `MoveContext` row internally consistent (`rec`, `st`, `st_prev`, `ref` from the same source row) |

**Contract and runner** (eager):

| # | property |
|---|---|
| C1 | for every subset of features (`st.sets(st.sampled_from(Feature))`), one `evaluate` call costs $\le 1$ network call, and returns exactly the requested features |
| C2 | `start_record` raises on a partial `Record` |
| E1 | `AuditingEvaluate` row totals match §17 for drawn $(I, N, S)$, with and without moves; no duplicate raises |
| E2 | `proposal is target` and `Zero()` give `log_w == 0` exactly |
| E3 | with `ReplayKernel` and `Never()`, the runner's `log_w` equals a plain-loop reference along the replayed path: $C_{\text{final}} + \sum_k [\log P - \log Q](x_k \mid x_{k-1})$. The runner starts from `log_w` $= C_{\text{start}}$, the start point's own tilt (FK's $G_T$; `TiltEnds`'s $-\ell_{\text{start}}$), so $C_{\text{start}}$ is **not** subtracted |
| E4 | `Never()` keeps row order (final $x$ = replayed path end, per row) |
| E5 | `move_at(k) == False` for all $k$ gives results identical to `move=None` |

**CREPE:**

| # | property |
|---|---|
| X1 | with replayed $y$, $z$ paths: `PathExchange`'s log-acceptance equals the four-kernel formula, computed directly |
| X2 | with $F^Q = F^P$ and $B^Q = B^P$: it reduces to endpoint tilts; with $F^Q = F^P$ only: to the $B$-only formula of §16 |
| X3 | the replica invariant (fixed level times, exactly `replica_needs`) holds after any number of blocks |
| X4 | evaluation counts per sweep match §17 for both `EvenOdd` and `SequentialSweep` |

Three worked examples:

```python
@pytest.mark.parametrize("bk", BACKENDS, ids=str)
@pytest.mark.parametrize("make_schedule", SCHEDULES)
@given(data=st.data())
def test_K3_posterior_is_bayes_inverse_of_noising(bk, make_schedule, data):
    xp, xps, sch = bk.xp, xps_for(bk), make_schedule()
    B = data.draw(st.integers(1, 4))
    D = data.draw(xps.array_shapes(min_dims=1, max_dims=2, max_side=3))
    arr = lambda: data.draw(xps.arrays(xp.float64, (B, *D), elements=elems()))
    x0, x_t, x_s = arr(), arr(), arr()
    u_t, u_s = data.draw(u_pair())
    t, s = xp.full((B,), sch.time(u_t), dtype=xp.float64), xp.full((B,), sch.time(u_s), dtype=xp.float64)
    marg = lambda x, tt: Gaussian(bcast(sch.mean(tt), x0) * x0, sch.std(tt) ** 2).log_prob(x)
    lhs = Noising(sch).at(Point(x_t, t), s).log_prob(x_s) + marg(x_t, t) - marg(x_s, s)
    rhs = PosteriorStep(sch).at(Record(x_s, s, x0=x0), t).log_prob(x_t)
    assert xp.all(xp.abs(lhs - rhs) <= 1e-8 * (1 + xp.abs(rhs)))

@pytest.mark.parametrize("bk", BACKENDS, ids=str)
@pytest.mark.parametrize("Pot", [FKDiff, FKMax, FKSum])
@given(data=st.data(), c=st.floats(0.0, 5.0))
def test_P1_fk_potentials_telescope(bk, Pot, data, c):
    xp, xps = bk.xp, xps_for(bk)
    I, N = data.draw(leading()); S = data.draw(st.integers(2, 6))
    r = data.draw(xps.arrays(xp.float64, (S, I, N), elements=elems()))
    t = xp.linspace(1.0, 0.0, S, dtype=xp.float64)
    rec = lambda k: Record(xp.zeros((I, N, 1), dtype=xp.float64), xp.broadcast_to(t[k], (I, N)), reward=r[k])
    pot = Pot(constant(c))
    st_ = pot.init(rec(0))
    for k in range(1, S):
        st_ = pot.update(st_, rec(k), last=(k == S - 1))
    final = pot.log_tilt(st_)
    assert xp.all(xp.abs(final - c * r[S - 1]) <= 1e-9 * (1 + xp.abs(c * r[S - 1])))

@pytest.mark.parametrize("bk", BACKENDS, ids=str)
@given(data=st.data(), rho=st.floats(0.01, 0.99))
def test_M1_pcn_is_reversible_wrt_reference(bk, data, rho):
    xp, xps = bk.xp, xps_for(bk)
    B = data.draw(st.integers(1, 4)); D = data.draw(xps.array_shapes(min_dims=1, max_dims=2, max_side=3))
    arr = lambda: data.draw(xps.arrays(xp.float64, (B, *D), elements=elems()))
    mu, x, x_hat = arr(), arr(), arr()
    tau2 = data.draw(xps.arrays(xp.float64, (B,), elements=elems(0.01, 4.0)))
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
| DAS | `Guided(P, tilt(lam))` proposal, `FKDiff(lam)`, `ESSBelow(0.5)`, `lam = das_lambda(t_grid, gamma=0.008, alpha_kl=0.5)`; $N = 256$, $I = 16$ |
| FK + pCN | as FK with `FKDiff`, plus `PCN(n_steps=2, step=constant(0.3))` and `move_at=lambda k: k % 5 == 0` |
| RpCN | $K = 3$, `betas = linspace(0, 2, 4)`, `step_sizes = (0.95, 0.95, 0.95, 0.95)`, `PCN(n_steps=20, step=ladder(step_sizes), pairs=EvenOdd())` at **every** step, `Never()`; $I = 2^{12}$; the estimate uses rung $K$ only. RpCN is exact only as mixing becomes complete, so its bound adds $0.02$ for that bias: $4\sqrt{\mathrm{SE}^2 + \mathrm{SE}_{\text{ref}}^2} + 0.02$ |
| CREPE | `u_levels = (0.0, 0.25, 0.5, 0.75, 1.0)`, $K = 10$, `level_lambda(..., beta=2.0, rho=1.0)` (so $\lambda = 2$ at level 0 and $0$ at the top), the default kernels of §16, `Langevin(h=lambda t: 0.01 * sch.std(t)**2)` on levels 0–3, `EvenOdd()`; $I = 2^{10}$ chains, 400 blocks, the first 100 discarded, level-0 samples pooled. **Reference:** the plain sampler using `cfg.B_P` on CREPE's own grid (the `fine_grid`s of every gap concatenated, from level $M$ down to level 0), reweighted by $e^{\lambda_0 \hat r}$ at level 0, where $\lambda_0 = 2$ |
| pass condition | $\lvert \bar m - \bar m_{\text{ref}} \rvert \le 4\sqrt{\mathrm{SE}^2 + \mathrm{SE}_{\text{ref}}^2}$ for both coordinates, with each standard error taken from the spread across the 8 seeds |
| sanity check | for each tested $\lambda$, the same tolerance must **fail** for $\lambda = 0$ against that $\lambda$'s reference, so the test can detect a missing tilt |
