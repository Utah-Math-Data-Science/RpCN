import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")

with app.setup:
    from collections import defaultdict
    from dataclasses import dataclass
    from functools import partial
    import math
    from typing import Any, Callable

    import marimo as mo
    import polars as pl
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    torch.use_deterministic_algorithms(True)
    import torch.distributions as D
    import ttg
    from ttg.backend import TorchBackend

    import toy.prob_path
    from toy.model import EnsembleDenoiser

    pt = 1/72.27
    page_width = 397.48499*pt
    golden_ratio = (1 + 5**0.5) / 2
    try:
        plt.style.use('style.mplstyle')
    except (RuntimeError, OSError):
        plt.rcParams.update({
            'mathtext.fontset': 'cm',
            'font.family': 'serif',
        })


@app.function
def tensordict_to(d, device='cpu'):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in d.items()}


@app.function
def make_seeds(rng_seed):
    # Independent torch seeds for each consumer, all derived from one user-facing seed.
    def np_to_torch_seed(child):
        return int(child.generate_state(1, dtype=np.uint64)[0] % (2**63 - 1))
    randbits = int('0x196719077368225158373397262628859969570', base=16)
    rng_seed_root = np.random.SeedSequence([randbits, rng_seed])
    names = ('training_samples', 'denoiser', 'fk', 'das', 'crepe', 'rpcn')
    return dict(zip(names, map(np_to_torch_seed, rng_seed_root.spawn(len(names)))))


@app.function
def build_gmm(n_modes, x_spacing, device='cuda'):
    default_means = {}
    default_stds = defaultdict(lambda: 4.0)
    default_weights = defaultdict(lambda: 1.0)
    if n_modes % 2 == 0:
        gmm_indices = list(range(1, (n_modes // 2) + 1))
        gmm_means = [default_means.get(i, i - 0.5) * 3 * x_spacing for i in gmm_indices]
    else:
        gmm_indices = list(range((n_modes + 1) // 2))
        gmm_means = [default_means.get(i, i) * 3 * x_spacing for i in gmm_indices]
    gmm_indices.extend([-i for i in gmm_indices])
    gmm_indices = sorted(set(gmm_indices))
    gmm_means.extend([-i for i in gmm_means])
    gmm_means = sorted(set(gmm_means))
    gmm_stds = [default_stds[i] for i in gmm_indices]
    gmm_weights = [default_weights[i] for i in gmm_indices]
    gmm_means, gmm_stds, gmm_weights = [torch.tensor(t, device=device, dtype=torch.float32) for t in (gmm_means, gmm_stds, gmm_weights)]
    mix = D.Categorical(gmm_weights)
    comp = D.Normal(gmm_means, gmm_stds)
    return D.MixtureSameFamily(mix, comp)


@app.function
def reward_fn(x):
    def reward_gauss(x, s=.5):
        return torch.exp(-(x / s).square())
    def gauss_low_high(x, m):
        return 1.6615 * reward_gauss(x - m) - 2 * reward_gauss(x - m + 1, s=1)
    r = gauss_low_high(x, 3) + gauss_low_high(-x, 3)
    return r.squeeze(-1)


@app.class_definition
@dataclass
class Problem:
    """The steering problem: GMM prior, reward tilt, ensemble denoiser, and the ttg pieces every method shares."""
    device: str
    tilt: float
    seeds: dict
    gmm: Any
    bins: np.ndarray
    bin_centers: torch.Tensor
    x_grid: torch.Tensor
    log_normalizing_constant: torch.Tensor
    training_samples: torch.Tensor
    prob_path: Any
    backend: Any
    dtype: torch.dtype
    event_shape: tuple
    schedule: Any
    reverse_step: Any
    evaluate: Callable

    def tilted_log_prob_unnormalized(self, x):
        return self.gmm.log_prob(x) + self.tilt * reward_fn(x)

    def tilted_log_prob(self, x):
        return self.tilted_log_prob_unnormalized(x) - self.log_normalizing_constant

    def u_grid(self, n_steps):
        # Sampling grid of FK, DAS and RpCN (CREPE uses its own levels).
        return self.backend.xp.linspace(1.0, 0.0, n_steps, dtype=self.dtype, device=self.device)

    def prior(self, batch_shape):
        return self.schedule.prior(batch_shape, self.event_shape, xp=self.backend.xp, dtype=self.dtype, device=self.device)

    def kl_divergence(self, samples, eps=1e-8):
        # KL(target || samples), both histogrammed on `bins`.
        target_density = np.asarray(self.tilted_log_prob(self.bin_centers).exp().cpu(), dtype=np.float64)
        ground_truth_hist = target_density * np.diff(self.bins)
        ground_truth_hist += eps
        ground_truth_hist /= ground_truth_hist.sum()
        steering_hist = np.histogram(samples, bins=self.bins)[0].astype(np.float64)
        steering_hist += eps
        steering_hist = steering_hist / steering_hist.sum()
        return (ground_truth_hist * (np.log(ground_truth_hist) - np.log(steering_hist))).sum()

    def get_metrics(self, samples, nfe):
        return dict(
            kl=float(self.kl_divergence(samples)),
            mean_reward=reward_fn(samples.to(self.device)).mean().item(),
            nfe=nfe,
        )

    @property
    def metrics_target(self):
        # Reference: E[R] under the tilted target, by quadrature on the plot grid.
        x_grid = self.x_grid
        return dict(
            kl=0.0,
            mean_reward=((self.tilted_log_prob(x_grid).exp() * reward_fn(x_grid[:, None])).sum() * (x_grid[1] - x_grid[0])).item(),
            nfe=None,
        )


@app.function
def build_problem(
    *,
    rng_seed=31,
    n_gmm_modes=1,
    x_spacing=3.0,
    tilt=1.0,
    n_grid=1200,
    n_training_samples=9984,
    eval_batch_size=9984,
    device='cuda',
):
    """Build the `Problem`. `eval_batch_size` sets the chunking of denoiser evaluations (memory, not results)."""
    seeds = make_seeds(rng_seed)
    gmm = build_gmm(n_gmm_modes, x_spacing, device)

    # Histogram bins (for KL) and the quadrature/plot grid.
    bin_range_low = (gmm.component_distribution.loc.min() - 5 * 3).item()
    bin_range_high = (gmm.component_distribution.loc.max() + 5 * 3).item()
    bins = np.histogram(np.zeros(2), bins=60, range=(bin_range_low, bin_range_high))[1]
    bin_centers = torch.as_tensor(bins[:-1] + np.diff(bins) / 2, device=device)
    x_grid = torch.linspace(bin_range_low, bin_range_high, n_grid, device=device)

    # Normalizing constant of the tilted target, by quadrature on x_grid.
    log_densities = gmm.log_prob(x_grid) + tilt * reward_fn(x_grid)
    max_log_density = log_densities.max()
    log_normalizing_constant = (
        (log_densities - max_log_density).exp().sum().log() + max_log_density + (x_grid[1] - x_grid[0]).log()
    )

    # Ensemble denoiser over training samples from the GMM.
    with torch.random.fork_rng(devices=[device]):
        torch.manual_seed(seed=seeds['training_samples'])
        training_samples = gmm.sample((n_training_samples,))[:, None]
    prob_path = toy.prob_path.EDM(data_std=gmm.stddev, sigma_min=1e-3, sigma_max=10)
    # Batched over (noise level, particle), the `network(x [B, *D], t [B])` shape ttg expects.
    denoiser = torch.vmap(
        partial(EnsembleDenoiser.denoise, training_samples, smoothing_std=0.0, generator=torch.Generator(device=device).manual_seed(seeds['denoiser'])),
        in_dims=(0, 0), randomness='different',
        chunk_size=eval_batch_size // 16,
    )

    backend = TorchBackend(device)
    # Same EDM path as `prob_path` (t = sigma, unit scale), so the denoiser's noise level is ttg's time.
    schedule = ttg.EDM(s_min=prob_path.sigma_min, s_max=prob_path.sigma_max, rho=prob_path.rho)
    evaluate_rows = ttg.make_evaluate_torch(
        lambda x, t: denoiser(t, x),  # the ensemble denoiser predicts x0
        lambda x, t, x0, s: (x0, ttg.eps_from_x0(x, t, x0, s)),
        schedule,
        ttg.on_x0(reward_fn),  # the builders call reward(x0, x, t); this reward reads x0 only
    )
    def evaluate(x, t, needs, row_chunk_size=eval_batch_size):
        # Rows are independent (evaluation contract C3), so chunking them bounds the memory autograd holds
        # for REWARD_GRAD across the ensemble of training samples.
        records = [
            evaluate_rows(x_chunk, t_chunk, needs)
            for x_chunk, t_chunk in zip(x.split(row_chunk_size), t.split(row_chunk_size))
        ]
        return ttg.ops.tree_map(lambda *chunks: torch.cat(chunks), *records)

    return Problem(
        device=device,
        tilt=tilt,
        seeds=seeds,
        gmm=gmm,
        bins=bins,
        bin_centers=bin_centers,
        x_grid=x_grid,
        log_normalizing_constant=log_normalizing_constant,
        training_samples=training_samples,
        prob_path=prob_path,
        backend=backend,
        dtype=training_samples.dtype,
        event_shape=tuple(training_samples.shape[1:]),
        schedule=schedule,
        # Reverse-SDE Euler-Maruyama step, as in `Euler.step` in toy.model.
        reverse_step=ttg.EulerMaruyama(ttg.ReverseSDE(schedule), ttg.Direction.BACKWARD),
        evaluate=evaluate,
    )


@app.function
def counting(evaluate, batch_size):
    # Wraps `evaluate` to count denoiser evaluations (rows); NFE per sample = rows / batch size.
    rows = [0]
    def counted(x, t, needs):
        rows[0] += x.shape[0]
        return evaluate(x, t, needs)
    return counted, lambda: rows[0] / batch_size


@app.function
def measure(problem, runner, batch_size, **kwargs):
    """Run `runner(problem, batch_size=..., evaluate=<counted>, **kwargs)`; return (samples on CPU, metrics)."""
    evaluate, nfe = counting(problem.evaluate, batch_size)
    samples = runner(problem, batch_size=batch_size, evaluate=evaluate, **kwargs).cpu()
    return samples, problem.get_metrics(samples, nfe())


@app.function
def fk_budget(nfe_budget, n_steps):
    # FK, DAS: N S  (one evaluation per particle per step). Returns (ensemble size N, expected NFE).
    ensemble_size = nfe_budget // n_steps
    return ensemble_size, ensemble_size * n_steps


@app.function
def rpcn_nfe(n_steps, n_replicas, pcn_steps):
    # RpCN: K S (1 + M)  (K rungs; each step evaluates once, then M pCN proposals; also at the start)
    return n_replicas * n_steps * (1 + pcn_steps)


@app.function
def rpcn_budget(nfe_budget, n_steps, n_replicas):
    # Returns (pCN steps M, expected NFE).
    pcn_steps = nfe_budget // (n_replicas * n_steps) - 1
    return pcn_steps, rpcn_nfe(n_steps, n_replicas, pcn_steps)


@app.function
def crepe_budget(nfe_budget, n_levels, fine_steps, thin=1, burn_in=0, n_tail=None):
    # CREPE:   1 + (L - 1) K_f          (prior draw at the top, carried down K_f fine steps per gap)
    #          + (b + n) W / 2 (2 K_f (L - 1) + 2 (L - 1) + 1)
    #                                   (b burn-in and n sampling blocks of W = lcm(2, thin) EvenOdd sweeps; per
    #                                    two sweeps, each of the L - 1 pairs exchanges once, up and down; a
    #                                    Langevin move on L - 1 levels per sweep; one top refresh)
    #          + n (W / thin) T         (with the finish, n_tail = T: T tail steps for each of the chain's n W / thin emits)
    # Returns (n blocks, kept samples per chain, expected NFE).
    L, K_f = n_levels, fine_steps
    W = math.lcm(2, int(thin))
    emits_per_block = W // int(thin)
    init = 1 + (L - 1) * K_f
    block = W // 2 * (2 * K_f * (L - 1) + 2 * (L - 1) + 1)
    finish_per_block = emits_per_block * n_tail if n_tail is not None else 0
    burn = int(burn_in) * block
    n_blocks = (nfe_budget - init - burn) // (block + finish_per_block)
    return n_blocks, n_blocks * emits_per_block, init + burn + n_blocks * (block + finish_per_block)


@app.function
def run_fk(problem, *, u_grid, batch_size, ensemble_size, evaluate=None, seed=None):
    p = problem
    seed = p.seeds['fk'] if seed is None else seed
    key_smc, key_select = p.backend.split(p.backend.key(seed), 2)
    fk = ttg.smc(
        p.prior((batch_size, ensemble_size)),
        u_grid,
        backend=p.backend,
        evaluate=p.evaluate if evaluate is None else evaluate,
        schedule=p.schedule,
        target=p.reverse_step,
        proposal=p.reverse_step,
        potential=ttg.FKDiff(ttg.constant(p.tilt)),
        policy=ttg.ESSBelow(0.5),
        key=key_smc,
    )
    # One particle per instance, drawn from the normalized final weights.
    selected = ttg.categorical(p.backend, fk.log_weights, key_select)
    return fk.final.x0[torch.arange(batch_size, device=p.device), selected]


@app.function
def run_das(problem, *, u_grid, batch_size, ensemble_size, gamma, evaluate=None, seed=None):
    p = problem
    seed = p.seeds['das'] if seed is None else seed
    key_smc, key_select = p.backend.split(p.backend.key(seed), 2)
    # Tempered tilt that reaches 1 / alpha_kl = beta at the last step.
    lam = ttg.das_lambda(p.schedule.time(u_grid), growth=gamma, alpha_kl=1 / p.tilt)
    das = ttg.smc(
        p.prior((batch_size, ensemble_size)),
        u_grid,
        backend=p.backend,
        evaluate=p.evaluate if evaluate is None else evaluate,
        schedule=p.schedule,
        target=p.reverse_step,
        proposal=ttg.Guided(p.reverse_step, ttg.tilt(lam)),
        potential=ttg.FKDiff(lam),
        policy=ttg.ESSBelow(0.5),
        sampler=ttg.ssp,
        key=key_smc,
    )
    selected = ttg.categorical(p.backend, das.log_weights, key_select)
    return das.final.x0[torch.arange(batch_size, device=p.device), selected]


@app.function
def run_crepe(problem, *, batch_size, n_levels, fine_steps, n_blocks, thin=1, burn_in=0, finish=None, evaluate=None, seed=None):
    # batch_size: number of chains, one instance each.
    # finish: None, or dict(u0=..., n_tail=..., exact=...) to start the levels at u0 > 0 and finish to u = 0
    p = problem
    seed = p.seeds['crepe'] if seed is None else seed
    u0 = 0.0 if finish is None else finish['u0']
    u_levels = tuple(u0 + (1.0 - u0) * m / (n_levels - 1) for m in range(n_levels))
    print(u_levels)
    lam = ttg.level_lambda(p.schedule.time(p.backend.xp.asarray(u_levels, dtype=p.dtype, device=p.device)), beta=p.tilt, rho=1.0)
    cfg = ttg.CREPEConfig(
        u_levels=u_levels,
        steps_per_gap=fine_steps,
        model_noising=ttg.Noising(p.schedule),
        proposal_noising=ttg.Noising(p.schedule),
        model_denoising=p.reverse_step,
        proposal_denoising=ttg.Guided(p.reverse_step, ttg.tilt(lam)),
        lam=lam,
        local_move=ttg.LangevinMove(ttg.Guided(ttg.Langevin(step_size=lambda t: 0.01 * p.schedule.std(t) ** 2), ttg.tilt(lam, 0.5))),
        local_levels=tuple(range(n_levels - 1)),
    )
    key_crepe, key_select = p.backend.split(p.backend.key(seed), 2)
    # One instance per chain, its particles the chain's post-burn-in samples at u = 0: finished from u0 (weighted
    # if exact, resampling only within a chain) when u0 > 0, otherwise CREPE's level-0 samples with log_weights = 0.
    out, _ = ttg.crepe_sample(
        cfg, batch_size, p.event_shape, n_blocks,
        n_tail=None if finish is None else finish['n_tail'],
        exact=None if finish is None else finish['exact'],
        backend=p.backend, evaluate=p.evaluate if evaluate is None else evaluate, schedule=p.schedule,
        dtype=p.dtype, device=p.device, key=key_crepe,
        thin=thin, burn_in=burn_in,
    )
    if finish is None:
        return out.final.x0[:, -1]  # plain CREPE: each chain's last kept sample, denoised
    # One particle per chain, drawn from its normalized weights.
    selected = ttg.categorical(p.backend, out.log_weights, key_select)
    return out.final.x0[torch.arange(batch_size, device=p.device), selected]


@app.function
def run_rpcn(problem, *, u_grid, batch_size, n_replicas, pcn_steps, evaluate=None, seed=None):
    p = problem
    seed = p.seeds['rpcn'] if seed is None else seed
    # Temperature ladder 0 = beta_0 < ... < beta_K = beta on the particle axis; no resampling.
    inverse_temperatures = p.tilt * torch.linspace(0.0, 1, n_replicas, device=p.device, dtype=p.dtype)**2
    if n_replicas == 1:
        inverse_temperatures += p.tilt
    step_sizes = torch.full_like(inverse_temperatures, .5)
    rpcn = ttg.smc(
        p.prior((batch_size, n_replicas)),
        u_grid,
        backend=p.backend,
        evaluate=p.evaluate if evaluate is None else evaluate,
        schedule=p.schedule,
        target=p.reverse_step,
        proposal=p.reverse_step,
        potential=ttg.FKDiff(ttg.ladder(inverse_temperatures)),
        policy=ttg.Never(),
        move=ttg.PCN(n_proposals=pcn_steps, step_size=ttg.ladder(step_sizes), pairs=ttg.SequentialSweep()),
        # move_at=lambda s: s >= 2,
        key=p.backend.key(seed),
    )
    return rpcn.final.x0[:, -1]


@app.cell
def _():
    device = 'cuda'
    rng_seed = mo.ui.number(value=31, label='RNG seed')
    n_gmm_modes = mo.ui.number(start=1, step=1, value=1, label='Num. GMM modes')
    x_spacing = mo.ui.number(start=0.0, step=0.5, value=3.0, label="x_spacing")
    n_grid = mo.ui.number(start=100, stop=4000, step=100, value=1200, label="plot_grid")

    tilt = mo.ui.number(start=0, stop=25, step=1, value=1, label=r'Reward tilt $\beta$')

    # Evaluation
    batch_size = mo.ui.number(start=128, stop=32768, step=128, value=9984, label='Batch size')

    # Denoiser
    sm_n_training_samples = mo.ui.number(start=128, stop=32768, step=128, value=9984, label="sm_n_training_samples")
    sm_n_sampling_steps = mo.ui.number(start=20, stop=1000, step=10, value=50, label="sm_n_sampling_steps")

    # Methods: every method gets the same number of denoiser evaluations (NFE) per sample
    nfe_budget = mo.ui.number(start=100, stop=100000, step=50, value=2000, label='NFE per sample')

    # DAS
    das_gamma = mo.ui.number(start=0.001, stop=1.0, step=0.001, value=0.008, label=r'$\gamma$')

    # RpCN
    rpcn_replicas = mo.ui.number(start=1, stop=32, step=1, value=4, label='Replicas')

    # CREPE
    crepe_n_levels = mo.ui.number(start=2, stop=32, step=1, value=5, label='Levels')
    crepe_fine_steps = mo.ui.number(start=1, stop=100, step=1, value=10, label='Fine steps per gap')
    crepe_thin = mo.ui.number(start=1, stop=100, step=1, value=1, label='Thin (sweeps per emit)')
    crepe_burn_in = mo.ui.number(start=0, stop=10000, step=1, value=0, label='Burn-in blocks')
    # u_0 > 0 puts level 0 above the data end, so its samples are finished down to u = 0 (crepe_sample does this)
    crepe_u0 = mo.ui.number(start=0.0, stop=0.9, step=0.01, value=0.0, label='Lowest level $u_0$ ($> 0$: finish)')
    crepe_n_tail = mo.ui.number(start=1, stop=200, step=1, value=10, label='Finish steps')
    crepe_finish_exact = mo.ui.checkbox(value=False, label='Exact (weighted) finish')

    mo.vstack([
        rng_seed,
        mo.hstack([n_gmm_modes, x_spacing, tilt]),
        mo.md('### Evaluation'),
        mo.hstack([batch_size]),
        mo.md('### Denoiser'),
        mo.hstack([sm_n_training_samples, sm_n_sampling_steps]),
        mo.md('### Methods'),
        mo.hstack([nfe_budget]),
        mo.md('#### DAS'),
        mo.hstack([das_gamma]),
        mo.md('#### RpCN'),
        mo.hstack([rpcn_replicas]),
        mo.md('#### CREPE'),
        mo.hstack([crepe_n_levels, crepe_fine_steps]),
        mo.hstack([crepe_thin, crepe_burn_in]),
        mo.hstack([crepe_u0, crepe_n_tail, crepe_finish_exact]),
    ])
    return (
        batch_size,
        crepe_burn_in,
        crepe_fine_steps,
        crepe_finish_exact,
        crepe_n_levels,
        crepe_n_tail,
        crepe_thin,
        crepe_u0,
        das_gamma,
        device,
        n_gmm_modes,
        n_grid,
        nfe_budget,
        rng_seed,
        rpcn_replicas,
        sm_n_sampling_steps,
        sm_n_training_samples,
        tilt,
        x_spacing,
    )


@app.cell
def _(
    batch_size,
    device,
    n_gmm_modes,
    n_grid,
    rng_seed,
    sm_n_training_samples,
    tilt,
    x_spacing,
):
    problem = build_problem(
        rng_seed=rng_seed.value,
        n_gmm_modes=n_gmm_modes.value,
        x_spacing=x_spacing.value,
        tilt=tilt.value,
        n_grid=n_grid.value,
        n_training_samples=sm_n_training_samples.value,
        eval_batch_size=batch_size.value,
        device=device,
    )
    return (problem,)


@app.cell
def _(problem, sm_n_sampling_steps):
    # The sampling grid of FK, DAS and RpCN, in its own cell so CREPE (which uses its own levels) doesn't depend on it.
    u_grid = problem.u_grid(sm_n_sampling_steps.value)
    return (u_grid,)


@app.cell
def _(nfe_budget, sm_n_sampling_steps):
    # Each method's settings are derived in its own cell from the shared NFE budget, so changing one method's
    # parameters reruns only that method.
    fk_ensemble_size, fk_expected_nfe = fk_budget(nfe_budget.value, sm_n_sampling_steps.value)
    mo.stop(fk_ensemble_size < 1, mo.md(f'**NFE budget {nfe_budget.value} is too small for FK/DAS.**'))
    mo.md(f'FK/DAS: {fk_ensemble_size} particles')
    return fk_ensemble_size, fk_expected_nfe


@app.cell
def _(nfe_budget, rpcn_replicas, sm_n_sampling_steps):
    rpcn_pcn_steps, rpcn_expected_nfe = rpcn_budget(nfe_budget.value, sm_n_sampling_steps.value, rpcn_replicas.value)
    mo.stop(rpcn_pcn_steps < 1, mo.md(f'**NFE budget {nfe_budget.value} is too small for RpCN with {rpcn_replicas.value} replicas.**'))
    mo.md(f'RpCN: {rpcn_pcn_steps} pCN steps')
    return rpcn_expected_nfe, rpcn_pcn_steps


@app.cell
def _(
    crepe_burn_in,
    crepe_fine_steps,
    crepe_n_levels,
    crepe_n_tail,
    crepe_thin,
    crepe_u0,
    nfe_budget,
):
    crepe_n_blocks, _emits_per_chain, crepe_expected_nfe = crepe_budget(
        nfe_budget.value, crepe_n_levels.value, crepe_fine_steps.value,
        thin=crepe_thin.value, burn_in=crepe_burn_in.value,
        n_tail=crepe_n_tail.value if crepe_u0.value > 0 else None,
    )
    mo.stop(crepe_n_blocks < 1, mo.md(f'**NFE budget {nfe_budget.value} is too small for these CREPE settings.**'))
    mo.md(f'CREPE: {crepe_n_blocks} blocks, {_emits_per_chain} kept samples per chain')
    return crepe_expected_nfe, crepe_n_blocks


@app.cell
def _(crepe_expected_nfe, fk_expected_nfe, nfe_budget, rpcn_expected_nfe):
    # Summary only; no method depends on this cell.
    _expected_nfe = {'FK': fk_expected_nfe, 'DAS': fk_expected_nfe, 'CREPE': crepe_expected_nfe, 'RpCN': rpcn_expected_nfe}
    mo.vstack([
        mo.md(f'Expected NFE per sample, from a budget of {nfe_budget.value}:'),
        pl.DataFrame(dict(method=list(_expected_nfe), expected_nfe=list(_expected_nfe.values()))),
    ])
    return


@app.cell
def _(batch_size, fk_ensemble_size, problem, u_grid):
    samples_fk, metrics_fk = measure(
        problem, run_fk, batch_size.value, u_grid=u_grid, ensemble_size=fk_ensemble_size,
    )
    metrics_fk
    return metrics_fk, samples_fk


@app.cell
def _(batch_size, das_gamma, fk_ensemble_size, problem, u_grid):
    samples_das, metrics_das = measure(
        problem, run_das, batch_size.value, u_grid=u_grid, ensemble_size=fk_ensemble_size, gamma=das_gamma.value,
    )
    metrics_das
    return metrics_das, samples_das


@app.cell
def _(
    batch_size,
    crepe_burn_in,
    crepe_fine_steps,
    crepe_finish_exact,
    crepe_n_blocks,
    crepe_n_levels,
    crepe_n_tail,
    crepe_thin,
    crepe_u0,
    problem,
):
    samples_crepe, metrics_crepe = measure(
        problem, run_crepe, batch_size.value,
        n_levels=crepe_n_levels.value, fine_steps=crepe_fine_steps.value, n_blocks=crepe_n_blocks,
        thin=int(crepe_thin.value), burn_in=int(crepe_burn_in.value),
        finish=(
            dict(u0=crepe_u0.value, n_tail=int(crepe_n_tail.value), exact=crepe_finish_exact.value)
            if crepe_u0.value > 0 else None
        ),
    )
    metrics_crepe
    return metrics_crepe, samples_crepe


@app.cell
def _(batch_size, problem, rpcn_pcn_steps, rpcn_replicas, u_grid):
    samples_rpcn, metrics_rpcn = measure(
        problem, run_rpcn, batch_size.value, u_grid=u_grid, n_replicas=rpcn_replicas.value, pcn_steps=rpcn_pcn_steps,
    )
    metrics_rpcn
    return metrics_rpcn, samples_rpcn


@app.cell
def _(
    metrics_crepe,
    metrics_das,
    metrics_fk,
    metrics_rpcn,
    problem,
    samples_crepe,
    samples_das,
    samples_fk,
    samples_rpcn,
):
    from toy.plots import save_all_subfigures
    from pathlib import Path
    SAVE_DIR = Path('~/Overleaf/RpCN/imgs/toy_example').expanduser()
    def _methods_plot():
        results = {
            'FK': (samples_fk, metrics_fk),
            'DAS': (samples_das, metrics_das),
            'CREPE': (samples_crepe, metrics_crepe),
            'RpCN': (samples_rpcn, metrics_rpcn),
        }
        df_summary = pl.DataFrame([
            dict(method=method, **metrics)
            for method, metrics in [*((m, mt) for m, (_, mt) in results.items()), ('Target', problem.metrics_target)]
        ])
        x_grid = problem.x_grid
        target_density = problem.tilted_log_prob(x_grid).exp()
        df_samples = pl.concat([
            pl.DataFrame(dict(x=samples.squeeze(-1).numpy())).with_columns(method=pl.lit(method))
            for method, (samples, _) in results.items()
        ])
        df_density = pl.DataFrame(tensordict_to(dict(x=x_grid, density=target_density)))
        df_density_untilted = pl.DataFrame(tensordict_to(dict(x=x_grid, density=problem.gmm.log_prob(x_grid).exp())))
        ax_width = page_width / 4
        aspect = 1
        g = (
            sns.FacetGrid(
                df_samples,
                col='method',
                col_order=list(results),
                height=ax_width / aspect,
                aspect=aspect,
                despine=False,
            )
            .set(
                xlim=(-13, 13),
                xticks=[-10, -3, 3, 10],  # the reward peaks at +/-3; no 0 tick, whose label would collide with them
                # ylim=(0, .2),
                # yscale='log',
            )
            .set_titles('')
        )
        g.map_dataframe(sns.histplot, x='x', bins=problem.bins, stat='density', element='step', lw=.5)
        for ax, (method, (samples, metrics)) in zip(g.axes.flat, results.items()):
            sns.lineplot(data=df_density, x='x', y='density', ax=ax, color='tab:orange', lw=1, zorder=-1)
            # Heights of the tallest histogram bar left and right of x = 0 (same bins and density normalization as the
            # histplot above). Each line runs from its side's axes edge to the outer edge of that peak's bar.
            heights = np.histogram(samples.squeeze(-1).numpy(), bins=problem.bins, density=True)[0]
            centers = (problem.bins[:-1] + problem.bins[1:]) / 2
            x_left, x_right = ax.get_xlim()
            for side in ('left', 'right'):
                idx = np.flatnonzero(centers < 0 if side == 'left' else centers > 0)
                i = idx[heights[idx].argmax()]
                peak = heights[i]
                x_edge, x_bar = (x_left, problem.bins[i]) if side == 'left' else (x_right, problem.bins[i + 1])
                ax.hlines(peak, min(x_edge, x_bar), max(x_edge, x_bar), color='tab:red', ls='--', lw=.75)
                ax.annotate(
                    f'{peak:.3f}', xy=(x_edge, peak),
                    xytext=(2 if side == 'left' else -2, -3), textcoords='offset points',  # gap below the line
                    ha=side, va='top', color='tab:red', fontsize='x-small',
                )
            # sns.lineplot(data=df_density_untilted, x='x', y='density', ax=ax, color='tab:red', lw=1, zorder=1)
            ax.set_xlabel('')
            xlabel = rf'{method} (KL = ${metrics["kl"]:.3e}$, $\bar{{R}} = {metrics["mean_reward"]:.3e}$)'
            print(xlabel)
            # ax.set_xlabel(xlabel)
        g.set(ylabel='')
        return df_summary, g
    _df_summary, _g = _methods_plot()
    save_all_subfigures(_g, SAVE_DIR/'NiceHistogram')
    mo.vstack([_df_summary, _g.figure])
    return


if __name__ == "__main__":
    app.run()
