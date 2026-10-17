import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")

with app.setup:
    from collections import defaultdict
    from functools import partial
    import math

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

    # Methods
    fk_ensemble_size = mo.ui.number(start=2, stop=256, step=1, value=20, label='FK/DAS particles')
    das_gamma = mo.ui.number(start=0.001, stop=1.0, step=0.001, value=0.008, label=r'DAS $\gamma$')
    rpcn_replicas = mo.ui.number(start=1, stop=32, step=1, value=4, label='RpCN replicas')
    rpcn_pcn_steps = mo.ui.number(start=1, stop=100, step=1, value=8, label='RpCN pCN steps')
    crepe_n_levels = mo.ui.number(start=2, stop=32, step=1, value=5, label='CREPE levels')
    crepe_fine_steps = mo.ui.number(start=1, stop=100, step=1, value=10, label='CREPE fine steps per gap')
    crepe_n_blocks = mo.ui.number(start=1, stop=200, step=1, value=10, label='CREPE blocks')

    mo.vstack([
        rng_seed,
        mo.hstack([n_gmm_modes, x_spacing, tilt]),
        mo.md('### Evaluation'),
        mo.hstack([batch_size]),
        mo.md('### Denoiser'),
        mo.hstack([sm_n_training_samples, sm_n_sampling_steps]),
        mo.md('### Methods'),
        mo.hstack([fk_ensemble_size, das_gamma]),
        mo.hstack([rpcn_replicas, rpcn_pcn_steps]),
        mo.hstack([crepe_n_levels, crepe_fine_steps, crepe_n_blocks]),
    ])
    return (
        batch_size,
        crepe_fine_steps,
        crepe_n_blocks,
        crepe_n_levels,
        das_gamma,
        device,
        fk_ensemble_size,
        n_gmm_modes,
        n_grid,
        rng_seed,
        rpcn_pcn_steps,
        rpcn_replicas,
        sm_n_sampling_steps,
        sm_n_training_samples,
        tilt,
        x_spacing,
    )


@app.cell
def _(rng_seed):
    def np_to_torch_seed(child):
        return int(child.generate_state(1, dtype=np.uint64)[0] % (2**63 - 1))
    randbits = int('0x196719077368225158373397262628859969570', base=16)
    rng_seed_root = np.random.SeedSequence([randbits, rng_seed.value])
    (
        rng_seed_training_samples,
        rng_seed_denoiser,
        rng_seed_fk,
        rng_seed_das,
        rng_seed_crepe,
        rng_seed_rpcn,
    ) = map(np_to_torch_seed, rng_seed_root.spawn(6))
    return (
        rng_seed_crepe,
        rng_seed_das,
        rng_seed_denoiser,
        rng_seed_fk,
        rng_seed_rpcn,
        rng_seed_training_samples,
    )


@app.cell
def _(device, n_gmm_modes, n_grid, x_spacing):
    def _build_gmm():
        default_means = {}
        default_stds = defaultdict(lambda: 3.0)
        default_weights = defaultdict(lambda: 1.0)
        if n_gmm_modes.value % 2 == 0:
            gmm_indices = list(range(1, (n_gmm_modes.value // 2) + 1))
            gmm_means = [default_means.get(i, i - 0.5) * 3 * x_spacing.value for i in gmm_indices]
        else:
            gmm_indices = list(range((n_gmm_modes.value + 1) // 2))
            gmm_means = [default_means.get(i, i) * 3 * x_spacing.value for i in gmm_indices]
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
    gmm = _build_gmm()
    _bin_range_low = gmm.component_distribution.loc.min() - 5 * gmm.component_distribution.scale.max()
    _bin_range_high = gmm.component_distribution.loc.max() + 5 * gmm.component_distribution.scale.max()
    bins = np.histogram(np.zeros(2), bins=60, range=(_bin_range_low.item(), _bin_range_high.item()))[1]
    bin_centers = torch.as_tensor(bins[:-1] + np.diff(bins) / 2, device=device)
    x_grid = torch.linspace(_bin_range_low, _bin_range_high, n_grid.value, device=device)
    return bin_centers, bins, gmm, x_grid


@app.cell
def _(gmm, tilt, x_grid):
    def _reward_gauss(x, s=.5):
        return torch.exp(-(x / s).square())
    def _gauss_low_high(x, m):
        return 1.6615 * _reward_gauss(x - m) - 2 * _reward_gauss(x - m + 1, s=1)
    def reward_fn(x):
        r = _gauss_low_high(x, 3) + _gauss_low_high(-x, 3)
        return r.squeeze(-1)
    def tilted_log_prob_unnormalized(x):
        return gmm.log_prob(x) + tilt.value * reward_fn(x)
    def _log_normalizing_constant(x_grid):
        log_densities = tilted_log_prob_unnormalized(x_grid)
        max_log_density = log_densities.max()
        log_densities_shifted = log_densities - max_log_density
        return (log_densities_shifted.exp().sum().log() + max_log_density) + (x_grid[1] - x_grid[0]).log()
    def tilted_log_prob(x):
        return tilted_log_prob_unnormalized(x) - log_normalizing_constant
    log_normalizing_constant = _log_normalizing_constant(x_grid)
    return reward_fn, tilted_log_prob


@app.cell
def _(bin_centers, bins, device, reward_fn, tilted_log_prob, x_grid):
    def get_kl_divergence(samples, density_func=lambda x: tilted_log_prob(x).exp().cpu(), eps=1e-8):
        target_density = np.asarray(density_func(bin_centers), dtype=np.float64)
        ground_truth_hist = target_density * np.diff(bins)
        ground_truth_hist += eps
        ground_truth_hist /= ground_truth_hist.sum()
        steering_hist = np.histogram(samples, bins=bins)[0].astype(np.float64)
        steering_hist += eps
        steering_hist = steering_hist / steering_hist.sum()
        return (ground_truth_hist * (np.log(ground_truth_hist) - np.log(steering_hist))).sum()
    def get_metrics(samples):
        return dict(
            kl=float(get_kl_divergence(samples)),
            mean_reward=reward_fn(samples.to(device)).mean().item(),
        )
    # Reference: E[R] under the tilted target, by quadrature on the plot grid.
    metrics_target = dict(
        kl=0.0,
        mean_reward=((tilted_log_prob(x_grid).exp() * reward_fn(x_grid[:, None])).sum() * (x_grid[1] - x_grid[0])).item(),
    )
    return get_metrics, metrics_target


@app.cell
def _(
    device,
    gmm,
    rng_seed_denoiser,
    rng_seed_training_samples,
    sm_n_training_samples,
):
    prob_path = toy.prob_path.EDM(data_std=gmm.stddev, sigma_min=1e-3, sigma_max=10)
    def get_denoiser(smoothing_std=0.16, vmap_chunk_size=None):
        # Batched over (noise level, particle), the `network(x [B, *D], t [B])` shape ttg expects.
        return torch.vmap(
            partial(EnsembleDenoiser.denoise, training_samples, smoothing_std=smoothing_std, generator=torch.Generator(device=device).manual_seed(rng_seed_denoiser)),
            in_dims=(0, 0), randomness='different',
            chunk_size=vmap_chunk_size,
        )
    with torch.random.fork_rng(devices=[device]):
        torch.manual_seed(seed=rng_seed_training_samples)
        training_samples = gmm.sample((sm_n_training_samples.value,))[:, None]
    return get_denoiser, prob_path, training_samples


@app.cell
def _(
    batch_size,
    device,
    get_denoiser,
    prob_path,
    reward_fn,
    sm_n_sampling_steps,
    training_samples,
):
    bk = TorchBackend(device)
    dtype = training_samples.dtype
    event_shape = tuple(training_samples.shape[1:])
    # Same EDM path as `prob_path` (t = sigma, unit scale), so the denoiser's noise level is ttg's time.
    sch = ttg.EDM(s_min=prob_path.sigma_min, s_max=prob_path.sigma_max, rho=prob_path.rho)
    _denoiser = get_denoiser(vmap_chunk_size=batch_size.value // 16)
    _evaluate_rows = ttg.make_evaluate_torch(
        lambda x, t: _denoiser(t, x),  # the ensemble denoiser predicts x0
        lambda x, t, x0, s: (x0, ttg.eps_from_x0(x, t, x0, s)),
        sch,
        reward_fn,
    )
    def evaluate(x, t, needs, row_chunk_size=batch_size.value):
        # Rows are independent (evaluation contract C3), so chunking them bounds the memory autograd holds
        # for REWARD_GRAD across the ensemble of training samples.
        records = [
            _evaluate_rows(x_chunk, t_chunk, needs)
            for x_chunk, t_chunk in zip(x.split(row_chunk_size), t.split(row_chunk_size))
        ]
        return ttg.ops.tree_map(lambda *chunks: torch.cat(chunks), *records)
    # Reverse-SDE Euler-Maruyama step, as in `Euler.step` in toy.model.
    reverse_step = ttg.EulerMaruyama(ttg.ReverseSDE(sch), ttg.Direction.BACKWARD)
    u_grid = bk.xp.linspace(1.0, 0.0, sm_n_sampling_steps.value, dtype=dtype, device=device)
    return bk, dtype, evaluate, event_shape, reverse_step, sch, u_grid


@app.cell
def _(
    batch_size,
    bk,
    device,
    dtype,
    evaluate,
    event_shape,
    fk_ensemble_size,
    get_metrics,
    reverse_step,
    rng_seed_fk,
    sch,
    tilt,
    u_grid,
):
    def run_fk(batch_size, ensemble_size):
        key_smc, key_select = bk.split(bk.key(rng_seed_fk), 2)
        fk = ttg.smc(
            sch.prior((batch_size, ensemble_size), event_shape, xp=bk.xp, dtype=dtype, device=device),
            u_grid,
            bk=bk,
            evaluate=evaluate,
            schedule=sch,
            target=reverse_step,
            proposal=reverse_step,
            potential=ttg.FKDiff(ttg.constant(tilt.value)),
            policy=ttg.ESSBelow(0.5),
            key=key_smc,
        )
        # One particle per instance, drawn from the normalized final weights.
        selected = ttg.categorical(bk, fk.log_w, key_select)
        return fk.final.x[torch.arange(batch_size, device=device), selected]
    samples_fk = run_fk(batch_size.value, fk_ensemble_size.value).cpu()
    metrics_fk = get_metrics(samples_fk)
    metrics_fk
    return metrics_fk, samples_fk


@app.cell
def _(
    batch_size,
    bk,
    das_gamma,
    device,
    dtype,
    evaluate,
    event_shape,
    fk_ensemble_size,
    get_metrics,
    reverse_step,
    rng_seed_das,
    sch,
    tilt,
    u_grid,
):
    def run_das(batch_size, ensemble_size):
        key_smc, key_select = bk.split(bk.key(rng_seed_das), 2)
        # Tempered tilt that reaches 1 / alpha_kl = beta at the last step.
        lam = ttg.das_lambda(sch.time(u_grid), gamma=das_gamma.value, alpha_kl=1 / tilt.value)
        das = ttg.smc(
            sch.prior((batch_size, ensemble_size), event_shape, xp=bk.xp, dtype=dtype, device=device),
            u_grid,
            bk=bk,
            evaluate=evaluate,
            schedule=sch,
            target=reverse_step,
            proposal=ttg.Guided(reverse_step, ttg.tilt(lam)),
            potential=ttg.FKDiff(lam),
            policy=ttg.ESSBelow(0.5),
            key=key_smc,
        )
        selected = ttg.categorical(bk, das.log_w, key_select)
        return das.final.x[torch.arange(batch_size, device=device), selected]
    samples_das = run_das(batch_size.value, fk_ensemble_size.value).cpu()
    metrics_das = get_metrics(samples_das)
    metrics_das
    return metrics_das, samples_das


@app.cell
def _(
    batch_size,
    bk,
    crepe_fine_steps,
    crepe_n_blocks,
    crepe_n_levels,
    device,
    dtype,
    evaluate,
    event_shape,
    get_metrics,
    reverse_step,
    rng_seed_crepe,
    sch,
    tilt,
):
    def run_crepe(n_chains, n_levels, fine_steps, n_blocks):
        u_levels = tuple(m / (n_levels - 1) for m in range(n_levels))
        lam = ttg.level_lambda(sch.time(bk.xp.asarray(u_levels, dtype=dtype, device=device)), beta=tilt.value, rho=1.0)
        cfg = ttg.CREPEConfig(
            u_levels=u_levels,
            K=fine_steps,
            F_P=ttg.Noising(sch),
            F_Q=ttg.Noising(sch),
            B_P=reverse_step,
            B_Q=ttg.Guided(reverse_step, ttg.tilt(lam)),
            lam=lam,
            local_move=ttg.LangevinMove(ttg.Guided(ttg.Langevin(h=lambda t: 0.01 * sch.std(t) ** 2), ttg.tilt(lam, 0.5))),
            local_levels=tuple(range(n_levels - 1)),
        )
        samples, _ = ttg.crepe(
            cfg, n_chains, event_shape, n_blocks, lambda r: r.x,
            bk=bk, evaluate=evaluate, schedule=sch, dtype=dtype, device=device, key=bk.key(rng_seed_crepe),
        )
        # [n_blocks, period, I, 1, *D]: the last sweep's level-0 replica of each independent chain.
        return samples[-1, -1, :, 0]
    samples_crepe = run_crepe(batch_size.value, crepe_n_levels.value, crepe_fine_steps.value, crepe_n_blocks.value).cpu()
    metrics_crepe = get_metrics(samples_crepe)
    metrics_crepe
    return metrics_crepe, samples_crepe


@app.cell
def _(
    batch_size,
    bk,
    device,
    dtype,
    evaluate,
    event_shape,
    get_metrics,
    reverse_step,
    rng_seed_rpcn,
    rpcn_pcn_steps,
    rpcn_replicas,
    sch,
    tilt,
    u_grid,
):
    def run_rpcn(batch_size, n_replicas, pcn_steps):
        # Temperature ladder 0 = beta_0 < ... < beta_K = beta on the particle axis; no resampling.
        inverse_temperatures = torch.linspace(0.0, tilt.value, n_replicas, device=device, dtype=dtype)
        rotations = torch.linspace(math.pi / 2 - 1e-3, 0.05, n_replicas, device=device, dtype=dtype)
        if n_replicas == 1:
            inverse_temperatures += tilt.value
            rotations += 0.05
        rpcn = ttg.smc(
            sch.prior((batch_size, n_replicas), event_shape, xp=bk.xp, dtype=dtype, device=device),
            u_grid,
            bk=bk,
            evaluate=evaluate,
            schedule=sch,
            target=reverse_step,
            proposal=reverse_step,
            potential=ttg.FKDiff(ttg.ladder(inverse_temperatures)),
            policy=ttg.Never(),
            move=ttg.PCN(n_steps=pcn_steps, step=ttg.ladder(rotations.sin()), pairs=ttg.SequentialSweep()),
            key=bk.key(rng_seed_rpcn),
        )
        return rpcn.final.x[:, -1]  # the beta_K = beta replica
    samples_rpcn = run_rpcn(batch_size.value, rpcn_replicas.value, rpcn_pcn_steps.value).cpu()
    metrics_rpcn = get_metrics(samples_rpcn)
    metrics_rpcn
    return metrics_rpcn, samples_rpcn


@app.cell
def _(
    bins,
    metrics_crepe,
    metrics_das,
    metrics_fk,
    metrics_rpcn,
    metrics_target,
    samples_crepe,
    samples_das,
    samples_fk,
    samples_rpcn,
    tilted_log_prob,
    x_grid,
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
            for method, metrics in [*((m, mt) for m, (_, mt) in results.items()), ('Target', metrics_target)]
        ])
        target_density = tilted_log_prob(x_grid).exp()
        df_samples = pl.concat([
            pl.DataFrame(dict(x=samples.squeeze(-1).numpy())).with_columns(method=pl.lit(method))
            for method, (samples, _) in results.items()
        ])
        df_density = pl.DataFrame(tensordict_to(dict(x=x_grid, density=target_density)))
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
            )
            .set_titles('')
        )
        g.map_dataframe(sns.histplot, x='x', bins=bins, stat='density', element='step', lw=.5)
        for ax, (method, (_, metrics)) in zip(g.axes.flat, results.items()):
            sns.lineplot(data=df_density, x='x', y='density', ax=ax, color='tab:orange', lw=1, zorder=-1)
            ax.set_xlabel('')
            xlabel = rf'{method} (KL = ${metrics["kl"]:.3f}$, $\bar{{R}} = {metrics["mean_reward"]:.3f}$)'
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
