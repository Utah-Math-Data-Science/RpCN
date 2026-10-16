import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")

with app.setup:
    from collections import defaultdict
    from functools import partial
    import itertools
    import math
    import secrets
    from pathlib import Path
    from types import SimpleNamespace

    import marimo as mo
    import duckdb
    import polars as pl
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    torch.use_deterministic_algorithms(True) 
    import torch.distributions as D
    import spt_guidance.torch.fk_steering
    import spt_guidance.torch.rpcn

    import toy.prob_path
    from toy.model import EnsembleDenoiser, Euler

    pt = 1/72.27
    page_width = 397.48499*pt
    golden_ratio = (1 + 5**0.5) / 2
    try:
        plt.style.use('style.mplstyle')
    except RuntimeError:
        plt.rcParams.update({
            'mathtext.fontset': 'cm',
            'font.family': 'serif',
        })


@app.cell
def _():
    device = 'cuda'
    rng_seed = mo.ui.number(value=31, label='Num. GMM modes')
    n_gmm_modes = mo.ui.number(start=1, step=1, value=1, label='Num. GMM modes')
    x_spacing = mo.ui.number(start=0.0, step=0.5, value=3.0, label="x_spacing")
    n_grid = mo.ui.number(start=100, stop=4000, step=100, value=1200, label="plot_grid")

    tilt = mo.ui.number(start=0, stop=25, step=1, value=1, label=r'Reward tilt $\beta$')

    # Evaluation
    batch_size = mo.ui.number(start=128, stop=32768, step=128, value=9984, label='Batch size')

    # Denoiser
    sm_n_training_samples = mo.ui.number(start=128, stop=32768, step=128, value=9984, label="sm_n_training_samples")
    sm_n_sampling_steps = mo.ui.number(start=20, stop=1000, step=10, value=50, label="sm_n_sampling_steps")


    mo.vstack([
        rng_seed,
        mo.hstack([n_gmm_modes, x_spacing, tilt]),
        mo.md('### Evaluation'),
        mo.hstack([batch_size]),
        mo.md('### Denoiser'),
        mo.hstack([
            sm_n_training_samples,
            # sm_epsilon_alpha,
            # sm_epsilon_beta,
        ]),
        mo.hstack([
            # sm_vmap_chunk_size,
            # sm_max_score_norm,
            sm_n_sampling_steps,
        ])
    ])
    return (
        batch_size,
        device,
        n_gmm_modes,
        n_grid,
        rng_seed,
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
        rng_seed_base,
        rng_seed_fk_steering,
        rng_seed_rpcn,
    ) = map(np_to_torch_seed, rng_seed_root.spawn(5))
    return (
        rng_seed_base,
        rng_seed_denoiser,
        rng_seed_fk_steering,
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
        gmm = D.MixtureSameFamily(mix, comp)
        return gmm
    gmm = _build_gmm()
    _bin_range_low = gmm.component_distribution.loc.min() - 5 * gmm.component_distribution.scale.max()
    _bin_range_high = gmm.component_distribution.loc.max() + 5 * gmm.component_distribution.scale.max()
    bins = np.histogram(np.zeros(2), bins=60, range=(_bin_range_low.item(), _bin_range_high.item()))[1]
    bin_centers = torch.as_tensor(bins[:-1] + np.diff(bins) / 2, device=device)
    x_grid = torch.linspace(_bin_range_low, _bin_range_high, n_grid.value, device=device)
    gmm.component_distribution.loc
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
        log_normalizing_constant = (log_densities_shifted.exp().sum().log() + max_log_density) + (x_grid[1] - x_grid[0]).log()
        return log_normalizing_constant
    def tilted_log_prob(x):
        return tilted_log_prob_unnormalized(x) - log_normalizing_constant
    log_normalizing_constant = _log_normalizing_constant(x_grid)
    log_normalizing_constant.exp()
    return reward_fn, tilted_log_prob


@app.function
def tensordict_to(d, device='cpu'):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in d.items()}


@app.cell
def _(gmm, tilted_log_prob, x_grid):
    def _gmm_plot(x_grid):
        base_log_density = gmm.log_prob(x_grid)
        df_base = pl.concat([
            pl.DataFrame(
                tensordict_to(dict(x=x_grid, y=base_log_density.exp())),
            ).with_columns(src=pl.lit(r'$p(x)$')),
            pl.DataFrame(
                tensordict_to(dict(x=x_grid, y=tilted_log_prob(x_grid).exp())),
            ).with_columns(src=pl.lit(r'$p(x)\exp(\beta R(x))/Z$')),
        ])
        ax_width = page_width#/ len(df_base['src'].unique())
        ax_height = ax_width / golden_ratio
        g = (
            sns.relplot(
                kind='line',
                data=df_base,
                x='x', y='y',
                hue='src',
                col='src',
                # color='tab:red',
                linewidth=1.5,
                facet_kws=dict(
                    # sharey=False,
                    despine=False,
                ),
                height=ax_height,
                aspect=golden_ratio,
            )
            .set_titles('')
            .set(ylabel='')
        )
        # fig_width = page_width * len(df_base['src'].unique())
        # fig_height = fig_width / golden_ratio / 1.0
        # g.fig.set_figwidth(fig_width)
        # g.fig.set_figheight(fig_height)
        sns.move_legend(
            g,
            loc='upper center',
            ncol=len(df_base['src'].unique()),
            # bbox_to_anchor=(.49, 1.13),
            title='',
        )
        return g
    _g = _gmm_plot(x_grid)
    _g
    return


@app.cell
def _(
    device,
    gmm,
    rng_seed_denoiser,
    rng_seed_training_samples,
    sm_n_training_samples,
):
    def _build_prob_path():
        # prob_path = Bao2024EnsembleScoreMatching(cfg)
        # prob_path = toy.prob_path.VarianceExploding(data_std=gmm.stddev, sigma_min=1e-3, sigma_max=10)
        prob_path = toy.prob_path.EDM(data_std=gmm.stddev, sigma_min=1e-3, sigma_max=10)
        return prob_path
    def get_denoiser(smoothing_std=0.16, vmap_chunk_size=None):
        denoiser = torch.vmap(
            partial(EnsembleDenoiser.denoise, training_samples, smoothing_std=smoothing_std, generator=torch.Generator(device=device).manual_seed(rng_seed_denoiser)),
            in_dims=(None, 0), randomness='different',
            chunk_size=vmap_chunk_size,
        )
        return denoiser
    with torch.random.fork_rng(devices=[device]):
        torch.manual_seed(seed=rng_seed_training_samples)
        training_samples = gmm.sample((sm_n_training_samples.value,))[:, None]
    prob_path = _build_prob_path()
    return get_denoiser, prob_path, training_samples


@app.cell
def _(bin_centers, bins, tilted_log_prob):
    def get_kl_divergence(samples, training_samples=None, density_func=lambda x: tilted_log_prob(x).exp().cpu(), eps=1e-8):
        if training_samples is not None:
            ground_truth_hist = np.histogram(training_samples, bins=bins)[0].astype(np.float64)
            ground_truth_hist += eps
            ground_truth_hist = ground_truth_hist / ground_truth_hist.sum()
        elif density_func is not None:
            target_density = np.asarray(density_func(bin_centers), dtype=np.float64)
            ground_truth_hist = target_density * np.diff(bins)
            ground_truth_hist += eps
            ground_truth_hist /= ground_truth_hist.sum()
        else:
            raise ValueError('Please provide training_samples or density_func')
        steering_hist = np.histogram(samples, bins=bins)[0].astype(np.float64)
        steering_hist += eps
        steering_hist = steering_hist / steering_hist.sum()
        kl_divergence = (ground_truth_hist * (np.log(ground_truth_hist) - np.log(steering_hist))).sum()
        return kl_divergence

    return (get_kl_divergence,)


@app.cell
def _(
    batch_size,
    bins,
    device,
    get_denoiser,
    get_kl_divergence,
    gmm,
    prob_path,
    rng_seed_base,
    training_samples,
):
    # kl_divergences = []
    # m = 40
    # smoothing_stds = [i / 100 for i in range(m + 1)]
    # for i, smoothing_std in enumerate(smoothing_stds):
    #     print(i)
    samples_base = EnsembleDenoiser.sample(
        denoiser=get_denoiser(),
        prob_path=prob_path,
        training_samples=training_samples,
        solver=Euler.step,
        count=batch_size.value,
        i0=1,
        i1=0,
        steps=50,
        generator=torch.Generator(device=device).manual_seed(rng_seed_base),
    ).cpu()
    #     kl_divergence = get_kl_divergence(**tensordict_to(dict(samples=samples_base)), density_func=lambda x: gmm.log_prob(x).exp().cpu())
    #     kl_divergences.append(kl_divergence)
    # df = pl.DataFrame(dict(smoothing_std=smoothing_stds, KL=kl_divergences))
    # print(df.sort('KL'))
    # sns.lineplot(data=df, x='smoothing_std', y='KL')
    _ax = (
        sns.histplot(
            bins=bins,
            data=pl.DataFrame(tensordict_to(dict(x=training_samples.squeeze()))),
            stat='density',
            x='x',
        )
    )
    _ax = (
        sns.histplot(
            bins=bins,
            data=pl.DataFrame(tensordict_to(dict(x=samples_base.squeeze()))),
            stat='density',
            x='x',
            ax=_ax,
        )
    )
    _ax.set_ylim(0, .2)
    print(get_kl_divergence(**tensordict_to(dict(samples=samples_base, training_samples=training_samples))))
    print(get_kl_divergence(**tensordict_to(dict(samples=samples_base)), density_func=lambda x: gmm.log_prob(x).exp().cpu()))
    _ax
    return


@app.cell
def _(device, prob_path, sm_n_sampling_steps, training_samples):
    dtype = training_samples.dtype
    indices = torch.linspace(1, 0, sm_n_sampling_steps.value, device=device, dtype=dtype)
    times = prob_path.t(indices)
    noise_levels = prob_path.noise_level(times)
    return dtype, noise_levels, times


@app.cell
def _(
    batch_size,
    bins,
    device,
    dtype,
    get_denoiser,
    get_kl_divergence,
    noise_levels,
    prob_path,
    reward_fn,
    rng_seed_fk_steering,
    tilt,
    tilted_log_prob,
    times,
    training_samples,
    x_grid,
):
    def run_fk_steering(batch_size, ensemble_size, pt_replicas=1, vmap_chunk_size=None):
        denoiser = get_denoiser(vmap_chunk_size=vmap_chunk_size)
        inverse_temperatures = torch.linspace(0.0, tilt.value, pt_replicas, device=device, dtype=dtype)
        if len(inverse_temperatures) == 1:
            inverse_temperatures += tilt.value
        fk_steering = spt_guidance.torch.fk_steering.FeynmanKac(
            init_param_fn=lambda: noise_levels[0] * torch.randn(training_samples.shape[1:], device=device, dtype=dtype),
            reward_fn=reward_fn,
            propose_fn=partial(EnsembleDenoiser.next_fn, denoiser, prob_path, noise_sampler=partial(torch.randn, device=device, dtype=dtype)),
            resample_fn=spt_guidance.torch.fk_steering.ResampleFn.adaptive_resample,
            potential=spt_guidance.torch.fk_steering.Potential.DIFFERENCE,
            intermediate_reward=spt_guidance.torch.fk_steering.IntermediateReward.EXPECTED_SAMPLE,
            inverse_temperature=inverse_temperatures,
            kwargs_intermediate_reward=dict(expected_sample_fn=partial(EnsembleDenoiser.tweedie, denoiser, prob_path)),
        )
        with torch.random.fork_rng(devices=[device]):
            torch.manual_seed(rng_seed_fk_steering)
            samples, log_weights = fk_steering.run(
                times,
                ensemble_size,
                batch_size=batch_size,
            )
            samples = samples[-1]
            log_weights = log_weights[-1]
            samples = spt_guidance.torch.fk_steering.SelectFn.weighted_sample(samples, log_weights)
        return samples
    samples_fk = run_fk_steering(batch_size.value, 20, vmap_chunk_size=batch_size.value // 16).cpu()
    _ax = (
        sns.histplot(
            bins=bins,
            data=pl.DataFrame(tensordict_to(dict(x=samples_fk.squeeze()))),
            stat='density',
            x='x',
        )
    )
    print(get_kl_divergence(**tensordict_to(dict(samples=samples_fk))))
    (
        sns.lineplot(
            data=pl.DataFrame(tensordict_to(dict(x=x_grid, density=tilted_log_prob(x_grid).exp()))),
            x='x', y='density',
            ax=_ax,
            color='tab:orange',
            lw=1.5,
        )
    )
    _ax
    return


@app.cell
def _(
    batch_size,
    bins,
    device,
    dtype,
    get_denoiser,
    get_kl_divergence,
    prob_path,
    reward_fn,
    rng_seed_rpcn,
    tilt,
    tilted_log_prob,
    times,
    x_grid,
):
    def run_rpcn(batch_size, pcn_steps, pt_replicas):
        inverse_temperatures = torch.linspace(0.0, tilt.value, pt_replicas, device=device, dtype=dtype)
        rotations = torch.linspace(math.pi / 2 - 1e-3, 0.05, pt_replicas, device=device, dtype=dtype)
        if pt_replicas == 1:
            inverse_temperatures += tilt.value
            rotations += 0.05
        kernel = spt_guidance.torch.rpcn.KernelPreconditionedCrankNicolson(
            1, device=device, dtype=dtype,
        )
        denoiser = get_denoiser()
        rpcn = spt_guidance.torch.rpcn.ResidualMCMCSteering(
            kernel,
            rotations.sin(),
            reward_fn=reward_fn,
            expected_next_fn=partial(EnsembleDenoiser.next_fn, denoiser, prob_path),
            std_fn=partial(EnsembleDenoiser.std_fn, prob_path),
            intermediate_reward=spt_guidance.torch.rpcn.IntermediateReward.EXPECTED_SAMPLE,
            inverse_temperature=inverse_temperatures,
            kwargs_intermediate_reward=dict(expected_sample_fn=partial(EnsembleDenoiser.tweedie, denoiser, prob_path)),
        )
        with torch.random.fork_rng(devices=[device]):
            torch.manual_seed(rng_seed_rpcn)
            samples = rpcn.run(
                times,
                pcn_update_count=pcn_steps,
                batch_size=batch_size,
            )
        return samples[-1]
    samples_rpcn = run_rpcn(batch_size.value, 8, 4)
    print(get_kl_divergence(**tensordict_to(dict(samples=samples_rpcn))))
    _ax = (
        sns.histplot(
            bins=bins,
            data=pl.DataFrame(tensordict_to(dict(x=samples_rpcn.squeeze()))),
            stat='density',
            x='x',
        )
    )
    (
        sns.lineplot(
            data=pl.DataFrame(tensordict_to(dict(x=x_grid, density=tilted_log_prob(x_grid).exp()))),
            x='x', y='density',
            ax=_ax,
            color='tab:orange',
            lw=1.5,
        )
    )
    _ax
    return


if __name__ == "__main__":
    app.run()
