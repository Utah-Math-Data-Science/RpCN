import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")

with app.setup:
    from collections import defaultdict
    from functools import partial
    import itertools
    import math
    from pathlib import Path
    from types import SimpleNamespace

    import marimo as mo
    import duckdb
    import polars as pl
    import seaborn as sns
    import matplotlib.pyplot as plt
    import torch
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
    rng_seed = mo.ui.number(value=31, label='Num. GMM modes')
    n_gmm_modes = mo.ui.number(start=1, step=1, value=1, label='Num. GMM modes')
    x_spacing = mo.ui.number(start=0.0, step=0.5, value=3.0, label="x_spacing")
    n_grid = mo.ui.number(start=100, stop=4000, step=100, value=1200, label="plot_grid")

    tilt = mo.ui.number(start=0, stop=25, step=1, value=1, label=r'Reward tilt $\beta$')

    # Denoiser
    sm_n_training_samples = mo.ui.number(start=128, stop=32768, step=128, value=1024, label="sm_n_training_samples")
    # sm_epsilon_alpha = mo.ui.number(start=1e-6, stop=0.2, step=1e-4, value=1e-3, label="SM epsilon_alpha")
    # sm_epsilon_beta = mo.ui.number(start=1e-8, stop=0.2, step=1e-5, value=1e-5, label="SM epsilon_beta")
    # sm_vmap_chunk_size = mo.ui.number(start=8, stop=100_000, step=8, value=100_000, label="SM vmap_chunk_size")
    # sm_max_score_norm = mo.ui.number(start=1.0, stop=1000.0, step=1.0, value=100.0, label='SM max score norm')
    sm_n_sampling_steps = mo.ui.number(start=20, stop=1000, step=10, value=50, label="sm_n_sampling_steps")


    mo.vstack([
        rng_seed,
        mo.hstack([n_gmm_modes, x_spacing, tilt]),
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
        n_gmm_modes,
        n_grid,
        sm_n_sampling_steps,
        sm_n_training_samples,
        tilt,
        x_spacing,
    )


@app.cell
def _(n_gmm_modes, x_spacing):
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
        gmm_means, gmm_stds, gmm_weights = [torch.tensor(t, dtype=torch.float32) for t in (gmm_means, gmm_stds, gmm_weights)]
        mix = D.Categorical(gmm_weights)
        comp = D.Normal(gmm_means, gmm_stds)
        gmm = D.MixtureSameFamily(mix, comp)
        return gmm
    gmm = _build_gmm()
    gmm.component_distribution.loc
    return (gmm,)


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
        return gmm.log_prob(x_grid) + tilt.value * reward_fn(x_grid)
    def _log_normalizing_constant(x_grid):
        log_densities = tilted_log_prob_unnormalized(x_grid)
        max_log_density = log_densities.max()
        log_densities_shifted = log_densities - max_log_density
        log_normalizing_constant = (log_densities_shifted.exp().sum().log() + max_log_density) + (x_grid[1] - x_grid[0]).log()
        return log_normalizing_constant
    log_normalizing_constant = _log_normalizing_constant(x_grid)
    log_normalizing_constant.exp()
    return log_normalizing_constant, reward_fn, tilted_log_prob_unnormalized


@app.cell
def _(gmm, n_grid):
    _means = gmm.component_distribution.loc
    _stds = gmm.component_distribution.scale
    x_grid = torch.linspace(_means[0] - 3 * _stds[0], _means[-1] + 3 * _stds[-1], n_grid.value)
    return (x_grid,)


@app.cell
def _(gmm, log_normalizing_constant, tilted_log_prob_unnormalized, x_grid):
    def _gmm_plot(x_grid):
        base_log_density = gmm.log_prob(x_grid)
        df_base = pl.concat([
            pl.DataFrame(
                dict(x=x_grid, y=base_log_density.exp()),
            ).with_columns(src=pl.lit(r'$p(x)$')),
            pl.DataFrame(
                dict(x=x_grid, y=(tilted_log_prob_unnormalized(x_grid) - log_normalizing_constant).exp()),
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
def _(gmm, sm_n_training_samples):
    def _build_prob_path():
        # prob_path = Bao2024EnsembleScoreMatching(cfg)
        # prob_path = toy.prob_path.VarianceExploding(data_std=gmm.stddev, sigma_min=1e-3, sigma_max=10)
        prob_path = toy.prob_path.EDM(data_std=gmm.stddev, sigma_min=1e-3, sigma_max=10)
        return prob_path
    training_samples = gmm.sample((sm_n_training_samples.value,))[:, None]
    prob_path = _build_prob_path()
    denoiser = torch.vmap(partial(EnsembleDenoiser.denoise, training_samples), in_dims=(None, 0))
    return denoiser, prob_path, training_samples


@app.cell
def _(denoiser, prob_path, training_samples):
    samples = EnsembleDenoiser.sample(
        denoiser=denoiser,
        prob_path=prob_path,
        training_samples=training_samples,
        solver=Euler.step,
        count=100,
        i0=1,
        i1=0,
        steps=50,
    )
    _ax = (
        sns.histplot(
            bins=64,
            data=pl.DataFrame(dict(x=samples.squeeze())),
            stat='density',
            x='x',
        )
    )
    (
        sns.histplot(
            bins=64,
            data=pl.DataFrame(dict(x=training_samples.squeeze())),
            stat='density',
            x='x',
        )
    )
    _ax
    return


@app.cell
def _(
    denoiser,
    prob_path,
    reward_fn,
    sm_n_sampling_steps,
    tilt,
    training_samples,
):
    device, dtype = training_samples.device, training_samples.dtype
    indices = torch.linspace(1, 0, sm_n_sampling_steps.value, device=device, dtype=dtype)
    times = prob_path.t(indices)
    noise_levels = prob_path.noise_level(times)
    def run_fk_steering(batch_size, ensemble_size, pt_replicas=1):
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
        samples, log_weights = fk_steering.run(
            times,
            ensemble_size,
            batch_size=batch_size,
        )
        samples = samples[-1]
        log_weights = log_weights[-1]
        samples = spt_guidance.torch.fk_steering.SelectFn.weighted_sample(samples, log_weights)
        return samples
    samples_fk = run_fk_steering(1000, 4)
    _ax = (
        sns.histplot(
            bins=64,
            data=pl.DataFrame(dict(x=samples_fk.squeeze())),
            stat='density',
            x='x',
        )
    )
    _ax
    return device, dtype, times


@app.cell
def _(denoiser, device, dtype, prob_path, reward_fn, tilt, times):
    def run_rpcn(batch_size, pcn_steps, pt_replicas):
        inverse_temperatures = torch.linspace(0.0, tilt.value, pt_replicas, device=device, dtype=dtype)
        rotations = torch.linspace(math.pi / 2 - 1e-3, 0.05, pt_replicas, device=device, dtype=dtype)
        if pt_replicas == 1:
            inverse_temperatures += tilt.value
            rotations += 0.05
        kernel = spt_guidance.torch.rpcn.KernelPreconditionedCrankNicolson(
            1, device=device, dtype=dtype,
        )
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
        samples = rpcn.run(
            times,
            pcn_update_count=pcn_steps,
            batch_size=batch_size,
        )
        return samples[-1]
    samples_rpcn = run_rpcn(1000, 8, 4)
    _ax = (
        sns.histplot(
            bins=64,
            data=pl.DataFrame(dict(x=samples_rpcn.squeeze())),
            stat='density',
            x='x',
        )
    )
    _ax
    return


if __name__ == "__main__":
    app.run()
