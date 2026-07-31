import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full")


@app.cell
def _():
    import functools
    import math
    import numpy as np

    import marimo as mo
    import polars as pl
    import seaborn as sns
    sns.set_theme(style="white", font_scale=2)
    import spt_guidance.torch.crepe as crepe
    import spt_guidance.torch.fk_steering as fk_steering
    import spt_guidance.torch.rpcn as rpcn
    import torch

    from toy.model import (
        Bao2024EnsembleScoreMatching,
        EnsembleScoreModel,
        metropolis_hastings_sample,
    )
    import toy.plots

    return (
        Bao2024EnsembleScoreMatching,
        EnsembleScoreModel,
        crepe,
        fk_steering,
        functools,
        math,
        metropolis_hastings_sample,
        mo,
        np,
        pl,
        rpcn,
        sns,
        torch,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # 1D Steering Explorer

    Define a 1D Gaussian mixture and a reward function `reward_fn(x)`, then run
    unsteered, FK, DAS, RpCN, CREPE, and MH reference samplers independently.
    """)
    return


@app.cell(hide_code=True)
def _(gmm_density, means, mo, pl, sns, stds, weights, x_grid):
    gmm_df = pl.DataFrame(
        {
            "x": x_grid.cpu().numpy().tolist(),
            "density": gmm_density.cpu().numpy().tolist(),
        }
    )
    modes_df = pl.DataFrame(
        {
            "mean": means.cpu().numpy().tolist(),
            "std": stds.cpu().numpy().tolist(),
            "weight": weights.cpu().numpy().tolist(),
        }
    ).with_row_index("mode")

    gmm_plot = sns.relplot(
        data=gmm_df.to_dict(as_series=False),
        x="x",
        y="density",
        kind="line",
        height=3.5,
        aspect=2.8,
        color="tab:red",
        linewidth=2.0,
    )
    gmm_ax = gmm_plot.axes.flat[0]
    zero_y = [0.0] * len(modes_df)
    sns.scatterplot(x=modes_df["mean"].to_numpy(), y=zero_y, ax=gmm_ax, color="#1971c2", s=60, label="mode means")
    for row in modes_df.iter_rows(named=True):
        gmm_ax.text(float(row["mean"]), 0.0, f"w={row['weight']:.2f}, s={row['std']:.2f}", fontsize=8, ha="center", va="bottom")
    gmm_ax.set_title("GMM Preview")
    gmm_ax.set_xlabel("x")
    gmm_ax.set_ylabel("density")
    gmm_ax.grid(alpha=0.2)
    gmm_ax.legend(loc="upper right")
    gmm_plot.figure.tight_layout()
    gmm_view = mo.vstack([mo.md("## GMM Preview"), mo.as_html(gmm_plot.figure)])
    gmm_view
    return


@app.cell
def _(mo):
    n_modes = mo.ui.number(start=1, stop=8, step=1, value=1, label="n_modes")
    return (n_modes,)


@app.cell(hide_code=True)
def _(mo, n_modes):
    mode_count = int(n_modes.value)
    default_std = 3
    if mode_count % 2 == 0:
        mean_inits_half = [i - .5 for i in range(1, (mode_count // 2) + 1)]
    else:
        mean_inits_half = [i for i in range((mode_count + 1) // 2)]
    mean_inits = sorted(set([-i for i in mean_inits_half] + mean_inits_half))
    mean_array = mo.ui.array(
        [
            mo.ui.number(start=-20.0, stop=20.0, step=0.1, value=float(2 * 3 * default_std * mean_inits[idx]), label=f"mean_{idx}")
            for idx in range(mode_count)
        ]
    )
    std_array = mo.ui.array(
        [
            mo.ui.number(start=0.05, stop=6.0, step=0.05, value=default_std, label=f"std_{idx}")
            for idx in range(mode_count)
        ]
    )
    weight_array = mo.ui.array(
        [
            mo.ui.slider(start=0.01, stop=10.0, step=0.01, value=1.0, label=f"weight_{idx}")
            for idx in range(mode_count)
        ]
    )

    gmm_modes_view = mo.vstack(
        [
            mo.md("## 1D GMM Modes"),
            n_modes,
            mean_array,
            std_array,
            weight_array,
        ]
    )
    gmm_modes_view
    return mean_array, std_array, weight_array


@app.cell(hide_code=True)
def _(mo, pl, reward_values, sns, x_grid):
    reward_df = pl.DataFrame(
        {
            "x": x_grid.cpu().numpy().tolist(),
            "reward": reward_values.cpu().numpy().tolist(),
        }
    )
    reward_plot = sns.relplot(
        data=reward_df.to_dict(as_series=False),
        x="x",
        y="reward",
        kind="line",
        height=3.5,
        # aspect=10 / 3.5,
        color="tab:red",
        linewidth=2.0,
    ).set(xlim=(-5, 5))
    reward_ax = reward_plot.axes.flat[0]
    # reward_ax.set_title("Reward Preview")
    reward_ax.set_xlabel("x")
    reward_ax.set_ylabel("reward")
    reward_ax.grid(alpha=0.2)
    reward_plot.figure.tight_layout()
    reward_view = mo.vstack([mo.md("## Reward Preview"), mo.as_html(reward_plot.figure)])
    reward_view
    return


@app.cell(hide_code=True)
def _(gmm_density, mh_inverse_temp, mo, pl, reward_values, sns, torch, x_grid):
    tilted_reward_values = torch.exp(
        float(mh_inverse_temp.value) * reward_values
    )
    unnormalized_tilted_density = gmm_density * tilted_reward_values
    tilted_density_df = pl.DataFrame(
        {
            "x": x_grid.cpu().numpy().tolist(),
            "unnormalized_density": (unnormalized_tilted_density / 1e9).cpu().numpy().tolist(),
        }
    )
    tilted_density_plot = sns.relplot(
        data=tilted_density_df.to_dict(as_series=False),
        x="x",
        y="unnormalized_density",
        kind="line",
        height=3.5,
        # aspect=10 / 3.5,
        color="tab:purple",
        linewidth=2.0,
    ).set(xlim=(-5, 5), xlabel='', ylabel='', xticks=[-3.1, 0, 3.1])
    tilted_density_ax = tilted_density_plot.axes.flat[0]
    tilted_density_ax.set_title(
        r"$\propto p(x)\exp(\beta R(x))$"
        # f"Unnormalized Reward-Tilted GMM (beta={float(mh_inverse_temp.value):g})"
    )
    # tilted_density_ax.set_xlabel("x")
    # tilted_density_ax.set_ylabel("unnormalized density")
    tilted_density_ax.grid(alpha=0.2)
    tilted_density_plot.figure.tight_layout()
    tilted_density_view = mo.vstack(
        [
            mo.md("## Unnormalized Reward-Tilted Density"),
            mo.md(r"$\tilde{p}(x) = p_{\mathrm{GMM}}(x)\exp(\beta r(x))$"),
            mo.as_html(tilted_density_plot.figure),
        ]
    )
    tilted_density_view
    return


@app.cell
def _(torch):
    def reward_gauss(x, s=.5):
        return torch.exp(
            -(
                x / s
            ).square()
        )
    def gauss_low_high(x, m):
        return 1.6615 * reward_gauss(x - m) - 2 * reward_gauss(x - m + 1, s=1)
    def reward_fn(x):
        if x.ndim < 1 or x.shape[-1] != 1:
            raise ValueError("reward_fn expects states with shape (..., 1)")
        r = gauss_low_high(x, 3) + gauss_low_high(-x, 3)
        return r.squeeze(-1)

    return (reward_fn,)


@app.cell
def _(np, x_max, x_min):
    hist_bin_count = 60
    hist_range = (float(x_min.value), float(x_max.value))
    hist_bin_edges = np.linspace(hist_range[0], hist_range[1], hist_bin_count + 1)
    return hist_bin_edges, hist_range


@app.cell(hide_code=True)
def _(
    crepe_final,
    das_final,
    fk_final,
    gmm_density,
    hist_bin_edges,
    hist_range,
    mh_final,
    mo,
    pl,
    reward_values,
    rpcn_final,
    sns,
    unsteered_samples,
    x_grid,
):
    has_samples_plot = not (unsteered_samples is None and fk_final is None and das_final is None and rpcn_final is None and crepe_final is None and mh_final is None)
    plot_histogram = None
    plot_bin_count = len(hist_bin_edges) - 1
    if has_samples_plot:
        method_order = ["Base", "MCMC Reference", "FK Steering", "DAS Steering", "RpCN Steering", "CREPE"]
        sample_frames = []
        if unsteered_samples is not None:
            sample_frames.append(
                pl.DataFrame({"method": ["Base"] * unsteered_samples.shape[0], "x": unsteered_samples[:, 0].cpu().numpy().tolist()})
            )
        if fk_final is not None:
            sample_frames.append(
                pl.DataFrame({"method": ["FK Steering"] * fk_final.shape[0], "x": fk_final[:, 0].cpu().numpy().tolist()})
            )
        if das_final is not None:
            sample_frames.append(
                pl.DataFrame({"method": ["DAS Steering"] * das_final.shape[0], "x": das_final[:, 0].cpu().numpy().tolist()})
            )
        if rpcn_final is not None:
            sample_frames.append(
                pl.DataFrame({"method": ["RpCN Steering"] * rpcn_final.shape[0], "x": rpcn_final[:, 0].cpu().numpy().tolist()})
            )
        if crepe_final is not None:
            sample_frames.append(
                pl.DataFrame({"method": ["CREPE"] * crepe_final.shape[0], "x": crepe_final[:, 0].cpu().numpy().tolist()})
            )
        if mh_final is not None:
            mh_df = pl.DataFrame({"method": ["MCMC Reference"] * mh_final.shape[0], "x": mh_final[:, 0].cpu().numpy().tolist()})
            sample_frames.append(mh_df)
        samples_df = pl.concat(sample_frames)

        reward_scaled = reward_values.cpu().numpy()
        reward_scaled = (reward_scaled - reward_scaled.min()) / (reward_scaled.max() - reward_scaled.min() + 1e-8)
        reward_scaled = reward_scaled * gmm_density.max().item()
        overlay_df = pl.DataFrame(
            {
                "x": x_grid.cpu().numpy().tolist(),
                "density": gmm_density.cpu().numpy().tolist(),
                "reward_scaled": reward_scaled.tolist(),
            }
        )
        comparison_plot = (
            sns.displot(
                kind='hist',
                data=samples_df,
                x='x',
                hue='method',
                col='method',
                palette={
                    'Base': 'tab:grey',
                    'FK Steering': 'tab:blue',
                    'DAS Steering': 'tab:green',
                    'RpCN Steering': 'tab:orange',
                    'CREPE': 'tab:purple',
                    'MCMC Reference': 'tab:grey',
                },
                alpha=.6,
                col_order=method_order,
                bins=hist_bin_edges,
                stat='density',
                element='step',
                common_norm=False,
                legend=False,
            )
            .set(
                # ylim=(0, gmm_density.max() * 1.504),
                ylim=(0, 0.8),
            )
            .set_titles('')
        )
        for (row_index, column_index, hue_index), _ in comparison_plot.facet_data():
            ax = comparison_plot.axes[row_index, column_index]
            method_name = comparison_plot.col_names[column_index]
            sns.lineplot(
                data=overlay_df,
                x='x',
                y='density',
                color='black',
                ax=ax,
                linewidth=3,
                label='Base density',
            )
            sns.lineplot(
                data=overlay_df,
                x='x',
                y='reward_scaled',
                color='tab:red',
                ax=ax,
                linewidth=3,
                linestyle='--',
                label='Reward (scaled)',
            )
            if mh_final is not None and method_name != "MCMC Reference":
                sns.histplot(
                    data=mh_df,
                    x='x',
                    ax=ax,
                    zorder=-1,
                    stat='density',
                    bins=hist_bin_edges,
                    color='tab:grey',
                    element='step',
                    label='Reference tilted target',
                )
            ax.set_xlabel('')
            ax.get_legend().remove()
        handles, legend_labels = comparison_plot.axes.flat[0].get_legend_handles_labels()
        comparison_plot.add_legend(dict(zip(legend_labels, handles)), loc="upper center", ncol=3,
                      bbox_to_anchor=(0.3, 1.04),
                      frameon=True, fancybox=True)
        comparison_view = mo.vstack(
            [
                mo.md("## Sample Comparison"),
                mo.md(f"Histogram spec (shared with KL): bins={plot_bin_count}, range=[{hist_range[0]:.2f}, {hist_range[1]:.2f}]"),
                mo.as_html(comparison_plot),
            ]
        )
        plot_histogram = comparison_plot
    else:
        comparison_view = mo.md("Run one or more samplers to generate samples.")
    comparison_view
    return


@app.cell
def _():
    # if plot_histogram is not None:
    #     toy.plots.save_all_subfigures(plot_histogram, 'Histogram')
    return


@app.cell(hide_code=True)
def _(mo):
    seed = mo.ui.number(start=0, stop=100000, value=31, label="seed")
    device = mo.ui.dropdown(options=["cpu", "cuda"], value="cuda", label="device")
    batch_size = mo.ui.number(start=1, stop=4096, step=1, value=1024, label="batch_size")
    n_steps = mo.ui.number(start=20, stop=1000, step=10, value=50, label="n_steps")
    n_reference = mo.ui.number(start=128, stop=32768, step=128, value=1024, label="n_reference")

    epsilon_alpha = mo.ui.number(start=1e-6, stop=0.2, step=1e-4, value=1e-3, label="epsilon_alpha")
    epsilon_beta = mo.ui.number(start=1e-8, stop=0.2, step=1e-5, value=1e-5, label="epsilon_beta")
    vmap_chunk_size = mo.ui.number(start=8, stop=100_000, step=8, value=100_000, label="vmap_chunk_size")
    sampling_max_score_norm = mo.ui.number(
        start=1.0,
        stop=1000.0,
        step=1.0,
        value=100.0,
        label="sampling_max_score_norm",
    )

    x_min = mo.ui.number(start=-30.0, stop=0.0, step=0.5, value=-10.0, label="x_min")
    x_max = mo.ui.number(start=0.0, stop=30.0, step=0.5, value=10.0, label="x_max")
    n_grid = mo.ui.number(start=100, stop=4000, step=100, value=1200, label="plot_grid")

    fk_n_chains = mo.ui.number(start=1, stop=64, step=1, value=20, label="fk_n_chains")
    fk_inverse_temperature = mo.ui.number(start=0.00, step=0.05, value=25.0, label="fk_inverse_temperature")
    fk_potential = mo.ui.dropdown(
        options=['difference', 'max', 'sum'],
        value="max",
        label="fk_potential",
    )
    fk_temperature_count = mo.ui.number(start=1, stop=64, step=1, value=1, label="fk_temperature_count")
    fk_resample_threshold = mo.ui.number(start=0.05, stop=0.95, step=0.05, value=0.5, label="fk_ess_threshold")
    fk_check_every = mo.ui.number(start=1, stop=100, step=1, value=1, label="fk_check_every")

    das_n_particles = mo.ui.number(start=2, stop=128, step=1, value=20, label="das_n_particles")
    das_inverse_temperature = mo.ui.number(start=0.00, step=0.05, value=25.0, label="das_inverse_temperature")
    das_proposal_scale = mo.ui.number(start=0.0, stop=1.0, step=0.01, value=1.0, label="das_proposal_scale")
    das_tempering_gamma = mo.ui.number(start=0.0, stop=1.0, value=0.001, label="das_tempering_gamma")
    das_resample_threshold = mo.ui.number(start=0.05, stop=0.95, step=0.05, value=0.5, label="das_ess_threshold")

    rpcn_n_chains = mo.ui.number(start=2, stop=128, step=1, value=4, label="rpcn_n_chains")
    rpcn_update_count = mo.ui.number(start=1, stop=200, step=1, value=8, label="rpcn_update_count")
    rpcn_angle_min = mo.ui.number(start=0.001, stop=1.0, step=0.001, value=0.05, label="rpcn_angle_min")
    rpcn_inverse_temp_max = mo.ui.number(start=0.00, step=0.05, value=25.0, label="rpcn_inverse_temp_max")
    rpcn_check_every = mo.ui.number(start=1, stop=100, step=1, value=1, label="rpcn_check_every")
    rpcn_burn_in_frac = mo.ui.number(start=0, stop=1, step=0.01, value=0.23, label="rpcn_burn_in_frac")

    mh_num_chains = mo.ui.number(start=1, stop=256, step=1, value=64, label="mh_num_chains")
    mh_num_samples = mo.ui.number(start=10, stop=10000, step=10, value=2000, label="mh_num_samples")
    mh_burn_in = mo.ui.number(start=0, stop=20000, step=50, value=2000, label="mh_burn_in")
    mh_thinning = mo.ui.number(start=1, stop=50, step=1, value=2, label="mh_thinning")
    mh_proposal_std = mo.ui.number(start=0.01, stop=10.0, step=0.01, value=0.14, label="mh_proposal_std")
    mh_global_proposal_std = mo.ui.number(start=0.01, stop=30.0, step=0.01, value=3.0, label="mh_global_proposal_std")
    mh_global_proposal_probability = mo.ui.number(start=0.0, stop=1.0, step=0.01, value=0.1, label="mh_global_proposal_probability")
    mh_inverse_temp = mo.ui.number(start=0.00, step=0.05, value=25.0, label="mh_inverse_temp")

    crepe_num_samples = mo.ui.number(start=1, stop=1000, step=1, value=1, label="crepe_num_samples")
    crepe_burn_in = mo.ui.number(start=0, stop=10000, step=1, value=49, label="crepe_burn_in")
    crepe_thinning = mo.ui.number(start=1, stop=1000, step=1, value=1, label="crepe_thinning")
    crepe_inverse_temp = mo.ui.number(start=0.0, step=0.05, value=25.0, label="crepe_inverse_temperature")
    crepe_reward_power = mo.ui.number(start=0.05, stop=10.0, step=0.05, value=1.0, label="crepe_reward_schedule_power")
    crepe_reward_reference = mo.ui.number(start=0.0, stop=1.0, step=0.01, value=0.0, label="crepe_reward_at_reference")
    crepe_reward_target = mo.ui.number(start=0.0, stop=1.0, step=0.01, value=1.0, label="crepe_reward_at_target")
    crepe_refresh_reference = mo.ui.checkbox(value=True, label="crepe_refresh_reference")

    crepe_replica_count = mo.ui.number(start=2, stop=1000, step=1, value=9, label="crepe_replica_count")
    crepe_substeps = mo.ui.number(start=1, stop=64, step=1, value=5, label="crepe_edge_substeps_K")
    crepe_time_min = mo.ui.number(start=0.0, stop=0.99, step=0.001, value=0.00, label="crepe_time_min")
    crepe_time_max = mo.ui.number(start=0.01, stop=1.0, step=0.001, value=1.0, label="crepe_time_max")
    crepe_time_spacing = mo.ui.dropdown(options=["linear", "karras"], value="karras", label="crepe_time_spacing")
    crepe_karras_rho = mo.ui.number(start=0.1, stop=20.0, step=0.1, value=7.0, label="crepe_karras_rho")
    # crepe_continuation_steps = mo.ui.number(start=1, stop=1000, step=1, value=20, label="crepe_continuation_steps")
    crepe_ttr_score_scale = mo.ui.number(start=0.0, stop=5.0, step=0.05, value=0.0, label="crepe_target_to_reference_score_scale")
    crepe_rtt_score_scale = mo.ui.number(start=0.0, stop=5.0, step=0.05, value=1.0, label="crepe_reference_to_target_score_scale")
    crepe_ttr_reward_scale = mo.ui.number(start=0.0, stop=5.0, step=0.05, value=0.0, label="crepe_target_to_reference_reward_gradient_scale")
    crepe_rtt_reward_scale = mo.ui.number(start=0.0, stop=5.0, step=0.05, value=0.0, label="crepe_reference_to_target_reward_gradient_scale")
    crepe_gradient_clip = mo.ui.number(start=0.0, stop=10000.0, step=1.0, value=1000.0, label="crepe_reward_gradient_clip")
    crepe_initialization = mo.ui.dropdown(
        options=["denoising path", "independent marginals", "shared clean ancestor"],
        value="denoising path",
        label="crepe_initialization",
    )
    crepe_rne_method = mo.ui.dropdown(
        options=["exact transition", "ito approximation", "gaussian stabilized"],
        value="exact transition",
        label="crepe_rne_method",
    )
    crepe_local_exploration = mo.ui.checkbox(value=False, label="crepe_local_ULA")
    crepe_local_steps = mo.ui.number(start=1, stop=100, step=1, value=1, label="crepe_local_steps")
    crepe_local_scale = mo.ui.number(start=0.0, stop=10.0, step=0.05, value=1.0, label="crepe_local_step_scale")

    run_unsteered = mo.ui.run_button(label="Unsteered")
    run_fk = mo.ui.run_button(label="FK")
    run_das = mo.ui.run_button(label="DAS")
    run_rpcn = mo.ui.run_button(label="RpCN")
    run_crepe = mo.ui.run_button(label="CREPE")
    run_mh = mo.ui.run_button(label="MH")

    controls = mo.vstack(
        [
            mo.hstack([mh_num_chains, mh_num_samples, mh_burn_in, mh_thinning, mh_proposal_std, mh_global_proposal_std, mh_global_proposal_probability, mh_inverse_temp, run_mh], justify="start"),
            mo.md("## Experiment Controls"),
            mo.hstack([seed, device, batch_size, n_steps, n_reference, run_unsteered, run_fk, run_das, run_rpcn], justify="start"),
            mo.hstack([epsilon_alpha, epsilon_beta, vmap_chunk_size, sampling_max_score_norm], justify="start"),
            mo.hstack([x_min, x_max, n_grid], justify="start"),
            mo.hstack([fk_n_chains, fk_inverse_temperature, fk_potential, fk_temperature_count, fk_resample_threshold, fk_check_every], justify="start"),
            mo.hstack([das_n_particles, das_inverse_temperature, das_proposal_scale, das_tempering_gamma, das_resample_threshold], justify="start"),
            mo.md("DAS uses the paper's exponential schedule `lambda_i=min((1+gamma)^i-1, 1)` and enforces terminal `lambda=1`. Set `das_proposal_scale=1` for the undamped Gaussian proposal; smaller values retain the exact importance correction while improving finite-particle stability for this non-concave reward."),
            mo.hstack([rpcn_n_chains, rpcn_update_count, rpcn_angle_min, rpcn_inverse_temp_max, rpcn_burn_in_frac, rpcn_check_every], justify="start"),
            mo.hstack([mo.md("### CREPE"), run_crepe]),
            mo.hstack([crepe_num_samples, crepe_burn_in, crepe_thinning, crepe_inverse_temp, crepe_refresh_reference], justify="start"),
            mo.hstack([crepe_replica_count, crepe_substeps, crepe_time_min, crepe_time_max, crepe_time_spacing, crepe_karras_rho,
                       # crepe_continuation_steps,
                      ], justify="start"),
            mo.hstack([crepe_reward_target, crepe_reward_reference, crepe_reward_power], justify="start"),
            mo.hstack([crepe_ttr_score_scale, crepe_rtt_score_scale, crepe_ttr_reward_scale, crepe_rtt_reward_scale, crepe_gradient_clip], justify="start"),
            mo.hstack([crepe_initialization, crepe_rne_method, crepe_local_exploration, crepe_local_steps, crepe_local_scale], justify="start"),
            mo.md("CREPE times are stored target-to-reference. `crepe_replica_count` controls levels, `K` controls Euler substeps per edge, and each iteration is one checkerboard half-sweep. If `time_min > 0`, retained target-replica states are denoised to `t=0`. Forced clean-edge acceptance is intentionally excluded because it violates Eq. (13)."),
            mo.md(r"""
            #### CREPE Parameter Reference

            CREPE stores diffusion times in increasing target-to-reference order: `time_min` is the least-noisy target-side replica and `time_max` is the noisiest reference-side replica. One CREPE iteration is one checkerboard parity half-sweep, so each edge is attempted on alternating iterations.

            **Sampling and execution**

            | Control | Description |
            |---|---|
            | `seed` | Seeds PyTorch before initialization and sampling. CREPE shares this control with the other samplers. |
            | `device` | Runs the score model and CREPE on CPU or CUDA. CUDA falls back to CPU when unavailable. |
            | `batch_size` | Number of independent CREPE chains evolved together. The final histogram receives `crepe_num_samples * batch_size` correlated MCMC samples. |
            | `crepe_num_samples` | Number of retained target-replica batches per chain. This is not the raw half-sweep count. |
            | `crepe_burn_in` | Number of initial checkerboard half-sweeps discarded. The first retained batch is recorded after half-sweep index `burn_in`. |
            | `crepe_thinning` | Number of half-sweeps between retained batches after burn-in. The run uses `burn_in + (num_samples - 1) * thinning + 1` raw half-sweeps. |

            **Diffusion ladder and path discretization**

            | Control | Description |
            |---|---|
            | `crepe_replica_count` | Number of simultaneous diffusion-time replicas, including both ladder endpoints. There are `replica_count - 1` communication edges. More replicas shorten individual exchanges but increase memory and score evaluations per half-sweep. |
            | `crepe_edge_substeps_K` | Euler-Maruyama transitions simulated in each direction for one edge proposal. The full discretization contains `(replica_count - 1) * K + 1` points, and every `K`-th point is a replica. Simulation and RNE evaluation use the same transitions. |
            | `crepe_time_min` | Least-noisy time used by replica exchange. If greater than zero, each retained target replica is subsequently denoised to `t=0`; that continuation is not Metropolis-corrected by CREPE. |
            | `crepe_time_max` | Noisiest reference-side time. It must be greater than `time_min` and at most one for this probability path. |
            | `crepe_time_spacing` | `linear` spaces Gaussian-path times uniformly. `karras` spaces the effective EDM noise `sigma_eff(t)=beta(t)/alpha(t)` with the Karras power schedule and then maps those noise levels back to Gaussian-path times. |
            | `crepe_karras_rho` | Exponent in `sigma_i = (sigma_min^(1/rho) + ramp_i * (sigma_max^(1/rho) - sigma_min^(1/rho)))^rho`, where `sigma_min/max = beta(time_min/max) / alpha(time_min/max)`. Larger values allocate relatively more resolution near low effective noise. It is ignored for linear spacing. |
            | Continuation steps | Derived automatically so continuation from `time_min` to `t=0` uses intervals no larger than the fine-grid spacing. Linear ladders match diffusion-time spacing; Karras ladders match spacing in `sigma_eff(t)^(1/rho)`. |

            **Reward-tilted target**

            The noisy reward is `r_t(x) = inverse_temperature * gamma(t) * reward_fn(Tweedie(x, t))`, where
            `gamma(t) = reward_at_reference + (reward_at_target - reward_at_reference) * ((time_max - t) / (time_max - time_min))^power`.

            | Control | Description |
            |---|---|
            | `crepe_inverse_temperature` | Overall multiplier on the clean reward. Larger values define a more strongly tilted target and usually make communication harder. |
            | `crepe_reward_at_target` | Reward-schedule value at `time_min`. The paper normally uses one. |
            | `crepe_reward_at_reference` | Reward-schedule value at `time_max`. The paper normally uses zero so the reference distribution remains directly sampleable. Exact reference refresh requires this control to be zero. |
            | `crepe_reward_schedule_power` | Curvature of the target-to-reference reward interpolation. Larger powers suppress the reward over more of the noisy ladder and turn it on closer to the target side. |

            **Proposal dynamics**

            Each directional proposal drift has the form `f - g^2 * (score_scale * model_score + reward_scale * grad(r_t))`. These controls change proposal efficiency, not the reward-tilted target used in the Metropolis acceptance ratio.

            | Control | Description |
            |---|---|
            | `crepe_target_to_reference_score_scale` | Model-score strength in the increasing-time/noising proposal. Zero gives the base forward/noising drift, matching the natural official-code convention. |
            | `crepe_reference_to_target_score_scale` | Model-score strength in the decreasing-time/denoising proposal. One gives the usual learned reverse-SDE drift. |
            | `crepe_target_to_reference_reward_gradient_scale` | Annealed Tweedie reward-gradient strength in the increasing-time proposal. Zero omits reward guidance in this direction. |
            | `crepe_reference_to_target_reward_gradient_scale` | Annealed Tweedie reward-gradient strength in the decreasing-time proposal. The Metropolis correction remains active when this differs from one or is zero. |
            | `crepe_reward_gradient_clip` | Symmetric componentwise clamp applied to reward gradients before proposal and local-move scaling. Zero disables clipping. Clipping changes the proposal but not the target ratio. |

            **Initialization, RNE, and local moves**

            | Control | Description |
            |---|---|
            | `crepe_initialization` | `denoising path` samples the top marginal and follows one guided reference-to-target trajectory while retaining replica levels. `independent marginals` samples every base diffusion marginal independently. `shared clean ancestor` noises one shared clean reference sample independently at every level. Initialization affects burn-in, not the stationary target. |
            | `crepe_rne_method` | `exact transition` sums exact Gaussian log-density ratios for the discretized Euler kernels. `ito approximation` uses the report's two-endpoint Ito approximation. `gaussian stabilized` divides transition ratios by a tractable Gaussian process with marginal variance `alpha(t)^2 + beta(t)^2` and restores its endpoint ratio. |
            | `crepe_refresh_reference` | Replaces the noisiest replica after every half-sweep with an exact sample from the model probability-path marginal. It is valid for the controlled target only when `reward_at_reference=0`. The refreshed top replica does not also receive a local ULA move. |
            | `crepe_local_ULA` | Applies unadjusted Langevin exploration independently at each non-refreshed replica after communication. ULA can improve same-level mixing but introduces finite-step discretization bias. |
            | `crepe_local_steps` | Number of ULA updates applied to each eligible replica after every half-sweep. It is ignored when local ULA is disabled. |
            | `crepe_local_step_scale` | Multiplier on the denoising-aligned ULA step `h = 0.5 * g(t)^2 * delta_t * scale`; the update uses `x + h * target_score + sqrt(2h) * noise`. Zero leaves replicas unchanged. |

            Acceptance diagnostics are reported per edge using each edge's own attempt count. Forced clean-edge acceptance from parts of the official experiment code is intentionally unavailable because it bypasses the CREPE Metropolis rule and can bias the target.
            """),
        ]
    )
    controls
    return (
        batch_size,
        crepe_burn_in,
        crepe_gradient_clip,
        crepe_initialization,
        crepe_inverse_temp,
        crepe_karras_rho,
        crepe_local_exploration,
        crepe_local_scale,
        crepe_local_steps,
        crepe_num_samples,
        crepe_refresh_reference,
        crepe_replica_count,
        crepe_reward_power,
        crepe_reward_reference,
        crepe_reward_target,
        crepe_rne_method,
        crepe_rtt_reward_scale,
        crepe_rtt_score_scale,
        crepe_substeps,
        crepe_thinning,
        crepe_time_max,
        crepe_time_min,
        crepe_time_spacing,
        crepe_ttr_reward_scale,
        crepe_ttr_score_scale,
        das_inverse_temperature,
        das_n_particles,
        das_proposal_scale,
        das_resample_threshold,
        das_tempering_gamma,
        device,
        epsilon_alpha,
        epsilon_beta,
        fk_check_every,
        fk_inverse_temperature,
        fk_n_chains,
        fk_potential,
        fk_resample_threshold,
        fk_temperature_count,
        mh_burn_in,
        mh_global_proposal_probability,
        mh_global_proposal_std,
        mh_inverse_temp,
        mh_num_chains,
        mh_num_samples,
        mh_proposal_std,
        mh_thinning,
        n_grid,
        n_reference,
        n_steps,
        rpcn_angle_min,
        rpcn_burn_in_frac,
        rpcn_check_every,
        rpcn_inverse_temp_max,
        rpcn_n_chains,
        rpcn_update_count,
        run_crepe,
        run_das,
        run_fk,
        run_mh,
        run_rpcn,
        run_unsteered,
        sampling_max_score_norm,
        seed,
        vmap_chunk_size,
        x_max,
        x_min,
    )


@app.cell(hide_code=True)
def _(
    crepe_final,
    crepe_inverse_temp,
    das_final,
    das_inverse_temperature,
    fk_final,
    fk_inverse_temperature,
    hist_bin_edges,
    hist_range,
    means,
    mh_final,
    mh_inverse_temp,
    mo,
    np,
    pl,
    reward_fn,
    rpcn_final,
    rpcn_inverse_temp_max,
    stds,
    torch,
    unsteered_samples,
    weights,
):
    has_samples_stats = not (unsteered_samples is None and fk_final is None and das_final is None and rpcn_final is None and crepe_final is None and mh_final is None)
    stats_bin_count = len(hist_bin_edges) - 1
    if has_samples_stats:
        method_frames = []
        if unsteered_samples is not None:
            method_frames.append(pl.DataFrame({"method": ["Unsteered"] * unsteered_samples.shape[0], "x": unsteered_samples[:, 0].cpu().numpy().tolist()}))
        if fk_final is not None:
            method_frames.append(pl.DataFrame({"method": ["FK"] * fk_final.shape[0], "x": fk_final[:, 0].cpu().numpy().tolist()}))
        if das_final is not None:
            method_frames.append(pl.DataFrame({"method": ["DAS"] * das_final.shape[0], "x": das_final[:, 0].cpu().numpy().tolist()}))
        if rpcn_final is not None:
            method_frames.append(pl.DataFrame({"method": ["RpCN"] * rpcn_final.shape[0], "x": rpcn_final[:, 0].cpu().numpy().tolist()}))
        if crepe_final is not None:
            method_frames.append(pl.DataFrame({"method": ["CREPE"] * crepe_final.shape[0], "x": crepe_final[:, 0].cpu().numpy().tolist()}))
        if mh_final is not None:
            method_frames.append(pl.DataFrame({"method": ["MH"] * mh_final.shape[0], "x": mh_final[:, 0].cpu().numpy().tolist()}))
        stats_df = pl.concat(method_frames)

        reward_x = torch.tensor(stats_df["x"].to_numpy(), dtype=torch.float32).unsqueeze(-1)
        reward_vals = reward_fn(reward_x).detach().cpu().numpy().tolist()
        stats_df = stats_df.with_columns(pl.Series("reward", reward_vals))

        summary = (
            stats_df.group_by("method")
            .agg(
                pl.len().alias("count"),
                # pl.col("reward").quantile(0.05).alias("q05"),
                # pl.col("reward").quantile(0.50).alias("q50"),
                # pl.col("reward").quantile(0.95).alias("q95"),
                pl.col("reward").mean().alias("reward_mean"),
                pl.col("reward").max().alias("reward_max"),
                pl.col("reward").min().alias("reward_min"),
            )
            .sort("method", descending=True)
        )

        stats_kl_eps = 1e-8

        def stats_histogram_probability(stats_values):
            stats_counts, _ = np.histogram(stats_values, bins=hist_bin_edges)
            stats_probability = stats_counts.astype(np.float64) + stats_kl_eps
            return stats_probability / stats_probability.sum()

        def stats_discrete_kl(stats_p, stats_q):
            stats_positive_p = stats_p > 0
            return float(
                np.sum(
                    stats_p[stats_positive_p]
                    * (
                        np.log(stats_p[stats_positive_p])
                        - np.log(stats_q[stats_positive_p])
                    )
                )
            )

        stats_method_names = summary["method"].to_list()
        stats_method_probabilities = {
            stats_method_name: stats_histogram_probability(
                stats_df.filter(pl.col("method") == stats_method_name)["x"].to_numpy()
            )
            for stats_method_name in stats_method_names
        }

        kl_by_method = {}
        if mh_final is not None:
            stats_mh_probability = stats_method_probabilities["MH"]
            for stats_method_name in stats_method_names:
                kl_by_method[stats_method_name] = stats_discrete_kl(
                    stats_mh_probability,
                    stats_method_probabilities[stats_method_name],
                )
        else:
            for stats_method_name in stats_method_names:
                kl_by_method[stats_method_name] = None

        summary = summary.with_columns(
            pl.Series("kl_to_mh", [kl_by_method[m] for m in stats_method_names])
        )

        stats_edges = torch.as_tensor(hist_bin_edges, dtype=torch.float64)
        stats_standardized_edges = (
            stats_edges[:, None] - means.to(dtype=torch.float64)[None, :]
        ) / stds.to(dtype=torch.float64)[None, :]
        stats_component_cdf = 0.5 * (
            1.0 + torch.erf(stats_standardized_edges / np.sqrt(2.0))
        )
        stats_base_cdf = (
            stats_component_cdf * weights.to(dtype=torch.float64)[None, :]
        ).sum(dim=1)
        stats_base_probability = (
            stats_base_cdf[1:] - stats_base_cdf[:-1]
        ).clamp_min(1e-300).cpu().numpy()
        stats_base_probability /= stats_base_probability.sum()

        stats_beta_by_method = {
            "Unsteered": 0.0,
            "FK": float(fk_inverse_temperature.value),
            "DAS": float(das_inverse_temperature.value),
            "RpCN": float(rpcn_inverse_temp_max.value),
            "CREPE": float(crepe_inverse_temp.value),
            "MH": float(mh_inverse_temp.value),
        }
        stats_kl_to_untilted = {}
        stats_objective = {}
        for stats_method_name in stats_method_names:
            stats_untilted_kl = stats_discrete_kl(
                stats_method_probabilities[stats_method_name],
                stats_base_probability,
            )
            stats_kl_to_untilted[stats_method_name] = stats_untilted_kl
            stats_method_beta = stats_beta_by_method[stats_method_name]
            stats_reward_mean = summary.filter(
                pl.col("method") == stats_method_name
            )["reward_mean"].item()
            stats_objective[stats_method_name] = (
                stats_reward_mean - stats_untilted_kl / stats_method_beta
                if stats_method_beta > 0.0
                else None
            )

        summary = summary.with_columns(
            pl.Series(
                "reward_minus_kl_over_beta",
                [stats_objective[m] for m in stats_method_names],
            ),
            pl.Series(
                "kl_to_untilted",
                [stats_kl_to_untilted[m] for m in stats_method_names],
            ),
        )

        stats_view = mo.vstack(
            [
                mo.md("## Summary Statistics"),
                mo.md(f"Histogram spec (shared with plot): bins={stats_bin_count}, range=[{hist_range[0]:.2f}, {hist_range[1]:.2f}]"),
                mo.ui.table(summary.to_dicts()),
            ]
        )
    else:
        stats_view = mo.md("Summary statistics will appear after running at least one sampler.")
    stats_view
    return


@app.cell(hide_code=True)
def _(mh_diagnostics, mo):
    if mh_diagnostics is None:
        mh_diag_view = mo.vstack([mo.md("## MH Diagnostics"), mo.md("Run MH to populate diagnostics.")])
    else:
        acceptance_rate = float(mh_diagnostics["acceptance_rate"])
        if acceptance_rate < 0.15:
            status_label = "Too low"
            hint = "Acceptance is low; try decreasing mh_proposal_std."
        elif acceptance_rate > 0.60:
            status_label = "Too high"
            hint = "Acceptance is high; try increasing mh_proposal_std."
        else:
            status_label = "Good"
            hint = "Acceptance is in a healthy range."

        mh_local_acceptance = mh_diagnostics["local_acceptance_rate"]
        mh_global_acceptance = mh_diagnostics["global_acceptance_rate"]
        mh_diag_table = [
            {"metric": "acceptance_rate", "value": f"{acceptance_rate:.4f}"},
            {"metric": "local_acceptance_rate", "value": "disabled" if mh_local_acceptance is None else f"{mh_local_acceptance:.4f}"},
            {"metric": "global_acceptance_rate", "value": "disabled" if mh_global_acceptance is None else f"{mh_global_acceptance:.4f}"},
            {"metric": "accepted", "value": int(mh_diagnostics["accepted"])},
            {"metric": "proposed", "value": int(mh_diagnostics["proposed"])},
            {"metric": "global_accepted", "value": int(mh_diagnostics["global_accepted"])},
            {"metric": "global_proposed", "value": int(mh_diagnostics["global_proposed"])},
            {"metric": "num_chains", "value": int(mh_diagnostics["num_chains"])},
            {"metric": "num_samples_per_chain", "value": int(mh_diagnostics["num_samples"])},
            {"metric": "burn_in", "value": int(mh_diagnostics["burn_in"])},
            {"metric": "thinning", "value": int(mh_diagnostics["thinning"])},
            {"metric": "proposal_std", "value": float(mh_diagnostics["proposal_std"])},
            {"metric": "global_proposal_std", "value": float(mh_diagnostics["global_proposal_std"])},
            {"metric": "global_proposal_probability", "value": float(mh_diagnostics["global_proposal_probability"])},
            {"metric": "init_source", "value": mh_diagnostics["init_source"]},
        ]
        mh_diag_view = mo.vstack(
            [
                mo.md("## MH Diagnostics"),
                mo.md(f"**Status:** {status_label}"),
                mo.ui.table(mh_diag_table),
                mo.md(hint),
            ]
        )
    mh_diag_view
    return


@app.cell(hide_code=True)
def _(crepe_diagnostics, crepe_is_stale, mo):
    if crepe_diagnostics is None:
        crepe_diag_view = mo.vstack(
            [mo.md("## CREPE Diagnostics"), mo.md("Run CREPE to populate diagnostics.")]
        )
    elif crepe_is_stale:
        crepe_diag_view = mo.vstack(
            [mo.md("## CREPE Diagnostics"), mo.md("Controls changed; rerun CREPE.")]
        )
    else:
        crepe_diag_summary = [
            {"metric": "mean_edge_acceptance", "value": f"{crepe_diagnostics['acceptance_rate_mean']:.4f}"},
            {"metric": "replica_count", "value": crepe_diagnostics["replica_count"]},
            {"metric": "edge_count", "value": crepe_diagnostics["edge_count"]},
            {"metric": "communication_substeps_K", "value": crepe_diagnostics["substeps"]},
            {"metric": "raw_half_sweeps", "value": crepe_diagnostics["iteration_count"]},
            {"metric": "retained_batches", "value": crepe_diagnostics["retained_batches"]},
            {"metric": "chain_batch_size", "value": crepe_diagnostics["batch_size"]},
            {"metric": "initialization", "value": crepe_diagnostics["initialization"]},
            {"metric": "RNE method", "value": crepe_diagnostics["rne_method"]},
            {"metric": "time_spacing", "value": crepe_diagnostics["time_spacing"]},
            {"metric": "time_range", "value": f"[{crepe_diagnostics['time_min']:.4g}, {crepe_diagnostics['time_max']:.4g}]"},
            {"metric": "continuation_steps", "value": crepe_diagnostics["continuation_steps"]},
            {"metric": "Karras rho", "value": crepe_diagnostics["karras_rho"]},
            {"metric": "inverse_temperature", "value": crepe_diagnostics["inverse_temperature"]},
            {"metric": "reward schedule target/reference/power", "value": f"{crepe_diagnostics['reward_at_target']}/{crepe_diagnostics['reward_at_reference']}/{crepe_diagnostics['reward_power']}"},
            {"metric": "target-to-reference score/reward", "value": f"{crepe_diagnostics['target_to_reference_score_scale']}/{crepe_diagnostics['target_to_reference_reward_scale']}"},
            {"metric": "reference-to-target score/reward", "value": f"{crepe_diagnostics['reference_to_target_score_scale']}/{crepe_diagnostics['reference_to_target_reward_scale']}"},
            {"metric": "local_exploration", "value": crepe_diagnostics["local_exploration"]},
            {"metric": "reference_refresh", "value": crepe_diagnostics["reference_refresh"]},
        ]
        crepe_diag_times = crepe_diagnostics["replica_times"]
        crepe_diag_accepts = crepe_diagnostics["acceptance_count"].sum(dim=1)
        crepe_diag_attempts = crepe_diagnostics["attempt_count"].sum(dim=1)
        crepe_diag_rates = crepe_diag_accepts / crepe_diag_attempts.clamp(min=1)
        crepe_diag_edges = [
            {
                "edge": crepe_diag_idx,
                "time_lower": float(crepe_diag_times[crepe_diag_idx]),
                "time_upper": float(crepe_diag_times[crepe_diag_idx + 1]),
                "accepted": int(crepe_diag_accepts[crepe_diag_idx]),
                "attempted": int(crepe_diag_attempts[crepe_diag_idx]),
                "acceptance_rate": float(crepe_diag_rates[crepe_diag_idx]),
            }
            for crepe_diag_idx in range(crepe_diagnostics["edge_count"])
        ]
        crepe_diag_view = mo.vstack(
            [
                mo.md("## CREPE Diagnostics"),
                mo.ui.table(crepe_diag_summary),
                mo.md("### Edge Communication"),
                mo.ui.table(crepe_diag_edges),
            ]
        )
    crepe_diag_view
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Helpers
    """)
    return


@app.cell
def _(
    batch_size,
    math,
    reference_data,
    reward_fn,
    rpcn,
    rpcn_angle_min,
    rpcn_burn_in_frac,
    rpcn_check_every,
    rpcn_inverse_temp_max,
    rpcn_n_chains,
    rpcn_update_count,
    run_rpcn,
    score_model,
    seed,
    set_rpcn_samples,
    timesteps,
    torch,
):
    if run_rpcn.value:
        torch.manual_seed(int(seed.value))
        kernel = rpcn.KernelPreconditionedCrankNicolson(
            1,
            device=reference_data.device,
            dtype=reference_data.dtype,
        )
        steering = rpcn.ResidualMCMCSteering(
            kernel,
            torch.linspace(
                math.pi / 2 - 1e-3,
                float(rpcn_angle_min.value),
                int(rpcn_n_chains.value),
                device=reference_data.device,
                dtype=reference_data.dtype,
            ).sin(),
            reward_fn=reward_fn,
            expected_next_fn=score_model.expected_next_fn,
            std_fn=score_model.std_fn,
            intermediate_reward=rpcn.IntermediateReward.EXPECTED_SAMPLE,
            inverse_temperature=torch.linspace(
                0.0,
                float(rpcn_inverse_temp_max.value),
                int(rpcn_n_chains.value),
                device=reference_data.device,
                dtype=reference_data.dtype,
            ),
            kwargs_intermediate_reward={"expected_sample_fn": score_model.get_tweedie},
        )

        def rpcn_update_fn(step_idx, t0, t1, reward):
            _ = (t0, t1, reward)
            return (step_idx >= timesteps.numel() * float(rpcn_burn_in_frac.value)) and (step_idx % int(rpcn_check_every.value) == 0)

        with torch.no_grad():
            set_rpcn_samples(
                steering.run(
                    timesteps,
                    pcn_should_update_fn=rpcn_update_fn,
                    pcn_update_count=int(rpcn_update_count.value),
                    batch_size=int(batch_size.value),
                )
            )
    return


@app.cell
def _(
    batch_size,
    crepe,
    crepe_burn_in,
    crepe_config_signature,
    crepe_fine_times,
    crepe_gradient_clip,
    crepe_initialization,
    crepe_inverse_temp,
    crepe_karras_rho,
    crepe_local_exploration,
    crepe_local_scale,
    crepe_local_steps,
    crepe_num_samples,
    crepe_refresh_reference,
    crepe_replica_times,
    crepe_reward_power,
    crepe_reward_reference,
    crepe_reward_target,
    crepe_rne_method,
    crepe_rtt_reward_scale,
    crepe_rtt_score_scale,
    crepe_substeps,
    crepe_thinning,
    crepe_time_spacing,
    crepe_ttr_reward_scale,
    crepe_ttr_score_scale,
    functools,
    math,
    reference_data,
    reward_fn,
    run_crepe,
    score_model,
    seed,
    set_crepe_diagnostics,
    set_crepe_samples,
    torch,
):
    if run_crepe.value:
        torch.manual_seed(int(seed.value))
        crepe_reward_beta = float(crepe_inverse_temp.value)
        crepe_reward_exponent = float(crepe_reward_power.value)
        crepe_reward_floor = float(crepe_reward_reference.value)
        crepe_reward_ceiling = float(crepe_reward_target.value)
        if crepe_refresh_reference.value and crepe_reward_floor != 0.0:
            raise ValueError(
                "Exact reference refresh requires crepe_reward_at_reference=0"
            )
        crepe_ttr_score = float(crepe_ttr_score_scale.value)
        crepe_rtt_score = float(crepe_rtt_score_scale.value)
        crepe_ttr_reward = float(crepe_ttr_reward_scale.value)
        crepe_rtt_reward = float(crepe_rtt_reward_scale.value)
        crepe_clip = float(crepe_gradient_clip.value)
        crepe_min_time = crepe_replica_times[0]
        crepe_max_time = crepe_replica_times[-1]
        crepe_edge_substeps = int(crepe_substeps.value)

        def crepe_schedule_value(crepe_schedule_time):
            crepe_schedule_fraction = (
                (crepe_max_time - crepe_schedule_time)
                / (crepe_max_time - crepe_min_time)
            ).clamp(0.0, 1.0)
            return crepe_reward_floor + (
                crepe_reward_ceiling - crepe_reward_floor
            ) * crepe_schedule_fraction.pow(crepe_reward_exponent)

        def crepe_tweedie_value(crepe_tweedie_time, crepe_tweedie_state):
            crepe_tweedie_score = score_model._score_particles(
                crepe_tweedie_time, crepe_tweedie_state
            )
            crepe_tweedie_alpha = score_model.prob_path.alpha(crepe_tweedie_time)
            crepe_tweedie_beta = score_model.prob_path.beta(crepe_tweedie_time)
            return (
                crepe_tweedie_state
                + crepe_tweedie_beta.square() * crepe_tweedie_score
            ) / crepe_tweedie_alpha

        def crepe_log_reward(crepe_reward_time, crepe_reward_state):
            return (
                crepe_reward_beta
                * crepe_schedule_value(crepe_reward_time)
                * reward_fn(crepe_tweedie_value(crepe_reward_time, crepe_reward_state))
            )

        def crepe_reward_gradient(crepe_gradient_time, crepe_gradient_state):
            def crepe_summed_reward(crepe_gradient_input):
                return crepe_log_reward(
                    crepe_gradient_time, crepe_gradient_input
                ).sum()

            crepe_gradient = torch.func.grad(crepe_summed_reward)(crepe_gradient_state)
            if crepe_clip > 0.0:
                crepe_gradient = crepe_gradient.clamp(-crepe_clip, crepe_clip)
            return crepe_gradient.detach()

        def crepe_model_ttr_drift(crepe_drift_time, crepe_drift_state):
            return score_model.prob_path.f(crepe_drift_time, crepe_drift_state)

        def crepe_model_rtt_drift(crepe_drift_time, crepe_drift_state):
            crepe_drift_g = score_model.prob_path.g(crepe_drift_time)
            return score_model.prob_path.f(
                crepe_drift_time, crepe_drift_state
            ) - crepe_drift_g.square() * score_model._score_particles(
                crepe_drift_time, crepe_drift_state
            )

        def crepe_proposal_ttr_drift(crepe_drift_time, crepe_drift_state):
            crepe_drift_g = score_model.prob_path.g(crepe_drift_time)
            crepe_drift_score = score_model._score_particles(
                crepe_drift_time, crepe_drift_state
            )
            crepe_drift_reward = (
                crepe_reward_gradient(crepe_drift_time, crepe_drift_state)
                if crepe_ttr_reward != 0.0
                else torch.zeros_like(crepe_drift_state)
            )
            return score_model.prob_path.f(
                crepe_drift_time, crepe_drift_state
            ) - crepe_drift_g.square() * (
                crepe_ttr_score * crepe_drift_score
                + crepe_ttr_reward * crepe_drift_reward
            )

        def crepe_proposal_rtt_drift(crepe_drift_time, crepe_drift_state):
            crepe_drift_g = score_model.prob_path.g(crepe_drift_time)
            crepe_drift_score = score_model._score_particles(
                crepe_drift_time, crepe_drift_state
            )
            crepe_drift_reward = (
                crepe_reward_gradient(crepe_drift_time, crepe_drift_state)
                if crepe_rtt_reward != 0.0
                else torch.zeros_like(crepe_drift_state)
            )
            return score_model.prob_path.f(
                crepe_drift_time, crepe_drift_state
            ) - crepe_drift_g.square() * (
                crepe_rtt_score * crepe_drift_score
                + crepe_rtt_reward * crepe_drift_reward
            )

        def crepe_gaussian_log_prob(
            crepe_gaussian_value,
            crepe_gaussian_mean,
            crepe_gaussian_variance,
        ):
            crepe_gaussian_log = -0.5 * (
                (crepe_gaussian_value - crepe_gaussian_mean).square()
                / crepe_gaussian_variance
                + torch.log(2.0 * torch.pi * crepe_gaussian_variance)
            )
            return crepe_gaussian_log.flatten(1).sum(dim=1)

        def crepe_transition_logs(
            crepe_path_times,
            crepe_path_states,
            crepe_ttr_drift_fn,
            crepe_rtt_drift_fn,
        ):
            crepe_ttr_logs = []
            crepe_rtt_logs = []
            for crepe_path_idx in range(crepe_path_times.numel() - 1):
                crepe_path_t0 = crepe_path_times[crepe_path_idx]
                crepe_path_t1 = crepe_path_times[crepe_path_idx + 1]
                crepe_path_dt = crepe_path_t1 - crepe_path_t0
                crepe_path_lower = crepe_path_states[crepe_path_idx]
                crepe_path_upper = crepe_path_states[crepe_path_idx + 1]
                crepe_ttr_variance = (
                    crepe_path_dt * score_model.prob_path.g(crepe_path_t0).square()
                )
                crepe_rtt_variance = (
                    crepe_path_dt * score_model.prob_path.g(crepe_path_t1).square()
                )
                crepe_ttr_mean = crepe_path_lower + crepe_path_dt * crepe_ttr_drift_fn(
                    crepe_path_t0, crepe_path_lower
                )
                crepe_rtt_mean = crepe_path_upper - crepe_path_dt * crepe_rtt_drift_fn(
                    crepe_path_t1, crepe_path_upper
                )
                crepe_ttr_logs.append(
                    crepe_gaussian_log_prob(
                        crepe_path_upper, crepe_ttr_mean, crepe_ttr_variance
                    )
                )
                crepe_rtt_logs.append(
                    crepe_gaussian_log_prob(
                        crepe_path_lower, crepe_rtt_mean, crepe_rtt_variance
                    )
                )
            return torch.stack(crepe_ttr_logs), torch.stack(crepe_rtt_logs)

        def crepe_stabilizing_rtt_drift(crepe_stable_time, crepe_stable_state):
            crepe_stable_alpha = score_model.prob_path.alpha(crepe_stable_time)
            crepe_stable_beta = score_model.prob_path.beta(crepe_stable_time)
            crepe_stable_variance = crepe_stable_alpha.square() + crepe_stable_beta.square()
            crepe_stable_score = -crepe_stable_state / crepe_stable_variance
            crepe_stable_g = score_model.prob_path.g(crepe_stable_time)
            return score_model.prob_path.f(
                crepe_stable_time, crepe_stable_state
            ) - crepe_stable_g.square() * crepe_stable_score

        def crepe_stabilizing_marginal_log_prob(
            crepe_stable_time, crepe_stable_state
        ):
            crepe_stable_alpha = score_model.prob_path.alpha(crepe_stable_time)
            crepe_stable_beta = score_model.prob_path.beta(crepe_stable_time)
            crepe_stable_variance = crepe_stable_alpha.square() + crepe_stable_beta.square()
            return crepe_gaussian_log_prob(
                crepe_stable_state,
                torch.zeros_like(crepe_stable_state),
                crepe_stable_variance,
            )

        def crepe_path_log_rne(
            crepe_path_times,
            crepe_path_states,
            crepe_ttr_drift_fn,
            crepe_rtt_drift_fn,
        ):
            if crepe_rne_method.value == "ito approximation":
                crepe_ito_terms = []
                for crepe_ito_idx in range(crepe_path_times.numel() - 1):
                    crepe_ito_times = crepe_path_times[crepe_ito_idx : crepe_ito_idx + 2]
                    crepe_ito_path = crepe_path_states[crepe_ito_idx : crepe_ito_idx + 2]
                    crepe_ito_ttr_drift = torch.stack(
                        [
                            crepe_ttr_drift_fn(crepe_ito_times[0], crepe_ito_path[0]),
                            crepe_ttr_drift_fn(crepe_ito_times[1], crepe_ito_path[1]),
                        ]
                    )
                    crepe_ito_rtt_drift = torch.stack(
                        [
                            crepe_rtt_drift_fn(crepe_ito_times[0], crepe_ito_path[0]),
                            crepe_rtt_drift_fn(crepe_ito_times[1], crepe_ito_path[1]),
                        ]
                    )
                    crepe_ito_diffusion = torch.stack(
                        [
                            score_model.prob_path.g(crepe_ito_times[0]),
                            score_model.prob_path.g(crepe_ito_times[1]),
                        ]
                    )
                    crepe_ito_terms.append(
                        crepe.PathLogRatio.from_ito_integrals(
                            crepe_ito_times,
                            crepe_ito_path,
                            crepe_ito_ttr_drift,
                            crepe_ito_rtt_drift,
                            crepe_ito_diffusion,
                        )
                    )
                return torch.stack(crepe_ito_terms).sum(dim=0)

            crepe_primary_ttr_logs, crepe_primary_rtt_logs = crepe_transition_logs(
                crepe_path_times,
                crepe_path_states,
                crepe_ttr_drift_fn,
                crepe_rtt_drift_fn,
            )
            if crepe_rne_method.value == "exact transition":
                return crepe.PathLogRatio.from_transition_log_probabilities(
                    crepe_primary_ttr_logs, crepe_primary_rtt_logs
                )

            crepe_stable_ttr_logs, crepe_stable_rtt_logs = crepe_transition_logs(
                crepe_path_times,
                crepe_path_states,
                crepe_model_ttr_drift,
                crepe_stabilizing_rtt_drift,
            )
            return crepe.PathLogRatio.from_reference_stabilized_transition_log_probabilities(
                crepe_primary_ttr_logs,
                crepe_primary_rtt_logs,
                crepe_stable_ttr_logs,
                crepe_stable_rtt_logs,
                crepe_stabilizing_marginal_log_prob(
                    crepe_path_times[0], crepe_path_states[0]
                ),
                crepe_stabilizing_marginal_log_prob(
                    crepe_path_times[-1], crepe_path_states[-1]
                ),
            )

        def crepe_summarize_path(crepe_path_times, crepe_path_states):
            crepe_model_rne = crepe_path_log_rne(
                crepe_path_times,
                crepe_path_states,
                crepe_model_ttr_drift,
                crepe_model_rtt_drift,
            )
            crepe_proposal_rne = crepe_path_log_rne(
                crepe_path_times,
                crepe_path_states,
                crepe_proposal_ttr_drift,
                crepe_proposal_rtt_drift,
            )
            crepe_target_ratio = crepe.TargetLogRatio.reward_tilt(
                crepe_model_rne,
                crepe_log_reward(crepe_path_times[0], crepe_path_states[0]),
                crepe_log_reward(crepe_path_times[-1], crepe_path_states[-1]),
            )
            return crepe_target_ratio.detach(), crepe_proposal_rne.detach()

        def crepe_edge_grid(crepe_grid_t0, crepe_grid_t1):
            _ = crepe_grid_t1
            crepe_grid_start = torch.argmin(
                (crepe_fine_times - crepe_grid_t0).abs()
            )
            crepe_grid_offsets = torch.arange(
                crepe_edge_substeps + 1, device=crepe_fine_times.device
            )
            return crepe_fine_times[crepe_grid_start + crepe_grid_offsets]

        def crepe_target_to_reference_path_fn(
            crepe_path_t0, crepe_path_t1, crepe_path_lower
        ):
            crepe_path_times = crepe_edge_grid(crepe_path_t0, crepe_path_t1)
            crepe_path_values = [crepe_path_lower.detach()]
            crepe_path_current = crepe_path_lower.detach()
            for crepe_path_idx in range(crepe_path_times.numel() - 1):
                crepe_path_now = crepe_path_times[crepe_path_idx]
                crepe_path_next = crepe_path_times[crepe_path_idx + 1]
                crepe_path_dt = crepe_path_next - crepe_path_now
                crepe_path_current = (
                    crepe_path_current
                    + crepe_path_dt
                    * crepe_proposal_ttr_drift(crepe_path_now, crepe_path_current)
                    + crepe_path_dt.sqrt()
                    * score_model.prob_path.g(crepe_path_now)
                    * torch.randn_like(crepe_path_current)
                ).detach()
                crepe_path_values.append(crepe_path_current)
            crepe_path_states = torch.stack(crepe_path_values)
            crepe_target_ratio, crepe_proposal_rne = crepe_summarize_path(
                crepe_path_times, crepe_path_states
            )
            return crepe_path_current, crepe_target_ratio, crepe_proposal_rne

        def crepe_reference_to_target_path_fn(
            crepe_path_t0, crepe_path_t1, crepe_path_upper
        ):
            crepe_path_times = crepe_edge_grid(crepe_path_t0, crepe_path_t1)
            crepe_reverse_values = [crepe_path_upper.detach()]
            crepe_path_current = crepe_path_upper.detach()
            for crepe_path_idx in range(crepe_path_times.numel() - 1, 0, -1):
                crepe_path_now = crepe_path_times[crepe_path_idx]
                crepe_path_next = crepe_path_times[crepe_path_idx - 1]
                crepe_path_dt = crepe_path_now - crepe_path_next
                crepe_path_current = (
                    crepe_path_current
                    - crepe_path_dt
                    * crepe_proposal_rtt_drift(crepe_path_now, crepe_path_current)
                    + crepe_path_dt.sqrt()
                    * score_model.prob_path.g(crepe_path_now)
                    * torch.randn_like(crepe_path_current)
                ).detach()
                crepe_reverse_values.append(crepe_path_current)
            crepe_path_states = torch.stack(crepe_reverse_values[::-1])
            crepe_target_ratio, crepe_proposal_rne = crepe_summarize_path(
                crepe_path_times, crepe_path_states
            )
            return crepe_path_current, crepe_target_ratio, crepe_proposal_rne

        def crepe_reference_sample(crepe_reference_time):
            crepe_reference_index = torch.randint(
                reference_data.shape[0], (), device=reference_data.device
            )
            crepe_reference_clean = reference_data[crepe_reference_index]
            return (
                score_model.prob_path.alpha(crepe_reference_time)
                * crepe_reference_clean
                + score_model.prob_path.beta(crepe_reference_time)
                * torch.randn_like(crepe_reference_clean)
            )

        def crepe_init_path_fn(crepe_init_times):
            if crepe_initialization.value == "independent marginals":
                crepe_init_indices = torch.randint(
                    reference_data.shape[0],
                    (crepe_init_times.numel(),),
                    device=reference_data.device,
                )
                crepe_init_clean = reference_data[crepe_init_indices]
                return (
                    score_model.prob_path.alpha(crepe_init_times)[:, None]
                    * crepe_init_clean
                    + score_model.prob_path.beta(crepe_init_times)[:, None]
                    * torch.randn_like(crepe_init_clean)
                )
            if crepe_initialization.value == "shared clean ancestor":
                crepe_init_index = torch.randint(
                    reference_data.shape[0], (), device=reference_data.device
                )
                crepe_init_clean = reference_data[crepe_init_index]
                return (
                    score_model.prob_path.alpha(crepe_init_times)[:, None]
                    * crepe_init_clean
                    + score_model.prob_path.beta(crepe_init_times)[:, None]
                    * torch.randn(
                        crepe_init_times.numel(),
                        crepe_init_clean.numel(),
                        device=reference_data.device,
                        dtype=reference_data.dtype,
                    )
                )

            crepe_init_current = crepe_reference_sample(crepe_init_times[-1])
            crepe_init_reverse = [crepe_init_current]
            for crepe_init_idx in range(crepe_fine_times.numel() - 1, 0, -1):
                crepe_init_now = crepe_fine_times[crepe_init_idx]
                crepe_init_next = crepe_fine_times[crepe_init_idx - 1]
                crepe_init_dt = crepe_init_now - crepe_init_next
                crepe_init_current = (
                    crepe_init_current
                    - crepe_init_dt
                    * crepe_proposal_rtt_drift(crepe_init_now, crepe_init_current[None])[0]
                    + crepe_init_dt.sqrt()
                    * score_model.prob_path.g(crepe_init_now)
                    * torch.randn_like(crepe_init_current)
                ).detach()
                if (crepe_init_idx - 1) % crepe_edge_substeps == 0:
                    crepe_init_reverse.append(crepe_init_current)
            return torch.stack(crepe_init_reverse[::-1])

        def crepe_local_exploration_fn(crepe_local_time, crepe_local_state):
            crepe_local_index = torch.argmin(
                (crepe_fine_times - crepe_local_time).abs()
            )
            crepe_local_index = crepe_local_index.clamp(0, crepe_fine_times.numel() - 2)
            crepe_local_dt = (
                crepe_fine_times[crepe_local_index + 1]
                - crepe_fine_times[crepe_local_index]
            )
            crepe_local_result = crepe_local_state.detach()
            for crepe_local_idx in range(int(crepe_local_steps.value)):
                crepe_local_target_score = score_model._score_particles(
                    crepe_local_time, crepe_local_result
                ) + crepe_reward_gradient(crepe_local_time, crepe_local_result)
                crepe_local_h = (
                    0.5
                    * score_model.prob_path.g(crepe_local_time).square()
                    * crepe_local_dt
                    * float(crepe_local_scale.value)
                )
                crepe_local_result = (
                    crepe_local_result
                    + crepe_local_h * crepe_local_target_score
                    + (2.0 * crepe_local_h).sqrt()
                    * torch.randn_like(crepe_local_result)
                ).detach()
            return crepe_local_result

        crepe_sampler = crepe.CREPE(
            init_path_fn=crepe_init_path_fn,
            target_to_reference_path_fn=crepe_target_to_reference_path_fn,
            reference_to_target_path_fn=crepe_reference_to_target_path_fn,
            local_exploration_fn=(
                crepe_local_exploration_fn
                if crepe_local_exploration.value
                else None
            ),
            refresh_reference_fn=(
                crepe_reference_sample if crepe_refresh_reference.value else None
            ),
        )
        crepe_iteration_count = int(crepe_burn_in.value) + (
            int(crepe_num_samples.value) - 1
        ) * int(crepe_thinning.value) + 1
        crepe_result = crepe_sampler.run(
            crepe_replica_times,
            crepe_iteration_count,
            batch_size=int(batch_size.value),
            should_save_sample_fn=functools.partial(
                crepe.SaveSampleFn.burn_in_and_thin,
                burn_in=int(crepe_burn_in.value),
                thin=int(crepe_thinning.value),
            ),
        )

        crepe_output_samples = crepe_result.samples.detach()
        crepe_continuation_steps = 0
        if crepe_min_time.item() > 0.0:
            if crepe_time_spacing.value == "linear":
                crepe_continuation_interval = (
                    crepe_fine_times[1] - crepe_fine_times[0]
                )
                crepe_continuation_steps = math.ceil(
                    float(crepe_min_time / crepe_continuation_interval)
                )
                crepe_continuation_times = torch.linspace(
                    crepe_min_time,
                    0.0,
                    crepe_continuation_steps + 1,
                    device=crepe_fine_times.device,
                    dtype=crepe_fine_times.dtype,
                )
            else:
                crepe_continuation_rho = float(crepe_karras_rho.value)

                def crepe_continuation_effective_sigma(crepe_continuation_time):
                    return (
                        score_model.prob_path.beta(crepe_continuation_time)
                        / score_model.prob_path.alpha(crepe_continuation_time)
                    )

                crepe_fine_schedule = crepe_continuation_effective_sigma(
                    crepe_fine_times
                ).pow(1.0 / crepe_continuation_rho)
                crepe_continuation_interval = (
                    crepe_fine_schedule[1] - crepe_fine_schedule[0]
                )
                crepe_zero_time = crepe_fine_times.new_zeros(())
                crepe_zero_schedule = crepe_continuation_effective_sigma(
                    crepe_zero_time
                ).pow(1.0 / crepe_continuation_rho)
                crepe_min_schedule = crepe_fine_schedule[0]
                crepe_continuation_steps = math.ceil(
                    float(
                        (crepe_min_schedule - crepe_zero_schedule)
                        / crepe_continuation_interval
                    )
                )
                crepe_continuation_schedule = torch.linspace(
                    crepe_min_schedule,
                    crepe_zero_schedule,
                    crepe_continuation_steps + 1,
                    device=crepe_fine_times.device,
                    dtype=crepe_fine_times.dtype,
                )
                crepe_continuation_sigma = crepe_continuation_schedule.pow(
                    crepe_continuation_rho
                )

                crepe_continuation_time_lower = torch.zeros_like(
                    crepe_continuation_sigma
                )
                crepe_continuation_time_upper = torch.full_like(
                    crepe_continuation_sigma, float(crepe_min_time.item())
                )
                for crepe_continuation_bisection_idx in range(48):
                    crepe_continuation_time_midpoint = (
                        crepe_continuation_time_lower
                        + crepe_continuation_time_upper
                    ) / 2
                    crepe_continuation_midpoint_sigma = (
                        crepe_continuation_effective_sigma(
                            crepe_continuation_time_midpoint
                        )
                    )
                    crepe_continuation_time_lower = torch.where(
                        crepe_continuation_midpoint_sigma
                        < crepe_continuation_sigma,
                        crepe_continuation_time_midpoint,
                        crepe_continuation_time_lower,
                    )
                    crepe_continuation_time_upper = torch.where(
                        crepe_continuation_midpoint_sigma
                        < crepe_continuation_sigma,
                        crepe_continuation_time_upper,
                        crepe_continuation_time_midpoint,
                    )
                crepe_continuation_interior_times = (
                    crepe_continuation_time_lower
                    + crepe_continuation_time_upper
                )[1:-1] / 2
                crepe_continuation_times = torch.cat(
                    (
                        crepe_min_time.reshape(1),
                        crepe_continuation_interior_times,
                        crepe_zero_time.reshape(1),
                    )
                )

            if crepe_continuation_times.numel() != crepe_continuation_steps + 1:
                raise ValueError("CREPE continuation has the wrong number of points")
            if not bool(torch.isfinite(crepe_continuation_times).all()):
                raise ValueError("CREPE continuation times must be finite")
            if not bool(
                (
                    crepe_continuation_times[:-1]
                    > crepe_continuation_times[1:]
                ).all()
            ):
                raise ValueError(
                    "CREPE continuation times must be strictly decreasing"
                )
            crepe_output_shape = crepe_output_samples.shape
            crepe_continuation = crepe_output_samples.reshape(-1, 1)
            for crepe_cont_idx in range(crepe_continuation_times.numel() - 1):
                crepe_cont_now = crepe_continuation_times[crepe_cont_idx]
                crepe_cont_next = crepe_continuation_times[crepe_cont_idx + 1]
                crepe_cont_dt = crepe_cont_now - crepe_cont_next
                crepe_continuation = (
                    crepe_continuation
                    - crepe_cont_dt
                    * crepe_proposal_rtt_drift(crepe_cont_now, crepe_continuation)
                    + crepe_cont_dt.sqrt()
                    * score_model.prob_path.g(crepe_cont_now)
                    * torch.randn_like(crepe_continuation)
                ).detach()
            crepe_output_samples = crepe_continuation.reshape(crepe_output_shape)

        crepe_attempted = crepe_result.attempt_count > 0
        crepe_acceptance_mean = float(
            crepe_result.acceptance_rate[crepe_attempted].mean().item()
        )
        set_crepe_samples(crepe_output_samples)
        set_crepe_diagnostics(
            {
                "config_signature": crepe_config_signature,
                "acceptance_rate_mean": crepe_acceptance_mean,
                "acceptance_rate": crepe_result.acceptance_rate.detach().cpu(),
                "acceptance_count": crepe_result.acceptance_count.detach().cpu(),
                "attempt_count": crepe_result.attempt_count.detach().cpu(),
                "replica_times": crepe_replica_times.detach().cpu(),
                "fine_times": crepe_fine_times.detach().cpu(),
                "replica_count": int(crepe_replica_times.numel()),
                "edge_count": int(crepe_replica_times.numel() - 1),
                "substeps": crepe_edge_substeps,
                "iteration_count": crepe_iteration_count,
                "retained_batches": int(crepe_output_samples.shape[0]),
                "batch_size": int(crepe_output_samples.shape[1]),
                "initialization": crepe_initialization.value,
                "rne_method": crepe_rne_method.value,
                "time_spacing": crepe_time_spacing.value,
                "time_min": float(crepe_replica_times[0].item()),
                "time_max": float(crepe_replica_times[-1].item()),
                "continuation_steps": crepe_continuation_steps,
                "karras_rho": float(crepe_karras_rho.value),
                "reward_at_target": crepe_reward_ceiling,
                "reward_at_reference": crepe_reward_floor,
                "reward_power": crepe_reward_exponent,
                "inverse_temperature": crepe_reward_beta,
                "target_to_reference_score_scale": crepe_ttr_score,
                "reference_to_target_score_scale": crepe_rtt_score,
                "target_to_reference_reward_scale": crepe_ttr_reward,
                "reference_to_target_reward_scale": crepe_rtt_reward,
                "local_exploration": bool(crepe_local_exploration.value),
                "reference_refresh": bool(crepe_refresh_reference.value),
            }
        )
    return


@app.cell
def _(
    device,
    gmm_log_prob,
    gmm_sample,
    metropolis_hastings_sample,
    mh_burn_in,
    mh_global_proposal_probability,
    mh_global_proposal_std,
    mh_inverse_temp,
    mh_num_chains,
    mh_num_samples,
    mh_proposal_std,
    mh_thinning,
    reward_fn,
    run_mh,
    seed,
    set_mh_diagnostics,
    set_mh_samples,
    torch,
):
    if run_mh.value:
        torch.manual_seed(int(seed.value))
        selected_device_mh = device.value
        if selected_device_mh == "cuda" and not torch.cuda.is_available():
            selected_device_mh = "cpu"
        selected_device_obj = torch.device(selected_device_mh)

        def log_target_fn(x: torch.Tensor) -> torch.Tensor:
            x_flat = x[:, 0]
            return gmm_log_prob(x_flat) + float(mh_inverse_temp.value) * reward_fn(x)

        mh_chain_samples, mh_diag = metropolis_hastings_sample(
            log_target_fn=log_target_fn,
            num_chains=int(mh_num_chains.value),
            num_samples=int(mh_num_samples.value),
            dim=1,
            burn_in=int(mh_burn_in.value),
            proposal_std=float(mh_proposal_std.value),
            global_proposal_std=float(mh_global_proposal_std.value),
            global_proposal_probability=float(mh_global_proposal_probability.value),
            thinning=int(mh_thinning.value),
            initial_x=gmm_sample(int(mh_num_chains.value), selected_device_mh),
            device=selected_device_obj,
            dtype=torch.float32,
        )
        mh_diag.update(
            {
                "num_chains": int(mh_num_chains.value),
                "num_samples": int(mh_num_samples.value),
                "burn_in": int(mh_burn_in.value),
                "thinning": int(mh_thinning.value),
                "proposal_std": float(mh_proposal_std.value),
                "global_proposal_std": float(mh_global_proposal_std.value),
                "global_proposal_probability": float(mh_global_proposal_probability.value),
                "init_source": "gmm_sample",
            }
        )
        set_mh_samples(mh_chain_samples)
        set_mh_diagnostics(mh_diag)
    return


@app.cell
def _(mean_array, std_array, torch, weight_array):
    means = torch.tensor(mean_array.value, dtype=torch.float32)
    stds = torch.tensor(std_array.value, dtype=torch.float32)
    raw_weights = torch.tensor(weight_array.value, dtype=torch.float32)
    weights = raw_weights / raw_weights.sum()
    return means, stds, weights


@app.cell
def _(
    batch_size,
    crepe_burn_in,
    crepe_gradient_clip,
    crepe_initialization,
    crepe_inverse_temp,
    crepe_karras_rho,
    crepe_local_exploration,
    crepe_local_scale,
    crepe_local_steps,
    crepe_num_samples,
    crepe_refresh_reference,
    crepe_replica_count,
    crepe_reward_power,
    crepe_reward_reference,
    crepe_reward_target,
    crepe_rne_method,
    crepe_rtt_reward_scale,
    crepe_rtt_score_scale,
    crepe_substeps,
    crepe_thinning,
    crepe_time_max,
    crepe_time_min,
    crepe_time_spacing,
    crepe_ttr_reward_scale,
    crepe_ttr_score_scale,
    device,
    epsilon_alpha,
    epsilon_beta,
    means,
    n_reference,
    n_steps,
    sampling_max_score_norm,
    seed,
    stds,
    vmap_chunk_size,
    weights,
):
    crepe_config_signature = (
        int(seed.value),
        device.value,
        int(n_reference.value),
        float(epsilon_alpha.value),
        float(epsilon_beta.value),
        int(vmap_chunk_size.value),
        float(sampling_max_score_norm.value),
        tuple(float(value) for value in means.tolist()),
        tuple(float(value) for value in stds.tolist()),
        tuple(float(value) for value in weights.tolist()),
        int(batch_size.value),
        int(n_steps.value),
        int(crepe_num_samples.value),
        int(crepe_burn_in.value),
        int(crepe_thinning.value),
        float(crepe_inverse_temp.value),
        float(crepe_reward_power.value),
        float(crepe_reward_reference.value),
        float(crepe_reward_target.value),
        bool(crepe_refresh_reference.value),
        int(crepe_replica_count.value),
        int(crepe_substeps.value),
        float(crepe_time_min.value),
        float(crepe_time_max.value),
        crepe_time_spacing.value,
        float(crepe_karras_rho.value),
        # crepe_continuation_steps,
        float(crepe_ttr_score_scale.value),
        float(crepe_rtt_score_scale.value),
        float(crepe_ttr_reward_scale.value),
        float(crepe_rtt_reward_scale.value),
        float(crepe_gradient_clip.value),
        crepe_initialization.value,
        crepe_rne_method.value,
        bool(crepe_local_exploration.value),
        int(crepe_local_steps.value),
        float(crepe_local_scale.value),
    )
    return (crepe_config_signature,)


@app.cell
def _(means, n_grid, stds, torch, weights, x_max, x_min):
    x_grid = torch.linspace(float(x_min.value), float(x_max.value), int(n_grid.value), dtype=torch.float32)

    def gmm_log_prob(x_values: torch.Tensor) -> torch.Tensor:
        x_col = x_values.unsqueeze(-1)
        mean_row = means.to(device=x_values.device, dtype=x_values.dtype).unsqueeze(0)
        std_row = stds.to(device=x_values.device, dtype=x_values.dtype).unsqueeze(0)
        weight_row = weights.to(device=x_values.device, dtype=x_values.dtype).unsqueeze(0)
        log_norm = -0.5 * torch.log(2.0 * torch.pi * std_row.square())
        log_comp = log_norm - 0.5 * ((x_col - mean_row) / std_row).square() + torch.log(weight_row)
        return torch.logsumexp(log_comp, dim=1)

    def gmm_sample(num_points: int, device_str: str) -> torch.Tensor:
        device_obj = torch.device(device_str)
        probs = weights.to(device_obj)
        centers = means.to(device_obj)
        scales = stds.to(device_obj)
        indices = torch.multinomial(probs, num_points, replacement=True)
        eps = torch.randn(num_points, device=device_obj)
        samples = centers[indices] + scales[indices] * eps
        return samples.unsqueeze(1)

    gmm_density = torch.exp(gmm_log_prob(x_grid))
    return gmm_density, gmm_log_prob, gmm_sample, x_grid


@app.cell
def _(reward_fn, torch, x_grid):
    reward_values = reward_fn(x_grid.unsqueeze(-1))
    if reward_values.ndim != 1:
        raise ValueError("reward_fn must map (..., 1) states to (...) rewards")
    if not torch.isfinite(reward_values).all():
        raise ValueError("reward_fn returned non-finite values on preview grid")
    return (reward_values,)


@app.cell
def _(mo):
    get_unsteered_samples, set_unsteered_samples = mo.state(None)
    get_fk_samples, set_fk_samples = mo.state(None)
    get_das_samples, set_das_samples = mo.state(None)
    get_rpcn_samples, set_rpcn_samples = mo.state(None)
    get_crepe_samples, set_crepe_samples = mo.state(None)
    get_crepe_diagnostics, set_crepe_diagnostics = mo.state(None)
    get_mh_samples, set_mh_samples = mo.state(None)
    get_mh_diagnostics, set_mh_diagnostics = mo.state(None)
    return (
        get_crepe_diagnostics,
        get_crepe_samples,
        get_das_samples,
        get_fk_samples,
        get_mh_diagnostics,
        get_mh_samples,
        get_rpcn_samples,
        get_unsteered_samples,
        set_crepe_diagnostics,
        set_crepe_samples,
        set_das_samples,
        set_fk_samples,
        set_mh_diagnostics,
        set_mh_samples,
        set_rpcn_samples,
        set_unsteered_samples,
    )


@app.cell
def _(
    Bao2024EnsembleScoreMatching,
    EnsembleScoreModel,
    device,
    epsilon_alpha,
    epsilon_beta,
    gmm_sample,
    n_reference,
    n_steps,
    sampling_max_score_norm,
    seed,
    torch,
    vmap_chunk_size,
):
    selected_device = device.value
    if selected_device == "cuda" and not torch.cuda.is_available():
        selected_device = "cpu"

    torch.manual_seed(int(seed.value))
    cfg = type("Cfg", (), {})()
    cfg.epsilon_alpha = float(epsilon_alpha.value)
    cfg.epsilon_beta = float(epsilon_beta.value)
    cfg.sampling_time_step_count = int(n_steps.value)
    cfg.vmap_chunk_size = int(vmap_chunk_size.value)
    cfg.sampling_max_score_norm = float(sampling_max_score_norm.value)
    cfg.renormalize_sampled_noise = False
    cfg.sample_noise_scale_std = True
    cfg.sample_noise_add_mean = True

    prob_path = Bao2024EnsembleScoreMatching(cfg)
    score_model = EnsembleScoreModel(cfg, prob_path)

    reference_data = gmm_sample(int(n_reference.value), selected_device)
    score_model.set_training_data(reference_data)

    timesteps = score_model.prob_path.linspace_time(cfg.sampling_time_step_count, device=reference_data.device)
    return reference_data, score_model, timesteps


@app.cell
def _(
    crepe_karras_rho,
    crepe_replica_count,
    crepe_substeps,
    crepe_time_max,
    crepe_time_min,
    crepe_time_spacing,
    reference_data,
    score_model,
    torch,
):
    crepe_ladder_min = float(crepe_time_min.value)
    crepe_ladder_max = float(crepe_time_max.value)
    if not 0.0 <= crepe_ladder_min < crepe_ladder_max <= 1.0:
        raise ValueError("CREPE requires 0 <= time_min < time_max <= 1")
    crepe_ladder_replica_count = int(crepe_replica_count.value)
    crepe_ladder_substeps = int(crepe_substeps.value)
    crepe_ladder_point_count = (
        crepe_ladder_replica_count - 1
    ) * crepe_ladder_substeps + 1
    crepe_ladder_ramp = torch.linspace(
        0.0,
        1.0,
        crepe_ladder_point_count,
        device=reference_data.device,
        dtype=reference_data.dtype,
    )
    if crepe_time_spacing.value == "linear":
        crepe_fine_times = crepe_ladder_min + (
            crepe_ladder_max - crepe_ladder_min
        ) * crepe_ladder_ramp
    else:
        crepe_ladder_rho = float(crepe_karras_rho.value)
        crepe_ladder_min_tensor = reference_data.new_tensor(crepe_ladder_min)
        crepe_ladder_max_tensor = reference_data.new_tensor(crepe_ladder_max)

        def crepe_effective_sigma(crepe_sigma_time):
            return (
                score_model.prob_path.beta(crepe_sigma_time)
                / score_model.prob_path.alpha(crepe_sigma_time)
            )

        crepe_ladder_sigma_min = crepe_effective_sigma(crepe_ladder_min_tensor)
        crepe_ladder_sigma_max = crepe_effective_sigma(crepe_ladder_max_tensor)
        crepe_ladder_sigma_grid = (
            crepe_ladder_sigma_min.pow(1.0 / crepe_ladder_rho)
            + (
                crepe_ladder_sigma_max.pow(1.0 / crepe_ladder_rho)
                - crepe_ladder_sigma_min.pow(1.0 / crepe_ladder_rho)
            )
            * crepe_ladder_ramp
        ).pow(crepe_ladder_rho)

        crepe_ladder_time_lower = torch.full_like(
            crepe_ladder_sigma_grid, crepe_ladder_min
        )
        crepe_ladder_time_upper = torch.full_like(
            crepe_ladder_sigma_grid, crepe_ladder_max
        )
        for crepe_ladder_bisection_idx in range(48):
            crepe_ladder_time_midpoint = (
                crepe_ladder_time_lower + crepe_ladder_time_upper
            ) / 2
            crepe_ladder_midpoint_sigma = crepe_effective_sigma(
                crepe_ladder_time_midpoint
            )
            crepe_ladder_time_lower = torch.where(
                crepe_ladder_midpoint_sigma < crepe_ladder_sigma_grid,
                crepe_ladder_time_midpoint,
                crepe_ladder_time_lower,
            )
            crepe_ladder_time_upper = torch.where(
                crepe_ladder_midpoint_sigma < crepe_ladder_sigma_grid,
                crepe_ladder_time_upper,
                crepe_ladder_time_midpoint,
            )
        crepe_ladder_interior_times = (
            crepe_ladder_time_lower + crepe_ladder_time_upper
        )[1:-1] / 2
        crepe_fine_times = torch.cat(
            (
                crepe_ladder_min_tensor[None],
                crepe_ladder_interior_times,
                crepe_ladder_max_tensor[None],
            )
        )
    crepe_replica_times = crepe_fine_times[::crepe_ladder_substeps]
    return crepe_fine_times, crepe_replica_times


@app.cell
def _(
    crepe_config_signature,
    get_crepe_diagnostics,
    get_crepe_samples,
    get_das_samples,
    get_fk_samples,
    get_mh_diagnostics,
    get_mh_samples,
    get_rpcn_samples,
    get_unsteered_samples,
):
    crepe_diagnostics = get_crepe_diagnostics()
    crepe_samples = get_crepe_samples()
    crepe_is_stale = (
        crepe_diagnostics is not None
        and crepe_diagnostics.get("config_signature") != crepe_config_signature
    )
    if crepe_is_stale:
        crepe_samples = None
    das_samples = get_das_samples()
    fk_samples = get_fk_samples()
    mh_diagnostics = get_mh_diagnostics()
    mh_samples = get_mh_samples()
    rpcn_samples = get_rpcn_samples()
    unsteered_samples = get_unsteered_samples()
    return (
        crepe_diagnostics,
        crepe_is_stale,
        crepe_samples,
        das_samples,
        fk_samples,
        mh_diagnostics,
        mh_samples,
        rpcn_samples,
        unsteered_samples,
    )


@app.cell
def _(
    batch_size,
    fk_check_every,
    fk_inverse_temperature,
    fk_n_chains,
    fk_potential,
    fk_resample_threshold,
    fk_steering,
    fk_temperature_count,
    reference_data,
    reward_fn,
    run_fk,
    score_model,
    seed,
    set_fk_samples,
    timesteps,
    torch,
):
    if run_fk.value:
        torch.manual_seed(int(seed.value))
        target_inverse_temperature = float(fk_inverse_temperature.value)
        temperature_count = int(fk_temperature_count.value)
        if temperature_count == 1:
            fk_inverse_temperatures = target_inverse_temperature
        else:
            fk_inverse_temperatures = torch.linspace(
                0.0,
                target_inverse_temperature,
                temperature_count,
                device=reference_data.device,
                dtype=reference_data.dtype,
            )

        def propose_step(t0, t1, z):
            mean = score_model.expected_next_fn(t0, t1, z)
            if t1 > 0:
                std = score_model.std_fn(t0, t1)
                return mean + std * torch.randn_like(z)
            return mean

        def adaptive_resample_safe(time_step, ensemble_size, log_weights):
            identity_indices = torch.arange(ensemble_size, device=log_weights.device)
            if time_step % int(fk_check_every.value) != 0:
                did_resample = torch.tensor(False, device=log_weights.device)
                return identity_indices, did_resample, did_resample
            safe_log_weights = torch.nan_to_num(log_weights, nan=0.0)
            probs = torch.softmax(safe_log_weights, dim=0)
            ess = 1.0 / probs.square().sum()
            threshold = float(fk_resample_threshold.value) * ensemble_size
            did_resample = ess < threshold
            ancestor_indices = torch.where(
                did_resample,
                fk_steering.ResampleFn.do_resample(ensemble_size, probs),
                identity_indices,
            )
            return ancestor_indices, did_resample, torch.zeros_like(did_resample)

        fk = fk_steering.FeynmanKac(
            init_param_fn=lambda: torch.randn(
                1,
                device=reference_data.device,
                dtype=reference_data.dtype,
            ),
            reward_fn=reward_fn,
            propose_fn=propose_step,
            resample_fn=adaptive_resample_safe,
            potential=fk_steering.Potential(fk_potential.value),
            intermediate_reward=fk_steering.IntermediateReward.EXPECTED_SAMPLE,
            inverse_temperature=fk_inverse_temperatures,
            kwargs_intermediate_reward={"expected_sample_fn": score_model.get_tweedie},
        )

        with torch.no_grad():
            fk_particles, fk_log_weights = fk.run(
                timesteps,
                int(fk_n_chains.value),
                batch_size=int(batch_size.value),
            )
            if temperature_count == 1:
                target_particles = fk_particles
                target_log_weights = fk_log_weights
            else:
                target_particles = fk_particles[-1]
                target_log_weights = fk_log_weights[-1]
            set_fk_samples(
                fk_steering.SelectFn.weighted_sample(target_particles, target_log_weights)
            )
    return


@app.cell
def _(
    batch_size,
    das_inverse_temperature,
    das_n_particles,
    das_proposal_scale,
    das_resample_threshold,
    das_tempering_gamma,
    np,
    reference_data,
    reward_fn,
    run_das,
    score_model,
    seed,
    set_das_samples,
    ssp_resample_indices,
    timesteps,
    torch,
):
    if run_das.value:
        torch.manual_seed(int(seed.value))
        das_ssp_rng = np.random.default_rng(int(seed.value))
        das_batch_count = int(batch_size.value)
        das_particle_count = int(das_n_particles.value)
        das_inverse_temp = float(das_inverse_temperature.value)
        das_guidance_scale = float(das_proposal_scale.value)
        das_tempering_steps = torch.arange(
            timesteps.numel(),
            device=reference_data.device,
            dtype=reference_data.dtype,
        )
        das_tempering = ((1.0 + float(das_tempering_gamma.value)) ** das_tempering_steps - 1.0).clamp(max=1.0)
        das_tempering = torch.cat([das_tempering[:-1], torch.ones_like(das_tempering[-1:])])

        def das_tweedie_fn(das_time, das_latents):
            das_score = score_model._score_particles(das_time, das_latents)
            das_alpha = score_model.prob_path.alpha(das_time)
            das_beta = score_model.prob_path.beta(das_time)
            return (das_latents + das_beta.square() * das_score) / das_alpha

        def das_log_gaussian_fn(das_value, das_mean, das_std):
            das_variance = das_std.square()
            das_dimension = das_value.shape[-1]
            das_squared_distance = (das_value - das_mean).square().sum(dim=-1)
            return -0.5 * (
                das_squared_distance / das_variance
                + das_dimension * torch.log(2.0 * torch.pi * das_variance)
            )

        das_particles = torch.randn(
            das_batch_count,
            das_particle_count,
            1,
            device=reference_data.device,
            dtype=reference_data.dtype,
        )
        das_log_weights = torch.full(
            (das_batch_count, das_particle_count),
            -torch.log(torch.tensor(float(das_particle_count), device=reference_data.device, dtype=reference_data.dtype)),
            device=reference_data.device,
            dtype=reference_data.dtype,
        )

        for das_step_idx, das_time_pair in enumerate(timesteps.unfold(0, 2, 1)):
            das_t0 = das_time_pair[0]
            das_t1 = das_time_pair[1]
            das_log_normalized = das_log_weights - torch.logsumexp(das_log_weights, dim=1, keepdim=True)
            das_probabilities = das_log_normalized.exp()
            das_ess = 1.0 / das_probabilities.square().sum(dim=1)
            das_should_resample = das_ess < float(das_resample_threshold.value) * das_particle_count
            das_identity_indices = torch.arange(das_particle_count, device=reference_data.device).expand(das_batch_count, -1)
            das_ancestor_indices = das_identity_indices.clone()
            for das_batch_idx in torch.nonzero(das_should_resample, as_tuple=False).flatten().tolist():
                das_ssp_indices = ssp_resample_indices(
                    das_probabilities[das_batch_idx].detach().cpu().numpy(),
                    das_particle_count,
                    das_ssp_rng,
                )
                das_ancestor_indices[das_batch_idx] = torch.as_tensor(
                    das_ssp_indices,
                    device=reference_data.device,
                    dtype=torch.long,
                )
            das_particles = torch.gather(das_particles, 1, das_ancestor_indices[:, :, None])
            das_resampled_log_weights = torch.where(
                das_should_resample[:, None],
                -torch.log(torch.tensor(float(das_particle_count), device=reference_data.device, dtype=reference_data.dtype)),
                das_log_normalized,
            )

            das_current = das_particles.reshape(-1, 1).detach().requires_grad_(True)
            das_base_mean = score_model.expected_next_fn(das_t0, das_t1, das_current)
            das_current_tweedie = das_tweedie_fn(das_t0, das_current)
            das_current_reward = reward_fn(das_current_tweedie).reshape(-1)
            das_reward_gradient = torch.autograd.grad(das_current_reward.sum(), das_current)[0]
            das_transition_std = score_model.std_fn(das_t0, das_t1)
            das_next_lambda = das_tempering[das_step_idx + 1]
            das_current_lambda = das_tempering[das_step_idx]
            das_proposal_mean = (
                das_base_mean
                + das_guidance_scale
                * das_transition_std.square()
                * das_next_lambda
                * das_inverse_temp
                * das_reward_gradient
            )

            with torch.no_grad():
                das_next = das_proposal_mean + das_transition_std * torch.randn_like(das_current)
                das_next_tweedie = das_tweedie_fn(das_t1, das_next)
                das_next_reward = reward_fn(das_next_tweedie).reshape(-1)
                das_incremental_log_weights = (
                    das_log_gaussian_fn(das_next, das_base_mean, das_transition_std)
                    - das_log_gaussian_fn(das_next, das_proposal_mean, das_transition_std)
                    + das_next_lambda * das_inverse_temp * das_next_reward
                    - das_current_lambda * das_inverse_temp * das_current_reward
                )
                das_log_weights = das_resampled_log_weights + das_incremental_log_weights.reshape(das_batch_count, das_particle_count)
                das_particles = das_next.reshape(das_batch_count, das_particle_count, 1)

        with torch.no_grad():
            das_final_probabilities = torch.softmax(das_log_weights, dim=1)
            das_final_indices = torch.multinomial(das_final_probabilities, 1)
            das_selected = torch.gather(das_particles, 1, das_final_indices[:, :, None]).squeeze(1)
            set_das_samples(das_selected)
    return


@app.cell
def _(np):
    def ssp_resample_indices(weights, output_size, rng):
        weights = np.asarray(weights, dtype=np.float64)
        if weights.ndim != 1 or weights.size == 0:
            raise ValueError("SSP weights must be a non-empty one-dimensional array")
        if not np.isfinite(weights).all() or np.any(weights < 0.0):
            raise ValueError("SSP weights must be finite and non-negative")
        weight_sum = weights.sum()
        if weight_sum <= 0.0:
            raise ValueError("SSP weights must have positive sum")
        weights = weights / weight_sum

        particle_count = weights.shape[0]
        scaled_weights = output_size * weights
        offspring_counts = np.floor(scaled_weights).astype(np.int64)
        fractional_counts = scaled_weights - offspring_counts
        if particle_count == 1:
            offspring_counts[0] = output_size
            return np.arange(particle_count, dtype=np.int64).repeat(offspring_counts)

        uniforms = rng.random(particle_count - 1)
        i, j = 0, 1
        for step_idx in range(particle_count - 1):
            delta_i = min(fractional_counts[j], 1.0 - fractional_counts[i])
            delta_j = min(fractional_counts[i], 1.0 - fractional_counts[j])
            delta_sum = delta_i + delta_j
            increase_i_probability = delta_i / delta_sum if delta_sum > 0.0 else 0.0
            if uniforms[step_idx] < increase_i_probability:
                j, i = i, j
                delta_i = delta_j
            if fractional_counts[j] < 1.0 - fractional_counts[i]:
                fractional_counts[i] += delta_i
                j = step_idx + 2
            else:
                fractional_counts[j] -= delta_i
                offspring_counts[i] += 1
                i = step_idx + 2

        offspring_total = offspring_counts.sum()
        if offspring_total == output_size - 1:
            last_index = i if j == particle_count else j
            if fractional_counts[last_index] > 0.99:
                offspring_counts[last_index] += 1
                offspring_total += 1
        if offspring_total != output_size:
            raise ValueError("SSP resampling produced the wrong number of offspring")
        return np.arange(particle_count, dtype=np.int64).repeat(offspring_counts)

    return (ssp_resample_indices,)


@app.cell
def _(
    n_steps,
    reference_data,
    run_unsteered,
    score_model,
    seed,
    set_unsteered_samples,
    torch,
):
    if run_unsteered.value:
        torch.manual_seed(int(seed.value))
        with torch.no_grad():
            set_unsteered_samples(score_model.sample(int(n_steps.value), reference_data))
    return


@app.cell
def _(fk_samples):
    fk_final = None
    if fk_samples is not None:
        if fk_samples.ndim == 1:
            fk_final = fk_samples.unsqueeze(0).unsqueeze(1)
        elif fk_samples.ndim == 2:
            fk_final = fk_samples
        elif fk_samples.ndim == 3:
            fk_final = fk_samples[-1]
        else:
            raise ValueError(f"Unexpected FK sample shape: {fk_samples.shape}")
    fk_final = fk_final.detach().cpu() if fk_final is not None else None
    return (fk_final,)


@app.cell
def _(das_samples):
    das_final = None
    if das_samples is not None:
        if das_samples.ndim != 2 or das_samples.shape[-1] != 1:
            raise ValueError(f"Unexpected DAS sample shape: {das_samples.shape}")
        das_final = das_samples
    das_final = das_final.detach().cpu() if das_final is not None else None
    return (das_final,)


@app.cell
def _(rpcn_samples):
    rpcn_final = None
    if rpcn_samples is not None:
        if rpcn_samples.ndim == 2:
            rpcn_final = rpcn_samples
        elif rpcn_samples.ndim == 3:
            rpcn_final = rpcn_samples[-1]
        else:
            raise ValueError(f"Unexpected RpCN sample shape: {rpcn_samples.shape}")
    rpcn_final = rpcn_final.detach().cpu() if rpcn_final is not None else None
    return (rpcn_final,)


@app.cell
def _(crepe_samples):
    crepe_final = None
    if crepe_samples is not None:
        if crepe_samples.ndim != 3 or crepe_samples.shape[-1] != 1:
            raise ValueError(f"Unexpected CREPE sample shape: {crepe_samples.shape}")
        crepe_final = crepe_samples.reshape(-1, 1)
    crepe_final = crepe_final.detach().cpu() if crepe_final is not None else None
    return (crepe_final,)


@app.cell
def _(mh_samples):
    mh_final = None
    if mh_samples is not None:
        if mh_samples.ndim != 3 or mh_samples.shape[-1] != 1:
            raise ValueError(f"Unexpected MH sample shape: {mh_samples.shape}")
        mh_final = mh_samples.reshape(-1, 1)
    mh_final = mh_final.detach().cpu() if mh_final is not None else None
    return (mh_final,)


@app.cell
def _(
    crepe_diagnostics,
    crepe_final,
    crepe_is_stale,
    das_final,
    fk_final,
    mh_diagnostics,
    mh_final,
    mo,
    rpcn_final,
    unsteered_samples,
):
    statuses = []
    statuses.append("Unsteered: ready" if unsteered_samples is not None else "Unsteered: not run")
    statuses.append("FK: ready" if fk_final is not None else "FK: not run")
    statuses.append("DAS: ready" if das_final is not None else "DAS: not run")
    statuses.append("RpCN: ready" if rpcn_final is not None else "RpCN: not run")
    if crepe_is_stale:
        statuses.append("CREPE: stale")
    else:
        statuses.append("CREPE: ready" if crepe_final is not None else "CREPE: not run")
    statuses.append("MH: ready" if mh_final is not None else "MH: not run")
    if crepe_diagnostics is not None and not crepe_is_stale:
        statuses.append(f"CREPE accept={crepe_diagnostics['acceptance_rate_mean']:.3f}")
    if mh_diagnostics is not None:
        statuses.append(f"MH accept={mh_diagnostics['acceptance_rate']:.3f}")
    status_view = mo.vstack([mo.md("## Sampler Status"), mo.md(" | ".join(statuses))])
    status_view
    return


@app.cell
def _(
    crepe_final,
    das_final,
    fk_final,
    mh_final,
    mo,
    rpcn_final,
    unsteered_samples,
):
    debug_rows = []
    if unsteered_samples is not None:
        debug_rows.append({"sampler": "Unsteered", "shape": str(tuple(unsteered_samples.shape))})
    if fk_final is not None:
        debug_rows.append({"sampler": "FK", "shape": str(tuple(fk_final.shape))})
    if das_final is not None:
        debug_rows.append({"sampler": "DAS", "shape": str(tuple(das_final.shape))})
    if rpcn_final is not None:
        debug_rows.append({"sampler": "RpCN", "shape": str(tuple(rpcn_final.shape))})
    if crepe_final is not None:
        debug_rows.append({"sampler": "CREPE", "shape": str(tuple(crepe_final.shape))})
    if mh_final is not None:
        debug_rows.append({"sampler": "MH", "shape": str(tuple(mh_final.shape))})
    if debug_rows:
        shape_view = mo.vstack([mo.md("## Output Shapes"), mo.ui.table(debug_rows)])
    else:
        shape_view = mo.md("No sampler outputs yet.")
    shape_view
    return


if __name__ == "__main__":
    app.run()
