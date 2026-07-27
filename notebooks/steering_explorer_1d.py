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
        toy,
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
        aspect=10 / 3.5,
        color="tab:red",
        linewidth=2.0,
    )
    reward_ax = reward_plot.axes.flat[0]
    reward_ax.set_title("Reward Preview")
    reward_ax.set_xlabel("x")
    reward_ax.set_ylabel("reward")
    reward_ax.grid(alpha=0.2)
    reward_plot.figure.tight_layout()
    reward_view = mo.vstack([mo.md("## Reward Preview"), mo.as_html(reward_plot.figure)])
    reward_view
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
                ylim=(0, gmm_density.max() * 1.504),
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
    return (plot_histogram,)


@app.cell
def _(plot_histogram, toy):
    if plot_histogram is not None:
        toy.plots.save_all_subfigures(plot_histogram, 'Histogram')
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
    fk_temperature_count = mo.ui.number(start=1, stop=64, step=1, value=1, label="fk_temperature_count")
    fk_resample_threshold = mo.ui.number(start=0.05, stop=0.95, step=0.05, value=0.5, label="fk_ess_threshold")
    fk_check_every = mo.ui.number(start=1, stop=100, step=1, value=1, label="fk_check_every")

    das_n_particles = mo.ui.number(start=2, stop=128, step=1, value=20, label="das_n_particles")
    das_inverse_temperature = mo.ui.number(start=0.00, step=0.05, value=25.0, label="das_inverse_temperature")
    das_proposal_scale = mo.ui.number(start=0.0, stop=1.0, step=0.01, value=1.0, label="das_proposal_scale")
    das_tempering_gamma = mo.ui.number(start=0.0, stop=1.0, value=0.003, label="das_tempering_gamma")
    das_resample_threshold = mo.ui.number(start=0.05, stop=0.95, step=0.05, value=0.5, label="das_ess_threshold")

    rpcn_n_chains = mo.ui.number(start=2, stop=128, step=1, value=4, label="rpcn_n_chains")
    rpcn_update_count = mo.ui.number(start=1, stop=200, step=1, value=5, label="rpcn_update_count")
    rpcn_angle_min = mo.ui.number(start=0.001, stop=1.0, step=0.001, value=0.05, label="rpcn_angle_min")
    rpcn_inverse_temp_max = mo.ui.number(start=0.00, step=0.05, value=25.0, label="rpcn_inverse_temp_max")
    rpcn_check_every = mo.ui.number(start=1, stop=100, step=1, value=1, label="rpcn_check_every")

    crepe_n_chains = mo.ui.number(start=1, stop=1024, step=1, value=64, label="crepe_n_chains")
    crepe_num_samples = mo.ui.number(start=1, stop=1000, step=1, value=16, label="crepe_num_samples")
    crepe_burn_in = mo.ui.number(start=0, stop=10000, step=2, value=500, label="crepe_burn_in")
    crepe_thinning = mo.ui.number(start=2, stop=100, step=2, value=2, label="crepe_thinning")
    crepe_inverse_temp = mo.ui.number(start=0.00, step=0.05, value=25.0, label="crepe_inverse_temp")
    crepe_reward_power = mo.ui.number(start=0.5, stop=20.0, step=0.5, value=5.0, label="crepe_reward_rho")
    crepe_refresh_reference = mo.ui.checkbox(value=True, label="crepe_refresh_reference")
    crepe_replica_count = mo.ui.number(start=4, stop=1000, step=1, value=50, label="crepe_replica_count")
    crepe_pt_min_time = mo.ui.number(start=1e-6, stop=0.1, step=1e-4, value=1e-4, label="crepe_pt_min_time")
    crepe_denoise_steps = mo.ui.number(start=1, stop=1000, step=1, value=32, label="crepe_denoise_steps")
    crepe_karras_rho = mo.ui.number(start=1.0, stop=20.0, step=0.5, value=7.0, label="crepe_karras_rho")

    mh_num_chains = mo.ui.number(start=1, stop=256, step=1, value=64, label="mh_num_chains")
    mh_num_samples = mo.ui.number(start=10, stop=10000, step=10, value=2000, label="mh_num_samples")
    mh_burn_in = mo.ui.number(start=0, stop=20000, step=50, value=2000, label="mh_burn_in")
    mh_thinning = mo.ui.number(start=1, stop=50, step=1, value=2, label="mh_thinning")
    mh_proposal_std = mo.ui.number(start=0.01, stop=10.0, step=0.01, value=0.4, label="mh_proposal_std")
    mh_inverse_temp = mo.ui.number(start=0.00, step=0.05, value=25.0, label="mh_inverse_temp")

    run_unsteered = mo.ui.run_button(label="Unsteered")
    run_fk = mo.ui.run_button(label="FK")
    run_das = mo.ui.run_button(label="DAS")
    run_rpcn = mo.ui.run_button(label="RpCN")
    run_crepe = mo.ui.run_button(label="CREPE")
    run_mh = mo.ui.run_button(label="MH")

    controls = mo.vstack(
        [
            mo.md("## Experiment Controls"),
            mo.hstack([seed, device, batch_size, n_steps, n_reference, run_unsteered, run_fk, run_das, run_rpcn, run_crepe], justify="start"),
            mo.hstack([epsilon_alpha, epsilon_beta, vmap_chunk_size, sampling_max_score_norm], justify="start"),
            mo.hstack([x_min, x_max, n_grid], justify="start"),
            mo.hstack([fk_n_chains, fk_inverse_temperature, fk_temperature_count, fk_resample_threshold, fk_check_every], justify="start"),
            mo.hstack([das_n_particles, das_inverse_temperature, das_proposal_scale, das_tempering_gamma, das_resample_threshold], justify="start"),
            mo.md("DAS uses the paper's exponential schedule `lambda_i=min((1+gamma)^i-1, 1)` and enforces terminal `lambda=1`. Set `das_proposal_scale=1` for the undamped Gaussian proposal; smaller values retain the exact importance correction while improving finite-particle stability for this non-concave reward."),
            mo.hstack([rpcn_n_chains, rpcn_update_count, rpcn_angle_min, rpcn_inverse_temp_max, rpcn_check_every], justify="start"),
            mo.hstack([crepe_n_chains, crepe_num_samples, crepe_burn_in, crepe_thinning, crepe_inverse_temp, crepe_reward_power, crepe_refresh_reference], justify="start"),
            mo.hstack([crepe_replica_count, crepe_pt_min_time, crepe_denoise_steps, crepe_karras_rho], justify="start"),
            mo.md("CREPE follows the paper's Karras schedule and runs PT only on `[crepe_pt_min_time, 1]`; retained states are then reverse-diffused to `t=0`. Each iteration is one checkerboard half-sweep. Retained output count is `crepe_num_samples * crepe_n_chains`."),
            mo.hstack([mh_num_chains, mh_num_samples, mh_burn_in, mh_thinning, mh_proposal_std, mh_inverse_temp, run_mh], justify="start"),
        ]
    )
    controls
    return (
        batch_size,
        crepe_burn_in,
        crepe_denoise_steps,
        crepe_inverse_temp,
        crepe_karras_rho,
        crepe_n_chains,
        crepe_num_samples,
        crepe_pt_min_time,
        crepe_refresh_reference,
        crepe_replica_count,
        crepe_reward_power,
        crepe_thinning,
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
        fk_resample_threshold,
        fk_temperature_count,
        mh_burn_in,
        mh_inverse_temp,
        mh_num_chains,
        mh_num_samples,
        mh_proposal_std,
        mh_thinning,
        n_grid,
        n_reference,
        n_steps,
        rpcn_angle_min,
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
    das_final,
    fk_final,
    hist_bin_edges,
    hist_range,
    mh_final,
    mo,
    np,
    pl,
    reward_fn,
    rpcn_final,
    torch,
    unsteered_samples,
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

        kl_by_method = {}
        if mh_final is not None:
            eps = 1e-8
            mh_x = mh_final[:, 0].cpu().numpy()
            mh_counts, _ = np.histogram(mh_x, bins=hist_bin_edges)
            p_mh = (mh_counts.astype(np.float64) + eps)
            p_mh = p_mh / p_mh.sum()

            for stats_method_name in summary["method"].to_list():
                if stats_method_name == "MH":
                    kl_by_method[stats_method_name] = 0.0
                    continue
                method_x = stats_df.filter(pl.col("method") == stats_method_name)["x"].to_numpy()
                method_counts, _ = np.histogram(method_x, bins=hist_bin_edges)
                p_method = (method_counts.astype(np.float64) + eps)
                p_method = p_method / p_method.sum()
                kl_by_method[stats_method_name] = float(np.sum(p_mh * (np.log(p_mh) - np.log(p_method))))
        else:
            for stats_method_name in summary["method"].to_list():
                kl_by_method[stats_method_name] = None

        summary = summary.with_columns(
            pl.Series("kl_to_mh", [kl_by_method[m] for m in summary["method"].to_list()])
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

        mh_diag_table = [
            {"metric": "acceptance_rate", "value": f"{acceptance_rate:.4f}"},
            {"metric": "accepted", "value": int(mh_diagnostics["accepted"])},
            {"metric": "proposed", "value": int(mh_diagnostics["proposed"])},
            {"metric": "num_chains", "value": int(mh_diagnostics["num_chains"])},
            {"metric": "num_samples_per_chain", "value": int(mh_diagnostics["num_samples"])},
            {"metric": "burn_in", "value": int(mh_diagnostics["burn_in"])},
            {"metric": "thinning", "value": int(mh_diagnostics["thinning"])},
            {"metric": "proposal_std", "value": float(mh_diagnostics["proposal_std"])},
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
        crepe_diag_view = mo.vstack([mo.md("## CREPE Diagnostics"), mo.md("Run CREPE to populate diagnostics.")])
    else:
        if crepe_is_stale:
            crepe_status_label = "Stale"
            crepe_status_hint = "Controls changed after this run. Run CREPE again before comparing samples."
        elif crepe_diagnostics["cold_edge_acceptance_rate"] < 0.05:
            crepe_status_label = "Cold-edge bottleneck"
            crepe_status_hint = "Replica 0 is barely moving. Increase burn-in, reduce inverse temperature, or refine the reward ladder."
        elif crepe_diagnostics["minimum_edge_acceptance_rate"] < 0.05:
            crepe_status_label = "Ladder bottleneck"
            crepe_status_hint = f"Communication is blocked near edge {crepe_diagnostics['bottleneck_edge']}. Increase the replica count or reduce inverse temperature."
        elif crepe_diagnostics["chains_without_cold_acceptance_fraction"] > 0.25:
            crepe_status_label = "Under-mixed"
            crepe_status_hint = "Many chains never accepted a target-edge proposal. Increase burn-in before interpreting the histogram."
        else:
            crepe_status_label = "Communicating"
            crepe_status_hint = "Cold-edge communication is active; confirm stability across longer burn-in runs."
        crepe_diag_table = [
            {"metric": "acceptance_rate_mean", "value": f"{crepe_diagnostics['acceptance_rate_mean']:.4f}"},
            {"metric": "cold_edge_acceptance_rate", "value": f"{crepe_diagnostics['cold_edge_acceptance_rate']:.4f}"},
            {"metric": "minimum_edge_acceptance_rate", "value": f"{crepe_diagnostics['minimum_edge_acceptance_rate']:.4f}"},
            {"metric": "bottleneck_edge", "value": str(crepe_diagnostics["bottleneck_edge"])},
            {"metric": "chains_without_cold_acceptance", "value": f"{crepe_diagnostics['chains_without_cold_acceptance_fraction']:.4f}"},
            {"metric": "retained_duplicate_fraction", "value": f"{crepe_diagnostics['duplicate_fraction']:.4f}"},
            {"metric": "accepted", "value": int(crepe_diagnostics["accepted"])},
            {"metric": "attempted", "value": int(crepe_diagnostics["attempted"])},
            {"metric": "iterations_half_sweeps", "value": int(crepe_diagnostics["iteration_count"])},
            {"metric": "full_sweep_equivalents", "value": float(crepe_diagnostics["full_sweep_equivalents"])},
            {"metric": "saved_iterations", "value": int(crepe_diagnostics["saved_iterations"])},
            {"metric": "chain_count", "value": int(crepe_diagnostics["chain_count"])},
            {"metric": "replica_count", "value": int(crepe_diagnostics["replica_count"])},
            {"metric": "pt_min_time", "value": crepe_diagnostics["pt_min_time"]},
            {"metric": "karras_rho", "value": crepe_diagnostics["karras_rho"]},
            {"metric": "final_denoise_steps", "value": int(crepe_diagnostics["final_denoise_steps"])},
            {"metric": "minimum_time_step", "value": crepe_diagnostics["minimum_time_step"]},
            {"metric": "maximum_time_step", "value": crepe_diagnostics["maximum_time_step"]},
        ]
        crepe_edge_table = [
            {
                "edge": f"({edge_index}, {edge_index + 1})",
                "acceptance_rate": edge_rate,
                "accepted": edge_accepted,
                "attempted": edge_attempted,
            }
            for edge_index, (edge_rate, edge_accepted, edge_attempted) in enumerate(
                zip(
                    crepe_diagnostics["acceptance_rate_by_edge"],
                    crepe_diagnostics["accepted_by_edge"],
                    crepe_diagnostics["attempted_by_edge"],
                )
            )
        ]
        crepe_diag_view = mo.vstack(
            [
                mo.md("## CREPE Diagnostics"),
                mo.md(f"**Status:** {crepe_status_label}"),
                mo.ui.table(crepe_diag_table),
                mo.md(crepe_status_hint),
                mo.md("### Communication By Edge"),
                mo.ui.table(crepe_edge_table),
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
            return (step_idx >= 5) and (step_idx % int(rpcn_check_every.value) == 0)

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
    crepe,
    crepe_burn_in,
    crepe_config_signature,
    crepe_denoise_steps,
    crepe_inverse_temp,
    crepe_karras_rho,
    crepe_n_chains,
    crepe_num_samples,
    crepe_pt_min_time,
    crepe_refresh_reference,
    crepe_replica_count,
    crepe_reward_power,
    crepe_thinning,
    functools,
    reference_data,
    reward_fn,
    run_crepe,
    score_model,
    seed,
    set_crepe_diagnostics,
    set_crepe_samples,
    timesteps,
    torch,
):
    if run_crepe.value:
        torch.manual_seed(int(seed.value))
        crepe_pt_min_time_value = float(crepe_pt_min_time.value)
        crepe_karras_rho_value = float(crepe_karras_rho.value)
        crepe_karras_fraction = torch.linspace(
            0.0,
            1.0,
            int(crepe_replica_count.value),
            device=timesteps.device,
            dtype=torch.float64,
        )
        crepe_descending_times = (
            1.0
            + crepe_karras_fraction
            * (crepe_pt_min_time_value ** (1.0 / crepe_karras_rho_value) - 1.0)
        ).pow(crepe_karras_rho_value)
        crepe_times = crepe_descending_times.flip(0).to(dtype=timesteps.dtype)
        crepe_inverse_temperature_value = float(crepe_inverse_temp.value)
        crepe_reward_power_value = float(crepe_reward_power.value)

        def crepe_diffusion(crepe_time):
            return score_model.prob_path.g(crepe_time)

        def crepe_model_forward_drift(crepe_time, crepe_batch):
            return score_model.prob_path.f(crepe_time, crepe_batch)

        def crepe_model_backward_drift(crepe_time, crepe_batch):
            crepe_sigma = crepe_diffusion(crepe_time)
            crepe_score = score_model._score_particles(crepe_time, crepe_batch)
            return crepe_model_forward_drift(crepe_time, crepe_batch) - crepe_sigma.square() * crepe_score

        def crepe_proposal_forward_drift(crepe_time, crepe_batch):
            return crepe_model_backward_drift(crepe_time, crepe_batch)

        def crepe_proposal_backward_drift(crepe_time, crepe_batch):
            return crepe_model_forward_drift(crepe_time, crepe_batch)

        def crepe_annealed_reward(crepe_time, crepe_batch):
            crepe_predicted_clean = score_model.get_tweedie(crepe_time, crepe_batch)
            crepe_level_progress = (
                crepe_time.pow(1.0 / crepe_karras_rho_value) - 1.0
            ) / (
                crepe_pt_min_time_value ** (1.0 / crepe_karras_rho_value) - 1.0
            )
            crepe_schedule = crepe_level_progress.clamp(0.0, 1.0).pow(
                crepe_reward_power_value
            )
            return crepe_inverse_temperature_value * crepe_schedule * reward_fn(crepe_predicted_clean)

        def crepe_log_gaussian(crepe_value, crepe_mean, crepe_std):
            crepe_variance = crepe_std.square()
            crepe_event_size = crepe_value.shape[-1]
            return -0.5 * (
                (crepe_value - crepe_mean).square().sum(dim=-1) / crepe_variance
                + crepe_event_size * torch.log(2.0 * torch.pi * crepe_variance)
            )

        def crepe_summarize_path(crepe_t0, crepe_t1, crepe_less_noisy, crepe_more_noisy):
            crepe_dt = crepe_t1 - crepe_t0
            crepe_proposal_forward_mean = (
                crepe_less_noisy
                + crepe_dt
                * crepe_proposal_forward_drift(crepe_t0, crepe_less_noisy)
            )
            crepe_proposal_backward_mean = (
                crepe_more_noisy
                - crepe_dt
                * crepe_proposal_backward_drift(crepe_t1, crepe_more_noisy)
            )
            crepe_proposal_forward_log_probability = crepe_log_gaussian(
                crepe_more_noisy,
                crepe_proposal_forward_mean,
                crepe_dt.sqrt() * crepe_diffusion(crepe_t0),
            )
            crepe_proposal_backward_log_probability = crepe_log_gaussian(
                crepe_less_noisy,
                crepe_proposal_backward_mean,
                crepe_dt.sqrt() * crepe_diffusion(crepe_t1),
            )
            crepe_proposal_log_rne = crepe.PathLogRatio.from_transition_log_probabilities(
                crepe_proposal_forward_log_probability.unsqueeze(0),
                crepe_proposal_backward_log_probability.unsqueeze(0),
            )
            crepe_model_forward_mean = (
                crepe_less_noisy
                + crepe_dt * crepe_model_forward_drift(crepe_t0, crepe_less_noisy)
            )
            crepe_model_backward_mean = (
                crepe_more_noisy
                - crepe_dt * crepe_model_backward_drift(crepe_t1, crepe_more_noisy)
            )
            crepe_model_forward_log_probability = crepe_log_gaussian(
                crepe_more_noisy,
                crepe_model_forward_mean,
                crepe_dt.sqrt() * crepe_diffusion(crepe_t0),
            )
            crepe_model_backward_log_probability = crepe_log_gaussian(
                crepe_less_noisy,
                crepe_model_backward_mean,
                crepe_dt.sqrt() * crepe_diffusion(crepe_t1),
            )
            crepe_model_log_rne = crepe.PathLogRatio.from_transition_log_probabilities(
                crepe_model_forward_log_probability.unsqueeze(0),
                crepe_model_backward_log_probability.unsqueeze(0),
            )
            crepe_target_log_ratio = crepe.TargetLogRatio.reward_tilt(
                crepe_model_log_rne,
                crepe_annealed_reward(crepe_t0, crepe_less_noisy),
                crepe_annealed_reward(crepe_t1, crepe_more_noisy),
            )
            return crepe_target_log_ratio, crepe_proposal_log_rne

        def crepe_forward_path_fn(crepe_t0, crepe_t1, crepe_less_noisy):
            crepe_dt = crepe_t1 - crepe_t0
            crepe_more_noisy = (
                crepe_less_noisy
                + crepe_dt * crepe_proposal_forward_drift(crepe_t0, crepe_less_noisy)
                + crepe_dt.sqrt() * crepe_diffusion(crepe_t0) * torch.randn_like(crepe_less_noisy)
            )
            crepe_target_log_ratio, crepe_proposal_log_rne = crepe_summarize_path(
                crepe_t0, crepe_t1, crepe_less_noisy, crepe_more_noisy
            )
            return crepe_more_noisy, crepe_target_log_ratio, crepe_proposal_log_rne

        def crepe_backward_path_fn(crepe_t0, crepe_t1, crepe_more_noisy):
            crepe_dt = crepe_t1 - crepe_t0
            crepe_less_noisy = (
                crepe_more_noisy
                - crepe_dt * crepe_proposal_backward_drift(crepe_t1, crepe_more_noisy)
                + crepe_dt.sqrt() * crepe_diffusion(crepe_t1) * torch.randn_like(crepe_more_noisy)
            )
            crepe_target_log_ratio, crepe_proposal_log_rne = crepe_summarize_path(
                crepe_t0, crepe_t1, crepe_less_noisy, crepe_more_noisy
            )
            return crepe_less_noisy, crepe_target_log_ratio, crepe_proposal_log_rne

        def crepe_sample_model_marginal(crepe_time):
            crepe_reference_index = torch.randint(
                reference_data.shape[0], (), device=reference_data.device
            )
            crepe_reference_weights = torch.nn.functional.one_hot(
                crepe_reference_index, reference_data.shape[0]
            ).to(dtype=reference_data.dtype)
            crepe_datum = (crepe_reference_weights[:, None] * reference_data).sum(dim=0)
            return (
                score_model.prob_path.mean(crepe_time, crepe_datum)
                + score_model.prob_path.std(crepe_time, crepe_datum) * torch.randn_like(crepe_datum)
            )

        def crepe_init_path_fn(crepe_ladder_times):
            crepe_descending_times = crepe_ladder_times.flip(0)
            crepe_state = crepe_sample_model_marginal(crepe_descending_times[0])
            crepe_descending_states = [crepe_state]
            for crepe_time_pair in crepe_descending_times.unfold(0, 2, 1):
                crepe_more_noisy_time = crepe_time_pair[0]
                crepe_less_noisy_time = crepe_time_pair[1]
                crepe_init_dt = crepe_less_noisy_time - crepe_more_noisy_time
                crepe_state_batch = crepe_state.unsqueeze(0)
                crepe_state = (
                    crepe_state
                    + crepe_init_dt
                    * crepe_model_backward_drift(crepe_more_noisy_time, crepe_state_batch)[0]
                    + crepe_init_dt.abs().sqrt()
                    * crepe_diffusion(crepe_more_noisy_time)
                    * torch.randn_like(crepe_state)
                )
                crepe_descending_states.append(crepe_state)
            return torch.stack(crepe_descending_states[::-1])

        crepe_sampler = crepe.CREPE(
            init_path_fn=crepe_init_path_fn,
            forward_path_fn=crepe_forward_path_fn,
            backward_path_fn=crepe_backward_path_fn,
            refresh_reference_fn=(
                crepe_sample_model_marginal if crepe_refresh_reference.value else None
            ),
        )
        crepe_iteration_count = (
            int(crepe_burn_in.value)
            + 1
            + (int(crepe_num_samples.value) - 1) * int(crepe_thinning.value)
        )
        with torch.no_grad():
            crepe_result = crepe_sampler.run(
                crepe_times,
                crepe_iteration_count,
                batch_size=int(crepe_n_chains.value),
                should_save_sample_fn=functools.partial(
                    crepe.SaveSampleFn.burn_in_and_thin,
                    burn_in=int(crepe_burn_in.value),
                    thin=int(crepe_thinning.value),
                ),
            )
            crepe_final_denoise_fraction = torch.linspace(
                0.0,
                1.0,
                int(crepe_denoise_steps.value) + 1,
                device=crepe_times.device,
                dtype=crepe_times.dtype,
            )
            crepe_final_denoise_times = (
                crepe_pt_min_time_value ** (1.0 / crepe_karras_rho_value)
                * (1.0 - crepe_final_denoise_fraction)
            ).pow(crepe_karras_rho_value)
            crepe_final_samples = crepe_result.samples.reshape(-1, *crepe_result.samples.shape[2:])
            for crepe_denoise_time_pair in crepe_final_denoise_times.unfold(0, 2, 1):
                crepe_denoise_t0 = crepe_denoise_time_pair[0]
                crepe_denoise_t1 = crepe_denoise_time_pair[1]
                crepe_denoise_dt = crepe_denoise_t1 - crepe_denoise_t0
                crepe_final_samples = (
                    crepe_final_samples
                    + crepe_denoise_dt
                    * crepe_model_backward_drift(crepe_denoise_t0, crepe_final_samples)
                    + crepe_denoise_dt.abs().sqrt()
                    * crepe_diffusion(crepe_denoise_t0)
                    * torch.randn_like(crepe_final_samples)
                )
            crepe_final_samples = crepe_final_samples.reshape(crepe_result.samples.shape)
        if not torch.isfinite(crepe_final_samples).all():
            raise ValueError("CREPE final denoising produced non-finite samples")
        set_crepe_samples(crepe_final_samples)
        crepe_accepted_by_edge = crepe_result.acceptance_count.sum(dim=1)
        crepe_attempted_by_edge = crepe_result.attempt_count.sum(dim=1)
        crepe_acceptance_by_edge = (
            crepe_accepted_by_edge / crepe_attempted_by_edge.clamp(min=1)
        )
        crepe_cold_acceptance_by_chain = crepe_result.acceptance_count[0]
        if crepe_result.samples.shape[0] > 1:
            crepe_duplicate_fraction = float(
                (crepe_result.samples[1:] == crepe_result.samples[:-1])
                .flatten(start_dim=2)
                .all(dim=2)
                .to(torch.float32)
                .mean()
                .item()
            )
        else:
            crepe_duplicate_fraction = 0.0
        set_crepe_diagnostics(
            {
                "config_signature": crepe_config_signature,
                "acceptance_rate_mean": float(crepe_result.acceptance_rate.mean().item()),
                "acceptance_rate_min": float(crepe_result.acceptance_rate.min().item()),
                "acceptance_rate_by_edge": crepe_acceptance_by_edge.cpu().tolist(),
                "minimum_edge_acceptance_rate": float(crepe_acceptance_by_edge.min().item()),
                "bottleneck_edge": int(crepe_acceptance_by_edge.argmin().item()),
                "accepted": int(crepe_result.acceptance_count.sum().item()),
                "attempted": int(crepe_result.attempt_count.sum().item()),
                "accepted_by_edge": crepe_accepted_by_edge.cpu().tolist(),
                "attempted_by_edge": crepe_attempted_by_edge.cpu().tolist(),
                "cold_edge_acceptance_rate": float(crepe_acceptance_by_edge[0].item()),
                "cold_edge_accepted": int(crepe_accepted_by_edge[0].item()),
                "cold_edge_attempted": int(crepe_attempted_by_edge[0].item()),
                "chains_without_cold_acceptance_fraction": float(
                    (crepe_cold_acceptance_by_chain == 0)
                    .to(torch.float32)
                    .mean()
                    .item()
                ),
                "duplicate_fraction": crepe_duplicate_fraction,
                "iteration_count": crepe_iteration_count,
                "full_sweep_equivalents": crepe_iteration_count / 2,
                "saved_iterations": int(crepe_result.samples.shape[0]),
                "chain_count": int(crepe_result.samples.shape[1]),
                "replica_count": int(crepe_times.numel()),
                "pt_min_time": crepe_pt_min_time_value,
                "karras_rho": crepe_karras_rho_value,
                "final_denoise_steps": int(crepe_denoise_steps.value),
                "minimum_time_step": float((crepe_times[1:] - crepe_times[:-1]).min().item()),
                "maximum_time_step": float((crepe_times[1:] - crepe_times[:-1]).max().item()),
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
    crepe_burn_in,
    crepe_denoise_steps,
    crepe_inverse_temp,
    crepe_karras_rho,
    crepe_n_chains,
    crepe_num_samples,
    crepe_pt_min_time,
    crepe_refresh_reference,
    crepe_replica_count,
    crepe_reward_power,
    crepe_thinning,
    device,
    epsilon_alpha,
    epsilon_beta,
    means,
    n_reference,
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
        int(crepe_n_chains.value),
        int(crepe_num_samples.value),
        int(crepe_burn_in.value),
        int(crepe_thinning.value),
        float(crepe_inverse_temp.value),
        float(crepe_reward_power.value),
        bool(crepe_refresh_reference.value),
        int(crepe_replica_count.value),
        float(crepe_pt_min_time.value),
        int(crepe_denoise_steps.value),
        float(crepe_karras_rho.value),
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
            potential=fk_steering.Potential.DIFFERENCE,
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
