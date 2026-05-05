import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full")


@app.cell
def _():
    import math
    import numpy as np

    import marimo as mo
    import polars as pl
    import seaborn as sns
    sns.set_theme(style="white", font_scale=2)
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
        fk_steering,
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
    unsteered, FK, RpCN, and MH reference samplers independently.
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

    _g = sns.relplot(
        data=gmm_df.to_dict(as_series=False),
        x="x",
        y="density",
        kind="line",
        height=3.5,
        aspect=2.8,
        color="tab:red",
        linewidth=2.0,
    )
    gmm_ax = _g.axes.flat[0]
    zero_y = [0.0] * len(modes_df)
    sns.scatterplot(x=modes_df["mean"].to_numpy(), y=zero_y, ax=gmm_ax, color="#1971c2", s=60, label="mode means")
    for row in modes_df.iter_rows(named=True):
        gmm_ax.text(float(row["mean"]), 0.0, f"w={row['weight']:.2f}, s={row['std']:.2f}", fontsize=8, ha="center", va="bottom")
    gmm_ax.set_title("GMM Preview")
    gmm_ax.set_xlabel("x")
    gmm_ax.set_ylabel("density")
    gmm_ax.grid(alpha=0.2)
    gmm_ax.legend(loc="upper right")
    _g.figure.tight_layout()
    gmm_view = mo.vstack([mo.md("## GMM Preview"), mo.as_html(_g.figure)])
    gmm_view
    return


@app.cell
def _(mo):
    n_modes = mo.ui.number(start=1, stop=8, step=1, value=1, label="n_modes")
    return (n_modes,)


@app.cell
def _(mo, n_modes):
    mode_count = int(n_modes.value)
    default_std = 3
    if mode_count % 2 == 0:
        mean_inits_half = [i - .5 for i in range(1, (mode_count // 2) + 1)]
    else:
        mean_inits_half = [i for i in range((mode_count + 1) // 2)]
    mean_inits = sorted(set([-i for i in mean_inits_half] + mean_inits_half))
    print(mean_inits)
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
    _g = sns.relplot(
        data=reward_df.to_dict(as_series=False),
        x="x",
        y="reward",
        kind="line",
        height=3.5,
        aspect=10 / 3.5,
        color="tab:red",
        linewidth=2.0,
    )
    reward_ax = _g.axes.flat[0]
    reward_ax.set_title("Reward Preview")
    reward_ax.set_xlabel("x")
    reward_ax.set_ylabel("reward")
    reward_ax.grid(alpha=0.2)
    _g.figure.tight_layout()
    reward_view = mo.vstack([mo.md("## Reward Preview"), mo.as_html(_g.figure)])
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
        r = gauss_low_high(x, 3) + gauss_low_high(-x, 3)
        return r.squeeze()
    
    # def reward_fn(x: torch.Tensor) -> torch.Tensor:
    #     q = 0
    #     # for i, m in enumerate((3, 6, 9)):
    #     for i, m in enumerate((3,)):
    #         q = q + (-reward_gauss(x + m, s=1) * 2 + reward_gauss(x - m) * 1.6615) * (i + 1)
    #         q = q + reward_gauss(x + 4) * 1.6615
    #         q = q - 2 * reward_gauss(x - 2, s=1)
    #         # q = q + reward_gauss(-x, 6)
    #     r = q
    #     # r = torch.exp(-0.5 * ((x - 3.0) / 0.8) ** 2) - 0.35 * torch.exp(-0.5 * ((x + 2.0) / 0.7) ** 2)
    #     # r = 1e6 * x**2
    #     # r = 1000 * r
    #     return r.squeeze()
    return (reward_fn,)


@app.cell
def _(np, x_max, x_min):
    hist_bin_count = 60
    hist_range = (float(x_min.value), float(x_max.value))
    hist_bin_edges = np.linspace(hist_range[0], hist_range[1], hist_bin_count + 1)
    return hist_bin_edges, hist_range


@app.cell(hide_code=True)
def _(
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
    has_samples_plot = not (unsteered_samples is None and fk_final is None and rpcn_final is None and mh_final is None)
    plot_histogram = None
    plot_bin_count = len(hist_bin_edges) - 1
    if has_samples_plot:
        # method_order = ["Base", "MCMC Reference", "FK Steering", "RpCN Steering"]
        method_order = ["MCMC Reference", "FK Steering", "RpCN Steering"]
        sample_frames = []
        if unsteered_samples is not None:
            sample_frames.append(
                pl.DataFrame({"method": ["Base"] * unsteered_samples.shape[0], "x": unsteered_samples[:, 0].cpu().numpy().tolist()})
            )
        if fk_final is not None:
            sample_frames.append(
                pl.DataFrame({"method": ["FK Steering"] * fk_final.shape[0], "x": fk_final[:, 0].cpu().numpy().tolist()})
            )
        if rpcn_final is not None:
            sample_frames.append(
                pl.DataFrame({"method": ["RpCN Steering"] * rpcn_final.shape[0], "x": rpcn_final[:, 0].cpu().numpy().tolist()})
            )
        if mh_final is not None:
            sample_frames.append(
                pl.DataFrame({"method": ["MCMC Reference"] * mh_final.shape[0], "x": mh_final[:, 0].cpu().numpy().tolist()})
            )
        samples_df = pl.concat(sample_frames) if sample_frames else pl.DataFrame({"method": [], "x": []})

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
        _g = (
            sns.displot(
                kind='hist',
                data=samples_df,
                x='x',
                hue='method',
                col='method',
                palette={
                    'Base': 'tab:grey',
                    'FK Steering': 'tab:blue',
                    'RpCN Steering': 'tab:orange',
                    'MCMC Reference': 'tab:grey',
                },
                alpha=.7,
                col_order=method_order,
                bins=hist_bin_edges,
                stat='density',
                element='step',
                common_norm=False,
                # aspect=.6,
                legend=False,
            )
            .set(
                ylim=(0, gmm_density.max() * 1.504),
                # yscale='log',
            )
            .set_titles('')
        )
        for (r, c, h), _ in _g.facet_data():
            ax = _g.axes[r, c]
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
            ax.set_xlabel('')
            ax.get_legend().remove()
        # _g.axes.flat[0].set_ylabel("Density")
        handles, legend_labels = _g.axes.flat[0].get_legend_handles_labels()
        _g.add_legend(dict(zip(legend_labels, handles)), loc="upper center", ncol=3,
                      bbox_to_anchor=(0.4, 1.04), 
                      frameon=True, fancybox=True)
        # _g.figure.tight_layout(rect=[0, 0, 1, 1])
        comparison_view = mo.vstack(
            [
                mo.md("## Sample Comparison"),
                mo.md(f"Histogram spec (shared with KL): bins={plot_bin_count}, range=[{hist_range[0]:.2f}, {hist_range[1]:.2f}]"),
                mo.as_html(_g),
            ]
        )
        plot_histogram = _g
    else:
        comparison_view = mo.md("Run one or more samplers to generate samples.")
    comparison_view
    return (plot_histogram,)


@app.cell
def _(plot_histogram, toy):
    if plot_histogram is not None:
        toy.plots.save_all_subfigures(plot_histogram, 'Histogram')
    return


@app.cell
def _():
    # comp_fig.savefig('Toy.pdf', format='pdf', bbox_inches='tight', pad_inches=0.06)
    return


@app.cell
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
    fk_resample_threshold = mo.ui.number(start=0.05, stop=0.95, step=0.05, value=0.5, label="fk_ess_threshold")
    fk_check_every = mo.ui.number(start=1, stop=100, step=1, value=1, label="fk_check_every")

    rpcn_n_chains = mo.ui.number(start=2, stop=128, step=1, value=4, label="rpcn_n_chains")
    rpcn_update_count = mo.ui.number(start=1, stop=200, step=1, value=5, label="rpcn_update_count")
    rpcn_angle_min = mo.ui.number(start=0.001, stop=1.0, step=0.001, value=0.05, label="rpcn_angle_min")
    rpcn_inverse_temp_max = mo.ui.number(start=0.00, step=0.05, value=25.0, label="rpcn_inverse_temp_max")
    rpcn_check_every = mo.ui.number(start=1, stop=100, step=1, value=1, label="rpcn_check_every")

    mh_num_chains = mo.ui.number(start=1, stop=256, step=1, value=64, label="mh_num_chains")
    mh_num_samples = mo.ui.number(start=10, stop=10000, step=10, value=2000, label="mh_num_samples")
    mh_burn_in = mo.ui.number(start=0, stop=20000, step=50, value=2000, label="mh_burn_in")
    mh_thinning = mo.ui.number(start=1, stop=50, step=1, value=2, label="mh_thinning")
    mh_proposal_std = mo.ui.number(start=0.01, stop=10.0, step=0.01, value=0.4, label="mh_proposal_std")
    mh_inverse_temp = mo.ui.number(start=0.00, step=0.05, value=25.0, label="mh_inverse_temp")

    run_unsteered = mo.ui.run_button(label="Unsteered")
    run_fk = mo.ui.run_button(label="FK")
    run_rpcn = mo.ui.run_button(label="RpCN")
    run_mh = mo.ui.run_button(label="MH")

    controls = mo.vstack(
        [
            mo.hstack([seed, device, batch_size, n_steps, n_reference, run_unsteered, run_fk, run_rpcn], justify="start"),
            mo.hstack([epsilon_alpha, epsilon_beta, vmap_chunk_size, sampling_max_score_norm], justify="start"),
            mo.hstack([x_min, x_max, n_grid], justify="start"),
            mo.hstack([fk_n_chains, fk_inverse_temperature, fk_resample_threshold, fk_check_every], justify="start"),
            mo.hstack([rpcn_n_chains, rpcn_update_count, rpcn_angle_min, rpcn_inverse_temp_max, rpcn_check_every], justify="start"),
            mo.hstack([mh_num_chains, mh_num_samples, mh_burn_in, mh_thinning, mh_proposal_std, mh_inverse_temp, run_mh], justify="start"),
        ]
    )
    mo.md("## Experiment Controls")
    controls
    return (
        batch_size,
        device,
        epsilon_alpha,
        epsilon_beta,
        fk_check_every,
        fk_inverse_temperature,
        fk_n_chains,
        fk_resample_threshold,
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


@app.cell
def _(
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
    has_samples_stats = not (unsteered_samples is None and fk_final is None and rpcn_final is None and mh_final is None)
    stats_bin_count = len(hist_bin_edges) - 1
    if has_samples_stats:
        method_frames = []
        if unsteered_samples is not None:
            method_frames.append(pl.DataFrame({"method": ["Unsteered"] * unsteered_samples.shape[0], "x": unsteered_samples[:, 0].cpu().numpy().tolist()}))
        if fk_final is not None:
            method_frames.append(pl.DataFrame({"method": ["FK"] * fk_final.shape[0], "x": fk_final[:, 0].cpu().numpy().tolist()}))
        if rpcn_final is not None:
            method_frames.append(pl.DataFrame({"method": ["RpCN"] * rpcn_final.shape[0], "x": rpcn_final[:, 0].cpu().numpy().tolist()}))
        if mh_final is not None:
            method_frames.append(pl.DataFrame({"method": ["MH"] * mh_final.shape[0], "x": mh_final[:, 0].cpu().numpy().tolist()}))
        stats_df = pl.concat(method_frames)

        reward_x = torch.tensor(stats_df["x"].to_numpy(), dtype=torch.float32)
        reward_vals = torch.vmap(reward_fn)(reward_x).detach().cpu().numpy().tolist()
        stats_df = stats_df.with_columns(pl.Series("reward", reward_vals))

        summary = (
            stats_df.group_by("method")
            .agg(
                pl.len().alias("count"),
                pl.col("reward").quantile(0.05).alias("q05"),
                pl.col("reward").quantile(0.50).alias("q50"),
                pl.col("reward").quantile(0.95).alias("q95"),
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

            for method_name in summary["method"].to_list():
                if method_name == "MH":
                    kl_by_method[method_name] = 0.0
                    continue
                method_x = stats_df.filter(pl.col("method") == method_name)["x"].to_numpy()
                method_counts, _ = np.histogram(method_x, bins=hist_bin_edges)
                p_method = (method_counts.astype(np.float64) + eps)
                p_method = p_method / p_method.sum()
                kl_by_method[method_name] = float(np.sum(p_mh * (np.log(p_mh) - np.log(p_method))))
        else:
            for method_name in summary["method"].to_list():
                kl_by_method[method_name] = None

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
            return step_idx % int(rpcn_check_every.value) == 0

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
            return gmm_log_prob(x_flat) + float(mh_inverse_temp.value) * reward_fn(x_flat)

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
    reward_values = torch.stack([torch.as_tensor(reward_fn(xi), dtype=torch.float32) for xi in x_grid])
    if reward_values.ndim != 1:
        raise ValueError("reward_fn must return scalar values for scalar tensor inputs")
    if not torch.isfinite(reward_values).all():
        raise ValueError("reward_fn returned non-finite values on preview grid")
    return (reward_values,)


@app.cell
def _(mo):
    get_unsteered_samples, set_unsteered_samples = mo.state(None)
    get_fk_samples, set_fk_samples = mo.state(None)
    get_rpcn_samples, set_rpcn_samples = mo.state(None)
    get_mh_samples, set_mh_samples = mo.state(None)
    get_mh_diagnostics, set_mh_diagnostics = mo.state(None)
    return (
        get_fk_samples,
        get_mh_diagnostics,
        get_mh_samples,
        get_rpcn_samples,
        get_unsteered_samples,
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

    # timesteps = torch.linspace(1.0, 0.0, int(n_steps.value), device=reference_data.device, dtype=reference_data.dtype)
    timesteps = score_model.prob_path.linspace_time(cfg.sampling_time_step_count, device=reference_data.device)
    return reference_data, score_model, timesteps


@app.cell
def _(
    get_fk_samples,
    get_mh_diagnostics,
    get_mh_samples,
    get_rpcn_samples,
    get_unsteered_samples,
):
    fk_samples = get_fk_samples()
    mh_diagnostics = get_mh_diagnostics()
    mh_samples = get_mh_samples()
    rpcn_samples = get_rpcn_samples()
    unsteered_samples = get_unsteered_samples()
    return (
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
        def propose_step(t0, t1, z):
            mean = score_model.expected_next_fn(t0, t1, z)
            if t1 > 0:
                std = score_model.std_fn(t0, t1)
                return mean + std * torch.randn_like(z)
            return mean

        def adaptive_resample_safe(time_step, ensemble, potentials):
            if time_step % int(fk_check_every.value) != 0:
                return ensemble
            weights = torch.nan_to_num(potentials, nan=0.0, posinf=0.0, neginf=0.0)
            weights = weights.clamp_min(0.0)
            tiny = 1e-9
            weights = torch.where(weights.sum() > tiny, weights, 1.)
            probs = weights / weights.sum()
            ess = 1.0 / probs.square().sum()
            threshold = float(fk_resample_threshold.value) * ensemble.shape[0]
            return torch.where(ess < threshold, fk_steering.ResampleFn.do_resample(ensemble, probs), ensemble)

        fk = fk_steering.FeynmannKac(
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
            inverse_temperature=float(fk_inverse_temperature.value),
            kwargs_intermediate_reward={"expected_sample_fn": score_model.get_tweedie},
        )

        with torch.no_grad():
            set_fk_samples(
                fk.run(
                    timesteps,
                    int(fk_n_chains.value),
                    batch_size=int(batch_size.value),
                )
            )
    return


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
def _(mh_samples):
    mh_final = None
    if mh_samples is not None:
        if mh_samples.ndim != 3 or mh_samples.shape[-1] != 1:
            raise ValueError(f"Unexpected MH sample shape: {mh_samples.shape}")
        mh_final = mh_samples.reshape(-1, 1)
    mh_final = mh_final.detach().cpu() if mh_final is not None else None
    return (mh_final,)


@app.cell
def _(fk_final, mh_diagnostics, mh_final, mo, rpcn_final, unsteered_samples):
    statuses = []
    statuses.append("Unsteered: ready" if unsteered_samples is not None else "Unsteered: not run")
    statuses.append("FK: ready" if fk_final is not None else "FK: not run")
    statuses.append("RpCN: ready" if rpcn_final is not None else "RpCN: not run")
    statuses.append("MH: ready" if mh_final is not None else "MH: not run")
    if mh_diagnostics is not None:
        statuses.append(f"MH accept={mh_diagnostics['acceptance_rate']:.3f}")
    status_view = mo.vstack([mo.md("## Sampler Status"), mo.md(" | ".join(statuses))])
    status_view
    return


@app.cell
def _(fk_final, mh_final, mo, rpcn_final, unsteered_samples):
    debug_rows = []
    if unsteered_samples is not None:
        debug_rows.append({"sampler": "Unsteered", "shape": str(tuple(unsteered_samples.shape))})
    if fk_final is not None:
        debug_rows.append({"sampler": "FK", "shape": str(tuple(fk_final.shape))})
    if rpcn_final is not None:
        debug_rows.append({"sampler": "RpCN", "shape": str(tuple(rpcn_final.shape))})
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
