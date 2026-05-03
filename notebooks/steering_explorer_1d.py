import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full")


@app.cell
def _():
    import math

    import matplotlib.pyplot as plt
    import marimo as mo
    import spt_guidance.torch.fk_steering as fk_steering
    import spt_guidance.torch.rpcn as rpcn
    import torch

    from toy.model import Bao2024EnsembleScoreMatching, EnsembleScoreModel

    return (
        Bao2024EnsembleScoreMatching,
        EnsembleScoreModel,
        fk_steering,
        math,
        mo,
        plt,
        rpcn,
        torch,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # 1D Steering Explorer

    Define a 1D Gaussian mixture and a reward function `reward_fn(x)`, then run
    unsteered, FK, and RpCN samplers independently.
    """)
    return


@app.cell(hide_code=True)
def _(gmm_density, means, mo, plt, stds, torch, weights, x_grid):
    gmm_fig, gmm_ax = plt.subplots(figsize=(10, 3.5))
    gmm_ax.plot(x_grid.numpy(), gmm_density.numpy(), color="#0b7285", lw=2.0, label="GMM density")
    marker_y = torch.zeros_like(means)
    gmm_ax.scatter(means.numpy(), marker_y.numpy(), c="#1971c2", s=60, label="mode means")
    for i in range(means.shape[0]):
        gmm_ax.text(float(means[i]), 0.0, f"w={weights[i]:.2f}, s={stds[i]:.2f}", fontsize=8, ha="center", va="bottom")
    gmm_ax.set_title("GMM Preview")
    gmm_ax.set_xlabel("x")
    gmm_ax.set_ylabel("density")
    gmm_ax.grid(alpha=0.2)
    gmm_ax.legend(loc="upper right")
    gmm_fig.tight_layout()
    gmm_view = mo.vstack([mo.md("## GMM Preview"), mo.as_html(gmm_fig)])
    gmm_view
    return


@app.cell
def _(mo):
    n_modes = mo.ui.number(start=1, stop=8, step=1, value=4, label="n_modes")
    return (n_modes,)


@app.cell
def _(mo, n_modes):
    mode_count = int(n_modes.value)
    mean_array = mo.ui.array(
        [
            mo.ui.number(start=-20.0, stop=20.0, step=0.1, value=float(-6.0 + 4.0 * idx), label=f"mean_{idx}")
            for idx in range(mode_count)
        ]
    )
    std_array = mo.ui.array(
        [
            mo.ui.number(start=0.05, stop=6.0, value=0.7, label=f"std_{idx}")
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
def _(mo, plt, reward_values, x_grid):
    reward_fig, reward_ax = plt.subplots(figsize=(10, 3.5))
    reward_ax.plot(x_grid.numpy(), reward_values.numpy(), color="#c92a2a", lw=2.0)
    reward_ax.set_title("Reward Preview")
    reward_ax.set_xlabel("x")
    reward_ax.set_ylabel("reward")
    reward_ax.grid(alpha=0.2)
    reward_fig.tight_layout()
    reward_view = mo.vstack([mo.md("## Reward Preview"), mo.as_html(reward_fig)])
    reward_view
    return


@app.cell
def _(torch):
    def reward_fn(x: torch.Tensor) -> torch.Tensor:
        r = torch.exp(-0.5 * ((x - 3.0) / 0.8) ** 2) - 0.35 * torch.exp(-0.5 * ((x + 2.0) / 0.7) ** 2)
        # r = 1e6 * x**2
        # r = 1000 * r
        return r.squeeze()

    return (reward_fn,)


@app.cell
def _(
    fk_final,
    gmm_density,
    mo,
    plt,
    reward_values,
    rpcn_final,
    unsteered_samples,
    x_grid,
):
    import seaborn as sns
    sns.set_theme(style='white', font_scale=1.8)
    has_samples_plot = not (unsteered_samples is None and fk_final is None and rpcn_final is None)
    if has_samples_plot:
        sample_dicts = [
            # dict(label="Base", samples=unsteered_samples, color="tab:grey"),
            dict(label="FK Steering", samples=fk_final, color="tab:blue"),
            dict(label="RpCN Steering", samples=rpcn_final, color="tab:orange"),
        ]
        comp_fig, comp_axes = plt.subplots(1, len(sample_dicts), figsize=(10, 5), sharey=True)
        # labels = ["Base", "FK Steering", "RpCN Steering"]
        # sample_sets = [unsteered_samples, fk_final, rpcn_final]
        # colors = ["tab:grey", "tab:blue", "tab:orange"]
        for comp_ax, d in zip(comp_axes, sample_dicts):
            label = d['label']
            samples = d['samples']
            color = d['color']
            if samples is not None:
                comp_ax.hist(samples[:, 0].cpu().numpy(), bins=60, density=True, alpha=0.7, color=color,
                             # label=f"{label} samples",
                             histtype='stepfilled',
                             linewidth=0)
            else:
                comp_ax.text(0.5, 0.5, "Not run", transform=comp_ax.transAxes, ha="center", va="center", fontsize=12)
            comp_ax.plot(x_grid.numpy(), gmm_density.numpy(), color="black", lw=3.0, label="Base density")
            reward_scaled = reward_values.numpy()
            reward_scaled = (reward_scaled - reward_scaled.min()) / (reward_scaled.max() - reward_scaled.min() + 1e-8)
            reward_scaled = reward_scaled * gmm_density.max().item()
            comp_ax.plot(x_grid.numpy(), reward_scaled, color="tab:red", lw=3, linestyle="--", label="Reward (scaled)")
            # comp_ax.set_title(label)
            # comp_ax.set_xlabel("x")
            comp_ax.grid()
            # comp_ax.grid(alpha=0.2)
            # comp_ax.set_ylim(0, gmm_density.max() + min(gmm_density.max() * .05, .05))
            comp_ax.yaxis.set_tick_params(labelleft=True)
            comp_ax.margins(0)
        comp_axes[0].set_ylabel("Density")
        handles, legend_labels = comp_axes[0].get_legend_handles_labels()
        comp_fig.legend(handles, legend_labels, loc="upper center", ncol=3, frameon=True, fancybox=True)
        comp_fig.tight_layout(rect=[0, 0, 1, 0.9])
        comparison_view = mo.vstack([mo.md("## Sample Comparison"), mo.as_html(comp_fig)])
    else:
        comparison_view = mo.md("Run one or more samplers to generate samples.")
    comparison_view
    return (comp_fig,)


@app.cell
def _(comp_fig):
    comp_fig.savefig('Toy.pdf', format='pdf', bbox_inches='tight', pad_inches=0.06)
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

    x_min = mo.ui.number(start=-30.0, stop=0.0, step=0.5, value=-12.0, label="x_min")
    x_max = mo.ui.number(start=0.0, stop=30.0, step=0.5, value=12.0, label="x_max")
    n_grid = mo.ui.number(start=100, stop=4000, step=100, value=1200, label="plot_grid")

    fk_n_chains = mo.ui.number(start=1, stop=64, step=1, value=20, label="fk_n_chains")
    fk_inverse_temperature = mo.ui.number(start=0.00, stop=20.0, step=0.05, value=1.0, label="fk_inverse_temperature")
    fk_resample_threshold = mo.ui.number(start=0.05, stop=0.95, step=0.05, value=0.5, label="fk_ess_threshold")
    fk_check_every = mo.ui.number(start=1, stop=100, step=1, value=1, label="fk_check_every")

    rpcn_n_chains = mo.ui.number(start=2, stop=128, step=1, value=20, label="rpcn_n_chains")
    rpcn_update_count = mo.ui.number(start=1, stop=200, step=1, value=5, label="rpcn_update_count")
    rpcn_angle_min = mo.ui.number(start=0.001, stop=1.0, step=0.001, value=0.05, label="rpcn_angle_min")
    rpcn_inverse_temp_max = mo.ui.number(start=0.00, stop=20.0, step=0.05, value=1.0, label="rpcn_inverse_temp_max")
    rpcn_check_every = mo.ui.number(start=1, stop=100, step=1, value=1, label="rpcn_check_every")

    run_unsteered = mo.ui.run_button(label="Unsteered")
    run_fk = mo.ui.run_button(label="FK")
    run_rpcn = mo.ui.run_button(label="RpCN")

    controls = mo.vstack(
        [
            mo.hstack([seed, device, batch_size, n_steps, n_reference, run_unsteered, run_fk, run_rpcn], justify="start"),
            mo.hstack([epsilon_alpha, epsilon_beta, vmap_chunk_size, sampling_max_score_norm], justify="start"),
            mo.hstack([x_min, x_max, n_grid], justify="start"),
            mo.hstack([fk_n_chains, fk_inverse_temperature, fk_resample_threshold, fk_check_every], justify="start"),
            mo.hstack([rpcn_n_chains, rpcn_update_count, rpcn_angle_min, rpcn_inverse_temp_max, rpcn_check_every], justify="start"),
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
        n_grid,
        n_reference,
        n_steps,
        rpcn_angle_min,
        rpcn_check_every,
        rpcn_inverse_temp_max,
        rpcn_n_chains,
        rpcn_update_count,
        run_fk,
        run_rpcn,
        run_unsteered,
        sampling_max_score_norm,
        seed,
        vmap_chunk_size,
        x_max,
        x_min,
    )


@app.cell
def _(fk_final, mo, reward_fn, rpcn_final, torch, unsteered_samples):
    has_samples_stats = not (unsteered_samples is None and fk_final is None and rpcn_final is None)
    if has_samples_stats:
        def reward_mean(samples):
            vals = []
            for i in range(samples.shape[0]):
                vals.append(torch.as_tensor(reward_fn(samples[i, 0]), dtype=torch.float32))
            return float(torch.stack(vals).mean().item())

        def reward_max(samples):
            vals = []
            for i in range(samples.shape[0]):
                vals.append(torch.as_tensor(reward_fn(samples[i, 0]), dtype=torch.float32))
            return float(torch.stack(vals).max().item())

        def stats_dict(name, samples):
            x = samples[:, 0].cpu()
            q = torch.quantile(x, torch.tensor([0.05, 0.5, 0.95]))
            return {
                "method": name,
                "count": int(samples.shape[0]),
                # "mean": float(x.mean().item()),
                # "std": float(x.std(unbiased=False).item()),
                "q05": float(q[0].item()),
                "q50": float(q[1].item()),
                "q95": float(q[2].item()),
                "reward_mean": reward_mean(samples),
                "reward_max": reward_max(samples),
            }

        stats_rows = []
        if unsteered_samples is not None:
            stats_rows.append(stats_dict("Unsteered", unsteered_samples))
        if fk_final is not None:
            stats_rows.append(stats_dict("FK", fk_final))
        if rpcn_final is not None:
            stats_rows.append(stats_dict("RpCN", rpcn_final))
        stats_view = mo.vstack([mo.md("## Summary Statistics"), mo.ui.table(stats_rows)])
    else:
        stats_view = mo.md("Summary statistics will appear after running at least one sampler.")
    stats_view
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
        mean_row = means.unsqueeze(0)
        std_row = stds.unsqueeze(0)
        weight_row = weights.unsqueeze(0)
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
    return gmm_density, gmm_sample, x_grid


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
    return (
        get_fk_samples,
        get_rpcn_samples,
        get_unsteered_samples,
        set_fk_samples,
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

    timesteps = torch.linspace(1.0, 0.0, int(n_steps.value), device=reference_data.device, dtype=reference_data.dtype)
    return reference_data, score_model, timesteps


@app.cell
def _(get_fk_samples, get_rpcn_samples, get_unsteered_samples):
    fk_samples = get_fk_samples()
    rpcn_samples = get_rpcn_samples()
    unsteered_samples = get_unsteered_samples()
    return fk_samples, rpcn_samples, unsteered_samples


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
            potential=fk_steering.Potential.MAX,
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
def _(fk_final, mo, rpcn_final, unsteered_samples):
    statuses = []
    statuses.append("Unsteered: ready" if unsteered_samples is not None else "Unsteered: not run")
    statuses.append("FK: ready" if fk_final is not None else "FK: not run")
    statuses.append("RpCN: ready" if rpcn_final is not None else "RpCN: not run")
    status_view = mo.vstack([mo.md("## Sampler Status"), mo.md(" | ".join(statuses))])
    status_view
    return


@app.cell
def _(fk_final, mo, rpcn_final, unsteered_samples):
    debug_rows = []
    if unsteered_samples is not None:
        debug_rows.append({"sampler": "Unsteered", "shape": str(tuple(unsteered_samples.shape))})
    if fk_final is not None:
        debug_rows.append({"sampler": "FK", "shape": str(tuple(fk_final.shape))})
    if rpcn_final is not None:
        debug_rows.append({"sampler": "RpCN", "shape": str(tuple(rpcn_final.shape))})
    if debug_rows:
        shape_view = mo.vstack([mo.md("## Output Shapes"), mo.ui.table(debug_rows)])
    else:
        shape_view = mo.md("No sampler outputs yet.")
    shape_view
    return


if __name__ == "__main__":
    app.run()
