import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full")


@app.cell
def _():
    import matplotlib.pyplot as plt
    import marimo as mo
    import numpy as np
    import spt_guidance.torch.fk_steering as fk_steering
    import spt_guidance.torch.rpcn as rpcn_lib
    import torch

    from toy.steer_toy.analytic_helpers import analytic_helpers
    from toy.steer_toy.log_reward import LogReward
    from toy.steer_toy.model.model import DiffusionModel
    from toy.steer_toy.utils import DIR_ROOT, get_reward_gmm_and_subset

    return (
        DIR_ROOT,
        DiffusionModel,
        LogReward,
        analytic_helpers,
        fk_steering,
        get_reward_gmm_and_subset,
        mo,
        np,
        plt,
        rpcn_lib,
        torch,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # RpCN / FK Steering Explorer

    Interactive controls for reproducing and comparing:
    - `src/toy/steer_toy/RpCN_sample.py`
    - `src/toy/steer_toy/FK_sample.py`

    The left panel shows the analytical posterior density
    $p(x \mid j \in S)$ where $S$ is the target Gaussian index subset.
    Use the **Run RpCN** and **Run FK** buttons to generate/cache samples.
    """)
    return


@app.cell
def _(mo):
    seed = mo.ui.number(start=0, stop=10_000, value=31, label="seed")
    batch_size = mo.ui.number(start=64, stop=4096, step=64, value=1024, label="batch_size")
    subset_indices_text = mo.ui.text(value="0,1,4,7", label="subset indices S")

    rpcn_n_chains = mo.ui.number(start=2, stop=64, value=20, label="rpcn n_chains")
    rpcn_update_count = mo.ui.number(start=1, stop=200, value=50, label="rpcn pcn_update_count")
    rpcn_step_min = mo.ui.slider(start=0.01, stop=0.5, step=0.01, value=0.05, label="rpcn min angle")
    rpcn_inverse_temp_max = mo.ui.slider(
        start=0.25,
        stop=10.0,
        step=0.05,
        value=5.0,
        label="rpcn inverse temperature max",
    )

    fk_n_chains = mo.ui.number(start=1, stop=32, value=4, label="fk chains")
    fk_inverse_temperature = mo.ui.slider(
        start=0.1,
        stop=10.0,
        step=0.1,
        value=5.0,
        label="fk inverse temperature",
    )
    fk_resample_threshold = mo.ui.slider(
        start=0.05,
        stop=0.95,
        step=0.05,
        value=0.5,
        label="fk ESS threshold",
    )

    run_rpcn = mo.ui.run_button(label="Run RpCN")
    run_fk = mo.ui.run_button(label="Run FK")

    x_min = mo.ui.slider(start=-12.0, stop=-2.0, step=0.5, value=-10.0, label="x min")
    x_max = mo.ui.slider(start=2.0, stop=12.0, step=0.5, value=10.0, label="x max")
    y_min = mo.ui.slider(start=-12.0, stop=-2.0, step=0.5, value=-10.0, label="y min")
    y_max = mo.ui.slider(start=2.0, stop=12.0, step=0.5, value=10.0, label="y max")
    grid_points = mo.ui.number(start=80, stop=600, step=20, value=220, label="density grid")
    contour_levels = mo.ui.number(start=8, stop=50, value=18, label="contour levels")
    point_size = mo.ui.slider(start=2, stop=40, step=1, value=8, label="point size")
    point_alpha = mo.ui.slider(start=0.05, stop=1.0, step=0.05, value=0.45, label="point alpha")
    metric_grid = mo.ui.number(start=16, stop=80, step=4, value=36, label="metric grid")
    sinkhorn_epsilon = mo.ui.slider(start=0.1, stop=20.0, step=0.1, value=4.0, label="sinkhorn epsilon")
    sinkhorn_iters = mo.ui.number(start=20, stop=400, step=20, value=120, label="sinkhorn iters")
    kl_epsilon = mo.ui.number(start=1e-12, stop=1e-2, step=1e-6, value=1e-8, label="kl epsilon")

    controls_panel = mo.vstack(
        [
            mo.hstack([seed, batch_size, subset_indices_text], justify="start"),
            mo.hstack(
                [
                    rpcn_n_chains,
                    rpcn_update_count,
                    rpcn_step_min,
                    rpcn_inverse_temp_max,
                    run_rpcn,
                ],
                justify="start",
            ),
            mo.hstack(
                [
                    fk_n_chains,
                    fk_inverse_temperature,
                    fk_resample_threshold,
                    run_fk,
                ],
                justify="start",
            ),
            mo.hstack(
                [
                    x_min,
                    x_max,
                    y_min,
                    y_max,
                    grid_points,
                    contour_levels,
                    point_size,
                    point_alpha,
                ],
                justify="start",
            ),
            mo.hstack(
                [metric_grid, sinkhorn_epsilon, sinkhorn_iters, kl_epsilon],
                justify="start",
            ),
        ]
    )
    return (
        batch_size,
        contour_levels,
        controls_panel,
        fk_inverse_temperature,
        fk_n_chains,
        fk_resample_threshold,
        grid_points,
        kl_epsilon,
        metric_grid,
        point_alpha,
        point_size,
        rpcn_inverse_temp_max,
        rpcn_n_chains,
        rpcn_step_min,
        rpcn_update_count,
        run_fk,
        run_rpcn,
        seed,
        sinkhorn_epsilon,
        sinkhorn_iters,
        subset_indices_text,
        x_max,
        x_min,
        y_max,
        y_min,
    )


@app.cell
def _():
    import hashlib
    import json

    def parse_subset_indices(text_value):
        values = [v.strip() for v in text_value.split(",") if v.strip()]
        if not values:
            return (0, 1, 4, 7)
        parsed = tuple(int(v) for v in values)
        return parsed

    def make_cache_key(name, params):
        payload = json.dumps({"name": name, "params": params}, sort_keys=True)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    return make_cache_key, parse_subset_indices


@app.cell
def _():
    MODEL_CACHE = {}
    SAMPLE_CACHE = {}
    return MODEL_CACHE, SAMPLE_CACHE


@app.cell
def _(
    DIR_ROOT,
    DiffusionModel,
    MODEL_CACHE,
    analytic_helpers,
    batch_size,
    get_reward_gmm_and_subset,
    parse_subset_indices,
    seed,
    subset_indices_text,
    torch,
):
    torch.manual_seed(int(seed.value))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    subset_indices = parse_subset_indices(subset_indices_text.value)
    gmm, subset_indices_tensor = get_reward_gmm_and_subset(device, subset_indices=subset_indices)

    model_cache_key = (device,)
    if model_cache_key not in MODEL_CACHE:
        checkpoint_path = DIR_ROOT / "src/toy/steer_toy/model/diffusion_gmm_model.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = DiffusionModel(input_dim=2, output_dim=2, hidden_dim=512, n_hidden=3).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        MODEL_CACHE[model_cache_key] = {
            "model": model,
            "alphas_cumprod": checkpoint["alphas_cumprod"],
        }

    model = MODEL_CACHE[model_cache_key]["model"]
    alphas_cumprod = MODEL_CACHE[model_cache_key]["alphas_cumprod"]
    toy_class = analytic_helpers(alphas_cumprod, model, batch_size=int(batch_size.value), device=device)
    return device, gmm, subset_indices, subset_indices_tensor, toy_class


@app.cell
def _(LogReward, gmm, subset_indices_tensor):
    reward_fn = LogReward(gmm, subset_indices_tensor)
    return (reward_fn,)


@app.cell
def _(
    SAMPLE_CACHE,
    device,
    make_cache_key,
    reward_fn,
    rpcn_inverse_temp_max,
    rpcn_lib,
    rpcn_n_chains,
    rpcn_step_min,
    rpcn_update_count,
    run_rpcn,
    seed,
    subset_indices,
    torch,
    toy_class,
):
    def run_rpcn_sampler():
        n_chains = int(rpcn_n_chains.value)
        kernel = rpcn_lib.KernelPreconditionedCrankNicolson(
            toy_class.data_dim,
            device=toy_class.device,
            dtype=toy_class.dtype,
        )
        steering = rpcn_lib.ResidualMCMCSteering(
            kernel,
            torch.linspace(torch.pi / 2 - 1e-3, float(rpcn_step_min.value), n_chains, device=device).sin(),
            reward_fn=reward_fn.forward,
            expected_next_fn=toy_class.expected_next_fn,
            std_fn=toy_class.std_fn,
            intermediate_reward=rpcn_lib.IntermediateReward.EXPECTED_SAMPLE,
            inverse_temperature=torch.linspace(0.0, float(rpcn_inverse_temp_max.value), n_chains, device=device),
            kwargs_intermediate_reward={"expected_sample_fn": toy_class.get_tweedie},
        )

        def pcn_update_fn(timestep, t0, t1, reward):
            _ = (t0, t1, reward)
            return True
            # return timestep % toy_class.check_every == 0

        with torch.no_grad():
            final_samples = steering.run(
                toy_class.timesteps.to(toy_class.device),
                pcn_should_update_fn=pcn_update_fn,
                pcn_update_count=int(rpcn_update_count.value),
                batch_size=toy_class.batch_size,
            )
        return final_samples[-1].detach().cpu()

    rpcn_params = {
        "batch_size": int(toy_class.batch_size),
        "seed": int(seed.value),
        "subset_indices": list(subset_indices),
        "n_chains": int(rpcn_n_chains.value),
        "pcn_update_count": int(rpcn_update_count.value),
        "step_min": float(rpcn_step_min.value),
        "inverse_temp_max": float(rpcn_inverse_temp_max.value),
    }
    rpcn_key = make_cache_key("rpcn", rpcn_params)

    if run_rpcn.value and rpcn_key not in SAMPLE_CACHE:
        SAMPLE_CACHE[rpcn_key] = run_rpcn_sampler()

    rpcn_samples = SAMPLE_CACHE.get(rpcn_key)
    rpcn_cache_hit = rpcn_key in SAMPLE_CACHE
    return rpcn_cache_hit, rpcn_params, rpcn_samples


@app.cell
def _(
    SAMPLE_CACHE,
    fk_inverse_temperature,
    fk_n_chains,
    fk_resample_threshold,
    fk_steering,
    make_cache_key,
    reward_fn,
    run_fk,
    seed,
    subset_indices,
    torch,
    toy_class,
):
    from functools import partial

    def extract_final_samples(samples):
        out = samples
        if isinstance(out, (tuple, list)):
            out = out[-1]
        if not torch.is_tensor(out):
            raise TypeError(f"Expected tensor output, got {type(out)}")
        if out.ndim >= 3:
            out = out[-1]
        if out.ndim == 3 and out.shape[-1] == 2:
            out = out.reshape(-1, 2)
        if out.ndim != 2 or out.shape[-1] != 2:
            raise ValueError(f"Unexpected sample shape: {tuple(out.shape)}")
        return out.detach().cpu()

    def run_fk_sampler():
        def propose_step(t0, t1, z):
            mean = toy_class.expected_next_fn(t0, t1, z)
            if t1 >= 0:
                std = toy_class.std_fn(t0, t1)
                return mean + std * torch.randn_like(z)
            return mean

        def adaptive_resample_safe(time_step, ensemble, potentials):
            _ = time_step
            weights = torch.nan_to_num(potentials, nan=0.0, posinf=0.0, neginf=0.0)
            weights = weights.clamp_min(0.0)
            weight_sum = weights.sum()
            safe_denominator = torch.clamp_min(weight_sum, torch.finfo(weights.dtype).tiny)
            normalized_probabilities = weights / safe_denominator
            uniform_probabilities = torch.ones_like(weights) / weights.numel()
            valid_probabilities = torch.isfinite(weight_sum) & (weight_sum > 0)
            probabilities = torch.where(valid_probabilities, normalized_probabilities, uniform_probabilities)

            effective_sample_size = 1.0 / torch.square(probabilities).sum()
            threshold = ensemble.shape[0] * float(fk_resample_threshold.value)
            return torch.where(
                effective_sample_size < threshold,
                fk_steering.ResampleFn.do_resample(ensemble, probabilities),
                ensemble,
            )

        fk = fk_steering.FeynmannKac(
            init_param_fn=partial(torch.randn, toy_class.data_dim, device=toy_class.device),
            reward_fn=reward_fn.forward,
            propose_fn=propose_step,
            resample_fn=adaptive_resample_safe,
            potential=fk_steering.Potential.MAX,
            intermediate_reward=fk_steering.IntermediateReward.EXPECTED_SAMPLE,
            inverse_temperature=float(fk_inverse_temperature.value),
            kwargs_intermediate_reward={"expected_sample_fn": toy_class.get_tweedie},
        )

        with torch.no_grad():
            final_samples = fk.run(
                toy_class.timesteps,
                int(fk_n_chains.value),
                batch_size=toy_class.batch_size,
            )
        return extract_final_samples(final_samples)

    fk_params = {
        "batch_size": int(toy_class.batch_size),
        "seed": int(seed.value),
        "subset_indices": list(subset_indices),
        "n_chains": int(fk_n_chains.value),
        "inverse_temperature": float(fk_inverse_temperature.value),
        "ess_threshold": float(fk_resample_threshold.value),
    }
    fk_key = make_cache_key("fk", fk_params)

    if run_fk.value and fk_key not in SAMPLE_CACHE:
        SAMPLE_CACHE[fk_key] = run_fk_sampler()

    fk_samples = SAMPLE_CACHE.get(fk_key)
    fk_cache_hit = fk_key in SAMPLE_CACHE
    return fk_cache_hit, fk_params, fk_samples


@app.cell
def _(
    contour_levels,
    fk_cache_hit,
    fk_params,
    fk_samples,
    gmm,
    grid_points,
    kl_epsilon,
    metric_grid,
    mo,
    np,
    plt,
    point_alpha,
    point_size,
    rpcn_cache_hit,
    rpcn_params,
    rpcn_samples,
    sinkhorn_epsilon,
    sinkhorn_iters,
    subset_indices,
    torch,
    x_max,
    x_min,
    y_max,
    y_min,
):
    def posterior_density_grid(gmm_obj, subset, xmin, xmax, ymin, ymax, n_grid):
        x = torch.linspace(xmin, xmax, n_grid, device=gmm_obj.device)
        y = torch.linspace(ymin, ymax, n_grid, device=gmm_obj.device)
        xx, yy = torch.meshgrid(x, y, indexing="xy")
        xy = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)

        subset_tensor = torch.as_tensor(subset, device=gmm_obj.device, dtype=torch.long)
        probs_subset = gmm_obj.probs[subset_tensor]
        probs_subset = probs_subset / probs_subset.sum()
        centers_subset = gmm_obj.centers[subset_tensor]

        diff = xy[:, None, :] - centers_subset[None, :, :]
        sq_dist = (diff * diff).sum(dim=-1)
        norm_const = 1.0 / (2.0 * torch.pi * (gmm_obj.std ** 2))
        comp_pdf = norm_const * torch.exp(-0.5 * sq_dist / (gmm_obj.std ** 2))
        density = (comp_pdf * probs_subset[None, :]).sum(dim=1)
        density = density.reshape(n_grid, n_grid)

        return xx.detach().cpu().numpy(), yy.detach().cpu().numpy(), density.detach().cpu().numpy()

    def matrix_sqrt_psd(matrix):
        eigvals, eigvecs = torch.linalg.eigh(matrix)
        eigvals = torch.clamp(eigvals, min=0.0)
        return eigvecs @ torch.diag(torch.sqrt(eigvals)) @ eigvecs.T

    def gaussian_w2_distance(mean_p, cov_p, mean_q, cov_q):
        mean_q = mean_q.to(device=mean_p.device, dtype=mean_p.dtype)
        cov_q = cov_q.to(device=cov_p.device, dtype=cov_p.dtype)
        cov_p_sqrt = matrix_sqrt_psd(cov_p)
        middle = cov_p_sqrt @ cov_q @ cov_p_sqrt
        middle_sqrt = matrix_sqrt_psd(middle)
        mean_term = torch.sum((mean_p - mean_q) ** 2)
        cov_term = torch.trace(cov_p + cov_q - 2.0 * middle_sqrt)
        return float(torch.sqrt(torch.clamp(mean_term + cov_term, min=0.0)).item())

    def subset_moments(gmm_obj, subset):
        subset_tensor = torch.as_tensor(subset, dtype=torch.long, device=gmm_obj.device)
        weights = gmm_obj.probs[subset_tensor]
        weights = weights / weights.sum()
        means = gmm_obj.centers[subset_tensor]
        mean = (weights[:, None] * means).sum(dim=0)

        covariance = torch.zeros((2, 2), dtype=torch.float32, device=gmm_obj.device)
        identity = torch.eye(2, dtype=torch.float32, device=gmm_obj.device)
        for idx in range(weights.shape[0]):
            delta = (means[idx] - mean).view(2, 1)
            covariance = covariance + weights[idx] * ((gmm_obj.std ** 2) * identity + delta @ delta.T)
        return mean, covariance

    def empirical_moments(samples):
        sample_tensor = samples.to(dtype=torch.float32)
        mean = sample_tensor.mean(dim=0)
        centered = sample_tensor - mean
        denom = max(sample_tensor.shape[0] - 1, 1)
        covariance = centered.T @ centered / denom
        return mean, covariance

    def discrete_posterior_mass(gmm_obj, subset, xmin, xmax, ymin, ymax, n_bins):
        x_edges = np.linspace(xmin, xmax, n_bins + 1)
        y_edges = np.linspace(ymin, ymax, n_bins + 1)
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
        xx_c, yy_c = np.meshgrid(x_centers, y_centers, indexing="xy")

        xy_centers = torch.as_tensor(
            np.stack([xx_c.reshape(-1), yy_c.reshape(-1)], axis=1),
            dtype=torch.float32,
            device=gmm_obj.device,
        )
        subset_tensor = torch.as_tensor(subset, device=gmm_obj.device, dtype=torch.long)
        probs_subset = gmm_obj.probs[subset_tensor]
        probs_subset = probs_subset / probs_subset.sum()
        centers_subset = gmm_obj.centers[subset_tensor]

        diff = xy_centers[:, None, :] - centers_subset[None, :, :]
        sq_dist = (diff * diff).sum(dim=-1)
        norm_const = 1.0 / (2.0 * torch.pi * (gmm_obj.std ** 2))
        component_pdf = norm_const * torch.exp(-0.5 * sq_dist / (gmm_obj.std ** 2))
        density = (component_pdf * probs_subset[None, :]).sum(dim=1)

        dx = x_edges[1] - x_edges[0]
        dy = y_edges[1] - y_edges[0]
        mass = (density * (dx * dy)).detach().cpu().numpy()
        mass = np.maximum(mass, 0.0)
        mass = mass / np.maximum(mass.sum(), 1e-16)
        coords = np.stack([xx_c.reshape(-1), yy_c.reshape(-1)], axis=1)
        return mass, coords, x_edges, y_edges

    def empirical_hist_mass(samples, x_edges, y_edges):
        hist, _, _ = np.histogram2d(samples[:, 0].numpy(), samples[:, 1].numpy(), bins=[x_edges, y_edges])
        mass = hist.reshape(-1)
        mass = mass / np.maximum(mass.sum(), 1e-16)
        return mass

    def kl_divergence(p, q, eps):
        p_safe = np.clip(p, eps, None)
        q_safe = np.clip(q, eps, None)
        p_safe = p_safe / p_safe.sum()
        q_safe = q_safe / q_safe.sum()
        return float(np.sum(p_safe * np.log(p_safe / q_safe)))

    def sinkhorn_w2(p, q, coords, epsilon, iters):
        p_t = torch.as_tensor(p, dtype=torch.float64)
        q_t = torch.as_tensor(q, dtype=torch.float64)
        x_t = torch.as_tensor(coords, dtype=torch.float64)
        cost = torch.cdist(x_t, x_t, p=2) ** 2
        kernel = torch.exp(-cost / epsilon)
        tiny = torch.finfo(torch.float64).tiny
        kernel = torch.clamp(kernel, min=tiny)

        u = torch.ones_like(p_t)
        v = torch.ones_like(q_t)

        for _ in range(iters):
            u = p_t / torch.clamp(kernel @ v, min=tiny)
            v = q_t / torch.clamp(kernel.T @ u, min=tiny)

        transport = u[:, None] * kernel * v[None, :]
        w2_squared = torch.sum(transport * cost)
        return float(torch.sqrt(torch.clamp(w2_squared, min=0.0)).item())

    xmin = float(x_min.value)
    xmax = float(x_max.value)
    ymin = float(y_min.value)
    ymax = float(y_max.value)
    n_grid = int(grid_points.value)
    levels = int(contour_levels.value)
    size = float(point_size.value)
    alpha = float(point_alpha.value)

    xx, yy, density = posterior_density_grid(gmm, subset_indices, xmin, xmax, ymin, ymax, n_grid)

    posterior_mass, metric_coords, metric_x_edges, metric_y_edges = discrete_posterior_mass(
        gmm,
        subset_indices,
        xmin,
        xmax,
        ymin,
        ymax,
        int(metric_grid.value),
    )

    posterior_mean, posterior_cov = subset_moments(gmm, subset_indices)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

    for ax in axes:
        ax.contourf(xx, yy, density, levels=levels, cmap="viridis")
        ax.contour(xx, yy, density, levels=levels, colors="white", linewidths=0.35, alpha=0.6)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)

    axes[0].set_title("Analytical posterior p(x | j in S)")

    axes[1].set_title("RpCN samples")
    if rpcn_samples is not None:
        axes[1].scatter(rpcn_samples[:, 0].numpy(), rpcn_samples[:, 1].numpy(), s=size, alpha=alpha, c="#ef6c00")
    else:
        axes[1].text(0.5, 0.5, "Press Run RpCN", ha="center", va="center", transform=axes[1].transAxes, c='white')

    axes[2].set_title("FK samples")
    if fk_samples is not None:
        axes[2].scatter(fk_samples[:, 0].numpy(), fk_samples[:, 1].numpy(), s=size, alpha=alpha, c="tab:red")
    else:
        axes[2].text(0.5, 0.5, "Press Run FK", ha="center", va="center", transform=axes[2].transAxes, c='white')

    fig.tight_layout()
    plt.close(fig)

    diagnostics = {
        "subset_indices": subset_indices,
        "rpcn_cached_for_params": rpcn_cache_hit,
        "fk_cached_for_params": fk_cache_hit,
        "rpcn_params": rpcn_params,
        "fk_params": fk_params,
        "rpcn_sample_count": int(rpcn_samples.shape[0]) if rpcn_samples is not None else 0,
        "fk_sample_count": int(fk_samples.shape[0]) if fk_samples is not None else 0,
        "rpcn_mean": np.round(rpcn_samples.mean(dim=0).numpy(), 3).tolist() if rpcn_samples is not None else None,
        "fk_mean": np.round(fk_samples.mean(dim=0).numpy(), 3).tolist() if fk_samples is not None else None,
    }

    metrics = {
        "metric_grid_bins": int(metric_grid.value),
        "sinkhorn_epsilon": float(sinkhorn_epsilon.value),
        "sinkhorn_iters": int(sinkhorn_iters.value),
        "kl_epsilon": float(kl_epsilon.value),
        "rpcn": None,
        "fk": None,
    }

    if rpcn_samples is not None:
        rpcn_hist = empirical_hist_mass(rpcn_samples, metric_x_edges, metric_y_edges)
        rpcn_mean, rpcn_cov = empirical_moments(rpcn_samples)
        metrics["rpcn"] = {
            "w2_sinkhorn_to_posterior": sinkhorn_w2(
                posterior_mass,
                rpcn_hist,
                metric_coords,
                float(sinkhorn_epsilon.value),
                int(sinkhorn_iters.value),
            ),
            "w2_gaussian_to_posterior": gaussian_w2_distance(posterior_mean, posterior_cov, rpcn_mean, rpcn_cov),
            "kl_posterior_to_empirical": kl_divergence(
                posterior_mass,
                rpcn_hist,
                float(kl_epsilon.value),
            ),
            "kl_empirical_to_posterior": kl_divergence(
                rpcn_hist,
                posterior_mass,
                float(kl_epsilon.value),
            ),
        }

    if fk_samples is not None:
        fk_hist = empirical_hist_mass(fk_samples, metric_x_edges, metric_y_edges)
        fk_mean, fk_cov = empirical_moments(fk_samples)
        metrics["fk"] = {
            "w2_sinkhorn_to_posterior": sinkhorn_w2(
                posterior_mass,
                fk_hist,
                metric_coords,
                float(sinkhorn_epsilon.value),
                int(sinkhorn_iters.value),
            ),
            "w2_gaussian_to_posterior": gaussian_w2_distance(posterior_mean, posterior_cov, fk_mean, fk_cov),
            "kl_posterior_to_empirical": kl_divergence(
                posterior_mass,
                fk_hist,
                float(kl_epsilon.value),
            ),
            "kl_empirical_to_posterior": kl_divergence(
                fk_hist,
                posterior_mass,
                float(kl_epsilon.value),
            ),
        }

    def format_metric(v):
        if v is None:
            return "-"
        return f"{v:.6f}"

    def best_bold(rpcn_value, fk_value, lower_is_better=True):
        if rpcn_value is None and fk_value is None:
            return "-", "-"
        if rpcn_value is None:
            return "-", f"**{format_metric(fk_value)}**"
        if fk_value is None:
            return f"**{format_metric(rpcn_value)}**", "-"

        rpcn_fmt = format_metric(rpcn_value)
        fk_fmt = format_metric(fk_value)
        if abs(rpcn_value - fk_value) < 1e-12:
            return f"**{rpcn_fmt}**", f"**{fk_fmt}**"

        rpcn_better = rpcn_value < fk_value if lower_is_better else rpcn_value > fk_value
        if rpcn_better:
            return f"**{rpcn_fmt}**", fk_fmt
        return rpcn_fmt, f"**{fk_fmt}**"

    rpcn_metrics = metrics["rpcn"] if metrics["rpcn"] is not None else {}
    fk_metrics = metrics["fk"] if metrics["fk"] is not None else {}

    rows = []
    metric_specs = [
        ("W2 (Sinkhorn)", "w2_sinkhorn_to_posterior"),
        ("W2 (Gaussian approx)", "w2_gaussian_to_posterior"),
        (r"KL(posterior \|\| empirical)", "kl_posterior_to_empirical"),
        (r"KL(empirical \|\| posterior)", "kl_empirical_to_posterior"),
    ]
    for label, key in metric_specs:
        rpcn_cell, fk_cell = best_bold(rpcn_metrics.get(key), fk_metrics.get(key), lower_is_better=True)
        rows.append(f"| {label} | {rpcn_cell} | {fk_cell} |")

    leaderboard_md = "\n".join(
        [
            "### Quantitative Comparison",
            "Lower is better for all metrics. Best value per row is bolded.",
            "",
            "| Metric | RpCN | FK |",
            "|---|---:|---:|",
            *rows,
        ]
    )

    leaderboard = mo.md(leaderboard_md)
    return diagnostics, fig, leaderboard


@app.cell
def _(controls_panel, diagnostics, fig, leaderboard, mo):
    diagnostics_md = "\n".join(
        [
            "**Diagnostics**",
            f"- subset `S`: `{tuple(diagnostics['subset_indices'])}`",
            f"- RpCN cached: `{diagnostics['rpcn_cached_for_params']}` | samples: `{diagnostics['rpcn_sample_count']}`",
            f"- FK cached: `{diagnostics['fk_cached_for_params']}` | samples: `{diagnostics['fk_sample_count']}`",
        ]
    )
    dashboard = mo.vstack(
        [
            controls_panel,
            leaderboard,
            fig,
            mo.md(diagnostics_md),
        ]
    )
    dashboard
    return


@app.cell
def _(mo):
    mo.md("""
    Quick start:
    1. `uv sync`
    2. `uv run marimo edit notebooks/steering_explorer.py`
    """)
    return


if __name__ == "__main__":
    app.run()
