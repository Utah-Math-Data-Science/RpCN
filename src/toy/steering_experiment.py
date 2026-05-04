import math
from typing import Callable

import polars as pl
import spt_guidance.torch.fk_steering as fk_steering
import spt_guidance.torch.rpcn as rpcn
import torch

import conf.setting
import conf.steering
import toy.utils
from toy.model import Bao2024EnsembleScoreMatching, EnsembleScoreModel, metropolis_hastings_sample


log = toy.utils.get_logger_by_filename(__file__)


def _build_reward_fn(cfg_exp: conf.setting.GMMExperiment) -> Callable[[torch.Tensor], torch.Tensor]:
    if cfg_exp.reward_kind != conf.setting.RewardKind.SIGNED_GAUSSIAN_PAIR:
        raise ValueError(f'Unsupported reward kind: {cfg_exp.reward_kind}')

    def reward_gauss(x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-((x / cfg_exp.reward_sigma).square()))

    def reward_fn(x: torch.Tensor) -> torch.Tensor:
        return (
            reward_gauss(x + 4) - 2 * reward_gauss(x + 3) + reward_gauss(x - 3)
        ).squeeze()

    return reward_fn


def _build_gmm(cfg_exp: conf.setting.GMMExperiment):
    means = torch.tensor(cfg_exp.parse_csv_floats(cfg_exp.means), dtype=torch.float32)
    stds = torch.tensor(cfg_exp.parse_csv_floats(cfg_exp.stds), dtype=torch.float32)
    raw_weights = torch.tensor(cfg_exp.parse_csv_floats(cfg_exp.weights), dtype=torch.float32)
    weights = raw_weights / raw_weights.sum()

    def gmm_sample(num_points: int, device_str: str) -> torch.Tensor:
        device_obj = torch.device(device_str)
        probs = weights.to(device_obj)
        centers = means.to(device_obj)
        scales = stds.to(device_obj)
        indices = torch.multinomial(probs, num_points, replacement=True)
        eps = torch.randn(num_points, device=device_obj)
        samples = centers[indices] + scales[indices] * eps
        return samples.unsqueeze(1)

    def gmm_log_prob(x_values: torch.Tensor) -> torch.Tensor:
        x_col = x_values.unsqueeze(-1)
        mean_row = means.to(device=x_values.device, dtype=x_values.dtype).unsqueeze(0)
        std_row = stds.to(device=x_values.device, dtype=x_values.dtype).unsqueeze(0)
        weight_row = weights.to(device=x_values.device, dtype=x_values.dtype).unsqueeze(0)
        log_norm = -0.5 * torch.log(2.0 * torch.pi * std_row.square())
        log_comp = log_norm - 0.5 * ((x_col - mean_row) / std_row).square() + torch.log(weight_row)
        return torch.logsumexp(log_comp, dim=1)

    reward_fn = _build_reward_fn(cfg_exp)
    return gmm_sample, gmm_log_prob, reward_fn


def _build_score_model(cfg):
    selected_device = cfg.device
    if selected_device == 'cuda' and not torch.cuda.is_available():
        selected_device = 'cpu'

    model_cfg = type('_score_model_cfg', (), {})()
    model_cfg.epsilon_alpha = cfg.score_model.epsilon_alpha
    model_cfg.epsilon_beta = cfg.score_model.epsilon_beta
    model_cfg.sampling_time_step_count = cfg.score_model.n_steps
    model_cfg.vmap_chunk_size = cfg.score_model.vmap_chunk_size
    model_cfg.sampling_max_score_norm = cfg.score_model.sampling_max_score_norm
    model_cfg.renormalize_sampled_noise = False
    model_cfg.sample_noise_scale_std = True
    model_cfg.sample_noise_add_mean = True

    prob_path = Bao2024EnsembleScoreMatching(model_cfg)
    score_model = EnsembleScoreModel(model_cfg, prob_path)
    return selected_device, score_model


def _run_unsteered(cfg, score_model, reference_data):
    torch.manual_seed(cfg.seed)
    with torch.no_grad():
        return score_model.sample(cfg.score_model.n_steps, reference_data)


def _run_fk(cfg, score_model, reference_data, reward_fn, timesteps):
    cfg_fk: conf.steering.FK = cfg.steering
    torch.manual_seed(cfg.seed)

    def propose_step(t0, t1, z):
        mean = score_model.expected_next_fn(t0, t1, z)
        if t1 > 0:
            std = score_model.std_fn(t0, t1)
            return mean + std * torch.randn_like(z)
        return mean

    def adaptive_resample_safe(time_step, ensemble, potentials):
        if time_step % cfg_fk.check_every != 0:
            return ensemble
        weights = torch.nan_to_num(potentials, nan=0.0, posinf=0.0, neginf=0.0)
        weights = weights.clamp_min(0.0)
        tiny = 1e-9
        weights = torch.where(weights.sum() > tiny, weights, 1.0)
        probs = weights / weights.sum()
        ess = 1.0 / probs.square().sum()
        threshold = float(cfg_fk.resample_threshold) * ensemble.shape[0]
        return torch.where(ess < threshold, fk_steering.ResampleFn.do_resample(ensemble, probs), ensemble)

    fk = fk_steering.FeynmannKac(
        init_param_fn=lambda: torch.randn(1, device=reference_data.device, dtype=reference_data.dtype),
        reward_fn=reward_fn,
        propose_fn=propose_step,
        resample_fn=adaptive_resample_safe,
        potential=cfg_fk.potential,
        intermediate_reward=fk_steering.IntermediateReward.EXPECTED_SAMPLE,
        inverse_temperature=cfg_fk.inverse_temperature,
        kwargs_intermediate_reward={'expected_sample_fn': score_model.get_tweedie},
    )
    with torch.no_grad():
        fk_samples = fk.run(timesteps, cfg_fk.n_chains, batch_size=cfg.batch_size)
    if fk_samples.ndim == 1:
        return fk_samples.unsqueeze(0).unsqueeze(1)
    if fk_samples.ndim == 2:
        return fk_samples
    if fk_samples.ndim == 3:
        return fk_samples[-1]
    raise ValueError(f'Unexpected FK sample shape: {fk_samples.shape}')


def _run_rpcn(cfg, score_model, reference_data, reward_fn, timesteps):
    cfg_rpcn: conf.steering.RpCN = cfg.steering
    torch.manual_seed(cfg.seed)
    kernel = rpcn.KernelPreconditionedCrankNicolson(1, device=reference_data.device, dtype=reference_data.dtype)
    steering = rpcn.ResidualMCMCSteering(
        kernel,
        torch.linspace(
            math.pi / 2 - 1e-3,
            0.05,
            cfg_rpcn.n_chains,
            device=reference_data.device,
            dtype=reference_data.dtype,
        ).sin(),
        reward_fn=reward_fn,
        expected_next_fn=score_model.expected_next_fn,
        std_fn=score_model.std_fn,
        intermediate_reward=rpcn.IntermediateReward.EXPECTED_SAMPLE,
        inverse_temperature=torch.linspace(
            0.0,
            cfg_rpcn.inverse_temperature,
            cfg_rpcn.n_chains,
            device=reference_data.device,
            dtype=reference_data.dtype,
        ),
        kwargs_intermediate_reward={'expected_sample_fn': score_model.get_tweedie},
    )

    def rpcn_update_fn(step_idx, t0, t1, reward):
        _ = (t0, t1, reward)
        return step_idx % cfg_rpcn.check_every == 0

    with torch.no_grad():
        rpcn_samples = steering.run(
            timesteps,
            pcn_should_update_fn=rpcn_update_fn,
            pcn_update_count=cfg_rpcn.pcn_update_count,
            batch_size=cfg.batch_size,
        )
    if rpcn_samples.ndim == 2:
        return rpcn_samples
    if rpcn_samples.ndim == 3:
        return rpcn_samples[-1]
    raise ValueError(f'Unexpected RpCN sample shape: {rpcn_samples.shape}')


def _run_mh(cfg, selected_device, gmm_sample, gmm_log_prob, reward_fn):
    cfg_mh: conf.steering.MH = cfg.steering
    torch.manual_seed(cfg.seed)

    def log_target_fn(x: torch.Tensor) -> torch.Tensor:
        x_flat = x[:, 0]
        return gmm_log_prob(x_flat) + cfg_mh.inverse_temperature * reward_fn(x_flat)

    mh_samples, mh_diag = metropolis_hastings_sample(
        log_target_fn=log_target_fn,
        num_chains=cfg_mh.n_chains,
        num_samples=cfg_mh.num_samples,
        dim=1,
        burn_in=cfg_mh.burn_in,
        proposal_std=cfg_mh.proposal_std,
        thinning=cfg_mh.thinning,
        initial_x=gmm_sample(cfg_mh.n_chains, selected_device),
        device=torch.device(selected_device),
        dtype=torch.float32,
    )
    retained_total = cfg_mh.n_chains * cfg_mh.num_samples
    log.info(
        'MH diagnostics: acceptance_rate=%.4f accepted=%d proposed=%d retained_samples=%d',
        mh_diag['acceptance_rate'],
        mh_diag['accepted'],
        mh_diag['proposed'],
        retained_total,
    )
    log.info(
        'MH parameters: n_chains=%d num_samples=%d burn_in=%d thinning=%d proposal_std=%.4f inverse_temperature=%.4f',
        cfg_mh.n_chains,
        cfg_mh.num_samples,
        cfg_mh.burn_in,
        cfg_mh.thinning,
        cfg_mh.proposal_std,
        cfg_mh.inverse_temperature,
    )
    return mh_samples.reshape(-1, 1)


def run_and_make_samples_df(cfg) -> pl.DataFrame:
    gmm_sample, gmm_log_prob, reward_fn = _build_gmm(cfg.setting)
    selected_device, score_model = _build_score_model(cfg)
    reference_data = gmm_sample(cfg.score_model.n_reference, selected_device)
    score_model.set_training_data(reference_data)
    # timesteps = torch.linspace(1.0, 0.0, cfg.score_model.n_steps, device=reference_data.device, dtype=reference_data.dtype)
    timesteps = score_model.prob_path.linspace_time(cfg.score_model.n_steps, device=reference_data.device)

    if isinstance(cfg.steering, conf.steering.Unsteered):
        final_samples = _run_unsteered(cfg, score_model, reference_data)
    elif isinstance(cfg.steering, conf.steering.FK):
        final_samples = _run_fk(cfg, score_model, reference_data, reward_fn, timesteps)
    elif isinstance(cfg.steering, conf.steering.RpCN):
        final_samples = _run_rpcn(cfg, score_model, reference_data, reward_fn, timesteps)
    elif isinstance(cfg.steering, conf.steering.MH):
        final_samples = _run_mh(cfg, selected_device, gmm_sample, gmm_log_prob, reward_fn)
    else:
        raise ValueError(f'Unsupported steering config type: {type(cfg.steering).__name__}')

    final_samples = final_samples.detach().cpu()
    x_values = final_samples[:, 0]
    rewards = torch.vmap(reward_fn)(x_values).detach().cpu()
    sample_ids = torch.arange(x_values.shape[0], dtype=torch.int64)
    return pl.DataFrame(
        {
            'sample_id': sample_ids.tolist(),
            'x': x_values.tolist(),
            'reward': rewards.tolist(),
        }
    )
