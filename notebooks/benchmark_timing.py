"""Benchmark unsteered, best-of-N, RpCN, DAS, FK, and CREPE sampling."""

import argparse
import functools
import math
import statistics
from time import perf_counter

import spt_guidance.torch.crepe as crepe
import spt_guidance.torch.fk_steering as fk_steering
import spt_guidance.torch.rpcn as rpcn
import torch

from toy.model import Bao2024EnsembleScoreMatching, EnsembleScoreModel


METHODS = ("unsteered", "best_of_n", "rpcn", "das", "fk", "crepe")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument("--n-reference", type=int, default=1024)
    parser.add_argument("--best-of-n-members", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))

    parser.add_argument("--fk-chains", type=int, default=20)
    parser.add_argument("--fk-inverse-temperature", type=float, default=25.0)
    parser.add_argument("--fk-potential", choices=("difference", "max", "sum"), default="difference")
    parser.add_argument("--fk-temperature-count", type=int, default=1)
    parser.add_argument("--fk-resample-threshold", type=float, default=0.5)
    parser.add_argument("--fk-check-every", type=int, default=1)

    parser.add_argument("--das-particles", type=int, default=20)
    parser.add_argument("--das-inverse-temperature", type=float, default=25.0)
    parser.add_argument("--das-proposal-scale", type=float, default=1.0)
    parser.add_argument("--das-tempering-gamma", type=float, default=0.003)
    parser.add_argument("--das-resample-threshold", type=float, default=0.5)

    parser.add_argument("--rpcn-chains", type=int, default=4)
    parser.add_argument("--rpcn-updates", type=int, default=8)
    parser.add_argument("--rpcn-angle-min", type=float, default=0.05)
    parser.add_argument("--rpcn-inverse-temperature", type=float, default=25.0)
    parser.add_argument("--rpcn-check-every", type=int, default=1)
    parser.add_argument("--rpcn-burn-in-frac", type=float, default=0.0)

    parser.add_argument("--crepe-num-samples", type=int, default=1)
    parser.add_argument("--crepe-burn-in", type=int, default=24)
    parser.add_argument("--crepe-thinning", type=int, default=1)
    parser.add_argument("--crepe-inverse-temperature", type=float, default=25.0)
    parser.add_argument("--crepe-reward-power", type=float, default=1.0)
    parser.add_argument("--crepe-reward-reference", type=float, default=0.0)
    parser.add_argument("--crepe-reward-target", type=float, default=1.0)
    parser.add_argument("--crepe-replicas", type=int, default=17)
    parser.add_argument("--crepe-substeps", type=int, default=5)
    parser.add_argument("--crepe-time-min", type=float, default=0.02)
    parser.add_argument("--crepe-time-max", type=float, default=1.0)
    parser.add_argument("--crepe-time-spacing", choices=("linear", "karras"), default="karras")
    parser.add_argument("--crepe-karras-rho", type=float, default=7.0)
    parser.add_argument("--crepe-gradient-clip", type=float, default=1000.0)
    parser.add_argument(
        "--crepe-initialization",
        choices=("denoising path", "independent marginals", "shared clean ancestor"),
        default="denoising path",
    )
    parser.add_argument(
        "--crepe-rne-method",
        choices=("exact transition", "ito approximation", "gaussian stabilized"),
        default="exact transition",
    )
    parser.add_argument(
        "--no-crepe-refresh-reference",
        action="store_false",
        dest="crepe_refresh_reference",
    )
    parser.add_argument("--crepe-local-exploration", action="store_true")
    parser.add_argument("--crepe-local-steps", type=int, default=1)
    parser.add_argument("--crepe-local-scale", type=float, default=1.0)
    args = parser.parse_args()

    positive = {
        "batch size": args.batch_size,
        "number of steps": args.n_steps,
        "reference size": args.n_reference,
        "best-of-N members": args.best_of_n_members,
        "repeats": args.repeats,
        "FK chains": args.fk_chains,
        "FK temperature count": args.fk_temperature_count,
        "FK check interval": args.fk_check_every,
        "DAS particles": args.das_particles,
        "RpCN chains": args.rpcn_chains,
        "RpCN updates": args.rpcn_updates,
        "RpCN check interval": args.rpcn_check_every,
        "CREPE samples": args.crepe_num_samples,
        "CREPE thinning": args.crepe_thinning,
        "CREPE replicas": args.crepe_replicas,
        "CREPE substeps": args.crepe_substeps,
        "CREPE local steps": args.crepe_local_steps,
    }
    for name, value in positive.items():
        if value < 1:
            parser.error(f"{name} must be positive")
    if args.n_steps < 2:
        parser.error("number of steps must be at least 2")
    if args.crepe_replicas < 2:
        parser.error("CREPE replicas must be at least 2")
    if args.rpcn_chains < 2:
        parser.error("RpCN chains must be at least 2")
    if args.warmups < 0 or args.crepe_burn_in < 0:
        parser.error("warmups and CREPE burn-in must be nonnegative")
    if not 0 <= args.crepe_time_min < args.crepe_time_max <= 1:
        parser.error("CREPE requires 0 <= time-min < time-max <= 1")
    if args.crepe_karras_rho <= 0:
        parser.error("CREPE Karras rho must be positive")
    if args.crepe_local_scale < 0:
        parser.error("CREPE local scale must be nonnegative")
    if args.das_proposal_scale < 0:
        parser.error("DAS proposal scale must be nonnegative")
    if not 0 <= args.rpcn_burn_in_frac <= 1:
        parser.error("RpCN burn-in fraction must be between 0 and 1")
    return args


def reward_fn(x):
    if x.ndim < 1 or x.shape[-1] != 1:
        raise ValueError("reward_fn expects states with shape (..., 1)")

    def reward_gauss(value, scale=0.5):
        return torch.exp(-((value / scale).square()))

    def gauss_low_high(value, mean):
        return 1.6615 * reward_gauss(value - mean) - 2 * reward_gauss(
            value - mean + 1, scale=1
        )

    return (gauss_low_high(x, 3) + gauss_low_high(-x, 3)).squeeze(-1)


def build_score_model(args):
    device_name = args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA is unavailable; using CPU.")
        device_name = "cpu"
    device = torch.device(device_name)
    cfg = type("ScoreModelConfig", (), {})()
    cfg.epsilon_alpha = 1e-3
    cfg.epsilon_beta = 1e-5
    cfg.sampling_time_step_count = args.n_steps
    cfg.vmap_chunk_size = 100_000
    cfg.sampling_max_score_norm = 100.0
    cfg.renormalize_sampled_noise = False
    cfg.sample_noise_scale_std = True
    cfg.sample_noise_add_mean = True

    torch.manual_seed(args.seed)
    probability_path = Bao2024EnsembleScoreMatching(cfg)
    score_model = EnsembleScoreModel(cfg, probability_path)
    weights = torch.ones(1, device=device)
    means = torch.zeros(1, device=device)
    stds = torch.full((1,), 3.0, device=device)
    component_indices = torch.multinomial(weights, args.n_reference, replacement=True)
    reference_data = (
        means[component_indices]
        + stds[component_indices] * torch.randn(args.n_reference, device=device)
    ).unsqueeze(1)
    score_model.set_training_data(reference_data)
    timesteps = probability_path.linspace_time(args.n_steps, device=device)
    return score_model, reference_data, timesteps


def run_unsteered(args, score_model, reference_data, timesteps):
    del timesteps
    ancestor_indices = torch.randint(
        reference_data.shape[0],
        (args.batch_size,),
        device=reference_data.device,
    )
    initial_data = reference_data[ancestor_indices]
    with torch.no_grad():
        return score_model.sample(args.n_steps, initial_data)


def run_best_of_n(args, score_model, reference_data, timesteps):
    del timesteps
    candidate_count = args.best_of_n_members * args.batch_size
    ancestor_indices = torch.randint(
        reference_data.shape[0],
        (candidate_count,),
        device=reference_data.device,
    )
    initial_data = reference_data[ancestor_indices]
    with torch.no_grad():
        candidates = score_model.sample(args.n_steps, initial_data).reshape(
            args.best_of_n_members, args.batch_size, -1
        )
        return fk_steering.SelectFn.highest_reward(candidates, reward_fn)


def run_rpcn(args, score_model, reference_data, timesteps):
    kernel = rpcn.KernelPreconditionedCrankNicolson(
        1, device=reference_data.device, dtype=reference_data.dtype
    )
    steering = rpcn.ResidualMCMCSteering(
        kernel,
        torch.linspace(
            math.pi / 2 - 1e-3,
            args.rpcn_angle_min,
            args.rpcn_chains,
            device=reference_data.device,
            dtype=reference_data.dtype,
        ).sin(),
        reward_fn=reward_fn,
        expected_next_fn=score_model.expected_next_fn,
        std_fn=score_model.std_fn,
        intermediate_reward=rpcn.IntermediateReward.EXPECTED_SAMPLE,
        inverse_temperature=torch.linspace(
            0.0,
            args.rpcn_inverse_temperature,
            args.rpcn_chains,
            device=reference_data.device,
            dtype=reference_data.dtype,
        ),
        kwargs_intermediate_reward={"expected_sample_fn": score_model.get_tweedie},
    )

    def should_update(step_idx, t0, t1, reward):
        del t0, t1, reward
        return (
            step_idx >= timesteps.numel() * args.rpcn_burn_in_frac
            and step_idx % args.rpcn_check_every == 0
        )

    with torch.no_grad():
        samples = steering.run(
            timesteps,
            pcn_should_update_fn=should_update,
            pcn_update_count=args.rpcn_updates,
            batch_size=args.batch_size,
        )
    return samples[-1] if samples.ndim == 3 else samples


def run_fk(args, score_model, reference_data, timesteps):
    target_temperature = args.fk_inverse_temperature
    inverse_temperatures = (
        target_temperature
        if args.fk_temperature_count == 1
        else torch.linspace(
            0.0,
            target_temperature,
            args.fk_temperature_count,
            device=reference_data.device,
            dtype=reference_data.dtype,
        )
    )

    def propose_step(t0, t1, state):
        mean = score_model.expected_next_fn(t0, t1, state)
        if t1 > 0:
            return mean + score_model.std_fn(t0, t1) * torch.randn_like(state)
        return mean

    def adaptive_resample(time_step, ensemble_size, log_weights):
        identity = torch.arange(ensemble_size, device=log_weights.device)
        if time_step % args.fk_check_every != 0:
            did_resample = torch.tensor(False, device=log_weights.device)
            return identity, did_resample, did_resample
        probabilities = torch.softmax(torch.nan_to_num(log_weights, nan=0.0), dim=0)
        did_resample = (
            1 / probabilities.square().sum()
            < args.fk_resample_threshold * ensemble_size
        )
        ancestors = torch.where(
            did_resample,
            fk_steering.ResampleFn.do_resample(ensemble_size, probabilities),
            identity,
        )
        return ancestors, did_resample, torch.zeros_like(did_resample)

    steering = fk_steering.FeynmanKac(
        init_param_fn=lambda: torch.randn(
            1, device=reference_data.device, dtype=reference_data.dtype
        ),
        reward_fn=reward_fn,
        propose_fn=propose_step,
        resample_fn=adaptive_resample,
        potential=fk_steering.Potential(args.fk_potential),
        intermediate_reward=fk_steering.IntermediateReward.EXPECTED_SAMPLE,
        inverse_temperature=inverse_temperatures,
        kwargs_intermediate_reward={"expected_sample_fn": score_model.get_tweedie},
    )
    with torch.no_grad():
        particles, log_weights = steering.run(
            timesteps, args.fk_chains, batch_size=args.batch_size
        )
        if args.fk_temperature_count > 1:
            particles = particles[-1]
            log_weights = log_weights[-1]
        return fk_steering.SelectFn.weighted_sample(particles, log_weights)


def ssp_resample_indices(weights, output_size):
    if weights.ndim != 2 or weights.shape[1] == 0:
        raise ValueError("SSP weights must have shape (batch, particles)")
    weights = weights / weights.sum(dim=1, keepdim=True)
    batch_count, particle_count = weights.shape
    scaled_weights = output_size * weights
    offspring_counts = scaled_weights.floor().to(torch.long)
    fractional_counts = scaled_weights - offspring_counts
    if particle_count == 1:
        return torch.zeros(
            batch_count,
            output_size,
            device=weights.device,
            dtype=torch.long,
        )

    uniforms = torch.rand(
        batch_count,
        particle_count - 1,
        device=weights.device,
        dtype=weights.dtype,
    )
    rows = torch.arange(batch_count, device=weights.device)
    i = torch.zeros(batch_count, device=weights.device, dtype=torch.long)
    j = torch.ones(batch_count, device=weights.device, dtype=torch.long)
    for step_idx in range(particle_count - 1):
        fraction_i = fractional_counts[rows, i]
        fraction_j = fractional_counts[rows, j]
        delta_i = torch.minimum(fraction_j, 1 - fraction_i)
        delta_j = torch.minimum(fraction_i, 1 - fraction_j)
        delta_sum = delta_i + delta_j
        increase_i_probability = torch.where(
            delta_sum > 0, delta_i / delta_sum, 0
        )
        swap = uniforms[:, step_idx] < increase_i_probability
        active_i = torch.where(swap, j, i)
        active_j = torch.where(swap, i, j)
        active_delta = torch.where(swap, delta_j, delta_i)
        increase_fraction = (
            fractional_counts[rows, active_j]
            < 1 - fractional_counts[rows, active_i]
        )
        fractional_counts[rows, active_i] += torch.where(
            increase_fraction, active_delta, 0
        )
        fractional_counts[rows, active_j] -= torch.where(
            increase_fraction, 0, active_delta
        )
        offspring_counts[rows, active_i] += ~increase_fraction
        next_index = torch.full_like(i, step_idx + 2)
        i = torch.where(increase_fraction, active_i, next_index)
        j = torch.where(increase_fraction, next_index, active_j)

    offspring_total = offspring_counts.sum(dim=1)
    last_index = torch.where(j == particle_count, i, j)
    add_last = (offspring_total == output_size - 1) & (
        fractional_counts[rows, last_index] > 0.99
    )
    offspring_counts[rows, last_index] += add_last
    cumulative_counts = offspring_counts.cumsum(dim=1)
    offspring_positions = torch.arange(output_size, device=weights.device).expand(
        batch_count, -1
    ).contiguous()
    return torch.searchsorted(
        cumulative_counts.contiguous(), offspring_positions, right=True
    )


def run_das(args, score_model, reference_data, timesteps):
    batch_count = args.batch_size
    particle_count = args.das_particles
    steps = torch.arange(
        timesteps.numel(), device=reference_data.device, dtype=reference_data.dtype
    )
    tempering = ((1 + args.das_tempering_gamma) ** steps - 1).clamp(max=1)
    tempering = torch.cat((tempering[:-1], torch.ones_like(tempering[-1:])))

    def tweedie(time, latents, score=None):
        if score is None:
            score = score_model._score_particles(time, latents)
        return (
            latents + score_model.prob_path.beta(time).square() * score
        ) / score_model.prob_path.alpha(time)

    def log_gaussian(value, mean, std):
        variance = std.square()
        dimension = value.shape[-1]
        squared_distance = (value - mean).square().sum(dim=-1)
        return -0.5 * (
            squared_distance / variance
            + dimension * torch.log(2 * torch.pi * variance)
        )

    particles = torch.randn(
        batch_count,
        particle_count,
        1,
        device=reference_data.device,
        dtype=reference_data.dtype,
    )
    uniform_log_weight = -torch.log(reference_data.new_tensor(float(particle_count)))
    log_weights = torch.full(
        (batch_count, particle_count), uniform_log_weight, device=reference_data.device
    )

    for step_idx, time_pair in enumerate(timesteps.unfold(0, 2, 1)):
        t0, t1 = time_pair
        normalized = log_weights - torch.logsumexp(log_weights, dim=1, keepdim=True)
        probabilities = normalized.exp()
        ess = 1 / probabilities.square().sum(dim=1)
        should_resample = ess < args.das_resample_threshold * particle_count
        identity = torch.arange(
            particle_count, device=reference_data.device
        ).expand(batch_count, -1)
        resampled = ssp_resample_indices(probabilities, particle_count)
        ancestors = torch.where(should_resample[:, None], resampled, identity)
        particles = torch.gather(particles, 1, ancestors[:, :, None])
        resampled_log_weights = torch.where(
            should_resample[:, None], uniform_log_weight, normalized
        )

        current = particles.reshape(-1, 1).detach().requires_grad_(True)
        current_score = score_model._score_particles(t0, current)
        diffusion = score_model.prob_path.g(t0)
        base_mean = current + (t1 - t0) * (
            score_model.prob_path.f(t0, current)
            - diffusion.square() * current_score
        )
        current_reward = reward_fn(tweedie(t0, current, current_score)).reshape(-1)
        reward_gradient = torch.autograd.grad(current_reward.sum(), current)[0]
        transition_std = score_model.std_fn(t0, t1)
        next_temperature = tempering[step_idx + 1]
        current_temperature = tempering[step_idx]
        proposal_mean = (
            base_mean
            + args.das_proposal_scale
            * transition_std.square()
            * next_temperature
            * args.das_inverse_temperature
            * reward_gradient
        )
        with torch.no_grad():
            following = proposal_mean + transition_std * torch.randn_like(current)
            following_reward = reward_fn(tweedie(t1, following)).reshape(-1)
            increment = (
                log_gaussian(following, base_mean, transition_std)
                - log_gaussian(following, proposal_mean, transition_std)
                + next_temperature * args.das_inverse_temperature * following_reward
                - current_temperature * args.das_inverse_temperature * current_reward
            )
            log_weights = resampled_log_weights + increment.reshape(
                batch_count, particle_count
            )
            particles = following.reshape(batch_count, particle_count, 1)

    with torch.no_grad():
        probabilities = torch.softmax(log_weights, dim=1)
        indices = torch.multinomial(probabilities, 1)
        return torch.gather(particles, 1, indices[:, :, None]).squeeze(1)


def build_crepe_times(args, score_model, reference_data):
    point_count = (args.crepe_replicas - 1) * args.crepe_substeps + 1
    ramp = torch.linspace(
        0,
        1,
        point_count,
        device=reference_data.device,
        dtype=reference_data.dtype,
    )
    if args.crepe_time_spacing == "linear":
        fine_times = args.crepe_time_min + (
            args.crepe_time_max - args.crepe_time_min
        ) * ramp
    else:
        time_min = reference_data.new_tensor(args.crepe_time_min)
        time_max = reference_data.new_tensor(args.crepe_time_max)

        def effective_sigma(time):
            return score_model.prob_path.beta(time) / score_model.prob_path.alpha(time)

        sigma_min = effective_sigma(time_min)
        sigma_max = effective_sigma(time_max)
        sigma_grid = (
            sigma_min.pow(1 / args.crepe_karras_rho)
            + (
                sigma_max.pow(1 / args.crepe_karras_rho)
                - sigma_min.pow(1 / args.crepe_karras_rho)
            )
            * ramp
        ).pow(args.crepe_karras_rho)
        lower = torch.full_like(sigma_grid, args.crepe_time_min)
        upper = torch.full_like(sigma_grid, args.crepe_time_max)
        for _ in range(48):
            midpoint = (lower + upper) / 2
            move_lower = effective_sigma(midpoint) < sigma_grid
            lower = torch.where(move_lower, midpoint, lower)
            upper = torch.where(move_lower, upper, midpoint)
        fine_times = torch.cat(
            (time_min[None], ((lower + upper) / 2)[1:-1], time_max[None])
        )
    return fine_times, fine_times[:: args.crepe_substeps]


def run_crepe(args, score_model, reference_data, timesteps):
    del timesteps
    if args.crepe_refresh_reference and args.crepe_reward_reference != 0:
        raise ValueError(
            "Exact reference refresh requires crepe_reward_reference=0"
        )
    fine_times, replica_times = build_crepe_times(args, score_model, reference_data)
    time_min = replica_times[0]
    time_max = replica_times[-1]

    def schedule_value(time):
        fraction = ((time_max - time) / (time_max - time_min)).clamp(0, 1)
        return args.crepe_reward_reference + (
            args.crepe_reward_target - args.crepe_reward_reference
        ) * fraction.pow(args.crepe_reward_power)

    def tweedie(time, state):
        score = score_model._score_particles(time, state)
        return (
            state + score_model.prob_path.beta(time).square() * score
        ) / score_model.prob_path.alpha(time)

    def log_reward(time, state):
        return (
            args.crepe_inverse_temperature
            * schedule_value(time)
            * reward_fn(tweedie(time, state))
        )

    def reward_gradient(time, state):
        gradient = torch.func.grad(lambda value: log_reward(time, value).sum())(state)
        if args.crepe_gradient_clip > 0:
            gradient = gradient.clamp(
                -args.crepe_gradient_clip, args.crepe_gradient_clip
            )
        return gradient.detach()

    def model_ttr_drift(time, state):
        return score_model.prob_path.f(time, state)

    def model_rtt_drift(time, state):
        diffusion = score_model.prob_path.g(time)
        return score_model.prob_path.f(time, state) - diffusion.square() * score_model._score_particles(time, state)

    def gaussian_log_prob(value, mean, variance):
        log_probability = -0.5 * (
            (value - mean).square() / variance
            + torch.log(2 * torch.pi * variance)
        )
        return log_probability.flatten(1).sum(dim=1)

    def transition_means(
        path_times, path_states, ttr_means=None, rtt_means=None
    ):
        computed_ttr_means = []
        computed_rtt_means = []
        for index in range(path_times.numel() - 1):
            t0, t1 = path_times[index : index + 2]
            dt = t1 - t0
            lower, upper = path_states[index : index + 2]
            if ttr_means is None:
                computed_ttr_means.append(lower + dt * model_ttr_drift(t0, lower))
            if rtt_means is None:
                computed_rtt_means.append(upper - dt * model_rtt_drift(t1, upper))
        if ttr_means is None:
            ttr_means = torch.stack(computed_ttr_means)
        if rtt_means is None:
            rtt_means = torch.stack(computed_rtt_means)
        return ttr_means, rtt_means

    def transition_logs(path_times, path_states, ttr_means, rtt_means):
        ttr_logs = []
        rtt_logs = []
        for index in range(path_times.numel() - 1):
            t0, t1 = path_times[index : index + 2]
            dt = t1 - t0
            lower, upper = path_states[index : index + 2]
            ttr_variance = dt * score_model.prob_path.g(t0).square()
            rtt_variance = dt * score_model.prob_path.g(t1).square()
            ttr_logs.append(
                gaussian_log_prob(upper, ttr_means[index], ttr_variance)
            )
            rtt_logs.append(
                gaussian_log_prob(lower, rtt_means[index], rtt_variance)
            )
        return torch.stack(ttr_logs), torch.stack(rtt_logs)

    def stabilizing_rtt_drift(time, state):
        alpha = score_model.prob_path.alpha(time)
        beta = score_model.prob_path.beta(time)
        score = -state / (alpha.square() + beta.square())
        diffusion = score_model.prob_path.g(time)
        return score_model.prob_path.f(time, state) - diffusion.square() * score

    def stabilizing_log_prob(time, state):
        variance = (
            score_model.prob_path.alpha(time).square()
            + score_model.prob_path.beta(time).square()
        )
        return gaussian_log_prob(state, torch.zeros_like(state), variance)

    def path_log_rne(path_times, path_states, ttr_means=None, rtt_means=None):
        if args.crepe_rne_method == "ito approximation":
            terms = []
            for index in range(path_times.numel() - 1):
                times = path_times[index : index + 2]
                path = path_states[index : index + 2]
                ttr_drift = torch.stack(
                    [
                        model_ttr_drift(times[0], path[0]),
                        model_ttr_drift(times[1], path[1]),
                    ]
                )
                rtt_drift = torch.stack(
                    [
                        model_rtt_drift(times[0], path[0]),
                        model_rtt_drift(times[1], path[1]),
                    ]
                )
                diffusion = torch.stack(
                    [
                        score_model.prob_path.g(times[0]),
                        score_model.prob_path.g(times[1]),
                    ]
                )
                terms.append(
                    crepe.PathLogRatio.from_ito_integrals(
                        times, path, ttr_drift, rtt_drift, diffusion
                    )
                )
            return torch.stack(terms).sum(dim=0)

        ttr_means, rtt_means = transition_means(
            path_times, path_states, ttr_means, rtt_means
        )
        ttr_logs, rtt_logs = transition_logs(
            path_times, path_states, ttr_means, rtt_means
        )
        if args.crepe_rne_method == "exact transition":
            return crepe.PathLogRatio.from_transition_log_probabilities(
                ttr_logs, rtt_logs
            )

        stable_rtt_means = []
        for index in range(path_times.numel() - 1):
            t0, t1 = path_times[index : index + 2]
            dt = t1 - t0
            _, upper = path_states[index : index + 2]
            stable_rtt_means.append(
                upper - dt * stabilizing_rtt_drift(t1, upper)
            )
        stable_ttr_logs, stable_rtt_logs = transition_logs(
            path_times,
            path_states,
            ttr_means,
            torch.stack(stable_rtt_means),
        )
        return crepe.PathLogRatio.from_reference_stabilized_transition_log_probabilities(
            ttr_logs,
            rtt_logs,
            stable_ttr_logs,
            stable_rtt_logs,
            stabilizing_log_prob(path_times[0], path_states[0]),
            stabilizing_log_prob(path_times[-1], path_states[-1]),
        )

    def summarize_path(path_times, path_states, ttr_means=None, rtt_means=None):
        model_rne = path_log_rne(path_times, path_states)
        proposal_rne = path_log_rne(
            path_times, path_states, ttr_means, rtt_means
        )
        target_ratio = crepe.TargetLogRatio.reward_tilt(
            model_rne,
            log_reward(path_times[0], path_states[0]),
            log_reward(path_times[-1], path_states[-1]),
        )
        return target_ratio.detach(), proposal_rne.detach()

    def edge_grid(t0, t1):
        del t1
        start = torch.argmin((fine_times - t0).abs())
        offsets = torch.arange(args.crepe_substeps + 1, device=fine_times.device)
        return fine_times[start + offsets]

    def target_to_reference_path(t0, t1, lower):
        path_times = edge_grid(t0, t1)
        states = [lower.detach()]
        ttr_means = []
        current = lower.detach()
        for index in range(path_times.numel() - 1):
            now, following = path_times[index : index + 2]
            dt = following - now
            mean = current + dt * model_ttr_drift(now, current)
            current = (
                mean
                + dt.sqrt() * score_model.prob_path.g(now) * torch.randn_like(current)
            ).detach()
            ttr_means.append(mean)
            states.append(current)
        target_ratio, proposal_rne = summarize_path(
            path_times, torch.stack(states), ttr_means=torch.stack(ttr_means)
        )
        return current, target_ratio, proposal_rne

    def reference_to_target_path(t0, t1, upper):
        path_times = edge_grid(t0, t1)
        reverse_states = [upper.detach()]
        reverse_rtt_means = []
        current = upper.detach()
        for index in range(path_times.numel() - 1, 0, -1):
            now = path_times[index]
            following = path_times[index - 1]
            dt = now - following
            mean = current - dt * model_rtt_drift(now, current)
            current = (
                mean
                + dt.sqrt() * score_model.prob_path.g(now) * torch.randn_like(current)
            ).detach()
            reverse_rtt_means.append(mean)
            reverse_states.append(current)
        target_ratio, proposal_rne = summarize_path(
            path_times,
            torch.stack(reverse_states[::-1]),
            rtt_means=torch.stack(reverse_rtt_means[::-1]),
        )
        return current, target_ratio, proposal_rne

    def reference_sample(time):
        index = torch.randint(reference_data.shape[0], (), device=reference_data.device)
        clean = reference_data[index]
        return (
            score_model.prob_path.alpha(time) * clean
            + score_model.prob_path.beta(time) * torch.randn_like(clean)
        )

    def initialize_path(times):
        if args.crepe_initialization == "independent marginals":
            indices = torch.randint(
                reference_data.shape[0],
                (times.numel(),),
                device=reference_data.device,
            )
            clean = reference_data[indices]
            return (
                score_model.prob_path.alpha(times)[:, None] * clean
                + score_model.prob_path.beta(times)[:, None] * torch.randn_like(clean)
            )
        if args.crepe_initialization == "shared clean ancestor":
            index = torch.randint(reference_data.shape[0], (), device=reference_data.device)
            clean = reference_data[index]
            return (
                score_model.prob_path.alpha(times)[:, None] * clean
                + score_model.prob_path.beta(times)[:, None]
                * torch.randn(
                    times.numel(),
                    clean.numel(),
                    device=reference_data.device,
                    dtype=reference_data.dtype,
                )
            )

        current = reference_sample(times[-1])
        reverse_states = [current]
        for index in range(fine_times.numel() - 1, 0, -1):
            now = fine_times[index]
            following = fine_times[index - 1]
            dt = now - following
            current = (
                current
                - dt * model_rtt_drift(now, current[None])[0]
                + dt.sqrt() * score_model.prob_path.g(now) * torch.randn_like(current)
            ).detach()
            if (index - 1) % args.crepe_substeps == 0:
                reverse_states.append(current)
        return torch.stack(reverse_states[::-1])

    def local_exploration(time, state):
        index = torch.argmin((fine_times - time).abs()).clamp(
            0, fine_times.numel() - 2
        )
        dt = fine_times[index + 1] - fine_times[index]
        result = state.detach()
        for _ in range(args.crepe_local_steps):
            target_score = score_model._score_particles(time, result) + reward_gradient(
                time, result
            )
            step_size = (
                0.5
                * score_model.prob_path.g(time).square()
                * dt
                * args.crepe_local_scale
            )
            result = (
                result
                + step_size * target_score
                + (2 * step_size).sqrt() * torch.randn_like(result)
            ).detach()
        return result

    sampler = crepe.CREPE(
        init_path_fn=initialize_path,
        target_to_reference_path_fn=target_to_reference_path,
        reference_to_target_path_fn=reference_to_target_path,
        local_exploration_fn=(
            local_exploration if args.crepe_local_exploration else None
        ),
        refresh_reference_fn=(
            reference_sample if args.crepe_refresh_reference else None
        ),
    )
    iteration_count = args.crepe_burn_in + (
        args.crepe_num_samples - 1
    ) * args.crepe_thinning + 1
    result = sampler.run(
        replica_times,
        iteration_count,
        batch_size=args.batch_size,
        should_save_sample_fn=functools.partial(
            crepe.SaveSampleFn.burn_in_and_thin,
            burn_in=args.crepe_burn_in,
            thin=args.crepe_thinning,
        ),
    )
    output = result.samples.detach()

    if time_min.item() > 0:
        if args.crepe_time_spacing == "linear":
            interval = fine_times[1] - fine_times[0]
            continuation_steps = math.ceil(float(time_min / interval))
            continuation_times = torch.linspace(
                time_min,
                0,
                continuation_steps + 1,
                device=fine_times.device,
                dtype=fine_times.dtype,
            )
        else:
            rho = args.crepe_karras_rho

            def effective_sigma(time):
                return score_model.prob_path.beta(time) / score_model.prob_path.alpha(time)

            fine_schedule = effective_sigma(fine_times).pow(1 / rho)
            interval = fine_schedule[1] - fine_schedule[0]
            zero_time = fine_times.new_zeros(())
            zero_schedule = effective_sigma(zero_time).pow(1 / rho)
            min_schedule = fine_schedule[0]
            continuation_steps = math.ceil(
                float((min_schedule - zero_schedule) / interval)
            )
            continuation_schedule = torch.linspace(
                min_schedule,
                zero_schedule,
                continuation_steps + 1,
                device=fine_times.device,
                dtype=fine_times.dtype,
            )
            continuation_sigma = continuation_schedule.pow(rho)
            lower = torch.zeros_like(continuation_sigma)
            upper = torch.full_like(continuation_sigma, float(time_min.item()))
            for _ in range(48):
                midpoint = (lower + upper) / 2
                move_lower = effective_sigma(midpoint) < continuation_sigma
                lower = torch.where(move_lower, midpoint, lower)
                upper = torch.where(move_lower, upper, midpoint)
            continuation_times = torch.cat(
                (
                    time_min.reshape(1),
                    ((lower + upper) / 2)[1:-1],
                    zero_time.reshape(1),
                )
            )

        if continuation_times.numel() != continuation_steps + 1:
            raise ValueError("CREPE continuation has the wrong number of points")
        if not bool(torch.isfinite(continuation_times).all()):
            raise ValueError("CREPE continuation times must be finite")
        if not bool((continuation_times[:-1] > continuation_times[1:]).all()):
            raise ValueError("CREPE continuation times must be strictly decreasing")
        output_shape = output.shape
        continuation = output.reshape(-1, 1)
        for index in range(continuation_times.numel() - 1):
            now = continuation_times[index]
            following = continuation_times[index + 1]
            dt = now - following
            continuation = (
                continuation
                - dt * model_rtt_drift(now, continuation)
                + dt.sqrt() * score_model.prob_path.g(now) * torch.randn_like(continuation)
            ).detach()
        output = continuation.reshape(output_shape)
    return output.reshape(-1, 1)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_method(name, runner, args, score_model, reference_data, timesteps):
    timings = []
    output_shape = None
    for run_index in range(args.warmups + args.repeats):
        torch.manual_seed(args.seed)
        synchronize(reference_data.device)
        start = perf_counter()
        output = runner(args, score_model, reference_data, timesteps)
        synchronize(reference_data.device)
        elapsed = perf_counter() - start
        output_shape = tuple(output.shape)
        if run_index >= args.warmups:
            timings.append(elapsed)
    return {
        "method": name,
        "mean": statistics.fmean(timings),
        "std": statistics.stdev(timings) if len(timings) > 1 else 0.0,
        "min": min(timings),
        "shape": output_shape,
    }


def main():
    args = parse_args()
    score_model, reference_data, timesteps = build_score_model(args)
    runners = {
        "unsteered": run_unsteered,
        "best_of_n": run_best_of_n,
        "rpcn": run_rpcn,
        "das": run_das,
        "fk": run_fk,
        "crepe": run_crepe,
    }
    results = [
        benchmark_method(
            method,
            runners[method],
            args,
            score_model,
            reference_data,
            timesteps,
        )
        for method in dict.fromkeys(args.methods)
    ]
    print(
        f"device={reference_data.device} batch_size={args.batch_size} "
        f"n_steps={args.n_steps} warmups={args.warmups} repeats={args.repeats}"
    )
    print(f"{'method':<10} {'mean (s)':>12} {'std (s)':>12} {'min (s)':>12}  output")
    print("-" * 68)
    for result in results:
        print(
            f"{result['method']:<10} {result['mean']:>12.6f} "
            f"{result['std']:>12.6f} {result['min']:>12.6f}  {result['shape']}"
        )


if __name__ == "__main__":
    main()
