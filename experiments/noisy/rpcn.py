import argparse
import os
import time
from pathlib import Path

import torch
from ttg import (
    FKDiff,
    Guided,
    Never,
    PCN,
    DDIMStep,
    SequentialSweep,
    ladder,
    make_evaluate_torch,
    normalized,
    on_x0,
    smc,
    x0_from_eps,
)
from ttg.backend import TorchBackend

from experiments.common import (
    DEFAULT_DATA_DIR,
    DEFAULT_MEASUREMENT_ROOT,
    DEFAULT_OUTPUT_ROOT,
    DiscreteVPSchedule,
    degradation_sha256,
    effective_sample_seed,
    measurement_metrics,
    output_dir,
    prepare_problem,
)
from utils import load_image_by_id, load_model_and_sampler, normalize_tensor, plot_samples


def parse_args():
    parser = argparse.ArgumentParser(
        description="Solve a noisy FFHQ inverse problem with the packaged RpCN solver."
    )
    parser.add_argument("--img_id", type=int, default=49000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--operator_seed",
        type=int,
        default=42,
        help="Global NumPy seed used to reproduce the DAPS degradation.",
    )
    parser.add_argument(
        "--operator",
        choices=("down_sampling", "gaussian_blur", "motion_blur", "inpainting_random"),
        default="motion_blur",
    )
    parser.add_argument("--noise_sigma", type=float, default=0.05)
    parser.add_argument("--measurement_seed", type=int, default=42)
    parser.add_argument(
        "--measurement_root", type=Path, default=DEFAULT_MEASUREMENT_ROOT
    )
    parser.add_argument("--n_chains", type=int, default=3)
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--resampling_frequency", type=int, default=25)
    parser.add_argument("--pcn_update_count", type=int, default=10)
    parser.add_argument(
        "--max_inverse_temperature",
        type=float,
        default=1.0,
        help="Inverse temperature of the coldest replica.",
    )
    parser.add_argument(
        "--guidance_step_size",
        type=float,
        default=1.0,
        help="Euclidean length of each normalized-gradient guidance shift.",
    )
    parser.add_argument(
        "--guidance_start",
        type=float,
        default=0.5,
        help="Denoising progress in [0, 1] at which normalized guidance starts.",
    )
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    if args.noise_sigma <= 0:
        parser.error("--noise_sigma must be positive")
    if args.n_chains < 2:
        parser.error("--n_chains must be at least 2")
    if args.diffusion_steps < 2 or args.diffusion_steps > 1000:
        parser.error("--diffusion_steps must be between 2 and 1000")
    if args.resampling_frequency < 1:
        parser.error("--resampling_frequency must be at least 1")
    if args.pcn_update_count < 1:
        parser.error("--pcn_update_count must be at least 1")
    if args.max_inverse_temperature <= 0:
        parser.error("--max_inverse_temperature must be positive")
    if args.guidance_step_size < 0:
        parser.error("--guidance_step_size must be nonnegative")
    if not 0 <= args.guidance_start <= 1:
        parser.error("--guidance_start must be in [0, 1]")
    return args


def main():
    args = parse_args()
    started_at = time.time()
    device = torch.device("cuda")
    dtype = torch.float32
    data_shape = (3, 256, 256)
    effective_seed = effective_sample_seed(args.seed, args.img_id)
    torch.manual_seed(effective_seed)

    model, ddpm = load_model_and_sampler()
    reference = load_image_by_id(
        args.data_dir, args.img_id, device=device, save=True
    ).detach().to(dtype=dtype)
    operator, measurement, cache_path = prepare_problem(
        reference,
        args.operator,
        operator_seed=args.operator_seed,
        noisy=True,
        noise_sigma=args.noise_sigma,
        measurement_seed=args.measurement_seed,
        image_id=args.img_id,
        measurement_root=args.measurement_root,
        create_measurement=False,
    )
    print(f"Measurement cache: {cache_path}")
    fingerprint = degradation_sha256(operator)
    if fingerprint is not None:
        print(f"Degradation SHA256: {fingerprint}")

    schedule = DiscreteVPSchedule(
        torch.as_tensor(ddpm.alphas_cumprod, device=device, dtype=dtype)
    )

    def epsilon_network(x, t):
        timestep = torch.round(t).to(torch.long)
        output = model(x, ddpm._scale_timesteps(timestep))
        return output[:, : x.shape[1]]

    def reward(x0):
        return operator.log_likelihood(x0, measurement)

    def to_x0_eps(x, t, eps, schedule):
        x0 = x0_from_eps(x, t, eps, schedule).clamp(-1, 1)
        return x0, eps

    evaluate = make_evaluate_torch(
        epsilon_network, to_x0_eps, schedule, on_x0(reward)
    )

    def guidance_step_length(t):
        progress = 1.0 - t / (schedule.num_timesteps - 1)
        enabled = progress >= args.guidance_start
        return torch.full_like(t, args.guidance_step_size) * enabled

    backend = TorchBackend(device=device)
    betas = args.max_inverse_temperature * torch.linspace(
        0.0,
        1.0,
        args.n_chains,
        device=device,
        dtype=dtype,
    ).pow(2.0)
    pcn_step_sizes = 0.5 * torch.ones(
        args.n_chains,
        device=device,
        dtype=dtype,
    )
    base_kernel = DDIMStep(schedule, eta=1.0)
    guided_kernel = Guided(base_kernel, normalized(guidance_step_length))
    move = PCN(
        n_proposals=args.pcn_update_count,
        step_size=ladder(pcn_step_sizes),
        pairs=SequentialSweep(),
    )
    start = schedule.prior(
        (1, args.n_chains), data_shape, xp=backend.xp, dtype=dtype, device=device
    )
    u_grid = torch.linspace(
        1.0, 0.0, args.diffusion_steps, device=device, dtype=dtype
    )

    print(
        f"Generating noisy reconstruction for FFHQ image {args.img_id} "
        f"under operator {args.operator} with sigma {args.noise_sigma:g}"
    )
    result = smc(
        start,
        u_grid,
        backend=backend,
        evaluate=evaluate,
        schedule=schedule,
        target=guided_kernel,
        proposal=guided_kernel,
        potential=FKDiff(ladder(betas)),
        policy=Never(),
        move=move,
        move_at=lambda step: (step > 0) & (step % args.resampling_frequency == 0),
        key=backend.key(effective_seed),
    )

    final_sample = result.final.x0[:, -1]
    metrics = measurement_metrics(
        operator, final_sample, measurement, noise_sigma=args.noise_sigma
    )
    print(f"Measurement MSE: {metrics['measurement_mse'].item():.8f}")
    print(f"Normalized residual: {metrics['normalized_residual'].item():.8f}")
    print(f"Negative log-likelihood: {metrics['negative_log_likelihood'].item():.8f}")
    print(f"Total time: {time.time() - started_at:.2f} seconds")

    if args.save:
        save_dir = output_dir(
            args.output_root,
            "noisy",
            "RpCN",
            args.operator,
            args.seed,
            args.noise_sigma,
        )
        os.makedirs(save_dir, exist_ok=True)
        torch.save(final_sample, save_dir / f"{args.img_id}.pt")

    if args.plot:
        os.makedirs("generated/figures", exist_ok=True)
        plot_samples(
            normalize_tensor(final_sample),
            args.img_id,
            f"{args.operator}_noisy_rpcn_sigma_{args.noise_sigma:g}",
            effective_seed,
        )


if __name__ == "__main__":
    main()
