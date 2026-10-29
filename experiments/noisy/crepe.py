import argparse
import os
import time
from pathlib import Path

from experiments.common import (
    DEFAULT_DATA_DIR,
    DEFAULT_MEASUREMENT_ROOT,
    DEFAULT_OUTPUT_ROOT,
    OPERATORS,
)

def parse_levels(value):
    try:
        levels = tuple(float(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("levels must be comma-separated numbers") from error
    if len(levels) < 2:
        raise argparse.ArgumentTypeError("at least two levels are required")
    if levels[0] < 0 or levels[-1] != 1.0:
        raise argparse.ArgumentTypeError("levels must start at or above 0 and end at 1")
    if any(left >= right for left, right in zip(levels, levels[1:])):
        raise argparse.ArgumentTypeError("levels must be strictly increasing")
    return levels


def parse_args():
    parser = argparse.ArgumentParser(
        description="Solve a noisy FFHQ inverse problem with the packaged CREPE solver."
    )
    parser.add_argument("--img_id", type=int, default=49000)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--operator_seed", type=int, default=42)
    parser.add_argument("--operator", choices=OPERATORS, default="motion_blur")
    parser.add_argument("--noise_sigma", type=float, default=0.05)
    parser.add_argument("--measurement_seed", type=int, default=42)
    parser.add_argument(
        "--measurement_root", type=Path, default=DEFAULT_MEASUREMENT_ROOT
    )
    parser.add_argument("--n_chains", type=int, default=1)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--thin", type=int, default=1)
    parser.add_argument(
        "--levels", type=parse_levels, default=parse_levels("0,0.25,0.5,0.75,1")
    )
    parser.add_argument("--tail_steps", type=int, default=50)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--lambda_rho", type=float, default=0.5)
    parser.add_argument(
        "--guidance",
        choices=("tilt", "clipped_tilt", "normalized", "none"),
        default="clipped_tilt",
    )
    parser.add_argument("--proposal_scale", type=float, default=0.3)
    parser.add_argument("--max_shift_norm", type=float, default=1.0)
    parser.add_argument(
        "--clamp_x0", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--langevin_scale", type=float, default=0.01)
    parser.add_argument("--local_move", action="store_true")
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    for name in ("n_chains", "n_blocks", "thin", "tail_steps"):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be at least 1")
    if args.noise_sigma <= 0:
        parser.error("--noise_sigma must be positive")
    if args.beta <= 0:
        parser.error("--beta must be positive")
    if args.lambda_rho <= 0:
        parser.error("--lambda_rho must be positive")
    if args.langevin_scale <= 0:
        parser.error("--langevin_scale must be positive")
    if args.proposal_scale < 0:
        parser.error("--proposal_scale must be nonnegative")
    if args.max_shift_norm <= 0:
        parser.error("--max_shift_norm must be positive")

    return args


def main():
    args = parse_args()

    import torch
    from ttg import (
        CREPEConfig,
        DDIMStep,
        Guided,
        Langevin,
        LangevinMove,
        Noising,
        crepe_sample,
        level_lambda,
        make_evaluate_torch,
        normalized,
        on_x0,
        tilt,
        x0_from_eps,
    )
    from ttg.backend import TorchBackend

    from experiments.common import (
        DiscreteVPSchedule,
        effective_sample_seed,
        measurement_metrics,
        output_dir,
        prepare_problem,
    )
    from utils import (
        load_image_by_id,
        load_model_and_sampler,
        normalize_tensor,
        plot_samples,
    )

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

    schedule = DiscreteVPSchedule(
        torch.as_tensor(ddpm.alphas_cumprod, device=device, dtype=dtype)
    )

    def epsilon_network(x, t):
        timestep = torch.round(t).to(torch.long).clamp(0, schedule.num_timesteps - 1)
        output = model(x, ddpm._scale_timesteps(timestep))
        return output[:, : x.shape[1]]

    def reward(x0):
        return operator.log_likelihood(x0, measurement)

    def to_x0_eps(x, t, eps, sch):
        x0 = x0_from_eps(x, t, eps, sch)
        if args.clamp_x0:
            x0 = x0.clamp(-1, 1)
        return x0, eps

    evaluate = make_evaluate_torch(
        epsilon_network,
        to_x0_eps,
        schedule,
        on_x0(reward),
    )
    backend = TorchBackend(device=device)
    u_levels = torch.tensor(args.levels, device=device, dtype=dtype)
    lam = level_lambda(schedule.time(u_levels), beta=args.beta, rho=args.lambda_rho)
    model_noising = Noising(schedule)
    model_denoising = DDIMStep(schedule, eta=1.0)

    if args.guidance == "tilt":
        proposal_denoising = Guided(
            model_denoising, tilt(lam, factor=args.proposal_scale)
        )
    elif args.guidance == "clipped_tilt":

        def clipped_tilt(t, grad, var):
            raw_scale = args.proposal_scale * var * lam(t)
            event_dims = tuple(range(var.ndim, grad.ndim))
            grad_norm = torch.linalg.vector_norm(grad, dim=event_dims)
            capped_scale = args.max_shift_norm / grad_norm.clamp_min(1e-12)
            return torch.minimum(raw_scale, capped_scale)

        proposal_denoising = Guided(model_denoising, clipped_tilt)
    elif args.guidance == "normalized":

        def normalized_step(t):
            return args.proposal_scale * lam(t) / args.beta

        proposal_denoising = Guided(model_denoising, normalized(normalized_step))
    else:
        proposal_denoising = model_denoising

    local_move = None
    local_levels = ()
    if args.local_move:
        local_move = LangevinMove(
            Guided(
                Langevin(lambda t: args.langevin_scale * schedule.std(t).square()),
                tilt(lam, 0.5),
            )
        )
        local_levels = tuple(range(len(args.levels) - 1))

    steps_per_level = int(1000 / (len(args.levels) - 1))
    config = CREPEConfig(
        u_levels=args.levels,
        steps_per_gap=steps_per_level,
        model_noising=model_noising,
        model_denoising=model_denoising,
        proposal_noising=model_noising,
        proposal_denoising=proposal_denoising,
        lam=lam,
        local_move=local_move,
        local_levels=local_levels,
    )

    print(
        f"Generating noisy CREPE reconstruction for FFHQ image {args.img_id} "
        f"under operator {args.operator} with sigma={args.noise_sigma:g}"
    )
    print(
        f"Gradient config: guidance={args.guidance}, "
        f"proposal_scale={args.proposal_scale:g}, "
        f"max_shift_norm={args.max_shift_norm:g}, beta={args.beta:g}, "
        f"rho={args.lambda_rho:g}, local_move={args.local_move}"
    )
    needs_tail = args.levels[0] > 0
    result, _ = crepe_sample(
        config,
        args.n_chains,
        data_shape,
        args.n_blocks,
        n_tail=args.tail_steps if needs_tail else None,
        exact=False,
        backend=backend,
        evaluate=evaluate,
        schedule=schedule,
        dtype=dtype,
        device=device,
        key=backend.key(effective_seed),
        thin=args.thin,
        burn_in=args.n_blocks - 1,
    )

    final_sample = result.final.x0[:, -1]
    reconstruction = (final_sample * 0.5 + 0.5).clamp(0, 1)
    reference_01 = (reference * 0.5 + 0.5).clamp(0, 1)
    mse = torch.mean((reconstruction - reference_01) ** 2)
    psnr = -10 * torch.log10(mse)
    metrics = measurement_metrics(
        operator, final_sample, measurement, noise_sigma=args.noise_sigma
    )
    print(f"Emitted samples per chain: {result.final.x0.shape[1]}")
    print(f"PSNR: {psnr.item():.4f} dB")
    print(f"Measurement MSE: {metrics['measurement_mse'].item():.8f}")
    print(f"Normalized residual: {metrics['normalized_residual'].item():.8f}")
    print(f"NLL: {metrics['negative_log_likelihood'].item():.4f}")
    print(
        f"Sample range: [{final_sample.min().item():.6f}, "
        f"{final_sample.max().item():.6f}]"
    )
    print(f"Total time: {time.time() - started_at:.2f} seconds")

    if args.save:
        save_dir = output_dir(
            args.output_root,
            "noisy",
            "CREPE",
            args.operator,
            args.seed,
            sigma=args.noise_sigma,
        )
        os.makedirs(save_dir, exist_ok=True)
        torch.save(final_sample, save_dir / f"{args.img_id}.pt")

    if args.plot:
        os.makedirs("generated/figures", exist_ok=True)
        plot_samples(
            normalize_tensor(final_sample[:1]),
            args.img_id,
            f"{args.operator}_noisy_sigma_{args.noise_sigma:g}_crepe",
            effective_seed,
        )


if __name__ == "__main__":
    main()
