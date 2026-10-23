import argparse
import os
import time

import numpy as np
import torch
from ttg import (
    ESSBelow,
    FKDiff,
    Guided,
    PosteriorStep,
    normalized,
    DDIMStep,
    das_lambda,
    make_evaluate_torch,
    on_x0,
    smc,
    tilt,
    x0_from_eps,
    categorical
)
from ttg.backend import TorchBackend

from RpCN_sample_noiseless import DiscreteVPSchedule
from rewards.DAPS_operators import get_operator
from utils import load_image_by_id, load_model_and_sampler, normalize_tensor, plot_samples


def parse_args():
    parser = argparse.ArgumentParser(
        description="Solve a noiseless FFHQ inverse problem with the packaged DAS solver."
    )
    parser.add_argument("--img_id", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument(
        "--operator",
        choices=("down_sampling", "gaussian_blur", "motion_blur", "inpainting_random"),
        default="motion_blur",
    )
    parser.add_argument("--n_particles", type=int, default=3)
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--lambda_growth", type=float, default=0.008)
    parser.add_argument(
        "--max_inverse_temperature",
        type=float,
        default=10.0,
        help="Max inverse temperature from RpCN, 1/alpha in DAS",
    )
    parser.add_argument("--ess_threshold", type=float, default=0.5)
    parser.add_argument("--output_dir", default="generated/tensors/DAS")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    if args.n_particles < 1:
        parser.error("--n_particles must be at least 1")
    if args.diffusion_steps < 2 or args.diffusion_steps > 1000:
        parser.error("--diffusion_steps must be between 2 and 1000")
    if args.lambda_growth <= 0:
        parser.error("--lambda_growth must be positive")
    if not 0 < args.ess_threshold <= 1:
        parser.error("--ess_threshold must be in (0, 1]")

    return args


def main():
    args = parse_args()
    started_at = time.time()
    device = torch.device("cuda")
    dtype = torch.float32
    data_shape = (3, 256, 256)
    validation_path = "~/.cache/kagglehub/datasets/weideng056/ffhq-1k-valid/versions/1/samples"

    effective_seed = args.seed + 2435 * args.img_id
    torch.manual_seed(effective_seed)
    np.random.seed(effective_seed % (2**32))

    model, ddpm = load_model_and_sampler()
    reference = load_image_by_id(
        validation_path, args.img_id, device=device, save=True
    ).detach().to(dtype=dtype)
    operator = get_operator(args.operator, device=device)
    measurement = operator(reference).detach()
    schedule = DiscreteVPSchedule(
        torch.as_tensor(ddpm.alphas_cumprod, device=device, dtype=dtype)
    )

    def epsilon_network(x, t):
        timestep = torch.round(t).to(torch.long)
        output = model(x, ddpm._scale_timesteps(timestep))
        return output[:, : x.shape[1]]

    def reward(x0):
        return - operator.loss(x0, measurement)

    def to_x0_eps(x,t,eps, schedule): 
        x0 = x0_from_eps(x, t, eps, schedule).clamp(-1, 1)
        return x0, eps
    


    evaluate = make_evaluate_torch(
        epsilon_network,
        to_x0_eps,
        schedule,
        on_x0(reward),
    )
    backend = TorchBackend(device=device)
    u_grid = torch.linspace(
        1.0, 0.0, args.diffusion_steps, device=device, dtype=dtype
    )
    lam = das_lambda(
        schedule.time(u_grid), growth=args.lambda_growth, alpha_kl=1/args.max_inverse_temperature
    )
    target = DDIMStep(schedule, eta=1.0)
    proposal = Guided(target, normalized(lam))
    # proposal = target

    start = schedule.prior(
        (1, args.n_particles), data_shape, xp=backend.xp, dtype=dtype, device=device
    )

    print(
        f"Generating noiseless DAS reconstruction for FFHQ image {args.img_id} "
        f"under operator {args.operator}"
    )

    key_root = backend.key(effective_seed)
    key_smc, key_select = backend.split(key_root, 2)


    result = smc(
        start,
        u_grid,
        backend=backend,
        evaluate=evaluate,
        schedule=schedule,
        target=target,
        proposal=proposal,
        potential=FKDiff(lam),
        policy=ESSBelow(args.ess_threshold),
        key=key_smc,
    )
    print(f'final img stats')
    print(result.final.x0.max(), result.final.x0.min())

    selected = categorical(backend, result.log_weights, key_select)
    final_sample = result.final.x0[0, selected]

    reconstruction = (final_sample * 0.5 + 0.5).clamp(0, 1)
    reference_01 = (reference * 0.5 + 0.5).clamp(0, 1)
    mse = torch.mean((reconstruction - reference_01) ** 2)
    psnr = -10 * torch.log10(mse)
    measurement_mse = torch.mean((operator(final_sample) - measurement) ** 2)
    print(f"PSNR: {psnr.item():.4f} dB")
    print(f"Measurement MSE: {measurement_mse.item():.8f}")
    print(f"Total time: {time.time() - started_at:.2f} seconds")

    if args.save:
        os.makedirs(args.output_dir, exist_ok=True)
        save_path = os.path.join(
            args.output_dir,
            f"{args.img_id}_{args.operator}_noiseless_das_{effective_seed}.pt",
        )
        torch.save(final_sample, save_path)

    if args.plot:
        os.makedirs("generated/figures", exist_ok=True)
        plot_samples(
            normalize_tensor(final_sample),
            args.img_id,
            f"{args.operator}_noiseless_das",
            effective_seed,
        )


if __name__ == "__main__":
    main()
