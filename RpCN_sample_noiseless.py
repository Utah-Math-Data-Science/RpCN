import argparse
import hashlib
import os
import time

import numpy as np
import torch
from ttg import (
    FKDiff,
    Guided,
    Never,
    PCN,
    PosteriorStep,
    DDIMStep,
    SequentialSweep,
    constant,
    ladder,
    make_evaluate_torch,
    normalized,
    on_x0,
    smc,
    x0_from_eps,
)
from ttg.backend import TorchBackend
from ttg.kernels import Gaussian

from rewards.DAPS_operators import get_operator
from utils import load_image_by_id, normalize_tensor, plot_samples
from utils import load_model_and_sampler


def degradation_sha256(operator):
    artifact = getattr(operator, "mask", None)
    if artifact is None and hasattr(operator, "conv"):
        artifact = operator.conv.seq[1].weight
    if artifact is None:
        return None
    data = artifact.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


class DiscreteVPSchedule:
    """Expose the trained DDPM's discrete VP schedule through the ttg API."""

    def __init__(self, alphas_cumprod):
        self.alphas_cumprod = alphas_cumprod
        self.num_timesteps = alphas_cumprod.shape[0]

    def time(self, u):
        return u * (self.num_timesteps - 1)

    def _alpha_bar(self, t):
        index = torch.round(t).to(torch.long).clamp(0, self.num_timesteps - 1)
        return self.alphas_cumprod[index]

    def mean(self, t):
        return self._alpha_bar(t).sqrt()

    def std(self, t):
        return (1.0 - self._alpha_bar(t)).sqrt()

    def log_snr(self, t):
        alpha_bar = self._alpha_bar(t)
        return torch.log(alpha_bar) - torch.log1p(-alpha_bar)

    def log_snr_diff(self, t_a, t_b):
        return self.log_snr(t_a) - self.log_snr(t_b)

    def prior(self, leading_shape, event_shape, *, xp, dtype, device=None):
        return Gaussian(
            xp.zeros((*leading_shape, *event_shape), dtype=dtype, device=device),
            xp.ones(tuple(leading_shape), dtype=dtype, device=device),
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Solve a noiseless FFHQ inverse problem with the packaged RpCN solver."
    )
    parser.add_argument("--img_id", type=int, default=49000)
    parser.add_argument("--seed", type=int, default=42)
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
    parser.add_argument("--n_chains", type=int, default=3)
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--resampling_frequency", type=int, default=25)
    parser.add_argument("--pcn_update_count", type=int, default=10)

    parser.add_argument(
        "--max_inverse_temperature",
        type=float,
        default=10.0,
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
    parser.add_argument("--output_dir", default="generated/tensors/RpCN")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

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
    validation_path = "~/DAPS/dataset/test-ffhq"

    effective_seed = args.seed + 2435 * args.img_id
    torch.manual_seed(effective_seed)

    model, ddpm = load_model_and_sampler()
    reference = load_image_by_id(
        validation_path, args.img_id, device=device, save=True
    ).detach().to(dtype=dtype)
    np.random.seed(args.operator_seed % (2**32))
    operator = get_operator(args.operator, device=device)
    measurement = operator(reference).detach()
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
        return - operator.loss(x0, measurement)

    # to_x0_eps = lambda x, t, eps, sch: (x0_from_eps(x, t, eps, sch), eps)

    def to_x0_eps(x,t,eps, schedule): 
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

    pcn_step_sizes =  0.5 * torch.ones(
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
        f"Generating noiseless reconstruction for FFHQ image {args.img_id} "
        f"under operator {args.operator}"
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


    print(f"Total time: {time.time() - started_at:.2f} seconds")

    if args.save:
        os.makedirs(args.output_dir, exist_ok=True)
        save_path = os.path.join(
            args.output_dir,
            f"{args.img_id}_{args.operator}_noiseless_rpcn_{effective_seed}.pt",
        )
        torch.save(final_sample, save_path)

    if args.plot:
        os.makedirs("generated/figures", exist_ok=True)
        plot_samples(
            normalize_tensor(final_sample),
            args.img_id,
            f"{args.operator}_noiseless_rpcn",
            effective_seed,
        )


if __name__ == "__main__":
    main()
