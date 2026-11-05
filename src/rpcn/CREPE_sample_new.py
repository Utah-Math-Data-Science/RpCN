import argparse
import time

import torch

from .sampling_common import (
    BATCH_SIZE,
    NUM_INFERENCE_STEPS,
    StableDiffusionContext,
    add_common_arguments,
    build_sampling_components,
    compute_seed,
    decode_and_score,
    get_prompt_by_id,
    save_results,
)
from ttg import (
    CREPEConfig,
    DDIMStep,
    Noising,
    SequentialSweep,
    crepe_sample,
    level_lambda,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Stable Diffusion samples with the updated CREPE implementation."
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--beta",
        default=10.0,
        type=float,
        help="Reward tilt at the lowest-noise level",
    )
    parser.add_argument(
        "--rho",
        default=1.0,
        type=float,
        help="Exponent controlling the tilt across noise levels",
    )


    args = parser.parse_args()

    prompt = get_prompt_by_id(args.img_id)
    if prompt is None:
        raise ValueError(f"Prompt ID not found: {args.img_id}")
    if args.beta <= 0:
        raise ValueError("--beta must be positive")
    if args.rho <= 0:
        raise ValueError("--rho must be positive")



    seed = compute_seed(args.img_id, 31)
    torch.manual_seed(seed)

    image_class = StableDiffusionContext(
        prompt=prompt,
        batch_size=BATCH_SIZE,
        num_inference_steps=NUM_INFERENCE_STEPS,
        model=args.model,
    )

    backend, schedule, evaluate = build_sampling_components(image_class)
    xp = backend.xp

    u_levels = (0, 1 / 3, 2 / 3, 1.0)
    level_times = schedule.time(
        xp.asarray(u_levels, dtype=image_class.dtype, device=image_class.device)
    )
    lam = level_lambda(level_times, beta=args.beta, rho=args.rho)
    noising = Noising(schedule)
    denoising = DDIMStep(schedule, eta=1.0)

    config = CREPEConfig(
        u_levels=u_levels,
        steps_per_gap=28,
        model_noising=noising,
        proposal_noising=noising,
        model_denoising=denoising,
        proposal_denoising=denoising,
        lam=lam,
    )

    # 3 gaps x 28 steps with two burn-in blocks and one emitting block:
    # (1 + 3 * 28) + 3 * (2 * 3 * 28 + 1) = 592 NFE per output.
    blocks = 3

    print(f"Generating CREPE samples for prompt: {args.img_id}")
    start_time = time.time()
    result, _ = crepe_sample(
        config,
        BATCH_SIZE,
        image_class.data_dim,
        1,
        backend=backend,
        evaluate=evaluate,
        schedule=schedule,
        dtype=image_class.dtype,
        device=image_class.device,
        key=backend.key(seed),
        burn_in=blocks - 1,
    )

    with torch.no_grad():
        final_latents = result.final.x0[:, -1]
        decoded_images, rewards = decode_and_score(
            image_class, prompt, final_latents
        )

    print(f"Total time: {time.time() - start_time}")

    save_results(
        args.model,
        f"{args.img_id}_crepe",
        decoded_images,
        rewards,
        "CREPE",
    )


if __name__ == "__main__":
    main()
