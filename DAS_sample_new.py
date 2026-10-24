import argparse
import time

import torch

from sampling_common import (
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
from ttg import DDIMStep, ESSBelow, FKDiff, Guided, das_lambda, smc, tilt


def main():
    parser = argparse.ArgumentParser(
        description="Generate Stable Diffusion samples with the updated DAS implementation."
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--growth",
        default=0.008,
        type=float,
        help="Growth rate of the DAS tempering schedule",
    )
    parser.add_argument(
        "--alpha_kl",
        default=0.01,
        type=float,
        help="KL regularization strength",
    )
    parser.add_argument(
        "--ess_threshold",
        default=0.5,
        type=float,
        help="Resample when ESS falls below this fraction of the particle count",
    )
    args = parser.parse_args()

    prompt = get_prompt_by_id("prompts/prompts.json", args.img_id)
    if prompt is None:
        raise ValueError(f"Prompt ID not found: {args.img_id}")
    if args.growth <= 0:
        raise ValueError("--growth must be positive")
    if args.alpha_kl <= 0:
        raise ValueError("--alpha_kl must be positive")
    if not 0 < args.ess_threshold <= 1:
        raise ValueError("--ess_threshold must be in (0, 1]")

    seed = compute_seed(args.img_id, 31)
    torch.manual_seed(seed)

    image_class = StableDiffusionContext(
        prompt=prompt,
        batch_size=BATCH_SIZE,
        num_inference_steps=NUM_INFERENCE_STEPS,
        model=args.model,
    )

    image_class.pipe.unet.requires_grad_(False)
    image_class.pipe.vae.requires_grad_(False)
    image_class.pipe.text_encoder.requires_grad_(False)
    image_class.ir_model.requires_grad_(False)

    backend, schedule, evaluate = build_sampling_components(image_class)
    xp = backend.xp
    u_grid = xp.linspace(
        1.0,
        0.0,
        NUM_INFERENCE_STEPS,
        dtype=image_class.dtype,
        device=image_class.device,
    )
    lam = das_lambda(
        schedule.time(u_grid), growth=args.growth, alpha_kl=args.alpha_kl
    )
    target = DDIMStep(schedule, eta=1.0)

    print(f"Generating DAS samples for prompt: {args.img_id}")
    start_time = time.time()
    result = smc(
        schedule.prior(
            (1, BATCH_SIZE),
            image_class.data_dim,
            xp=xp,
            dtype=image_class.dtype,
            device=image_class.device,
        ),
        u_grid,
        backend=backend,
        evaluate=evaluate,
        schedule=schedule,
        target=target,
        proposal=Guided(target, tilt(lam)),
        potential=FKDiff(lam),
        policy=ESSBelow(args.ess_threshold),
        key=backend.key(seed),
    )

    with torch.no_grad():
        final_latents = result.final.x0.reshape((-1, *image_class.data_dim))
        decoded_images, rewards = decode_and_score(
            image_class, prompt, final_latents
        )

    print(f"Total time: {time.time() - start_time}")

    save_results(
        args.model,
        f"{args.img_id}_das",
        decoded_images,
        rewards,
        "DAS",
    )


if __name__ == "__main__":
    main()
