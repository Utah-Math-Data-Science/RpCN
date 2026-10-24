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

from ttg import (  # noqa: E402
    DDIMStep,
    FKDiff,
    Never,
    PCN,
    SequentialSweep,
    ladder,
    smc,
)


N_CHAINS = 4


def main():
    parser = argparse.ArgumentParser(
        description="Generate Stable Diffusion samples with the updated RpCN implementation."
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--resample_freq",
        default=20,
        type=int,
        help="How often to apply the RpCN move",
    )
    parser.add_argument(
        "--pCN_ct",
        default=10,
        type=int,
        help="Number of pCN proposals per move",
    )
    args = parser.parse_args()

    prompt = get_prompt_by_id("prompts/prompts.json", args.img_id)
    if prompt is None:
        raise ValueError(f"Prompt ID not found: {args.img_id}")
    if args.resample_freq <= 0:
        raise ValueError("--resample_freq must be positive")
    if args.pCN_ct <= 0:
        raise ValueError("--pCN_ct must be positive")

    seed = compute_seed(args.img_id, 31)
    torch.manual_seed(seed)

    image_class = StableDiffusionContext(
        prompt=prompt,
        batch_size=BATCH_SIZE,
        num_inference_steps=NUM_INFERENCE_STEPS,
        check_every=args.resample_freq,
        model=args.model,
    )

    backend, schedule, evaluate = build_sampling_components(image_class)
    xp = backend.xp
    u_grid = xp.linspace(
        1.0,
        0.0,
        NUM_INFERENCE_STEPS,
        dtype=image_class.dtype,
        device=image_class.device,
    )

    betas = xp.linspace(
        0.0, 10.0, N_CHAINS, dtype=image_class.dtype, device=image_class.device
    )
    step_sizes = torch.linspace(
        torch.pi / 2 - 1e-3,
        0.05,
        N_CHAINS,
        dtype=image_class.dtype,
        device=image_class.device,
    ).sin()

    target = DDIMStep(schedule, eta=1.0)

    print(f"Generating reward aligned samples for prompt: {args.img_id}")
    start_time = time.time()
    with torch.no_grad():
        result = smc(
            schedule.prior(
                (BATCH_SIZE, N_CHAINS),
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
            proposal=target,
            potential=FKDiff(ladder(betas)),
            policy=Never(),
            move=PCN(
                n_proposals=args.pCN_ct,
                step_size=ladder(step_sizes),
                pairs=SequentialSweep(),
            ),
            move_at=lambda step: step % image_class.check_every == 0,
            key=backend.key(seed),
        )

        coldest_latents = result.final.x0[:, -1]
        decoded_images, rewards = decode_and_score(
            image_class, prompt, coldest_latents
        )

    print(f"Total time: {time.time() - start_time}")

    save_results(args.model, args.img_id, decoded_images, rewards, "RpCN")


if __name__ == "__main__":
    main()
