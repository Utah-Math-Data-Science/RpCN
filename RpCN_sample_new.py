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
    parser.add_argument(
        "--final_state",
        choices=("x", "x0", "both"),
        default="x0",
        help="Which terminal representation to decode",
    )
    parser.add_argument(
        "--pcn_step_size",
        type=float,
        help="Use this constant pCN step size instead of the default ladder",
    )
    parser.add_argument(
        "--pcn_step_ladder",
        help="Comma-separated pCN step sizes, one per chain",
    )
    parser.add_argument(
        "--beta_schedule",
        choices=("linear", "quadratic"),
        default="linear",
        help="Inverse-temperature ladder shape",
    )
    parser.add_argument(
        "--output_label",
        default="RpCN",
        help="Output directory label",
    )
    parser.add_argument(
        "--move_steps",
        help="Comma-separated zero-based denoising transitions at which to apply RpCN",
    )
    args = parser.parse_args()

    prompt = get_prompt_by_id("prompts/prompts.json", args.img_id)
    if prompt is None:
        raise ValueError(f"Prompt ID not found: {args.img_id}")
    if args.resample_freq <= 0:
        raise ValueError("--resample_freq must be positive")
    if args.pCN_ct <= 0:
        raise ValueError("--pCN_ct must be positive")
    if args.pcn_step_size is not None and not 0 < args.pcn_step_size < 1:
        raise ValueError("--pcn_step_size must be in (0, 1)")
    if args.pcn_step_size is not None and args.pcn_step_ladder is not None:
        raise ValueError("Specify only one of --pcn_step_size and --pcn_step_ladder")

    custom_step_ladder = None
    if args.pcn_step_ladder is not None:
        custom_step_ladder = [
            float(value) for value in args.pcn_step_ladder.split(",")
        ]
        if len(custom_step_ladder) != N_CHAINS:
            raise ValueError(f"--pcn_step_ladder must contain {N_CHAINS} values")
        if any(not 0 < value <= 1 for value in custom_step_ladder):
            raise ValueError("All --pcn_step_ladder values must be in (0, 1]")

    move_steps = None
    if args.move_steps is not None:
        move_steps = frozenset(int(value) for value in args.move_steps.split(","))
        if not move_steps:
            raise ValueError("--move_steps must not be empty")
        if any(step < 0 or step >= NUM_INFERENCE_STEPS - 1 for step in move_steps):
            raise ValueError(
                f"--move_steps values must be in [0, {NUM_INFERENCE_STEPS - 2}]"
            )

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

    beta_fraction = xp.linspace(
        0.0, 1.0, N_CHAINS, dtype=image_class.dtype, device=image_class.device
    )
    betas = 10.0 * (
        beta_fraction.square()
        if args.beta_schedule == "quadratic"
        else beta_fraction
    )
    if custom_step_ladder is not None:
        step_sizes = xp.asarray(
            custom_step_ladder,
            dtype=image_class.dtype,
            device=image_class.device,
        )
    elif args.pcn_step_size is None:
        step_sizes = torch.linspace(
            torch.pi / 2 - 1e-3,
            0.05,
            N_CHAINS,
            dtype=image_class.dtype,
            device=image_class.device,
        ).sin()
    else:
        step_sizes = xp.full(
            (N_CHAINS,),
            args.pcn_step_size,
            dtype=image_class.dtype,
            device=image_class.device,
        )

    target = DDIMStep(schedule, eta=1.0)

    def should_move(step):
        transition = int(step) - 1
        if move_steps is not None:
            return transition in move_steps
        return transition % image_class.check_every == 0

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
            move_at=should_move,
            key=backend.key(seed),
        )

    print(f"Total time: {time.time() - start_time}")

    endpoints = ("x", "x0") if args.final_state == "both" else (args.final_state,)
    with torch.no_grad():
        for endpoint in endpoints:
            final = result.final.x if endpoint == "x" else result.final.x0
            decoded_images, rewards = decode_and_score(
                image_class, prompt, final[:, -1]
            )
            label = (
                f"{args.output_label}_{endpoint}"
                if args.final_state == "both"
                else args.output_label
            )
            save_results(args.model, args.img_id, decoded_images, rewards, label)


if __name__ == "__main__":
    main()
