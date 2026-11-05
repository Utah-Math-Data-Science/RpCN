#!/usr/bin/env python3

import argparse
import csv
import sys
from importlib.resources import files
from pathlib import Path

import clip
import torch
from PIL import Image
from tabulate import tabulate


MODELS = ("v1.4", "v1.5", "v2.1")
SAMPLERS = (
    ("CREPE", "CREPE", "_crepe"),
    ("DAS (alpha_kl=0.1)", "DAS_0.1", "_das"),
    ("DAS (alpha_kl=0.01)", "DAS_0.01", "_das"),
    ("RpCN", "RpCN", ""),
)
IMAGES_PER_PROMPT = 4


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate reward and CLIP diversity for all T2I experiments."
    )
    parser.add_argument(
        "--generated_dir",
        type=Path,
        default=Path.cwd() / "generated",
        help="Generated results directory",
    )
    parser.add_argument(
        "--prompt_file",
        type=Path,
        default=files("rpcn.data").joinpath("prompt_ids.csv"),
        help="CSV containing the evaluation prompt IDs",
    )
    parser.add_argument(
        "--clip_model",
        default="ViT-L/14",
        help="CLIP model used for image diversity",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device used for CLIP evaluation",
    )
    parser.add_argument(
        "--allow_missing",
        action="store_true",
        help="Evaluate only complete prompts instead of failing on missing results",
    )
    return parser.parse_args()


def load_prompt_ids(path):
    with path.open(newline="") as file:
        prompt_ids = [row["id"] for row in csv.DictReader(file)]
    if not prompt_ids:
        raise ValueError(f"No prompt IDs found in {path}")
    return prompt_ids


def collect_complete_prompts(result_dir, prompt_ids, stem_suffix):
    complete = []
    missing = []
    for prompt_id in prompt_ids:
        reward_path = result_dir / "csv" / f"{prompt_id}{stem_suffix}_rewards.csv"
        image_paths = [
            result_dir / "images" / f"{prompt_id}{stem_suffix}_{index}.png"
            for index in range(IMAGES_PER_PROMPT)
        ]
        if reward_path.is_file() and all(path.is_file() for path in image_paths):
            complete.append((prompt_id, reward_path, image_paths))
        else:
            missing.append(prompt_id)
    return complete, missing


def max_reward(path):
    with path.open(newline="") as file:
        rewards = [float(row["reward"]) for row in csv.DictReader(file)]
    if not rewards:
        raise ValueError(f"No rewards found in {path}")
    return max(rewards)


def clip_diversity(image_paths, clip_model, preprocess, device):
    images = []
    for path in image_paths:
        with Image.open(path) as image:
            images.append(preprocess(image.convert("RGB")))

    image_batch = torch.stack(images).to(device)
    features = clip_model.encode_image(image_batch).float()
    features = features / features.norm(dim=1, keepdim=True)
    pairwise_distances = torch.cdist(features, features).square()
    return pairwise_distances.triu(diagonal=1).sum().item() * (
        2 / (IMAGES_PER_PROMPT * (IMAGES_PER_PROMPT - 1))
    )


def evaluate_configuration(
    result_dir,
    prompt_ids,
    stem_suffix,
    clip_model,
    preprocess,
    device,
    allow_missing,
):
    complete, missing = collect_complete_prompts(
        result_dir, prompt_ids, stem_suffix
    )
    if missing and not allow_missing:
        raise FileNotFoundError(
            f"{result_dir}: {len(missing)} of {len(prompt_ids)} prompts are incomplete; "
            "rerun generation or pass --allow_missing"
        )
    if not complete:
        raise FileNotFoundError(f"{result_dir}: no complete prompts found")
    if missing:
        print(
            f"Warning: {result_dir} uses {len(complete)} of {len(prompt_ids)} prompts",
            file=sys.stderr,
        )

    reward_values = []
    diversity_values = []
    with torch.inference_mode():
        for _, reward_path, image_paths in complete:
            reward_values.append(max_reward(reward_path))
            diversity_values.append(
                clip_diversity(image_paths, clip_model, preprocess, device)
            )

    return (
        sum(reward_values) / len(reward_values),
        sum(diversity_values) / len(diversity_values),
    )


def main():
    args = parse_args()
    prompt_ids = load_prompt_ids(args.prompt_file)
    clip_model, preprocess = clip.load(
        args.clip_model, device=args.device, jit=False
    )
    clip_model.eval()

    for model in MODELS:
        rows = []
        for display_name, result_label, stem_suffix in SAMPLERS:
            result_dir = args.generated_dir / model / result_label
            reward, diversity = evaluate_configuration(
                result_dir,
                prompt_ids,
                stem_suffix,
                clip_model,
                preprocess,
                args.device,
                args.allow_missing,
            )
            rows.append((display_name, reward, diversity))

        print(f"\n{model}")
        print(
            tabulate(
                rows,
                headers=(
                    "Sampler",
                    "Average max-image reward",
                    "Mean CLIP diversity",
                ),
                tablefmt="github",
                floatfmt=".4f",
            )
        )


if __name__ == "__main__":
    main()
