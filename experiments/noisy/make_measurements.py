import argparse
from pathlib import Path

import numpy as np
import torch

from experiments.common import (
    DEFAULT_DATA_DIR,
    DEFAULT_MEASUREMENT_ROOT,
    DAPS_PIXEL_FFHQ_IDS,
    MEASUREMENT_FORMAT_VERSION,
    MEASUREMENT_RNG_SCHEME,
    OPERATORS,
    OPERATOR_CONFIGS,
    _atomic_torch_save,
    build_operator,
    degradation_sha256,
    measurement_path,
)
from utils import load_image_by_id


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create DAPS pixel_ffhq-compatible noisy observations as one "
            "ordered CUDA batch per operator."
        )
    )
    parser.add_argument("--ids_file", type=Path, default=Path("ids.csv"))
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--operators", nargs="+", choices=OPERATORS, default=OPERATORS)
    parser.add_argument("--noise_sigma", type=float, default=0.05)
    parser.add_argument("--operator_seed", type=int, default=42)
    parser.add_argument("--measurement_seed", type=int, default=42)
    parser.add_argument(
        "--measurement_root", type=Path, default=DEFAULT_MEASUREMENT_ROOT
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_ids(path):
    lines = path.read_text().splitlines()
    return [int(line.strip()) for line in lines[1:] if line.strip()]


def main():
    args = parse_args()
    if args.noise_sigma <= 0:
        raise ValueError("--noise_sigma must be positive")
    image_ids = load_ids(args.ids_file)
    if tuple(image_ids) != DAPS_PIXEL_FFHQ_IDS:
        raise ValueError(
            f"{args.ids_file} must list the DAPS pixel_ffhq IDs 49000-49099 "
            "in sorted order"
        )

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("DAPS-compatible noisy observations require a CUDA device")
    if device.index is None:
        device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    references = torch.cat(
        [
            load_image_by_id(args.data_dir, image_id, device=device, save=False)
            for image_id in image_ids
        ],
        dim=0,
    )

    for operator_name in args.operators:
        # DAPS starts each operator command in a fresh seed-42 process.
        np.random.seed(args.measurement_seed)
        torch.manual_seed(args.measurement_seed)
        torch.cuda.manual_seed_all(args.measurement_seed)
        torch.backends.cudnn.deterministic = True

        # DAPS DiffusionData.get_data(..., sigma=0) still consumes this draw.
        torch.randn_like(references)

        operator = build_operator(
            operator_name,
            device,
            args.operator_seed,
            args.noise_sigma,
        )
        clean = operator(references).detach()
        noise = torch.randn_like(clean)
        measurements = clean + args.noise_sigma * noise
        fingerprint = degradation_sha256(operator)

        for dataset_index, image_id in enumerate(image_ids):
            path = measurement_path(
                args.measurement_root,
                args.noise_sigma,
                operator_name,
                image_id,
            )
            artifact = {
                "measurement_format_version": MEASUREMENT_FORMAT_VERSION,
                "rng_scheme": MEASUREMENT_RNG_SCHEME,
                "image_id": image_id,
                "dataset_index": dataset_index,
                "source_filename": f"{image_id}.png",
                "batch_size": len(image_ids),
                "operator": operator_name,
                "operator_seed": args.operator_seed,
                "measurement_seed": args.measurement_seed,
                "noise_sigma": args.noise_sigma,
                "operator_config": OPERATOR_CONFIGS[operator_name],
                "degradation_sha256": fingerprint,
                "measurement": measurements[dataset_index : dataset_index + 1].cpu(),
            }
            _atomic_torch_save(artifact, path)
            print(path)


if __name__ == "__main__":
    main()
