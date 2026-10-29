import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from experiments.common import (
    DEFAULT_MEASUREMENT_ROOT,
    OPERATORS,
    OPERATOR_CONFIGS,
    build_operator,
    degradation_sha256,
    measurement_path,
)


DAPS_OPERATOR_NAMES = {
    "down_sampling": "down_sampling",
    "gaussian_blur": "gaussian_blur",
    "motion_blur": "motion_blur",
    "inpainting_random": "inpainting",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Numerically compare cached observations with DAPS pixel_ffhq."
    )
    parser.add_argument("--daps_root", type=Path, default=Path.home() / "DAPS")
    parser.add_argument("--ids_file", type=Path, default=Path("ids.csv"))
    parser.add_argument(
        "--measurement_root", type=Path, default=DEFAULT_MEASUREMENT_ROOT
    )
    parser.add_argument("--noise_sigma", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--operators", nargs="+", choices=OPERATORS, default=OPERATORS)
    return parser.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, str(args.daps_root))
    from data import ImageDataset
    from forward_operator import get_operator

    image_ids = [
        int(line)
        for line in args.ids_file.read_text().splitlines()[1:]
        if line.strip()
    ]
    device = torch.device(args.device)
    if device.index is None:
        device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    dataset = ImageDataset(
        root=args.daps_root / "dataset/test-ffhq",
        resolution=256,
        device=device,
        start_id=0,
        end_id=len(image_ids),
    )

    failed = False
    for operator_name in args.operators:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        references = dataset.get_data(len(image_ids), 0)

        np.random.seed(args.seed)
        config = dict(OPERATOR_CONFIGS[operator_name])
        daps_name = DAPS_OPERATOR_NAMES[operator_name]
        operator = get_operator(
            name=daps_name,
            device=device,
            sigma=args.noise_sigma,
            **config,
        )
        expected = operator.measure(references)
        artifacts = [
            torch.load(
                    measurement_path(
                        args.measurement_root,
                        args.noise_sigma,
                        operator_name,
                        image_id,
                    ),
                    map_location=device,
                    weights_only=True,
                )
            for image_id in image_ids
        ]
        cached = torch.cat(
            [artifact["measurement"] for artifact in artifacts],
            dim=0,
        )
        local_operator = build_operator(
            operator_name, device, args.seed, 0.0
        )
        cached_clean = local_operator(references)
        expected_clean = operator(references)
        maximum_error = (cached - expected).abs().max().item()
        exact = torch.equal(cached, expected)
        clean_maximum_error = (cached_clean - expected_clean).abs().max().item()
        clean_exact = torch.equal(cached_clean, expected_clean)
        fingerprint = degradation_sha256(operator)
        print(
            f"{operator_name}: clean_exact={clean_exact}, "
            f"clean_max_abs_error={clean_maximum_error:.9g}, "
            f"noisy_exact={exact}, noisy_max_abs_error={maximum_error:.9g}, "
            f"degradation_sha256={fingerprint}"
        )
        failed |= not (clean_exact and exact)

    if failed:
        raise SystemExit("At least one cached observation differs from DAPS")


if __name__ == "__main__":
    main()
