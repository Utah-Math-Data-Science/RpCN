import argparse
import csv
from pathlib import Path

import torch
from piq import LPIPS
from tabulate import tabulate

from DAPS_metrics import (
    DEFAULT_TRUE_DIR,
    calculate_lpips,
    calculate_psnr_ssim,
    discover_tensor_files,
    load_generated_tensor,
    load_true_tensor,
)


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate PSNR, SSIM, and LPIPS for guidance ablations."
    )
    parser.add_argument("--operator", default="down_sampling")
    parser.add_argument(
        "--ablation-root", type=Path, default=ROOT / "generated/ablations"
    )
    parser.add_argument(
        "--configs",
        type=Path,
        default=None,
        help="Optional CSV with a name column; otherwise discover output directories.",
    )
    parser.add_argument("--true-dir", type=Path, default=DEFAULT_TRUE_DIR)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV destination for the metric table.",
    )
    return parser.parse_args()


def load_config_names(path):
    with path.open(newline="") as file:
        return [row["name"] for row in csv.DictReader(file)]


def main():
    args = parse_args()
    operator_root = args.ablation_root / args.operator
    if args.configs is None:
        config_names = sorted(
            path.name
            for path in operator_root.iterdir()
            if (path / "tensors").is_dir()
        )
    else:
        config_names = load_config_names(args.configs)
    if not config_names:
        raise FileNotFoundError(f"No ablation configurations found in {operator_root}")
    config_files = {}
    expected_ids = None

    for name in config_names:
        tensor_dir = args.ablation_root / args.operator / name / "tensors"
        files = discover_tensor_files(tensor_dir, args.operator)
        image_ids = set(files)
        if expected_ids is None:
            expected_ids = image_ids
        elif image_ids != expected_ids:
            raise ValueError(
                f"Image IDs for {name} do not match the other configurations"
            )
        config_files[name] = files

    image_ids = sorted(expected_ids)
    real_images = torch.stack(
        [load_true_tensor(args.true_dir, image_id) for image_id in image_ids]
    )
    lpips = LPIPS(replace_pooling=True, reduction="none").to(args.device).eval()

    rows = []
    for name in config_names:
        generated_images = torch.stack(
            [load_generated_tensor(config_files[name][image_id]) for image_id in image_ids]
        )
        psnr_score, ssim_score = calculate_psnr_ssim(
            real_images, generated_images, args.device, args.batch_size
        )
        lpips_score = calculate_lpips(
            real_images, generated_images, lpips, args.device, args.batch_size
        )
        rows.append(
            {
                "configuration": name,
                "psnr": psnr_score,
                "ssim": ssim_score,
                "lpips": lpips_score,
            }
        )

    print(
        tabulate(
            rows,
            headers="keys",
            tablefmt="github",
            floatfmt=".4f",
        )
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
