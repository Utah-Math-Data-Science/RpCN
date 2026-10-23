import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from piq import FID, LPIPS, psnr, ssim
from tabulate import tabulate
from torchvision.models import Inception_V3_Weights, inception_v3
from torchvision.transforms.functional import pil_to_tensor


OPERATORS = (
    ("down_sampling", "down_sampling"),
    ("inpainting_random", "inpaint_random"),
    ("gaussian_blur", "deblur_gauss"),
    ("motion_blur", "deblur_motion"),
)
DEFAULT_TRUE_DIR = (
    Path.home()
    / ".cache/kagglehub/datasets/weideng056/ffhq-1k-valid/versions/1/samples"
)


def parse_args():
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    parser = argparse.ArgumentParser(
        description="Compute FID, mean PSNR, SSIM, and LPIPS for all DAPS operators."
    )
    parser.add_argument(
        "--tensors-dir",
        type=Path,
        default=project_dir / "generated" / "tensors",
        help="Root directory containing one subdirectory per sampling method.",
    )
    parser.add_argument(
        "--method",
        default="RpCN",
        help="Sampling method subdirectory to evaluate (for example, RpCN or DAS).",
    )
    parser.add_argument(
        "--true-dir",
        type=Path,
        default=DEFAULT_TRUE_DIR,
        help="Directory containing ground-truth images named <image_id>.png.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (default: cuda when available, otherwise cpu).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N image IDs (useful for a quick check).",
    )
    return parser.parse_args()


def discover_tensor_files(tensors_dir, operator):
    files_by_id = {}
    for path in tensors_dir.glob(f"*_{operator}_*.pt"):
        try:
            image_id = int(path.name.split("_", 1)[0])
        except ValueError as error:
            raise ValueError(f"Could not read an image ID from {path.name}") from error
        if image_id in files_by_id:
            raise ValueError(
                f"Multiple tensors found for {operator}, image {image_id}: "
                f"{files_by_id[image_id].name} and {path.name}"
            )
        files_by_id[image_id] = path

    if not files_by_id:
        raise FileNotFoundError(f"No tensors found for {operator} in {tensors_dir}")
    return files_by_id


def validate_and_select_files(tensors_dir, limit):
    operator_files = {
        operator: discover_tensor_files(tensors_dir, operator)
        for operator, _ in OPERATORS
    }
    reference_operator = OPERATORS[0][0]
    expected_ids = set(operator_files[reference_operator])
    for operator, files_by_id in operator_files.items():
        ids = set(files_by_id)
        if ids != expected_ids:
            missing = sorted(expected_ids - ids)
            extra = sorted(ids - expected_ids)
            raise ValueError(
                f"Tensor IDs for {operator} do not match {reference_operator}. "
                f"Missing: {missing}; extra: {extra}"
            )

    image_ids = sorted(expected_ids)
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be at least 1")
        image_ids = image_ids[:limit]
    return operator_files, image_ids


def load_generated_tensor(path):
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    while tensor.ndim > 3 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim != 3:
        raise ValueError(f"Expected a CHW tensor in {path}, got shape {tuple(tensor.shape)}")
    if tensor.shape[0] != 3 and tensor.shape[-1] == 3:
        tensor = tensor.permute(2, 0, 1)
    if tensor.shape[0] != 3:
        raise ValueError(f"Expected three channels in {path}, got shape {tuple(tensor.shape)}")

    return tensor.float()


def load_true_tensor(true_dir, image_id):
    path = true_dir / f"{image_id}.png"
    if not path.is_file():
        raise FileNotFoundError(f"Ground-truth image not found: {path}")
    with Image.open(path) as image:
        tensor = pil_to_tensor(image.convert("RGB")).float() / 255
    return tensor * 2 - 1


def inception_preprocess(images):
    images = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)
    mean = images.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = images.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    images = images * 0.5 + 0.5
    return (images - mean) / std


def daps_metric_preprocess(images):
    return (images * 0.5 + 0.5).clamp(0, 1)


@torch.inference_mode()
def extract_features(images, model, device, batch_size):
    features = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size].to(device)
        features.append(model(inception_preprocess(batch)).cpu())
    return torch.cat(features)


@torch.inference_mode()
def calculate_lpips(real_images, generated_images, model, device, batch_size):
    values = []
    for start in range(0, len(real_images), batch_size):
        real_batch = daps_metric_preprocess(
            real_images[start : start + batch_size].to(device)
        )
        generated_batch = daps_metric_preprocess(
            generated_images[start : start + batch_size].to(device)
        )
        values.append(model(real_batch, generated_batch).reshape(-1).cpu())
    return torch.cat(values).mean().item()


@torch.inference_mode()
def calculate_psnr_ssim(real_images, generated_images, device, batch_size):
    psnr_values = []
    ssim_values = []
    for start in range(0, len(real_images), batch_size):
        real_batch = daps_metric_preprocess(
            real_images[start : start + batch_size].to(device)
        )
        generated_batch = daps_metric_preprocess(
            generated_images[start : start + batch_size].to(device)
        )
        psnr_values.append(
            psnr(real_batch, generated_batch, data_range=1.0, reduction="none").cpu()
        )
        ssim_values.append(
            ssim(real_batch, generated_batch, data_range=1.0, reduction="none").cpu()
        )
    return (
        torch.cat(psnr_values).mean().item(),
        torch.cat(ssim_values).mean().item(),
    )


def main():
    args = parse_args()
    method_dir = args.tensors_dir / args.method
    if not method_dir.is_dir():
        raise FileNotFoundError(
            f"Tensor directory for method {args.method!r} not found: {method_dir}"
        )
    operator_files, image_ids = validate_and_select_files(method_dir, args.limit)
    print(
        f"Evaluating {args.method}: {len(image_ids)} images per operator "
        f"on {args.device}..."
    )

    real_images = torch.stack(
        [load_true_tensor(args.true_dir, image_id) for image_id in image_ids]
    )
    inception = inception_v3(weights=Inception_V3_Weights.DEFAULT)
    inception.fc = torch.nn.Identity()
    inception = inception.to(args.device).eval()
    lpips = LPIPS(replace_pooling=True, reduction="none").to(args.device).eval()
    fid = FID()

    real_features = extract_features(real_images, inception, args.device, args.batch_size)
    rows = []
    for operator, display_name in OPERATORS:
        generated_images = torch.stack(
            [load_generated_tensor(operator_files[operator][image_id]) for image_id in image_ids]
        )
        if generated_images.shape != real_images.shape:
            raise ValueError(
                f"Shape mismatch for {operator}: generated {tuple(generated_images.shape)}, "
                f"ground truth {tuple(real_images.shape)}"
            )

        generated_features = extract_features(
            generated_images, inception, args.device, args.batch_size
        )
        fid_score = fid.compute_metric(real_features, generated_features).item()
        psnr_score, ssim_score = calculate_psnr_ssim(
            real_images, generated_images, args.device, args.batch_size
        )
        lpips_score = calculate_lpips(
            real_images, generated_images, lpips, args.device, args.batch_size
        )
        rows.append(
            {
                "operator": display_name,
                "psnr": psnr_score,
                "ssim": ssim_score,
                "lpips": lpips_score,
                "fid": fid_score,
            }
        )

    print(
        tabulate(
            rows,
            headers={
                "operator": "Operator",
                "psnr": "PSNR",
                "ssim": "SSIM",
                "lpips": "LPIPS",
                "fid": "FID",
            },
            tablefmt="github",
            floatfmt=".4f",
        )
    )


if __name__ == "__main__":
    main()
