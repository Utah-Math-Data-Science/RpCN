import hashlib
import os
from pathlib import Path

import numpy as np
import torch
from ttg.kernels import Gaussian

from rewards.DAPS_operators import get_operator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = Path.home() / "DAPS/dataset/test-ffhq"
DEFAULT_MEASUREMENT_ROOT = PROJECT_ROOT / "generated/measurements"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "generated/inverse_problems"
MEASUREMENT_FORMAT_VERSION = 3
MEASUREMENT_RNG_SCHEME = "daps_cuda_batch_v1"
DAPS_PIXEL_FFHQ_IDS = tuple(range(49000, 49100))
OPERATORS = (
    "down_sampling",
    "gaussian_blur",
    "motion_blur",
    "inpainting_random",
)
OPERATOR_CONFIGS = {
    "down_sampling": {"resolution": 256, "scale_factor": 4},
    "gaussian_blur": {"kernel_size": 61, "intensity": 3.0},
    "motion_blur": {"kernel_size": 61, "intensity": 0.5},
    "inpainting_random": {
        "mask_type": "random",
        "mask_prob_range": [0.70, 0.71],
        "resolution": 256,
    },
}


class DiscreteVPSchedule:
    """Expose the trained DDPM's discrete VP schedule through the TTG API."""

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


def effective_sample_seed(seed, image_id):
    return seed + 2435 * image_id


def degradation_sha256(operator):
    artifact = getattr(operator, "mask", None)
    if artifact is None and hasattr(operator, "conv"):
        artifact = operator.conv.seq[1].weight
    if artifact is None:
        return None
    data = artifact.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def _sigma_name(sigma):
    return f"sigma_{sigma:g}"


def measurement_path(root, sigma, operator, image_id):
    return Path(root) / "noisy" / _sigma_name(sigma) / operator / f"{image_id}.pt"


def output_dir(root, regime, method, operator, sample_seed, sigma=None):
    path = Path(root) / regime
    if sigma is not None:
        path /= _sigma_name(sigma)
    return path / method / operator / f"sample_seed_{sample_seed}"


def _atomic_torch_save(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def build_operator(operator_name, device, operator_seed, sigma):
    np.random.seed(operator_seed % (2**32))
    return get_operator(
        operator_name,
        device=device,
        sigma=sigma,
        **OPERATOR_CONFIGS[operator_name],
    )


def prepare_problem(
    reference,
    operator_name,
    *,
    operator_seed,
    noisy,
    noise_sigma=0.05,
    measurement_seed=42,
    image_id,
    measurement_root=DEFAULT_MEASUREMENT_ROOT,
    create_measurement=True,
):
    """Construct a reproducible operator and its exact or cached noisy observation."""
    operator_sigma = noise_sigma if noisy else 0.0
    operator = build_operator(
        operator_name, reference.device, operator_seed, operator_sigma
    )
    clean_measurement = operator(reference).detach()
    fingerprint = degradation_sha256(operator)

    if not noisy:
        return operator, clean_measurement, None

    cache_path = measurement_path(
        measurement_root, noise_sigma, operator_name, image_id
    )
    expected = {
        "measurement_format_version": MEASUREMENT_FORMAT_VERSION,
        "rng_scheme": MEASUREMENT_RNG_SCHEME,
        "image_id": image_id,
        "dataset_index": DAPS_PIXEL_FFHQ_IDS.index(image_id),
        "source_filename": f"{image_id}.png",
        "batch_size": len(DAPS_PIXEL_FFHQ_IDS),
        "operator": operator_name,
        "operator_seed": operator_seed,
        "measurement_seed": measurement_seed,
        "noise_sigma": noise_sigma,
        "degradation_sha256": fingerprint,
        "operator_config": OPERATOR_CONFIGS[operator_name],
    }

    if cache_path.is_file():
        artifact = torch.load(cache_path, map_location="cpu", weights_only=True)
        for key, value in expected.items():
            if artifact.get(key) != value:
                raise ValueError(
                    f"Cached measurement metadata mismatch for {cache_path}: "
                    f"{key}={artifact.get(key)!r}, expected {value!r}"
                )
        measurement = artifact["measurement"].to(
            device=reference.device, dtype=reference.dtype
        )
        return operator, measurement, cache_path

    if not create_measurement:
        raise FileNotFoundError(
            f"No cached measurement at {cache_path}; run "
            "python -m experiments.noisy.make_measurements first"
        )
    raise FileNotFoundError(
        f"DAPS-compatible observations must be generated as one ordered batch. "
        f"No cached measurement at {cache_path}; run "
        "python -m experiments.noisy.make_measurements first"
    )


def measurement_metrics(operator, sample, measurement, noise_sigma=None):
    residual = operator(sample) - measurement
    mse = residual.square().mean()
    values = {"measurement_mse": mse}
    if noise_sigma is not None:
        values["normalized_residual"] = mse / noise_sigma**2
        values["negative_log_likelihood"] = (
            residual.square().flatten(1).sum(-1) / (2 * noise_sigma**2)
        ).mean()
    return values
