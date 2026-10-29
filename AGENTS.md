# Repository Notes

## Setup and execution

- This repository contains experiment modules, not an installable package. Install dependencies with `pip install -r requirements.txt`.
- The samplers use the separately packaged `ttg` library. In this workspace install it with `pip install ../RpCN_updated/RpCN`.
- Do not add or use `spt_guidance`; all RpCN, DAS, and CREPE experiments use TTG.
- Run experiment modules from the repository root: `python -m experiments.noisy.rpcn --operator motion_blur --img_id 49000`.
- Sampling is CUDA-only and expensive. Use `python -m compileall -q experiments utils.py model rewards analysis` for an asset-free syntax check.

## Experiment organization

- `experiments/noisy/` contains TTG RpCN, DAS, and CREPE entry points for Gaussian-noise measurements.
- `experiments/noiseless/` contains the corresponding exact-observation entry points.
- `experiments/common.py` owns the discrete VP schedule, seeds, operator construction, measurement cache, output paths, and consistency metrics.
- Noisy measurements use `y = A(x) + sigma * epsilon` and are cached under `generated/measurements/noisy/sigma_<sigma>/<operator>/<id>.pt`.
- `experiments.noisy.make_measurements` reproduces DAPS `pixel_ffhq` batching and CUDA RNG order; do not replace it with per-image noise draws.
- `python -m experiments.verify_daps_compatibility` checks bitwise equality against `~/DAPS` for clean and noisy observations.
- Noisy rewards use the calibrated Gaussian log likelihood `-||A(x)-y||^2/(2*sigma^2)`.
- Generated samples are stored under `generated/inverse_problems/<regime>/[sigma_<sigma>/]<method>/<operator>/sample_seed_<seed>/<id>.pt`.
- GNU Parallel launchers are grouped under each regime's `jobs/` directory and derive the repository root from their own location.

## Required assets and paths

- `utils.load_model_and_sampler()` loads `model/checkpoint/ffhq_10m.pt` and the tracked YAML files in that directory.
- The default FFHQ directory is `~/DAPS/dataset/test-ffhq`; samplers accept `--data_dir`.
- The four core registry names are `down_sampling`, `gaussian_blur`, `motion_blur`, and `inpainting_random`.
- Model-space images are in `[-1, 1]`; noisy `sigma=0.05` corresponds to approximately `6.4/255` in display space.
- Sampling seeds, operator seeds, and measurement-noise seeds are independent. Keep operator and measurement seeds identical across methods for fair comparisons.

## Analysis

- `analysis/DAPS_metrics.py` reads the organized experiment hierarchy and reports PSNR, SSIM, LPIPS, and FID.
- Select outputs with `--regime`, `--noise-sigma`, and `--sample-seed`; the evaluator compares DAS and RpCN together.
- A complete metric run requires the same image-ID set for all four operators.
- Analysis imports undeclared packages including `piq` and `tabulate`; install them separately when needed.
