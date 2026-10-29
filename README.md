## Overview

This repository evaluates TTG implementations of RpCN, DAS, and CREPE on
noisy and noiseless FFHQ inverse problems. All methods use the same local DDPM
and DAPS forward operators. The experiment entry points are separated by
observation regime:

```text
experiments/
  common.py
  noisy/
    rpcn.py
    das.py
    crepe.py
    make_measurements.py
    jobs/
  noiseless/
    rpcn.py
    das.py
    crepe.py
    jobs/
```

The samplers use the `ttg` package only; `spt_guidance` is not required.

## Installation

Install the declared dependencies and the local TTG package used in this
workspace:

```bash
pip install -r requirements.txt
pip install ../RpCN_updated/RpCN
```

Place the DPS FFHQ checkpoint at:

```text
model/checkpoint/ffhq_10m.pt
```

The default dataset directory is `~/DAPS/dataset/test-ffhq`, containing images
named `<id>.png`. Override it with `--data_dir`.

Run all commands from the repository root so the model configuration and
operator assets resolve consistently.

## Noisy Measurements

Noisy experiments use

```text
y = A(x) + sigma * epsilon, epsilon ~ N(0, I)
```

with default model-space noise `sigma=0.05`. Measurements are cached once and
shared by every method. Generate the full cache with:

```bash
python -m experiments.noisy.make_measurements \
    --noise_sigma 0.05 \
    --operator_seed 42 \
    --measurement_seed 42
```

The generator intentionally loads all IDs from `ids.csv` and draws one CUDA
noise batch per operator, matching `DAPS/commands/pixel_ffhq.sh`. Cached
artifacts are written under `generated/measurements/noisy/sigma_0.05/` and
contain the noisy observation, seeds, operator configuration, dataset index,
and degradation fingerprint.

Numerically verify the cache against the implementation in `~/DAPS` with:

```bash
python -m experiments.verify_daps_compatibility
```

The verifier reconstructs `pixel_ffhq` through DAPS's dataset and operator
modules and requires bitwise equality for both clean and noisy observations.

## Sampling

Run noisy methods with the same cached measurement:

```bash
python -m experiments.noisy.rpcn --img_id 49000 --operator motion_blur --save
python -m experiments.noisy.das --img_id 49000 --operator motion_blur --save
python -m experiments.noisy.crepe --img_id 49000 --operator motion_blur --save
```

Noisy samplers require the cached batched observation and fail if it is
missing. This prevents accidental fallback to a per-image RNG scheme that
would not match DAPS.

Run the corresponding noiseless methods with:

```bash
python -m experiments.noiseless.rpcn --img_id 49000 --operator motion_blur --save
python -m experiments.noiseless.das --img_id 49000 --operator motion_blur --save
python -m experiments.noiseless.crepe --img_id 49000 --operator motion_blur --save
```

Outputs are isolated by regime, noise level, method, operator, and sampling
seed:

```text
generated/inverse_problems/
  noisy/sigma_0.05/RpCN/motion_blur/sample_seed_42/49000.pt
  noiseless/CREPE/down_sampling/sample_seed_31/49000.pt
```

GNU Parallel launchers are under `experiments/noisy/jobs/` and
`experiments/noiseless/jobs/`. The noisy `run_all.sh` compares RpCN and DAS;
the noiseless launcher also includes CREPE. Both cover all four core operators
and IDs in `ids.csv`.

## Noisy Experiment Protocol

The intended `sigma=0.05` evaluation is staged to avoid expensive invalid
full runs:

1. Generate the full 100-image DAPS-compatible measurement cache.
2. Run all three methods with reduced budgets and verify finite samples,
   measurement MSE, normalized residual, and runtime.
3. Run the first 12 IDs over all four operators with full budgets.
4. Repeat the 12-ID pilot with three sampling seeds to assess variability.
5. Run all 100 IDs only after selecting one configuration per method.

The noisy reward is the calibrated Gaussian log likelihood:

```text
-||A(x) - y||^2 / (2 * sigma^2)
```

This matches the declared observation model. DAPS's pixel-space Langevin
sampler separately uses `tau=0.01`; that sampler hyperparameter does not
redefine the shared observation noise.

Each run reports measurement MSE, measurement MSE divided by `sigma^2`, and
negative log likelihood in addition to image-space metrics.

## Evaluation

Evaluate a complete method/seed output with:

```bash
python analysis/DAPS_metrics.py \
    --regime noisy \
    --noise-sigma 0.05 \
    --sample-seed 42
```

Use `--regime noiseless` for exact-observation outputs. The evaluator compares
DAS and RpCN in one table and reports PSNR, SSIM, LPIPS, and FID for each
operator.

## External Assets

- The pretrained FFHQ DDPM checkpoint is from DPS.
- Forward operator implementations are adapted from DAPS.
- The FFHQ validation set is distributed through Kaggle.
