## Overview

This repository evaluates RpCN, DAS, and CREPE on
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

## Virtual Environment 

To setup a virtual environment for this branch run 
```bash    uv sync   ```


## Pretrained Model 
Download the FFHQ model from https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh


After downloading, place the DPS FFHQ checkpoint at:
```text
model/checkpoint/ffhq_10m.pt
```


Run all commands from the repository root so the model configuration and
operator assets resolve consistently.


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



## Evaluation

Evaluate a complete method/seed output with:

```bash
python analysis/DAPS_metrics.py \
    --regime noiseless \
    --noise-sigma 0.05 \
    --sample-seed 42
```

Use `--regime noiseless` for exact-observation outputs. The evaluator compares
DAS and RpCN in one table and reports PSNR, SSIM, LPIPS, and FID for each
operator.



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
