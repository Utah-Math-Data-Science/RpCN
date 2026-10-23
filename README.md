## Overview

This branch provides an implementation of **RpCN** for solving image inverse problems on the **FFHQ dataset**.

## Installation

Install the RpCN package by running ``pip install ../main/.``

## Before Running

Before running the code, download the pretrained FFHQ DDPM checkpoint and the FFHQ 1K validation set.

### Pretrained FFHQ DDPM Checkpoint

We use the FFHQ DDPM checkpoint released with **DPS** so that differences in inverse problem performance are attributable to the sampling/guidance method rather than to differences in the pretrained generative model.

To download it, download `ffhq_10.pt` from the following Google Drive folder
```text
https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh
```
and the model weights in the following folder.

```text
model/checkpoint/
```
### FFHQ 1K Validation Set

Download the FFHQ 1K validation set from Kaggle:

```text
https://www.kaggle.com/datasets/weideng056/ffhq-1k-valid
```

Follow the instructions on the Kaggle page to download the dataset directory.

### Forward Operators

To ensure a fair and reproducible comparison with DAPS, we use the same forward operator implementations as the DAPS codebase for the inverse problem experiments. This avoids discrepancies arising from differences in degradation models, such as blur kernel construction, masking procedures, or super-resolution/downsampling conventions.

## Usage

To run the sampling procedure with the hyperparameters used in the paper, specify the desired inverse problem using the `--operator` argument.

For example, to run RpCN on the motion deblurring task:

```bash
python RpCN_sample.py --operator motion_blur
```

Supported operators include:

```text
down_sampling (super resolution)
gaussian_blur
motion_blur
random_inpainting
```

## Guidance ablation

`guidance_ablation.sh` runs early, middle, late, and constant schedules at
guidance strengths `0.25`, `0.5`, `1.25`, `2.5`, and `5.0`. Preview the 240
commands for the 12 IDs in `ids.csv` without running them:

```bash
./guidance_ablation.sh --operator down_sampling --dry-run
```

Execute the matrix on one or more GPUs. Each GPU receives a sequential queue,
and completed tensors are skipped when the command is resumed:

```bash
./guidance_ablation.sh \
    --operator down_sampling \
    --gpus 0,1,2,3
```

Results are stored under
`generated/ablations/<operator>/<configuration>/tensors`. Evaluate all complete
configurations with:

```bash
python analysis/guidance_ablation_metrics.py \
    --operator down_sampling \
    --output generated/ablations/down_sampling/metrics.csv
```

Use `--limit 3` for a cheaper screening pass. Override the default strengths
with, for example, `--strengths "0.5 1.0 2.0"`. The script resumes by skipping
existing tensors; pass `--rerun` to replace them.

## External Assets and Attribution

This code uses several external assets:

- The pretrained FFHQ DDPM checkpoint is from **DPS**.
- The forward operator implementations are adapted from **DAPS**.
- The FFHQ 1K validation set is provided through Kaggle.

Please refer to the original DPS and DAPS repositories/papers for additional details, licenses, and usage terms.
