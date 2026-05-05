## Overview

This branch provides an implementation of **RpCN** for solving image inverse problems on the **FFHQ dataset**.

## Before Running

Before running the code, download the pretrained FFHQ DDPM model and the FFHQ 1K validation set.

### Pretrained FFHQ Model

To download the pretrained FFHQ model, run:

```bash
bash download_model.sh
```

### FFHQ 1K Validation Set

Download the FFHQ 1K validation set from Kaggle:

```text
https://www.kaggle.com/datasets/weideng056/ffhq-1k-valid
```

Follow the instructions on the Kaggle page to download and place the dataset in the expected data directory.

## Usage

To run the sampling procedure with the hyperparameters used in the paper, specify the desired inverse problem using the `--operator` argument.

For example, to run RpCN on the motion deblurring task:

```bash
python RpCN_sample.py --operator motion_deblur
```

Supported operators include:

```text
super_resolution
gaussian_blur
motion_blur
random_inpainting
```