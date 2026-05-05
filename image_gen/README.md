## Overview

This branch provides an implementation of **RpCN** for **text-to-image generation with Stable Diffusion**.

## Usage

To run the sampling procedure with the hyperparameters used in the paper, specify:

- the Stable Diffusion variant using the `--model` argument
- the desired prompt identifier using the `--img_id` argument

For example:

```bash
python RpCN_sample.py --model v1.5 --img_id 004971-0071
```

## Reproducing Paper Figures

To reproduce the qualitative image-generation examples from the paper, run:

```bash
python RpCN_sample.py --model v1.5 --img_id 004971-0071
python RpCN_sample.py --model v1.5 --img_id 000562-0011
```

These commands generate samples for the following prompts:

| Prompt ID | Prompt |
|---|---|
| `004971-0071` | `a sports car, motion blur` |
| `000562-0011` | `a coffee cup made of cardboard` |

## Reproducing Paper Results

To reproduce the quantitative image-generation results reported in the paper, run:

```bash
bash bash_scripts/image_gen_job.sh
```

Before running the script, ensure that the `--model` argument is set to the desired Stable Diffusion variant.

Supported model options include:

```text
v1.4
v1.5
v2.1
```