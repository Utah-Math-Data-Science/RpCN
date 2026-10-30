## Overview

This branch provides an implementation of **RpCN** for **text-to-image generation with Stable Diffusion**.

## Installation

Install the RpCN package by running ``pip install ../main/.``

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

## DAS Experiments

Run the complete RpCN, CREPE, and DAS sweep with:

```bash
bash image_gen_job.sh
```

The job runs all three Stable Diffusion models. DAS is run with both
`alpha_kl=0.1` and `alpha_kl=0.01`.

Results are kept separate by configuration:

```text
generated/<model>_2/DAS_0.1/
generated/<model>_2/DAS_0.01/
```

For a single prompt, `--output_label` is optional because it is derived from
`--alpha_kl`:

```bash
python DAS_sample_new.py --model v1.5 --img_id 004971-0071 --alpha_kl 0.01
```

Evaluate all methods and configurations in ordered tables with:

```bash
python evaluate_results.py
```

The evaluator reports the mean of each prompt's maximum image reward and the
mean pairwise CLIP diversity among its four samples. It requires every prompt
to be complete by default; pass `--allow_missing` to evaluate partial runs.


## Reward Model

Our text-to-image experiments use **ImageReward** as the reward model. The reward model implementation is taken from the official ImageReward codebase and used to score generated images during test-time guidance.

Please refer to the original ImageReward repository and paper for installation details, licensing information, and usage terms.
