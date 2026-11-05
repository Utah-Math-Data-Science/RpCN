## Overview

This branch provides an implementation of **RpCN** for **text-to-image generation with Stable Diffusion**.

## Virtual Environment

Setup a virtual environment with all required dependencies by running 
```uv sync ``` 

## Usage

To run the sampling procedure with the hyperparameters used in the paper, specify:

- the Stable Diffusion variant using the `--model` argument
- the desired prompt identifier using the `--img_id` argument

For example:

```bash
rpcn-sample --model v1.5 --img_id 004971-0071
```

## Reproducing Paper Figures

To reproduce the qualitative image-generation examples from the paper, run:

```bash
rpcn-sample --model v1.5 --img_id 004971-0071
rpcn-sample --model v1.5 --img_id 000562-0011
```

These commands generate samples for the following prompts:

| Prompt ID | Prompt |
|---|---|
| `004971-0071` | `a sports car, motion blur` |
| `000562-0011` | `a coffee mug made of cardboard` |

## Reproducing Paper Results

To reproduce the quantitative image-generation results reported in the paper, run:

```bash
bash image_gen_job.sh
```


The job runs all three Stable Diffusion models. DAS is run with both
`alpha_kl=0.1` and `alpha_kl=0.01`.

Results are kept separate by configuration:

```text
generated/<model>/DAS_0.1/
generated/<model>/DAS_0.01/
```

## Evaluation

Evaluate all methods and configurations in ordered tables with:

```bash
rpcn-evaluate
```

The evaluator reports the mean of each prompt's maximum image reward and the
mean pairwise CLIP diversity among its four samples. It requires every prompt
to be complete by default; pass `--allow_missing` to evaluate partial runs.


## Reward Model

Our text-to-image experiments use **ImageReward** as the reward model. The reward model implementation is taken from the official ImageReward codebase and used to score generated images during test-time guidance.

Please refer to the original ImageReward repository and paper for installation details, licensing information, and usage terms.
