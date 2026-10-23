# Repository Notes

## Setup and execution

- This directory is an experiment script, not an installable package. Install its declared dependencies with `pip install -r requirements.txt`.
- `RpCN_sample.py` also imports the separately packaged `spt_guidance`, which is not in `requirements.txt`. In this workspace it is the sibling `../RpCN_main`; install it with `pip install ../RpCN_main`.
- Run sampling from the repository root because model, data-output, and operator paths are relative to the current working directory: `python RpCN_sample.py --operator motion_blur [--img_id 60083] [--seed 31] [--save] [--plot]`.
- Sampling is CUDA-only as currently wired (`model.cuda()`, operator defaults, and the explicit `"cuda"` device). A full run is expensive: the checked-in diffusion config uses 1,000 steps and RpCN performs 10 updates every 25 steps across 3 chains.
- There is no configured test, lint, formatter, or CI suite. Use `python -m compileall -q RpCN_sample.py inverse_utils.py utils.py model rewards analysis` as the asset-free syntax check; a sampling smoke test requires CUDA and the assets below.

## Required assets and paths

- The model loader is authoritative over the README: `utils.load_model_and_sampler()` hardcodes `model/checkpoint/ffhq_10m.pt` plus the two tracked YAML files in that directory. The README's `ffhq_10.pt` name will not load unless the code is changed.
- FFHQ images are hardcoded to `~/.cache/kagglehub/datasets/weideng056/ffhq-1k-valid/versions/1/samples/<id>.png` in both sampling and analysis code.
- Sampling always writes `generated/figures/<id>_ground_truth.pdf`, even without `--plot`. `--save` writes `generated/tensors/<id>_<operator>_<seed+2435*id>.pt`; `--plot` writes a reconstruction PDF under `generated/figures/`. These paths are gitignored.

## Code map and conventions

- `RpCN_sample.py` owns experiment hyperparameters and connects `spt_guidance.torch.rpcn` to `inverse_utils.inverse_utils`; `model/` is the local FFHQ DDPM implementation; `rewards/DAPS_operators/__init__.py` is the operator registry and likelihood implementation.
- Use registry names exactly. The core CLI choices are `down_sampling`, `gaussian_blur`, `motion_blur`, and `inpainting_random` (not README's `random_inpainting`). The registry also contains `down_sampling_16x`, `phase_retrieval`, `high_dynamic_range`, and `nonlinear_blur`; the last references an untracked `bkse` subtree and is not runnable from this checkout.
- `--seed` only calls `torch.manual_seed`. Random inpainting masks and motion-blur kernels use NumPy randomness, so do not claim runs of those operators are fully reproducible without also seeding NumPy.
- Operator outputs and model samples use `[B, C, H, W]`, 256x256 RGB tensors. Model-space images are normalized to `[-1, 1]`; metric helpers convert back to `[0, 1]`.
- Analysis scripts assume they are launched from `analysis/` because they read `../ids.csv` and `../generated/tensors`. Their imports require undeclared packages including `pandas`, `piq`, `prettytable`, `wandb`, and `torchmetrics`.
- `inverse_problem_job.sh` is a machine-specific GNU Parallel launcher: it depends on `env_parallel` being initialized by `~/.bashrc`, reads `ids.csv`, and hardcodes GPU IDs 2-7 and the `inpainting_random` operator. Adjust those deliberately rather than treating it as a portable default.
