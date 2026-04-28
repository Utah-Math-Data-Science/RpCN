# AGENTS

## Repo Reality (verify first)
- This is a small Python project managed with `uv` (`pyproject.toml`, `uv.lock`), with source under `src/toy/`.
- `README.rst` is empty; treat code + `pyproject.toml` as source of truth.
- There are currently no test, lint, typecheck, CI, or pre-commit configs in this repo.

## Setup / Dependencies
- Python requirement is `>=3.13`.
- Install with `uv sync`.
- `spt-guidance` is a local editable dependency at `../Mean_Flow_Steering`; many steering scripts depend on it. If that sibling repo is missing, steering code will not run.
- `torch` is resolved from the custom `pytorch` index (`https://download.pytorch.org/whl/cu128`).

## Entry Points That Matter
- Training code lives in `src/toy/train_toy/`.
- Steering/sampling code lives in `src/toy/steer_toy/`.
- Do not rely on `uv run toy`: `pyproject.toml` points to `toy:main`, but no `main()` exists in `src/toy/`.

## Import + Execution Gotchas
- Use Python import grouping/order consistently: built-in imports first, then third-party imports, then local/package imports, with one blank line between groups.
- **Important:** use absolute package imports in training code (e.g., `from toy.train_toy.distributions import ...`, `from toy.train_toy.model import ...`).
- Running training as a file path (e.g., `python src/toy/train_toy/train.py`) requires the `toy` package to be importable in the active environment (typically via `uv sync`).
- Use absolute package imports in steering code (e.g., `from toy.steer_toy.log_reward import ...`, `from toy.steer_toy.model.model import ...`).
- Running steering as a file path (e.g., `python src/toy/steer_toy/RpCN_sample.py`) requires the `toy` package to be importable in the active environment (typically via `uv sync`).
- `src/toy/train_toy/` and `src/toy/steer_toy/model/` each define a `DiffusionModel`; keep architectures in sync if either copy changes.

## Artifact / Path Assumptions
- Training saves `diffusion_gmm_model.pt` in the current working directory.
- Steering scripts load checkpoint from `model/diffusion_gmm_model.pt` relative to their runtime working directory.
- `plot_samples` in `src/toy/steer_toy/utils.py` writes to `generated/<name>` and does not create the directory.
