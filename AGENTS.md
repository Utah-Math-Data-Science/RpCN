# AGENTS

## Current repo reality
- Active code is Hydra + hydra-orm steering experiments under `src/toy/` and `src/conf/`.
- `README.rst` is empty; trust code/config over docs.
- `pyproject.toml` script entrypoints are stale: `toy = "toy:main"` and `toy-train = "toy.train_toy.train:main"` do not exist in this tree. Run file paths directly with `uv run python ...`.

## Setup and dependencies
- Python `>=3.13`; install with `uv sync`.
- `spt-guidance` is a local editable dependency at `../Mean_Flow_Steering`; steering runs fail if that sibling repo is missing.
- `torch` is pinned to the custom `pytorch` index in `pyproject.toml`.

## Real entrypoint and command shape
- Main entrypoint: `src/toy/main.py`.
- Canonical run pattern:
  - `uv run python src/toy/main.py +experiment=GMM steering=RpCN steering.pcn_update_count=50`
- `experiment` is a preset group (`conf/experiment/GMM.yaml`) that selects a `setting` schema (`conf/setting/GMMExperiment.yaml` -> `_GMMExperiment`).
- The experiment preset file must stay `# @package _global_`; otherwise Hydra composition can fail with key errors.
- Keep defaults in structured configs (`src/conf/*.py`), not duplicated in preset YAML wrappers.
- `steering` is required (`Conf.defaults` sets it to `MISSING`), so always pass `steering=Unsteered|FK|RpCN` or provide it via a preset.

## Hydra/hydra-orm wiring quirks
- Structured config tables are in `src/conf/conf.py`, `src/conf/experiment.py`, `src/conf/steering.py`.
- `Conf.setting` is persisted and typed as `GMMExperiment`; `Conf.score_model` is a separate persisted table.
- `vmap_chunk_size` is intentionally non-persisted (`field(default=8192)`), not an ORM column.
- `get_run_dir()` rejects Hydra multirun flags (`-m` / `--multirun`); use single-run invocations.

## Outputs and logging
- Runs persist config rows via `orm.instantiate_and_insert_config(...)` and write to `generated/runs/<alt_id>/`.
- Current artifact contract is a single file: `samples.csv` with columns `sample_id,x,reward`.
- Logging style mirrors DAFM: command, pretty-printed resolved config, output directory, saved artifact path.

## Verification commands that reflect current state
- Syntax check: `uv run python -m compileall src`
- Steering smoke test: `uv run python src/toy/main.py +experiment=GMM steering=RpCN steering.pcn_update_count=5 device=cpu`
- `uv run pytest -q` currently has no active test modules after removing stale `train_toy`/`steer_toy` tests.
