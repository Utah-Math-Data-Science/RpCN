# AGENTS

## Layout
- Flat uv project (`[tool.uv] package = false`): every `.py` file lives in the repo root; there is no `src/` package.
- Notebooks (marimo):
  - `ttg_1d.py`: 1D GMM steering comparison (FK, DAS, CREPE, RpCN) at a matched NFE budget.
  - `rpcn_sweep_1d.py`: RpCN replicas x pCN-steps sweep; imports `build_problem`, `measure`, `rpcn_nfe`, `run_rpcn` from `ttg_1d.py`.
- Helper modules: `model.py` (`EnsembleDenoiser`), `prob_path.py` (`EDM`), `plots.py` (`save_all_subfigures`).
- `style.mplstyle` is loaded by relative path, so run notebooks from the repo root.

## Setup
- Python `>=3.13`; install with `uv sync`.
- `ttg` is a git dependency on the `main` branch of `https://github.com/Utah-Math-Data-Science/RpCN` (see `[tool.uv.sources]` in `pyproject.toml`).
- `torch` is pinned to the custom `pytorch` index in `pyproject.toml`.

## Running
- `uv run marimo edit ttg_1d.py` (or `rpcn_sweep_1d.py`). Both default to `device='cuda'`.
- Figures are saved to `~/Overleaf/RpCN/imgs/toy_example/`.
- Syntax check: `uv run python -m compileall -q .`

## Marimo notebook editing guardrails
- Use `@app.cell` function cells only (`def _(...): ... return ...`); avoid raw top-level code blocks.
- Do not introduce shared variable names across cells; use unique per-cell locals.
- Prefer a single final `return`; avoid early `return` branches where practical.
- Make every used symbol an explicit dependency in the cell signature.
- Keep shared artifacts (for example histogram bins/edges) in one dedicated cell and pass them downstream.
- After notebook edits, run a quick audit:
  - no `app._unparsable_cell(...)`
  - no duplicate names introduced across edited cells
  - no unresolved symbol dependencies between edited cells
- If Marimo runtime still errors after file fixes, reload notebook from disk and run all cells to clear stale DAG state.
