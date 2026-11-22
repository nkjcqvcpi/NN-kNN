# AGENTS.md

## Branch scope

This checkout is the `nnknn/` worktree on `main`. Follow the parent workspace AGENTS.md. Neural CBR/T1, RL/T2 (including NN-kNN-RL), and LLM/T3 work belongs to `../neural-cbr/`, `../rl/`, and `../llm/` respectively; do not add stage implementations here.

`main` owns the shared NN-kNN core. Read `docs/NNKNN_CORE.md` before editing any file listed in `tools/check_nnknn_core.py`. Core changes are made here first, in commits that touch only core files, then copied unchanged to the stage branches (stage branches follow `main`).

## Repository expectations

- This repository is a Python classification/regression research workflow centered on `model/` and `datasets/`.
- For regression work, prefer the helpers in `model/regression_workflow.py` instead of calling low-level training code directly.
- Regression configs must explicitly set `task_type="regression"`. The notebook does this already; ad hoc scripts should too.
- Core files must not import stage code (`model.t1`, `model.t2`, `model.t3`, RL workflows, `datasets.rl_tasks`), even inside functions. `tests/test_nnknn_core.py` enforces this.
- After changing a core file, refresh the manifest with `uv run python tools/check_nnknn_core.py --write` and record in the commit message that the stage branches need the same files.
- `model/nnknn_model_1_4.py`, `model/nnknn_model_with_post_mlp.py`, `model/nn_cdh_old.py`, `model/cls_model.py`, `model/reg_model.py`, `model/module.py`, and `older versions/` are legacy. They are outside the core and are kept on `main` only.
- Obsolete tracked content is deleted, not archived; git history keeps it.
- Use synthetic datasets for quick validation unless a task specifically requires the larger real datasets.
- `datasets/` in the repo holds loader code only (`datasets/*.py`). Data files and preprocessed caches are not tracked: they live in the shared `NN-KNN/datasets/` next to the worktrees (or `$NNKNN_DATASETS`), found through `datasets.paths.datasets_root()`; datasets that a library downloads (scikit-learn's California Housing, OpenML) stay in that library's default location instead (`~/scikit_learn_data` or `$SCIKIT_LEARN_DATA`; the `openml` cache), and torchvision images, which have no library default, stay in the shared directory. Never add data files to git (`.gitignore` blocks everything under `datasets/` except `*.py`). No model weights are kept in the repo.

## Layout and project rules

- This project has no `CHANGELOG.md` or `LICENSE`; `README.md` covers the venv and dataset setup. Durable guidance is in this file, `HANDOFF.md`, and `docs/` (`docs/NNKNN_AGENT_GUIDE.md` is the long agent guide).
- Notebooks live in `notebooks/` (`nnknn_sample_classification.ipynb`, `nnknn_sample_regression.ipynb`). They assume the repo root as the working directory.
- Environment and dependencies are managed by uv only. The single `pyproject.toml`, `uv.lock` and `.python-version` (Python 3.14) live on `main`; one venv at `NN-KNN/.venv` is shared by all worktrees (`NN-KNN/{main,rl,llm,neural-cbr}`). Set the user environment variable `UV_PROJECT_ENVIRONMENT` to its absolute path once per device. In `main` run `uv sync` and `uv run python ...`; in another worktree run `uv run --project ../main --no-sync python ...`. torch is the Intel Arc XPU build (Windows x86_64 only; `environments` in the toml). No pip, no `requirements*.txt`, no hand-made venv. wandb was removed; experiment results are to be logged with TensorBoard (TODO, not built yet).

## Setup

- Install with `uv sync` in `main` (creates or updates `NN-KNN/.venv`). Use the same venv from every worktree as described above.
- The data must exist at `NN-KNN/datasets/` (or set `NNKNN_DATASETS`); loaders raise `FileNotFoundError` instead of downloading into a missing directory.
- Cloud and Codex setup scripts no longer exist (`codex/setup.sh`, `codex/maintenance.sh` were deleted); only `codex/smoke_test.py` remains.

## Validation

- Fast import check: `uv run python codex/smoke_test.py --mode imports`
- Fast training smoke tests: `uv run python codex/smoke_test.py --mode train` and `--mode classification`
- Core contract: `uv run python -m pytest tests -q --basetemp=<new-temp-directory>`
- Cross-branch core check: `uv run python tools/check_nnknn_core.py --branches main neural-cbr rl LLM`

## Notes

- `checkpoints/` and transient artifacts are expected and are gitignored.
- Completed prior-paper reruns are filed under `results/prior_baseline/`; see `results/README.md`.
- `matplotlib` should run headlessly in cloud tasks with `MPLBACKEND=Agg`.
- Avoid full multi-run benchmarks for routine verification; they are intentionally expensive.
