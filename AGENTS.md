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
- Environment and dependencies are managed by uv only. The single `pyproject.toml`, `uv.lock` and `.python-version` (Python 3.14) live on `main`; one venv at `NN-KNN/.venv` is shared by all worktrees (`NN-KNN/{main,rl,llm,neural-cbr}`). Set the user environment variable `UV_PROJECT_ENVIRONMENT` to its absolute path once per device. In `main` run `uv sync` and `uv run python ...`; in another worktree run `uv run --project ../main --no-sync python ...`. torch 2.15.0 is the Intel Arc XPU build on Windows x86_64 (g234) and the standard PyPI build (CUDA 13.2) on Linux x86_64 (r760, used there on CPU); both platforms are in `environments` in the toml and every other package has the same version on both. On r760 the venv is `/mnt/largedata/NN-KNN/.venv` and jobs first `source /mnt/largedata/NN-KNN/env.sh`. No pip, no `requirements*.txt`, no hand-made venv. wandb was removed; training data is always recorded in TensorBoard event files by `model/common/tracking.py` (see `docs/NNKNN_CORE.md`, "Run tracking"); summaries are files under `results/`.

## Setup

- Install with `uv sync` in `main` (creates or updates `NN-KNN/.venv`). Use the same venv from every worktree as described above.
- The data must exist at `NN-KNN/datasets/` (or set `NNKNN_DATASETS`); loaders raise `FileNotFoundError` instead of downloading into a missing directory.
- Cloud and Codex setup scripts no longer exist (`codex/setup.sh`, `codex/maintenance.sh` were deleted); only `codex/smoke_test.py` remains.

## Validation

- Fast import check: `uv run python codex/smoke_test.py --mode imports`
- Fast training smoke tests: `uv run python codex/smoke_test.py --mode train --study <name>` and `--mode classification --study <name>` (point `NNKNN_RUNS` at a scratch run root; a complete smoke unit is reported, not retrained)
- Core contract: `uv run python -m pytest tests -q --basetemp=<new-temp-directory>`
- Cross-branch core check: `uv run python tools/check_nnknn_core.py --branches main neural-cbr rl LLM`

## Notes

- Owner's rules (2026-11-22, final; details in `docs/NNKNN_CORE.md`, "Owner's rules"). R1: Table 1 (`tools/table1_nnknn_kfold.py`, `tools/run_table1_kfold.py`, `tools/resume_table1_kfold.py`, and the `train_model` / regression-workflow path it calls without a run) is the upstream authors' published benchmark: its code path, outputs, checkpoint files and numbers do not change, and it has no run tracking. R2: only training data (per-epoch curves, per-run final metrics, predictions, weights) goes to TensorBoard. O1: recording is mandatory: every entry point that trains or evaluates requires `--study` (library functions require `run=` / `unit_run=`) and writes per-run data and weights under `runs/` only; there is no file-only mode. O2: summary, report and plotting scripts read only `runs/` and write to `<worktree>/results/` (names, formats, columns unchanged). O3: reports written by agents or people go to `<worktree>/reports/` (in git); generated files never go to `reports/` or `docs/`. O4: never retrain: a complete unit is detected before any training and its weights (and, where later code depends on it, its RNG state) are loaded from the unit. O5: resume files are kept after a unit completes; the device is not part of unit identity.
- Where things go: `NN-KNN/runs/` (shared, not in git) holds every run unit (`runs/core/<study>/...` for the core tools); `<worktree>/results/` (git-ignored) holds generated summaries; `<worktree>/reports/` (in git) holds hand-written reports. Tests point `NNKNN_RUNS` at a temporary directory.
- `checkpoints/` exists only for Table 1: `train_model` without a run writes its checkpoint files there as it always did, and `tools/resume_table1_kfold.py` resumes from them. Nothing else writes checkpoint files.
- Old result files on the experiment host (`results/prior_baseline/`, indexed by `results/README.md` beside them) are not touched, moved or converted, and no code reads them.
- The notebooks are not converted to mandatory recording: their cells that train without a run (`run_single_nnknn_classification_experiment`, `run_repeated_classification_model_benchmarks`, `run_repeated_regression_model_benchmarks` with `nnknn`, `run_repeated_nnknn_regression_experiments`) now raise until they pass a run.
- `matplotlib` should run headlessly in cloud tasks with `MPLBACKEND=Agg`.
- Avoid full multi-run benchmarks for routine verification; they are intentionally expensive.
