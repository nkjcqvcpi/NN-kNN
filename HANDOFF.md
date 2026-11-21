# Handoff

Branch `main` (`nnknn/` worktree): the standalone NN-kNN model and the shared core. Neural CBR (T1), RL (T2), and LLM (T3) status lives in the handoffs on `neural-cbr`, `rl`, and `LLM`. The RL status that used to be recorded here is preserved in `rl`'s `HANDOFF.md` and in the history of this file.

## Current Status

- The maintained NN-kNN core supports both regression and classification.
- 2026-10-10: `main` adopted the reconciled shared core from `neural-cbr` (`model/nnknn_model.py`, `model/nn_cdh.py`, `model/device_utils.py`, `model/common/`). The core now has a written contract (`docs/NNKNN_CORE.md`), a hash manifest (`tools/nnknn_core_manifest.json`), a checker (`tools/check_nnknn_core.py`), and tests (`tests/test_nnknn_core.py`).
- The same date fixed a stage dependency in the core: the classification adapter path imported `model.t1.nominal`, which exists only on `neural-cbr`. It now imports `model.common.nominal`.
- Classification uses normalized case activation as class probability mass with NLL loss; it does not restore the older class-weight formulation.
- `model/classification_workflow.py` and `datasets/classification_data.py` provide tabular and image classification workflows and representative baselines. `model/regression_workflow.py` and `tools/table1_nnknn_kfold.py` provide regression benchmarks and Table 1 reruns.
- `notebooks/nnknn_sample_classification.ipynb` and `notebooks/nnknn_sample_regression.ipynb` are the maintained notebook entry points; they call the Python workflows. They assume the repo root as the working directory, so launch them with the repo root as cwd.
- 2026-10-12 (commit `e1b1d00`): accelerator support was added; the core's device handling (`model/device_utils.py`) covers CPU, CUDA, and XPU (see `docs/NNKNN_CORE.md`).
- A June 1 NN-kNN-only 10-fold rerun confirmed the recorded Iris and Zebra representative results exactly. These are current-core functionality checks, not exact reproductions of IJCAI-25 results from the older classification formulation. The run is in `results/prior_baseline/ijcai2025_classification/classification_nnknn_rerun_20260601_103353/`.

## Current Work Focus

- Classification support is functional; work on the core focuses on improving scores on the earlier IJCAI-25 classification tasks while keeping the maintained IJCAI-26-based retrieval implementation.
- Do not restore the retired legacy case-weight classification path solely to reproduce old numbers. Improvements should come from the current workflow, dataset protocol checks, hyperparameter tuning, or appropriate current-core changes.
- The June 1 rerun (10-fold stratified CV, `softmax`, `tau=0.5`, 50 epochs, seed 42) reproduced:
  - `iris`: `0.9600 +/- 0.0562`
  - `zebra` / Zebra (a): `0.5182 +/- 0.1488`
  - `zebra_special` / Zebra (b): `0.5227 +/- 0.0890`
- Zebra (a) and (b) are close to chance and are the immediate debugging priority. Inspect their alternating-boundary structure, preprocessing/splitting protocol, retrieved cases, feature weights, temperature, normalizer choice, and possible current-core feature mapping. Use `"zebra"` for Zebra (a) and `"zebra_special"` for Zebra (b) in the classification notebook.
- Before expanding to full prior-paper comparisons, establish better representative results on Zebra, then rerun the remaining small classification suite (`wine`, `breast_cancer`, `balance`, `digits`).
- T1 mechanisms (retention, reuse, synchronization, MCB, revision) are developed on `neural-cbr`. When T1, T2, or T3 work needs a core change, make it here first and propagate it (see `docs/NNKNN_CORE.md`).

## Pending Core Propagation

- `rl` and `LLM` receive the core from this branch directly. Stage branches follow `main`.
- `neural-cbr` is running the T1 full study (started 2026-10-09). Its manifests record `git HEAD` and a source fingerprint over `model/ tools/ configs/ tests/ datasets/`, so the import fix and core contract are staged for it and applied only after the study finishes. Until then, `tools/check_nnknn_core.py --branches neural-cbr` reports `model/nnknn_model.py` and the contract files as different.

## Current Local Artifacts

- `checkpoints/` and transient smoke-test artifacts are expected and gitignored.
- Root checkpoint files such as `nnknn_regression_best.pth` and `nnknn_regression_best_retr.pth` may change when experiments are run.
- Classification runs write checkpoints beneath `checkpoints/` by default. The CLI creates a fresh `results/classification_<suite>_<timestamp>/` folder with summary, per-run, and manifest files.
- Completed prior-paper reruns are filed under `results/prior_baseline/`; see `results/README.md`. Keep old run folders for comparison and postmortem context rather than overwriting them.

## Where Durable Guidance Lives

Read `docs/NNKNN_AGENT_GUIDE.md` for model descriptions, workflow entry points, Table 1 protocol details, output schemas, restart/resume patterns, and machine-specific caveats. Read `docs/NNKNN_CORE.md` before changing any shared core file.

## Environment (uv only)

Environment and dependencies are managed by uv only. The single `pyproject.toml`, `uv.lock` and `.python-version` (Python 3.14) live on `main`; one venv at `NN-KNN/.venv` is shared by all worktrees (`NN-KNN/{main,rl,llm,neural-cbr}`). Set the user environment variable `UV_PROJECT_ENVIRONMENT` to its absolute path once per device. In `main` run `uv sync` and `uv run python ...`; in another worktree run `uv run --project ../main --no-sync python ...`. torch is the Intel Arc XPU build (Windows x86_64 only; `environments` in the toml). No pip, no `requirements*.txt`, no hand-made venv. wandb was removed; experiment results are to be logged with TensorBoard (TODO, not built yet).

## 2026-11-21: repository layout cleanup

- Removed: `README.md`, `requirements.txt`, `requirements-latest.txt`, `verify_packages.py`, `codex/README.md`, `codex/setup.sh`, `codex/maintenance.sh` (`codex/smoke_test.py` is kept), `tools/Sync-CodexHistory.ps1`, `tools/codex-history-sync.md`, `tmp_eval_section.txt`, `model/desktop.ini`, `model/NN-kNN.code-workspace`, `notes/testFile.md`, `.DS_Store`. Obsolete tracked content is deleted (git history keeps it), not archived.
- Moved: `python_version.txt` to `.python-version`; `nnknn_sample_*.ipynb` to `notebooks/`; `NNKNN_AGENT_GUIDE.md` to `docs/NNKNN_AGENT_GUIDE.md`; `config.yaml`, `nnknn_reg_yu.py`, `selected_indices*.json`, `tools/create_nnknn_orientation_deck.py` and `model/Copy_nnknn_model.py` to `archive/legacy/`; deleted again in the 2026-11-21 uv step (below).
- Decisions (owner): this project has no README, CHANGELOG, or LICENSE; uv is the only way to manage environments and dependencies.
- The notebooks in `notebooks/` must be launched with the repo root as the working directory.

## 2026-11-21: shared uv project

- Added `pyproject.toml`, `uv.lock`, `.python-version` (Python 3.14.7). Locked: torch 2.15.0+xpu, torchvision 0.29.2+xpu, triton-xpu 3.9.0, numpy 2.5.5, scipy 1.18.1, pandas 3.1.1, scikit-learn 1.9.1, gymnasium 1.4.0, matplotlib 3.11.2, transformers 5.24.0, pytest 9.2.0. Only libraries the code imports are dependencies; `notebook` group = ipykernel, ipywidgets.
- Removed: wandb (dependency, `model/train_GMM3.py` with a hard-coded key, `archive/legacy/`). The key is public in GitHub history and must be revoked by its owner.
- Worktree directory `nnknn` renamed to `main` (g234). The three stage branches lost their own `pyproject.toml`, `uv.lock`, `.python-version`.
- Known gaps: LunarLander (Box2D) has no Python 3.14 wheel and `gymnasium[box2d]` needs MSVC + swig, so it is not installed. On torch 2.15.0+xpu two neural-cbr tests fail (`test_t1_tuning_features.py` accelerator tests, `UR_RESULT_ERROR_UNKNOWN` in the glocal feature weighting); they pass on torch 2.14.0+xpu / Python 3.12. TensorBoard logging is not implemented (TODO: rework the experiment flow).
