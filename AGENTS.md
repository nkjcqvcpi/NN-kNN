# Branch scope: NN-kNN core on main

This checkout is `nnknn/` on `main`. Follow the parent workspace AGENTS.md. Neural CBR/T1, RL/T2 (including NN-kNN-RL), and LLM/T3 work belongs to `../neural-cbr/`, `../rl/`, and `../llm/` respectively; do not add stage implementations here.

`main` owns the shared NN-kNN core. Read `docs/NNKNN_CORE.md` before editing any file listed in `tools/check_nnknn_core.py`. Core changes are made here first, in commits that touch only core files, then copied unchanged to the stage branches.

# AGENTS.md

## Repository expectations

- This repository is a Python classification/regression research workflow centered on `model/` and `datasets/`.
- For regression work, prefer the helpers in `model/regression_workflow.py` instead of calling low-level training code directly.
- Regression configs must explicitly set `task_type="regression"`. The notebook does this already; ad hoc scripts should too.
- Core files must not import stage code (`model.t1`, `model.t2`, `model.t3`, RL workflows, `datasets.rl_tasks`), even inside functions. `tests/test_nnknn_core.py` enforces this.
- After changing a core file, refresh the manifest with `python tools/check_nnknn_core.py --write` and record in the commit message that the stage branches need the same files.
- `model/Copy_nnknn_model.py`, `model/nnknn_model_1_4.py`, `model/nnknn_model_with_post_mlp.py`, `model/nn_cdh_old.py`, `model/train_GMM3.py`, `model/cls_model.py`, `model/reg_model.py`, `model/module.py`, and `older versions/` are legacy. They are outside the core and are kept on `main` only.
- Use synthetic datasets for quick validation unless a task specifically requires the larger real datasets in `datasets/`.

## Setup

- For Codex cloud tasks, use `bash codex/setup.sh` as the setup script.
- For cached Codex cloud environments, use `bash codex/maintenance.sh` as the maintenance script.
- The cloud setup assumes Python `3.11.9`, matching `python_version.txt`.
- On the Windows workstation, `main` has no `.venv`; run checks with a stage checkout's interpreter, for example `..\neural-cbr\.venv\Scripts\python.exe`.

## Validation

- Fast import check: `python codex/smoke_test.py --mode imports`
- Fast training smoke tests: `python codex/smoke_test.py --mode train` and `--mode classification`
- Core contract: `python -m pytest tests -q --basetemp=<new-temp-directory>`
- Cross-branch core check: `python tools/check_nnknn_core.py --branches main neural-cbr rl LLM`

## Notes

- `checkpoints/` and transient artifacts are expected and are gitignored.
- Completed prior-paper reruns are filed under `results/prior_baseline/`; see `results/README.md`.
- `matplotlib` should run headlessly in cloud tasks with `MPLBACKEND=Agg`.
- Avoid full multi-run benchmarks for routine verification; they are intentionally expensive.
