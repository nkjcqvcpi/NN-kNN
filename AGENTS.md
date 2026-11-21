# Repository expectations

This branch is `neural-cbr` (`neural-cbr/`): the shared NN-kNN core plus full-cycle neural CBR (T1). Use `tools/t1_run.py`, `tools/t1_reviewer_ui.py`, `configs/t1/`, and `tests/test_t1_*.py`. Other stages belong to their own branches (`main` is the core; `rl` and `llm` are the other stage branches); standalone NN-kNN material belongs to main. Cross-stage progress reports live in `reports/` (renamed from `handoff/` on 2026-11-21).

- The shared NN-kNN core (files listed in `tools/check_nnknn_core.py`) is owned by main. A core fix found here is made on main first and then synced (`docs/NNKNN_CORE.md`); `tests/test_nnknn_core.py` fails if the core drifts. T1-only code stays in `model/t1/`.
- File results and logs by T1 handoff phase under `results/t1/<phase>/` and `logs/t1/<phase>/` (see `results/README.md`).
- Use `model/regression_workflow.py` for regression and set `task_type="regression"`.
- Environment and dependencies are managed by uv only. The single `pyproject.toml`, `uv.lock` and `.python-version` (Python 3.14) live on `main`; one venv at `NN-KNN/.venv` is shared by all worktrees (`NN-KNN/{main,rl,llm,neural-cbr}`). Set the user environment variable `UV_PROJECT_ENVIRONMENT` to its absolute path once per device. In `main` run `uv sync` and `uv run python ...`; in another worktree run `uv run --project ../main --no-sync python ...`. torch is the Intel Arc XPU build (Windows x86_64 only; `environments` in the toml). No pip, no `requirements*.txt`, no hand-made venv. wandb was removed; experiment results are to be logged with TensorBoard (TODO, not built yet).
- This project keeps no README, CHANGELOG or LICENSE. Obsolete tracked content is deleted (Git history keeps it), not archived.
- Use synthetic datasets for quick validation; run `uv run python codex/smoke_test.py --mode imports` and the branch tests. Avoid full benchmarks for routine checks.
- `checkpoints/`, `*.pth`, and `*.pt` are local artifacts.
- Historical provenance hashes and commit IDs in experiment evidence describe the original experiment, not the reorganized source.
