# Repository expectations

This branch is `LLM` (`llm/`): the shared NN-kNN core plus LLM retrieval and host integration (T3). Use `tools/t3_*.py`, `model/t3/`, and `tests/test_t3_*.py`. Other stages belong to their own branches; standalone NN-kNN material belongs to main.

- The shared NN-kNN core (files listed in `tools/check_nnknn_core.py`) is owned by main. Do not edit it here; change it on main and sync it (`docs/NNKNN_CORE.md`). `tests/test_nnknn_core.py` fails if it drifts.
- File results and logs by T3 handoff stage under `results/t3/<stage>/` and `logs/t3/<stage>/` (see `results/README.md`). The `tools/t3_*.py` scripts require an explicit `--output`/`--out`; point it there.
- Use `model/regression_workflow.py` for regression and set `task_type="regression"`.
- Environment and dependencies are managed by uv only. The single `pyproject.toml`, `uv.lock` and `.python-version` (Python 3.14) live on `main`; one venv at `NN-KNN/.venv` is shared by all worktrees (`NN-KNN/{main,rl,llm,neural-cbr}`). Set the user environment variable `UV_PROJECT_ENVIRONMENT` to its absolute path once per device. In `main` run `uv sync` and `uv run python ...`; in another worktree run `uv run --project ../main --no-sync python ...`. torch is the Intel Arc XPU build (Windows x86_64 only; `environments` in the toml). No pip, no `requirements*.txt`, no hand-made venv. wandb was removed; experiment results are to be logged with TensorBoard (TODO, not built yet).
- This project has no README, CHANGELOG or LICENSE by owner decision (2026-11-21); obsolete tracked content is deleted, not archived (Git history keeps it).
- Use synthetic datasets for quick validation; run `uv run python codex/smoke_test.py --mode imports` and the branch tests (`uv run python -m pytest`). Avoid full benchmarks for routine checks.
- `checkpoints/`, `*.pth`, and `*.pt` are local artifacts.
- Historical provenance hashes and commit IDs in experiment evidence describe the original experiment, not the reorganized source.
