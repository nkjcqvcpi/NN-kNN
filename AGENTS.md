# Repository expectations

This branch is `neural-cbr` (`neural-cbr/`): the shared NN-kNN core plus full-cycle neural CBR (T1). Use `tools/t1_run.py`, `tools/t1_reviewer_ui.py`, `configs/t1/`, and `tests/test_t1_*.py`. Other stages belong to their own branches; standalone NN-kNN material belongs to main.

- The shared NN-kNN core (files listed in `tools/check_nnknn_core.py`) is owned by main. A core fix found here is made on main first and then synced (`docs/NNKNN_CORE.md`); `tests/test_nnknn_core.py` fails if the core drifts. T1-only code stays in `model/t1/`.
- File results and logs by T1 handoff phase under `results/t1/<phase>/` and `logs/t1/<phase>/` (see `results/README.md`).
- Use `model/regression_workflow.py` for regression and set `task_type="regression"`.
- Use synthetic datasets for quick validation; run `.venv/Scripts/python.exe codex/smoke_test.py --mode imports` and the branch tests. Avoid full benchmarks for routine checks.
- `checkpoints/`, `*.pth`, and `*.pt` are local artifacts.
- Historical provenance hashes and commit IDs in experiment evidence describe the original experiment, not the reorganized source.
