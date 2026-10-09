# Repository expectations

This branch is `neural-cbr` (Neural CBR). Use `tools/t1_run.py`, `tools/t1_reviewer_ui.py`, `configs/t1/`, and `tests/test_t1_*.py`. Other stages belong to their own branches.

- Keep shared neural components in `model/common/`, `model/nnknn_model.py`, `model/nn_cdh.py`, `model/device_utils.py`, and `model/feature_extractors.py`.
- Use `model/regression_workflow.py` for regression and set `task_type="regression"`.
- Use synthetic datasets for quick validation; run `.venv/Scripts/python.exe codex/smoke_test.py --mode imports` and the branch tests. Avoid full benchmarks for routine checks.
- `checkpoints/`, `*.pth`, and `*.pt` are local artifacts.
- Historical provenance hashes and commit IDs in experiment evidence describe the original experiment, not the reorganized source.
