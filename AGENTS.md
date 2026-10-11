# Repository expectations

This branch is `LLM` (`llm/`): the shared NN-kNN core plus LLM retrieval and host integration (T3). Use `tools/t3_*.py`, `model/t3/`, and `tests/test_t3_*.py`. Other stages belong to their own branches; standalone NN-kNN material belongs to main.

- The shared NN-kNN core (files listed in `tools/check_nnknn_core.py`) is owned by main. Do not edit it here; change it on main and sync it (`docs/NNKNN_CORE.md`). `tests/test_nnknn_core.py` fails if it drifts.
- File results and logs by T3 handoff stage under `results/t3/<stage>/` and `logs/t3/<stage>/` (see `results/README.md`). The `tools/t3_*.py` scripts require an explicit `--output`/`--out`; point it there.
- Use `model/regression_workflow.py` for regression and set `task_type="regression"`.
- Use synthetic datasets for quick validation; run `.venv/Scripts/python.exe codex/smoke_test.py --mode imports` and the branch tests. Avoid full benchmarks for routine checks.
- `checkpoints/`, `*.pth`, and `*.pt` are local artifacts.
- Historical provenance hashes and commit IDs in experiment evidence describe the original experiment, not the reorganized source.
