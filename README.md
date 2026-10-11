# Neural CBR

This is the `neural-cbr` branch (`neural-cbr/`): the shared NN-kNN core plus full-cycle neural CBR (T1). The standalone model lives on `main` (`nnknn/`); RL (T2) on `rl`; LLM retrieval (T3) on `LLM`.

Stage entry points: `tools/t1_run.py`, `tools/t1_reviewer_ui.py`, `configs/t1/`, and `tests/test_t1_*.py`.

The shared NN-kNN core (`model/nnknn_model.py`, `model/common/`, and the files listed in `tools/check_nnknn_core.py`) is copied unchanged from `main`; read [docs/NNKNN_CORE.md](docs/NNKNN_CORE.md) before changing it. See [branch organization](docs/BRANCH_ORGANIZATION.md) for ownership and Git recovery details.

Results and logs are filed by T1 handoff phase (`T1_FULL_CYCLE_IMPLEMENTATION_PLAN.md` §13; copy in `docs/t1_spec/`) under `results/t1/<phase>/` and `logs/t1/<phase>/`. See [results/README.md](results/README.md).

Use the existing Python environment or the checked-in `pyproject.toml`/`uv.lock` for the Windows XPU environment. `requirements.txt` and `codex/setup.sh` describe the separate CPU/CUDA cloud environment; their versions differ.

```powershell
.\.venv\Scripts\python.exe codex/smoke_test.py --mode imports
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe tools/check_nnknn_core.py
```

Checkpoints (`*.pth`, `*.pt`) remain local and are ignored. Expensive benchmarks are opt-in; use synthetic smoke checks first.
