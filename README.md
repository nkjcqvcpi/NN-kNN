# Reinforcement learning

This is the `rl` branch of NN-kNN. The other stages live on `neural-cbr`, `rl`, and `LLM`.

Stage entry points: `tools/run_rl_*.py`, `tools/t2_role_pilot.py`, `tools/t2_retention_pilot.py`, and `tests/test_t2_*.py`.

The shared neural model lives in `model/nnknn_model.py`; shared construction, geometry, and artifact helpers live in `model/common/`. See [branch organization](docs/BRANCH_ORGANIZATION.md) for ownership and Git recovery details.

Use the existing Python environment or the checked-in `pyproject.toml`/`uv.lock` for the Windows XPU environment. `requirements.txt` and `codex/setup.sh` describe the separate CPU/CUDA cloud environment; their versions differ.

```powershell
.\.venv\Scripts\python.exe codex/smoke_test.py --mode imports
.\.venv\Scripts\python.exe -m pytest tests -q
```

Checkpoints (`*.pth`, `*.pt`) remain local and are ignored. Expensive benchmarks are opt-in; use synthetic smoke checks first.
