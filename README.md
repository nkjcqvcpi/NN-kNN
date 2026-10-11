# Reinforcement learning

This is the `rl` branch (`rl/`): the shared NN-kNN core plus reinforcement learning (T2), including NN-kNN-RL. The standalone model lives on `main` (`nnknn/`); neural CBR (T1) on `neural-cbr`; LLM retrieval (T3) on `LLM`.

Stage entry points: `tools/run_rl_*.py`, `tools/t2_role_pilot.py`, `tools/t2_retention_pilot.py`, and `tests/test_t2_*.py`.

The shared NN-kNN core (`model/nnknn_model.py`, `model/common/`, and the files listed in `tools/check_nnknn_core.py`) is copied unchanged from `main`; read [docs/NNKNN_CORE.md](docs/NNKNN_CORE.md) before changing it. See [branch organization](docs/BRANCH_ORGANIZATION.md) for ownership and Git recovery details.

Results and logs are filed by T2 handoff step (`T2_RL_IMPLEMENTATION_PLAN.md` in `career-2027/03-research-architecture/implementation-handoff/`) under `results/t2/<step>/` and `logs/t2/<step>/`. See [results/README.md](results/README.md).

Use the existing Python environment or the checked-in `pyproject.toml`/`uv.lock` for the Windows XPU environment. `requirements.txt` and `codex/setup.sh` describe the separate CPU/CUDA cloud environment; their versions differ.

```powershell
.\.venv\Scripts\python.exe codex/smoke_test.py --mode imports
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe tools/check_nnknn_core.py
```

Checkpoints (`*.pth`, `*.pt`) remain local and are ignored. Expensive benchmarks are opt-in; use synthetic smoke checks first.
