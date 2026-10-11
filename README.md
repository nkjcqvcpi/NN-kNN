# LLM retrieval and host integration

This is the `LLM` branch (`llm/`): the shared NN-kNN core plus LLM/agent retrieval (T3). The standalone model lives on `main` (`nnknn/`); neural CBR (T1) on `neural-cbr`; RL (T2) on `rl`.

Stage entry points: `tools/t3_*.py`, `model/t3/`, and `tests/test_t3_*.py`.

The shared NN-kNN core (`model/nnknn_model.py`, `model/common/`, and the files listed in `tools/check_nnknn_core.py`) is copied unchanged from `main`; read [docs/NNKNN_CORE.md](docs/NNKNN_CORE.md) before changing it. See [branch organization](docs/BRANCH_ORGANIZATION.md) for ownership and Git recovery details.

Results and logs are filed by T3 handoff stage (`T3_LLM_AGENT_IMPLEMENTATION_PLAN.md`; copy in `docs/t1_spec/`) under `results/t3/<stage>/` and `logs/t3/<stage>/`. See [results/README.md](results/README.md).

Use the existing Python environment or the checked-in `pyproject.toml`/`uv.lock` for the Windows XPU environment. `requirements.txt` and `codex/setup.sh` describe the separate CPU/CUDA cloud environment; their versions differ.

```powershell
.\.venv\Scripts\python.exe codex/smoke_test.py --mode imports
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe tools/check_nnknn_core.py
```

Checkpoints (`*.pth`, `*.pt`) remain local and are ignored. Expensive benchmarks are opt-in; use synthetic smoke checks first.
