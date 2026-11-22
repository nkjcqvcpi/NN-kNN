# NN-kNN: environment setup

One uv project (this directory) and one venv for all worktrees. Layout on every device:

```
NN-KNN/
  .venv/         shared venv, created from main/pyproject.toml
  main/  rl/  llm/  neural-cbr/
```

The venv must live at `../.venv` relative to `main/` (that is, `NN-KNN/.venv`). uv cannot take this path from `pyproject.toml`, so set it once per device with an absolute path:

- Windows (PowerShell): `[Environment]::SetEnvironmentVariable('UV_PROJECT_ENVIRONMENT', 'C:\path\to\NN-KNN\.venv', 'User')`, then open a new shell.
- Linux/macOS: `export UV_PROJECT_ENVIRONMENT=/path/to/NN-KNN/.venv` (put it in the shell profile).

Create or update the venv (Python 3.14, Intel Arc XPU build of torch, Windows x86_64 only):

```
cd NN-KNN/main
uv sync
uv run python -c "import torch; print(torch.__version__, torch.xpu.is_available())"
```

Run from any worktree with the shared venv:

```
cd NN-KNN/neural-cbr
uv run --project ../main --no-sync python -m pytest tests -q
```

## Datasets

Data files and preprocessed caches are not in git. They live in one directory shared by all worktrees, `NN-KNN/datasets/` (next to `main/`, `rl/`, ...; see its `README.md`). To use another location set `NNKNN_DATASETS` to its absolute path. The `datasets/` folder inside each worktree holds only the loader code.

Only `main` has `pyproject.toml`, `uv.lock` and `.python-version`. Do not create another venv, and do not use pip or requirements files.
