# Codex Cloud Environment

Use this repository with a Codex web cloud environment configured as follows:

- Base image: `universal`
- Python version: `3.11.9`
- Setup script: `bash codex/setup.sh`
- Maintenance script: `bash codex/maintenance.sh`

Recommended environment variables:

- `MPLBACKEND=Agg`
- `PYTHONUNBUFFERED=1`
- `NNKNN_DEVICE=cpu`

Quick validation commands inside a task:

- `.venv/bin/python codex/smoke_test.py --mode imports`
- `.venv/bin/python codex/smoke_test.py --mode train`
- `.venv/bin/python codex/smoke_test.py --mode classification`
- `.venv/bin/python -m pytest tests -q --basetemp=<new-temp-directory>`
- `.venv/bin/python tools/check_nnknn_core.py`

This branch (`main`) holds the standalone NN-kNN model. RL, neural CBR, and
LLM smoke modes live on the `rl`, `neural-cbr`, and `LLM` branches.
