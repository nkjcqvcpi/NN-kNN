"""Shared fixtures of the T2 tests: recording is mandatory, so every training in a test is a run unit."""
from __future__ import annotations

import os

os.environ.setdefault("NNKNN_DEVICE", "cpu")

import pytest


@pytest.fixture
def unit_train(tmp_path, monkeypatch):
    """``unit_train(name, train, task, cfg, **kw)``: ``train(task, cfg, run=<run>, **kw)`` in the unit
    ``runs/t2/test/<name>`` of a fresh run root (completed when it returns); returns the training's state, whose
    ``run_dir`` is the unit. Read its per-run files with ``model.t2.runs.RunFiles(state["run_dir"])``."""
    from model.common import tracking as tr

    runs = tmp_path / "runs"
    runs.mkdir(exist_ok=True)
    monkeypatch.setenv("NNKNN_RUNS", str(runs))

    def train_unit(name, train, task, cfg, **kwargs):
        with tr.open_run(tr.RunKey("t2", "test", (name,)), config={"test": name}) as run:
            return train(task, cfg, run=run, **kwargs)

    return train_unit
