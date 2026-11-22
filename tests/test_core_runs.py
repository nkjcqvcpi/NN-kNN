"""Run tracking at the shared-core call sites (train_retrieval, train_model; CPU).

A real run must not change what is computed, and an interrupted and resumed training must equal the
uninterrupted one bit for bit (result_digest and the state digests of the named states).
"""
from __future__ import annotations

import os

os.environ.setdefault("NNKNN_DEVICE", "cpu")

import pytest
import torch

from model.common import tracking as tr
from model.common.core import CoreConfig, build_model, train_retrieval
from model.nnknn_model import default_args, train_model


class Crash(Exception):
    pass


@pytest.fixture
def root(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setenv("NNKNN_RUNS", str(runs))
    for name in ("NNKNN_RESUME", "NNKNN_RESUME_EVERY_SECONDS", "NNKNN_RESUME_EVERY_STEPS"):
        monkeypatch.delenv(name, raising=False)
    return runs


@pytest.fixture(autouse=True)
def keep_global_torch_flags():
    """The regression baselines call seed_everything(deterministic=True); later test files must not inherit it."""
    flags = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    yield
    torch.use_deterministic_algorithms(flags[0], warn_only=flags[1])


@pytest.fixture
def crash_at(monkeypatch):
    """crash_at(step): the next log_step call at that step raises Crash (once), before it writes anything."""
    original, armed = tr.Run.log_step, {}

    def log_step(self, step, values):
        if armed.pop(step, False):
            raise Crash(f"interrupted at step {step}")
        return original(self, step, values)

    monkeypatch.setattr(tr.Run, "log_step", log_step)
    return lambda step: armed.__setitem__(step, True)


def key(name):
    return tr.RunKey("core", "s2", (name,))


def open_unit(name, every=3):
    return tr.open_run(key(name), config={"case": name.split("-")[0]}, checkpoint_every_steps=every,
                       checkpoint_every_seconds=0)


def sd_digest(model):
    return tr.state_digest(model.state_dict())


# ---------------------------------------------------------------- train_retrieval

def _retrieval(task, run=None, **kw):
    torch.manual_seed(0)
    X = torch.randn(40, 4)
    y = torch.randint(0, 3, (40,)) if task == "classification" else X @ torch.tensor([1.0, -1.0, 0.5, 0.0])
    Xv = torch.randn(16, 4)
    yv = torch.randint(0, 3, (16,)) if task == "classification" else Xv @ torch.tensor([1.0, -1.0, 0.5, 0.0])
    cfg = CoreConfig(task_type=task, representation="mlp", mlp_dims=(8,), mlp_dropout=0.2, top_k=4, epochs=8,
                     patience=100, batch_size=8, seed=0)
    model = build_model(X, y, cfg, 3 if task == "classification" else None)
    extra = {} if run is None else {"run": run, "resumable": True}
    res = train_retrieval(model, X, y, Xv, yv, cfg, **extra, **kw)
    return res


@pytest.mark.parametrize("task,kw", [("classification", {}), ("regression", {"include_initial": True})])
def test_train_retrieval_with_a_run_computes_the_same(root, task, kw):
    plain = _retrieval(task, **kw)
    with open_unit(f"{task}-run") as run:
        tracked = _retrieval(task, run=run, **kw)
        run.save_state("final", tracked.model.state_dict())
    assert sd_digest(tracked.model) == sd_digest(plain.model)
    assert tracked.history == plain.history and tracked.best_epoch == plain.best_epoch
    assert tr.state_digest(tracked.optimizer.state_dict()) == tr.state_digest(plain.optimizer.state_dict())
    data = tr.load_run(root / "core" / "s2" / f"{task}-run")
    assert [v for _, v in data.series("train/loss")] == [h["train_loss"] for h in plain.history]
    assert [v for _, v in data.series("val/metric")] == [h["val_metric"] for h in plain.history]


@pytest.mark.parametrize("task,kw", [("classification", {}), ("regression", {"include_initial": True})])
def test_train_retrieval_resumed_equals_uninterrupted(root, crash_at, task, kw):
    with open_unit(f"{task}-ref") as run:
        ref = _retrieval(task, run=run, **kw)
        run.save_state("final", ref.model.state_dict())
        run.log_final({"best_epoch": ref.best_epoch, "best_val": ref.best_val})
    crash_at(4)  # checkpoints every 3 epochs (steps 2, 5): the resume from step 2 cuts what step 3 logged
    with pytest.raises(Crash), open_unit(f"{task}-cut") as run:
        _retrieval(task, run=run, **kw)
    assert tr.run_status(root / "core" / "s2" / f"{task}-cut").status == "failed"
    with open_unit(f"{task}-cut") as run:
        assert run.resumed_from is not None and run.resumed_from["step"] == 2
        res = _retrieval(task, run=run, **kw)
        run.save_state("final", res.model.state_dict())
        run.log_final({"best_epoch": res.best_epoch, "best_val": res.best_val})
    a, b = (tr.load_run(root / "core" / "s2" / f"{task}-{n}") for n in ("ref", "cut"))
    assert a.info.result_digest == b.info.result_digest
    assert a.states()["final"]["state_sha256"] == b.states()["final"]["state_sha256"] == sd_digest(_retrieval(task, **kw).model)
    assert b.info.sessions == 2 and b.info.resumed


def test_train_retrieval_resumable_rejects_a_checkpoint_hook():
    with pytest.raises(ValueError, match="checkpoint_hook"):
        _retrieval("classification", run=tr.NullRun(), checkpoint_hook=lambda *a: None, checkpoint_epochs={2})


# ---------------------------------------------------------------- train_model

def _train_model_cfg(task):
    cfg = dict(default_args)
    cfg.update({"training_epochs": 6, "patience": 100, "batch_size": 8, "top_k": 4, "bias_manual_set": True,
                "bias_manual_value": 1.0, "case_score_mode": "bias_minus_distance", "normalize_over_cases": True,
                "post_mlp_enabled": True, "post_mlp_dims": (8,), "post_mlp_dropout": 0.2,
                "freeze_feature_extractor_epochs": 3})
    if task == "classification":
        cfg.update({"task_type": "classification", "classification_loss": "nll_class_mass"})
    else:
        cfg.update({"task_type": "regression", "use_nn_cdh": True, "nn_cdh_pretrain": False})
    return cfg


def _train(task, run=None, cfg_updates=None):
    torch.manual_seed(0)
    X, Xv = torch.randn(48, 5), torch.randn(16, 5)
    if task == "classification":
        y, yv = (X[:, 0] > 0).long() + (X[:, 1] > 0.5).long(), (Xv[:, 0] > 0).long() + (Xv[:, 1] > 0.5).long()
    else:
        w = torch.tensor([1.0, -2.0, 0.5, 0.0, 1.5])
        y, yv = X @ w, Xv @ w
    cfg = _train_model_cfg(task)
    cfg.update(cfg_updates or {})
    extra = {} if run is None else {"run": run, "resumable": True}
    metric, _, model = train_model(X, y, Xv, yv, None, cfg, **extra)
    return metric, model


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_train_model_with_a_run_computes_the_same_and_keeps_named_states(root, task, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # nothing may be written outside the run
    before = set(tmp_path.iterdir())
    m0, plain = _train(task)
    with open_unit(f"{task}-tm") as run:
        m1, tracked = _train(task, run=run)
    assert set(tmp_path.iterdir()) == before
    assert m0 == m1 and sd_digest(plain) == sd_digest(tracked)
    data = tr.load_run(root / "core" / "s2" / f"{task}-tm")
    states = data.states()
    expected = {"best_retrieval", "final"} | ({"best_adapter"} if task == "regression" else set())
    assert set(states) == expected
    assert states["final"]["state_sha256"] == sd_digest(plain)
    tag = "val/acc" if task == "classification" else "val/r2"
    stages = [v for _, v in data.series("train/stage")]
    assert stages == [1] * 6 + ([2] * 6 if task == "regression" else [])
    assert [s for s, _ in data.series(tag)] == list(range(len(stages)))
    final = tr.load_state(data.path, "final")
    assert all(torch.equal(final[k], v) for k, v in plain.state_dict().items())


@pytest.mark.parametrize("task,crash", [
    ("regression", 2),   # stage 1, before the extractor is unfrozen (resume from step 1)
    ("regression", 4),   # stage 1, after the unfreeze at epoch 3 (resume from step 3)
    ("regression", 6),   # first stage-2 epoch (resume from the stage-1/stage-2 checkpoint at step 5)
    ("regression", 10),  # stage 2 (resume from step 9)
    ("classification", 4),
])
def test_train_model_resumed_equals_uninterrupted(root, crash_at, task, crash):
    with open_unit(f"{task}-tmref", every=2) as run:
        m_ref, ref = _train(task, run=run)
    crash_at(crash)
    with pytest.raises(Crash), open_unit(f"{task}-tmcut{crash}", every=2) as run:
        _train(task, run=run)
    with open_unit(f"{task}-tmcut{crash}", every=2) as run:
        assert run.resumed_from is not None and run.resumed_from["step"] == crash - 1
        m_cut, cut = _train(task, run=run)
    a, b = (tr.load_run(root / "core" / "s2" / n) for n in (f"{task}-tmref", f"{task}-tmcut{crash}"))
    assert m_ref == m_cut and sd_digest(ref) == sd_digest(cut)
    assert a.info.result_digest == b.info.result_digest
    assert {n: s["state_sha256"] for n, s in a.states().items()} == {n: s["state_sha256"] for n, s in b.states().items()}
    assert dict(cut.named_parameters()).keys() == dict(ref.named_parameters()).keys()
    assert {n: p.requires_grad for n, p in cut.named_parameters()} == {n: p.requires_grad for n, p in ref.named_parameters()}


def test_train_model_early_stop_resumes_too(root, crash_at):
    upd = {"patience": 0, "training_epochs": 8}
    with open_unit("regression-es", every=1) as run:
        m_ref, ref = _train("regression", run=run, cfg_updates=upd)
    plain_metric, plain = _train("regression", cfg_updates=upd)
    assert m_ref == plain_metric and sd_digest(ref) == sd_digest(plain)
    n = len(tr.load_run(root / "core" / "s2" / "regression-es").series("train/stage"))
    crash_at(n - 1)
    with pytest.raises(Crash), open_unit("regression-es2", every=1) as run:
        _train("regression", run=run, cfg_updates=upd)
    with open_unit("regression-es2", every=1) as run:
        m_cut, cut = _train("regression", run=run, cfg_updates=upd)
    a, b = (tr.load_run(root / "core" / "s2" / u) for u in ("regression-es", "regression-es2"))
    assert a.info.result_digest == b.info.result_digest and sd_digest(cut) == sd_digest(ref)
