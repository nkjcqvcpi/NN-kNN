"""Run tracking at the shared-core call sites (train_retrieval, train_model, workflows, benchmark tools; CPU).

A real run must not change what is computed, and an interrupted and resumed training must equal the
uninterrupted one bit for bit (result_digest and the state digests of the named states).
"""
from __future__ import annotations

import math
import os
import random

os.environ.setdefault("NNKNN_DEVICE", "cpu")

import numpy as np
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

def _retrieval(task, run=None, patience=100, epochs=8, bias_seed=None, **kw):
    torch.manual_seed(0)
    X = torch.randn(40, 4)
    y = torch.randint(0, 3, (40,)) if task == "classification" else X @ torch.tensor([1.0, -1.0, 0.5, 0.0])
    Xv = torch.randn(16, 4)
    yv = torch.randint(0, 3, (16,)) if task == "classification" else Xv @ torch.tensor([1.0, -1.0, 0.5, 0.0])
    cfg = CoreConfig(task_type=task, representation="mlp", mlp_dims=(8,), mlp_dropout=0.2, top_k=4, epochs=epochs,
                     patience=patience, batch_size=8, seed=0)
    model = build_model(X, y, cfg, 3 if task == "classification" else None)
    if bias_seed is not None:  # a setup that left case_default_bias different (the checkpoint must restore it)
        model.case_default_bias += bias_seed
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
    assert res.history == ref.history and res.epochs_run == ref.epochs_run and res.best_epoch == ref.best_epoch


def _first_bad_epoch(history):
    """First epoch without improvement (patience counter 1) after which training continues, or None."""
    best = math.inf
    for rec in history[:-1]:
        if rec["val_loss"] < best - 1e-9:
            best = rec["val_loss"]
        else:
            return rec["epoch"]
    return None


def test_train_retrieval_resume_keeps_the_patience_counter_and_the_case_bias(root, crash_at):
    """Crash right after a checkpoint taken while the patience counter is above 0; resume with another case bias."""
    kw = dict(patience=2, epochs=30)
    plain = _retrieval("regression", **kw)
    with open_unit("pat-ref", every=1) as run:
        ref = _retrieval("regression", run=run, **kw)
    assert ref.history == plain.history
    bad_ep = _first_bad_epoch(plain.history)
    assert bad_ep is not None
    crash_at(bad_ep + 1)  # the checkpoint of epoch bad_ep holds bad >= 1
    with pytest.raises(Crash), open_unit("pat-cut", every=1) as run:
        _retrieval("regression", run=run, **kw)
    with open_unit("pat-cut", every=1) as run:
        assert run.resumed_from["step"] == bad_ep
        res = _retrieval("regression", run=run, bias_seed=0.5, **kw)
    assert res.history == ref.history and res.epochs_run == ref.epochs_run
    assert sd_digest(res.model) == sd_digest(ref.model) and res.model.case_default_bias == ref.model.case_default_bias
    a, b = (tr.load_run(root / "core" / "s2" / n) for n in ("pat-ref", "pat-cut"))
    assert a.info.result_digest == b.info.result_digest


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


def _train(task, run=None, cfg_updates=None, setup_seed=None, nan_val=False, random_val=False):
    torch.manual_seed(0)
    X, Xv = torch.randn(48, 5), torch.randn(16, 5)
    if task == "classification":
        y, yv = (X[:, 0] > 0).long() + (X[:, 1] > 0.5).long(), (Xv[:, 0] > 0).long() + (Xv[:, 1] > 0.5).long()
    else:
        w = torch.tensor([1.0, -2.0, 0.5, 0.0, 1.5])
        y, yv = X @ w, Xv @ w
        if nan_val:
            yv[0] = math.nan  # every validation loss and R2 is NaN
        if random_val:
            yv = 3 * torch.randn(16)  # unrelated to X: the validation loss goes up and down
    if setup_seed is not None:  # the RNG state the model is built from (case_default_bias, initial weights)
        torch.manual_seed(setup_seed)
        np.random.seed(setup_seed)
        random.seed(setup_seed)
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


def test_train_model_resume_keeps_the_patience_counter_and_the_case_bias(root, crash_at):
    """Patience 2: crash after a checkpoint taken while patience_counter > 0. The model's case_default_bias
    is drawn from the RNG (bias_manual_set=False) and the resumed session builds it from another RNG state."""
    upd = {"patience": 2, "training_epochs": 12, "bias_manual_set": False}
    with open_unit("regression-pat", every=1) as run:
        m_ref, ref = _train("regression", run=run, cfg_updates=upd, setup_seed=1, random_val=True)
    m_plain, plain = _train("regression", cfg_updates=upd, setup_seed=1, random_val=True)
    assert m_ref == m_plain and sd_digest(ref) == sd_digest(plain)
    _, other = _train("regression", cfg_updates=dict(upd, training_epochs=0), setup_seed=2, random_val=True)
    assert other.case_default_bias != ref.case_default_bias  # the setup really differs
    data = tr.load_run(root / "core" / "s2" / "regression-pat")
    points = [(step, loss, stage) for (step, loss), (_, stage) in zip(data.series("val/loss"), data.series("train/stage"))]
    best, crash = {}, None
    for (step, loss, stage), nxt in zip(points, points[1:]):  # first epoch without improvement in a stage that goes on
        if loss < best.get(stage, math.inf):
            best[stage] = loss
        elif nxt[2] == stage:
            crash = step + 1
            break
    assert crash is not None
    crash_at(crash)
    with pytest.raises(Crash), open_unit("regression-pat2", every=1) as run:
        _train("regression", run=run, cfg_updates=upd, setup_seed=1, random_val=True)
    with open_unit("regression-pat2", every=1) as run:
        assert run.resumed_from["step"] == crash - 1
        m_cut, cut = _train("regression", run=run, cfg_updates=upd, setup_seed=2, random_val=True)
    b = tr.load_run(root / "core" / "s2" / "regression-pat2")
    assert m_cut == m_ref and sd_digest(cut) == sd_digest(ref) and cut.case_default_bias == ref.case_default_bias
    assert data.info.result_digest == b.info.result_digest


def _early_stopped_post_mlp(task, run):
    """Trainable post-MLP whose best epoch lies before the last one (random validation targets, patience 2)."""
    torch.manual_seed(0)
    X, Xv = torch.randn(64, 5), torch.randn(24, 5)
    if task == "classification":
        y, yv = (X[:, 0] > 0).long() + (X[:, 1] > 0.5).long(), torch.randint(0, 3, (24,))
    else:
        y, yv = X @ torch.tensor([1.0, -2.0, 0.5, 0.0, 1.5]), 3 * torch.randn(24)
    cfg = _train_model_cfg(task)
    cfg.update({"training_epochs": 20, "patience": 2, "freeze_feature_extractor_epochs": 0,
                "feature_extractor_lr": 5e-3, "use_nn_cdh": False})
    metric, _, model = train_model(X, y, Xv, yv, None, cfg, run=run, resumable=True)
    return metric, model, Xv, yv, cfg["batch_size"]


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_train_model_returned_model_predicts_like_the_final_state(root, task):
    """Restoring the best state also restores the extractor the cases are evaluated with: the returned model,
    a model rebuilt from the saved "final" state and the best validation epoch all give the same predictions."""
    import copy

    with open_unit(f"{task}-final") as run:
        metric, model, Xv, yv, bs = _early_stopped_post_mlp(task, run)
    data = tr.load_run(root / "core" / "s2" / f"{task}-final")
    assert data.states()["best_retrieval"]["step"] < len(data.series("train/stage")) - 1  # an earlier state was restored
    rebuilt = copy.deepcopy(model)
    rebuilt._invalidate_case_cache()
    rebuilt.load_state_dict(tr.load_state(data.path, "final"))
    model.eval()
    rebuilt.eval()
    with torch.no_grad():
        outs = [[m(Xv[s:s + bs]) for s in range(0, len(Xv), bs)] for m in (model, rebuilt)]
    scores, preds = ([torch.cat([o[i] for o in out]) for out in outs] for i in (0, 1))
    assert torch.equal(scores[0], scores[1]) and torch.equal(preds[0], preds[1])
    if task == "classification":  # the recorded best validation accuracy is the returned model's
        assert int((preds[0] == yv).sum()) / len(yv) == metric
    else:
        r2 = 1 - torch.sum((yv - preds[0]) ** 2).item() / torch.sum((yv - yv.mean()) ** 2).item()
        assert r2 == metric


def test_train_model_early_stopped_post_mlp_resumes_exactly(root, crash_at):
    import copy

    with open_unit("classification-esref", every=1) as run:
        m_ref, ref, Xv, _, _ = _early_stopped_post_mlp("classification", run)
    n = len(tr.load_run(root / "core" / "s2" / "classification-esref").series("train/stage"))
    crash_at(n - 2)
    with pytest.raises(Crash), open_unit("classification-escut", every=1) as run:
        _early_stopped_post_mlp("classification", run)
    with open_unit("classification-escut", every=1) as run:
        assert run.resumed_from is not None
        m_cut, cut, _, _, _ = _early_stopped_post_mlp("classification", run)
    a, b = (tr.load_run(root / "core" / "s2" / u) for u in ("classification-esref", "classification-escut"))
    assert m_ref == m_cut and sd_digest(ref) == sd_digest(cut) and a.info.result_digest == b.info.result_digest
    fresh = copy.deepcopy(cut)
    fresh._invalidate_case_cache()
    with torch.no_grad():
        assert torch.equal(ref.eval()(Xv)[0], cut.eval()(Xv)[0]) and torch.equal(cut(Xv)[0], fresh.eval()(Xv)[0])


def test_train_model_with_nan_validation_completes_its_run(root):
    """A diverged unit (NaN val loss, so NaN best-state meta) must complete, not fail in save_state."""
    with open_unit("regression-nan") as run:
        m, model = _train("regression", run=run, nan_val=True)
    assert math.isnan(m)
    data = tr.load_run(root / "core" / "s2" / "regression-nan")
    assert data.info.status == "complete"
    assert math.isnan(data.states()["best_retrieval"]["meta"]["val_loss"])
    assert set(data.states()) == {"best_retrieval", "best_adapter", "final"}


# ---------------------------------------------------------------- workflows and tools

def test_classification_tool_records_units_skips_complete_ones_and_matches_the_workflow(root, capsys):
    from model.classification_workflow import make_classification_cfg, run_repeated_classification_model_benchmarks
    from tools.run_classification_benchmarks import record_summary_report, run_benchmarks
    from tools.table1_nnknn_kfold import read_frame

    cfg = make_classification_cfg({"training_epochs": 3, "batch_size": 16})
    kw = dict(methods=["nnknn", "knn", "mlp"], runs=2, seed=3)
    summary, runs_df, log = run_benchmarks(["iris"], cfg, study="cls", **kw)
    assert [u["source"] for u in log] == ["fresh"] * 6
    _, plain_runs, _ = run_repeated_classification_model_benchmarks(["iris"], cfg, methods=kw["methods"], num_runs=2, base_seed=3)
    assert runs_df.to_dict("records") == plain_runs.to_dict("records")
    unit = root / "core" / "cls" / "iris" / "nnknn" / "holdout-1"
    data = tr.load_run(unit)
    assert set(data.states()) == {"best_retrieval", "final"} and len(data.series("val/acc")) == 3
    assert data.array("predictions").shape == (30,)

    summary2, runs2, log2 = run_benchmarks(["iris"], cfg, study="cls", **kw)
    assert [u["source"] for u in log2] == ["complete"] * 6
    assert runs2.equals(runs_df) and summary2.equals(summary)
    report = record_summary_report(summary, runs_df, study="cls", manifest={"x": 1}, units=log2)
    rep = tr.load_run(report)
    assert rep.records("summary") == summary.to_dict("records")
    assert read_frame(rep, "runs").equals(runs_df) and list(read_frame(rep, "summary").columns) == list(summary.columns)
    assert [u["result_digest"] for u in rep.record("units")] == [tr.run_status(u["path"]).result_digest for u in log2]
    with pytest.raises(tr.ConfigMismatchError):
        run_benchmarks(["iris"], make_classification_cfg({"training_epochs": 4, "batch_size": 16}), study="cls", **kw)
    _, runs3, log3 = run_benchmarks(["iris"], cfg, study="cls", **dict(kw, runs=3))  # a holdout study grows
    assert [u["source"] for u in log3] == ["complete"] * 6 + ["fresh"] * 3
    assert runs3.iloc[:6].equals(runs_df)


def test_image_units_reuse_the_classifier_stored_by_a_skipped_unit(root, monkeypatch):
    from datasets.paths import datasets_root
    import model.classification_workflow as cw
    from model.classification_workflow import make_classification_cfg
    from tools.run_classification_benchmarks import run_benchmarks

    if not (datasets_root() / "MNIST" / "raw" / "train-images-idx3-ubyte").is_file():
        pytest.skip("MNIST is not in the datasets directory")
    cfg = make_classification_cfg({"training_epochs": 1, "batch_size": 16})
    common = dict(runs=1, seed=4, method_cfgs={"convnet": {"epochs": 2, "batch_size": 32}},
                  dataset_kwargs_map={"mnist": {"max_train_samples": 120, "max_eval_samples": 40, "download": False}})
    methods = ["nnknn_conv_frozen", "convnet", "knn_conv_frozen"]
    _, ref, _ = run_benchmarks(["mnist"], cfg, study="imgref", methods=methods, **common)
    owner = root / "core" / "imgref" / "mnist" / "nnknn_conv_frozen" / "official-1"
    assert set(tr.load_run(owner).states()) == {"best_retrieval", "final", "classifier"}

    run_benchmarks(["mnist"], cfg, study="imgcut", methods=methods[:1], **common)
    calls, train = [], cw._train_torch_classifier
    monkeypatch.setattr(cw, "_train_torch_classifier", lambda *a, **k: calls.append(1) or train(*a, **k))
    _, cut, log = run_benchmarks(["mnist"], cfg, study="imgcut", methods=methods, **common)
    assert [u["source"] for u in log] == ["complete", "fresh", "fresh"] and calls == []  # loaded, not retrained
    assert cut.to_dict("records") == ref.to_dict("records")
    for method in methods[1:]:
        a, b = (tr.load_run(root / "core" / s / "mnist" / method / "official-1") for s in ("imgref", "imgcut"))
        assert np.array_equal(a.array("predictions"), b.array("predictions"))


def test_table1_study_skips_complete_units_resumes_incomplete_ones_and_reports(root, monkeypatch):
    from tools.table1_nnknn_kfold import read_table1_report, record_table1_report, run_table1_kfold

    kw = dict(num_folds=2, base_seed=5, nnknn_base_cfg={"training_epochs": 3, "patience": 100},
              baseline_method_cfgs={"mlp": {"epochs": 2}, "mlkr_knn": {"max_iter": 3}})
    log: list = []
    summary, runs_df, art = run_table1_kfold(["diabets"], study="t1ref", unit_log=log, **kw)
    assert len(log) == 2 * 4 + 2 * 3 and {u["source"] for u in log} == {"fresh"}
    assert len(runs_df) == 2 * 4 * 2 + 2 * 3
    unit = root / "core" / "t1ref" / "diabets" / "softmax" / "f1"
    data = tr.load_run(unit)
    assert set(data.states()) == {"best_retrieval", "best_adapter", "final"}
    assert len(data.records("rows")) == 2 and "rmse_post_raw" in data.final()

    plain_summary, plain_runs, _ = run_table1_kfold(["diabets"], **kw)  # untracked: same columns, same order
    assert list(runs_df.columns) == list(plain_runs.columns) and list(summary.columns) == list(plain_summary.columns)
    baseline_rows = runs_df["source"] == "baseline"
    assert runs_df[baseline_rows].equals(plain_runs[baseline_rows])

    log2: list = []  # everything complete: nothing trains, the tables are read back unchanged
    summary2, runs2, art2 = run_table1_kfold(["diabets"], study="t1ref", unit_log=log2, **kw)
    assert {u["source"] for u in log2} == {"complete"}
    assert runs2.equals(runs_df) and summary2.equals(summary)
    fresh, skipped = art["nnknn"]["diabets"]["softmax"][0], art2["nnknn"]["diabets"]["softmax"][0]
    assert "model" not in skipped  # not rebuilt unless asked
    for k in ("rmse_pre_raw", "rmse_post_raw", "rmse_pre_model_space", "rmse_post_model_space", "best_metric"):
        assert skipped[k] == fresh[k]
    assert all(torch.equal(skipped[k], fresh[k]) for k in ("y_pred_pre_z", "y_pred_post_z"))
    _, _, art3 = run_table1_kfold(["diabets"], study="t1ref", run_baselines=False, rebuild_skipped=True, **kw)
    assert torch.equal(art3["nnknn"]["diabets"]["softmax"][0]["model"].biases, fresh["model"].biases.cpu())

    monkeypatch.setenv("NNKNN_RESUME_EVERY_STEPS", "1")  # crash the third NN-kNN unit in stage 2
    calls = {"n": 0}
    plain_log_step = tr.Run.log_step

    def crash_third(self, step, values):
        calls["n"] += step == 0
        if calls["n"] == 3 and step == 4 and not calls.get("done"):
            calls["done"] = True
            raise Crash("interrupted")
        return plain_log_step(self, step, values)

    monkeypatch.setattr(tr.Run, "log_step", crash_third)
    with pytest.raises(Crash):
        run_table1_kfold(["diabets"], study="t1cut", run_baselines=False, **kw)
    log3: list = []
    summary3, runs3, _ = run_table1_kfold(["diabets"], study="t1cut", run_baselines=False, unit_log=log3, **kw)
    assert [u["source"] for u in log3] == ["complete", "complete", "resumed"] + ["fresh"] * 5
    nn = runs_df[runs_df["source"] == "nnknn"].reset_index(drop=True)
    assert runs3.to_dict("records") == nn[list(runs3.columns)].to_dict("records")
    for u_ref, u_cut in zip(log[:8], log3):
        assert tr.run_status(u_ref["path"]).result_digest == tr.run_status(u_cut["path"]).result_digest

    report = record_table1_report(summary, runs_df, study="t1ref", metadata={"note": "test"}, units=log)
    tables = read_table1_report(report)
    assert tables["summary_long"]["rmse_raw_table"].tolist() == summary["rmse_raw_table"].tolist()
    assert list(tables["runs_long"].columns) == list(runs_df.columns) and tables["summary_long"].equals(summary)
    assert len(tables["transposed"]) == len(set(summary["entry_label"])) and tables["transposed"].columns[0] == "Model"
