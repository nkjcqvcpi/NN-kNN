"""Never retrained (owner's rule O4): a partly rerun T1 study trains nothing complete and equals a straight run (CPU).

For every experiment type of ``tools/t1_run.py``: the same dataset and seed are run once straight through (study
``a``) and once crashed at a chosen run and then rerun (study ``b``). The rerun must not train the shared core,
the shared adapters or stages, or any complete run, and every record, curve and named state of every run (``rng``
and the ID counter positions ``ids`` included) must equal the straight run. Each attempt starts like a fresh process
(ID counters at zero). A rerun of a finished study must train and load nothing.
"""
import contextlib
import io
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from model.common import tracking as tr
from model.t1 import records
from model.t1.data import T1Data
import tools.t1_run as t1_run
from tools.t1_run import Runner, load_config

from t1_units import stage_units, units

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "t1"
IRIS = {"name": "iris", "task_type": "classification", "max_train": 24}


def shrink(cfg, **core):
    cfg["core"].update(epochs=3, patience=3, batch_size=8, top_k=3, bias_init_k=3, **core)
    for part in ("reuse", "sync"):
        if part in cfg:
            cfg[part].update(epochs=3, patience=3)
    if "reuse" in cfg:
        cfg["reuse"].update(batch_size=8, hidden_dims=[8, 4])
    return cfg


def synthetic_revision_data(ds, seed):
    """A small classification task with known corrupted labels (``case_truth``/``true_labels``) for ``revise``."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(40, 3, generator=g)
    truth = (X[:, 0] > 0).long()
    y = truth.clone()
    corrupted = np.zeros(28, dtype=bool)
    corrupted[[1, 5, 9, 13]] = True
    y[:28][torch.as_tensor(corrupted)] = 1 - y[:28][torch.as_tensor(corrupted)]
    return T1Data(ds["name"], "classification", X[:28], y[:28], X[28:34], truth[28:34], X[34:], truth[34:],
                  num_classes=2, case_truth={"corrupted": corrupted}, meta={"true_labels": truth[:28].numpy()})


def config(kind):
    """``(cfg, dataset entry, data_for or None)`` of a small study of experiment type ``kind``."""
    data_for = None
    if kind == "retention":  # an MLP encoder with dropout: the finetunes consume the global RNG
        cfg = shrink(load_config(CONFIGS / "alignment_candidates_small.yaml"), representation="mlp", mlp_dims=[8],
                     mlp_dropout=0.2)
        cfg["conditions"] = [{"policy": "full_memory"}, {"policy": "random"}, {"policy": "provenance_only"}]
        cfg["protocol"]["finetune_epochs"] = 2
        ds = IRIS
    elif kind == "reuse":
        cfg = shrink(load_config(CONFIGS / "p2_reuse.yaml"))
        cfg["conditions"] = cfg["conditions"][:3]
        ds = IRIS
    elif kind == "sync":
        cfg = shrink(load_config(CONFIGS / "alignment_sync.yaml"))
        cfg["sync"].update(block_epochs=1, maintenance_epochs=[2])
        ds = IRIS
    elif kind == "encoder_ablation":
        cfg = shrink(load_config(CONFIGS / "ema_ablation_20261010.yaml"), mlp_dims=[8], mlp_dropout=0.2)
        cfg["arms"] = cfg["arms"][1:3]  # shared encoder, EMA case encoder
        cfg["sync"].update(block_epochs=1)
        ds = IRIS
    elif kind == "removal_retraining":
        cfg = shrink(load_config(CONFIGS / "alignment_retraining.yaml"))
        cfg["treatments"] = ["full_frozen", "random_matched", "removal_frozen", "removal_retrained"]
        cfg["retraining"]["epochs_per_candidate"] = 1
        ds = IRIS
    elif kind == "adapted_retrained_removal":
        cfg = shrink(load_config(CONFIGS / "alignment_adapted_retraining.yaml"))
        cfg["conditions"] = cfg["conditions"][:3]
        ds = {**IRIS, "max_train": 18}
    elif kind == "reuse_retention":
        cfg = shrink(load_config(CONFIGS / "alignment_adapted_candidates.yaml"))
        cfg["conditions"] = [{"policy": "full_memory"}, {"policy": "provenance_only"}, {"policy": "removal_cached"}]
        ds = IRIS
    elif kind == "mcb":
        cfg = shrink(load_config(CONFIGS / "smoke_mcb.yaml"), mlp_dims=[8])
        cfg["core"]["epochs"] = 4
        cfg["maintenance_epochs"] = [2]
        cfg["conditions"].append({"mcb": False, "policy": "random", "K": 0.5})
        ds = IRIS
    elif kind == "revise":
        cfg = shrink(load_config(CONFIGS / "smoke_revise.yaml"))
        cfg["revise"].update(budgets=[2, 6], methods=["provenance_only", "random"], retrain_epochs=2,
                             matched_training_control=True)
        ds, data_for = {"name": "tiny_revise", "task_type": "classification"}, synthetic_revision_data
    else:
        raise KeyError(kind)
    cfg["datasets"], cfg["seeds"] = [ds], [3]
    return cfg, ds, data_for


class Counter:
    """Counts the trainings a rerun must skip: cores, adapters, stages and the runs themselves."""

    def __init__(self, monkeypatch):
        self.calls = {}
        for owner, name in ((Runner, "trained_core"), (Runner, "observed_core"), (Runner, "_finish"),
                            (t1_run, "train_classification_adapter"), (t1_run, "flagging_ablation")):
            self.wrap(monkeypatch, owner, name)
        import model.t1.retraining as retraining
        self.wrap(monkeypatch, retraining, "run_retrained_removal")
        stage_unit = records.stage_unit

        def counted_stage(*args, **kwargs):  # a stage computed (trained) rather than loaded
            objects, loaded = stage_unit(*args, **kwargs)
            name = "stage_loaded" if loaded else "stage_trained"
            self.calls[name] = self.calls.get(name, 0) + 1
            return objects, loaded
        monkeypatch.setattr(records, "stage_unit", counted_stage)

    def wrap(self, monkeypatch, owner, name):
        original = getattr(owner, name)

        def counted(*args, **kwargs):
            self.calls[name] = self.calls.get(name, 0) + 1
            return original(*args, **kwargs)
        monkeypatch.setattr(owner, name, counted)

    def trained(self):
        return {k: v for k, v in self.calls.items() if v}


def runner(kind, study, base):
    cfg, ds, data_for = config(kind)
    r = Runner(cfg, base / f"inputs-{study}", study)
    if data_for is not None:
        r.data_for = data_for
    return r, ds


def run(r, ds, kind):
    """One process's worth of work: the process-wide ID counters start at zero, as in a fresh process."""
    records.restore_ids({f"{module}.{attr}": 0 for module, attr in records.ID_COUNTERS})
    with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()):
        r.check_recorded(ds, 3)
        getattr(r, f"run_{kind}")(ds, 3)
    return out.getvalue()


def same(a, b, path=""):
    """Equal values (NaN equal to NaN), ignoring wall times and the per-process source snapshot path."""
    if isinstance(a, dict) and isinstance(b, dict):
        keys = {k for k in a.keys() | b.keys() if "seconds" not in str(k)
                and k not in ("timestamp", "timestamp_utc", "source_snapshot", "created_at_utc")}
        return all(k in a and k in b and same(a[k], b[k], f"{path}.{k}") for k in keys)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y, f"{path}[{i}]") for i, (x, y) in enumerate(zip(a, b)))
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def first_difference(a, b, path=""):
    """Where ``same`` first fails (for the assertion message)."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(a.keys() | b.keys(), key=str):
            if "seconds" in str(k) or k in ("timestamp", "timestamp_utc", "source_snapshot", "created_at_utc"):
                continue
            if k not in a or k not in b:
                return f"{path}.{k} missing"
            if not same(a[k], b[k]):
                return first_difference(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{path} lengths {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            if not same(x, y):
                return first_difference(x, y, f"{path}[{i}]")
    return f"{path}: {a!r} != {b!r}"[:300]


def assert_units_equal(left, right):
    a, b = units(left), units(right)
    assert [u.row["run_id"] for u in a] == [u.row["run_id"] for u in b] and a
    for ua, ub in zip(a, b):
        names = sorted(n for n in ua.data._single if n != "config")
        assert names == sorted(n for n in ub.data._single if n != "config"), ua.condition
        for name in names:
            if name.startswith("state/"):  # weights and RNG: the state digest (all tensors and values)
                assert ua.data._single[name]["state_sha256"] == ub.data._single[name]["state_sha256"], (ua.condition, name)
            else:
                x, y = records.record(ua.data, name), records.record(ub.data, name)
                assert same(x, y), (ua.condition, name, first_difference(x, y))
        assert sorted(ua.data._rows) == sorted(ub.data._rows)
        for name in ua.data._rows:
            x, y = records.rows(ua.data, name), records.rows(ub.data, name)
            assert same(x, y), (ua.condition, name, first_difference(x, y))
        for tag in ua.data._x:
            if "seconds" not in tag:
                assert same(ua.data.series(tag), ub.data.series(tag)), (ua.condition, tag)


KINDS = {  # experiment type -> the run (1-based finish call) at which the crashed attempt fails
    "retention": 2, "reuse": 2, "sync": 2, "encoder_ablation": 6, "removal_retraining": 3,
    "adapted_retrained_removal": 2, "reuse_retention": 2, "mcb": 2, "revise": 3,
}
STAGELESS = {"mcb"}  # every mcb condition trains its own core: nothing is shared


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_partly_rerun_study_trains_nothing_complete_and_equals_a_straight_run(kind, tmp_path, monkeypatch):
    (tmp_path / "runs").mkdir()
    monkeypatch.setenv("NNKNN_RUNS", str(tmp_path / "runs"))
    for name in ("NNKNN_RESUME", "NNKNN_RESUME_EVERY_SECONDS", "NNKNN_RESUME_EVERY_STEPS"):
        monkeypatch.delenv(name, raising=False)
    root = tmp_path / "runs" / "t1"

    straight, ds = runner(kind, "a", tmp_path)
    run(straight, ds, kind)
    total = len(units(root / "a"))
    assert total >= KINDS[kind] + 1

    crashed, ds = runner(kind, "b", tmp_path)
    original, calls = Runner._finish, []

    def crash(self, *args, **kwargs):
        calls.append(kwargs["cond_label"])
        if len(calls) == KINDS[kind]:
            raise RuntimeError("simulated crash")
        return original(self, *args, **kwargs)
    with monkeypatch.context() as m:
        m.setattr(Runner, "_finish", crash)
        with pytest.raises(RuntimeError, match="simulated crash"):
            run(crashed, ds, kind)
    done_before = len(units(root / "b"))
    assert done_before == KINDS[kind] - 1
    stages = stage_units(root / "b")
    assert bool(stages) == (kind not in STAGELESS), "shared training is recorded as a stage before any run uses it"

    with monkeypatch.context() as m:
        counter = Counter(m)
        rerun, ds = runner(kind, "b", tmp_path)
        printed = run(rerun, ds, kind)
    trained = counter.trained()
    # the core and every stage trained before the crash (initial adapters, retrained removal, revise ablation) are
    # loaded, never trained again; only the runs that were not complete train (e.g. their own reuse adapters)
    for name in ("trained_core", "observed_core", "flagging_ablation", "stage_trained"):
        assert name not in trained, (name, trained)
    # stages loaded: those with an incomplete run; the first EMA arm finished before the crash, so its core stays put
    loaded = 0 if kind in STAGELESS else 1 if kind == "encoder_ablation" else len(stages)
    assert trained.get("stage_loaded", 0) == loaded, (trained, stages)
    if kind == "removal_retraining":  # the crash came after K's retrained removal: its stage is loaded
        assert "run_retrained_removal" not in trained
    assert trained.get("_finish") == total - done_before  # only the runs that were not complete
    assert printed.count("complete; not trained again") == done_before
    assert ("loaded, not trained again" in printed) == (kind not in STAGELESS)
    assert_units_equal(root / "a", root / "b")

    with monkeypatch.context() as m:  # a finished study: nothing is trained or loaded
        counter = Counter(m)
        again, ds = runner(kind, "a", tmp_path)
        printed = run(again, ds, kind)
    assert counter.trained() == {}, counter.trained()
    assert printed.count("complete; not trained again") == total and "loaded" not in printed
    assert_units_equal(root / "a", root / "b")


def test_stage_objects_continue_exactly_what_was_trained(tmp_path, monkeypatch):
    """The stage's objects are the trained ones: weights, optimizer moments (bound to the model), history, modes."""
    (tmp_path / "runs").mkdir()
    monkeypatch.setenv("NNKNN_RUNS", str(tmp_path / "runs"))
    r, ds = runner("retention", "s", tmp_path)
    data = r.data_for(ds, 3)
    with contextlib.redirect_stdout(io.StringIO()):
        made = r.core_stage(ds, 3, data, [("never", {"x": 1})])
        loaded = r.core_stage(ds, 3, data, [("never", {"x": 1})])
    assert made is not loaded
    for key in ("model", "tr"):
        assert type(made[key]) is type(loaded[key])
    assert tr.state_digest(made["model"].state_dict()) == tr.state_digest(loaded["model"].state_dict())
    assert tr.state_digest(made["tr"].optimizer.state_dict()) == tr.state_digest(loaded["tr"].optimizer.state_dict())
    params = {id(p) for p in loaded["model"].parameters()}
    assert all(id(p) in params for g in loaded["tr"].optimizer.param_groups for p in g["params"])
    assert loaded["tr"].model is loaded["model"] and loaded["tr"].history == made["tr"].history
    assert [m.training for m in loaded["model"].modules()] == [m.training for m in made["model"].modules()]
    assert loaded["model"].t1_bias_init == made["model"].t1_bias_init
    stage = stage_units(tmp_path / "runs")[0]
    data_ = tr.load_run(stage)
    assert {"objects", "rng", "model"} <= set(data_.states())
    assert data_.series("core/train_loss") and records.record(data_, "history")["core"]


def test_another_config_is_refused_before_any_training(tmp_path, monkeypatch):
    (tmp_path / "runs").mkdir()
    monkeypatch.setenv("NNKNN_RUNS", str(tmp_path / "runs"))
    r, ds = runner("reuse", "c", tmp_path)
    run(r, ds, "reuse")
    changed, ds = runner("reuse", "c", tmp_path)
    changed.cfg["status"] = "changed"
    with monkeypatch.context() as m:
        counter = Counter(m)
        with pytest.raises(tr.ConfigMismatchError):
            run(changed, ds, "reuse")
    assert counter.trained() == {}


def test_runner_and_cli_require_a_study(tmp_path, monkeypatch):
    cfg, _, _ = config("reuse")
    with pytest.raises(TypeError):
        Runner(cfg, tmp_path, None)
    import sys
    monkeypatch.setattr(sys, "argv", ["t1_run.py", str(CONFIGS / "smoke_legacy.yaml")])
    with pytest.raises(SystemExit) as exc, contextlib.redirect_stderr(io.StringIO()):
        t1_run.main()
    assert exc.value.code == 2
