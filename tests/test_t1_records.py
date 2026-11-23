"""T1 run units (--study): what the T1 tools record, and the readers on units versus the legacy files (CPU).

The same runs written as files and recorded as units must hold the same numbers, rows, histories and weights,
and every converted summary must give byte-identical outputs from either input.
"""
import contextlib
import io
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from model.common import tracking as tr
from model.t1 import records
from model.t1.data import T1Data
from tools.t1_run import Runner, load_config

EXP = "records_tiny"
STUDY = "rec"


def tiny_data(ds, seed):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(16, 3, generator=g)
    if ds["task_type"] == "classification":
        y = (X[:, 0] > 0).long()
        return T1Data(ds["name"], "classification", X, y, X[:8] + .05, y[:8], X[8:] - .05, y[8:], num_classes=2)
    y = X[:, 0] + .1 * X[:, 1]
    return T1Data(ds["name"], "regression", X, y, X[:8] + .01, y[:8], X[8:] + .02, y[8:])


def tiny_config():
    cfg = load_config(Path("configs/t1/confirm_final_20261013/regression__wine.yaml"))
    cfg["experiment"] = EXP
    cfg["core"].update(epochs=2, patience=2, batch_size=4, top_k=3, bias_init_k=3)
    cfg["core_by_dataset"]["regression__wine"].update(top_k=3, batch_size=4)
    cfg["reuse"].update(epochs=2, patience=2, hidden_dims=[8, 4])
    cfg["sync"].update(epochs=2, patience=2, block_epochs=1)
    cfg["conditions"][0]["adapter_warmup_epochs"] = 1
    cfg["datasets"] = [{"name": "wine", "task_type": "regression", "qualified": True},
                       {"name": "wine", "task_type": "classification"}]
    return cfg


def run_jobs(runner, cfg, seeds=(40, 41)):
    with contextlib.redirect_stdout(io.StringIO()) as out:
        for ds in cfg["datasets"]:
            for seed in seeds:
                runner.run_encoder_ablation(ds, seed)
    return out.getvalue()


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    """The same jobs once as files (``files/<EXP>``) and once recorded (``runs/t1/<STUDY>/<EXP>``)."""
    base = tmp_path_factory.mktemp("pair")
    runs = base / "runs"
    runs.mkdir()
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("NNKNN_RUNS", str(runs))
        for name in ("NNKNN_RESUME", "NNKNN_RESUME_EVERY_SECONDS", "NNKNN_RESUME_EVERY_STEPS"):
            mp.delenv(name, raising=False)
        cfg = tiny_config()
        files = Runner(cfg, base / "files")
        files.data_for = tiny_data
        run_jobs(files, cfg)
        units = Runner(tiny_config(), base / "units", study=STUDY)
        units.data_for = tiny_data
        printed = run_jobs(units, tiny_config())
    return {"base": base, "runs": runs, "files": base / "files" / EXP, "units_out": base / "units" / EXP,
            "units": runs / "t1" / STUDY / EXP, "printed": printed}


@pytest.fixture
def runs_root(pair, monkeypatch):
    monkeypatch.setenv("NNKNN_RUNS", str(pair["runs"]))
    return pair["runs"]


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def same(a, b):
    """Equal values (tensors by value, NaN equal to NaN), ignoring wall times."""
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        a, b = torch.as_tensor(np.asarray(a)) if not isinstance(a, torch.Tensor) else a, b
        return a.dtype == b.dtype and a.shape == b.shape and torch.equal(torch.nan_to_num(a), torch.nan_to_num(b))
    if isinstance(a, dict) and isinstance(b, dict):
        keys = {k for k in a.keys() | b.keys() if "seconds" not in str(k) and k not in ("timestamp", "source_snapshot")}
        return all(k in a and k in b and same(a[k], b[k]) for k in keys)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float) and a != a and b != b:
        return True
    return a == b


# ----------------------------------------------------------------------------- helpers


def test_unit_component_keeps_valid_names_and_hashes_others():
    assert records.unit_component("alternating_rr__kappa=10") == "alternating_rr__kappa=10"
    for name in ("a b/c", "x" * 80, "nul", "tuned::F", "x.held_out.y"):
        part = records.unit_component(name)
        assert tr._COMPONENT.match(part) and len(part) <= records.MAX_COMPONENT and "held_out" not in part
        assert part == records.unit_component(name) and part != records.unit_component(name + "!")


def test_ordered_records_keep_key_order_and_values():
    row = {"b": 1, "a": float("nan"), "c": {"z": [1, {"y": 2, "x": None}], "w": True}}
    back = records.unordered(tr.decode_json(tr.canonical_json(records.ordered(row))))
    assert json.dumps(back) == json.dumps(row)
    assert list(back["c"]) == ["z", "w"] and list(back["c"]["z"][1]) == ["y", "x"]


def test_storable_converts_numpy_and_refuses_objects():
    state = records.storable({"a": np.arange(3), "b": np.float32(1.5), "c": (1, "x"), "d": Path("p"), 3: None})
    assert torch.equal(state["a"], torch.arange(3)) and state["b"] == 1.5 and state["c"] == (1, "x")
    assert state["d"] == "p" and tr.state_digest(state)
    with pytest.raises(TypeError):
        records.storable({"f": object()})


# ----------------------------------------------------------------------------- tools/t1_run.py


def test_recorded_runs_equal_the_files_of_the_same_runs(pair):
    rows = {(r["dataset"], r["condition"], r["seed"]): r for r in records.shard_rows(pair["files"])}
    units = records.condition_units(pair["units"])
    assert len(units) == len(rows) == 2 * 2 * 4  # 2 tasks x 2 seeds x (retrieval only, reuse, F, default)
    for row, data in units:
        d = pair["files"] / row["dataset"] / row["condition"] / f"s{row['seed']}"
        old = rows[(row["dataset"], row["condition"], row["seed"])]
        assert json.dumps(row) == json.dumps(old)  # values and key order
        metrics = records.record(data, "metrics")
        assert json.dumps(metrics) == json.dumps(json.loads((d / "metrics.json").read_text()))
        assert data.final() == records.numeric(metrics)
        assert json.dumps(records.record(data, "history")) == json.dumps(json.loads((d / "history.json").read_text()))
        for name in ("case_statistics", "case_maintenance", "retrieval_events"):
            assert json.dumps(records.rows(data, name)) == json.dumps(jsonl(d / f"{name}.jsonl"))
        man = records.record(data, "manifest")
        assert same(man, json.loads((d / "run_manifest.json").read_text()))
        pt = sorted(p.stem for p in d.glob("*.pt"))
        assert sorted(data.states()) == pt
        for stem in pt:
            assert same(torch.load(d / f"{stem}.pt", weights_only=False), tr.load_state(data.path, stem))
        for f in d.glob("*.json"):
            if f.name not in ("metrics.json", "history.json", "run_manifest.json"):
                assert json.dumps(records.record(data, f.stem)) == json.dumps(json.loads(f.read_text()))
        if records.record(data, "history").get("core"):
            assert len(data.series("core/train_loss")) == len(records.record(data, "history")["core"])


def test_recording_writes_no_per_run_files_and_prints_the_same(pair):
    out = pair["units_out"]
    assert not list(out.glob("runs__*.jsonl"))
    assert sorted(p.name for p in out.rglob("*") if p.is_file() and not p.name.startswith("source_snapshot")) == \
        ["data_snapshot_s40.pt"] * 2 + ["data_snapshot_s41.pt"] * 2
    done = sorted(line for line in pair["printed"].splitlines() if line.startswith("[done]"))
    assert len(done) == 16


def test_rerun_skips_complete_units_and_refuses_another_config(pair, runs_root):
    cfg = tiny_config()
    runner = Runner(cfg, pair["base"] / "rerun", study=STUDY)
    runner.data_for = tiny_data
    before = sorted(str(p) for p in pair["units"].rglob("events.out.tfevents.*"))
    with contextlib.redirect_stdout(io.StringIO()) as out:
        runner.run_encoder_ablation(cfg["datasets"][1], 40)
    assert out.getvalue().count("[recorded]") == 4 and "[done]" not in out.getvalue()
    assert sorted(str(p) for p in pair["units"].rglob("events.out.tfevents.*")) == before
    cfg["status"] = "changed"
    runner = Runner(cfg, pair["base"] / "rerun2", study=STUDY)
    runner.data_for = tiny_data
    with pytest.raises(tr.ConfigMismatchError), contextlib.redirect_stdout(io.StringIO()):
        runner.run_encoder_ablation(cfg["datasets"][1], 40)


def test_legacy_reference_records_train_model_states(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "runs").mkdir()
    monkeypatch.setenv("NNKNN_RUNS", str(tmp_path / "runs"))
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs/t1/smoke_legacy.yaml")
    cfg["legacy"]["overrides"].update(training_epochs=3, patience=3, batch_size=4)
    ds = {"name": "tiny", "task_type": "classification"}
    cfg["datasets"] = [ds]
    results = []
    for study in (None, "leg"):
        runner = Runner(cfg, tmp_path / f"out-{study}", study=study)
        runner.data_for = tiny_data
        with contextlib.redirect_stdout(io.StringIO()):
            runner.run_legacy_reference(ds, 0)
        results.append(runner)
    d = tmp_path / "out-None" / cfg["experiment"] / "tiny" / "legacy" / "s0"
    assert (d / "legacy_best.pth").exists()
    unit = tmp_path / "runs" / "t1" / "leg" / cfg["experiment"] / "tiny" / "legacy" / "s0"
    data = tr.load_run(unit)
    assert {"best_retrieval", "final", "checkpoint"} <= set(data.states())
    assert data.series("val/loss") and data.series("train/stage")
    assert same(json.loads((d / "metrics.json").read_text()), records.record(data, "metrics"))
    old = torch.load(d / "checkpoint.pt", weights_only=False)
    new = tr.load_state(unit, "checkpoint")
    assert same(old["model_state"], new["model_state"])
    assert not list((tmp_path / "out-leg").rglob("*.pth")) and not (tmp_path / "checkpoints").exists()


# ----------------------------------------------------------------------------- readers


def _summarize(src, out, *extra):
    from tools import t1_summarize
    argv = ["t1_summarize.py", str(src), "--metric", "test_accuracy_pre", "--metric", "test_rmse_pre",
            "--metric", "test_rmse_post", "--reference", "tuned__retrieval_only", *extra]
    with pytest.MonkeyPatch.context() as mp, contextlib.redirect_stdout(io.StringIO()):
        mp.setattr(sys, "argv", argv + (["--out", str(out)] if out else []))
        t1_summarize.main()


def test_summaries_are_byte_identical_from_units_and_files(pair, tmp_path):
    files = Path(shutil.copytree(pair["files"], tmp_path / "files"))
    _summarize(files, None)
    _summarize(pair["units"], tmp_path / "from_units")
    for name in ("summary.csv", "paired_vs_tuned__retrieval_only.csv"):
        assert (files / name).read_bytes() == (tmp_path / "from_units" / name).read_bytes()
    with pytest.raises(SystemExit):
        _summarize(pair["units"], None)  # a unit tree needs --out


def test_schedule_search_rows_come_from_units_in_shard_order(pair):
    from tools.t1_schedule_search import load_rows
    assert json.dumps(load_rows(pair["units"])) == json.dumps(load_rows(pair["files"]))


def test_cbm_diagnostics_are_identical_from_units_and_files(pair, tmp_path):
    from tools.t1_cbm_diagnostics import main
    outs = {}
    for name, src in (("files", pair["files"]), ("units", pair["units"])):
        with contextlib.redirect_stderr(io.StringIO()):
            assert main([str(src), "--all-seeds", "--draws", "3", "--json", str(tmp_path / f"{name}.json"),
                         "--md", str(tmp_path / f"{name}.md")]) == 0
        outs[name] = json.loads((tmp_path / f"{name}.json").read_text())
    assert outs["files"]["runs"] and outs["files"]["summary"] == outs["units"]["summary"]
    assert [r["metrics"] for r in outs["files"]["runs"]] == [r["metrics"] for r in outs["units"]["runs"]]
    md = [(tmp_path / f"{n}.md").read_text().replace(str(pair[n]), "ROOT") for n in ("files", "units")]
    assert md[0] == md[1]


def test_confirm_audit_checks_units(pair, tmp_path, runs_root):
    from tools.t1_confirm_audit import audit_units, main
    report = audit_units(pair["units"])
    assert report["ok"], (report["problems"], report["unit_self_check"]["mismatches"][:3])
    assert report["unit_self_check"]["exact"] == report["rows"] == 16
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "job.log").write_text(pair["printed"], encoding="utf-8")
    (logs / "job.json").write_text(json.dumps({"exit_code": 0}))
    assert audit_units(pair["units"], logs=logs)["ok"]
    (logs / "job.log").write_text("", encoding="utf-8")
    assert not audit_units(pair["units"], logs=logs)["ok"]
    with contextlib.redirect_stdout(io.StringIO()):
        assert main([str(pair["units"])]) == 0
        with pytest.raises(SystemExit):
            main([str(pair["units"]), "--repair", "--logs", str(logs)])


def test_confirm_report_reads_rows_and_predictions_from_units(pair):
    from tools.t1_confirm_report import events_predictions, experiment_rows
    legacy = [(row, src) for src, row, _ in experiment_rows(pair["files"])]
    units = [(row, data) for _, row, data in experiment_rows(pair["units"])]
    assert json.dumps([r for r, _ in legacy]) == json.dumps([r for r, _ in units])
    for (row, _), (_, data) in zip(legacy, units):
        if row["task_type"] == "classification":
            d = pair["files"] / row["dataset"] / row["condition"] / f"s{row['seed']}" / "retrieval_events.jsonl"
            assert events_predictions(d, "final_prediction") == events_predictions(data, "final_prediction")


# ----------------------------------------------------------------------------- grid, ablation, reviewer


def test_grid_trial_units_reuse_and_failures(tmp_path, monkeypatch):
    import tools.t1_grid_search as search
    (tmp_path / "runs").mkdir()
    monkeypatch.setenv("NNKNN_RUNS", str(tmp_path / "runs"))
    x = torch.arange(24).float().reshape(-1, 2)
    y = torch.arange(12) % 2
    data = T1Data("tiny", "classification", x[:6], y[:6], x[6:9], y[6:9], x[9:], y[9:], num_classes=2)
    job_files, job_units = tmp_path / "out/classification__tiny/knn", tmp_path / "grid/classification__tiny/knn"
    params = {"n_neighbors": 3, "weights": "uniform"}
    with contextlib.redirect_stdout(io.StringIO()):
        old = search.run_trial(job_files, "knn", "c1", params, "evaluation", lambda s: data, 30, 1, 1, "src", test=True)
        new = search.run_trial(job_units, "knn", "c1", params, "evaluation", lambda s: data, 30, 1, 1, "src", test=True,
                               study="g")
        again = search.run_trial(job_units, "knn", "c1", params, "evaluation", lambda s: None, 30, 1, 1, "src", test=True,
                                 study="g")
    assert same(old, new) and json.dumps(again) == json.dumps(new)
    assert not job_units.exists()
    unit = records.trial_unit("g", "classification__tiny", "knn", "evaluation", "c1", 30)
    run = tr.load_run(unit)
    saved = torch.load(job_files / "evaluation/c1/s30/test_predictions.pt", weights_only=True)
    assert np.array_equal(run.array("test_prediction"), saved["prediction"].numpy())
    assert np.array_equal(run.array("test_target"), saved["target"].numpy())
    assert run.final()["validation_error"] == new["validation_error"]
    model = search.trial_model(unit)
    assert np.array_equal(model.predict_proba(data.X_test.numpy()), saved["prediction"].numpy())
    from tools.t1_confirm_report import saved_predictions
    assert all(np.array_equal(a, b) for a, b in zip(saved_predictions(unit),
                                                     saved_predictions(job_files / "evaluation/c1/s30/test_predictions.pt")))

    def broken(seed):
        raise RuntimeError("no data")
    with contextlib.redirect_stdout(io.StringIO()):
        failed = search.run_trial(job_units, "knn", "c2", params, "screen", broken, 20, 1, 1, "src", study="g")
        assert failed["status"] == "failed"
        bad = records.trial_unit("g", "classification__tiny", "knn", "screen", "c2", 20)
        assert tr.run_status(bad).status == "failed"
        refit = search.run_trial(job_units, "knn", "c2", params, "screen", lambda s: data, 20, 1, 1, "src", study="g")
    assert refit["status"] == "completed" and tr.run_status(bad).status == "complete"


def test_grid_prefill_reads_trial_units(tmp_path, monkeypatch):
    import tools.t1_grid_search as search
    from tools.t1_grid_prefill import receipt
    (tmp_path / "runs").mkdir()
    monkeypatch.setenv("NNKNN_RUNS", str(tmp_path / "runs"))
    x = torch.arange(24).float().reshape(-1, 2)
    y = x[:, 0] * .5
    data = T1Data("tiny", "regression", x[:6], y[:6], x[6:9], y[6:9], x[9:], y[9:])
    job = tmp_path / "grid/regression__tiny/knn"
    assert receipt(job, "screen", "c", 20, run_study="g") is None
    with contextlib.redirect_stdout(io.StringIO()):
        row = search.run_trial(job, "knn", "c", {"n_neighbors": 2}, "screen", lambda s: data, 20, 1, 1, "src", study="g")
    assert json.dumps(receipt(job, "screen", "c", 20, run_study="g")) == json.dumps(row)


def test_core_ablation_trial_unit_matches_receipt(tmp_path, monkeypatch):
    import tools.t1_core_ablation as abl
    (tmp_path / "runs").mkdir()
    monkeypatch.setenv("NNKNN_RUNS", str(tmp_path / "runs"))
    data = tiny_data({"task_type": "regression", "name": "tiny"}, 3)
    settings = dict(epochs=2, patience=2, batch_size=4, top_k=3, bias_init_k=3)
    with contextlib.redirect_stdout(io.StringIO()):
        old = abl.run_trial(tmp_path / "out/regression__tiny/base", "base", settings, lambda s: data, 30, "src")
        new = abl.run_trial(tmp_path / "u/regression__tiny/base", "base", settings, lambda s: data, 30, "src", "abl")
        again = abl.run_trial(tmp_path / "u/regression__tiny/base", "base", settings, lambda s: None, 30, "src", "abl")
    assert old["status"] == new["status"] == "completed" and same(old, new) and json.dumps(again) == json.dumps(new)
    unit = tr.unit_dir(records.key("abl", "regression__tiny", "base", "s30"))
    run = tr.load_run(unit)
    saved = torch.load(tmp_path / "out/regression__tiny/base/s30/test_predictions.pt", weights_only=True)
    assert np.array_equal(run.array("test_prediction"), saved["prediction"].numpy())
    assert "model" in run.states() and run.series("core/train_loss")


def test_core_ablation_uses_inlined_grid_winners():
    import yaml
    from tools.t1_core_ablation import grid_winner, task_key
    study = yaml.safe_load(Path("configs/t1/core_ablation_20261027.yaml").read_text(encoding="utf-8"))
    study["grid"] = "/nonexistent"
    for ds in study["datasets"]:
        assert grid_winner(study, ds) == study["grid_winners"][task_key(ds)]["parameters"]


def test_reviewer_session_records_version_states(tmp_path, monkeypatch):
    from model.t1.reviewer_ui import ReviewerSession
    (tmp_path / "runs").mkdir()
    monkeypatch.setenv("NNKNN_RUNS", str(tmp_path / "runs"))
    plain = ReviewerSession(tmp_path / "plain", epochs=2)
    with tr.open_run(records.key("rv", "reviewer", "s1"), config={"kind": "test"}) as run:
        recorded = ReviewerSession(tmp_path / "session", epochs=2, run=run)
    assert plain.snapshot == recorded.snapshot
    data = tr.load_run(run.path)
    assert list(data.states()) == ["v0000-M0"] and records.record(data, "view/v0000") == json.loads(
        (tmp_path / "plain/v0000-M0/view.json").read_text(encoding="utf-8"))
    assert same(torch.load(tmp_path / "plain/v0000-M0/checkpoint.pt", weights_only=False)["model_state"],
                tr.load_state(run.path, "v0000-M0")["model_state"])
    assert not list((tmp_path / "session").glob("v0000*")) and data.series("m0/train/loss")


def test_frozen_confirmation_configs_name_their_selection_files():
    """The pre-registered confirm_final_20261013 configs stay as frozen: each names its grid selection.json by path."""
    import yaml
    paths = sorted(Path("configs/t1/confirm_final_20261013").glob("*.yaml"))
    assert len(paths) == 30
    for path in paths:
        sources = yaml.safe_load(path.read_text(encoding="utf-8"))["core_selection_source"]
        assert sources and all(isinstance(v, str) and v.endswith("selection.json") for v in sources.values()), path
