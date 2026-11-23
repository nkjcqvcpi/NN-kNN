"""tools/t1_confirm_audit.py on genuine tools/t1_run.py run units (tiny confirmation-config runs)."""
import contextlib
import io
import json
import shutil
from pathlib import Path

import pytest
import torch

from model.t1.data import T1Data
from tools.t1_confirm_audit import audit_units, check_fingerprints, main
from tools.t1_run import Runner, load_config

EXP = "confirm_tiny"
STUDY = "audit"


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
                       {"name": "wine", "task_type": "classification"}, {"name": "tiny", "task_type": "regression"}]
    return cfg


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """One experiment from a frozen confirmation config recorded as units: wine in two roles plus an unqualified
    task, two seeds, the confirmation's four condition types (4 runs and a core stage per task and seed). Each (dataset,
    seed) job's stdout is kept as tools/t1_parallel.py keeps it: ``logs/<task>__s<seed>.log`` beside a receipt."""
    base = tmp_path_factory.mktemp("built")
    (base / "runs").mkdir()
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("NNKNN_RUNS", str(base / "runs"))
        cfg = tiny_config()
        runner = Runner(cfg, base / "inputs", STUDY)
        runner.data_for = tiny_data
        (base / "logs").mkdir()
        for ds in cfg["datasets"]:
            for seed in (40, 41):
                with contextlib.redirect_stdout(io.StringIO()) as stdout:
                    runner.run_encoder_ablation(ds, seed)
                log = base / "logs" / f"{ds['task_type']}__{ds['name']}__s{seed}.log"
                log.write_text(stdout.getvalue(), encoding="utf-8")
                log.with_suffix(".json").write_text(json.dumps({"dataset": ds["name"], "seed": seed, "exit_code": 0}))
    return base


@pytest.fixture
def root(built, tmp_path, monkeypatch):
    """A copy of the run root; returns the experiment's unit tree (its job logs are ``<tmp>/logs``)."""
    shutil.copytree(built / "runs", tmp_path / "runs")
    shutil.copytree(built / "logs", tmp_path / "logs")
    monkeypatch.setenv("NNKNN_RUNS", str(tmp_path / "runs"))
    return tmp_path / "runs" / "t1" / STUDY / EXP


def test_every_unit_rebuilds_its_row_exactly(root, tmp_path):
    report = audit_units(root, settle_seconds=0)
    assert report["ok"], report["problems"]
    assert report["rows"] == report["unit_self_check"]["exact"] == 3 * 2 * 4
    assert report["stage_units"] == 3 * 2  # the trained core: shared training, no row
    assert report["run_dirs"] == report["rows"] + report["stage_units"]
    out = tmp_path / "audit.json"
    with contextlib.redirect_stdout(io.StringIO()):
        assert main([str(root), "--settle-seconds", "0", "--json", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["ok"] is True


def test_done_lines_are_required_with_logs(root, tmp_path):
    logs = tmp_path / "logs"
    assert audit_units(root, logs=logs, settle_seconds=0)["ok"]
    victim = sorted(logs.glob("*.log"))[0]
    victim.write_text("", encoding="utf-8")
    report = audit_units(root, logs=logs, settle_seconds=0)
    assert not report["ok"] and len(report["done_lines_missing"]) == 4
    victim.with_suffix(".json").write_text(json.dumps({"exit_code": 1}))
    assert not audit_units(root, logs=logs, settle_seconds=0)["ok"]


def test_crashed_run_fails_after_the_settle_window(tmp_path, monkeypatch):
    (tmp_path / "runs").mkdir()
    monkeypatch.setenv("NNKNN_RUNS", str(tmp_path / "runs"))
    cfg = tiny_config()
    cfg["datasets"] = cfg["datasets"][2:]
    runner = Runner(cfg, tmp_path / "inputs", STUDY)
    runner.data_for = tiny_data
    original = Runner._finish

    def crash(self, *args, **kwargs):
        if kwargs["cond_label"].endswith("alternating_default_matched"):
            raise RuntimeError("simulated crash")
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Runner, "_finish", crash)
    with pytest.raises(RuntimeError), contextlib.redirect_stdout(io.StringIO()):
        runner.run_encoder_ablation(cfg["datasets"][0], 40)
    root = tmp_path / "runs" / "t1" / STUDY / EXP
    assert audit_units(root, settle_seconds=3600)["ok"]  # recent: may still be writing
    report = audit_units(root, settle_seconds=0)
    assert not report["ok"] and [i["status"] for i in report["incomplete_run_dirs"]] == ["failed"]


def test_source_fingerprint_expectations(root):
    fp = next(iter(audit_units(root, settle_seconds=0)["source_fingerprints"]["regression__tiny"]["fingerprints"]))
    rep = audit_units(root, settle_seconds=0, expect=[("*wine", fp), ("regression__*", "f" * 12), ("missing_task", fp)])
    wine, regression, none = rep["fingerprint_expectations"]
    assert wine["passed"] and wine["tasks"] == ["classification__wine", "regression__wine"]
    assert not regression["passed"] and regression["other_fingerprints"]["regression__tiny"] == {fp: 8}
    assert not none["passed"] and none["tasks"] == []
    with contextlib.redirect_stdout(io.StringIO()):
        assert main([str(root), "--settle-seconds", "0", "--expect-fingerprint", f"tiny={'f' * 12}"]) == 1
        assert main([str(root), "--settle-seconds", "0", "--expect-fingerprint", f"classification__*={fp}"]) == 0


def test_a_folder_without_run_units_is_refused(tmp_path):
    legacy = tmp_path / "results" / EXP
    legacy.mkdir(parents=True)
    (legacy / "runs__tiny.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        audit_units(legacy)


def test_fingerprint_expectation_uses_the_most_specific_pattern():
    fps = {"regression__wine": {"dataset_keys": ["regression__wine"], "fingerprints": {"bbbb" * 4: 3}},
           "classification__wine": {"dataset_keys": ["wine"], "fingerprints": {"aaaa" * 4: 5}},
           "regression__yacht": {"dataset_keys": ["yacht"], "fingerprints": {"aaaa" * 4: 2}}}
    for expect in ([("*", "aaaa" * 3), ("regression__wine", "bbbb" * 3)],
                   [("regression__wine", "bbbb" * 3), ("*", "aaaa" * 3)]):
        got = {e["pattern"]: e for e in check_fingerprints(fps, expect)}
        assert got["*"]["tasks"] == ["classification__wine", "regression__yacht"] and got["*"]["passed"]
        assert got["*"]["overridden"] == ["regression__wine"]
        assert got["regression__wine"]["tasks"] == ["regression__wine"] and got["regression__wine"]["passed"]
    # dataset key beats a glob; the glob alone would fail classification wine
    got = check_fingerprints(fps, [("*wine", "bbbb" * 3), ("wine", "aaaa" * 3)])
    assert [e["passed"] for e in got] == [True, True] and got[1]["tasks"] == ["classification__wine"]
    got = check_fingerprints(fps, [("*", "bbbb" * 3)])
    assert not got[0]["passed"] and sorted(got[0]["other_fingerprints"]) == ["classification__wine", "regression__yacht"]
