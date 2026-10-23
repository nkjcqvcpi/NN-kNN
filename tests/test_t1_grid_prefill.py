import concurrent.futures
import json
import shutil
from pathlib import Path

import pytest
import torch
import yaml

import tools.t1_grid_prefill as prefill
import tools.t1_grid_search as search

REPO = Path(__file__).resolve().parents[1]
KEY = "classification__iris"


def receipts(root):
    return {p.relative_to(root).as_posix(): json.loads(p.read_text()) for p in root.rglob("receipt.json")}


def test_prefilled_receipts_reproduce_the_sequential_grid(tmp_path, monkeypatch, request):
    threads = torch.get_num_threads()
    request.addfinalizer(lambda: torch.set_num_threads(threads))  # the grid's own run uses threads here
    monkeypatch.delenv("NNKNN_DETERMINISTIC", raising=False)
    study = dict(development_seeds=[20, 21], evaluation_seeds=[40], screen_epochs=2, screen_patience=1,
        refit_epochs=3, refit_patience=1, finalists=2, torch_threads=1, splits=dict(test_frac=.2, val_frac=.2),
        families=["nnknn", "knn", "mlp"], datasets=[dict(name="iris", task_type="classification", max_train=60)],
        grids=dict(nnknn=dict(representation=["raw"], top_k=[3], lr_case=[0.01], glocal_fw_set_num=[1], tau=[0.5, 1.0]),
                   knn=dict(n_neighbors=[1, 3, 5], weights=["distance"]),
                   mlp=dict(width=[8], layers=[1], lr=[0.003, 0.01], weight_decay=[0.0], dropout=[0.0])))
    config, sequential, parallel = tmp_path / "grid.yaml", tmp_path / "sequential", tmp_path / "parallel"
    config.write_text(yaml.safe_dump(study))
    monkeypatch.setattr(search, "ProcessPoolExecutor", concurrent.futures.ThreadPoolExecutor)
    search.main([str(config), "--out", str(sequential), "--workers", "1"])
    # A grid that was started (study.json written) and then stopped before any trial.
    parallel.mkdir()
    shutil.copy(sequential / "study.json", parallel / "study.json")
    jobs = [f"{KEY}/nnknn", f"{KEY}/mlp", f"{KEY}/knn"]
    # Real worker processes: each loads the grid module from --grid-repo itself.
    assert prefill.main(["--grid-repo", str(REPO), "--out", str(parallel), "--jobs", *jobs, "--workers", "3"]) == 0
    expected, got = receipts(sequential), receipts(parallel)
    # knn 6 screen + 4 refit + 1 evaluation; mlp and nnknn 4 + 4 + 1 each
    assert sorted(got) == sorted(expected) and len(got) == 11 + 9 + 9
    drop = {"training_seconds", "inference_seconds"}
    for path, row in expected.items():
        assert {k: v for k, v in got[path].items() if k not in drop} == {k: v for k, v in row.items() if k not in drop}
    # Selection records exist before the grid reruns, the winner's before any test-stream trial ran.
    for family in ("nnknn", "knn", "mlp"):
        for name in ("shortlist.json", "selection.json"):
            assert (parallel / KEY / family / name).read_bytes() == (sequential / KEY / family / name).read_bytes()
        evaluated = min(p.stat().st_mtime for p in (parallel / KEY / family / "evaluation").rglob("receipt.json"))
        assert (parallel / KEY / family / "selection.json").stat().st_mtime <= evaluated
    # The grid's own run now only reads receipts and writes the job summaries.
    monkeypatch.setattr(search, "fit_candidate", lambda *args: pytest.fail("the grid must reuse every receipt"))
    search.main([str(config), "--out", str(parallel), "--workers", "1"])
    for family in ("nnknn", "knn", "mlp"):
        for name in ("shortlist.json", "selection.json"):
            assert (json.loads((parallel / KEY / family / name).read_text())
                    == json.loads((sequential / KEY / family / name).read_text()))
        done = json.loads((parallel / KEY / family / "completed.json").read_text())
        assert done["status"] == "completed"
    with pytest.raises(ValueError, match="already completed"):
        prefill.main(["--grid-repo", str(REPO), "--out", str(parallel), "--jobs", f"{KEY}/knn"])


def test_refuses_another_source_or_an_unknown_job(tmp_path, monkeypatch):
    monkeypatch.delenv("NNKNN_DETERMINISTIC", raising=False)
    gs = prefill.grid_module(REPO)
    study = dict(development_seeds=[20], families=["knn"], datasets=[dict(name="iris", task_type="classification")],
                 grids=dict(knn=dict(n_neighbors=[1])))
    (tmp_path / "study.json").write_text(json.dumps(dict(configuration=study, configuration_id=gs.config_id(study),
                                                         source_sha256="not this source")))
    with pytest.raises(ValueError, match="not the source"):
        prefill.main(["--grid-repo", str(REPO), "--out", str(tmp_path), "--jobs", f"{KEY}/knn"])
    monkeypatch.setattr(gs, "source_fingerprint", lambda repo: "not this source")
    with pytest.raises(ValueError, match="not a job"):
        prefill.main(["--grid-repo", str(REPO), "--out", str(tmp_path), "--jobs", f"{KEY}/tabm"])
    monkeypatch.setenv("NNKNN_DETERMINISTIC", "1")
    with pytest.raises(ValueError, match="NNKNN_DETERMINISTIC"):
        prefill.main(["--grid-repo", str(REPO), "--out", str(tmp_path), "--jobs", f"{KEY}/knn"])


def test_legacy_trial_matches_the_grid_trial(tmp_path, monkeypatch, request):
    """Grids without run_trial (79dbb31) go through legacy_trial, which must write the same receipts."""
    threads = torch.get_num_threads()
    request.addfinalizer(lambda: torch.set_num_threads(threads))
    monkeypatch.delenv("NNKNN_DETERMINISTIC", raising=False)
    study = dict(development_seeds=[20, 21], evaluation_seeds=[40], screen_epochs=2, screen_patience=1,
        refit_epochs=3, refit_patience=1, finalists=1, torch_threads=1,
        splits=dict(test_frac=.2, val_frac=.2, max_train=60), families=["nnknn", "knn"],
        datasets=[dict(name="iris", task_type="classification")],
        grids=dict(nnknn=dict(representation=["raw"], top_k=[3], lr_case=[0.01], glocal_fw_set_num=[1], tau=[0.5, 1.0]),
                   knn=dict(n_neighbors=[1, 5], weights=["distance"])))
    config, sequential, legacy = tmp_path / "grid.yaml", tmp_path / "sequential", tmp_path / "legacy"
    config.write_text(yaml.safe_dump(study))
    monkeypatch.setattr(search, "ProcessPoolExecutor", concurrent.futures.ThreadPoolExecutor)
    search.main([str(config), "--out", str(sequential), "--workers", "1"])
    legacy.mkdir()
    shutil.copy(sequential / "study.json", legacy / "study.json")
    gs = prefill.grid_module(REPO)
    monkeypatch.delattr(gs, "run_trial")  # in-process workers share this module
    monkeypatch.setattr(prefill, "ProcessPoolExecutor", concurrent.futures.ThreadPoolExecutor)
    calls = []
    original = prefill.legacy_trial
    monkeypatch.setattr(prefill, "legacy_trial", lambda *a: calls.append(a[7]) or original(*a))
    assert prefill.main(["--grid-repo", str(REPO), "--out", str(legacy), "--jobs", f"{KEY}/nnknn", f"{KEY}/knn",
                         "--workers", "1"]) == 0
    expected, got = receipts(sequential), receipts(legacy)
    assert sorted(got) == sorted(expected) and len(calls) == len(got) == (4 + 2 + 1) * 2
    drop = {"training_seconds", "inference_seconds"}
    for path, row in expected.items():
        assert ({k: v for k, v in search.compact_row(got[path]).items() if k not in drop}
                == {k: v for k, v in row.items() if k not in drop})
