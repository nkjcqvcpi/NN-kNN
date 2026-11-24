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


def receipts(tree):
    """{trial unit path below the study: its ``trial`` record} of a grid's complete trial units."""
    from model.common.tracking import iter_runs, load_run
    from model.t1.records import record
    return {Path(i.path).relative_to(tree).as_posix(): record(load_run(i.path), "trial") for i in iter_runs(tree)}


def with_study(src, dst, study):
    """Copy a grid's study.json to ``dst`` for a grid recording into ``study``."""
    frozen = json.loads((src / "study.json").read_text())
    (dst / "study.json").write_text(json.dumps({**frozen, "study": study}))


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
    (tmp_path / "runs").mkdir()
    monkeypatch.setenv("NNKNN_RUNS", str(tmp_path / "runs"))  # inherited by the worker processes
    monkeypatch.setattr(search, "ProcessPoolExecutor", concurrent.futures.ThreadPoolExecutor)
    search.main([str(config), "--out", str(sequential), "--workers", "1", "--study", "S1"])
    # A grid that was started (study.json written, recording into its own study S2) and stopped before any trial.
    parallel.mkdir()
    with_study(sequential, parallel, "S2")
    jobs = [f"{KEY}/nnknn", f"{KEY}/mlp", f"{KEY}/knn"]
    # Real worker processes: each loads the grid module from --grid-repo itself.
    assert prefill.main(["--grid-repo", str(REPO), "--out", str(parallel), "--jobs", *jobs, "--workers", "3"]) == 0
    expected, got = receipts(tmp_path / "runs/t1/S1"), receipts(tmp_path / "runs/t1/S2")
    # knn 6 screen + 4 refit + 1 evaluation; mlp and nnknn 4 + 4 + 1 each
    assert sorted(got) == sorted(expected) and len(got) == 11 + 9 + 9
    drop = {"training_seconds", "inference_seconds"}
    for path, row in expected.items():
        assert {k: v for k, v in got[path].items() if k not in drop} == {k: v for k, v in row.items() if k not in drop}
    # Selection records exist before the grid reruns, the winner's before any test-stream trial ran.
    for family in ("nnknn", "knn", "mlp"):
        for name in ("shortlist.json", "selection.json"):
            assert (parallel / KEY / family / name).read_bytes() == (sequential / KEY / family / name).read_bytes()
        evaluated = min(p.stat().st_mtime for p in (tmp_path / "runs/t1/S2" / KEY / family / "evaluation").rglob(
            "events.out.tfevents.*"))
        assert (parallel / KEY / family / "selection.json").stat().st_mtime <= evaluated
    # The grid's own run now only reads the complete trials and writes the job summaries.
    monkeypatch.setattr(search, "fit_candidate", lambda *args: pytest.fail("the grid must reuse every trial"))
    search.main([str(config), "--out", str(parallel), "--workers", "1", "--study", "S2"])
    for family in ("nnknn", "knn", "mlp"):
        for name in ("shortlist.json", "selection.json"):
            assert (json.loads((parallel / KEY / family / name).read_text())
                    == json.loads((sequential / KEY / family / name).read_text()))
        done = json.loads((parallel / KEY / family / "completed.json").read_text())
        assert done["status"] == "completed"
    with pytest.raises(ValueError, match="already completed"):
        prefill.main(["--grid-repo", str(REPO), "--out", str(parallel), "--jobs", f"{KEY}/knn"])


def test_another_source_is_accepted_an_unknown_job_refused(tmp_path, monkeypatch):
    """The grid's recorded source is provenance (owner's order 2026-11-23): another code state prefills the same
    grid; unknown jobs and another determinism setting are still refused."""
    monkeypatch.delenv("NNKNN_DETERMINISTIC", raising=False)
    gs = prefill.grid_module(REPO)
    study = dict(development_seeds=[20], families=["knn"], datasets=[dict(name="iris", task_type="classification")],
                 grids=dict(knn=dict(n_neighbors=[1])))
    (tmp_path / "study.json").write_text(json.dumps(dict(configuration=study, configuration_id=gs.config_id(study),
                                                         source_sha256="not this source", study="S")))
    loaded = prefill.load_study(gs, REPO, tmp_path)
    assert loaded["source_sha256"] == "not this source" and gs.current_source() != "not this source"
    with pytest.raises(ValueError, match="not a job"):
        prefill.main(["--grid-repo", str(REPO), "--out", str(tmp_path), "--jobs", f"{KEY}/tabm"])
    monkeypatch.setenv("NNKNN_DETERMINISTIC", "1")
    with pytest.raises(ValueError, match="NNKNN_DETERMINISTIC"):
        prefill.main(["--grid-repo", str(REPO), "--out", str(tmp_path), "--jobs", f"{KEY}/knn"])
    path = tmp_path / "selection.json"  # a selection recorded under another source is the same selection
    path.write_text(json.dumps(dict(candidate_id="c", source_sha256="old")))
    prefill.write_frozen(gs, path, dict(candidate_id="c", source_sha256="new"))
    with pytest.raises(ValueError, match="disagrees"):
        prefill.write_frozen(gs, path, dict(candidate_id="d", source_sha256="old"))


def test_a_grid_that_records_no_study_is_refused(tmp_path, monkeypatch):
    """Grids that wrote trial files (no ``"study"`` in their study.json) cannot be prefilled."""
    monkeypatch.delenv("NNKNN_DETERMINISTIC", raising=False)
    gs = prefill.grid_module(REPO)
    study = dict(development_seeds=[20], families=["knn"], datasets=[dict(name="iris", task_type="classification")],
                 grids=dict(knn=dict(n_neighbors=[1])))
    (tmp_path / "study.json").write_text(json.dumps(dict(configuration=study, configuration_id=gs.config_id(study),
                                                         source_sha256="src")))
    with pytest.raises(ValueError, match="no run study"):
        prefill.main(["--grid-repo", str(REPO), "--out", str(tmp_path), "--jobs", f"{KEY}/knn"])
