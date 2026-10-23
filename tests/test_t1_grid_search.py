import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from model.t1 import data as data_module
from tools.t1_grid_search import expand_grid, rank_candidates, dataset_family, numeric_metrics


def test_rank_requires_every_seed_and_never_uses_test_scores():
    rows = [dict(candidate_id="a", seed=s, status="completed", validation_error=.2,
                 validation_loss=.3, test_accuracy=0.) for s in (20, 21)]
    rows += [dict(candidate_id="b", seed=20, status="completed", validation_error=0., validation_loss=0.)]
    assert rank_candidates(rows, ["a", "b"], [20, 21]) == ["a"]
    for row in rows:
        row["test_accuracy"] = 1.
    assert rank_candidates(rows, ["a", "b"], [20, 21]) == ["a"]
    assert len(expand_grid(dict(a=[1, 2], b=[3, 4]))) == 4


def test_categories_and_numeric_scaling_use_training_only(monkeypatch):
    frame = pd.DataFrame(dict(number=np.arange(100), category=[f"category{i}" for i in range(100)]))
    target = np.tile([0, 1], 50)
    monkeypatch.setattr(data_module, "_load_raw", lambda *args: (frame, target, {}))
    data = data_module.make_splits("fake", "classification", 3, test_frac=.2, val_frac=.2)
    assert data.X_train.shape[1] == 1 + len(data.y_train)
    assert torch.all(data.X_test[:, 1:] == 0)
    assert torch.all(data.X_val[:, 1:] == 0)
    assert abs(float(data.X_train[:, 0].mean())) < 1e-6
    assert not set(data.meta["split_indices"]["test"]) & set(data.meta["split_indices"]["train"])


def test_worker_freezes_selection_before_test_prediction(tmp_path, monkeypatch):
    import tools.t1_grid_search as search
    from model.t1.data import T1Data
    x = torch.arange(12).float().reshape(-1, 1)
    y = torch.arange(12).float()
    data = T1Data("tiny", "regression", x[:6], y[:6], x[6:9], y[6:9], x[9:], y[9:])
    monkeypatch.setattr(search, "make_splits", lambda *args, **kwargs: data)
    seen = []
    def fit(family, params, allowed, seed, epochs, patience):
        assert "X_test" not in allowed and "y_test" not in allowed
        seen.append(seed)
        return torch.nn.Linear(1, 1), {}
    monkeypatch.setattr(search, "fit_candidate", fit)
    original = search.prediction
    def predict(model, family, X, task):
        if torch.equal(X, data.X_test):
            assert (tmp_path / "regression__tiny/mlp/selection.json").exists()
        return X.numpy().reshape(-1)
    monkeypatch.setattr(search, "prediction", predict)
    study = dict(torch_threads=1, grids={"mlp": {"width": [4, 8]}}, development_seeds=[20, 21],
        evaluation_seeds=[30], screen_epochs=1, screen_patience=1, refit_epochs=1,
        refit_patience=1, finalists=1, source_sha256="test", splits={})
    result = dataset_family(study, dict(name="tiny", task_type="regression"), "mlp", tmp_path)
    assert result["status"] == "completed"
    assert seen.count(30) == 1
    receipts = list((tmp_path / "regression__tiny/mlp/screen").rglob("receipt.json"))
    assert len(receipts) == 4
    assert all(not json.loads(p.read_text())["test_stream_evaluated"] for p in receipts)


def test_regression_metric_records_original_target_units():
    metrics = numeric_metrics(np.array([1., 2.]), np.array([0., 1.]), "regression", (10., 3.))
    assert metrics["rmse_standardized"] == 1.
    assert metrics["rmse_original"] == 3.


def test_dataset_max_train_overrides_the_study_cap():
    import tools.t1_grid_search as search
    splits = dict(test_frac=.2, val_frac=.2, maint_frac=0., reg_cohort_bins=5, max_train=3000)
    plain = dict(name="iris", task_type="classification")
    assert search.dataset_splits(dict(splits=splits), plain) == splits
    assert search.dataset_splits(dict(splits=splits), {**plain, "max_train": 40}) == {**splits, "max_train": 40}
    assert search.dataset_splits(dict(splits=splits), {**plain, "max_train": None})["max_train"] is None
    assert search.dataset_splits(dict(splits=dict(test_frac=.2)), plain) == dict(test_frac=.2)
    assert splits["max_train"] == 3000
    load = lambda ds: search.split_loader(dict(splits=dict(test_frac=.2, val_frac=.2)), ds)(30)  # noqa: E731
    capped, full = load({**plain, "max_train": 40}), load(plain)
    assert (len(capped.y_train), len(full.y_train)) == (40, 90)
    assert capped.meta["split_indices"]["test"] == full.meta["split_indices"]["test"]
    assert set(capped.meta["split_indices"]["train"]) < set(full.meta["split_indices"]["train"])


def test_receipts_hash_split_indices_and_old_receipts_still_resume(tmp_path, monkeypatch):
    import tools.t1_grid_search as search
    from model.t1.data import T1Data
    x = torch.arange(12).float().reshape(-1, 1)
    y = torch.arange(12).float()
    indices = dict(train=list(range(6)), validation=[6, 7, 8], test=[9, 10, 11], maintenance=[])
    data = T1Data("tiny", "regression", x[:6], y[:6], x[6:9], y[6:9], x[9:], y[9:], meta=dict(split_indices=indices))
    params = dict(n_neighbors=2, weights="uniform", p=2)
    row = search.run_trial(tmp_path, "knn", "c", params, "screen", lambda seed: data, 20, 1, 1, "src")
    stored = json.loads((tmp_path / "screen/c/s20/receipt.json").read_text())
    assert row == stored and "deterministic_algorithms" not in stored
    assert "split_indices" not in stored["dataset"]
    assert stored["dataset"]["split_indices_sha256"] == search.split_digest(indices)
    assert search.split_digest(dict(reversed(indices.items()))) == search.split_digest(indices)
    assert search.split_digest({k: [np.int64(i) for i in v] for k, v in indices.items()}) == search.split_digest(indices)
    assert search.split_digest({**indices, "test": [9, 11, 10]}) != search.split_digest(indices)
    # A receipt written before hashing is reused unchanged on disk and compacted in memory.
    old = dict(stored, dataset={k: v for k, v in stored["dataset"].items() if k != "split_indices_sha256"} | dict(split_indices=indices))
    old_path = tmp_path / "screen/d/s20/receipt.json"
    old_path.parent.mkdir(parents=True)
    old_path.write_text(json.dumps(old))
    def no_data(seed):
        raise AssertionError("a reused receipt must not reload data")
    reused = search.run_trial(tmp_path, "knn", "d", params, "screen", no_data, 20, 1, 1, "src")
    assert reused["dataset"] == stored["dataset"] and json.loads(old_path.read_text()) == old
    monkeypatch.setattr(search.torch, "are_deterministic_algorithms_enabled", lambda: True)
    flagged = search.run_trial(tmp_path, "knn", "e", params, "screen", lambda seed: data, 20, 1, 1, "src")
    assert flagged["deterministic_algorithms"] is True


def test_deterministic_environment_flag(monkeypatch):
    import os
    import tools.t1_grid_search as search
    calls = []
    monkeypatch.setattr(search.torch, "use_deterministic_algorithms", calls.append)
    monkeypatch.delenv("NNKNN_DETERMINISTIC", raising=False)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "")  # records the original value for teardown
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG")
    search.apply_determinism()
    assert calls == [] and "CUBLAS_WORKSPACE_CONFIG" not in os.environ
    monkeypatch.setenv("NNKNN_DETERMINISTIC", "1")
    search.apply_determinism()
    # Same convention as tools/t1_run.py: deterministic cuBLAS needs a fixed workspace.
    assert calls == [True] and os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    search.apply_determinism()
    assert calls == [True, True] and os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"


def test_evaluate_refits_the_frozen_winner_without_reselection(tmp_path, monkeypatch, request):
    import concurrent.futures
    import shutil
    import pytest
    import yaml
    import tools.t1_grid_search as search
    threads = torch.get_num_threads()
    request.addfinalizer(lambda: torch.set_num_threads(threads))  # workers run in this process here
    monkeypatch.delenv("NNKNN_DETERMINISTIC", raising=False)
    study = dict(development_seeds=[20], evaluation_seeds=[30], screen_epochs=2, screen_patience=1, refit_epochs=3,
        refit_patience=1, finalists=1, torch_threads=1, splits=dict(test_frac=.2, val_frac=.2),
        families=["knn", "mlp", "catboost"], datasets=[dict(name="iris", task_type="classification", max_train=60)],
        grids=dict(knn=dict(n_neighbors=[3, 5], weights=["distance"], p=[2]),
                   mlp=dict(width=[8], layers=[1], lr=[0.003], weight_decay=[0.0], dropout=[0.0]),
                   catboost=dict(depth=[2], iterations=[5])))
    config, grid, out = tmp_path / "grid.yaml", tmp_path / "grid", tmp_path / "evaluate"
    config.write_text(yaml.safe_dump(study))
    # Worker threads keep the test in one process; one worker keeps redirect_stdout unshared.
    monkeypatch.setattr(search, "ProcessPoolExecutor", concurrent.futures.ThreadPoolExecutor)
    search.main([str(config), "--out", str(grid), "--workers", "1", "--only-family", "knn"])
    search.main([str(config), "--out", str(grid), "--workers", "1", "--only-family", "mlp"])
    fits = []
    original_fit = search.fit_candidate
    def fit(family, params, allowed, seed, epochs, patience):
        fits.append((family, params, seed, epochs, patience))
        return original_fit(family, params, allowed, seed, epochs, patience)
    monkeypatch.setattr(search, "fit_candidate", fit)
    monkeypatch.setattr(search, "rank_candidates", lambda *args: pytest.fail("evaluate must not rank"))
    evaluate = ["evaluate", "--grid", str(grid), "--out", str(out), "--workers", "1"]
    search.main(evaluate + ["--families", "knn", "mlp", "catboost", "--seeds", "30", "31"])
    key = "classification__iris"
    statuses = {r["family"]: r["status"] for r in json.loads((out / "results.json").read_text())}
    assert statuses == dict(knn="completed", mlp="completed", catboost="no_selection")
    for family in ("knn", "mlp"):
        completed = json.loads((grid / key / family / "completed.json").read_text())
        evaluated = json.loads((out / key / family / "evaluation.json").read_text())
        assert evaluated["winner"] == completed["winner"] and evaluated["status"] == "completed"
        assert [r["seed"] for r in evaluated["evaluation"]] == [30, 31]
        before, after = completed["evaluation"][0], evaluated["evaluation"][0]
        assert set(before) == set(after) and before["dataset"] == after["dataset"]
        assert before["test_metrics"] == after["test_metrics"]
        assert before["validation_metrics"] == after["validation_metrics"]
        assert after["dataset"]["n_train"] == 60 and "split_indices_sha256" in after["dataset"]
        assert sorted(p.name for p in (out / key / family).iterdir()) == ["evaluation", "evaluation.json", "training.log"]
        assert [p.name for p in (out / key / family / "evaluation").iterdir()] == [completed["winner"]]
        grid_params = completed["parameters"]
        assert [f[1:] for f in fits if f[0] == family] == [(grid_params, s, 3, 1) for s in (30, 31)]
    # Resuming adds seeds, reuses receipts and keeps earlier seeds in evaluation.json.
    fits.clear()
    search.main(evaluate + ["--families", "knn", "--seeds", "32", "--datasets", key])
    assert [f[2] for f in fits] == [32]
    assert [r["seed"] for r in json.loads((out / key / "knn/evaluation.json").read_text())["evaluation"]] == [30, 31, 32]
    assert len(json.loads((out / "evaluate.json").read_text())["invocations"]) == 2
    with pytest.raises(ValueError, match="Not in the grid"):
        search.main(evaluate + ["--families", "tabm", "--seeds", "30"])
    with pytest.raises(ValueError, match="outside the grid"):
        search.main(["evaluate", "--grid", str(grid), "--out", str(grid / "x"), "--families", "knn", "--seeds", "30"])
    # Another grid's directory holds stale evaluation receipts: it is refused and left untouched.
    foreign = tmp_path / "foreign"
    shutil.copytree(grid, foreign)
    assert list((foreign / key / "knn/evaluation").rglob("s30/receipt.json"))
    snapshot = {p: p.read_bytes() for p in foreign.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="not empty"):
        search.main(["evaluate", "--grid", str(grid), "--out", str(foreign), "--families", "knn", "--seeds", "30"])
    assert {p: p.read_bytes() for p in foreign.rglob("*") if p.is_file()} == snapshot
    empty = tmp_path / "empty"
    empty.mkdir()
    search.main(["evaluate", "--grid", str(grid), "--out", str(empty), "--workers", "1", "--families", "knn",
                 "--seeds", "30", "--datasets", key])
    assert (json.loads((empty / key / "knn/evaluation.json").read_text())["evaluation"][0]["test_metrics"]
            == json.loads((grid / key / "knn/completed.json").read_text())["evaluation"][0]["test_metrics"])
    # NNKNN_DETERMINISTIC is recorded and an output directory may not be resumed under another setting.
    determinism = []
    monkeypatch.setattr(search.torch, "use_deterministic_algorithms", determinism.append)
    assert "deterministic_algorithms" not in json.loads((grid / "study.json").read_text())
    monkeypatch.setenv("NNKNN_DETERMINISTIC", "1")
    with pytest.raises(ValueError, match="NNKNN_DETERMINISTIC"):
        search.main(evaluate + ["--families", "knn", "--seeds", "30"])
    with pytest.raises(ValueError, match="NNKNN_DETERMINISTIC"):
        search.main([str(config), "--out", str(grid), "--workers", "1", "--only-family", "knn"])
    deterministic_grid = tmp_path / "deterministic_grid"
    search.main([str(config), "--out", str(deterministic_grid), "--workers", "1", "--only-dataset", "none"])
    assert json.loads((deterministic_grid / "study.json").read_text())["deterministic_algorithms"] is True
    monkeypatch.delenv("NNKNN_DETERMINISTIC")
    with pytest.raises(ValueError, match="NNKNN_DETERMINISTIC"):
        search.main([str(config), "--out", str(deterministic_grid), "--workers", "1", "--only-dataset", "none"])
    assert determinism == []
    monkeypatch.setattr(search, "source_fingerprint", lambda repo: "changed")
    with pytest.raises(ValueError, match="fresh output"):
        search.main(evaluate + ["--families", "knn", "--seeds", "30"])
