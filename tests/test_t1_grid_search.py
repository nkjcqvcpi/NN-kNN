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
