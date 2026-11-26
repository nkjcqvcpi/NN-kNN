"""Pack p4 of the 2026-11-26 handoff batch (configs/t1/handoff_20261126/p4-baselines-stress/): TabPFN v2 as an extra
comparator of the baselines grid (opt-in family ``tabpfn``, opt-in ``selection: fixed``) and the covariate-shift stress
split (opt-in ``splits.split_mode: covariate_pc1``, also ``t1_grid_search.py evaluate --split-mode``). Defaults stay as
they were: the random split, the grid's screen/refit selection and evaluate's frozen splits.
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

os.environ.setdefault("NNKNN_DEVICE", "cpu")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from model.t1 import data as data_module  # noqa: E402
from model.t1 import records  # noqa: E402
from model.t1 import tabpfn_baseline as tp  # noqa: E402

P4 = REPO / "configs/t1/handoff_20261126/p4-baselines-stress"
CHOICE = "exploratory, our choice, not a PI decision"
SHIFT = {"split_mode": "covariate_pc1", "shift_quantile": 0.7}


# --------------------------------------------------------------------------- covariate_pc1 split


def test_pc1_order_hand_example():
    # one informative direction (x0 = x1), a constant column (dropped) and a missing value (column median)
    X = np.array([[3.0, 3.0, 1.0], [1.0, 1.0, 1.0], [2.0, 2.0, 1.0], [2.0, np.nan, 1.0], [4.0, 4.0, 1.0]])
    order, info = data_module.covariate_pc1_order(X)
    # row 3 is imputed with the median of column 1 (2.5): (2, 2.5) sits between rows 2 and 0 on PC1
    assert order.tolist() == [1, 2, 3, 0, 4]
    assert info["features"] == ["0", "1"] and all(v > 0 for v in info["loadings"])
    assert 0.9 < info["explained_variance_ratio"] <= 1.0
    # the sign is fixed by the largest loading, so flipping every feature reverses the order
    flipped, _ = data_module.covariate_pc1_order(-np.nan_to_num(X, nan=2.5))
    assert flipped.tolist() == [4, 0, 3, 2, 1]
    # ties keep the row index order; a frame uses its numeric columns only
    frame = pd.DataFrame({"a": [1.0, 1.0, 0.0, 2.0], "c": ["x", "y", "x", "y"]})
    assert data_module.covariate_pc1_order(frame)[0].tolist() == [2, 0, 1, 3]
    with pytest.raises(ValueError, match="non-constant"):
        data_module.covariate_pc1_order(np.ones((4, 2)))


def _fake(monkeypatch, n=100, task="classification"):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 3)).astype(np.float32)
    y = (np.arange(n) % 3).astype(np.int64) if task == "classification" else rng.normal(size=n).astype(np.float32)
    monkeypatch.setattr(data_module, "_load_raw", lambda *a: (X, y, {"outer_train_indices": [0], "outer_test_indices": [1],
                                                                       "outer_repeat": 0, "outer_fold": 0}))
    return X, y


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_shift_split_trains_on_the_lower_side(monkeypatch, task):
    X, y = _fake(monkeypatch, task=task)
    order, info = data_module.covariate_pc1_order(X)
    scores = info["scores"]
    kw = dict(test_frac=0.2, val_frac=0.2, maint_frac=0.0, reg_cohort_bins=5)
    datas = [data_module.make_splits("fake", task, s, **kw, **SHIFT) for s in (30, 31)]
    for d in datas:
        idx = d.meta["split_indices"]
        side = idx["train"] + idx["validation"]
        assert len(side) == 70 and len(idx["test"]) == 30 and len(idx["validation"]) == round(0.25 * 70)
        assert max(scores[side]) <= min(scores[idx["test"]])
        assert set(side) == set(order[:70].tolist())
        sh = d.meta["shift"]
        assert sh["split_mode"] == "covariate_pc1" and sh["shift_quantile"] == 0.7
        assert sh["threshold"] == pytest.approx(scores[order[69]])
        assert "outer_repeat" not in d.meta and "covariate-shift" in d.meta["protocol"]
        near, far = d.test_groups["shift_near"], d.test_groups["shift_far"]
        assert near.sum() == 15 and far.sum() == 15 and not (near & far).any()
        test_scores = scores[idx["test"]]
        assert test_scores[near].max() <= test_scores[far].min()
        if task == "classification":
            assert sh["validation_stratified"] is True and sh["classes_missing_from_training"] == []
    # the test side is fixed by PC1; the seed only moves rows between training and validation
    assert datas[0].meta["split_indices"]["test"] == datas[1].meta["split_indices"]["test"]
    assert datas[0].meta["split_indices"]["validation"] != datas[1].meta["split_indices"]["validation"]


def test_shift_split_falls_back_to_unstratified_and_records_missing_classes(monkeypatch):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 2)).astype(np.float32)
    order, _ = data_module.covariate_pc1_order(X)
    y = np.zeros(40, dtype=np.int64)
    y[order[-5:]] = 2          # class 2 only on the test side
    y[order[0]] = 1            # class 1 has a single training-side row
    monkeypatch.setattr(data_module, "_load_raw", lambda *a: (X, y, {}))
    d = data_module.make_splits("fake", "classification", 3, test_frac=0.2, val_frac=0.2, **SHIFT)
    assert d.meta["shift"]["validation_stratified"] is False
    assert 2 in d.meta["shift"]["classes_missing_from_training"] and d.num_classes == 3


def test_shift_split_refusals(monkeypatch):
    _fake(monkeypatch)
    kw = dict(test_frac=0.2, val_frac=0.2)
    with pytest.raises(ValueError, match="unknown split_mode"):
        data_module.make_splits("fake", "classification", 0, split_mode="pc2", **kw)
    for q in (0.0, 1.0, True):
        with pytest.raises(ValueError, match="shift_quantile"):
            data_module.make_splits("fake", "classification", 0, split_mode="covariate_pc1", shift_quantile=q, **kw)
    with pytest.raises(ValueError, match="fixed designs"):
        data_module.make_splits("synthetic_diag", "classification", 0, split_mode="covariate_pc1", **kw)


def test_random_split_is_unchanged_by_the_new_keywords():
    kw = dict(test_frac=0.2, val_frac=0.2, maint_frac=0.0, reg_cohort_bins=5)
    a = data_module.make_splits("iris", "classification", 30, **kw)
    b = data_module.make_splits("iris", "classification", 30, split_mode="random", shift_quantile=0.5, **kw)
    assert a.meta == b.meta and a.test_groups == b.test_groups == {}
    for name in ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test"):
        assert torch.equal(getattr(a, name), getattr(b, name))
    assert "shift" not in a.meta


def test_real_shift_split_of_iris():
    d = data_module.make_splits("iris", "classification", 30, test_frac=0.2, val_frac=0.2, **SHIFT)
    assert (len(d.y_train), len(d.y_val), len(d.y_test)) == (78, 27, 45)
    assert d.meta["shift"]["features"] == ["0", "1", "2", "3"]


def test_runner_passes_the_split_mode_only_when_configured(tmp_path, monkeypatch):
    import tools.t1_run as T
    seen = []
    monkeypatch.setattr(T, "make_splits", lambda *a, **kw: seen.append(kw))
    runner = T.Runner.__new__(T.Runner)
    ds = {"name": "iris", "task_type": "classification"}
    base = {"test_frac": 0.2, "val_frac": 0.2, "maint_frac": 0.0, "reg_cohort_bins": 5}
    runner.cfg = {"splits": dict(base)}
    runner.data_for(ds, 30)
    runner.cfg = {"splits": {**base, **SHIFT}}
    runner.data_for(ds, 30)
    assert "split_mode" not in seen[0] and "shift_quantile" not in seen[0]
    assert seen[1]["split_mode"] == "covariate_pc1" and seen[1]["shift_quantile"] == 0.7


# --------------------------------------------------------------------------- grid: selection fixed, evaluate --split-mode


def _grid_study(**over):
    study = dict(development_seeds=[20], evaluation_seeds=[30], screen_epochs=2, screen_patience=1, refit_epochs=3,
                 refit_patience=1, finalists=1, torch_threads=1, splits=dict(test_frac=.2, val_frac=.2),
                 families=["knn"], datasets=[dict(name="iris", task_type="classification", max_train=60)],
                 grids=dict(knn=dict(n_neighbors=[3, 5], weights=["distance"], p=[2])))
    study.update(over)
    return study


def test_fixed_selection_evaluates_the_single_candidate_without_screening(tmp_path, monkeypatch, request):
    import tools.t1_grid_search as search
    threads = torch.get_num_threads()
    request.addfinalizer(lambda: torch.set_num_threads(threads))
    (tmp_path / "runs").mkdir()
    monkeypatch.setenv("NNKNN_RUNS", str(tmp_path / "runs"))
    monkeypatch.setattr(search, "ProcessPoolExecutor", concurrent.futures.ThreadPoolExecutor)
    monkeypatch.setattr(search, "rank_candidates", lambda *a: pytest.fail("fixed selection must not rank"))
    study = _grid_study(selection="fixed", development_seeds=[], evaluation_seeds=[30, 31],
                        grids=dict(knn=dict(n_neighbors=[5], weights=["distance"], p=[2])))
    config = tmp_path / "fixed.yaml"
    config.write_text(yaml.safe_dump(study))
    search.main([str(config), "--out", str(tmp_path / "out"), "--workers", "1", "--study", "F"])
    job = tmp_path / "out" / "classification__iris" / "knn"
    selection = json.loads((job / "selection.json").read_text())
    assert selection["selection"] == "fixed" and selection["development_seeds"] == []
    assert not (job / "shortlist.json").exists()
    done = json.loads((job / "completed.json").read_text())
    assert done["status"] == "completed" and [e["seed"] for e in done["evaluation"]] == [30, 31]
    from model.common.tracking import iter_runs
    stages = {p.name for p in (tmp_path / "runs/t1/F/classification__iris/knn").iterdir()}
    assert stages == {"evaluation"}
    assert len(list(iter_runs(tmp_path / "runs/t1/F"))) == 2
    # more than one candidate, or an unknown selection mode, is refused
    with pytest.raises(ValueError, match="exactly one"):
        search.dataset_family(_grid_study(selection="fixed", source_sha256="x"), study["datasets"][0], "knn",
                              tmp_path / "out2", "F2")
    with pytest.raises(ValueError, match="selection must be"):
        search.fixed_selection({"selection": "best"})
    assert search.fixed_selection({}) is False


def test_evaluate_split_mode_is_opt_in_and_recorded(tmp_path, monkeypatch, request):
    import tools.t1_grid_search as search
    threads = torch.get_num_threads()
    request.addfinalizer(lambda: torch.set_num_threads(threads))
    monkeypatch.delenv("NNKNN_DETERMINISTIC", raising=False)
    (tmp_path / "runs").mkdir()
    monkeypatch.setenv("NNKNN_RUNS", str(tmp_path / "runs"))
    monkeypatch.setattr(search, "ProcessPoolExecutor", concurrent.futures.ThreadPoolExecutor)
    config, grid = tmp_path / "grid.yaml", tmp_path / "grid"
    config.write_text(yaml.safe_dump(_grid_study()))
    search.main([str(config), "--out", str(grid), "--workers", "1", "--study", "G"])
    key = "classification__iris"
    base = ["evaluate", "--grid", str(grid), "--workers", "1", "--families", "knn", "--seeds", "30"]
    search.main(base + ["--out", str(tmp_path / "plain"), "--study", "P"])
    search.main(base + ["--out", str(tmp_path / "shift"), "--study", "S", "--split-mode", "covariate_pc1"])
    assert "splits_override" not in json.loads((tmp_path / "plain/evaluate.json").read_text())
    assert json.loads((tmp_path / "shift/evaluate.json").read_text())["splits_override"] == SHIFT
    plain = records.trial_row("P", key, "knn", json.loads((tmp_path / "plain" / key / "knn/evaluation.json").read_text())["evaluation"][0])
    shifted = records.trial_row("S", key, "knn", json.loads((tmp_path / "shift" / key / "knn/evaluation.json").read_text())["evaluation"][0])
    graded = records.trial_row("G", key, "knn", json.loads((grid / key / "knn/completed.json").read_text())["evaluation"][0])
    assert plain["test_metrics"] == graded["test_metrics"] and "shift" not in plain["dataset"]
    assert shifted["dataset"]["shift"]["split_mode"] == "covariate_pc1"
    assert shifted["dataset"]["split_indices_sha256"] != plain["dataset"]["split_indices_sha256"]
    # per-half test metrics only on the shift split; the random split's rows are unchanged
    assert "test_group_metrics" not in plain and "test_group_metrics" not in graded
    halves = shifted["test_group_metrics"]
    assert set(halves) == {"shift_near", "shift_far"}
    n = [halves[g]["n"] for g in ("shift_near", "shift_far")]
    assert sum(n) == 45 and n[0] == 23
    weighted = sum(halves[g]["n"] * halves[g]["accuracy"] for g in halves) / 45
    assert weighted == pytest.approx(shifted["test_metrics"]["accuracy"])
    # the shifted trial unit carries the split mode in its config (unit identity)
    from model.common.tracking import iter_runs, load_run
    (unit,) = list(iter_runs(tmp_path / "runs/t1/S"))
    assert load_run(unit.path, arrays=False).config["data"]["splits"]["split_mode"] == "covariate_pc1"
    with pytest.raises(ValueError, match="split-mode"):
        search.main(base + ["--out", str(tmp_path / "shift"), "--study", "S"])
    with pytest.raises(SystemExit):
        search.main(base + ["--out", str(tmp_path / "q"), "--study", "S", "--shift-quantile", "0.6"])


def test_group_metrics_per_half():
    import tools.t1_grid_search as search
    y = np.array([0, 0, 1, 1])
    p = np.array([[.9, .1], [.2, .8], [.3, .7], [.6, .4]])
    groups = {"shift_near": np.array([True, True, False, False]), "shift_far": np.array([False, False, True, True]),
              "empty": np.zeros(4, dtype=bool)}
    out = search.group_metrics(p, y, "classification", None, groups)
    # a half with one class has no ROC AUC (it would raise), but accuracy and log loss
    assert out["shift_near"] == pytest.approx(dict(n=2, accuracy=.5, loss=-(np.log(.9) + np.log(.2)) / 2))
    assert "roc_auc" not in out["shift_far"] and out["empty"] == {"n": 0}
    both = search.group_metrics(p, y, "classification", None, {"all": np.ones(4, dtype=bool)})["all"]
    assert both == pytest.approx(dict(n=4, **search.numeric_metrics(p, y, "classification")))
    r = search.group_metrics(np.array([0., 1., 2., 4.]), np.array([0., 0., 2., 2.]), "regression", (0., 2.),
                             {"far": np.array([False, False, True, True])})["far"]
    assert r == pytest.approx(dict(n=2, rmse_standardized=2 ** .5, rmse_original=2 * 2 ** .5, loss=2.))
    with pytest.raises(ValueError, match="align"):
        search.group_metrics(p, y, "classification", None, {"bad": np.ones(3, dtype=bool)})


def test_regression_test_groups_are_reported_only_when_present():
    from model.common.core import CoreConfig, build_model
    from model.t1.reuse import evaluate_regression_reuse, train_regression_adapter, ReuseConfig
    import tools.t1_run as T
    torch.manual_seed(0)
    d = data_module.make_splits("yacht", "regression", 30, test_frac=0.2, val_frac=0.2, **SHIFT)
    plain = data_module.make_splits("yacht", "regression", 30, test_frac=0.2, val_frac=0.2)
    assert plain.test_groups == {} and set(d.test_groups) == {"shift_near", "shift_far"}
    model = build_model(d.X_train, d.y_train, CoreConfig(task_type="regression", top_k=3, seed=1), None)
    m = T.test_metrics(model, d)
    e = T.evaluate(model, d.X_test, d.y_test)
    for g, mask in d.test_groups.items():
        mt = torch.as_tensor(mask)
        assert m[f"test_n_{g}"] == int(mt.sum())
        assert m[f"test_rmse_{g}"] == pytest.approx(float(((e["pred_pre"][mt] - d.y_test[mt]) ** 2).mean()) ** .5)
    n = [m[f"test_n_{g}"] for g in ("shift_near", "shift_far")]
    assert m["test_rmse_pre"] ** 2 == pytest.approx(sum(k * m[f"test_rmse_{g}"] ** 2 for k, g in zip(n, ("shift_near", "shift_far"))) / sum(n), rel=1e-5)
    pm = T.test_metrics(build_model(plain.X_train, plain.y_train, CoreConfig(task_type="regression", top_k=3, seed=1),
                                    None), plain)
    assert not any(k.startswith(("test_n_", "test_rmse_shift")) for k in pm)
    adapter, _ = train_regression_adapter(model, d, ReuseConfig("nominal_residual_scores", "combined", 1., 1., "softmax",
                                                                      hidden_dims=(8, 4), epochs=2, patience=1))
    base = evaluate_regression_reuse(model, adapter, d.X_test, d.y_test, success_tolerance=.5)
    grouped = evaluate_regression_reuse(model, adapter, d.X_test, d.y_test, success_tolerance=.5, test_groups=d.test_groups)
    assert {k: v for k, v in grouped.items() if k in base} == base
    extra = set(grouped) - set(base)
    assert extra == {f"{a}_{g}" for g in d.test_groups for a in ("n", "rmse_pre", "rmse_post")}
    for suffix in ("pre", "post"):
        assert base[f"rmse_{suffix}"] ** 2 == pytest.approx(sum(
            grouped[f"n_{g}"] * grouped[f"rmse_{suffix}_{g}"] ** 2 for g in d.test_groups) / len(d.y_test), rel=1e-5)


# --------------------------------------------------------------------------- TabPFN family


def test_tabpfn_helpers():
    p = np.array([[0.2, 0.8], [0.6, 0.4]])
    assert tp.expand_proba(p, [0, 1], 2) is not None and np.array_equal(tp.expand_proba(p, [0, 1], 2), p)
    assert tp.expand_proba(p, [0, 2], 3).tolist() == [[0.2, 0.0, 0.8], [0.6, 0.0, 0.4]]
    params = {"model_version": "v2", "n_estimators": 4, "max_train_rows": 3000, "max_features": 500}
    tp.check_parameters(params)
    assert tp.in_regime(3000, 500, 10, params) is None
    assert "training rows" in tp.in_regime(3001, 5, 2, params)
    assert "features" in tp.in_regime(10, 501, 2, params)
    assert "classes" in tp.in_regime(10, 5, 11, params)
    for bad in ({**params, "model_version": "v3"}, {**params, "n_estimators": 0}, {**params, "lr": 1}):
        with pytest.raises(ValueError):
            tp.check_parameters(bad)


def test_tabpfn_checkpoint_is_pinned(tmp_path, monkeypatch):
    monkeypatch.setenv("NNKNN_TABPFN_DIR", str(tmp_path))
    assert tp.weights_dir() == tmp_path
    with pytest.raises(FileNotFoundError, match="t1_tabpfn_weights"):
        tp.checkpoint("classification")
    (tmp_path / "tabpfn-v2-classifier.ckpt").write_bytes(b"not the checkpoint")
    with pytest.raises(ValueError, match="pinned"):
        tp.checkpoint("classification")
    monkeypatch.delenv("NNKNN_TABPFN_DIR")
    from datasets.paths import datasets_root
    assert tp.weights_dir() == datasets_root().parent / "models" / "tabpfn"


def test_tabpfn_refuses_a_task_outside_the_regime():
    params = {"model_version": "v2", "n_estimators": 1, "max_train_rows": 5, "max_features": 500}
    data = dict(X_train=torch.zeros(6, 2), y_train=torch.zeros(6, dtype=torch.long), task="classification", classes=2)
    with pytest.raises(ValueError, match="outside the declared TabPFN regime"):
        tp.fit(params, data, 0)


def test_tabpfn_trial_records_its_settings_and_no_weights(tmp_path, monkeypatch, request):
    """A real TabPFN v2 trial on a tiny task through the grid's ``tabpfn`` family (skipped without the package or the
    pinned checkpoints on this host)."""
    pytest.importorskip("tabpfn")
    try:
        tp.checkpoint("classification")
        tp.check_version()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        pytest.skip(str(exc))
    import tools.t1_grid_search as search
    threads = torch.get_num_threads()
    request.addfinalizer(lambda: torch.set_num_threads(threads))
    (tmp_path / "runs").mkdir()
    monkeypatch.setenv("NNKNN_RUNS", str(tmp_path / "runs"))
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64)
    monkeypatch.setattr(data_module, "_load_raw", lambda *a: (X, y, {}))
    study = dict(selection="fixed", development_seeds=[], evaluation_seeds=[50], screen_epochs=0, screen_patience=0,
                 refit_epochs=0, refit_patience=0, finalists=1, torch_threads=1, source_sha256="t",
                 splits=dict(test_frac=.2, val_frac=.2), families=["tabpfn"],
                 grids=dict(tabpfn=dict(model_version=["v2"], n_estimators=[1], max_train_rows=[3000], max_features=[500])))
    result = search.dataset_family(study, dict(name="fake", task_type="classification"), "tabpfn", tmp_path / "out", "T")
    assert result["status"] == "completed"
    row = records.trial_row("T", "classification__fake", "tabpfn", result["evaluation"][0])
    assert set(row["test_metrics"]) == {"accuracy", "loss", "roc_auc"} and row["test_metrics"]["accuracy"] > 0.6
    from model.common.tracking import iter_runs, load_run
    (unit,) = list(iter_runs(tmp_path / "runs/t1/T"))
    state = load_run(unit.path, arrays=False).load_state("model")
    assert state["format"] == "tabpfn_reference" and state["checkpoint_sha256"] == tp.CHECKPOINTS["classification"][2]
    assert state["n_estimators"] == 1 and state["tabpfn_version"] == tp.PACKAGE_VERSION


# --------------------------------------------------------------------------- pack configs


def test_pack_p4_tabpfn_configs():
    from tools.t1_grid_search import expand_grid, fixed_selection
    files = sorted((P4 / "tabpfn").glob("*.yaml"))
    assert len(files) == 6
    covered = set()
    for f in files:
        text = f.read_text(encoding="utf-8")
        assert CHOICE in text
        cfg = yaml.safe_load(text)
        group = "legacy" if "legacy" in f.name else "transfer"
        src = yaml.safe_load((REPO / "configs/t1" / ("grid_search_20261009.yaml" if group == "legacy"
                                                     else "grid_transfer_20261013.yaml")).read_text())
        assert fixed_selection(cfg) and cfg["families"] == ["tabpfn"] and cfg["development_seeds"] == []
        assert len(cfg["evaluation_seeds"]) == 1 and cfg["evaluation_seeds"][0] in (50, 51, 52)
        assert cfg["splits"] == src["splits"]  # the recorded baselines' splits
        (cand,) = expand_grid(cfg["grids"]["tabpfn"])
        tp.check_parameters(cand)
        assert cand == {"model_version": "v2", "n_estimators": 4, "max_train_rows": 3000, "max_features": 500}
        for d in cfg["datasets"]:
            assert d in src["datasets"]
            covered.add((f"{d['task_type']}__{d['name']}", cfg["evaluation_seeds"][0]))
    confirm = {p.stem for p in (REPO / "configs/t1/confirm_final_20261013").glob("*.yaml")} - {"regression__tabarena_363697"}
    tasks = {k for k, _ in covered}
    infeasible = {"classification__tabarena_363681", "regression__tabarena_363705", "regression__tabarena_363631"}
    assert tasks == confirm - {"regression__car", "regression__wine"} - infeasible and len(tasks) == 24
    assert len(covered) == 24 * 3


def test_pack_p4_shift_configs_differ_from_their_sources_only_in_declared_keys():
    from tools.t1_run import load_config, retention_cfg_for
    masked = REPO / "configs/t1/handoff_20261126/p1-allcase/maskedF"
    files = sorted((P4 / "shift" / "shiftF").glob("*.yaml"))
    assert len(files) == 20
    for f in files:
        assert CHOICE in f.read_text(encoding="utf-8")
        new, src = load_config(f), load_config(masked / f.name)
        diff = {k for k in set(new) | set(src) if new.get(k) != src.get(k)}
        assert diff == {"experiment", "status", "seeds", "splits"}
        assert new["seeds"] == [30, 31, 32] and new["experiment"] == "shiftF_20261126"
        assert new["splits"] == {**src["splits"], **SHIFT} and new["type"] == "encoder_ablation"
    tasks = {f.stem for f in files}
    retention = {}
    for f in sorted((P4 / "shift").glob("shift_retention_*.yaml")):
        assert CHOICE in f.read_text(encoding="utf-8")
        cfg = load_config(f)
        assert cfg["type"] == "retention" and cfg["K"] == [0.5] and cfg["seeds"] == [30, 31, 32]
        assert cfg["splits"]["split_mode"] == "covariate_pc1" and cfg["splits"]["shift_quantile"] == 0.7
        assert [c["policy"] for c in cfg["conditions"]] == ["full_memory", "random", "stratified", "bias_current",
                                                             "provenance_only", "bias_normalized", "kcenter",
                                                             "coverage_reachability"]
        for c in cfg["conditions"]:
            retention_cfg_for(c, cfg, 30).validate()
        for d in cfg["datasets"]:
            retention[f"{d['task_type']}__{d['name']}"] = d
            data_module.SPLIT_MODES  # noqa: B018 - the mode is a known one
        assert cfg["splits"]["split_mode"] in data_module.SPLIT_MODES
    assert set(retention) == tasks and len(tasks) == 20
    assert not any(k.startswith("classification__synthetic") for k in retention)
