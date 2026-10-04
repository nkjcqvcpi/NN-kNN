"""Synthetic factors must preserve paired observations and truthful annotations."""
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from model.t1.data import make_synthetic, make_splits
from model.t1.core import CoreConfig, build_model
from model.t1.revise import InterventionLog, evaluate_m0_m1_m2


@pytest.mark.parametrize("seed,digest", [
    (5, "a4bd42c71027060c74f524f87a3321b249bc5ab2bc276be3f84946c7cba514fe"),
    (6, "8fdc6ab17717d28c7e668138f560ce92c127cd17e496a179085d84367ca6f639"),
    (7, "e567211d8135dc218b34daa59dd0f34f5e6fe46e40da9ca91698841c07356cd6")])
def test_legacy_tensor_bytes_preserved(seed, digest):
    d = make_synthetic("synthetic_diag", seed)
    actual = hashlib.sha256(b"".join(t.numpy().tobytes() for t in
        [d.X_train, d.y_train, d.X_val, d.y_val, d.X_test, d.y_test])).hexdigest()
    assert actual == digest


def test_corruption_does_not_change_any_features_or_clean_holdout():
    a = make_synthetic("synthetic_noise", 8, generator_version="independent_v1", corruption_rate=0.)
    b = make_synthetic("synthetic_noise", 8, generator_version="independent_v1", corruption_rate=.4)
    for name in ("X_train", "X_val", "y_val", "X_test", "y_test"):
        assert torch.equal(getattr(a, name), getattr(b, name))
    assert a.meta["duplicate_source_ids"] == b.meta["duplicate_source_ids"]
    assert np.array_equal(b.case_truth["corrupted"], b.y_train.numpy() != b.meta["true_labels"])
    assert b.case_truth["corrupted"][360:420].any()  # inherited noisy labels remain marked
    assert np.array_equal(a.meta["true_labels"], b.meta["true_labels"])


def test_training_coverage_factors_preserve_raw_holdout():
    from sklearn.preprocessing import StandardScaler
    a = make_synthetic("synthetic", 9, generator_version="independent_v1", duplicate_count=0, rare_count=0)
    b = make_synthetic("synthetic", 9, generator_version="independent_v1", duplicate_count=120, rare_count=24)
    # Different train-only scalers are expected. Undo their affine transformations
    # through corresponding base rows, then verify validation/test are unchanged.
    xa, xb = a.X_train[:360].numpy(), b.X_train[:360].numpy()
    slope = xa.std(0) / xb.std(0)
    intercept = xa.mean(0) - xb.mean(0) * slope
    for name in ("X_val", "X_test"):
        np.testing.assert_allclose(getattr(a, name).numpy(), getattr(b, name).numpy()*slope+intercept, atol=2e-6)
    assert torch.equal(a.y_test, b.y_test)
    assert a.case_truth["rare"].sum() == 0
    assert a.test_groups["rare"].sum() == 20
    assert not (a.test_groups["rare"] & a.test_groups["boundary"]).any()
    assert all(len(mask) == len(b.y_train) for mask in b.case_truth.values())


@pytest.mark.parametrize("kwargs", [dict(corruption_rate=1.1), dict(within_std=0),
    dict(center_scale=float("nan")), dict(duplicate_count=400), dict(rare_count=-1),
    dict(duplicate_count=1.5), dict(generator_version="mystery"), dict(shift=float("inf"))])
def test_invalid_factors_fail_before_generation(kwargs):
    with pytest.raises(ValueError):
        make_synthetic("synthetic", 0, **kwargs)


def test_runner_routes_parameters_and_saves_safe_snapshot(tmp_path):
    from tools.t1_run import Runner, load_config, REPO
    cfg = load_config(REPO / "configs/t1/alignment_revision_matched.yaml")
    runner = Runner(cfg, tmp_path)
    ds = dict(name="synthetic_zero", task_type="classification", synthetic=dict(
        generator_version="independent_v1", corruption_rate=0, duplicate_count=0, rare_count=0))
    data = runner.data_for(ds, 3)
    cc = CoreConfig(task_type="classification")
    model = build_model(data.X_train, data.y_train, cc, data.num_classes)
    runner.finish(ds=ds, seed=3, cond_label="snapshot", cond={}, data=data, model=model,
        store=None, archive=None, metrics={}, history=[], events=[], components={}, budgets={},
        maintenance={}, model_desc={"core": cc.__dict__})
    snapshot = torch.load(runner.out / ds["name"] / "data_snapshot_s3.pt", weights_only=True)
    assert len(snapshot["y_train"]) == 360
    assert snapshot["meta"]["synthetic_parameters"]["corruption_rate"] == 0
    assert not snapshot["case_truth"]["corrupted"].any()
    assert snapshot["test_groups"]["rare"].sum() == 20


def test_subgroups_come_from_each_actual_stage_prediction():
    X = torch.tensor([[0.], [.1], [2.], [2.1]])
    y = torch.tensor([0, 0, 1, 1])
    data = SimpleNamespace(X_train=X, y_train=y, X_val=X, y_val=y,
        X_test=X[:2]+.03, y_test=y[:2], test_groups={"target": [True, True], "empty": [False, False]})
    cc = CoreConfig(task_type="classification", top_k=1)
    model = build_model(X, y, cc, 2)
    log = InterventionLog("subgroups")
    def edit(m, lg):
        lg.relabel(m, 0, 1, actor="test", reason="deliberate local change")
        lg.relabel(m, 1, 1, actor="test", reason="deliberate local change")
        return [0, 1]
    r = evaluate_m0_m1_m2(model, data, cc, edit, log, retrain_epochs=0)
    assert r["M0"]["accuracy_target"] == 1
    assert r["M1"]["accuracy_target"] == r["M2"]["accuracy_target"] == 0
    for stage in ("M0", "M1", "M2"):
        assert r[stage]["n_empty"] == 0
        assert "accuracy_empty" not in r[stage]
