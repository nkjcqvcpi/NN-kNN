"""Plan-required calibration replay and component-isolated optimization."""
from pathlib import Path

import pytest
import torch

from model.common.core import CoreConfig, build_model
from model.t1.calibration import calibrate_free_radius
from model.t1.data import T1Data
from model.t1.sync import SyncConfig, train_synchronized
from tools.t1_run import Runner, load_config


def toy():
    x = torch.tensor([[0.], [.1], [.3], [.8], [.9], [1.2]])
    y = torch.tensor([0., .1, .2, .8, 1., 1.2])
    return T1Data(name="tiny", task_type="regression", X_train=x, y_train=y,
                  X_val=x + .01, y_val=y, X_test=x + .02, y_test=y)


@pytest.mark.parametrize("method", ["trained_bias_region", "zero", "initial_bias_region", "exact_kth_pair"])
def test_calibration_saved_pairs_reproduce_threshold(method):
    data = toy()
    cc = CoreConfig(task_type="regression", top_k=3, bias_init_k=3)
    model = build_model(data.X_train, data.y_train, cc, None)
    fr, snap = calibrate_free_radius(model, s_task=.5, snapshot_step=7, method=method,
                                     initial_bias=model.biases.detach().clone(), neighbor_k=3)
    mask = snap["pair_mask"]
    assert not mask.diag().any()
    if method in {"trained_bias_region", "initial_bias_region"}:
        reconstructed = snap["bias_snapshot"][None] - snap["distance_snapshot"] >= 0
        reconstructed.fill_diagonal_(False)
        assert torch.equal(reconstructed, mask)
    labels = snap["labels_snapshot"].flatten()
    dy = (labels[:, None] - labels[None, :]).abs()
    expected = float(dy[mask].mean()) if mask.any() else 0.
    assert fr.tau_task == pytest.approx(expected)
    assert fr.method == method
    frozen_labels = snap["labels_snapshot"].clone()
    with torch.no_grad():
        model.labels.add_(10.)
    assert torch.equal(snap["labels_snapshot"], frozen_labels)
    if method == "exact_kth_pair":
        assert mask.sum() == len(labels)


def test_updates_are_isolated_and_all_terms_are_logged():
    data = toy()
    cc = CoreConfig(task_type="regression", batch_size=3, top_k=3)
    model = build_model(data.X_train, data.y_train, cc, None)
    fr, _ = calibrate_free_radius(model, s_task=.5, snapshot_step=0)
    cfg = SyncConfig("alternating_rr", 1., .5, .1, 1., 1., .1,
                     output_mode="aggregate_regression", epochs=4, block_epochs=1, patience=4)
    _, info = train_synchronized(model, data, cc, cfg, fr, near_scale=1.)
    required = {"L_pre", "L_post", "L_near", "L_delta", "L_small", "correction_norm"}
    for row in info["history"]:
        assert required == row["train_before_update"].keys()
        assert required == row["train_after_update"].keys() == row["validation_terms"].keys()
        if row["phase"] == "adapter":
            # Adapter-only updates cannot change retrieval loss/locality.
            for key in ("L_pre", "L_near"):
                assert row["train_after_update"][key] == pytest.approx(row["train_before_update"][key], abs=1e-8)
            assert row["train_after_update"]["L_post"] != row["train_before_update"]["L_post"]


def test_runner_saves_post_core_and_maintenance_calibration(tmp_path):
    cfg = load_config(Path("configs/t1/alignment_sync_plan_20261009.yaml"))
    cfg["core"].update(epochs=2, patience=2, batch_size=3, top_k=3, bias_init_k=3)
    cfg["sync"].update(epochs=4, block_epochs=1, maintenance_epochs=[2], patience=4, K=.75)
    cfg["conditions"] = [{"name": "rrr_primary", "schedule": "alternating_rrr"}]
    cfg["retention"]["min_per_cohort"] = 0
    runner = Runner(cfg, tmp_path)
    data = toy()
    runner.data_for = lambda ds, seed: data
    runner.run_sync({"name": "tiny", "task_type": "regression"}, 8)
    path = next(tmp_path.rglob("calibration_snapshots.pt"))
    snapshots = torch.load(path, weights_only=True)
    assert len(snapshots) == 2
    assert [s["active_case_count"] for s in snapshots] == [6, 4]
    for snap in snapshots:
        model = build_model(data.X_train, data.y_train, CoreConfig(**cfg["core"], task_type="regression", seed=8), None)
        model.set_active_case_count(snap["active_case_count"])
        model.load_state_dict(snap["model_state"])
        fr, replay = calibrate_free_radius(model, s_task=.5, snapshot_step=snap["free_radius"]["snapshot_step"])
        assert fr.to_dict() == snap["free_radius"]
        assert torch.equal(replay["distance_snapshot"], snap["distance_snapshot"])
        assert torch.equal(replay["case_ids"], snap["case_ids"])


@pytest.mark.parametrize("transform", ["identity", "shrunk", "clipped", "cohort_normalized"])
@pytest.mark.parametrize("aggregation", ["mean", "median"])
def test_declared_calibration_diagnostics_leave_model_unchanged(transform, aggregation):
    data = toy()
    model = build_model(data.X_train, data.y_train, CoreConfig(task_type="regression", top_k=3), None)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    fr, snap = calibrate_free_radius(model, s_task=.5, snapshot_step=2,
                                    aggregation=aggregation, bias_transform=transform)
    labels = snap["labels_snapshot"].flatten()
    dy = (labels[:, None] - labels[None, :]).abs()[snap["pair_mask"]]
    expected = float(dy.mean() if aggregation == "mean" else dy.median()) if dy.numel() else 0.
    assert fr.tau_task == pytest.approx(expected)
    assert all(torch.equal(before[k], model.state_dict()[k]) for k in before)


def test_large_snapshot_preserves_reconstruction_hashes():
    import hashlib
    from model.t1.calibration import compact_calibration_snapshot
    data = toy()
    model = build_model(data.X_train, data.y_train, CoreConfig(task_type="regression", top_k=3), None)
    _, snapshot = calibrate_free_radius(model, s_task=.5, snapshot_step=0)
    snapshot.update(model_state=model.state_dict(), active_case_count=model.case_count())
    compact = compact_calibration_snapshot(snapshot, max_matrix_bytes=0)
    assert compact["matrix_storage"] == "reconstruct_from_frozen_model"
    assert "distance_snapshot" not in compact and "pair_mask" not in compact
    for key in ("distance_snapshot", "pair_mask"):
        assert compact["matrix_evidence"][key]["sha256"] == hashlib.sha256(memoryview(snapshot[key].numpy())).hexdigest()
    assert compact["active_case_count"] == model.case_count()
