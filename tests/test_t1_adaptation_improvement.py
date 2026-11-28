"""Paired local experiments: shared core/RNG, recording and honest summaries."""
import copy
import shutil
import pytest
import torch

from model.common import tracking
from model.t1 import records
from model.t1.data import T1Data
import tools.t1_adaptation_improvement as comparison
import tools.t1_run as t1_run


def test_suite_preserves_task_selected_schedule_and_loss_weights():
    configs = comparison.suite()
    assert len(configs) == 29
    assert "regression__tabarena_363697" not in configs
    assert sum(c["datasets"][0]["task_type"] == "classification" for c in configs.values()) == 14
    for task, cfg in configs.items():
        source = t1_run.load_config(comparison.DEFAULT_CONFIGS / f"{task}.yaml")
        frozen = task == "regression__diabets"
        assert cfg["adapter_comparison"]["frozen_reuse"] == frozen
        field = "reuse_conditions" if frozen else "conditions"
        old, new = cfg[field]
        original = source["reuse_conditions"][0] if frozen else next(c for c in source["conditions"] if c.get("name") == "F")
        for key, value in original.items():
            if key != "name":
                assert old[key] == new[key] == value
        assert new["adapter_design"] == "magnitude_direction"
        assert new["initial_scale"] == .01
        for key in ("core", "core_by_dataset", "splits", "sync", "reuse"):
            assert cfg[key] == source[key]


def toy_data(task_type, name="toy"):
    g = torch.Generator().manual_seed(73)
    x = torch.randn(40, 3, generator=g)
    y = (x[:, 0] > 0).long() if task_type == "classification" else .8 * x[:, 0] - .2 * x[:, 1]
    return T1Data(name, task_type, x[:24], y[:24], x[24:32], y[24:32], x[32:], y[32:],
        num_classes=2 if task_type == "classification" else None,
        meta={"source": "synthetic_test", "split_indices": {"train": list(range(24)),
              "validation": list(range(24, 32)), "test": list(range(32, 40))}})


def test_preflight_fails_before_any_runner_or_training(monkeypatch):
    configs = comparison.suite(tasks=["classification__iris", "regression__yacht"])
    def load(self, ds, seed):
        if ds["task_type"] == "regression":
            raise FileNotFoundError("missing local data")
        return toy_data("classification", ds["name"])
    monkeypatch.setattr(t1_run.Runner, "data_for", load)
    def forbidden(*args, **kwargs):
        pytest.fail("preflight must not construct a runner or train")
    monkeypatch.setattr(comparison.ComparisonRunner, "__init__", forbidden)
    monkeypatch.setattr(t1_run, "train_retrieval", forbidden)
    with pytest.raises(ValueError, match="no training was started"):
        comparison.preflight(configs, [40])


def test_fingerprint_detects_validation_features_and_targets():
    data = toy_data("regression")
    before = comparison.data_fingerprint(data)
    changed = copy.deepcopy(data)
    changed.X_val[0, 0] += .1
    assert comparison.data_fingerprint(changed) != before
    changed = copy.deepcopy(data)
    changed.y_test[0] += .1
    assert comparison.data_fingerprint(changed) != before


@pytest.mark.parametrize("task", ["classification__iris", "regression__yacht", "regression__diabets"])
def test_local_pair_records_once_and_resumes_without_core_training(task, tmp_path, monkeypatch):
    runs = tmp_path / "r"
    runs.mkdir()
    monkeypatch.setenv("NNKNN_RUNS", str(runs))
    monkeypatch.setenv("NNKNN_DEVICE", "cpu")
    cfg = comparison.suite(tasks=[task])[task]
    cfg["core_by_dataset"] = {}
    cfg["core"].update(epochs=2, patience=2, batch_size=8, top_k=3, bias_init_k=3, glocal_fw_set_num=1)
    cfg["sync"].update(epochs=3, patience=3, block_epochs=1, hidden_dims=[8, 4])
    cfg["reuse"].update(epochs=3, patience=3, batch_size=8, hidden_dims=[8, 4])
    for cond in cfg["conditions"]:
        cond["adapter_warmup_epochs"] = 1
    ds = cfg["datasets"][0]
    monkeypatch.setattr(t1_run.Runner, "data_for", lambda self, entry, seed: toy_data(entry["task_type"], entry["name"]))
    fingerprints = comparison.preflight({task: cfg}, [40])[task]
    runner = comparison.ComparisonRunner(cfg, tmp_path / "inputs", "pair", fingerprints)
    starts = []
    method = "reuse_condition" if cfg["adapter_comparison"]["frozen_reuse"] else "sync_condition"
    original = getattr(runner, method)
    def record_start(*args, **kwargs):
        starts.append((tracking.state_digest(args[3].state_dict()),
                       tracking.state_digest(args[5].optimizer.state_dict()), torch.get_rng_state().clone()))
        return original(*args, **kwargs)
    monkeypatch.setattr(runner, method, record_start)
    runner.run_pair(ds, 40)
    assert starts[0][:2] == starts[1][:2]
    assert torch.equal(starts[0][2], starts[1][2])
    first = comparison.summarize("pair", {task: cfg}, [40])
    assert first["complete_seed_pairs"] == 1
    assert not first["missing_units"]
    assert first["groups"][ds["task_type"]]["n_tasks"] == 1
    _, adaptations = runner.pair_units(ds)
    new_label, new_cond = adaptations[1][1]
    new_key = runner.unit_key(ds, new_label, 40)
    new_unit = tracking.load_unit(new_key, config=runner.unit_config(runner.run_config(ds, 40, new_cond)))
    new_row = records.record(new_unit, "row")
    new_path = new_unit.path.resolve()
    assert new_path.is_relative_to(tmp_path.resolve())
    shutil.rmtree(new_path)
    def forbidden(*args, **kwargs):
        pytest.fail("the completed shared retrieval core must never be trained again")
    monkeypatch.setattr(runner, "trained_core", forbidden)
    monkeypatch.setattr(runner, "observed_core", forbidden)
    runner.run_pair(ds, 40)
    assert starts[0][:2] == starts[2][:2]
    assert torch.equal(starts[0][2], starts[2][2])
    restored = tracking.load_unit(new_key, config=runner.unit_config(runner.run_config(ds, 40, new_cond)))
    restored_row = records.record(restored, "row")
    assert {k: v for k, v in new_row.items() if k.startswith(("test_", "val_"))} == {
        k: v for k, v in restored_row.items() if k.startswith(("test_", "val_"))}
    monkeypatch.setattr(runner, "data_for", forbidden)
    monkeypatch.setattr(runner, "core_stage", forbidden)
    monkeypatch.setattr(runner, "sync_condition", forbidden)
    monkeypatch.setattr(runner, "reuse_condition", forbidden)
    runner.run_pair(ds, 40)  # finished study neither loads data nor trains
    runner.fingerprints = {40: "changed-data"}
    with pytest.raises(tracking.ConfigMismatchError):
        runner.run_pair(ds, 40)


def test_summary_separates_families_and_explicitly_handles_zero_errors():
    def pair(task, kind, old, new):
        return {"task": task, "task_type": kind, "existing_error": old, "new_error": new,
                "retrieval_error": old, "existing_pre_error": old, "new_pre_error": old}
    result = comparison.aggregate_pairs([
        pair("classification__a", "classification", 0, 0),
        pair("classification__b", "classification", .2, .1),
        pair("regression__a", "regression", .5, 1),
    ])
    classification = result["groups"]["classification"]
    assert classification["wins"] == classification["ties"] == 1
    assert classification["gmr_positive_task_means"] == pytest.approx(.5)
    assert classification["gmr_excluded_zero_error_tasks"] == ["classification__a"]
    assert result["groups"]["regression"]["losses"] == 1
    assert result["groups"]["regression"]["gmr_positive_task_means"] == pytest.approx(2)


def test_summary_missing_units_does_not_train(tmp_path, monkeypatch):
    runs = tmp_path / "r"
    runs.mkdir()
    monkeypatch.setenv("NNKNN_RUNS", str(runs))
    monkeypatch.setattr(t1_run.Runner, "data_for", lambda *a: pytest.fail("summary loaded data"))
    report = comparison.summarize("missing", comparison.suite(tasks=["classification__iris"]), [40])
    assert report["complete_seed_pairs"] == 0
    assert len(report["missing_units"]) == 3
