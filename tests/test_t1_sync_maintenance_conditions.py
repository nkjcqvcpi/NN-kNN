"""Per-condition maintenance in synchronized training (F7): policy, K and epochs per condition."""
import json
import warnings
from pathlib import Path

import pytest

from tools.t1_run import Runner, load_config, retention_cfg_for, sync_maintenance_for

IRIS = {"name": "iris", "task_type": "classification", "max_train": 30}


def tiny_cfg():
    cfg = load_config(Path("configs/t1/alignment_sync_plan_20261009.yaml"))
    cfg["core"].update(epochs=3, patience=3, batch_size=8, top_k=3, bias_init_k=3)
    cfg["sync"].update(epochs=4, block_epochs=1, patience=4, maintenance_epochs=[2], K=.75)
    cfg["datasets"] = [IRIS]
    return cfg


def read_runs(root):
    """{condition: (row, manifest, events, history)} of one tiny iris run."""
    rows = {r["condition"]: r for r in map(json.loads, next(root.rglob("runs__iris.jsonl")).read_text().splitlines())}
    out = {}
    for label, row in rows.items():
        run_dir = next(root.rglob(f"iris/{label}/s8"))
        events = [json.loads(line) for line in (run_dir / "case_maintenance.jsonl").read_text().splitlines()]
        out[label] = (row, json.loads((run_dir / "run_manifest.json").read_text()), events,
                      json.loads((run_dir / "history.json").read_text())["sync"])
    return out


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    cfg = tiny_cfg()
    cfg["conditions"] = [{"name": "prov", "schedule": "joint_rrr", "policy": "provenance_only"},
                         {"name": "rand", "schedule": "joint_rrr", "policy": "random"},
                         {"name": "k05", "schedule": "joint_rrr", "K": .5, "maintenance_epochs": [3]}]
    root = tmp_path_factory.mktemp("sync_conditions")
    Runner(cfg, root).run_sync(IRIS, 8)
    return read_runs(root)


def test_condition_policy_reaches_the_rrr_hook(runs):
    archived = {}
    for label, policy in (("prov", "provenance_only"), ("rand", "random")):
        row, manifest, events, _ = runs[label]
        assert events and {e["policy_version"].split("/")[0] for e in events} == {policy}
        assert manifest["maintenance"]["policy"] == policy
        assert manifest["maintenance"]["in_training"]["retention"]["policy"] == policy
        assert row["policy"] == policy and row["n_cases"] == 22
        archived[label] = {e["case_id"] for e in events if e["action"] == "archive"}
    assert len(archived["prov"]) == len(archived["rand"]) == 8
    assert archived["prov"] != archived["rand"]


def test_condition_K_and_maintenance_epochs_take_effect(runs):
    for label, K, epoch in (("prov", 22, 2), ("k05", 15, 3)):
        row, manifest, events, history = runs[label]
        assert {e["step"] for e in events} == {epoch}
        assert {e["capacity_after"] for e in events} == {K} and row["n_cases"] == K
        assert [h["epoch"] for h in history if "maintenance" in h] == [epoch]
        assert manifest["maintenance"]["in_training"]["K"] == K
        assert manifest["maintenance"]["in_training"]["maintenance_epochs"] == [epoch]
    # the default condition inherits retention and keeps its row free of maintenance keys
    assert runs["k05"][1]["maintenance"]["in_training"]["retention"]["policy"] == "provenance_only"
    assert "maintenance_epochs" not in runs["k05"][0] and "K" not in runs["prov"][0]


@pytest.mark.parametrize("key", ["polciy", "seed", "probability_mode", "capacity_match"])
def test_unread_condition_key_raises_before_training(tmp_path, key):
    cfg = tiny_cfg()
    cfg["conditions"] = [{"schedule": "joint_rrr"}, {"name": "typo", "schedule": "joint_rrr", key: "random"}]
    runner = Runner(cfg, tmp_path)
    runner.data_for = lambda ds, seed: pytest.fail("trained before checking conditions")
    with pytest.raises(ValueError, match=rf"'typo'.*unknown key\(s\) \['{key}'\]"):
        runner.run_sync(IRIS, 8)


def test_bad_policy_raises_and_inert_keys_raise():
    cfg = tiny_cfg()
    with pytest.raises(ValueError, match="Unknown policy"):
        sync_maintenance_for({"schedule": "alternating_rrr", "policy": "radnom"}, cfg, 0)
    with pytest.raises(ValueError, match=r"\['K', 'policy'\] have no effect"):
        sync_maintenance_for({"schedule": "joint_rr", "policy": "random", "K": .5}, cfg, 0)
    retention, K, epochs = sync_maintenance_for({"schedule": "joint_rr"}, cfg, 0)
    assert retention is None and K == .75 and epochs == {2}


def test_existing_sync_conditions_resolve_as_before():
    """Every sync condition under configs/t1 keeps the hook's earlier settings: the global
    retention block, capacity_match.K else sync.K else 1.0, and sync.maintenance_epochs."""
    checked = rrr = 0
    for path in sorted(Path("configs/t1").rglob("*.yaml")):
        cfg = load_config(path)
        if cfg.get("type") not in {"sync", "encoder_ablation"} or not cfg.get("conditions"):
            continue
        capacity_match = cfg.get("capacity_match")
        legacy_K = capacity_match["K"] if capacity_match else cfg["sync"].get("K", 1.0)
        legacy_epochs = set(cfg["sync"].get("maintenance_epochs", []))
        for cond in cfg["conditions"]:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                retention, K, epochs = sync_maintenance_for(cond, cfg, 3)
            assert (K, epochs) == (legacy_K, legacy_epochs), (path, cond)
            if "rrr" in {**cfg["sync"], **cond}["schedule"]:
                assert retention == retention_cfg_for({}, cfg, 3), (path, cond)
                rrr += 1
            else:
                assert retention is None, (path, cond)
            checked += 1
    assert checked > 300 and rrr >= 10
