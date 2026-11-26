"""Pack p0 of the 2026-11-26 handoff batch (configs/t1/handoff_20261126/p0/): every config loads through the
runner's loader, keeps 3 seeds, and each value it changes reaches the object the runner builds from it (no training,
no data). The configs are new files; no recorded unit's configuration changes.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import yaml

os.environ.setdefault("NNKNN_DEVICE", "cpu")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
P0 = REPO / "configs/t1/handoff_20261126/p0"
T1_CONFIGS = sorted(p for p in P0.glob("*.yaml") if not p.name.startswith("pretopk_"))


def test_pack_has_every_config_and_declares_our_choice():
    names = {p.name for p in P0.glob("*.yaml")}
    assert {"synthetic_protect_rare10.yaml", "reuse_sweep_classification.yaml", "reuse_sweep_regression.yaml",
            "sync_sens_classification.yaml", "sync_sens_regression.yaml", "pretopk_bik3.yaml", "pretopk_bik10.yaml",
            "pretopk_bik20.yaml"} <= names
    assert len([n for n in names if n.startswith("mcbmom")]) == 8
    for p in P0.glob("*.yaml"):
        header = p.read_text(encoding="utf-8").split("\nstatus:")[0]
        assert "exploratory, our choice, not a PI decision" in header, p.name
        assert yaml.safe_load(p.read_text())["status"] == "exploratory_handoff_20261126_p0"


@pytest.mark.parametrize("path", T1_CONFIGS, ids=lambda p: p.stem)
def test_t1_config_values_reach_the_runner_objects(path):
    from tools.t1_run import load_config, retention_cfg_for, sync_maintenance_for, core_cfg_for, SYNC_RUNNER_KEYS
    from model.t1.reuse import ReuseConfig
    from model.t1.sync import SyncConfig
    cfg = load_config(path)
    assert cfg["seeds"] == [30, 31, 32]
    assert not any("363697" in d["name"] for d in cfg["datasets"])
    assert cfg["core"]["pre_topk_mask"] is True
    labels = []
    for cond in cfg["conditions"]:
        if cfg["type"] == "retention":
            r = retention_cfg_for(cond, cfg, 30)
            r.validate()
            assert r.protect_rare_cohort_below == 10 and r.boundary_protect_fraction == 0.0
            labels += [f"{cond['policy']}_K{k}" for k in cfg["K"]]
        elif cfg["type"] == "reuse":
            rc = ReuseConfig(seed=30, **{**cfg["reuse"], **{k: v for k, v in cond.items() if k != "name"}})
            rc.validate()
            assert all(getattr(rc, k) == v for k, v in cond.items() if k != "name")
            assert (rc.lambda_diff, rc.lambda_cls, rc.probability_mode) != (1.0, 1.0, "softmax")
            labels.append(cond["name"])
        elif cfg["type"] == "sync":
            sync_maintenance_for(cond, cfg, 30)
            merged = {**cfg["sync"], **cond}
            for task in {d["task_type"] for d in cfg["datasets"]}:
                merged.setdefault("output_mode", "aggregate_regression" if task == "regression" else "nominal_residual_scores")
                SyncConfig(seed=30, **{k: v for k, v in merged.items()
                                       if k in SyncConfig.__dataclass_fields__ and k != "seed"}).validate(task)
            assert all(k in SyncConfig.__dataclass_fields__ or k in SYNC_RUNNER_KEYS for k in cond)
            labels.append(cond.get("name", cond["schedule"]))
        else:
            assert cfg["type"] == "mcb" and cond["mcb"] is True
            cc = core_cfg_for(cfg, cfg["datasets"][0]["task_type"], 30, mcb_enabled=True)
            assert cc.mcb_momentum in (0.9, 0.999)
            labels.append(f"{cond['policy']}_K{cond['K']}")
    assert len(labels) == len(set(labels))
    if cfg["experiment"].startswith("mcbmom"):
        assert cfg["core"]["mcb_momentum"] in (0.9, 0.999)
    if cfg["experiment"].startswith("sync_sens"):
        changed = [k for c in cfg["conditions"] for k in c if k not in ("name", "schedule", "free_radius_method")]
        assert {"s_task", "radius_aggregation", "bias_transform", "core_pretrain"} <= set(changed)


@pytest.mark.parametrize("k", [3, 10, 20])
def test_pretopk_configs_change_only_bias_init_k(k):
    from model.common.core import CoreConfig
    new = yaml.safe_load((P0 / f"pretopk_bik{k}.yaml").read_text())
    old = yaml.safe_load((REPO / "configs/t1/pre_topk_expanded_20261030.yaml").read_text())
    assert {key: v for key, v in new.items() if key not in ("status", "core")} == \
           {key: v for key, v in old.items() if key not in ("status", "core")}
    assert {key for key in old["core"] if old["core"][key] != new["core"][key]} == {"bias_init_k"}
    assert CoreConfig(task_type="regression", seed=70, pre_topk_mask=False, **new["core"]).bias_init_k == k
