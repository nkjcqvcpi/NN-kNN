"""Pack p1 of the 2026-11-26 handoff batch (configs/t1/handoff_20261126/p1-allcase/): the revised all-case baseline
declared by us (DECLARATION.md). Every config loads through the runner's loader, keeps 3 seeds, differs from its
source (frozen confirm_final_20261013 F config, or the plan_full core built by tools/t1_execute_study.py) only in the
declared keys, and each declared value reaches the object the runner builds from it. No training, no data; the
configs are new files and no recorded unit's configuration changes.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import yaml

os.environ.setdefault("NNKNN_DEVICE", "cpu")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
P1 = REPO / "configs/t1/handoff_20261126/p1-allcase"
FROZEN = REPO / "configs/t1/confirm_final_20261013"
TASKS = sorted(p.stem for p in (P1 / "allcaseF").glob("*.yaml"))
PLAN = sorted((P1 / "plan").glob("*.yaml"))
SELECTION = json.loads((P1 / "allcase_grid_selection.json").read_text())["tasks"]
CHOICE = "exploratory, our choice, not a PI decision"
# plan_full group -> (source config of tools/t1_execute_study.group, tuned core)
GROUPS = {"retention": ("p1_retention_classification.yaml", True), "reuse_classification": ("p2_reuse.yaml", True),
          "reuse_regression": ("p2_reuse.yaml", True), "synchronization": ("full_sync_ablation_20261009.yaml", True),
          "sync_encoder_control": ("sync_encoder_control_20261009.yaml", False),
          "sync_mcb": ("sync_mcb_interaction_20261009.yaml", False), "mcb_retention": ("p5_mcb.yaml", False),
          "exact_removal_diagnostics": ("p1_retention_classification.yaml", True),
          "removal_retraining_diagnostics": ("alignment_retraining.yaml", True),
          "adapted_retraining_diagnostics": ("alignment_adapted_retraining.yaml", True),
          "nominal_reuse": ("alignment_nominal_reuse.yaml", False), "nominal_sync": ("alignment_nominal_sync.yaml", False),
          "revision_stress": ("alignment_synthetic_reliability.yaml", False)}


def test_pack_has_every_file_and_declares_our_choice():
    assert len(TASKS) == 29 and "regression__tabarena_363697" not in TASKS
    assert TASKS == sorted(p.stem for p in FROZEN.glob("*.yaml") if "363697" not in p.stem)
    assert sorted(p.stem for p in (P1 / "maskedF").glob("*.yaml")) == TASKS
    assert len(PLAN) == 195 and set(SELECTION) == set(TASKS)
    assert all(v["recomputed_from_runs"] for v in SELECTION.values())
    counts = {}
    for p in PLAN:
        counts[p.name.split("__")[0]] = counts.get(p.name.split("__")[0], 0) + 1
    assert counts == {"retention": 21, "reuse_classification": 10, "reuse_regression": 11, "synchronization": 21,
                      "sync_encoder_control": 21, "sync_mcb": 21, "mcb_retention": 21, "exact_removal_diagnostics": 20,
                      "removal_retraining_diagnostics": 20, "adapted_retraining_diagnostics": 20, "nominal_reuse": 1,
                      "nominal_sync": 1, "revision_stress": 7}
    assert CHOICE in (P1 / "DECLARATION.md").read_text(encoding="utf-8")
    for p in [*(P1 / "allcaseF").glob("*.yaml"), *(P1 / "maskedF").glob("*.yaml"), *PLAN]:
        text = p.read_text(encoding="utf-8")
        assert CHOICE in text.split("\nbase:")[0].split("\nstatus:")[0], p
        assert not any("363697" in d["name"] for d in yaml.safe_load(text)["datasets"])


def _f_arm(cfg):
    return [c for c in cfg["conditions"] if c["name"] == "F"]


def _sync(cfg, cond, task):
    from model.t1.sync import SyncConfig
    merged = {**cfg["sync"], **cond}
    merged.setdefault("output_mode", "aggregate_regression" if task == "regression" else "nominal_residual_scores")
    sc = SyncConfig(seed=50, **{k: v for k, v in merged.items() if k in SyncConfig.__dataclass_fields__ and k != "seed"})
    sc.validate(task)
    return sc, merged


@pytest.mark.parametrize("key", TASKS)
def test_allcase_f_applies_the_declaration_and_nothing_else(key):
    from tools.t1_run import load_config, core_cfg_for, sync_maintenance_for
    from tools.t1_grid_search import nnknn_core_settings
    task = key.split("__")[0]
    new, frozen = load_config(P1 / "allcaseF" / f"{key}.yaml"), load_config(FROZEN / f"{key}.yaml")
    assert new["seeds"] == [50, 51, 52] and new["status"] == "exploratory_handoff_20261126_p1_allcaseF"
    assert new["experiment"] == "allcaseF_20261126"
    diff = {k for k in set(new) | set(frozen) if new.get(k) != frozen.get(k)}
    assert diff <= {"experiment", "status", "seeds", "core", "core_by_dataset", "core_selection_source", "conditions"}
    assert {k for k in frozen["core"] if frozen["core"][k] != new["core"][k]} == {"pre_topk_mask"}
    assert new["core"]["pre_topk_mask"] is False
    winner = nnknn_core_settings(SELECTION[key]["parameters"])
    assert new["core_by_dataset"] == {key: winner} and winner["pre_topk_mask"] is False
    cc = core_cfg_for(new, task, 50, **new["core_by_dataset"][key])  # what Runner.core_config builds
    assert cc.pre_topk_mask is False and (cc.epochs, cc.patience) == (300, 40)
    assert all(getattr(cc, k) == v for k, v in winner.items())
    F = _f_arm(frozen)
    if key == "regression__diabets":  # its F is the frozen-core reuse arm
        assert not F and new["conditions"] == [] and new["reuse_conditions"] == frozen["reuse_conditions"]
        return
    assert new["conditions"][0] == F[0]
    sc_f, merged = _sync(new, F[0], task)
    sync_maintenance_for(F[0], new, 50)
    assert merged.get("free_radius_method", "trained_bias_region") == "trained_bias_region"
    if sc_f.v_small > 0:
        assert [c["name"] for c in new["conditions"]] == ["F", "F_zero_radius"]
        zero = new["conditions"][1]
        sync_maintenance_for(zero, new, 50)
        sc_z, merged_z = _sync(new, zero, task)
        assert sc_z == sc_f and merged_z["free_radius_method"] == "zero"
        assert {k for k in set(zero) | set(F[0]) if zero.get(k) != F[0].get(k)} == {"name", "free_radius_method"}
    else:
        assert [c["name"] for c in new["conditions"]] == ["F"]


@pytest.mark.parametrize("key", TASKS)
def test_masked_f_is_the_frozen_f_arm(key):
    from tools.t1_run import load_config, core_cfg_for
    task = key.split("__")[0]
    new, frozen = load_config(P1 / "maskedF" / f"{key}.yaml"), load_config(FROZEN / f"{key}.yaml")
    assert new["seeds"] == [50, 51, 52] and new["experiment"] == "maskedF_20261126"
    diff = {k for k in set(new) | set(frozen) if new.get(k) != frozen.get(k)}
    assert diff == {"experiment", "status", "seeds", "conditions"}
    assert new["conditions"] == _f_arm(frozen)
    assert core_cfg_for(new, task, 50, **new["core_by_dataset"][key]).pre_topk_mask is True
    allcase = load_config(P1 / "allcaseF" / f"{key}.yaml")
    assert allcase["datasets"] == new["datasets"] and allcase["splits"] == new["splits"]  # same data, same seeds


@pytest.mark.parametrize("path", PLAN, ids=lambda p: p.stem)
def test_plan_twin_switches_only_the_mask_and_the_tuned_core(path):
    from tools.t1_run import load_config, core_cfg_for, retention_cfg_for, sync_maintenance_for
    from tools.t1_grid_search import nnknn_core_settings
    cfg = load_config(path)
    group = path.name.split("__")[0]
    source, tuned = GROUPS[group]
    assert cfg["status"] == "exploratory_handoff_20261126_p1_allcaseplan" and cfg["seeds"] == [30, 31, 32]
    assert cfg["experiment"] == f"{group}__{cfg['datasets'][0]['task_type']}"
    plan_core = {**load_config(REPO / "configs/t1" / source)["core"], "epochs": 300, "patience": 40}
    assert cfg["core"] == {**plan_core, "pre_topk_mask": False}
    ds = cfg["datasets"][0]
    key = f"{ds['task_type']}__{ds['name']}"
    assert ("core_by_dataset" in cfg) is tuned
    if tuned:
        assert cfg["core_by_dataset"] == {k: nnknn_core_settings(SELECTION[k]["parameters"]) for k in cfg["core_by_dataset"]}
    cc = core_cfg_for(cfg, ds["task_type"], 30, **cfg.get("core_by_dataset", {}).get(key, {}))
    assert cc.pre_topk_mask is False
    for arm in cfg.get("arms") or []:
        assert "pre_topk_mask" not in arm
    if cfg["type"] == "sync":
        for cond in cfg["conditions"]:
            sync_maintenance_for(cond, cfg, 30)
            _sync(cfg, cond, ds["task_type"])
    if cfg["type"] == "retention":
        for cond in cfg["conditions"]:
            retention_cfg_for(cond, cfg, 30).validate()


def test_declared_mask_and_penalty_reach_the_model_and_the_calibration():
    """The all-case core builds a model without the pre-Top-K mask, and the declared variant's free_radius_method
    'zero' gives an allowance of 0 (squared L2 from zero) where the default region gives a positive one."""
    import torch
    from tools.t1_run import load_config, core_cfg_for
    from model.common.core import build_model
    from model.t1.calibration import calibrate_free_radius
    cfg = load_config(P1 / "allcaseF" / "regression__student_performance.yaml")
    over = dict(cfg["core_by_dataset"]["regression__student_performance"], representation="raw")
    cc = core_cfg_for(cfg, "regression", 50, **over)
    g = torch.Generator().manual_seed(0)
    X = torch.randn(40, 3, generator=g)
    y = X[:, :1] + 0.1 * torch.randn(40, 1, generator=g)
    model = build_model(X, y, cc, None)
    assert model.config["pre_topk_mask"] is False
    default, _ = calibrate_free_radius(model, s_task=0.5, snapshot_step=0, method="trained_bias_region")
    zero, _ = calibrate_free_radius(model, s_task=0.5, snapshot_step=0, method="zero")
    assert zero.tau_task == 0.0 and default.tau_task > 0.0
