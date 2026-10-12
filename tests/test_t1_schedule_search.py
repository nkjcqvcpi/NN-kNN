"""Condition grids and validation-only schedule selection."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tools.t1_run import expand_condition_grid, load_config
from tools.t1_schedule_search import condition_id, rank_conditions, write_confirmation

STAGE = Path("configs/t1/schedule_search_20261011.yaml")


def test_stage_grid_expands_to_unique_named_conditions():
    cfg = load_config(STAGE)
    names = [c["name"] for c in cfg["conditions"]]
    assert "condition_grid" not in cfg and len(names) == len(set(names)) == 26
    scratch = [c for c in cfg["conditions"] if c["name"].startswith("joint_scratch")]
    assert len(scratch) == 2 and all(c["core_pretrain"] is False and c["epochs"] == 300 for c in scratch)
    assert {"joint__joint_objective=summed__adapter_warmup_epochs=5__retrieval_lr_scale=0.3",
            "alternating__block_epochs=1__retrieval_lr_scale=1.0", "alternating_default"} <= set(names)


def test_duplicate_grid_names_are_rejected():
    cfg = {"conditions": [{"name": "a__x=1"}], "condition_grid": [{"name": "a", "grid": {"x": [1]}}]}
    with pytest.raises(ValueError):
        expand_condition_grid(cfg)


def row(task, cond, seed, val_err, val_loss, task_type="classification", arm="tuned", **extra):
    r = {"dataset": task, "condition": f"{arm}__{cond}", "arm": arm, "seed": seed, "task_type": task_type,
         "val_L_post": val_loss, "test_accuracy_post": 0.0, "test_rmse_post": 99.0, **extra}
    if task_type == "classification":
        r["val_accuracy_post"] = 1 - val_err
    else:
        r["val_rmse_post"] = val_err
    if not cond.startswith(("nominal_residual", "aggregate_regression")):
        r["schedule"] = "x"
    return r


def test_reuse_rows_share_an_identity_across_task_types():
    a = row("t", "nominal_residual_scores_combined", 1, .1, .1, loss="combined")
    b = row("u", "aggregate_regression_combined", 1, .1, .1, task_type="regression", loss="combined")
    assert condition_id(a) == condition_id(b) == "tuned::frozen_reuse__combined"


def test_selection_uses_validation_error_then_loss_and_ignores_test():
    rows = []
    for seed in (20, 21):
        rows += [row("a", "joint", seed, .10, .50), row("a", "alt", seed, .10, .40),
                 row("b", "joint", seed, .20, .90, "regression"), row("b", "alt", seed, .30, .10, "regression"),
                 row("a", "retrieval_only", seed, .0, .0)]
    rows[0]["test_accuracy_post"] = 1.0  # test outcomes never change selection
    table, winners, glob, n = rank_conditions(rows)
    assert n == 2 and winners == {"a": "tuned::alt", "b": "tuned::joint"}
    assert {g["condition"] for g in glob} == {"tuned::alt", "tuned::joint"}
    assert all(g["mean_rank"] == 1.5 for g in glob)
    with pytest.raises(SystemExit):
        rank_conditions(rows[:-2])  # unequal seeds for one condition


def test_confirmation_config_carries_finalist_definitions(tmp_path):
    args = SimpleNamespace(stage_config=[str(STAGE)], confirm_template="configs/t1/schedule_search_confirm_20261011.yaml",
                           confirm_config=str(tmp_path / "confirm.yaml"), out=str(tmp_path))
    finalists = ["tuned::joint__joint_objective=separated__adapter_warmup_epochs=0__retrieval_lr_scale=0.3",
                 "tuned::frozen_reuse__combined", "tuned::alternating_default"]
    write_confirmation(args, finalists)
    conf = yaml.safe_load((tmp_path / "confirm.yaml").read_text())
    assert conf["seeds"] == [30, 31, 32, 33, 34] and conf["arms"] == [{"name": "tuned"}]
    assert [c["name"] for c in conf["conditions"]] == [finalists[0].split("::")[1], "alternating_default"]
    assert conf["conditions"][0]["retrieval_lr_scale"] == 0.3 and conf["reuse_conditions"]


def test_tuned_cores_reuse_the_grid_training_settings(tmp_path):
    import json
    from tools.t1_grid_search import nnknn_core_settings
    from tools.t1_schedule_search import prepare
    params = {"representation": "raw", "top_k": 10, "lr_case": 0.01, "glocal_fw_set_num": 4, "tau": 0.5}
    settings = nnknn_core_settings(params)
    assert settings["batch_size"] == 64 and settings["lr_glocal"] == .01 and settings["lr_feature"] == pytest.approx(.001)
    template = tmp_path / "template.yaml"
    template.write_text(yaml.safe_dump({"experiment": "x", "datasets": [{"name": "iris", "task_type": "classification"}]}))
    sel = tmp_path / "grid" / "classification__iris" / "nnknn"
    sel.mkdir(parents=True)
    (sel / "selection.json").write_text(json.dumps({"parameters": params, "test_stream_used": False}))
    out = tmp_path / "tuned.yaml"
    prepare(SimpleNamespace(template=str(template), grid=str(tmp_path / "grid"), out=str(out), experiment="y"))
    tuned = yaml.safe_load(out.read_text())
    assert tuned["core_by_dataset"]["classification__iris"] == settings and tuned["experiment"] == "y"
