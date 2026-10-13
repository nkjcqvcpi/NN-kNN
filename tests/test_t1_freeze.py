"""Freeze logic: schedule alignment across task types, matched references, global selection."""
import pytest

from tests.test_t1_schedule_search import full_row
from tools.t1_freeze import aligned, global_winner, matched_reference, spec_for

SEEDS = [20, 21, 22]


def test_alignment_drops_the_classification_score_scale_marker():
    assert aligned("tuned::joint_x__kappa10") == "joint_x" == aligned("tuned::joint_x")
    assert aligned("tuned::frozen_reuse__combined__kappa=10") == aligned("tuned::frozen_reuse__combined") == "frozen_reuse__combined"


def test_specs_and_matched_reference_follow_f():
    dicts = {"joint_x__kappa10": {"schedule": "joint_rr", "score_scale": 10.0, "epochs": 300, "patience": 60,
                                  "early_stopping": "val_error", "name": "joint_x__kappa10"},
             "joint_x": {"schedule": "joint_rr", "epochs": 300, "patience": 60, "early_stopping": "val_error", "name": "joint_x"}}
    cls, reg = spec_for("tuned::joint_x", dicts, "classification"), spec_for("tuned::joint_x__kappa10", dicts, "regression")
    assert cls["score_scale"] == 10.0 and "score_scale" not in reg
    ref = matched_reference(cls, "classification")
    assert ref == {"name": "alternating_default_matched", "schedule": "alternating_rr", "epochs": 300, "patience": 60,
                   "early_stopping": "val_error", "score_scale": 10.0}
    assert "score_scale" not in matched_reference(reg, "regression")
    assert spec_for("tuned::frozen_reuse__combined", dicts, "regression") == {"reuse": True}
    assert matched_reference({"reuse": True}, "regression") == {"name": "alternating_default_matched", "schedule": "alternating_rr"}
    with pytest.raises(SystemExit):
        spec_for("tuned::unknown", dicts, "regression")


def test_global_winner_needs_every_task_feasibility_and_shared_schedules():
    rows = []
    for s in SEEDS:
        for t, tt, suffix in (("c1", "classification", "__kappa10"), ("r1", "regression", "")):
            rows.append(full_row(t, "retrieval_only", s, .30, test_err=.3, task_type=tt))
            rows.append(full_row(t, "safe" + suffix, s, .20, val_loss=.2, task_type=tt))
            # best on validation, but degrades retrieval on the regression task: ineligible everywhere
            rows.append(full_row(t, "hide" + suffix, s, .10, val_loss=.1, task_type=tt, pre_err=.9 if tt == "regression" else .3))
        rows.append(full_row("c1", "cls_only__kappa10", s, .05, val_loss=.05))  # not on every task
    for r in rows:
        if r["condition"] != "tuned__retrieval_only":
            r["schedule"] = "joint_rr"
    name, top, excluded = global_winner(rows, SEEDS, 1.02)
    assert name == "tuned::safe" and "tuned::hide" in excluded
    assert all("cls_only" not in g["condition"] for g in top)
