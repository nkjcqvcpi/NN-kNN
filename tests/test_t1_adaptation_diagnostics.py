import numpy as np
import pytest
import json

from tools.t1_adaptation_diagnostics import (
    regression_diagnostics, classification_diagnostics, correct_scale_interval,
)


def test_regression_distinguishes_beneficial_and_destructive_overshoot():
    y = np.array([1., 1., 1., 1., 1., 1., 0., 1.])
    correction = np.array([-.5, .5, 1., 1.5, 2., 2.5, 1., 0.])
    summary, rows, _ = regression_diagnostics(np.zeros(8), correction, y)
    assert [r["category"] for r in rows] == [
        "wrong_direction", "toward_target_short", "target_reached", "overshoot_still_helps",
        "overshoot_equal_error", "overshoot_harms", "moved_from_exact", "unchanged"]
    assert rows[3]["squared_error_change"] < 0
    assert rows[5]["squared_error_change"] > 0
    assert summary["post_mse"]-summary["pre_mse"] == pytest.approx(summary["mean_c2"]-summary["mean_2rc"])
    assert sum(v for k,v in summary.items() if k.startswith("count_")) == len(y)


def test_scale_oracle_cannot_repair_conflicting_directions():
    summary, _, _ = regression_diagnostics([0, 0], [.5, .5], [1, -1])
    assert summary["oracle_nonnegative_global_scale"] == 0
    assert summary["oracle_scale_mse"] == 1
    assert summary["oracle_sign_mse"] == .25


def test_classifier_needs_larger_scale_and_boundary_is_not_a_fix():
    summary, rows, _ = classification_diagnostics([[.7, .3]], [[.6, .4]], [1])
    assert rows[0]["category"] == "needs_larger_scale"
    assert rows[0]["min_scale_to_correct"] == pytest.approx(2)
    assert not rows[0]["rescuable_up_to_2"] or rows[0]["min_scale_to_correct"] < 2
    assert summary["pre_wrong"] == summary["post_wrong"] == 1
    json.dumps(summary, allow_nan=False)  # export must contain native scalars


def test_classifier_multiclass_overshoot_has_a_finite_correct_interval():
    p, c = np.array([.4, .45, .15]), np.array([.3, 0., .8])
    lo, hi = correct_scale_interval(p, c, 0)
    assert lo == pytest.approx(1/6)
    assert hi == pytest.approx(.5)
    _, rows, _ = classification_diagnostics([p], [p+c], [0])
    assert rows[0]["category"] == "needs_smaller_scale"
    assert np.argmax(p+.3*c) == 0
    assert np.argmax(p+c) == 2


def test_classifier_misaligned_direction_is_not_fixed_by_more_scale():
    p, c = np.array([.2, .5, .3]), np.array([.2, -.1, .4])
    assert correct_scale_interval(p, c, 0) is None
    _, rows, _ = classification_diagnostics([p], [p+c], [0])
    assert rows[0]["category"] == "no_positive_scale_rescues"


def test_classifier_common_score_offset_has_no_decision_effect():
    summary, rows, _ = classification_diagnostics([[.3, .7]], [[2.3, 2.7]], [0])
    assert summary["common_offset_energy_fraction"] == pytest.approx(1)
    assert rows[0]["category"] == "no_positive_scale_rescues"


def test_classifier_flip_counts_reconcile_and_ties_follow_argmax():
    summary, rows, _ = classification_diagnostics([[.7, .3], [.3, .7]], [[.3, .7], [.7, .3]], [0, 0])
    assert summary["count_harmful_flip"] == summary["count_fixed"] == 1
    assert summary["pre_wrong"] == summary["post_wrong"] == 1
    assert correct_scale_interval(np.array([.5, .5]), np.array([1., 1.]), 1) is None
    assert correct_scale_interval(np.array([.5, .5]), np.array([1., 1.]), 0) is not None
