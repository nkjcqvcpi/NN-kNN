"""Independent Adam continuations and retrained-removal budget semantics."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from model.t1.candidates import MaintenanceReference
from model.common.core import CoreConfig, build_model, clone_optimizer, make_optimizer
from model.t1.provenance import CaseStatisticsStore, ScoreConfig
from model.t1.retraining import continued_trial, reference_loss, run_retrained_removal
from model.t1.retention import RetentionConfig


def setup():
    X = torch.tensor([[0.], [.1], [.2], [1.], [1.1], [1.2]])
    y = torch.tensor([0, 0, 0, 1, 1, 1])
    cfg = CoreConfig(task_type="classification", top_k=3, batch_size=3)
    model = build_model(X, y, cfg, 2)
    data = SimpleNamespace(X_train=X, y_train=y, X_val=X + .01, y_val=y, reg_bins=None)
    opt = make_optimizer(model, cfg)
    # Materialize nonzero Adam moments before cloning.
    continued, opt, _ = continued_trial(model, opt, data, cfg, epochs=1, lr_scale=.1)
    return continued, opt, data, cfg, MaintenanceReference(X, y, torch.arange(6))


def assert_state_equal(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_state_equal(a[k], b[k])
    elif isinstance(a, list):
        for x, y in zip(a, b):
            assert_state_equal(x, y)
    else:
        assert a == b


def test_optimizer_forks_do_not_alias_or_change_source():
    model, opt, data, cfg, _ = setup()
    original = copy.deepcopy(opt.state_dict())
    fork = clone_optimizer(opt, copy.deepcopy(model), cfg)
    for key in original["state"]:
        for name in ("exp_avg", "exp_avg_sq"):
            assert fork.state_dict()["state"][key][name].data_ptr() != opt.state_dict()["state"][key][name].data_ptr()
    rng = torch.get_rng_state().clone()
    first, _, _ = continued_trial(model, opt, data, cfg, epochs=2, lr_scale=.1, drop_slot=0)
    continued_trial(model, opt, data, cfg, epochs=2, lr_scale=.1, drop_slot=3)
    repeat, _, _ = continued_trial(model, opt, data, cfg, epochs=2, lr_scale=.1, drop_slot=0)
    assert_state_equal(opt.state_dict(), original)
    assert_state_equal(first.state_dict(), repeat.state_dict())
    assert torch.equal(torch.get_rng_state(), rng)


def test_retrained_removal_trials_match_budget_and_source_stays_unchanged():
    model, opt, data, cfg, ref = setup()
    state, ostate = copy.deepcopy(model.state_dict()), copy.deepcopy(opt.state_dict())
    result, _, archive, info = run_retrained_removal(
        model, opt, data, cfg, ref, CaseStatisticsStore(), ScoreConfig(1, 0, 0),
        RetentionConfig("removal_influence", min_per_cohort=1, allowed_loss_increase=10),
        4, epochs=2, lr_scale=.1, run_id="test")
    assert result.case_count() == 4 and len(archive) == 2
    assert info["summary"]["accepted_model_training_epochs"] == 4
    assert_state_equal(model.state_dict(), state)
    assert_state_equal(opt.state_dict(), ostate)
    for row in info["scoring_trace"]:
        assert row["reference_denominator"] == 6
        assert len(row["control_history"]) == 2
        for score in row["candidate_scores"]:
            assert score["epochs_run"] == 2
            assert score["influence"] == pytest.approx(score["loss_after_retraining"] - row["starting_reference_loss"])
            assert score["delta_vs_matched_training"] == pytest.approx(score["loss_after_retraining"] - row["matched_full_memory_loss"])
    assert info["summary"]["final_reference_loss"] == pytest.approx(reference_loss(result, ref))


def test_retraining_cannot_select_on_validation_reference():
    model, opt, data, cfg, ref = setup()
    ref.stream = "validation"
    with pytest.raises(ValueError, match="test or validation"):
        run_retrained_removal(model, opt, data, cfg, ref, CaseStatisticsStore(), ScoreConfig(1, 0, 0),
                              RetentionConfig("removal_influence", 1, allowed_loss_increase=.1),
                              4, epochs=2, lr_scale=.1, run_id="invalid")
