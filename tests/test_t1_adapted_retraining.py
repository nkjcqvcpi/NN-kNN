"""Actual adapted updates, isolated moments and fixed training-stream semantics."""
import copy
from types import SimpleNamespace

import pytest
import torch

from model.t1.adapted_retraining import continued_adapted_trial, run_adapted_retrained_removal, matched_random_continuation
from model.t1.candidates import MaintenanceReference
from model.t1.provenance import CaseStatisticsStore, ScoreConfig
from model.t1.retention import RetentionConfig
from model.t1.retraining import reference_loss
from model.t1.core import CoreConfig, build_model, make_optimizer
from model.t1.reuse import ReuseConfig, train_classification_adapter


def equal(a, b):
    if torch.is_tensor(a):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def prepared(mode="nominal_residual_scores", nominal=False):
    torch.manual_seed(7)
    X = torch.tensor([[0.], [.1], [.3], [.8], [.9], [1.2]])
    y = torch.tensor([0, 0, 1, 1, 0, 1])
    data = SimpleNamespace(X_train=X, y_train=y, X_val=X + .01, y_val=y,
                           num_classes=2, nominal_train=None)
    kwargs = {}
    if nominal:
        data.nominal_train = {"color": ["red", "blue", None, "red", "blue", "red"]}
        data.nominal_val = {"color": ["amber", "blue", None, "red", "blue", "red"]}
        kwargs = {"nominal_fields": ({"name": "color", "covered_by_representation": False},),
                  "nominal_fields_covered_by_representation": False}
    cc = CoreConfig("classification", batch_size=3, top_k=3, seed=7)
    model = build_model(X, y, cc, 2)
    rc = ReuseConfig(mode, "combined", 1., 1., "softmax", epochs=2,
                     patience=2, batch_size=3, hidden_dims=(8, 4), seed=7, **kwargs)
    ad, info = train_classification_adapter(model, data, rc)
    return model, make_optimizer(model, cc), ad, info["optimizer_state"], data, cc, rc


@pytest.mark.parametrize("mode", ["nominal_residual_scores", "logit_residual"])
@pytest.mark.parametrize("budgets", [(1, 0), (0, 1), (1, 1)])
def test_scope_updates_are_real_isolated_and_reproducible(mode, budgets):
    m, opt, ad, ast, data, cc, rc = prepared(mode)
    before = copy.deepcopy((m.state_dict(), opt.state_dict(), ad.state_dict(), ast))
    # Deliberately poison held-out arrays: continuation must never read them.
    data.X_val = data.y_val = data.X_test = data.y_test = None
    rng = torch.get_rng_state().clone()
    kwargs = dict(retrieval_epochs=budgets[0], adapter_epochs=budgets[1], lr_scale=.1)
    first = continued_adapted_trial(m, opt, ad, ast, data, cc, rc, **kwargs)
    repeat = continued_adapted_trial(m, opt, ad, ast, data, cc, rc, **kwargs)
    assert equal(before, (m.state_dict(), opt.state_dict(), ad.state_dict(), ast))
    assert torch.equal(rng, torch.get_rng_state())
    for i in (0, 1, 2, 3):
        assert equal(first[i].state_dict(), repeat[i].state_dict())
    assert equal(before[0], first[0].state_dict()) == (budgets[0] == 0)
    assert equal(before[2], first[2].state_dict()) == (budgets[1] == 0)
    assert first[4]["optimizer_updates"] == {"retrieval": 2 * budgets[0], "adapter": 2 * budgets[1]}
    assert len(first[4]["history"]) == sum(budgets)
    assert all(h["examples"] == 6 for h in first[4]["history"])


def test_nominal_compaction_keeps_fitted_schema_and_original_query_ids():
    m, opt, ad, ast, data, cc, rc = prepared(nominal=True)
    schema = copy.deepcopy(ad.nominal_schema.state_dict())
    # Validation-only amber remains an unknown category, with no refit.
    data.X_val = data.y_val = data.nominal_val = None
    result = continued_adapted_trial(m, opt, ad, ast, data, cc, rc,
        retrieval_epochs=1, adapter_epochs=1, lr_scale=.1, drop_slot=2)
    tm, _, ta, _, info = result
    assert tm.active_case_ids().tolist() == [0, 1, 3, 4, 5]
    assert ta.nominal_schema.state_dict() == schema
    assert equal(ta.nominal_case_ids, ad.nominal_case_ids)
    assert info["training_examples_per_epoch"] == 6
    assert info["case_count"] == 5
    assert info["retrieval_query_case_pairs"] == 60


def test_invalid_budget_or_ambiguous_ids_rejected_before_mutation():
    m, opt, ad, ast, data, cc, rc = prepared()
    before = copy.deepcopy(m.state_dict())
    for kwargs in ({"retrieval_epochs": True, "adapter_epochs": 0},
                   {"retrieval_epochs": 0, "adapter_epochs": 0},
                   {"retrieval_epochs": 1, "adapter_epochs": 0, "training_case_ids": [0] * 6},
                   {"retrieval_epochs": 1, "adapter_epochs": 0, "drop_slot": 99}):
        with pytest.raises(ValueError):
            continued_adapted_trial(m, opt, ad, ast, data, cc, rc, lr_scale=.1, **kwargs)
    assert equal(before, m.state_dict())


def test_actual_removal_candidates_use_their_own_adapter_and_full_reference():
    m, opt, ad, ast, data, cc, rc = prepared(nominal=True)
    data.reg_bins = None
    reference = MaintenanceReference(data.X_train, data.y_train, torch.arange(6),
        adapter=ad, query_nominal=ad.nominal_schema.encode(data.nominal_train, rows=6))
    before = copy.deepcopy((m.state_dict(), opt.state_dict(), ad.state_dict(), ast))
    result = run_adapted_retrained_removal(m, opt, ad, ast, data, cc, rc, reference,
        CaseStatisticsStore(), ScoreConfig(1, 0, 0),
        RetentionConfig("removal_influence", min_per_cohort=1, allowed_loss_increase=10),
        4, retrieval_epochs=1, adapter_epochs=1, lr_scale=.1, run_id="adapted-test")
    final, _, fa, _, archive, report = result
    assert final.case_count() == 4 and len(archive) == 2
    assert equal(before, (m.state_dict(), opt.state_dict(), ad.state_dict(), ast))
    fresh_reference = copy.copy(reference)
    fresh_reference.adapter = fa
    assert report["summary"]["final_reference_loss"] == pytest.approx(reference_loss(final, fresh_reference))
    for row in report["scoring_trace"]:
        assert row["reference_denominator"] == 6 and row["accepted"]
        for score in row["candidate_scores"]:
            cid = score["case_id"]
            assert score["influence"] == pytest.approx(score["loss_after_retraining"] - row["starting_reference_loss"])
            assert score["delta_vs_matched_training"] == pytest.approx(score["loss_after_retraining"] - row["matched_full_memory_loss"])
            assert score["continuation"]["case_count"] == len(row["active_case_ids"]) - 1
            assert score["continuation"]["optimizer_updates"] == {"retrieval": 2, "adapter": 2}
        # Independently replay the first actual candidate from the unchanged M0.
        if row["iteration"] == 0:
            score = row["candidate_scores"][0]
            slot = m.active_case_ids().tolist().index(score["case_id"])
            tm, _, ta, _, _ = continued_adapted_trial(m, opt, ad, ast, data, cc, rc,
                retrieval_epochs=1, adapter_epochs=1, lr_scale=.1, drop_slot=slot)
            fresh_reference.adapter = ta
            assert score["loss_after_retraining"] == pytest.approx(reference_loss(tm, fresh_reference))


def test_adapted_removal_rejects_held_out_reference_before_training():
    m, opt, ad, ast, data, cc, rc = prepared()
    reference = MaintenanceReference(data.X_train, data.y_train, stream="test")
    with pytest.raises(ValueError, match="test or validation"):
        run_adapted_retrained_removal(m, opt, ad, ast, data, cc, rc, reference,
            CaseStatisticsStore(), ScoreConfig(1, 0, 0), RetentionConfig("removal_influence", 1),
            4, retrieval_epochs=1, adapter_epochs=1, lr_scale=.1, run_id="bad")


def test_rejected_candidates_preserve_accepted_model_and_adapter(monkeypatch):
    import model.t1.retraining as retraining
    m, opt, ad, ast, data, cc, rc = prepared()
    data.reg_bins = None
    reference = MaintenanceReference(data.X_train, data.y_train, torch.arange(6), adapter=ad)
    real_loss = retraining.reference_loss
    # A deliberate loss stress makes every deletion fail the cumulative budget;
    # the candidate optimizer still performs its actual gradient updates.
    def stress_loss(model, ref):
        return real_loss(model, ref) + (100. if model.case_count() < 6 else 0.)
    monkeypatch.setattr(retraining, "reference_loss", stress_loss)
    result = run_adapted_retrained_removal(m, opt, ad, ast, data, cc, rc, reference,
        CaseStatisticsStore(), ScoreConfig(1, 0, 0), RetentionConfig("removal_influence", 1, allowed_loss_increase=0),
        4, retrieval_epochs=1, adapter_epochs=1, lr_scale=.1, run_id="reject")
    assert result[-1]["summary"]["stop_reason"] == "cumulative_loss_budget"
    assert result[-1]["events"] == [] and len(result[4]) == 0
    assert result[-1]["summary"]["optimizer_updates"]["accepted"] == {"retrieval": 0, "adapter": 0}
    assert equal(result[0].state_dict(), m.state_dict())
    assert equal(result[2].state_dict(), ad.state_dict())
    assert equal(result[1].state_dict(), opt.state_dict())
    assert equal(result[3], ast)


def test_random_control_matches_capacity_updates_and_is_repeatable():
    m, opt, ad, ast, data, cc, rc = prepared()
    data.reg_bins = None
    before = copy.deepcopy((m.state_dict(), opt.state_dict(), ad.state_dict(), ast))
    args = (m, opt, ad, ast, data, cc, rc, CaseStatisticsStore(), ScoreConfig(1, 0, 0),
            RetentionConfig("removal_influence", min_per_cohort=1, allowed_loss_increase=0), 4)
    first = matched_random_continuation(*args, retrieval_epochs=1, adapter_epochs=1, lr_scale=.1, seed=42)
    second = matched_random_continuation(*args, retrieval_epochs=1, adapter_epochs=1, lr_scale=.1, seed=42)
    assert first[0].case_count() == 4
    assert first[-1]["optimizer_updates"] == {"retrieval": 4, "adapter": 4}
    assert first[-1]["removed_case_ids"] == second[-1]["removed_case_ids"]
    assert first[-1]["loss_budget_used_for_selection"] is False
    assert len(first[-1]["history"]) == 2
    assert len(first[0].labels[:4].argmax(1).unique()) == 2
    for i in (0, 1, 2):
        assert equal(first[i].state_dict(), second[i].state_dict())
    assert equal(first[3], second[3])
    assert equal(before, (m.state_dict(), opt.state_dict(), ad.state_dict(), ast))


def test_random_control_cannot_break_cohort_protection():
    m, opt, ad, ast, data, cc, rc = prepared()
    data.reg_bins = None
    with pytest.raises(ValueError, match="protected floors"):
        matched_random_continuation(m, opt, ad, ast, data, cc, rc, CaseStatisticsStore(),
            ScoreConfig(1, 0, 0), RetentionConfig("removal_influence", min_per_cohort=1,
                                              protect_rare_cohort_below=4, allowed_loss_increase=0),
            4, retrieval_epochs=1, adapter_epochs=1, lr_scale=.1, seed=42)
