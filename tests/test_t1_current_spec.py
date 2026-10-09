"""Behavioral checks for the pinned September 20 PI decisions."""

import copy
import os

os.environ.setdefault("NNKNN_DEVICE", "cpu")

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from model.nn_cdh import ClassificationNNCDHAdapter
from model.t1.candidates import MaintenanceReference, coverage_reachability, removal_influence
from model.common.core import CoreConfig, build_model, evaluate
from model.t1.maintenance import CaseArchive, run_maintenance
from model.t1.provenance import CaseStatisticsStore, ScoreConfig, active_cohorts, audit_provenance, score_cases
from model.t1.retention import RetentionConfig, select_active_set


def toy():
    X = torch.tensor([[0., 0.], [0.1, 0.], [1., 1.], [1.1, 1.]])
    y = torch.tensor([0, 0, 1, 1])
    m = build_model(X, y, CoreConfig(task_type="classification", top_k=2), 2)
    return m, X, y


SCFG = ScoreConfig(smoothing=1, min_retrieval_count=0, min_activation_mass=0)


class FixedAdapter(torch.nn.Module):
    output_mode = "nominal_residual_scores"

    def __init__(self, label):
        super().__init__()
        self.label = label

    def forward(self, dz, p0, du):
        out = torch.zeros_like(p0)
        out[:, self.label] = 5
        return out - p0, out


@pytest.mark.parametrize("correct", [True, False])
def test_credit_uses_final_query_outcome_even_for_opposing_labels(correct):
    m, _, _ = toy()
    X, y = torch.tensor([[.02, 0.]]), torch.tensor([1])
    ad = FixedAdapter(1 if correct else 0)
    r = m.retrieve(X, exclude_identical=False)
    # Neighbors have stored label 0, but an adapter can produce correct class 1.
    assert (r["weights"] @ m.labels[:4]).argmax(1).item() == 0
    st = CaseStatisticsStore()
    audit_provenance(m, X, y, st, exclude_identical=False, adapter=ad, counterfactual=False)
    for i in range(4):
        assert st.get(i).positive_support == pytest.approx(float(r["weights"][0, i]) if correct else 0)
        assert st.get(i).harmful_support == pytest.approx(0 if correct else float(r["weights"][0, i]))
        assert st.get(i).activation_mass == pytest.approx(st.get(i).positive_support + st.get(i).harmful_support)
    assert sum(s.positive_support + s.harmful_support for _, s in st.items()) == pytest.approx(1)


def test_single_case_receives_whole_query_credit():
    m = build_model(torch.tensor([[0., 0.]]), torch.tensor([0]), CoreConfig(task_type="classification", bias_init="manual"), 2)
    st = CaseStatisticsStore()
    audit_provenance(m, torch.tensor([[1., 1.]]), torch.tensor([1]), st,
                     exclude_identical=False, adapter=FixedAdapter(1), counterfactual=False)
    assert st.get(0).positive_support == 1 and st.get(0).harmful_support == 0


def test_identity_loo_retains_duplicates_after_compaction():
    m, _, _ = toy()
    m.cases[1].copy_(m.cases[0])
    q = m.cases[:1].clone()
    m.compact_cases(torch.tensor([1, 0, 3, 2]))
    r = m.retrieve(q, exclude_identical=False, query_case_ids=torch.tensor([0]))
    assert r["weights"][0, 1] == 0  # ID 0 moved to slot 1
    assert r["weights"][0, 0] > 0  # identical input with a different ID stays eligible


def test_mcb_identity_loo_does_not_depend_on_encoder_agreement():
    X = torch.tensor([[0., 0.], [.1, 0.], [1., 1.], [1.1, 1.]])
    m = build_model(X, torch.tensor([0, 0, 1, 1]), CoreConfig(task_type="classification", representation="mlp", mcb_enabled=True), 2)
    with torch.no_grad():
        for p in m.momentum_encoder.parameters():
            p.add_(0.2)
    r = m.retrieve(X, exclude_identical=False, query_case_ids=torch.arange(4))
    assert torch.equal(r["weights"].diagonal(), torch.zeros(4))


def test_removal_uses_final_adapted_loss_and_preserves_parameters_modes_rng():
    m, X, y = toy()
    ad = ClassificationNNCDHAdapter(2, 2, 0, (4, 4), "nominal_residual_scores")
    m.classification_adapter = ad
    ad.eval()
    m.train()
    ad.eval()  # preserve a mixed mode configuration
    before = copy.deepcopy(m.state_dict())
    rng = torch.random.get_rng_state()
    ref = MaintenanceReference(X, y, torch.arange(4))
    result = removal_influence(m, ref)
    assert m.training and not ad.training
    m.eval()
    with torch.no_grad():
        baseline = F.cross_entropy(m(X, exclude_identical=False, query_case_ids=torch.arange(4))[0], y, reduction="none")
        for i in range(4):
            mask = torch.ones(4, dtype=torch.bool)
            mask[i] = False
            without = F.cross_entropy(m(X, exclude_identical=False, query_case_ids=torch.arange(4), case_mask=mask)[0], y, reduction="none")
            assert result["full"][i] == pytest.approx(float((without - baseline).mean()), abs=1e-6)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert all(torch.equal(v, before[k]) for k, v in m.state_dict().items())


def test_evaluation_restores_mixed_modes_on_error():
    m, X, y = toy()
    ad = FixedAdapter(0)
    m.train()
    ad.eval()
    bad = MaintenanceReference(X, y, torch.arange(4), adapter=ad)
    # Threshold too high: no query has a solver; explicitly reject zeros.
    with pytest.raises(ValueError, match="Zero reachability"):
        coverage_reachability(m, bad, activation_threshold=1, zero_reachability="error")
    assert m.training and not ad.training


def test_cached_removal_uses_full_denominator_and_reports_approximation_error():
    m, X, y = toy()
    threshold = .55
    ref = MaintenanceReference(X, y, torch.arange(4))
    result = removal_influence(m, ref, cached_threshold=threshold)
    m.eval()
    r = m.retrieve(X, exclude_identical=False, query_case_ids=torch.arange(4))
    labels = m.labels[:4]
    base = F.nll_loss((r["weights"] @ labels).clamp_min(1e-8).log(), y, reduction="none")
    for i in range(4):
        mask = torch.ones(4, dtype=torch.bool)
        mask[i] = False
        w = m.retrieve(X, exclude_identical=False, query_case_ids=torch.arange(4), case_mask=mask)["weights"]
        loss = F.nll_loss((w @ labels).clamp_min(1e-8).log(), y, reduction="none")
        hits = r["weights"][:, i] > threshold
        expected = float((loss - base)[hits].sum()) / len(y)
        assert result["cached"][i] == pytest.approx(expected, abs=1e-6)
    assert result["cache_max_absolute_error"] == pytest.approx(float(np.max(np.abs(result["full"] - result["cached"]))))
    fast = removal_influence(m, ref, cached_threshold=threshold, validate_cache=False)
    assert np.allclose(fast["cached"], result["cached"])
    assert fast["full"] is None and fast["validation_rerun_queries"] == 0
    assert fast["candidate_rerun_queries"] < len(y) * m.case_count()


def test_ratio_solve_uses_final_outcome_and_excludes_self():
    m, X, y = toy()
    ref = MaintenanceReference(X, y, torch.arange(4), adapter=FixedAdapter(1))
    result = coverage_reachability(m, ref, activation_threshold=.1, zero_reachability="zero")
    m.eval()
    w = m.retrieve(X, exclude_identical=False, query_case_ids=torch.arange(4))["weights"]
    solve = (w > .1) & (y == 1)[:, None]
    assert np.array_equal(result["coverage"], solve.sum(0).numpy())
    assert np.array_equal(result["reachability"], solve.sum(1).numpy())
    assert result["zero_reachability_count"] == 2


def test_regression_requires_declared_success_and_uses_final_adapter():
    m = build_model(torch.tensor([[0.], [1.], [2.]]), torch.tensor([0., 1., 2.]), CoreConfig(task_type="regression"), None)
    with pytest.raises(ValueError, match="explicit finite"):
        audit_provenance(m, torch.tensor([[.5]]), torch.tensor([0.]), CaseStatisticsStore("regression"), exclude_identical=False, counterfactual=False)
    st = CaseStatisticsStore("regression")
    audit_provenance(m, torch.tensor([[.5]]), torch.tensor([0.]), st, exclude_identical=False,
                     counterfactual=False, regression_success_tolerance=10)
    assert sum(s.positive_support for _, s in st.items()) == pytest.approx(1)


def test_capacity_that_cannot_preserve_floors_is_rejected_before_mutation():
    m, X, y = toy()
    st = CaseStatisticsStore()
    audit_provenance(m, X, y, st, exclude_identical=False, query_case_ids=torch.arange(4), counterfactual=False)
    with pytest.raises(ValueError, match="cannot satisfy"):
        run_maintenance(m, st, CaseArchive(), 1, RetentionConfig(policy="random", min_per_cohort=1), SCFG, step=0, run_id="test")
    assert m.case_count() == 4


def test_q_ranking_is_independent_of_bias():
    m, X, y = toy()
    st = CaseStatisticsStore()
    audit_provenance(m, X, y, st, exclude_identical=False, counterfactual=False)
    scores = score_cases(np.arange(4), active_cohorts(m), np.arange(4), st, SCFG)
    cfg = RetentionConfig(policy="provenance_only", min_per_cohort=0)
    before = select_active_set(scores, 2, cfg, None).keep_slots
    scores.B[:] = 1 - scores.B
    scores.bias[:] = -scores.bias
    assert np.array_equal(before, select_active_set(scores, 2, cfg, None).keep_slots)


@pytest.mark.parametrize("budget", [.05, .3])
def test_sequential_removal_refreshes_and_preserves_original_loss_budget(budget):
    m, X, y = toy()
    st = CaseStatisticsStore()
    audit_provenance(m, X, y, st, exclude_identical=False, counterfactual=False)
    ref = MaintenanceReference(torch.tensor([[.05, 0.], [1.05, 1.]]), torch.tensor([0, 1]), stream="maintenance_split")
    baseline = evaluate(m, ref.X, ref.y)["loss_pre"]
    archive = CaseArchive()
    res = run_maintenance(m, st, archive, 2, RetentionConfig(policy="removal_influence", min_per_cohort=1, allowed_loss_increase=budget), SCFG, step=0, run_id="test", reference=ref)
    assert evaluate(m, ref.X, ref.y)["loss_pre"] - baseline <= budget + 1e-6
    trace = res["scoring_trace"]
    if budget == .3:
        assert len(trace) >= 2 and len(trace[1]["active_case_ids"]) == 3
    else:
        assert res["summary"]["stop_reason"] == "cumulative_loss_budget" and m.case_count() == 4
    assert len(archive) == 4 - m.case_count()
    assert sum(e["action"] == "archive" for e in res["events"]) == len(archive)


def test_test_stream_is_rejected_for_candidate_selection():
    m, X, y = toy()
    with pytest.raises(ValueError, match="cannot use test"):
        removal_influence(m, MaintenanceReference(X, y, stream="test"))


def test_regression_adapter_changes_outcome_and_removal_loss():
    class RegressionAdapter(torch.nn.Module):
        def forward_aggregate(self, query_features, case_features, case_labels, case_weights):
            return 2 * (case_weights @ case_labels)

    m = build_model(torch.tensor([[0.], [1.], [2.]]), torch.tensor([0., 1., 2.]), CoreConfig(task_type="regression"), None)
    m.nn_cdh = RegressionAdapter()
    X, y = torch.tensor([[.5]]), torch.tensor([1.])
    st = CaseStatisticsStore("regression")
    info = audit_provenance(m, X, y, st, exclude_identical=False, counterfactual=False, regression_success_tolerance=.01)
    ref = MaintenanceReference(X, y, regression_success_tolerance=.01, stream="maintenance_split")
    result = removal_influence(m, ref)
    m.eval()
    with torch.no_grad():
        base = (m(X, exclude_identical=False)[0].view(-1) - y).square()
        assert info["mean_loss_final"] == pytest.approx(float(base.mean()))
        for i in range(3):
            mask = torch.ones(3, dtype=torch.bool)
            mask[i] = False
            loss = (m(X, exclude_identical=False, case_mask=mask)[0].view(-1) - y).square()
            assert result["full"][i] == pytest.approx(float((loss - base).mean()), abs=1e-6)


def test_retrieval_trace_contains_actual_final_adapter_decision():
    from model.common.core import retrieval_events
    m, _, _ = toy()
    rows = retrieval_events(m, torch.tensor([[.02, 0.]]), torch.tensor([1]),
                            stream="test", run_id="x", adapter=FixedAdapter(1))
    assert rows[0]["pre_adaptation_prediction"] == 0
    assert rows[0]["final_prediction"] == 1
    assert sum(rows[0]["activations"]) == pytest.approx(1)
