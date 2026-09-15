from __future__ import annotations

import copy
import math
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.nn_cdh import (
    ClassificationNNCDHAdapter,
    compute_classification_adaptation_inputs,
)
from model.nnknn_model import NN_KNN_Model, default_args
from model.t1_maintenance import (
    BiasOnlyPolicy,
    CaseArchiveStore,
    CaseStatistics,
    CaseStatisticsStore,
    DownsampleRandomPolicy,
    FullMemoryPolicy,
    ProvenanceBiasCoveragePolicy,
    ProvenanceOnlyPolicy,
    StratifiedRandomPolicy,
    TrustworthinessOnlyPolicy,
    audit_regression_provenance_counterfactual,
    compute_trustworthiness,
    compute_within_cohort_percentile_biases,
    get_maintenance_policy,
)
from model.t1_mcb import (
    MCBEncoder,
    MCBProjectionHead,
    build_mcb_encoder_pair,
    compute_neighborhood_churn,
    compute_representation_drift,
    compute_selection_churn,
)
from model.t1_revise import HumanReviseManager, NeuralCaseReviser
from model.t1_synchronization import (
    ComponentSynchronizer,
    calibrate_free_correction_radius,
    compute_minimal_adaptation_penalty,
)
from model.t1_workflow import (
    T1Config,
    build_t1_model,
    evaluate_t1_model,
    run_t1_maintenance_step,
    train_leave_one_out_adapter,
)


def test_zero_exposure_quality_score():
    """Zero exposure yields Q_i = 0.5 under symmetric smoothing s > 0."""
    stats = CaseStatistics(case_id=1, correct_support=0.0, incorrect_support=0.0)
    for s in [0.5, 1.0, 2.0, 10.0]:
        q = stats.quality_score(smoothing=s)
        assert abs(q - 0.5) < 1e-6, f"Expected 0.5 for s={s}, got {q}"


def test_increasing_correct_support_raises_q():
    """Increasing correct activation raises Q_i; increasing incorrect lowers it."""
    stats = CaseStatistics(case_id=1, correct_support=1.0, incorrect_support=1.0)
    base_q = stats.quality_score(smoothing=1.0)
    assert abs(base_q - 0.5) < 1e-6

    stats.correct_support += 5.0
    q_higher = stats.quality_score(smoothing=1.0)
    assert q_higher > base_q

    stats.incorrect_support += 10.0
    q_lower = stats.quality_score(smoothing=1.0)
    assert q_lower < q_higher


def test_geometric_score_log_space_equivalence():
    """Geometric score computation matches direct and log-space forms."""
    q_vals = [0.1, 0.5, 0.8, 0.99]
    b_vals = [0.2, 0.5, 0.7, 0.95]
    alpha = 0.6

    for q in q_vals:
        for b in b_vals:
            direct_t = (q ** alpha) * (b ** (1.0 - alpha))
            log_t = compute_trustworthiness(q, b, alpha=alpha)
            assert math.isclose(direct_t, log_t, rel_tol=1e-5), f"direct={direct_t}, log={log_t}"


def test_within_cohort_percentile_biases():
    """Percentile normalization produces values in [0, 1] within cohorts."""
    biases = torch.tensor([1.0, 3.0, 2.0, 10.0, 20.0])
    cohorts = [0, 0, 0, 1, 1]
    norm_b = compute_within_cohort_percentile_biases(biases, cohorts)

    assert math.isclose(norm_b[0], 0.0, abs_tol=1e-5)
    assert math.isclose(norm_b[1], 1.0, abs_tol=1e-5)
    assert math.isclose(norm_b[2], 0.5, abs_tol=1e-5)
    assert math.isclose(norm_b[3], 0.0, abs_tol=1e-5)
    assert math.isclose(norm_b[4], 1.0, abs_tol=1e-5)


def test_compaction_preserves_case_ids_and_parameters():
    """Compaction preserves statistics, labels, biases, glocal weights, and stable IDs."""
    cases = torch.randn(10, 4)
    labels = F.one_hot(torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]), num_classes=2).float()
    cfg = T1Config(case_capacity=10)
    model, stats_store, _, _ = build_t1_model(cases, labels, cfg)

    for i in range(10):
        model.biases.data[i] = float(i * 1.5)
        st = stats_store.get(i)
        assert st is not None
        st.retrieval_count = i * 2
        st.correct_support = float(i)

    keep_indices = [2, 5, 8]
    expected_ids = [2, 5, 8]
    expected_biases = [float(i * 1.5) for i in expected_ids]

    removed = model.compact_cases(keep_indices)
    assert removed == 7
    assert model.case_count() == 3

    active_ids = model.active_case_ids().cpu().tolist()
    assert active_ids == expected_ids

    active_biases = model.biases[:3].detach().cpu().tolist()
    for act_b, exp_b in zip(active_biases, expected_biases):
        assert math.isclose(act_b, exp_b, abs_tol=1e-5)

    for cid in expected_ids:
        st = stats_store.get(cid)
        assert st is not None
        assert st.retrieval_count == cid * 2
        assert math.isclose(st.correct_support, float(cid), abs_tol=1e-5)


def test_protected_cases_and_minimum_coverage():
    """Protected cases and minimum per-class floor cannot be evicted."""
    cases = torch.randn(10, 4)
    labels = F.one_hot(torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]), num_classes=2).float()
    cfg = T1Config(case_capacity=10)
    model, stats_store, archive_store, _ = build_t1_model(cases, labels, cfg)

    stats_store.get(0).protected = True
    stats_store.get(9).protected = True

    policy = ProvenanceBiasCoveragePolicy(min_per_class=2)
    removed, actions = run_t1_maintenance_step(
        model, stats_store, archive_store, policy, target_capacity=4, step=1
    )

    kept_ids = model.active_case_ids().cpu().tolist()
    assert len(kept_ids) == 4
    assert 0 in kept_ids, "Protected case 0 was evicted!"
    assert 9 in kept_ids, "Protected case 9 was evicted!"

    kept_labels = model.labels[:len(kept_ids)].argmax(dim=-1).cpu().tolist()
    assert kept_labels.count(0) >= 2, "Class 0 floor violated!"
    assert kept_labels.count(1) >= 2, "Class 1 floor violated!"


def test_archive_restore_reproduces_state():
    """Archived cases preserve all tensors, parameters, and provenance for exact restoration."""
    archive = CaseArchiveStore()
    case_t = torch.tensor([1.0, 2.0, 3.0])
    label_t = torch.tensor([0.0, 1.0])
    bias = 1.234
    stats = CaseStatistics(case_id=7, retrieval_count=15, correct_support=12.5)

    archive.archive_case(7, case_t, label_t, bias, stats, step=10, reason="low_trustworthiness")
    assert archive.contains(7)
    assert archive.count() == 1

    restored = archive.restore(7)
    assert restored is not None
    assert torch.allclose(restored["case_tensor"], case_t)
    assert torch.allclose(restored["label_tensor"], label_t)
    assert math.isclose(restored["bias"], bias, abs_tol=1e-5)
    assert restored["stats"].retrieval_count == 15
    assert restored["stats"].correct_support == 12.5
    assert not archive.contains(7)


def test_mcb_encoder_gradient_isolation_and_ema():
    """MCB memory encoder has requires_grad=False and updates via EMA."""
    online_enc, mem_enc = build_mcb_encoder_pair(
        base_extractor=None,
        in_dim=4,
        hidden_dim=16,
        proj_dim=8,
        normalize=True,
    )

    # Check gradient isolation
    for p in mem_enc.parameters():
        assert not p.requires_grad, "Memory encoder parameter has requires_grad=True!"

    x = torch.randn(5, 4)
    z_online = online_enc(x)
    z_mem = mem_enc(x)
    assert z_online.shape == (5, 8)
    assert z_mem.shape == (5, 8)

    # Simulate gradient update on online encoder
    opt = torch.optim.SGD(online_enc.parameters(), lr=0.1)
    loss = z_online.sum()
    loss.backward()
    opt.step()

    # Memory encoder should NOT change until EMA update
    z_mem_after_grad = mem_enc(x)
    assert torch.allclose(z_mem, z_mem_after_grad, atol=1e-6)

    # Test EMA step in NN_KNN_Model
    cases = torch.randn(6, 4)
    labels = F.one_hot(torch.tensor([0, 1, 0, 1, 0, 1]), num_classes=2).float()
    cfg = T1Config(mcb_enabled=True, mcb_momentum=0.9)
    model, _, _, _ = build_t1_model(cases, labels, cfg)

    assert model.momentum_encoder is not None
    orig_param = next(model.momentum_encoder.parameters()).clone()

    # Step online extractor
    for p in model.feature_extractor.parameters():
        p.data.add_(1.0)

    model.update_momentum_encoder(momentum=0.9)
    updated_param = next(model.momentum_encoder.parameters())
    assert not torch.allclose(orig_param, updated_param)


def test_representation_drift_and_neighborhood_churn():
    """Drift and churn metrics correctly quantify representation changes and neighborhood shifts."""
    # Drift
    f1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    f2 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    assert compute_representation_drift(f1, f2) == 0.0

    f3 = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    drift = compute_representation_drift(f1, f3)
    assert drift > 0.5

    # Neighborhood churn
    n1 = [[0, 1, 2], [3, 4, 5]]
    n2 = [[0, 1, 2], [3, 4, 5]]
    assert compute_neighborhood_churn(n1, n2) == 0.0

    n3 = [[6, 7, 8], [9, 10, 11]]
    assert compute_neighborhood_churn(n1, n3) == 1.0

    # Selection churn
    assert compute_selection_churn([1, 2, 3], [1, 2, 3]) == 0.0
    assert compute_selection_churn([1, 2, 3], [4, 5, 6]) == 1.0


def test_free_radius_calibration_and_penalty():
    """Free radius calibration identifies P_bias and applies zero penalty within threshold."""
    cases = torch.tensor([[0.0, 0.0], [0.1, 0.0], [10.0, 10.0]])
    labels = F.one_hot(torch.tensor([0, 0, 1]), num_classes=2).float()
    cfg = T1Config(case_capacity=3)
    model, _, _, _ = build_t1_model(cases, labels, cfg)

    # Set bias high enough to activate neighbor 1 from case 0
    model.biases.data[0] = 0.5
    model.biases.data[1] = 0.5
    model.biases.data[2] = 0.0

    calib = calibrate_free_correction_radius(model, cases, labels, max_pairs=100)
    assert calib.tau_task >= 0.0
    assert calib.s_task > 0.0

    # Test penalty function
    pre_out = torch.tensor([[0.8, 0.2]])
    # Change is 0.05 (within tau if tau >= 0.1)
    post_out_small = torch.tensor([[0.75, 0.25]])
    pen_zero = compute_minimal_adaptation_penalty(pre_out, post_out_small, tau_task=0.5, s_task=1.0)
    assert pen_zero.item() == 0.0

    # Change exceeds threshold
    post_out_large = torch.tensor([[0.1, 0.9]])
    pen_large = compute_minimal_adaptation_penalty(pre_out, post_out_large, tau_task=0.1, s_task=1.0)
    assert pen_large.item() > 0.0


def test_component_synchronizer_losses():
    """ComponentSynchronizer computes all separable terms in L_R and L_A."""
    sync = ComponentSynchronizer(
        w_pre=1.0, w_post=0.5, w_near=0.01,
        v_post=1.0, v_delta=0.5, v_small=0.1,
        free_radius=0.1, scale_task=1.0,
    )
    p0_q = torch.tensor([[0.7, 0.3]])
    s_q = torch.tensor([[1.2, 0.4]])
    targets = torch.tensor([[1.0, 0.0]])

    l_retrieval, l_adapter, terms = sync.compute_losses(
        p0_q=p0_q,
        s_q=s_q,
        target_labels=targets,
        residual_hat=torch.tensor([[0.1, -0.1]]),
        residual_target=torch.tensor([[0.3, -0.3]]),
        query_to_case_dist=torch.tensor([0.25]),
    )

    assert terms.loss_pre > 0.0
    assert terms.loss_post > 0.0
    assert terms.loss_near > 0.0
    assert terms.loss_delta > 0.0
    assert terms.loss_retrieval_total > 0.0
    assert terms.loss_adapter_total > 0.0


def test_neural_revise_and_human_intervention():
    """NeuralCaseReviser and HumanReviseManager detect anomalies and measure causal flips."""
    # Test NeuralCaseReviser
    reviser = NeuralCaseReviser(k_neighbors=2, conflict_threshold=0.5, confidence_threshold=0.6)
    cases = torch.tensor([[0.0, 0.0], [0.05, 0.0], [0.02, 0.01]])  # cluster together
    labels = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])  # case 2 is conflicting anomaly

    rev_cases, rev_labels, rev_biases, records = reviser.revise_case_base(cases, labels)
    assert len(records) > 0
    assert records[0].action == "relabel"
    assert records[0].proposed_label == 0

    # Test HumanReviseManager
    mgr = HumanReviseManager(revision_enabled=True)
    cfg = T1Config(case_capacity=3)
    model, _, _, _ = build_t1_model(cases, labels, cfg)
    rec = mgr.edit_case_label(model, case_idx=2, new_label=0)
    assert rec.proposed_label == 0
    assert model.labels[2].argmax().item() == 0


def test_actor_critic_namespaces_never_collide():
    """Actor and critic case namespaces remain distinct."""
    actor_stats = CaseStatisticsStore()
    critic_stats = CaseStatisticsStore()

    actor_stats.register_case(case_id=1, initial_bias=0.1, case_role="actor")
    critic_stats.register_case(case_id=1, initial_bias=0.9, case_role="critic")

    assert actor_stats.get(1).case_role == "actor"
    assert critic_stats.get(1).case_role == "critic"
    assert actor_stats.get(1).initial_bias != critic_stats.get(1).initial_bias


if __name__ == "__main__":
    tests = [
        test_zero_exposure_quality_score,
        test_increasing_correct_support_raises_q,
        test_geometric_score_log_space_equivalence,
        test_within_cohort_percentile_biases,
        test_compaction_preserves_case_ids_and_parameters,
        test_protected_cases_and_minimum_coverage,
        test_archive_restore_reproduces_state,
        test_mcb_encoder_gradient_isolation_and_ema,
        test_representation_drift_and_neighborhood_churn,
        test_free_radius_calibration_and_penalty,
        test_component_synchronizer_losses,
        test_neural_revise_and_human_intervention,
        test_actor_critic_namespaces_never_collide,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"[FAIL] {t.__name__}: {e}")
            raise
    print(f"\nAll {passed}/{len(tests)} T1 tests passed successfully!")
