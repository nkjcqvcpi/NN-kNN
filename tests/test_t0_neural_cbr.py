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
import torch.nn.functional as F

from model.nn_cdh import (
    ClassificationNNCDHAdapter,
    compute_classification_adaptation_inputs,
)
from model.nnknn_model import NN_KNN_Model, default_args
from model.t0_maintenance import (
    BiasOnlyPolicy,
    CaseArchiveStore,
    CaseStatistics,
    CaseStatisticsStore,
    ProvenanceBiasCoveragePolicy,
    audit_regression_provenance_counterfactual,
    compute_trustworthiness,
    compute_within_cohort_percentile_biases,
)
from model.t0_workflow import (
    T0Config,
    build_t0_model,
    evaluate_t0_model,
    load_t0_checkpoint,
    run_t0_maintenance_step,
    save_t0_checkpoint,
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

    # More correct support raises Q
    stats.correct_support += 5.0
    q_higher = stats.quality_score(smoothing=1.0)
    assert q_higher > base_q

    # More incorrect support lowers Q
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

    # In cohort 0: 1.0 -> 0.0, 2.0 -> 0.5, 3.0 -> 1.0
    assert math.isclose(norm_b[0], 0.0, abs_tol=1e-5)
    assert math.isclose(norm_b[1], 1.0, abs_tol=1e-5)
    assert math.isclose(norm_b[2], 0.5, abs_tol=1e-5)
    # In cohort 1: 10.0 -> 0.0, 20.0 -> 1.0
    assert math.isclose(norm_b[3], 0.0, abs_tol=1e-5)
    assert math.isclose(norm_b[4], 1.0, abs_tol=1e-5)


def test_compaction_preserves_case_ids_and_parameters():
    """Compaction preserves statistics, labels, biases, glocal weights, and stable IDs."""
    cases = torch.randn(10, 4)
    labels = F.one_hot(torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]), num_classes=2).float()
    cfg = T0Config(case_capacity=10)
    model, stats_store, _, _ = build_t0_model(cases, labels, cfg)

    # Set specific biases
    for i in range(10):
        model.biases.data[i] = float(i * 10)

    # Compact to keep indices [2, 5, 8]
    keep_indices = [2, 5, 8]
    expected_ids = [2, 5, 8]
    expected_biases = [20.0, 50.0, 80.0]

    removed = model.compact_cases(keep_indices)
    assert removed == 7
    assert model.case_count() == 3
    assert model.active_case_ids().tolist() == expected_ids
    for idx, b in enumerate(expected_biases):
        assert math.isclose(model.biases[idx].item(), b, abs_tol=1e-4)


def test_protected_cases_and_minimum_coverage():
    """Protected cases and minimum cohort coverage cannot be violated by the retention policy."""
    cases = torch.randn(12, 4)
    # 6 cases of class 0, 6 cases of class 1
    classes = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    labels = F.one_hot(classes, num_classes=2).float()
    cfg = T0Config(case_capacity=12, target_capacity=4, case_min_per_cohort=2)
    model, stats_store, archive_store, policy = build_t0_model(cases, labels, cfg)

    # Set all class 0 biases artificially low to test if coverage floor protects them
    model.biases.data[:6] = -100.0
    model.biases.data[6:] = 100.0

    new_count, actions = run_t0_maintenance_step(
        model, stats_store, archive_store, policy, target_capacity=4
    )

    assert new_count == 4
    remaining_classes = model.labels[:new_count].argmax(dim=-1).tolist()
    # Must preserve at least 2 cases of class 0 and 2 cases of class 1
    assert remaining_classes.count(0) == 2
    assert remaining_classes.count(1) == 2


def test_archive_restore_reproduces_state():
    """Archiving an evicted case and then restoring it reproduces case state."""
    archive = CaseArchiveStore()
    case_t = torch.tensor([1.0, 2.0, 3.0])
    label_t = torch.tensor([1.0, 0.0])
    stats = CaseStatistics(case_id=42, correct_support=3.5, incorrect_support=0.5)

    archive.archive_case(
        case_id=42,
        case_tensor=case_t,
        label_tensor=label_t,
        bias=1.5,
        stats=stats,
        step=10,
        reason="test_eviction",
    )

    assert archive.count() == 1
    assert archive.contains(42)

    restored = archive.restore(42)
    assert restored is not None
    assert archive.count() == 0
    assert torch.equal(restored["case_tensor"], case_t)
    assert torch.equal(restored["label_tensor"], label_t)
    assert restored["bias"] == 1.5
    assert restored["stats"].correct_support == 3.5


def test_deterministic_policy_decisions():
    """Deterministic inputs and seed produce identical keep/evict decisions."""
    cases = torch.randn(10, 4)
    labels = F.one_hot(torch.randint(0, 3, (10,)), num_classes=3).float()

    cfg1 = T0Config(case_capacity=10, target_capacity=5, seed=123)
    model1, stats1, arc1, pol1 = build_t0_model(cases, labels, cfg1)

    cfg2 = T0Config(case_capacity=10, target_capacity=5, seed=123)
    model2, stats2, arc2, pol2 = build_t0_model(cases, labels, cfg2)

    keep_ids1, act1 = pol1.select_keep_case_ids(
        model1.active_case_ids().tolist(), model1.cases, model1.labels, model1.biases, stats1, 5
    )
    keep_ids2, act2 = pol2.select_keep_case_ids(
        model2.active_case_ids().tolist(), model2.cases, model2.labels, model2.biases, stats2, 5
    )

    assert keep_ids1 == keep_ids2
    assert [a.action for a in act1] == [a.action for a in act2]


def test_target_residual_single_correct_case():
    """With one retrieved case of the correct class, target residual is all zeros."""
    y_target = F.one_hot(torch.tensor([1]), num_classes=3).float()
    p0 = F.one_hot(torch.tensor([1]), num_classes=3).float()
    r_star = y_target - p0
    assert torch.equal(r_star, torch.zeros(1, 3))


def test_target_residual_single_incorrect_case():
    """With one retrieved case of class j and target class k, residual has -1 at j and +1 at k."""
    # Target is class 2, retrieved is class 0
    y_target = F.one_hot(torch.tensor([2]), num_classes=3).float()
    p0 = F.one_hot(torch.tensor([0]), num_classes=3).float()
    r_star = y_target - p0
    expected = torch.tensor([[-1.0, 0.0, 1.0]])
    assert torch.equal(r_star, expected)


def test_aggregate_target_residual_sums_to_zero():
    """For any aggregate neighborhood, every target residual sums to zero within tolerance."""
    # Random probability mass for 10 queries over 4 classes
    p0 = F.softmax(torch.randn(10, 4), dim=-1)
    targets = F.one_hot(torch.randint(0, 4, (10,)), num_classes=4).float()
    r_star = targets - p0
    row_sums = r_star.sum(dim=-1)
    assert torch.allclose(row_sums, torch.zeros(10), atol=1e-6)


def test_disabled_adapter_returns_retrieval_only():
    """A disabled adapter returns the retrieval-only classification output exactly."""
    cases = torch.randn(8, 4)
    labels = F.one_hot(torch.tensor([0, 1, 0, 1, 0, 1, 0, 1]), num_classes=2).float()
    cfg = T0Config(case_capacity=8, classification_adapter_enabled=False)
    model, _, _, _ = build_t0_model(cases, labels, cfg)

    query = torch.randn(3, 4)
    out, pred, pre_adapt, _, _, _ = model(query)
    # When disabled, pre_adapted_solution is None and out is the direct class mass
    assert pre_adapt is None
    assert out.shape == (3, 2)
    assert torch.allclose(out.sum(dim=1), torch.ones(3), atol=1e-5)


def test_adapter_input_dimensions_and_isolation():
    """The aggregate adapter input strictly contains Delta_z_q, Delta_u_q, and p0_q."""
    adapter = ClassificationNNCDHAdapter(
        feature_dim=16,
        num_classes=3,
        nominal_dim=6,
        hidden_dims=(32, 16),
        output_mode="nominal_residual_scores",
    )

    B = 5
    Delta_z = torch.randn(B, 16)
    p0 = F.softmax(torch.randn(B, 3), dim=-1)
    Delta_u = torch.randn(B, 6)

    r_hat, s_q = adapter(Delta_z, p0, Delta_u)
    assert r_hat.shape == (B, 3)
    assert s_q.shape == (B, 3)

    # Residuals must be in [-1, 1] due to tanh
    assert (r_hat >= -1.0 - 1e-6).all() and (r_hat <= 1.0 + 1e-6).all()


def test_nominal_grouped_differences_sum_to_zero():
    """Each nominal field difference Delta_u_qj sums to zero within tolerance."""
    # 2 nominal fields: field 1 has 3 categories, field 2 has 2 categories
    u_q1 = F.one_hot(torch.tensor([1, 2]), num_classes=3).float()
    u_q2 = F.one_hot(torch.tensor([0, 1]), num_classes=2).float()
    nominal_q = torch.cat([u_q1, u_q2], dim=-1)  # [2, 5]

    # Retrieved cases
    K = 4
    u_ret1 = F.one_hot(torch.randint(0, 3, (2, K)), num_classes=3).float()
    u_ret2 = F.one_hot(torch.randint(0, 2, (2, K)), num_classes=2).float()
    nominal_ret = torch.cat([u_ret1, u_ret2], dim=-1)  # [2, K, 5]

    w = F.softmax(torch.randn(2, K), dim=-1)
    q_feats = torch.randn(2, 4)
    ret_feats = torch.randn(2, K, 4)
    labels = F.one_hot(torch.randint(0, 2, (2, K)), num_classes=2).float()

    Delta_z, p0, Delta_u = compute_classification_adaptation_inputs(
        q_feats, ret_feats, w, labels, nominal_q, nominal_ret
    )

    assert Delta_u is not None
    # Check field 1 sum (first 3 columns)
    diff_field1_sum = Delta_u[:, :3].sum(dim=-1)
    assert torch.allclose(diff_field1_sum, torch.zeros(2), atol=1e-5)
    # Check field 2 sum (next 2 columns)
    diff_field2_sum = Delta_u[:, 3:].sum(dim=-1)
    assert torch.allclose(diff_field2_sum, torch.zeros(2), atol=1e-5)


def test_pre_adaptation_statistics_invariance():
    """Pre-adaptation statistics are identical whether the adapter is enabled or disabled."""
    cases = torch.randn(8, 4)
    labels = F.one_hot(torch.tensor([0, 1, 0, 1, 0, 1, 0, 1]), num_classes=2).float()
    queries = torch.randn(4, 4)

    # Model without adapter
    cfg_off = T0Config(case_capacity=8, classification_adapter_enabled=False)
    m_off, stats_off, _, _ = build_t0_model(cases, labels, cfg_off)
    eval_off = evaluate_t0_model(m_off, queries, labels[:4], cfg_off, stats_store=stats_off)

    # Model with adapter
    cfg_on = T0Config(case_capacity=8, classification_adapter_enabled=True)
    m_on, stats_on, _, _ = build_t0_model(cases, labels, cfg_on)
    # Keep weights identical
    m_on.biases.data.copy_(m_off.biases.data)
    eval_on = evaluate_t0_model(m_on, queries, labels[:4], cfg_on, stats_store=stats_on)

    # Pre-adaptation accuracy and stats must match exactly
    assert math.isclose(eval_off["pre_accuracy"], eval_on["pre_accuracy"], abs_tol=1e-6)
    for cid in range(8):
        st_off = stats_off.get(cid)
        st_on = stats_on.get(cid)
        assert st_off.retrieval_count == st_on.retrieval_count
        assert math.isclose(st_off.activation_mass, st_on.activation_mass, abs_tol=1e-5)


def test_flip_analysis_accounting():
    """ClassificationNNCDHAdapter.analyze_flips properly counts correct, harmful, and total flips."""
    # 4 queries, 2 classes
    # Target: [0, 0, 1, 1]
    targets = torch.tensor([0, 0, 1, 1])
    # p0 predictions: [0, 1, 0, 1]  (q0: correct, q1: incorrect, q2: incorrect, q3: correct)
    p0 = torch.tensor([
        [0.8, 0.2],  # pred 0 (correct)
        [0.2, 0.8],  # pred 1 (incorrect)
        [0.7, 0.3],  # pred 0 (incorrect)
        [0.1, 0.9],  # pred 1 (correct)
    ])
    # adapted predictions: [1, 0, 0, 1]
    # q0: was 0 (correct), now 1 (incorrect) -> harmful flip!
    # q1: was 1 (incorrect), now 0 (correct) -> correct flip!
    # q2: was 0 (incorrect), now 0 (incorrect) -> no flip
    # q3: was 1 (correct), now 1 (correct) -> no flip
    adapted = torch.tensor([
        [0.3, 0.7],  # pred 1
        [0.9, 0.1],  # pred 0
        [0.6, 0.4],  # pred 0
        [0.2, 0.8],  # pred 1
    ])

    flips = ClassificationNNCDHAdapter.analyze_flips(p0, adapted, targets)
    assert flips["total_flips"] == 2
    assert flips["correct_flips"] == 1
    assert flips["harmful_flips"] == 1
    assert flips["net_flip_benefit"] == 0
    assert flips["pre_accuracy"] == 0.5
    assert flips["post_accuracy"] == 0.5


def test_regression_counterfactual_auditing():
    """Verify positive Delta_i for helpful cases and negative Delta_i for harmful cases in regression auditing."""
    cases = torch.tensor([[0.0], [1.0], [0.2]])
    labels = torch.tensor([[0.0], [1.0], [50.0]])
    cfg = T0Config(task_type="regression", case_capacity=3, top_k=3, tau=1.0)
    model, stats_store, _, _ = build_t0_model(cases, labels, cfg)

    audit_queries = torch.tensor([[0.2]])
    audit_targets = torch.tensor([[0.2]])

    res = audit_regression_provenance_counterfactual(
        model=model,
        audit_queries=audit_queries,
        audit_targets=audit_targets,
        stats_store=stats_store,
        step=1,
    )
    assert res["status"] == "audited"
    st2 = stats_store.get(2)
    assert st2.incorrect_support > st2.correct_support


def test_statistics_only_mode_invariance():
    """Forward/backward outputs and weights are completely unchanged when statistics are active vs disabled."""
    torch.manual_seed(42)
    cases = torch.randn(8, 4)
    labels = F.one_hot(torch.randint(0, 2, (8,)), num_classes=2).float()
    queries = torch.randn(4, 4)
    targets = torch.tensor([0, 1, 0, 1])

    cfg1 = T0Config(case_capacity=8)
    m1, stats1, _, _ = build_t0_model(cases, labels, cfg1)

    cfg2 = T0Config(case_capacity=8)
    m2, _, _, _ = build_t0_model(cases, labels, cfg2)
    m2.load_state_dict(m1.state_dict())

    eval1 = evaluate_t0_model(m1, queries, targets, cfg1, stats_store=stats1)
    eval2 = evaluate_t0_model(m2, queries, targets, cfg2, stats_store=None)

    assert torch.allclose(eval1["scores"], eval2["scores"])
    assert eval1["pre_accuracy"] == eval2["pre_accuracy"]
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.allclose(p1, p2)


def test_actor_critic_namespaces_and_statistics_isolation():
    """Verify actor and critic stats stores never share IDs or cross-contaminate."""
    from model.nnknn_rl_workflow import NNKNNRLConfig, _build_actor_critic_models
    cfg = NNKNNRLConfig(
        actor_type="nnknn",
        critic_type="nnknn",
        case_capacity=16,
        critic_case_capacity=16,
        case_maintenance_policy="provenance_bias_coverage",
    )
    actor, critic = _build_actor_critic_models(obs_dim=4, action_dim=2, cfg=cfg, device=torch.device("cpu"))

    assert actor.stats_store is not critic.stats_store
    assert actor.archive_store is not critic.archive_store

    obs = torch.randn(4, 4)
    actions = torch.tensor([0, 1, 0, 1])
    actor.add_cases(obs, actions)

    critic_obs = torch.randn(4, 4)
    critic_vals = torch.tensor([1.0, 2.0, 3.0, 4.0])
    critic.add_cases(critic_obs, critic_vals)

    adv = torch.tensor([1.0, -1.0, 0.5, -0.5])
    actor.observe_actor_experience(obs, actions, adv, step=1)

    assert len(critic.stats_store._stats) == 0
    assert len(actor.stats_store._stats) > 0

    target_returns = torch.tensor([[1.5], [2.5], [3.0], [4.5]])
    critic.observe_critic_experience(critic_obs, target_returns, step=1)

    actor.stats_store.ensure(999).correct_support += 10.0
    assert 999 not in critic.stats_store._stats


def test_target_critic_alignment_after_maintenance():
    """Verify target critic tensors match online critic after compaction."""
    from model.nnknn_rl_workflow import (
        NNKNNRLConfig,
        _align_nnknn_target_case_store,
        _build_actor_critic_models,
        _build_nnknn_target_value_model,
    )
    cfg = NNKNNRLConfig(
        critic_type="nnknn",
        critic_case_capacity=10,
        case_maintenance_policy="bias_only",
        case_prune_quantile=0.5,
    )
    _, value_model = _build_actor_critic_models(obs_dim=4, action_dim=2, cfg=cfg, device=torch.device("cpu"))
    target_value_model = _build_nnknn_target_value_model(value_model, device=torch.device("cpu"))

    obs = torch.randn(6, 4)
    vals = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    value_model.add_cases(obs, vals)
    _align_nnknn_target_case_store(value_model, target_value_model)
    assert target_value_model.case_entries == 6

    pruned = value_model.prune_cases(force=True)
    assert pruned > 0
    new_online_count = value_model.case_entries
    assert new_online_count < 6

    _align_nnknn_target_case_store(value_model, target_value_model)
    assert target_value_model.case_entries == new_online_count
    assert torch.allclose(
        target_value_model.nnknn_model.cases[:new_online_count],
        value_model.nnknn_model.cases[:new_online_count],
    )
    assert torch.allclose(
        target_value_model.nnknn_model.labels[:new_online_count],
        value_model.nnknn_model.labels[:new_online_count],
    )


def test_evaluation_test_data_isolation():
    """Verify evaluation on test data never modifies training provenance store."""
    cases = torch.randn(10, 4)
    labels = F.one_hot(torch.randint(0, 2, (10,)), num_classes=2).float()
    cfg = T0Config(case_capacity=10)
    model, train_stats, _, _ = build_t0_model(cases, labels, cfg)

    train_queries = torch.randn(5, 4)
    train_targets = torch.randint(0, 2, (5,))
    evaluate_t0_model(model, train_queries, train_targets, cfg, stats_store=train_stats)

    train_snapshot = copy.deepcopy(train_stats.snapshot())
    assert len(train_snapshot) > 0

    test_queries = torch.randn(20, 4)
    test_targets = torch.randint(0, 2, (20,))
    evaluate_t0_model(model, test_queries, test_targets, cfg, stats_store=None)

    post_snapshot = train_stats.snapshot()
    assert len(train_snapshot) == len(post_snapshot)
    for cid, st_pre in train_snapshot.items():
        st_post = post_snapshot[cid]
        assert st_pre.retrieval_count == st_post.retrieval_count
        assert math.isclose(st_pre.activation_mass, st_post.activation_mass, abs_tol=1e-6)
        assert math.isclose(st_pre.correct_support, st_post.correct_support, abs_tol=1e-6)


def test_checkpoint_save_and_reload_full_state():
    """Verify save_t0_checkpoint and load_t0_checkpoint cleanly recover all weights, active counts, archives, and stats."""
    import tempfile
    cases = torch.randn(8, 4)
    labels = F.one_hot(torch.tensor([0, 1, 0, 1, 0, 1, 0, 1]), num_classes=2).float()
    cfg = T0Config(case_capacity=8, classification_adapter_enabled=True)
    model, stats_store, archive_store, policy = build_t0_model(cases, labels, cfg)

    model.biases.data[0] = 5.0
    model.biases.data[1] = -2.0

    stats_store.ensure(0).retrieval_count = 12.0
    stats_store.ensure(0).correct_support = 10.0
    stats_store.ensure(1).retrieval_count = 3.0
    stats_store.ensure(1).incorrect_support = 2.5

    evicted_stats = CaseStatistics(case_id=7, retrieval_count=1.0, incorrect_support=1.0)
    archive_store.archive_case(
        case_id=7,
        case_tensor=torch.ones(4),
        label_tensor=torch.tensor([1.0, 0.0]),
        bias=-1.5,
        stats=evicted_stats,
        step=5,
        reason="test_eviction",
    )

    with tempfile.TemporaryDirectory() as td:
        ckpt_path = Path(td) / "t0_ckpt.pt"
        save_t0_checkpoint(ckpt_path, model, stats_store, archive_store, cfg)

        loaded_model, loaded_stats, loaded_archive, loaded_cfg = load_t0_checkpoint(
            ckpt_path,
            X_sample=cases[:2],
            y_sample=labels[:2],
        )

        assert loaded_model.case_count() == model.case_count()
        assert torch.allclose(loaded_model.cases[:8], model.cases[:8])
        assert torch.allclose(loaded_model.biases[:8], model.biases[:8])
        assert loaded_cfg.classification_adapter_enabled == cfg.classification_adapter_enabled
        assert loaded_stats.get(0).retrieval_count == 12.0
        assert loaded_stats.get(0).correct_support == 10.0
        assert loaded_stats.get(1).incorrect_support == 2.5
        assert len(loaded_archive) == 1
        assert loaded_archive.has_case(7)
        arc7 = loaded_archive.get(7)
        assert arc7["bias"] == -1.5
        assert arc7["eviction_reason"] == "test_eviction"


def test_class_permutation_invariance():
    """Class-index permutation produces corresponding permutation of adapter inputs and outputs."""
    perm = [1, 2, 0]
    q_feats = torch.randn(2, 4)
    ret_feats = torch.randn(2, 3, 4)
    weights = F.softmax(torch.randn(2, 3), dim=-1)
    ret_labels = F.one_hot(torch.tensor([[0, 1, 2], [2, 1, 0]]), num_classes=3).float()

    _, p0_orig, _ = compute_classification_adaptation_inputs(
        query_features=q_feats,
        retrieved_features=ret_feats,
        case_weights=weights,
        retrieved_labels=ret_labels,
    )
    ret_labels_perm = ret_labels[:, :, perm]
    _, p0_permuted, _ = compute_classification_adaptation_inputs(
        query_features=q_feats,
        retrieved_features=ret_feats,
        case_weights=weights,
        retrieved_labels=ret_labels_perm,
    )
    assert torch.allclose(p0_orig[:, perm], p0_permuted, atol=1e-6)


def test_frozen_retrieval_parameter_invariance():
    """Frozen-retrieval training changes adapter parameters without changing retrieval parameters."""
    torch.manual_seed(42)
    cases = torch.randn(10, 4)
    labels = F.one_hot(torch.randint(0, 2, (10,)), num_classes=2).float()
    cfg = T0Config(
        case_capacity=10,
        classification_adapter_enabled=True,
        classification_adapter_freeze_retrieval_first=True,
        classification_adapter_joint_finetune=False,
        adapter_epochs=5,
    )
    model, _, _, _ = build_t0_model(cases, labels, cfg)

    orig_biases = model.biases.detach().clone()
    orig_cases = model.cases.detach().clone()
    adapter = getattr(model, "classification_adapter")
    orig_adapter_params = [p.detach().clone() for p in adapter.parameters()]

    train_leave_one_out_adapter(model, cases, labels, cfg)

    assert torch.allclose(model.biases, orig_biases)
    assert torch.allclose(model.cases, orig_cases)
    changed = any(not torch.allclose(p, p_orig) for p, p_orig in zip(adapter.parameters(), orig_adapter_params))
    assert changed


if __name__ == "__main__":
    tests = [
        test_zero_exposure_quality_score,
        test_increasing_correct_support_raises_q,
        test_geometric_score_log_space_equivalence,
        test_within_cohort_percentile_biases,
        test_compaction_preserves_case_ids_and_parameters,
        test_protected_cases_and_minimum_coverage,
        test_archive_restore_reproduces_state,
        test_deterministic_policy_decisions,
        test_target_residual_single_correct_case,
        test_target_residual_single_incorrect_case,
        test_aggregate_target_residual_sums_to_zero,
        test_disabled_adapter_returns_retrieval_only,
        test_adapter_input_dimensions_and_isolation,
        test_nominal_grouped_differences_sum_to_zero,
        test_pre_adaptation_statistics_invariance,
        test_flip_analysis_accounting,
        test_regression_counterfactual_auditing,
        test_statistics_only_mode_invariance,
        test_actor_critic_namespaces_and_statistics_isolation,
        test_target_critic_alignment_after_maintenance,
        test_evaluation_test_data_isolation,
        test_checkpoint_save_and_reload_full_state,
        test_class_permutation_invariance,
        test_frozen_retrieval_parameter_invariance,
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
    print(f"\nAll {passed}/{len(tests)} T0 tests passed successfully!")


