"""Handoff 2026-11-26 pack p2: established maintenance methods and data-valuation baselines as fixed-K retention
policies, the C = RC(i), H = 0 special-case recovery, latency/memory measurement and the C/H reliability diagnostic.

Most policy tests run on hand-written score matrices (``ActivationEngine.from_arrays``; one row per query, one column
per case, -inf = excluded) whose outcome is checked by hand in the comments. The engine itself is checked against
``model.retrieve`` and ``candidates.removal_influence`` on a real core.
"""
import itertools
import math
import os
from pathlib import Path

os.environ.setdefault("NNKNN_DEVICE", "cpu")

import numpy as np
import pytest
import torch
import yaml

from model.common.core import CoreConfig, build_model
from model.t1.baselines import (ActivationEngine, baseline_priority, coverage_liability, knn_shapley, policy_priority,
                                relative_coverage)
from model.t1.candidates import MaintenanceReference, removal_influence
from model.t1.efficiency import case_memory_bytes, check_efficiency_cfg, efficiency_metrics
from model.t1.maintenance import CaseArchive, run_maintenance
from model.t1.outcomes import final_prediction
from model.t1.provenance import CaseScores, CaseStatisticsStore, ScoreConfig, audit_provenance
from model.t1.reliability import ch_reliability, check_reliability_cfg, extreme_overlap, spearman
from model.t1.retention import BASELINE_POLICIES, POLICIES, RetentionConfig, select_active_set

from t1_units import units

NEG = -np.inf
REPO = Path(__file__).resolve().parents[1]


def case_scores(labels, ids=None, protected=None):
    n = len(labels)
    ids = np.arange(n) if ids is None else np.asarray(ids)
    z = np.zeros(n)
    extra = {"human_protected": np.zeros(n, dtype=bool) if protected is None else np.asarray(protected)}
    return CaseScores(ids, [f"class_{int(c)}" for c in labels], z, z, z, z, z + 0.5, z + 0.5, z.astype(bool), z, extra)


def rcfg(policy, **kw):
    return RetentionConfig(policy=policy, min_per_cohort=kw.pop("min_per_cohort", 0), **kw)


def line_engine(x, labels, top_k=1):
    """Case problems on a line: score = -|x_q - x_c|, own case excluded."""
    x = np.asarray(x, dtype=float)
    Z = -np.abs(x[:, None] - x[None, :])
    np.fill_diagonal(Z, NEG)
    return ActivationEngine.from_arrays(Z, labels, labels, top_k=top_k)


# --------------------------------------------------------------------------- the engine against the real core


def tiny_core(task, topk, n=30, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, 3, generator=g)
    if task == "classification":
        y = (X[:, 0] > 0).long() + (X[:, 1] > 0.5).long()
    else:
        y = X[:, 0] * 2 + 0.1 * torch.randn(n, generator=g)
    cc = CoreConfig(task_type=task, top_k=4, pre_topk_mask=topk, tau=0.7, seed=seed)
    model = build_model(X, y, cc, 3 if task == "classification" else None)
    with torch.no_grad():
        model.biases.copy_(torch.randn(n, generator=g) * 0.3)
    return model, X, y


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("topk", [True, False])
def test_engine_masks_and_loo_match_the_model(task, topk):
    model, X, y = tiny_core(task, topk)
    ids = torch.arange(len(y))
    eng = ActivationEngine(model, X, y, ids)
    assert eng.check_max_abs_diff < 1e-6
    mask = np.ones(len(y), dtype=bool)
    mask[[1, 5, 7, 20]] = False
    model.eval()
    with torch.no_grad():
        r = model.retrieve(X, exclude_identical=False, case_mask=torch.as_tensor(mask), query_case_ids=ids)
        _, p, _ = final_prediction(model, X, r, case_mask=torch.as_tensor(mask), query_case_ids=ids)
    assert np.abs(eng.predict(mask) - p.double().numpy().reshape(len(y), -1)).max() < 1e-6
    ri = removal_influence(model, MaintenanceReference(X, y, ids, 0.5))
    v, base = eng.removal_values(np.ones(len(y), dtype=bool), "nll" if task == "classification" else "squared_error")
    assert np.abs(v - ri["full"]).max() < 1e-6 and abs(base - ri["baseline_loss"]) < 1e-6


def test_engine_refuses_an_adapter_path():
    model, X, y = tiny_core("classification", True)
    model.config["case_score_mode"] = "sigmoid"
    with pytest.raises(ValueError, match="softmax"):
        ActivationEngine(model, X, y, torch.arange(len(y)))


# --------------------------------------------------------------------------- special-case recovery C = RC, H = 0


def test_special_case_recovery_C_equals_RC_and_H_is_zero():
    """Uniform activation over exactly the successful solvers (survey section 3): equal biases, duplicates at three
    sites far apart and a temperature so low that every other case gets activation 0. Two A cases (class 0), three B
    cases (class 1), one C case (class 2) no query reaches; 3 queries at A, 2 at B (a separate maintenance split).
    RC(A) = 3 * 1/2 = 1.5, RC(B) = 2 * 1/3, RC(C) = 0 and the audit gives the same C with H = 0."""
    X = torch.tensor([[0.0, 0.0]] * 2 + [[10.0, 10.0]] * 3 + [[-10.0, 10.0]])
    y = torch.tensor([0, 0, 1, 1, 1, 2])
    model = build_model(X, y, CoreConfig(task_type="classification", top_k=6, pre_topk_mask=False, tau=0.01, seed=0), 3)
    with torch.no_grad():
        model.biases.fill_(0.0)
    Xq = torch.tensor([[0.0, 0.0]] * 3 + [[10.0, 10.0]] * 2)
    yq = torch.tensor([0, 0, 0, 1, 1])
    eng = ActivationEngine(model, Xq, yq, None)
    members, w = eng.weights(np.ones(6, dtype=bool), np.arange(5))
    act = np.zeros((5, 6))
    np.put_along_axis(act, members, w, 1)
    assert np.allclose(act[:3, :2], 0.5) and np.allclose(act[3:, 2:5], 1 / 3)
    assert act[:3, 2:].sum() == 0 and act[3:, [0, 1, 5]].sum() == 0  # zero activation outside the solver set
    rc, reach = relative_coverage(eng, threshold=0.0)
    assert np.allclose(rc, [1.5, 1.5, 2 / 3, 2 / 3, 2 / 3, 0.0]) and np.allclose(reach, [2, 2, 2, 3, 3])
    store = CaseStatisticsStore("classification")
    audit_provenance(model, Xq, yq, store, exclude_identical=False, counterfactual=False)
    C = np.array([store.get(i).positive_support for i in range(6)])
    H = np.array([store.get(i).harmful_support for i in range(6)])
    assert np.allclose(C, rc, atol=1e-6) and np.allclose(H, 0.0)


def test_relative_coverage_hand_example():
    """Survey section 3: c0 covers problems with 1, 2 and 4 solvers -> RC = 1 + 1/2 + 1/4 = 1.75."""
    Z = np.array([[0, NEG, NEG, NEG, NEG],
                  [0, 0, NEG, NEG, NEG],
                  [0, 0, 0, 0, NEG],
                  [0, NEG, NEG, NEG, 0]], dtype=float)  # q3 (class 1): c0 is retrieved but disagrees
    eng = ActivationEngine.from_arrays(Z, [0, 0, 0, 0, 1], [0, 0, 0, 1])
    rc, reach = relative_coverage(eng, threshold=0.1)
    assert np.allclose(rc, [1.75, 0.75, 0.25, 0.25, 1.0]) and np.allclose(reach, [1, 2, 4, 1])


# --------------------------------------------------------------------------- the policies on hand examples


def test_rc_cnn_hand_example():
    """1-NN on a line, x = 0, 1, 3 (class 0), 10, 11, 13 (class 1). Nearest other case: q0->c1, q1->c0, q2->c1,
    q3->c4, q4->c3, q5->c4, so RC = 1, 2, 0, 1, 2, 0 and the order is c1, c4, c0, c3, c2, c5. CNN adds c1 (empty set),
    adds c4 (c1 answers 0 for x = 11), and every other problem is then answered correctly: condensed set {c1, c4}."""
    eng = line_engine([0, 1, 3, 10, 11, 13], [0, 0, 0, 1, 1, 1])
    cfg = rcfg("rc_cnn", activation_threshold=0.0)
    extra, summary = policy_priority(cfg, case_scores([0, 0, 0, 1, 1, 1]), 4, eng, eng)
    assert np.allclose(extra["relative_coverage"], [1, 2, 0, 1, 2, 0])
    assert summary["baseline_set_size"] == 2 and summary["cnn_passes"] == 2
    assert list(np.argsort(-extra["rc_cnn"], kind="stable")) == [1, 4, 0, 3, 2, 5]
    sc = case_scores([0, 0, 0, 1, 1, 1])
    sc.extra.update(extra)
    sel = select_active_set(sc, 4, cfg, None)
    assert sorted(sel.keep_slots.tolist()) == [0, 1, 3, 4]


def test_crr_hand_example():
    """Same line. Cov sizes 1, 2, 0, 1, 2, 0 (c0 covers q1; c1 covers q0, q2; c3 covers q4; c4 covers q3, q5).
    Increasing order c2, c5, c0, c3, c1, c4: c2 and c5 are kept, c0 is kept and removes c1, c3 is kept and removes c4."""
    eng = line_engine([0, 1, 3, 10, 11, 13], [0, 0, 0, 1, 1, 1])
    cfg = rcfg("crr", activation_threshold=0.0)
    extra, summary = policy_priority(cfg, case_scores([0, 0, 0, 1, 1, 1]), 4, eng, eng)
    assert np.allclose(extra["coverage_size"], [1, 2, 0, 1, 2, 0])
    assert summary["baseline_set_size"] == 4
    assert list(np.argsort(-extra["crr"], kind="stable")) == [2, 5, 0, 3, 1, 4]


NOISY = np.array([[NEG, 1, 0, 3, 2],
                  [3, NEG, 0, 1, 2],
                  [1, 0, NEG, 3, 2],
                  [1, 0, 3, NEG, 2],
                  [3, 1, 2, 0, NEG]], dtype=float)
NOISY_LABELS = [0, 0, 1, 0, 1]


def test_bbnr_hand_example():
    """1-NN, labels 0, 0, 1, 0, 1. q0->c3 (right), q1->c0 (right), q2->c3 (wrong), q3->c2 (wrong), q4->c0 (wrong):
    Cov(c0) = {q1}, Cov(c3) = {q0}; Liability(c0) = {q4}, Liability(c2) = {q3}, Liability(c3) = {q2}.
    Order c0, c2, c3 (ties by ID). Without c0, q1 falls to c4 (wrong): c0 stays. c2 covers nothing: deleted.
    Without c2 and c3, q0 falls to c4 (wrong): c3 stays. Only c2 is deleted."""
    eng = ActivationEngine.from_arrays(NOISY, NOISY_LABELS, NOISY_LABELS, top_k=1)
    cov, liab, correct = coverage_liability(eng, 0.0)
    assert cov == [[1], [], [], [0], []] and liab == [[4], [], [3], [2], []]
    assert correct.tolist() == [True, True, False, False, False]
    cfg = rcfg("bbnr", activation_threshold=0.0, fill_policy="random", seed=3)
    extra, summary = policy_priority(cfg, case_scores(NOISY_LABELS), 3, eng, eng)
    assert extra["bbnr_deleted"].tolist() == [0, 0, 1, 0, 0]
    assert summary == dict(baseline_set_size=4, bbnr_candidates=3, bbnr_deleted=1)
    assert extra["bbnr"][2] == extra["bbnr"].min()  # the deleted case is ranked last


def test_enn_hand_example_and_fill_order():
    """Wilson editing on the same matrix: problems q2, q3, q4 are answered wrongly, so c2, c3, c4 are edited.
    Survivors come first; inside each group the order of the existing ``random`` policy (same seed)."""
    eng = ActivationEngine.from_arrays(NOISY, NOISY_LABELS, NOISY_LABELS, top_k=1)
    cfg = rcfg("enn", fill_policy="random", seed=7)
    sc = case_scores(NOISY_LABELS)
    extra, summary = policy_priority(cfg, sc, 2, eng, eng)
    assert extra["enn_edited"].tolist() == [0, 0, 1, 1, 1] and summary["enn_edited"] == 3
    draw = np.random.default_rng(7).random(5)
    order = list(np.argsort(-extra["enn"], kind="stable"))
    assert set(order[:2]) == {0, 1}
    assert order[:2] == sorted([0, 1], key=lambda j: -draw[j]) and order[2:] == sorted([2, 3, 4], key=lambda j: -draw[j])
    sc.extra.update(extra)
    assert sorted(select_active_set(sc, 2, cfg, None).keep_slots.tolist()) == [0, 1]


def test_loo_value_hand_example():
    """Uniform activation over c0, c1 (class 0) and c2 (class 1), one query of class 0: P = (2/3, 1/3).
    Removing c0 gives (1/2, 1/2): value log(4/3); removing c2 gives (1, 0): value log(2/3)."""
    eng = ActivationEngine.from_arrays(np.zeros((1, 3)), [0, 0, 1], [0])
    extra, summary = policy_priority(rcfg("loo_value"), case_scores([0, 0, 1]), 2, eng)
    assert np.allclose(extra["loo_value"], [math.log(4 / 3), math.log(4 / 3), math.log(2 / 3)])
    assert np.isclose(summary["reference_loss"], -math.log(2 / 3))


def brute_shapley(Z, M, K):
    """Exact Shapley values (all orderings) of U(S) = (1/K) sum of M over the K highest-scoring members of S."""
    nq, n = Z.shape
    out = np.zeros(n)
    perms = list(itertools.permutations(range(n)))
    for q in range(nq):
        def U(S):
            S = [c for c in S if np.isfinite(Z[q, c])]
            top = sorted(S, key=lambda c: (-Z[q, c], c))[:K]
            return sum(M[q, c] for c in top) / K
        for p in perms:
            for i, c in enumerate(p):
                out[c] += U(p[:i + 1]) - U(p[:i])
    return out / (len(perms) * nq)


@pytest.mark.parametrize("K", [1, 2, 3])
def test_knn_shapley_equals_brute_force(K):
    rng = np.random.default_rng(K)
    Z = rng.normal(size=(4, 5))
    Z[0, 1] = Z[2, 3] = NEG  # leave-one-out style exclusions
    labels, y = np.array([0, 1, 0, 1, 1]), np.array([0, 1, 1, 0])
    eng = ActivationEngine.from_arrays(Z, labels, y, top_k=K)
    M = (labels[None, :] == y[:, None]).astype(float)
    assert np.allclose(knn_shapley(eng), brute_shapley(Z, M, K))


def test_knn_shapley_regression_uses_the_solve_tolerance():
    Z = np.array([[3.0, 2.0, 1.0]])
    eng = ActivationEngine.from_arrays(Z, [0.0, 0.4, 2.0], [0.1], task_type="regression", top_k=1)
    M = np.array([[1.0, 1.0, 0.0]])  # |error| <= 0.5 solves
    assert np.allclose(knn_shapley(eng, tolerance=0.5), brute_shapley(Z, M, 1))


def hinge(P, y, margin):
    wrong = np.delete(P, y)
    return max(0.0, margin + wrong.max() - P[y])


@pytest.mark.parametrize("top_k", [2, None])
def test_energy_compress_matches_a_brute_force_deletion_sequence(top_k):
    """Each step removes the eligible case whose removal raises the mean hinge loss least (recomputed from scratch
    with engine.predict here); cohort floors are never crossed."""
    rng = np.random.default_rng(5)
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2])
    Z = rng.normal(size=(12, 8))
    y = rng.integers(0, 3, 12)
    eng = ActivationEngine.from_arrays(Z, labels, y, top_k=top_k)
    cfg = rcfg("energy_compress", energy_margin=0.2, min_per_cohort=1)
    extra, summary = policy_priority(cfg, case_scores(labels), 3, eng)
    active = np.ones(8, dtype=bool)
    order = []
    while active.sum() > 3:
        def mean_loss(mask):
            P = eng.predict(mask)
            return np.mean([hinge(P[q], y[q], 0.2) for q in range(12)])
        base = mean_loss(active)
        best = None
        for c in np.flatnonzero(active):
            if (labels[active] == labels[c]).sum() <= 1:
                continue
            m = active.copy()
            m[c] = False
            comp = mean_loss(m) - base
            if best is None or comp <= best[0] + 1e-12:  # exact ties: the largest slot (tie key) goes first
                best = (comp, c)
        order.append(best[1])
        active[best[1]] = False
    assert extra["energy_removal_step"][order].tolist() == list(range(5))
    assert summary["energy_steps"] == 5 and summary["baseline_set_size"] == 3
    assert sorted(np.argsort(-extra["energy_compress"])[:3].tolist()) == sorted(np.flatnonzero(active).tolist())
    assert np.isclose(summary["energy_final_reference_loss"],
                      np.mean([hinge(eng.predict(active)[q], y[q], 0.2) for q in range(12)]))


def test_energy_compress_breaks_exact_ties_like_the_retention_ranking():
    """Cases 2..5 enter no top-1 neighborhood (competence exactly 0, also after earlier deletions): they leave in
    decreasing tie-key order, so the smaller keys survive, as in select_active_set (smaller key kept first)."""
    labels = np.array([0, 1, 0, 1, 0, 1])
    Z = np.full((4, 6), -5.0)
    Z[[0, 1], 0] = 0.0
    Z[[2, 3], 1] = 0.0
    y = np.array([0, 0, 1, 1])
    eng = ActivationEngine.from_arrays(Z, labels, y, top_k=1)
    extra, summary = policy_priority(rcfg("energy_compress", energy_margin=0.2), case_scores(labels), 4, eng)
    step = extra["energy_removal_step"]
    assert step[5] == 0 and step[4] == 1 and step[[0, 1, 2, 3]].tolist() == [-1, -1, -1, -1]
    ids = np.array([50, 40, 30, 20, 10, 0])  # reversed keys reverse the order among the zero-competence cases
    extra, _ = policy_priority(rcfg("energy_compress", energy_margin=0.2), case_scores(labels, ids=ids), 4, eng)
    assert extra["energy_removal_step"][2] == 0 and extra["energy_removal_step"][3] == 1


# --------------------------------------------------------------------------- config validation and runner glue


def test_policy_names_and_settings_are_opt_in():
    assert set(BASELINE_POLICIES) <= set(POLICIES)
    for name in ("solve_tolerance", "fill_policy", "energy_margin"):
        assert getattr(rcfg("provenance_only"), name) is None
    with pytest.raises(ValueError, match="read only by"):
        rcfg("provenance_only", fill_policy="random").validate()
    with pytest.raises(ValueError, match="fill_policy"):
        rcfg("enn").validate()
    with pytest.raises(ValueError, match="activation_threshold"):
        rcfg("crr").validate()
    with pytest.raises(ValueError, match="priorities are computed"):
        select_active_set(case_scores([0, 1]), 1, rcfg("knn_shapley"), None)
    rcfg("enn", fill_policy="bias_current", solve_tolerance=0.5).validate()


def test_regression_policies_need_a_solve_tolerance_and_energy_a_margin():
    model, X, y = tiny_core("regression", True)
    ref = MaintenanceReference(X, y, torch.arange(len(y)), 0.5)
    sc = case_scores(np.zeros(len(y)))
    with pytest.raises(ValueError, match="solve_tolerance"):
        baseline_priority(model, ref, rcfg("enn", fill_policy="random"), sc, 10)
    cm, Xc, yc = tiny_core("classification", True)
    with pytest.raises(ValueError, match="energy_margin"):
        baseline_priority(cm, MaintenanceReference(Xc, yc, torch.arange(len(yc)), None), rcfg("energy_compress"),
                          case_scores(yc.numpy()), 10)


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("policy", BASELINE_POLICIES)
def test_every_policy_runs_through_run_maintenance(task, policy):
    model, X, y = tiny_core(task, True, n=40)
    ref = MaintenanceReference(X, y, torch.arange(len(y)), 0.5)
    reg_bins = None if task == "classification" else np.quantile(y.numpy(), [0.5])
    kw = dict(activation_threshold=0.05, fill_policy="random", solve_tolerance=0.5 if task == "regression" else None,
              energy_margin=0.2 if task == "classification" else None)
    cfg = RetentionConfig(policy=policy, min_per_cohort=2, seed=1, **kw)
    store = CaseStatisticsStore(task)
    audit_provenance(model, X, y, store, exclude_identical=True, counterfactual=False, query_case_ids=torch.arange(40),
                     reg_bins=reg_bins, regression_success_tolerance=0.5)
    res = run_maintenance(model, store, CaseArchive(), 12, cfg, ScoreConfig(1.0, 2, 0.1), step=0, run_id="t",
                          reg_bins=reg_bins, reference=ref)
    assert model.case_count() == 12 and res["summary"]["n_after"] == 12
    assert res["summary"]["engine_check_max_abs_diff"] < 1e-5 and res["summary"]["n_reference_queries"] == 40
    if policy in {"rc_cnn", "crr", "bbnr", "enn"}:
        assert res["summary"]["problem_engine_shared"] == 1
    kept = {e["case_id"] for e in res["events"] if e["action"] == "keep"}
    assert kept == set(res["kept_case_ids"]) and all(policy in e for e in res["events"])


# --------------------------------------------------------------------------- efficiency and reliability


def test_case_memory_and_efficiency_scalars():
    model, X, y = tiny_core("classification", True, n=20)
    # float32 cases 20x3, labels 20x3, biases 20, negative_weights 20, glocal_weights 20x1; int64 case_ids 20
    assert model.glocal_weights.shape == (20, 1)
    expected = 4 * (60 + 60 + 20 + 20 + 20) + 8 * 20
    assert case_memory_bytes(model) == expected == 880
    with pytest.raises(ValueError):
        check_efficiency_cfg({"lookup_repeats": 2})
    small = build_model(X[:5], y[:5], CoreConfig(task_type="classification", top_k=4, seed=0), 3)
    m = efficiency_metrics(small, model, X, {"lookup_repeats": 2, "single_queries": 3}, audit_seconds=0.5,
                           maintenance_seconds=0.25)
    assert m["eff_maintenance_overhead_seconds"] == 0.75 and m["eff_lookup_single_count"] == 6
    assert m["eff_lookup_single_median_seconds"] > 0 and m["eff_fullmem_lookup_batch_queries"] == 20
    assert m["eff_memory_saved_bytes"] == expected - case_memory_bytes(small)


def test_spearman_and_extreme_overlap():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 1, 1], [1, 2, 3]) is None
    assert spearman([1, 2, 2, 3], [1, 3, 2, 4]) == pytest.approx(0.9486832980505138)
    full, sub = np.arange(10.0), np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 8.0])
    tie = np.arange(10)
    assert extreme_overlap(full, sub, tie, 0.2, top=True) == 1.0 and extreme_overlap(full, sub, tie, 0.1, top=True) == 0.0
    assert extreme_overlap(full, sub, tie, 0.1, top=False) == 1.0


def test_ch_reliability_full_fraction_is_exact_and_subsets_are_nested():
    model, X, y = tiny_core("classification", True, n=40)
    ref = MaintenanceReference(X, y, torch.arange(40), None)
    rel = {"fractions": [0.25, 0.5, 1.0], "repeats": 2, "decile": 0.1}
    check_reliability_cfg(rel)
    with pytest.raises(ValueError):
        check_reliability_cfg({"fractions": [0.5], "repeats": 1, "decile": 0.1})
    m, details, store = ch_reliability(model, ref, ScoreConfig(1.0, 2, 0.1), rel, seed=3, tie=np.arange(40))
    assert m["rel_f100_spearman_Q"] == pytest.approx(1.0) and m["rel_f100_top_overlap_C"] == 1.0
    assert m["rel_f25_n_queries"] == 10 and m["rel_f50_n_queries"] == 20
    assert m["rel_f25_evidenced_fraction"] <= m["rel_f50_evidenced_fraction"] <= m["rel_f100_evidenced_fraction"]
    total = sum(store.get(i).positive_support + store.get(i).harmful_support for i in range(40))
    assert total == pytest.approx(40.0)  # the full-set store: one unit of activation per reference query
    assert len(details["per_repeat"]["0.25"]) == 2
    m2, _, _ = ch_reliability(model, ref, ScoreConfig(1.0, 2, 0.1), rel, seed=3, tie=np.arange(40))
    assert m2 == m  # deterministic per seed


# --------------------------------------------------------------------------- end to end through the runner


def small_cfg(conditions, **extra):
    from tools.t1_run import load_config
    cfg = load_config(REPO / "configs" / "t1" / "alignment_candidates_small.yaml")
    cfg["core"].update(epochs=3, patience=3, batch_size=8, top_k=3, bias_init_k=3)
    cfg["protocol"]["finetune_epochs"] = 2
    cfg["datasets"] = [{"name": "iris", "task_type": "classification", "max_train": 24}]
    cfg["seeds"] = [0]
    cfg["K"] = [0.5]
    cfg["conditions"] = conditions
    cfg.update(extra)
    return cfg


def test_runner_opt_in_keys_add_records_and_change_no_policy_number(tmp_path, monkeypatch):
    from t1_units import study_runner
    plain = small_cfg([{"policy": "random"}, {"policy": "provenance_only"}])
    ra = study_runner(plain, tmp_path / "a", monkeypatch, study="a")
    ra.run_retention(plain["datasets"][0], 0)
    keyed = small_cfg([{"policy": "random"}, {"policy": "provenance_only"}],
                      efficiency={"lookup_repeats": 2, "single_queries": 2},
                      reliability={"fractions": [0.5, 1.0], "repeats": 2, "decile": 0.1})
    rb = study_runner(keyed, tmp_path / "b", monkeypatch, study="b")
    rb.run_retention(keyed["datasets"][0], 0)
    ua = {u.condition: u for u in units(tmp_path / "a" / "runs")}
    ub = {u.condition: u for u in units(tmp_path / "b" / "runs")}
    assert set(ua) == {"random_K0.5", "provenance_only_K0.5"} and set(ub) == set(ua) | {"ch_reliability"}
    for c in ua:
        ma, mb = ua[c].json("metrics.json"), ub[c].json("metrics.json")
        assert not any(k.startswith("eff_") for k in ma)
        assert mb["eff_lookup_single_count"] == 4 and mb["eff_case_memory_bytes"] < mb["eff_fullmem_case_memory_bytes"]
        assert {k: v for k, v in mb.items() if not k.startswith("eff_")} == ma
    rel = ub["ch_reliability"].json("metrics.json")
    assert rel["rel_f100_spearman_C"] == pytest.approx(1.0) and rel["rel_f50_n_queries"] == 12
    assert ub["ch_reliability"].json("reliability.json")["repeats"] == 2


@pytest.mark.parametrize("policy", ["rc_cnn", "energy_compress", "knn_shapley"])
def test_runner_records_a_baseline_policy_unit(policy, tmp_path, monkeypatch):
    from t1_units import study_runner
    cfg = small_cfg([{"policy": policy}])
    cfg["retention"].update(activation_threshold=0.05, energy_margin=0.2)
    if policy != "energy_compress":
        cfg["retention"].pop("energy_margin")
    r = study_runner(cfg, tmp_path, monkeypatch)
    r.run_retention(cfg["datasets"][0], 0)
    (u,) = units(tmp_path / "runs")
    m = u.json("metrics.json")
    assert u.condition == f"{policy}_K0.5" and m["n_cases"] == 12 and m["sel_engine_check_max_abs_diff"] < 1e-5
    events = u.jsonl("case_maintenance.jsonl")
    assert sum(e["action"] == "keep" for e in events) == 12 and all(policy in e for e in events)


def test_pack_p2_configs_read_every_key():
    """Every config of the pack: the baseline settings are read by its conditions, the opt-in keys are valid and
    the 8 recorded policies are absent from the new-policy files."""
    root = REPO / "configs" / "t1" / "handoff_20261126" / "p2-retention"
    files = sorted(root.glob("*.yaml"))
    if not files:
        pytest.skip("pack p2 configs not written")
    from tools.t1_run import load_config, retention_cfg_for
    for f in files:
        cfg = load_config(f)
        assert cfg["seeds"] == [30, 31, 32] and cfg["type"] == "retention"
        assert all(d["name"] != "tabarena_363697" for d in cfg["datasets"])
        policies = [c["policy"] for c in cfg["conditions"]]
        if "efficiency" in cfg:
            check_efficiency_cfg(cfg["efficiency"])
        if "reliability" in cfg:
            check_reliability_cfg(cfg["reliability"])
        for c in cfg["conditions"]:
            retention_cfg_for(c, cfg, 30).validate()
        if set(policies) & set(BASELINE_POLICIES):
            assert set(policies) == set(BASELINE_POLICIES)
            assert cfg["retention"]["fill_policy"] == "random"
            reg = all(d["task_type"] == "regression" for d in cfg["datasets"])
            assert ("solve_tolerance" in cfg["retention"]) == reg and ("energy_margin" in cfg["retention"]) == (not reg)
        text = f.read_text()
        assert "exploratory, our choice, not a PI decision" in text
