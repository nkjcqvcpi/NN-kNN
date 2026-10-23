"""RetentionConfig.tie_break: frozen case-ID default and the seeded_random option (review F3).

The snapshots were computed from the pre-option implementation; ``case_id`` must
reproduce them exactly because step G (docs/T1_PREREG_20261013.md) is frozen on it.
"""

import hashlib
import json
import os

os.environ.setdefault("NNKNN_DEVICE", "cpu")

import numpy as np
import pytest
import torch

import model.t1.candidates as candidates
from model.common.core import CoreConfig, build_model
from model.t1.maintenance import CaseArchive, run_maintenance
from model.t1.provenance import CaseScores, CaseStatisticsStore, ScoreConfig
from model.t1.retention import RetentionConfig, select_active_set


def scores_fixture(seed=0, n=40):
    """Class-ordered rows, unsorted stable IDs, exact ties in Q, bias, B, coverage and distance."""
    rng = np.random.default_rng(seed)
    ids = rng.permutation(np.arange(100, 100 + 3 * n))[:n]
    cohorts = ["class_0"] * 17 + ["class_1"] * 20 + ["class_2"] * 3
    R = rng.integers(0, 4, n).astype(float)
    A = np.where(R > 0, rng.random(n), 0.0)
    C = np.floor(R * rng.random(n))
    H = R - C
    Q = (C + 1) / (C + H + 2)
    evidenced = (R >= 2) & (A >= 0.1)
    bias = np.where(rng.random(n) < 0.5, 0.0, rng.normal(size=n))
    B = np.round(rng.random(n), 1)
    cov = np.where(rng.random(n) < 0.4, 0.0, rng.random(n))
    human = np.zeros(n, dtype=bool)
    human[5] = True
    pts = rng.normal(size=(n, 2))
    pts[1], pts[7] = pts[0], pts[3]
    dist = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
    extra = {"human_protected": human, "coverage_reachability": cov}
    return CaseScores(ids, cohorts, R, A, C, H, Q, B, evidenced, bias, extra), dist


def permuted(sc, dist, p):
    extra = {k: v[p] for k, v in sc.extra.items()}
    return CaseScores(sc.case_ids[p], [sc.cohorts[i] for i in p], sc.R[p], sc.A[p], sc.C[p], sc.H[p], sc.Q[p],
                      sc.B[p], sc.evidenced[p], sc.bias[p], extra), dist[np.ix_(p, p)]


CASES = {
    "provenance_only": (dict(policy="provenance_only", min_per_cohort=2), 15, False),
    "provenance_only_boundary": (dict(policy="provenance_only", min_per_cohort=0, boundary_protect_fraction=0.1), 10, True),
    "bias_current_rare": (dict(policy="bias_current", min_per_cohort=2, protect_rare_cohort_below=3), 15, False),
    "bias_normalized": (dict(policy="bias_normalized", min_per_cohort=1), 12, False),
    "kcenter": (dict(policy="kcenter", min_per_cohort=2), 15, True),
    "coverage_reachability": (dict(policy="coverage_reachability", min_per_cohort=1, activation_threshold=0.1,
                                   zero_reachability="zero"), 12, False),
    "random": (dict(policy="random", min_per_cohort=2, seed=3), 15, False),
    "stratified": (dict(policy="stratified", min_per_cohort=1, seed=4), 20, False),
    "full_memory": (dict(policy="full_memory", min_per_cohort=0), 5, False),
}
# Pre-option implementation: (keep_slots, digest of the selection-owned record fields).
SNAPSHOT = {
    "provenance_only": ([2, 4, 5, 6, 7, 9, 10, 11, 18, 22, 25, 33, 36, 37, 39], "a36ab7f36101da19"),
    "provenance_only_boundary": ([4, 5, 6, 11, 12, 18, 22, 25, 33, 39], "cfcf999ae6e5b46d"),
    "bias_current_rare": ([0, 2, 4, 5, 6, 10, 13, 19, 20, 27, 31, 33, 37, 38, 39], "aa6b7b1d5e800d63"),
    "bias_normalized": ([1, 2, 4, 5, 13, 17, 20, 21, 23, 24, 26, 39], "59617ff98ac480a4"),
    "kcenter": ([2, 4, 5, 14, 16, 21, 23, 24, 29, 30, 33, 34, 36, 37, 38], "709c24322045d09f"),
    "coverage_reachability": ([1, 3, 5, 6, 9, 14, 15, 18, 22, 23, 24, 37], "832c4f5430eb08e3"),
    "random": ([2, 5, 8, 14, 15, 17, 18, 21, 24, 27, 29, 32, 33, 37, 38], "4e44726e5634e54b"),
    "stratified": ([0, 2, 5, 6, 8, 10, 14, 16, 17, 19, 20, 21, 22, 27, 31, 32, 33, 35, 37, 38], "8e7703f631db1e46"),
    "full_memory": (list(range(40)), "86c9772f90eef0ba"),
}
# random/stratified rank by row-indexed draws, so only the tie key is order-free there.
ORDER_FREE = [c for c in CASES if c not in {"random", "stratified"}]
TIED = ["provenance_only", "provenance_only_boundary", "bias_current_rare", "bias_normalized", "kcenter",
        "coverage_reachability"]
TIED_AT_CUT = {"provenance_only_boundary": 12, "coverage_reachability": 30}  # snapshot K cuts above the ties


def record_digest(decisions):
    rows = [(d["case_id"], d["action"], d["protected"], d["policy_score"], d["reason_codes"], d["tie_break"])
            for d in decisions]
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()[:16]


def select(name, sc=None, dist=None, K=None, **over):
    kw, K0, use_dist = CASES[name]
    if sc is None:
        sc, dist = scores_fixture()
    return select_active_set(sc, K or K0, RetentionConfig(**{**kw, **over}), dist if use_dist else None)


def kept_ids(sc, sel):
    return sorted(sc.case_ids[sel.keep_slots].tolist())


# ------------------------------------------------------------------ fixed-K selection


@pytest.mark.parametrize("name", list(CASES))
@pytest.mark.parametrize("over", [{}, {"tie_break": "case_id"}])
def test_case_id_default_matches_pre_option_snapshot(name, over):
    sel = select(name, **over)
    assert sel.keep_slots.tolist() == SNAPSHOT[name][0]
    assert record_digest(sel.decisions) == SNAPSHOT[name][1]
    assert {d["tie_break"] for d in sel.decisions} == {"case_id_ascending"}


def test_tie_break_is_validated():
    with pytest.raises(ValueError, match="tie_break"):
        RetentionConfig("provenance_only", 0, tie_break="row_order").validate()
    for seed in (-1, 1.5, True, "3"):
        with pytest.raises(ValueError, match="nonnegative integer seed"):
            RetentionConfig("provenance_only", 0, seed=seed, tie_break="seeded_random").validate()
    RetentionConfig("provenance_only", 0, seed=np.int64(4), tie_break="seeded_random").validate()
    RetentionConfig("provenance_only", 0, seed=-1).validate()  # seed only matters to seeded_random here


@pytest.mark.parametrize("name", TIED)
def test_seeded_random_is_deterministic_and_recorded(name):
    a, b = select(name, seed=7, tie_break="seeded_random"), select(name, seed=7, tie_break="seeded_random")
    assert np.array_equal(a.keep_slots, b.keep_slots)
    assert {d["tie_break"] for d in a.decisions} == {"seeded_random(seed=7)"}


@pytest.mark.parametrize("name", TIED)
def test_seeded_random_differs_across_seeds_when_ties_decide(name):
    K = TIED_AT_CUT.get(name)
    kept = {tuple(select(name, K=K, seed=s, tie_break="seeded_random").keep_slots) for s in range(6)}
    assert len(kept) > 1


@pytest.mark.parametrize("name", ORDER_FREE)
def test_seeded_random_is_invariant_to_case_order(name):
    sc, dist = scores_fixture()
    psc, pdist = permuted(sc, dist, np.random.default_rng(11).permutation(len(sc.case_ids)))
    for seed in (0, 5):
        a = select(name, sc, dist, seed=seed, tie_break="seeded_random")
        b = select(name, psc, pdist, seed=seed, tie_break="seeded_random")
        assert kept_ids(sc, a) == kept_ids(psc, b)


def test_seeded_random_never_changes_untied_selections():
    sc, dist = scores_fixture()
    rng = np.random.default_rng(1)
    n = len(sc.case_ids)
    sc.bias[:], sc.B[:], sc.extra["coverage_reachability"][:] = rng.normal(size=n), rng.random(n), rng.random(n)
    sc.Q[:], sc.evidenced[:] = rng.random(n), True
    dist = np.abs(dist + rng.normal(scale=1e-3, size=dist.shape))
    for name in ORDER_FREE + ["random", "stratified"]:
        if name == "kcenter":  # its first pick and floors rank an all-zero score: always tied
            continue
        for seed in (0, 9):  # random/stratified draw their ranking from the same seed
            base = select(name, sc, dist, seed=seed)
            alt = select(name, sc, dist, seed=seed, tie_break="seeded_random")
            assert np.array_equal(base.keep_slots, alt.keep_slots), name
            assert [(d["action"], d["reason_codes"], d["policy_score"]) for d in alt.decisions] == \
                   [(d["action"], d["reason_codes"], d["policy_score"]) for d in base.decisions]


# ------------------------------------------------------------------ dynamic maintenance


SCFG = ScoreConfig(smoothing=1, min_retrieval_count=0, min_activation_mass=0)
MODEL_IDS = np.array([7, 3, 11, 0, 9, 5, 2, 10, 4, 8, 1, 6]) + 20  # rows 0-5 class 0, rows 6-11 class 1
ROW_PERM = np.random.default_rng(2).permutation(len(MODEL_IDS))


def model_fixture(order=None):
    g = torch.Generator().manual_seed(0)
    X = torch.cat([torch.randn(6, 2, generator=g) + 2, torch.randn(6, 2, generator=g) - 2])
    y = torch.tensor([0] * 6 + [1] * 6)
    ids = torch.as_tensor(MODEL_IDS)
    if order is not None:
        X, y, ids = X[order], y[order], ids[order]
    m = build_model(X, y, CoreConfig(task_type="classification", top_k=2), 2)
    with torch.no_grad():
        m.case_ids[:len(ids)] = ids
    return m


def fake_removal(values_of):
    def removal_influence(model, reference, *, cached_threshold=None):
        ids = model.active_case_ids().cpu().numpy()
        v = values_of(ids)
        return dict(full=v, cached=v, baseline_loss=0.0, n_reference_queries=1,
                    cache_query_counts=np.zeros(len(ids), dtype=int), cache_query_ids=[[] for _ in ids],
                    candidate_rerun_queries=0, validation_rerun_queries=0, cache_max_absolute_error=0.0)
    return removal_influence


def sync_run(order=None, **over):
    """All-tied provenance_only (every Q = 0.5) through run_maintenance."""
    m = model_fixture(order)
    res = run_maintenance(m, CaseStatisticsStore(), CaseArchive(), 6, RetentionConfig("provenance_only", 2, **over),
                          SCFG, step=0, run_id="t")
    return sorted(m.active_case_ids().tolist()), res


def removal_run(order=None, **over):
    m = model_fixture(order)
    res = run_maintenance(m, CaseStatisticsStore(), CaseArchive(), 6,
                          RetentionConfig("removal_influence", 2, allowed_loss_increase=1.0, **over), SCFG,
                          step=0, run_id="t", reference=object())
    return sorted(res["kept_case_ids"]), res


def test_maintenance_case_id_default_matches_pre_option_snapshot(monkeypatch):
    for over in ({}, {"tie_break": "case_id"}):
        kept, res = sync_run(**over)
        assert kept == [20, 21, 22, 23, 24, 25] and record_digest(res["events"]) == "0dca858ddd3a1da3"
    monkeypatch.setattr(candidates, "removal_influence", fake_removal(lambda ids: np.zeros(len(ids))))
    for over in ({}, {"tie_break": "case_id"}):
        _, res = removal_run(**over)
        assert res["kept_case_ids"] == [27, 31, 29, 30, 28, 26]
        assert [t["candidate_case_id"] for t in res["scoring_trace"]] == [20, 21, 22, 23, 24, 25]
        assert {e["tie_break"] for e in res["events"]} == {"case_id_ascending"}


def test_sync_maintenance_passes_seeded_random_through():
    kept, res = sync_run(seed=3, tie_break="seeded_random")
    assert {e["tie_break"] for e in res["events"]} == {"seeded_random(seed=3)"}
    assert kept == sync_run(seed=3, tie_break="seeded_random")[0] == sync_run(ROW_PERM, seed=3, tie_break="seeded_random")[0]
    assert len({tuple(sync_run(seed=s, tie_break="seeded_random")[0]) for s in range(6)}) > 1


def test_sequential_removal_passes_seeded_random_through(monkeypatch):
    monkeypatch.setattr(candidates, "removal_influence", fake_removal(lambda ids: np.zeros(len(ids))))
    kept, res = removal_run(seed=3, tie_break="seeded_random")
    assert {e["tie_break"] for e in res["events"]} == {"seeded_random(seed=3)"}
    assert kept == removal_run(seed=3, tie_break="seeded_random")[0] == removal_run(ROW_PERM, seed=3, tie_break="seeded_random")[0]
    assert len({tuple(removal_run(seed=s, tie_break="seeded_random")[0]) for s in range(6)}) > 1
    # untied removal scores: the tie key is never consulted
    monkeypatch.setattr(candidates, "removal_influence", fake_removal(lambda ids: (ids * 7919 % 101) / 101.0))
    base = removal_run()[1]["scoring_trace"]
    alt = removal_run(seed=3, tie_break="seeded_random")[1]["scoring_trace"]
    assert [t["candidate_case_id"] for t in base] == [t["candidate_case_id"] for t in alt]


def test_run_maintenance_reports_the_kept_ids_it_kept():
    """kept_case_ids must be read before compact_cases overwrites the case_ids buffer (a CPU view)."""
    for over in ({}, {"tie_break": "seeded_random", "seed": 3}):
        kept, res = sync_run(**over)
        assert sorted(res["kept_case_ids"]) == kept
