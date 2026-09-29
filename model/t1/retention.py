"""Constrained retention under fixed capacity K (T1 plan section 10).

One policy interface for every case role. A policy only *ranks*; the shared
``select_active_set`` enforces the three stages:

  Stage 1  protection and evidence: coverage floors per cohort, rare /
           boundary indicators, insufficient-exposure marking, trust
           components, observed utility, redundancy in the learned geometry.
  Stage 2  routine eviction eligibility (reason codes are logged per case).
  Stage 3  fill capacity: protected cases first, then the policy's ranking,
           with deterministic, logged tie-breaking (case_id ascending).

Policies compared (section 10, in order):
  full_memory, random, stratified, bias_current, provenance_only, bias_normalized,
  trust, utility_redundancy, trust_utility, trust_utility_coverage
plus the optional selection baseline ``kcenter`` (greedy core-set).

Weights for the combined policies are unresolved PI decisions; they are passed
explicitly through ``RetentionConfig`` and never defaulted silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .provenance import CaseScores

POLICIES = (
    "full_memory",
    "random",
    "stratified",
    "bias_current",
    "provenance_only",
    "bias_normalized",
    "trust",
    "utility_redundancy",
    "trust_utility",
    "trust_utility_coverage",
    "kcenter",
)


@dataclass
class RetentionConfig:
    policy: str
    min_per_cohort: int  # coverage floor per class/cohort
    protect_rare_cohort_below: int = 0  # cohorts with <= this many cases are fully protected
    boundary_protect_fraction: float = 0.0  # fraction of cases closest to another cohort that are protected
    w_trust: float | None = None
    w_utility: float | None = None
    w_redundancy: float | None = None
    w_coverage: float | None = None
    seed: int = 0
    notes: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.policy not in POLICIES:
            raise ValueError(f"Unknown retention policy {self.policy!r}; choose from {POLICIES}")
        need = {
            "utility_redundancy": ("w_utility", "w_redundancy"),
            "trust_utility": ("w_trust", "w_utility"),
            "trust_utility_coverage": ("w_trust", "w_utility", "w_redundancy", "w_coverage"),
        }.get(self.policy, ())
        missing = [k for k in need if getattr(self, k) is None]
        if missing:
            raise ValueError(f"Policy {self.policy} requires explicit {missing} (unresolved PI decision; no hidden default).")


@dataclass
class SelectionResult:
    keep_slots: np.ndarray  # active slots kept, in selection order
    decisions: list[dict[str, Any]]  # one per active case (reason codes, ranks, component scores)
    summary: dict[str, Any]


def _percentile(values: np.ndarray) -> np.ndarray:
    n = len(values)
    if n <= 1:
        return np.full(n, 0.5)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(n)
    ranks[order] = np.arange(n)
    uniq, inv = np.unique(values, return_inverse=True)
    mid = np.bincount(inv, weights=ranks) / np.bincount(inv)
    return mid[inv] / (n - 1)


def select_active_set(
    scores: CaseScores,
    K: int,
    cfg: RetentionConfig,
    dist: np.ndarray | None,
) -> SelectionResult:
    """Choose which active slots to keep. ``dist`` = [N, N] learned case-case distance."""
    cfg.validate()
    n = len(scores.case_ids)
    ids = scores.case_ids.astype(np.int64)
    cohorts = np.asarray(scores.cohorts)
    rng = np.random.default_rng(cfg.seed)
    tie = ids  # deterministic tie-break key: smaller case_id wins

    # ---------------- Stage 1: protection and evidence ----------------
    protected = np.zeros(n, dtype=bool)
    reasons: list[list[str]] = [[] for _ in range(n)]
    uniq = sorted(set(cohorts.tolist()))
    cohort_sizes = {c: int((cohorts == c).sum()) for c in uniq}
    # rare cohorts
    for c in uniq:
        if cohort_sizes[c] <= cfg.protect_rare_cohort_below:
            for j in np.nonzero(cohorts == c)[0]:
                protected[j] = True
                reasons[j].append("protect_rare_cohort")
    # boundary cases: smallest distance to a case of another cohort
    nearest_enemy = np.full(n, np.inf)
    if dist is not None and len(uniq) > 1:
        for j in range(n):
            other = cohorts != cohorts[j]
            if other.any():
                nearest_enemy[j] = float(dist[j, other].min())
        if cfg.boundary_protect_fraction > 0:
            m = int(np.floor(cfg.boundary_protect_fraction * n))
            if m > 0:
                order = np.lexsort((tie, nearest_enemy))[:m]
                for j in order:
                    if not protected[j]:
                        protected[j] = True
                        reasons[j].append("protect_boundary")
    # redundancy: max similarity to another case of the same cohort, in learned geometry
    redundancy = np.zeros(n)
    nearest_same = np.full(n, np.inf)
    if dist is not None:
        for j in range(n):
            same = (cohorts == cohorts[j]).copy()
            same[j] = False
            if same.any():
                nearest_same[j] = float(dist[j, same].min())
        finite = np.isfinite(nearest_same)
        scale = np.median(nearest_same[finite]) if finite.any() else 1.0
        scale = scale if scale > 0 else 1.0
        redundancy = np.where(finite, np.exp(-nearest_same / scale), 0.0)
    utility = _percentile(scores.A)  # observed utility = activation mass percentile
    evidenced = scores.evidenced
    Q_eff = np.where(evidenced, scores.Q, 0.5)  # unobserved evidence is not negative evidence
    from .provenance import trustworthiness

    alpha = float(cfg.notes.get("alpha", 0.5))
    trust_mode = str(cfg.notes.get("trust_mode", "geometric"))
    T_eff = np.array([trustworthiness(Q_eff[j], scores.B[j], alpha, trust_mode) for j in range(n)])

    # ---------------- Stage 2: eligibility reason codes ----------------
    red_hi = redundancy >= np.median(redundancy) if n else redundancy
    util_lo = utility <= 0.25
    trust_lo = evidenced & (T_eff <= np.quantile(T_eff, 0.25) if n else False)
    eligible = np.zeros(n, dtype=bool)
    for j in range(n):
        if protected[j]:
            continue
        if not evidenced[j]:
            reasons[j].append("insufficient_exposure")
        if util_lo[j] and red_hi[j]:
            eligible[j] = True
            reasons[j].append("eligible_low_utility_high_redundancy")
        if trust_lo[j] and (util_lo[j] or red_hi[j]):
            eligible[j] = True
            reasons[j].append("eligible_low_trust")

    # ---------------- Stage 3: fill capacity ----------------
    K = int(min(K, n))
    if cfg.policy == "full_memory":
        K = n
    keep: list[int] = []
    chosen = np.zeros(n, dtype=bool)

    def take(j: int, why: str) -> None:
        chosen[j] = True
        keep.append(j)
        reasons[j].append(why)

    # coverage floor per cohort first (deterministic: best policy score within cohort)
    base_score = _policy_static_score(cfg, scores, utility, redundancy, T_eff, rng, n)
    for j in np.nonzero(protected)[0]:
        if len(keep) < K:
            take(int(j), "keep_protected")
    for c in uniq:
        members = np.nonzero((cohorts == c) & ~chosen)[0]
        have = int(((cohorts == c) & chosen).sum())
        need = max(0, min(cfg.min_per_cohort, cohort_sizes[c]) - have)
        if need and members.size:
            order = members[np.lexsort((tie[members], -base_score[members]))][:need]
            for j in order:
                if len(keep) < K:
                    take(int(j), "keep_coverage_floor")

    if cfg.policy in ("utility_redundancy", "trust_utility_coverage", "kcenter") and dist is not None:
        _greedy_fill(cfg, keep, chosen, K, dist, scores, utility, T_eff, tie, reasons)
    elif cfg.policy == "stratified":
        _stratified_fill(keep, chosen, K, cohorts, rng, tie, reasons)
    else:
        rest = np.nonzero(~chosen)[0]
        order = rest[np.lexsort((tie[rest], -base_score[rest]))]
        for j in order:
            if len(keep) >= K:
                break
            take(int(j), "keep_policy_rank")

    rank_of = {j: r for r, j in enumerate(keep)}
    decisions = []
    for j in range(n):
        d = scores.row(j)
        d.update(
            {
                "slot_before": int(j),
                "action": "keep" if chosen[j] else "archive",
                "protected": bool(protected[j]),
                "eligible_for_routine_eviction": bool(eligible[j]),
                "utility": float(utility[j]),
                "redundancy": float(redundancy[j]),
                "nearest_enemy_distance": float(nearest_enemy[j]) if np.isfinite(nearest_enemy[j]) else None,
                "T_effective": float(T_eff[j]),
                "policy_score": float(base_score[j]),
                "selection_rank": rank_of.get(j),
                "reason_codes": reasons[j] + ([] if chosen[j] else ["evicted_by_capacity"]),
                "tie_break": "case_id_ascending",
            }
        )
        decisions.append(d)
    summary = {
        "policy": cfg.policy,
        "n_before": n,
        "K": K,
        "n_after": len(keep),
        "n_protected": int(protected.sum()),
        "n_evidenced": int(evidenced.sum()),
        "n_eligible": int(eligible.sum()),
        "evicted_eligible": int((~chosen & eligible).sum()),
        "evicted_ineligible": int((~chosen & ~eligible).sum()),
    }
    return SelectionResult(np.asarray(keep, dtype=np.int64), decisions, summary)


def _policy_static_score(cfg, scores, utility, redundancy, T_eff, rng, n):
    policy = cfg.policy
    if policy in ("full_memory",):
        return np.zeros(n)
    if policy == "random" or policy == "stratified":
        return rng.random(n)
    if policy == "bias_current":
        return scores.bias.astype(float)
    if policy == "provenance_only":
        return np.where(scores.evidenced, scores.Q, 0.5)
    if policy == "bias_normalized":
        return scores.B.astype(float)
    if policy == "trust":
        return T_eff
    if policy in ("trust_utility", "trust_utility_coverage"):
        return cfg.w_trust * T_eff + cfg.w_utility * utility  # coverage/redundancy added greedily
    if policy == "utility_redundancy":
        return cfg.w_utility * utility - cfg.w_redundancy * redundancy
    if policy == "kcenter":
        return np.zeros(n)
    raise ValueError(policy)


def _stratified_fill(keep, chosen, K, cohorts, rng, tie, reasons):
    """Proportional-to-cohort random selection (true stratification)."""
    n = len(cohorts)
    uniq = sorted(set(cohorts.tolist()))
    remaining = K - len(keep)
    if remaining <= 0:
        return
    avail = {c: [j for j in np.nonzero((cohorts == c) & ~chosen)[0]] for c in uniq}
    total = sum(len(v) for v in avail.values())
    quota = {c: remaining * len(avail[c]) / max(total, 1) for c in uniq}
    base = {c: int(np.floor(quota[c])) for c in uniq}
    left = remaining - sum(base.values())
    for c in sorted(uniq, key=lambda c: (-(quota[c] - base[c]), c))[:left]:
        base[c] += 1
    for c in uniq:
        pool = np.asarray(avail[c], dtype=np.int64)
        if pool.size == 0:
            continue
        perm = pool[rng.permutation(pool.size)][: base[c]]
        for j in perm:
            chosen[j] = True
            keep.append(int(j))
            reasons[j].append("keep_stratified_random")


def _greedy_fill(cfg, keep, chosen, K, dist, scores, utility, T_eff, tie, reasons):
    """Greedy selection with redundancy/coverage measured against the *already kept* set."""
    n = dist.shape[0]
    finite = dist[np.isfinite(dist) & (dist > 0)]
    scale = float(np.median(finite)) if finite.size else 1.0
    cohorts = np.asarray(scores.cohorts)
    while len(keep) < K:
        cand = np.nonzero(~chosen)[0]
        if cand.size == 0:
            break
        if keep:
            kept = np.asarray(keep)
            d_to_kept = dist[np.ix_(cand, kept)]
            min_all = d_to_kept.min(1)
            same = cohorts[cand][:, None] == cohorts[kept][None, :]
            d_same = np.where(same, d_to_kept, np.inf).min(1)
        else:
            min_all = np.full(cand.size, np.inf)
            d_same = np.full(cand.size, np.inf)
        red = np.where(np.isfinite(d_same), np.exp(-d_same / scale), 0.0)
        cov = np.where(np.isfinite(min_all), 1.0 - np.exp(-min_all / scale), 1.0)
        if cfg.policy == "kcenter":
            s = np.where(np.isfinite(min_all), min_all, 1e30)
        elif cfg.policy == "utility_redundancy":
            s = cfg.w_utility * utility[cand] - cfg.w_redundancy * red
        else:  # trust_utility_coverage
            s = cfg.w_trust * T_eff[cand] + cfg.w_utility * utility[cand] - cfg.w_redundancy * red + cfg.w_coverage * cov
        best = cand[np.lexsort((tie[cand], -s))[0]]
        chosen[best] = True
        keep.append(int(best))
        reasons[best].append("keep_greedy")
