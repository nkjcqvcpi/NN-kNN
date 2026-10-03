"""Fixed-K standalone candidates with explicit coverage protection.

Q and B are alternatives. September 20 supersedes geometric trust and weighted
Q/B mixtures. Dynamic removal is handled by maintenance.run_maintenance.
"""

from dataclasses import dataclass, field

import numpy as np


POLICIES = ("full_memory", "random", "stratified", "bias_current",
            "provenance_only", "bias_normalized", "kcenter",
            "coverage_reachability", "removal_influence", "removal_cached")
RETIRED_POLICIES = {"trust", "trust_utility", "trust_utility_coverage", "utility_redundancy"}


@dataclass
class RetentionConfig:
    policy: str
    min_per_cohort: int
    protect_rare_cohort_below: int = 0
    boundary_protect_fraction: float = 0.0
    activation_threshold: float | None = None
    zero_reachability: str | None = None
    cache_activation_threshold: float | None = None
    allowed_loss_increase: float | None = None
    seed: int = 0
    notes: dict = field(default_factory=dict)

    def validate(self):
        if self.policy in RETIRED_POLICIES:
            raise ValueError("PI September 20 supersedes Q/B combination policies; choose standalone candidates")
        if self.policy not in POLICIES:
            raise ValueError(f"Unknown policy {self.policy!r}; choose from {POLICIES}")
        if self.min_per_cohort < 0 or self.protect_rare_cohort_below < 0 or not 0 <= self.boundary_protect_fraction <= 1:
            raise ValueError("Protection settings must be nonnegative; boundary fraction is in [0,1]")
        if self.policy == "coverage_reachability":
            if self.activation_threshold is None or self.zero_reachability is None:
                raise ValueError("Coverage/reachability needs an explicit activation threshold and zero policy")
        if self.policy.startswith("removal"):
            if self.allowed_loss_increase is None or not np.isfinite(self.allowed_loss_increase) or self.allowed_loss_increase < 0:
                raise ValueError("Removal needs an explicit finite cumulative loss-increase budget")
            if self.policy == "removal_cached" and self.cache_activation_threshold is None:
                raise ValueError("Cached removal needs an explicit activation threshold")


@dataclass
class SelectionResult:
    keep_slots: np.ndarray
    decisions: list
    summary: dict


def protection_requirements(scores, cfg, dist=None):
    cohorts = np.asarray(scores.cohorts)
    n = len(cohorts)
    protected = np.zeros(n, dtype=bool)
    reasons = [[] for _ in range(n)]
    floors = {}
    for c in sorted(set(cohorts.tolist())):
        members = np.flatnonzero(cohorts == c)
        floors[c] = min(cfg.min_per_cohort, len(members))
        if len(members) <= cfg.protect_rare_cohort_below:
            protected[members] = True
            for j in members:
                reasons[j].append("protect_rare_cohort")
    if cfg.boundary_protect_fraction > 0:
        if dist is None:
            raise ValueError("Boundary protection requires learned distances")
        enemy = np.full(n, np.inf)
        for j in range(n):
            other = cohorts != cohorts[j]
            if other.any():
                enemy[j] = dist[j, other].min()
        count = int(np.floor(cfg.boundary_protect_fraction * n))
        for j in np.lexsort((scores.case_ids, enemy))[:count]:
            protected[j] = True
            reasons[j].append("protect_boundary")
    return protected, floors, reasons


def candidate_rank(scores, cfg):
    if cfg.policy == "full_memory":
        return np.zeros(len(scores.case_ids))
    if cfg.policy in {"random", "stratified"}:
        return np.random.default_rng(cfg.seed).random(len(scores.case_ids))
    if cfg.policy == "bias_current":
        return scores.bias.copy()
    if cfg.policy == "bias_normalized":
        return scores.B.copy()
    if cfg.policy == "provenance_only":
        return np.where(scores.evidenced, scores.Q, 0.5)
    if cfg.policy == "kcenter":
        return np.zeros(len(scores.case_ids))
    return scores.extra[cfg.policy].copy()


def select_active_set(scores, K, cfg, dist):
    cfg.validate()
    if cfg.policy.startswith("removal"):
        raise ValueError("Removal policies require sequential recomputation through run_maintenance")
    n = len(scores.case_ids)
    if not isinstance(K, (int, np.integer)) or K < 1:
        raise ValueError("Capacity K must be a positive integer")
    K = n if cfg.policy == "full_memory" else min(K, n)
    protected, floors, reasons = protection_requirements(scores, cfg, dist)
    cohorts = np.asarray(scores.cohorts)
    required = sum(max(floors[c], int(protected[cohorts == c].sum())) for c in floors)
    if required > K:
        raise ValueError(f"Capacity {K} cannot satisfy {required} protected/floor cases")
    ranking = candidate_rank(scores, cfg)
    chosen = protected.copy()
    for c, floor in floors.items():
        members = np.flatnonzero((cohorts == c) & ~chosen)
        need = max(0, floor - int(chosen[cohorts == c].sum()))
        order = members[np.lexsort((scores.case_ids[members], -ranking[members]))]
        for j in order[:need]:
            chosen[j] = True
            reasons[j].append("keep_coverage_floor")
    if cfg.policy == "stratified":
        pool = np.flatnonzero(~chosen)
        remaining = K - int(chosen.sum())
        quotas = {c: remaining * int((cohorts[pool] == c).sum()) / max(len(pool), 1) for c in floors}
        counts = {c: int(np.floor(q)) for c, q in quotas.items()}
        for c in sorted(floors, key=lambda c: (-(quotas[c] - counts[c]), c))[:remaining - sum(counts.values())]:
            counts[c] += 1
        for c, count in counts.items():
            members = pool[cohorts[pool] == c]
            chosen[members[np.lexsort((scores.case_ids[members], -ranking[members]))][:count]] = True
    elif cfg.policy == "kcenter":
        if dist is None:
            raise ValueError("kcenter needs learned distances")
        while chosen.sum() < K:
            pool, kept = np.flatnonzero(~chosen), np.flatnonzero(chosen)
            d = dist[np.ix_(pool, kept)].min(1) if len(kept) else np.zeros(len(pool))
            j = pool[np.lexsort((scores.case_ids[pool], -d))[0]]
            chosen[j] = True
    else:
        pool = np.flatnonzero(~chosen)
        order = pool[np.lexsort((scores.case_ids[pool], -ranking[pool]))]
        chosen[order[:K - int(chosen.sum())]] = True
    decisions = []
    for j in range(n):
        d = scores.row(j)
        d.update(slot_before=j, action="keep" if chosen[j] else "archive",
                 protected=bool(protected[j]), policy_score=float(ranking[j]),
                 reason_codes=reasons[j] + (["keep_policy_rank"] if chosen[j] else ["evicted_by_capacity"]),
                 tie_break="case_id_ascending")
        decisions.append(d)
    return SelectionResult(np.flatnonzero(chosen), decisions,
                           dict(policy=cfg.policy, n_before=n, K=K, n_after=int(chosen.sum()),
                                n_protected=int(protected.sum()), n_evidenced=int(scores.evidenced.sum())))
