from __future__ import annotations

import copy
import csv
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class CaseStatistics:
    """Per-case runtime provenance and bookkeeping counters."""
    case_id: int
    retrieval_count: int = 0
    activation_mass: float = 0.0
    correct_support: float = 0.0
    incorrect_support: float = 0.0
    initial_bias: float = 0.0
    current_bias: float = 0.0
    insertion_step: int = 0
    last_retrieved_step: int = 0
    cohort_id: str | int | None = None
    protected: bool = False

    @property
    def total_evidence(self) -> float:
        return self.correct_support + self.incorrect_support

    def quality_score(self, smoothing: float = 1.0) -> float:
        """Smoothed provenance-quality score Q_i (T0 Eq. from handoff).
        
        Q_i = (C_i + s) / (C_i + H_i + 2s), with s > 0.
        When C_i = H_i = 0, Q_i = 0.5 (symmetric prior).
        """
        s = max(float(smoothing), 1e-6)
        return float((self.correct_support + s) / (self.correct_support + self.incorrect_support + 2.0 * s))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CaseScoringResult:
    """Detailed multi-component evaluation for a single case during maintenance."""
    case_id: int
    cohort_id: str | int | None
    protected: bool
    retrieval_count: int
    activation_mass: float
    correct_support: float
    incorrect_support: float
    quality_score: float  # Q_i
    normalized_bias: float  # B_i
    trustworthiness: float  # T_i
    redundancy: float
    composite_score: float
    has_sufficient_exposure: bool


@dataclass
class MaintenanceAction:
    """Record of a single case maintenance decision."""
    case_id: int
    action: str  # 'keep', 'archive', 'restore', 'protect', 'insert', 'replace'
    reason: str
    step: int
    cohort_id: str | int | None = None
    quality_score: float | None = None
    normalized_bias: float | None = None
    trustworthiness: float | None = None
    activation_mass: float | None = None
    composite_score: float | None = None
    protected: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CaseArchiveStore:
    """Reversible archival store for evicted cases.
    
    Preserves case tensors, parameters, and provenance so evictions can be audited
    or restored without destructive data loss.
    """

    def __init__(self) -> None:
        self._archive: dict[int, dict[str, Any]] = {}

    def archive_case(
        self,
        case_id: int,
        case_tensor: torch.Tensor,
        label_tensor: torch.Tensor,
        bias: float,
        stats: CaseStatistics,
        step: int,
        reason: str = "routine_eviction",
        **extra_metadata: Any,
    ) -> None:
        self._archive[case_id] = {
            "case_id": int(case_id),
            "case_tensor": case_tensor.detach().cpu().clone(),
            "label_tensor": label_tensor.detach().cpu().clone(),
            "bias": float(bias),
            "stats": copy.deepcopy(stats),
            "archived_at_step": int(step),
            "eviction_reason": str(reason),
            "extra": extra_metadata,
        }

    def contains(self, case_id: int) -> bool:
        return case_id in self._archive

    def get(self, case_id: int) -> dict[str, Any] | None:
        return self._archive.get(case_id)

    def restore(self, case_id: int) -> dict[str, Any] | None:
        return self._archive.pop(case_id, None)

    def count(self) -> int:
        return len(self._archive)

    def all_case_ids(self) -> list[int]:
        return sorted(self._archive.keys())

    def clear(self) -> None:
        self._archive.clear()


class CaseStatisticsStore:
    """Maintains CaseStatistics indexed by stable case_id across compactions."""

    def __init__(self) -> None:
        self._stats: dict[int, CaseStatistics] = {}

    def register_case(
        self,
        case_id: int,
        initial_bias: float = 0.0,
        cohort_id: str | int | None = None,
        insertion_step: int = 0,
        protected: bool = False,
    ) -> CaseStatistics:
        stats = CaseStatistics(
            case_id=case_id,
            initial_bias=float(initial_bias),
            current_bias=float(initial_bias),
            cohort_id=cohort_id,
            insertion_step=int(insertion_step),
            last_retrieved_step=int(insertion_step),
            protected=bool(protected),
        )
        self._stats[case_id] = stats
        return stats

    def get(self, case_id: int) -> CaseStatistics | None:
        return self._stats.get(case_id)

    def ensure(
        self,
        case_id: int,
        initial_bias: float = 0.0,
        cohort_id: str | int | None = None,
        insertion_step: int = 0,
    ) -> CaseStatistics:
        if case_id not in self._stats:
            return self.register_case(case_id, initial_bias, cohort_id, insertion_step)
        return self._stats[case_id]

    def update_bias(self, case_id: int, current_bias: float) -> None:
        if case_id in self._stats:
            self._stats[case_id].current_bias = float(current_bias)

    def observe_retrieval(
        self,
        retrieved_case_ids: Sequence[int] | torch.Tensor,
        activations: Sequence[float] | torch.Tensor,
        is_correct: Sequence[bool] | torch.Tensor,
        step: int = 0,
    ) -> None:
        """Observe a batch of query-case retrievals and accumulate statistics."""
        ids_list = (
            retrieved_case_ids.cpu().tolist()
            if isinstance(retrieved_case_ids, torch.Tensor)
            else list(retrieved_case_ids)
        )
        acts_list = (
            activations.cpu().tolist()
            if isinstance(activations, torch.Tensor)
            else [float(a) for a in activations]
        )
        corr_list = (
            is_correct.cpu().tolist()
            if isinstance(is_correct, torch.Tensor)
            else [bool(c) for c in is_correct]
        )
        for cid, act, corr in zip(ids_list, acts_list, corr_list):
            cid_int = int(cid)
            if cid_int < 0:
                continue
            stats = self.ensure(cid_int)
            stats.retrieval_count += 1
            stats.activation_mass += float(act)
            if corr:
                stats.correct_support += float(act)
            else:
                stats.incorrect_support += float(act)
            stats.last_retrieved_step = max(stats.last_retrieved_step, int(step))

    def prune_untracked(self, active_case_ids: Sequence[int]) -> None:
        """Remove statistics for cases that were permanently deleted."""
        active_set = {int(cid) for cid in active_case_ids if int(cid) >= 0}
        to_delete = [cid for cid in self._stats if cid not in active_set]
        for cid in to_delete:
            del self._stats[cid]

    def snapshot(self) -> dict[int, CaseStatistics]:
        return copy.deepcopy(self._stats)

    def count(self) -> int:
        return len(self._stats)


def compute_within_cohort_percentile_biases(
    biases: torch.Tensor | np.ndarray,
    cohorts: Sequence[str | int | None],
) -> np.ndarray:
    """Normalize raw case biases to [0, 1] within their respective cohorts.
    
    Higher bias within a cohort yields B_i closer to 1.0.
    """
    biases_arr = (
        biases.detach().cpu().numpy()
        if isinstance(biases, torch.Tensor)
        else np.asarray(biases, dtype=np.float32)
    )
    cohort_list = list(cohorts)
    norm_biases = np.full(len(biases_arr), 0.5, dtype=np.float32)
    cohort_indices: dict[Any, list[int]] = {}
    for idx, cohort in enumerate(cohort_list):
        cohort_indices.setdefault(cohort, []).append(idx)

    for cohort, indices in cohort_indices.items():
        if len(indices) <= 1:
            norm_biases[indices] = 0.5
            continue
        c_biases = biases_arr[indices]
        order = np.argsort(c_biases)
        ranks = np.empty_like(order, dtype=np.float32)
        ranks[order] = np.arange(len(indices), dtype=np.float32)
        norm_biases[indices] = ranks / float(len(indices) - 1)
    return norm_biases


def compute_trustworthiness(
    quality_score: float | np.ndarray,
    normalized_bias: float | np.ndarray,
    alpha: float = 0.5,
    eps: float = 1e-8,
) -> float | np.ndarray:
    """Compute primary combined trustworthiness score T_i = Q_i^alpha * B_i^(1-alpha) in log-space."""
    alpha_clamped = min(max(float(alpha), 0.0), 1.0)
    q = np.clip(np.asarray(quality_score, dtype=np.float32), eps, 1.0)
    b = np.clip(np.asarray(normalized_bias, dtype=np.float32), eps, 1.0)
    log_t = alpha_clamped * np.log(q) + (1.0 - alpha_clamped) * np.log(b)
    t = np.exp(log_t)
    return float(t) if np.ndim(t) == 0 else t


class CaseMaintenancePolicy:
    """Abstract interface for T0 case maintenance policies."""

    def select_keep_case_ids(
        self,
        active_case_ids: Sequence[int],
        cases: torch.Tensor,
        labels: torch.Tensor,
        biases: torch.Tensor,
        stats_store: CaseStatisticsStore,
        target_capacity: int,
        step: int = 0,
    ) -> tuple[list[int], list[MaintenanceAction]]:
        raise NotImplementedError


class ProvenanceBiasCoveragePolicy(CaseMaintenancePolicy):
    """Primary T0 retention policy combining provenance, bias, utility, redundancy, and coverage."""

    def __init__(
        self,
        smoothing: float = 1.0,
        alpha: float = 0.5,
        min_retrieval_count: int = 2,
        min_activation_mass: float = 0.05,
        min_per_cohort: int = 2,
        redundancy_weight: float = 0.2,
        utility_weight: float = 0.1,
    ) -> None:
        self.smoothing = float(smoothing)
        self.alpha = float(alpha)
        self.min_retrieval_count = int(min_retrieval_count)
        self.min_activation_mass = float(min_activation_mass)
        self.min_per_cohort = int(min_per_cohort)
        self.redundancy_weight = float(redundancy_weight)
        self.utility_weight = float(utility_weight)

    def select_keep_case_ids(
        self,
        active_case_ids: Sequence[int],
        cases: torch.Tensor,
        labels: torch.Tensor,
        biases: torch.Tensor,
        stats_store: CaseStatisticsStore,
        target_capacity: int,
        step: int = 0,
    ) -> tuple[list[int], list[MaintenanceAction]]:
        active_ids = [int(cid) for cid in active_case_ids if int(cid) >= 0]
        n_active = len(active_ids)
        if n_active <= target_capacity:
            actions = [
                MaintenanceAction(cid, "keep", "within_capacity", step)
                for cid in active_ids
            ]
            return active_ids, actions

        # 1. Resolve cohorts (class labels from labels tensor)
        cohorts: list[str | int | None] = []
        labels_cpu = labels[:n_active].detach().cpu()
        for i, cid in enumerate(active_ids):
            st = stats_store.get(cid)
            if st is not None and st.cohort_id is not None:
                cohorts.append(st.cohort_id)
            elif labels_cpu[i].numel() > 1:
                cohorts.append(int(labels_cpu[i].argmax().item()))
            else:
                cohorts.append(int(labels_cpu[i].item()))

        # 2. Normalize biases within cohort
        b_norm = compute_within_cohort_percentile_biases(biases[:n_active], cohorts)

        # 3. Compute pairwise redundancy (cosine similarity in flattened representation)
        cases_flat = cases[:n_active].detach().cpu().view(n_active, -1).float()
        cases_norm = F.normalize(cases_flat, p=2, dim=-1)
        sim_matrix = torch.matmul(cases_norm, cases_norm.T).numpy()

        # 4. Evaluate trustworthiness and composite score for each case
        scored: list[CaseScoringResult] = []
        for i, cid in enumerate(active_ids):
            st = stats_store.ensure(cid, initial_bias=float(biases[i].item()), cohort_id=cohorts[i])
            q_i = st.quality_score(self.smoothing)
            b_i = float(b_norm[i])
            t_i = float(compute_trustworthiness(q_i, b_i, self.alpha))

            has_exposure = bool(
                st.retrieval_count >= self.min_retrieval_count
                and st.activation_mass >= self.min_activation_mass
            )

            # Intra-cohort redundancy (max similarity to other cases of same cohort)
            same_cohort_idx = [j for j in range(n_active) if j != i and cohorts[j] == cohorts[i]]
            if same_cohort_idx:
                max_sim = float(sim_matrix[i, same_cohort_idx].max())
                redundancy = max(0.0, max_sim)
            else:
                redundancy = 0.0

            # Composite score balances trustworthiness, utility, and diversity
            norm_util = min(st.activation_mass / 5.0, 1.0)
            composite = t_i - self.redundancy_weight * redundancy + self.utility_weight * norm_util

            scored.append(
                CaseScoringResult(
                    case_id=cid,
                    cohort_id=cohorts[i],
                    protected=st.protected,
                    retrieval_count=st.retrieval_count,
                    activation_mass=st.activation_mass,
                    correct_support=st.correct_support,
                    incorrect_support=st.incorrect_support,
                    quality_score=q_i,
                    normalized_bias=b_i,
                    trustworthiness=t_i,
                    redundancy=redundancy,
                    composite_score=composite,
                    has_sufficient_exposure=has_exposure,
                )
            )

        # 5. Protected cases (ensure min_per_cohort)
        cohort_groups: dict[Any, list[CaseScoringResult]] = {}
        for res in scored:
            cohort_groups.setdefault(res.cohort_id, []).append(res)

        keep_set: set[int] = set()
        actions: list[MaintenanceAction] = []

        # First guarantee coverage per cohort
        for cohort, group in cohort_groups.items():
            # Sort cohort candidates by composite_score descending
            sorted_group = sorted(group, key=lambda x: (x.protected, x.composite_score), reverse=True)
            floor = min(self.min_per_cohort, len(sorted_group))
            for item in sorted_group[:floor]:
                keep_set.add(item.case_id)
                actions.append(
                    MaintenanceAction(
                        case_id=item.case_id,
                        action="protect",
                        reason="cohort_coverage_floor",
                        step=step,
                        cohort_id=cohort,
                        quality_score=item.quality_score,
                        normalized_bias=item.normalized_bias,
                        trustworthiness=item.trustworthiness,
                        activation_mass=item.activation_mass,
                        composite_score=item.composite_score,
                        protected=True,
                    )
                )

        # 6. Fill remaining capacity up to target_capacity
        remaining_slots = target_capacity - len(keep_set)
        if remaining_slots > 0:
            remaining_candidates = [
                res for res in scored if res.case_id not in keep_set
            ]
            # Sort by composite score descending, break ties deterministically by case_id
            remaining_candidates.sort(key=lambda x: (x.composite_score, -x.case_id), reverse=True)
            for item in remaining_candidates[:remaining_slots]:
                keep_set.add(item.case_id)
                actions.append(
                    MaintenanceAction(
                        case_id=item.case_id,
                        action="keep",
                        reason="high_composite_score",
                        step=step,
                        cohort_id=item.cohort_id,
                        quality_score=item.quality_score,
                        normalized_bias=item.normalized_bias,
                        trustworthiness=item.trustworthiness,
                        activation_mass=item.activation_mass,
                        composite_score=item.composite_score,
                        protected=False,
                    )
                )

        # 7. Mark evictions
        evicted_candidates = [res for res in scored if res.case_id not in keep_set]
        for item in evicted_candidates:
            reason = "low_trustworthiness" if item.trustworthiness < 0.4 else "budget_capacity"
            if item.redundancy > 0.85:
                reason = "high_redundancy"
            actions.append(
                MaintenanceAction(
                    case_id=item.case_id,
                    action="archive",
                    reason=reason,
                    step=step,
                    cohort_id=item.cohort_id,
                    quality_score=item.quality_score,
                    normalized_bias=item.normalized_bias,
                    trustworthiness=item.trustworthiness,
                    activation_mass=item.activation_mass,
                    composite_score=item.composite_score,
                    protected=False,
                )
            )

        # Preserve original relative order among kept cases
        keep_ids = [cid for cid in active_ids if cid in keep_set]
        return keep_ids, actions


class BiasOnlyPolicy(CaseMaintenancePolicy):
    """Baseline policy: retains top K cases strictly by case bias."""

    def select_keep_case_ids(
        self,
        active_case_ids: Sequence[int],
        cases: torch.Tensor,
        labels: torch.Tensor,
        biases: torch.Tensor,
        stats_store: CaseStatisticsStore,
        target_capacity: int,
        step: int = 0,
    ) -> tuple[list[int], list[MaintenanceAction]]:
        active_ids = [int(cid) for cid in active_case_ids if int(cid) >= 0]
        n_active = len(active_ids)
        if n_active <= target_capacity:
            return active_ids, [
                MaintenanceAction(cid, "keep", "within_capacity", step) for cid in active_ids
            ]
        b_vals = biases[:n_active].detach().cpu().numpy()
        order = np.argsort(b_vals)[::-1]  # descending
        keep_indices = set(order[:target_capacity].tolist())
        keep_ids = [active_ids[i] for i in range(n_active) if i in keep_indices]
        actions = []
        for i, cid in enumerate(active_ids):
            if i in keep_indices:
                actions.append(MaintenanceAction(cid, "keep", "top_bias", step))
            else:
                actions.append(MaintenanceAction(cid, "archive", "low_bias", step))
        return keep_ids, actions


class ProvenanceOnlyPolicy(CaseMaintenancePolicy):
    """Ablation policy: retains top K cases strictly by provenance score Q_i."""

    def __init__(self, smoothing: float = 1.0) -> None:
        self.smoothing = float(smoothing)

    def select_keep_case_ids(
        self,
        active_case_ids: Sequence[int],
        cases: torch.Tensor,
        labels: torch.Tensor,
        biases: torch.Tensor,
        stats_store: CaseStatisticsStore,
        target_capacity: int,
        step: int = 0,
    ) -> tuple[list[int], list[MaintenanceAction]]:
        active_ids = [int(cid) for cid in active_case_ids if int(cid) >= 0]
        n_active = len(active_ids)
        if n_active <= target_capacity:
            return active_ids, [
                MaintenanceAction(cid, "keep", "within_capacity", step) for cid in active_ids
            ]
        q_scores = []
        for cid in active_ids:
            st = stats_store.ensure(cid)
            q_scores.append(st.quality_score(self.smoothing))
        order = np.argsort(q_scores)[::-1]
        keep_indices = set(order[:target_capacity].tolist())
        keep_ids = [active_ids[i] for i in range(n_active) if i in keep_indices]
        actions = [
            MaintenanceAction(cid, "keep" if i in keep_indices else "archive", "provenance_q", step, quality_score=q_scores[i])
            for i, cid in enumerate(active_ids)
        ]
        return keep_ids, actions


class TrustworthinessOnlyPolicy(CaseMaintenancePolicy):
    """Ablation policy: retains top K cases strictly by geometric score T_i."""

    def __init__(self, smoothing: float = 1.0, alpha: float = 0.5) -> None:
        self.smoothing = float(smoothing)
        self.alpha = float(alpha)

    def select_keep_case_ids(
        self,
        active_case_ids: Sequence[int],
        cases: torch.Tensor,
        labels: torch.Tensor,
        biases: torch.Tensor,
        stats_store: CaseStatisticsStore,
        target_capacity: int,
        step: int = 0,
    ) -> tuple[list[int], list[MaintenanceAction]]:
        active_ids = [int(cid) for cid in active_case_ids if int(cid) >= 0]
        n_active = len(active_ids)
        if n_active <= target_capacity:
            return active_ids, [
                MaintenanceAction(cid, "keep", "within_capacity", step) for cid in active_ids
            ]
        labels_cpu = labels[:n_active].detach().cpu()
        cohorts = [
            int(labels_cpu[i].argmax().item()) if labels_cpu[i].numel() > 1 else int(labels_cpu[i].item())
            for i in range(n_active)
        ]
        b_norm = compute_within_cohort_percentile_biases(biases[:n_active], cohorts)
        t_scores = []
        for i, cid in enumerate(active_ids):
            st = stats_store.ensure(cid)
            q_i = st.quality_score(self.smoothing)
            t_scores.append(float(compute_trustworthiness(q_i, b_norm[i], self.alpha)))
        order = np.argsort(t_scores)[::-1]
        keep_indices = set(order[:target_capacity].tolist())
        keep_ids = [active_ids[i] for i in range(n_active) if i in keep_indices]
        actions = [
            MaintenanceAction(cid, "keep" if i in keep_indices else "archive", "trustworthiness_t", step, trustworthiness=t_scores[i])
            for i, cid in enumerate(active_ids)
        ]
        return keep_ids, actions


class StratifiedRandomPolicy(CaseMaintenancePolicy):
    """Baseline policy: random stratified selection down to capacity K."""

    def __init__(self, seed: int = 0) -> None:
        self.seed = int(seed)

    def select_keep_case_ids(
        self,
        active_case_ids: Sequence[int],
        cases: torch.Tensor,
        labels: torch.Tensor,
        biases: torch.Tensor,
        stats_store: CaseStatisticsStore,
        target_capacity: int,
        step: int = 0,
    ) -> tuple[list[int], list[MaintenanceAction]]:
        active_ids = [int(cid) for cid in active_case_ids if int(cid) >= 0]
        n_active = len(active_ids)
        if n_active <= target_capacity:
            return active_ids, [
                MaintenanceAction(cid, "keep", "within_capacity", step) for cid in active_ids
            ]
        rng = np.random.default_rng(self.seed + step)
        selected_idx = set(rng.choice(n_active, size=target_capacity, replace=False).tolist())
        keep_ids = [active_ids[i] for i in range(n_active) if i in selected_idx]
        actions = [
            MaintenanceAction(cid, "keep" if i in selected_idx else "archive", "random_stratified", step)
            for i, cid in enumerate(active_ids)
        ]
        return keep_ids, actions


def write_maintenance_artifacts(
    output_dir: str | Path,
    stats_store: CaseStatisticsStore,
    actions_log: list[MaintenanceAction],
    archive_store: CaseArchiveStore | None = None,
) -> None:
    """Save case_statistics.csv, case_maintenance.csv, and case_selection_summary.json."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. case_statistics.csv
    stats_dict = stats_store.snapshot()
    if stats_dict:
        fieldnames = list(asdict(next(iter(stats_dict.values()))).keys())
        with (out_path / "case_statistics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for st in stats_dict.values():
                writer.writerow(asdict(st))

    # 2. case_maintenance.csv
    if actions_log:
        fieldnames = list(actions_log[0].to_dict().keys())
        with (out_path / "case_maintenance.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for act in actions_log:
                writer.writerow(act.to_dict())

    # 3. case_selection_summary.json
    action_counts: dict[str, int] = {}
    for act in actions_log:
        action_counts[act.action] = action_counts.get(act.action, 0) + 1

    summary = {
        "total_active_cases": stats_store.count(),
        "archived_cases": archive_store.count() if archive_store is not None else 0,
        "maintenance_actions": action_counts,
        "total_maintenance_events": len(actions_log),
    }
    with (out_path / "case_selection_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
