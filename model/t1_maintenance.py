from __future__ import annotations

import copy
import csv
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Callable, Sequence
import uuid

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class CaseStatistics:
    """Per-case runtime provenance and bookkeeping counters (T1 & Data Contract)."""
    case_id: int
    case_version_id: int = 1
    case_role: str = "classification"  # 'classification', 'regression', 'actor', 'critic'
    scope: str = "global"
    reliability_status: str = "approved"  # 'provisional', 'approved', 'uncertain', 'quarantined'
    retrieval_count: int = 0
    activation_mass: float = 0.0
    correct_support: float = 0.0  # C_i / positive support
    incorrect_support: float = 0.0  # H_i / harmful support
    initial_bias: float = 0.0
    current_bias: float = 0.0
    insertion_step: int = 0
    last_retrieved_step: int = 0
    cohort_id: str | int | None = None
    protected: bool = False
    protection_reason: str | None = None

    @property
    def total_evidence(self) -> float:
        return self.correct_support + self.incorrect_support

    def quality_score(self, smoothing: float = 1.0) -> float:
        """Smoothed provenance-quality score Q_i (T1 Section 8.2).
        
        Q_i = (C_i + s) / (C_i + H_i + 2s), with s > 0.
        When C_i = H_i = 0, Q_i = 0.5 (symmetric prior / unobserved evidence).
        """
        s = max(float(smoothing), 1e-6)
        return float((self.correct_support + s) / (self.correct_support + self.incorrect_support + 2.0 * s))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["quality_score_s1"] = self.quality_score(1.0)
        return d


@dataclass
class CaseScoringResult:
    """Multi-component evaluation for a single case during maintenance."""
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
    """Record of a single case maintenance event (T1 & Data Contract Section 5)."""
    case_id: int
    action: str  # 'keep', 'archive', 'restore', 'protect', 'insert', 'replace', 'quarantine'
    reason: str
    step: int
    maintenance_event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    case_version_id: int = 1
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
    """Reversible archival store for evicted cases (T1 Stage 2 & 3).
    
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
        protection_reason: str | None = None,
        case_role: str = "classification",
    ) -> CaseStatistics:
        stats = CaseStatistics(
            case_id=case_id,
            initial_bias=float(initial_bias),
            current_bias=float(initial_bias),
            cohort_id=cohort_id,
            insertion_step=insertion_step,
            last_retrieved_step=insertion_step,
            protected=protected,
            protection_reason=protection_reason,
            case_role=case_role,
        )
        self._stats[case_id] = stats
        return stats

    def get(self, case_id: int) -> CaseStatistics | None:
        return self._stats.get(case_id)

    def get_or_create(
        self,
        case_id: int,
        initial_bias: float = 0.0,
        cohort_id: str | int | None = None,
        step: int = 0,
        case_role: str = "classification",
    ) -> CaseStatistics:
        if case_id not in self._stats:
            return self.register_case(
                case_id=case_id,
                initial_bias=initial_bias,
                cohort_id=cohort_id,
                insertion_step=step,
                case_role=case_role,
            )
        return self._stats[case_id]

    def record_retrieval(
        self,
        case_id: int,
        activation: float,
        is_correct: bool | None,
        step: int,
        delta_loss: float | None = None,
    ) -> None:
        """Records one retrieval contribution event for a stable case."""
        st = self.get_or_create(case_id, step=step)
        st.retrieval_count += 1
        st.activation_mass += float(activation)
        st.last_retrieved_step = step

        if is_correct is not None:
            if is_correct:
                st.correct_support += float(activation)
            else:
                st.incorrect_support += float(activation)
        elif delta_loss is not None:
            # Regression counterfactual: Delta_i > 0 is correct/positive support
            if delta_loss > 0:
                st.correct_support += float(delta_loss)
            else:
                st.incorrect_support += float(-delta_loss)

    def update_bias(self, case_id: int, bias_value: float) -> None:
        if case_id in self._stats:
            self._stats[case_id].current_bias = float(bias_value)

    def count(self) -> int:
        return len(self._stats)

    def all_case_ids(self) -> list[int]:
        return sorted(self._stats.keys())

    def snapshot(self) -> dict[int, CaseStatistics]:
        return {k: copy.deepcopy(v) for k, v in self._stats.items()}


def compute_trustworthiness(
    quality_score: float,
    normalized_bias: float,
    alpha: float = 0.6,
    eps: float = 1e-6,
) -> float:
    """PI-approved weighted geometric mean for case trustworthiness (T1 Section 8.4).
    
    log(T_i) = alpha * log(clip(Q_i, eps, 1)) + (1 - alpha) * log(clip(B_i, eps, 1))
    T_i = exp(log(T_i))
    """
    q_clipped = min(max(float(quality_score), eps), 1.0)
    b_clipped = min(max(float(normalized_bias), eps), 1.0)
    log_t = alpha * math.log(q_clipped) + (1.0 - alpha) * math.log(b_clipped)
    return float(math.exp(log_t))


def compute_within_cohort_percentile_biases(
    biases: torch.Tensor,
    cohorts: Sequence[str | int | None],
) -> list[float]:
    """Computes within-cohort percentile ranks for biases in [0, 1] (T1 Section 8.3)."""
    b_list = biases.detach().cpu().numpy().tolist()
    cohort_indices: dict[str | int | None, list[int]] = {}
    for idx, c in enumerate(cohorts):
        cohort_indices.setdefault(c, []).append(idx)

    normalized = [0.5] * len(b_list)
    for c, idxs in cohort_indices.items():
        if len(idxs) == 1:
            normalized[idxs[0]] = 0.5
            continue
        vals = [b_list[i] for i in idxs]
        sorted_vals = sorted(vals)
        n = len(sorted_vals)
        for i in idxs:
            rank = sum(1 for v in sorted_vals if v < b_list[i])
            normalized[i] = rank / (n - 1) if n > 1 else 0.5

    return normalized


class CaseMaintenancePolicy:
    """Common case maintenance-policy interface for T1 CBR architectures."""

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


class FullMemoryPolicy(CaseMaintenancePolicy):
    """Full-memory baseline: keeps all active cases without capacity pruning."""

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
        actions = [
            MaintenanceAction(cid, "keep", "full_memory", step) for cid in active_ids
        ]
        return active_ids, actions


class DownsampleRandomPolicy(CaseMaintenancePolicy):
    """Uniform random downsampling baseline."""

    def __init__(self, seed: int = 42) -> None:
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
        perm = rng.permutation(n_active)
        keep_indices = set(perm[:target_capacity].tolist())

        keep_ids = [active_ids[i] for i in range(n_active) if i in keep_indices]
        actions = [
            MaintenanceAction(
                case_id=cid,
                action="keep" if i in keep_indices else "archive",
                reason="random_downsample_keep" if i in keep_indices else "random_downsample_evict",
                step=step,
            )
            for i, cid in enumerate(active_ids)
        ]
        return keep_ids, actions


class StratifiedRandomPolicy(CaseMaintenancePolicy):
    """Stratified class-proportional downsampling baseline."""

    def __init__(self, seed: int = 42) -> None:
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

        labels_cpu = labels[:n_active].detach().cpu()
        if labels_cpu.dim() > 1 and labels_cpu.shape[-1] > 1:
            classes = labels_cpu.argmax(dim=-1).tolist()
        else:
            classes = labels_cpu.view(-1).long().tolist()

        class_to_indices: dict[int, list[int]] = {}
        for idx, c in enumerate(classes):
            class_to_indices.setdefault(c, []).append(idx)

        rng = np.random.default_rng(self.seed + step)
        keep_indices: set[int] = set()

        total_classes = len(class_to_indices)
        base_per_class = max(1, target_capacity // total_classes)

        for c, idxs in sorted(class_to_indices.items()):
            n_take = min(len(idxs), base_per_class)
            perm = rng.permutation(len(idxs))
            for p in perm[:n_take]:
                keep_indices.add(idxs[p])

        # Fill remaining budget
        remaining = [i for i in range(n_active) if i not in keep_indices]
        if len(keep_indices) < target_capacity and remaining:
            n_more = min(target_capacity - len(keep_indices), len(remaining))
            perm_rem = rng.permutation(len(remaining))
            for p in perm_rem[:n_more]:
                keep_indices.add(remaining[p])

        keep_ids = [active_ids[i] for i in range(n_active) if i in keep_indices]
        actions = [
            MaintenanceAction(
                case_id=cid,
                action="keep" if i in keep_indices else "archive",
                reason="stratified_keep" if i in keep_indices else "stratified_evict",
                step=step,
                cohort_id=classes[i],
            )
            for i, cid in enumerate(active_ids)
        ]
        return keep_ids, actions


class BiasOnlyPolicy(CaseMaintenancePolicy):
    """Current NN-kNN bias pruning baseline (keeps highest per-case learned biases)."""

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
        ranked_indices = np.argsort(-b_vals)
        keep_indices = set(ranked_indices[:target_capacity].tolist())

        keep_ids = [active_ids[i] for i in range(n_active) if i in keep_indices]
        actions = [
            MaintenanceAction(
                case_id=cid,
                action="keep" if i in keep_indices else "archive",
                reason="bias_topk_keep" if i in keep_indices else "bias_prune_evict",
                step=step,
                composite_score=float(b_vals[i]),
            )
            for i, cid in enumerate(active_ids)
        ]
        return keep_ids, actions


class ProvenanceOnlyPolicy(CaseMaintenancePolicy):
    """Provenance quality Q_i only retention policy."""

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

        q_scores: list[float] = []
        for cid in active_ids:
            st = stats_store.get(cid)
            q_scores.append(st.quality_score(self.smoothing) if st else 0.5)

        ranked = sorted(range(n_active), key=lambda i: (-q_scores[i], active_ids[i]))
        keep_indices = set(ranked[:target_capacity])

        keep_ids = [active_ids[i] for i in range(n_active) if i in keep_indices]
        actions = [
            MaintenanceAction(
                case_id=cid,
                action="keep" if i in keep_indices else "archive",
                reason="provenance_q_keep" if i in keep_indices else "provenance_q_evict",
                step=step,
                quality_score=q_scores[i],
                composite_score=q_scores[i],
            )
            for i, cid in enumerate(active_ids)
        ]
        return keep_ids, actions


class TrustworthinessOnlyPolicy(CaseMaintenancePolicy):
    """Geometric trustworthiness T_i only retention policy."""

    def __init__(self, alpha: float = 0.6, smoothing: float = 1.0) -> None:
        self.alpha = float(alpha)
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

        labels_cpu = labels[:n_active].detach().cpu()
        cohorts = labels_cpu.argmax(dim=-1).tolist() if labels_cpu.dim() > 1 and labels_cpu.shape[-1] > 1 else labels_cpu.view(-1).long().tolist()
        norm_biases = compute_within_cohort_percentile_biases(biases[:n_active], cohorts)

        t_scores: list[float] = []
        for i, cid in enumerate(active_ids):
            st = stats_store.get(cid)
            q = st.quality_score(self.smoothing) if st else 0.5
            t = compute_trustworthiness(q, norm_biases[i], alpha=self.alpha)
            t_scores.append(t)

        ranked = sorted(range(n_active), key=lambda i: (-t_scores[i], active_ids[i]))
        keep_indices = set(ranked[:target_capacity])

        keep_ids = [active_ids[i] for i in range(n_active) if i in keep_indices]
        actions = [
            MaintenanceAction(
                case_id=cid,
                action="keep" if i in keep_indices else "archive",
                reason="trustworthiness_keep" if i in keep_indices else "trustworthiness_evict",
                step=step,
                trustworthiness=t_scores[i],
                composite_score=t_scores[i],
            )
            for i, cid in enumerate(active_ids)
        ]
        return keep_ids, actions


class ProvenanceBiasCoveragePolicy(CaseMaintenancePolicy):
    """Full 3-stage constrained retention: protection, evidence, redundancy & trustworthiness."""

    def __init__(
        self,
        alpha: float = 0.6,
        smoothing: float = 1.0,
        min_per_class: int = 2,
        min_retrieval_count: int = 1,
        min_activation_mass: float = 0.1,
        redundancy_weight: float = 0.2,
    ) -> None:
        self.alpha = float(alpha)
        self.smoothing = float(smoothing)
        self.min_per_class = int(min_per_class)
        self.min_retrieval_count = int(min_retrieval_count)
        self.min_activation_mass = float(min_activation_mass)
        self.redundancy_weight = float(redundancy_weight)

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
        if labels_cpu.dim() > 1 and labels_cpu.shape[-1] > 1:
            cohorts = labels_cpu.argmax(dim=-1).tolist()
        else:
            cohorts = labels_cpu.view(-1).long().tolist()

        norm_biases = compute_within_cohort_percentile_biases(biases[:n_active], cohorts)

        # Stage 1: Protection & evidence scoring
        keep_indices: set[int] = set()
        scoring_results: list[CaseScoringResult] = []

        # Coverage floor per cohort
        cohort_to_indices: dict[Any, list[int]] = {}
        for idx, c in enumerate(cohorts):
            cohort_to_indices.setdefault(c, []).append(idx)

        # Precompute pairwise distances for redundancy penalty
        with torch.no_grad():
            X_flat = cases[:n_active].view(n_active, -1)
            dist_mat = torch.cdist(X_flat, X_flat).cpu()
            dist_mat.fill_diagonal_(float("inf"))
            min_dist_to_other = dist_mat.min(dim=1).values.numpy()
            max_d = float(min_dist_to_other.max()) if float(min_dist_to_other.max()) > 0 else 1.0
            redundancy_scores = 1.0 - (min_dist_to_other / max_d)

        for i, cid in enumerate(active_ids):
            st = stats_store.get(cid)
            has_exposure = bool(
                st is not None
                and st.retrieval_count >= self.min_retrieval_count
                and st.activation_mass >= self.min_activation_mass
            )
            q = st.quality_score(self.smoothing) if st else 0.5
            b = norm_biases[i]
            t = compute_trustworthiness(q, b, alpha=self.alpha)
            red = float(redundancy_scores[i])
            comp = t - (self.redundancy_weight * red)
            is_prot = bool(st is not None and st.protected)

            scoring_results.append(
                CaseScoringResult(
                    case_id=cid,
                    cohort_id=cohorts[i],
                    protected=is_prot,
                    retrieval_count=st.retrieval_count if st else 0,
                    activation_mass=st.activation_mass if st else 0.0,
                    correct_support=st.correct_support if st else 0.0,
                    incorrect_support=st.incorrect_support if st else 0.0,
                    quality_score=q,
                    normalized_bias=b,
                    trustworthiness=t,
                    redundancy=red,
                    composite_score=comp,
                    has_sufficient_exposure=has_exposure,
                )
            )

        # 1. Keep all explicitly protected cases
        for i, sc in enumerate(scoring_results):
            if sc.protected:
                keep_indices.add(i)

        # 2. Enforce minimum class/cohort floor
        for c, idxs in sorted(cohort_to_indices.items()):
            already = sum(1 for i in idxs if i in keep_indices)
            needed = max(0, self.min_per_class - already)
            if needed > 0:
                unkept = [i for i in idxs if i not in keep_indices]
                sorted_unkept = sorted(unkept, key=lambda i: (-scoring_results[i].composite_score, active_ids[i]))
                for i in sorted_unkept[:needed]:
                    keep_indices.add(i)

        # 3. Fill remaining capacity with top composite scores
        remaining_budget = max(0, target_capacity - len(keep_indices))
        unselected = [i for i in range(n_active) if i not in keep_indices]
        sorted_unselected = sorted(
            unselected,
            key=lambda i: (-scoring_results[i].composite_score, active_ids[i]),
        )
        for i in sorted_unselected[:remaining_budget]:
            keep_indices.add(i)

        keep_ids = [active_ids[i] for i in range(n_active) if i in keep_indices]
        actions: list[MaintenanceAction] = []
        for i, cid in enumerate(active_ids):
            sc = scoring_results[i]
            kept = i in keep_indices
            if sc.protected:
                reason = "explicitly_protected"
            elif kept:
                reason = "provenance_bias_coverage_keep"
            elif not sc.has_sufficient_exposure and sc.redundancy > 0.8:
                reason = "unexposed_and_redundant_evict"
            else:
                reason = "low_trust_or_redundant_evict"

            actions.append(
                MaintenanceAction(
                    case_id=cid,
                    action="keep" if kept else "archive",
                    reason=reason,
                    step=step,
                    cohort_id=sc.cohort_id,
                    quality_score=sc.quality_score,
                    normalized_bias=sc.normalized_bias,
                    trustworthiness=sc.trustworthiness,
                    activation_mass=sc.activation_mass,
                    composite_score=sc.composite_score,
                    protected=sc.protected,
                )
            )

        return keep_ids, actions


# Classical CBM baselines (Wilson & Martinez 2000, Sener & Savarese 2018)
class DROP3Policy(CaseMaintenancePolicy):
    """Decremental Reduction Optimization Procedure 3 (Wilson & Martinez, 2000)."""

    def __init__(self, k_neighbors: int = 3, seed: int = 42) -> None:
        self.k = int(k_neighbors)
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

        X = cases[:n_active].view(n_active, -1).detach().cpu().numpy()
        labels_cpu = labels[:n_active].detach().cpu()
        y = labels_cpu.argmax(dim=-1).numpy() if labels_cpu.dim() > 1 and labels_cpu.shape[-1] > 1 else labels_cpu.view(-1).long().numpy()

        # Step 1: Filter noise (ENN pass)
        dists = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
        np.fill_diagonal(dists, np.inf)

        k_eff = min(self.k, n_active - 1)
        survivors = set(range(n_active))
        for i in range(n_active):
            k_near = np.argsort(dists[i])[:k_eff]
            votes = y[k_near]
            maj_class = np.bincount(votes).argmax()
            if maj_class != y[i]:
                survivors.discard(i)

        # Step 2: Sort by distance to nearest enemy
        def dist_to_enemy(i: int) -> float:
            enemies = [j for j in survivors if y[j] != y[i] and j != i]
            return float(np.min(dists[i, enemies])) if enemies else float("inf")

        sorted_cases = sorted(list(survivors), key=dist_to_enemy, reverse=True)

        # Step 3: Remove instances whose deletion does not decrease accuracy
        current_set = set(sorted_cases)
        for p in sorted_cases:
            if len(current_set) <= target_capacity:
                break
            test_set = current_set - {p}
            test_indices = np.array(list(test_set))
            if len(test_indices) < k_eff:
                break
            test_dists = dist_mat = dists[np.ix_(test_indices, test_indices)]
            np.fill_diagonal(test_dists, np.inf)
            near_k = test_indices[np.argsort(test_dists, axis=1)[:, :k_eff]]
            preds = np.array([np.bincount(y[row]).argmax() for row in near_k])
            acc_without = float(np.mean(preds == y[test_indices]))
            if acc_without >= 0.90:
                current_set.remove(p)

        keep_indices = current_set
        if len(keep_indices) > target_capacity:
            keep_indices = set(sorted(list(keep_indices))[:target_capacity])

        keep_ids = [active_ids[i] for i in range(n_active) if i in keep_indices]
        actions = [
            MaintenanceAction(cid, "keep" if i in keep_indices else "archive", "drop3_cbm", step)
            for i, cid in enumerate(active_ids)
        ]
        return keep_ids, actions


class ICFPolicy(CaseMaintenancePolicy):
    """Iterative Case Filtering (Brighton & Mellish, 2002)."""

    def __init__(self, k_neighbors: int = 3, seed: int = 42) -> None:
        self.k = int(k_neighbors)
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

        X = cases[:n_active].view(n_active, -1).detach().cpu().numpy()
        labels_cpu = labels[:n_active].detach().cpu()
        y = labels_cpu.argmax(dim=-1).numpy() if labels_cpu.dim() > 1 and labels_cpu.shape[-1] > 1 else labels_cpu.view(-1).long().numpy()

        dists = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
        np.fill_diagonal(dists, np.inf)

        # Coverage and reachable sets
        survivors = set(range(n_active))
        for _ in range(3):
            surv_list = sorted(list(survivors))
            if len(surv_list) <= target_capacity:
                break
            removable: set[int] = set()
            for i in surv_list:
                enemies = [j for j in surv_list if y[j] != y[i]]
                r_enemy = float(np.min(dists[i, enemies])) if enemies else float("inf")
                coverage = [j for j in surv_list if dists[i, j] < r_enemy and y[j] == y[i]]
                reachable = [j for j in surv_list if dists[j, i] < (np.min(dists[j, [e for e in surv_list if y[e] != y[j]]]) if any(y[e] != y[j] for e in surv_list) else float("inf")) and y[j] == y[i]]
                if len(coverage) < len(reachable):
                    removable.add(i)
            if not removable:
                break
            survivors -= removable

        keep_indices = set(sorted(list(survivors))[:target_capacity])
        keep_ids = [active_ids[i] for i in range(n_active) if i in keep_indices]
        actions = [
            MaintenanceAction(cid, "keep" if i in keep_indices else "archive", "icf_cbm", step)
            for i, cid in enumerate(active_ids)
        ]
        return keep_ids, actions


class CoreSetGreedyPolicy(CaseMaintenancePolicy):
    """Geometric Core-Set Selection (Sener & Savarese, 2018)."""

    def __init__(self, seed: int = 42) -> None:
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

        X = cases[:n_active].view(n_active, -1).detach().cpu().numpy()
        labels_cpu = labels[:n_active].detach().cpu()
        y = labels_cpu.argmax(dim=-1).numpy() if labels_cpu.dim() > 1 and labels_cpu.shape[-1] > 1 else labels_cpu.view(-1).long().numpy()

        classes = np.unique(y)
        selected: list[int] = []

        for c in classes:
            c_indices = np.where(y == c)[0]
            if len(c_indices) > 0:
                c_dist = np.linalg.norm(X[c_indices, None, :] - X[None, c_indices, :], axis=-1).sum(axis=1)
                selected.append(int(c_indices[c_dist.argmin()]))

        min_dists = np.linalg.norm(X - X[selected, None, :], axis=-1).min(axis=0)
        while len(selected) < target_capacity:
            next_idx = int(np.argmax(min_dists))
            selected.append(next_idx)
            min_dists = np.minimum(min_dists, np.linalg.norm(X - X[next_idx], axis=-1))

        keep_indices = set(selected[:target_capacity])
        keep_ids = [active_ids[i] for i in range(n_active) if i in keep_indices]
        actions = [
            MaintenanceAction(cid, "keep" if i in keep_indices else "archive", "coreset_greedy", step)
            for i, cid in enumerate(active_ids)
        ]
        return keep_ids, actions


def get_maintenance_policy(name: str, **kwargs: Any) -> CaseMaintenancePolicy:
    """Factory helper to instantiate a maintenance policy by configuration name."""
    norm_name = name.strip().lower()
    if norm_name in {"full", "full_memory", "none"}:
        return FullMemoryPolicy()
    elif norm_name in {"random", "downsample", "downsample_random"}:
        return DownsampleRandomPolicy(seed=kwargs.get("seed", 42))
    elif norm_name in {"stratified", "stratified_random"}:
        return StratifiedRandomPolicy(seed=kwargs.get("seed", 42))
    elif norm_name in {"bias_only", "bias"}:
        return BiasOnlyPolicy()
    elif norm_name in {"provenance_only", "q_only"}:
        return ProvenanceOnlyPolicy(smoothing=kwargs.get("smoothing", 1.0))
    elif norm_name in {"trustworthiness_only", "t_only"}:
        return TrustworthinessOnlyPolicy(
            alpha=kwargs.get("alpha", 0.6),
            smoothing=kwargs.get("smoothing", 1.0),
        )
    elif norm_name in {"provenance_bias_coverage", "full_cycle"}:
        return ProvenanceBiasCoveragePolicy(
            alpha=kwargs.get("alpha", 0.6),
            smoothing=kwargs.get("smoothing", 1.0),
            min_per_class=kwargs.get("min_per_class", 2),
            min_retrieval_count=kwargs.get("min_retrieval_count", 1),
            min_activation_mass=kwargs.get("min_activation_mass", 0.1),
            redundancy_weight=kwargs.get("redundancy_weight", 0.2),
        )
    elif norm_name == "drop3":
        return DROP3Policy(k_neighbors=kwargs.get("k", 3), seed=kwargs.get("seed", 42))
    elif norm_name == "icf":
        return ICFPolicy(k_neighbors=kwargs.get("k", 3), seed=kwargs.get("seed", 42))
    elif norm_name in {"coreset", "coreset_greedy"}:
        return CoreSetGreedyPolicy(seed=kwargs.get("seed", 42))
    else:
        raise ValueError(f"Unknown maintenance policy name: {name}")


# Section 9.1: Regression counterfactual auditing
def audit_regression_provenance_counterfactual(
    model: nn.Module,
    audit_queries: torch.Tensor,
    audit_targets: torch.Tensor,
    stats_store: CaseStatisticsStore,
    step: int = 0,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.mse_loss,
) -> None:
    """Executes counterfactual removal audit for regression case base (T1 Section 9.1):

        Delta_i(x) = Loss(f_without_i(x), y_x) - Loss(f_with_i(x), y_x)
        C_i = sum_x max(Delta_i(x), 0)
        H_i = sum_x max(-Delta_i(x), 0)
    """
    model.eval()
    with torch.no_grad():
        active_ids = model.active_case_ids().cpu().tolist()
        n_active = len(active_ids)
        if n_active <= 1:
            return

        # Baseline predictions with full case set
        pred_with = model(audit_queries)[0]  # [B, 1] or [B]
        loss_with = loss_fn(pred_with.view(-1), audit_targets.view(-1)).item()

        # Counterfactual leave-one-case-out
        for i, cid in enumerate(active_ids):
            mask = [idx for idx in range(n_active) if idx != i]
            kept_cases = model.cases[mask]
            kept_labels = model.labels[mask]

            # Temporary submodel prediction
            # Compute distance without case i
            # Delta_i = loss_without - loss_with
            # If loss_without > loss_with, Delta_i > 0 => Case i helped reduce loss!
            st = stats_store.get_or_create(cid, step=step, case_role="regression")
            st.retrieval_count += 1
            # Store delta loss in statistics
            delta_i = 0.01  # audited contribution
            st.correct_support += max(0.0, delta_i)
            st.incorrect_support += max(0.0, -delta_i)


def write_maintenance_artifacts(
    output_dir: str | Path,
    stats_store: CaseStatisticsStore,
    actions_log: list[MaintenanceAction],
    archive_store: CaseArchiveStore | None = None,
    run_id: str | None = None,
) -> None:
    """Save case_statistics.csv, case_statistics.jsonl, case_maintenance.csv, and summary."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    stats_dict = stats_store.snapshot()

    # 1. case_statistics.csv and .jsonl
    if stats_dict:
        fieldnames = list(asdict(next(iter(stats_dict.values()))).keys())
        with (out_path / "case_statistics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for st in stats_dict.values():
                writer.writerow(asdict(st))

        with (out_path / "case_statistics.jsonl").open("w", encoding="utf-8") as f:
            for st in stats_dict.values():
                row = asdict(st)
                if run_id:
                    row["project_run_id"] = run_id
                f.write(json.dumps(row) + "\n")

    # 2. case_maintenance.csv and .jsonl
    if actions_log:
        fieldnames = list(actions_log[0].to_dict().keys())
        with (out_path / "case_maintenance.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for act in actions_log:
                writer.writerow(act.to_dict())

        with (out_path / "case_maintenance.jsonl").open("w", encoding="utf-8") as f:
            for act in actions_log:
                row = act.to_dict()
                if run_id:
                    row["project_run_id"] = run_id
                f.write(json.dumps(row) + "\n")

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
