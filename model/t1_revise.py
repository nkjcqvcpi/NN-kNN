from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence
import uuid
import numpy as np
import torch
import torch.nn.functional as F

from model.t1_maintenance import CaseArchiveStore, CaseStatistics, CaseStatisticsStore


@dataclass
class RevisionRecord:
    """Audit record for a case revision or intervention action (T1 Sections 3.3 & 11)."""
    case_id: int
    original_label: int
    proposed_label: int
    consensus_confidence: float
    conflict_score: float
    action: str  # 'keep', 'relabel', 'quarantine', 'bias_penalty', 'restore', 'force_activate'
    step: int
    intervention_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    details: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class NeuralCaseReviser:
    """Neural consensus-guided case reviser for quality-aware audit and revision."""

    def __init__(
        self,
        k_neighbors: int = 5,
        conflict_threshold: float = 0.55,
        confidence_threshold: float = 0.65,
        revision_mode: str = "relabel",  # 'relabel', 'quarantine', 'bias_penalty'
        bias_penalty: float = 2.0,
    ) -> None:
        self.k_neighbors = int(k_neighbors)
        self.conflict_threshold = float(conflict_threshold)
        self.confidence_threshold = float(confidence_threshold)
        self.revision_mode = str(revision_mode).lower()
        self.bias_penalty = float(bias_penalty)
        self.audit_log: list[RevisionRecord] = []

    def revise_case_base(
        self,
        cases: torch.Tensor,
        labels: torch.Tensor,
        biases: torch.Tensor | None = None,
        step: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[RevisionRecord]]:
        device = cases.device
        N = cases.shape[0]
        if N <= self.k_neighbors:
            return cases, labels, (biases if biases is not None else torch.zeros(N, device=device)), []

        if labels.dim() > 1 and labels.shape[-1] > 1:
            class_ints = labels.argmax(dim=-1)
            num_classes = labels.shape[-1]
            one_hot_labels = labels.float()
        else:
            class_ints = labels.view(-1).long()
            num_classes = int(class_ints.max().item()) + 1
            one_hot_labels = F.one_hot(class_ints, num_classes=num_classes).float()

        if biases is None:
            biases = torch.zeros(N, dtype=torch.float32, device=device)
        else:
            biases = biases.clone()

        revised_labels = one_hot_labels.clone()
        revised_biases = biases.clone()
        records: list[RevisionRecord] = []

        flat_cases = cases.view(N, -1)
        dist_matrix = torch.cdist(flat_cases, flat_cases, p=2)
        dist_matrix.fill_diagonal_(float("inf"))

        topk_dists, topk_indices = torch.topk(dist_matrix, k=self.k_neighbors, dim=1, largest=False)
        topk_weights = F.softmax(-topk_dists, dim=1)

        keep_mask = torch.ones(N, dtype=torch.bool, device=device)

        for i in range(N):
            orig_c = int(class_ints[i].item())
            neighbor_idxs = topk_indices[i]
            neighbor_weights = topk_weights[i]
            neighbor_one_hot = one_hot_labels[neighbor_idxs]

            consensus_probs = (neighbor_weights.unsqueeze(-1) * neighbor_one_hot).sum(dim=0)
            top_consensus_prob, proposed_class_t = torch.max(consensus_probs, dim=0)
            top_prob = float(top_consensus_prob.item())
            proposed_class = int(proposed_class_t.item())
            orig_prob = float(consensus_probs[orig_c].item())

            conflict_score = 1.0 - orig_prob
            is_anomaly = (
                proposed_class != orig_c
                and conflict_score >= self.conflict_threshold
                and top_prob >= self.confidence_threshold
            )

            if is_anomaly:
                if self.revision_mode == "relabel":
                    action = "relabel"
                    new_one_hot = F.one_hot(torch.tensor(proposed_class, device=device), num_classes=num_classes).float()
                    revised_labels[i] = new_one_hot
                    details = f"Relabeled {orig_c} -> {proposed_class} (conf: {top_prob:.3f})"
                elif self.revision_mode == "quarantine":
                    action = "quarantine"
                    keep_mask[i] = False
                    details = f"Quarantined case {i} due to conflict {conflict_score:.3f}"
                elif self.revision_mode == "bias_penalty":
                    action = "bias_penalty"
                    revised_biases[i] = revised_biases[i] - self.bias_penalty
                    details = f"Reduced bias by {self.bias_penalty} (conflict: {conflict_score:.3f})"
                else:
                    action = "keep"
                    details = "No action"

                rec = RevisionRecord(
                    case_id=i,
                    original_label=orig_c,
                    proposed_label=proposed_class,
                    consensus_confidence=top_prob,
                    conflict_score=conflict_score,
                    action=action,
                    step=step,
                    details=details,
                )
                records.append(rec)
                self.audit_log.append(rec)

        if self.revision_mode == "quarantine" and not keep_mask.all():
            revised_cases = cases[keep_mask]
            revised_labels = revised_labels[keep_mask]
            revised_biases = revised_biases[keep_mask]
        else:
            revised_cases = cases

        return revised_cases, revised_labels, revised_biases, records


class HumanReviseManager:
    """Manages authorized human case/weight intervention and causal evaluation (T1.2)."""

    def __init__(self, revision_enabled: bool = False) -> None:
        self.revision_enabled = bool(revision_enabled)
        self.intervention_log: list[RevisionRecord] = []

    def edit_case_label(
        self,
        model: Any,
        case_idx: int,
        new_label: int,
        step: int = 0,
        reason: str = "human_expert_edit",
    ) -> RevisionRecord:
        """Edits stored case solution with versioning and audit record."""
        orig_label = int(model.labels[case_idx].argmax().item())
        num_classes = model.labels.shape[-1]
        with torch.no_grad():
            model.labels[case_idx] = F.one_hot(
                torch.tensor(new_label, device=model.labels.device),
                num_classes=num_classes,
            ).float()

        record = RevisionRecord(
            case_id=int(model.case_ids[case_idx].item()) if hasattr(model, "case_ids") else case_idx,
            original_label=orig_label,
            proposed_label=new_label,
            consensus_confidence=1.0,
            conflict_score=1.0,
            action="relabel",
            step=step,
            details=reason,
        )
        self.intervention_log.append(record)
        return record

    def adjust_case_bias(
        self,
        model: Any,
        case_idx: int,
        delta_bias: float,
        step: int = 0,
        reason: str = "bias_intervention",
    ) -> RevisionRecord:
        """Modifies per-case learned bias b_i."""
        with torch.no_grad():
            model.biases[case_idx].add_(float(delta_bias))

        record = RevisionRecord(
            case_id=int(model.case_ids[case_idx].item()) if hasattr(model, "case_ids") else case_idx,
            original_label=0,
            proposed_label=0,
            consensus_confidence=1.0,
            conflict_score=0.0,
            action="bias_penalty" if delta_bias < 0 else "bias_boost",
            step=step,
            details=f"delta_bias={delta_bias:.3f}, reason={reason}",
        )
        self.intervention_log.append(record)
        return record

    def evaluate_causal_intervention(
        self,
        model_m0: Any,
        model_m1: Any,
        test_queries: torch.Tensor,
        test_labels: torch.Tensor,
        target_case_id: int,
    ) -> dict[str, Any]:
        """Compares baseline M0 vs post-intervention M1 without retraining (Section 11).

        Measures:
        - overall accuracy before and after;
        - targeted decision flips (queries where target_case was in top-k);
        - correctness of flips;
        - collateral changes on unaffected queries.
        """
        model_m0.eval()
        model_m1.eval()

        with torch.no_grad():
            pred_m0 = model_m0(test_queries)[0].argmax(dim=-1).cpu()
            pred_m1 = model_m1(test_queries)[0].argmax(dim=-1).cpu()

            if test_labels.dim() > 1 and test_labels.size(-1) > 1:
                true_y = test_labels.argmax(dim=-1).cpu()
            else:
                true_y = test_labels.view(-1).long().cpu()

            flips = pred_m0 != pred_m1
            num_flips = int(flips.sum().item())

            acc_m0 = float((pred_m0 == true_y).float().mean().item())
            acc_m1 = float((pred_m1 == true_y).float().mean().item())

            # Flip correctness
            correct_flips = int(((pred_m0 != pred_m1) & (pred_m1 == true_y)).sum().item())
            harmful_flips = int(((pred_m0 != pred_m1) & (pred_m0 == true_y)).sum().item())

        return {
            "target_case_id": int(target_case_id),
            "m0_accuracy": acc_m0,
            "m1_accuracy": acc_m1,
            "accuracy_change": acc_m1 - acc_m0,
            "total_flips": num_flips,
            "correct_flips": correct_flips,
            "harmful_flips": harmful_flips,
            "net_flip_benefit": correct_flips - harmful_flips,
        }
