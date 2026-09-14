"""Neural Revise Stage for Full-Cycle Case-Based Reasoning (T0).

Implements the 3rd R ('Revise') of the 4R CBR cycle through neural consensus-based
validation and anomaly correction. Detects noisy, conflicting, or poisoned cases
using neighborhood agreement and proposes calibrated label corrections or quarantine
penalties with an audit log.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class RevisionRecord:
    """Audit record for a case revision action."""
    case_id: int
    original_label: int
    proposed_label: int
    consensus_confidence: float
    conflict_score: float
    action: str  # 'keep', 'relabel', 'quarantine', 'bias_penalty'
    step: int
    details: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class NeuralCaseReviser:
    """Neural consensus-guided case reviser.
    
    Evaluates each case against the soft consensus of its nearest neighbors in
    the learned latent representation space. Cases whose stored solution conflicts
    strongly with high-confidence neighborhood consensus are revised or quarantined.
    """

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
        """Perform neural revision on a case base.
        
        Args:
            cases: [N, D] case feature representations.
            labels: [N, C] one-hot or integer labels.
            biases: [N] case learned biases (optional).
            step: current step or epoch for audit logging.
            
        Returns:
            revised_cases: [N', D] cases tensor.
            revised_labels: [N', C] labels tensor.
            revised_biases: [N'] biases tensor.
            records: list of RevisionRecord documenting all actions.
        """
        device = cases.device
        N = cases.shape[0]
        if N <= self.k_neighbors:
            return cases, labels, (biases if biases is not None else torch.zeros(N, device=device)), []

        # Convert labels to class integers and one-hot
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

        # Compute pairwise Euclidean distance in representation space
        flat_cases = cases.view(N, -1)
        dist_matrix = torch.cdist(flat_cases, flat_cases, p=2)
        # Mask self-distance with infinity
        dist_matrix.fill_diagonal_(float("inf"))

        # Find k nearest neighbors for each case
        topk_dists, topk_indices = torch.topk(dist_matrix, k=self.k_neighbors, dim=1, largest=False)

        # Softmax weights based on negative distance
        topk_weights = F.softmax(-topk_dists, dim=1)  # [N, k]

        keep_mask = torch.ones(N, dtype=torch.bool, device=device)

        for i in range(N):
            orig_c = int(class_ints[i].item())
            neighbor_idxs = topk_indices[i]  # [k]
            neighbor_weights = topk_weights[i]  # [k]
            neighbor_one_hot = one_hot_labels[neighbor_idxs]  # [k, C]

            # Weighted neighborhood consensus distribution
            consensus_probs = (neighbor_weights.unsqueeze(-1) * neighbor_one_hot).sum(dim=0)  # [C]
            
            top_consensus_prob, proposed_class_t = torch.max(consensus_probs, dim=0)
            top_prob = float(top_consensus_prob.item())
            proposed_class = int(proposed_class_t.item())
            orig_prob = float(consensus_probs[orig_c].item())

            conflict_score = 1.0 - orig_prob

            # Anomaly condition: neighborhood strongly favors another class and disagrees with orig
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
                    revised_biases[i] -= self.bias_penalty
                    details = f"Applied bias penalty -{self.bias_penalty} (new bias: {revised_biases[i]:.3f})"
                else:
                    action = "keep"
                    details = "No-op revision"
            else:
                action = "keep"
                details = "Consistent with neighborhood"

            record = RevisionRecord(
                case_id=i,
                original_label=orig_c,
                proposed_label=proposed_class if is_anomaly else orig_c,
                consensus_confidence=top_prob,
                conflict_score=conflict_score,
                action=action,
                step=step,
                details=details,
            )
            records.append(record)
            self.audit_log.append(record)

        if self.revision_mode == "quarantine":
            return cases[keep_mask], revised_labels[keep_mask], revised_biases[keep_mask], records

        return cases, revised_labels, revised_biases, records


def inject_synthetic_noise(
    labels: torch.Tensor | np.ndarray,
    noise_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[torch.Tensor, np.ndarray]:
    """Inject controlled label noise for counterfactual revision testing.
    
    Args:
        labels: tensor of labels (integers or one-hot).
        noise_ratio: fraction of labels to perturb.
        seed: random seed for reproducibility.
        
    Returns:
        noisy_labels: tensor of corrupted labels.
        corrupted_mask: boolean numpy array marking corrupted indices.
    """
    is_tensor = isinstance(labels, torch.Tensor)
    if is_tensor:
        if labels.dim() > 1 and labels.shape[-1] > 1:
            arr = labels.argmax(dim=-1).cpu().numpy()
            is_one_hot = True
            num_classes = labels.shape[-1]
        else:
            arr = labels.view(-1).cpu().numpy()
            is_one_hot = False
            num_classes = int(arr.max()) + 1
    else:
        arr = np.asarray(labels)
        is_one_hot = False
        num_classes = int(arr.max()) + 1

    rng = np.random.default_rng(seed)
    n = len(arr)
    n_corrupt = int(round(n * noise_ratio))
    corrupted_indices = rng.choice(n, size=n_corrupt, replace=False)
    corrupted_mask = np.zeros(n, dtype=bool)
    corrupted_mask[corrupted_indices] = True

    noisy_arr = arr.copy()
    for idx in corrupted_indices:
        orig = arr[idx]
        other_classes = [c for c in range(num_classes) if c != orig]
        noisy_arr[idx] = rng.choice(other_classes)

    if is_tensor:
        noisy_t = torch.tensor(noisy_arr, dtype=torch.long, device=labels.device)
        if is_one_hot:
            noisy_t = F.one_hot(noisy_t, num_classes=num_classes).float()
        return noisy_t, corrupted_mask
    return torch.tensor(noisy_arr), corrupted_mask
