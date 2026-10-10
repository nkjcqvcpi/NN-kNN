"""Free-correction radius (plan section 7).

    b_i*     = stop_gradient(b_i^live(t*))            frozen trained-bias snapshot
    d*       = stop_gradient(d_theta(t*))             matching learned distance
    P_bias   = {(i, j): i != j and b_i* - d*(x_j, x_i) >= 0}
               (case i positively activated by training query x_j)
    tau_task = mean_{(i,j) in P_bias} d_y(y_i, y_j)
    L_small  = mean_q [ (max(0, c_q - tau_task) / (s_task + eps))^2 ]

Training cases only; self pairs excluded; computed only at declared outer
checkpoints. Degenerate regions are reported, never silently patched.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import hashlib

import torch

from .geometry import case_case_distance


@dataclass
class FreeRadius:
    tau_task: float
    s_task: float
    n_pairs: int
    n_possible_pairs: int
    pair_fraction: float
    status: str  # ok | empty_region | overly_broad | near_zero
    bias_snapshot_mean: float
    bias_snapshot_std: float
    snapshot_step: int
    label_metric: str
    method: str = "trained_bias_region"
    aggregation: str = "mean"
    bias_transform: str = "identity"

    def to_dict(self) -> dict:
        return asdict(self)


@torch.no_grad()
def calibrate_free_radius(
    model,
    *,
    s_task: float,
    snapshot_step: int,
    broad_fraction: float = 0.5,
    near_zero: float = 1e-6,
    method: str = "trained_bias_region",
    initial_bias: torch.Tensor | None = None,
    neighbor_k: int = 5,
    aggregation: str = "mean",
    bias_transform: str = "identity",
    shrink_factor: float = 0.5,
    clipping_quantile: float = 0.95,
) -> tuple[FreeRadius, dict[str, Any]]:
    if s_task <= 0:
        raise ValueError("s_task must be > 0 (plan: divide by s_task, not tau_task)")
    n = model.case_count()
    b = model.biases[:n].detach().clone()  # b_i*
    trained_b = b.clone()
    D = case_case_distance(model)  # [query j, case i]
    if aggregation not in {"mean", "median"}:
        raise ValueError("aggregation must be mean or median")
    if bias_transform != "identity" and method != "trained_bias_region":
        raise ValueError("trained-bias transformations apply only to trained_bias_region")
    if bias_transform == "shrunk":
        if not 0 < shrink_factor <= 1:
            raise ValueError("shrink_factor must be in (0,1]")
        b = b * shrink_factor
    elif bias_transform == "clipped":
        if not 0 < clipping_quantile <= 1:
            raise ValueError("clipping_quantile must be in (0,1]")
        b = b.clamp(min=0, max=b.quantile(clipping_quantile).clamp_min(0))
    elif bias_transform == "cohort_normalized":
        from .provenance import active_cohorts
        cohorts = active_cohorts(model)
        global_scale = b.abs().mean().clamp_min(1e-8)
        for cohort in sorted(set(cohorts)):
            mask = torch.tensor([c == cohort for c in cohorts], device=b.device)
            b[mask] = b[mask] * global_scale / trained_b[mask].abs().mean().clamp_min(1e-8)
    elif bias_transform != "identity":
        raise ValueError("Unknown trained-bias transform")
    if method == "initial_bias_region":
        if initial_bias is None or initial_bias.numel() != n:
            raise ValueError("initial_bias_region requires biases aligned with active case IDs")
        b = initial_bias.to(b).detach().clone()
    elif method not in {"trained_bias_region", "zero", "exact_kth_pair"}:
        raise ValueError(f"Unknown free-radius method: {method}")
    act = (b.unsqueeze(0) - D) >= 0  # [j, i]
    act.fill_diagonal_(False)
    if method == "exact_kth_pair":
        if not 1 <= neighbor_k < n:
            raise ValueError("neighbor_k must be between 1 and n_cases - 1")
        nonself = D.clone()
        nonself.fill_diagonal_(float("inf"))
        kth = nonself.argsort(dim=1, stable=True)[:, neighbor_k - 1]
        act.zero_()
        act[torch.arange(n, device=D.device), kth] = True
    elif method == "zero":
        act.zero_()
    labels = model.labels[:n].detach().float().clone()
    if model.task_type == "classification":
        same = labels @ labels.t()  # 1 if same class
        dy = torch.sqrt((2.0 * (1.0 - same)).clamp_min(0))
        metric = "one_hot_euclidean"
    else:
        y = labels.view(n, -1)[:, 0]
        dy = (y.unsqueeze(0) - y.unsqueeze(1)).abs()
        metric = "absolute_difference"
    n_pairs = int(act.sum())
    possible = n * (n - 1)
    tau = float(dy[act].mean() if aggregation == "mean" else dy[act].median()) if n_pairs else 0.0
    frac = n_pairs / max(possible, 1)
    status = "ok"
    if method == "zero":
        status = "zero_threshold_ablation"
    elif n_pairs == 0:
        status = "empty_region"
    elif frac > broad_fraction:
        status = "overly_broad"
    elif tau < near_zero:
        status = "near_zero"
    fr = FreeRadius(tau, float(s_task), n_pairs, possible, frac, status, float(b.mean()), float(b.std()) if n > 1 else 0.0, snapshot_step, metric, method, aggregation, bias_transform)
    # Save the actual measuring scale and pair identities, not two floating-point
    # sums that cannot reconstruct the frozen activation regions.
    return fr, {"bias_snapshot": b.cpu(), "trained_bias_snapshot": trained_b.cpu(), "distance_snapshot": D.cpu(),
                "labels_snapshot": labels.cpu(), "pair_mask": act.cpu(),
                "case_ids": model.active_case_ids().detach().cpu().clone(),
                "free_radius": fr.to_dict(),
                "transformation_parameters": {"shrink_factor": shrink_factor, "clipping_quantile": clipping_quantile},
                "distance_snapshot_hash": torch.tensor([float(D.sum()), float((D ** 2).sum())])}


def l_small(c_q: torch.Tensor, fr: FreeRadius, eps: float = 1e-8) -> torch.Tensor:
    return (torch.clamp(c_q - fr.tau_task, min=0.0) / (fr.s_task + eps)).pow(2).mean()


def compact_calibration_snapshot(snapshot: dict, max_matrix_bytes: int = 64 * 1024**2) -> dict:
    """Keep exact matrices for small runs, or frozen-state reconstruction evidence.

    Large matrices can be regenerated using model_state, active_case_count,
    model_description in checkpoint.pt, and the bundled source. Cryptographic
    digests verify the regenerated distance and pair identity tensors.
    """
    if "model_state" not in snapshot or "active_case_count" not in snapshot:
        raise ValueError("Compact calibration requires a frozen model and active case count")
    matrices = ("distance_snapshot", "pair_mask")
    size = sum(snapshot[k].numel() * snapshot[k].element_size() for k in matrices)
    result = dict(snapshot)
    result["matrix_storage"] = "exact" if size <= max_matrix_bytes else "reconstruct_from_frozen_model"
    result["matrix_evidence"] = {}
    for key in matrices:
        tensor = snapshot[key].cpu().contiguous()
        result["matrix_evidence"][key] = dict(shape=list(tensor.shape), dtype=str(tensor.dtype),
            sha256=hashlib.sha256(memoryview(tensor.numpy())).hexdigest())
        if size > max_matrix_bytes:
            del result[key]
    return result
