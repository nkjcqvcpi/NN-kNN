from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FreeRadiusCalibrationResult:
    """Artifact storing the calibrated free-correction radius and snapshot metadata."""
    tau_task: float
    s_task: float
    num_positive_pairs: int
    total_candidate_pairs: int
    bias_snapshot_mean: float
    bias_snapshot_std: float
    metric_name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "tau_task": float(self.tau_task),
            "s_task": float(self.s_task),
            "num_positive_pairs": int(self.num_positive_pairs),
            "total_candidate_pairs": int(self.total_candidate_pairs),
            "bias_snapshot_mean": float(self.bias_snapshot_mean),
            "bias_snapshot_std": float(self.bias_snapshot_std),
            "metric_name": str(self.metric_name),
        }


def calibrate_free_correction_radius(
    model: nn.Module,
    cases: torch.Tensor,
    labels: torch.Tensor,
    task_type: str = "classification",
    label_metric: str = "one_hot_euclidean",
    max_pairs: int = 5000,
    seed: int = 42,
) -> FreeRadiusCalibrationResult:
    """Calibrates task-specific free-correction radius tau_task (T1 Section 7).

    Evaluates learned distance and frozen trained bias on training case pairs (i != j):
        P_bias = {(i, j) : i != j and b_i^* - d_theta^*(x_i, x_j) >= 0}
        tau_task = mean_{(i, j) in P_bias} d_y(y_i, y_j)
    """
    rng = np.random.default_rng(seed)
    n_cases = cases.size(0)

    # 1. Capture stop-gradient snapshots
    with torch.no_grad():
        biases = model.biases[:n_cases].detach().cpu().clone()
        b_mean = float(biases.mean().item())
        b_std = float(biases.std().item()) if n_cases > 1 else 0.0

        # Sample non-self pairs
        if n_cases < 2:
            return FreeRadiusCalibrationResult(
                tau_task=0.0,
                s_task=1.0,
                num_positive_pairs=0,
                total_candidate_pairs=0,
                bias_snapshot_mean=b_mean,
                bias_snapshot_std=b_std,
                metric_name=label_metric,
            )

        n_pairs = min(max_pairs, n_cases * (n_cases - 1) // 2)
        idx_i = rng.integers(0, n_cases, size=n_pairs * 2)
        idx_j = rng.integers(0, n_cases, size=n_pairs * 2)
        valid = idx_i != idx_j
        idx_i = idx_i[valid][:n_pairs]
        idx_j = idx_j[valid][:n_pairs]

        t_i = torch.from_numpy(idx_i).to(cases.device)
        t_j = torch.from_numpy(idx_j).to(cases.device)

        # Compute learned distance between pairs
        cases_i = cases[t_i]
        cases_j = cases[t_j]

        if hasattr(model, "feature_extractor") and model.feature_extractor is not None:
            feat_i = model.feature_extractor(cases_i)
            feat_j = model.feature_extractor(cases_j)
            if getattr(model, "mcb_normalize_embeddings", False):
                feat_i = F.normalize(feat_i, p=2, dim=-1)
                feat_j = F.normalize(feat_j, p=2, dim=-1)
        else:
            feat_i = cases_i.view(cases_i.size(0), -1)
            feat_j = cases_j.view(cases_j.size(0), -1)

        elem_dist = (feat_i - feat_j) ** 2
        if hasattr(model, "glocal_weightor") and model.glocal_weightor is not None:
            gw = model.glocal_weights[t_i]
            elem_dist = model.glocal_weightor(elem_dist.unsqueeze(1), gw).squeeze(1)

        pair_dists = torch.sqrt(torch.relu(torch.sum(elem_dist, dim=-1))).cpu()
        pair_biases = biases[idx_i]

        # Positive activation condition: b_i^* - d_theta^*(x_i, x_j) >= 0
        in_region = (pair_biases - pair_dists) >= 0

        # Compute label distance d_y
        labels_i = labels[idx_i].cpu()
        labels_j = labels[idx_j].cpu()

        if task_type == "classification":
            if label_metric == "one_hot_euclidean":
                dy = torch.norm(labels_i.float() - labels_j.float(), p=2, dim=-1)
            else:  # 0-1 disagreement
                cls_i = labels_i.argmax(dim=-1)
                cls_j = labels_j.argmax(dim=-1)
                dy = (cls_i != cls_j).float()
        else:  # regression
            dy = torch.abs(labels_i.float() - labels_j.float()).view(-1)

        all_dy = dy.numpy()
        s_task = float(np.std(all_dy)) if len(all_dy) > 1 else 1.0
        if s_task < 1e-4:
            s_task = 1.0

        n_pos = int(in_region.sum().item())
        if n_pos > 0:
            tau_task = float(dy[in_region].mean().item())
        else:
            # Fallback when region is empty: mean of smallest 5% distances
            tau_task = 0.0

    return FreeRadiusCalibrationResult(
        tau_task=tau_task,
        s_task=s_task,
        num_positive_pairs=n_pos,
        total_candidate_pairs=len(idx_i),
        bias_snapshot_mean=b_mean,
        bias_snapshot_std=b_std,
        metric_name=label_metric,
    )


def compute_minimal_adaptation_penalty(
    pre_adaptation_output: torch.Tensor,
    post_adaptation_output: torch.Tensor,
    tau_task: float,
    s_task: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Computes scale-normalized minimal-adaptation penalty L_small (T1 Eq.):

        c_q = d_y(output_pre_q, output_post_q)
        L_small = mean_q [ (max(0, c_q - tau_task) / (s_task + eps))^2 ]
    """
    diff = post_adaptation_output - pre_adaptation_output
    if diff.dim() > 1 and diff.size(-1) > 1:
        c_q = torch.norm(diff, p=2, dim=-1)
    else:
        c_q = torch.abs(diff).squeeze(-1)

    excess = torch.relu(c_q - float(tau_task))
    normalized_excess = excess / (float(s_task) + eps)
    return torch.mean(normalized_excess ** 2)


@dataclass
class SynchronizedLossTerms:
    """Separately inspectable loss terms for CBR component synchronization (T1 Section 6)."""
    loss_pre: float
    loss_post: float
    loss_near: float
    loss_delta: float
    loss_small: float
    loss_retrieval_total: float
    loss_adapter_total: float

    def to_dict(self) -> dict[str, float]:
        return {
            "loss_pre": self.loss_pre,
            "loss_post": self.loss_post,
            "loss_near": self.loss_near,
            "loss_delta": self.loss_delta,
            "loss_small": self.loss_small,
            "loss_retrieval_total": self.loss_retrieval_total,
            "loss_adapter_total": self.loss_adapter_total,
        }


class ComponentSynchronizer:
    """Coordinates retrieval and adaptation learning via multi-term synchronized losses."""

    def __init__(
        self,
        w_pre: float = 1.0,
        w_post: float = 0.5,
        w_near: float = 0.01,
        v_post: float = 1.0,
        v_delta: float = 0.5,
        v_small: float = 0.1,
        free_radius: float = 0.0,
        scale_task: float = 1.0,
    ) -> None:
        self.w_pre = float(w_pre)
        self.w_post = float(w_post)
        self.w_near = float(w_near)
        self.v_post = float(v_post)
        self.v_delta = float(v_delta)
        self.v_small = float(v_small)
        self.free_radius = float(free_radius)
        self.scale_task = float(scale_task)

    def compute_losses(
        self,
        p0_q: torch.Tensor,
        s_q: torch.Tensor,
        target_labels: torch.Tensor,
        residual_hat: torch.Tensor | None = None,
        residual_target: torch.Tensor | None = None,
        query_to_case_dist: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, SynchronizedLossTerms]:
        """Computes separable L_R and L_A objectives.

        L_R = w_pre * L_pre + w_post * L_post + w_near * L_near
        L_A = v_post * L_post + v_delta * L_delta + v_small * L_small
        """
        # L_pre: task loss before adaptation
        if target_labels.dim() > 1 and target_labels.size(-1) > 1:
            target_cls = target_labels.argmax(dim=-1)
        else:
            target_cls = target_labels.long().view(-1)

        l_pre = F.cross_entropy(torch.log(torch.clamp(p0_q, 1e-8, 1.0)), target_cls)
        l_post = F.cross_entropy(s_q, target_cls)

        # L_near: query-to-retrieved-case proximity
        if query_to_case_dist is not None:
            l_near = torch.mean(query_to_case_dist)
        else:
            l_near = torch.tensor(0.0, device=p0_q.device)

        # L_delta: difference / residual prediction loss
        if residual_hat is not None and residual_target is not None:
            l_delta = F.mse_loss(residual_hat, residual_target)
        else:
            l_delta = torch.tensor(0.0, device=p0_q.device)

        # L_small: free-correction radius penalty
        l_small = compute_minimal_adaptation_penalty(
            p0_q,
            F.softmax(s_q, dim=-1),
            tau_task=self.free_radius,
            s_task=self.scale_task,
        )

        l_retrieval = self.w_pre * l_pre + self.w_post * l_post + self.w_near * l_near
        l_adapter = self.v_post * l_post + self.v_delta * l_delta + self.v_small * l_small

        terms = SynchronizedLossTerms(
            loss_pre=float(l_pre.item()),
            loss_post=float(l_post.item()),
            loss_near=float(l_near.item()),
            loss_delta=float(l_delta.item()),
            loss_small=float(l_small.item()),
            loss_retrieval_total=float(l_retrieval.item()),
            loss_adapter_total=float(l_adapter.item()),
        )
        return l_retrieval, l_adapter, terms
