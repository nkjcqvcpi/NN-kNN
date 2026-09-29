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
) -> tuple[FreeRadius, dict[str, torch.Tensor]]:
    if s_task <= 0:
        raise ValueError("s_task must be > 0 (plan: divide by s_task, not tau_task)")
    n = model.case_count()
    b = model.biases[:n].detach().clone()  # b_i*
    D = case_case_distance(model)  # [query j, case i]
    act = (b.unsqueeze(0) - D) >= 0  # [j, i]
    act.fill_diagonal_(False)
    labels = model.labels[:n].float()
    if model.task_type == "classification":
        same = labels @ labels.t()  # 1 if same class
        dy = torch.sqrt(2.0 * (1.0 - same)).clamp_min(0)
        metric = "one_hot_euclidean"
    else:
        y = labels.view(n, -1)[:, 0]
        dy = (y.unsqueeze(0) - y.unsqueeze(1)).abs()
        metric = "absolute_difference"
    n_pairs = int(act.sum())
    possible = n * (n - 1)
    tau = float(dy[act].mean()) if n_pairs else 0.0
    frac = n_pairs / max(possible, 1)
    status = "ok"
    if n_pairs == 0:
        status = "empty_region"
    elif frac > broad_fraction:
        status = "overly_broad"
    elif tau < near_zero:
        status = "near_zero"
    fr = FreeRadius(tau, float(s_task), n_pairs, possible, frac, status, float(b.mean()), float(b.std()) if n > 1 else 0.0, snapshot_step, metric)
    return fr, {"bias_snapshot": b.cpu(), "distance_snapshot_hash": torch.tensor([float(D.sum()), float((D ** 2).sum())])}


def l_small(c_q: torch.Tensor, fr: FreeRadius, eps: float = 1e-8) -> torch.Tensor:
    return (torch.clamp(c_q - fr.tau_task, min=0.0) / (fr.s_task + eps)).pow(2).mean()
