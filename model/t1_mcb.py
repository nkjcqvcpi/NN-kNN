from __future__ import annotations

import copy
from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F


class MCBProjectionHead(nn.Module):
    """Two-layer projection head with LayerNorm and GELU from AAAI 2027 Eq. (3).

    Maps backbone features (or flat tabular features) to normalized embeddings:
        z = W_2 * GELU(LN(W_1 * v(x)))
        f(x) = z / ||z||_2  (if normalize=True)
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 64,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.normalize = bool(normalize)
        self.feature_dim = self.out_dim

        self.net = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.out_dim),
            nn.LayerNorm(self.out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        z = self.net(x.float())
        if self.normalize:
            z = F.normalize(z, p=2, dim=-1)
        return z


class MCBEncoder(nn.Module):
    """Combines an optional base extractor with an MCB projection head."""

    def __init__(self, base_extractor: nn.Module | None, proj_head: MCBProjectionHead) -> None:
        super().__init__()
        self.base_extractor = base_extractor
        self.proj_head = proj_head
        self.feature_dim = int(proj_head.feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.base_extractor is not None:
            x = self.base_extractor(x)
        return self.proj_head(x)


def build_mcb_encoder_pair(
    base_extractor: nn.Module | None,
    in_dim: int,
    hidden_dim: int = 64,
    proj_dim: int = 64,
    normalize: bool = True,
) -> tuple[MCBEncoder, MCBEncoder]:
    """Constructs (online_encoder, memory_encoder) pair for Momentum Case Base (MCB-R).

    The online encoder is trained with gradients.
    The memory encoder is initialized as an exact deepcopy, frozen with requires_grad=False,
    and kept in eval mode.
    """
    proj_head = MCBProjectionHead(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=proj_dim,
        normalize=normalize,
    )
    online_encoder = MCBEncoder(base_extractor, proj_head)
    memory_encoder = copy.deepcopy(online_encoder)
    memory_encoder.requires_grad_(False)
    memory_encoder.eval()
    return online_encoder, memory_encoder


def compute_representation_drift(
    features_t1: torch.Tensor,
    features_t2: torch.Tensor,
    metric: str = "cosine",
) -> float:
    """Computes representation drift between two feature snapshots of the same data.

    Returns:
        float: Mean drift across instances.
    """
    if features_t1.shape != features_t2.shape:
        raise ValueError(f"Shape mismatch: {features_t1.shape} vs {features_t2.shape}")

    f1 = features_t1.detach().float()
    f2 = features_t2.detach().float()

    if metric == "cosine":
        cos_sim = F.cosine_similarity(f1, f2, dim=-1)
        drift = 1.0 - cos_sim.mean().item()
        return float(max(0.0, drift))
    elif metric == "l2":
        dist = torch.norm(f1 - f2, p=2, dim=-1)
        return float(dist.mean().item())
    else:
        raise ValueError(f"Unknown metric: {metric}")


def compute_neighborhood_churn(
    top_k_indices_t1: torch.Tensor | Sequence[Sequence[int]],
    top_k_indices_t2: torch.Tensor | Sequence[Sequence[int]],
) -> float:
    """Computes retrieved-neighborhood churn using mean Jaccard distance:

        churn = 1 - (|N_1 cap N_2| / |N_1 cup N_2|)

    A churn of 0 means identical retrieved neighborhoods; 1 means disjoint neighborhoods.
    """
    if isinstance(top_k_indices_t1, torch.Tensor):
        list1 = top_k_indices_t1.detach().cpu().tolist()
    else:
        list1 = [list(x) for x in top_k_indices_t1]

    if isinstance(top_k_indices_t2, torch.Tensor):
        list2 = top_k_indices_t2.detach().cpu().tolist()
    else:
        list2 = [list(x) for x in top_k_indices_t2]

    if len(list1) != len(list2):
        raise ValueError(f"Length mismatch: {len(list1)} vs {len(list2)}")

    jaccard_dists: list[float] = []
    for s1, s2 in zip(list1, list2):
        set1 = set(s1)
        set2 = set(s2)
        union = len(set1 | set2)
        if union == 0:
            jaccard_dists.append(0.0)
        else:
            intersection = len(set1 & set2)
            jaccard_dists.append(1.0 - (intersection / union))

    return float(sum(jaccard_dists) / max(len(jaccard_dists), 1))


def compute_selection_churn(
    kept_ids_t1: Sequence[int],
    kept_ids_t2: Sequence[int],
) -> float:
    """Computes case-base maintenance selection churn between consecutive maintenance cycles."""
    set1 = set(kept_ids_t1)
    set2 = set(kept_ids_t2)
    union = len(set1 | set2)
    if union == 0:
        return 0.0
    return float(1.0 - (len(set1 & set2) / union))
