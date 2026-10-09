"""Shared learned-distance interface (plan section 7 / handoff: one versioned geometry).

Every T1 component that needs problem-space proximity (redundancy, coverage,
L_near, free-correction pairs, drift/churn) must go through these helpers so it
uses the *same* learned representation, glocal feature weights and distance as
NN-kNN retrieval, never an unrelated raw-input metric.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def active_case_features(model) -> torch.Tensor:
    """Representation of all active cases as seen by retrieval (eval mode)."""
    was = model.training
    model.eval()
    n = model.case_count()
    idx = torch.arange(n, device=model.cases.device)
    feats = model._extract_features(idx).detach().clone()
    if was:
        model.train()
    return feats


@torch.no_grad()
def query_features(model, X: torch.Tensor) -> torch.Tensor:
    import torch.nn.functional as F

    was = model.training
    model.eval()
    X = X.to(model.cases.device)
    if model.feature_extractor is not None:
        q = model.feature_extractor(X)
        if getattr(model, "mcb_normalize_embeddings", False):
            q = F.normalize(q, p=2, dim=-1)
    else:
        q = X
    if was:
        model.train()
    return q


def learned_distance(model, query_feats: torch.Tensor, case_feats: torch.Tensor, case_slots: torch.Tensor) -> torch.Tensor:
    """Distance used by retrieval between query features [B, D] and cases [M, D] at active slots.

    The glocal weighting of the *case* (target) side is applied, exactly as in
    ``NN_KNN_Model.retrieve``.
    """
    q = query_feats.unsqueeze(1).expand(-1, case_feats.size(0), -1)
    c = case_feats.unsqueeze(0).expand(query_feats.size(0), -1, -1)
    if model.feature_distance_metric is None:
        el = (q - c) ** 2
    else:
        el = model.feature_distance_metric(q, c)
    if model.glocal_weightor is not None:
        el = model.glocal_weightor(el, model.glocal_weights[case_slots])
    return torch.sqrt(torch.relu(el.sum(-1)))


@torch.no_grad()
def case_case_distance(model, chunk: int = 512) -> torch.Tensor:
    """[N, N] learned distance between active cases (row = query case, col = stored case)."""
    feats = active_case_features(model)
    n = feats.size(0)
    slots = torch.arange(n, device=feats.device)
    out = torch.empty(n, n, device=feats.device)
    for s in range(0, n, chunk):
        out[s : s + chunk] = learned_distance(model, feats[s : s + chunk], feats, slots)
    return out
