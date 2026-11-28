"""Non-NN-CDH adapters: vote attention, residual memory and anchor transport."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from .pairwise_adaptation import predict_pairs


def normalize_probabilities(value):
    value = np.maximum(np.asarray(value, dtype=np.float64), 0.)
    total = value.sum(1, keepdims=True)
    if not np.isfinite(value).all() or np.any(total <= 0):
        raise ValueError("cannot normalize invalid class scores")
    return value / total


def point_predict(model, z, kind, output_dim):
    if kind == "regression":
        return model.predict(z).reshape(-1, 1)
    result = np.zeros((len(z), output_dim), dtype=np.float64)
    result[:, np.asarray(model.classes_, dtype=int)] = model.predict_proba(z)
    return result


def retrieved_pairs(context):
    part = context["train"]
    n, k = part["indices"].shape
    qi, ci = np.repeat(np.arange(n), k), part["indices"].reshape(-1)
    weight = np.asarray(part["weights"], dtype=np.float64).reshape(-1)
    keep = weight > 0
    qi, ci, weight = qi[keep], ci[keep], weight[keep]
    if np.any(qi == ci) or not np.allclose(np.bincount(qi, weights=weight), 1., atol=1e-6):
        raise ValueError("retrieved pairs violate self-exclusion or equal query mass")
    targets = context["case_targets"]
    target = ((targets[qi].argmax(1) == targets[ci].argmax(1)).astype(np.float32)[:, None]
        if context["task_type"] == "classification" else targets[qi] - targets[ci])
    return dict(a=part["query_z"][qi], b=context["case_z"][ci], target=target.astype(np.float32),
        weight=(weight / weight.mean()).astype(np.float32), query_index=qi, case_index=ci)


def context_descriptor(context, part):
    q, indices = part["query_z"], part["indices"]
    weights = np.asarray(part["weights"], dtype=np.float64)
    weights = weights / weights.sum(1, keepdims=True)
    distance = np.linalg.norm(np.asarray(q, dtype=np.float64)[:, None] - context["case_z"][indices], axis=-1)
    nearest = np.where(weights > 0, distance, np.inf).min(1)
    mean = (weights * distance).sum(1)
    concentration = np.square(weights).sum(1)
    if context["task_type"] == "classification":
        disagreement = np.maximum(1 - np.square(part["p0"]).sum(1), 0.)
    else:
        y = context["case_targets"][indices, 0]
        disagreement = np.sqrt((weights * np.square(y - part["p0"])).sum(1))
    return np.concatenate((q, part["p0"], np.stack((nearest, mean, concentration, disagreement), axis=1)), axis=1)


def make_error_memory(context):
    descriptor = context_descriptor(context, context["train"])
    mean, scale = descriptor.mean(0), descriptor.std(0)
    scale = np.where(scale < 1e-6, 1., scale)
    return dict(z=(descriptor - mean) / scale, mean=mean, scale=scale,
        residual=np.asarray(context["case_targets"], dtype=np.float64) - context["train"]["p0"])


def error_memory_predict(model, context, part, settings, *, query_ids=None):
    from sklearn.neighbors import NearestNeighbors
    z = (context_descriptor(context, part) - model["mean"]) / model["scale"]
    n = len(model["z"])
    k = min(settings["neighbors"], n - (query_ids is not None))
    finder = NearestNeighbors(n_neighbors=k + (query_ids is not None), algorithm="brute", n_jobs=1).fit(model["z"])
    distance, indices = finder.kneighbors(z)
    if query_ids is not None:
        keep = indices != np.asarray(query_ids)[:, None]
        distance = np.stack([row[mask][:k] for row, mask in zip(distance, keep)])
        indices = np.stack([row[mask][:k] for row, mask in zip(indices, keep)])
    exact = distance <= 1e-12
    weights = np.where(exact.any(1, keepdims=True), exact.astype(float), 1 / distance.clip(1e-12))
    weights /= weights.sum(1, keepdims=True)
    delta = (weights[..., None] * model["residual"][indices]).sum(1)
    final = part["p0"] + settings["multiplier"] * delta
    return normalize_probabilities(final) if context["task_type"] == "classification" else final


def transport_predict(model, context, part):
    indices, w = part["indices"], np.asarray(part["weights"], dtype=np.float64)
    folds = model["fold_of_case"][indices]
    delta = -(w[..., None] * model["anchor_predictions"][indices]).sum(1)
    for fold, forest in enumerate(model["models"]):
        mass = (w * (folds == fold)).sum(1, keepdims=True)
        delta += mass * point_predict(forest, part["query_z"], context["task_type"], context["case_targets"].shape[1])
    final = part["p0"] + delta
    return normalize_probabilities(final) if context["task_type"] == "classification" else final


class VoteAttention(nn.Module):
    """Shared case scorer; the final prediction stays in the neighbor target hull."""
    def __init__(self, feature_dim, output_dim, *, hidden=(64, 32), dropout=0., score_cap=4.):
        super().__init__()
        self.score_cap = score_cap
        self.net = nn.Sequential(nn.Linear(3 * feature_dim + output_dim, hidden[0]), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden[1], 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, q, cases, targets, weights, p0, *, return_weights=False):
        queries = q[:, None].expand_as(cases)
        features = torch.cat((queries, cases, queries - cases, targets - p0[:, None]), dim=-1)
        scores = self.score_cap * torch.tanh(self.net(features).squeeze(-1))
        logits = weights.clamp_min(1e-30).log() + scores
        logits = logits.masked_fill(weights <= 0, -torch.inf)
        adjusted = torch.softmax(logits, dim=1)
        prediction = (adjusted[..., None] * targets).sum(1)
        return (prediction, adjusted) if return_weights else prediction


def attention_tensors(context, part):
    i = part["indices"]
    return tuple(torch.as_tensor(v, dtype=torch.float32) for v in (
        part["query_z"], context["case_z"][i], context["case_targets"][i], part["weights"], part["p0"]))


def attention_predict(model, context, part):
    model.eval()
    tensors = attention_tensors(context, part)
    with torch.no_grad():
        result = torch.cat([model(*(t[start:start + 256] for t in tensors))
            for start in range(0, len(tensors[0]), 256)]).numpy().astype(np.float64)
    return normalize_probabilities(result) if context["task_type"] == "classification" else result


def predict_alternative(model, context, part, family, settings):
    if family == "direct_trees":
        result = point_predict(model, part["query_z"], context["task_type"], context["case_targets"].shape[1])
    elif family == "local_pair_trees":
        result = predict_pairs(model, context, part, family="pair_trees")
    elif family == "anchor_transport":
        result = transport_predict(model, context, part)
    elif family == "error_memory":
        result = error_memory_predict(model, context, part, settings)
    elif family == "vote_attention":
        result = attention_predict(model, context, part)
    else:
        raise ValueError(f"unknown alternative {family}")
    if not np.isfinite(result).all():
        raise ValueError("non-finite alternative prediction")
    return result
