"""Individual-case alternatives to the aggregate-input NN-CDH adapter.

Prediction routines only consume query features, training cases/labels and
retrieval weights. Query labels are used solely by the training helpers.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


def pair_features(a, b):
    return np.concatenate((a, b, a-b), axis=-1)


def pair_training_data(context, *, seed, random_anchors=8):
    """Equal query mass; half retrieved neighbors, half uniform non-self anchors."""
    z = np.asarray(context["case_z"], dtype=np.float32)
    t = np.asarray(context["case_targets"], dtype=np.float32)
    part = context["train"]
    n = len(z)
    if n < 2 or random_anchors < 1:
        raise ValueError("need at least two cases and one random anchor")
    query_z = np.asarray(part["query_z"], dtype=np.float32)
    if not np.allclose(query_z, z, atol=1e-5, rtol=1e-5):
        raise ValueError("training queries must align exactly with the case representation")
    qi = np.repeat(np.arange(n), part["indices"].shape[1])
    ci = np.asarray(part["indices"]).reshape(-1)
    weight = .5*np.asarray(part["weights"], dtype=np.float64).reshape(-1)
    keep = weight > 0
    qi, ci, weight = qi[keep], ci[keep], weight[keep]
    if np.any(qi == ci):
        raise ValueError("retrieved training pairs contain a positive-weight self pair")
    rng = np.random.default_rng(seed)
    random_i = rng.integers(0, n-1, size=(n, random_anchors))
    random_i += random_i >= np.arange(n)[:, None]
    qi = np.concatenate((qi, np.repeat(np.arange(n), random_anchors)))
    ci = np.concatenate((ci, random_i.reshape(-1)))
    weight = np.concatenate((weight, np.full(n*random_anchors, .5/random_anchors)))
    if not np.allclose(np.bincount(qi, weights=weight, minlength=n), 1., atol=1e-6):
        raise ValueError("pair weights must give every query equal mass")
    a, b = query_z[qi], z[ci]
    if context["task_type"] == "classification":
        target = (t[qi].argmax(1) == t[ci].argmax(1)).astype(np.float32)[:, None]
    else:
        target = t[qi]-t[ci]
    # Mean weight one makes minibatch mean loss an unbiased weighted objective.
    weight = (weight/weight.mean()).astype(np.float32)
    return dict(a=a, b=b, target=target, weight=weight, query_index=qi, case_index=ci)


def classification_from_affinity(affinity, anchor_targets, weights, prior):
    """PDL per-anchor distributions, combined using fixed NN-kNN activations."""
    s = np.asarray(affinity, dtype=np.float64)
    target = np.asarray(anchor_targets, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    prior = np.asarray(prior, dtype=np.float64)
    if len(prior) < 2 or np.any(prior <= 0) or np.any(prior >= 1):
        raise ValueError("all classification priors must be strictly between zero and one")
    if np.any((s < 0) | (s > 1)) or s.shape != w.shape:
        raise ValueError("same-class probabilities and weights must align")
    anchor_prior = (target*prior).sum(-1)
    per_case = (1-s)[..., None] * prior / (1-anchor_prior)[..., None]
    per_case = per_case*(1-target) + s[..., None]*target
    final = (w[..., None]*per_case).sum(1)
    return final/final.sum(1, keepdims=True).clip(1e-12)


def local_ridge_predict(context, part, ridge, *, batch_size=256):
    """Weighted affine ridge using a K x K dual solve; intercept is unpenalized."""
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be positive")
    z = np.asarray(context["case_z"], dtype=np.float64)
    targets = np.asarray(context["case_targets"], dtype=np.float64)
    result = []
    for start in range(0, len(part["query_z"]), batch_size):
        sl = slice(start, start+batch_size)
        q = np.asarray(part["query_z"][sl], dtype=np.float64)
        i, w = part["indices"][sl], np.asarray(part["weights"][sl], dtype=np.float64)
        x, y = z[i], targets[i]
        xmean, ymean = (w[..., None]*x).sum(1), (w[..., None]*y).sum(1)
        rootw = np.sqrt(w)[..., None]
        u = rootw*(x-xmean[:, None, :])
        v = rootw*(y-ymean[:, None, :])
        gram = u@np.swapaxes(u, 1, 2)
        gram += ridge*np.eye(gram.shape[-1])[None]
        coefficients = np.linalg.solve(gram, v)
        delta = ((q-xmean)[:, None, :]@np.swapaxes(u, 1, 2)@coefficients)[:, 0]
        final = np.asarray(part["p0"][sl], dtype=np.float64)+delta
        if context["task_type"] == "classification":
            final = np.maximum(final, 0.)
            final /= final.sum(1, keepdims=True).clip(1e-12)
        result.append(final)
    return np.concatenate(result)


class PairNetwork(nn.Module):
    """Symmetric relation or antisymmetric target-difference network."""
    def __init__(self, feature_dim, task_type, *, dropout=0., hidden=(64, 32)):
        super().__init__()
        if task_type not in {"classification", "regression"} or not 0 <= dropout < 1:
            raise ValueError("unsupported task or dropout")
        self.task_type = task_type
        self.net = nn.Sequential(nn.Linear(3*feature_dim, hidden[0]), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden[1], 1))

    def forward(self, a, b):
        ab = self.net(torch.cat((a, b, a-b), -1))
        ba = self.net(torch.cat((b, a, b-a), -1))
        if self.task_type == "classification":
            return .5*(torch.sigmoid(ab)+torch.sigmoid(ba))
        difference = .5*(ab-ba)
        return torch.where((a == b).all(-1, keepdim=True), torch.zeros_like(difference), difference)


def forest_pair_predict(model, a, b, task_type):
    if task_type == "classification":
        classes = np.asarray(model.classes_)
        def positive(x):
            if 1 not in classes:
                return np.zeros(len(x))
            return model.predict_proba(x)[:, int(np.flatnonzero(classes == 1)[0])]
        return .5*(positive(pair_features(a, b))+positive(pair_features(b, a)))
    return .5*(model.predict(pair_features(a, b))-model.predict(pair_features(b, a)))


def predict_pairs(model, context, part, *, family, batch_size=256):
    """Transform each individual case, then aggregate with unchanged weights."""
    z = np.asarray(context["case_z"], dtype=np.float32)
    targets = np.asarray(context["case_targets"], dtype=np.float64)
    kind = context["task_type"]
    if family == "pair_neural":
        model.eval()
    results = []
    for start in range(0, len(part["query_z"]), batch_size):
        sl = slice(start, start+batch_size)
        i, w = part["indices"][sl], np.asarray(part["weights"][sl], dtype=np.float64)
        q = np.asarray(part["query_z"][sl], dtype=np.float32)
        a = np.broadcast_to(q[:, None, :], (*i.shape, q.shape[-1])).reshape(-1, q.shape[-1]).copy()
        b = z[i].reshape(a.shape)
        if family == "pair_neural":
            with torch.no_grad():
                values = model(torch.from_numpy(a), torch.from_numpy(b)).cpu().numpy().reshape(i.shape)
        elif family == "pair_trees":
            values = forest_pair_predict(model, a, b, kind).reshape(i.shape)
        else:
            raise ValueError("unknown pair family")
        if kind == "classification":
            final = classification_from_affinity(values, targets[i], w, context["prior"])
        else:
            final = np.asarray(part["p0"][sl], dtype=np.float64)+(w*values).sum(1, keepdims=True)
        if not np.isfinite(final).all():
            raise ValueError("non-finite pair prediction")
        results.append(final)
    return np.concatenate(results)


def metrics(prediction, target, task_type, p0=None):
    prediction, target = np.asarray(prediction), np.asarray(target)
    if task_type == "classification":
        wrong = prediction.argmax(1) != target.astype(int)
        error = float(wrong.mean())
        loss = float(-np.log(np.clip(prediction[np.arange(len(target)), target.astype(int)], 1e-12, 1.)).mean())
        before = np.asarray(p0).argmax(1) != target if p0 is not None else wrong
        helped, harmed = before & ~wrong, ~before & wrong
    else:
        target = target.reshape(prediction.shape)
        squared = np.square(prediction-target).mean(1)
        loss, error = float(squared.mean()), float(np.sqrt(squared.mean()))
        before = np.square(np.asarray(p0)-target).mean(1) if p0 is not None else squared
        helped, harmed = squared < before-1e-8, squared > before+1e-8
    return dict(error=error, task_loss=loss, n=len(target), helped=int(helped.sum()), harmed=int(harmed.sum()))
