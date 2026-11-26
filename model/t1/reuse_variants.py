"""Opt-in reuse variants beside the aggregate NN-CDH adapters of ``reuse.py`` (handoff 2026-11-26, pack p3).

Selected by ``ReuseConfig.output_mode``; the default modes never reach this module.

``per_case_2022`` -- the historical per-case NN-CDH adapter of the 2022 unified paper, as an ablation
(T1_IMPLEMENTATION_HANDOFF "Classification-adapter baselines and ablations", item 2; T1_FULL_CYCLE_NNKNN "The
maintained code implements aggregate and per-case adapter variants"). Every retrieved case ``i`` (normalized
activation ``a_qi > 0``) is adapted separately from its own problem context and the query-minus-case difference,
then the adapted case solutions are combined with the retrieval activations:

    classification  r_i = tanh(g([z_i, e(u_i), z_q - z_i, e(u_q) - e(u_i)]))   target e(y_q) - e(y_i)
                    s_q = sum_i a_qi (e(y_i) + r_i) = p0_q + sum_i a_qi r_i      y_hat = argmax s_q
    regression      dy_i = g([z_i, z_q - z_i])                                  target y_q - y_i
                    y_hat = sum_i a_qi (y_i + dy_i)                             (the maintained NNCDHAdapter pair net)

``e(u)`` are the explicit one-hot groups of nominal fields the representation does not cover (none otherwise).
The plus-sign convention (query minus case) of the current design is kept; the 2022 paper's retrieved-minus-query
sign only flips the learned map. ``loss: diff_only`` is the historical objective (mean squared pair difference,
uniform over retrieved pairs); ``combined`` adds the final task loss (cross-entropy of ``score_scale * s_q`` /
squared error of ``y_hat``) and ``cls_only`` uses the final task loss alone. Unlike the aggregate adapter, this
variant receives the retrieved problem context ``z_i``: it is the historical reference, not the v0 design.

``direct_head`` -- a direct neural classification head on the query representation ``z_q`` (plus the query's
uncovered nominal one-hot groups): a capacity reference, not an adaptation method, reported separately
(handoff: "A direct neural classification head may be included as a capacity reference ... reported separately").
It returns class logits as scores and trains with cross-entropy only (``loss: cls_only``).
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.nn_cdh import NNCDHAdapter

from .chunked import retrieve_chunked
from .nominal import NominalClassificationAdapter, fit_nominal_schema

PER_CASE = "per_case_2022"
DIRECT_HEAD = "direct_head"
VARIANT_MODES = frozenset({PER_CASE, DIRECT_HEAD})


def neighbor_slots(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Positions (into the retrieval's case columns) and activations of every case with ``a_qi > 0``.

    Rows are padded to the largest neighborhood of the batch with zero activations, which drop out of every
    weighted sum and every masked pair loss.
    """
    k = max(1, int((w > 0).sum(1).max()))
    wk, idx = torch.topk(w, k, dim=1)
    return idx, wk


def _mlp(in_dim: int, out_dim: int, hidden_dims, dropout: float) -> nn.Sequential:
    h1, h2 = hidden_dims
    layers: list[nn.Module] = [nn.Linear(in_dim, h1), nn.ReLU()]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers += [nn.Linear(h1, h2), nn.ReLU(), nn.Linear(h2, out_dim)]
    return nn.Sequential(*layers)


def _check_query_nominal(schema, query, rows: int, device) -> torch.Tensor:
    if query is None:
        raise ValueError("uncovered nominal fields require explicit encoded query values")
    q = torch.as_tensor(query, dtype=torch.float32, device=device).detach()
    if q.shape != (rows, schema.dimension) or not torch.isfinite(q).all():
        raise ValueError("nominal queries must align with this retrieval event")
    for group in schema.groups:
        g = q[:, group["offset"]:group["offset"] + group["width"]]
        if not ((g == 0) | (g == 1)).all() or not (g.sum(1) == 1).all():
            raise ValueError("each nominal field must retain its own one-hot group")
    return q


class PerCaseClassificationAdapter(nn.Module):
    """Historical per-case nominal-difference adapter (``per_case_2022``); see the module docstring."""

    output_mode = PER_CASE

    def __init__(self, feature_dim: int, num_classes: int, hidden_dims=(64, 32), dropout: float = 0.0,
                 schema=None, case_ids=None, case_values=None):
        super().__init__()
        self.feature_dim, self.num_classes = int(feature_dim), int(num_classes)
        self.nominal_dim = 0 if schema is None else int(schema.dimension)
        self.net = _mlp(2 * self.feature_dim + 2 * self.nominal_dim, self.num_classes, hidden_dims, dropout)
        if schema is not None:
            self.nominal_schema = schema
            ids = torch.as_tensor(case_ids, dtype=torch.long).view(-1)
            values = torch.as_tensor(case_values, dtype=torch.float32)
            if not len(ids) or (ids < 0).any() or ids.unique().numel() != len(ids) or \
                    values.shape != (len(ids), schema.dimension) or not torch.isfinite(values).all():
                raise ValueError("nominal case bindings require unique stable IDs and aligned groups")
            order = ids.argsort()
            self.register_buffer("nominal_case_ids", ids[order].clone())
            self.register_buffer("nominal_case_values", values[order].clone())

    values_for_ids = NominalClassificationAdapter.values_for_ids
    nominal_delta = NominalClassificationAdapter.nominal_delta  # aggregate Delta_u (diagnostic callers only)

    def pair_residuals(self, zq, zc, uq=None, uc=None) -> torch.Tensor:
        """``r_i`` [B, k, C] for queries ``zq`` [B, D] and their retrieved cases ``zc`` [B, k, D]."""
        zq_e = zq.unsqueeze(1).expand_as(zc)
        parts = [zc]
        if self.nominal_dim:
            parts.append(uc)
        parts.append(zq_e - zc)
        if self.nominal_dim:
            parts.append(uq.unsqueeze(1).expand_as(uc) - uc)
        return torch.tanh(self.net(torch.cat(parts, dim=-1)))

    def combine(self, r, wk, yc) -> tuple[torch.Tensor, torch.Tensor]:
        """``(sum_i a_qi r_i, s_q)`` with ``s_q = sum_i a_qi (e(y_i) + r_i)``."""
        residual = (wk.unsqueeze(-1) * r).sum(1)
        p0 = (wk.unsqueeze(-1) * yc).sum(1)
        p0 = p0 / p0.sum(1, keepdim=True).clamp_min(1e-12)
        return residual, p0 + residual

    def forward_retrieval(self, model, retrieval, query_nominal=None) -> tuple[torch.Tensor, torch.Tensor]:
        w = retrieval["weights"]
        idx, wk = neighbor_slots(w)
        zc = retrieval["case_features"][idx]
        yc = model.labels[retrieval["case_indices"]].float()[idx]
        uq = uc = None
        if self.nominal_dim:
            uq = _check_query_nominal(self.nominal_schema, query_nominal, w.shape[0], w.device)
            ids = model.active_case_ids()[retrieval["case_indices"]].to(w.device)
            uc = self.values_for_ids(ids[idx].reshape(-1)).to(w).view(*idx.shape, self.nominal_dim)
        return self.combine(self.pair_residuals(retrieval["query_features"], zc, uq, uc), wk, yc)


class DirectHeadAdapter(nn.Module):
    """Direct classification head on ``z_q`` (``direct_head``): a capacity reference, not adaptation."""

    output_mode = DIRECT_HEAD
    nominal_dim = 0  # the query's own nominal groups are an input here, never a neighborhood difference

    def __init__(self, feature_dim: int, num_classes: int, hidden_dims=(64, 32), dropout: float = 0.0, schema=None):
        super().__init__()
        self.feature_dim, self.num_classes = int(feature_dim), int(num_classes)
        self.query_nominal_dim = 0 if schema is None else int(schema.dimension)
        if schema is not None:
            self.nominal_schema = schema
        self.net = _mlp(self.feature_dim + self.query_nominal_dim, self.num_classes, hidden_dims, dropout)

    def logits(self, zq, uq=None) -> torch.Tensor:
        return self.net(zq if not self.query_nominal_dim else torch.cat([zq, uq], dim=-1))

    def forward_retrieval(self, model, retrieval, query_nominal=None) -> tuple[torch.Tensor, torch.Tensor]:
        w = retrieval["weights"]
        uq = None
        if self.query_nominal_dim:
            uq = _check_query_nominal(self.nominal_schema, query_nominal, w.shape[0], w.device)
        s = self.logits(retrieval["query_features"], uq)
        p0 = w @ model.labels[retrieval["case_indices"]].float()
        p0 = p0 / p0.sum(1, keepdim=True).clamp_min(1e-12)
        return s - p0, s


class PerCaseRegressionAdapter(NNCDHAdapter):
    """Per-case regression NN-CDH (``per_case_2022``): the maintained pair net, activation-weighted.

    Only the pair net trains; the inherited aggregate net stays frozen and unused.
    """

    output_mode = PER_CASE

    def __init__(self, feature_dim: int, hidden_dims=(64, 32)):
        super().__init__(feature_dim, 1, tuple(hidden_dims))
        self.adapt_net_final.requires_grad_(False)

    def pair_deltas(self, zq, zc) -> torch.Tensor:
        return self.adapt_net_pair(self.build_pair_features(zc, zq.unsqueeze(1).expand_as(zc)))

    def forward_aggregate(self, query_features, case_features, case_labels, case_weights):
        idx, wk = neighbor_slots(case_weights)
        yc = case_labels.view(case_features.shape[0], -1)[idx]
        dy = self.pair_deltas(query_features, case_features[idx])
        return (wk.unsqueeze(-1) * (yc + dy)).sum(1)


# ----------------------------------------------------------------------------- training


def _stream(model, X, ids, *, query_nominal=None):
    """Frozen-retrieval neighborhoods of one stream, kept compact (k retrieved columns per query)."""
    model.eval()
    device = model.cases.device
    with torch.no_grad():
        r = retrieve_chunked(model, X.to(device), exclude_identical=False,
                             query_case_ids=None if ids is None else ids.to(device))
        w = r["weights"]
        idx, wk = neighbor_slots(w)
        labels = model.labels[r["case_indices"]].float()
        p0 = w @ labels
        if model.task_type == "classification":
            p0 = p0 / p0.sum(1, keepdim=True).clamp_min(1e-12)
        case_ids = model.active_case_ids()[r["case_indices"]].to(device)
        return {"zq": r["query_features"].detach(), "idx": idx, "wk": wk.detach(), "p0": p0.detach(),
                "case_features": r["case_features"].detach(), "labels": labels, "case_ids": case_ids,
                "uq": None if query_nominal is None else query_nominal.to(device)}


def _fit(adapter, cfg, n_train: int, batch_loss, val_loss, device):
    """Shared early-stopping loop (validation loss, patience) with the selected Adam state, as reuse.py."""
    from .reuse import adapter_optimizer
    opt = adapter_optimizer([p for p in adapter.parameters() if p.requires_grad], cfg.lr, cfg.weight_decay, device)
    g = torch.Generator().manual_seed(cfg.seed)
    best, bad, hist = (float("inf"), None, -1, None), 0, []
    for ep in range(1, cfg.epochs + 1):
        adapter.train()
        perm = torch.randperm(n_train, generator=g)
        for s in range(0, n_train, cfg.batch_size):
            b = perm[s:s + cfg.batch_size].to(device)
            L = batch_loss(b)
            if not torch.isfinite(L["loss"]):
                raise ValueError("Nonfinite reuse-variant training loss")
            opt.zero_grad()
            L["loss"].backward()
            opt.step()
        adapter.eval()
        with torch.no_grad():
            Lv = val_loss()
        hist.append({"epoch": ep, "val_loss": float(Lv["loss"]), "val_l_diff": float(Lv["l_diff"]),
                     "val_l_cls": float(Lv["l_cls"])})
        if float(Lv["loss"]) < best[0] - 1e-9:
            best, bad = (float(Lv["loss"]), copy.deepcopy(adapter.state_dict()), ep, copy.deepcopy(opt.state_dict())), 0
        else:
            bad += 1
            if bad > cfg.patience:
                break
    adapter.load_state_dict(best[1])
    adapter.eval()
    return {"history": hist, "best_epoch": best[2], "epochs_run": len(hist), "optimizer_state": best[3],
            "optimizer_checkpoint_epoch": best[2], "n_train_examples": int(n_train)}


def _total(cfg, l_diff, l_cls):
    if cfg.loss == "diff_only":
        return l_diff
    if cfg.loss == "cls_only":
        return l_cls
    return cfg.lambda_diff * l_diff + cfg.lambda_cls * l_cls


def _masked_pair_mse(pred, target, wk):
    """Mean squared pair difference over retrieved pairs (``a_qi > 0``), uniform over pairs and output units."""
    mask = (wk > 0).unsqueeze(-1).to(pred)
    return ((pred - target) ** 2 * mask).sum() / (mask.sum() * pred.shape[-1]).clamp_min(1.0)


def train_classification_variant(model, data, cfg) -> tuple[nn.Module, dict[str, Any]]:
    """Train a ``per_case_2022`` or ``direct_head`` classification adapter on the frozen core (training LOO)."""
    cfg.validate()
    if cfg.output_mode not in VARIANT_MODES:
        raise ValueError(cfg.output_mode)
    torch.manual_seed(cfg.seed)
    C = int(data.num_classes)
    device = model.cases.device
    schema = train_nominal = val_nominal = None
    if cfg.nominal_fields:
        schema = fit_nominal_schema(data, cfg.nominal_fields)
        if schema.dimension:
            train_nominal = schema.encode(data.nominal_train, rows=len(data.y_train))
            val_nominal = schema.encode(data.nominal_val, rows=len(data.y_val))
        else:
            schema = None  # every field covered: no explicit channel
    elif getattr(data, "nominal_train", None):
        raise ValueError("nominal data requires explicit field-by-field representation coverage")
    n = len(data.y_train)
    tr = _stream(model, data.X_train, torch.arange(n), query_nominal=train_nominal)
    va = _stream(model, data.X_val, None, query_nominal=val_nominal)
    ytr, yva = data.y_train.to(device).long(), data.y_val.to(device).long()
    D = tr["zq"].shape[1]
    if cfg.output_mode == DIRECT_HEAD:
        adapter = DirectHeadAdapter(D, C, cfg.hidden_dims, cfg.dropout, schema=schema).to(device)

        def losses(st, b, y):
            s = adapter.logits(st["zq"][b], None if st["uq"] is None else st["uq"][b])
            l_cls = F.cross_entropy(cfg.score_scale * s, y)
            return {"loss": l_cls, "l_diff": torch.zeros((), device=device), "l_cls": l_cls}
    else:
        adapter = PerCaseClassificationAdapter(D, C, cfg.hidden_dims, cfg.dropout, schema=schema,
                                               case_ids=None if schema is None else torch.arange(n),
                                               case_values=train_nominal).to(device)

        def losses(st, b, y):
            idx, wk = st["idx"][b], st["wk"][b]
            zc = st["case_features"][idx]
            yc = st["labels"][idx]
            uq = uc = None
            if schema is not None:
                uq = st["uq"][b]
                uc = adapter.values_for_ids(st["case_ids"][idx].reshape(-1)).to(zc).view(*idx.shape, -1)
            r = adapter.pair_residuals(st["zq"][b], zc, uq, uc)
            target = F.one_hot(y, C).float().unsqueeze(1) - yc
            l_diff = _masked_pair_mse(r, target, wk)
            _, s = adapter.combine(r, wk, yc)
            l_cls = F.cross_entropy(cfg.score_scale * s, y)
            return {"loss": _total(cfg, l_diff, l_cls), "l_diff": l_diff, "l_cls": l_cls}
    all_val = torch.arange(len(yva), device=device)
    info = _fit(adapter, cfg, n, lambda b: losses(tr, b, ytr[b]), lambda: losses(va, all_val, yva), device)
    info.update(neighborhoods="leave_one_out", actual_output_mode=cfg.output_mode,
                pair_weighting="uniform_over_retrieved_pairs" if cfg.output_mode == PER_CASE else None)
    if schema is not None:
        info["nominal_manifest"] = schema.state_dict()
    return adapter, info


def train_per_case_regression_adapter(model, data, cfg) -> tuple[nn.Module, dict[str, Any]]:
    """Per-case regression NN-CDH on the frozen core (training LOO), ``loss`` as in the module docstring."""
    cfg.validate()
    if cfg.output_mode != PER_CASE or model.task_type != "regression":
        raise ValueError("per-case regression reuse needs output_mode per_case_2022 on a regression core")
    torch.manual_seed(cfg.seed)
    device = model.cases.device
    n = len(data.y_train)
    tr = _stream(model, data.X_train, torch.arange(n))
    va = _stream(model, data.X_val, None)
    ytr, yva = data.y_train.to(device).float().view(-1, 1), data.y_val.to(device).float().view(-1, 1)
    adapter = PerCaseRegressionAdapter(tr["zq"].shape[1], cfg.hidden_dims).to(device)

    def losses(st, b, y):
        idx, wk = st["idx"][b], st["wk"][b]
        yc = st["labels"].view(st["labels"].shape[0], -1)[idx]
        dy = adapter.pair_deltas(st["zq"][b], st["case_features"][idx])
        l_diff = _masked_pair_mse(dy, y.unsqueeze(1) - yc, wk)
        final = (wk.unsqueeze(-1) * (yc + dy)).sum(1)
        l_cls = F.mse_loss(final, y)
        return {"loss": _total(cfg, l_diff, l_cls), "l_diff": l_diff, "l_cls": l_cls}
    all_val = torch.arange(len(yva), device=device)
    info = _fit(adapter, cfg, n, lambda b: losses(tr, b, ytr[b]), lambda: losses(va, all_val, yva), device)
    info.update(neighborhoods="stable_ID_leave_one_out", actual_output_mode="per_case_regression_2022",
                pair_weighting="uniform_over_retrieved_pairs",
                unused_classification_options=["probability_mode"])
    return adapter, info
