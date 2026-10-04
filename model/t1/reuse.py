"""Classification reuse: aggregate, retrieved-label-conditioned NN-CDH (plan section 5).

Training protocol (section 5.5):
  1. retrieval core and case memory are trained and frozen;
  2. adapter examples come from leave-one-out neighborhoods of training cases;
  3. the adapter never alters retrieval;
  4. retrieval-only vs adapted decisions are compared on identical case sets.

Adapter input follows the information-flow rule: ``[Delta_z_q, (Delta_u_q), p0_q]``.
Raw z_q, individual z_i, z_bar_q and raw cases are never passed.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from model.nn_cdh import ClassificationNNCDHAdapter

from .core import expected_calibration_error
from .nominal import NominalSchema, NominalClassificationAdapter, nominal_difference, fit_nominal_schema


@dataclass
class ReuseConfig:
    output_mode: str  # nominal_residual_scores | logit_residual
    loss: str  # combined | diff_only | cls_only
    lambda_diff: float
    lambda_cls: float
    probability_mode: str  # softmax | clip_normalize : how s_q is turned into probabilities for ECE
    hidden_dims: tuple[int, int] = (64, 32)
    lr: float = 1e-3
    epochs: int = 200
    patience: int = 20
    batch_size: int = 64
    seed: int = 0
    nominal_fields_covered_by_representation: bool = True  # declared, not inferred (section 5.2)
    nominal_fields: tuple[dict[str, Any], ...] = ()  # explicit per-field coverage

    def validate(self) -> None:
        if self.output_mode not in {"nominal_residual_scores", "logit_residual"}:
            raise ValueError(self.output_mode)
        if self.loss not in {"combined", "diff_only", "cls_only"}:
            raise ValueError(self.loss)
        if self.loss == "combined" and not (self.lambda_diff > 0 and self.lambda_cls > 0):
            raise ValueError("combined loss requires lambda_diff > 0 and lambda_cls > 0 (plan 5.4)")
        if self.epochs < 1 or self.patience < 0 or self.batch_size < 1:
            raise ValueError("adapter training needs positive epochs/batch size and nonnegative patience")
        if not self.nominal_fields and not self.nominal_fields_covered_by_representation:
            raise ValueError("uncovered nominal inputs require explicit field-by-field declarations")
        if self.nominal_fields:
            NominalSchema(self.nominal_fields)
            if self.nominal_fields_covered_by_representation != all(f["covered_by_representation"] for f in self.nominal_fields):
                raise ValueError("aggregate coverage flag must agree with the explicit per-field declarations")


@torch.no_grad()
def neighborhood_inputs(model, X: torch.Tensor, *, exclude_identical: bool, X_nominal: torch.Tensor | None = None, nominal_case: torch.Tensor | None = None, query_case_ids=None, nominal_adapter=None) -> dict[str, torch.Tensor]:
    """Frozen-retrieval neighborhood quantities for queries X."""
    model.eval()
    r = model.retrieve(X.to(model.cases.device), exclude_identical=exclude_identical if query_case_ids is None else False, query_case_ids=query_case_ids)
    w = r["weights"]
    labels = model.labels[r["case_indices"]].float()
    p0 = w @ labels
    p0 = p0 / p0.sum(1, keepdim=True).clamp_min(1e-12)
    z_bar = w @ r["case_features"]
    dz = r["query_features"] - z_bar
    du = None
    if nominal_adapter is not None:
        du = nominal_difference(nominal_adapter, model, r, X_nominal)
    elif X_nominal is not None or nominal_case is not None:
        raise ValueError("nominal inputs require a fitted schema and stable case-ID binding via nominal_adapter")
    return {"dz": dz.detach(), "p0": p0.detach(), "du": du, "weights": w.detach()}


def _probs_from_scores(s: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "softmax":
        return F.softmax(s, dim=1)
    if mode == "clip_normalize":
        c = s.clamp_min(0)
        return c / c.sum(1, keepdim=True).clamp_min(1e-12)
    raise ValueError(mode)


def _adapter_losses(adapter, out_r, out_s, p0, y, cfg: ReuseConfig) -> dict[str, torch.Tensor]:
    C = p0.size(1)
    y1 = F.one_hot(y.long(), C).float()
    r_star = y1 - p0
    if cfg.output_mode == "nominal_residual_scores":
        l_diff = F.mse_loss(out_r, r_star)
        l_cls = F.cross_entropy(out_s, y.long())  # plan 5.4: cross_entropy(s_q, y_q)
    else:
        # logit mode: residual target is not a literal nominal difference; measure the
        # implied probability change against r* for the diff term.
        l_diff = F.mse_loss(out_s - p0, r_star)
        l_cls = F.nll_loss(torch.log(out_s.clamp_min(1e-8)), y.long())
    if cfg.loss == "diff_only":
        total = l_diff
    elif cfg.loss == "cls_only":
        total = l_cls
    else:
        total = cfg.lambda_diff * l_diff + cfg.lambda_cls * l_cls
    return {"loss": total, "l_diff": l_diff, "l_cls": l_cls}


def train_classification_adapter(model, data, cfg: ReuseConfig) -> tuple[ClassificationNNCDHAdapter, dict[str, Any]]:
    cfg.validate()
    torch.manual_seed(cfg.seed)
    C = int(data.num_classes)
    schema, train_nominal, val_nominal = None, None, None
    if cfg.nominal_fields:
        schema = fit_nominal_schema(data, cfg.nominal_fields)
        train_nominal = schema.encode(data.nominal_train, rows=len(data.y_train))
        val_nominal = schema.encode(data.nominal_val, rows=len(data.y_val))
    elif getattr(data, "nominal_train", None):
        raise ValueError("nominal data requires explicit field-by-field representation coverage")
    tr = neighborhood_inputs(model, data.X_train, exclude_identical=True, query_case_ids=torch.arange(len(data.y_train)))
    kwargs = dict(feature_dim=tr["dz"].shape[1], num_classes=C, hidden_dims=cfg.hidden_dims, output_mode=cfg.output_mode)
    if schema is not None:
        adapter = NominalClassificationAdapter(schema=schema, case_ids=torch.arange(len(data.y_train)),
                                              case_values=train_nominal, **kwargs)
    else:
        adapter = ClassificationNNCDHAdapter(nominal_dim=0, **kwargs)
    adapter = adapter.to(tr["dz"].device)
    if schema is not None:
        tr = neighborhood_inputs(model, data.X_train, exclude_identical=True, query_case_ids=torch.arange(len(data.y_train)),
                                 X_nominal=train_nominal, nominal_adapter=adapter)
    va = neighborhood_inputs(model, data.X_val, exclude_identical=False, X_nominal=val_nominal, nominal_adapter=adapter)
    opt = torch.optim.Adam(adapter.parameters(), lr=cfg.lr)
    ytr = data.y_train.to(tr["dz"].device)
    yva = data.y_val.to(tr["dz"].device)
    g = torch.Generator().manual_seed(cfg.seed)
    best = (float("inf"), None, -1)
    bad = 0
    hist = []
    for ep in range(1, cfg.epochs + 1):
        adapter.train()
        perm = torch.randperm(ytr.numel(), generator=g)
        for s in range(0, ytr.numel(), cfg.batch_size):
            b = perm[s : s + cfg.batch_size]
            r, sc = adapter(tr["dz"][b], tr["p0"][b], None if tr["du"] is None else tr["du"][b])
            L = _adapter_losses(adapter, r, sc, tr["p0"][b], ytr[b], cfg)
            opt.zero_grad()
            L["loss"].backward()
            opt.step()
        adapter.eval()
        with torch.no_grad():
            r, sc = adapter(va["dz"], va["p0"], va["du"])
            Lv = _adapter_losses(adapter, r, sc, va["p0"], yva, cfg)
        hist.append({"epoch": ep, "val_loss": float(Lv["loss"]), "val_l_diff": float(Lv["l_diff"]), "val_l_cls": float(Lv["l_cls"])})
        if float(Lv["loss"]) < best[0] - 1e-9:
            best, bad = (float(Lv["loss"]), copy.deepcopy(adapter.state_dict()), ep), 0
        else:
            bad += 1
            if bad > cfg.patience:
                break
    adapter.load_state_dict(best[1])
    adapter.eval()
    info = {"history": hist, "best_epoch": best[2], "n_train_examples": int(ytr.numel()), "neighborhoods": "leave_one_out"}
    if schema is not None:
        info["nominal_manifest"] = schema.state_dict()
    return adapter, info


@torch.no_grad()
def evaluate_reuse(model, adapter, X: torch.Tensor, y: torch.Tensor, cfg: ReuseConfig, *, query_nominal=None, test_groups=None) -> dict[str, Any]:
    """Pre (retrieval-only) vs post (adapted) on the same case set; flip accounting."""
    nb = neighborhood_inputs(model, X, exclude_identical=False, X_nominal=query_nominal, nominal_adapter=adapter)
    p0 = nb["p0"]
    y = y.to(p0.device).long()
    r, s = adapter(nb["dz"], p0, nb["du"])
    pre = p0.argmax(1)
    post = s.argmax(1)
    probs_post = s if cfg.output_mode == "logit_residual" else _probs_from_scores(s, cfg.probability_mode)
    ok_pre, ok_post = pre == y, post == y
    flips = {
        "wrong_to_correct": int((~ok_pre & ok_post).sum()),
        "correct_to_wrong": int((ok_pre & ~ok_post).sum()),
        "wrong_to_other_wrong": int((~ok_pre & ~ok_post & (pre != post)).sum()),
        "unchanged": int((pre == post).sum()),
    }
    mag = (s - p0).abs().sum(1) if cfg.output_mode == "nominal_residual_scores" else (probs_post - p0).abs().sum(1)
    metrics = {
        "accuracy_pre": float(ok_pre.float().mean()),
        "accuracy_post": float(ok_post.float().mean()),
        "ece_pre": expected_calibration_error(p0.cpu(), y.cpu()),
        "ece_post": expected_calibration_error(probs_post.cpu(), y.cpu()),
        "probability_mode": cfg.probability_mode if cfg.output_mode == "nominal_residual_scores" else "softmax(log p0 + delta)",
        "flips": flips,
        "net_beneficial_flips": flips["wrong_to_correct"] - flips["correct_to_wrong"],
        "correction_l1_mean": float(mag.mean()),
        "n": int(y.numel()),
    }
    for group, mask in (test_groups or {}).items():
        mask = torch.as_tensor(mask, dtype=torch.bool, device=y.device)
        if mask.shape != y.shape:
            raise ValueError("test group masks must align with targets")
        metrics[f"n_{group}"] = int(mask.sum())
        if mask.any():
            metrics[f"accuracy_pre_{group}"] = float(ok_pre[mask].float().mean())
            metrics[f"accuracy_post_{group}"] = float(ok_post[mask].float().mean())
    return metrics


@torch.no_grad()
def evaluate_regression_reuse(model, adapter, X, y, *, success_tolerance):
    """Actual aggregate NN-CDH pre/post outputs, in training-standardized units."""
    from .outcomes import final_prediction, frozen_evaluation, prediction_loss, query_success

    device = model.cases.device
    pre_rows, post_rows = [], []
    with frozen_evaluation(model, adapter):
        for start in range(0, len(y), 256):
            xb = X[start:start + 256].to(device)
            r = model.retrieve(xb, exclude_identical=False)
            pre, post, _ = final_prediction(model, xb, r, adapter=adapter)
            pre_rows.append(pre.view(-1))
            post_rows.append(post.view(-1))
    pre, post = torch.cat(pre_rows), torch.cat(post_rows)
    target = y.to(device).view(-1)
    errors_pre, errors_post = (pre - target).abs(), (post - target).abs()
    metrics = {"n": len(y), "correction_l1_mean": float((post - pre).abs().mean()),
               "improved_queries": int((errors_post < errors_pre).sum()),
               "worsened_queries": int((errors_post > errors_pre).sum()),
               "success_tolerance": success_tolerance}
    for suffix, pred in (("pre", pre), ("post", post)):
        mse = float(prediction_loss(pred, target, "squared_error").mean())
        metrics.update({f"loss_{suffix}": mse, f"rmse_{suffix}": mse ** .5,
                        f"mae_{suffix}": float((pred - target).abs().mean()),
                        f"success_rate_{suffix}": float(query_success(pred, target, "regression", success_tolerance).float().mean())})
    return metrics
