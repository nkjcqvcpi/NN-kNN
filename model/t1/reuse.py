"""Supervised aggregate reuse, including classification NN-CDH (plan section 5).

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
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from model.device_utils import adam_kwargs_for_device
from model.nn_cdh import ClassificationNNCDHAdapter, NNCDHAdapter

from .core import expected_calibration_error
from .chunked import retrieve_chunked
from .nominal import NominalSchema, NominalClassificationAdapter, nominal_difference, fit_nominal_schema


def adapter_optimizer(params, lr: float, weight_decay: float, device) -> torch.optim.Optimizer:
    """Adam without weight decay (unchanged default); decoupled AdamW otherwise."""
    if weight_decay == 0:
        return torch.optim.Adam(params, lr=lr, **adam_kwargs_for_device(device))
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, **adam_kwargs_for_device(device))


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
    score_scale: float = 1.0  # kappa: l_cls = cross_entropy(kappa * s, y) (nominal_residual_scores only)
    weight_decay: float = 0.0  # > 0 switches the adapter optimizer from Adam to AdamW
    dropout: float = 0.0  # classification adapters only

    def validate(self) -> None:
        if self.output_mode not in {"nominal_residual_scores", "logit_residual"}:
            raise ValueError(self.output_mode)
        if self.loss not in {"combined", "diff_only", "cls_only"}:
            raise ValueError(self.loss)
        if self.loss == "combined" and not (self.lambda_diff > 0 and self.lambda_cls > 0):
            raise ValueError("combined loss requires lambda_diff > 0 and lambda_cls > 0 (plan 5.4)")
        if self.epochs < 1 or self.patience < 0 or self.batch_size < 1:
            raise ValueError("adapter training needs positive epochs/batch size and nonnegative patience")
        if not (math.isfinite(self.score_scale) and self.score_scale > 0):
            raise ValueError("score_scale must be finite and positive")
        if self.score_scale != 1.0 and self.output_mode != "nominal_residual_scores":
            raise ValueError("score_scale applies to nominal_residual_scores only")
        if not self.weight_decay >= 0 or not 0 <= self.dropout < 1:
            raise ValueError("adapter weight decay must be nonnegative and dropout in [0, 1)")
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
    r = retrieve_chunked(model, X.to(model.cases.device), exclude_identical=exclude_identical if query_case_ids is None else False, query_case_ids=query_case_ids)
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


def calibration_logits(s: torch.Tensor, cfg: ReuseConfig) -> torch.Tensor:
    """Logits of the adapted output: kappa * s for scores, log p for ``logit_residual`` probabilities."""
    if cfg.output_mode == "logit_residual":
        return torch.log(s.clamp_min(1e-8))
    return cfg.score_scale * s


def separated(logits: torch.Tensor, y: torch.Tensor) -> bool:
    """True when no row's true-class logit is exceeded by another class's logit.

    Then every row's NLL term log(1 + sum_j exp(-(z_y - z_j) / T)) is nonincreasing
    as T falls, so the NLL of softmax(logits / T) has no minimizer at a positive T.
    """
    z = logits.detach().cpu()
    t = y.detach().cpu().long().view(-1, 1)
    return bool((z <= z.gather(1, t)).all())


def calibrate_temperature(logits: torch.Tensor, y: torch.Tensor, *, bounds: tuple[float, float] = (1e-2, 1e2),
                          max_iter: int = 100) -> tuple[float, bool]:
    """``(T, degenerate)``: one temperature T > 0 minimizing the NLL of softmax(logits / T) on the given stream.

    LBFGS on log T in float64 on the CPU, started at T = 1 and kept within
    ``bounds``. The fit is degenerate, and T = 1 (no rescaling), when the NLL has
    no minimizer inside the bounds: the stream is ``separated`` (it would drive T
    toward 0, so any fitted value is wherever the optimizer stops), the fit ends
    on a bound, or it is not finite. A fit that does not lower the NLL also gives
    T = 1. Dividing by T > 0 never changes an argmax.
    """
    z = logits.detach().cpu().double()
    t = y.detach().cpu().long().view(-1)
    if separated(z, t):
        return 1.0, True
    lo, hi = math.log(bounds[0]), math.log(bounds[1])

    def nll(log_t):
        return F.cross_entropy(z / log_t.clamp(lo, hi).exp(), t)

    with torch.enable_grad():
        log_t = torch.zeros((), dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([log_t], lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = nll(log_t)
            loss.backward()
            return loss

        opt.step(closure)
    with torch.no_grad():
        fitted = log_t.detach().clamp(lo, hi)
        if not torch.isfinite(fitted) or not lo < float(fitted) < hi:
            return 1.0, True
        if not float(nll(fitted)) < float(nll(torch.zeros((), dtype=torch.float64))):
            return 1.0, False
    return math.exp(float(fitted)), False


def fit_temperature(logits: torch.Tensor, y: torch.Tensor, **kwargs: Any) -> float:
    """The temperature of :func:`calibrate_temperature` (1 when the fit is degenerate)."""
    return calibrate_temperature(logits, y, **kwargs)[0]


def _adapter_losses(adapter, out_r, out_s, p0, y, cfg: ReuseConfig) -> dict[str, torch.Tensor]:
    C = p0.size(1)
    y1 = F.one_hot(y.long(), C).float()
    r_star = y1 - p0
    if cfg.output_mode == "nominal_residual_scores":
        l_diff = F.mse_loss(out_r, r_star)
        l_cls = F.cross_entropy(cfg.score_scale * out_s, y.long())  # plan 5.4: cross_entropy(s_q, y_q), kappa = 1
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
    kwargs = dict(feature_dim=tr["dz"].shape[1], num_classes=C, hidden_dims=cfg.hidden_dims, output_mode=cfg.output_mode,
                  dropout=cfg.dropout)
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
    opt = adapter_optimizer(adapter.parameters(), cfg.lr, cfg.weight_decay, tr["dz"].device)
    ytr = data.y_train.to(tr["dz"].device)
    yva = data.y_val.to(tr["dz"].device)
    g = torch.Generator().manual_seed(cfg.seed)
    best = (float("inf"), None, -1, None)
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
            best, bad = (float(Lv["loss"]), copy.deepcopy(adapter.state_dict()), ep,
                         copy.deepcopy(opt.state_dict())), 0
        else:
            bad += 1
            if bad > cfg.patience:
                break
    adapter.load_state_dict(best[1])
    adapter.eval()
    info = {"history": hist, "best_epoch": best[2], "epochs_run": len(hist),
            "optimizer_state": best[3], "optimizer_checkpoint_epoch": best[2],
            "n_train_examples": int(ytr.numel()), "neighborhoods": "leave_one_out"}
    if schema is not None:
        info["nominal_manifest"] = schema.state_dict()
    return adapter, info


def regression_adapter_losses(final, p0, y, cfg):
    """Additive residual and final-output MSE coincide; keep weighting explicit."""
    target = y.float().view_as(final)
    l_cls = F.mse_loss(final, target)
    l_diff = F.mse_loss(final-p0, target-p0)
    total = l_diff if cfg.loss == "diff_only" else l_cls if cfg.loss == "cls_only" else (
        cfg.lambda_diff*l_diff+cfg.lambda_cls*l_cls)
    return {"loss": total, "l_diff": l_diff, "l_cls": l_cls}


def train_regression_adapter(model, data, cfg: ReuseConfig):
    """Frozen-core, training-LOO aggregate residual fit with selected Adam state.

    Numeric ReuseConfig budgets/weights are shared. Classification output and
    probability modes are unused; the actual regression interface is recorded.
    """
    cfg.validate()
    if model.task_type != "regression" or data.task_type != "regression" or cfg.nominal_fields:
        raise ValueError("Regression reuse needs an explicit regression task and continuous inputs")
    if cfg.dropout > 0 or cfg.score_scale != 1.0:
        raise ValueError("adapter dropout and score_scale apply to classification adapters only")
    device = model.cases.device
    torch.manual_seed(cfg.seed)
    def inputs(X, ids=None):
        with torch.no_grad():
            model.eval()
            r = retrieve_chunked(model, X.to(device), exclude_identical=False, query_case_ids=ids)
            p0 = r["weights"] @ model.labels[r["case_indices"]].float()
            dz = r["query_features"]-r["weights"] @ r["case_features"]
        return dz.detach(), p0.detach()
    tr = inputs(data.X_train, torch.arange(len(data.y_train), device=device))
    va = inputs(data.X_val)
    adapter = NNCDHAdapter(tr[0].shape[1], 1, cfg.hidden_dims).to(device)
    adapter.adapt_net_pair.requires_grad_(False)
    opt = adapter_optimizer([p for p in adapter.parameters() if p.requires_grad], cfg.lr, cfg.weight_decay, device)
    ytr, yva = data.y_train.to(device), data.y_val.to(device)
    generator = torch.Generator().manual_seed(cfg.seed)
    best, bad, history = (float("inf"), None, None, -1), 0, []
    def prediction(values):
        dz, p0 = values
        return p0+adapter.adapt_net_final(torch.cat((dz, p0), dim=1))
    for epoch in range(1, cfg.epochs+1):
        adapter.train()
        perm = torch.randperm(len(ytr), generator=generator)
        for start in range(0, len(ytr), cfg.batch_size):
            b = perm[start:start+cfg.batch_size].to(device)
            final = prediction((tr[0][b], tr[1][b]))
            loss = regression_adapter_losses(final, tr[1][b], ytr[b], cfg)["loss"]
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite regression adapter training loss")
            opt.zero_grad()
            loss.backward()
            opt.step()
        adapter.eval()
        with torch.no_grad():
            losses = regression_adapter_losses(prediction(va), va[1], yva, cfg)
        value = float(losses["loss"])
        if not np.isfinite(value):
            raise ValueError("Nonfinite regression validation loss")
        history.append(dict(epoch=epoch, val_loss=value, val_l_diff=float(losses["l_diff"]), val_l_cls=float(losses["l_cls"])))
        if value < best[0]-1e-9:
            best, bad = (value, copy.deepcopy(adapter.state_dict()), copy.deepcopy(opt.state_dict()), epoch), 0
        else:
            bad += 1
            if bad > cfg.patience:
                break
    adapter.load_state_dict(best[1])
    return adapter.eval(), dict(history=history, best_epoch=best[3], epochs_run=len(history),
        optimizer_state=best[2], optimizer_checkpoint_epoch=best[3], n_train_examples=len(ytr),
        neighborhoods="stable_ID_leave_one_out", actual_output_mode="aggregate_regression_residual",
        unused_classification_options=["output_mode", "probability_mode"])


@torch.no_grad()
def reuse_logits(model, adapter, X: torch.Tensor, cfg: ReuseConfig, *, query_nominal=None) -> torch.Tensor:
    """``calibration_logits`` of the adapted output on one stream (frozen retrieval, no LOO)."""
    nb = neighborhood_inputs(model, X, exclude_identical=False, X_nominal=query_nominal, nominal_adapter=adapter)
    return calibration_logits(adapter(nb["dz"], nb["p0"], nb["du"])[1], cfg)


@torch.no_grad()
def evaluate_reuse(model, adapter, X: torch.Tensor, y: torch.Tensor, cfg: ReuseConfig, *, query_nominal=None, test_groups=None,
                   temperature: float | None = None) -> dict[str, Any]:
    """Pre (retrieval-only) vs post (adapted) on the same case set; flip accounting.

    ``temperature`` (fitted on another stream) adds ``ece_post_T``/``nll_post_T`` of
    softmax(calibration_logits / T); predictions (argmax) are unchanged.
    """
    nb = neighborhood_inputs(model, X, exclude_identical=False, X_nominal=query_nominal, nominal_adapter=adapter)
    p0 = nb["p0"]
    y = y.to(p0.device).long()
    r, s = adapter(nb["dz"], p0, nb["du"])
    pre = p0.argmax(1)
    post = s.argmax(1)
    # kappa * s is the trained softmax input; 1.0 * s == s keeps the default exact
    probs_post = s if cfg.output_mode == "logit_residual" else _probs_from_scores(cfg.score_scale * s, cfg.probability_mode)
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
    if temperature is not None:
        if not (math.isfinite(temperature) and temperature > 0):
            raise ValueError("temperature must be finite and positive")
        # float64 on the CPU, as in calibrate_temperature: some accelerators (Arc A770) lack fp64
        z = calibration_logits(s, cfg).detach().cpu().double() / temperature
        y_cpu = y.cpu()
        metrics["ece_post_T"] = expected_calibration_error(F.softmax(z, dim=1), y_cpu)
        metrics["nll_post_T"] = float(F.cross_entropy(z, y_cpu))
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
