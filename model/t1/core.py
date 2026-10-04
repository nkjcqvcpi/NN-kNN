"""Retrieval-core construction, training and evaluation for T1 experiments.

Built on the maintained ``NN_KNN_Model`` (plan section 4: preserve current
behavior as the selectable reference). What this module adds:

* case-bias warm start from the mean k-th nearest non-self *learned* distance
  (plan section 7, k=5), instead of the core default manual bias 0;
* leave-one-out retrieval during training (``ignore_identical_in_training``);
* early stopping / checkpoint selection on the validation stream only;
* declared maintenance checkpoints (epoch boundaries) with optimizer realignment;
* optional MCB: a no-gradient EMA copy of the online encoder represents stored
  cases (plan section 3.5).
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

from model.nnknn_model import (
    GlocalFeatureWeight,
    MLPFeatureProjector,
    NN_KNN_Model,
    classification_class_mass_loss,
    default_args,
)

from .geometry import case_case_distance


@dataclass
class CoreConfig:
    task_type: str
    representation: str = "raw"  # raw | mlp
    mlp_dims: tuple[int, ...] = (32, 16)
    mlp_dropout: float = 0.0
    glocal_fw_set_num: int = 1
    tau: float = 1.0
    top_k: int = 5
    pre_topk_mask: bool = True
    case_normalizer: str = "softmax"
    lr_case: float = 1e-2
    lr_glocal: float = 1e-2
    lr_feature: float = 1e-3
    weight_decay: float = 1e-5
    lambda_case_bias: float = 0.0
    epochs: int = 200
    patience: int = 30
    batch_size: int = 32
    bias_init: str = "knn_mean"  # knn_mean | manual
    bias_init_k: int = 5
    bias_manual_value: float = 0.0
    mcb_enabled: bool = False
    mcb_momentum: float = 0.99
    normalize_embeddings: bool = False  # L2-normalize learned embeddings (guards against metric collapse)
    grad_clip: float = 0.5
    seed: int = 0

    def to_model_kwargs(self) -> dict[str, Any]:
        kw = dict(default_args)
        kw.update(
            {
                "task_type": self.task_type,
                "normalize_over_cases": True,
                "case_normalizer": self.case_normalizer,
                "case_score_mode": "bias_minus_distance",
                "tau": self.tau,
                "top_k": self.top_k,
                "pre_topk_mask": self.pre_topk_mask,
                "glocal_fw_set_num": self.glocal_fw_set_num,
                "ignore_identical_in_training": True,
                "bias_manual_set": True,
                "bias_manual_value": 0.0,  # replaced below by the declared initializer
                "sampling_cases_flag": False,
                "use_mcb": self.mcb_enabled,
                "mcb_momentum": self.mcb_momentum,
                "mcb_normalize_embeddings": bool(self.normalize_embeddings),
                "explanation_mode": False,
                "regression_locality": False,
            }
        )
        return kw


def _labels_tensor(y: torch.Tensor, task_type: str, num_classes: int | None) -> torch.Tensor:
    if task_type == "classification":
        return torch.nn.functional.one_hot(y.long(), num_classes=int(num_classes)).float()
    return y.float().view(-1, 1)


def build_model(X_train: torch.Tensor, y_train: torch.Tensor, cfg: CoreConfig, num_classes: int | None) -> NN_KNN_Model:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    fe = None
    if cfg.representation == "mlp":
        fe = MLPFeatureProjector(hidden_dims=cfg.mlp_dims, dropout=cfg.mlp_dropout)
        with torch.no_grad():
            fe(X_train[:2])  # materialize LazyLinear
    elif cfg.representation != "raw":
        raise ValueError(cfg.representation)
    feat_dim = X_train.shape[1] if fe is None else fe.feature_dim
    glocal = GlocalFeatureWeight(feat_dim, cfg.glocal_fw_set_num)
    mom = None
    if cfg.mcb_enabled:
        if fe is None:
            raise ValueError("MCB needs a trainable encoder (representation='mlp'); raw features have nothing to average.")
        mom = copy.deepcopy(fe)
    model = NN_KNN_Model(
        X_train,
        _labels_tensor(y_train, cfg.task_type, num_classes),
        feature_extractor=fe,
        glocal_weightor=glocal,
        momentum_encoder=mom,
        **cfg.to_model_kwargs(),
    )
    init_bias = cfg.bias_manual_value
    if cfg.bias_init == "knn_mean":
        init_bias = kth_neighbor_learned_distance(model, cfg.bias_init_k)
    with torch.no_grad():
        model.biases.fill_(float(init_bias))
    model.case_default_bias = float(init_bias)
    model.t1_bias_init = {"rule": cfg.bias_init, "k": cfg.bias_init_k, "value": float(init_bias)}
    return model


@torch.no_grad()
def kth_neighbor_learned_distance(model: NN_KNN_Model, k: int) -> float:
    """b_default^(0) = mean_i r_i^(k): k-th nearest non-self learned distance (plan section 7)."""
    D = case_case_distance(model)
    D.fill_diagonal_(float("inf"))
    D = torch.where(D < 1e-8, torch.full_like(D, float("inf")), D)  # exact duplicates are not neighbours
    kk = min(k, D.shape[1] - 1)
    r = torch.topk(D, kk, dim=1, largest=False).values[:, -1]
    r = r[torch.isfinite(r)]
    return float(r.mean().item()) if r.numel() else 0.0


def make_optimizer(model: NN_KNN_Model, cfg: CoreConfig) -> torch.optim.Optimizer:
    fe = list(model.feature_extractor.parameters()) if model.feature_extractor is not None else []
    gw = list(model.glocal_weightor.parameters()) if model.glocal_weightor is not None else []
    shared = {id(p) for p in fe + gw}
    mom = {id(p) for p in (model.momentum_encoder.parameters() if model.momentum_encoder is not None else [])}
    adapter = {id(p) for p in (model.nn_cdh.parameters() if model.nn_cdh is not None else [])}
    cls_ad = getattr(model, "classification_adapter", None)
    if cls_ad is not None:
        adapter |= {id(p) for p in cls_ad.parameters()}
    case = [p for p in model.parameters() if id(p) not in shared | mom | adapter and p.requires_grad]
    groups = [{"params": case, "lr": cfg.lr_case, "name": "case"}, {"params": gw, "lr": cfg.lr_glocal, "name": "glocal"}]
    if fe:
        groups.append({"params": fe, "lr": cfg.lr_feature, "name": "feature"})
    return torch.optim.Adam(groups, weight_decay=cfg.weight_decay)


def clone_optimizer(src: torch.optim.Optimizer, model: NN_KNN_Model, cfg: CoreConfig) -> torch.optim.Optimizer:
    """Optimizer for a deep-copied model that continues ``src``'s Adam state.

    Continuing training of a converged core with a *fresh* Adam is unstable: the
    first bias-corrected steps are ~lr * sign(grad) for every case bias and feature
    weight at once, which can collapse retrieval (observed: digits 0.98 -> 0.10 in a
    few epochs). All post-core phases (fine-tune after maintenance, synchronization,
    M2 retraining) therefore continue the core optimizer's state.
    """
    opt = make_optimizer(model, cfg)
    # load_state_dict can retain same-device moment tensors by reference.
    # Independent candidate continuations must not mutate the source Adam state.
    opt.load_state_dict(copy.deepcopy(src.state_dict()))
    return opt


def task_loss(model: NN_KNN_Model, final_predictions: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if model.task_type == "classification":
        return classification_class_mass_loss(final_predictions, y)
    return torch.mean((final_predictions.view(-1) - y.float().view(-1)) ** 2)


@dataclass
class TrainResult:
    model: NN_KNN_Model
    optimizer: torch.optim.Optimizer
    history: list[dict[str, Any]] = field(default_factory=list)
    best_epoch: int = -1
    best_val: float = float("nan")
    epochs_run: int = 0


def train_retrieval(
    model: NN_KNN_Model,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    cfg: CoreConfig,
    *,
    epochs: int | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    checkpoint_hook: Callable[[int, NN_KNN_Model, torch.optim.Optimizer], dict[str, Any] | None] | None = None,
    checkpoint_epochs: set[int] | None = None,
    select_best: bool = True,
    lr_scale: float = 1.0,
    include_initial: bool = False,
) -> TrainResult:
    """Train retrieval (biases, glocal weights, optional encoder) with LOO on the training stream.

    ``checkpoint_hook`` runs *after* the listed epochs (safe maintenance checkpoints).
    Early stopping and best-state selection use the validation stream only. When a
    maintenance checkpoint actually changes model state or active capacity, the
    best-state tracker restarts. No-op hooks preserve the matching best model
    and optimizer. Only executable checkpoints within this budget delay stopping.
    """
    device = model.cases.device
    opt = optimizer or make_optimizer(model, cfg)
    if lr_scale != 1.0:
        # fine-tuning phases of an already-trained core use a declared smaller step
        base = {"case": cfg.lr_case, "glocal": cfg.lr_glocal, "feature": cfg.lr_feature}
        for grp in opt.param_groups:
            grp["lr"] = base.get(grp.get("name"), grp["lr"]) * lr_scale
    epochs = cfg.epochs if epochs is None else epochs
    if epochs < 1:
        raise ValueError("Training epochs must be positive")
    g = torch.Generator().manual_seed(cfg.seed)
    Xtr, ytr = X_train.to(device), y_train.to(device)
    best = (math.inf, None, -1)
    best_optimizer = None
    if include_initial and select_best:
        # the starting state is a candidate: fine-tuning can only be kept if validation improves
        v0 = evaluate(model, X_val, y_val)
        best = (v0["loss_pre"], copy.deepcopy(model.state_dict()), 0)
        best_optimizer = copy.deepcopy(opt.state_dict())
    bad = 0
    res = TrainResult(model, opt)
    checkpoint_epochs = checkpoint_epochs or set()
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(Xtr.size(0), generator=g)
        tot, nb = 0.0, 0
        for s in range(0, Xtr.size(0), cfg.batch_size):
            b = perm[s : s + cfg.batch_size].to(device)
            # queries are training cases: the model's LOO mask drops their own case
            out = model(Xtr[b], exclude_identical=False, query_case_ids=b)
            loss = task_loss(model, out[0], ytr[b])
            if cfg.lambda_case_bias:
                loss = loss + cfg.lambda_case_bias * (model.biases[: model.case_count()] ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
            opt.step()
            if cfg.mcb_enabled:
                model.update_momentum_encoder(cfg.mcb_momentum)
            tot += float(loss.item())
            nb += 1
        val = evaluate(model, X_val, y_val)
        rec = {"epoch": ep, "train_loss": tot / max(nb, 1), "val_loss": val["loss_pre"], "val_metric": val["metric_pre"], "n_cases": model.case_count()}
        if ep in checkpoint_epochs and checkpoint_hook is not None:
            before_count = model.case_count()
            before_state = copy.deepcopy(model.state_dict())
            info = checkpoint_hook(ep, model, opt)
            rec["checkpoint"] = info
            after_state = model.state_dict()
            changed = (model.case_count() != before_count or
                       set(before_state) != set(after_state) or
                       any(not torch.equal(value, after_state[key])
                           for key, value in before_state.items()))
            rec["checkpoint_state_changed"] = changed
            if changed:
                best, bad = (math.inf, None, -1), 0  # real state change: restart selection
                best_optimizer = None
                val = evaluate(model, X_val, y_val)
                rec["val_loss_after_checkpoint"] = val["loss_pre"]
        res.history.append(rec)
        if val["loss_pre"] < best[0] - 1e-9:
            best = (val["loss_pre"], copy.deepcopy(model.state_dict()), ep)
            best_optimizer = copy.deepcopy(opt.state_dict()) if select_best else None
            bad = 0
        else:
            bad += 1
            pending = checkpoint_hook is not None and any(ep < e <= epochs for e in checkpoint_epochs)
            if bad > cfg.patience and not pending:
                break
    res.epochs_run = ep
    if select_best and best[1] is not None:
        model.load_state_dict(best[1])
        model._invalidate_case_cache()
        opt.load_state_dict(best_optimizer)
    res.best_epoch, res.best_val = best[2], best[0]
    return res


# ---------------------------------------------------------------------------
# Evaluation (pre-adaptation retrieval-only output)
# ---------------------------------------------------------------------------


def expected_calibration_error(probs: torch.Tensor, y: torch.Tensor, n_bins: int = 15) -> float:
    conf, pred = probs.max(1)
    acc = (pred == y).float()
    edges = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += float(m.float().mean() * (acc[m].mean() - conf[m].mean()).abs())
    return ece


@torch.no_grad()
def evaluate(model: NN_KNN_Model, X: torch.Tensor, y: torch.Tensor, *, batch_size: int = 512, case_mask=None) -> dict[str, Any]:
    """Retrieval-only (pre-adaptation) metrics on an evaluation stream; no LOO (queries are not cases)."""
    was = model.training
    model.eval()
    device = model.cases.device
    preds = []
    for s in range(0, X.size(0), batch_size):
        r = model.retrieve(X[s : s + batch_size].to(device), exclude_identical=False, case_mask=case_mask)
        labels = model.labels[r["case_indices"]].float()
        f = r["weights"] @ labels
        preds.append(f)
    F = torch.cat(preds)
    y = y.to(device)
    if was:
        model.train()
    if model.task_type == "classification":
        p = F / F.sum(1, keepdim=True).clamp_min(1e-12)
        loss = float(classification_class_mass_loss(p, y).item())
        acc = float((p.argmax(1) == y).float().mean().item())
        return {"loss_pre": loss, "metric_pre": acc, "accuracy_pre": acc, "ece_pre": expected_calibration_error(p.cpu(), y.cpu()), "pred_pre": p.argmax(1).cpu(), "prob_pre": p.cpu()}
    f = F.view(-1)
    mse = float(torch.mean((f - y.float().view(-1)) ** 2).item())
    mae = float(torch.mean((f - y.float().view(-1)).abs()).item())
    return {"loss_pre": mse, "metric_pre": -mse, "rmse_pre": mse ** 0.5, "mae_pre": mae, "pred_pre": f.cpu()}


@torch.no_grad()
def retrieval_events(model, X, y, *, stream, run_id, top=5, adapter=None):
    """Complete active contributions and final decisions from one retrieval event."""
    from .outcomes import final_prediction, frozen_evaluation
    device = model.cases.device
    out = []
    ids = model.active_case_ids()
    with frozen_evaluation(model, adapter):
        for start in range(0, len(X), 256):
            xb = X[start:start + 256].to(device)
            r = model.retrieve(xb, exclude_identical=False)
            w = r["weights"]
            pre, final, loss_kind = final_prediction(model, xb, r, adapter=adapter)
            for b in range(len(xb)):
                sel = torch.nonzero(w[b] > 0).view(-1)
                out.append({
                    "retrieval_event_id": f"{run_id}-{stream}-{start + b}",
                    "stream": stream, "query_index": start + b,
                    "case_ids": ids[r["case_indices"][sel]].tolist(),
                    "activations": w[b, sel].cpu().tolist(),
                    "distances": r["distances"][b, sel].cpu().tolist(),
                    "case_biases": model.biases[r["case_indices"][sel]].detach().cpu().tolist(),
                    "pre_adaptation_prediction": int(pre[b].argmax()) if model.task_type == "classification" else float(pre[b, 0]),
                    "final_prediction": int(final[b].argmax()) if model.task_type == "classification" else float(final[b].view(-1)[0]),
                    "final_prediction_loss_kind": loss_kind,
                    "target": int(y[start + b]) if model.task_type == "classification" else float(y[start + b]),
                })
    return out
