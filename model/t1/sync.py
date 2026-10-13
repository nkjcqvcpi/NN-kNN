"""Component synchronization for classification/regression (plan section 6).

    L_pre   = NLL of the retrieval-only class mass p0
    L_post  = task loss after adaptation
    L_near  = activation-weighted learned query-to-case distance (declared locality metric)
    L_delta = MSE(r_hat, r*)
    L_small = free-correction penalty (calibration.l_small)

    L_R = w_pre * L_pre + w_post * L_post + w_near * L_near   -> retrieval parameters only
    L_A = v_post * L_post + v_delta * L_delta + v_small * L_small -> adapter parameters only

Schedules compared (Phase 4):
    independent        retrieval on L_pre, then adapter on (L_post, L_delta) with retrieval frozen
    alternating_rr     alternate blocks of L_R and L_A updates (retrieval-reuse synchronized)
    alternating_rrr    alternating_rr + checkpoint-discrete maintenance (retrieval-reuse-retain)
    joint_rr           end-to-end: every minibatch updates retrieval and adapter together
    joint_rrr          joint_rr + checkpoint-discrete maintenance

Joint schedules share one forward pass per minibatch, so L_post's gradient
reaches retrieval through the current adapter. ``joint_objective="separated"``
keeps the plan's split (L_R -> retrieval parameters, L_A -> adapter parameters);
``"summed"`` backpropagates L_R + L_A into every trainable parameter.
``adapter_warmup_epochs`` trains a fresh adapter alone before joint updates.

Every term is logged separately each epoch. Training queries are training cases
with leave-one-out retrieval. Early stopping uses the validation stream only:
``val_L_post`` (default) or ``val_error`` (post-adaptation error, then L_post).
``retrieval_guard`` g makes a checkpoint eligible only while validation L_pre
stays within (1 + g) of its value before any synchronized update.
"""

from __future__ import annotations

import copy
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn.functional as F

from model.nn_cdh import ClassificationNNCDHAdapter, NNCDHAdapter

from .calibration import FreeRadius, l_small
from .core import CoreConfig, make_optimizer
from .nominal import NominalSchema, NominalClassificationAdapter, nominal_difference, fit_nominal_schema
from .reuse import adapter_optimizer


SCHEDULES = {"independent", "alternating_rr", "alternating_rrr", "joint_rr", "joint_rrr"}
EARLY_STOPPING = {"val_L_post", "val_error"}
ERROR_TERMS = {"err_pre", "err_post"}  # logged as val_err_*, kept out of the per-term loss logs


@dataclass
class SyncConfig:
    schedule: str  # independent | alternating_rr | alternating_rrr | joint_rr | joint_rrr
    w_pre: float
    w_post: float
    w_near: float
    v_post: float
    v_delta: float
    v_small: float
    output_mode: str = "nominal_residual_scores"
    block_epochs: int = 2
    epochs: int = 100
    patience: int = 20
    adapter_lr: float = 1e-3
    hidden_dims: tuple[int, int] = (64, 32)
    seed: int = 0
    nominal_fields: tuple[dict[str, Any], ...] = ()
    joint_objective: str = "separated"  # separated | summed (joint schedules only)
    adapter_warmup_epochs: int = 0  # joint schedules: adapter-only epochs before joint updates
    # Logging-only full passes over the training stream before/after each epoch's
    # update: every epoch (1), every k-th epoch and epoch 1 (k), or never (0).
    # They run in eval mode without gradients and never change training.
    diagnostics_every: int = 1
    adapter_weight_decay: float = 0.0  # > 0 switches the adapter optimizer from Adam to AdamW
    adapter_dropout: float = 0.0  # classification adapters only
    score_scale: float = 1.0  # kappa: L_post = cross_entropy(kappa * s, y) (nominal_residual_scores only)
    early_stopping: str = "val_L_post"  # val_L_post | val_error (lexicographic on (error, L_post))
    retrieval_guard: float | None = None  # eligible iff val L_pre <= (1 + guard) * epoch-0 val L_pre

    @property
    def maintains(self) -> bool:
        return self.schedule.endswith("_rrr")

    def validate(self, task_type):
        if self.schedule not in SCHEDULES:
            raise ValueError("Unknown synchronization schedule")
        alternating = self.schedule.startswith("alternating")
        if self.epochs < 2 or self.block_epochs < 1 or (alternating and self.block_epochs >= self.epochs) or self.patience < 0:
            raise ValueError("Synchronization must include retrieval and adapter updates")
        if self.joint_objective not in {"separated", "summed"}:
            raise ValueError("joint_objective must be 'separated' or 'summed'")
        if self.adapter_warmup_epochs < 0 or self.adapter_warmup_epochs >= self.epochs:
            raise ValueError("adapter warm-up must leave joint epochs")
        if self.diagnostics_every < 0:
            raise ValueError("diagnostics_every must be nonnegative")
        if self.adapter_warmup_epochs and not self.schedule.startswith("joint"):
            raise ValueError("adapter warm-up applies to joint schedules only")
        if min(self.w_pre, self.w_post, self.w_near, self.v_post, self.v_delta, self.v_small) < 0:
            raise ValueError("Synchronization loss weights must be nonnegative")
        if self.v_post + self.v_delta <= 0 or self.w_pre + self.w_post + self.w_near <= 0:
            raise ValueError("Each component needs a learning objective")
        expected = {"nominal_residual_scores", "logit_residual"} if task_type == "classification" else {"aggregate_regression"}
        if self.output_mode not in expected:
            raise ValueError(f"Output mode {self.output_mode!r} is invalid for {task_type}")
        if not self.adapter_weight_decay >= 0 or not 0 <= self.adapter_dropout < 1:
            raise ValueError("adapter weight decay must be nonnegative and dropout in [0, 1)")
        if self.adapter_dropout > 0 and task_type != "classification":
            raise ValueError("adapter dropout applies to classification adapters only")
        if not (math.isfinite(self.score_scale) and self.score_scale > 0):
            raise ValueError("score_scale must be finite and positive")
        if self.score_scale != 1.0 and self.output_mode != "nominal_residual_scores":
            raise ValueError("score_scale applies to nominal_residual_scores only")
        if self.early_stopping not in EARLY_STOPPING:
            raise ValueError(f"early_stopping must be one of {sorted(EARLY_STOPPING)}")
        if self.retrieval_guard is not None and not self.retrieval_guard >= 0:
            raise ValueError("retrieval_guard must be nonnegative (or None)")
        if self.nominal_fields:
            if task_type != "classification":
                raise ValueError("nominal classification synchronization cannot be used for regression")
            NominalSchema(self.nominal_fields)


@contextmanager
def frozen_parameters(module):
    """Preserve intentionally frozen cases, labels, target encoders and features."""
    states = [(p, p.requires_grad) for p in module.parameters()]
    try:
        for p, _ in states:
            p.requires_grad_(False)
        yield
    finally:
        for p, state in states:
            p.requires_grad_(state)


def _forward_terms(model, adapter, X, y, fr: FreeRadius | None, *, output_mode: str, near_scale: float, query_case_ids=None, query_nominal=None, score_scale: float = 1.0) -> dict[str, torch.Tensor]:
    """Loss terms of one forward pass, plus no-grad pre/post errors.

    Errors are the argmax error rate (classification) or RMSE in training-
    standardized units (regression); they never enter a gradient.
    """
    r = model.retrieve(X, exclude_identical=False, query_case_ids=query_case_ids)
    w = r["weights"]
    labels = model.labels[r["case_indices"]].float()
    L_near = (w * r["distances"]).sum(1).mean() / max(near_scale, 1e-8)
    if model.task_type == "regression":
        p0 = w @ labels.view(w.shape[1], -1)
        s = adapter.forward_aggregate(r["query_features"], r["case_features"], labels, w)
        target = y.float().view_as(p0)
        L_pre, L_post = F.mse_loss(p0, target), F.mse_loss(s, target)
        L_delta = F.mse_loss(s - p0, target - p0)
        c_q = torch.linalg.vector_norm(s - p0, dim=1)
        with torch.no_grad():
            err_pre, err_post = F.mse_loss(p0, target).sqrt(), F.mse_loss(s, target).sqrt()
    else:
        p0 = (w @ labels) / (w @ labels).sum(1, keepdim=True).clamp_min(1e-12)
        L_pre = F.nll_loss(torch.log(p0.clamp_min(1e-8)), y.long())
        dz = r["query_features"] - w @ r["case_features"]
        du = nominal_difference(adapter, model, r, query_nominal)
        rh, s = adapter(dz, p0, du)
        y1 = F.one_hot(y.long(), p0.size(1)).float()
        if output_mode == "nominal_residual_scores":
            L_post = F.cross_entropy(score_scale * s, y.long())  # x * 1.0 is exact: default unchanged
            L_delta = F.mse_loss(rh, y1 - p0)
        else:
            L_post = F.nll_loss(torch.log(s.clamp_min(1e-8)), y.long())
            L_delta = F.mse_loss(s - p0, y1 - p0)
        c_q = torch.linalg.vector_norm(s - p0, dim=1)
        with torch.no_grad():
            err_pre = (p0.argmax(1) != y.long()).float().mean()
            err_post = (s.argmax(1) != y.long()).float().mean()
    L_sm = l_small(c_q, fr) if fr is not None else torch.zeros((), device=p0.device)
    return {"L_pre": L_pre, "L_post": L_post, "L_near": L_near, "L_delta": L_delta, "L_small": L_sm, "correction_norm": c_q.mean(),
            "err_pre": err_pre.detach(), "err_post": err_post.detach()}


def joint_backward(T: dict[str, torch.Tensor], cfg: SyncConfig, retrieval_params, adapter_params) -> None:
    """Accumulate one joint step's gradients from a single forward pass ``T``.

    ``separated``: L_A reaches adapter parameters only and L_R retrieval
    parameters only; L_post still differentiates through the adapter into
    retrieval. ``summed``: L_R + L_A reaches every trainable parameter.
    """
    L_R = cfg.w_pre * T["L_pre"] + cfg.w_post * T["L_post"] + cfg.w_near * T["L_near"]
    L_A = cfg.v_post * T["L_post"] + cfg.v_delta * T["L_delta"] + cfg.v_small * T["L_small"]
    if cfg.joint_objective == "summed":
        (L_R + L_A).backward()
    else:
        L_A.backward(inputs=list(adapter_params), retain_graph=True)
        L_R.backward(inputs=list(retrieval_params))


def train_synchronized(
    model,
    data,
    core_cfg: CoreConfig,
    cfg: SyncConfig,
    fr: FreeRadius | None,
    *,
    near_scale: float,
    maintenance_hook: Callable[[int, Any, torch.optim.Optimizer, Any], dict[str, Any] | None] | None = None,
    maintenance_epochs: set[int] | None = None,
    recalibrate: Callable[[int], FreeRadius] | None = None,
    core_optimizer: torch.optim.Optimizer | None = None,
    epoch_observer: Callable[[int, Any, Any], dict[str, Any]] | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    cfg.validate(model.task_type)
    started = time.perf_counter()
    torch.manual_seed(cfg.seed)
    device = model.cases.device
    X, y = data.X_train.to(device), data.y_train.to(device)
    feat_dim = model.retrieve(X[:1], exclude_identical=False)["query_features"].shape[1]
    nominal_train = nominal_val = None
    if model.task_type == "classification":
        if cfg.nominal_fields:
            schema = fit_nominal_schema(data, cfg.nominal_fields)
            nominal_train = schema.encode(data.nominal_train, rows=len(data.y_train)).to(device)
            nominal_val = schema.encode(data.nominal_val, rows=len(data.y_val)).to(device)
            adapter = NominalClassificationAdapter(schema=schema, case_ids=torch.arange(len(data.y_train)),
                case_values=nominal_train, feature_dim=feat_dim, num_classes=int(data.num_classes),
                hidden_dims=cfg.hidden_dims, output_mode=cfg.output_mode, dropout=cfg.adapter_dropout).to(device)
        else:
            if getattr(data, "nominal_train", None):
                raise ValueError("nominal synchronization requires explicit per-field coverage declarations")
            adapter = ClassificationNNCDHAdapter(feat_dim, int(data.num_classes), 0, cfg.hidden_dims, cfg.output_mode,
                                                 dropout=cfg.adapter_dropout).to(device)
    else:
        adapter = NNCDHAdapter(feat_dim, 1, cfg.hidden_dims).to(device)
        # The preserved foundation contains a historical pair network; the T1
        # aggregate path never trains or uses that extra information channel.
        for p in adapter.adapt_net_pair.parameters():
            p.requires_grad_(False)
    ropt = core_optimizer if core_optimizer is not None else make_optimizer(model, core_cfg)
    aopt = adapter_optimizer([p for p in adapter.parameters() if p.requires_grad], cfg.adapter_lr,
                             cfg.adapter_weight_decay, device)
    g = torch.Generator().manual_seed(cfg.seed)
    hist: list[dict[str, Any]] = []
    # (val L_post, model, adapter, epoch, retrieval Adam, adapter Adam, free radius, val error)
    empty = (float("inf"), None, None, -1, None, None, None, float("inf"))
    best = empty
    bad = 0
    maintenance_epochs = maintenance_epochs or set()
    X_val, y_val = data.X_val.to(device), data.y_val.to(device)

    def validation_terms():
        model.eval()
        adapter.eval()
        with torch.no_grad():
            return _forward_terms(model, adapter, X_val, y_val, fr, output_mode=cfg.output_mode,
                                  near_scale=near_scale, query_nominal=nominal_val, score_scale=cfg.score_scale)

    # Reference for the retrieval guard, before any synchronized update. Eval
    # mode without gradients changes no parameter, RNG stream or optimizer state.
    V0 = validation_terms()
    v_pre0, v_err_pre0 = float(V0["L_pre"]), float(V0["err_pre"])

    def improves(v_err, v_post):
        if cfg.early_stopping == "val_L_post":
            return v_post < best[0] - 1e-9
        return v_err < best[7] - 1e-9 or (v_err <= best[7] + 1e-9 and v_post < best[0] - 1e-9)

    retrieval_params = [p for p in model.parameters() if p.requires_grad]
    adapter_params = [p for p in adapter.parameters() if p.requires_grad]

    def phase_of(ep: int) -> str:
        if cfg.schedule == "independent":
            return "retrieval" if ep <= cfg.epochs // 2 else "adapter"
        if cfg.schedule.startswith("joint"):
            return "adapter" if ep <= cfg.adapter_warmup_epochs else "joint"
        return "retrieval" if ((ep - 1) // cfg.block_epochs) % 2 == 0 else "adapter"

    for ep in range(1, cfg.epochs + 1):
        ph = phase_of(ep)
        # Same train-only LOO stream before/after the update; eval mode keeps
        # dropout and mutable encoders from confounding the comparison.
        def diagnostic_terms():
            model.eval()
            adapter.eval()
            with torch.no_grad():
                totals = {}
                for start in range(0, len(y), core_cfg.batch_size):
                    stop = min(start + core_cfg.batch_size, len(y))
                    terms = _forward_terms(model, adapter, X[start:stop], y[start:stop], fr,
                        output_mode=cfg.output_mode, near_scale=near_scale,
                        query_case_ids=torch.arange(start, stop, device=device),
                        query_nominal=None if nominal_train is None else nominal_train[start:stop],
                        score_scale=cfg.score_scale)
                    for key, value in terms.items():
                        if key not in ERROR_TERMS:
                            totals[key] = totals.get(key, 0.) + float(value) * (stop - start) / len(y)
            return totals
        log_terms = bool(cfg.diagnostics_every) and (ep == 1 or ep % cfg.diagnostics_every == 0)
        before_update = diagnostic_terms() if log_terms else None
        model.train()
        adapter.train()
        perm = torch.randperm(y.numel(), generator=g)
        agg = {k: 0.0 for k in ("L_pre", "L_post", "L_near", "L_delta", "L_small", "correction_norm")}
        nb = 0
        query_case_pairs = 0
        for s in range(0, y.numel(), core_cfg.batch_size):
            b = perm[s : s + core_cfg.batch_size].to(device)
            query_case_pairs += len(b) * model.case_count()
            if ph == "retrieval":
                adapter.eval()  # frozen adapter: no dropout noise (LayerNorm/ReLU nets are mode-free otherwise)
                with frozen_parameters(adapter):
                    T = _forward_terms(model, adapter, X[b], y[b], fr, output_mode=cfg.output_mode, near_scale=near_scale, query_case_ids=b, query_nominal=None if nominal_train is None else nominal_train[b], score_scale=cfg.score_scale)
                adapter.train()
                if cfg.schedule == "independent":
                    L = T["L_pre"]
                else:
                    L = cfg.w_pre * T["L_pre"] + cfg.w_post * T["L_post"] + cfg.w_near * T["L_near"]
                ropt.zero_grad()
                aopt.zero_grad()
                L.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), core_cfg.grad_clip)
                ropt.step()  # adapter grads are discarded: L_R updates retrieval only
                if core_cfg.mcb_enabled:
                    model.update_momentum_encoder(core_cfg.mcb_momentum)
            elif ph == "joint":
                T = _forward_terms(model, adapter, X[b], y[b], fr, output_mode=cfg.output_mode, near_scale=near_scale, query_case_ids=b, query_nominal=None if nominal_train is None else nominal_train[b], score_scale=cfg.score_scale)
                ropt.zero_grad()
                aopt.zero_grad()
                joint_backward(T, cfg, retrieval_params, adapter_params)
                torch.nn.utils.clip_grad_norm_(model.parameters(), core_cfg.grad_clip)
                ropt.step()
                aopt.step()
                if core_cfg.mcb_enabled:
                    model.update_momentum_encoder(core_cfg.mcb_momentum)
            else:
                model.eval()  # frozen retrieval, LOO still enforced explicitly
                with frozen_parameters(model):
                    T = _forward_terms(model, adapter, X[b], y[b], fr, output_mode=cfg.output_mode, near_scale=near_scale, query_case_ids=b, query_nominal=None if nominal_train is None else nominal_train[b], score_scale=cfg.score_scale)
                if cfg.schedule == "independent":
                    L = cfg.v_post * T["L_post"] + cfg.v_delta * T["L_delta"]  # v0 adapter losses, no L_small
                else:
                    L = cfg.v_post * T["L_post"] + cfg.v_delta * T["L_delta"] + cfg.v_small * T["L_small"]
                aopt.zero_grad()
                L.backward()
                aopt.step()
                model.train()
            for k in agg:
                agg[k] += float(T[k].detach())
            nb += 1
        rec = {"epoch": ep, "phase": ph, "optimizer_updates": nb,
               "training_examples": len(y), "retrieval_query_case_pairs": query_case_pairs,
               **({"train_before_update": before_update, "train_after_update": diagnostic_terms()} if log_terms else {}),
               **{k: v / max(nb, 1) for k, v in agg.items()}}
        if ep in maintenance_epochs and maintenance_hook is not None and cfg.maintains:
            before_count = model.case_count()
            before_state = copy.deepcopy(model.state_dict())
            rec["maintenance"] = maintenance_hook(ep, model, ropt, adapter)
            after_state = model.state_dict()
            changed = (before_count != model.case_count() or before_state.keys() != after_state.keys() or
                       any(not torch.equal(v, after_state[k]) for k, v in before_state.items()))
            rec["maintenance_state_changed"] = changed
            if recalibrate is not None:
                fr = recalibrate(ep)
                rec["free_radius"] = fr.to_dict()
            # Earlier predictions are invalid after a real case/parameter edit;
            # an audit or no-op selection does not invalidate a better state.
            if changed:
                best, bad = empty, 0
            rec["train_after_maintenance"] = diagnostic_terms()
        # validation: post-adaptation task loss on the untouched validation stream
        V = validation_terms()
        v_post, v_pre, v_err = float(V["L_post"]), float(V["L_pre"]), float(V["err_post"])
        guard_rejected = cfg.retrieval_guard is not None and not v_pre <= (1 + cfg.retrieval_guard) * v_pre0
        rec.update({"val_L_post": v_post, "val_L_pre": v_pre,
                    "validation_terms": {k: float(v) for k, v in V.items() if k not in ERROR_TERMS}, "n_cases": model.case_count(),
                    "val_err_pre": float(V["err_pre"]), "val_err_post": v_err, "guard_rejected": guard_rejected})
        if epoch_observer is not None:
            rec["stability"] = epoch_observer(ep, model, adapter)
        hist.append(rec)
        # select only after the adapter has trained at least once; guard-rejected
        # epochs are never selected but still count toward patience
        adapter_trained = any(h["phase"] in {"adapter", "joint"} for h in hist)
        if adapter_trained and not guard_rejected and improves(v_err, v_post):
            best, bad = (v_post, copy.deepcopy(model.state_dict()), copy.deepcopy(adapter.state_dict()), ep,
                         copy.deepcopy(ropt.state_dict()), copy.deepcopy(aopt.state_dict()), copy.deepcopy(fr), v_err), 0
        elif adapter_trained:
            bad += 1
            pending_maintenance = (cfg.maintains and maintenance_hook is not None and
                                   any(ep < e <= cfg.epochs for e in maintenance_epochs))
            if bad > cfg.patience and not pending_maintenance:
                break
    if best[1] is not None:
        model.load_state_dict(best[1])
        model._invalidate_case_cache()
        adapter.load_state_dict(best[2])
        ropt.load_state_dict(best[4])
        aopt.load_state_dict(best[5])
        fr = best[6]
    adapter.eval()
    return adapter, {"history": hist, "best_epoch": best[3], "best_val_L_post": best[0],
                     "best_val_err": best[7], "v_pre0": v_pre0, "v_err_pre0": v_err_pre0,
                     "guard_rejected_epochs": sum(h["guard_rejected"] for h in hist),
                     "adapter_optimizer": type(aopt).__name__,
                     "elapsed_seconds": time.perf_counter() - started,
                     "retrieval_optimizer_state": copy.deepcopy(ropt.state_dict()),
                     "adapter_optimizer_state": copy.deepcopy(aopt.state_dict()),
                     "optimizer_checkpoint_epoch": best[3],
                     "free_radius": None if fr is None else fr.to_dict()}
