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

For the controlled adapter study, ``comparison_scheme`` optionally executes
``joint_rr`` settings as frozen-retrieval sequential training or as one full
retrieval pass followed by one full adapter pass per cycle. It preserves the
effective summed/separated objective and maximum per-component pass budget.
Historical schedule behavior is unchanged when this override is absent.

Every term is logged separately each epoch. Training queries are training cases
with leave-one-out retrieval. Early stopping uses the validation stream only:
``val_L_post`` (default) or ``val_error`` (post-adaptation error, then L_post).
``retrieval_guard`` g makes a checkpoint eligible only while validation L_pre
stays within (1 + g) of its value before any synchronized update. If the guard
leaves no eligible checkpoint, the state before any synchronized update (epoch
0: the starting retrieval, the untrained adapter and both optimizers' starting
states) is returned with ``best_epoch = 0`` and ``no_eligible_checkpoint``; after
a maintenance edit that state no longer exists and training raises instead.
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
from .chunked import retrieve_chunked
from .core import CoreConfig, make_optimizer
from .nominal import NominalSchema, NominalClassificationAdapter, nominal_difference, fit_nominal_schema
from .reuse import adapter_optimizer
from .magnitude_direction import make_magnitude_direction_adapter, neighborhood_statistics
from .query_aware import QUERY_AWARE_DESIGNS, is_query_aware, query_adapter_type
from .adapter_regularization import add_hidden_dropout, validate_dropout_placement


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
    adapter_dropout: float = 0.0  # legacy first hidden; all_hidden override covers both hidden layers
    score_scale: float = 1.0  # kappa: L_post = cross_entropy(kappa * s, y) (nominal_residual_scores only)
    early_stopping: str = "val_L_post"  # val_L_post | val_error (lexicographic on (error, L_post))
    retrieval_guard: float | None = None  # eligible iff val L_pre <= (1 + guard) * epoch-0 val L_pre
    adapter_design: str = "residual"  # opt-in; historical adapter stays unchanged
    initial_scale: float = 0.01  # magnitude in standardized target / class-score units

    @property
    def maintains(self) -> bool:
        return self.schedule.endswith("_rrr")

    def validate(self, task_type):
        if self.adapter_design not in {"residual", "magnitude_direction"} | QUERY_AWARE_DESIGNS:
            raise ValueError("Unknown adapter_design")
        if self.adapter_design in QUERY_AWARE_DESIGNS and (self.nominal_fields or self.output_mode == "logit_residual"):
            raise ValueError("query-aware synchronization requires represented numeric inputs and additive residuals")
        if not math.isfinite(self.initial_scale) or self.initial_scale <= 0:
            raise ValueError("initial_scale must be finite and positive")
        if self.adapter_design == "magnitude_direction":
            if self.v_delta <= 0 or self.v_post <= 0:
                raise ValueError("Magnitude/direction needs positive raw-proposal and final-output losses")
            if self.output_mode == "logit_residual":
                raise ValueError("Magnitude/direction classification uses nominal_residual_scores")
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
        if self.adapter_dropout > 0 and task_type != "classification" and self.adapter_design == "magnitude_direction":
            raise ValueError("regression dropout is supported for residual adapters")
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
    r = retrieve_chunked(model, X, exclude_identical=False, query_case_ids=query_case_ids)
    w = r["weights"]
    labels = model.labels[r["case_indices"]].float()
    L_near = (w * r["distances"]).sum(1).mean() / max(near_scale, 1e-8)
    split_output = getattr(adapter, "adapter_design", "residual") == "magnitude_direction"
    parts = None
    if model.task_type == "regression":
        p0 = w @ labels.view(w.shape[1], -1)
        if split_output:
            parts = adapter.forward_parts(r["query_features"] - w @ r["case_features"], p0,
                neighborhood_statistics(r, labels, task_type="regression"))
            s = parts.final
        else:
            s = adapter.forward_aggregate(r["query_features"], r["case_features"], labels, w)
        target = y.float().view_as(p0)
        L_pre, L_post = F.mse_loss(p0, target), F.mse_loss(s, target)
        # The raw proposal supplies the gradient blocked by scalar sign(v).
        L_delta = F.mse_loss(parts.raw_proposal if split_output else s - p0, target - p0)
        c_q = torch.linalg.vector_norm(s - p0, dim=1)
        with torch.no_grad():
            err_pre, err_post = F.mse_loss(p0, target).sqrt(), F.mse_loss(s, target).sqrt()
    else:
        p0 = (w @ labels) / (w @ labels).sum(1, keepdim=True).clamp_min(1e-12)
        L_pre = F.nll_loss(torch.log(p0.clamp_min(1e-8)), y.long())
        dz = r["query_features"] - w @ r["case_features"]
        du = nominal_difference(adapter, model, r, query_nominal)
        if split_output:
            parts = adapter.forward_parts(dz, p0,
                neighborhood_statistics(r, labels, task_type="classification"), du)
            rh, s = parts.raw_proposal, parts.final
        else:
            extra = {"query_features":r["query_features"]} if is_query_aware(adapter) else {}
            rh, s = adapter(dz, p0, du, **extra)
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
    terms = {"L_pre": L_pre, "L_post": L_post, "L_near": L_near, "L_delta": L_delta, "L_small": L_sm, "correction_norm": c_q.mean(),
             "err_pre": err_pre.detach(), "err_post": err_post.detach()}
    if parts is not None:
        terms.update(predicted_magnitude=parts.magnitude.mean(),
                     raw_proposal_norm=torch.linalg.vector_norm(parts.raw_proposal, dim=1).mean(),
                     direction_norm=torch.linalg.vector_norm(parts.direction, dim=1).mean())
        if not all(bool(torch.isfinite(value).all()) for value in terms.values()):
            raise ValueError("Nonfinite magnitude/direction loss or diagnostic")
    return terms


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
    run=None,
    comparison_scheme: str | None = None,
    dropout_placement: str | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    cfg.validate(model.task_type)
    validate_dropout_placement(dropout_placement, cfg.adapter_design)
    constructor_dropout = 0.0 if dropout_placement == "all_hidden" else cfg.adapter_dropout
    if comparison_scheme is not None and (comparison_scheme not in {"sequential", "alternating"} or cfg.schedule != "joint_rr"):
        raise ValueError("controlled scheme comparison requires joint_rr settings and sequential or alternating execution")
    if run is not None and (getattr(run, "run_id", None) is None or cfg.maintains):
        raise ValueError("recorded epoch resume requires an open run and a schedule without maintenance")
    started = time.perf_counter()
    torch.manual_seed(cfg.seed)
    device = model.cases.device
    X, y = data.X_train.to(device), data.y_train.to(device)
    if cfg.adapter_design in QUERY_AWARE_DESIGNS or comparison_scheme is not None or dropout_placement is not None:
        # A shape probe must not seed the eval feature cache with an autograd
        # graph that would be reused across frozen-retrieval warm-up batches.
        model._invalidate_case_cache()
        with torch.no_grad():
            feat_dim = model.retrieve(X[:1], exclude_identical=False)["query_features"].shape[1]
    else:
        feat_dim = model.retrieve(X[:1], exclude_identical=False)["query_features"].shape[1]
    nominal_train = nominal_val = None
    if cfg.adapter_design == "magnitude_direction":
        schema = None
        if cfg.nominal_fields:
            schema = fit_nominal_schema(data, cfg.nominal_fields)
            nominal_train = schema.encode(data.nominal_train, rows=len(data.y_train)).to(device)
            nominal_val = schema.encode(data.nominal_val, rows=len(data.y_val)).to(device)
        elif getattr(data, "nominal_train", None):
            raise ValueError("nominal synchronization requires explicit per-field coverage declarations")
        adapter = make_magnitude_direction_adapter(model.task_type, feat_dim,
            num_classes=data.num_classes, hidden_dims=cfg.hidden_dims, output_mode=cfg.output_mode,
            dropout=cfg.adapter_dropout, initial_magnitude=cfg.initial_scale, schema=schema,
            case_ids=torch.arange(len(data.y_train)) if schema is not None else None,
            case_values=nominal_train).to(device)
    elif model.task_type == "classification":
        if cfg.nominal_fields:
            schema = fit_nominal_schema(data, cfg.nominal_fields)
            nominal_train = schema.encode(data.nominal_train, rows=len(data.y_train)).to(device)
            nominal_val = schema.encode(data.nominal_val, rows=len(data.y_val)).to(device)
            adapter = NominalClassificationAdapter(schema=schema, case_ids=torch.arange(len(data.y_train)),
                case_values=nominal_train, feature_dim=feat_dim, num_classes=int(data.num_classes),
                hidden_dims=cfg.hidden_dims, output_mode=cfg.output_mode, dropout=constructor_dropout).to(device)
        else:
            if getattr(data, "nominal_train", None):
                raise ValueError("nominal synchronization requires explicit per-field coverage declarations")
            adapter_type = query_adapter_type("classification", cfg.adapter_design) if cfg.adapter_design in QUERY_AWARE_DESIGNS else ClassificationNNCDHAdapter
            adapter = adapter_type(feat_dim, int(data.num_classes), 0, cfg.hidden_dims, cfg.output_mode,
                                   dropout=constructor_dropout).to(device)
    else:
        adapter_type = query_adapter_type("regression", cfg.adapter_design) if cfg.adapter_design in QUERY_AWARE_DESIGNS else NNCDHAdapter
        adapter = adapter_type(feat_dim, 1, cfg.hidden_dims).to(device)
        # The preserved foundation contains a historical pair network; the T1
        # aggregate path never trains or uses that extra information channel.
        for p in adapter.adapt_net_pair.parameters():
            p.requires_grad_(False)
    if dropout_placement == "all_hidden" or model.task_type == "regression":
        add_hidden_dropout(adapter, cfg.adapter_dropout, all_hidden=dropout_placement == "all_hidden")
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
    # The guard's fallback when no epoch is eligible: this epoch-0 state, whose
    # L_pre is the reference itself. Copies only; unguarded runs take none.
    guarded = cfg.retrieval_guard is not None
    epoch0 = (copy.deepcopy(model.state_dict()), copy.deepcopy(adapter.state_dict()), copy.deepcopy(ropt.state_dict()),
              copy.deepcopy(aopt.state_dict()), copy.deepcopy(fr)) if guarded else None
    maintenance_edited = False

    def improves(v_err, v_post):
        if cfg.early_stopping == "val_L_post":
            return v_post < best[0] - 1e-9
        return v_err < best[7] - 1e-9 or (v_err <= best[7] + 1e-9 and v_post < best[0] - 1e-9)

    retrieval_params = [p for p in model.parameters() if p.requires_grad]
    adapter_params = [p for p in adapter.parameters() if p.requires_grad]

    def phase_of(ep: int) -> str:
        if comparison_scheme == "sequential":
            return "adapter"
        if comparison_scheme == "alternating":
            return "adapter" if ep <= cfg.adapter_warmup_epochs else "alternating"
        if cfg.schedule == "independent":
            return "retrieval" if ep <= cfg.epochs // 2 else "adapter"
        if cfg.schedule.startswith("joint"):
            return "adapter" if ep <= cfg.adapter_warmup_epochs else "joint"
        return "retrieval" if ((ep - 1) // cfg.block_epochs) % 2 == 0 else "adapter"

    start, already_stopped = 1, False
    if run is not None:
        from . import records
        resumed = run.load_resume()
        if resumed is not None:
            state = records.unpack(resumed.state["sync_state"])
            if state.get("comparison_scheme") != comparison_scheme:
                raise ValueError("resume comparison scheme differs")
            if state.get("dropout_placement") != dropout_placement:
                raise ValueError("resume dropout placement differs")
            model.load_state_dict(state["model"])
            model._invalidate_case_cache()
            adapter.load_state_dict(state["adapter"])
            ropt.load_state_dict(state["retrieval_optimizer"])
            aopt.load_state_dict(state["adapter_optimizer"])
            best, bad, hist, fr = state["best"], state["bad"], state["history"], state["free_radius"]
            start, already_stopped = state["epoch"]+1, state["stopped"]
            resumed.restore_rng({"shuffle":g})
    for ep in range(start, cfg.epochs + 1) if not already_stopped else ():
        ph = phase_of(ep)
        phases = ("retrieval", "adapter") if ph == "alternating" else (ph,)
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
        if cfg.adapter_design == "magnitude_direction":
            agg.update(predicted_magnitude=0.0, raw_proposal_norm=0.0, direction_norm=0.0)
        nb = 0
        query_case_pairs = 0
        component_updates = {"retrieval": 0, "adapter": 0}
        for phase in phases:
            for s in range(0, y.numel(), core_cfg.batch_size):
                b = perm[s : s + core_cfg.batch_size].to(device)
                query_case_pairs += len(b) * model.case_count()
                if phase == "retrieval":
                    adapter.eval()  # frozen adapter: no dropout noise (LayerNorm/ReLU nets are mode-free otherwise)
                    with frozen_parameters(adapter):
                        T = _forward_terms(model, adapter, X[b], y[b], fr, output_mode=cfg.output_mode, near_scale=near_scale, query_case_ids=b, query_nominal=None if nominal_train is None else nominal_train[b], score_scale=cfg.score_scale)
                    adapter.train()
                    if cfg.schedule == "independent":
                        L = T["L_pre"]
                    else:
                        L = cfg.w_pre * T["L_pre"] + cfg.w_post * T["L_post"] + cfg.w_near * T["L_near"]
                    if comparison_scheme is not None and cfg.joint_objective == "summed":
                        L_A = cfg.v_post*T["L_post"] + cfg.v_delta*T["L_delta"] + cfg.v_small*T["L_small"]
                        L = L + L_A
                    ropt.zero_grad()
                    aopt.zero_grad()
                    L.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), core_cfg.grad_clip)
                    ropt.step()  # adapter grads are discarded: L_R updates retrieval only
                    if core_cfg.mcb_enabled:
                        model.update_momentum_encoder(core_cfg.mcb_momentum)
                elif phase == "joint":
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
                    if comparison_scheme is not None and ep > cfg.adapter_warmup_epochs and cfg.joint_objective == "summed":
                        L_R = cfg.w_pre*T["L_pre"] + cfg.w_post*T["L_post"] + cfg.w_near*T["L_near"]
                        L = L_R + L
                    aopt.zero_grad()
                    L.backward()
                    aopt.step()
                    model.train()
                for k in agg:
                    agg[k] += float(T[k].detach())
                nb += 1
                for component in component_updates:
                    if phase in {component, "joint"}:
                        component_updates[component] += 1
        rec = {"epoch": ep, "phase": ph, "optimizer_updates": nb,
               "training_examples": len(y)*len(phases), "retrieval_query_case_pairs": query_case_pairs,
               **({"train_before_update": before_update, "train_after_update": diagnostic_terms()} if log_terms else {}),
               **{k: v / max(nb, 1) for k, v in agg.items()}}
        if comparison_scheme is not None:
            rec.update(retrieval_updates=component_updates["retrieval"], adapter_updates=component_updates["adapter"])
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
                maintenance_edited = True
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
        adapter_trained = any(h["phase"] in {"adapter", "joint", "alternating"} for h in hist)
        should_stop = False
        if adapter_trained and not guard_rejected and improves(v_err, v_post):
            best, bad = (v_post, copy.deepcopy(model.state_dict()), copy.deepcopy(adapter.state_dict()), ep,
                         copy.deepcopy(ropt.state_dict()), copy.deepcopy(aopt.state_dict()), copy.deepcopy(fr), v_err), 0
        elif adapter_trained:
            bad += 1
            pending_maintenance = (cfg.maintains and maintenance_hook is not None and
                                   any(ep < e <= cfg.epochs for e in maintenance_epochs))
            if bad > cfg.patience and not pending_maintenance:
                should_stop = True
        if run is not None:
            run.log_step(ep, {k:v for k,v in rec.items() if isinstance(v, (int, float))})
            run.save_resume({"sync_state":records.pack(dict(epoch=ep, model=model.state_dict(),
                adapter=adapter.state_dict(), retrieval_optimizer=ropt.state_dict(),
                adapter_optimizer=aopt.state_dict(), best=best, bad=bad, history=hist,
                free_radius=fr, stopped=should_stop, comparison_scheme=comparison_scheme,
                dropout_placement=dropout_placement))}, step=ep, generators={"shuffle":g})
        if should_stop:
            break
    no_eligible = best[1] is None
    if no_eligible and guarded:
        # Never return a guard-violating state: fall back to epoch 0 (eligible by
        # definition), unless a maintenance edit has since replaced it.
        if maintenance_edited:
            raise RuntimeError("retrieval_guard rejected every checkpoint after a maintenance edit; the epoch-0 "
                               "state cannot be restored without undoing that edit")
        best = (float(V0["L_post"]), *epoch0[:2], 0, *epoch0[2:], float(V0["err_post"]))
    if best[1] is not None:
        model.load_state_dict(best[1])
        model._invalidate_case_cache()
        adapter.load_state_dict(best[2])
        ropt.load_state_dict(best[4])
        aopt.load_state_dict(best[5])
        fr = best[6]
    adapter.eval()
    # Without the guard, no selected epoch is possible only with nonfinite
    # validation losses; that case keeps the last state, as before.
    return adapter, {"history": hist, "best_epoch": best[3], "best_val_L_post": best[0],
                     "best_val_err": best[7], "v_pre0": v_pre0, "v_err_pre0": v_err_pre0,
                     "no_eligible_checkpoint": no_eligible,
                     "guard_rejected_epochs": sum(h["guard_rejected"] for h in hist),
                     "adapter_optimizer": type(aopt).__name__,
                     "elapsed_seconds": time.perf_counter() - started,
                     "retrieval_optimizer_state": copy.deepcopy(ropt.state_dict()),
                     "adapter_optimizer_state": copy.deepcopy(aopt.state_dict()),
                     "optimizer_checkpoint_epoch": best[3],
                     "free_radius": None if fr is None else fr.to_dict()}
