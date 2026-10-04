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

Every term is logged separately each epoch. Training queries are training cases
with leave-one-out retrieval. Early stopping uses the validation stream only.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn.functional as F

from model.nn_cdh import ClassificationNNCDHAdapter, NNCDHAdapter

from .calibration import FreeRadius, l_small
from .core import CoreConfig, make_optimizer
from .nominal import NominalSchema, NominalClassificationAdapter, nominal_difference, fit_nominal_schema


@dataclass
class SyncConfig:
    schedule: str  # independent | alternating_rr | alternating_rrr
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

    def validate(self, task_type):
        if self.schedule not in {"independent", "alternating_rr", "alternating_rrr"}:
            raise ValueError("Unknown synchronization schedule")
        if self.epochs < 2 or self.block_epochs < 1 or (self.schedule != "independent" and self.block_epochs >= self.epochs) or self.patience < 0:
            raise ValueError("Synchronization must include retrieval and adapter updates")
        if min(self.w_pre, self.w_post, self.w_near, self.v_post, self.v_delta, self.v_small) < 0:
            raise ValueError("Synchronization loss weights must be nonnegative")
        if self.v_post + self.v_delta <= 0 or self.w_pre + self.w_post + self.w_near <= 0:
            raise ValueError("Each component needs a learning objective")
        expected = {"nominal_residual_scores", "logit_residual"} if task_type == "classification" else {"aggregate_regression"}
        if self.output_mode not in expected:
            raise ValueError(f"Output mode {self.output_mode!r} is invalid for {task_type}")
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


def _forward_terms(model, adapter, X, y, fr: FreeRadius | None, *, output_mode: str, near_scale: float, query_case_ids=None, query_nominal=None) -> dict[str, torch.Tensor]:
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
    else:
        p0 = (w @ labels) / (w @ labels).sum(1, keepdim=True).clamp_min(1e-12)
        L_pre = F.nll_loss(torch.log(p0.clamp_min(1e-8)), y.long())
        dz = r["query_features"] - w @ r["case_features"]
        du = nominal_difference(adapter, model, r, query_nominal)
        rh, s = adapter(dz, p0, du)
        y1 = F.one_hot(y.long(), p0.size(1)).float()
        if output_mode == "nominal_residual_scores":
            L_post = F.cross_entropy(s, y.long())
            L_delta = F.mse_loss(rh, y1 - p0)
        else:
            L_post = F.nll_loss(torch.log(s.clamp_min(1e-8)), y.long())
            L_delta = F.mse_loss(s - p0, y1 - p0)
        c_q = torch.linalg.vector_norm(s - p0, dim=1)
    L_sm = l_small(c_q, fr) if fr is not None else torch.zeros((), device=p0.device)
    return {"L_pre": L_pre, "L_post": L_post, "L_near": L_near, "L_delta": L_delta, "L_small": L_sm, "correction_norm": c_q.mean()}


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
) -> tuple[torch.nn.Module, dict[str, Any]]:
    cfg.validate(model.task_type)
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
                hidden_dims=cfg.hidden_dims, output_mode=cfg.output_mode).to(device)
        else:
            if getattr(data, "nominal_train", None):
                raise ValueError("nominal synchronization requires explicit per-field coverage declarations")
            adapter = ClassificationNNCDHAdapter(feat_dim, int(data.num_classes), 0, cfg.hidden_dims, cfg.output_mode).to(device)
    else:
        adapter = NNCDHAdapter(feat_dim, 1, cfg.hidden_dims).to(device)
        # The preserved foundation contains a historical pair network; the T1
        # aggregate path never trains or uses that extra information channel.
        for p in adapter.adapt_net_pair.parameters():
            p.requires_grad_(False)
    ropt = core_optimizer if core_optimizer is not None else make_optimizer(model, core_cfg)
    aopt = torch.optim.Adam([p for p in adapter.parameters() if p.requires_grad], lr=cfg.adapter_lr)
    g = torch.Generator().manual_seed(cfg.seed)
    hist: list[dict[str, Any]] = []
    best = (float("inf"), None, None, -1, None, None, None)
    bad = 0
    maintenance_epochs = maintenance_epochs or set()

    def phase_of(ep: int) -> str:
        if cfg.schedule == "independent":
            return "retrieval" if ep <= cfg.epochs // 2 else "adapter"
        return "retrieval" if ((ep - 1) // cfg.block_epochs) % 2 == 0 else "adapter"

    for ep in range(1, cfg.epochs + 1):
        ph = phase_of(ep)
        model.train()
        adapter.train()
        perm = torch.randperm(y.numel(), generator=g)
        agg = {k: 0.0 for k in ("L_pre", "L_post", "L_near", "L_delta", "L_small", "correction_norm")}
        nb = 0
        for s in range(0, y.numel(), core_cfg.batch_size):
            b = perm[s : s + core_cfg.batch_size].to(device)
            if ph == "retrieval":
                with frozen_parameters(adapter):
                    T = _forward_terms(model, adapter, X[b], y[b], fr, output_mode=cfg.output_mode, near_scale=near_scale, query_case_ids=b, query_nominal=None if nominal_train is None else nominal_train[b])
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
            else:
                model.eval()  # frozen retrieval, LOO still enforced explicitly
                with frozen_parameters(model):
                    T = _forward_terms(model, adapter, X[b], y[b], fr, output_mode=cfg.output_mode, near_scale=near_scale, query_case_ids=b, query_nominal=None if nominal_train is None else nominal_train[b])
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
        rec = {"epoch": ep, "phase": ph, **{k: v / max(nb, 1) for k, v in agg.items()}}
        if ep in maintenance_epochs and maintenance_hook is not None and cfg.schedule == "alternating_rrr":
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
                best, bad = (float("inf"), None, None, -1, None, None, None), 0
        # validation: post-adaptation task loss on the untouched validation stream
        model.eval()
        adapter.eval()
        with torch.no_grad():
            V = _forward_terms(model, adapter, data.X_val.to(device), data.y_val.to(device),
                               fr, output_mode=cfg.output_mode, near_scale=near_scale, query_nominal=nominal_val)
            v_post, v_pre = float(V["L_post"]), float(V["L_pre"])
        rec.update({"val_L_post": v_post, "val_L_pre": v_pre, "n_cases": model.case_count()})
        hist.append(rec)
        # select on post loss only after the adapter has trained at least once
        if any(h["phase"] == "adapter" for h in hist) and v_post < best[0] - 1e-9:
            best, bad = (v_post, copy.deepcopy(model.state_dict()), copy.deepcopy(adapter.state_dict()), ep,
                         copy.deepcopy(ropt.state_dict()), copy.deepcopy(aopt.state_dict()), copy.deepcopy(fr)), 0
        elif any(h["phase"] == "adapter" for h in hist):
            bad += 1
            pending_maintenance = (cfg.schedule == "alternating_rrr" and maintenance_hook is not None and
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
    return adapter, {"history": hist, "best_epoch": best[3],
                     "retrieval_optimizer_state": copy.deepcopy(ropt.state_dict()),
                     "adapter_optimizer_state": copy.deepcopy(aopt.state_dict()),
                     "optimizer_checkpoint_epoch": best[3],
                     "free_radius": None if fr is None else fr.to_dict()}
