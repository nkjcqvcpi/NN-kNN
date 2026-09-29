"""Component synchronization for classification (plan section 6, Phase 4).

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
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn.functional as F

from model.nn_cdh import ClassificationNNCDHAdapter

from .calibration import FreeRadius, l_small
from .core import CoreConfig, make_optimizer


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


def _forward_terms(model, adapter, X, y, fr: FreeRadius | None, *, output_mode: str, near_scale: float) -> dict[str, torch.Tensor]:
    r = model.retrieve(X, exclude_identical=True)  # LOO for training-case queries
    w = r["weights"]
    labels = model.labels[r["case_indices"]].float()
    p0 = (w @ labels) / (w @ labels).sum(1, keepdim=True).clamp_min(1e-12)
    L_pre = F.nll_loss(torch.log(p0.clamp_min(1e-8)), y.long())
    L_near = (w * r["distances"]).sum(1).mean() / max(near_scale, 1e-8)
    dz = r["query_features"] - w @ r["case_features"]
    rh, s = adapter(dz, p0, None)
    y1 = F.one_hot(y.long(), p0.size(1)).float()
    if output_mode == "nominal_residual_scores":
        L_post = F.cross_entropy(s, y.long())
        L_delta = F.mse_loss(rh, y1 - p0)
        c_q = torch.linalg.vector_norm(s - p0, dim=1)
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
    maintenance_hook: Callable[[int, Any, torch.optim.Optimizer], dict[str, Any] | None] | None = None,
    maintenance_epochs: set[int] | None = None,
    recalibrate: Callable[[int], FreeRadius] | None = None,
    core_optimizer: torch.optim.Optimizer | None = None,
) -> tuple[ClassificationNNCDHAdapter, dict[str, Any]]:
    torch.manual_seed(cfg.seed)
    device = model.cases.device
    X, y = data.X_train.to(device), data.y_train.to(device)
    C = int(data.num_classes)
    feat_dim = model.retrieve(X[:1], exclude_identical=False)["query_features"].shape[1]
    adapter = ClassificationNNCDHAdapter(feat_dim, C, 0, cfg.hidden_dims, cfg.output_mode).to(device)
    ropt = core_optimizer if core_optimizer is not None else make_optimizer(model, core_cfg)
    aopt = torch.optim.Adam(adapter.parameters(), lr=cfg.adapter_lr)
    g = torch.Generator().manual_seed(cfg.seed)
    hist: list[dict[str, Any]] = []
    best = (float("inf"), None, None, -1)
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
                T = _forward_terms(model, adapter, X[b], y[b], fr, output_mode=cfg.output_mode, near_scale=near_scale)
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
                for p in model.parameters():
                    p.requires_grad_(False)
                T = _forward_terms(model, adapter, X[b], y[b], fr, output_mode=cfg.output_mode, near_scale=near_scale)
                for p in model.parameters():
                    p.requires_grad_(True)
                if model.momentum_encoder is not None:
                    for p in model.momentum_encoder.parameters():
                        p.requires_grad_(False)
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
            rec["maintenance"] = maintenance_hook(ep, model, ropt)
            if recalibrate is not None:
                fr = recalibrate(ep)
                rec["free_radius"] = fr.to_dict()
            best, bad = (float("inf"), None, None, -1), 0
        # validation: post-adaptation task loss on the untouched validation stream
        model.eval()
        adapter.eval()
        with torch.no_grad():
            r = model.retrieve(data.X_val.to(device), exclude_identical=False)
            w = r["weights"]
            labels = model.labels[r["case_indices"]].float()
            p0 = (w @ labels) / (w @ labels).sum(1, keepdim=True).clamp_min(1e-12)
            _, s_ = adapter(r["query_features"] - w @ r["case_features"], p0, None)
            yv = data.y_val.to(device).long()
            v_post = float(F.cross_entropy(s_, yv) if cfg.output_mode == "nominal_residual_scores" else F.nll_loss(torch.log(s_.clamp_min(1e-8)), yv))
            v_pre = float(F.nll_loss(torch.log(p0.clamp_min(1e-8)), yv))
        rec.update({"val_L_post": v_post, "val_L_pre": v_pre, "n_cases": model.case_count()})
        hist.append(rec)
        # select on post loss only after the adapter has trained at least once
        if any(h["phase"] == "adapter" for h in hist) and v_post < best[0] - 1e-9:
            best, bad = (v_post, copy.deepcopy(model.state_dict()), copy.deepcopy(adapter.state_dict()), ep), 0
        elif any(h["phase"] == "adapter" for h in hist):
            bad += 1
            if bad > cfg.patience and not any(e > ep for e in maintenance_epochs):
                break
    if best[1] is not None:
        model.load_state_dict(best[1])
        model._invalidate_case_cache()
        adapter.load_state_dict(best[2])
    adapter.eval()
    return adapter, {"history": hist, "best_epoch": best[3], "free_radius": None if fr is None else fr.to_dict()}
