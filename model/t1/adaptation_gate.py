"""Experimental apply-or-skip head, retaining the full old NN-CDH proposal.

No retrieval training is performed here. Gate labels compare both branches on
training targets; inference receives only representation/neighborhood evidence.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F

from .calibration import FreeRadius, l_small
from . import records

MODES = ("full_control", "probability_gate", "utility_gate")


@dataclass
class GateParts:
    proposal: torch.Tensor
    full: torch.Tensor
    logit: torch.Tensor
    apply: torch.Tensor
    final: torch.Tensor


class ApplyOrSkipAdapter(nn.Module):
    """Two heads share the old adapter trunk; inference uses an exact 0/1 gate."""
    def __init__(self, network, task_type, mode):
        super().__init__()
        if task_type not in {"classification", "regression"} or mode not in MODES:
            raise ValueError("unknown task or gate mode")
        self.task_type, self.mode = task_type, mode
        self.network = copy.deepcopy(network)
        if not isinstance(self.network[-1], nn.Linear):
            raise ValueError("old residual network must end in a linear layer")
        hidden, labels = self.network[-1].in_features, self.network[-1].out_features
        self.gate = nn.Linear(hidden + 4 + 2*labels, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        if mode == "full_control":
            self.gate.requires_grad_(False)
        self.register_buffer("feature_mean", torch.zeros(self.gate.in_features))
        self.register_buffer("feature_std", torch.ones(self.gate.in_features))
        self.register_buffer("normalizer_fitted", torch.tensor(False))

    def _features(self, x, p0, statistics):
        if statistics.shape != (len(x), 4) or not torch.isfinite(statistics).all() or (statistics < 0).any():
            raise ValueError("four finite nonnegative neighborhood statistics required")
        hidden = self.network[:-1](x)
        raw = self.network[-1](hidden)
        proposal = raw.tanh() if self.task_type == "classification" else raw
        features = torch.cat((hidden, torch.log1p(statistics), p0, proposal), dim=1)
        return proposal, features

    @torch.no_grad()
    def fit_normalizer(self, train):
        self.eval()
        _, features = self._features(train["x"], train["p0"], train["statistics"])
        self.feature_mean.copy_(features.mean(0))
        self.feature_std.copy_(features.std(0, unbiased=False).clamp_min(1e-3))
        self.normalizer_fitted.fill_(True)

    def forward(self, x, p0, statistics):
        if not bool(self.normalizer_fitted):
            raise ValueError("fit the gate normalizer on training inputs first")
        proposal, features = self._features(x, p0, statistics)
        z = ((features-self.feature_mean)/self.feature_std).clamp(-10, 10)
        logit = self.gate(z).squeeze(1)
        apply = torch.ones_like(logit, dtype=torch.bool) if self.mode == "full_control" else logit > 0
        full = p0+proposal
        final = torch.where(apply[:, None], full, p0)
        return GateParts(proposal, full, logit, apply, final)


def losses_per_example(scores, target, task_type, score_scale=1.):
    if task_type == "regression":
        return (scores-target.float().reshape_as(scores)).square().mean(1)
    return F.cross_entropy(score_scale*scores, target.long(), reduction="none")


def gate_loss(logit, before_loss, full_loss, *, weighted):
    # No gradient into the target/weight calculation. Otherwise the adapter
    # could alter the supervision instead of improving the decision.
    benefit = (before_loss-full_loss).detach()
    loss = F.binary_cross_entropy_with_logits(logit, (benefit > 0).float(), reduction="none")
    if weighted:
        weight = benefit.abs()
        return (weight*loss).sum()/weight.sum().clamp_min(1e-12)
    return loss.mean()


def objective(model, values, settings):
    p = model(values["x"], values["p0"], values["statistics"])
    y, p0 = values["target"], values["p0"]
    pre = losses_per_example(p0, y, model.task_type, settings["score_scale"])
    full = losses_per_example(p.full, y, model.task_type, settings["score_scale"])
    target = F.one_hot(y.long(), p0.shape[1]).float() if model.task_type == "classification" else y.float().reshape_as(p0)
    residual = F.mse_loss(p.proposal, target-p0)
    small = l_small(p.proposal.norm(dim=1), FreeRadius(**settings["free_radius"])) if settings["free_radius"] is not None else full.mean()*0
    adapter = settings["v_post"]*full.mean()+settings["v_delta"]*residual+settings["v_small"]*small
    gate = p.logit.sum()*0 if model.mode == "full_control" else gate_loss(
        p.logit, pre, full, weighted=model.mode == "utility_gate")
    return adapter+gate, {"adapter_loss": adapter.detach(), "gate_loss": gate.detach()}


@torch.no_grad()
def outcome(model, values, settings):
    model.eval()
    p = model(values["x"], values["p0"], values["statistics"])
    target, p0 = values["target"], values["p0"]
    cls = model.task_type == "classification"
    def metric(s):
        return float((s.argmax(1) != target.long()).float().mean()) if cls else float(
            losses_per_example(s, target, model.task_type).mean().sqrt())
    pre_loss = losses_per_example(p0, target, model.task_type, settings["score_scale"])
    full_loss = losses_per_example(p.full, target, model.task_type, settings["score_scale"])
    final_loss = losses_per_example(p.final, target, model.task_type, settings["score_scale"])
    if cls:
        before = p0.argmax(1) != target.long()
        after = p.final.argmax(1) != target.long()
        full_error = p.full.argmax(1) != target.long()
        helped, harmed = before & ~after, ~before & after
        oracle = float(torch.minimum(before, full_error).float().mean())
    else:
        before = losses_per_example(p0,target,"regression")
        after = losses_per_example(p.final,target,"regression")
        helped, harmed = after < before-1e-8, after > before+1e-8
        oracle = float(torch.minimum(before, losses_per_example(p.full,target,"regression")).mean().sqrt())
    result = dict(n=len(target), error=metric(p.final), pre_error=metric(p0), full_error=metric(p.full),
                  task_loss=float(final_loss.mean()), apply_rate=float(p.apply.float().mean()),
                  helped=int(helped.sum()), harmed=int(harmed.sum()), oracle_branch_error=oracle)
    result["surrogate_help_precision"] = float((full_loss[p.apply]<pre_loss[p.apply]).float().mean()) if p.apply.any() else None
    if not cls:
        c, r = (p.final-p0).flatten(), (target.float().view_as(p0)-p0).flatten()
        result.update(wrong_direction=int(((c*r<0)&(c.abs()>1e-8)&(r.abs()>1e-8)).sum()),
                      harmful_overshoot=int(((c*r>0)&(c.abs()>2*r.abs())&(r.abs()>1e-8)).sum()))
    return result, p


def selection_key(result, task_type):
    return (result["error"], result["task_loss"]) if task_type == "classification" else (result["task_loss"],)


def train_continuation(model, train, val, settings, seed, *, run):
    """Record every continuation and resume without replaying completed epochs."""
    if run is None or getattr(run, "run_id", None) is None:
        raise TypeError("an open recorded run is mandatory")
    torch.manual_seed(seed+1701)
    generator = torch.Generator().manual_seed(seed+1701)
    model.fit_normalizer(train)
    opt_class = torch.optim.AdamW if settings["weight_decay"] else torch.optim.Adam
    optimizer = opt_class([p for p in model.parameters() if p.requires_grad],
                          lr=settings["lr"], weight_decay=settings["weight_decay"])
    initial, _ = outcome(model, val, settings)
    best = dict(key=selection_key(initial, model.task_type), epoch=0,
                model=copy.deepcopy(model.state_dict()), optimizer=copy.deepcopy(optimizer.state_dict()))
    history, bad, start = [], 0, 1
    resumed = run.load_resume()
    if resumed is not None:
        state = resumed.state
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        best, history, bad, start = state["best"], state["history"], state["bad"], state["epoch"]+1
        resumed.restore_rng({"shuffle": generator})
    for epoch in range(start, settings["epochs"]+1):
        if bad > settings["patience"]:
            break
        model.train()
        order = torch.randperm(len(train["target"]), generator=generator)
        total, n = 0., 0
        for begin in range(0, len(order), settings["batch_size"]):
            idx = order[begin:begin+settings["batch_size"]]
            batch = {k:v[idx] for k,v in train.items()}
            loss, terms = objective(model, batch, settings)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite gate/adapter loss")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach())*len(idx)
            n += len(idx)
        validation, _ = outcome(model, val, settings)
        key = selection_key(validation, model.task_type)
        if key < tuple(best["key"]):
            best = dict(key=key, epoch=epoch, model=copy.deepcopy(model.state_dict()),
                        optimizer=copy.deepcopy(optimizer.state_dict()))
            bad = 0
        else:
            bad += 1
        row = dict(epoch=epoch, train_loss=total/n, val_error=validation["error"],
                   val_loss=validation["task_loss"], val_apply_rate=validation["apply_rate"])
        history.append(row)
        run.log_step(epoch, {k:v for k,v in row.items() if k!="epoch"})
        # Small heads: retaining an epoch checkpoint is cheap and avoids losing work.
        run.save_resume(dict(epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict(),
                             best=best, history=history, bad=bad), step=epoch, generators={"shuffle":generator})
    model.load_state_dict(best["model"])
    model.eval()
    run.log_record("history", records.ordered(history))
    run.save_state("checkpoint", records.storable(dict(model=best["model"], optimizer=best["optimizer"],
                    selected_epoch=best["epoch"], mode=model.mode, task_type=model.task_type, settings=settings)))
    return dict(selected_epoch=best["epoch"], epochs_run=len(history))
