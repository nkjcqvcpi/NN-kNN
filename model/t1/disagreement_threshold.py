"""One learned disagreement cutoff; predictions use full adaptation or retrieval.

Only the scalar cutoff is learned by the routing objective. Its temperature and
normalization are fixed. Branch costs are detached; the joint arm trains the
residual network with its original supervised objective, even on skipped cases.
"""
from __future__ import annotations

import copy
import torch
from torch import nn
from torch.nn import functional as F

from . import records
from .adaptation_gate import GateParts, losses_per_example, outcome, selection_key
from .calibration import FreeRadius, l_small

MODES = ("threshold_frozen", "threshold_joint")


class DisagreementThresholdAdapter(nn.Module):
    def __init__(self, network, task_type, mode, *, temperature=.25):
        super().__init__()
        if task_type not in {"classification", "regression"} or mode not in MODES:
            raise ValueError("unknown task or threshold mode")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.network = copy.deepcopy(network)
        self.network.requires_grad_(mode == "threshold_joint")
        self.task_type, self.mode = task_type, mode
        self.threshold = nn.Parameter(torch.tensor(0.))
        self.register_buffer("temperature", torch.tensor(float(temperature)))
        self.register_buffer("disagreement_mean", torch.tensor(0.))
        self.register_buffer("disagreement_std", torch.tensor(1.))
        self.register_buffer("normalizer_fitted", torch.tensor(False))

    @staticmethod
    def disagreement(statistics):
        if statistics.ndim != 2 or statistics.shape[1] != 4:
            raise ValueError("four neighborhood statistics required")
        d = statistics[:, 3]
        if not torch.isfinite(d).all() or (d < 0).any():
            raise ValueError("disagreement must be finite and nonnegative")
        return d

    @torch.no_grad()
    def fit_normalizer(self, train):
        d = self.disagreement(train["statistics"])
        self.disagreement_mean.copy_(d.mean())
        self.disagreement_std.copy_(d.std(unbiased=False).clamp_min(1e-6))
        self.threshold.copy_((d.median()-self.disagreement_mean)/self.disagreement_std)
        self.normalizer_fitted.fill_(True)

    def logits(self, statistics):
        if not bool(self.normalizer_fitted):
            raise ValueError("fit on training statistics first")
        z = (self.disagreement(statistics)-self.disagreement_mean)/self.disagreement_std
        return (z-self.threshold)/self.temperature

    @property
    def raw_threshold(self):
        return self.disagreement_mean+self.disagreement_std*self.threshold

    def forward(self, x, p0, statistics):
        raw = self.network(x)
        proposal = raw.tanh() if self.task_type == "classification" else raw
        full = p0+proposal
        logit = self.logits(statistics)
        apply = logit > 0
        return GateParts(proposal, full, logit, apply, torch.where(apply[:, None], full, p0))


def routing_costs(p0, full, target, task_type):
    """Direct classification error or MSE, not a differentiable label surrogate."""
    if task_type == "classification":
        before = (p0.argmax(1) != target.long()).float()
        after = (full.argmax(1) != target.long()).float()
    else:
        before = losses_per_example(p0, target, task_type)
        after = losses_per_example(full, target, task_type)
    return before.detach(), after.detach()


def routing_loss(logits, before, full):
    """Expected branch-cost improvement under a soft Bernoulli router.

    The fixed positive normalization improves numerical scale, without changing
    the optimum. This is not the loss of an interpolated prediction.
    """
    difference = (full-before).detach()
    return (logits.sigmoid()*difference).mean()/difference.abs().mean().clamp_min(1e-8)


def adapter_loss(model, batch, settings):
    p = model(batch["x"], batch["p0"], batch["statistics"])
    y, p0 = batch["target"], batch["p0"]
    full = losses_per_example(p.full, y, model.task_type, settings["score_scale"]).mean()
    target = F.one_hot(y.long(), p0.shape[1]).float() if model.task_type == "classification" else y.float().reshape_as(p0)
    residual = F.mse_loss(p.proposal, target-p0)
    small = l_small(p.proposal.norm(dim=1), FreeRadius(**settings["free_radius"])) if settings["free_radius"] is not None else full*0
    return settings["v_post"]*full+settings["v_delta"]*residual+settings["v_small"]*small


@torch.no_grad()
def routing_diagnostics(model, train):
    model.eval()
    p = model(train["x"], train["p0"], train["statistics"])
    before, full = routing_costs(train["p0"], p.full, train["target"], model.task_type)
    return dict(threshold=float(model.raw_threshold), threshold_standardized=float(model.threshold),
                train_apply_rate=float(p.apply.float().mean()),
                train_routing_cost=float(torch.where(p.apply, full, before).mean()),
                train_soft_cost=float((before+p.logit.sigmoid()*(full-before)).mean()),
                train_retrieval_cost=float(before.mean()), train_full_cost=float(full.mean()),
                train_benefit_fraction=float((full<before).float().mean()),
                train_harm_fraction=float((full>before).float().mean()))


def train_threshold(model, train, val, settings, seed, *, run):
    if run is None or getattr(run, "run_id", None) is None:
        raise TypeError("an open recorded run is mandatory")
    torch.manual_seed(seed+1701)
    generator = torch.Generator().manual_seed(seed+1701)
    model.fit_normalizer(train)
    joint = model.mode == "threshold_joint"
    gate_optimizer = torch.optim.Adam([model.threshold], lr=settings["threshold_lr"])
    opt_class = torch.optim.AdamW if settings["weight_decay"] else torch.optim.Adam
    optimizer = opt_class(model.network.parameters(), lr=settings["lr"],
                          weight_decay=settings["weight_decay"]) if joint else None
    initial, _ = outcome(model, val, settings)

    def snapshot(epoch, key):
        return dict(epoch=epoch, key=key, model=copy.deepcopy(model.state_dict()),
                    optimizer=copy.deepcopy(optimizer.state_dict()) if joint else None,
                    gate_optimizer=copy.deepcopy(gate_optimizer.state_dict()))

    best = snapshot(0, selection_key(initial, model.task_type))
    history, start, bad = [], 1, 0
    resumed = run.load_resume()
    if resumed is not None:
        state = resumed.state
        model.load_state_dict(state["model"])
        gate_optimizer.load_state_dict(state["gate_optimizer"])
        if joint:
            optimizer.load_state_dict(state["optimizer"])
        best, history, bad, start = state["best"], state["history"], state["bad"], state["epoch"]+1
        resumed.restore_rng({"shuffle": generator})
    for epoch in range(start, settings["epochs"]+1):
        if joint and bad > settings["patience"]:
            break
        total, n = 0., 0
        if joint:
            model.train()
            order = torch.randperm(len(train["target"]), generator=generator)
            for begin in range(0, len(order), settings["batch_size"]):
                idx = order[begin:begin+settings["batch_size"]]
                batch = {k:v[idx] for k,v in train.items()}
                loss = adapter_loss(model, batch, settings)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite adapter loss")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += float(loss.detach())*len(idx)
                n += len(idx)
        model.eval()
        with torch.no_grad():
            p = model(train["x"], train["p0"], train["statistics"])
            before, full = routing_costs(train["p0"], p.full, train["target"], model.task_type)
        for _ in range(settings["threshold_steps"]):
            loss = routing_loss(model.logits(train["statistics"]), before, full)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite threshold loss")
            gate_optimizer.zero_grad()
            loss.backward()
            gate_optimizer.step()
        validation, _ = outcome(model, val, settings)
        key = selection_key(validation, model.task_type)
        # Frozen arm uses the final training iterate: validation cannot choose its threshold.
        if not joint or key < tuple(best["key"]):
            best = snapshot(epoch, key)
            bad = 0
        else:
            bad += 1
        row = dict(epoch=epoch, adapter_train_loss=total/n if n else 0.,
                   threshold_loss=float(loss.detach()), val_error=validation["error"],
                   val_loss=validation["task_loss"], val_apply_rate=validation["apply_rate"],
                   **routing_diagnostics(model, train))
        history.append(row)
        run.log_step(epoch, {k:v for k,v in row.items() if k != "epoch"})
        run.save_resume(dict(epoch=epoch, model=model.state_dict(),
                             optimizer=optimizer.state_dict() if joint else None,
                             gate_optimizer=gate_optimizer.state_dict(), best=best,
                             history=history, bad=bad), step=epoch, generators={"shuffle":generator})
    model.load_state_dict(best["model"])
    model.eval()
    run.log_record("history", records.ordered(history))
    run.save_state("checkpoint", records.storable(dict(**best, mode=model.mode,
                    task_type=model.task_type, settings=settings)))
    return dict(selected_epoch=best["epoch"], epochs_run=len(history),
                **routing_diagnostics(model, train))
