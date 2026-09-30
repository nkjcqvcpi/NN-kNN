"""Authorized case revision and causal evaluation (plan sections 3.3 and 11; T1.2).

Revise is OFF in the T1.1 pipeline (``revise_enabled = false``). This module
provides the mechanism and evaluation harness for T1.2:

* ``InterventionLog`` - every edit is logged with actor/authority, reason,
  before/after state and restoration data, linked by ``intervention_id``.
* edit operations: relabel (correct solution), adjust bias, quarantine /
  release, archive-remove / restore, protect.
* ``evaluate_m0_m1_m2`` - M0 saved baseline, M1 edit only (no gradients),
  M2 edit + controlled retraining under a fixed, reported budget; flip matrix,
  targeted and collateral effects.
* ``flagging_ablation`` - the initial ablation matrix of T1_REVISE_RETAIN.md:
  provenance-only, bias-only, combined T, random (matched budget) and oracle
  flagging, with a simulated reviewer that corrects only truly corrupted cases
  it inspects (a controlled stand-in for the later human study).

Automatic relabeling by neighbor consensus is deliberately NOT provided:
routing a suspected-bad case to an action requires review (plan 3.3).
"""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch

from .core import CoreConfig, clone_optimizer, evaluate, train_retrieval
from .maintenance import realign_optimizer_state

_ids = itertools.count()


@dataclass
class InterventionLog:
    run_id: str
    records: list[dict[str, Any]] = field(default_factory=list)
    quarantined: set[int] = field(default_factory=set)

    def _slot(self, model, case_id: int) -> int:
        ids = model.active_case_ids().tolist()
        return ids.index(int(case_id))

    def _record(self, op: str, case_id: int, before: Any, after: Any, actor: str, reason: str) -> str:
        iid = f"{self.run_id}-i{next(_ids)}"
        self.records.append({"intervention_id": iid, "operation": op, "case_id": int(case_id), "before": before, "after": after, "actor": actor, "reason": reason})
        return iid

    @torch.no_grad()
    def relabel(self, model, case_id: int, new_label: int, *, actor: str, reason: str) -> str:
        s = self._slot(model, case_id)
        before = int(model.labels[s].argmax())
        model.labels[s].zero_()
        model.labels[s, int(new_label)] = 1.0
        model._rebuild_class_to_cases()
        return self._record("relabel", case_id, before, int(new_label), actor, reason)

    @torch.no_grad()
    def adjust_bias(self, model, case_id: int, delta: float, *, actor: str, reason: str) -> str:
        s = self._slot(model, case_id)
        before = float(model.biases[s])
        model.biases[s] += float(delta)
        return self._record("adjust_bias", case_id, before, float(model.biases[s]), actor, reason)

    def quarantine(self, model, case_id: int, *, actor: str, reason: str) -> str:
        self.quarantined.add(int(case_id))
        return self._record("quarantine", case_id, "active", "quarantined", actor, reason)

    def release(self, model, case_id: int, *, actor: str, reason: str) -> str:
        self.quarantined.discard(int(case_id))
        return self._record("release_quarantine", case_id, "quarantined", "active", actor, reason)

    def case_mask(self, model) -> torch.Tensor | None:
        """Quarantined cases are ineligible for retrieval regardless of match (contract 12)."""
        if not self.quarantined:
            return None
        ids = model.active_case_ids().tolist()
        return torch.tensor([cid not in self.quarantined for cid in ids], dtype=torch.bool, device=model.cases.device)


def flip_matrix(pred_a: torch.Tensor, pred_b: torch.Tensor, y: torch.Tensor) -> dict[str, int]:
    a, b = pred_a == y, pred_b == y
    return {
        "wrong_to_correct": int((~a & b).sum()),
        "correct_to_wrong": int((a & ~b).sum()),
        "wrong_to_other_wrong": int((~a & ~b & (pred_a != pred_b)).sum()),
        "unchanged": int((pred_a == pred_b).sum()),
    }


def evaluate_m0_m1_m2(
    model,
    data,
    core_cfg: CoreConfig,
    edit: Callable[[Any, InterventionLog], list[int]],
    log: InterventionLog,
    *,
    retrain_epochs: int,
    influenced_mask: Callable[[Any, list[int]], torch.Tensor] | None = None,
    y_train: torch.Tensor | None = None,
    core_optimizer: torch.optim.Optimizer | None = None,
    retrain_lr_scale: float = 1.0,
) -> dict[str, Any]:
    """M0 = saved trained model; M1 = after ``edit`` without gradients; M2 = M1 + fixed retraining budget.

    ``edit(model, log)`` applies the intervention and returns the edited case_ids.
    Targeted/collateral effects use the test queries whose top-k retrieval included
    an edited case at M0 (``influenced``) versus the rest.
    """
    m0 = copy.deepcopy(model)
    e0 = evaluate(m0, data.X_test, data.y_test)
    m1 = copy.deepcopy(model)
    edited = edit(m1, log)
    mask = log.case_mask(m1)
    e1 = evaluate(m1, data.X_test, data.y_test, case_mask=mask)
    m2 = copy.deepcopy(m1)
    opt2 = clone_optimizer(core_optimizer, m2, core_cfg) if core_optimizer is not None else None
    if mask is not None:
        # quarantined cases leave the active set before retraining (reversible via log/archive)
        keep = torch.nonzero(mask).view(-1)
        n_old = m2.case_count()
        m2.compact_cases(keep)
        if opt2 is not None:
            realign_optimizer_state(opt2, m2, keep.cpu().numpy(), n_old)
    if retrain_epochs > 0:
        # fixed, reported budget; corrected cases are also training queries, so retrain on corrected targets
        ft_cfg = copy.copy(core_cfg)
        ft_cfg.patience = retrain_epochs + 1
        train_retrieval(m2, data.X_train, data.y_train if y_train is None else y_train, data.X_val, data.y_val, ft_cfg, epochs=retrain_epochs, optimizer=opt2, select_best=False, lr_scale=retrain_lr_scale)
    e2 = evaluate(m2, data.X_test, data.y_test)
    y = data.y_test
    out: dict[str, Any] = {"edited_case_ids": edited, "retrain_epochs": retrain_epochs}
    for name, e in (("M0", e0), ("M1", e1), ("M2", e2)):
        out[name] = {k: v for k, v in e.items() if not k.startswith(("pred", "prob"))}
    if model.task_type == "classification":
        out["flips_M0_M1"] = flip_matrix(e0["pred_pre"], e1["pred_pre"], y)
        out["flips_M0_M2"] = flip_matrix(e0["pred_pre"], e2["pred_pre"], y)
        if influenced_mask is not None:
            infl = influenced_mask(m0, edited).cpu()
            for tag, m in (("influenced", infl), ("collateral", ~infl)):
                if m.any():
                    out[f"{tag}_flips_M0_M1"] = flip_matrix(e0["pred_pre"][m], e1["pred_pre"][m], y[m])
                    out[f"{tag}_n"] = int(m.sum())
    return out


@torch.no_grad()
def influenced_by(model, X: torch.Tensor, case_ids: list[int]) -> torch.Tensor:
    r = model.retrieve(X.to(model.cases.device), exclude_identical=False)
    ids = model.active_case_ids()[r["case_indices"]]
    hit = torch.isin(ids, torch.tensor(case_ids, device=ids.device))
    return ((r["weights"] > 0) & hit.unsqueeze(0)).any(1)


def flag_ranking(method: str, scores, rng: np.random.Generator, truth: np.ndarray | None = None) -> np.ndarray:
    """Return active-slot indices ordered most-suspicious first."""
    n = len(scores.case_ids)
    tie = scores.case_ids
    if method == "provenance_only":
        key = np.where(scores.evidenced, scores.Q, 0.5)
    elif method == "bias_only":
        key = scores.B
    elif method == "combined_T":
        key = scores.T
    elif method == "random":
        key = rng.random(n)
    elif method == "oracle":
        if truth is None:
            raise ValueError("oracle flagging needs ground truth")
        key = np.where(truth, 0.0, 1.0)
    else:
        raise ValueError(method)
    return np.lexsort((tie, key))


def flagging_ablation(
    model,
    data,
    scores,
    core_cfg: CoreConfig,
    *,
    truth_corrupted: np.ndarray,
    true_labels: np.ndarray,
    budgets: list[int],
    methods: list[str],
    retrain_epochs: int,
    run_id: str,
    seed: int,
    core_optimizer: torch.optim.Optimizer | None = None,
    retrain_lr_scale: float = 1.0,
) -> list[dict[str, Any]]:
    """Simulated review: the reviewer inspects the top-b flagged cases and repairs the corrupted ones."""
    rng = np.random.default_rng(seed)
    ids = scores.case_ids
    if model.case_count() != len(data.y_train) or not np.array_equal(ids, np.arange(len(data.y_train))):
        raise ValueError("flagging_ablation expects the full-memory model whose slots equal training rows")
    rows = []
    for method in methods:
        order = flag_ranking(method, scores, rng, truth_corrupted)
        for b in budgets:
            inspected = order[:b]
            hits = inspected[truth_corrupted[inspected]]
            log = InterventionLog(run_id)

            def edit(m, lg, hits=hits):
                for slot in hits:
                    lg.relabel(m, int(ids[slot]), int(true_labels[slot]), actor="simulated_oracle_reviewer", reason=f"flagged_by_{method}")
                return [int(ids[s]) for s in hits]

            y_fixed = data.y_train.clone()
            if len(hits):
                y_fixed[torch.as_tensor(hits)] = torch.as_tensor(true_labels[hits], dtype=y_fixed.dtype)
            res = evaluate_m0_m1_m2(
                model, data, core_cfg, edit, log, retrain_epochs=retrain_epochs,
                influenced_mask=lambda m, e: influenced_by(m, data.X_test, e), y_train=y_fixed,
                core_optimizer=core_optimizer, retrain_lr_scale=retrain_lr_scale,
            )
            n_bad = int(truth_corrupted.sum())
            rows.append(
                {
                    "method": method,
                    "review_budget": b,
                    "precision_at_budget": float(truth_corrupted[inspected].mean()) if b else 0.0,
                    "recall_at_budget": float(truth_corrupted[inspected].sum() / max(n_bad, 1)),
                    "cases_inspected": int(b),
                    "corrections": len(hits),
                    **{f"{k}": v for k, v in res.items() if k != "edited_case_ids"},
                    "interventions": log.records,
                }
            )
    return rows
