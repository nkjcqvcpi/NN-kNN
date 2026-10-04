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
  provenance-only, bias-only, random (matched budget) and oracle
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
from .maintenance import ArchivedCase, CaseArchive, realign_optimizer_state
from .outcomes import final_prediction, frozen_evaluation

_ids = itertools.count()


@dataclass
class InterventionLog:
    run_id: str
    records: list[dict[str, Any]] = field(default_factory=list)
    quarantined: set[int] = field(default_factory=set)
    protected: set[int] = field(default_factory=set)
    archive: CaseArchive = field(default_factory=CaseArchive)
    decisions: list[dict[str, Any]] = field(default_factory=list)

    def _slot(self, model, case_id: int) -> int:
        ids = model.active_case_ids().tolist()
        return ids.index(int(case_id))

    def _record(self, op: str, case_id: int, before: Any, after: Any, actor: str, reason: str) -> str:
        iid = f"{self.run_id}-i{next(_ids)}"
        existing_ids = {r["intervention_id"] for r in self.records}
        while iid in existing_ids:
            iid = f"{self.run_id}-i{next(_ids)}"
        self.records.append({"schema_version": "reviewer_intervention/v2", "run_id": self.run_id,
                             "version": len(self.records) + 1, "intervention_id": iid,
                             "parent_intervention_id": self.records[-1]["intervention_id"] if self.records else None,
                             "operation": op, "case_id": None if case_id is None else int(case_id),
                             "before": before, "after": after, "actor": actor, "reason": reason})
        return iid

    @torch.no_grad()
    def relabel(self, model, case_id: int, new_label: int, *, actor: str, reason: str) -> str:
        s = self._slot(model, case_id)
        if model.task_type != "classification" or not 0 <= int(new_label) < model.labels.shape[1] or int(new_label) != new_label:
            raise ValueError("relabel requires a valid classification label")
        before = int(model.labels[s].argmax())
        model.labels[s].zero_()
        model.labels[s, int(new_label)] = 1.0
        model._rebuild_class_to_cases()
        return self._record("relabel", case_id, before, int(new_label), actor, reason)

    @torch.no_grad()
    def adjust_bias(self, model, case_id: int, delta: float, *, actor: str, reason: str) -> str:
        s = self._slot(model, case_id)
        if not np.isfinite(delta):
            raise ValueError("Bias delta must be finite")
        before = float(model.biases[s])
        if not torch.isfinite(model.biases[s] + float(delta)):
            raise ValueError("Bias edit would overflow")
        model.biases[s] += float(delta)
        return self._record("adjust_bias", case_id, before, float(model.biases[s]), actor, reason)

    def quarantine(self, model, case_id: int, *, actor: str, reason: str) -> str:
        self._slot(model, case_id)
        before = "quarantined" if int(case_id) in self.quarantined else "active"
        self.quarantined.add(int(case_id))
        return self._record("quarantine", case_id, before, "quarantined", actor, reason)

    def release(self, model, case_id: int, *, actor: str, reason: str) -> str:
        self._slot(model, case_id)
        before = "quarantined" if int(case_id) in self.quarantined else "active"
        self.quarantined.discard(int(case_id))
        return self._record("release_quarantine", case_id, before, "active", actor, reason)

    def protect(self, model, case_id: int, store, *, actor: str, reason: str) -> str:
        self._slot(model, case_id)
        st = store.ensure(case_id)
        before = {"protected": st.protected, "reason": st.protected_reason}
        self.protected.add(int(case_id))
        st.protected, st.protected_reason = True, reason
        return self._record("protect", case_id, before, {"protected": True, "reason": reason}, actor, reason)

    def unprotect(self, model, case_id: int, store, *, actor: str, reason: str) -> str:
        self._slot(model, case_id)
        st = store.ensure(case_id)
        before = {"protected": st.protected, "reason": st.protected_reason}
        self.protected.discard(int(case_id))
        st.protected, st.protected_reason = False, None
        return self._record("unprotect", case_id, before, {"protected": False, "reason": None}, actor, reason)

    @torch.no_grad()
    def set_feature_weights(self, model, weights, *, actor: str, reason: str) -> str:
        if model.glocal_weightor is None:
            raise ValueError("Model has no learned feature-weight parameter")
        p = model.glocal_weightor.feature_weights
        value = torch.as_tensor(weights, device=p.device, dtype=p.dtype)
        if value.shape != p.shape or not torch.isfinite(value).all():
            raise ValueError("Feature weights must be finite and match the learned parameter shape")
        before = p.detach().cpu().tolist()
        p.copy_(value)
        iid = self._record("set_feature_weights", None, before, p.cpu().tolist(), actor, reason)
        self.records[-1]["parameterization"] = "raw glocal_weightor.feature_weights; distance uses leaky_relu(raw, 0.001)"
        return iid

    @torch.no_grad()
    def set_case_weights(self, model, case_id: int, weights, *, actor: str, reason: str) -> str:
        p = model.glocal_weights[self._slot(model, case_id)]
        value = torch.as_tensor(weights, device=p.device, dtype=p.dtype)
        if value.shape != p.shape or not torch.isfinite(value).all():
            raise ValueError("Case weights must be finite and match the glocal mixture shape")
        before = p.detach().cpu().tolist()
        p.copy_(value)
        return self._record("set_case_weights", case_id, before, p.cpu().tolist(), actor, reason)

    @torch.no_grad()
    def undo_parameter_edit(self, model, intervention_id: str, *, actor: str, reason: str) -> str:
        record = next((r for r in self.records if r["intervention_id"] == intervention_id), None)
        if record is None:
            raise ValueError("Unknown intervention ID")
        op, cid = record["operation"], record["case_id"]
        if op == "set_feature_weights":
            p = model.glocal_weightor.feature_weights
        elif op == "set_case_weights":
            p = model.glocal_weights[self._slot(model, cid)]
        elif op == "adjust_bias":
            p = model.biases[self._slot(model, cid)]
        elif op == "relabel":
            p = model.labels[self._slot(model, cid)]
            if int(p.argmax()) != record["after"]:
                raise ValueError("Edit is stale; restore a full checkpoint after intervening changes")
            iid = self.relabel(model, cid, record["before"], actor=actor, reason=reason)
            self.records[-1]["undo_of"] = intervention_id
            return iid
        else:
            raise ValueError("Use explicit release/unprotect/restore for eligibility edits")
        after = torch.as_tensor(record["after"], device=p.device, dtype=p.dtype)
        if not torch.equal(p, after):
            raise ValueError("Edit is stale; restore a full checkpoint after intervening changes")
        p.copy_(torch.as_tensor(record["before"], device=p.device, dtype=p.dtype))
        iid = self._record("undo_" + op, cid, record["after"], record["before"], actor, reason)
        self.records[-1]["undo_of"] = intervention_id
        return iid

    def archive_remove(self, model, case_id: int, store, *, actor: str, reason: str, optimizer=None, step=0) -> str:
        slot = self._slot(model, case_id)
        if int(case_id) in self.protected or store.ensure(case_id).protected:
            raise ValueError("Unprotect a case explicitly before archive removal")
        if model.case_count() <= 1:
            raise ValueError("Cannot remove the last active case")
        entry = self.archive.add(model, slot, store.get(case_id).to_dict(), step, [reason],
                                 status="quarantined" if int(case_id) in self.quarantined else "archived")
        old_count = model.case_count()
        keep = np.delete(np.arange(old_count), slot)
        model.compact_cases(torch.as_tensor(keep))
        if optimizer is not None:
            realign_optimizer_state(optimizer, model, keep, old_count)
        iid = self._record("archive_remove", case_id, "active", entry.status, actor, reason)
        self.records[-1]["restoration"] = {"archive_key": int(case_id), "optimizer": "restored rows have zero moments"}
        return iid

    def restore(self, model, case_id: int, *, actor: str, reason: str, optimizer=None) -> str:
        entry = self.archive.entries[int(case_id)]
        before = entry.status
        self.archive.restore(model, [case_id], optimizer)
        # Quarantine remains in force after restoration until explicit release.
        if before == "quarantined":
            self.quarantined.add(int(case_id))
        return self._record("restore", case_id, before, "active_quarantined" if int(case_id) in self.quarantined else "active", actor, reason)

    def controlled_decision(self, model, X, case_ids, *, query_ids, activation_floor, actor, reason, adapter=None, query_case_ids=None):
        if len(query_ids) != len(X):
            raise ValueError("query_ids must identify every controlled query")
        if model.task_type == "regression" and model.nn_cdh is not None and adapter is None:
            raise ValueError("Controlled regression decisions require an external aggregate adapter or retrieval-only model")
        with frozen_evaluation(model, adapter):
            xb = X.to(model.cases.device)
            r = model.retrieve(xb, exclude_identical=False, case_mask=self.case_mask(model),
                               query_case_ids=query_case_ids, force_case_ids=case_ids,
                               forced_weight_floor=activation_floor)
            pre, final, kind = final_prediction(model, xb, r, adapter=adapter)
            iid = self._record("force_eligible_cases", None, None, r["override"], actor, reason)
            for b, qid in enumerate(query_ids):
                slots = r["case_indices"][r["weights"][b] > 0]
                w = r["weights"][b, r["weights"][b] > 0]
                self.decisions.append({"schema_version": "controlled_retrieval/v1", "run_id": self.run_id,
                                       "retrieval_event_id": f"{iid}-q{b}", "intervention_id": iid, "query_id": qid,
                                       "override": r["override"], "case_ids": model.case_ids[slots].tolist(),
                                       "activations": w.cpu().tolist(), "distances": r["distances"][b, r["weights"][b] > 0].cpu().tolist(),
                                       "case_biases": model.biases[slots].cpu().tolist(),
                                       "pre_prediction": pre[b].cpu().tolist(), "final_prediction": final[b].cpu().tolist(),
                                       "loss_kind": kind, "eligibility_gates": ["active", "quarantine", "self_id", "top_k"],
                                       "actor": actor, "reason": reason})
            return pre.detach(), final.detach(), iid

    def case_mask(self, model) -> torch.Tensor | None:
        """Quarantined cases are ineligible for retrieval regardless of match (contract 12)."""
        if not self.quarantined:
            return None
        ids = model.active_case_ids().tolist()
        return torch.tensor([cid not in self.quarantined for cid in ids], dtype=torch.bool, device=model.cases.device)

    def state_dict(self):
        return {"run_id": self.run_id, "records": copy.deepcopy(self.records),
                "decisions": copy.deepcopy(self.decisions), "quarantined": sorted(self.quarantined),
                "protected": sorted(self.protected), "archive": self.archive.state_dict()}

    @classmethod
    def from_state_dict(cls, state):
        log = cls(state["run_id"], records=copy.deepcopy(state["records"]),
                  quarantined=set(state["quarantined"]), protected=set(state["protected"]),
                  decisions=copy.deepcopy(state.get("decisions", [])))
        for cid, entry in state["archive"].items():
            log.archive.entries[int(cid)] = ArchivedCase(int(cid), **copy.deepcopy(entry))
        return log


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
    capture_artifacts: bool = False,
    matched_training_control: bool = False,
) -> dict[str, Any]:
    """M0 = saved trained model; M1 = after ``edit`` without gradients; M2 = M1 + fixed retraining budget.

    ``edit(model, log)`` applies the intervention and returns the edited case_ids.
    Targeted/collateral effects use the test queries whose top-k retrieval included
    an edited case at M0 (``influenced``) versus the rest.
    """
    if retrain_epochs < 0:
        raise ValueError("retrain_epochs must be nonnegative")
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
    retrain_history = []
    control, control_optimizer, control_history = None, None, []
    if retrain_epochs > 0:
        # fixed, reported budget; corrected cases are also training queries, so retrain on corrected targets
        ft_cfg = copy.copy(core_cfg)
        ft_cfg.patience = retrain_epochs + 1
        device = m2.cases.device
        device_type = device.type if device.type in {"cuda", "xpu"} else "cuda"
        devices = [device.index or 0] if device.type in {"cuda", "xpu"} else []
        # Independent forks preserve the common RNG stream, including dropout.
        with torch.random.fork_rng(devices=devices, device_type=device_type):
            tr2 = train_retrieval(m2, data.X_train, data.y_train if y_train is None else y_train, data.X_val, data.y_val, ft_cfg, epochs=retrain_epochs, optimizer=opt2, select_best=False, lr_scale=retrain_lr_scale)
        opt2, retrain_history = tr2.optimizer, tr2.history
        if matched_training_control:
            control = copy.deepcopy(m0)
            control_optimizer = clone_optimizer(core_optimizer, control, core_cfg) if core_optimizer is not None else None
            with torch.random.fork_rng(devices=devices, device_type=device_type):
                ctr = train_retrieval(control, data.X_train, data.y_train, data.X_val, data.y_val, ft_cfg,
                                      epochs=retrain_epochs, optimizer=control_optimizer, select_best=False,
                                      lr_scale=retrain_lr_scale)
            control_optimizer, control_history = ctr.optimizer, ctr.history
    e2 = evaluate(m2, data.X_test, data.y_test)
    y = data.y_test
    out: dict[str, Any] = {"edited_case_ids": edited, "retrain_epochs": retrain_epochs}
    if capture_artifacts:
        # Keep tensors/models outside the public metrics JSON. Each saved stage
        # represents exactly the predictor used for that stage's evaluation.
        def snapshot(m, opt, stage_mask=None):
            return {"model_state": copy.deepcopy(m.state_dict()),
                    "active_case_count": m.case_count(),
                    "optimizer_state": None if opt is None else copy.deepcopy(opt.state_dict()),
                    "requires_grad": {n: p.requires_grad for n, p in m.named_parameters()},
                    "case_mask": None if stage_mask is None else stage_mask.detach().cpu().clone()}
        out["_artifacts"] = {"final_model": m2, "final_optimizer": opt2,
                             "history": retrain_history,
                             "stages": {"M0": snapshot(m0, core_optimizer),
                                        "M1": snapshot(m1, core_optimizer, mask),
                                        "M2": snapshot(m2, opt2)},
                             "y_train_corrected": (data.y_train if y_train is None else y_train).detach().cpu().clone()}
        out["_artifacts"]["intervention_log"] = log.state_dict()
        if control is not None:
            out["_artifacts"]["stages"]["MC"] = snapshot(control, control_optimizer)
            out["_artifacts"]["control_history"] = control_history
    for name, e in (("M0", e0), ("M1", e1), ("M2", e2)):
        out[name] = {k: v for k, v in e.items() if not k.startswith(("pred", "prob"))}
    if control is not None:
        ec = evaluate(control, data.X_test, data.y_test)
        out["MC"] = {k: v for k, v in ec.items() if not k.startswith(("pred", "prob"))}
        out["matched_control_epochs"] = retrain_epochs
        out["M2_minus_MC_loss"] = e2["loss_pre"] - ec["loss_pre"]
        if model.task_type == "classification":
            out["flips_MC_M2"] = flip_matrix(ec["pred_pre"], e2["pred_pre"], y)
    if model.task_type == "classification":
        out["flips_M0_M1"] = flip_matrix(e0["pred_pre"], e1["pred_pre"], y)
        out["flips_M0_M2"] = flip_matrix(e0["pred_pre"], e2["pred_pre"], y)
        if influenced_mask is not None:
            infl = influenced_mask(m0, edited).cpu()
            for tag, m in (("influenced", infl), ("collateral", ~infl)):
                if m.any():
                    out[f"{tag}_flips_M0_M1"] = flip_matrix(e0["pred_pre"][m], e1["pred_pre"][m], y[m])
                    out[f"{tag}_flips_M0_M2"] = flip_matrix(e0["pred_pre"][m], e2["pred_pre"][m], y[m])
                    if control is not None:
                        out[f"{tag}_flips_MC_M2"] = flip_matrix(ec["pred_pre"][m], e2["pred_pre"][m], y[m])
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
        raise ValueError("combined_T was superseded by the PI September 20 Q/B alternatives")
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
    capture_artifacts: bool = False,
    matched_training_control: bool = False,
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
            log = InterventionLog(f"{run_id}-{method}-b{b}")

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
                capture_artifacts=capture_artifacts,
                matched_training_control=matched_training_control,
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
