"""Reversible case-base maintenance at declared safe checkpoints.

``run_maintenance`` = audit scores -> ``select_active_set`` -> archive evicted
cases (exact restorable state) -> compact the model -> realign optimizer state
for per-case parameters -> write one maintenance event per case.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .geometry import case_case_distance
from .provenance import CaseStatisticsStore, ScoreConfig, active_cohorts, score_cases
from .retention import RetentionConfig, select_active_set

PER_CASE_PARAMS = ("biases", "negative_weights", "glocal_weights")
PER_CASE_BUFFERS = ("cases", "labels", "case_ids")


@dataclass
class ArchivedCase:
    case_id: int
    state: dict[str, torch.Tensor]
    stats: dict[str, Any]
    archived_step: int
    reason_codes: list[str]
    status: str = "archived"  # archived | quarantined


@dataclass
class CaseArchive:
    entries: dict[int, ArchivedCase] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.entries)

    def add(self, model, slot: int, stats: dict[str, Any], step: int, reasons: list[str], status: str = "archived") -> ArchivedCase:
        state = {name: getattr(model, name)[slot].detach().cpu().clone() for name in PER_CASE_BUFFERS + PER_CASE_PARAMS}
        cid = int(state["case_ids"].item())
        entry = ArchivedCase(cid, state, dict(stats), step, list(reasons), status)
        self.entries[cid] = entry
        return entry

    def restore(self, model, case_ids: list[int], optimizer: torch.optim.Optimizer | None = None) -> list[int]:
        """Put archived cases back into free active slots with their exact saved state."""
        slots = []
        for cid in case_ids:
            entry = self.entries.pop(int(cid))
            slot = model.case_count()
            if slot >= model.case_capacity():
                self.entries[int(cid)] = entry
                raise RuntimeError("No free case slot to restore into; raise capacity or evict first.")
            with torch.no_grad():
                for name, val in entry.state.items():
                    getattr(model, name)[slot].copy_(val.to(getattr(model, name).device))
            model.set_active_case_count(slot + 1)
            if optimizer is not None:
                _zero_optimizer_rows(optimizer, model, [slot])
            slots.append(slot)
        return slots

    def state_dict(self) -> dict[str, Any]:
        return {
            cid: {"state": e.state, "stats": e.stats, "archived_step": e.archived_step, "reason_codes": e.reason_codes, "status": e.status}
            for cid, e in self.entries.items()
        }


def _param_objects(model) -> dict[str, torch.nn.Parameter]:
    return {name: getattr(model, name) for name in PER_CASE_PARAMS}


def realign_optimizer_state(optimizer: torch.optim.Optimizer, model, keep_slots: np.ndarray, old_count: int) -> None:
    """Move per-case optimizer moments with their cases after ``compact_cases``.

    Rows ``keep_slots`` move to ``[0, len(keep))``; freed rows are zeroed. Adam's
    ``step`` counter is shared per tensor and left unchanged.
    """
    keep = torch.as_tensor(np.asarray(keep_slots), dtype=torch.long)
    for p in _param_objects(model).values():
        st = optimizer.state.get(p)
        if not st:
            continue
        for key, buf in st.items():
            if not torch.is_tensor(buf) or buf.dim() == 0 or buf.shape[0] != p.shape[0]:
                continue
            k = keep.to(buf.device)
            moved = buf[k].clone()
            buf[: k.numel()] = moved
            buf[k.numel() : old_count] = 0


def _zero_optimizer_rows(optimizer, model, slots: list[int]) -> None:
    for p in _param_objects(model).values():
        st = optimizer.state.get(p)
        if not st:
            continue
        for buf in st.values():
            if torch.is_tensor(buf) and buf.dim() > 0 and buf.shape[0] == p.shape[0]:
                buf[slots] = 0


_event_counter = itertools.count()


def run_maintenance(
    model,
    store: CaseStatisticsStore,
    archive: CaseArchive,
    K: int,
    retention_cfg: RetentionConfig,
    score_cfg: ScoreConfig,
    *,
    step: int,
    run_id: str,
    optimizer: torch.optim.Optimizer | None = None,
    reg_bins: np.ndarray | None = None,
    compute_distance: bool = True,
    reference=None,
) -> dict[str, Any]:
    retention_cfg.validate()
    if retention_cfg.policy.startswith("removal"):
        return _sequential_removal(model, store, archive, K, retention_cfg, score_cfg,
                                   step=step, run_id=run_id, optimizer=optimizer,
                                   reg_bins=reg_bins, reference=reference)
    n_before = model.case_count()
    ids = model.active_case_ids().detach().cpu().numpy()
    cohorts = active_cohorts(model, reg_bins)
    biases = model.biases[:n_before].detach().cpu().numpy()
    scores = score_cases(ids, cohorts, biases, store, score_cfg)
    dist = case_case_distance(model).cpu().numpy() if compute_distance else None
    if retention_cfg.policy == "coverage_reachability":
        from .candidates import coverage_reachability
        if reference is None:
            raise ValueError("Coverage/reachability needs designated maintenance queries")
        result = coverage_reachability(model, reference,
                                      activation_threshold=retention_cfg.activation_threshold,
                                      zero_reachability=retention_cfg.zero_reachability)
        scores.extra.update({k: v for k, v in result.items() if isinstance(v, np.ndarray)})
    sel = select_active_set(scores, K, retention_cfg, dist)
    keep = np.sort(sel.keep_slots)  # preserve relative slot order among kept cases
    slot_after = {int(s): i for i, s in enumerate(keep)}
    events = []
    for d in sel.decisions:
        slot = d["slot_before"]
        ev = dict(d)
        ev.update(
            {
                "maintenance_event_id": f"{run_id}-m{next(_event_counter)}",
                "run_id": run_id,
                "step": int(step),
                "policy_version": f"{retention_cfg.policy}/pi-20260920-v2",
                "slot_after": slot_after.get(slot),
                "capacity_before": int(n_before),
                "capacity_after": int(len(keep)),
                "restoration": None,
            }
        )
        if d["action"] == "archive":
            archive.add(model, slot, d, step, d["reason_codes"])
            ev["restoration"] = {"archive_key": int(d["case_id"])}
        events.append(ev)
    if len(keep) < n_before:
        model.compact_cases(torch.as_tensor(keep, dtype=torch.long))
        if optimizer is not None:
            realign_optimizer_state(optimizer, model, keep, n_before)
    return {"summary": sel.summary, "events": events, "kept_case_ids": ids[keep].tolist()}


def _sequential_removal(model, store, archive, K, cfg, score_cfg, *, step, run_id,
                        optimizer, reg_bins, reference):
    """Recompute after each removal; enforce loss against the original memory."""
    from .candidates import removal_influence
    from .retention import protection_requirements

    if reference is None:
        raise ValueError("Removal needs designated maintenance reference queries")
    if K < 1:
        raise ValueError("K must be positive")
    n_before = model.case_count()
    original_ids = model.active_case_ids().cpu().numpy().copy()
    scores = score_cases(original_ids, active_cohorts(model, reg_bins),
                         model.biases[:n_before].detach().cpu().numpy(), store, score_cfg)
    distances = case_case_distance(model).cpu().numpy() if cfg.boundary_protect_fraction > 0 else None
    protected, floors, reasons = protection_requirements(scores, cfg, distances)
    protected_ids = set(original_ids[protected].tolist())
    cohorts_original = np.asarray(scores.cohorts)
    required = sum(max(floors[c], int(protected[cohorts_original == c].sum())) for c in floors)
    if min(K, n_before) < required:
        raise ValueError("Capacity cannot satisfy protected cases and cohort floors")
    events_by_id = {}
    for j, cid in enumerate(original_ids):
        events_by_id[int(cid)] = {**scores.row(j), "slot_before": j, "protected": bool(protected[j]),
                                 "reason_codes": reasons[j], "action": "keep"}
    trace = []
    original_loss = None
    stop_reason = "capacity_reached"
    while model.case_count() > K:
        ids = model.active_case_ids().cpu().numpy()
        cohorts = np.asarray(active_cohorts(model, reg_bins))
        eligible = np.array([int(cid) not in protected_ids and int((cohorts == cohorts[j]).sum()) > floors[cohorts[j]]
                             for j, cid in enumerate(ids)])
        if not eligible.any():
            stop_reason = "protection_limit"
            break
        result = removal_influence(model, reference,
                                   cached_threshold=cfg.cache_activation_threshold if cfg.policy == "removal_cached" else None)
        if original_loss is None:
            original_loss = result["baseline_loss"]
        values = result["cached"] if cfg.policy == "removal_cached" else result["full"]
        pool = np.flatnonzero(eligible)
        j = int(pool[np.lexsort((ids[pool], values[pool]))[0]])
        cumulative = result["baseline_loss"] + float(result["full"][j]) - original_loss
        trace.append({"iteration": len(trace), "active_case_ids": ids.tolist(),
                      "scores": values.tolist(), "full_scores": result["full"].tolist(),
                      "cache_query_counts": result["cache_query_counts"].tolist(),
                      "cache_query_ids": result["cache_query_ids"],
                      "candidate_rerun_queries": result["candidate_rerun_queries"],
                      "validation_rerun_queries": result["validation_rerun_queries"],
                      "cache_max_absolute_error": result["cache_max_absolute_error"],
                      "candidate_case_id": int(ids[j]), "cumulative_loss_increase": cumulative,
                      "reference_denominator": result["n_reference_queries"],
                      "cache_refresh": "current_checkpoint_and_case_base"})
        if cumulative > cfg.allowed_loss_increase + 1e-10:
            stop_reason = "cumulative_loss_budget"
            break
        cid = int(ids[j])
        ev = events_by_id[cid]
        ev.update(action="archive", policy_score=float(values[j]), removal_iteration=len(trace) - 1,
                  full_removal_influence=float(result["full"][j]), cumulative_loss_increase=cumulative,
                  reason_codes=ev["reason_codes"] + ["lowest_eligible_removal_score"])
        archive.add(model, j, ev, step, ev["reason_codes"])
        keep = np.delete(np.arange(len(ids)), j)
        model.compact_cases(torch.as_tensor(keep))
        if optimizer is not None:
            realign_optimizer_state(optimizer, model, keep, len(ids))
    final_ids = model.active_case_ids().tolist()
    for cid, ev in events_by_id.items():
        ev.update(maintenance_event_id=f"{run_id}-m{next(_event_counter)}", run_id=run_id, step=int(step),
                  policy_version=f"{cfg.policy}/pi-20260920-v2", capacity_before=n_before,
                  capacity_after=len(final_ids), slot_after=final_ids.index(cid) if cid in final_ids else None,
                  restoration={"archive_key": cid} if ev["action"] == "archive" else None,
                  tie_break="case_id_ascending")
        if ev["action"] == "keep":
            ev["reason_codes"] += ["kept_" + stop_reason]
    return {"summary": {"policy": cfg.policy, "n_before": n_before, "K": K,
                        "n_after": len(final_ids), "capacity_reached": len(final_ids) <= K,
                        "stop_reason": stop_reason, "allowed_loss_increase": cfg.allowed_loss_increase,
                        "original_reference_loss": original_loss},
            "events": list(events_by_id.values()), "kept_case_ids": final_ids, "scoring_trace": trace}
