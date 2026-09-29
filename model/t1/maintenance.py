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
) -> dict[str, Any]:
    n_before = model.case_count()
    ids = model.active_case_ids().detach().cpu().numpy()
    cohorts = active_cohorts(model, reg_bins)
    biases = model.biases[:n_before].detach().cpu().numpy()
    scores = score_cases(ids, cohorts, biases, store, score_cfg)
    dist = case_case_distance(model).cpu().numpy() if compute_distance else None
    retention_cfg.notes.setdefault("alpha", score_cfg.alpha)
    retention_cfg.notes.setdefault("trust_mode", score_cfg.trust_mode)
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
                "policy_version": f"{retention_cfg.policy}/v1",
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
