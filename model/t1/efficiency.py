"""Retrieval latency and case-memory size of one model (handoff 2026-11-26 pack p2; opt-in ``efficiency`` key).

Lookup = retrieval plus the reuse prediction (``final_prediction``) of the frozen model, timed with
``time.perf_counter`` after one untimed warm-up call. Wall times on a shared host are noisy; medians over repeated
lookups are recorded. Memory = bytes of the active cases' stored buffers and per-case parameters (the state
``CaseArchive`` saves per case), not process memory.
"""

from __future__ import annotations

import time

import numpy as np

from .maintenance import PER_CASE_BUFFERS, PER_CASE_PARAMS
from .outcomes import final_prediction, frozen_evaluation

EFFICIENCY_KEYS = frozenset({"lookup_repeats", "single_queries"})


def check_efficiency_cfg(eff):
    unknown = sorted(set(eff) - EFFICIENCY_KEYS)
    if unknown or not EFFICIENCY_KEYS <= set(eff):
        raise ValueError(f"efficiency takes exactly {sorted(EFFICIENCY_KEYS)}; got {sorted(eff)}")
    if int(eff["lookup_repeats"]) < 1 or int(eff["single_queries"]) < 1:
        raise ValueError("efficiency.lookup_repeats and single_queries must be positive")


def case_memory_bytes(model) -> int:
    n = model.case_count()
    total = 0
    for name in PER_CASE_BUFFERS + PER_CASE_PARAMS:
        t = getattr(model, name, None)
        if t is not None:
            total += int(t[:n].numel()) * int(t.element_size())
    return total


def lookup_latency(model, X, *, repeats, single_queries):
    """Median wall seconds of one batch lookup of all of ``X`` and of single-query lookups of its first rows."""
    device = model.cases.device
    X = X.to(device)
    single = X[:min(int(single_queries), len(X))]

    def lookup(xb):
        final_prediction(model, xb, model.retrieve(xb, exclude_identical=False))

    batch_t, single_t = [], []
    with frozen_evaluation(model):
        lookup(X[:1])
        for _ in range(int(repeats)):
            t0 = time.perf_counter()
            lookup(X)
            batch_t.append(time.perf_counter() - t0)
            for i in range(len(single)):
                t0 = time.perf_counter()
                lookup(single[i:i + 1])
                single_t.append(time.perf_counter() - t0)
    return {"lookup_batch_median_seconds": float(np.median(batch_t)),
            "lookup_batch_queries": int(len(X)),
            "lookup_single_median_seconds": float(np.median(single_t)),
            "lookup_single_p90_seconds": float(np.quantile(single_t, 0.9)),
            "lookup_single_count": int(len(single_t))}


def efficiency_metrics(model, base, X, eff, *, audit_seconds, maintenance_seconds):
    """``eff_*`` scalars of one retention unit: maintenance overhead, lookup latency and case memory of the
    maintained model and of the full-memory core it came from (measured in the same unit, back to back)."""
    check_efficiency_cfg(eff)
    kw = dict(repeats=eff["lookup_repeats"], single_queries=eff["single_queries"])
    out = {"eff_maintenance_audit_seconds": float(audit_seconds),
           "eff_maintenance_select_seconds": float(maintenance_seconds),
           "eff_maintenance_overhead_seconds": float(audit_seconds + maintenance_seconds)}
    out.update({f"eff_{k}": v for k, v in lookup_latency(model, X, **kw).items()})
    out.update({f"eff_fullmem_{k}": v for k, v in lookup_latency(base, X, **kw).items()})
    mem, full = case_memory_bytes(model), case_memory_bytes(base)
    out.update(eff_case_memory_bytes=mem, eff_fullmem_case_memory_bytes=full, eff_memory_saved_bytes=full - mem,
               eff_memory_saved_fraction=(full - mem) / full if full else 0.0,
               eff_lookup_single_speedup=out["eff_fullmem_lookup_single_median_seconds"]
               / max(out["eff_lookup_single_median_seconds"], 1e-12))
    return out
