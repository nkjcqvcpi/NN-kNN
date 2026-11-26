"""C/H reliability as evidence accumulates (handoff 2026-11-26 pack p2; opt-in ``reliability`` key).

Open checklist item: "Evaluate whether C/H guides maintenance more reliably as queries accumulate". Exploratory design,
our choice, not a PI decision: on the frozen trained core, C, H and Q (and the provenance_only rank, Q where evidenced
and 0.5 elsewhere) are recomputed by the ordinary audit (``audit_provenance``) on nested random subsets of the
maintenance reference queries (one permutation per repeat; the first ceil(f |D|) queries for each fraction f), and
compared with the same statistics on the full reference set: Spearman rank correlation (average ranks; undefined when
either vector is constant) and the overlap of the top and bottom ``decile`` of cases (exact ties ordered by the
retention tie key).
"""

from __future__ import annotations

import math

import numpy as np

from .provenance import CaseStatisticsStore, active_cohorts, audit_provenance, score_cases

RELIABILITY_KEYS = frozenset({"fractions", "repeats", "decile"})
STATISTICS = ("C", "H", "Q", "Qe")


def check_reliability_cfg(rel):
    unknown = sorted(set(rel) - RELIABILITY_KEYS)
    if unknown or not RELIABILITY_KEYS <= set(rel):
        raise ValueError(f"reliability takes exactly {sorted(RELIABILITY_KEYS)}; got {sorted(rel)}")
    fr = [float(f) for f in rel["fractions"]]
    if not fr or any(not 0 < f <= 1 for f in fr) or sorted(set(fr)) != fr or fr[-1] != 1.0:
        raise ValueError("reliability.fractions: increasing values in (0, 1] ending with 1.0")
    if int(rel["repeats"]) < 1 or not 0 < float(rel["decile"]) <= 0.5:
        raise ValueError("reliability.repeats must be positive and decile in (0, 0.5]")


def average_ranks(v):
    v = np.asarray(v, dtype=float)
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(len(v))
    ranks[order] = np.arange(1, len(v) + 1)
    _, inv = np.unique(v, return_inverse=True)
    return (np.bincount(inv, weights=ranks) / np.bincount(inv))[inv]


def spearman(a, b):
    ra, rb = average_ranks(a), average_ranks(b)
    if len(ra) < 2 or ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def extreme_overlap(full, sub, tie, frac, top):
    m = max(1, math.ceil(frac * len(full)))
    sign = -1.0 if top else 1.0
    a = set(np.lexsort((tie, sign * np.asarray(full, dtype=float)))[:m].tolist())
    b = set(np.lexsort((tie, sign * np.asarray(sub, dtype=float)))[:m].tolist())
    return len(a & b) / m


def _statistics(model, reference, rows, score_cfg, reg_bins, counterfactual):
    store = CaseStatisticsStore(model.task_type)
    qids = None if reference.query_case_ids is None else reference.query_case_ids[rows]
    nominal = None if reference.query_nominal is None else reference.query_nominal[rows]
    audit_provenance(model, reference.X[rows], reference.y[rows], store,
                     exclude_identical=reference.stream == "train_loo", counterfactual=counterfactual,
                     reg_bins=reg_bins, query_case_ids=qids, adapter=reference.adapter,
                     regression_success_tolerance=reference.regression_success_tolerance, query_nominal=nominal)
    n = model.case_count()
    sc = score_cases(model.active_case_ids().detach().cpu().numpy().copy(), active_cohorts(model, reg_bins),
                     model.biases[:n].detach().cpu().numpy(), store, score_cfg)
    stats = {"C": sc.C, "H": sc.H, "Q": sc.Q, "Qe": np.where(sc.evidenced, sc.Q, 0.5)}
    return stats, float(sc.evidenced.mean()) if n else 0.0, store


def ch_reliability(model, reference, score_cfg, rel, *, seed, tie, reg_bins=None, counterfactual=False):
    """``(metrics, details, full_store)``; metrics ``rel_f<pct>_<stat>_<C|H|Q|Qe>`` are means over repeats."""
    check_reliability_cfg(rel)
    reference.validate()
    import torch
    nq = len(reference.y)
    everything = torch.arange(nq)
    full, full_evidenced, full_store = _statistics(model, reference, everything, score_cfg, reg_bins, counterfactual)
    rng = np.random.default_rng(seed)
    fractions = [float(f) for f in rel["fractions"]]
    decile = float(rel["decile"])
    per = {f: [] for f in fractions}
    for r in range(int(rel["repeats"])):
        perm = rng.permutation(nq)
        for f in fractions:
            m = max(1, math.ceil(f * nq))
            if m >= nq:
                sub, evid = full, full_evidenced
            else:
                rows = torch.as_tensor(np.sort(perm[:m]))
                sub, evid = _statistics(model, reference, rows, score_cfg, reg_bins, counterfactual)[:2]
            rec = {"repeat": r, "n_queries": m, "evidenced_fraction": evid}
            for s in STATISTICS:
                rec[f"spearman_{s}"] = spearman(full[s], sub[s])
                rec[f"top_overlap_{s}"] = extreme_overlap(full[s], sub[s], tie, decile, True)
                rec[f"bottom_overlap_{s}"] = extreme_overlap(full[s], sub[s], tie, decile, False)
            per[f].append(rec)
    metrics = {"rel_n_reference_queries": nq, "rel_n_cases": int(model.case_count()), "rel_repeats": int(rel["repeats"]),
               "rel_decile": decile}
    for f, recs in per.items():
        tag = f"rel_f{round(100 * f)}"
        metrics[f"{tag}_n_queries"] = recs[0]["n_queries"]
        metrics[f"{tag}_evidenced_fraction"] = float(np.mean([x["evidenced_fraction"] for x in recs]))
        for key in recs[0]:
            if key in ("repeat", "n_queries", "evidenced_fraction"):
                continue
            vals = [x[key] for x in recs if x[key] is not None]
            metrics[f"{tag}_{key}"] = float(np.mean(vals)) if vals else None
            if key.startswith("spearman"):
                metrics[f"{tag}_{key}_sd"] = float(np.std(vals)) if vals else None
                metrics[f"{tag}_{key}_undefined"] = len(recs) - len(vals)
    details = {"design": "nested random subsets of the maintenance reference queries, one permutation per repeat "
                         "(numpy default_rng(run seed)); statistics recomputed by audit_provenance on each subset",
               "fractions": fractions, "repeats": int(rel["repeats"]), "decile": decile,
               "stream": reference.stream, "per_repeat": {str(f): v for f, v in per.items()}}
    return metrics, details, full_store
