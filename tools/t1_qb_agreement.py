"""Q-B agreement of the recorded retention runs (read-only; nothing is trained, nothing under the run root is written).

    python tools/t1_qb_agreement.py [TREE ...] [--source-condition full_memory_K0.5]
        [--check-condition provenance_only_K0.5] [--seeds 30 31 32] [--out DIR]

TREE is a ``tools/t1_run.py`` retention run-unit tree (default: ``runs/t1/plan_full_r760/retention__classification``
and ``.../retention__regression``). Every ``retention`` condition unit records, in its ``case_maintenance`` rows, the
score that ``model.t1.provenance.score_cases`` gave each active case before maintenance: the smoothed outcome
quality ``Q = (C + s) / (C + H + 2s)``, the within-cohort bias percentile ``B`` and the evidence flag
(``R >= min_retrieval_count`` and ``A >= min_activation_mass``). Before maintenance these scores depend only on the
trained core and the step-0 audit, so all policy units of a (dataset, seed) carry the same values; the tool reads them
from ``--source-condition`` and, when ``--check-condition`` is recorded too, reports the largest |dQ| and |dB| between
the two as a consistency check (expected 0). The ``_core`` stage unit records no per-case scores.

Per (dataset, seed), over the evidenced cases only:
- ``spearman``: Spearman rank correlation of Q and B (ties get average ranks; NaN when fewer than 3 cases or a
  constant column);
- median split: ``high`` means strictly above the median of the evidenced cases of that run, ``low`` at or below it
  (Q often ties at the median, e.g. every case with H = 0 and the same C); counts and shares of the quadrants
  ``hiQ_hiB`` and ``loQ_loB`` (agreement), ``hiQ_loB`` (good outcomes, low bias) and ``loQ_hiB`` (harmful outcomes,
  high bias), plus ``agree_share`` = share of the two agreement quadrants.

Outputs (default ``<worktree>/results/t1/<study>/qb_agreement/``): ``qb_agreement_per_seed.csv``,
``qb_agreement_by_dataset.csv`` (mean over seeds) and ``qb_agreement.md``. Table 1 code and the existing summary
tools are not used or changed.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from model.common.tracking import TrackingError, load_run, run_status, runs_root  # noqa: E402
from model.t1.records import RESULTS, record as read_record, rows as read_rows  # noqa: E402

DEFAULT_STUDY = "plan_full_r760"
DEFAULT_EXPERIMENTS = ("retention__classification", "retention__regression")
QUADRANTS = ("hiQ_hiB", "loQ_loB", "hiQ_loB", "loQ_hiB")


def event_hosts(unit: Path) -> list[str]:
    """Host names in a unit's event file names (``events.out.tfevents.<time>.<host>.<pid>.<n>.s<k>``)."""
    hosts = set()
    for f in unit.glob("events.out.tfevents.*"):
        parts = f.name.split(".")
        if len(parts) >= 8:
            hosts.add(".".join(parts[4:-3]))
    return sorted(hosts)


def qb_stats(Q, B, evidenced) -> dict:
    """Spearman and median-split quadrants of Q and B over the evidenced cases."""
    Q, B = np.asarray(Q, dtype=float), np.asarray(B, dtype=float)
    ev = np.asarray(evidenced, dtype=bool)
    q, b = Q[ev], B[ev]
    n = int(ev.sum())
    out = {"n_cases": int(len(Q)), "n_evidenced": n}
    if n >= 3 and np.ptp(q) > 0 and np.ptp(b) > 0:
        out["spearman"] = float(stats.spearmanr(q, b)[0])
    else:
        out["spearman"] = math.nan
    if n == 0:
        out.update({f"n_{k}": 0 for k in QUADRANTS})
        out.update({f"share_{k}": math.nan for k in QUADRANTS})
        out.update(q_median=math.nan, b_median=math.nan, agree_share=math.nan)
        return out
    qm, bm = float(np.median(q)), float(np.median(b))
    hq, hb = q > qm, b > bm
    counts = {"hiQ_hiB": hq & hb, "loQ_loB": ~hq & ~hb, "hiQ_loB": hq & ~hb, "loQ_hiB": ~hq & hb}
    out.update(q_median=qm, b_median=bm)
    for k in QUADRANTS:
        out[f"n_{k}"] = int(counts[k].sum())
        out[f"share_{k}"] = out[f"n_{k}"] / n
    out["agree_share"] = out["share_hiQ_hiB"] + out["share_loQ_loB"]
    return out


def unit_scores(unit: Path):
    """(row, Q, B, evidenced, case_ids) of a complete condition unit, or None when it is not complete."""
    if run_status(unit).status != "complete":
        return None
    data = load_run(unit, arrays=False)
    row = read_record(data, "row")
    events = read_rows(data, "case_maintenance")
    if not events or "Q" not in events[0]:
        raise ValueError(f"{unit}: case_maintenance holds no Q/B scores")
    ids = np.array([int(e["case_id"]) for e in events])
    order = np.argsort(ids, kind="mergesort")
    pick = lambda k, t: np.array([e[k] for e in events], dtype=t)[order]  # noqa: E731
    return row, pick("Q", float), pick("B", float), pick("evidenced", bool), ids[order]


def analyse_tree(tree: Path, source: str, check: str | None, seeds) -> tuple[list[dict], list[str]]:
    out, notes = [], []
    for ds_dir in sorted(p for p in tree.iterdir() if p.is_dir()):
        src_dir = ds_dir / source
        if not src_dir.is_dir():
            notes.append(f"{tree.name}/{ds_dir.name}: no {source}")
            continue
        for seed_dir in sorted(src_dir.glob("s*")):
            try:
                seed = int(seed_dir.name[1:])
            except ValueError:
                continue
            if seeds and seed not in seeds:
                continue
            try:
                got = unit_scores(seed_dir)
            except (TrackingError, KeyError, ValueError) as e:
                notes.append(f"{tree.name}/{ds_dir.name}/{source}/{seed_dir.name}: {type(e).__name__}: {e}")
                continue
            if got is None:
                notes.append(f"{tree.name}/{ds_dir.name}/{source}/{seed_dir.name}: not complete")
                continue
            row, Q, B, ev, ids = got
            rec = {"experiment": tree.name, "dataset": ds_dir.name, "task_type": row.get("task_type") or "",
                   "seed": seed, "source_condition": source, "host": "+".join(event_hosts(seed_dir)),
                   **qb_stats(Q, B, ev)}
            rec["check_max_abs_dQ"] = rec["check_max_abs_dB"] = math.nan
            if check:
                chk = ds_dir / check / seed_dir.name
                try:
                    other = unit_scores(chk) if chk.is_dir() else None
                except (TrackingError, KeyError, ValueError) as e:
                    notes.append(f"{tree.name}/{ds_dir.name}/{check}/{seed_dir.name}: {type(e).__name__}: {e}")
                    other = None
                if other is not None:
                    if not np.array_equal(other[4], ids):
                        notes.append(f"{tree.name}/{ds_dir.name}/{seed_dir.name}: case ids differ from {check}")
                    else:
                        rec["check_max_abs_dQ"] = float(np.max(np.abs(other[1] - Q))) if len(Q) else 0.0
                        rec["check_max_abs_dB"] = float(np.max(np.abs(other[2] - B))) if len(B) else 0.0
            out.append(rec)
    return out, notes


def by_dataset(per_seed: pd.DataFrame) -> pd.DataFrame:
    cols = ["spearman", "n_evidenced", "n_cases", "agree_share", *[f"share_{k}" for k in QUADRANTS]]
    g = per_seed.groupby(["experiment", "dataset", "task_type"], sort=True)
    agg = g[cols].mean()
    agg["spearman_sd"] = g["spearman"].std(ddof=1)
    agg["spearman_min"] = g["spearman"].min()
    agg["spearman_max"] = g["spearman"].max()
    agg["n_seeds"] = g["seed"].nunique()
    agg["seeds"] = g["seed"].apply(lambda s: " ".join(str(int(x)) for x in sorted(s)))
    agg["host"] = g["host"].apply(lambda s: "+".join(sorted(set("+".join(s).split("+")) - {""})))
    agg["check_max_abs_dQ"] = g["check_max_abs_dQ"].max()
    agg["check_max_abs_dB"] = g["check_max_abs_dB"].max()
    return agg.reset_index()


def markdown(per_ds: pd.DataFrame, trees, source, check, notes) -> str:
    f = lambda x, d=3: "nan" if pd.isna(x) else f"{x:.{d}f}"  # noqa: E731
    lines = ["# Q-B agreement (evidenced cases, mean over seeds)", "",
             f"Run trees: {', '.join(str(t) for t in trees)}. Scores from `{source}`"
             + (f", checked against `{check}`." if check else "."), "",
             "| task | dataset | host | seeds | evidenced / cases | Spearman (sd) [min, max] | hiQ_hiB | loQ_loB | hiQ_loB "
             "| loQ_hiB | agree | max dQ, dB |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in per_ds.iterrows():
        lines.append(
            f"| {r.task_type} | {r.dataset} | {r.host} | {r.seeds} | {f(r.n_evidenced, 0)} / {f(r.n_cases, 0)} "
            f"| {f(r.spearman)} ({f(r.spearman_sd)}) [{f(r.spearman_min)}, {f(r.spearman_max)}] "
            f"| {f(r.share_hiQ_hiB)} | {f(r.share_loQ_loB)} | {f(r.share_hiQ_loB)} | {f(r.share_loQ_hiB)} "
            f"| {f(r.agree_share)} | {f(r.check_max_abs_dQ, 2)}, {f(r.check_max_abs_dB, 2)} |")
    lines += ["", "Quadrants: median split over the evidenced cases of each run (high = strictly above the median). "
              "Shares are of the evidenced cases; agree = hiQ_hiB + loQ_loB."]
    if notes:
        lines += ["", "Notes:", *[f"- {n}" for n in notes]]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("trees", nargs="*", help="retention run-unit trees (default: plan_full_r760 retention, both tasks)")
    ap.add_argument("--source-condition", default="full_memory_K0.5")
    ap.add_argument("--check-condition", default="provenance_only_K0.5", help="'' to skip the check")
    ap.add_argument("--seeds", nargs="+", type=int, default=[30, 31, 32])
    ap.add_argument("--out", help="output directory (default <worktree>/results/t1/<study>/qb_agreement)")
    args = ap.parse_args(argv)
    trees = [Path(t) for t in args.trees] or [runs_root() / "t1" / DEFAULT_STUDY / e for e in DEFAULT_EXPERIMENTS]
    rows, notes = [], []
    for t in trees:
        if not t.is_dir():
            notes.append(f"{t}: missing")
            continue
        r, n = analyse_tree(t, args.source_condition, args.check_condition or None, set(args.seeds))
        rows += r
        notes += n
    if not rows:
        print("no units found", *notes, sep="\n", file=sys.stderr)
        return 1
    out = Path(args.out) if args.out else RESULTS / "t1" / trees[0].parent.name / "qb_agreement"
    out.mkdir(parents=True, exist_ok=True)
    per_seed = pd.DataFrame(rows).sort_values(["experiment", "dataset", "seed"])
    per_ds = by_dataset(per_seed)
    per_seed.to_csv(out / "qb_agreement_per_seed.csv", index=False)
    per_ds.to_csv(out / "qb_agreement_by_dataset.csv", index=False)
    (out / "qb_agreement.md").write_text(markdown(per_ds, trees, args.source_condition, args.check_condition, notes),
                                         encoding="utf-8")
    print(f"{len(per_seed)} runs, {len(per_ds)} datasets -> {out}")
    for n in notes:
        print("note:", n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
