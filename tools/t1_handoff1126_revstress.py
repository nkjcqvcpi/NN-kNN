"""Summary of the revision_stress part of handoff pack p1 (2026-11-26): the all-case twin
(``handoff_allcaseplan_r760``) and the recorded masked rows (``plan_full_r760``) of experiment
``revision_stress__classification`` on the 7 synthetic factor sets, seeds 30-32. Exploratory, our choices, 3 seeds.

Two steps, both read-only on the run root:

    # 1. on r760 (source ../env.sh first); reads ONLY runs/
    python tools/t1_handoff1126_revstress.py dump --out revstress_units.jsonl
    # 2. anywhere: tables from that dump
    python tools/t1_handoff1126_revstress.py summarize revstress_units.jsonl --out results/t1/handoff_20261126/revstress

``dump`` writes one JSON line per unit directory (study, dataset, run, seed, status) and, for complete units, the
scalar fields of the ``row`` record (condition units) and the numeric ``final/`` scalars. ``summarize`` writes csv
files and ``tables.md``.

What a row is (model/t1/revise.py, ``flagging_ablation``): the trained full-memory core (M0) is flagged by a method
(provenance_only, bias_only, random, oracle); a simulated reviewer inspects the top-b flagged training cases
(b = 10 or 30) and relabels the truly corrupted ones. M1 = edit only; M2 = edit + 5 retraining epochs on the corrected
labels; MC = M0 + the same 5 epochs without edits (matched control).

Choices (ours, exploratory, not PI decisions):
- metrics: precision_at_budget, test_M2_accuracy_pre (primary), gain_M2_M0 = M2 - M0 test accuracy,
  gain_M2_MC = M2 - MC test accuracy, M2_minus_MC_loss (lower is better), test_M0_accuracy_pre (the core);
- comparison A, within each study: every flagging method vs ``random`` at the same budget, task and seed;
- comparison B, all-case twin vs masked plan_full, same condition, task and seed;
- tie = |3-seed mean paired difference| <= 0.005 (metric units); a task enters a comparison only with all 3 seeds for
  both arms; ± is the sd over the 3 seeds of the per-seed mean over tasks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

STUDIES = {"masked": "plan_full_r760", "allcase": "handoff_allcaseplan_r760"}
EXP = "revision_stress__classification"
SEEDS = ("s30", "s31", "s32")
METHODS = ("provenance_only", "bias_only", "random", "oracle")
BUDGETS = ("10", "30")
TIE = 0.005
# metric -> lower is better
METRICS = {"precision_at_budget": False, "corrections": False, "test_M2_accuracy_pre": False, "gain_M2_M0": False,
           "gain_M2_MC": False, "M2_minus_MC_loss": True, "test_M1_accuracy_pre": False, "test_M0_accuracy_pre": False}


def _md(df, floatfmt=".4f") -> str:
    def f(v):
        return format(v, floatfmt) if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(f(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def _read_unit(u: str) -> dict:
    from model.common.tracking import TrackingError, load_run, run_status
    from model.t1.records import unordered
    try:  # load_run runs the full verification scan and raises unless the unit is complete
        data = load_run(u, arrays=False)
    except TrackingError:
        return {"status": run_status(u).status}
    out = {"status": "complete",
           "final": {k: v for k, v in data.final().items() if isinstance(v, (int, float)) and not isinstance(v, bool)}}
    try:
        row = unordered(data.record("row"))
        out["row"] = {k: v for k, v in row.items() if isinstance(v, (int, float, str, bool)) or v is None}
    except KeyError:
        pass
    return out


def dump(args: argparse.Namespace) -> None:
    from concurrent.futures import ProcessPoolExecutor
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from model.common.tracking import runs_root
    root = runs_root() / "t1"
    todo = []
    for arm, study in STUDIES.items():
        ed = root / study / EXP
        for dd in sorted(p for p in ed.iterdir() if p.is_dir()):
            for rd in sorted(p for p in dd.iterdir() if p.is_dir()):
                for u in sorted(p for p in rd.iterdir() if p.is_dir() and p.name in SEEDS):
                    todo.append(({"arm": arm, "study": study, "dataset": dd.name, "run": rd.name, "seed": u.name}, str(u)))
    with open(args.out, "w") as fh, ProcessPoolExecutor(args.workers) as ex:
        for rec, res in zip((t[0] for t in todo), ex.map(_read_unit, [t[1] for t in todo], chunksize=4)):
            fh.write(json.dumps({**rec, **res}) + "\n")
    print(len(todo), "units ->", args.out, file=sys.stderr)


def _paired(a, b, metric, lower):
    """a, b: frames with task, seed, metric. Returns summary dict, per-task frame, or None."""
    m = a.merge(b, on=["task", "seed"], suffixes=("", "_ref"))
    good = m.groupby("task").seed.nunique()
    tasks = sorted(good[good == 3].index)
    dropped = sorted((set(a.task) | set(b.task)) - set(tasks))
    m = m[m.task.isin(tasks)].copy()
    if not tasks:
        return None, None
    m["d"] = m[metric] - m[metric + "_ref"]
    pt = m.groupby("task").agg(mean_diff=("d", "mean"), std_diff=("d", "std"), arm_mean=(metric, "mean"),
                               ref_mean=(metric + "_ref", "mean"))
    w = int(((pt.mean_diff < -TIE) if lower else (pt.mean_diff > TIE)).sum())
    l_ = int(((pt.mean_diff > TIE) if lower else (pt.mean_diff < -TIE)).sum())
    s = {"n_tasks": len(tasks), "seeds": 3,
         "arm_mean": m[metric].mean(), "arm_std_over_seeds": m.groupby("seed")[metric].mean().std(ddof=1),
         "ref_mean": m[metric + "_ref"].mean(), "ref_std_over_seeds": m.groupby("seed")[metric + "_ref"].mean().std(ddof=1),
         "mean_diff": m.d.mean(), "std_diff_over_seeds": m.groupby("seed").d.mean().std(ddof=1),
         "wins": w, "ties": len(pt) - w - l_, "losses": l_, "dropped_tasks": ";".join(dropped)}
    return s, pt.reset_index()


def summarize(args: argparse.Namespace) -> None:
    import pandas as pd

    rows = [json.loads(l) for l in open(args.dump) if l.strip()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    units = pd.DataFrame(rows)
    units["kind"] = units.run.where(units.run.str.startswith("_"), "condition")
    comp = units.groupby(["arm", "kind", "status"]).size().unstack(fill_value=0)
    comp.to_csv(out / "completeness.csv")
    cnt = units[units.kind == "condition"].groupby(["arm", "dataset"]).apply(
        lambda g: int((g.status == "complete").sum())).unstack(0)
    md = ["# revision_stress tables (generated by tools/t1_handoff1126_revstress.py; numbers only)", "",
          "## Unit completeness (condition units: 8 conditions x 3 seeds = 24 per set)", "", _md(comp.reset_index()), "",
          _md(cnt.reset_index()), ""]

    recs = []
    for r in rows:
        if r.get("status") != "complete" or "row" not in r:
            continue
        x = r["row"]
        rec = {"arm": r["arm"], "task": r["dataset"], "seed": r["seed"], "method": x["method"],
               "budget": str(x["review_budget"]), "condition": r["run"]}
        for k in ("precision_at_budget", "recall_at_budget", "corrections", "M2_minus_MC_loss",
                  "test_M0_accuracy_pre", "test_M1_accuracy_pre", "test_M2_accuracy_pre", "test_MC_accuracy_pre",
                  "test_M0_loss_pre", "test_M2_loss_pre", "test_MC_loss_pre"):
            rec[k] = x.get(k)
        rec["gain_M2_M0"] = rec["test_M2_accuracy_pre"] - rec["test_M0_accuracy_pre"]
        rec["gain_M2_MC"] = rec["test_M2_accuracy_pre"] - rec["test_MC_accuracy_pre"]
        recs.append(rec)
    df = pd.DataFrame(recs)
    df.to_csv(out / "units.csv", index=False)

    # sanity: M0 and MC are the same model for every condition of a task, seed and arm
    chk = df.groupby(["arm", "task", "seed"]).agg(m0_spread=("test_M0_accuracy_pre", lambda s: s.max() - s.min()),
                                                  mc_spread=("test_MC_accuracy_pre", lambda s: s.max() - s.min()))
    md += [f"M0 test accuracy spread across the 8 conditions of one task/seed/arm: max {chk.m0_spread.max():.3g}; "
           f"MC spread max {chk.mc_spread.max():.3g}", ""]

    # per arm x method x budget: mean over tasks of 3-seed means
    agg = df.groupby(["arm", "method", "budget"]).agg(
        n_tasks=("task", "nunique"), n_units=("seed", "size"),
        **{f"{k}": (k, "mean") for k in METRICS}).reset_index()
    agg.to_csv(out / "per_arm_method_budget.csv", index=False)
    md += ["## Means over tasks and seeds", "", _md(agg), ""]
    pertask = df.groupby(["arm", "task", "method", "budget"]).agg(
        n=("seed", "nunique"), **{f"{k}_mean": (k, "mean") for k in METRICS},
        **{f"{k}_std": (k, "std") for k in METRICS}).reset_index()
    pertask.to_csv(out / "per_task.csv", index=False)

    comp_rows, task_rows = [], []
    # A: method vs random within each arm
    for arm in STUDIES:
        for meth in METHODS:
            if meth == "random":
                continue
            for b in BUDGETS:
                for metric, lower in METRICS.items():
                    if metric == "test_M0_accuracy_pre":
                        continue
                    a_ = df[(df.arm == arm) & (df.method == meth) & (df.budget == b)][["task", "seed", metric]]
                    r_ = df[(df.arm == arm) & (df.method == "random") & (df.budget == b)][["task", "seed", metric]]
                    s, pt = _paired(a_, r_, metric, lower)
                    if s is None:
                        continue
                    key = {"comparison": "method_vs_random", "arm": arm, "reference": f"{arm}:random",
                           "method": meth, "budget": b, "metric": metric, "lower_is_better": lower}
                    comp_rows.append({**key, **s})
                    task_rows += [{**key, **t} for t in pt.to_dict("records")]
    # B: all-case vs masked, same condition
    for meth in METHODS:
        for b in BUDGETS:
            for metric, lower in METRICS.items():
                a_ = df[(df.arm == "allcase") & (df.method == meth) & (df.budget == b)][["task", "seed", metric]]
                r_ = df[(df.arm == "masked") & (df.method == meth) & (df.budget == b)][["task", "seed", metric]]
                s, pt = _paired(a_, r_, metric, lower)
                if s is None:
                    continue
                key = {"comparison": "allcase_vs_masked", "arm": "allcase", "reference": "masked",
                       "method": meth, "budget": b, "metric": metric, "lower_is_better": lower}
                comp_rows.append({**key, **s})
                task_rows += [{**key, **t} for t in pt.to_dict("records")]
    cmp_df = pd.DataFrame(comp_rows)
    cmp_df.to_csv(out / "paired.csv", index=False)
    pd.DataFrame(task_rows).to_csv(out / "paired_per_task.csv", index=False)
    cols = ["method", "budget", "n_tasks", "arm_mean", "arm_std_over_seeds", "ref_mean", "ref_std_over_seeds",
            "mean_diff", "std_diff_over_seeds", "wins", "ties", "losses"]
    for comparison, arms in (("method_vs_random", list(STUDIES)), ("allcase_vs_masked", ["allcase"])):
        for arm in arms:
            for metric in METRICS:
                sub = cmp_df[(cmp_df.comparison == comparison) & (cmp_df.arm == arm) & (cmp_df.metric == metric)]
                if sub.empty:
                    continue
                md += [f"## {comparison} ({arm}): {metric}{' (lower better)' if METRICS[metric] else ''}", "",
                       _md(sub[cols]), ""]
    (out / "tables.md").write_text("\n".join(md))
    print("wrote", out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--out", required=True)
    d.add_argument("--workers", type=int, default=4)
    s = sub.add_parser("summarize")
    s.add_argument("dump")
    s.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results/t1/handoff_20261126/revstress"))
    a = ap.parse_args()
    dump(a) if a.cmd == "dump" else summarize(a)


if __name__ == "__main__":
    main()
