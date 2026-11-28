"""Summary of handoff pack p2 (2026-11-26): the 7 case-selection / data-valuation policies (study handoff_newpol_r760)
against the 8 recorded plan_full_r760 retention policies, finetune recovery, lookup latency / maintenance overhead
(eff_*, study handoff_latency_r760) and C/H reliability (ch_reliability units). Exploratory, our choices, 3 seeds.

Two steps, both read-only on the run root:

    # 1. on the host whose run root holds the units (r760: source ../env.sh first); reads ONLY runs/
    python tools/t1_handoff1126_p2.py dump --out units.jsonl
    # 2. anywhere: tables from that dump (every number is a recorded final/ scalar of a complete unit)
    python tools/t1_handoff1126_p2.py summarize units.jsonl --out results/t1/handoff_20261126/p2

``dump`` writes one JSON line per unit directory: study, experiment, dataset, run (condition), seed, status and, for
complete units, the numeric ``final/`` scalars (``RunData.final``). ``summarize`` writes csv files and ``tables.md``.

Choices (ours, exploratory, not PI decisions):
- primary metric: test accuracy (classification) and test RMSE (regression, lower is better) after the protocol's
  finetune (``test_accuracy_pre`` / ``test_rmse_pre``), plus the same without finetune (``test_nofinetune_*``);
  the runner's ``test_metric_pre`` is -MSE for regression, so it is not used there;
- recovery = metric after finetune minus metric without finetune (what the finetune gives back), signed so that > 0
  is better; retained = metric after finetune / the full-memory metric of the same core (RMSE ratio for regression);
- only seeds s30-s32 (plan_full also holds s33/s34 for some regression tasks);
- references: ``provenance_only`` (the project's rule) and ``random`` at the same K, same task, same seed;
- tie = |paired difference of the 3-seed means| <= 0.005 (metric units), else win/loss by direction;
- a task enters a comparison only with all 3 seeds for both arms (incomplete tasks are listed, not averaged).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

STUDIES = {
    "handoff_newpol_r760": None,
    "handoff_latency_r760": None,
    "plan_full_r760": ("retention__classification", "retention__regression"),
}
SEEDS = ("s30", "s31", "s32")
NEW = ("rc_cnn", "crr", "bbnr", "enn", "energy_compress", "knn_shapley", "loo_value")
OLD = ("full_memory", "random", "stratified", "bias_current", "provenance_only", "bias_normalized", "kcenter",
       "coverage_reachability")
KS = ("0.2", "0.5", "0.75")
TIE = 0.005


def _md(df, index=True, floatfmt=".4f") -> str:
    """Markdown table without the tabulate dependency."""
    d = df.reset_index() if index else df
    def f(v):
        return format(v, floatfmt) if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
    lines += ["| " + " | ".join(f(v) for v in row) + " |" for row in d.itertuples(index=False)]
    return "\n".join(lines)


def _read_unit(u: str) -> dict:
    from model.common.tracking import TrackingError, load_run, run_status
    try:  # load_run runs the full verification scan and raises unless the unit is complete
        fin = load_run(u, arrays=False).final()
        return {"status": "complete",
                "final": {k: v for k, v in fin.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}}
    except TrackingError:
        return {"status": run_status(u).status}


def dump(args: argparse.Namespace) -> None:
    from concurrent.futures import ProcessPoolExecutor
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from model.common.tracking import runs_root
    root = runs_root() / "t1"
    todo, missing = [], []
    for study, exps in STUDIES.items():
        sd = root / study
        if not sd.is_dir():
            missing.append({"study": study, "status": "missing_study"})
            continue
        for ed in sorted(p for p in sd.iterdir() if p.is_dir() and (exps is None or p.name in exps)):
            for dd in sorted(p for p in ed.iterdir() if p.is_dir()):
                for rd in sorted(p for p in dd.iterdir() if p.is_dir()):
                    for u in sorted(p for p in rd.iterdir() if p.is_dir() and p.name.startswith("s")):
                        todo.append(({"study": study, "experiment": ed.name, "dataset": dd.name, "run": rd.name,
                                      "seed": u.name}, str(u)))
    with open(args.out, "w") as fh, ProcessPoolExecutor(args.workers) as ex:
        for m in missing:
            fh.write(json.dumps(m) + "\n")
        for n, (rec, res) in enumerate(zip((t[0] for t in todo), ex.map(_read_unit, [t[1] for t in todo], chunksize=8)), 1):
            fh.write(json.dumps({**rec, **res}) + "\n")
            if n % 200 == 0:
                print(n, "/", len(todo), "units", file=sys.stderr, flush=True)
    print(len(todo), "units ->", args.out, file=sys.stderr)


def summarize(args: argparse.Namespace) -> None:
    import numpy as np
    import pandas as pd

    rows = [json.loads(l) for l in open(args.dump) if l.strip()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    units = pd.DataFrame([r for r in rows if "run" in r])
    units["task_type"] = units["experiment"].str.rsplit("__", n=1).str[1]
    units["task"] = units["task_type"].str[:3] + ":" + units["dataset"]
    # --exclude study:task:seed (e.g. handoff_newpol_r760:cla:wine:s32, a seed being rerun): counted as missing
    for ex in args.exclude or []:
        st, tt_, ds, sd = ex.split(":")
        hit = (units.study == st) & (units.task == f"{tt_}:{ds}") & (units.seed == sd)
        units.loc[hit, "status"] = "excluded_rerun"
        units.loc[hit, "final"] = None
        print(f"excluded {int(hit.sum())} units for {ex}", file=sys.stderr)

    # completeness per study/experiment
    comp = units.groupby(["study", "experiment", "status"]).size().unstack(fill_value=0)
    comp.to_csv(out / "completeness.csv")
    md = ["# Pack p2 tables (generated by tools/t1_handoff1126_p2.py; numbers only)", "",
          "## Unit completeness", "", _md(comp), ""]

    def flat(study, exps_prefix):
        sel = units[(units.study == study) & (units.status == "complete")
                    & units.experiment.str.startswith(exps_prefix)]
        recs = []
        for _, r in sel.iterrows():
            f = r["final"]
            if not isinstance(f, dict):
                continue
            rec = {"task": r["task"], "task_type": r["task_type"], "run": r["run"], "seed": r["seed"], **f}
            recs.append(rec)
        df = pd.DataFrame(recs)
        return df[df.seed.isin(SEEDS)].copy()  # plan_full also holds s33/s34 on some tasks; p2 ran s30-s32

    new = flat("handoff_newpol_r760", "newpol")
    old = flat("plan_full_r760", "retention")
    lat = flat("handoff_latency_r760", "latency")

    # per-task seed completeness of the newpol policy arms
    pol = new[new.run.str.contains("_K")]
    seedcount = pol.groupby(["task", "run"]).seed.nunique().unstack(fill_value=0)
    expected = [f"{p}_K{k}" for p in NEW for k in KS]
    all_tasks = sorted(set(old.task))
    sc = pd.DataFrame(0, index=all_tasks, columns=expected)
    for t in seedcount.index:
        for c in seedcount.columns:
            if c in sc.columns:
                sc.loc[t, c] = seedcount.loc[t, c]
    sc["arms_with_3_seeds"] = (sc[expected] == 3).sum(axis=1)
    sc["units"] = sc[expected].sum(axis=1)
    sc.to_csv(out / "newpol_seed_completeness.csv")
    rel_n = new[new.run == "ch_reliability"].groupby("task").seed.nunique()
    md += ["## newpol: complete policy units per task (21 arms x 3 seeds = 63 expected)", "",
           _md(sc[["arms_with_3_seeds", "units"]].assign(
               ch_reliability_seeds=rel_n.reindex(sc.index).fillna(0).astype(int))), ""]

    # fullmem check: the newpol core must reproduce the recorded one before pairing
    fm = []
    for (t, s), g in new[new.run.str.contains("_K")].groupby(["task", "seed"]):
        ref = old[(old.task == t) & (old.seed == s)]
        if ref.empty:
            continue
        for k in ("fullmem_metric_pre", "fullmem_loss_pre"):
            fm.append({"task": t, "seed": s, "metric": k,
                       "maxdiff": float(np.nanmax(np.abs(g[k].to_numpy()[:, None] - ref[k].to_numpy()[None, :])))})
    fm = pd.DataFrame(fm)
    fm.to_csv(out / "fullmem_check.csv", index=False)
    md += ["## fullmem check (newpol vs plan_full, same task and seed)", "",
           f"pairs checked {len(fm)}, max |diff| {fm.maxdiff.max() if len(fm) else float('nan'):.3g}", ""]

    both = pd.concat([new.assign(src="newpol"), old.assign(src="plan_full")], ignore_index=True)
    both = both[both.run.str.contains("_K")].copy()
    both["policy"] = both.run.str.rsplit("_K", n=1).str[0]
    both["K"] = both.run.str.rsplit("_K", n=1).str[1]
    reg = both.task_type == "regression"
    # primary metric: accuracy (classification, higher better), RMSE (regression, lower better; the runner's
    # test_metric_pre is -MSE there)
    both["m"] = np.where(reg, both.get("test_rmse_pre"), both["test_accuracy_pre"])
    both["m_nofinetune"] = np.where(reg, both.get("test_nofinetune_rmse_pre"), both["test_nofinetune_accuracy_pre"])
    both["m_fullmem"] = np.where(reg, both.get("fullmem_rmse_pre"), both["fullmem_accuracy_pre"])
    sign = np.where(reg, -1.0, 1.0)
    both["recovery"] = sign * (both["m"] - both["m_nofinetune"])
    both["retained"] = both["m"] / both["m_fullmem"]
    both.to_csv(out / "policy_units.csv", index=False)

    # per task x policy x K: mean, std over 3 seeds
    metrics = ["m", "m_nofinetune", "recovery", "retained"]
    agg = both.groupby(["task_type", "task", "policy", "K"]).agg(
        n=("seed", "nunique"), **{f"{m}_mean": (m, "mean") for m in metrics},
        **{f"{m}_std": (m, "std") for m in metrics}).reset_index()
    agg.to_csv(out / "per_task_policy_K.csv", index=False)

    # paired comparisons against references
    comp_rows, wtl_rows = [], []
    for ref in ("provenance_only", "random"):
        for tt in ("classification", "regression"):
            lower = tt == "regression"
            for metric in ("m", "m_nofinetune"):
                for pol_name in list(NEW) + [p for p in OLD if p not in (ref, "full_memory")]:
                    for K in KS:
                        a = both[(both.task_type == tt) & (both.policy == pol_name) & (both.K == K)]
                        b = both[(both.task_type == tt) & (both.policy == ref) & (both.K == K)]
                        m = a.merge(b, on=["task", "seed"], suffixes=("", "_ref"))
                        good = m.groupby("task").seed.nunique()
                        tasks = sorted(good[good == 3].index)
                        dropped = sorted((set(a.task) | set(b.task)) - set(tasks))
                        m = m[m.task.isin(tasks)]
                        if not tasks:
                            continue
                        m["d"] = m[metric] - m[metric + "_ref"]
                        pt = m.groupby("task").d.mean()
                        w = int(((pt < -TIE) if lower else (pt > TIE)).sum())
                        l_ = int(((pt > TIE) if lower else (pt < -TIE)).sum())
                        tie = len(pt) - w - l_
                        sm = m.groupby("seed").d.mean()  # per-seed mean over tasks
                        comp_rows.append({
                            "reference": ref, "task_type": tt, "metric": metric, "policy": pol_name, "K": K,
                            "n_tasks": len(tasks), "seeds": 3,
                            "policy_mean": m[metric].mean(), "ref_mean": m[metric + "_ref"].mean(),
                            "policy_std_over_seeds": m.groupby("seed")[metric].mean().std(ddof=1),
                            "ref_std_over_seeds": m.groupby("seed")[metric + "_ref"].mean().std(ddof=1),
                            "mean_diff": m.d.mean(), "std_diff_over_seeds": sm.std(ddof=1),
                            "wins": w, "ties": tie, "losses": l_, "dropped_tasks": ";".join(dropped)})
                        for t, d in pt.items():
                            wtl_rows.append({"reference": ref, "metric": metric, "policy": pol_name, "K": K,
                                             "task": t, "mean_diff": d,
                                             "std_diff": m[m.task == t].d.std(ddof=1)})
    cmp_df = pd.DataFrame(comp_rows)
    cmp_df.to_csv(out / "paired_vs_reference.csv", index=False)
    pd.DataFrame(wtl_rows).to_csv(out / "paired_per_task.csv", index=False)
    for ref in ("provenance_only", "random"):
        for tt in ("classification", "regression"):
            sub = cmp_df[(cmp_df.reference == ref) & (cmp_df.task_type == tt) & (cmp_df.metric == "m")]
            md += [f"## {tt}: {'RMSE' if tt == 'regression' else 'accuracy'} after finetune, paired vs {ref} (W/T/L per task, tie |d|<={TIE})", "",
                   _md(sub[["policy", "K", "n_tasks", "policy_mean", "policy_std_over_seeds", "ref_mean",
                            "ref_std_over_seeds", "mean_diff", "std_diff_over_seeds",
                            "wins", "ties", "losses"]], index=False, floatfmt=".4f"), ""]

    # recovery
    rec = agg[agg.n == 3].groupby(["task_type", "policy", "K"]).agg(  # only tasks with all 3 seeds
        n_tasks=("task", "nunique"), recovery=("recovery_mean", "mean"),
        retained=("retained_mean", "mean")).reset_index()
    rec.to_csv(out / "recovery.csv", index=False)
    md += ["## Recovery (finetune gain in accuracy / RMSE reduction, > 0 better) and retained (metric after finetune / full-memory metric; RMSE ratio for regression, lower better), mean over tasks of 3-seed means", "",
           _md(rec, index=False, floatfmt=".4f"), ""]

    # efficiency: newpol (eff_* in newpol units) + latency study (8 recorded policies)
    effcols = ["eff_maintenance_overhead_seconds", "eff_lookup_single_median_seconds",
               "eff_fullmem_lookup_single_median_seconds", "eff_lookup_single_speedup", "eff_lookup_batch_median_seconds",
               "eff_memory_saved_fraction", "sel_baseline_seconds"]
    effu = pd.concat([new.assign(src="newpol"), lat.assign(src="latency")], ignore_index=True)
    effu = effu[effu.run.str.contains("_K")].copy()
    effu["policy"] = effu.run.str.rsplit("_K", n=1).str[0]
    effu["K"] = effu.run.str.rsplit("_K", n=1).str[1]
    cols = [c for c in effcols if c in effu.columns]
    eff = effu.groupby(["src", "policy", "K"]).agg(n_units=("seed", "size"), n_tasks=("task", "nunique"),
                                                    **{c: (c, "median") for c in cols}).reset_index()
    eff.to_csv(out / "efficiency.csv", index=False)
    md += ["## Efficiency (median over all units of the policy x K; seconds)", "",
           _md(eff, index=False, floatfmt=".4g"), ""]

    # latency study must agree with plan_full on non-eff metrics
    chk = lat[lat.run.str.contains("_K")].merge(old, on=["task", "run", "seed"], suffixes=("", "_pf"))
    lat_diff = float(np.nanmax(np.abs(chk["test_metric_pre"] - chk["test_metric_pre_pf"]))) if len(chk) else float("nan")
    lat_nf = float(np.nanmax(np.abs(chk["test_nofinetune_metric_pre"] - chk["test_nofinetune_metric_pre_pf"]))) if len(chk) else float("nan")
    md += [f"latency units paired with plan_full on test_metric_pre: {len(chk)} units, max |diff| {lat_diff:.3g} (no-finetune {lat_nf:.3g})", ""]

    # C/H reliability
    rel = new[new.run == "ch_reliability"]
    rcols = [c for c in rel.columns if c.startswith("rel_f") and ("spearman" in c or "overlap" in c)
             and not c.endswith("_sd") and not c.endswith("_undefined")]
    rtab = []
    for c in rcols:
        f, kind = c[len("rel_"):].split("_", 1)
        per_task = rel.groupby("task")[c].mean()
        rtab.append({"fraction": f, "stat": kind, "n_tasks": per_task.notna().sum(), "mean_over_tasks": per_task.mean(),
                     "min_task": per_task.min(), "max_task": per_task.max()})
    rtab = pd.DataFrame(rtab)
    rtab.to_csv(out / "ch_reliability.csv", index=False)
    md += ["## C/H reliability (subset of reference queries vs full set; mean over seeds then tasks)", "",
           _md(rtab, index=False, floatfmt=".3f"), ""]
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
    s.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results/t1/handoff_20261126/p2"))
    s.add_argument("--exclude", nargs="*", default=[],
                   help="study:task_prefix:dataset:seed counted as missing, e.g. handoff_newpol_r760:cla:wine:s32")
    a = ap.parse_args()
    dump(a) if a.cmd == "dump" else summarize(a)


if __name__ == "__main__":
    main()
