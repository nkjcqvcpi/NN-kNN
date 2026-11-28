"""Read-only summary of handoff pack p0 (2026-11-26) against its recorded reference rows.

Two steps, because the run units live in the ``runs/`` tree of the host that holds them (r760 is the single copy):

    # 1. on the host with runs/ (reads only runs/, writes one jsonl)
    python tools/t1_handoff_p0_summary.py dump --runs <NN-KNN>/runs --out <dir>/p0_units.jsonl [--jobs 8]
    # 2. anywhere (reads only the dump)
    python tools/t1_handoff_p0_summary.py analyze --dump <dir>/p0_units.jsonl --out results/t1/handoff_20261126/p0

``dump`` scans every unit of the p0 studies and of their reference experiments (status from the event files, the
``row`` record of t1_run units or the ``result`` record of pre-Top-K units, scalars only, plus the event-file host).
``analyze`` pairs each swept condition with its reference by (dataset, seed) and writes per-dataset mean/sd over seeds,
paired differences and better/tie/worse/identical counts, plus unit completeness counts. Nothing is trained or written
under runs/. Exploratory: our own choices, 3 seeds; the counts are consistency counts, not significance tests.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# (study, experiment) of every p0 condition tree
P0 = [
    ("handoff_reusesweep_r760", "reuse_sweep__classification"),
    ("handoff_reusesweep_r760", "reuse_sweep__regression"),
    ("handoff_syncsens_r760", "sync_sens__classification"),
    ("handoff_syncsens_r760", "sync_sens__regression"),
    *[("handoff_mcbmom_r760", f"mcbmom{m}_{g}__{t}") for m in ("0p9", "0p999") for g in ("retention", "sync")
      for t in ("classification", "regression")],
    ("handoff_protect_r760", "synthetic_protect_rare10__classification"),
    ("handoff_protect_r760", "retention_protect_rare10__classification"),
]
REFS = [
    ("plan_full_r760", "reuse_classification__classification"),
    ("plan_full_r760", "reuse_regression__regression"),
    ("plan_full_r760", "synchronization__classification"),
    ("plan_full_r760", "synchronization__regression"),
    ("plan_full_r760", "mcb_retention__classification"),
    ("plan_full_r760", "mcb_retention__regression"),
    ("plan_full_r760", "sync_mcb__classification"),
    ("plan_full_r760", "sync_mcb__regression"),
    ("handoff_synret_r760", "synthetic_retention__classification"),
    ("plan_full_r760", "retention__classification"),
]
PRETOPK = ["handoff_bik3_r760", "handoff_bik10_r760", "handoff_bik20_r760", "pretopk_expanded_r760"]


def _scalar(v):
    return isinstance(v, (int, float, str, bool)) or v is None


def _unit(args):
    root, path = args
    sys.path.insert(0, str(REPO))
    from model.common.tracking import load_run
    from model.t1.records import unordered
    rel = os.path.relpath(path, root)
    ev = sorted(glob.glob(os.path.join(path, "events.out.tfevents.*")))
    rec = {"_path": rel, "_host": ev[0].split("/")[-1].split(".")[4] if ev else None, "_nev": len(ev)}
    try:
        d = load_run(path, arrays=False, allow_incomplete=True)
        rec["_status"] = d.info.status
        name = "result" if rel.split("/")[0] in PRETOPK else "row"
        if not rel.split("/")[-1].startswith("_") and "/_core/" not in rel:
            row = unordered(d.record(name))
            rec.update({k: v for k, v in row.items() if _scalar(v)})
            if name == "result":
                rec["bias_init_k"] = (row.get("bias_initialization") or {}).get("k")
                rec["core_bias_init_k"] = (row.get("core") or {}).get("bias_init_k")
    except Exception as e:  # noqa: BLE001 - recorded, never hidden
        rec["_err"] = repr(e)
    return rec


def dump(args):
    root = os.path.join(args.runs, "t1")
    units = []
    for study, exp in P0 + REFS:
        for p in sorted(glob.glob(os.path.join(root, study, exp, "*", "*", "s*"))):
            units.append(p)
    for study in PRETOPK:
        units += sorted(glob.glob(os.path.join(root, study, "*", "s*", "*")))
    units = [u for u in units if "sel_fresh" not in u and "held_out" not in u and os.path.isdir(u)]
    with ProcessPoolExecutor(args.jobs) as ex, open(args.out, "w") as f:
        for rec in ex.map(_unit, [(root, u) for u in units], chunksize=4):
            f.write(json.dumps(rec) + "\n")
    print(f"dumped {len(units)} units to {args.out}")


# ---------------------------------------------------------------- analysis

def lower_better(metric):
    return any(t in metric.lower() for t in ("error", "loss", "rmse", "mae", "ece", "nll", "corrupted", "duplicate"))


def sd(x):
    if len(x) < 2:
        return float("nan")
    m = sum(x) / len(x)
    return math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - 1))


def verdict(ds, lower):
    """better/tie/worse/identical over the paired seeds (rule of the 2026-11-25 batch reports)."""
    if all(d == 0 for d in ds):
        return "identical"
    m = sum(ds) / len(ds)
    good = [(-d if lower else d) > 0 for d in ds]
    bad = [(-d if lower else d) < 0 for d in ds]
    if ((-m if lower else m) > 0) and sum(good) >= 2:
        return "better"
    if ((-m if lower else m) < 0) and sum(bad) >= 2:
        return "worse"
    return "tie"


def _split(path):
    p = path.split("/")
    if p[0] in PRETOPK:  # study/ds/sNN/arm
        return {"study": p[0], "experiment": "pretopk", "dataset": p[1], "condition": p[3], "seed": int(p[2][1:])}
    return {"study": p[0], "experiment": p[1], "dataset": p[2], "condition": p[3], "seed": int(p[4][1:])}


def comparisons():
    """(name, cand (study, exp), ref (study, exp), cond -> ref cond, metrics) for every p0 comparison."""
    out = []
    rs_cls = {}
    for om in ("nominal_residual_scores", "logit_residual"):
        for lp in ("ld1.0_lc0.3", "ld0.3_lc1.0", "ld3.0_lc1.0", "ld1.0_lc3.0"):
            rs_cls[f"{om}_combined_{lp}"] = f"{om}_combined"
    for loss in ("combined", "cls_only", "diff_only"):
        rs_cls[f"nominal_residual_scores_{loss}_clipnorm"] = f"nominal_residual_scores_{loss}"
    out.append(("reuse_sweep cls", ("handoff_reusesweep_r760", "reuse_sweep__classification"),
                ("plan_full_r760", "reuse_classification__classification"), rs_cls,
                ["test_accuracy_post", "test_ece_post", "test_nll_post_T", "test_accuracy_pre"]))
    out.append(("reuse_sweep reg", ("handoff_reusesweep_r760", "reuse_sweep__regression"),
                ("plan_full_r760", "reuse_regression__regression"),
                {f"aggregate_regression_combined_{lp}": "aggregate_regression_combined"
                 for lp in ("ld1.0_lc0.3", "ld0.3_lc1.0", "ld3.0_lc1.0", "ld1.0_lc3.0")}, ["test_rmse_post"]))
    for t, metrics in (("classification", ["test_accuracy_post", "test_ece_post"]), ("regression", ["test_rmse_post"])):
        m = {f"rr_{k}_matched_K0.75": "alternating_rr_matched_K0.75"
             for k in ("s_task0.25", "s_task1.0", "median_radius", "shrunk_bias_radius", "clipped_bias_radius",
                       "cohortnorm_bias_radius", "tstar0_no_pretrain")}
        m.update({f"rr_zero_radius_{k}_matched_K0.75": "rr_zero_radius_matched_K0.75" for k in ("s_task0.25", "s_task1.0")})
        out.append((f"sync_sens {t[:3]}", ("handoff_syncsens_r760", f"sync_sens__{t}"),
                    ("plan_full_r760", f"synchronization__{t}"), m, metrics))
        for mom in ("0p9", "0p999"):
            out.append((f"mcbmom {mom} retention {t[:3]}", ("handoff_mcbmom_r760", f"mcbmom{mom}_retention__{t}"),
                        ("plan_full_r760", f"mcb_retention__{t}"),
                        {c: c for c in ("mcb1_random_K0.5", "mcb1_provenance_only_K0.5")},
                        ["test_accuracy_pre"] if t == "classification" else ["test_rmse_pre"]))
            out.append((f"mcbmom {mom} sync {t[:3]}", ("handoff_mcbmom_r760", f"mcbmom{mom}_sync__{t}"),
                        ("plan_full_r760", f"sync_mcb__{t}"),
                        {c: c for c in ("independent_matched_K0.75", "alternating_rr_matched_K0.75",
                                        "alternating_rrr_matched_K0.75")}, metrics))
    pol = {f"{p}_K{k}": f"{p}_K{k}" for p in ("random", "bias_current", "provenance_only", "coverage_reachability")
           for k in ("0.2", "0.5", "0.75")}
    out.append(("protect rare10 synthetic", ("handoff_protect_r760", "synthetic_protect_rare10__classification"),
                ("handoff_synret_r760", "synthetic_retention__classification"), pol,
                ["test_accuracy_pre", "kept_fraction_rare", "kept_fraction_corrupted", "sel_n_protected"]))
    out.append(("protect rare10 anneal", ("handoff_protect_r760", "retention_protect_rare10__classification"),
                ("plan_full_r760", "retention__classification"), pol, ["test_accuracy_pre", "sel_n_protected"]))
    for k in (3, 10, 20):
        out.append((f"pretopk bias_init_k {k}", (f"handoff_bik{k}_r760", "pretopk"), ("pretopk_expanded_r760", "pretopk"),
                    {"all_cases": "all_cases", "masked": "masked"}, ["test_error"]))
    return out


def analyze(args):
    rows = [json.loads(line) for line in open(args.dump)]
    for r in rows:
        r.update(_split(r["_path"]))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    idx = {}
    for r in rows:
        if r["condition"] != "_core":
            idx[(r["study"], r["experiment"], r["dataset"], r["condition"], r["seed"])] = r
    # completeness of every p0 unit (condition units and _core units)
    comp = {}
    for r in rows:
        if r["study"].startswith("handoff_") and r["study"] != "handoff_synret_r760":
            k = (r["study"], r["experiment"])
            c = comp.setdefault(k, {"units": 0, "complete": 0, "core_units": 0, "errors": 0, "multi_event": 0,
                                    "datasets": set(), "seeds": set(), "hosts": set()})
            c["units"] += 1
            c["core_units"] += r["condition"] == "_core"
            c["complete"] += r.get("_status") == "complete"
            c["errors"] += "_err" in r
            c["multi_event"] += r["_nev"] != 1
            c["datasets"].add(r["dataset"]); c["seeds"].add(r["seed"]); c["hosts"].add(r["_host"])
    with open(out / "completeness.csv", "w") as f:
        f.write("study,experiment,units,complete,core_units,load_errors,units_not_one_event_file,datasets,seeds,hosts\n")
        for (s, e), c in sorted(comp.items()):
            f.write(f"{s},{e},{c['units']},{c['complete']},{c['core_units']},{c['errors']},{c['multi_event']},"
                    f"{len(c['datasets'])},{'/'.join(map(str, sorted(c['seeds'])))},{'/'.join(sorted(map(str, c['hosts'])))}\n")
    per_ds, agg, missing = [], [], []
    for name, (cs, ce), (rs, re_), cmap, metrics in comparisons():
        dss = sorted({r["dataset"] for r in rows if r["study"] == cs and r["experiment"] == ce})
        for cond, rcond in cmap.items():
            for m in metrics:
                lower = lower_better(m) if m != "sel_n_protected" else False
                counts = {"better": 0, "tie": 0, "worse": 0, "identical": 0}
                cand_means, ref_means, diffs_all, n_ds = [], [], [], 0
                for ds in dss:
                    seeds = sorted({r["seed"] for r in rows if r["study"] == cs and r["experiment"] == ce
                                    and r["dataset"] == ds and r["condition"] == cond})
                    if not seeds:
                        continue
                    pairs = []
                    for s in seeds:
                        a, b = idx.get((cs, ce, ds, cond, s)), idx.get((rs, re_, ds, rcond, s))
                        if a is None or b is None or a.get(m) is None or b.get(m) is None:
                            missing.append(f"{name},{ds},{cond},{s},{m},cand={a is not None},ref={b is not None}")
                            continue
                        if a.get("_status") != "complete" or b.get("_status") != "complete":
                            missing.append(f"{name},{ds},{cond},{s},{m},incomplete")
                            continue
                        pairs.append((float(a[m]), float(b[m])))
                    if not pairs:
                        continue
                    ca, rb = [p[0] for p in pairs], [p[1] for p in pairs]
                    d = [x - y for x, y in pairs]
                    v = verdict(d, lower)
                    counts[v] += 1
                    n_ds += 1
                    cand_means.append(sum(ca) / len(ca)); ref_means.append(sum(rb) / len(rb)); diffs_all.append(sum(d) / len(d))
                    per_ds.append({"comparison": name, "condition": cond, "reference": f"{rs}/{re_}/{rcond}", "metric": m,
                                   "dataset": ds, "n": len(pairs), "cand_mean": sum(ca) / len(ca), "cand_sd": sd(ca),
                                   "ref_mean": sum(rb) / len(rb), "ref_sd": sd(rb), "diff_mean": sum(d) / len(d),
                                   "diff_sd": sd(d), "verdict": v, "diffs": " ".join(f"{x:.6g}" for x in d)})
                if n_ds:
                    agg.append({"comparison": name, "condition": cond, "metric": m, "lower_is_better": lower,
                                "tasks": n_ds, "cand_mean": sum(cand_means) / n_ds, "ref_mean": sum(ref_means) / n_ds,
                                "mean_diff": sum(diffs_all) / n_ds, **counts})
    _csv(out / "per_dataset.csv", per_ds)
    _csv(out / "aggregate.csv", agg)
    (out / "missing_pairs.txt").write_text("\n".join(missing) + ("\n" if missing else ""))
    print(f"{len(rows)} units; {len(per_ds)} dataset cells; {len(agg)} aggregate rows; {len(missing)} missing pairs -> {out}")


def _csv(path, rows):
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0])
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(f"{r[k]:.6g}" if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--runs", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--jobs", type=int, default=8)
    a = sub.add_parser("analyze")
    a.add_argument("--dump", required=True)
    a.add_argument("--out", default=str(REPO / "results/t1/handoff_20261126/p0"))
    args = ap.parse_args()
    dump(args) if args.cmd == "dump" else analyze(args)


if __name__ == "__main__":
    main()
