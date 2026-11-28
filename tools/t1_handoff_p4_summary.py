"""Read-only summary of handoff 2026-11-26 pack p4 (TabPFN baseline, covariate_pc1 shift stress).

Two steps, both read only:
  dump    walks the run root (``runs/``) and writes one JSON line per unit (status, verify problems, scalar row or
          grid-trial fields) to ``<out>/units.jsonl``. Run it where the units are (r760 holds every host's copy).
  report  reads ``units.jsonl`` and writes the comparison tables (csv + md) to ``<out>``.

Error convention (as tools/t1_confirm_report.py): classification 1 - accuracy, regression standardized-target RMSE.
NN-kNN F and reuse arms use the post-adaptation output, retrieval_only the pre output; grid baselines use test_metrics
(accuracy / rmse_standardized). Per task, a paired difference is taken per seed and averaged over the 3 seeds; a task is a
win / tie / loss for the reference by the sign of that mean (|mean| <= TIE counts as a tie). Exploratory, our choices.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TIE = 1e-9
DEFAULT_OUT = ROOT / "results" / "t1" / "handoff_20261126" / "p4"
P4 = ("handoff_tabpfn_r760", "handoff_shift_r760")
FAM = ("knn", "mlp", "catboost", "tabm")
SMALL7 = {"classification__iris", "classification__wine", "classification__zebra", "classification__zebra_special",
          "regression__body_fat", "regression__diabets", "regression__yacht"}

PATTERNS = {  # name -> glob below runs/t1 (unit dirs)
    "tabpfn": "handoff_tabpfn_r760/*__*/tabpfn/evaluation/*/s*",
    "ref_grid": "handoff_allcaseF_r760/*__*/*/evaluation/*/s5[012]",
    "ref_grid_small": "baselines_r760/*__*/*/evaluation/*/s5[012]",
    "maskedF": "handoff_allcaseF_r760/maskedF_20261126/*/tuned__*/s5[012]",
    # allcaseF holds 115 GB of event files: only the F unit of the TabPFN tasks is read (see dump)
    "allcaseF": "handoff_allcaseF_r760/allcaseF_20261126/*/tuned__F/s5[012]",
    "shift_grid": "handoff_shift_r760/*__*/*/evaluation/*/s*",
    "shiftF": "handoff_shift_r760/shiftF_20261126/*/*/s*",
    "shift_ret": "handoff_shift_r760/shift_retention__*/*/*/s*",
    "plan_ret": "plan_full_r760/retention__*/*/*_K0.5/s3[012]",
}


def _scalars(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_scalars(v, f"{prefix}{k}."))
        elif isinstance(v, (int, float, str, bool)) or v is None:
            out[prefix + k] = v
    return out


def dump(runs: Path, out: Path) -> None:
    from model.common.tracking import load_run, run_status, verify_run  # noqa: F401
    from model.t1.records import unordered
    out.mkdir(parents=True, exist_ok=True)
    base = runs / "t1"
    tabpfn_names = {p.split("/")[-5].split("__", 1)[1] for p in glob.glob(str(base / PATTERNS["tabpfn"]))}
    n = 0
    with open(out / "units.jsonl", "w") as f:
        for name, pat in PATTERNS.items():
            for p in sorted(glob.glob(str(base / pat))):
                rel = os.path.relpath(p, base)
                if any(w in rel.lower() for w in ("sel_fresh", "held_out")):
                    continue
                if name == "ref_grid_small" and rel.split("/")[1] not in SMALL7:
                    continue
                if name == "ref_grid" and rel.split("/")[2] not in FAM:
                    continue
                if name == "allcaseF" and rel.split("/")[2] not in tabpfn_names:
                    continue
                ev = sorted(glob.glob(p + "/events.out.tfevents.*"))
                rec = {"_set": name, "_path": rel, "_host": ev[0].split("/")[-1].split(".")[4] if ev else None}
                try:  # one scan: load_run(allow_incomplete) carries the unit's status
                    d = load_run(p, arrays=False, allow_incomplete=True)
                    rec["_status"] = d.info.status
                except Exception:
                    d, rec["_status"] = None, run_status(p).status
                if rel.startswith(P4):
                    rec["_verify"] = verify_run(p)
                if d is not None and rec["_status"] == "complete":
                    for key in ("row", "trial"):
                        if key in d._single:
                            rec.update(_scalars(unordered(d.record(key))))
                    man = d._single.get("manifest")
                    if isinstance(man, dict):
                        rec.update({f"manifest.{k}": v for k, v in _scalars(unordered(man)).items()
                                    if "sha256" in k or "split" in k})
                f.write(json.dumps(rec) + "\n")
                n += 1
    print(f"dumped {n} units to {out / 'units.jsonl'}")


# ---------------------------------------------------------------- report

def task_of(rec: dict) -> str:
    parts = rec["_path"].split("/")
    if rec["_set"] in ("tabpfn", "ref_grid", "ref_grid_small", "shift_grid"):
        return parts[1]
    return f"{rec['task_type']}__{rec['dataset_name']}"


def method_of(rec: dict) -> str | None:
    parts = rec["_path"].split("/")
    s = rec["_set"]
    if s in ("tabpfn", "ref_grid", "ref_grid_small", "shift_grid"):
        return parts[2]
    if s in ("maskedF", "allcaseF", "shiftF"):
        c = parts[3]
        if c.startswith("_"):
            return None
        tag = {"maskedF": "maskedF", "allcaseF": "allcaseF", "shiftF": "shiftNN"}[s]
        return f"{tag}:{c.removeprefix('tuned__')}"
    if s in ("shift_ret", "plan_ret"):
        return None if parts[3].startswith("_") else f"{s}:{parts[3]}"
    return None


def grid_err(rec: dict, group: str | None = None) -> float | None:
    cls = rec.get("dataset.task_type") == "classification"
    if group:
        k = f"test_group_metrics.{group}."
        v = rec.get(k + ("accuracy" if cls else "rmse_standardized"))
    else:
        v = rec.get("test_metrics." + ("accuracy" if cls else "rmse_standardized"))
    return None if v is None else (1 - v if cls else v)


def nn_err(rec: dict, group: str | None = None) -> float | None:
    out = "pre" if rec["condition"].endswith("retrieval_only") else "post"
    cls = rec["task_type"] == "classification"
    # retrieval_only rows record the halves without the pre/post tag (test_accuracy_shift_near)
    k = f"test_{'accuracy' if cls else 'rmse'}_" + (f"{group}" if group and out == "pre" else
                                                     f"{out}_{group}" if group else out)
    v = rec.get(k)
    return None if v is None else (1 - v if cls else v)


def ret_err(rec: dict, prefix: str = "test", group: str | None = None) -> float | None:
    cls = rec["task_type"] == "classification"
    k = f"{prefix}_{'accuracy' if cls else 'rmse'}" + (f"_{group}" if group else "_pre")
    v = rec.get(k)
    return None if v is None else (1 - v if cls else v)


def err_of(rec: dict, group: str | None = None) -> float | None:
    if rec["_set"] in ("tabpfn", "ref_grid", "ref_grid_small", "shift_grid"):
        return grid_err(rec, group)
    if rec["_set"] in ("shift_ret", "plan_ret"):
        return ret_err(rec, "test", group)
    return nn_err(rec, group)


def table(recs, methods, ref, tasks, seeds, group=None):
    """per (method, task): seed->err; plus paired stats vs ref."""
    E = defaultdict(dict)
    for r in recs:
        m = r["_m"]
        if m in methods and r["_task"] in tasks and r.get("seed") in seeds:
            e = err_of(r, group)
            if e is not None and math.isfinite(e):
                E[(m, r["_task"])][r["seed"]] = e
    rows, summary = [], []
    for m in methods:
        full = [t for t in tasks if len(E[(m, t)]) == len(seeds) and len(E[(ref, t)]) == len(seeds)]
        for t in tasks:
            e = E[(m, t)]
            rr = E[(ref, t)]
            d = [e[s] - rr[s] for s in seeds if s in e and s in rr]
            rows.append(dict(method=m, task=t, n_seeds=len(e), mean_err=np.mean(list(e.values())) if e else None,
                             std_err=np.std(list(e.values()), ddof=1) if len(e) > 1 else None,
                             mean_diff_vs_ref=np.mean(d) if len(d) == len(seeds) else None))
        cls_split = {}
        for tt in ("classification", "regression"):
            ft = [t for t in full if t.startswith(tt)]
            if not ft:
                continue
            per_seed = [np.mean([E[(m, t)][s] for t in ft]) for s in seeds]
            per_seed_ref = [np.mean([E[(ref, t)][s] for t in ft]) for s in seeds]
            dmeans = [np.mean([E[(m, t)][s] - E[(ref, t)][s] for s in seeds]) for t in ft]
            # wins for the reference: method error higher than ref
            w = sum(x > TIE for x in dmeans); l = sum(x < -TIE for x in dmeans); ti = len(dmeans) - w - l
            seed_d = [a - b for a, b in zip(per_seed, per_seed_ref)]
            cls_split[tt] = dict(method=m, task_type=tt, n_tasks=len(ft), mean=np.mean(per_seed),
                                 std=np.std(per_seed, ddof=1), ref_mean=np.mean(per_seed_ref),
                                 diff_mean=np.mean(seed_d), diff_std=np.std(seed_d, ddof=1),
                                 ref_wins=w, ties=ti, ref_losses=l)
        summary.extend(cls_split.values())
    return rows, summary, E


def fmt(x, nd=3):
    return "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}f}"


def md_summary(summary, ref_label):
    lines = [f"| method | type | tasks | mean err | std (3 seeds) | {ref_label} err, same tasks | diff vs {ref_label} (mean ± std) | {ref_label} W/T/L |",
             "|---|---|---|---|---|---|---|---|"]
    for s in summary:
        lines.append(f"| {s['method']} | {s['task_type'][:3]} | {s['n_tasks']} | {fmt(s['mean'])} | {fmt(s['std'])} | {fmt(s['ref_mean'])} | "
                     f"{s['diff_mean']:+.3f} ± {s['diff_std']:.3f} | {s['ref_wins']}/{s['ties']}/{s['ref_losses']} |")
    return "\n".join(lines)


def write_csv(path, rows):
    import csv
    if not rows:
        return
    keys = list(rows[0])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def report(out: Path) -> None:
    recs = [json.loads(line) for line in open(out / "units.jsonl")]
    md = ["# Handoff 2026-11-26 pack p4: summary tables (generated by tools/t1_handoff_p4_summary.py)", ""]
    # completeness
    comp = defaultdict(lambda: defaultdict(int))
    for r in recs:
        comp[r["_set"]][r["_status"]] += 1
        if r.get("_verify"):
            comp[r["_set"]]["verify_problems"] += 1
    md += ["## Unit completeness (all unit dirs matched, incl. _core units)", "", "| set | " + " | ".join(
        ["complete", "other status", "verify problems"]) + " |", "|---|---|---|---|"]
    for s, c in comp.items():
        md.append(f"| {s} | {c['complete']} | {sum(v for k, v in c.items() if k not in ('complete', 'verify_problems'))}"
                  f" | {c.get('verify_problems', 0) if s in ('tabpfn', 'shift_grid', 'shiftF', 'shift_ret') else 'n/a'} |")
    recs = [r for r in recs if r["_status"] == "complete"]
    for r in recs:  # _core units (no row) carry no method
        r["_m"] = method_of(r)
        r["_task"] = task_of(r) if r["_m"] else None
    recs = [r for r in recs if r["_m"]]
    for r in recs:  # grid sets: the ref family name is the method
        if r["_set"] in ("ref_grid", "ref_grid_small"):
            r["_m"] = "ref:" + r["_m"]
        elif r["_set"] == "shift_grid":
            r["_m"] = "shift:" + r["_m"]
    # diabets: F is the reuse arm
    for r in recs:
        if r["_task"] == "regression__diabets" and r["_m"].endswith(":aggregate_regression_combined"):
            r2 = dict(r); r2["_m"] = r["_m"].split(":")[0] + ":F"; recs.append(r2)

    # --- A. TabPFN
    tab_tasks = sorted({r["_task"] for r in recs if r["_m"] == "tabpfn"})
    seedsA = [50, 51, 52]
    mA = ["tabpfn", "maskedF:F", "allcaseF:F", "ref:knn", "ref:mlp", "ref:catboost", "ref:tabm"]
    rowsA, sumA, EA = table(recs, mA, "tabpfn", tab_tasks, seedsA)
    write_csv(out / "tabpfn_per_task.csv", rowsA); write_csv(out / "tabpfn_summary.csv", sumA)
    md += ["", "## A. TabPFN vs NN-kNN F and baselines (seeds 50-52, reference TabPFN; diff = method - TabPFN error)",
           "", md_summary(sumA, "TabPFN")]
    md += ["", "Per task, mean error over seeds (TabPFN | maskedF F | allcaseF F | kNN | MLP | CatBoost | TabM):", "",
           "| task | " + " | ".join(mA) + " |", "|---|" + "---|" * len(mA)]
    for t in tab_tasks:
        md.append(f"| {t} | " + " | ".join(fmt(np.mean(list(EA[(m, t)].values())) if EA[(m, t)] else None) + (
            "" if len(EA[(m, t)]) in (0, 3) else f" (n={len(EA[(m, t)])})") for m in mA) + " |")

    # --- B. shift
    seedsB = [30, 31, 32]
    sh_tasks = sorted({r["_task"] for r in recs if r["_set"] == "shiftF"})
    reuse = {t: ("shiftNN:aggregate_regression_combined" if t.startswith("regression") else
                 "shiftNN:nominal_residual_scores_combined") for t in sh_tasks}
    for r in recs:
        if r["_set"] == "shiftF" and r["_m"] == reuse.get(r["_task"]):
            r2 = dict(r); r2["_m"] = "shiftNN:reuse"; recs.append(r2)
    mB = ["shiftNN:F", "shiftNN:retrieval_only", "shiftNN:reuse", "shift:knn", "shift:mlp", "shift:catboost"]
    for grp, label in ((None, "overall"), ("shift_near", "near half"), ("shift_far", "far half")):
        rowsB, sumB, EB = table(recs, mB, "shiftNN:F", sh_tasks, seedsB, grp)
        write_csv(out / f"shift_{label.split()[0]}_per_task.csv", rowsB); write_csv(out / f"shift_{label.split()[0]}_summary.csv", sumB)
        md += ["", f"## B. Shift (covariate_pc1, seeds 30-32), {label}: reference shifted NN-kNN F", "", md_summary(sumB, "F")]
    # far - near gap per method
    gap_rows = []
    _, _, En = table(recs, mB, "shiftNN:F", sh_tasks, seedsB, "shift_near")
    _, _, Ef = table(recs, mB, "shiftNN:F", sh_tasks, seedsB, "shift_far")
    md += ["", "## B2. Far minus near error (mean over tasks with 3 seeds; per-seed across-task means, mean ± std)", "",
           "| method | type | tasks | far-near | tasks far worse / better |", "|---|---|---|---|---|"]
    for m in mB:
        for tt in ("classification", "regression"):
            ft = [t for t in sh_tasks if t.startswith(tt) and len(En[(m, t)]) == 3 and len(Ef[(m, t)]) == 3]
            if not ft:
                continue
            ps = [np.mean([Ef[(m, t)][s] - En[(m, t)][s] for t in ft]) for s in seedsB]
            per_t = [np.mean([Ef[(m, t)][s] - En[(m, t)][s] for s in seedsB]) for t in ft]
            gap_rows.append(dict(method=m, task_type=tt, n_tasks=len(ft), gap_mean=np.mean(ps), gap_std=np.std(ps, ddof=1)))
            md.append(f"| {m} | {tt[:3]} | {len(ft)} | {np.mean(ps):+.3f} ± {np.std(ps, ddof=1):.3f} | "
                      f"{sum(x > TIE for x in per_t)} / {sum(x < -TIE for x in per_t)} |")
    write_csv(out / "shift_far_minus_near.csv", gap_rows)

    # --- C. retention under shift vs unshifted plan_full, K 0.5
    pols = sorted({r["_m"].split(":")[1] for r in recs if r["_set"] == "shift_ret"})
    ret_tasks = sorted({r["_task"] for r in recs if r["_set"] == "shift_ret"})
    crow, md_c = [], ["", "## C. Retention at K 0.5: policy error minus full_memory_K0.5 error, shifted vs unshifted "
                      "(plan_full_r760, same seeds 30-32, different test sets)", "",
                      "| policy | type | tasks (shift / plan) | shift: diff ± std | shift F.M. W/T/L | plan: diff ± std | plan F.M. W/T/L |",
                      "|---|---|---|---|---|---|---|"]
    res = {}
    for st in ("shift_ret", "plan_ret"):
        ms = [f"{st}:{p}" for p in pols]
        rows, summ, _ = table(recs, ms, f"{st}:full_memory_K0.5", ret_tasks, seedsB)
        write_csv(out / f"retention_{st}_per_task.csv", rows); write_csv(out / f"retention_{st}_summary.csv", summ)
        for s in summ:
            res[(st, s["method"].split(":")[1], s["task_type"])] = s
    for p in pols:
        if p == "full_memory_K0.5":
            continue
        for tt in ("classification", "regression"):
            a, b = res.get(("shift_ret", p, tt)), res.get(("plan_ret", p, tt))
            if not a:
                continue
            f = lambda s: (f"{s['diff_mean']:+.3f} ± {s['diff_std']:.3f}", f"{s['ref_wins']}/{s['ties']}/{s['ref_losses']}") if s else ("-", "-")
            md_c.append(f"| {p} | {tt[:3]} | {a['n_tasks']} / {b['n_tasks'] if b else 0} | {f(a)[0]} | {f(a)[1]} | {f(b)[0]} | {f(b)[1]} |")
    md += md_c
    md += ["", "full_memory_K0.5 mean error: " + "; ".join(
        f"{st} {tt[:3]} {fmt(res[(st, pols[0], tt)]['ref_mean'])}" for st in ("shift_ret", "plan_ret")
        for tt in ("classification", "regression") if pols and (st, pols[0], tt) in res)]
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=["dump", "report"])
    ap.add_argument("--runs", default=None, help="run root (default: $NNKNN_RUNS or <NN-KNN>/runs)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    a = ap.parse_args()
    if a.step == "dump":
        from model.common.tracking import runs_root
        dump(runs_root(a.runs), Path(a.out))
    else:
        report(Path(a.out))


if __name__ == "__main__":
    main()
