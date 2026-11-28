"""Read-only summary of the 2026-11-26 handoff pack p3 (core and reuse ablations) against its recorded references.

    # 1. dump (on the host whose run root holds the units; reads only runs/)
    python tools/t1_handoff1126_p3.py dump --runs-root /mnt/largedata/NN-KNN/runs --out p3_dump.jsonl
    # 2. analyse a dump (writes csv/json/md under results/t1/handoff_20261126/p3/)
    python tools/t1_handoff1126_p3.py analyze p3_dump.jsonl [--out DIR]

The dump walks the p3 studies and their reference trees, records every unit's status (complete, incomplete, ...), the
recording host and commit, and, for complete units, the condition ``row`` (t1_run.py) or the pilot ``result``
(t1_pre_topk_pilot.py) scalars. Nothing is trained and nothing under the run root is written.

Comparisons (each p3 condition against the recorded reference of the same dataset and seed):
  pretopk   handoff_pct{10,25,50}_r760 all-case arm  vs pretopk_expanded_r760 all-case arm (knn_mean k 5), test_error
  pctsync   pct{10,25,50}__<cond>_matched_K0.75       vs plan_full <cond>_matched_K0.75 (knn_mean k 5)
  pvdist    <cond>_pvdist                              vs plan_full <cond> (one_hot_euclidean label metric)
  percase   per_case_2022*_<loss>                      vs plan_full aggregate adapter with the same loss
  directhead direct_head_cls_only                      vs plan_full nominal_residual_scores_cls_only (capacity reference)
  nodu      no_du__<cond>                              vs plan_full <cond> (with Delta_u_q)
Per (dataset, comparison) cell: d_s = condition - reference per seed; "better" when the mean of d favours the condition
and at least 2 of 3 seeds favour it, "worse" symmetrically, "identical" when all d_s are 0, otherwise "tie".
With 3 seeds this is a consistency count, not a significance test. Exploratory runs, our own choices.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

STUDIES = {  # study -> experiments (None: the pre-top-K pilot layout <dataset>/s<seed>/<arm>)
    "handoff_pct10_r760": None, "handoff_pct25_r760": None, "handoff_pct50_r760": None, "pretopk_expanded_r760": None,
    "handoff_pctsync_r760": ["pctinit_sync__classification", "pctinit_sync__regression"],
    "handoff_pvdist_r760": ["pvdist_sync__classification"],
    "handoff_percase_r760": ["percase_reuse_classification__classification", "percase_reuse_regression__regression",
                             "percase_nominal_reuse__classification"],
    "handoff_directhead_r760": ["directhead_reuse_classification__classification",
                                "directhead_nominal_reuse__classification"],
    "handoff_nodu_r760": ["nodu_nominal_reuse__classification"],
    "plan_full_r760": ["synchronization__classification", "synchronization__regression",
                       "reuse_classification__classification", "reuse_regression__regression",
                       "nominal_reuse__classification"],
}
KEEP = ("test_accuracy_post", "test_accuracy_pre", "test_L_post", "test_nll_post_T", "test_ece_post", "test_ece_post_T",
        "test_rmse_post", "test_rmse_pre", "test_mae_post", "test_loss_post", "tau_task", "free_radius_status",
        "output_mode", "loss", "schedule", "test_n")
SEEDS = {30, 31, 32}
PILOT_SEEDS = {70, 71, 72}


# ----------------------------------------------------------------------------- dump

def dump(runs_root: Path, out: Path) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from model.common.tracking import iter_runs, load_run
    from model.t1.records import unordered
    base = runs_root / "t1"
    with open(out, "w") as fh:
        for study, exps in STUDIES.items():
            trees = [base / study] if exps is None else [base / study / e for e in exps]
            for tree in trees:
                if not tree.is_dir():
                    fh.write(json.dumps({"study": study, "tree": str(tree.relative_to(base)), "missing": True}) + "\n")
                    continue
                for info in iter_runs(tree, status=None):
                    rel = info.path.relative_to(base).parts
                    rec = {"study": study, "parts": list(rel), "status": info.status, "reason": info.reason}
                    if info.status == "complete":
                        d = load_run(info.path, arrays=False)
                        rec["host"], rec["commit"] = d.identity.get("host"), d.identity.get("git_commit")
                        if "row" in d._single:
                            row = unordered(d.record("row"))
                            rec["kind"] = "row"
                            rec["row"] = {k: row.get(k) for k in ("dataset", "task_type", "seed", "condition", "experiment")}
                            rec["row"].update({k: row[k] for k in KEEP if k in row})
                        elif "result" in d._single:
                            res = unordered(d.record("result"))
                            rec["kind"] = "pilot"
                            rec["row"] = {k: res.get(k) for k in ("dataset", "task_type", "seed", "masked", "test_error")}
                            rec["row"]["bias_init_value"] = (res.get("bias_initialization") or {}).get("value")
                            rec["row"]["bias_init_rule"] = (res.get("bias_initialization") or {}).get("rule")
                            rec["row"]["percentile"] = (res.get("bias_initialization") or {}).get("percentile")
                        else:
                            rec["kind"] = "stage"
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()


# ----------------------------------------------------------------------------- analyse

def lower_better(metric: str) -> bool:
    return any(t in metric for t in ("error", "rmse", "mae", "_L_", "nll", "ece", "loss"))


def verdict(d: list[float], lower: bool) -> str:
    if all(x == 0 for x in d):
        return "identical"
    fav = [(-x if lower else x) for x in d]
    m = sum(fav) / len(fav)
    if m > 0 and sum(f > 0 for f in fav) >= 2:
        return "better"
    if m < 0 and sum(f < 0 for f in fav) >= 2:
        return "worse"
    return "tie"


def mean_sd(x: list[float]) -> tuple[float, float]:
    n = len(x)
    m = sum(x) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in x) / (n - 1)) if n > 1 else float("nan")
    return m, sd


def analyze(dumps: list[Path], out: Path) -> None:
    recs = [json.loads(line) for p in dumps for line in open(p) if line.strip()]
    out.mkdir(parents=True, exist_ok=True)
    # unit completeness per study/experiment
    comp: dict = defaultdict(lambda: defaultdict(int))
    for r in recs:
        if r.get("missing"):
            comp[r["tree"]]["MISSING"] += 1
            continue
        key = "/".join(r["parts"][:2]) if STUDIES[r["study"]] else r["study"]
        comp[key][r["status"]] += 1
    # rows: (study, experiment or '', dataset, condition, seed) -> row
    rows = {}
    hosts = {}
    for r in recs:
        if r.get("status") != "complete" or r.get("kind") not in ("row", "pilot"):
            continue
        w = r["row"]
        if r["kind"] == "pilot":
            cond = "masked" if w["masked"] else "all_cases"
            k = (r["study"], "", w["dataset"], cond, w["seed"])
        else:
            k = (r["study"], r["parts"][1], w["dataset"], w["condition"], w["seed"])
        rows[k] = w
        hosts[k] = r.get("host")

    comparisons = []  # (comparison, cond key prefix, ref key prefix, metric list)
    for p in (10, 25, 50):
        comparisons.append((f"pretopk_pct{p}", ("handoff_pct%d_r760" % p, "", "all_cases"),
                            ("pretopk_expanded_r760", "", "all_cases"), ["test_error"], PILOT_SEEDS))
        comparisons.append((f"pretopk_pct{p}_maskedarm", ("handoff_pct%d_r760" % p, "", "masked"),
                            ("pretopk_expanded_r760", "", "masked"), ["test_error"], PILOT_SEEDS))
    for task, metrics in (("classification", ["test_accuracy_post", "test_L_post"]), ("regression", ["test_rmse_post"])):
        for p in (10, 25, 50):
            for c in ("independent", "alternating_rr", "rr_initial_bias_radius"):
                comparisons.append((f"pctsync_{task}_pct{p}_{c}",
                                    ("handoff_pctsync_r760", f"pctinit_sync__{task}", f"pct{p}__{c}_matched_K0.75"),
                                    ("plan_full_r760", f"synchronization__{task}", f"{c}_matched_K0.75"), metrics, SEEDS))
    for c, ref in (("rr_pvdist", "alternating_rr"), ("rr_initial_bias_radius_pvdist", "rr_initial_bias_radius"),
                   ("rr_exact_kth_radius_pvdist", "rr_exact_kth_radius")):
        comparisons.append((f"pvdist_{ref}", ("handoff_pvdist_r760", "pvdist_sync__classification", f"{c}_matched_K0.75"),
                            ("plan_full_r760", "synchronization__classification", f"{ref}_matched_K0.75"),
                            ["test_accuracy_post", "test_L_post"], SEEDS))
    for loss in ("diff_only", "combined"):
        comparisons.append((f"percase_cls_{loss}",
                            ("handoff_percase_r760", "percase_reuse_classification__classification", f"per_case_2022_{loss}"),
                            ("plan_full_r760", "reuse_classification__classification", f"nominal_residual_scores_{loss}"),
                            ["test_accuracy_post", "test_L_post"], SEEDS))
        comparisons.append((f"percase_nominal_{loss}",
                            ("handoff_percase_r760", "percase_nominal_reuse__classification", f"per_case_2022_{loss}"),
                            ("plan_full_r760", "nominal_reuse__classification", f"nominal_residual_scores_{loss}"),
                            ["test_accuracy_post", "test_L_post"], SEEDS))
        comparisons.append((f"percase_reg_{loss}",
                            ("handoff_percase_r760", "percase_reuse_regression__regression",
                             f"per_case_2022_regression_{loss}"),
                            ("plan_full_r760", "reuse_regression__regression", f"aggregate_regression_{loss}"),
                            ["test_rmse_post"], SEEDS))
    comparisons.append(("directhead_cls", ("handoff_directhead_r760", "directhead_reuse_classification__classification",
                                           "direct_head_cls_only"),
                        ("plan_full_r760", "reuse_classification__classification", "nominal_residual_scores_cls_only"),
                        ["test_accuracy_post", "test_L_post"], SEEDS))
    comparisons.append(("directhead_nominal", ("handoff_directhead_r760", "directhead_nominal_reuse__classification",
                                               "direct_head_cls_only"),
                        ("plan_full_r760", "nominal_reuse__classification", "nominal_residual_scores_cls_only"),
                        ["test_accuracy_post", "test_L_post"], SEEDS))
    for om in ("nominal_residual_scores", "logit_residual"):
        for loss in ("combined", "diff_only", "cls_only"):
            comparisons.append((f"nodu_{om}_{loss}", ("handoff_nodu_r760", "nodu_nominal_reuse__classification",
                                                      f"no_du__{om}_{loss}"),
                                ("plan_full_r760", "nominal_reuse__classification", f"{om}_{loss}"),
                                ["test_accuracy_post", "test_L_post"], SEEDS))

    per_task, overall, problems = [], [], []
    for name, ck, rk, metrics, seeds in comparisons:
        cds = sorted({k[2] for k in rows if k[:2] == ck[:2] and k[3] == ck[2]})
        if not cds:
            problems.append(f"{name}: no rows for condition {ck}")
            continue
        for m in metrics:
            counts = defaultdict(int)
            allc, allr = [], []
            for ds in cds:
                cv, rv, sd_used = [], [], []
                for s in sorted(seeds):
                    a, b = rows.get(ck[:2] + (ds, ck[2], s)), rows.get(rk[:2] + (ds, rk[2], s))
                    if a is None or b is None or a.get(m) is None or b.get(m) is None:
                        continue
                    ha, hb = hosts[ck[:2] + (ds, ck[2], s)], hosts[rk[:2] + (ds, rk[2], s)]
                    if ha != hb:
                        problems.append(f"{name} {ds} s{s}: condition host {ha} != reference host {hb}")
                    cv.append(float(a[m]))
                    rv.append(float(b[m]))
                    sd_used.append(s)
                if len(cv) != len(seeds):
                    problems.append(f"{name} {ds} {m}: only seeds {sd_used} paired")
                if not cv:
                    continue
                d = [x - y for x, y in zip(cv, rv)]
                v = verdict(d, lower_better(m))
                counts[v] += 1
                cm, cs = mean_sd(cv)
                rm, rs = mean_sd(rv)
                dm, dsd = mean_sd(d)
                allc.append(cm)
                allr.append(rm)
                per_task.append({"comparison": name, "metric": m, "dataset": ds, "n": len(cv), "seeds": sd_used,
                                 "cond_mean": cm, "cond_sd": cs, "ref_mean": rm, "ref_sd": rs,
                                 "diff_mean": dm, "diff_sd": dsd, "diffs": d, "verdict": v,
                                 "host": hosts[ck[:2] + (ds, ck[2], sd_used[0])]})
            overall.append({"comparison": name, "metric": m, "lower_is_better": lower_better(m), "tasks": len(allc),
                            "cond_mean_over_tasks": sum(allc) / len(allc), "ref_mean_over_tasks": sum(allr) / len(allr),
                            "mean_diff_over_tasks": (sum(allc) - sum(allr)) / len(allc),
                            "better": counts["better"], "tie": counts["tie"], "worse": counts["worse"],
                            "identical": counts["identical"]})

    def write_csv(path: Path, items: list[dict]) -> None:
        if not items:
            return
        cols = list(items[0].keys())
        with open(path, "w") as fh:
            fh.write(",".join(cols) + "\n")
            for it in items:
                fh.write(",".join(json.dumps(it[c]).replace(",", ";") if isinstance(it[c], (list, dict, str)) and c in (
                    "seeds", "diffs") else str(it[c]) for c in cols) + "\n")

    write_csv(out / "per_task.csv", per_task)
    write_csv(out / "overall.csv", overall)
    with open(out / "completeness.json", "w") as fh:
        json.dump({k: dict(v) for k, v in sorted(comp.items())}, fh, indent=1)
    with open(out / "problems.txt", "w") as fh:
        fh.write("\n".join(sorted(set(problems))) + "\n")
    # pilot bias-initialisation values (what the percentile did to b_default^(0) relative to knn_mean)
    binit = defaultdict(list)
    for k, w in rows.items():
        if k[1] == "" and k[3] == "all_cases" and w.get("bias_init_value") is not None:
            binit[(k[0], k[2])].append(w["bias_init_value"])
    with open(out / "pilot_bias_init.csv", "w") as fh:
        fh.write("study,dataset,mean_bias_init_value,n\n")
        for (st, ds), v in sorted(binit.items()):
            fh.write(f"{st},{ds},{sum(v) / len(v)},{len(v)}\n")
    print(f"{len(per_task)} per-task rows, {len(overall)} overall rows, {len(set(problems))} problems -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("dump")
    a.add_argument("--runs-root", required=True, type=Path)
    a.add_argument("--out", required=True, type=Path)
    b = sub.add_parser("analyze")
    b.add_argument("dumps", nargs="+", type=Path)
    b.add_argument("--out", type=Path,
                   default=Path(__file__).resolve().parents[1] / "results" / "t1" / "handoff_20261126" / "p3")
    args = ap.parse_args()
    if args.cmd == "dump":
        dump(args.runs_root, args.out)
    else:
        analyze(args.dumps, args.out)


if __name__ == "__main__":
    main()
