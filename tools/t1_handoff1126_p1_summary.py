"""Summary of handoff pack p1 (2026-11-26) from the dump that tools/t1_handoff1126_p1_dump.py read from runs/.

    python tools/t1_handoff1126_p1_summary.py <dump.jsonl> [--out results/t1/handoff_20261126/p1]

Error is 1 - test accuracy (classification) or the train-standardized test RMSE (regression), of the adapted output
(``*_post``) and of the pre-adaptation output for retrieval-only rows (as tools/t1_confirm_report.py defines it);
baselines use their trial's ``test_metrics`` (accuracy, rmse_standardized). Lower is better everywhere.

Per comparison (condition A against reference B, paired by task and seed): per task the 3-seed mean and sd of each
side, the mean paired difference d = e_A - e_B, and win / tie / loss of A by the sign of d (tie: |d| < 1e-9); over
tasks the counts and the geometric mean ratio GMR = exp(mean_task mean_seed log((e_A + eps) / (e_B + eps))), eps =
0.5 / n_test for classification, 0 for regression (as in the confirm report). Only tasks where both sides have all three
seeds are paired; the others are listed as incomplete. Exploratory: no test, no interval, no decision.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SEEDS_F, SEEDS_PLAN = (50, 51, 52), (30, 31, 32)
TIE = 1e-9
REUSE_IS_F = {"regression__diabets"}  # DECLARATION.md: diabetes keeps its frozen F, the frozen-core reuse arm


def arm_name(cond: str) -> str:
    c = cond.removeprefix("tuned__")
    if c.startswith(("nominal_residual_scores", "aggregate_regression")):
        return "reuse"
    return c


def row_error(r: dict) -> float | None:
    """Adapted output when the row has one, else the pre-adaptation output (retrieval-only and retention rows)."""
    name = "accuracy" if r["task_type"] == "classification" else "rmse"
    outs = ("pre",) if r["condition"].endswith("retrieval_only") else ("post", "pre")
    v = next((r[f"test_{name}_{o}"] for o in outs if r.get(f"test_{name}_{o}") is not None), None)
    if v is None:
        return None
    return 1.0 - float(v) if name == "accuracy" else float(v)


def task_key(r: dict) -> str:
    name = r.get("dataset_name") or r["dataset"].removeprefix(r["task_type"] + "__")
    return f"{r['task_type']}__{name}"


def load(dump: Path):
    lines = [json.loads(x) for x in dump.open()]
    err = defaultdict(dict)  # (scope, task, method) -> seed -> error
    ntest = {}
    hosts = defaultdict(set)
    units = []
    for x in lines:
        units.append({"study": x["study"], "experiment": x["experiment"], "kind": x["kind"], "status": x["status"],
                      "rel": x["rel"], "host": x["host"]})
        if x["kind"] == "row":
            r = x["row"]
            e = row_error(r)
            if e is None:
                continue
            t = task_key(r)
            if x["study"] == "handoff_allcaseF_r760":
                scope, method = "F", ("masked" if x["experiment"].startswith("maskedF") else "allcase") + ":" + arm_name(r["condition"])
            else:
                scope, method = f"plan:{x['experiment']}", ("twin" if x["study"] == "handoff_allcaseplan_r760" else "plan_full") + ":" + r["condition"]
            s = int(r["seed"])
            old = err[scope, t, method].get(s)
            if old is not None and old != e:
                raise SystemExit(f"duplicate rows disagree: {scope} {t} {method} s{s}")
            err[scope, t, method][s] = e
            if scope == "F" and t in REUSE_IS_F and method.endswith(":reuse"):  # its declared F is the reuse arm
                err[scope, t, method.replace(":reuse", ":F")][s] = e
            if r.get("test_n"):
                ntest[t, s] = int(r["test_n"])
            hosts[scope, t, method.split(":")[0]].add(x["host"])
        elif x["kind"] == "trial":
            parts = x["rel"].split("/")  # study/task/family/evaluation/cid/sN
            t, fam = parts[1], parts[2]
            s = int(x["trial"]["seed"])
            m = x["test_metrics"]
            e = 1.0 - float(m["accuracy"]) if t.startswith("classification__") else float(m["rmse_standardized"])
            old = err["F", t, f"baseline:{fam}"].get(s)
            if old is not None and old != e:
                raise SystemExit(f"baseline trials disagree: {t} {fam} s{s}")
            err["F", t, f"baseline:{fam}"][s] = e
            hosts["F", t, "baseline"].add(x["host"])
            if x.get("n_test"):
                ntest.setdefault((t, s), int(x["n_test"]))
    return err, ntest, hosts, pd.DataFrame(units)


def compare(err, ntest, scope: str, a: str, b: str, seeds, tasks=None):
    rows, incomplete = [], []
    all_tasks = sorted({t for (sc, t, m) in err if sc == scope and m in (a, b)})
    for t in all_tasks if tasks is None else tasks:
        ea, eb = err.get((scope, t, a), {}), err.get((scope, t, b), {})
        if not all(s in ea for s in seeds) or not all(s in eb for s in seeds):
            incomplete.append({"task": t, "seeds_A": sorted(ea), "seeds_B": sorted(eb)})
            continue
        xa, xb = np.array([ea[s] for s in seeds]), np.array([eb[s] for s in seeds])
        d = xa - xb
        eps = np.array([0.5 / ntest[t, s] if t.startswith("classification__") and (t, s) in ntest else 0.0 for s in seeds])
        if t.startswith("classification__") and not all((t, s) in ntest for s in seeds):
            eps[:] = 1e-3
        with np.errstate(divide="ignore", invalid="ignore"):
            lr = float(np.mean(np.log((xa + eps) / (xb + eps))))
        md = float(d.mean())
        rows.append({"task": t, "A": a, "B": b, "mean_A": xa.mean(), "sd_A": xa.std(ddof=1), "mean_B": xb.mean(),
                     "sd_B": xb.std(ddof=1), "mean_diff": md, "sd_diff": d.std(ddof=1), "log_ratio": lr,
                     "seeds_A_better": int((d < -TIE).sum()), "seeds_tied": int((abs(d) <= TIE).sum()),
                     "seeds_A_worse": int((d > TIE).sum()),
                     "result": "win" if md < -TIE else ("loss" if md > TIE else "tie")})
    df = pd.DataFrame(rows)
    agg = {"scope": scope, "A": a, "B": b, "tasks_paired": len(df), "tasks_incomplete": len(incomplete)}
    if len(df):
        finite = df["log_ratio"][np.isfinite(df["log_ratio"])]
        agg.update(wins=int((df.result == "win").sum()), ties=int((df.result == "tie").sum()),
                   losses=int((df.result == "loss").sum()),
                   gmr=float(math.exp(finite.mean())) if len(finite) else float("nan"),
                   n_log_ratio_nonfinite=int((~np.isfinite(df["log_ratio"])).sum()),
                   median_mean_diff=float(df["mean_diff"].median()))
    return df, agg, incomplete


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--out", default=str(REPO / "results/t1/handoff_20261126/p1"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    err, ntest, hosts, units = load(Path(a.dump))

    # unit completeness
    comp = units.groupby(["study", "experiment", "kind", "status"]).size().rename("units").reset_index()
    comp.to_csv(out / "unit_status.csv", index=False)
    units[units.status != "complete"].to_csv(out / "units_not_complete.csv", index=False)

    # experiment 1
    cmp1 = [("allcase:F", "masked:F"), ("allcase:F", "allcase:retrieval_only"), ("allcase:F", "allcase:reuse"),
            ("allcase:F_zero_radius", "allcase:F"), ("allcase:F_no_penalty", "allcase:F"),
            ("allcase:retrieval_only", "masked:retrieval_only")]
    cmp1 += [("allcase:F", f"baseline:{f}") for f in ("knn", "mlp", "catboost", "tabm")]
    cmp1 += [("masked:F", f"baseline:{f}") for f in ("knn", "mlp", "catboost", "tabm")]
    per, aggs, inc = [], [], []
    for A, B in cmp1:
        df, agg, incomplete = compare(err, ntest, "F", A, B, SEEDS_F)
        per.append(df)
        aggs.append(agg)
        inc += [{"A": A, "B": B, **i} for i in incomplete]
    pd.concat(per).to_csv(out / "exp1_per_task.csv", index=False)
    pd.DataFrame(aggs).to_csv(out / "exp1_aggregate.csv", index=False)
    pd.DataFrame(inc).to_csv(out / "exp1_incomplete.csv", index=False)
    # per-task arm means
    cells = []
    for (sc, t, m), d in err.items():
        if sc != "F":
            continue
        x = np.array([d[s] for s in SEEDS_F if s in d])
        cells.append({"task": t, "method": m, "n_seeds": len(x), "mean": x.mean() if len(x) else np.nan,
                      "sd": x.std(ddof=1) if len(x) > 1 else np.nan, "seeds": sorted(d)})
    pd.DataFrame(cells).sort_values(["task", "method"]).to_csv(out / "exp1_cells.csv", index=False)

    # experiment 2: every (experiment, task, condition) of the twin against plan_full
    per2, agg2, inc2 = [], [], []
    scopes = sorted({sc for (sc, t, m) in err if sc.startswith("plan:")})
    for sc in scopes:
        conds = sorted({m.split(":", 1)[1] for (s2, t, m) in err if s2 == sc})
        for c in conds:
            df, agg, incomplete = compare(err, ntest, sc, f"twin:{c}", f"plan_full:{c}", SEEDS_PLAN)
            if len(df):
                df.insert(0, "experiment", sc[5:])
                per2.append(df)
            agg2.append({**agg, "experiment": sc[5:], "condition": c})
            inc2 += [{"experiment": sc[5:], "condition": c, **i} for i in incomplete]
    p2 = pd.concat(per2) if per2 else pd.DataFrame()
    p2.to_csv(out / "exp2_per_task.csv", index=False)
    pd.DataFrame(agg2).to_csv(out / "exp2_per_condition.csv", index=False)
    pd.DataFrame(inc2).to_csv(out / "exp2_incomplete.csv", index=False)
    if len(p2):
        g = p2.groupby("experiment")
        ex = pd.DataFrame({"cells": g.size(), "tasks": g["task"].nunique(), "wins": g.apply(lambda d: (d.result == "win").sum()),
                           "ties": g.apply(lambda d: (d.result == "tie").sum()), "losses": g.apply(lambda d: (d.result == "loss").sum()),
                           "gmr": g["log_ratio"].apply(lambda s: math.exp(s[np.isfinite(s)].mean())),
                           "median_rel_diff": g.apply(lambda d: float(np.median(d.mean_diff / d.mean_B.replace(0, np.nan))))})
        ex.reset_index().to_csv(out / "exp2_per_experiment.csv", index=False)
        print(ex.to_string())
    # host check: twin and plan_full of one task on the same host
    hc = [{"scope": sc, "task": t, "side": side, "hosts": sorted(h)} for (sc, t, side), h in hosts.items()]
    pd.DataFrame(hc).to_csv(out / "hosts.csv", index=False)
    print(pd.DataFrame(aggs).to_string())
    print(comp.to_string())


if __name__ == "__main__":
    main()
