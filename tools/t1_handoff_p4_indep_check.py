"""Independent check of t1_handoff_p4_summary.py headline numbers, from the raw TensorBoard event files only.

Uses tensorboard's EventFileLoader and no project code. Read only. Run where the units are (r760):
    python t1_handoff_p4_indep_check.py [run_root]
Checks: (1) shift overall, CatBoost - F and retrieval_only - F per task type with W/T/L (task = 3-seed mean diff);
(2) TabPFN vs CatBoost (seeds 50-52) per task type with W/T/L. Error: 1 - accuracy / standardized RMSE.
"""
import glob
import json
import sys

import numpy as np
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader

R = (sys.argv[1] if len(sys.argv) > 1 else "/mnt/largedata/NN-KNN/runs").rstrip("/") + "/t1/"


def un(o):
    if isinstance(o, dict) and set(o) == {"$items"}:
        return {k: un(v) for k, v in o["$items"]}
    if isinstance(o, dict):
        return {k: un(v) for k, v in o.items()}
    if isinstance(o, list):
        return [un(v) for v in o]
    return o


def rec(unit, name):
    out = None
    for f in sorted(glob.glob(unit + "/events.out.tfevents.*")):
        for ev in EventFileLoader(f).Load():
            for v in ev.summary.value:
                if v.tag == "rec/" + name and v.tensor.string_val:
                    out = un(json.loads(v.tensor.string_val[0]))
    return out


def gerr(t):
    m = t["test_metrics"]
    return 1 - m["accuracy"] if t["dataset"]["task_type"] == "classification" else m["rmse_standardized"]


def nerr(r, out):
    cls = r["task_type"] == "classification"
    v = r[f"test_{'accuracy' if cls else 'rmse'}_{out}"]
    return 1 - v if cls else v


def compare(A, B, seeds, label):
    """A, B: {(task, seed): err}; prints mean/std over seeds of across-task mean of A-B and W/T/L for B."""
    for tt in ("classification", "regression"):
        full = sorted({t for t, _ in A if t.startswith(tt) and all((t, s) in A and (t, s) in B for s in seeds)})
        d = [np.mean([A[(t, s)] - B[(t, s)] for t in full]) for s in seeds]
        per_t = [np.mean([A[(t, s)] - B[(t, s)] for s in seeds]) for t in full]
        w = sum(x > 1e-9 for x in per_t); lo = sum(x < -1e-9 for x in per_t)
        print(f"{label} {tt}: {len(full)} tasks, diff {np.mean(d):+.5f} +- {np.std(d, ddof=1):.5f}, "
              f"B W/T/L {w}/{len(full) - w - lo}/{lo}; B mean {np.mean([np.mean([B[(t, s)] for t in full]) for s in seeds]):.5f}")


# (1) shift: F, retrieval_only, CatBoost (seeds 30-32)
F, RO, C = {}, {}, {}
for u in glob.glob(R + "handoff_shift_r760/shiftF_20261126/*/tuned__*/s*"):
    cond, s = u.split("/")[-2:]
    r = rec(u, "row")
    task = f"{r['task_type']}__{r['dataset_name']}"
    want_f = "tuned__aggregate_regression_combined" if r["dataset_name"] == "diabets" else "tuned__F"
    if cond == want_f:
        F[(task, int(s[1:]))] = nerr(r, "post")
    if cond == "tuned__retrieval_only":
        RO[(task, int(s[1:]))] = nerr(r, "pre")
for u in glob.glob(R + "handoff_shift_r760/*__*/catboost/evaluation/*/s*"):
    t = rec(u, "trial")
    C[(u.split("/")[-5], t["seed"])] = gerr(t)
print("shift units: F", len(F), "retrieval_only", len(RO), "catboost", len(C))
compare(C, F, (30, 31, 32), "shift overall CatBoost - F")
compare(RO, F, (30, 31, 32), "shift overall retrieval_only - F")

# (2) TabPFN vs CatBoost, seeds 50-52, grid baselines of the TabPFN tasks
T, CB = {}, {}
for u in glob.glob(R + "handoff_tabpfn_r760/*__*/tabpfn/evaluation/*/s*"):
    t = rec(u, "trial")
    T[(u.split("/")[-5], t["seed"])] = gerr(t)
for task, s in list(T):
    for study in ("handoff_allcaseF_r760", "baselines_r760"):
        us = glob.glob(f"{R}{study}/{task}/catboost/evaluation/*/s{s}")
        if us:
            CB[(task, s)] = gerr(rec(us[0], "trial"))
            break
print("tabpfn units", len(T), "catboost matched", len(CB))
compare(CB, T, (50, 51, 52), "TabPFN-task CatBoost - TabPFN")
