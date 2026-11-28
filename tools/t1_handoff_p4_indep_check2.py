"""Second independent check (raw TensorBoard events, no project code), read only.
(1) TabPFN vs allcaseF F, seeds 50-52, per task type: mean+-std of across-task mean, paired diff, W/T/L.
(2) shift F (diabets: aggregate_regression_combined) far - near gap, and retrieval_only vs reuse arm overall, seeds 30-32.
"""
import glob, json, sys
import numpy as np
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader

R = sys.argv[1].rstrip("/") + "/t1/"


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


def stats(A, B, seeds, label):
    for tt in ("classification", "regression"):
        ts = sorted({t for t, _ in A if t.startswith(tt) and all((t, s) in A and (t, s) in B for s in seeds)})
        a = [np.mean([A[(t, s)] for t in ts]) for s in seeds]
        b = [np.mean([B[(t, s)] for t in ts]) for s in seeds]
        d = [x - y for x, y in zip(a, b)]
        pt = [np.mean([A[(t, s)] - B[(t, s)] for s in seeds]) for t in ts]
        w = sum(x > 1e-9 for x in pt); lo = sum(x < -1e-9 for x in pt)
        print(f"{label} {tt}: {len(ts)} tasks; A {np.mean(a):.5f}+-{np.std(a, ddof=1):.5f}; B {np.mean(b):.5f}+-{np.std(b, ddof=1):.5f}; "
              f"A-B {np.mean(d):+.5f}+-{np.std(d, ddof=1):.5f}; B W/T/L {w}/{len(ts)-w-lo}/{lo}")


# (1)
T, AF = {}, {}
for u in glob.glob(R + "handoff_tabpfn_r760/*__*/tabpfn/evaluation/*/s*"):
    t = rec(u, "trial"); m = t["test_metrics"]
    cls = t["dataset"]["task_type"] == "classification"
    T[(u.split("/")[-5], t["seed"])] = 1 - m["accuracy"] if cls else m["rmse_standardized"]
for (task, s) in list(T):
    u = f"{R}handoff_allcaseF_r760/allcaseF_20261126/{task.split('__', 1)[1]}/tuned__F/s{s}"
    if glob.glob(u + "/events.out.tfevents.*"):
        r = rec(u, "row")
        if r is None or f"{r['task_type']}__{r['dataset_name']}" != task:
            continue
        cls = r["task_type"] == "classification"
        v = r[f"test_{'accuracy' if cls else 'rmse'}_post"]
        AF[(task, s)] = 1 - v if cls else v
print("tabpfn", len(T), "allcaseF F", len(AF))
stats(AF, T, (50, 51, 52), "allcaseF F (A) vs TabPFN (B)")

# (2)
F = {}; FN = {}; FF = {}; RO = {}; RU = {}
for u in glob.glob(R + "handoff_shift_r760/shiftF_20261126/*/tuned__*/s*"):
    cond, s = u.split("/")[-2:]; s = int(s[1:])
    r = rec(u, "row")
    if r is None:
        continue
    cls = r["task_type"] == "classification"; key = "accuracy" if cls else "rmse"
    e = (lambda v: 1 - v) if cls else (lambda v: v)
    task = f"{r['task_type']}__{r['dataset_name']}"
    fcond = "tuned__aggregate_regression_combined" if r["dataset_name"] == "diabets" else "tuned__F"
    rcond = "tuned__nominal_residual_scores_combined" if cls else "tuned__aggregate_regression_combined"
    if cond == fcond:
        F[(task, s)] = e(r[f"test_{key}_post"])
        FN[(task, s)] = e(r[f"test_{key}_post_shift_near"]); FF[(task, s)] = e(r[f"test_{key}_post_shift_far"])
    if cond == "tuned__retrieval_only":
        RO[(task, s)] = e(r[f"test_{key}_pre"])
    if cond == rcond:
        RU[(task, s)] = e(r[f"test_{key}_post"])
print("shift F", len(F), "retrieval_only", len(RO), "reuse", len(RU))
stats(FF, FN, (30, 31, 32), "shift F far (A) vs near (B)")
stats(RO, F, (30, 31, 32), "shift retrieval_only (A) vs F (B)")
stats(RU, F, (30, 31, 32), "shift reuse (A) vs F (B)")
stats(RU, RO, (30, 31, 32), "shift reuse (A) vs retrieval_only (B)")
