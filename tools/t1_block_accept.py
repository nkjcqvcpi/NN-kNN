"""Pre-registered acceptance decisions for the step-D tuning blocks (docs/T1_PREREG_20261013.md).

    rules  compare selection rules on one candidate pool against the current rule
    block  score a tuning block: the held-out gain of the decided rule when the block's
           candidates join the pool

Both use leave-one-development-seed-out (LOSO): for each held-out seed the rule selects on
the validation rows of the other development seeds and the selection is scored on the
held-out seed's test stream (the pre-registered inner held-out use) as the log error ratio
against a reference condition. Paired differences are summarized over (seed, task) cells and,
for the decision, by the task-clustered standard error (SE of the per-task mean differences),
which allows for correlation between a task's seeds. A change is adopted only if the mean
paired difference is below -1 task-clustered SE.

Rows: ``--pool`` directories are read in order, keeping the first (task, condition, seed)
occurrence; ``--conditions-from`` restricts the pool to the conditions present in that
directory (plus retrieval-only). List the newest stage first so its rows take precedence
(for example uncapped abalone rows over earlier capped ones).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools.t1_schedule_search import (  # noqa: E402
    condition_id, guard_fallbacks, held_out_errors, load_rows, log_ratio, rank_conditions, selection_rows,
)

RULE_SCOPES = [("err", "global"), ("err", "pertask"), ("loss", "global"), ("loss", "pertask"),
               ("rank", "global"), ("rank", "pertask")]


def pool_rows(pool_dirs, conditions_from=None, seeds=None, prereg=None, exclude=None) -> list[dict]:
    rows = selection_rows(pool_dirs, exclude, seeds, prereg)
    if conditions_from:
        keep = {condition_id(r) for d in conditions_from for r in load_rows(Path(d))}
        rows = [r for r in rows if condition_id(r) in keep or condition_id(r).endswith("retrieval_only")]
    return [r for r in rows if seeds is None or r["seed"] in seeds]


def loso_cells(rows: list[dict], seeds: list[int], rule: str, scope: str, reference: str) -> dict:
    """(seed, task) -> held-out cell of the condition the rule selects without that seed."""
    test = held_out_errors(rows)
    ineligible = set(guard_fallbacks(rows))
    cells = {}
    for held in seeds:
        _, winners, glob, _ = rank_conditions(rows, seeds=[s for s in seeds if s != held], criterion=rule,
                                              ineligible=ineligible)
        for t, w in sorted(winners.items()):
            sel = glob[0]["condition"] if scope == "global" else w
            cells[held, t] = {**log_ratio(test, t, sel, reference, held), "selected": sel}
    return cells


def paired(a: dict, b: dict) -> dict:
    """Mean of (a - b) over common cells, its cell SE and its task-clustered SE."""
    keys = sorted(set(a) & set(b))
    if len(keys) != len(a) or len(keys) != len(b):
        raise SystemExit("the two evaluations do not cover the same (seed, task) cells")
    d = np.array([a[k]["log_ratio"] - b[k]["log_ratio"] for k in keys])
    tasks = sorted({t for _, t in keys})
    per_task = np.array([np.mean([a[k]["log_ratio"] - b[k]["log_ratio"] for k in keys if k[1] == t]) for t in tasks])
    se_task = float(per_task.std(ddof=1) / np.sqrt(len(tasks))) if len(tasks) > 1 else float("nan")
    return {"mean_difference": float(d.mean()), "se_cells": float(d.std(ddof=1) / np.sqrt(len(d))),
            "se_task_clustered": se_task, "z_task_clustered": float(d.mean() / se_task) if se_task else float("nan"),
            "tasks_better": int((per_task < 0).sum()), "tasks_worse": int((per_task > 0).sum()), "n_cells": len(d),
            "n_tasks": len(tasks), "adopt": bool(d.mean() < -se_task)}


def summary(cells: dict) -> dict:
    v = np.array([c["log_ratio"] for c in cells.values()])
    return {"mean_log_ratio": float(v.mean()), "gmr": float(np.exp(v.mean())), "n_cells": len(v)}


def rules_main(args) -> dict:
    seeds = sorted(args.seeds)
    rows = pool_rows(args.pool, args.conditions_from, seeds, args.prereg, args.exclude_condition)
    base = tuple(args.current)
    evals = {f"{r}/{s}": loso_cells(rows, seeds, r, s, args.reference) for r, s in RULE_SCOPES}
    cur = evals["/".join(base)]
    out = {"mode": "rules", "current": "/".join(base), "reference": args.reference, "seeds": seeds,
           "conditions": sorted({condition_id(r) for r in rows if not condition_id(r).endswith("retrieval_only")}),
           "tasks": sorted({r["dataset"] for r in rows}), "rules": {}}
    for name, cells in evals.items():
        out["rules"][name] = {**summary(cells), **({} if name == out["current"] else {"vs_current": paired(cells, cur)})}
    passing = {k: v["vs_current"]["mean_difference"] for k, v in out["rules"].items() if v.get("vs_current", {}).get("adopt")}
    out["decision"] = min(passing, key=passing.get) if passing else out["current"]
    out["decision_rule"] = ("adopt the rule/scope with the most negative mean paired held-out log ratio among those "
                            "beating the current rule by more than 1 task-clustered SE; otherwise keep the current rule")
    return out


def block_main(args) -> dict:
    seeds = sorted(args.seeds)
    base_rows = pool_rows(args.pool, args.conditions_from, seeds, args.prereg, args.exclude_condition)
    block_rows = [r for r in selection_rows(args.block, args.exclude_condition, seeds, args.prereg)
                  if not condition_id(r).endswith("retrieval_only")]
    have = {(r["dataset"], condition_id(r), r["seed"]) for r in base_rows}
    new = [r for r in block_rows if (r["dataset"], condition_id(r), r["seed"]) not in have]
    rule, scope = args.rule
    before = loso_cells(base_rows, seeds, rule, scope, args.reference)
    after = loso_cells(base_rows + new, seeds, rule, scope, args.reference)
    picked = sorted({c["selected"] for c in after.values()} - {c["selected"] for c in before.values()})
    return {"mode": "block", "rule": f"{rule}/{scope}", "reference": args.reference, "seeds": seeds,
            "pool_conditions": len({condition_id(r) for r in base_rows}) - 1,
            "block_conditions": sorted({condition_id(r) for r in new}),
            "before": summary(before), "after": summary(after), "after_vs_before": paired(after, before),
            "block_conditions_selected_held_out": picked,
            "selections_after": {f"{t} s{s}": c["selected"] for (s, t), c in sorted(after.items())},
            "decision_rule": "accept the block (its candidates join the pool) iff the mean paired held-out log ratio "
                             "after vs before is below -1 task-clustered SE"}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("rules", "block"):
        p = sub.add_parser(name)
        p.add_argument("--pool", nargs="+", required=True)
        p.add_argument("--conditions-from", nargs="*")
        p.add_argument("--seeds", type=int, nargs="+", required=True)
        p.add_argument("--reference", required=True)
        p.add_argument("--prereg")
        p.add_argument("--exclude-condition", action="append")
        p.add_argument("--out", required=True)
        if name == "rules":
            p.add_argument("--current", nargs=2, default=["err", "global"], metavar=("RULE", "SCOPE"))
        else:
            p.add_argument("--block", nargs="+", required=True)
            p.add_argument("--rule", nargs=2, required=True, metavar=("RULE", "SCOPE"))
    args = ap.parse_args()
    if args.prereg:
        args.prereg = str(args.prereg)
    out = rules_main(args) if args.cmd == "rules" else block_main(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    if args.cmd == "rules":
        for k, v in out["rules"].items():
            vs = v.get("vs_current")
            extra = "" if not vs else (f"  diff {vs['mean_difference']:+.4f} (task-SE {vs['se_task_clustered']:.4f}, "
                                       f"z {vs['z_task_clustered']:+.2f}, tasks {vs['tasks_better']}/{vs['tasks_worse']})"
                                       f"{' ADOPT-ELIGIBLE' if vs['adopt'] else ''}")
            print(f"{k:14s} GMR vs reference {v['gmr']:.3f}{extra}")
        print("decision:", out["decision"])
    else:
        d = out["after_vs_before"]
        print(f"{out['rule']}: GMR {out['before']['gmr']:.3f} -> {out['after']['gmr']:.3f}; diff {d['mean_difference']:+.4f} "
              f"(task-SE {d['se_task_clustered']:.4f}, z {d['z_task_clustered']:+.2f}, tasks {d['tasks_better']}/{d['tasks_worse']}); "
              f"accept: {d['adopt']}; block picks: {len(out['block_conditions_selected_held_out'])}")


if __name__ == "__main__":
    main()
