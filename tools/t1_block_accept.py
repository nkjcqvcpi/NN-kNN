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
    condition_id, guard_fallbacks, held_out_errors, load_rows, log_ratio, pre_feasibility, rank_conditions,
    selection_rows,
)

RULE_SCOPES = [("err", "global"), ("err", "pertask"), ("loss", "global"), ("loss", "pertask"),
               ("rank", "global"), ("rank", "pertask")]


def pool_rows(pool_dirs, conditions_from=None, seeds=None, prereg=None, exclude=None, tasks=None) -> list[dict]:
    rows = selection_rows(pool_dirs, exclude, seeds, prereg)
    if tasks:  # a block evaluated on the tasks it covers (e.g. classification-only candidates)
        rows = [r for r in rows if r["dataset"] in tasks]
    if conditions_from:
        keep = {condition_id(r) for d in conditions_from for r in load_rows(Path(d))}
        rows = [r for r in rows if condition_id(r) in keep or condition_id(r).endswith("retrieval_only")]
    return [r for r in rows if seeds is None or r["seed"] in seeds]


def infeasible(rows: list[dict], seeds: list[int], bound: float | None, scope: str):
    """Pre-adaptation feasibility from the selection seeds' validation rows only (prereg H1):
    on a task, mean validation pre-adaptation error <= bound x retrieval-only + 1/n_val.
    Returns (conditions excluded everywhere, task -> conditions excluded on that task)."""
    if bound is None:
        return set(), {}
    fails = pre_feasibility([r for r in rows if r["seed"] in seeds], bound)
    if scope == "global":  # the prereg's "on every task" form
        return set(fails), {}
    per_task: dict[str, set] = {}
    for c, ws in fails.items():
        for w in ws:
            per_task.setdefault(w["task"], set()).add(c)
    return set(), per_task


def loso_cells(rows: list[dict], seeds: list[int], rule: str, scope: str, reference: str,
               feasibility: float | None = None) -> dict:
    """(seed, task) -> held-out cell of the condition the rule selects without that seed."""
    test = held_out_errors(rows)
    fallback = set(guard_fallbacks(rows))
    cells = {}
    for held in seeds:
        selection_seeds = [s for s in seeds if s != held]
        everywhere, per_task = infeasible(rows, selection_seeds, feasibility, scope)
        _, winners, glob, _ = rank_conditions(rows, seeds=selection_seeds, criterion=rule,
                                              ineligible=fallback | everywhere, task_ineligible=per_task)
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
    rows = pool_rows(args.pool, args.conditions_from, seeds, args.prereg, args.exclude_condition, args.tasks)
    base = tuple(args.current)
    evals = {f"{r}/{s}": loso_cells(rows, seeds, r, s, args.reference, args.feasibility_pre) for r, s in RULE_SCOPES}
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
    base_rows = pool_rows(args.pool, args.conditions_from, seeds, args.prereg, args.exclude_condition, args.tasks)
    block_rows = [r for r in selection_rows(args.block, args.exclude_condition, seeds, args.prereg)
                  if not condition_id(r).endswith("retrieval_only") and (not args.tasks or r["dataset"] in args.tasks)]
    have = {(r["dataset"], condition_id(r), r["seed"]) for r in base_rows}
    new = [r for r in block_rows if (r["dataset"], condition_id(r), r["seed"]) not in have]
    rule, scope = args.rule
    before = loso_cells(base_rows, seeds, rule, scope, args.reference, args.feasibility_pre)
    after = loso_cells(base_rows + new, seeds, rule, scope, args.reference, args.feasibility_pre)
    picked = sorted({c["selected"] for c in after.values()} - {c["selected"] for c in before.values()})
    return {"mode": "block", "rule": f"{rule}/{scope}", "reference": args.reference, "seeds": seeds,
            "feasibility_pre": args.feasibility_pre, "tasks": sorted({r["dataset"] for r in base_rows}),
            "pool_conditions": len({condition_id(r) for r in base_rows}) - 1,
            "block_conditions": sorted({condition_id(r) for r in new}),
            "before": summary(before), "after": summary(after), "after_vs_before": paired(after, before),
            "block_conditions_selected_held_out": picked,
            "selections_after": {f"{t} s{s}": c["selected"] for (s, t), c in sorted(after.items())},
            "decision_rule": "accept the block (its candidates join the pool) iff the mean paired held-out log ratio "
                             "after vs before is below -1 task-clustered SE"}


def typed_rows(dirs, conditions_from, seeds, prereg, task_type, block=False):
    rows = (selection_rows(dirs, None, seeds, prereg) if block else pool_rows(dirs, conditions_from, seeds, prereg))
    return [r for r in rows if r["task_type"] == task_type and not (block and condition_id(r).endswith("retrieval_only"))]


def split_main(args) -> dict:
    """A block over both task types: classification on its pool (e.g. kappa 10), regression on its own;
    held-out cells of both are pooled into one paired, task-clustered comparison."""
    seeds = sorted(args.seeds)
    rule, scope = args.rule
    before, after, picks = {}, {}, set()
    for tt, pool, cfrom, block, ref in (("classification", args.cls_pool, None, args.cls_block, args.cls_reference),
                                        ("regression", args.reg_pool, args.reg_conditions_from, args.reg_block, args.reference)):
        base = typed_rows(pool, cfrom, seeds, args.prereg, tt)
        have = {(r["dataset"], condition_id(r), r["seed"]) for r in base}
        new = [r for r in typed_rows(block, None, seeds, args.prereg, tt, block=True)
               if (r["dataset"], condition_id(r), r["seed"]) not in have]
        b = loso_cells(base, seeds, rule, scope, ref, args.feasibility_pre)
        a = loso_cells(base + new, seeds, rule, scope, ref, args.feasibility_pre)
        before.update(b)
        after.update(a)
        picks |= {c["selected"] for c in a.values()} - {c["selected"] for c in b.values()}
    return {"mode": "split", "rule": f"{rule}/{scope}", "seeds": seeds, "feasibility_pre": args.feasibility_pre,
            "references": {"classification": args.cls_reference, "regression": args.reference},
            "before": summary(before), "after": summary(after), "after_vs_before": paired(after, before),
            "block_conditions_selected_held_out": sorted(picks),
            "selections_after": {f"{t} s{s}": c["selected"] for (s, t), c in sorted(after.items())},
            "decision_rule": "accept the block iff the mean paired held-out log ratio after vs before is below "
                             "-1 task-clustered SE, over both task types"}


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
        p.add_argument("--feasibility-pre", type=float, help="prereg H1 bound B (pre-adaptation feasibility)")
        p.add_argument("--tasks", nargs="*", help="restrict to these task (dataset) names")
        if name == "rules":
            p.add_argument("--current", nargs=2, default=["err", "global"], metavar=("RULE", "SCOPE"))
        else:
            p.add_argument("--block", nargs="+", required=True)
            p.add_argument("--rule", nargs=2, required=True, metavar=("RULE", "SCOPE"))
    sp = sub.add_parser("split")
    sp.add_argument("--cls-pool", nargs="+", required=True)
    sp.add_argument("--cls-block", nargs="+", required=True)
    sp.add_argument("--cls-reference", required=True)
    sp.add_argument("--reg-pool", nargs="+", required=True)
    sp.add_argument("--reg-conditions-from", nargs="*")
    sp.add_argument("--reg-block", nargs="+", required=True)
    sp.add_argument("--reference", required=True, help="regression reference")
    sp.add_argument("--rule", nargs=2, required=True, metavar=("RULE", "SCOPE"))
    sp.add_argument("--seeds", type=int, nargs="+", required=True)
    sp.add_argument("--feasibility-pre", type=float)
    sp.add_argument("--prereg")
    sp.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "split":
        out = split_main(args)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
        d = out["after_vs_before"]
        print(f"{out['rule']}: GMR {out['before']['gmr']:.3f} -> {out['after']['gmr']:.3f}; diff {d['mean_difference']:+.4f} "
              f"(task-SE {d['se_task_clustered']:.4f}, z {d['z_task_clustered']:+.2f}, tasks {d['tasks_better']}/{d['tasks_worse']}); "
              f"accept: {d['adopt']}; block picks: {len(out['block_conditions_selected_held_out'])}")
        return
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
