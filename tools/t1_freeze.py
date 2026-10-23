"""Freeze the final configuration F (docs/T1_PREREG_20261013.md, step D6) and emit confirmation configs.

Selection uses the adopted rule (per task: mean validation error, then validation loss) with the
per-task pre-adaptation feasibility constraint, on all development seeds, over the accepted pool:
classification at score scale 10 (H2c), regression at the default, plus accepted blocks.

F is per task on tuned tasks. Tasks without development runs (the transfer group, and car and
regression wine) get the single global schedule F_global: the best mean per-task rank under the
pre-registered "every task" feasibility, over schedules present on every tuned task, with the
classification score scale aligned (``__kappa10`` names map to their regression counterparts).

Each confirmation config holds one task: F (named ``F``), the alternating default at F's epochs,
patience, stopping rule and output scale (``alternating_default_matched``), F_global (descriptive on
tuned tasks), and frozen-core reuse; retrieval only is recorded by the runner.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools.t1_block_accept import infeasible, pool_rows  # noqa: E402
from tools.t1_grid_search import nnknn_core_settings  # noqa: E402
from tools.t1_run import load_config  # noqa: E402
from tools.t1_schedule_search import condition_id, rank_conditions, selection_rows  # noqa: E402

KAPPA_SUFFIX = "__kappa10"
REUSE = "frozen_reuse__combined"
SYNC_BUDGET_KEYS = ("epochs", "patience", "early_stopping", "score_scale")


def aligned(cid: str) -> str:
    """Schedule identity across task types: drop the classification score-scale marker."""
    name = cid.split("::", 1)[-1]
    if name.startswith(REUSE):
        return REUSE
    return name[: -len(KAPPA_SUFFIX)] if name.endswith(KAPPA_SUFFIX) else name


def rows_for(pool_dirs, conditions_from, block_dirs, seeds, prereg, task_type) -> list[dict]:
    """One task type's pool (stage directories also hold the other type's rows, e.g. kappa-1 classification)."""
    base = [r for r in pool_rows(pool_dirs, conditions_from, seeds, prereg) if r["task_type"] == task_type]
    have = {(r["dataset"], condition_id(r), r["seed"]) for r in base}
    if not block_dirs:
        return base
    new = [r for r in selection_rows(block_dirs, None, seeds, prereg) if r["task_type"] == task_type
           and not condition_id(r).endswith("retrieval_only") and (r["dataset"], condition_id(r), r["seed"]) not in have]
    return base + new


def per_task(rows, seeds, bound):
    _, excluded = infeasible(rows, seeds, bound, "pertask")
    table, winners, _, _ = rank_conditions(rows, seeds=seeds, criterion="err", task_ineligible=excluded)
    return winners, excluded, table


def global_winner(rows, seeds, bound):
    """Best mean per-task rank over aligned schedules present on every task, every-task feasibility."""
    al = []
    for r in rows:
        cid = condition_id(r)
        if cid.endswith("retrieval_only"):
            al.append(r)
            continue
        al.append({**r, "condition": f"{r.get('arm', 'tuned')}__{aligned(cid)}", "aligned_from": cid})
    tasks = {r["dataset"] for r in al}
    present = {}
    for r in al:
        present.setdefault(condition_id(r), set()).add(r["dataset"])
    keep = {c for c, ts in present.items() if ts == tasks}
    al = [r for r in al if condition_id(r) in keep or condition_id(r).endswith("retrieval_only")]
    everywhere, _ = infeasible(al, seeds, bound, "global")
    _, _, glob, _ = rank_conditions(al, seeds=seeds, criterion="err", ineligible=everywhere)
    return glob[0]["condition"], glob[:5], sorted(everywhere)


def condition_dicts(config_paths) -> dict[str, dict]:
    out = {}
    for p in config_paths:
        cfg = load_config(Path(p))
        for c in cfg.get("conditions", []):
            out[c.get("name", c["schedule"])] = c
    return out


def spec_for(name: str, dicts: dict, task_type: str) -> dict:
    """Condition dict of a selected schedule for one task type (classification gets score scale 10)."""
    base = aligned(name)
    if base == REUSE:
        return {"reuse": True}
    cands = [base + KAPPA_SUFFIX, base] if task_type == "classification" else [base]
    for n in cands:
        if n in dicts:
            c = copy.deepcopy(dicts[n])
            if task_type == "classification":
                c["score_scale"] = 10.0
            else:
                c.pop("score_scale", None)
            return c
    raise SystemExit(f"no condition definition for {name} ({task_type})")


def matched_reference(f: dict, task_type: str) -> dict:
    ref = {"name": "alternating_default_matched", "schedule": "alternating_rr"}
    if not f.get("reuse"):
        ref.update({k: f[k] for k in SYNC_BUDGET_KEYS if k in f})
    if task_type == "classification":
        ref["score_scale"] = 10.0
    return ref


def task_config(template: dict, ds: dict, core: dict, f: dict, fg: dict, seeds: list[int], experiment: str,
                base: str, core_source: str) -> dict:
    cfg = {k: copy.deepcopy(v) for k, v in template.items() if k not in ("conditions", "condition_grid", "datasets",
                                                                          "core_by_dataset", "core_selection_source")}
    key = f"{ds['task_type']}__{ds['name']}"
    cls = ds["task_type"] == "classification"
    reuse = {"output_mode": "nominal_residual_scores", "loss": "combined"}
    if cls:
        reuse["score_scale"] = 10.0
    conds = []
    if not f.get("reuse"):
        conds.append({**f, "name": "F"})
    conds.append(matched_reference(f, ds["task_type"]))
    if fg is not None and not fg.get("reuse"):
        conds.append({**fg, "name": "F_global"})
    cfg.update(base=base, experiment=experiment, status="confirmatory_prereg_20261013", seeds=seeds,
               datasets=[{k: v for k, v in ds.items() if v is not None}], core_by_dataset={key: core},
               core_selection_source={key: core_source}, conditions=conds, reuse_conditions=[reuse])
    cfg["sync"] = {**cfg["sync"], "diagnostics_every": 0}
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cls-pool", nargs="+", required=True)
    ap.add_argument("--cls-blocks", nargs="*", default=[])
    ap.add_argument("--reg-pool", nargs="+", required=True)
    ap.add_argument("--reg-conditions-from", nargs="+", required=True)
    ap.add_argument("--reg-blocks", nargs="*", default=[])
    ap.add_argument("--condition-configs", nargs="+", required=True)
    ap.add_argument("--legacy-cores", required=True, help="tuning config with core_by_dataset for tuned tasks")
    ap.add_argument("--legacy-grid", required=True, help="full-study grid dir (car, regression wine cores)")
    ap.add_argument("--transfer-grid", required=True)
    ap.add_argument("--prereg", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--feasibility-pre", type=float, default=1.02)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--record", required=True)
    ap.add_argument("--experiment", default="confirm_final_20261013")
    ap.add_argument("--small-tasks", nargs="*", default=[], help="task keys from tools/t1_seed_audit.py (n_test < 100)")
    args = ap.parse_args()
    seeds = sorted(args.seeds)
    prereg = json.loads(Path(args.prereg).read_text(encoding="utf-8"))
    cls_rows = rows_for(args.cls_pool, None, args.cls_blocks, seeds, args.prereg, "classification")
    reg_rows = rows_for(args.reg_pool, args.reg_conditions_from, args.reg_blocks, seeds, args.prereg, "regression")
    cls_win, cls_ex, cls_table = per_task(cls_rows, seeds, args.feasibility_pre)
    reg_win, reg_ex, reg_table = per_task(reg_rows, seeds, args.feasibility_pre)
    fg_name, fg_top, fg_excluded = global_winner(cls_rows + reg_rows, seeds, args.feasibility_pre)
    dicts = condition_dicts(args.condition_configs)
    legacy = load_config(Path(args.legacy_cores))
    template = yaml.safe_load(Path(args.legacy_cores).read_text(encoding="utf-8"))
    eval_seeds = prereg["seeds"]["evaluation"]
    small = set(args.small_tasks)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    import os
    base = Path(os.path.relpath(REPO / "configs" / "t1" / "_base.yaml", out.resolve())).as_posix()
    legacy_sources = yaml.safe_load(Path(args.legacy_cores).read_text(encoding="utf-8")).get("core_selection_source", {}).get("files", {})
    emitted, selections = {}, {}

    def seeds_for(key):
        return eval_seeds + (prereg["seeds"]["evaluation_small_tasks"] if key in small else [])

    tuned = {**{t: ("classification", w) for t, w in cls_win.items()}, **{t: ("regression", w) for t, w in reg_win.items()}}
    for d in legacy["datasets"]:
        key = f"{d['task_type']}__{d['name']}"
        tt, win = tuned[d["name"]]
        f, fg = spec_for(win, dicts, tt), spec_for(fg_name, dicts, tt)
        src = legacy_sources.get(key, {}).get("selection", args.legacy_cores)
        cfg = task_config(template, dict(d), legacy["core_by_dataset"][key], f, fg, seeds_for(key), args.experiment, base, src)
        emitted[key] = cfg
        selections[key] = {"F": win, "F_is_frozen_reuse": bool(f.get("reuse")), "group": "legacy_tuned"}
    for name, tt in (("car", "regression"), ("wine", "regression")):  # addendum 2: confirmation only
        key = f"{tt}__{name}"
        sel = json.loads((Path(args.legacy_grid) / key / "nnknn" / "selection.json").read_text())
        assert not sel["test_stream_used"]
        fg = spec_for(fg_name, dicts, tt)
        ds = {"name": name, "task_type": tt, **({"qualified": True} if name == "wine" else {})}
        emitted[key] = task_config(template, ds, nnknn_core_settings(sel["parameters"]), fg, None, seeds_for(key), args.experiment,
                                   base, str(Path(args.legacy_grid) / key / "nnknn" / "selection.json"))
        selections[key] = {"F": fg_name, "group": "legacy_untuned_global"}
    for t in prereg["tasks"]["primary_transfer"]:
        key = f"{t['task_type']}__{t['name']}"
        sel_path = Path(args.transfer_grid) / key / "nnknn" / "selection.json"
        if not sel_path.exists():
            selections[key] = {"F": fg_name, "group": "transfer", "pending": "no NN-kNN selection yet"}
            continue
        sel = json.loads(sel_path.read_text())
        assert not sel["test_stream_used"]
        fg = spec_for(fg_name, dicts, t["task_type"])
        ds = {"name": t["name"], "task_type": t["task_type"], "display": t.get("display"), "max_train": t["max_train"]}
        emitted[key] = task_config(template, ds, nnknn_core_settings(sel["parameters"]), fg, None, seeds_for(key), args.experiment,
                                   base, str(sel_path))
        selections[key] = {"F": fg_name, "group": "transfer"}
    hashes = {}
    for key, cfg in emitted.items():
        path = out / f"{key}.yaml"
        text = (f"# Frozen confirmation config ({args.experiment}) for {key}; generated by tools/t1_freeze.py.\n"
                + yaml.safe_dump(cfg, sort_keys=False))
        path.write_text(text, encoding="utf-8", newline="\n")
        hashes[str(path)] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    # Tasks whose F is frozen-core reuse have no condition named F: map them for tools/t1_confirm_report.py.
    final_map = {key: "tuned::" + REUSE + ("__kappa=10" if key.startswith("classification__") else "")
                 for key, v in selections.items() if v.get("F_is_frozen_reuse")}
    map_path = Path(args.record).with_name("final_map.json")
    map_path.parent.mkdir(parents=True, exist_ok=True)
    map_path.write_text(json.dumps(final_map, indent=2), encoding="utf-8")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    record = {"rule": "per task: mean validation error, then validation loss; per-task pre-adaptation feasibility "
                      f"{args.feasibility_pre}; development seeds {seeds}",
              "global_rule": "best mean per-task rank over schedules present on every tuned task, every-task feasibility",
              "F_global": fg_name, "F_global_top5": fg_top, "F_global_excluded_by_feasibility": fg_excluded,
              "per_task": selections, "classification_excluded": {t: sorted(v) for t, v in cls_ex.items()},
              "regression_excluded": {t: sorted(v) for t, v in reg_ex.items()},
              "configs": hashes, "code_head": head, "test_stream_used": False, "final_id": "tuned::F",
              "final_map": final_map,
              "pools": {"classification": args.cls_pool + args.cls_blocks, "regression": args.reg_pool + args.reg_blocks}}
    Path(args.record).parent.mkdir(parents=True, exist_ok=True)
    Path(args.record).write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"F_global: {fg_name}")
    for k, v in selections.items():
        print(f"  {k:40s} {v['group']:22s} {v['F']}{' (PENDING)' if v.get('pending') else ''}")
    print(f"wrote {len(emitted)} configs to {out}; record {args.record}")


if __name__ == "__main__":
    main()
