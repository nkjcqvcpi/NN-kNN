"""Validation-only search for the NN-kNN retrieval + reuse training schedule.

    prepare  embed each task's validation-selected NN-kNN core (HPO grid) into a search config
    select   rank conditions on development seeds by validation error, then validation
             post-adaptation loss; write the selection and a confirmation config
    loso     leave-one-development-seed-out check of selection rules, or of challengers
             against an incumbent, on the held-out seed's test stream
    report   summarize the confirmation on fresh evaluation seeds (test stream)

A condition is (arm, schedule-or-reuse). Frozen-core reuse rows are a schedule
("retrain nothing after the core"); retrieval-only rows are reported, not ranked.
Selection reads only ``val_*`` metrics; test metrics of development seeds are ignored,
except by ``loso``, the pre-registration's declared inner held-out use of them
(docs/T1_PREREG_20261013.md, H6). With ``--prereg``, rows of evaluation, retired or
reserved seeds are refused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools.t1_grid_search import nnknn_core_settings  # noqa: E402
from tools.t1_run import load_config  # noqa: E402
from tools.t1_seed_audit import FORBIDDEN_ROLES, load_prereg, seed_roles  # noqa: E402


# ----------------------------------------------------------------------------- rows

def load_rows(exp_dir: Path) -> list[dict]:
    rows = []
    for shard in sorted(exp_dir.glob("runs__*.jsonl")):
        rows += [json.loads(line) for line in shard.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"no runs__*.jsonl rows in {exp_dir}")
    return rows


def condition_id(row: dict) -> str:
    arm = row.get("arm", "")
    label = row["condition"]
    name = label[len(arm) + 2:] if arm and label.startswith(arm + "__") else label
    if name.endswith("retrieval_only"):
        name = "retrieval_only"
    elif "schedule" not in row:  # frozen-core reuse; regression and classification labels differ
        name = f"frozen_reuse__{row['loss']}"
    return f"{arm}::{name}" if arm else name


def error_of(row: dict, stream: str) -> float:
    post = "pre" if condition_id(row).endswith("retrieval_only") else "post"
    if row["task_type"] == "classification":
        return 1.0 - float(row[f"{stream}_accuracy_{post}"])
    return float(row[f"{stream}_rmse_{post}"])


def mean_by(rows, keyfn, valfn):
    out: dict = {}
    for r in rows:
        out.setdefault(keyfn(r), []).append(valfn(r))
    return out


def refuse_prereg_seeds(rows: list[dict], prereg: str, seeds=None) -> dict:
    """Stop on any row (or requested seed) of a seed the pre-registration keeps out of tuning."""
    roles = seed_roles(load_prereg(prereg))
    used = sorted({int(r["seed"]) for r in rows} | {int(s) for s in seeds or []})
    hit = {s: [x for x in roles.get(s, []) if x in FORBIDDEN_ROLES] for s in used}
    hit = {s: x for s, x in hit.items() if x}
    if hit:
        raise SystemExit(f"refusing pre-registered {' / '.join(FORBIDDEN_ROLES)} seeds: "
                         + ", ".join(f"{s} ({'+'.join(x)})" for s, x in hit.items()))
    return {"path": str(prereg), "sha256": hashlib.sha256(Path(prereg).read_bytes()).hexdigest(),
            "refused_roles": list(FORBIDDEN_ROLES)}


def selection_rows(exp_dirs, exclude=None, seeds=None, prereg=None) -> list[dict]:
    """Rows of the given stages, first occurrence per (task, condition, seed), minus excluded conditions."""
    rows = [r for d in exp_dirs for r in load_rows(Path(d))]
    if prereg:
        refuse_prereg_seeds(rows, prereg, seeds)
    # Retrieval-only rows repeat in every stage; other repeats are dropped by name.
    drop = set(exclude or [])
    seen, kept = set(), []
    for r in rows:
        key = (r["dataset"], condition_id(r), r["seed"])
        if condition_id(r) not in drop and key not in seen:
            seen.add(key)
            kept.append(r)
    if seeds is not None:
        absent = sorted(set(seeds) - {r["seed"] for r in kept})
        if absent:
            raise SystemExit(f"no rows for requested seeds {absent}")
        kept = [r for r in kept if r["seed"] in set(seeds)]
    return kept


def stream_sizes(rows: list[dict], stream: str) -> dict:
    """(dataset, seed) -> stream size from ``<stream>_n``; retrieval-only rows borrow it from their seed's other rows."""
    out: dict = {}
    for r in rows:
        if f"{stream}_n" in r:
            key, n = (r["dataset"], r["seed"]), int(r[f"{stream}_n"])
            if out.setdefault(key, n) != n:
                raise SystemExit(f"inconsistent {stream}_n for {key}: {out[key]} vs {n}")
    return out


# ----------------------------------------------------------------------------- prepare

def prepare(args) -> None:
    template = yaml.safe_load(Path(args.template).read_text(encoding="utf-8"))
    grid = Path(args.grid)
    core_by_dataset, sources = {}, {}
    for ds in template["datasets"]:
        key = f"{ds['task_type']}__{ds['name']}"
        path = grid / key / "nnknn" / "selection.json"
        if not path.exists():
            raise SystemExit(f"no validation-selected core for {key}: {path}")
        sel = json.loads(path.read_text())
        if sel["test_stream_used"]:
            raise SystemExit(f"{key}: test-based selection is prohibited")
        core_by_dataset[key] = nnknn_core_settings(sel["parameters"])
        sources[key] = {"selection": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    if args.experiment:
        template["experiment"] = args.experiment
    template["core_by_dataset"] = core_by_dataset
    template["core_selection_source"] = {"grid": str(grid), "files": sources}
    out = Path(args.out)
    out.write_text("# Generated by tools/t1_schedule_search.py prepare from " + Path(args.template).name + "\n"
                   + yaml.safe_dump(template, sort_keys=False), encoding="utf-8")
    print(f"wrote {out} with {len(core_by_dataset)} tuned cores")


# ----------------------------------------------------------------------------- select

GLOBAL_RULE = "global: mean per-task rank, then tasks won, then name"
RULES = {
    "err": "per task: mean validation error (1-accuracy_post or rmse_post), then mean validation "
           "post-adaptation loss, then name; " + GLOBAL_RULE,
    "loss": "per task: mean validation post-adaptation loss, then mean validation error, then name; " + GLOBAL_RULE,
    "rank": "per task: mean over seeds of the per-seed rank by (validation error, validation post-adaptation "
            "loss, name), then mean validation error, then mean loss, then name; " + GLOBAL_RULE,
}


def rank_conditions(rows: list[dict], seeds: list[int] | None = None, criterion: str = "err",
                    ineligible=frozenset()):
    """Per-task means over seeds, per-task ranks and the global mean rank (validation only).

    ``criterion`` orders a task's conditions (``RULES``). ``ineligible`` conditions must still
    be complete; they are listed in the table without a rank and never win.
    """
    pool = [r for r in rows if not condition_id(r).endswith("retrieval_only")]
    if seeds is not None:
        pool = [r for r in pool if r["seed"] in seeds]
    err = mean_by(pool, lambda r: (r["dataset"], condition_id(r)), lambda r: error_of(r, "val"))
    loss = mean_by(pool, lambda r: (r["dataset"], condition_id(r)), lambda r: float(r["val_L_post"]))
    counts = {k: len(v) for k, v in err.items()}
    expected = max(counts.values())
    incomplete = sorted(k for k, n in counts.items() if n != expected)
    tasks = sorted({t for t, _ in err})
    every = sorted({c for _, c in err})
    missing = sorted((t, c) for t in tasks for c in every if (t, c) not in err)
    if incomplete or missing:
        raise SystemExit(f"incomplete validation evidence: unequal seeds {incomplete[:5]}, missing {missing[:5]}")
    conds = [c for c in every if c not in ineligible]
    if not conds:
        raise SystemExit("no eligible condition to rank")
    if criterion == "err":
        key = lambda t, c: (np.mean(err[t, c]), np.mean(loss[t, c]), c)  # noqa: E731
    elif criterion == "loss":
        key = lambda t, c: (np.mean(loss[t, c]), np.mean(err[t, c]), c)  # noqa: E731
    elif criterion == "rank":
        seed_rank = per_seed_ranks(pool, tasks, conds)
        key = lambda t, c: (seed_rank[t, c], np.mean(err[t, c]), np.mean(loss[t, c]), c)  # noqa: E731
    else:
        raise ValueError(f"unknown selection criterion {criterion!r}")
    def entry(t, c, i):
        e = {"task": t, "condition": c, "rank": i, "val_error": float(np.mean(err[t, c])),
             "val_L_post": float(np.mean(loss[t, c])), "seeds": counts[t, c]}
        if criterion == "rank":
            e["mean_seed_rank"] = seed_rank.get((t, c))
        return e

    table, ranks = [], {c: [] for c in conds}
    winners = {}
    for t in tasks:
        order = sorted(conds, key=lambda c: key(t, c))
        winners[t] = order[0]
        for i, c in enumerate(order, 1):
            ranks[c].append(i)
            table.append(entry(t, c, i))
        table += [entry(t, c, None) for c in every if c in ineligible]
    glob = sorted(({"condition": c, "mean_rank": float(np.mean(ranks[c])),
                    "tasks_won": sum(w == c for w in winners.values())} for c in conds),
                  key=lambda g: (g["mean_rank"], -g["tasks_won"], g["condition"]))
    return table, winners, glob, expected


def per_seed_ranks(pool: list[dict], tasks: list[str], conds: list[str]) -> dict:
    """(task, condition) -> mean over seeds of its rank among ``conds`` by (validation error, L_post, name)."""
    err = mean_by(pool, lambda r: (r["dataset"], condition_id(r), r["seed"]), lambda r: error_of(r, "val"))
    loss = mean_by(pool, lambda r: (r["dataset"], condition_id(r), r["seed"]), lambda r: float(r["val_L_post"]))
    out = {}
    for t in tasks:
        seeds = sorted({s for (tt, _, s) in err if tt == t})
        gaps = [(c, s) for c in conds for s in seeds if (t, c, s) not in err]
        if gaps:
            raise SystemExit(f"rank criterion needs every seed of {t} for every condition; missing {gaps[:5]}")
        ranks = {c: [] for c in conds}
        for s in seeds:
            order = sorted(conds, key=lambda c: (np.mean(err[t, c, s]), np.mean(loss[t, c, s]), c))
            for i, c in enumerate(order, 1):
                ranks[c].append(i)
        out.update({(t, c): float(np.mean(v)) for c, v in ranks.items()})
    return out


FEASIBILITY_TOL = 1e-9  # float64 rounding of the means; classification errors are whole cases first


def pre_feasibility(rows: list[dict], bound: float) -> dict[str, list[dict]]:
    """Conditions whose mean validation pre-adaptation error exceeds ``bound`` x retrieval-only + 1/n_val on a task.

    Means are over the condition's seeds, paired with the same arm's retrieval-only rows;
    n_val is the validation size of those seeds (mean, if folds differ). Classification
    errors are first rounded to whole validation cases, round((1 - accuracy) * n_val) / n_val,
    because the runner stores accuracies as float32 (29/30 -> 0.96666664), which would
    otherwise put a candidate exactly on the bound (one extra error) a few 1e-8 above it.
    """
    n_val = stream_sizes(rows, "val")

    def val_pre(r):
        e, n = pre_error(r, "val"), n_val.get((r["dataset"], r["seed"]))
        if r["task_type"] != "classification" or n is None:  # a missing n_val is refused below
            return e
        cases = e * n
        if abs(cases - round(cases)) > 1e-2 + 1e-7 * n:  # float32 accuracy: relative error <= 6e-8
            raise SystemExit(f"val_n {n} of {r['dataset']} seed {r['seed']} does not fit its validation accuracy "
                             f"({cases:.4f} errors in {condition_id(r)})")
        return round(cases) / n

    ref = {(r["dataset"], r.get("arm", ""), r["seed"]): val_pre(r)
           for r in rows if condition_id(r).endswith("retrieval_only")}
    pre = mean_by([r for r in rows if not condition_id(r).endswith("retrieval_only")],
                  lambda r: (r["dataset"], condition_id(r)), lambda r: (r.get("arm", ""), r["seed"], val_pre(r)))
    out: dict[str, list[dict]] = {}
    for (t, c), values in sorted(pre.items()):
        gaps = [s for arm, s, _ in values if (t, arm, s) not in ref or (t, s) not in n_val]
        if gaps:
            raise SystemExit(f"--feasibility-pre needs retrieval-only rows and val_n for {t}, seeds {sorted(set(gaps))} ({c})")
        cand = float(np.mean([v for _, _, v in values]))
        base = float(np.mean([ref[t, arm, s] for arm, s, _ in values]))
        nv = float(np.mean([n_val[t, s] for _, s, _ in values]))
        limit = bound * base + 1.0 / nv
        if cand > limit + FEASIBILITY_TOL:
            out.setdefault(c, []).append({"task": t, "val_pre_error": cand, "retrieval_only_val_error": base,
                                          "n_val": nv, "limit": limit, "seeds": sorted(s for _, s, _ in values)})
    return out


def guard_fallbacks(rows: list[dict]) -> dict[str, list[dict]]:
    """Conditions with a run whose retrieval guard left no eligible checkpoint.

    Such a run returns its epoch-0 state (``sync_no_eligible_checkpoint``): post-adaptation
    metrics come from an untrained adapter and pre-adaptation error equals retrieval-only, so
    it would pass the pre-adaptation feasibility test without being a trained candidate.
    """
    out: dict[str, list[dict]] = {}
    for r in rows:
        if r.get("sync_no_eligible_checkpoint"):
            out.setdefault(condition_id(r), []).append({"task": r["dataset"], "seed": r["seed"],
                                                        "reason": "no eligible checkpoint under the retrieval guard"})
    return out


def select(args) -> None:
    seeds, prereg = getattr(args, "seeds", None), getattr(args, "prereg", None)
    criterion, bound = getattr(args, "criterion", None), getattr(args, "feasibility_pre", None)
    rows = selection_rows(args.exp_dir, args.exclude_condition, seeds, prereg)
    ineligible = pre_feasibility(rows, bound) if bound is not None else {}
    fallbacks = guard_fallbacks(rows)
    table, winners, glob, n_seeds = rank_conditions(rows, criterion=criterion or "err",
                                                    ineligible=set(ineligible) | set(fallbacks))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    finalists = [g["condition"] for g in glob[: args.finalists]]
    for c in list(winners.values()) + list(args.reference or []):
        if c not in finalists:
            finalists.append(c)
    selection = {
        "rule": RULES[criterion or "err"],
        "test_stream_used": False, "development_seeds": sorted({r["seed"] for r in rows}),
        "seeds_per_condition": n_seeds, "global_winner": glob[0]["condition"], "global_ranking": glob,
        "per_task_winner": winners, "finalists": finalists, "references": list(args.reference or []),
    }
    # Optional records appear only when their option is given, so default output is unchanged.
    if criterion:
        selection["criterion"] = criterion
    if seeds is not None:
        selection["seeds_requested"] = sorted(seeds)
    if prereg:
        selection["prereg"] = refuse_prereg_seeds(rows, prereg, seeds)
    if bound is not None:
        selection["feasibility_pre"] = {
            "bound": bound, "rule": "ineligible if on any task mean validation pre-adaptation error > "
                                    "bound * same-arm retrieval-only mean validation error + 1/n_val; "
                                    "classification errors counted in whole validation cases per seed",
            "ineligible": ineligible}
    if fallbacks:
        selection["guard_fallback_ineligible"] = fallbacks
    (out / "selection.json").write_text(json.dumps(selection, indent=2))
    import csv
    with (out / "validation_table.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    print(f"global winner: {glob[0]['condition']} (mean rank {glob[0]['mean_rank']:.2f} of {len(glob)})")
    for g in glob[:10]:
        print(f"  {g['mean_rank']:6.2f}  won {g['tasks_won']:2d}  {g['condition']}")
    if ineligible:
        print(f"{len(ineligible)} ineligible by --feasibility-pre {bound}:")
        for c, why in ineligible.items():
            print(f"  {c}: " + ", ".join(f"{w['task']} {w['val_pre_error']:.4f} > {w['limit']:.4f}" for w in why))
    if args.confirm_template:
        write_confirmation(args, finalists)


def write_confirmation(args, finalists: list[str]) -> None:
    stages = [load_config(Path(p)) for p in args.stage_config]
    stage = stages[0]
    conf = yaml.safe_load(Path(args.confirm_template).read_text(encoding="utf-8"))
    by_name = {}
    for st in stages:
        for c in st["conditions"]:
            name = c.get("name", c["schedule"])
            if name in by_name and by_name[name] != c:
                raise SystemExit(f"condition {name} differs between stages")
            by_name[name] = c
    arms = {a["name"]: a for a in stage.get("arms", [])}
    wanted_arms, sync_conds = [], []
    for f in finalists:
        arm, name = f.split("::", 1) if "::" in f else ("", f)
        if arm and arm not in wanted_arms:
            wanted_arms.append(arm)
        if name.startswith("frozen_reuse__") or name in [c.get("name") for c in sync_conds]:
            continue
        cond = dict(by_name[name])
        cond.setdefault("name", name)
        sync_conds.append(cond)
    for key in ("core", "core_by_dataset", "core_selection_source", "sync", "reuse", "reuse_conditions",
                "datasets", "statistics", "splits"):
        if key in stage and key not in conf:
            conf[key] = stage[key]
    conf["arms"] = [arms[a] for a in wanted_arms] if wanted_arms else stage.get("arms")
    conf["conditions"] = sync_conds
    conf["selection_source"] = str(Path(args.out) / "selection.json")
    if getattr(args, "confirm_experiment", None):
        conf["experiment"] = args.confirm_experiment
    # Conditions confirmed in an earlier batch are not rerun.
    skip = set(getattr(args, "skip_condition", None) or [])
    conf["conditions"] = [c for c in conf["conditions"] if c["name"] not in skip]
    if getattr(args, "skip_reuse", False):
        conf["reuse_conditions"] = []
    Path(args.confirm_config).write_text("# Generated by tools/t1_schedule_search.py select\n"
                                         + yaml.safe_dump(conf, sort_keys=False), encoding="utf-8")
    print(f"wrote {args.confirm_config}: {len(conf['arms'])} arm(s), {len(conf['conditions'])} schedules"
          f"{' + frozen reuse' if conf.get('reuse_conditions') else ''}")


# ----------------------------------------------------------------------------- loso

def held_out_errors(rows: list[dict]) -> dict:
    """(task, condition, seed) -> (test error, task type, n_test) of every row (retrieval-only included)."""
    n_test = stream_sizes(rows, "test")
    return {(r["dataset"], condition_id(r), r["seed"]):
            (error_of(r, "test"), r["task_type"], n_test.get((r["dataset"], r["seed"]))) for r in rows}


def log_ratio(test: dict, task: str, cond: str, ref: str, seed: int) -> dict:
    """Held-out log((e + eps) / (e_ref + eps)); eps = 0.5/n_test (classification) or 0 (regression)."""
    for c in (cond, ref):
        if (task, c, seed) not in test:
            raise SystemExit(f"no test row for {task} {c} seed {seed}")
    (e, kind, n), (e_ref, _, _) = test[task, cond, seed], test[task, ref, seed]
    if kind == "classification" and not n:
        raise SystemExit(f"no test_n for {task} seed {seed}")
    eps = 0.5 / n if kind == "classification" else 0.0
    if e + eps <= 0 or e_ref + eps <= 0:
        raise SystemExit(f"zero regression error for {task} seed {seed}; the log ratio is undefined")
    return {"seed": seed, "task": task, "error": e, "reference_error": e_ref, "log_ratio": float(np.log((e + eps) / (e_ref + eps)))}


def summarize_ratios(cells: list[dict], level: float = 0.8) -> dict:
    """Mean, standard error and two-sided ``level`` t interval over (seed, task) cells, plus per-task means."""
    v = np.array([c["log_ratio"] for c in cells], dtype=float)
    n, mean = len(v), float(v.mean())
    se = float(v.std(ddof=1) / np.sqrt(n)) if n > 1 else None
    half = float(stats.t.ppf(0.5 + level / 2, n - 1)) * se if se is not None else None
    per_task = {}
    for c in cells:
        per_task.setdefault(c["task"], []).append(c)
    return {"n": n, "mean_log_ratio": mean, "se": se, "gmr": float(np.exp(mean)),
            f"interval_{round(level * 100)}": None if half is None else [mean - half, mean + half],
            "per_task": {t: {"mean_log_ratio": float(np.mean([c["log_ratio"] for c in cs])), "cells": cs}
                         for t, cs in sorted(per_task.items())}}


def loso(args) -> None:
    seeds = sorted(set(args.seeds))
    if len(seeds) < 2:
        raise SystemExit("loso needs at least two development seeds")
    rows = selection_rows(args.exp_dir, args.exclude_condition, seeds, args.prereg)
    test = held_out_errors(rows)
    report = {"seeds": seeds, "selection_stream": "validation of the other development seeds",
              "held_out_stream": "test stream of the held-out development seed (declared inner held-out use)",
              "effect": "log((e + eps) / (e_ref + eps)), eps = 0.5/n_test (classification), 0 (regression); "
                        "summaries over (seed, task) cells", "results": []}
    if args.prereg:
        report["prereg"] = refuse_prereg_seeds(rows, args.prereg, seeds)
    if args.incumbent:
        if not args.challengers:
            raise SystemExit("--incumbent needs --challengers")
        tasks = sorted({t for t, c, s in test if c == args.incumbent and s in seeds})
        report.update(mode="compare", incumbent=args.incumbent, challengers=args.challengers)
        for ch in args.challengers:
            cells = [log_ratio(test, t, ch, args.incumbent, s) for s in seeds for t in tasks]
            report["results"].append({"challenger": ch, **summarize_ratios(cells)})
    else:
        if not args.reference:
            raise SystemExit("loso needs --reference (rule comparison) or --incumbent with --challengers")
        report.update(mode="rules", reference=args.reference)
        cells = {}
        for rule in args.rules:
            for held in seeds:
                _, winners, glob, _ = rank_conditions(rows, seeds=[s for s in seeds if s != held], criterion=rule,
                                                      ineligible=set(guard_fallbacks(rows)))
                for scope in args.scope:
                    for t, w in sorted(winners.items()):
                        sel = glob[0]["condition"] if scope == "global" else w
                        cells.setdefault((rule, scope), []).append({**log_ratio(test, t, sel, args.reference, held), "selected": sel})
        first = next(iter(cells))
        for (rule, scope), cs in cells.items():
            res = {"rule": rule, "scope": scope, **summarize_ratios(cs)}
            if (rule, scope) != first:  # paired by (seed, task) against the first rule/scope given
                base = {(b["seed"], b["task"]): b["log_ratio"] for b in cells[first]}
                diff = [{"seed": a["seed"], "task": a["task"], "log_ratio": a["log_ratio"] - base[a["seed"], a["task"]]}
                        for a in cs]
                res["difference_vs_first"] = {k: v for k, v in summarize_ratios(diff).items() if k != "per_task"}
            report["results"].append(res)
        report["first"] = {"rule": first[0], "scope": first[1]}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    for r in report["results"]:
        name = r.get("challenger") or f"{r['rule']}/{r['scope']}"
        se = "n/a" if r["se"] is None else f"{r['se']:.4f}"
        print(f"{name}: mean log ratio {r['mean_log_ratio']:+.4f} (SE {se}, GMR {r['gmr']:.3f}, n={r['n']})")
    print(f"wrote {out}")


# ----------------------------------------------------------------------------- report

def pre_error(row: dict, stream: str) -> float:
    """Pre-adaptation (retrieval-only) error of the same run."""
    if row["task_type"] == "classification":
        return 1.0 - float(row[f"{stream}_accuracy_pre"])
    return float(row[f"{stream}_rmse_pre"])


def report(args) -> None:
    rows, seen = [], set()
    for d in args.exp_dir:  # later batches may repeat the retrieval-only/reuse rows of earlier ones
        for r in load_rows(Path(d)):
            key = (r["dataset"], condition_id(r), r["seed"])
            if key not in seen:
                seen.add(key)
                rows.append(r)
    selection = json.loads(Path(args.selection).read_text())
    post = {(r["dataset"], condition_id(r), r["seed"]): error_of(r, "test") for r in rows}
    pre = {(r["dataset"], condition_id(r), r["seed"]): pre_error(r, "test") for r in rows}
    tasks = sorted({k[0] for k in post})
    conds = sorted({k[1] for k in post})

    def mean(store, t, c):
        v = [x for (tt, cc, _), x in store.items() if (tt, cc) == (t, c)]
        return float(np.mean(v)) if v else None

    def pick(name, t):
        return {"global_selected": selection["global_winner"],
                "per_task_selected": selection["per_task_winner"].get(t)}.get(name, name)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = [{"task": t, "condition": c, "test_error_mean": mean(post, t, c), "test_pre_error_mean": mean(pre, t, c),
                "n": sum(1 for k in post if k[:2] == (t, c))} for t in tasks for c in conds if mean(post, t, c) is not None]
    # Test ranks per task (reported only; selection was fixed beforehand).
    ranks = {c: [] for c in conds}
    for t in tasks:
        present = [c for c in conds if mean(post, t, c) is not None]
        for i, c in enumerate(sorted(present, key=lambda c: mean(post, t, c)), 1):
            ranks[c].append(i)
    base = args.baseline
    agg = []
    for c in conds:
        rel = [(mean(post, t, c) - mean(post, t, base)) / mean(post, t, base) for t in tasks
               if mean(post, t, c) is not None and mean(post, t, base)]
        rel_pre = [(mean(pre, t, c) - mean(pre, t, base)) / mean(pre, t, base) for t in tasks
                   if mean(pre, t, c) is not None and mean(pre, t, base)]
        agg.append({"condition": c, "mean_test_rank": float(np.mean(ranks[c])), "tasks": len(ranks[c]),
                    "tasks_better_than_baseline": sum(x < 0 for x in rel), "tasks_worse_than_baseline": sum(x > 0 for x in rel),
                    "median_relative_error_vs_baseline": float(np.median(rel)) if rel else float("nan"),
                    "median_relative_pre_error_vs_baseline": float(np.median(rel_pre)) if rel_pre else float("nan")})
    agg.sort(key=lambda g: g["mean_test_rank"])
    paired = []
    for ref in selection["references"] + [c for c in conds if c.endswith("retrieval_only")]:
        for name in ("global_selected", "per_task_selected"):
            per_task = []
            for t in tasks:
                c = pick(name, t)
                d = [post[t, c, s] - post[t, ref, s] for (tt, cc, s) in post if (tt, cc) == (t, c) and (t, ref, s) in post]
                if d:
                    per_task.append({"task": t, "condition": c, "mean_error_difference": float(np.mean(d)),
                                     "seeds_better": sum(x < 0 for x in d), "seeds_worse": sum(x > 0 for x in d), "n": len(d)})
            rel = [p["mean_error_difference"] / mean(post, p["task"], ref) for p in per_task if mean(post, p["task"], ref)]
            paired.append({"contender": name, "reference": ref,
                           "tasks_better": sum(p["mean_error_difference"] < 0 for p in per_task),
                           "tasks_worse": sum(p["mean_error_difference"] > 0 for p in per_task),
                           "tasks_tied": sum(p["mean_error_difference"] == 0 for p in per_task),
                           "median_relative_difference": float(np.median(rel)) if rel else float("nan"),
                           "wilcoxon_p": float(stats.wilcoxon(rel).pvalue) if len(rel) > 1 and any(rel) else float("nan"),
                           "per_task": per_task})
    (out / "test_summary.json").write_text(json.dumps({"rows": summary, "aggregate": agg, "paired": paired,
        "error": "1-accuracy (classification) or standardized RMSE (regression)", "baseline": base,
        "selection": args.selection}, indent=2))
    short = lambda c: c.split("::", 1)[-1]
    md = ["# Schedule confirmation (test stream, evaluation seeds)", "",
          "Error: 1 - accuracy (classification) or train-standardized RMSE (regression), mean over seeds; lower is better. "
          f"Relative columns are medians over tasks of (condition - `{short(base)}`) / `{short(base)}`; "
          "\"pre\" is the same runs' retrieval-only output. Selection used validation only, before these results.", "",
          "| condition | mean test rank | better / worse tasks vs baseline | median rel. error | median rel. pre-adaptation error |",
          "|---|---|---|---|---|"]
    for g in agg:
        md.append(f"| {short(g['condition'])} | {g['mean_test_rank']:.2f} | {g['tasks_better_than_baseline']} / "
                  f"{g['tasks_worse_than_baseline']} | {g['median_relative_error_vs_baseline']:+.1%} | "
                  f"{g['median_relative_pre_error_vs_baseline']:+.1%} |")
    key = [c for c in conds if c.endswith("retrieval_only")] + list(selection["references"]) + ["global_selected", "per_task_selected"]
    md += ["", "## Per task", "", "| task | " + " | ".join(short(k) for k in key) + " | per-task selection |",
           "|---" * (len(key) + 2) + "|"]
    for t in tasks:
        cells = [mean(post, t, pick(k, t)) for k in key]
        best = min(v for v in cells if v is not None)
        md.append(f"| {t} | " + " | ".join("" if v is None else (f"**{v:.4f}**" if v == best else f"{v:.4f}") for v in cells)
                  + f" | {short(selection['per_task_winner'].get(t, ''))} |")
    md += ["", "## Selected schedules vs references", ""]
    for p in paired:
        md.append(f"- {p['contender']} vs `{short(p['reference'])}`: better on {p['tasks_better']}, worse on {p['tasks_worse']}, "
                  f"tied on {p['tasks_tied']} tasks; median relative difference {p['median_relative_difference']:+.1%}, "
                  f"Wilcoxon p = {p['wilcoxon_p']:.3f}")
    (out / "test_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--template", required=True)
    p.add_argument("--grid", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--experiment", help="experiment name of the generated config")
    s = sub.add_parser("select")
    s.add_argument("exp_dir", nargs="+", help="one or more stage experiment directories (same seeds)")
    s.add_argument("--out", required=True)
    s.add_argument("--finalists", type=int, default=3)
    s.add_argument("--reference", action="append", help="condition id always confirmed, e.g. tuned::alternating_default")
    s.add_argument("--stage-config", action="append", help="every stage config whose conditions may be finalists")
    s.add_argument("--confirm-template")
    s.add_argument("--confirm-config")
    s.add_argument("--confirm-experiment")
    s.add_argument("--skip-condition", action="append", help="schedule name confirmed in an earlier batch")
    s.add_argument("--skip-reuse", action="store_true", help="frozen reuse confirmed in an earlier batch")
    s.add_argument("--exclude-condition", action="append", help="condition id left out of ranking (e.g. an exact repeat)")
    s.add_argument("--seeds", type=int, nargs="+", help="use only rows of these development seeds")
    s.add_argument("--prereg", help="pre-registration JSON; refuse rows of its evaluation/retired/reserved seeds")
    s.add_argument("--criterion", choices=sorted(RULES), help="per-task ordering (default err: error, then L_post)")
    s.add_argument("--feasibility-pre", type=float, metavar="B",
                   help="ineligible if on any task mean val pre-adaptation error > B * retrieval-only + 1/n_val")
    o = sub.add_parser("loso", help="leave-one-development-seed-out evaluation of selection rules")
    o.add_argument("exp_dir", nargs="+")
    o.add_argument("--seeds", type=int, nargs="+", required=True, help="development seeds")
    o.add_argument("--rules", nargs="+", choices=sorted(RULES), default=["err"])
    o.add_argument("--scope", nargs="+", choices=("global", "pertask"), default=["global"])
    o.add_argument("--reference", help="condition id each selection is scored against")
    o.add_argument("--incumbent", help="compare mode: condition id the challengers are scored against")
    o.add_argument("--challengers", nargs="+", help="compare mode: condition ids; nothing is selected")
    o.add_argument("--prereg")
    o.add_argument("--exclude-condition", action="append")
    o.add_argument("--out", required=True, help="output JSON")
    r = sub.add_parser("report")
    r.add_argument("exp_dir", nargs="+")
    r.add_argument("--selection", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--baseline", default="tuned::alternating_default", help="condition for relative columns")
    args = ap.parse_args()
    {"prepare": prepare, "select": select, "loso": loso, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
