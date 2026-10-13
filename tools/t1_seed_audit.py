"""Seed audit for the T1 pre-registration (docs/T1_PREREG_20261013.md).

    python tools/t1_seed_audit.py <config.yaml...> --prereg configs/t1/prereg_20261013.json [--check-splits]

OpenML tasks (``tabarena_<task id>``): seed s selects the outer split
((s // folds) % repeats, s % folds) of the task's own split dimensions, as in
``model.t1.data._load_raw``. Evaluation seeds must map to splits that no development,
retired or reserved seed uses, and to no split twice. Other tasks draw a fresh random
holdout per seed; their test sizes are reported and tasks under 100 test rows listed
(they get the pre-registration's extra small-task seeds). A pre-registered training cap
(``max_train``) must reach both code paths: the runner reads only the dataset entry's
``max_train``, the HPO grid only ``splits.max_train``. ``--check-splits`` builds every split
through the runner (``Runner.data_for``) and the grid's own call
(``make_splits(name, task_type, seed, **splits)``, ``tools/t1_grid_search.py``, which ignores
the dataset entry) and requires identical index hashes. Exits 1 on any violation.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

FORBIDDEN_ROLES = ("evaluation", "evaluation_small_tasks", "retired", "reserved")  # never tuned on
AVOIDED_ROLES = ("development", "development_reserve", "retired", "reserved")  # splits evaluation must not reuse
SMALL_TEST = 100


def load_prereg(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def seed_roles(prereg: dict) -> dict[int, list[str]]:
    """Seed -> every pre-registered role listing it."""
    out: dict[int, list[str]] = {}
    for role, seeds in prereg["seeds"].items():
        for s in seeds:
            out.setdefault(int(s), []).append(role)
    return dict(sorted(out.items()))


def role_violations(roles: dict[int, list[str]]) -> list[str]:
    return [f"seed {s} is listed under several roles: {', '.join(r)}" for s, r in roles.items() if len(r) > 1]


def outer_split(seed: int, repeats: int, folds: int) -> tuple[int, int]:
    """OpenML outer (repeat, fold) of a seed, as ``model.t1.data._load_raw`` maps it."""
    return (seed // folds) % repeats, seed % folds


def outer_split_audit(name: str, repeats: int, folds: int, roles: dict[int, list[str]]):
    """Mapping table of every pre-registered seed and the evaluation-seed violations."""
    table = [{"seed": s, "roles": r, "split": list(outer_split(s, repeats, folds))} for s, r in roles.items()]
    avoided: dict[tuple, list[str]] = {}
    for s, rs in roles.items():
        for role in rs:
            if role in AVOIDED_ROLES:
                avoided.setdefault(outer_split(s, repeats, folds), []).append(f"{role} seed {s}")
    seen: dict[tuple, int] = {}
    violations = []
    for s, rs in roles.items():
        if "evaluation" not in rs:
            continue
        split = outer_split(s, repeats, folds)
        if split in avoided:
            violations.append(f"{name}: evaluation seed {s} uses outer split {split} of {', '.join(avoided[split])}")
        if split in seen:
            violations.append(f"{name}: evaluation seeds {seen[split]} and {s} share outer split {split}")
        seen.setdefault(split, s)
    return table, violations


def openml_task_id(name: str) -> int | None:
    """Task id of a name whose split comes from an OpenML task; ``openml_<dataset>`` names use random holdouts."""
    return int(name.split("_")[1]) if name.startswith("tabarena_") else None


def openml_dimensions(task_id: int) -> dict:
    import openml
    task = openml.tasks.get_task(task_id, download_splits=True)
    repeats, folds, _ = task.get_split_dimensions()
    return {"task": task, "repeats": int(repeats), "folds": int(folds)}


def config_seeds(cfg: dict) -> list[int]:
    return sorted({int(s) for k in ("seeds", "development_seeds", "evaluation_seeds") for s in cfg.get(k) or []})


def runner_for(cfg: dict):
    """A runner whose ``data_for`` is the real one; ``__init__`` is skipped (it writes source snapshots)."""
    from tools.t1_run import Runner
    runner = object.__new__(Runner)
    runner.cfg = cfg
    return runner


def quiet(fn, *args, **kwargs):
    """Call a data builder without its loaders' progress prints."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def grid_splits(cfg: dict) -> dict:
    """Keyword arguments of the HPO grid's ``make_splits`` call: ``splits`` verbatim, as
    ``tools/t1_grid_search.py`` passes ``**study["splits"]``. A dataset entry's ``max_train``
    does not reach the grid."""
    return dict(cfg["splits"])


def split_hashes(data) -> dict[str, str]:
    idx = data.meta.get("split_indices")
    if idx is None:  # synthetic generators have no index split: hash the streams themselves
        streams = {s: (getattr(data, f"X_{s}"), getattr(data, f"y_{s}")) for s in ("train", "val", "test")}
        return {s: hashlib.sha256(b"".join(t.numpy().tobytes() for t in ts)).hexdigest() for s, ts in streams.items()}
    return {k: hashlib.sha256(np.asarray(v, dtype=np.int64).tobytes()).hexdigest() for k, v in idx.items()}


def prereg_caps(prereg: dict) -> dict[tuple, int | None]:
    caps = {}
    for group in ("primary_legacy", "primary_transfer"):
        for t in prereg.get("tasks", {}).get(group, []):
            caps[t["name"], t["task_type"]] = t.get("max_train")
    return caps


def audit(config_paths, prereg_path, check_splits=False, split_seeds=None) -> dict:
    from model.t1.data import make_splits
    from tools.t1_run import load_config_raw
    prereg = load_prereg(prereg_path)
    roles = seed_roles(prereg)
    by_role = {r: sorted(int(s) for s in v) for r, v in prereg["seeds"].items()}
    violations, notes = role_violations(roles), []
    configs = [(str(p), load_config_raw(Path(p))) for p in config_paths]
    entries = [(p, cfg, ds) for p, cfg in configs for ds in cfg.get("datasets", [])]
    caps = prereg_caps(prereg)
    report = {"prereg": str(prereg_path), "prereg_sha256": hashlib.sha256(Path(prereg_path).read_bytes()).hexdigest(),
              "seed_roles": by_role, "configs": {p: {"seeds": config_seeds(cfg), "seed_roles": {
                  s: roles.get(s, ["unlisted"]) for s in config_seeds(cfg)}} for p, cfg in configs},
              "openml_tasks": {}, "holdout_tasks": [], "small_tasks": [], "split_checks": []}
    evaluation = by_role.get("evaluation", [])
    holdout_cache: dict = {}
    for path, cfg, ds in entries:
        key = f"{ds['task_type']}__{ds['name']}"
        cap = caps.get((ds["name"], ds["task_type"]))
        if cap is not None:
            if ds.get("max_train") != cap:
                violations.append(f"{path} {key}: dataset-entry max_train {ds.get('max_train')} != pre-registered {cap} "
                                  "(Runner.data_for reads only the entry, not splits.max_train)")
            grid_cap = cfg.get("splits", {}).get("max_train")
            if grid_cap != cap:
                violations.append(f"{path} {key}: splits.max_train {grid_cap} != pre-registered {cap} "
                                  "(the HPO grid reads only splits.max_train, not the dataset entry)")
        tid = openml_task_id(ds["name"])
        if tid is not None:
            if key in report["openml_tasks"]:
                continue
            dims = openml_dimensions(tid)
            table, bad = outer_split_audit(key, dims["repeats"], dims["folds"], roles)
            n_test = {s: len(dims["task"].get_train_test_split_indices(repeat=r, fold=f)[1])
                      for s in evaluation for r, f in [outer_split(s, dims["repeats"], dims["folds"])]}
            report["openml_tasks"][key] = {"task_id": tid, "repeats": dims["repeats"], "folds": dims["folds"],
                                           "mapping": table, "n_test": n_test}
            violations += bad
            if n_test and min(n_test.values()) < SMALL_TEST:
                notes.append(f"{key}: OpenML task with fewer than {SMALL_TEST} test rows; extra seeds would reuse "
                             "development splits, so the small-task seeds do not apply")
            continue
        test_frac = cfg.get("splits", {}).get("test_frac")
        cache_key = (ds["name"], ds["task_type"], test_frac, json.dumps(ds.get("synthetic"), sort_keys=True))
        if cache_key not in holdout_cache:
            runner = runner_for(cfg)
            holdout_cache[cache_key] = {s: int(quiet(runner.data_for, ds, s).X_test.shape[0]) for s in evaluation}
        n_test = holdout_cache[cache_key]
        small = bool(n_test) and min(n_test.values()) < SMALL_TEST
        report["holdout_tasks"].append({"task": key, "config": path, "test_frac": test_frac, "n_test": n_test,
                                        "small": small})
        if small and key not in report["small_tasks"]:
            report["small_tasks"].append(key)
    listed = {f"{t}__{n}" for (n, t) in caps}
    present = {f"{ds['task_type']}__{ds['name']}" for _, _, ds in entries}
    if listed - present:
        notes.append(f"pre-registered tasks in no given config: {sorted(listed - present)}")
    if check_splits:
        base_seeds = sorted({s for r in ("development", "development_reserve", "evaluation") for s in by_role.get(r, [])})
        for path, cfg, ds in entries:
            key = f"{ds['task_type']}__{ds['name']}"
            if "splits" not in cfg:
                violations.append(f"{path} {key}: no splits section")
                continue
            seeds = split_seeds or sorted(set(base_seeds) | set(config_seeds(cfg))
                                          | (set(by_role.get("evaluation_small_tasks", [])) if key in report["small_tasks"] else set()))
            runner, kwargs, tid = runner_for(cfg), grid_splits(cfg), openml_task_id(ds["name"])
            for s in seeds:
                a, b = quiet(runner.data_for, ds, s), quiet(make_splits, ds["name"], ds["task_type"], s, **kwargs)
                ha, hb = split_hashes(a), split_hashes(b)
                check = {"config": path, "task": key, "seed": s, "identical": ha == hb, "hashes": ha,
                         "n": {k: len(v) for k, v in (a.meta.get("split_indices") or {}).items()}}
                if ha != hb:
                    differ = sorted(k for k in set(ha) | set(hb) if ha.get(k) != hb.get(k))
                    check["grid_hashes"] = hb
                    check["grid_n"] = {k: len(v) for k, v in (b.meta.get("split_indices") or {}).items()}
                    violations.append(f"{path} {key} seed {s}: runner and grid splits differ in {differ}")
                if tid is not None:
                    dims = report["openml_tasks"][key]
                    want = list(outer_split(s, dims["repeats"], dims["folds"]))
                    got = [a.meta.get("outer_repeat"), a.meta.get("outer_fold")]
                    check["outer_split"] = got
                    if got != want:
                        violations.append(f"{path} {key} seed {s}: data uses outer split {got}, audit mapping {want}")
                report["split_checks"].append(check)
    report["violations"], report["notes"] = violations, notes
    return report


def print_report(report: dict) -> None:
    print(f"pre-registration {report['prereg']}")
    for role, seeds in report["seed_roles"].items():
        print(f"  {role:24s} {seeds}")
    layouts: dict = {}
    for key, t in report["openml_tasks"].items():
        layouts.setdefault((t["repeats"], t["folds"]), []).append(key)
    for (repeats, folds), keys in layouts.items():
        mapping = report["openml_tasks"][keys[0]]["mapping"]
        print(f"\nOpenML outer split (repeat, fold) at {repeats} repeats x {folds} folds: {', '.join(keys)}")
        for m in mapping:
            print(f"  seed {m['seed']:3d}  {'+'.join(m['roles']):24s} {tuple(m['split'])}")
    if report["holdout_tasks"]:
        print(f"\nRandom-holdout tasks: test rows on evaluation seeds (small: < {SMALL_TEST})")
        for h in report["holdout_tasks"]:
            n = sorted(set(h["n_test"].values()))
            print(f"  {h['task']:40s} n_test {n}{'  SMALL' if h['small'] else ''}  ({Path(h['config']).name})")
    print(f"\nsmall tasks (extra evaluation seeds): {report['small_tasks'] or 'none'}")
    if report["split_checks"]:
        same = sum(c["identical"] for c in report["split_checks"])
        print(f"split checks: {same}/{len(report['split_checks'])} identical between runner and grid")
    for n in report["notes"]:
        print(f"note: {n}")
    for v in report["violations"]:
        print(f"VIOLATION: {v}")
    print("FAILED" if report["violations"] else "OK")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("configs", nargs="+", help="run or grid configs whose datasets are audited")
    ap.add_argument("--prereg", required=True)
    ap.add_argument("--check-splits", action="store_true", help="build every split through both code paths")
    ap.add_argument("--seeds", type=int, nargs="+", help="seeds for --check-splits (default: development, "
                    "reserve, evaluation, the configs' seeds, small-task seeds for small tasks)")
    ap.add_argument("--out", help="write the full report as JSON")
    args = ap.parse_args(argv)
    report = audit(args.configs, args.prereg, args.check_splits, args.seeds)
    print_report(report)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 1 if report["violations"] else 0


if __name__ == "__main__":
    sys.exit(main())
