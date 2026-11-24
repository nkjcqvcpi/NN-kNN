"""Audit the run-unit tree of a ``tools/t1_run.py`` experiment.

    python tools/t1_confirm_audit.py <NN-KNN/runs/t1/<study>/<experiment>> [--logs JOB_LOG_DIR] [--json OUT]
        [--expect-fingerprint TASK_OR_GLOB=SHA256 ...] [--exclude DATASET_GLOB ...]

``tools/t1_run.py --study`` records one unit per run and no second copy of its row, so rows cannot go missing or be
duplicated. The audit checks every unit's status (failed or incomplete units are crashed runs, unless they changed
within ``--settle-seconds``; invalid ones fail), rebuilds each complete run's row from its ``manifest`` and
``metrics`` records and requires the identical serialized ``row`` record, checks the unit path against the row and
the source fingerprints (provenance: several fingerprints, or a source that changed during a run, are listed, not
failed; owner's order 2026-11-23) and, with ``--logs`` (the t1_parallel job-log directory), that a job log whose receipt has
exit code 0 holds the run's ``[done]`` line, equal to the line rebuilt from its metrics. Stage units (``_<stage>``:
shared training such as the trained core, no row) are checked for their status only. Run manifests map each task
(``<task_type>__<name>``) to its source fingerprints; ``--expect-fingerprint`` (task key or dataset key, glob
allowed; a hex prefix of 12+ digits) fails a task with any other fingerprint; a task is checked against its most
specific pattern (exact task key, exact dataset key, then the first matching glob). Exits 1 when anything is
crashed, unrebuildable or unexpected. Legacy result folders (``runs__*.jsonl`` shards and run directories) are not
read; old results are left as they are.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.t1.records import require_unit_tree  # noqa: E402

DONE_KEYS = ("test_accuracy_pre", "test_accuracy_post", "test_rmse_pre", "test_accuracy_rare", "n_cases")
FINGERPRINT = re.compile(r"[0-9a-f]{12,64}")


class RebuildError(ValueError):
    """A run unit whose records cannot reproduce its row."""


# --------------------------------------------------------------------------- rebuild


def config_id(cfg) -> str:
    """``model.common.artifacts.config_id`` of a JSON-loaded config (already JSON-safe)."""
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def dataset_key(config: dict, ds: dict) -> str:
    """``Runner.dataset_key``: role-qualified when the name has several task types or ``qualified``."""
    roles = {d["task_type"] for d in config["datasets"] if d["name"] == ds["name"]}
    return f"{ds['task_type']}__{ds['name']}" if len(roles) > 1 or ds.get("qualified") else ds["name"]


def row_line(row: dict) -> str:
    """A row serialized as ``Runner.finish`` writes it, without the line terminator."""
    return json.dumps(row)


def done_line(run_id: str, metrics: dict) -> str:
    """The ``[done]`` line ``Runner.finish`` prints after appending the row (its expression, unchanged)."""
    return f"[done] {run_id} " + " ".join(f"{k}={metrics[k]:.4f}" for k in DONE_KEYS
                                          if isinstance(metrics.get(k), (int, float)))


# --------------------------------------------------------------------------- job logs


def done_lines(logs: Path) -> dict:
    """{run id: [(log, line)]} for every ``[done]`` line in ``logs/*.log`` (job logs and kept attempts)."""
    out = {}
    for log in sorted(logs.glob("*.log")):
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            at = line.find("[done] ")
            if at >= 0:
                out.setdefault(line[at:].split(" ", 2)[1], []).append((log, line[at:]))
    return out


def receipt_exit_code(log: Path):
    """``exit_code`` of the t1_parallel receipt beside ``log``; None without one (a kept attempt log)."""
    try:
        return json.loads(log.with_suffix(".json").read_text(encoding="utf-8"))["exit_code"]
    except (OSError, ValueError, KeyError, TypeError):
        return None


def done_evidence(index: dict, run_id: str, expected: str) -> dict:
    """Whether the job logs show ``finish`` completed for ``run_id``: a ``[done]`` line equal to
    ``expected`` (rebuilt from metrics.json, so the metrics are cross-checked) in a log whose receipt
    has exit code 0, and no ``[done]`` line for the run that differs."""
    lines = [{"log": log.name, "exit_code": receipt_exit_code(log), "matches": line == expected}
             for log, line in index.get(run_id, [])]
    if not lines:
        reason = f"no [done] {run_id} line in any job log: finish() may not have completed"
    elif not all(x["matches"] for x in lines):
        reason = "[done] line disagrees with metrics.json in " + ", ".join(x["log"] for x in lines if not x["matches"])
    elif not any(x["exit_code"] == 0 for x in lines):
        reason = "no log with the [done] line has a receipt with exit code 0"
    else:
        reason = None
    return {"done_lines": lines, "passed": reason is None, "reason": reason}


# --------------------------------------------------------------------------- scan


def _matches(name: str, patterns) -> bool:
    return any(fnmatch.fnmatchcase(name, p) for p in patterns)


def _value_diff(row: dict, rebuilt: dict) -> str:
    if row == rebuilt and list(row) != list(rebuilt):
        return "key order differs"
    fields = sorted(k for k in row.keys() | rebuilt.keys()
                    if k not in row or k not in rebuilt or json.dumps(row[k]) != json.dumps(rebuilt[k]))
    return f"fields differ: {', '.join(fields)}" if fields else "serialization differs"


def audit_units(root, *, logs=None, exclude=(), settle_seconds: float = 300.0, expect=(),
                fail_identical_duplicates: bool = False, now: float | None = None) -> dict:
    """The audit of a run-unit tree (see the module docstring)."""
    from model.common.tracking import iter_runs, load_run
    from model.t1.records import record, unit_component
    root = require_unit_tree(root)
    now = time.time() if now is None else now
    rel = lambda p: Path(p).relative_to(root).as_posix()  # noqa: E731
    if logs is not None and not Path(logs).is_dir():
        raise SystemExit(f"job-log directory {logs} does not exist")
    infos = [i for i in iter_runs(root, status=None) if not _matches(Path(i.path).relative_to(root).parts[0], exclude)]
    if not infos:
        raise SystemExit(f"no run units in {root}")
    fingerprints, incomplete, rebuild_errors, changed, check = {}, [], [], [], {"rows": 0, "exact": 0, "mismatches": []}
    index = None if logs is None else done_lines(Path(logs))
    unproven, stages = [], 0
    for info in infos:
        path = Path(info.path)
        if info.status in ("failed", "incomplete", "empty"):
            age = now - max(f.stat().st_mtime for f in path.glob("events.out.tfevents.*")) if info.status != "empty" else 0.0
            incomplete.append({"run": rel(path), "run_dir": rel(path), "status": info.status,
                               "missing_files": [f"unit {info.status}" + (f": {info.reason}" if info.reason else "")],
                               "recent": age < settle_seconds, "row": False})
            continue
        if info.status != "complete":
            rebuild_errors.append({"run_dir": rel(path), "error": f"unit {info.status}: {info.reason}"})
            continue
        try:
            data = load_run(path, arrays=False)
            if "stage" in ((data.config or {}).get("condition") or {}):
                stages += 1  # shared training of several runs (tools/t1_run.py Runner.stage): no row of its own
                continue
            man, metrics, row = record(data, "manifest"), record(data, "metrics"), record(data, "row")
            rc, run = man["resolved_config"], man["run"]
            ds, seed, exp = rc["dataset"], rc["seed"], rc["experiment"]
            key, label = dataset_key(rc["config"], ds), row["condition"]
            cid = config_id(rc)
            if run["configuration_id"] != cid or run["id"] != f"{exp}-{key}-{label}-s{seed}-{cid[:8]}":
                raise RebuildError(f"run id {run['id']} does not match its configuration and condition {label}")
            if path.parts[-3:] != (unit_component(key), unit_component(label), f"s{seed}"):
                raise RebuildError(f"unit path does not match {key}/{label}/s{seed}")
            rebuilt = {"run_id": run["id"], "experiment": exp, "dataset": key, "dataset_name": ds["name"],
                       "task_type": man["task"]["task_type"], "seed": seed, "condition": label,
                       **{k: v for k, v in rc["condition"].items() if not isinstance(v, (dict, list))},
                       **{k: v for k, v in metrics.items() if not isinstance(v, (dict, list))}}
        except (OSError, ValueError, KeyError, TypeError) as e:
            rebuild_errors.append({"run_dir": rel(path), "error": f"{type(e).__name__}: {e}"})
            continue
        task = f"{man['task']['task_type']}__{ds['name']}"
        fp = str(run.get("source_fingerprint_sha256"))
        entry = fingerprints.setdefault(task, {"dataset_keys": [], "fingerprints": {}})
        if key not in entry["dataset_keys"]:
            entry["dataset_keys"].append(key)
        entry["fingerprints"][fp] = entry["fingerprints"].get(fp, 0) + 1
        if run.get("source_changed_during_run"):
            changed.append(rel(path))
        check["rows"] += 1
        if row_line(rebuilt) == row_line(row):
            check["exact"] += 1
        else:
            check["mismatches"].append({"run_dir": rel(path), "reason": _value_diff(row, rebuilt)})
        if index is not None:
            evidence = done_evidence(index, run["id"], done_line(run["id"], metrics))
            if not evidence["passed"]:
                unproven.append({"run_dir": rel(path), "run_id": run["id"], "evidence": evidence})
    check["passed"] = check["rows"] > 0 and check["exact"] == check["rows"]
    report = {"root": str(root), "logs": None if logs is None else str(logs), "settle_seconds": settle_seconds,
              "exclude": list(exclude), "fail_identical_duplicates": fail_identical_duplicates, "units": True,
              "run_dirs": len(infos), "stage_units": stages, "rebuildable_run_dirs": check["rows"] + len(rebuild_errors),
              "rows": check["rows"], "shards": {}, "unit_self_check": check, "done_lines_missing": unproven,
              "missing_rows": [], "pending_rows": [], "incomplete_run_dirs": incomplete,
              "orphan_rows": [], "duplicates": {"identical": [], "conflicting": []}, "rebuild_errors": rebuild_errors,
              "source_fingerprints": dict(sorted(fingerprints.items())),
              "source_changed_during_run": changed,
              "fingerprint_expectations": check_fingerprints(fingerprints, expect)}
    report["problems"] = problems(report, fail_identical_duplicates)
    if check["exact"] < check["rows"]:
        report["problems"].append(f"unit self-check {check['exact']}/{check['rows']} rows rebuild exactly")
    if unproven:
        report["problems"].append(f"{len(unproven)} units without a matching [done] line in a job log with exit code 0")
    report["ok"] = not report["problems"]
    return report


def check_fingerprints(fingerprints: dict, expect) -> list[dict]:
    """Each ``(pattern, fingerprint)``: the tasks it decides and their other fingerprints.

    A task is decided by its most specific pattern, as in tools/t1_confirm_report.py: an exact task key, then
    an exact dataset key, then the first matching glob. A pattern that matches tasks only where a more
    specific one decides them lists those tasks as ``overridden``.
    """
    def decider(t, keys):
        for exact in (lambda p: p == t, lambda p: p in keys):
            for i, (pattern, _) in enumerate(expect):
                if exact(pattern):
                    return i
        for i, (pattern, _) in enumerate(expect):
            if _matches(t, [pattern]) or any(_matches(k, [pattern]) for k in keys):
                return i
        return None

    decided = {t: decider(t, e["dataset_keys"]) for t, e in sorted(fingerprints.items())}
    out = []
    for i, (pattern, want) in enumerate(expect):
        matched = [t for t, e in sorted(fingerprints.items())
                   if _matches(t, [pattern]) or any(_matches(k, [pattern]) for k in e["dataset_keys"])]
        tasks = [t for t in matched if decided[t] == i]
        other = {t: {fp: n for fp, n in fingerprints[t]["fingerprints"].items() if not fp.startswith(want)}
                 for t in tasks}
        other = {t: fps for t, fps in other.items() if fps}
        out.append({"pattern": pattern, "fingerprint": want, "tasks": tasks,
                    "overridden": [t for t in matched if decided[t] != i], "other_fingerprints": other,
                    "passed": bool(matched) and not other})
    return out


def problems(report: dict, fail_identical_duplicates: bool = False) -> list[str]:
    """Every finding that fails the audit. Units that changed within the settle window (in progress) do not."""
    out = []
    count = lambda items, what: out.append(f"{len(items)} {what}") if items else None  # noqa: E731
    count([i for i in report["incomplete_run_dirs"] if not i["recent"]],
          f"failed or incomplete units, unchanged for {report['settle_seconds']:.0f}s (crashed and not rerun)")
    count(report["rebuild_errors"], "units whose records cannot rebuild a row")
    for e in report["fingerprint_expectations"]:
        if not e["tasks"] and not e["overridden"]:
            out.append(f"--expect-fingerprint {e['pattern']} matched no task")
        for task, fps in e["other_fingerprints"].items():
            out.append(f"{task}: source fingerprints {', '.join(f'{fp[:12]} ({n})' for fp, n in fps.items())}"
                       f" besides {e['fingerprint'][:12]}")
    return out


# --------------------------------------------------------------------------- report


def print_report(report: dict) -> None:
    print(f"audit {report['root']}: {report['run_dirs']} units ({report['stage_units']} stage units), {report['rows']} rows")
    c = report["unit_self_check"]
    print(f"{'run units':<40} {c['rows']:>5} {c['exact']:>5}  {'pass' if c['passed'] else 'FAIL'}")
    for m in c["mismatches"][:5]:
        print(f"    {m['run_dir']}: {m['reason']}")
    for m in report["done_lines_missing"][:5]:
        print(f"    {m['run_dir']}: {m['evidence']['reason']}")
    sections = [("failed or incomplete units (recent ones in progress, ignored)", report["incomplete_run_dirs"],
                 lambda i: f"{i['run_dir']}{' (recent)' if i['recent'] else ''}: {', '.join(i['missing_files'])}"),
                ("rebuild errors", report["rebuild_errors"], lambda e: f"{e['run_dir']}: {e['error']}")]
    for title, items, fmt in sections:
        if items:
            print(f"{title} ({len(items)}):")
            for item in items:
                print(f"  {fmt(item)}")
    print("source fingerprints per task:")
    for task, e in report["source_fingerprints"].items():
        print(f"  {task:<40} " + ", ".join(f"{fp[:12]} x{n}" for fp, n in e["fingerprints"].items()))
    for e in report["fingerprint_expectations"]:
        print(f"expect {e['pattern']}={e['fingerprint'][:12]}: {len(e['tasks'])} tasks"
              + (f" ({len(e['overridden'])} decided by more specific patterns)" if e["overridden"] else "")
              + f", {'pass' if e['passed'] else 'FAIL'}")
    deferred = sum(i["recent"] for i in report["incomplete_run_dirs"])
    later = f" ({deferred} units changed within the settle window, unjudged; audit again later)" if deferred else ""
    print("result: " + ("OK" + later if report["ok"] else "FAIL - " + "; ".join(report["problems"]) + later))


def _expectation(text: str):
    pattern, sep, fp = text.partition("=")
    if not sep or not pattern or not FINGERPRINT.fullmatch(fp.lower()):
        raise argparse.ArgumentTypeError(f"expected TASK_OR_GLOB=<12-64 hex digits>, got {text!r}")
    return pattern, fp.lower()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("root", help="run-unit tree of one experiment: NN-KNN/runs/t1/<study>/<experiment>")
    ap.add_argument("--logs", help="t1_parallel job-log directory (<name>__s<seed>.log and .json receipts)")
    ap.add_argument("--json", help="write the whole audit as JSON (put it under <worktree>/results/)")
    ap.add_argument("--expect-fingerprint", type=_expectation, nargs="+", default=[], metavar="TASK_OR_GLOB=FP")
    ap.add_argument("--exclude", nargs="+", default=[], metavar="DATASET_GLOB", help="dataset keys to skip")
    ap.add_argument("--settle-seconds", type=float, default=300.0,
                    help="changes this recent mean a run may still be writing (default 300)")
    args = ap.parse_args(argv)
    report = audit_units(args.root, logs=args.logs, exclude=args.exclude, settle_seconds=args.settle_seconds,
                         expect=args.expect_fingerprint)
    print_report(report)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
