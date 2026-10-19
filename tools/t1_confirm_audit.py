"""Audit a ``tools/t1_run.py`` experiment directory: run directories against ``runs__*.jsonl`` rows.

    python tools/t1_confirm_audit.py <experiment dir> [--logs JOB_LOG_DIR] [--repair] [--json OUT]
        [--expect-fingerprint TASK_OR_GLOB=SHA256 ...] [--exclude DATASET_GLOB ...]

The experiment directory is ``<t1_run --out>/<experiment>``. ``Runner.finish`` writes the run
directory ``<dataset key>/<condition>/s<seed>/`` (manifest, metrics, history, case and retrieval
logs, checkpoint, the seed's data snapshot, then the condition's extra files), then appends the
run's row to ``runs__<dataset key>.jsonl`` and prints ``[done] <run id> <metrics>``. Before
``tools/t1_run.append_row`` it appended unlocked while ``tools/t1_parallel.py`` ran seeds of one
dataset concurrently; on Windows two appends could land at one offset, so a row could vanish or
leave a fragment. Reported: run directories without a row, rows without a run directory,
identical and conflicting duplicate rows, malformed lines. ``rebuild_row`` rebuilds a row from
the run's own run_manifest.json and metrics.json; the per-shard self-check rebuilds every
existing row and requires the identical serialized line.

A run directory lacking any file ``finish`` writes for every run before the row is incomplete,
not missing: in progress when it changed within ``--settle-seconds``, otherwise a crash that was
not rerun (or files removed), which fails the audit. A finished run without a row that changed
within the settle window is pending and does not fail yet.

``--repair`` (needs ``--logs``, the t1_parallel job-log directory) appends rebuilt rows for
missing ones, one write per row under ``append_row``'s lock (``shard_lock``), only to a shard
whose self-check passed on every existing row (at least one) and whose rebuild is verified for
the missing run's source fingerprint, and only for a run whose ``finish`` provably completed: a
job log holds its ``[done]`` line, equal to the line rebuilt from metrics.json, and that log's
receipt has exit code 0 (no ``[done]`` line for the run may differ). It refuses a shard that
changed within the settle window, since older runs append without the lock; it never rewrites
or removes lines. Run manifests map each task (``<task_type>__<name>``) to its source
fingerprints; ``--expect-fingerprint`` (task key or dataset key, glob allowed; a hex prefix of
12+ digits) fails a task with any other fingerprint; a task is checked against its most specific
pattern (exact task key, exact dataset key, then the first matching glob). Exits 1 when anything is missing,
conflicting, malformed, crashed or unexpected, judged on a re-scan after ``--repair``. Identical
duplicates (a bitwise-identical rerun) are reported but pass unless ``--fail-identical-duplicates``.
"""
from __future__ import annotations

import argparse
import contextlib
import fnmatch
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

SEED_DIR = re.compile(r"s(-?\d+)")
RUN_FILES = ("run_manifest.json", "metrics.json")  # enough to rebuild the row
# what Runner.finish writes for every run before the row (and <key>/data_snapshot_s<seed>.pt)
FINISH_FILES = (*RUN_FILES, "history.json", "case_statistics.jsonl", "case_maintenance.jsonl",
                "retrieval_events.jsonl", "checkpoint.pt")
DONE_KEYS = ("test_accuracy_pre", "test_accuracy_post", "test_rmse_pre", "test_accuracy_rare", "n_cases")
FINGERPRINT = re.compile(r"[0-9a-f]{12,64}")


class RebuildError(ValueError):
    """A run directory whose saved files cannot reproduce its row."""


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


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _rebuild(run_dir: Path, man: dict, metrics: dict) -> dict:
    rc, run = man["resolved_config"], man["run"]
    ds, seed, cond, exp = rc["dataset"], rc["seed"], rc["condition"], rc["experiment"]
    key, label = dataset_key(rc["config"], ds), run_dir.parent.name
    if (run_dir.parent.parent.name, run_dir.name) != (key, f"s{seed}"):
        raise RebuildError(f"manifest is for {key}/{label}/s{seed}")
    cid = config_id(rc)
    if run["configuration_id"] != cid or run["id"] != f"{exp}-{key}-{label}-s{seed}-{cid[:8]}":
        raise RebuildError(f"run id {run['id']} does not match its directory and configuration")
    # Runner.finish's row expression, unchanged; json keeps insertion order.
    return {"run_id": run["id"], "experiment": exp, "dataset": key, "dataset_name": ds["name"],
            "task_type": man["task"]["task_type"], "seed": seed, "condition": label,
            **{k: v for k, v in cond.items() if not isinstance(v, (dict, list))},
            **{k: v for k, v in metrics.items() if not isinstance(v, (dict, list))}}


def rebuild_row(run_dir) -> dict:
    """The row ``Runner.finish`` appended for ``run_dir``, rebuilt from the run's own files.

    run_manifest.json holds the run id and the dataset entry, seed, condition dict and config
    the run was given (``resolved_config``, so its ``dataset_key``) and ``task.task_type``
    (``data.task_type``); metrics.json holds the metrics in insertion order; the condition
    label is the directory name. Raises ``RebuildError`` when the directory, the run id and the
    recomputed configuration id disagree.
    """
    run_dir = Path(run_dir)
    return _rebuild(run_dir, _read_json(run_dir / "run_manifest.json"), _read_json(run_dir / "metrics.json"))


def done_line(run_id: str, metrics: dict) -> str:
    """The ``[done]`` line ``Runner.finish`` prints after appending the row (its expression, unchanged)."""
    return f"[done] {run_id} " + " ".join(f"{k}={metrics[k]:.4f}" for k in DONE_KEYS
                                          if isinstance(metrics.get(k), (int, float)))


def missing_files(run_dir: Path) -> list[str]:
    """``FINISH_FILES`` and the seed's data snapshot that ``run_dir`` lacks. The condition's extra
    files come after these and vary by condition; the ``[done]`` line vouches for them."""
    snapshot = run_dir.parent.parent / f"data_snapshot_{run_dir.name}.pt"
    return [f for f in FINISH_FILES if not (run_dir / f).is_file()] + ([] if snapshot.is_file() else [f"../../{snapshot.name}"])


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


def _age(path: Path, now: float) -> float:
    """Seconds since anything in ``path`` (a file, or a directory and its files) changed."""
    paths = [path, *path.iterdir()] if path.is_dir() else [path]
    return now - max(p.stat().st_mtime for p in paths)


def scan_run_dirs(root: Path, exclude=()) -> dict:
    """{(dataset key, condition, seed): directory} for every ``<key>/<condition>/s<seed>/``."""
    out = {}
    for ds_dir in sorted(p for p in root.iterdir() if p.is_dir() and not _matches(p.name, exclude)):
        for cond_dir in sorted(p for p in ds_dir.iterdir() if p.is_dir()):
            for run_dir in sorted(p for p in cond_dir.iterdir() if p.is_dir()):
                m = SEED_DIR.fullmatch(run_dir.name)
                if m:
                    out[(ds_dir.name, cond_dir.name, int(m.group(1)))] = run_dir
    return out


def _parse_row(text: str):
    try:
        row = json.loads(text)
    except ValueError as e:
        return None, f"not JSON ({e})"
    if not (isinstance(row, dict) and isinstance(row.get("dataset"), str) and isinstance(row.get("condition"), str)
            and type(row.get("seed")) is int):
        return None, "not a t1_run row (dataset, condition, seed)"
    return row, None


def read_shard(path: Path) -> dict:
    """A shard's lines as written: text without terminator, row or parse error; tail and terminators."""
    raw = path.read_bytes()
    parts = raw.split(b"\n")
    tail = parts.pop()  # b"" when the file ends with a newline
    lines, terminators = [], set()
    for number, part in enumerate(parts + ([tail] if tail else []), 1):
        terminated = number <= len(parts)
        if terminated:
            terminators.add("\r\n" if part.endswith(b"\r") else "\n")
            part = part.removesuffix(b"\r")
        text = part.decode("utf-8", errors="replace")
        row, error = (None, None) if not text.strip() else _parse_row(text)
        lines.append({"line": number, "text": text, "row": row, "error": error, "terminated": terminated})
    return {"lines": lines, "terminators": sorted(terminators), "unterminated_tail": bool(tail),
            "size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _value_diff(row: dict, rebuilt: dict) -> str:
    if row == rebuilt and list(row) != list(rebuilt):
        return "key order differs"
    fields = sorted(k for k in row.keys() | rebuilt.keys()
                    if k not in row or k not in rebuilt or json.dumps(row[k]) != json.dumps(rebuilt[k]))
    return f"fields differ: {', '.join(fields)}" if fields else "serialization differs"


def audit(root, *, logs=None, exclude=(), settle_seconds: float = 300.0, expect=(),
          fail_identical_duplicates: bool = False, now: float | None = None) -> dict:
    """Read-only audit of one experiment directory (see the module docstring); ``logs`` is the job-log
    directory that proves a missing row's run completed (needed by ``repair``)."""
    root = Path(root)
    now = time.time() if now is None else now
    rel = lambda p: Path(p).relative_to(root).as_posix()  # noqa: E731
    if logs is not None and not Path(logs).is_dir():
        raise SystemExit(f"job-log directory {logs} does not exist")
    dirs = scan_run_dirs(root, exclude)
    shard_paths = {p.name[len("runs__"):-len(".jsonl")]: p for p in sorted(root.glob("runs__*.jsonl"))
                   if not _matches(p.name[len("runs__"):-len(".jsonl")], exclude)}
    if not dirs and not shard_paths:
        raise SystemExit(f"no runs__*.jsonl and no <dataset>/<condition>/s<seed> directories in {root}")

    # every run directory that can rebuild its row, rebuilt once
    rebuilt, done, fingerprints, incomplete, rebuild_errors, run_fp, changed = {}, {}, {}, {}, [], {}, []
    for key, run_dir in dirs.items():
        lacking = missing_files(run_dir)
        if lacking:  # finish() did not get past these: in progress, crashed, or files removed
            incomplete[key] = {"run": list(key), "run_dir": rel(run_dir), "missing_files": lacking,
                               "recent": _age(run_dir, now) < settle_seconds}
        if not all((run_dir / f).is_file() for f in RUN_FILES):
            continue
        try:
            man, metrics = _read_json(run_dir / "run_manifest.json"), _read_json(run_dir / "metrics.json")
            task = f"{man['task']['task_type']}__{man['resolved_config']['dataset']['name']}"
            fp = str(man["run"].get("source_fingerprint_sha256"))
            entry = fingerprints.setdefault(task, {"dataset_keys": [], "fingerprints": {}})
            if key[0] not in entry["dataset_keys"]:
                entry["dataset_keys"].append(key[0])
            entry["fingerprints"][fp] = entry["fingerprints"].get(fp, 0) + 1
            run_fp[key] = fp
            if man["run"].get("source_changed_during_run"):
                changed.append(rel(run_dir))
            rebuilt[key] = _rebuild(run_dir, man, metrics)
            done[key] = done_line(man["run"]["id"], metrics)
        except (OSError, ValueError, KeyError, TypeError) as e:
            rebuild_errors.append({"run_dir": rel(run_dir), "error": f"{type(e).__name__}: {e}"})
    errors = {e["run_dir"]: e["error"] for e in rebuild_errors}

    # every row, keyed by its run
    shards, by_key = {}, {}
    for skey, path in shard_paths.items():
        content = read_shard(path)
        info = {"path": path.name, "size": content["size"], "sha256": content["sha256"],
                "terminators": content["terminators"], "unterminated_tail": content["unterminated_tail"],
                "recent": _age(path, now) < settle_seconds, "blank_lines": [], "malformed": [], "misplaced": [],
                "self_check": {"rows": 0, "exact": 0, "passed": False, "mismatches": []}, "verified_fingerprints": []}
        check = info["self_check"]
        for ln in content["lines"]:
            if not ln["text"].strip():
                info["blank_lines"].append(ln["line"])
                continue
            check["rows"] += 1
            where = {"shard": path.name, "line": ln["line"]}
            if ln["row"] is None or not ln["terminated"]:
                reason = ln["error"] or "no line terminator (interrupted write)"
                info["malformed"].append({"line": ln["line"], "reason": reason, "text": ln["text"][:120]})
                check["mismatches"].append({**where, "reason": reason})
                continue
            row = ln["row"]
            key = (row["dataset"], row["condition"], row["seed"])
            by_key.setdefault(key, []).append({**where, "text": ln["text"]})
            if row["dataset"] != skey:
                info["misplaced"].append({"line": ln["line"], "dataset": row["dataset"]})
            if key in rebuilt:
                if row_line(rebuilt[key]) == ln["text"]:
                    check["exact"] += 1
                    if run_fp[key] not in info["verified_fingerprints"]:
                        info["verified_fingerprints"].append(run_fp[key])
                    continue
                reason = _value_diff(row, rebuilt[key])
            elif key in dirs and rel(dirs[key]) in errors:
                reason = f"rebuild failed: {errors[rel(dirs[key])]}"
            else:
                reason = "run directory incomplete" if key in dirs else "no run directory"
            check["mismatches"].append({**where, "run": list(key), "reason": reason})
        check["passed"] = check["rows"] > 0 and check["exact"] == check["rows"]
        shards[skey] = info

    orphans, duplicates = [], {"identical": [], "conflicting": []}
    for key, rs in by_key.items():
        if key not in rebuilt and not (key in dirs and rel(dirs[key]) in errors):
            orphans += [{"shard": r["shard"], "line": r["line"], "run": list(key),
                         "reason": "run directory incomplete" if key in dirs else "no run directory"} for r in rs]
        if len(rs) > 1:
            kind = "identical" if len({r["text"] for r in rs}) == 1 else "conflicting"
            duplicates[kind].append({"run": list(key), "lines": [{"shard": r["shard"], "line": r["line"]} for r in rs]})
    for key, entry in incomplete.items():
        entry["row"] = key in by_key
    index = None if logs is None else done_lines(Path(logs))
    missing, pending = [], []
    for key, row in rebuilt.items():
        if key in by_key or key in incomplete:
            continue
        entry = {"run": list(key), "run_dir": rel(dirs[key]), "shard": f"runs__{row['dataset']}.jsonl",
                 "run_id": row["run_id"], "source_fingerprint_sha256": run_fp[key], "line": row_line(row),
                 "evidence": None if index is None else done_evidence(index, row["run_id"], done[key])}
        (pending if _age(dirs[key], now) < settle_seconds else missing).append(entry)

    report = {"root": str(root), "logs": None if logs is None else str(logs), "settle_seconds": settle_seconds,
              "exclude": list(exclude), "fail_identical_duplicates": fail_identical_duplicates,
              "run_dirs": len(dirs), "rebuildable_run_dirs": len(rebuilt) + len(rebuild_errors),
              "rows": sum(len(rs) for rs in by_key.values()), "shards": shards,
              "missing_rows": missing, "pending_rows": pending, "incomplete_run_dirs": list(incomplete.values()),
              "orphan_rows": orphans, "duplicates": duplicates, "rebuild_errors": rebuild_errors,
              "source_fingerprints": dict(sorted(fingerprints.items())),
              "source_changed_during_run": changed,
              "fingerprint_expectations": check_fingerprints(fingerprints, expect)}
    report["problems"] = problems(report, fail_identical_duplicates)
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
    """Every finding that fails the audit. Runs that changed within the settle window (incomplete:
    in progress; pending: a row may still be appended) do not, nor identical duplicates unless
    asked: a bitwise-identical rerun is accepted by tools/t1_confirm_report.py."""
    out = []
    count = lambda items, what: out.append(f"{len(items)} {what}") if items else None  # noqa: E731
    count(report["missing_rows"], "run directories without a row")
    count([i for i in report["incomplete_run_dirs"] if not i["recent"]],
          f"run directories without every file finish() writes before the row, unchanged for"
          f" {report['settle_seconds']:.0f}s (crashed and not rerun, or files removed)")
    count(report["orphan_rows"], "rows without a run directory")
    count(report["duplicates"]["conflicting"], "runs with conflicting duplicate rows")
    if fail_identical_duplicates:
        count(report["duplicates"]["identical"], "runs with identical duplicate rows")
    count(report["rebuild_errors"], "run directories whose files cannot rebuild a row")
    count(report["source_changed_during_run"], "runs whose source changed during the run")
    for s in report["shards"].values():
        if s["self_check"]["exact"] < s["self_check"]["rows"]:
            out.append(f"{s['path']}: self-check {s['self_check']['exact']}/{s['self_check']['rows']} rows rebuild exactly")
        count(s["malformed"], f"malformed lines in {s['path']}")
        count(s["blank_lines"], f"blank lines in {s['path']}")
        count(s["misplaced"], f"rows of another dataset in {s['path']}")
        if len(s["terminators"]) > 1:
            out.append(f"{s['path']}: mixed line terminators")
    for e in report["fingerprint_expectations"]:
        if not e["tasks"] and not e["overridden"]:
            out.append(f"--expect-fingerprint {e['pattern']} matched no task")
        for task, fps in e["other_fingerprints"].items():
            out.append(f"{task}: source fingerprints {', '.join(f'{fp[:12]} ({n})' for fp, n in fps.items())}"
                       f" besides {e['fingerprint'][:12]}")
    return out


# --------------------------------------------------------------------------- repair


@contextlib.contextmanager
def shard_lock(shard: Path, timeout: float = 60.0):
    """Exclusive OS lock on byte 0 of ``<shard>.lock``, the lock ``tools/t1_run.append_row`` takes.

    The lock file is never written (its position stays at byte 0) and never removed. Runs
    started before ``append_row`` existed append without it, hence the settle window.
    """
    fd = os.open(shard.with_name(shard.name + ".lock"), os.O_RDWR | os.O_CREAT, 0o666)
    try:
        deadline = time.monotonic() + timeout
        while True:
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except (PermissionError, BlockingIOError):  # EACCES (msvcrt) or EWOULDBLOCK (flock): held
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"{shard.name}.lock stayed locked for {timeout:.0f}s") from None
                time.sleep(0.01)
        try:
            yield
        finally:
            if os.name == "nt":
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def refusal(report: dict, shard_key: str, rows: list[dict]) -> str | None:
    """Why rebuilt rows must not be appended to ``runs__<shard_key>.jsonl`` (None: they may)."""
    info = report["shards"].get(shard_key)
    if info is None:
        return "shard file absent: no existing row to verify the rebuild against"
    check = info["self_check"]
    if not check["passed"]:
        return (f"self-check failed: {check['exact']}/{check['rows']} existing rows rebuild exactly"
                if check["rows"] else "no existing row to verify the rebuild against")
    if info["unterminated_tail"] or len(info["terminators"]) != 1:
        return "the shard's line terminators are inconsistent"
    if info["recent"] or any(p["shard"] == info["path"] for p in report["pending_rows"]) or any(
            i["recent"] and i["run_dir"].split("/")[0] == shard_key for i in report["incomplete_run_dirs"]):
        return f"the shard or its runs changed within {report['settle_seconds']:.0f}s (a run may still be writing)"
    unverified = sorted({r["source_fingerprint_sha256"] for r in rows} - set(info["verified_fingerprints"]))
    if unverified:
        return f"no existing row verifies the rebuild for source fingerprint {', '.join(f[:12] for f in unverified)}"
    return None


def run_refusal(missing: dict) -> str | None:
    """Why one missing row must not be appended: nothing shows its run's ``finish`` completed."""
    if missing["evidence"] is None:
        return "audited without job logs (--logs): nothing shows finish() completed"
    return missing["evidence"]["reason"]


def repair(root, report: dict, *, lock_timeout: float = 60.0) -> dict:
    """Append rebuilt rows for ``report``'s missing rows where ``refusal`` and ``run_refusal`` allow;
    never rewrites lines."""
    root = Path(root)
    wanted = {}
    for m in report["missing_rows"]:
        wanted.setdefault(m["shard"][len("runs__"):-len(".jsonl")], []).append(m)
    appended, refused = [], []
    for skey, rows in sorted(wanted.items()):
        reason = refusal(report, skey, rows)
        info = report["shards"].get(skey)
        if reason is None:
            refused += [{"shard": m["shard"], "reason": run_refusal(m), "runs": [m["run"]]} for m in rows if run_refusal(m)]
            rows = [m for m in rows if run_refusal(m) is None]
        if reason is None and rows:
            path = root / info["path"]
            try:
                lines = [row_line(rebuild_row(root / m["run_dir"])) for m in rows]
                if lines != [m["line"] for m in rows]:
                    raise ValueError("a run directory changed after the audit; audit again")
                with shard_lock(path, lock_timeout):
                    if hashlib.sha256(path.read_bytes()).hexdigest() != info["sha256"]:
                        reason = "the shard changed after the audit; audit again"
                    else:
                        fd = os.open(path, os.O_WRONLY | os.O_APPEND | getattr(os, "O_BINARY", 0))
                        try:
                            for m, line in zip(rows, lines):
                                view = memoryview((line + info["terminators"][0]).encode("utf-8"))
                                while view:  # one write per row; loops only after a short write
                                    view = view[os.write(fd, view):]
                                os.fsync(fd)
                                appended.append({"shard": info["path"], "run": m["run"], "run_id": m["run_id"],
                                                 "line": line})
                        finally:
                            os.close(fd)
            except (TimeoutError, ValueError, KeyError) as e:  # before any write
                reason = str(e)
        if reason is not None:
            refused.append({"shard": f"runs__{skey}.jsonl", "reason": reason, "runs": [m["run"] for m in rows]})
    return {"appended": appended, "refused": refused}


# --------------------------------------------------------------------------- report


def print_report(report: dict) -> None:
    print(f"audit {report['root']}: {report['run_dirs']} run directories, {report['rows']} rows")
    print(f"{'shard':<40} {'rows':>5} {'exact':>5}  self-check")
    for s in report["shards"].values():
        c = s["self_check"]
        print(f"{s['path']:<40} {c['rows']:>5} {c['exact']:>5}  {'pass' if c['passed'] else 'FAIL'}")
        for m in c["mismatches"][:5]:
            print(f"    line {m['line']}: {m['reason']}")
    proof = lambda m: "" if m["evidence"] is None else (  # noqa: E731
        " ([done] line verified)" if m["evidence"]["passed"] else f" ({m['evidence']['reason']})")
    sections = [("missing rows", report["missing_rows"], lambda m: f"{' '.join(map(str, m['run']))} -> {m['shard']}{proof(m)}"),
                ("pending rows (run finished within the settle window)", report["pending_rows"],
                 lambda m: f"{' '.join(map(str, m['run']))} -> {m['shard']}"),
                ("incomplete run directories (recent ones in progress, ignored)", report["incomplete_run_dirs"],
                 lambda i: f"{i['run_dir']}{' (recent)' if i['recent'] else ''}{' (has a row)' if i['row'] else ''}:"
                           f" lacks {', '.join(i['missing_files'])}"),
                ("orphan rows", report["orphan_rows"], lambda o: f"{o['shard']}:{o['line']} {' '.join(map(str, o['run']))}"),
                ("conflicting duplicates", report["duplicates"]["conflicting"],
                 lambda d: f"{' '.join(map(str, d['run']))}: {d['lines']}"),
                ("identical duplicates" + ("" if report["fail_identical_duplicates"] else " (pass)"),
                 report["duplicates"]["identical"],
                 lambda d: f"{' '.join(map(str, d['run']))}: {d['lines']}"),
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
    if "repair" in report:
        for a in report["repair"]["appended"]:
            print(f"appended to {a['shard']}: {a['run_id']}")
        for r in report["repair"]["refused"]:
            print(f"refused {r['shard']} ({len(r['runs'])} rows): {r['reason']}")
    deferred = len(report["pending_rows"]) + sum(i["recent"] for i in report["incomplete_run_dirs"])
    later = f" ({deferred} runs changed within the settle window, unjudged; audit again later)" if deferred else ""
    print("result: " + ("OK" + later if report["ok"] else "FAIL - " + "; ".join(report["problems"]) + later))


def _expectation(text: str):
    pattern, sep, fp = text.partition("=")
    if not sep or not pattern or not FINGERPRINT.fullmatch(fp.lower()):
        raise argparse.ArgumentTypeError(f"expected TASK_OR_GLOB=<12-64 hex digits>, got {text!r}")
    return pattern, fp.lower()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("root", help="experiment directory holding runs__*.jsonl (<t1_run --out>/<experiment>)")
    ap.add_argument("--logs", help="t1_parallel job-log directory (<name>__s<seed>.log and .json receipts)")
    ap.add_argument("--repair", action="store_true", help="append rebuilt rows for missing ones (see module doc)")
    ap.add_argument("--json", help="write the whole audit as JSON")
    ap.add_argument("--expect-fingerprint", type=_expectation, nargs="+", default=[], metavar="TASK_OR_GLOB=FP")
    ap.add_argument("--exclude", nargs="+", default=[], metavar="DATASET_GLOB", help="dataset keys to skip")
    ap.add_argument("--settle-seconds", type=float, default=300.0,
                    help="changes this recent mean a run may still be writing (default 300)")
    ap.add_argument("--fail-identical-duplicates", action="store_true", help="also fail on identical duplicate rows")
    ap.add_argument("--lock-timeout", type=float, default=60.0)
    args = ap.parse_args(argv)
    if args.repair and not args.logs:
        ap.error("--repair needs --logs: a row is rebuilt only for a run whose [done] line a job log holds")
    opts = dict(logs=args.logs, exclude=args.exclude, settle_seconds=args.settle_seconds,
                expect=args.expect_fingerprint, fail_identical_duplicates=args.fail_identical_duplicates)
    report = audit(args.root, **opts)
    if args.repair:
        done = repair(args.root, report, lock_timeout=args.lock_timeout)
        report = {**audit(args.root, **opts), "before_repair": {"problems": report["problems"]}, "repair": done}
    print_report(report)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
