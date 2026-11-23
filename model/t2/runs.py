"""Per-run output of the T2 RL workflows: plain files (as before) or a run unit (model/common/tracking.py).

Every RL training entry point (``train_dqn``, ``train_nec``, ``train_ppo``, ``train_td3``, ``train_nnknn_rl``) takes
``run=None, resumable=False``. Without a run it writes its run directory exactly as before: ``config.json``, the
metric CSVs, ``summary.json``, ``manifest.json``, ``checkpoint.pt`` and its extra event files and ``.pt`` snapshots.
With a run (an open ``tracking.Run``; the caller opens and closes it) it writes no file of its own. Each per-run file
becomes a record of the unit instead:

- ``<name>.csv`` -> row record ``rec/files/<name>.csv``; each row is a list of ``[column, value]`` pairs, so the
  column order of the CSV is kept (``ordered_row`` / ``plain_row``). Rows are flushed before every resume
  checkpoint and at the end, so a resumed run continues the same record.
- ``<name>.json`` -> record ``rec/files/<name>.json`` (canonical JSON: keys sorted); an embedded event list
  (``events``) goes to the row record ``rec/files/<name>.json/events``. ``config.json`` is recorded without
  ``created_at_utc``; readers take it from the first session's start time.
- ``checkpoint.pt`` -> named state ``checkpoint`` (``weights/checkpoint.pt``; numpy arrays stored as tensors);
  mid-training snapshots ``<dir>/<file>.pt`` -> named state ``<dir>.<file>``.
- ``manifest.json`` (it only listed the files) -> record ``rec/outputs``: the per-run files the run produced.

Curves for TensorBoard go to ``x/`` tags (``episode/*``, ``eval/*``, ``train/*``, ``holdout/*``) at the environment
step, and the headline numbers to ``x/final/*``. Readers rebuild the files' contents from the records:
``RunFiles(path)`` reads a run unit or a legacy run directory with the same calls, so summary scripts read both.
"""
from __future__ import annotations

import contextlib
import csv
import io
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import numpy as np

FILES = "files/"


def is_recording(run: Any) -> bool:
    """True for an open tracking run; False for None and NullRun (plain files)."""
    return run is not None and getattr(run, "path", None) is not None


def is_unit(path: str | Path) -> bool:
    """True if ``path`` is a run unit directory (it holds event files)."""
    p = Path(path)
    return p.is_dir() and any("tfevents" in f.name for f in p.iterdir() if f.is_file())


def ordered_row(row: Mapping[str, Any]) -> list[list[Any]]:
    return [[str(k), v] for k, v in row.items()]


def plain_row(pairs: Iterable[Iterable[Any]]) -> dict[str, Any]:
    return {str(k): v for k, v in pairs}


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def json_text(payload: Any) -> str:
    """The text the workflows' ``_write_json`` writes."""
    return json.dumps(payload, indent=2, default=_json_default) + "\n"


def csv_text(rows: list[Mapping[str, Any]]) -> str:
    """The text the workflows' ``_write_csv`` writes (columns in order of first appearance)."""
    if not rows:
        return ""
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


def write_json_file(path: Path, payload: Any) -> None:
    path.write_text(json_text(payload), encoding="utf-8")


def write_csv_file(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        handle.write(csv_text(rows))


def tensorize(obj: Any) -> Any:
    """A checkpoint with numpy arrays as tensors and numpy scalars as numbers (what save_state stores)."""
    import torch

    if isinstance(obj, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(obj).copy())
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Mapping):
        return {k: tensorize(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return tuple(tensorize(v) for v in obj)
    if isinstance(obj, list):
        return [tensorize(v) for v in obj]
    return obj


def numpyize(obj: Any) -> Any:
    """Inverse of ``tensorize`` for the arrays a workflow keeps as numpy (CPU tensors -> numpy arrays)."""
    import torch

    if torch.is_tensor(obj):
        return obj.detach().cpu().numpy().copy()
    if isinstance(obj, Mapping):
        return {k: numpyize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [numpyize(v) for v in obj]
    return obj


def state_name(relpath: str) -> str:
    """Named-state name of a per-run ``.pt`` path: ``checkpoint.pt`` -> ``checkpoint``, ``a/b.pt`` -> ``a.b``."""
    stem = relpath[:-3] if relpath.endswith(".pt") else relpath
    return re.sub(r"[^A-Za-z0-9._=+-]+", "_", stem.replace("/", ".").replace("\\", "."))


class RowLog(list):
    """The row list of one per-run file; in a recording run new rows go to its row record at each flush."""

    def __init__(self, name: str, ordered: bool = True):
        super().__init__()
        self.name, self.ordered, self.flushed = name, ordered, 0

    def flush(self, run: Any) -> None:
        new = self[self.flushed:]
        if new:
            run.log_records(FILES + self.name, [ordered_row(r) if self.ordered else r for r in new])
        self.flushed = len(self)


class RunOutput:
    """Where an RL workflow puts its per-run output: ``run_dir`` files (no run) or records of ``run``."""

    def __init__(self, run: Any, run_dir: Path | None, *, resumable: bool = False):
        self.recording = is_recording(run)
        self.run = run if self.recording else None
        self.dir = Path(run.path) if self.recording else Path(run_dir)
        if resumable and not self.recording:
            raise ValueError("resumable=True needs a recording run")
        self.resumable = resumable
        self._logs: dict[str, RowLog] = {}
        self._last: dict[str, int] = {}
        self._outputs: list[dict[str, Any]] = []
        self.snapshots: dict[str, dict] = {}  # recorded mid-training snapshots: name -> state record

    # ------------------------------------------------------------ rows and curves
    def rows(self, filename: str, ordered: bool = True) -> RowLog:
        """The row list for ``filename`` (a .csv, or ``<file>.json/events`` for an event list)."""
        log = self._logs[filename] = RowLog(filename, ordered)
        return log

    def curve(self, step: int, values: Mapping[str, Any]) -> None:
        """Display curves (``x/<tag>``) at ``step``; None values and repeated steps are skipped (rows hold all)."""
        if not self.recording:
            return
        keep = {}
        for tag, value in values.items():
            if value is None or isinstance(value, str) or (isinstance(value, float) and not math.isfinite(value)):
                continue
            if step <= self._last.get(tag, -1):
                continue
            keep[tag] = value
        if keep:
            self.run.log_step(int(step), keep)
            for tag in keep:
                self._last[tag] = int(step)

    def where(self, path: str | Path | None = None) -> str:
        """``str(path)`` for files; in a recording run the path relative to the unit (``.`` for the unit itself),
        so records depend neither on the run root nor on the unit name. RunFiles turns them back into full paths."""
        if not self.recording:
            return str(self.dir if path is None else path)
        return Path(self.dir if path is None else path).relative_to(self.dir).as_posix()

    def flush(self) -> None:
        if self.recording:
            for log in self._logs.values():
                log.flush(self.run)

    # ------------------------------------------------------------ resume
    def checkpoint_due(self, step: int) -> bool:
        return self.resumable and self.run.checkpoint_due(step)

    def save_resume(self, state: Mapping[str, Any], *, step: int) -> None:
        """Flush every row list, then write the resume checkpoint at ``step``."""
        self.flush()
        self.run.save_resume(state, step=step)

    def load_resume(self) -> Any:
        """The resume payload (None on a fresh unit); rebuilds the row lists and snapshot records from the stream."""
        if not self.recording:
            return None
        payload = self.run.load_resume()
        if payload is None:
            return None
        if not self.resumable:
            raise RuntimeError(f"{self.run.run_id} holds a resume checkpoint, but resumable=False")
        from model.common.tracking import load_run

        data = load_run(self.run.path, arrays=False, allow_incomplete=True)
        for name, log in self._logs.items():
            rows = data.records(FILES + name)
            log[:] = [plain_row(r) for r in rows] if log.ordered else rows
            log.flushed = len(log)
        self.snapshots = {k: v for k, v in data.states().items() if k != "checkpoint"}
        return payload

    # ------------------------------------------------------------ files
    def write_csv(self, filename: str, rows: list[Mapping[str, Any]]) -> None:
        if not self.recording:
            write_csv_file(self.dir / filename, rows)
            return
        log = self._logs.get(filename)
        if log is not None and rows is log:
            log.flush(self.run)
        elif rows:
            self.run.log_records(FILES + filename, [ordered_row(r) for r in rows])
        self._outputs.append({"name": filename, "kind": "csv"})

    def write_json(self, filename: str, payload: Mapping[str, Any]) -> None:
        if not self.recording:
            write_json_file(self.dir / filename, payload)
            return
        payload = dict(payload)
        events = payload.get("events")
        entry = {"name": filename, "kind": "json"}
        if isinstance(events, RowLog):
            events.flush(self.run)
            payload.pop("events")
            entry["events"] = events.name
        elif events is not None and f"{filename}/events" in self._logs:
            raise ValueError(f"{filename}: pass the RowLog itself as 'events'")
        if filename == "config.json":
            payload.pop("created_at_utc", None)  # the run's sessions record when it started
        self.run.log_record(FILES + filename, payload)
        self._outputs.append(entry)

    def save_checkpoint(self, filename: str, checkpoint: Mapping[str, Any]) -> Path:
        """``torch.save`` to ``run_dir/filename``, or the named state ``state_name(filename)``; returns the file."""
        if not self.recording:
            import torch

            path = self.dir / filename
            torch.save(checkpoint, path)
            return path
        record = self.run.save_state(state_name(filename), tensorize(checkpoint))
        self._outputs.append({"name": filename, "kind": "state", "state": state_name(filename)})
        return self.dir / record["file"]["relpath"]

    def save_snapshot(self, relpath: str, state: Any, *, once: bool = True) -> tuple[Path, str]:
        """A mid-training snapshot ``<dir>/<file>.pt`` (``state`` or a callable making it); returns (file, file
        SHA-256). ``once``: an existing snapshot of that name is kept (files) / reused (records); without it a file
        is overwritten, and a record name may not repeat."""
        import hashlib

        if not self.recording:
            import torch

            path = self.dir / relpath
            path.parent.mkdir(exist_ok=True)
            if not (once and path.exists()):
                torch.save(state() if callable(state) else state, path)
            return path, hashlib.sha256(path.read_bytes()).hexdigest()
        name = state_name(relpath)
        record = self.snapshots.get(name) if once else None
        if record is None:
            record = self.snapshots[name] = self.run.save_state(name, tensorize(state() if callable(state) else state))
        return self.dir / record["file"]["relpath"], record["file"]["sha256"]

    def finish(self, finals: Mapping[str, Any]) -> None:
        """Record the output list and the headline numbers (the caller completes the run)."""
        if not self.recording:
            return
        self.flush()
        self.run.log_record("outputs", self._outputs)
        self.run.log_final({k: v for k, v in finals.items() if v is not None})


def headline(summary: Mapping[str, Any]) -> dict[str, Any]:
    """The ``x/final/*`` numbers of an RL run summary."""
    final, last = summary.get("final_eval") or {}, summary.get("last_eval") or {}
    return {
        "selected_mean_return": final.get("mean_return"),
        "selected_std_return": final.get("std_return"),
        "last_mean_return": last.get("mean_return"),
        "selected_step": summary.get("selected_step"),
        "actual_timesteps": summary.get("actual_timesteps"),
        "passed": summary.get("passed"),
    }


# ---------------------------------------------------------------- resume helpers
def replay_state(buffer: Any) -> dict[str, Any]:
    """The filled part of a replay buffer (numpy array fields, ``pos``, ``size``) as tensors."""
    import torch

    size = int(buffer.size)
    arrays = {k: torch.from_numpy(np.ascontiguousarray(v[:size])) for k, v in vars(buffer).items()
              if isinstance(v, np.ndarray)}
    return {"pos": int(buffer.pos), "size": size, "arrays": arrays}


def load_replay_state(buffer: Any, state: Mapping[str, Any]) -> None:
    size = int(state["size"])
    for k, v in state["arrays"].items():
        getattr(buffer, k)[:size] = v.numpy()
    buffer.pos, buffer.size = int(state["pos"]), size


def space_rng_state(space: Any) -> dict[str, Any]:
    """State of a gymnasium space's sampler (``action_space.sample()`` draws from it)."""
    return dict(space.np_random.bit_generator.state)


def set_space_rng_state(space: Any, state: Mapping[str, Any]) -> None:
    space.np_random.bit_generator.state = dict(state)


def tracker_state(tracker: Any) -> dict[str, Any]:
    from dataclasses import asdict

    return asdict(tracker)


def load_tracker_state(tracker: Any, state: Mapping[str, Any]) -> None:
    for k, v in state.items():
        setattr(tracker, k, v)


# ---------------------------------------------------------------- readers
class RunFiles:
    """The per-run files of one RL run, from a run unit or from a legacy run directory, with the same calls.

    ``text(name)`` gives the file's text (for a unit rebuilt from its records: CSVs byte for byte as the workflow
    writes them, JSON with sorted keys), ``json(name)`` the parsed JSON, ``frame(name)`` the CSV read by pandas from
    that text, ``checkpoint_path()`` the checkpoint file (hash-checked for a unit) and ``load_checkpoint``.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.unit = is_unit(self.path)
        self.data = None
        if self.unit:
            from model.common.tracking import load_run

            self.data = load_run(self.path, arrays=False)
            self._outputs = {o["name"]: o for o in self.data.record("outputs")}
        elif not self.path.is_dir():
            raise FileNotFoundError(f"{self.path} is neither a run unit nor a run directory")

    def exists(self, name: str) -> bool:
        if self.unit:
            return name in self._outputs or (name == "manifest.json")
        return (self.path / name).is_file()

    def names(self) -> list[str]:
        if self.unit:
            return list(self._outputs)
        return sorted(p.name for p in self.path.iterdir() if p.is_file())

    def _rows(self, record: str, ordered: bool) -> list:
        rows = self.data.records(FILES + record)
        return [plain_row(r) for r in rows] if ordered else rows

    def json(self, name: str) -> Any:
        if not self.unit:
            return json.loads((self.path / name).read_text(encoding="utf-8"))
        if name == "manifest.json":
            return {"outputs": list(self._outputs)}
        entry = self._outputs.get(name)
        if entry is None or entry["kind"] != "json":
            raise FileNotFoundError(f"{self.path} has no {name}")
        payload = dict(self.data.record(FILES + name))
        for key in ("run_dir", "checkpoint_path"):  # recorded relative to the unit (RunOutput.where)
            if isinstance(payload.get(key), str):
                payload[key] = self.resolve(payload[key])
        if entry.get("events"):
            payload["events"] = self._rows(entry["events"], ordered=False)
        if name == "config.json":
            payload = {"created_at_utc": self.data.sessions()[0]["begin"]["started_utc"], **payload}
        return payload

    def resolve(self, where: str) -> str:
        """Full path of a path recorded by ``RunOutput.where``."""
        if not self.unit:
            return where
        return str(self.path if where == "." else self.path / where)

    def rows(self, name: str) -> list[dict[str, Any]]:
        """The rows of a CSV file: python values for a unit, strings (csv.DictReader) for a legacy file."""
        if not self.unit:
            with (self.path / name).open(newline="", encoding="utf-8") as fh:
                return list(csv.DictReader(fh))
        if name not in self._outputs:
            raise FileNotFoundError(f"{self.path} has no {name}")
        return self._rows(name, ordered=True)

    def text(self, name: str) -> str:
        """The file's text without newline translation (CSV rows end in \\r\\n, as csv writes them)."""
        if not self.unit:
            with (self.path / name).open(encoding="utf-8", newline="") as fh:
                return fh.read()
        if name.endswith(".csv"):
            return csv_text(self.rows(name))
        return json_text(self.json(name))

    def frame(self, name: str) -> Any:
        """``pandas.read_csv`` of the file (a unit's CSV text is rebuilt first, so both parse the same text)."""
        import pandas as pd

        if not self.unit:
            return pd.read_csv(self.path / name)
        return pd.read_csv(io.StringIO(self.text(name)))

    def train_seconds(self) -> float | None:
        """Wall time of the training call (the last session's, for a resumed unit; ``run_training``)."""
        if not self.unit:
            return None
        points = self.data.series("sys/train_seconds")
        return float(points[-1][1]) if points else None

    def checkpoint_path(self, name: str = "checkpoint.pt") -> Path:
        if not self.unit:
            return self.path / name
        from model.common.tracking import TrackingError, _file_sha256

        f = self.data.states()[state_name(name)]["file"]
        path = self.path / f["relpath"]
        if not path.is_file() or _file_sha256(path) != (f["sha256"], f["bytes"]):
            raise TrackingError(f"{path} does not match its record")
        return path

    def load_checkpoint(self, name: str = "checkpoint.pt", map_location: Any = "cpu", weights_only: bool = True) -> Any:
        import torch

        if not self.unit:
            return torch.load(self.path / name, map_location=map_location, weights_only=weights_only)
        from model.common.tracking import load_state

        return load_state(self.path, state_name(name), map_location=map_location)


def checkpoint_source(path: str | Path) -> tuple[Path, Any | None]:
    """(path, unit checkpoint) for the ``load_*_checkpoint`` functions: a run unit directory loads its
    hash-checked ``checkpoint`` state on the CPU; any other path is returned as is with None."""
    p = Path(path)
    if p.is_dir() and is_unit(p):
        return p, RunFiles(p).load_checkpoint()
    return p, None


# ---------------------------------------------------------------- tools
def unit_component(name: Any) -> str:
    """A name as a run-unit path component."""
    return re.sub(r"[^A-Za-z0-9._=+-]+", "_", str(name)).strip("._") or "unnamed"


def unit_parts(text: str) -> tuple[str, ...]:
    return tuple(unit_component(p) for p in str(text).replace("\\", "/").split("/") if p)


def check_study(study: str, root: Any = None) -> None:
    """Raise early (before any training) if the run root is missing or ``study`` is not a valid name."""
    from model.common.tracking import RunKey, unit_dir

    unit_dir(RunKey("t2", study, ("check",)), root)


def unit_config(algorithm: str, task: str, config: Mapping[str, Any], device: Any, **extra: Any) -> dict[str, Any]:
    """The ``rec/config`` of an RL training unit (what makes two runs the same run)."""
    return {"kind": "rl_training", "algorithm": algorithm, "task": task, "config": dict(config),
            "device": None if device is None else str(device), **extra}


@contextlib.contextmanager
def study_unit(study: str, parts: Iterable[str], config: Mapping[str, Any], *, root: Any = None,
               label: str = "t2") -> Iterator[tuple[str, Any]]:
    """Yield ``("complete", RunFiles)`` for a complete unit of ``runs/t2/<study>/<parts...>`` (written with this
    config; another config raises ConfigMismatchError) or ``(source, run)`` for an open run that is completed when
    the block ends (marked failed if it raises); ``source`` is ``fresh``, ``resumed`` or ``restarted``."""
    from model.common.tracking import RunKey, open_run, resolve_unit, unit_dir

    key = RunKey("t2", study, tuple(parts))
    if resolve_unit(key, config=config, root=root) is not None:
        print(f"[{label}] {key.run_id}: complete, not rerun", flush=True)
        yield "complete", RunFiles(unit_dir(key, root))
        return
    with open_run(key, config=config, root=root) as run:
        source = "restarted" if run.restarted else ("resumed" if run.resumed_from else "fresh")
        print(f"[{label}] {key.run_id}: {source}", flush=True)
        yield source, run


def run_training(study: str, parts: Iterable[str], config: Mapping[str, Any], train: Any, *, root: Any = None,
                 label: str = "t2") -> "RunFiles":
    """``train(run)`` in the unit ``runs/t2/<study>/<parts...>`` (resumable; skipped if the unit is complete);
    returns the RunFiles of the complete unit. The wall time of the training call of the last session is logged
    as ``x/sys/train_seconds`` (``RunFiles.train_seconds()``; ``x/sys/*`` is outside the result digest)."""
    import time

    with study_unit(study, parts, config, root=root, label=label) as (source, handle):
        if source != "complete":
            started = time.perf_counter()
            train(handle)
            handle.log_step(0, {"sys/train_seconds": time.perf_counter() - started})
    from model.common.tracking import RunKey, unit_dir

    return RunFiles(unit_dir(RunKey("t2", study, tuple(parts)), root))


def add_study_arguments(parser: Any) -> None:
    parser.add_argument("--study", default=None,
                        help="Record the run as a unit of NN-KNN/runs/t2/<study>/ (TensorBoard event files and weights, "
                             "no run directory); rerunning the same command skips a complete unit and resumes an "
                             "interrupted one. Off by default.")
    parser.add_argument("--unit", default=None,
                        help="Unit path inside the study (a/b/c); default <algorithm>/<task>/seed<seed>.")


def train_or_record(args: Any, algorithm: str, cfg: Any, train: Any, make_output_dir: Any) -> tuple[dict, str]:
    """The training step of tools/run_rl_*.py: a timestamped run directory as before, or with ``--study`` a run
    unit. Returns (summary, checkpoint path)."""
    if getattr(args, "study", None) is None:
        state = train(args.task, cfg, output_dir=make_output_dir(args.task, parent=args.output_dir),
                      device=args.device, progress=not args.quiet)
        return state["summary"], str(state["checkpoint_path"])
    parts = unit_parts(args.unit) if args.unit else (algorithm, unit_component(args.task), f"seed{cfg.seed}")
    config = unit_config(algorithm, args.task, cfg.to_dict(), args.device)
    files = run_training(args.study, parts, config,
                         lambda run: train(args.task, cfg, device=args.device, progress=not args.quiet,
                                           run=run, resumable=True), label=algorithm)
    return files.json("summary.json"), str(files.checkpoint_path())


def summary_of(state_or_files: Any) -> dict[str, Any]:
    """``summary.json`` of a finished training: the state a ``train_*`` returned, or a RunFiles / run path."""
    if isinstance(state_or_files, Mapping):
        return dict(state_or_files["summary"])
    files = state_or_files if isinstance(state_or_files, RunFiles) else RunFiles(state_or_files)
    return files.json("summary.json")
