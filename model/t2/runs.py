"""Per-run output of the T2 RL workflows: records of a run unit (model/common/tracking.py).

Recording is mandatory (owner's rule, 2026-11-22). Every RL training entry point (``train_dqn``, ``train_nec``,
``train_ppo``, ``train_td3``, ``train_nnknn_rl``) requires ``run``, an open ``tracking.Run`` (the caller opens and
closes it; None or a NullRun raises TypeError), and takes ``resumable=False``. It writes no file of its own: what the
retired file mode wrote as per-run files (metric CSVs, ``config.json``, ``summary.json``, ``checkpoint.pt``, event
files and ``.pt`` snapshots) becomes records and named states of the unit:

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
``RunFiles(path)`` reads a run unit (only a run unit: summary scripts read nothing but ``runs/``).
"""
from __future__ import annotations

import contextlib
import csv
import io
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import numpy as np

FILES = "files/"
TRAIN_SECONDS = "t2_train_seconds"  # key of the accumulated training time in a resume payload's state
IMAGE_RESUME_EVERY_SECONDS = 1800.0  # resume cadence of image (ALE) tasks; flat tasks use the core default


def is_recording(run: Any) -> bool:
    """True for an open tracking run; False for None and NullRun."""
    return run is not None and getattr(run, "path", None) is not None


def require_run(run: Any, who: str) -> None:
    """Raise TypeError unless ``run`` is an open tracking run (recording is mandatory)."""
    if not is_recording(run):
        raise TypeError(f"{who} needs run=<an open tracking.Run>: every T2 training is recorded as a run unit "
                        f"of NN-KNN/runs (got {type(run).__name__})")


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
    """The row list of one per-run file; new rows go to its row record at each flush."""

    def __init__(self, name: str, ordered: bool = True):
        super().__init__()
        self.name, self.ordered, self.flushed = name, ordered, 0

    def flush(self, run: Any) -> None:
        new = self[self.flushed:]
        if new:
            run.log_records(FILES + self.name, [ordered_row(r) if self.ordered else r for r in new])
        self.flushed = len(self)


class RunOutput:
    """Where an RL workflow puts its per-run output: records and named states of ``run`` (required)."""

    def __init__(self, run: Any, *, resumable: bool = False):
        require_run(run, "RunOutput")
        self.recording = True
        self.run = run
        self.dir = Path(run.path)
        self.resumable = resumable
        self._logs: dict[str, RowLog] = {}
        self._last: dict[str, int] = {}
        self._outputs: list[dict[str, Any]] = []
        self.snapshots: dict[str, dict] = {}  # recorded mid-training snapshots: name -> state record
        self._started, self._prior_seconds = time.perf_counter(), 0.0  # training time of this and earlier sessions

    def train_seconds(self) -> float:
        """Training time so far: this session since the output was made, plus what the earlier sessions had
        reached at the checkpoint this one resumed from (work redone after that checkpoint is counted once)."""
        return self._prior_seconds + time.perf_counter() - self._started

    # ------------------------------------------------------------ rows and curves
    def rows(self, filename: str, ordered: bool = True) -> RowLog:
        """The row list for ``filename`` (a .csv, or ``<file>.json/events`` for an event list)."""
        log = self._logs[filename] = RowLog(filename, ordered)
        return log

    def curve(self, step: int, values: Mapping[str, Any]) -> None:
        """Display curves (``x/<tag>``) at ``step``; None values and repeated steps are skipped (rows hold all)."""
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
        """The path relative to the unit (``.`` for the unit itself), so records depend neither on the run root
        nor on the unit name. RunFiles turns them back into full paths."""
        return Path(self.dir if path is None else path).relative_to(self.dir).as_posix()

    def flush(self) -> None:
        for log in self._logs.values():
            log.flush(self.run)

    # ------------------------------------------------------------ resume
    def checkpoint_due(self, step: int) -> bool:
        return self.resumable and self.run.checkpoint_due(step)

    def save_resume(self, state: Mapping[str, Any], *, step: int) -> None:
        """Flush every row list, then write the resume checkpoint at ``step`` (with the training time so far)."""
        self.flush()
        self.run.save_resume({**state, TRAIN_SECONDS: self.train_seconds()}, step=step)

    def load_resume(self) -> Any:
        """The resume payload (None on a fresh unit); rebuilds the row lists and snapshot records from the stream."""
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
        self._prior_seconds = float(payload.state.pop(TRAIN_SECONDS, 0.0))
        return payload

    # ------------------------------------------------------------ files
    def write_csv(self, filename: str, rows: list[Mapping[str, Any]]) -> None:
        log = self._logs.get(filename)
        if log is not None and rows is log:
            log.flush(self.run)
        elif rows:
            self.run.log_records(FILES + filename, [ordered_row(r) for r in rows])
        self._outputs.append({"name": filename, "kind": "csv"})

    def write_json(self, filename: str, payload: Mapping[str, Any]) -> None:
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
        """The named state ``state_name(filename)``; returns its file."""
        record = self.run.save_state(state_name(filename), tensorize(checkpoint))
        self._outputs.append({"name": filename, "kind": "state", "state": state_name(filename)})
        return self.dir / record["file"]["relpath"]

    def save_snapshot(self, relpath: str, state: Any, *, once: bool = True) -> tuple[Path, str]:
        """A mid-training snapshot ``<dir>/<file>.pt`` (``state`` or a callable making it) as the named state
        ``<dir>.<file>``; returns (file, file SHA-256). ``once``: an existing snapshot of that name is reused;
        without it a name may not repeat."""
        name = state_name(relpath)
        record = self.snapshots.get(name) if once else None
        if record is None:
            record = self.snapshots[name] = self.run.save_state(name, tensorize(state() if callable(state) else state))
        return self.dir / record["file"]["relpath"], record["file"]["sha256"]

    def finish(self, finals: Mapping[str, Any]) -> None:
        """Record the output list, the headline numbers and the training time over all sessions as
        ``x/sys/train_seconds`` at step 0 (``x/sys/*`` is outside the result digest); the caller completes the run."""
        self.flush()
        self.run.log_record("outputs", self._outputs)
        self.run.log_final({k: v for k, v in finals.items() if v is not None})
        self.run.log_step(0, {"sys/train_seconds": self.train_seconds()})


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
    """The per-run files of one RL run, rebuilt from the records of a complete run unit.

    ``text(name)`` gives the file's text (CSVs byte for byte as the retired file mode wrote them, JSON with sorted
    keys), ``json(name)`` the parsed JSON, ``frame(name)`` the CSV read by pandas from that text,
    ``checkpoint_path()`` the hash-checked checkpoint file and ``load_checkpoint``. Only run units are read: a
    legacy run directory (the file-mode output under results/) raises FileNotFoundError.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not is_unit(self.path):
            raise FileNotFoundError(f"{self.path} is not a run unit (summaries read only run units of NN-KNN/runs)")
        from model.common.tracking import load_run

        self.unit = True
        self.data = load_run(self.path, arrays=False)
        try:
            outputs = self.data.record("outputs")
        except KeyError:
            raise FileNotFoundError(f"{self.path} is not an RL training unit (no record 'outputs')") from None
        self._outputs = {o["name"]: o for o in outputs}

    def exists(self, name: str) -> bool:
        return name in self._outputs or (name == "manifest.json")

    def names(self) -> list[str]:
        return list(self._outputs)

    def _rows(self, record: str, ordered: bool) -> list:
        rows = self.data.records(FILES + record)
        return [plain_row(r) for r in rows] if ordered else rows

    def json(self, name: str) -> Any:
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
        return str(self.path if where == "." else self.path / where)

    def rows(self, name: str) -> list[dict[str, Any]]:
        """The rows of a CSV file (python values)."""
        if name not in self._outputs:
            raise FileNotFoundError(f"{self.path} has no {name}")
        return self._rows(name, ordered=True)

    def text(self, name: str) -> str:
        """The file's text without newline translation (CSV rows end in \\r\\n, as csv writes them)."""
        if name.endswith(".csv"):
            return csv_text(self.rows(name))
        return json_text(self.json(name))

    def frame(self, name: str) -> Any:
        """``pandas.read_csv`` of the CSV text rebuilt from the records."""
        import pandas as pd

        return pd.read_csv(io.StringIO(self.text(name)))

    def train_seconds(self) -> float | None:
        """Training time of the unit summed over its sessions (``RunOutput.train_seconds``). An interrupted session
        counts up to the checkpoint the next one resumed from."""
        points = self.data.series("sys/train_seconds")
        return float(points[-1][1]) if points else None

    def checkpoint_path(self, name: str = "checkpoint.pt") -> Path:
        from model.common.tracking import TrackingError, _file_sha256

        f = self.data.states()[state_name(name)]["file"]
        path = self.path / f["relpath"]
        if not path.is_file() or _file_sha256(path) != (f["sha256"], f["bytes"]):
            raise TrackingError(f"{path} does not match its record")
        return path

    def load_checkpoint(self, name: str = "checkpoint.pt", map_location: Any = "cpu") -> Any:
        return self.data.load_state(state_name(name), map_location=map_location)


def checkpoint_source(path: str | Path) -> tuple[Path, Any | None]:
    """(path, unit checkpoint) for the ``load_*_checkpoint`` functions: a run unit directory loads its
    hash-checked ``checkpoint`` state on the CPU; any other path (a weights file, e.g. a unit's
    ``weights/checkpoint.pt`` from ``RunFiles.checkpoint_path``) is returned as is with None."""
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


def resume_every_seconds(task: Any) -> float | None:
    """Seconds between resume checkpoints for ``task``: ``IMAGE_RESUME_EVERY_SECONDS`` for an image (ALE) task,
    whose checkpoints hold the filled replay buffer (GBs each), None (the core default, 300 s) otherwise or for an
    unknown task. ``NNKNN_RESUME_EVERY_SECONDS`` overrides both (core)."""
    if not task:
        return None
    from datasets.rl_tasks import get_rl_task_spec, is_image_observation_kind

    try:
        spec = get_rl_task_spec(str(task))
    except ValueError:
        return None
    return IMAGE_RESUME_EVERY_SECONDS if is_image_observation_kind(spec.observation_kind) else None


@contextlib.contextmanager
def study_unit(study: str, parts: Iterable[str], config: Mapping[str, Any], *, root: Any = None,
               label: str = "t2", checkpoint_every_seconds: Any = "auto") -> Iterator[tuple[str, Any]]:
    """Yield ``("complete", RunFiles)`` for a complete unit of ``runs/t2/<study>/<parts...>`` (written with this
    config; another config raises ConfigMismatchError) or ``(source, run)`` for an open run that is completed when
    the block ends (marked failed if it raises); ``source`` is ``fresh``, ``resumed`` or ``restarted``.
    ``checkpoint_every_seconds``: the resume cadence; ``"auto"`` takes ``resume_every_seconds(config["task"])``."""
    from model.common.tracking import RunKey, open_run, resolve_unit, unit_dir

    key = RunKey("t2", study, tuple(parts))
    if resolve_unit(key, config=config, root=root) is not None:
        print(f"[{label}] {key.run_id}: complete, not rerun", flush=True)
        yield "complete", RunFiles(unit_dir(key, root))
        return
    if checkpoint_every_seconds == "auto":
        checkpoint_every_seconds = resume_every_seconds(config.get("task"))
    with open_run(key, config=config, root=root, checkpoint_every_seconds=checkpoint_every_seconds) as run:
        source = "restarted" if run.restarted else ("resumed" if run.resumed_from else "fresh")
        print(f"[{label}] {key.run_id}: {source}", flush=True)
        yield source, run


def run_training(study: str, parts: Iterable[str], config: Mapping[str, Any], train: Any, *, root: Any = None,
                 label: str = "t2", checkpoint_every_seconds: Any = "auto") -> "RunFiles":
    """``train(run)`` in the unit ``runs/t2/<study>/<parts...>`` (resumable; skipped if the unit is complete);
    returns the RunFiles of the complete unit. The training time over all sessions is in
    ``RunFiles.train_seconds()`` (logged by the workflow's ``RunOutput.finish``)."""
    with study_unit(study, parts, config, root=root, label=label,
                    checkpoint_every_seconds=checkpoint_every_seconds) as (source, handle):
        if source != "complete":
            train(handle)
    from model.common.tracking import RunKey, unit_dir

    return RunFiles(unit_dir(RunKey("t2", study, tuple(parts)), root))


def add_study_arguments(parser: Any) -> None:
    parser.add_argument("--study", required=True,
                        help="Required: record the run as a unit of NN-KNN/runs/t2/<study>/ (TensorBoard event files "
                             "and weights; no other file is written). Rerunning the same command skips a complete "
                             "unit before any training and resumes an interrupted one.")
    parser.add_argument("--unit", default=None,
                        help="Unit path inside the study (a/b/c); default <algorithm>/<task>/seed<seed> "
                             "(eval/<algorithm>/<task>/<checkpoint>_seed<eval seed>_ep<episodes> with --eval-only).")


def check_study_or_exit(study: str) -> None:
    """``check_study`` for a CLI: SystemExit with the reason (before any data is loaded or any training)."""
    try:
        check_study(study)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"--study: {exc}")


def record_training(args: Any, algorithm: str, cfg: Any, train: Any) -> tuple[dict, str]:
    """The training step of tools/run_rl_*.py: the run unit ``runs/t2/<study>/<algorithm>/<task>/seed<N>`` (or
    ``--unit``); a complete unit is read back, not trained again. Returns (summary, checkpoint file)."""
    parts = unit_parts(args.unit) if args.unit else (algorithm, unit_component(args.task), f"seed{cfg.seed}")
    config = unit_config(algorithm, args.task, cfg.to_dict(), args.device)
    files = run_training(args.study, parts, config,
                         lambda run: train(args.task, cfg, device=args.device, progress=not args.quiet,
                                           run=run, resumable=True), label=algorithm)
    return files.json("summary.json"), str(files.checkpoint_path())


def record_evaluation(args: Any, algorithm: str, load: Any, evaluate: Any, options: Any = None) -> dict[str, Any]:
    """``--eval-only`` of tools/run_rl_*.py: evaluate the weights of the complete run unit ``--checkpoint`` and
    record the evaluation as the unit ``runs/t2/<study>/eval/<algorithm>/<task>/<checkpoint>_seed<S>_ep<E>`` (or
    ``--unit``): record ``files/eval_summary.json`` (what the file mode wrote as eval_summary.json, without
    ``created_at_utc``), row record ``files/eval_episodes.csv`` and ``x/final/*``. A complete evaluation unit is
    read back, not evaluated again. ``options(cfg)`` gives extra keyword arguments of ``evaluate`` (they are part of
    the unit config). Returns the metrics (without the episode rows)."""
    if not args.checkpoint:
        raise SystemExit("--checkpoint is required with --eval-only")
    source = Path(args.checkpoint)
    if not is_unit(source):
        raise SystemExit(f"--checkpoint {source}: not a run unit (evaluation reads only run units of NN-KNN/runs)")
    files = RunFiles(source)
    loaded = load(source, device=args.device)
    cfg = loaded["config"]
    task_name = loaded["task"]["name"]
    episodes = args.eval_episodes if args.eval_episodes is not None else cfg.eval_episodes
    eval_seed = args.eval_seed if args.eval_seed is not None else cfg.eval_seed
    extra = dict(options(cfg)) if options is not None else {}
    checkpoint_sha256 = files.data.states()["checkpoint"]["state_sha256"]
    config = {"kind": "rl_evaluation", "algorithm": algorithm, "task": task_name,
              "checkpoint_state_sha256": checkpoint_sha256, "episodes": int(episodes), "eval_seed": int(eval_seed),
              "device": None if args.device is None else str(args.device), **extra}
    parts = unit_parts(args.unit) if args.unit else (
        "eval", algorithm, unit_component(task_name), f"{checkpoint_sha256[:12]}_seed{eval_seed}_ep{episodes}")
    from model.common.tracking import RunKey, load_unit, open_run

    key = RunKey("t2", args.study, tuple(parts))
    done = load_unit(key, config=config, arrays=False)  # before any evaluation
    if done is not None:
        print(f"[{algorithm}-eval] {key.run_id}: complete, not rerun", flush=True)
        return dict(done.record(FILES + "eval_summary.json")["metrics"])
    with open_run(key, config=config) as run:
        print(f"[{algorithm}-eval] {key.run_id}: {'restarted' if run.restarted else 'fresh'}", flush=True)
        metrics = evaluate(task_name, loaded["model"], episodes=episodes, seed=eval_seed, device=loaded["device"],
                           **extra)
        summary = {k: v for k, v in metrics.items() if k != "episode_metrics"}
        run.log_record(FILES + "eval_summary.json", {
            "task": task_name, "checkpoint": str(source), "checkpoint_run_id": files.data.info.run_id,
            "episodes": episodes, "eval_seed": eval_seed, "metrics": summary})
        rows = metrics.get("episode_metrics") or []
        if rows:
            run.log_records(FILES + "eval_episodes.csv", [ordered_row(r) for r in rows])
        run.log_final({k: v for k, v in summary.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
    return summary


def tool_main(args: Any, algorithm: str, *, config: Any, train: Any, load: Any, evaluate: Any,
              eval_options: Any = None, before_training: Any = None) -> None:
    """main() of tools/run_rl_*.py: ``--study`` is required (argparse); the study is checked before anything else.
    ``before_training(args)`` runs checks before the unit is opened."""
    check_study_or_exit(args.study)
    if args.eval_only:
        metrics = record_evaluation(args, algorithm, load, evaluate, eval_options)
        print(json.dumps(metrics, indent=2))
        return
    cfg = config(args)
    if before_training is not None:
        before_training(args)
    summary, checkpoint = record_training(args, algorithm, cfg, train)
    print(json.dumps({k: v for k, v in summary.items() if k != "checkpoint_path"}, indent=2))
    print(f"Checkpoint: {checkpoint}")


def summary_of(state_or_files: Any) -> dict[str, Any]:
    """``summary.json`` of a finished training: the state a ``train_*`` returned, or a RunFiles / run unit path."""
    if isinstance(state_or_files, Mapping):
        return dict(state_or_files["summary"])
    files = state_or_files if isinstance(state_or_files, RunFiles) else RunFiles(state_or_files)
    return files.json("summary.json")
