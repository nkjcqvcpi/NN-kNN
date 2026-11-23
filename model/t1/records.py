"""T1 run units: recording helpers for the T1 tools, and the readers of run units.

Every T1 tool records each run as one unit under ``NN-KNN/runs/t1/S/...`` (``--study S`` is required) through
``model.common.tracking`` (docs/NNKNN_CORE.md, "Run tracking"). Summaries read only run units and write their outputs
under ``<worktree>/results/`` (``results_dir``); legacy result folders (shards, per-run files) are not read.

Every JSON record the T1 tools write is stored with ``ordered`` (each dict as ``{"$items": [[key, value], ...]}``)
and read back with ``record``/``rows`` (``unordered``): a record's canonical JSON sorts dictionary keys, and the
readers rebuild rows and lines whose key order matters (e.g. ``tools/t1_confirm_audit.py``).
"""

from __future__ import annotations

import dataclasses
import hashlib
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from model.common.tracking import RunKey, iter_runs, load_run, runs_root, unit_dir

WORKSTREAM = "t1"
_COMPONENT = re.compile(r"(?!(?i:con|prn|aux|nul|com\d|lpt\d)(\.|\Z))[A-Za-z0-9_][A-Za-z0-9._=+-]{0,63}(?<!\.)\Z")
_TAG_PART = re.compile(r"[A-Za-z0-9_.=+-]+\Z")
_EMBARGO = ("sel_fresh", "held_out")
MAX_COMPONENT = 48  # longer names get a hash suffix, so the event file path stays under the tracking limit


# ----------------------------------------------------------------------------- names


def unit_component(name: Any) -> str:
    """A valid unit path component for ``name``: unchanged when valid and short, else a readable prefix plus a hash.

    The original name is always kept in the unit's config and records, so the mapping only has to be stable.
    """
    text = str(name)
    if _COMPONENT.match(text) and len(text) <= MAX_COMPONENT and not any(w in text.lower() for w in _EMBARGO):
        return text
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    stem = re.sub(r"[^A-Za-z0-9._=+-]", "_", text)[: MAX_COMPONENT - 13].rstrip(".")
    for word in _EMBARGO:
        stem = re.sub(word, "x", stem, flags=re.I)
    if not stem or not re.match(r"[A-Za-z0-9_]", stem):
        stem = "u" + stem
    return f"{stem}-{digest}"


def check_study(study: str, root: Any = None) -> None:
    """Raise early (before any data is loaded) when the run root is missing or the study name is invalid."""
    unit_dir(RunKey(WORKSTREAM, study, ("check",)), root)


def key(study: str, *parts: Any) -> RunKey:
    return RunKey(WORKSTREAM, study, tuple(unit_component(p) for p in parts))


def event_path_length(study: str, *parts: Any, root: Any = None) -> int:
    """Length of the longest event file path ``open_run`` can create for the unit ``parts`` (a 7-digit pid,
    session 001); it refuses to open a unit whose path reaches ``tracking.MAX_PATH``."""
    import socket
    name = f"events.out.tfevents.{'0' * 10}.{socket.gethostname()}.{'0' * 7}.99999.s001"
    return len(str(unit_dir(key(study, *parts), root) / name))


def check_unit_paths(study: str, units: Any, root: Any = None) -> None:
    """ValueError, before any training, when one of ``units`` (tuples of unit parts) would have an event file path
    too long for the tracking module (Windows MAX_PATH)."""
    from model.common.tracking import MAX_PATH
    for parts in units:
        if (n := event_path_length(study, *parts, root=root)) >= MAX_PATH:
            raise ValueError(f"unit {'/'.join(map(str, parts))} of study {study!r} would have event file paths of {n} "
                             f"characters (limit {MAX_PATH}); use a shorter study name or run root")


# ----------------------------------------------------------------------------- values


def jsonable(obj: Any) -> Any:
    """JSON view of ``obj`` exactly as ``write_json``/``write_jsonl`` wrote it (numpy and tensors as lists, Path as str)."""
    from model.common.artifacts import _jsonable  # imports torch: only writers need it
    return _jsonable(obj)


def ordered(obj: Any) -> Any:
    """``obj`` (made JSON-able) with every dict as ``{"$items": [[key, value], ...]}``, so a record keeps key order."""
    obj = jsonable(obj)
    if isinstance(obj, dict):
        return {"$items": [[k, ordered(v)] for k, v in obj.items()]}
    if isinstance(obj, list):
        return [ordered(v) for v in obj]
    return obj


def unordered(obj: Any) -> Any:
    """Inverse of ``ordered``: the original dicts, in their original key order."""
    if isinstance(obj, dict) and set(obj) == {"$items"}:
        return {k: unordered(v) for k, v in obj["$items"]}
    if isinstance(obj, dict):
        return {k: unordered(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [unordered(v) for v in obj]
    return obj


def record(data: Any, name: str) -> Any:
    """The single record ``name`` of a RunData, as it was given to ``log_record`` (key order kept)."""
    return unordered(data.record(name))


def rows(data: Any, name: str) -> list:
    """The row record ``name`` of a RunData, as given to ``log_records`` (key order kept)."""
    return [unordered(r) for r in data.records(name)]


def storable(obj: Any) -> Any:
    """``obj`` as a state that ``Run.save_state`` accepts: tensors (on the CPU) and plain JSON scalars.

    numpy arrays become tensors (object arrays become lists), numpy scalars Python numbers, tuples stay tuples,
    paths, devices and dtypes become strings and dataclasses dicts. Anything else raises TypeError.
    """
    import torch

    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if obj is None or type(obj) in (bool, int, float, str):
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return [storable(v) for v in obj.tolist()] if obj.dtype == object else torch.from_numpy(np.ascontiguousarray(obj).copy())
    if isinstance(obj, dict):
        out = OrderedDict() if isinstance(obj, OrderedDict) else {}
        for k, v in obj.items():
            out[k if isinstance(k, (str, int)) and not isinstance(k, bool) else str(k)] = storable(v)
        return out
    if type(obj) is tuple:
        return tuple(storable(v) for v in obj)
    if isinstance(obj, (set, frozenset)):
        return [storable(v) for v in sorted(obj, key=repr)]
    if isinstance(obj, (list, tuple)):
        return [storable(v) for v in obj]
    if isinstance(obj, (Path, torch.device, torch.dtype)):
        return str(obj)
    if isinstance(obj, bool):
        return bool(obj)
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return float(obj)
    if isinstance(obj, str):
        return str(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return storable(dataclasses.asdict(obj))
    raise TypeError(f"{type(obj).__name__} cannot be stored in a run state")


def numeric(values: dict) -> dict:
    """The top-level numbers of ``values`` whose key is a valid tag, for ``log_final``/``log_step``."""
    out = {}
    for k, v in values.items():
        if isinstance(v, (bool, np.bool_)) or (isinstance(v, (int, float, np.integer, np.floating)) and not
                                                   (isinstance(v, int) and not -(2**63) <= v < 2**63)):
            if all(_TAG_PART.match(p) for p in str(k).split("/")) and len(str(k)) <= 150:
                out[str(k)] = v
    return out


def log_curves(run: Any, history: Any, group: str = "history") -> None:
    """Replay the numbers of every list-of-dict history as display curves ``<group>/<field>`` (``<section>/<field>``
    for a dict of lists). The step is the row's ``epoch`` when every row has one and they increase, else the row index.
    """
    if isinstance(history, dict):
        for section, rows in history.items():
            if isinstance(rows, list) and _TAG_PART.match(str(section)):
                log_curves(run, rows, str(section))
        return
    rows = [r for r in history if isinstance(r, dict)] if isinstance(history, list) else []
    if not rows or len(rows) != len(history):
        return
    epochs = [r.get("epoch") for r in rows]
    by_epoch = all(type(e) is int for e in epochs) and all(a < b for a, b in zip(epochs, epochs[1:]))
    for i, r in enumerate(rows):
        values = {f"{group}/{k}": v for k, v in numeric(r).items() if k != "epoch"}
        if values:
            run.log_step(epochs[i] if by_epoch else i, values)


# ----------------------------------------------------------------------------- readers

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"  # git-ignored: generated summaries only (reports written by people go to reports/)


def results_dir(tree: Any, root: Any = None) -> Path:
    """Default output directory of a summary of the run-unit tree ``tree``: ``<worktree>/results/`` plus the tree's
    path below the run root (e.g. ``results/t1/<study>/<experiment>``), or plus its name when it is elsewhere."""
    tree = Path(tree).resolve()
    try:
        rel = tree.relative_to(runs_root(root).resolve())
    except (FileNotFoundError, ValueError):
        rel = Path(tree.name)
    return RESULTS / rel


def require_unit_tree(path: Any) -> Path:
    """``path`` when it holds run units (event files below it); SystemExit otherwise (legacy result folders with
    ``runs__*.jsonl`` shards or per-run files are no longer read)."""
    path = Path(path)
    if not path.is_dir() or not any(path.rglob("events.out.tfevents.*")):
        raise SystemExit(f"{path} holds no run units; summaries read only run units under the run root "
                         f"(NN-KNN/runs/t1/<study>/...)")
    return path


def _started(data: Any) -> str:
    sessions = data.sessions()
    return str(sessions[0]["begin"].get("started_utc", "")) if sessions else ""


def condition_units(path: Path, arrays: bool = False) -> list[tuple[dict, Any]]:
    """``(row, RunData)`` of every complete ``tools/t1_run.py`` condition unit below ``path``.

    The order is that of the shards the same runs filled before recording (and so of every summary written then):
    by dataset key (as the shard file name ``runs__<dataset key>.jsonl`` sorted), then by the time the unit was first
    opened. Stage units (no ``row`` record) are left out.
    """
    out = []
    for info in iter_runs(Path(path)):
        data = load_run(info.path, arrays=arrays)
        try:
            row = unordered(data.record("row"))
        except KeyError:  # not a condition unit (e.g. a grid trial)
            continue
        # sorted like sorted(Path.glob("runs__*.jsonl")): by Path, so case-insensitively on Windows
        out.append((Path(f"runs__{row['dataset']}.jsonl"), _started(data), str(info.path), row, data))
    out.sort(key=lambda t: t[:3])
    return [(row, data) for *_, row, data in out]


def experiment_rows(path: Path) -> list[dict]:
    """Rows of an experiment's run-unit tree (its condition units, in ``condition_units`` order)."""
    return [row for row, _ in condition_units(require_unit_tree(path))]


def pack(objects: Any) -> dict:
    """``objects`` (models, optimizers, adapters, archives, rows: anything ``torch.save`` pickles) as a named state.

    A stage unit keeps the exact objects its training produced, so loading it continues with what an uninterrupted
    run holds in memory: parameters and buffers, optimizer moments, plain attributes, train/eval modes and Python
    value types, with shared references kept (an optimizer stays bound to its model's parameters).
    """
    import io
    import torch
    buf = io.BytesIO()
    torch.save(objects, buf)
    return {"format": "torch.save", "bytes": torch.frombuffer(bytearray(buf.getvalue()), dtype=torch.uint8)}


def unpack(state: dict) -> Any:
    """Inverse of ``pack``; tensors go to the runtime device (``NNKNN_DEVICE``), as freshly trained ones would."""
    import io
    import torch
    from model.device_utils import resolve_runtime_device
    if state.get("format") != "torch.save":
        raise ValueError(f"unknown stage object format {state.get('format')!r}")
    # our own stage file, already checked against its recorded SHA-256 by RunData.load_state
    return torch.load(io.BytesIO(state["bytes"].numpy().tobytes()), map_location=resolve_runtime_device(),
                      weights_only=False)


# Process-wide ID counters whose position later records depend on, like an RNG: maintenance event ids
# (``<run>-m<n>``) and intervention ids (``<run>-i<n>``) number on across the runs of one process.
ID_COUNTERS = (("model.t1.maintenance", "_event_counter"), ("model.t1.revise", "_ids"))


def capture_ids() -> dict:
    """Positions of the ``ID_COUNTERS`` (read from their repr: nothing is consumed)."""
    import sys
    out = {}
    for module, attr in ID_COUNTERS:
        if module in sys.modules:
            out[f"{module}.{attr}"] = int(re.fullmatch(r"count\((-?\d+)\)", repr(getattr(sys.modules[module], attr))).group(1))
    return out


def restore_ids(positions: dict) -> None:
    import importlib
    import itertools
    for name, position in positions.items():
        module, attr = name.rsplit(".", 1)
        setattr(importlib.import_module(module), attr, itertools.count(int(position)))


def save_rng(run: Any) -> tuple[dict, dict]:
    """Record the RNG state (``rng``) and the ID counter positions (``ids``) a unit's computation leaves; returns
    them for ``restore_after``."""
    from model.common.tracking import capture_rng
    rng, ids = capture_rng(), capture_ids()
    run.save_state("rng", rng, meta={"kind": "rng"})
    run.save_state("ids", ids, meta={"kind": "id_counters"})
    return rng, ids


def restore_after(saved: tuple[dict, dict]) -> None:
    """Put back what ``save_rng`` captured (writing a unit must not move the RNG or the counters)."""
    from model.common.tracking import restore_rng
    restore_rng(saved[0])
    restore_ids(saved[1])


def restore_rng_of(data: Any) -> bool:
    """Restore the RNG state and ID counter positions a complete unit recorded (named states ``rng``, ``ids``), so
    the code after a skipped unit computes what it computed in an uninterrupted run; False (and a warning) for a unit
    recorded without them."""
    import sys
    states = data.states()
    if "rng" in states:
        data.restore_rng()
    if "ids" in states:
        restore_ids(data.load_state("ids"))
    if "rng" in states and "ids" in states:
        return True
    print(f"[warning] {data.info.run_id} has no recorded RNG state or ID counters; what follows it may differ from an "
          f"uninterrupted run", file=sys.stderr, flush=True)
    return False


def stage_unit(key: RunKey, config: Any, compute: Any, describe: Any = None) -> tuple[dict, bool]:
    """Shared training recorded once and never repeated: ``(objects, loaded)``.

    When the unit ``key`` is complete with ``config`` its objects are loaded (``pack``) and the RNG state and ID
    counters its training left are restored; else ``compute()`` runs, and its dict of objects is recorded (named
    states ``objects``, ``rng``, ``ids`` and a readable state dict per ``torch.nn.Module``; ``describe(run, objects)``
    adds records) before it is returned. Writing the unit leaves the RNG and the counters as the computation left them.
    """
    import torch
    from model.common.tracking import load_unit, open_run
    data = load_unit(key, config=config, arrays=False)
    if data is not None:
        objects = unpack(data.load_state("objects"))
        restore_rng_of(data)
        return objects, True
    objects = compute()
    with open_run(key, config=config) as run:
        saved = save_rng(run)  # first: writing the unit must not move the RNG or the counters
        run.save_state("objects", pack(objects), meta={"kind": "objects", "keys": sorted(objects)})
        for part, obj in objects.items():
            if isinstance(obj, torch.nn.Module):
                run.save_state(part, storable(obj.state_dict()))
        if describe is not None:
            describe(run, objects)
    restore_after(saved)
    return objects, False


def unit_tree(study: str, *parts: Any, root: Any = None) -> Path:
    """Directory of a study (or a sub-tree of it, e.g. one experiment) below the run root."""
    return runs_root(root).joinpath(WORKSTREAM, study, *(unit_component(p) for p in parts))


def trial_unit(study: str, task: str, family: str, stage: str, cid: str, seed: int, root: Any = None) -> Path:
    """Unit directory of one ``tools/t1_grid_search.py`` trial."""
    return unit_dir(key(study, task, family, stage, cid, f"s{seed}"), root)
