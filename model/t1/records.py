"""T1 run units: recording helpers for the T1 tools and readers for both run units and legacy result files.

A T1 tool called with ``--study S`` records each run as one unit under ``NN-KNN/runs/t1/S/...`` through
``model.common.tracking`` (docs/NNKNN_CORE.md, "Run tracking"). Readers accept either such a unit tree or a legacy
result directory written before recording existed (``runs__*.jsonl`` shards, per-run files, ``test_predictions.pt``);
the legacy files are read as they are and never converted.

Every JSON record the T1 tools write is stored with ``ordered`` (each dict as ``{"$items": [[key, value], ...]}``)
and read back with ``record``/``rows`` (``unordered``): a record's canonical JSON sorts dictionary keys, and the
readers rebuild rows and lines whose key order matters (e.g. ``tools/t1_confirm_audit.py``).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
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


def has_shards(path: Path, pattern: str = "runs__*.jsonl") -> bool:
    return any(Path(path).glob(pattern))


def is_unit_tree(path: Path) -> bool:
    """True when ``path`` holds run units (event files below it) rather than legacy result files."""
    path = Path(path)
    return path.is_dir() and not has_shards(path) and any(path.rglob("events.out.tfevents.*"))


def shard_rows(path: Path, pattern: str = "runs__*.jsonl") -> list[dict]:
    """Rows of the legacy ``runs__*.jsonl`` shards of an experiment directory, in file and line order."""
    return [json.loads(line) for f in sorted(Path(path).glob(pattern))
            for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]


def _started(data: Any) -> str:
    sessions = data.sessions()
    return str(sessions[0]["begin"].get("started_utc", "")) if sessions else ""


def condition_units(path: Path, arrays: bool = False) -> list[tuple[dict, Any]]:
    """``(row, RunData)`` of every complete ``tools/t1_run.py`` condition unit below ``path``.

    The order is that of the legacy shards the same runs would have filled: by shard file name
    (``runs__<dataset key>.jsonl``), then by the time the unit was first opened (one shard row per finished run).
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


def experiment_rows(path: Path, pattern: str = "runs__*.jsonl") -> list[dict]:
    """Rows of an experiment: legacy shards when ``path`` has any, else the rows of its condition units."""
    path = Path(path)
    if has_shards(path, pattern):
        return shard_rows(path, pattern)
    return [row for row, _ in condition_units(path)] if path.is_dir() else []


def unit_tree(study: str, *parts: Any, root: Any = None) -> Path:
    """Directory of a study (or a sub-tree of it, e.g. one experiment) below the run root."""
    return runs_root(root).joinpath(WORKSTREAM, study, *(unit_component(p) for p in parts))


def trial_unit(study: str, task: str, family: str, stage: str, cid: str, seed: int, root: Any = None) -> Path:
    """Unit directory of one ``tools/t1_grid_search.py`` trial."""
    return unit_dir(key(study, task, family, stage, cid, f"s{seed}"), root)
