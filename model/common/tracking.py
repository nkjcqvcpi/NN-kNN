"""Run tracking on TensorBoard event files (shared core; see docs/NNKNN_CORE.md, "Run tracking").

Event files are the only record of a run; code reads results back with the reader half of this module, which needs
only tensorboard, numpy and the standard library (the writer side imports torch). ``runs/<workstream>/<study>/
<unit...>/`` is one run ("unit"); every process session that writes to it adds ONE event file ("segment",
``*.s<NNN>``) there, and weights stay in its ``weights/`` folder. The exact channel (tensor summaries with plugin
names ``nnknn.*``: ``_run/*`` session records, ``x/<tag>`` numbers, ``rec/<name>`` canonical JSON, ``arr/<name>``
arrays) is what readers use; the display channel (float32 scalars under the tag without ``x/``, a text card) is for
the TensorBoard UI only. ``torch.Tensor`` values raise ``TypeError``, so logging never syncs a device or uses RNG.

Sessions are ordered by the session number in ``_run/begin``. A resumed session cuts the data records that follow
its ``_run/checkpoint`` in earlier sessions, a restart in place cuts all earlier data records; session records are
never cut. ``result_digest`` (in ``_run/complete``) hashes the remaining logical stream without ``x/sys/*`` and
without weight-file bytes, so on CPU an interrupted and resumed run matches the uninterrupted one.

Display after a resume (checked with ``EventAccumulator``): each session starts with a SessionLog START at step 0
(fresh, restart) or checkpoint step + 1. TensorBoard's Python loader (``--load_fast=false``) then hides every earlier
point at or after that step, for every tag: a re-logged tag is one continuous curve, a tag that is not logged again
ends at the checkpoint step, a restart hides all earlier points. The Rust loader (``--load_fast=true``) ignores
SessionLog and keeps stale points of tags that are not logged again. CLI: ``python -m model.common.tracking
{ls|show|verify} <path>``.
"""
from __future__ import annotations

import contextlib
import functools
import hashlib
import io
import json
import math
import os
import random
import re
import socket
import struct
import sys
import time
import traceback
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import numpy as np
from tensorboard.backend.event_processing.event_file_loader import RawEventFileLoader  # no tensorboard: no tracking
from tensorboard.compat.proto import event_pb2, summary_pb2, tensor_pb2, tensor_shape_pb2, types_pb2
from tensorboard.compat.tensorflow_stub import dtypes as _tb_dtypes
from tensorboard.compat.tensorflow_stub.pywrap_tensorflow import masked_crc32c

ENV_VAR = "NNKNN_RUNS"
DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "runs"  # <NN-KNN>/<worktree>/model/common/tracking.py
WORKSTREAMS = ("core", "t1", "t2", "t3")
SCHEMA, MAX_PATH, SMALL_ARRAY, ROW_CHUNK_ROWS, ARRAY_CHUNK_BYTES = 1, 240, 4096, 5000, 64 << 20
P_RUN, P_EXACT, P_RECORD, P_ARRAY = "nnknn.run", "nnknn.exact", "nnknn.record", "nnknn.array"
_COMPONENT = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._=+-]{0,63}\Z")
_TAG = re.compile(r"[A-Za-z0-9_.=+-]+(/[A-Za-z0-9_.=+-]+)*\Z")
_EMBARGO = ("sel_fresh", "held_out")
_ARRAY_DTYPES = {np.dtype(t) for t in ("bool", "int8", "int16", "int32", "int64", "uint8", "float16", "float32", "float64")}
_TERMINAL = ("_run/complete", "_run/failed")


class TrackingError(RuntimeError):
    """Base class of every tracking error."""


class UnitCompleteError(TrackingError):
    """open_run on a complete unit (callers skip it through resolve_unit)."""


class UnitBusyError(TrackingError):
    """The latest unfinished session of the unit looks alive (same host, pid running)."""


class ConfigMismatchError(TrackingError):
    """The config differs from the one the unit was written with."""


class _Invalid(Exception):
    pass


def runs_root(root: str | Path | None = None) -> Path:
    """The run root: ``root``, else ``$NNKNN_RUNS``, else ``<NN-KNN>/runs``; raises if it does not exist."""
    env = os.environ.get(ENV_VAR, "").strip()
    path = Path(env if root is None else root).expanduser().resolve() if root is not None or env else DEFAULT_ROOT
    if not path.is_dir():
        source = "the root argument" if root is not None else (f"${ENV_VAR}" if env else f"the default (${ENV_VAR} unset)")
        raise FileNotFoundError(f"NN-kNN run root {path} does not exist (from {source}). Create it once per host, "
                                f"or set {ENV_VAR} to the directory that holds the runs.")
    return path


def _check_component(part: str, what: str = "path component") -> str:
    if not isinstance(part, str) or not _COMPONENT.match(part) or any(w in part.lower() for w in _EMBARGO):
        raise ValueError(f"invalid {what} {part!r}: use [A-Za-z0-9_][A-Za-z0-9._=+-]{{0,63}}, no embargoed names")
    return part


@dataclass(frozen=True)
class RunKey:
    workstream: str
    study: str
    unit: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "unit", (self.unit,) if isinstance(self.unit, str) else tuple(self.unit))
        if self.workstream not in WORKSTREAMS or not self.unit:
            raise ValueError(f"a run key needs a workstream in {WORKSTREAMS} and at least one unit component")
        if "weights" in [_check_component(part) for part in (self.study, *self.unit)]:
            raise ValueError("'weights' is reserved inside a unit")

    @property
    def run_id(self) -> str:
        return "/".join((self.workstream, self.study, *self.unit))


def _as_key(key: RunKey | str) -> RunKey:
    if isinstance(key, RunKey):
        return key
    if len(parts := key.split("/")) < 3:
        raise ValueError(f"run key {key!r} needs workstream/study/unit")
    return RunKey(parts[0], parts[1], tuple(parts[2:]))


def unit_dir(key: RunKey | str, root: str | Path | None = None) -> Path:
    key = _as_key(key)
    path = runs_root(root).joinpath(key.workstream, key.study, *key.unit)
    if any(word in str(path).lower() for word in _EMBARGO):
        raise ValueError(f"run path {path} names embargoed data")
    return path


def _is_tensor(obj: Any) -> bool:
    torch = sys.modules.get("torch")
    return torch is not None and isinstance(obj, torch.Tensor)


def canonicalize(obj: Any) -> Any:
    """Strict JSON normalisation; raises TypeError on anything ambiguous."""
    if obj is None or isinstance(obj, (bool, str, int)):
        return obj
    if isinstance(obj, np.generic):
        return canonicalize(obj.item())
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else {"$nonfinite": "nan" if math.isnan(obj) else ("inf" if obj > 0 else "-inf")}
    if isinstance(obj, Mapping):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            k = str(int(k)) if isinstance(k, (int, np.integer)) and not isinstance(k, bool) else k
            if not isinstance(k, str) or k == "$nonfinite" or k in out:
                raise TypeError(f"record key {k!r} is not a str, is reserved, or collides after int keys became str")
            out[k] = canonicalize(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [canonicalize(v) for v in obj]
    if isinstance(obj, np.ndarray) and obj.size <= SMALL_ARRAY and obj.dtype.kind in "biuf":
        return canonicalize(obj.tolist())
    raise TypeError(f"{type(obj).__name__} is not allowed in a record (tensors: convert at an existing sync point; "
                    "large arrays: log_array; sets: sort them; paths and dates: pass str)")


def canonical_json(obj: Any) -> bytes:
    return json.dumps(canonicalize(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def payload_sha256(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj)).hexdigest()


def decode_json(raw: bytes) -> Any:
    hook = lambda d: float(d["$nonfinite"]) if len(d) == 1 and "$nonfinite" in d else d  # noqa: E731
    return json.loads(raw.decode("utf-8"), object_hook=hook)


def _exact(value: Any) -> tuple[str, Any]:
    """(kind, python value) of a number for the exact channel."""
    if _is_tensor(value):
        raise TypeError("torch.Tensor value: convert with .item() at an existing sync point")
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value[()]
    if isinstance(value, (bool, np.bool_)):
        return "bool", bool(value)
    if isinstance(value, (int, np.integer)):
        if not -(2**63) <= int(value) < 2**63:
            raise OverflowError(f"{value} does not fit in int64")
        return "int64", int(value)
    if isinstance(value, (float, np.floating)) and not (isinstance(value, np.floating) and value.dtype.itemsize > 8):
        return "float64", float(value)
    raise TypeError(f"{type(value).__name__} is not a number that float64/int64/bool holds exactly; use log_record")


def _check_tag(tag: str) -> str:
    if not isinstance(tag, str) or not _TAG.match(tag) or len(tag) > 200:
        raise ValueError(f"invalid tag {tag!r}: use [A-Za-z0-9_.=+-] groups separated by '/'")
    return tag


def _value(tag: str, payload: Any, plugin: str, content: bytes = b"") -> Any:
    shape = tensor_shape_pb2.TensorShapeProto
    if isinstance(payload, bytes):
        tensor = tensor_pb2.TensorProto(dtype=types_pb2.DT_STRING, tensor_shape=shape(), string_val=[payload])
    elif isinstance(payload, np.ndarray):
        tensor = tensor_pb2.TensorProto(dtype=_tb_dtypes.as_dtype(payload.dtype).as_datatype_enum,
                                        tensor_shape=shape(dim=[shape.Dim(size=n) for n in payload.shape]),
                                        tensor_content=payload.tobytes())
    else:
        dtype, field = {"float64": (types_pb2.DT_DOUBLE, "double_val"), "int64": (types_pb2.DT_INT64, "int64_val"),
                        "bool": (types_pb2.DT_BOOL, "bool_val")}[payload[0]]
        tensor = tensor_pb2.TensorProto(dtype=dtype, tensor_shape=shape(), **{field: [payload[1]]})
    meta = summary_pb2.SummaryMetadata(plugin_data=summary_pb2.SummaryMetadata.PluginData(plugin_name=plugin,
                                                                                         content=content))
    return summary_pb2.Summary.Value(tag=tag, metadata=meta, tensor=tensor)


@dataclass
class _Rec:
    tag: str
    step: int
    kind: str  # float64 | int64 | bool | json | rows | array
    value: Any
    raw: bytes


def _decode(value: Any, step: int) -> _Rec:
    plugin, t = value.metadata.plugin_data.plugin_name, value.tensor
    if plugin == P_EXACT:
        for kind, dtype, vals, fmt in (("float64", types_pb2.DT_DOUBLE, t.double_val, "<d"),
                                       ("int64", types_pb2.DT_INT64, t.int64_val, "<q"),
                                       ("bool", types_pb2.DT_BOOL, t.bool_val, "<q")):
            if t.dtype == dtype and len(vals) == 1:
                v = bool(vals[0]) if kind == "bool" else vals[0]
                return _Rec(value.tag, step, kind, v, struct.pack(fmt, v))
    elif plugin in (P_RUN, P_RECORD) and t.dtype == types_pb2.DT_STRING and len(t.string_val) == 1:
        kind = "rows" if value.metadata.plugin_data.content == b"rows" else "json"
        return _Rec(value.tag, step, kind, decode_json(t.string_val[0]), t.string_val[0])
    elif plugin == P_ARRAY:
        dtype = np.dtype(_tb_dtypes.as_dtype(t.dtype).as_numpy_dtype).newbyteorder("<")
        arr = np.frombuffer(t.tensor_content, dtype).reshape([d.size for d in t.tensor_shape.dim]).copy()
        return _Rec(value.tag, step, "array", arr, arr.tobytes())
    raise _Invalid(f"undecodable exact record {value.tag!r} ({plugin})")


def _lp(b: bytes) -> bytes:
    return struct.pack("<Q", len(b)) + b


def _hash_items(r: _Rec) -> Iterator[tuple[bytes, ...]]:
    """Hash inputs (tag, i64 step, dtype, shape, payload) of a data record; each row of a row chunk is one item."""
    tag = r.tag.encode()
    if r.kind == "rows":
        for i, row in enumerate(r.value["rows"]):
            yield tag, struct.pack("<q", r.value["first"] + i), b"row", b"", canonical_json(row)
    elif r.kind == "array":
        yield tag, struct.pack("<q", r.step), r.value.dtype.str.encode(), json.dumps(list(r.value.shape)).encode(), r.raw
    elif r.kind == "json":  # state records are hashed without their file member (torch.save bytes may vary)
        state = canonical_json({k: v for k, v in r.value.items() if k != "file"}) if r.tag.startswith("rec/state/") else None
        yield tag, struct.pack("<q", r.step), b"json", b"[]", state or r.raw
    else:
        yield tag, struct.pack("<q", r.step), r.kind.encode(), b"[]", r.raw


def _result_digest(stream: list[_Rec]) -> tuple[str, int]:
    h, n = hashlib.sha256(b"nnknn.result.v1"), 0
    for r in stream:
        for item in _hash_items(r):
            n += 1
            if not r.tag.startswith("x/sys/"):
                h.update(b"".join(_lp(part) for part in item))
    return h.hexdigest(), n


@dataclass
class _Segment:
    file: Path
    records: list[_Rec]
    torn_bytes: int = 0
    damage: str | None = None
    session: int = 0
    begin: dict | None = None

    @property
    def terminal(self) -> _Rec | None:
        return self.records[-1] if self.records and self.records[-1].tag in _TERMINAL else None


def _read_segment(path: Path) -> _Segment:
    """One sequential pass with TensorBoard's raw loader; a torn tail (crash) is dropped, other damage reported."""
    seg, consumed = _Segment(path, []), 0
    try:
        for raw in RawEventFileLoader(str(path)).Load():  # stops silently at the first bad record
            consumed += len(raw) + 16
            event = event_pb2.Event.FromString(raw)
            for value in event.summary.value if event.HasField("summary") else ():
                if value.metadata.plugin_data.plugin_name.startswith("nnknn."):
                    seg.records.append(_decode(value, event.step))
    except Exception as exc:  # undecodable Event or exact record
        seg.damage = f"{path.name}: undecodable event ({exc})"
        return seg
    if rest := path.stat().st_size - consumed:  # a crash cuts the last record short (or leaves zeros); a whole
        with path.open("rb") as fh:                # record that fails its CRC, or a bad header CRC, is damage
            fh.seek(consumed)
            head = fh.read(12)
        cut = len(head) < 12 or not head.strip(b"\0") or (masked_crc32c(head[:8]) == struct.unpack("<I", head[8:])[0]
                                                         and 16 + struct.unpack("<Q", head[:8])[0] > rest)
        seg.torn_bytes, seg.damage = (rest, None) if cut else (0, f"{path.name}: corrupt record at byte {consumed}")
    return seg


@dataclass(frozen=True)
class RunInfo:
    path: Path
    run_id: str
    status: str  # complete | failed | incomplete | invalid | empty
    sessions: int
    resumed: bool
    restarts: int
    result_digest: str | None
    n_exact: int
    config_sha256: str | None
    reason: str | None


@dataclass
class _Unit:
    path: Path
    segments: list[_Segment]
    stream: list[_Rec]
    live_checkpoints: list[dict]
    info: RunInfo
    max_seq: int = 0


def _logical(segments: list[_Segment], pending: Mapping | None = None) -> tuple[list[_Rec], list[dict]]:
    """Kept data records (oldest first) and live checkpoints; ``pending`` applies a next session's _run/resume."""
    kept: list[_Rec] = []
    live: list[dict] = []
    pos: dict[int, int] = {}

    def cut(resume: Mapping) -> tuple[list, list]:
        idx = next((i for i, c in enumerate(live) if c["seq"] == resume.get("seq")), None)
        if idx is None and not resume.get("restart"):
            raise _Invalid(f"_run/resume names checkpoint seq {resume.get('seq')}, which is not live")
        return ([], []) if resume.get("restart") else (kept[: pos[resume["seq"]]], live[: idx + 1])

    for seg in segments:
        if seg.session > 1:
            if len(seg.records) < 2 or seg.records[1].tag != "_run/resume":
                raise _Invalid(f"session {seg.session} has no _run/resume after _run/begin")
            kept, live = cut(seg.records[1].value)
        for r in seg.records:
            if r.tag == "_run/checkpoint":
                pos[r.value["seq"]] = len(kept)
                live.append(dict(r.value, session=seg.session))
            elif not r.tag.startswith("_run/"):
                kept.append(r)
    return cut(pending) if pending is not None else (kept, live)


def _event_files(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.is_file() and "tfevents" in p.name) if path.is_dir() else []


def _scan(path: Path) -> _Unit:
    path = Path(path)
    unit, status, reason, kw = _Unit(path, [], [], [], None), "invalid", None, {}
    try:
        for seg in map(_read_segment, _event_files(path)):
            if seg.damage:
                raise _Invalid(seg.damage)
            if not seg.records:
                continue  # empty segment (file header only): ignored
            if seg.records[0].tag != "_run/begin" or not isinstance(seg.records[0].value.get("session"), int):
                raise _Invalid(f"{seg.file.name} does not start with _run/begin")
            if any(r.tag in _TERMINAL for r in seg.records[:-1]):
                raise _Invalid(f"records after a terminal record in {seg.file.name}")
            seg.begin, seg.session = seg.records[0].value, seg.records[0].value["session"]
            unit.segments.append(seg)
        unit.segments.sort(key=lambda s: s.session)
        if [s.session for s in unit.segments] != list(range(1, len(unit.segments) + 1)):
            raise _Invalid(f"session numbers {[s.session for s in unit.segments]} are not 1..n")
        if len({s.begin.get("run_uuid") for s in unit.segments}) > 1:
            raise _Invalid("sessions disagree on run_uuid")
        if any(s.terminal is not None and s.terminal.tag == "_run/complete" for s in unit.segments[:-1]):
            raise _Invalid("a session other than the last one completed")
        unit.max_seq = max((r.value["seq"] for s in unit.segments for r in s.records if r.tag == "_run/checkpoint"),
                           default=0)
        unit.stream, unit.live_checkpoints = _logical(unit.segments)
        for f in (r.value["file"] for r in unit.stream if r.tag.startswith("rec/state/")):
            if not (path / f["relpath"]).is_file() or (path / f["relpath"]).stat().st_size != f["bytes"]:
                raise _Invalid(f"weight file {f['relpath']} is missing or has the wrong size")
        resumes, (digest, n) = [s.records[1].value for s in unit.segments[1:]], _result_digest(unit.stream)
        kw = {"resumed": any(not r.get("restart") for r in resumes), "restarts": sum(bool(r.get("restart")) for r in resumes),
              "n": n}
        terminal = unit.segments[-1].terminal if unit.segments else None
        if not unit.segments:
            status, reason = "empty", "no exact records"
        elif terminal is None:
            status, reason = "incomplete", "no terminal record in the last session"
        elif terminal.tag == "_run/failed":
            status, reason = "failed", terminal.value.get("error")
        elif terminal.value.get("result_digest") != digest or terminal.value.get("n_exact") != n:
            raise _Invalid("result_digest or n_exact does not match the _run/complete record")
        else:
            status, kw["digest"] = "complete", digest
    except _Invalid as exc:
        status, reason, kw = "invalid", str(exc), {}
    begin = unit.segments[-1].begin if unit.segments else {}
    unit.info = RunInfo(path, begin.get("run_id", str(path)), status, len(unit.segments), kw.get("resumed", False),
                        kw.get("restarts", 0), kw.get("digest"), kw.get("n", 0), begin.get("config_sha256"), reason)
    return unit


def run_status(path: str | Path) -> RunInfo:
    """Status of a unit directory (complete, failed, incomplete, invalid or empty), from a full verification pass."""
    return _scan(Path(path)).info


class RunData:
    """Read view of a unit's logical stream (see load_run)."""

    def __init__(self, unit: _Unit, arrays: bool = True):
        self.info, self.path, self._unit, self._arrays_on = unit.info, unit.path, unit, arrays
        self.identity: dict = unit.segments[-1].begin if unit.segments else {}
        self._x, self._single, self._rows, self._arrays = {}, {}, {}, {}  # tag -> points, payload, rows, chunks
        for r in unit.stream:
            if r.tag.startswith("x/"):
                self._x.setdefault(r.tag[2:], []).append((r.step, r.value))
            elif r.kind == "rows":
                self._rows.setdefault(r.tag[4:], []).extend(r.value["rows"])
            elif r.kind == "json":
                self._single[r.tag[4:]] = r.value
            elif arrays:
                self._arrays.setdefault(r.tag[4:], []).append(r.value)
        self.config: dict | None = self._single.get("config")

    def final(self) -> dict[str, Any]:
        return {tag[6:]: pts[0][1] for tag, pts in self._x.items() if tag.startswith("final/")}

    def series(self, tag: str) -> list[tuple[int, Any]]:
        return list(self._x.get(tag[2:] if tag.startswith("x/") else tag, []))

    def record(self, name: str) -> Any:
        return self._single[name]

    def records(self, name: str) -> list:
        return list(self._rows.get(name, []))

    def array(self, name: str) -> np.ndarray:
        if not self._arrays_on:
            raise TrackingError("load_run(arrays=False) does not keep arrays")
        parts = self._arrays[name]
        return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)

    def table(self, name: str) -> dict[str, np.ndarray]:
        cols = [k for k in self._arrays if k.startswith(name + "/") and "/" not in k[len(name) + 1:]]
        if not cols:
            raise KeyError(name)
        return {k[len(name) + 1:]: self.array(k) for k in cols}

    def states(self) -> dict[str, dict]:
        return {k[6:]: v for k, v in self._single.items() if k.startswith("state/")}

    def state_path(self, name: str) -> Path:
        return self.path / self.states()[name]["file"]["relpath"]

    def sessions(self) -> list[dict]:
        return [{"session": s.session, "file": s.file.name, "begin": s.begin, "torn_bytes": s.torn_bytes,
                 "resume": s.records[1].value if s.session > 1 else None,
                 "checkpoints": [r.value for r in s.records if r.tag == "_run/checkpoint"],
                 "terminal": {"tag": s.terminal.tag, **s.terminal.value} if s.terminal else None}
                for s in self._unit.segments]


def load_run(path: str | Path, arrays: bool = True, allow_incomplete: bool = False) -> RunData:
    """RunData of a complete unit (or of any valid one with allow_incomplete); raises TrackingError otherwise."""
    unit = _scan(Path(path))
    if unit.info.status == "invalid" or (unit.info.status != "complete" and not allow_incomplete):
        raise TrackingError(f"{path} is {unit.info.status}" + (f": {unit.info.reason}" if unit.info.reason else ""))
    return RunData(unit, arrays=arrays)


def resolve_unit(key: RunKey | str, *, config: Mapping | None, root: str | Path | None = None) -> RunInfo | None:
    """The unit's RunInfo if it is complete, else None; ConfigMismatchError if complete with another config."""
    info = run_status(unit_dir(key, root))
    if info.status != "complete":
        return None
    if config is not None and info.config_sha256 != payload_sha256(config):
        raise ConfigMismatchError(f"{info.run_id} is complete with a different config")
    return info


def _file_sha256(path: Path) -> tuple[str, int]:
    with path.open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest(), os.fstat(fh.fileno()).st_size


def verify_run(path: str | Path) -> list[str]:
    """Problems of a unit: empty if it is complete and every weight file matches its SHA-256."""
    unit = _scan(Path(path))
    if unit.info.status != "complete":
        return [f"status {unit.info.status}" + (f": {unit.info.reason}" if unit.info.reason else "")]
    return [f"state {name}: {rec['file']['relpath']} does not match its record"
            for name, rec in RunData(unit, arrays=False).states().items()
            if not (unit.path / rec["file"]["relpath"]).is_file()
            or _file_sha256(unit.path / rec["file"]["relpath"]) != (rec["file"]["sha256"], rec["file"]["bytes"])]


def iter_runs(base: str | Path, status: set[str] | None = frozenset({"complete"})) -> Iterator[RunInfo]:
    """Units below ``base`` (units never nest), filtered by status (None: all)."""
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(d for d in dirnames if d != "weights" and not any(w in d.lower() for w in _EMBARGO))
        if any("tfevents" in f for f in filenames):
            dirnames[:] = []
            info = run_status(Path(dirpath))
            if status is None or info.status in status:
                yield info


def _tree_leaves(tree: Any, path: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(tree, Mapping) and tree:
        for k, v in tree.items():
            if isinstance(k, bool) or not isinstance(k, (str, int)):
                raise TypeError(f"state keys must be str or int, got {k!r}")
            yield from _tree_leaves(v, f"{path}/{k}" if path else str(k))
    elif isinstance(tree, (list, tuple)) and tree:
        for i, v in enumerate(tree):
            yield from _tree_leaves(v, f"{path}[{i}]")
    else:
        yield path, dict(tree) if isinstance(tree, Mapping) else (list(tree) if isinstance(tree, tuple) else tree)


def state_digest(state: Any) -> str:
    """SHA-256 over canonical tensor bytes of a state tree (independent of file framing, device and torch version)."""
    h = hashlib.sha256(b"nnknn.state.v1")
    for path, leaf in sorted(_tree_leaves(state), key=lambda p: p[0].encode()):
        h.update(_lp(path.encode()))
        if _is_tensor(leaf):
            t = leaf.detach().cpu().contiguous()  # sparse and quantized tensors raise here or in view()
            data = t.reshape(-1).view(sys.modules["torch"].uint8).numpy().tobytes() if t.numel() else b""
            h.update(b"T" + _lp(str(t.dtype).encode()) + _lp(json.dumps(list(t.shape)).encode()) + _lp(data))
        elif leaf is None or isinstance(leaf, (bool, int, float, str, dict, list)):
            h.update(b"J" + _lp(canonical_json(leaf)))
        else:
            raise TypeError(f"{path}: {type(leaf).__name__} is not a tensor or JSON scalar")
    return h.hexdigest()


def _to_cpu(tree: Any) -> Any:
    if isinstance(tree, Mapping):
        items = [(k, _to_cpu(v)) for k, v in tree.items()]
        return OrderedDict(items) if isinstance(tree, OrderedDict) else dict(items)
    if isinstance(tree, (list, tuple)):
        return (tuple if type(tree) is tuple else list)(_to_cpu(v) for v in tree)
    if _is_tensor(tree):
        return tree.detach().cpu()
    if tree is None or isinstance(tree, (bool, int, float, str)):
        return tree
    raise TypeError(f"{type(tree).__name__} cannot be stored in a weights file (tensors and JSON scalars only)")


def _replace(src: Path, dst: Path) -> None:
    """os.replace, retried on Windows sharing violations; then fsync of the directory on POSIX."""
    for delay in (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 0.0):
        try:
            os.replace(src, dst)
            break
        except PermissionError:
            if not delay:
                raise
            time.sleep(delay)
    if os.name == "posix":
        fd = os.open(dst.parent, os.O_RDONLY)
        os.fsync(fd)
        os.close(fd)


def _write_torch(obj: Any, tmp: Path, dst: Path) -> tuple[str, int]:
    import torch
    with tmp.open("xb") as fh:  # a file object, not a path: the zip prefix is then constant ("archive/")
        torch.save(obj, fh)
        fh.flush()
        os.fsync(fh.fileno())
    sha = _file_sha256(tmp)
    _replace(tmp, dst)
    return sha


def _load_torch(path: Path, sha256: str, size: int, map_location: Any) -> Any:
    import torch
    blob = path.read_bytes() if path.is_file() else b""
    if len(blob) != size or hashlib.sha256(blob).hexdigest() != sha256:
        raise TrackingError(f"{path} is missing or does not match its record")
    return torch.load(io.BytesIO(blob), map_location=map_location, weights_only=True)


def load_state(run_path: str | Path, name: str, map_location: Any = "cpu", allow_incomplete: bool = False) -> Any:
    """Load a named state after checking its file's size and SHA-256 against the record."""
    states = load_run(run_path, arrays=False, allow_incomplete=allow_incomplete).states()
    if name not in states:
        raise TrackingError(f"{run_path} has no state {name!r}")
    f = states[name]["file"]
    return _load_torch(Path(run_path) / f["relpath"], f["sha256"], f["bytes"], map_location)


def capture_rng(generators: Mapping[str, Any] | None = None) -> dict:
    """RNG states: python, numpy global, torch CPU, initialised CUDA/XPU, named torch.Generators. Consumes nothing."""
    import torch
    version, mt, gauss = random.getstate()
    _, key, pos, has_gauss, cached = np.random.get_state()
    out = {"python": [version, list(mt), gauss], "numpy": [key.tolist(), int(pos), int(has_gauss), float(cached)],
           "torch": torch.get_rng_state(), "generators": {n: g.get_state() for n, g in (generators or {}).items()}}
    for dev in ("cuda", "xpu"):
        mod = getattr(torch, dev, None)
        if mod is not None and mod.is_available() and getattr(mod, "is_initialized", lambda: False)():
            out[dev] = mod.get_rng_state_all()
    return out


def restore_rng(snapshot: Mapping, generators: Mapping[str, Any] | None = None) -> None:
    import torch
    generators = dict(generators or {})
    if set(generators) != set(snapshot["generators"]):
        raise TrackingError(f"generators {sorted(generators)} differ from the saved {sorted(snapshot['generators'])}")
    (version, mt, gauss), (key, pos, has_gauss, cached) = snapshot["python"], snapshot["numpy"]
    random.setstate((version, tuple(mt), gauss))
    np.random.set_state(("MT19937", np.array(key, dtype=np.uint32), pos, has_gauss, cached))
    torch.set_rng_state(snapshot["torch"])
    for dev in ("cuda", "xpu"):
        if dev in snapshot:
            getattr(torch, dev).set_rng_state_all(snapshot[dev])
    for name, g in generators.items():
        g.set_state(snapshot["generators"][name])


@dataclass
class ResumePayload:
    seq: int
    step: int
    state: dict
    rng: dict

    def restore_rng(self, generators: Mapping[str, Any] | None = None) -> None:
        """Call last: after rebuilding objects and loading their states, right before the loop continues."""
        restore_rng(self.rng, generators)


def _session_alive(begin: Mapping) -> bool:
    """True if the session ran on this host and its pid is still running (pid reuse can fool it)."""
    if (pid := int(begin.get("pid", -1))) < 0 or begin.get("host") != socket.gethostname():
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except OSError as exc:
            return not isinstance(exc, ProcessLookupError)
        return True
    import ctypes  # never os.kill on Windows: it terminates the process
    k32, code = ctypes.WinDLL("kernel32", use_last_error=True), ctypes.c_ulong()
    k32.OpenProcess.restype, k32.CloseHandle.argtypes = ctypes.c_void_p, (ctypes.c_void_p,)
    k32.GetExitCodeProcess.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong))
    handle = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return ctypes.get_last_error() == 5  # access denied: the process exists
    alive = bool(k32.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259  # STILL_ACTIVE
    k32.CloseHandle(handle)
    return alive


@functools.lru_cache(maxsize=4)
def _provenance(repo: Path) -> tuple[Any, Any, str]:
    """git commit, dirty flag and source fingerprint, once per process (a process's code does not change)."""
    from . import artifacts
    git = artifacts.git_state(repo)
    return git.get("commit"), git.get("dirty_code_worktree"), artifacts.source_fingerprint(repo)


class Run:
    """One writing session of a unit; create it with open_run. Not thread-safe."""

    def __init__(self, *, key: RunKey, path: Path, writer: Any, event_file: Path, session: int, run_uuid: str,
                 resume_from: dict | None, restarted: bool, kept: list[_Rec], max_seq: int,
                 every_seconds: float | None, every_steps: float | None):
        self.key, self.run_id, self.path, self.event_file = key, key.run_id, path, event_file
        self.session, self.run_uuid, self.resumed_from, self.restarted = session, run_uuid, resume_from, restarted
        self._writer, self._fw, self._closed, self._loaded = writer, writer._get_file_writer(), False, False
        self._holding = resume_from is not None  # setup re-execution: data writes are discarded until load_resume
        self._seq, self._every_seconds, self._every_steps = max_seq, every_seconds, int(every_steps or 0)
        self._last_ckpt_step, self._last_ckpt_time = (resume_from["step"] if resume_from else -1), time.monotonic()
        self._steps = {(r.tag, r.step) for r in kept if r.tag.startswith("x/")}
        self._singles = {r.tag for r in kept if r.kind == "json"}
        self._arrays = {r.tag for r in kept if r.kind == "array"}
        self._rows = {r.tag: [r.step + 1, r.value["first"] + len(r.value["rows"])] for r in kept if r.kind == "rows"}

    def _writing(self) -> bool:
        """Raise on a closed session; False while a resumed session is still in its setup phase."""
        if self._closed:
            raise TrackingError(f"session {self.session} of {self.run_id} is closed")
        return not self._holding

    def _emit(self, values: list[Any], step: int = 0) -> None:
        self._fw.add_summary(summary_pb2.Summary(value=values), step)

    def _emit_json(self, tag: str, obj: Any, plugin: str = P_RECORD, content: bytes = b"one", step: int = 0) -> None:
        self._emit([_value(tag, canonical_json(obj), plugin, content)], step)

    def _sync(self, close: bool = False) -> None:
        self._writer.flush()
        if close:
            self._writer.close()
            self._closed = True
        with self.event_file.open("ab") as fh:
            os.fsync(fh.fileno())

    def _record_tag(self, name: str) -> str:
        if name.startswith("state/") or name == "config":  # written by save_state and open_run
            raise ValueError(f"record name {name!r} is reserved")
        return "rec/" + _check_tag(name)

    def _numbers(self, items: list[tuple[str, str, Any]], step: int, display: bool) -> None:
        if not self._writing():
            return
        if dup := [t for t, _, _ in items if (t, step) in self._steps]:
            raise TrackingError(f"{dup} already logged at step {step}")
        self._emit([_value(t, (kind, v), P_EXACT) for t, kind, v in items], step)
        for t, _, v in items:
            self._steps.add((t, step))
            if display:
                self._writer.add_scalar(t[2:], float(v), step)

    def log_step(self, step: int, values: Mapping[str, Any]) -> None:
        """Numbers at ``step`` (keys like ``train/loss``): exact ``x/<tag>`` and display ``<tag>``, each once."""
        items = [("x/" + _check_tag(k), *_exact(v)) for k, v in values.items()]
        if any(t.startswith("x/final/") for t, _, _ in items):
            raise ValueError("final/ tags are written by log_final")
        self._numbers(items, int(step), display=True)

    def log_final(self, values: Mapping[str, Any]) -> None:
        """Final numbers ``x/final/<key>`` (step 0), each once per run; complete() writes their display copies."""
        self._numbers([("x/final/" + _check_tag(k), *_exact(v)) for k, v in values.items()], 0, display=False)

    def log_record(self, name: str, obj: Any) -> str:
        """A single JSON record ``rec/<name>``, written once; returns the payload's SHA-256."""
        tag, sha = self._record_tag(name), payload_sha256(obj)
        if self._writing():
            if tag in self._singles or tag in self._rows:
                raise TrackingError(f"{tag} was already logged")
            self._emit_json(tag, obj)
            self._singles.add(tag)
        return sha

    def log_records(self, name: str, rows: Iterable[Any]) -> None:
        """Append rows to the row record ``rec/<name>``; rows are numbered over the logical stream."""
        tag, rows = self._record_tag(name), [canonicalize(row) for row in rows]
        if not self._writing() or not rows:
            return
        if tag in self._singles:
            raise TrackingError(f"{tag} is a single record")
        counter = self._rows.setdefault(tag, [0, 0])
        for i in range(0, len(rows), ROW_CHUNK_ROWS):
            chunk = rows[i:i + ROW_CHUNK_ROWS]
            self._emit_json(tag, {"first": counter[1], "rows": chunk}, content=b"rows", step=counter[0])
            counter[0], counter[1] = counter[0] + 1, counter[1] + len(chunk)

    def log_array(self, name: str, array: np.ndarray) -> None:
        """A numeric array ``arr/<name>`` (dtype and shape kept), once per name; large arrays split along axis 0."""
        if not isinstance(array, np.ndarray) or array.dtype not in _ARRAY_DTYPES:
            raise TypeError(f"log_array takes a numpy array of numbers or bools, not {type(array).__name__}")
        tag, a = "arr/" + _check_tag(name), array.astype(array.dtype.newbyteorder("<"), order="C", copy=False)
        if not self._writing():
            return
        if tag in self._arrays:
            raise TrackingError(f"{tag} was already logged")
        per = max(1, ARRAY_CHUNK_BYTES // max(1, a[0].nbytes)) if a.ndim and len(a) else 0
        for i, part in enumerate([a] if not per else [a[j:j + per] for j in range(0, len(a), per)]):
            self._emit([_value(tag, part, P_ARRAY)], i)
        self._arrays.add(tag)

    def log_table(self, name: str, columns: Mapping[str, np.ndarray]) -> None:
        """Columns of equal length along axis 0, as ``arr/<name>/<column>``."""
        if len({np.shape(a)[:1] for a in columns.values()}) > 1 or any(np.ndim(a) == 0 for a in columns.values()):
            raise ValueError("table columns must be arrays of equal length")
        for col, a in columns.items():
            self.log_array(f"{name}/{_check_component(col, 'column')}", a)

    def save_state(self, name: str, state: Any, *, step: int | None = None, meta: Mapping | None = None) -> dict | None:
        """Save ``weights/<name>.pt`` (once per name), then bind it with ``rec/state/<name>``; returns that record."""
        if _check_component(name, "state name") == "resume" or ".superseded" in name:
            raise ValueError(f"state name {name!r} is reserved")
        tag, meta = "rec/state/" + name, canonicalize(dict(meta or {}))
        if not self._writing():
            return None
        wdir, cpu = self.path / "weights", _to_cpu(state)
        if tag in self._singles or (wdir / f"{name}.pt").exists():
            raise TrackingError(f"state {name!r} was already saved in this run, or {name}.pt exists unbound")
        wdir.mkdir(exist_ok=True)
        sha, size = _write_torch(cpu, wdir / f".tmp-{uuid.uuid4().hex[:8]}-{name}.pt", wdir / f"{name}.pt")
        record = {"schema": SCHEMA, "name": name, "step": step, "meta": meta, "state_sha256": state_digest(cpu),
                  "file": {"relpath": f"weights/{name}.pt", "sha256": sha, "bytes": size}}
        self._emit_json(tag, record)
        self._singles.add(tag)
        return record

    def checkpoint_due(self, step: int) -> bool:
        """True when a resume checkpoint is due (every N steps and/or every N seconds since the last one)."""
        if self._every_steps and step - self._last_ckpt_step >= self._every_steps:
            return True
        return bool(self._every_seconds and time.monotonic() - self._last_ckpt_time >= self._every_seconds)

    def save_resume(self, state: Mapping, *, step: int, generators: Mapping[str, Any] | None = None) -> int:
        """Replace ``weights/resume/last.pt`` (state, RNG, step), then bind it with ``_run/checkpoint``; returns seq."""
        if not self._writing():
            raise TrackingError("a resumed session must call load_resume() before save_resume()")
        payload = {"format": "nnknn.resume/1", "run_uuid": self.run_uuid, "seq": self._seq + 1, "step": int(step),
                   "state": _to_cpu(dict(state)), "rng": capture_rng(generators)}
        rdir = self.path / "weights" / "resume"
        rdir.mkdir(parents=True, exist_ok=True)
        sha, size = _write_torch(payload, rdir / f".tmp-{uuid.uuid4().hex[:8]}.pt", rdir / "last.pt")
        self._seq += 1
        self._emit_json("_run/checkpoint", {"seq": self._seq, "step": int(step), "session": self.session,
                                            "file": {"relpath": "weights/resume/last.pt", "sha256": sha, "bytes": size}},
                        plugin=P_RUN, content=b"", step=int(step))
        self._sync()
        self._last_ckpt_step, self._last_ckpt_time = int(step), time.monotonic()
        return self._seq

    def load_resume(self, map_location: Any = "cpu") -> ResumePayload | None:
        """The checkpoint this session resumes from (None on a fresh unit or a restart); ends the setup phase."""
        if self._loaded or self._closed:
            raise TrackingError("load_resume() was already called, or the session is closed")
        self._loaded, self._holding = True, False
        if self.resumed_from is None:
            return None
        f = self.resumed_from["file"]
        p = _load_torch(self.path / f["relpath"], f["sha256"], f["bytes"], map_location)
        if p.get("run_uuid") != self.run_uuid or p.get("seq") != self.resumed_from["seq"]:
            raise TrackingError("the resume checkpoint belongs to another run or checkpoint")
        return ResumePayload(p["seq"], p["step"], p["state"], p["rng"])

    def complete(self) -> str:
        """Write ``_run/complete`` with the result digest, then flush, close and fsync; returns result_digest."""
        if self.resumed_from is not None and not self._loaded:
            raise TrackingError("a resumed session must call load_resume() before complete()")
        self._writing()
        self._writer.flush()
        unit = _scan(self.path)
        if unit.info.status != "incomplete":
            raise TrackingError(f"cannot complete {self.run_id}: {unit.info.status} ({unit.info.reason})")
        digest, n = _result_digest(unit.stream)
        for k, v in RunData(unit, arrays=False).final().items():
            self._writer.add_scalar("final/" + k, float(v), 0)
        self._emit_json("_run/complete", {"status": "complete", "session": self.session, "sessions": len(unit.segments),
                                          "resumed": unit.info.resumed, "restarts": unit.info.restarts, "n_exact": n,
                                          "result_digest": digest}, plugin=P_RUN, content=b"")
        self._sync(close=True)
        return digest

    def fail(self, exc: BaseException | str) -> None:
        """Write ``_run/failed``, then flush, close and fsync (no-op on a closed session)."""
        if not self._closed:
            tb = "".join(traceback.format_exception(exc))[-8000:] if isinstance(exc, BaseException) else ""
            self._emit_json("_run/failed", {"status": "failed", "error": repr(exc)[:2000], "traceback": tb},
                            plugin=P_RUN, content=b"")
            self._sync(close=True)

    def __enter__(self) -> "Run":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is not None:
            self.fail(exc)
        elif not self._closed:
            try:
                self.complete()
            except BaseException as err:
                self.fail(err)
                raise
        return False


class NullRun(contextlib.nullcontext):
    """What open_run(enabled=False) returns, for tests: every Run method exists and writes nothing."""

    path, session, resumed_from, restarted = None, 0, None, False

    def __init__(self) -> None:
        super().__init__(self)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_") or not callable(getattr(Run, name, None)):
            raise AttributeError(name)
        return lambda *args, **kwargs: False if name == "checkpoint_due" else None


def open_run(key: RunKey | str, *, config: Mapping, root: str | Path | None = None, enabled: bool = True,
             resume: str = "auto", checkpoint_every_seconds: float | None = None,
             checkpoint_every_steps: int | None = None, repo: Path | None = None, flush_secs: int = 30) -> Run | NullRun:
    """Open the next session of a unit (fresh, resume or restart in place); the full rules are in docs/NNKNN_CORE.md.

    UnitCompleteError, UnitBusyError (a cheap double-launch guard, not a lock), ConfigMismatchError unless
    ``resume="restart"``. ``NNKNN_RESUME_EVERY_SECONDS`` / ``_STEPS`` override the cadence arguments (0: off).
    """
    if not enabled:
        return NullRun()
    from torch.utils.tensorboard import SummaryWriter

    key, config_sha = _as_key(key), payload_sha256(config)
    path = unit_dir(key, root)
    if (resume := "restart" if os.environ.get("NNKNN_RESUME", "").strip() == "restart" else resume) not in ("auto", "restart"):
        raise ValueError("resume must be 'auto' or 'restart'")
    prev = _scan(path)
    if prev.info.status in ("complete", "invalid"):
        raise (UnitCompleteError if prev.info.status == "complete" else TrackingError)(
            f"{key.run_id} is {prev.info.status}" + (f": {prev.info.reason}" if prev.info.reason else ""))
    session, run_uuid, resume_from, pending = 1, uuid.uuid4().hex, None, None
    if prev.segments:
        last = prev.segments[-1]
        if last.terminal is None and _session_alive(last.begin):
            raise UnitBusyError(f"{key.run_id}: session {last.session} (pid {last.begin.get('pid')}) is still running")
        if last.begin.get("config_sha256") != config_sha and resume != "restart":
            raise ConfigMismatchError(f"{key.run_id} was written with another config; pass resume='restart'")
        session, run_uuid = last.session + 1, last.begin["run_uuid"]
        ckpt = prev.live_checkpoints[-1] if prev.live_checkpoints else None
        last_pt = path / "weights" / "resume" / "last.pt"
        if resume != "restart" and ckpt is not None and last_pt.is_file() \
                and _file_sha256(last_pt) == (ckpt["file"]["sha256"], ckpt["file"]["bytes"]):
            resume_from, pending = ckpt, {"from_session": ckpt["session"], "seq": ckpt["seq"], "step": ckpt["step"]}
        else:
            pending = {"restart": True, "reason": "forced restart" if resume == "restart" else
                       ("no checkpoint" if ckpt is None else "checkpoint file does not match")}
    kept = _logical(prev.segments, pending)[0] if prev.segments else []
    host, pid, start = socket.gethostname(), os.getpid(), resume_from["step"] + 1 if resume_from else 0
    if len(str(longest := path / f"events.out.tfevents.{int(time.time()):010d}.{host}.{pid}.99999.s{session:03d}")) >= MAX_PATH:
        raise ValueError(f"event file path {longest} would have {len(str(longest))} characters (limit {MAX_PATH})")
    path.mkdir(parents=True, exist_ok=True)
    before = set(_event_files(path))
    writer = SummaryWriter(str(path), purge_step=start, flush_secs=flush_secs, filename_suffix=f".s{session:03d}")
    event_file = next(iter(set(_event_files(path)) - before))
    commit, dirty, fingerprint = _provenance(Path(repo) if repo is not None else Path(__file__).resolve().parents[2])
    begin = {"schema": SCHEMA, "run_id": key.run_id, "workstream": key.workstream, "study": key.study,
             "unit": list(key.unit), "run_uuid": run_uuid, "session": session, "host": host, "pid": pid,
             "argv": list(sys.argv), "started_utc": datetime.now(timezone.utc).isoformat(), "config_sha256": config_sha,
             "git_commit": commit, "dirty": dirty, "source_fingerprint": fingerprint, "event_file": event_file.name}
    run = Run(key=key, path=path, writer=writer, event_file=event_file, session=session, run_uuid=run_uuid, kept=kept,
              resume_from=resume_from, restarted=bool(pending and pending.get("restart")), max_seq=prev.max_seq,
              every_seconds=float(env_s) if (env_s := os.environ.get("NNKNN_RESUME_EVERY_SECONDS", "").strip()) else
              (300.0 if checkpoint_every_seconds is None else checkpoint_every_seconds),
              every_steps=float(env_n) if (env_n := os.environ.get("NNKNN_RESUME_EVERY_STEPS", "").strip()) else
              checkpoint_every_steps)
    bundle = [_value("_run/begin", canonical_json(begin), P_RUN)]
    if pending is not None:
        bundle.append(_value("_run/resume", canonical_json(pending), P_RUN))
    run._emit(bundle, start)  # one event: begin and resume are written whole or not at all
    mode = "fresh" if pending is None else ("restart in place" if pending.get("restart") else f"resume after step {start - 1}")
    writer.add_text("card/session", f"session {session} on {host} (pid {pid}): {mode}", start)
    run._sync()
    if session > 1:  # named states that no kept record binds step aside, after the bundle that cut them is on disk
        live, wdir = {r.tag[10:] for r in kept if r.tag.startswith("rec/state/")}, path / "weights"
        for f in sorted(wdir.glob("*.pt")) if wdir.is_dir() else []:
            stem = f.name[:-3]
            if stem not in live and ".superseded-" not in stem and not stem.startswith(".tmp-"):
                target = wdir / f"{stem}.superseded-s{session:03d}.pt"
                _replace(f, target if not target.exists() else target.with_suffix(f".{uuid.uuid4().hex[:8]}.pt"))
    if resume_from is None:
        run._emit_json("rec/config", config)
        run._singles.add("rec/config")
    return run


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2 or args[0] not in ("ls", "show", "verify"):
        print("usage: python -m model.common.tracking {ls|show|verify} <path>", file=sys.stderr)
        return 2
    command, path = args[0], Path(args[1])
    if command == "ls":
        for info in iter_runs(path, status=None):
            print(f"{info.status:<10} s{info.sessions:<3} {info.run_id}" + (f"  ({info.reason})" if info.reason else ""))
        return 0
    if command == "verify":
        problems = verify_run(path)
        print("\n".join(problems) or "OK")
        return 1 if problems else 0
    info = run_status(path)
    print(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in info.__dict__.items()}, indent=2))
    if info.status in ("invalid", "empty"):
        return 1
    data = load_run(path, allow_incomplete=True)
    for s in data.sessions():
        print(f"session {s['session']} {s['file']}: resume={s['resume']} end={(s['terminal'] or {}).get('tag')}")
    print("\n".join(f"x/{tag}: {len(pts)} point(s), last {pts[-1]}" for tag, pts in sorted(data._x.items())))
    print("records:", sorted(data._single), "row records:", sorted(data._rows), "arrays:", sorted(data._arrays))
    return 0


if __name__ == "__main__":
    sys.exit(main())
