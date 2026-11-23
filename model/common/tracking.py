"""Run tracking on TensorBoard event files (shared core; the rules are in docs/NNKNN_CORE.md, "Run tracking").

``runs/<workstream>/<study>/<unit...>/`` is one run ("unit"); each process session that writes to it adds ONE event
file (``*.s<NNN>``) there, weights stay in its ``weights/``. Readers use the exact channel (tensor summaries with
plugin names ``nnknn.*``) and need only tensorboard, numpy and the standard library; the display channel (float32
scalars, a text card) is for the TensorBoard UI. ``torch.Tensor`` values raise, so logging never syncs a device.
Sessions are ordered by the number in ``_run/begin``; a resume cuts the data records after its ``_run/checkpoint``,
a restart cuts all earlier data records; ``result_digest`` hashes what remains (without ``x/sys/*`` and weight-file
bytes), so on CPU an interrupted and resumed run matches the uninterrupted one.

Display after a resume (checked with ``EventAccumulator``): each session writes a SessionLog START at step 0 (fresh,
restart) or checkpoint step + 1, and TensorBoard's Python loader (``--load_fast=false``) hides every earlier point at or
after that step: a re-logged tag is one continuous curve, a tag not logged again ends at the checkpoint step, a restart
hides all earlier points. The Rust loader ignores SessionLog. CLI: ``python -m model.common.tracking {ls|show|verify}``.
"""
from __future__ import annotations

import functools
import hashlib
import io
import json
import math
import operator
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
from typing import Any, Iterable, Iterator, Mapping, NamedTuple

import numpy as np

# tensorboard is imported on first use (no tensorboard: no tracking), so importing this module for ``NullRun`` (the
# default of the training code, used by Table 1 and every plain run) neither needs nor loads tensorboard. Code here
# reaches these names as ``_tb.<name>``; tests may monkeypatch them on the module.
_TB_NAMES = {
    "RawEventFileLoader": ("tensorboard.backend.event_processing.event_file_loader", "RawEventFileLoader"),
    "event_pb2": ("tensorboard.compat.proto.event_pb2", None),
    "summary_pb2": ("tensorboard.compat.proto.summary_pb2", None),
    "tensor_pb2": ("tensorboard.compat.proto.tensor_pb2", None),
    "tensor_shape_pb2": ("tensorboard.compat.proto.tensor_shape_pb2", None),
    "types_pb2": ("tensorboard.compat.proto.types_pb2", None),
    "_tb_dtypes": ("tensorboard.compat.tensorflow_stub.dtypes", None),
    "masked_crc32c": ("tensorboard.compat.tensorflow_stub.pywrap_tensorflow", "masked_crc32c"),
}


def __getattr__(name: str) -> Any:
    if name not in _TB_NAMES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module, attr = _TB_NAMES[name]
    value = importlib.import_module(module)
    value = getattr(value, attr) if attr else value
    globals()[name] = value
    return value


_tb = sys.modules[__name__]

ENV_VAR = "NNKNN_RUNS"
DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "runs"  # <NN-KNN>/<worktree>/model/common/tracking.py
WORKSTREAMS = ("core", "t1", "t2", "t3")
SCHEMA, MAX_PATH, SMALL_ARRAY, ROW_CHUNK_ROWS, ARRAY_CHUNK_BYTES = 1, 240, 4096, 5000, 64 << 20
P_RUN, P_EXACT, P_RECORD, P_ARRAY = "nnknn.run", "nnknn.exact", "nnknn.record", "nnknn.array"
_COMPONENT = re.compile(r"(?!(?i:con|prn|aux|nul|com\d|lpt\d)(\.|\Z))[A-Za-z0-9_][A-Za-z0-9._=+-]{0,63}(?<!\.)\Z")
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
        raise FileNotFoundError(f"NN-kNN run root {path} does not exist: create it once per host, or set ${ENV_VAR}")
    return path


def _check_component(part: str, what: str = "path component") -> str:
    if not isinstance(part, str) or not _COMPONENT.match(part) or any(w in part.lower() for w in _EMBARGO):
        raise ValueError(f"invalid {what} {part!r}: use [A-Za-z0-9_][A-Za-z0-9._=+-]{{0,63}}, no trailing '.', "
                         "no embargoed or Windows device names")
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
    path = runs_root(root).joinpath((key := _as_key(key)).workstream, key.study, *key.unit)
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
    return json.dumps(canonicalize(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


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
    shape = _tb.tensor_shape_pb2.TensorShapeProto
    if isinstance(payload, bytes):
        tensor = _tb.tensor_pb2.TensorProto(dtype=_tb.types_pb2.DT_STRING, tensor_shape=shape(), string_val=[payload])
    elif isinstance(payload, np.ndarray):
        tensor = _tb.tensor_pb2.TensorProto(dtype=_tb._tb_dtypes.as_dtype(payload.dtype).as_datatype_enum,
                                        tensor_shape=shape(dim=[shape.Dim(size=n) for n in payload.shape]),
                                        tensor_content=payload.tobytes())
    else:
        dtype, field = {"float64": (_tb.types_pb2.DT_DOUBLE, "double_val"), "int64": (_tb.types_pb2.DT_INT64, "int64_val"),
                        "bool": (_tb.types_pb2.DT_BOOL, "bool_val")}[payload[0]]
        tensor = _tb.tensor_pb2.TensorProto(dtype=dtype, tensor_shape=shape(), **{field: [payload[1]]})
    meta = _tb.summary_pb2.SummaryMetadata(plugin_data={"plugin_name": plugin, "content": content})
    return _tb.summary_pb2.Summary.Value(tag=tag, metadata=meta, tensor=tensor)


_Rec = NamedTuple("_Rec", [("tag", str), ("step", int), ("kind", str), ("value", Any), ("raw", bytes)])  # raw: hashed


def _decode(value: Any, step: int) -> _Rec:
    plugin, t = value.metadata.plugin_data.plugin_name, value.tensor
    if plugin == P_EXACT:
        for kind, dtype, vals, fmt in (("float64", _tb.types_pb2.DT_DOUBLE, t.double_val, "<d"),
                                       ("int64", _tb.types_pb2.DT_INT64, t.int64_val, "<q"),
                                       ("bool", _tb.types_pb2.DT_BOOL, t.bool_val, "<q")):
            if t.dtype == dtype and len(vals) == 1:
                v = bool(vals[0]) if kind == "bool" else vals[0]
                return _Rec(value.tag, step, kind, v, struct.pack(fmt, v))
    elif plugin in (P_RUN, P_RECORD) and t.dtype == _tb.types_pb2.DT_STRING and len(t.string_val) == 1:
        kind = "rows" if value.metadata.plugin_data.content == b"rows" else "json"
        return _Rec(value.tag, step, kind, decode_json(t.string_val[0]), t.string_val[0])
    elif plugin == P_ARRAY:
        dtype = np.dtype(_tb._tb_dtypes.as_dtype(t.dtype).as_numpy_dtype).newbyteorder("<")
        arr = np.frombuffer(t.tensor_content, dtype).reshape([d.size for d in t.tensor_shape.dim])  # read-only view
        return _Rec(value.tag, step, "array", arr, t.tensor_content)
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
    damage: str | None = None
    session: int = 0
    begin: dict | None = None

    @property
    def terminal(self) -> _Rec | None:
        return self.records[-1] if self.records and self.records[-1].tag in _TERMINAL else None


def _read_segment(path: Path) -> _Segment:
    """One sequential pass with TensorBoard's raw loader; a torn tail (crash) is dropped, other damage reported."""
    seg, consumed, size = _Segment(path, []), 0, path.stat().st_size  # size first: the file may grow while read
    try:
        for raw in _tb.RawEventFileLoader(str(path)).Load():  # stops silently at the first bad record
            consumed += len(raw) + 16
            event = _tb.event_pb2.Event.FromString(raw)
            for value in event.summary.value if event.HasField("summary") else ():
                if value.metadata.plugin_data.plugin_name.startswith("nnknn."):
                    seg.records.append(_decode(value, event.step))
    except Exception as exc:  # undecodable Event or exact record
        seg.damage = f"{path.name}: undecodable event ({exc})"
        return seg
    if (rest := size - consumed) > 0:  # a crash cuts the last record short or leaves zeros (power cut: a valid header,
        with path.open("rb") as fh:  # then zeros); a record failing its CRC is damage
            head = (tail := (fh.seek(consumed), fh.read(rest))[1])[:12]
        cut = len(head) < 12 or not head.strip(b"\0") or (_tb.masked_crc32c(head[:8]) == struct.unpack("<I", head[8:])[0]
                                                         and (16 + struct.unpack("<Q", head[:8])[0] > rest
                                                              or not tail[12:].strip(b"\0")))
        seg.damage = None if cut else f"{path.name}: corrupt record at byte {consumed}"
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
    digest: str = ""


def _logical(segments: list[_Segment], pending: Mapping | None = None) -> tuple[list[_Rec], list[dict]]:
    """Kept data records (oldest first) and live checkpoints; ``pending`` applies a next session's _run/resume."""
    kept, live, pos = [], [], {}  # data records, live checkpoints, checkpoint seq -> len(kept) at that record

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
            if seg.terminal is not None and seg.terminal.value.get("withdrawn"):
                continue  # a concurrent launch that stepped back before writing anything (see open_run)
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
        unit.digest = digest
        kept = resumes[max((i for i, r in enumerate(resumes) if r.get("restart")), default=-1) + 1:]  # after a restart
        kw = {"resumed": bool(kept), "restarts": sum(bool(r.get("restart")) for r in resumes), "n": n}
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
        return parts[0].copy() if len(parts) == 1 else np.concatenate(parts, axis=0)

    def table(self, name: str) -> dict[str, np.ndarray]:
        cols = [k for k in self._arrays if k.startswith(name + "/") and "/" not in k[len(name) + 1:]]
        if not cols:
            raise KeyError(name)
        return {k[len(name) + 1:]: self.array(k) for k in cols}

    def states(self) -> dict[str, dict]:
        return {k[6:]: v for k, v in self._single.items() if k.startswith("state/")}

    def state_path(self, name: str) -> Path:
        """Path of a named state's file, NOT hash-checked (run_status checks sizes only); load_state checks it."""
        return self.path / self.states()[name]["file"]["relpath"]

    def load_state(self, name: str, map_location: Any = "cpu") -> Any:
        """A named state of this unit, after checking its file's size and SHA-256 against the record."""
        f = self.states()[name]["file"]
        return _load_torch(self.path / f["relpath"], f["sha256"], f["bytes"], map_location)

    def restore_rng(self, name: str = "rng", generators: Mapping[str, Any] | None = None) -> None:
        """Restore the RNG snapshot that ``Run.save_rng(name)`` stored (python, numpy, torch CPU, CUDA/XPU, generators),
        so code after a skipped unit continues from the RNG state the unit's training left."""
        restore_rng(self.load_state(name), generators)

    def sessions(self) -> list[dict]:
        return [{"session": s.session, "file": s.file.name, "begin": s.begin,
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
    info = run_status(unit_dir(key := _as_key(key), root))
    if info.status != "complete":
        return None
    if info.run_id != key.run_id:  # e.g. case aliasing on Windows
        raise TrackingError(f"{info.path} holds run {info.run_id}, not {key.run_id}")
    if config is not None and info.config_sha256 != payload_sha256(config):
        raise ConfigMismatchError(f"{info.run_id} is complete with a different config")
    return info


def load_unit(key: RunKey | str, *, config: Mapping | None, root: str | Path | None = None,
              arrays: bool = True) -> RunData | None:
    """RunData of the unit if it is complete (with this config), else None; ConfigMismatchError if it is complete
    with another config. Call it before any training: a complete unit is never trained again, its data and weights
    are read from the unit (``RunData.load_state``, ``RunData.restore_rng``)."""
    info = resolve_unit(key, config=config, root=root)
    return None if info is None else load_run(info.path, arrays=arrays)


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


def _tree_leaves(tree: Any, path: tuple = ()) -> Iterator[tuple[tuple, Any]]:
    if isinstance(tree, Mapping) and tree:
        for k, v in tree.items():
            if isinstance(k, bool) or not isinstance(k, (str, int)):
                raise TypeError(f"state keys must be str or int, got {k!r}")
            yield from _tree_leaves(v, (*path, k))
    elif isinstance(tree, (list, tuple)) and tree:
        for i, v in enumerate(tree):
            yield from _tree_leaves(v, (*path, [i]))
    else:
        yield path, dict(tree) if isinstance(tree, Mapping) else (list(tree) if isinstance(tree, tuple) else tree)


def state_digest(state: Any) -> str:
    """SHA-256 over canonical tensor bytes of a state tree (independent of file framing, device and torch version)."""
    h = hashlib.sha256(b"nnknn.state.v2")
    for path, leaf in sorted(((json.dumps(p).encode(), v) for p, v in _tree_leaves(state)), key=lambda p: p[0]):
        h.update(_lp(path))
        if _is_tensor(leaf):
            t = leaf.detach().cpu().contiguous()  # sparse and quantized tensors raise here or in view()
            data = t.reshape(-1).view(sys.modules["torch"].uint8).numpy().tobytes() if t.numel() else b""
            h.update(b"T" + _lp(str(t.dtype).encode()) + _lp(json.dumps(list(t.shape)).encode()) + _lp(data))
        elif leaf is None or isinstance(leaf, (bool, int, float, str, dict, list)):
            h.update(b"J" + _lp(canonical_json(leaf)))
        else:
            raise TypeError(f"{path.decode()}: {type(leaf).__name__} is not a tensor or JSON scalar")
    return h.hexdigest()


def _to_cpu(tree: Any) -> Any:
    if isinstance(tree, Mapping):
        items = [(k, _to_cpu(v)) for k, v in tree.items()]
        return OrderedDict(items) if isinstance(tree, OrderedDict) else dict(items)
    if isinstance(tree, (list, tuple)):
        return (tuple if type(tree) is tuple else list)(_to_cpu(v) for v in tree)
    if _is_tensor(tree):
        return tree.detach().cpu()
    if isinstance(tree, np.generic):  # numpy scalars (and other subclasses below) would fail weights_only loading
        tree = tree.item()
    if tree is None or type(tree) in (bool, int, float, str):
        return tree
    raise TypeError(f"{type(tree).__name__} cannot be stored in a weights file (tensors and plain JSON scalars only)")


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


def _write_torch(obj: Any, tmp: Path, dst: Path, keep: Path | None = None) -> tuple[str, int]:
    import torch
    with tmp.open("xb") as fh:  # a file object, not a path: the zip prefix is then constant ("archive/")
        torch.save(obj, fh)
        fh.flush()
        os.fsync(fh.fileno())
    sha = _file_sha256(tmp)
    if keep is not None and dst.exists():
        _replace(dst, keep)
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
    f = load_run(run_path, arrays=False, allow_incomplete=allow_incomplete).states()[name]["file"]
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
        if mod is not None and getattr(mod, "is_initialized", lambda: False)() and mod.is_available():
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
    torch.set_rng_state(snapshot["torch"].cpu())  # .cpu(): load_resume(map_location=<device>) moved the states
    for dev in ("cuda", "xpu"):
        if dev in snapshot:
            getattr(torch, dev).set_rng_state_all([s.cpu() for s in snapshot[dev]])
    for name, g in generators.items():
        g.set_state(snapshot["generators"][name].cpu())


@dataclass
class ResumePayload:
    seq: int
    step: int
    state: dict
    rng: dict

    def restore_rng(self, generators: Mapping[str, Any] | None = None) -> None:
        """Call last: after rebuilding objects and loading their states, right before the loop continues."""
        restore_rng(self.rng, generators)


def _proc_id(pid: int) -> str | None:
    """Boot id + start ticks (Linux) or creation time (Windows) of running ``pid``; None if gone, a zombie, or unknown."""
    if os.name == "nt":
        import ctypes  # never os.kill on Windows: it terminates the process
        k32, code, times = ctypes.WinDLL("kernel32"), ctypes.c_ulong(), [ctypes.c_ulonglong() for _ in range(4)]
        k32.OpenProcess.restype = ctypes.c_void_p  # PROCESS_QUERY_LIMITED_INFORMATION is granted on all of one's
        if not (handle := ctypes.c_void_p(k32.OpenProcess(0x1000, False, pid))).value:  # own processes
            return None
        ok = k32.GetExitCodeProcess(handle, ctypes.byref(code)) and code.value == 259 and k32.GetProcessTimes(
            handle, *map(ctypes.byref, times))  # 259: STILL_ACTIVE
        k32.CloseHandle(handle)
        return str(times[0].value) if ok else None
    try:
        fields = Path(f"/proc/{pid}/stat").read_bytes().rsplit(b")", 1)[1].split()
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return None
    return None if fields[0] in (b"Z", b"X") else f"{boot}:{fields[19].decode()}"


def _session_alive(begin: Mapping) -> bool:
    """True if the session ran on this host and its process (pid and start identity) still runs; see open_run."""
    if begin.get("host") != socket.gethostname() or os.environ.get("NNKNN_IGNORE_BUSY", "").strip() == "1":
        return False
    return (pid := int(begin.get("pid", -1))) > 0 and (now := _proc_id(pid)) is not None and now == begin.get("proc_id")


@functools.lru_cache(maxsize=4)
def _provenance(repo: Path) -> tuple[Any, Any, str]:
    """git commit, dirty flag and source fingerprint, once per process (a process's code does not change)."""
    from . import artifacts
    git = artifacts.git_state(repo)
    return git.get("commit"), git.get("dirty_code_worktree"), artifacts.source_fingerprint(repo)


class Run:
    """One writing session of a unit; create it with open_run. Not thread-safe."""

    def __init__(self, *, key: RunKey, path: Path, writer: Any, event_file: Path, session: int, run_uuid: str,
                 resume_from: dict | None, restarted: bool, kept: list[_Rec], max_seq: int, cadence: tuple):
        self.key, self.run_id, self.path, self.event_file = key, key.run_id, path, event_file
        self.session, self.run_uuid, self.resumed_from, self.restarted = session, run_uuid, resume_from, restarted
        self._writer, self._fw, self._closed, self._loaded = writer, writer._get_file_writer(), False, False
        self._holding = resume_from is not None  # setup re-execution: data writes are discarded until load_resume
        env_s, env_n = (os.environ.get(f"NNKNN_RESUME_EVERY_{unit}", "").strip() for unit in ("SECONDS", "STEPS"))
        self._every_seconds = float(env_s) if env_s else (300.0 if cadence[0] is None else cadence[0])
        self._seq, self._every_steps = max_seq, int(float(env_n)) if env_n else int(cadence[1] or 0)
        self._last_ckpt_step, self._last_ckpt_time = (resume_from["step"] if resume_from else -1), time.monotonic()
        self._last = {r.tag: r.step for r in kept if r.tag.startswith("x/")}  # steps increase per tag in the stream
        self._singles, self._arrays = ({r.tag for r in kept if r.kind == kind} for kind in ("json", "array"))
        self._rows = {r.tag: [r.step + 1, r.value["first"] + len(r.value["rows"])] for r in kept if r.kind == "rows"}

    def _writing(self) -> bool:
        """Raise on a closed session; False while a resumed session is still in its setup phase."""
        if self._closed:
            raise TrackingError(f"session {self.session} of {self.run_id} is closed")
        return not self._holding

    def _emit(self, values: list[Any], step: int = 0) -> None:
        self._fw.add_summary(_tb.summary_pb2.Summary(value=values), step)

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
        if dup := [t for t, _, _ in items if step <= self._last.get(t, -2**63 - 1)]:
            raise TrackingError(f"{dup} already logged at step {step} or later (steps must increase per tag)")
        self._emit([_value(t, (kind, v), P_EXACT) for t, kind, v in items], step)
        for t, _, v in items:
            self._last[t] = step
            if display:
                self._writer.add_scalar(t[2:], float(v), step)

    def log_step(self, step: int, values: Mapping[str, Any]) -> None:
        """Numbers at ``step`` (keys like ``train/loss``) as ``x/<tag>`` and display ``<tag>``; steps rise per tag."""
        items = [("x/" + _check_tag(k), *_exact(v)) for k, v in values.items()]
        if any(t.startswith("x/final/") for t, _, _ in items):
            raise ValueError("final/ tags are written by log_final")
        self._numbers(items, operator.index(step), display=True)

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
        tag, rows = self._record_tag(name), list(rows)
        canonicalize(rows)  # validate before writing; _emit_json canonicalizes once (twice breaks NaN markers)
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
        self._arrays.add(tag)  # first: a retry after a writer error raises instead of duplicating chunks
        per = max(1, ARRAY_CHUNK_BYTES // max(1, a[0].nbytes)) if a.ndim and len(a) else 0
        for i, part in enumerate([a] if not per else [a[j:j + per] for j in range(0, len(a), per)]):
            self._emit([_value(tag, part, P_ARRAY)], i)

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
        tag, meta = "rec/state/" + name, dict(meta or {})
        canonicalize(meta)  # validate before writing; _emit_json canonicalizes once (twice breaks NaN markers)
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

    def save_rng(self, name: str = "rng", *, generators: Mapping[str, Any] | None = None,
                 step: int | None = None) -> dict | None:
        """Save ``capture_rng(generators)`` as the named state ``name`` (meta ``{"kind": "rng"}``); consumes nothing.
        Record the RNG state the unit's training leaves when later code depends on it, so a rerun that skips the
        complete unit restores it with ``RunData.restore_rng(name)`` and computes the same numbers."""
        return self.save_state(name, capture_rng(generators), step=step, meta={"kind": "rng"})

    def checkpoint_due(self, step: int) -> bool:
        """True when a resume checkpoint is due (every N steps and/or every N seconds since the last one)."""
        return bool(self._every_steps and step - self._last_ckpt_step >= self._every_steps) or bool(
            self._every_seconds and time.monotonic() - self._last_ckpt_time >= self._every_seconds)

    def save_resume(self, state: Mapping, *, step: int, generators: Mapping[str, Any] | None = None) -> int:
        """Write ``weights/resume/last.pt`` (state, RNG, step; old one kept as ``prev.pt``), bind it; returns seq."""
        if not self._writing():
            raise TrackingError("a resumed session must call load_resume() before save_resume()")
        step, rdir = operator.index(step), self.path / "weights" / "resume"
        payload = {"format": "nnknn.resume/1", "run_uuid": self.run_uuid, "seq": self._seq + 1, "step": step,
                   "state": _to_cpu(dict(state)), "rng": capture_rng(generators)}
        rdir.mkdir(parents=True, exist_ok=True)
        sha, size = _write_torch(payload, rdir / f".tmp-{uuid.uuid4().hex[:8]}.pt", rdir / "last.pt", rdir / "prev.pt")
        self._seq, file = self._seq + 1, {"relpath": "weights/resume/last.pt", "sha256": sha, "bytes": size}
        self._emit_json("_run/checkpoint", {"seq": self._seq, "step": step, "session": self.session, "file": file},
                        P_RUN, b"", step)
        self._sync()
        self._last_ckpt_step, self._last_ckpt_time = step, time.monotonic()
        return self._seq

    def load_resume(self, map_location: Any = "cpu") -> ResumePayload | None:
        """The checkpoint this session resumes from (None on a fresh unit or a restart); ends the setup phase."""
        if self._loaded or self._closed:
            raise TrackingError("load_resume() was already called, or the session is closed")
        self._loaded, self._holding = True, False
        if self.resumed_from is None:
            return None
        f = self.resumed_from["file"]  # "found": last.pt, or prev.pt after a crash before last.pt was bound
        p = _load_torch(self.path / self.resumed_from["found"], f["sha256"], f["bytes"], map_location)
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
        for k, v in RunData(unit, arrays=False).final().items():
            self._writer.add_scalar("final/" + k, float(v), 0)
        digest, n = unit.digest, unit.info.n_exact
        self._emit_json("_run/complete", {"status": "complete", "session": self.session, "sessions": len(unit.segments),
                        "resumed": unit.info.resumed, "restarts": unit.info.restarts, "n_exact": n, "result_digest": digest},
                        P_RUN, b"")
        self._sync(close=True)
        return digest

    def fail(self, exc: BaseException | str) -> None:
        """Write ``_run/failed``, then flush, close and fsync (no-op on a closed session)."""
        if not self._closed:
            tb = "".join(traceback.format_exception(exc))[-8000:] if isinstance(exc, BaseException) else ""
            self._emit_json("_run/failed", {"status": "failed", "error": repr(exc)[:2000], "traceback": tb}, P_RUN, b"")
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


class NullRun:
    """What open_run(enabled=False) returns, for tests: every Run method exists and writes nothing."""

    path, session, resumed_from, restarted = None, 0, None, False
    __enter__, __exit__ = (lambda self: self), (lambda self, *exc: False)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_") or not callable(getattr(Run, name, None)):
            raise AttributeError(name)
        if name == "log_record":
            return lambda record_name, obj: payload_sha256(obj)
        return lambda *args, **kwargs: False if name == "checkpoint_due" else None


def open_run(key: RunKey | str, *, config: Mapping, root: str | Path | None = None, enabled: bool = True,
             resume: str = "auto", checkpoint_every_seconds: float | None = None,
             checkpoint_every_steps: int | None = None, repo: Path | None = None, flush_secs: int = 30) -> Run | NullRun:
    """Open the next session of a unit (fresh, resume or restart in place); the full rules are in docs/NNKNN_CORE.md.

    UnitBusyError: the last unfinished session's process still runs here (a cheap guard, not a lock;
    ``NNKNN_IGNORE_BUSY=1`` skips it), or a launch at the same moment also wrote its header (one that sees a rival
    steps back, its segment marked withdrawn). Also UnitCompleteError, ConfigMismatchError, TrackingError.
    """
    key = _as_key(key)
    if not enabled:
        return NullRun()
    from torch.utils.tensorboard import SummaryWriter

    path, config_sha = unit_dir(key, root), payload_sha256(config)
    if (resume := "restart" if os.environ.get("NNKNN_RESUME", "").strip() == "restart" else resume) not in ("auto", "restart"):
        raise ValueError("resume must be 'auto' or 'restart'")
    if nest := [str(a) for a in path.parents[:len(key.unit)] if _event_files(a)] or \
            [str(p) for p in (path.iterdir() if path.is_dir() else ()) if p.is_dir() and p.name != "weights"]:
        raise TrackingError(f"units cannot nest: {key.run_id} would be inside or around {nest[0]}")
    prev = _scan(path)
    if prev.segments and prev.info.run_id != key.run_id:  # e.g. case aliasing on Windows
        raise TrackingError(f"{path} holds run {prev.info.run_id}, not {key.run_id}")
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
        found = next((f"weights/resume/{n}" for n in ("last.pt", "prev.pt") if resume != "restart" and ckpt is not None
                      and (f := path / "weights" / "resume" / n).is_file() and f.stat().st_size == ckpt["file"]["bytes"]
                      and _file_sha256(f)[0] == ckpt["file"]["sha256"]), None)  # prev.pt: crash before last.pt was bound
        if found:
            resume_from = dict(ckpt, found=found)
            pending = {"from_session": ckpt["session"], "seq": ckpt["seq"], "step": ckpt["step"]}
        else:
            pending = {"restart": True, "reason": "forced restart" if resume == "restart" else
                       ("no checkpoint" if ckpt is None else "checkpoint file does not match")}
    kept = _logical(prev.segments, pending)[0] if prev.segments else []
    host, pid, start = socket.gethostname(), os.getpid(), resume_from["step"] + 1 if resume_from else 0
    if len(str(longest := path / f"events.out.tfevents.{int(time.time()):010d}.{host}.{pid}.99999.s{session:03d}")) >= MAX_PATH:
        raise ValueError(f"event file path {longest} would have {len(str(longest))} characters (limit {MAX_PATH})")
    path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(path), flush_secs=flush_secs, filename_suffix=f".s{session:03d}")
    event_file = Path(writer._get_file_writer().event_writer._file_name)
    commit, dirty, fingerprint = _provenance(Path(repo) if repo is not None else Path(__file__).resolve().parents[2])
    begin = {"schema": SCHEMA, "run_id": key.run_id, "workstream": key.workstream, "study": key.study,
             "unit": list(key.unit), "run_uuid": run_uuid, "session": session, "host": host, "pid": pid,
             "proc_id": _proc_id(pid), "argv": list(sys.argv), "started_utc": datetime.now(timezone.utc).isoformat(),
             "config_sha256": config_sha, "git_commit": commit, "dirty": dirty, "source_fingerprint": fingerprint,
             "event_file": event_file.name}
    run = Run(key=key, path=path, writer=writer, event_file=event_file, session=session, run_uuid=run_uuid, kept=kept,
              resume_from=resume_from, restarted=bool(pending and pending.get("restart")), max_seq=prev.max_seq,
              cadence=(checkpoint_every_seconds, checkpoint_every_steps))
    records = [("_run/begin", begin)] + ([("_run/resume", pending)] if pending is not None else [])
    run._emit([_value(t, canonical_json(v), P_RUN) for t, v in records], start)  # one event: whole or not at all
    run._sync()
    known = {s.file for s in prev.segments} | {event_file}
    if [s for s in map(_read_segment, sorted(set(_event_files(path)) - known)) if s.records and not s.damage and
            s.records[0].tag == "_run/begin" and s.records[0].value.get("session", 0) >= session and
            not (s.terminal and s.terminal.value.get("withdrawn"))]:  # a concurrent launch wrote its header too
        run._emit_json("_run/failed", {"status": "failed", "withdrawn": True, "error": "concurrent launch"}, P_RUN, b"")
        run._sync(close=True)
        raise UnitBusyError(f"{key.run_id}: another launch opened session {session} at the same time; this one stepped back")
    run._fw.add_event(_tb.event_pb2.Event(session_log=_tb.event_pb2.SessionLog(status=_tb.event_pb2.SessionLog.START)), step=start)
    mode = "fresh" if pending is None else ("restart in place" if pending.get("restart") else f"resume after step {start - 1}")
    writer.add_text("card/session", f"session {session} on {host} (pid {pid}): {mode}", start)
    wdir = path / "weights"
    for f in [*wdir.glob(".tmp-*"), *wdir.glob("resume/.tmp-*")]:  # partial files of crashed sessions
        f.unlink(missing_ok=True)
    if session > 1:  # named states that no kept record binds step aside, after the bundle that cut them is on disk
        live = {r.tag[10:] for r in kept if r.tag.startswith("rec/state/")}
        for f in sorted(wdir.glob("*.pt")):
            if (stem := f.name[:-3]) not in live and ".superseded-" not in stem:
                target = wdir / f"{stem}.superseded-s{session:03d}.pt"
                _replace(f, target if not target.exists() else target.with_suffix(f".{uuid.uuid4().hex[:8]}.pt"))
    if resume_from is None:
        run._emit_json("rec/config", config)
        run._singles.add("rec/config")
    return run


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2 or args[0] not in ("ls", "show", "verify") or any(w in args[1].lower() for w in _EMBARGO):
        print("usage: python -m model.common.tracking {ls|show|verify} <path> (no embargoed paths)", file=sys.stderr)
        return 2
    command, path = args[0], Path(args[1])
    if command == "ls":
        for info in iter_runs(path, status=None):
            print(f"{info.status:<10} s{info.sessions:<3} {os.path.relpath(info.path, path)}  [{info.run_id}]"
                  + (f"  ({info.reason})" if info.reason else ""))
        return 0
    if command == "verify":
        print("\n".join(problems := verify_run(path)) or "OK")
        return int(bool(problems))
    info = run_status(path)
    print(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in info.__dict__.items()}, indent=2))
    if info.status in ("invalid", "empty"):
        return 1
    data = load_run(path, allow_incomplete=True)
    print("\n".join([f"session {s['session']} {s['file']}: resume={s['resume']} end={(s['terminal'] or {}).get('tag')}"
                     for s in data.sessions()] + [f"x/{t}: {len(p)} point(s), last {p[-1]}" for t, p in sorted(data._x.items())]
                    + [f"records {sorted(data._single)}, row records {sorted(data._rows)}, arrays {sorted(data._arrays)}"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
