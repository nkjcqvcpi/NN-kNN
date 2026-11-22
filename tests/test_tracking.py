"""Tests for run tracking on TensorBoard event files (model/common/tracking.py; shared core, CPU only)."""
from __future__ import annotations

import enum
import math
import os
import random
import shutil
import socket
import struct
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("NNKNN_DEVICE", "cpu")

import numpy as np
import pytest
import torch

from model.common import tracking as tr

ROOT = Path(__file__).resolve().parents[1]
CFG = {"lr": 0.05, "marker": "zzzzzzzzzzzzzzzz"}


@pytest.fixture
def root(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setenv("NNKNN_RUNS", str(runs))
    for name in ("NNKNN_RESUME", "NNKNN_RESUME_EVERY_SECONDS", "NNKNN_RESUME_EVERY_STEPS"):
        monkeypatch.delenv(name, raising=False)
    return runs


@pytest.fixture
def dead(monkeypatch):
    """Earlier sessions of this process count as dead, so an abandoned Run simulates a crash."""
    monkeypatch.setattr(tr, "_session_alive", lambda begin: False)


def key(name="u1"):
    return tr.RunKey("core", "study", (name,))


def op(name="u1", every=None, config=CFG, **kw):
    return tr.open_run(key(name), config=config, checkpoint_every_steps=every, checkpoint_every_seconds=0, **kw)


def crash(run):
    """Abandon a session without a terminal record (what a crash after the last flush leaves)."""
    run._writer.close()
    run._closed = True


def upath(root, name="u1"):
    return root / "core" / "study" / name


def event_files(path):
    return sorted(p for p in path.iterdir() if "tfevents" in p.name)


def train(run, *, steps=12, crash_at=None, progress=False, sleep=0.0):
    """Tiny regression loop whose result depends on torch, numpy and python RNG (dropout, shuffling, noise)."""
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    X = torch.randn(48, 4)
    y = X @ torch.tensor([[1.0], [-2.0], [0.5], [3.0]]) + 0.1 * torch.randn(48, 1)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Dropout(0.3), torch.nn.Linear(8, 1))
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    run.log_record("data", {"n": 48, "x0": X[0].tolist()})
    run.log_array("x", X.numpy())
    start = 0
    payload = run.load_resume()
    if payload is not None:
        model.load_state_dict(payload.state["model"])
        opt.load_state_dict(payload.state["opt"])
        start = payload.step + 1
        payload.restore_rng()
    loss = None
    for step in range(start, steps):
        idx = torch.randperm(48)[:16]
        noise = random.random() + float(np.random.rand())
        model.train()
        loss = ((model(X[idx]) - y[idx]) ** 2).mean() * (1 + 1e-3 * noise)
        opt.zero_grad()
        loss.backward()
        opt.step()
        run.log_step(step, {"train/loss": loss.item(), "sys/wall": time.time()})
        run.log_records("trace", [{"step": step, "loss": loss.item()}])
        if step == 5:
            run.save_state("mid", model.state_dict(), step=step)
        if step % 4 == 0:
            run.log_array(f"w{step}", model[0].weight.detach().numpy().copy())
        if run.checkpoint_due(step):
            run.save_resume({"model": model.state_dict(), "opt": opt.state_dict()}, step=step)
        if progress:
            print(f"STEP {step}", flush=True)
            time.sleep(sleep)
        if step == crash_at:
            return None
    run.save_state("final", model.state_dict(), step=steps - 1)
    run.log_final({"loss": loss.item(), "steps": steps, "ok": True})
    return model


def finished(name, every=None):
    run = op(name, every)
    train(run)
    digest = run.complete()
    return digest, tr.load_run(upath(tr.runs_root(), name)).states()["final"]["state_sha256"]


# ----------------------------------------------------------------------------- roots and names
def test_runs_root_resolution(tmp_path, monkeypatch):
    monkeypatch.delenv("NNKNN_RUNS", raising=False)
    assert tr.DEFAULT_ROOT == ROOT.parent / "runs"
    with pytest.raises(FileNotFoundError):
        tr.runs_root(tmp_path / "missing")
    monkeypatch.setenv("NNKNN_RUNS", str(tmp_path))
    assert tr.runs_root() == tmp_path.resolve()
    other = tmp_path / "other"
    other.mkdir()
    assert tr.runs_root(other) == other.resolve()
    monkeypatch.setenv("NNKNN_RUNS", str(tmp_path / "nope"))
    with pytest.raises(FileNotFoundError):
        tr.runs_root()
    assert not (tmp_path / "nope").exists()


@pytest.mark.parametrize("bad", ["", ".hidden", "a/b", "x" * 65, "has space", "my_sel_fresh_set", "HELD_OUT", "weights",
                                 "lr=0.", "aux", "NUL.txt", "com1"])
def test_bad_components_rejected(root, bad):
    with pytest.raises(ValueError):
        tr.RunKey("core", "study", (bad,))


def test_key_forms_and_path_length(root):
    with pytest.raises(ValueError):
        tr.RunKey("t9", "study", ("u",))
    assert tr._as_key("t1/study/a/b") == tr.RunKey("t1", "study", ("a", "b"))
    assert tr.RunKey("core", "s", "u").unit == ("u",)
    assert tr.unit_dir("core/s/u") == root.resolve() / "core" / "s" / "u"
    with pytest.raises(ValueError):
        tr.open_run(tr.RunKey("core", "s", ("a" * 64,) * 3), config=CFG)


# ----------------------------------------------------------------------------- values and records
def test_exact_numbers_round_trip(root):
    values = {"f": 0.1234567890123456, "big": 2**62 + 1, "neg": -(2**63), "t": True, "f32": np.float32(0.1),
              "i8": np.int8(-5), "z": np.array(2.5), "nb": np.bool_(False)}
    with op() as run:
        run.log_final(values)
        run.log_step(3, {"x": 1 / 3})
    data = tr.load_run(upath(root))
    final = data.final()
    assert final == {k: (v.item() if isinstance(v, (np.generic, np.ndarray)) else v) for k, v in values.items()}
    assert type(final["big"]) is int and type(final["t"]) is bool and type(final["nb"]) is bool
    assert final["f32"] == float(np.float32(0.1)) and data.series("x") == [(3, 1 / 3)]


def test_non_finite_values(root):
    with op() as run:
        run.log_step(0, {"nan": math.nan, "inf": math.inf, "ninf": -math.inf})
        run.log_record("nf", {"a": [math.nan, math.inf, -math.inf]})
    data = tr.load_run(upath(root))
    assert math.isnan(data.series("nan")[0][1]) and data.series("inf") == [(0, math.inf)]
    a = data.record("nf")["a"]
    assert math.isnan(a[0]) and a[1:] == [math.inf, -math.inf]


def test_torch_tensors_and_strings_rejected(root):
    run = op()
    with pytest.raises(TypeError):
        run.log_step(0, {"a": torch.tensor(1.0)})
    with pytest.raises(TypeError):
        run.log_final({"a": torch.tensor(1)})
    with pytest.raises(TypeError):
        run.log_record("r", {"a": torch.zeros(2)})
    with pytest.raises(TypeError):
        run.log_array("a", torch.zeros(2))
    with pytest.raises(TypeError):
        run.log_step(0, {"a": "text"})
    with pytest.raises(OverflowError):
        run.log_step(0, {"a": 2**63})
    run.complete()


def test_canonical_json_rules():
    assert tr.canonical_json({"b": 1, "a": [1, 2.5, None, True], 3: "x"}) == b'{"3":"x","a":[1,2.5,null,true],"b":1}'
    assert tr.canonical_json({"u": "é"}) == '{"u":"é"}'.encode()
    assert tr.canonical_json((1, np.int64(2), np.float32(0.5), np.arange(3))) == b"[1,2,0.5,[0,1,2]]"
    assert tr.canonical_json(0.1) == b"0.1" and tr.canonical_json(math.inf) == b'{"$nonfinite":"inf"}'
    for bad in ({1: "a", "1": "b"}, {1, 2}, Path("x"), b"x", np.zeros(5000), {"$nonfinite": "nan"}, datetime.now()):
        with pytest.raises(TypeError):
            tr.canonical_json(bad)
    assert tr.decode_json(tr.canonical_json({"x": -math.inf})) == {"x": -math.inf}
    assert tr.payload_sha256({"a": 1, "b": 2}) == tr.payload_sha256({"b": 2, "a": 1})


def test_chunked_records_and_arrays(root, monkeypatch):
    monkeypatch.setattr(tr, "ROW_CHUNK_ROWS", 3)
    monkeypatch.setattr(tr, "ARRAY_CHUNK_BYTES", 40)
    arrays = {"a": np.arange(30, dtype=np.float64).reshape(10, 3), "h": np.arange(4, dtype=np.float16),
              "b": np.array([True, False]), "s": np.array(3, np.int64), "e": np.zeros((0, 2), np.float32),
              "u": np.arange(7, dtype=np.uint8)}
    with op() as run:
        run.log_records("rows", [{"i": i} for i in range(4)])
        run.log_records("rows", [{"i": i} for i in range(4, 7)])
        for name, a in arrays.items():
            run.log_array(name, a)
        run.log_table("t", {"x": np.arange(3), "y": np.ones(3, np.float32)})
        with pytest.raises(ValueError):
            run.log_table("bad", {"x": np.arange(3), "y": np.ones(2)})
        run.log_records("nonfinite", [{"x": math.nan, "y": [math.inf, 1.5]}, {"x": -math.inf, "y": []}])
        with pytest.raises(TypeError):  # rejected before anything is written
            run.log_records("bad", [{"ok": 1}, {"$nonfinite": "nan"}])
    data = tr.load_run(upath(root))
    assert data.records("rows") == [{"i": i} for i in range(7)]
    nf = data.records("nonfinite")
    assert math.isnan(nf[0]["x"]) and nf[0]["y"] == [math.inf, 1.5] and nf[1] == {"x": -math.inf, "y": []}
    assert "bad" not in data._rows
    assert sum(r.tag == "arr/a" for r in tr._scan(upath(root)).stream) == 10
    for name, a in arrays.items():
        got = data.array(name)
        assert got.dtype == a.dtype and got.shape == a.shape and np.array_equal(got, a)
    table = data.table("t")
    assert list(table) == ["x", "y"] and table["y"].dtype == np.float32


def test_row_chunking_does_not_change_digest(root, monkeypatch):
    rows = [{"i": i, "v": i / 7} for i in range(10)]
    with op("one") as run:
        run.log_records("rows", rows)
    monkeypatch.setattr(tr, "ROW_CHUNK_ROWS", 2)
    with op("many") as run:
        run.log_records("rows", rows[:3])
        run.log_records("rows", rows[3:])
    assert tr.run_status(upath(root, "one")).result_digest == tr.run_status(upath(root, "many")).result_digest


def test_duplicates_rejected(root):
    run = op()
    run.log_final({"a": 1})
    with pytest.raises(tr.TrackingError):
        run.log_final({"a": 2})
    run.log_step(0, {"g/a": 1})
    with pytest.raises(tr.TrackingError):
        run.log_step(0, {"g/a": 2})
    run.log_step(5, {"g/a": 2})
    with pytest.raises(tr.TrackingError):
        run.log_step(4, {"g/a": 3})  # steps must increase per tag
    with pytest.raises(TypeError):
        run.log_step(6.0, {"g/a": 3})  # a float step is not truncated
    with pytest.raises(ValueError):
        run.log_step(1, {"final/x": 1})
    run.log_record("r", {"v": 1})
    with pytest.raises(tr.TrackingError):
        run.log_record("r", {"v": 1})
    with pytest.raises(tr.TrackingError):
        run.log_records("r", [{"v": 1}])
    run.log_array("arr", np.zeros(2))
    with pytest.raises(tr.TrackingError):
        run.log_array("arr", np.zeros(2))
    with pytest.raises(ValueError):
        run.log_record("config", {})
    run.complete()


# ----------------------------------------------------------------------------- statuses
def test_complete_failed_incomplete_empty(root, dead):
    with op("ok") as run:
        run.log_step(0, {"a": 1.0})
    info = tr.run_status(upath(root, "ok"))
    assert (info.status, info.sessions, info.resumed, info.reason) == ("complete", 1, False, None)
    with pytest.raises(ValueError):
        with op("bad") as run:
            raise ValueError("boom")
    info = tr.run_status(upath(root, "bad"))
    assert info.status == "failed" and "boom" in info.reason
    run = op("crashed")
    run.log_step(0, {"a": 1.0})
    crash(run)
    assert tr.run_status(upath(root, "crashed")).status == "incomplete"
    with pytest.raises(tr.TrackingError):
        tr.load_run(upath(root, "crashed"))
    assert tr.load_run(upath(root, "crashed"), allow_incomplete=True).series("a") == [(0, 1.0)]
    assert tr.run_status(upath(root, "never")).status == "empty"
    from torch.utils.tensorboard import SummaryWriter

    SummaryWriter(str(upath(root, "hdr"))).close()
    assert tr.run_status(upath(root, "hdr")).status == "empty"


def test_truncated_tampered_and_missing_weights(root, tmp_path):
    with op() as run:
        run.log_step(0, {"a": 1.0})
        run.save_state("best", {"w": torch.ones(3)})
    src = upath(root)
    copies = {}
    for name in ("cut", "flip", "gone"):
        copies[name] = tmp_path / name
        shutil.copytree(src, copies[name])
    f = event_files(copies["cut"])[0]
    data = f.read_bytes()
    f.write_bytes(data[:-10])
    assert tr.run_status(copies["cut"]).status == "incomplete"
    f = event_files(copies["flip"])[0]
    data = bytearray(f.read_bytes())
    data[data.index(b"zzzzzzzz") + 3] ^= 0x01
    f.write_bytes(bytes(data))
    info = tr.run_status(copies["flip"])
    assert info.status == "invalid" and "corrupt" in info.reason
    (copies["gone"] / "weights" / "best.pt").rename(copies["gone"] / "weights" / "moved.bin")
    info = tr.run_status(copies["gone"])
    assert info.status == "invalid" and "best.pt" in info.reason


def test_save_state_and_load_state(root):
    state = {"w": torch.arange(6.0).reshape(2, 3), "b": torch.tensor([True, False]), "n": 3,
             "h": torch.ones(2, dtype=torch.bfloat16), "nested": [torch.zeros(1), {"k": None}]}
    with op() as run:
        ref = run.save_state("best", state, step=3, meta={"acc": 0.5})
        with pytest.raises(tr.TrackingError):
            run.save_state("best", state)
        with pytest.raises(TypeError):
            run.save_state("np", {"a": np.zeros(2)})
        run.save_state("diverged", {"w": torch.ones(1)}, meta={"val_loss": math.nan, "lo": -math.inf, "v": [math.inf]})
        with pytest.raises(TypeError):  # refused before the weight file is written
            run.save_state("badmeta", {"w": torch.ones(1)}, meta={"$nonfinite": "nan"})
    path = upath(root)
    meta = tr.load_run(path).states()["diverged"]["meta"]
    assert math.isnan(meta["val_loss"]) and meta["lo"] == -math.inf and meta["v"] == [math.inf]
    assert not (path / "weights" / "badmeta.pt").exists() and "badmeta" not in tr.load_run(path).states()
    loaded = tr.load_state(path, "best")
    assert torch.equal(loaded["w"], state["w"]) and torch.equal(loaded["h"], state["h"]) and loaded["n"] == 3
    rec = tr.load_run(path).states()["best"]
    assert ref["state_sha256"] == rec["state_sha256"] == tr.state_digest(loaded) == tr.state_digest(state)
    assert rec["meta"] == {"acc": 0.5} and rec["step"] == 3 and tr.load_run(path).state_path("best").is_file()
    assert tr.verify_run(path) == []
    f = path / "weights" / "best.pt"
    data = bytearray(f.read_bytes())
    data[len(data) // 2] ^= 0xFF
    f.write_bytes(bytes(data))
    assert tr.run_status(path).status == "complete"  # same size: only a hash check sees it
    with pytest.raises(tr.TrackingError):
        tr.load_state(path, "best")
    assert tr.verify_run(path) and "best" in tr.verify_run(path)[0]


# ----------------------------------------------------------------------------- resume
def test_resume_cuts_records_after_checkpoint(root, dead):
    run = op()
    run.log_step(0, {"a": 1.0})
    run.log_records("rows", [{"i": 0}])
    run.log_array("arr0", np.arange(3))
    run.save_resume({"x": torch.zeros(1)}, step=0)
    run.log_step(1, {"a": 2.0})
    run.log_final({"f": 1})
    run.log_record("late", {"v": 1})
    run.log_records("rows", [{"i": 1}])
    run.log_array("arr1", np.arange(2))
    crash(run)
    path = upath(root)
    assert tr.load_run(path, allow_incomplete=True).series("a") == [(0, 1.0), (1, 2.0)]
    run = op()
    assert run.resumed_from["step"] == 0 and not run.restarted and len(event_files(path)) == 2
    assert torch.equal(run.load_resume().state["x"], torch.zeros(1))
    data = tr.load_run(path, allow_incomplete=True)
    assert data.series("a") == [(0, 1.0)] and data.final() == {} and data.records("rows") == [{"i": 0}]
    with pytest.raises(KeyError):
        data.record("late")
    with pytest.raises(KeyError):
        data.array("arr1")
    with pytest.raises(tr.TrackingError):
        run.log_step(0, {"a": 9.0})  # kept step
    run.log_step(1, {"a": 3.0})
    run.log_final({"f": 2})
    run.log_records("rows", [{"i": "again"}])
    run.complete()
    data = tr.load_run(path)
    assert data.series("a") == [(0, 1.0), (1, 3.0)] and data.final() == {"f": 2}
    assert data.records("rows") == [{"i": 0}, {"i": "again"}] and np.array_equal(data.array("arr0"), np.arange(3))
    sessions = data.sessions()
    assert [s["session"] for s in sessions] == [1, 2] and sessions[1]["resume"] == {"from_session": 1, "seq": 1, "step": 0}
    assert data.info.resumed and data.info.sessions == 2


@pytest.mark.parametrize("every", [3, 5])
def test_interrupted_resume_matches_uninterrupted(root, dead, every):
    ref_digest, ref_state = finished("ref")  # no checkpoints at all
    run = op("u", every)
    assert train(run, crash_at=7) is None
    crash(run)
    run = op("u", every)
    assert run.resumed_from["step"] == (5 if every == 3 else 4)
    train(run)
    assert run.complete() == ref_digest
    data = tr.load_run(upath(root, "u"))
    assert data.states()["final"]["state_sha256"] == ref_state and data.info.resumed
    assert data.states()["mid"]["state_sha256"] == tr.load_run(upath(root, "ref")).states()["mid"]["state_sha256"]
    superseded = upath(root, "u") / "weights" / "mid.superseded-s002.pt"
    assert superseded.exists() == (every == 5)  # with every=5 the checkpoint (step 4) precedes "mid" (step 5)


def test_two_crashes_then_resume(root, dead):
    ref_digest, ref_state = finished("ref")
    for crash_at in (4, 7):
        run = op("u", 3)
        train(run, crash_at=crash_at)
        crash(run)
    run = op("u", 3)
    train(run)
    assert run.complete() == ref_digest
    assert tr.load_run(upath(root, "u")).info.sessions == 3


KILL_SCRIPT = r"""
import os, sys
sys.path[:0] = [sys.argv[1], os.path.join(sys.argv[1], "tests")]
import test_tracking as t
run = t.op("killed", 3)
t.train(run, progress=True, sleep=0.3)
print("DONE", flush=True)
"""


def test_killed_subprocess_resumes_exactly(root):
    ref_digest, ref_state = finished("ref")
    proc = subprocess.Popen([sys.executable, "-c", KILL_SCRIPT, str(ROOT)], stdout=subprocess.PIPE, text=True,
                            env={**os.environ, "NNKNN_RUNS": str(root)})
    try:
        for line in proc.stdout:
            if line.startswith("STEP 7"):
                break
        else:
            pytest.fail("child ended before step 7")
    finally:
        proc.kill()
        proc.wait()
        proc.stdout.close()
    assert tr.run_status(upath(root, "killed")).status == "incomplete"
    run = op("killed", 3)
    assert run.resumed_from is not None
    train(run)
    assert run.complete() == ref_digest
    assert tr.load_run(upath(root, "killed")).states()["final"]["state_sha256"] == ref_state


def test_restart_in_place_without_checkpoint(root, dead):
    run = op()
    run.log_step(0, {"a": 1.0})
    run.save_state("best", {"w": torch.ones(1)})
    crash(run)
    run = op()
    assert run.restarted and run.resumed_from is None and run.load_resume() is None
    assert (upath(root) / "weights" / "best.superseded-s002.pt").exists()
    run.log_step(0, {"a": 5.0})
    run.complete()
    data = tr.load_run(upath(root))
    assert data.series("a") == [(0, 5.0)] and data.config == CFG and data.states() == {}
    assert data.info.restarts == 1 and data.sessions()[1]["resume"]["restart"] is True


def test_stale_last_pt_restarts(root, dead, tmp_path):
    run = op(every=1)
    run.save_resume({"x": torch.zeros(1)}, step=0)
    crash(run)
    last = upath(root) / "weights" / "resume" / "last.pt"
    newer = tmp_path / "newer.pt"
    torch.save({"x": torch.ones(1)}, newer)
    os.replace(newer, last)  # a crash after os.replace and before the _run/checkpoint record
    run = op(every=1)
    assert run.restarted and run.resumed_from is None
    assert tr.load_run(upath(root), allow_incomplete=True).sessions()[1]["resume"]["reason"] == "checkpoint file does not match"
    run.complete()


def test_forced_restart_by_env(root, dead, monkeypatch):
    run = op(every=1)
    run.save_resume({}, step=0)
    crash(run)
    monkeypatch.setenv("NNKNN_RESUME", "restart")
    run = op(every=1)
    assert run.restarted
    run.complete()


def test_setup_writes_held_until_load_resume(root, dead):
    run = op(every=1)
    run.log_record("data", {"v": 1})
    run.log_step(0, {"a": 1.0})
    run.save_resume({}, step=0)
    crash(run)
    run = op(every=1)
    run.log_record("data", {"v": 999})  # setup re-execution: discarded
    run.log_step(0, {"a": 7.0})
    run.log_array("setup", np.ones(2))
    with pytest.raises(tr.TrackingError):
        run.save_resume({}, step=0)
    with pytest.raises(tr.TrackingError):
        run.complete()  # a resumed session must call load_resume first
    run.load_resume()
    run.log_step(1, {"a": 2.0})
    run.complete()
    data = tr.load_run(upath(root))
    assert data.record("data") == {"v": 1} and data.series("a") == [(0, 1.0), (1, 2.0)]
    with pytest.raises(KeyError):
        data.array("setup")


def test_named_generator_restored(root, dead):
    g = torch.Generator().manual_seed(7)
    run = op(every=1)
    run.save_resume({}, step=0, generators={"g": g})
    expected = torch.rand(3, generator=g)
    crash(run)
    run = op(every=1)
    payload = run.load_resume()
    other = torch.Generator().manual_seed(123)
    with pytest.raises(tr.TrackingError):
        payload.restore_rng({})  # the saved generators must all be supplied
    payload.restore_rng({"g": other})
    assert torch.equal(torch.rand(3, generator=other), expected)
    run.complete()


def test_failed_session_resumes_and_closed_session_refuses_writes(root):
    with pytest.raises(RuntimeError):
        with op(every=1) as run:
            run.log_step(0, {"a": 1.0})
            run.save_resume({}, step=0)
            raise RuntimeError("late failure")
    assert tr.run_status(upath(root)).status == "failed"
    run = op(every=1)
    assert run.resumed_from["step"] == 0
    run.load_resume()
    run.log_step(1, {"a": 2.0})
    run.complete()
    with pytest.raises(tr.TrackingError):
        run.log_step(2, {"a": 3.0})
    assert tr.load_run(upath(root)).series("a") == [(0, 1.0), (1, 2.0)]


def test_iter_runs_filters_by_status(root, dead):
    with op("a") as run:
        run.save_state("s", {"w": torch.ones(1)})
    crash(op("b"))
    assert [i.run_id for i in tr.iter_runs(root)] == ["core/study/a"]
    assert sorted((i.run_id, i.status) for i in tr.iter_runs(root, status=None)) == [
        ("core/study/a", "complete"), ("core/study/b", "incomplete")]


def test_unit_complete_busy_and_config_errors(root, monkeypatch):
    with op("done") as run:
        run.log_step(0, {"a": 1})
    with pytest.raises(tr.UnitCompleteError):
        op("done")
    live = op("live")  # this process is alive: a second launch is refused
    with pytest.raises(tr.UnitBusyError):
        op("live")
    live.complete()
    run = op("cfg")
    crash(run)
    monkeypatch.setattr(tr, "_session_alive", lambda begin: False)
    with pytest.raises(tr.ConfigMismatchError):
        op("cfg", config={"lr": 1.0})
    run = op("cfg", config={"lr": 1.0}, resume="restart")
    assert run.restarted
    run.complete()
    assert tr.load_run(upath(root, "cfg")).config == {"lr": 1.0}


def test_resolve_unit(root, dead):
    assert tr.resolve_unit(key("x"), config=CFG) is None
    run = op("x")
    crash(run)
    assert tr.resolve_unit(key("x"), config=CFG) is None
    with op("x") as run:
        run.load_resume()
    info = tr.resolve_unit(key("x"), config=CFG)
    assert info.status == "complete" and info.run_id == "core/study/x"
    with pytest.raises(tr.ConfigMismatchError):
        tr.resolve_unit(key("x"), config={"other": 1})
    assert [i.run_id for i in tr.iter_runs(root)] == ["core/study/x"]


def test_checkpoint_cadence(root, monkeypatch):
    run = op(every=4)
    assert [s for s in range(10) if run.checkpoint_due(s) and run.save_resume({}, step=s)] == [3, 7]
    run.complete()
    monkeypatch.setenv("NNKNN_RESUME_EVERY_SECONDS", "0.01")
    run = op("timed")
    time.sleep(0.02)
    assert run.checkpoint_due(0)
    run.complete()


def test_event_accumulator_view_after_resume(root, dead):
    """TensorBoard's Python loader: a re-logged tag is one curve; a tag not re-logged ends at the checkpoint."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    run = op()
    for s in range(8):
        run.log_step(s, {"train/loss": float(s), "other/x": float(s)})
        if s == 4:
            run.save_resume({}, step=4)
    crash(run)
    time.sleep(1.1)  # event file names start with the start second; display order follows name order
    run = op()
    run.load_resume()
    for s in range(5, 10):
        run.log_step(s, {"train/loss": 100.0 + s})
    run.log_final({"acc": 0.75})
    run.complete()
    acc = EventAccumulator(str(upath(root)), size_guidance={"scalars": 0})
    acc.Reload()
    loss = [(e.step, e.value) for e in acc.Scalars("train/loss")]
    assert loss == [(s, float(s)) for s in range(5)] + [(s, 100.0 + s) for s in range(5, 10)]
    assert [e.step for e in acc.Scalars("other/x")] == [0, 1, 2, 3, 4]
    assert [(e.step, e.value) for e in acc.Scalars("final/acc")] == [(0, 0.75)]


def test_logging_leaves_rng_and_results_untouched(root):
    states = (torch.get_rng_state(), np.random.get_state()[1].copy(), random.getstate())
    with op(every=1) as run:
        run.log_step(0, {"a": 1.0})
        run.log_final({"b": 2})
        run.log_record("r", {"c": 3})
        run.log_records("rows", [{"d": 4}])
        run.log_array("e", np.ones(3))
        run.save_state("s", {"w": torch.ones(2)})
        run.save_resume({"w": torch.ones(2)}, step=0)
    assert torch.equal(torch.get_rng_state(), states[0]) and np.array_equal(np.random.get_state()[1], states[1])
    assert random.getstate() == states[2]
    logged = op("logged", every=3)
    a = train(logged)
    logged.complete()
    b = train(tr.open_run(key("null"), config=CFG, enabled=False))
    assert all(torch.equal(p, q) for p, q in zip(a.state_dict().values(), b.state_dict().values()))


def test_null_run(root):
    run = tr.open_run(key("n"), config=CFG, enabled=False)
    assert isinstance(run, tr.NullRun) and run.load_resume() is None and not run.checkpoint_due(10**6)
    with run:
        run.log_step(0, {"a": 1.0})
        assert run.save_state("s", {"w": torch.ones(1)}) is None and run.save_resume({}, step=0) is None
    with pytest.raises(AttributeError):
        run.log_stpe(0, {})
    assert not upath(root, "n").exists()


def test_pinned_tensorboard_internals(tmp_path):
    from tensorboard.backend.event_processing import event_file_loader
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(str(tmp_path / "pin"), filename_suffix=".s001")
    fw = writer._get_file_writer()
    assert all(hasattr(fw, name) for name in ("add_summary", "add_event", "flush"))
    fw.add_event(tr.event_pb2.Event(session_log=tr.event_pb2.SessionLog(status=tr.event_pb2.SessionLog.START)), step=3)
    writer.close()
    (f,) = event_files(tmp_path / "pin")
    assert f.name.endswith(".s001") and Path(fw.event_writer._file_name) == f
    events = [tr.event_pb2.Event.FromString(r) for r in event_file_loader.RawEventFileLoader(str(f)).Load()]
    assert any(e.HasField("session_log") and e.step == 3 for e in events)


def test_cli_and_torch_free_reader(root):
    with op() as run:
        run.log_step(0, {"a": 1.0})
        run.save_state("best", {"w": torch.ones(1)})
    path = upath(root)
    assert tr.main(["ls", str(root)]) == 0 and tr.main(["show", str(path)]) == 0 and tr.main(["verify", str(path)]) == 0
    assert tr.main(["show", str(root / "a_held_out_set")]) == 2  # refused before anything is read
    code = "import sys, model.common.tracking as t; print(t.run_status(sys.argv[1]).status); assert 'torch' not in sys.modules"
    out = subprocess.run([sys.executable, "-c", code, str(path)], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.strip() == "complete", out.stderr
    out = subprocess.run([sys.executable, "-m", "model.common.tracking", "verify", str(path)], cwd=ROOT,
                         capture_output=True, text=True)
    assert out.returncode == 0 and "OK" in out.stdout, out.stderr


# ----------------------------------------------------------------------------- review fixes (regressions)
def test_numpy_scalars_in_states_load_back(root, dead):
    run = op(every=1)
    run.save_state("s", {"acc": np.float32(0.25), "w": torch.ones(1)})
    with pytest.raises(TypeError):
        run.save_state("e", {"k": enum.IntEnum("E", "A").A})  # weights_only loading would refuse it later
    run.save_resume({"best": np.float64(0.5), "n": np.int64(3), "ok": np.bool_(True), "w": torch.ones(1)}, step=0)
    crash(run)
    run = op(every=1)
    state = run.load_resume().state
    assert state["best"] == 0.5 and type(state["best"]) is float and type(state["n"]) is int and state["ok"] is True
    run.complete()
    assert tr.load_state(upath(root), "s")["acc"] == 0.25


def test_busy_guard_checks_process_identity(root, monkeypatch):
    me = {"host": socket.gethostname(), "pid": os.getpid(), "proc_id": tr._proc_id(os.getpid())}
    assert me["proc_id"] and tr._session_alive(me)
    assert not tr._session_alive({**me, "proc_id": "reused"})  # same pid, another process (pid reuse, reboot)
    assert not tr._session_alive({**me, "host": "elsewhere"}) and not tr._session_alive({**me, "pid": -1})
    live = op("live")
    monkeypatch.setenv("NNKNN_IGNORE_BUSY", "1")
    run = op("live")  # the explicit override
    assert run.session == 2 and run.restarted
    crash(live)
    run.complete()


def test_file_growing_during_scan_reads_incomplete(root, dead, monkeypatch):
    run = op()
    run.log_step(0, {"a": 1.0})
    crash(run)
    (f,) = event_files(upath(root))
    data, pos = f.read_bytes(), 0
    while pos < len(data):  # find the last whole frame
        frame = data[pos:pos + 16 + struct.unpack("<Q", data[pos:pos + 8])[0]]
        pos += len(frame)

    class Growing(tr.RawEventFileLoader):
        def Load(self):  # the writer appends a whole record right after the reader reached EOF
            yield from super().Load()
            with f.open("ab") as fh:
                fh.write(frame)

    monkeypatch.setattr(tr, "RawEventFileLoader", Growing)
    assert tr.run_status(upath(root)).status == "incomplete"


def test_crash_before_checkpoint_record_resumes_from_prev(root, dead, monkeypatch):
    run = op(every=1)
    run.log_step(0, {"a": 1.0})
    run.save_resume({"x": torch.zeros(1)}, step=0)
    run.log_step(1, {"a": 2.0})
    real = run._emit_json
    monkeypatch.setattr(run, "_emit_json", lambda tag, *a: (_ for _ in ()).throw(OSError("killed"))
                        if tag == "_run/checkpoint" else real(tag, *a))
    with pytest.raises(OSError):
        run.save_resume({"x": torch.ones(1)}, step=1)  # last.pt replaced, record never written
    crash(run)
    rdir = upath(root) / "weights" / "resume"
    (rdir / ".tmp-dead.pt").write_bytes(b"partial")
    run = op(every=1)
    assert run.resumed_from["step"] == 0 and run.resumed_from["found"] == "weights/resume/prev.pt"
    assert not (rdir / ".tmp-dead.pt").exists()  # partial files of crashed sessions are removed
    assert torch.equal(run.load_resume().state["x"], torch.zeros(1))
    run.log_step(1, {"a": 3.0})
    run.complete()
    assert tr.load_run(upath(root)).series("a") == [(0, 1.0), (1, 3.0)]


def test_zero_filled_tail_reads_as_crash(root, dead):
    run = op(every=1)
    run.log_step(0, {"a": 1.0})
    run.save_resume({}, step=0)
    crash(run)
    (f,) = event_files(upath(root))
    size = struct.pack("<Q", 40)  # a power cut: the header of the next record reached the disk, its data did not
    f.write_bytes(f.read_bytes() + size + struct.pack("<I", tr.masked_crc32c(size)) + b"\0" * 44)
    assert tr.run_status(upath(root)).status == "incomplete"
    run = op(every=1)
    assert run.resumed_from["seq"] == 1
    run.load_resume()
    run.complete()


def test_concurrent_launch_steps_back(root, dead, monkeypatch):
    first, real, calls = op(), tr._scan, []

    def late(path):  # the second launch scanned before the first one's header reached the file
        calls.append(path)
        return real(path.parent / "missing") if len(calls) == 1 else real(path)

    monkeypatch.setattr(tr, "_scan", late)
    with pytest.raises(tr.UnitBusyError):
        op()
    monkeypatch.setattr(tr, "_scan", real)
    first.log_step(0, {"a": 1.0})
    first.complete()
    info = tr.run_status(upath(root))
    assert info.status == "complete" and info.sessions == 1 and len(event_files(upath(root))) == 2


def test_nesting_and_aliased_run_ids_refused(root):
    with op("a"):
        pass
    with pytest.raises(tr.TrackingError):
        tr.open_run(tr.RunKey("core", "study", ("a", "b")), config=CFG)
    with tr.open_run(tr.RunKey("core", "study", ("p", "q")), config=CFG):
        pass
    with pytest.raises(tr.TrackingError):
        tr.open_run(tr.RunKey("core", "study", ("p",)), config=CFG)
    shutil.copytree(upath(root, "a"), upath(root, "alias"))  # what case aliasing on Windows produces
    with pytest.raises(tr.TrackingError):
        op("alias")
    with pytest.raises(tr.TrackingError):
        tr.resolve_unit(key("alias"), config=CFG)


def test_restart_after_resume_and_resume_after_restart(root, dead, monkeypatch):
    from torch.utils.tensorboard import SummaryWriter

    run = op(every=1)
    run.log_step(0, {"a": 1.0})
    run.save_resume({}, step=0)
    run.log_step(1, {"a": 2.0})
    crash(run)
    run = op(every=1)
    run.log_record("setup", {"v": 1})
    crash(run)  # dies in the setup phase of a resume
    run = op(every=1)
    assert run.resumed_from["seq"] == 1
    run.load_resume()
    run.log_step(1, {"a": 3.0})
    crash(run)
    assert tr.run_status(upath(root)).resumed
    monkeypatch.setenv("NNKNN_RESUME", "restart")
    run = op(every=1)
    run.load_resume()
    run.log_step(0, {"a": 10.0})
    run.save_resume({}, step=0)
    crash(run)
    info = tr.run_status(upath(root))
    assert not info.resumed and info.restarts == 1  # the restart cut the earlier resume
    monkeypatch.delenv("NNKNN_RESUME")
    SummaryWriter(str(upath(root))).close()  # a header-only segment in the middle is ignored
    run = op(every=1)
    assert run.session == 5 and run.resumed_from["seq"] == 2
    run.load_resume()
    run.log_step(1, {"a": 11.0})
    run.complete()
    data = tr.load_run(upath(root))
    assert data.series("a") == [(0, 10.0), (1, 11.0)] and data.info.resumed and data.info.restarts == 1


def test_restore_rng_brings_states_back_to_cpu(root, dead):
    run = op(every=1)
    run.save_resume({}, step=0)
    expected = torch.rand(2)
    crash(run)
    run = op(every=1)
    payload = run.load_resume()

    class OnDevice:  # what load_resume(map_location=<device>) leaves in the snapshot
        def __init__(self, t):
            self.t = t

        def cpu(self):
            return self.t

    payload.rng["torch"] = OnDevice(payload.rng["torch"])
    payload.restore_rng()
    assert torch.equal(torch.rand(2), expected)
    run.complete()


def test_small_fixes(root, monkeypatch, capsys):
    t = torch.ones(1)
    assert tr.state_digest({"a/b": t}) != tr.state_digest({"a": {"b": t}})
    assert len({tr.state_digest({0: t}), tr.state_digest({"0": t}), tr.state_digest([t])}) == 3
    null = tr.open_run(key("n"), config=CFG, enabled=False)
    assert null.log_record("r", {"a": 1}) == tr.payload_sha256({"a": 1})
    with pytest.raises(ValueError):
        tr.open_run(tr.RunKey("core", "s", ("bad name",)), config=CFG, enabled=False)
    monkeypatch.setattr(tr, "ARRAY_CHUNK_BYTES", 24)
    run, calls = op(), []
    real = run._emit
    monkeypatch.setattr(run, "_emit", lambda values, step=0: real(values, step) if calls.append(step) or len(calls) != 2
                        else (_ for _ in ()).throw(OSError("writer died")))
    with pytest.raises(OSError):
        run.log_array("big", np.zeros((10, 3)))
    with pytest.raises(tr.TrackingError):
        run.log_array("big", np.zeros((10, 3)))  # a retry raises instead of appending duplicate chunks
    run.fail("test")
    assert tr.main(["ls", str(root)]) == 0 and "[core/study/u1]" in capsys.readouterr().out
